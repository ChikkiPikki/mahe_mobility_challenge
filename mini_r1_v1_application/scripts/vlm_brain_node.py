#!/usr/bin/env python3
"""VLM Navigation Brain - Main ROS2 Node.

Subscribes to camera and odom, calls VLM API, publishes Twist commands.
Integrates VLMap spatial memory and sign detection for hybrid reasoning.
Runs the continuous perceive-reason-act loop.
"""

import base64
import io
import json
import math
import re
import signal
import threading
import time
from collections import deque

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from openai import OpenAI

import os, sys
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
import vlm_config as config

# Optional hybrid modules
_vlmap_available = False
_sign_detector_available = False
try:
    from vlmap_builder import VLMapBuilder, VLMAP_ENABLED
    _vlmap_available = True
except ImportError:
    VLMAP_ENABLED = False

try:
    from sign_detector import SignDetector, SIGN_DETECTOR_ENABLED
    _sign_detector_available = True
except ImportError:
    SIGN_DETECTOR_ENABLED = False


class NavigationBrain(Node):
    def __init__(self):
        super().__init__("navigation_brain")

        # --- Subscribers ---
        self.bridge = CvBridge()
        self.create_subscription(Image, config.CAMERA_TOPIC, self._camera_cb, 1)
        self.create_subscription(Odometry, config.ODOM_TOPIC, self._odom_cb, 1)
        # Tool data subscribers
        self.create_subscription(String, '/mini_r1/sign_detections', self._sign_detect_cb, 10)
        self.create_subscription(String, '/mini_r1/navigator/status', self._nav_status_cb, 10)

        # --- Publishers ---
        self.cmd_pub = self.create_publisher(Twist, config.CMD_VEL_TOPIC, 1)
        self.command_pub = self.create_publisher(String, '/vlm_brain/command', 10)
        self.status_pub = self.create_publisher(String, '/vlm_brain/status', 10)

        # --- Tool data ---
        self._latest_sign = ""
        self._sign_time = 0.0
        self._sign_history = []
        self._nav_status = ""

        # --- Thread-safe state ---
        self._lock = threading.Lock()
        self._latest_frame: bytes | None = None
        self._frame_time: float = 0.0
        self._odom_x: float = 0.0
        self._odom_y: float = 0.0
        self._odom_theta: float = 0.0

        # --- Navigation state ---
        self.objective: str = ""
        self.paused: bool = True  # Start paused until objective is set
        self.observations: deque = deque(maxlen=config.MAX_OBSERVATIONS)
        self.odom_history: deque = deque(maxlen=config.STUCK_CYCLE_COUNT)
        self.visited_positions: list = []
        self.cycle_count: int = 0
        self.api_call_count: int = 0
        self.current_provider: str = "nvidia"

        # --- Queued command for overlapped execution ---
        self._queued_command: dict | None = None
        self._queue_lock = threading.Lock()

        # --- VLM clients ---
        self._local_client = None
        self._nvidia_client = None
        self._openrouter_client = None
        if config.LOCAL_VLM_ENABLED:
            self._local_client = OpenAI(
                base_url=config.LOCAL_VLM_BASE_URL,
                api_key=config.LOCAL_VLM_API_KEY,
            )
            self.current_provider = "local"
        if config.NVIDIA_API_KEY:
            self._nvidia_client = OpenAI(
                base_url=config.NVIDIA_BASE_URL,
                api_key=config.NVIDIA_API_KEY,
            )
        if config.OPENROUTER_API_KEY:
            self._openrouter_client = OpenAI(
                base_url=config.OPENROUTER_BASE_URL,
                api_key=config.OPENROUTER_API_KEY,
            )

        # --- Load prompt template ---
        with open(config.PROMPT_FILE) as f:
            self.prompt_template = f.read()

        # --- Hybrid modules (VLMap + Sign Detector) ---
        self.vlmap: VLMapBuilder | None = None
        self.sign_detector: SignDetector | None = None

        # --- Dashboard callback (set by dashboard.py) ---
        self.on_reasoning: callable | None = None

        # --- Shutdown safety ---
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.get_logger().info("Navigation Brain initialized")

    # ─── ROS2 Callbacks ─────────────────────────────────────────────

    def _camera_cb(self, msg: Image):
        """Store latest camera frame as JPEG bytes."""
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        cv_image = cv2.resize(cv_image, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
        _, jpeg_buf = cv2.imencode(
            ".jpg", cv_image, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY]
        )
        with self._lock:
            self._latest_frame = jpeg_buf.tobytes()
            self._frame_time = time.time()

    def _odom_cb(self, msg: Odometry):
        """Store latest odometry."""
        q = msg.pose.pose.orientation
        # Convert quaternion to yaw
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        with self._lock:
            self._odom_x = msg.pose.pose.position.x
            self._odom_y = msg.pose.pose.position.y
            self._odom_theta = math.degrees(yaw)

    def _sign_detect_cb(self, msg):
        self._latest_sign = msg.data
        self._sign_time = time.time()
        self._sign_history.append({"direction": msg.data, "time": time.time()})
        if len(self._sign_history) > 10:
            self._sign_history = self._sign_history[-10:]

    def _nav_status_cb(self, msg):
        self._nav_status = msg.data

    def _build_tool_context(self) -> str:
        """Build tool/sensor context for the VLM prompt."""
        lines = []

        # Navigator status
        if self._nav_status:
            try:
                nav = json.loads(self._nav_status)
                lines.append(f"Navigator state: {nav.get('state', '?')}, behavior: {nav.get('behavior', '?')}")
            except json.JSONDecodeError:
                pass

        # Sign detections
        now = time.time()
        recent_signs = [s for s in self._sign_history if now - s['time'] < 10.0]
        if recent_signs:
            sign_text = ", ".join(f"{s['direction']} ({now - s['time']:.0f}s ago)" for s in recent_signs[-3:])
            lines.append(f"Detected signs: {sign_text}")
        else:
            lines.append("Detected signs: none nearby")

        # Available behaviors
        lines.append("")
        lines.append("Available behaviors you can command: turn_left_90, turn_right_90, turn_180, stop, reverse, explore_random, move_forward, move_cautious")
        lines.append("Available recoveries: dead_end_recovery, stuck_recovery, loop_recovery")

        return "\n".join(lines)

    # ─── Frame + Odom Access ────────────────────────────────────────

    def get_frame_b64(self) -> str | None:
        """Get latest frame as base64 string."""
        with self._lock:
            if self._latest_frame is None:
                return None
            return base64.b64encode(self._latest_frame).decode("utf-8")

    def get_odom(self) -> tuple[float, float, float]:
        """Get latest odom (x, y, theta_degrees)."""
        with self._lock:
            return self._odom_x, self._odom_y, self._odom_theta

    # ─── Stuck Detection ────────────────────────────────────────────

    def check_stuck(self) -> bool:
        """Check if robot is stuck based on odom history."""
        if len(self.odom_history) < config.STUCK_CYCLE_COUNT:
            return False

        positions = list(self.odom_history)
        first = positions[0]
        last = positions[-1]

        dx = last[0] - first[0]
        dy = last[1] - first[1]
        pos_delta = math.sqrt(dx * dx + dy * dy)

        heading_delta = abs(last[2] - first[2])
        if heading_delta > 180:
            heading_delta = 360 - heading_delta

        return (
            pos_delta < config.STUCK_POSITION_THRESHOLD
            and heading_delta < config.STUCK_HEADING_THRESHOLD
        )

    # ─── Revisit Detection ──────────────────────────────────────────

    def check_revisiting(self) -> bool:
        """Check if robot is near a previously visited position."""
        x, y, _ = self.get_odom()
        for vx, vy in self.visited_positions:
            dx = x - vx
            dy = y - vy
            if math.sqrt(dx * dx + dy * dy) < config.REVISIT_DISTANCE_THRESHOLD:
                return True
        return False

    def record_position(self):
        """Record current position for revisit detection."""
        x, y, _ = self.get_odom()
        self.visited_positions.append((x, y))

    # ─── VLM Call ───────────────────────────────────────────────────

    def _build_sign_context(self) -> str:
        """Get structured sign detections for the prompt."""
        if not self.sign_detector:
            return ""
        detections = self.sign_detector.get_latest_detections()
        signs = detections.get("signs", [])
        if not signs:
            return "  (no signs detected)"
        lines = []
        for s in signs:
            if s["type"] == "text":
                lines.append(f"  - Text sign: \"{s['content']}\" ({s['position']} side, conf={s['confidence']})")
            elif s["type"] == "arrow":
                lines.append(f"  - {s['color'].title()} arrow pointing {s['direction']} ({s['position']} side)")
        return "\n".join(lines)

    def _build_vlmap_context(self) -> str:
        """Query VLMap for spatial context relevant to the objective."""
        if not self.vlmap or not self.objective:
            return ""
        cells = self.vlmap.get_cells_filled()
        if cells == 0:
            return "  (map still building, no spatial data yet)"
        x, y, _ = self.get_odom()
        # Query for objective-related locations nearby
        wx, wy, score = self.vlmap.query_nearby(self.objective, x, y, radius=5.0)
        lines = [f"  Map cells explored: {cells}"]
        if score > 0.15:
            dx = wx - x
            dy = wy - y
            dist = math.sqrt(dx * dx + dy * dy)
            angle = math.degrees(math.atan2(dy, dx))
            lines.append(f"  Best match for \"{self.objective}\": {dist:.1f}m away at ~{angle:.0f} deg (score={score:.2f})")
        else:
            lines.append(f"  No strong spatial match for objective yet")
        return "\n".join(lines)

    def call_vlm(self, frame_b64: str) -> dict | None:
        """Call VLM API with frame and prompt. Returns parsed JSON or None."""
        x, y, theta = self.get_odom()

        # Build status
        status = "normal"
        if self.check_stuck():
            status = "appears_stuck — try a different direction"
        elif self.check_revisiting():
            status = "revisiting_area — you have been here before, explore a new direction"

        # Build prompt
        obs_text = "\n".join(
            f"  {i+1}. {obs}" for i, obs in enumerate(self.observations)
        ) or "  (none yet)"

        # Build tool context
        tool_context = self._build_tool_context()
        sign_context = self._build_sign_context()
        vlmap_context = self._build_vlmap_context()

        try:
            prompt = self.prompt_template.format(
                objective=self.objective,
                observations=obs_text,
                x=x, y=y, theta=theta,
                status=status,
                sign_detections=sign_context or "  (sign detector off)",
                spatial_memory=vlmap_context or "  (spatial map off)",
            )
        except KeyError:
            # Fallback if prompt template has different placeholders
            prompt = self.prompt_template

        # Append tool context
        prompt += f"\n\nSensor & tool data:\n{tool_context}"

        # Try primary, then fallback: local → nvidia → openrouter
        providers = []
        if self._local_client:
            providers.append(("local", self._local_client, config.LOCAL_VLM_MODEL))
        if self.current_provider == "nvidia" and self._nvidia_client:
            providers.append(("nvidia", self._nvidia_client, config.NVIDIA_MODEL))
        if self._openrouter_client:
            providers.append(("openrouter", self._openrouter_client, config.OPENROUTER_MODEL))
        if self.current_provider != "nvidia" and self._nvidia_client:
            providers.append(("nvidia", self._nvidia_client, config.NVIDIA_MODEL))

        for provider_name, client, model in providers:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{frame_b64}"
                                    },
                                },
                            ],
                        },
                    ],
                    max_tokens=300,
                    temperature=0.1,
                )

                raw = response.choices[0].message.content.strip()
                self.get_logger().info(f"[{provider_name}] Raw VLM response: {raw[:200]}")
                parsed = self._parse_vlm_response(raw)
                if parsed:
                    self.api_call_count += 1
                    self.current_provider = provider_name
                    return parsed

                self.get_logger().warn(
                    f"[{provider_name}] Bad JSON, retrying parse..."
                )

            except Exception as e:
                self.get_logger().error(f"[{provider_name}] API error: {e}")
                continue

        self.get_logger().error("All VLM providers failed")
        return None

    def _parse_vlm_response(self, raw: str) -> dict | None:
        """Parse VLM response with multiple fallback strategies."""
        # Strategy 1: strip markdown fences and parse JSON
        cleaned = re.sub(r"```(?:json)?\s*", "", raw)
        cleaned = cleaned.strip().rstrip("`")
        try:
            data = json.loads(cleaned)
            if "action" in data:
                if data["action"] not in config.VALID_ACTIONS:
                    data["action"] = "stop"
                return data
        except json.JSONDecodeError:
            pass

        # Strategy 2: find JSON object embedded in text
        json_match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                if "action" in data:
                    if data["action"] not in config.VALID_ACTIONS:
                        data["action"] = "stop"
                    return data
            except json.JSONDecodeError:
                pass

        # Strategy 3: parse markdown bold key-value pairs (Llama-style)
        fields = {}
        for match in re.finditer(r'\*\*(\w[\w\s]*?):\*\*\s*(.+)', raw):
            key = match.group(1).strip().lower().replace(" ", "_")
            val = match.group(2).strip()
            fields[key] = val

        if "action" in fields:
            try:
                data = {
                    "action": fields.get("action", "stop").lower(),
                    "speed": float(fields.get("speed", 0.3)),
                    "turn_angle": float(fields.get("turn_angle", 0.0)),
                    "duration": float(fields.get("duration", 1.0)),
                    "reasoning": fields.get("reasoning", ""),
                    "objective_complete": fields.get("objective_complete", "false").lower() == "true",
                    "observation": fields.get("observation", ""),
                }
                if data["action"] not in config.VALID_ACTIONS:
                    data["action"] = "stop"
                return data
            except (ValueError, KeyError):
                pass

        return None

    # ─── Command Execution ──────────────────────────────────────────

    def execute_command(self, cmd: dict):
        """Map VLM action to behavioral command for the navigator."""
        action = cmd.get("action", "stop")

        # Map VLM actions to navigator behaviors
        behavior_map = {
            "left": "turn_left_90",
            "right": "turn_right_90",
            "backward": "reverse",
            "stop": "stop",
        }

        if action == "forward":
            # "forward" = let the navigator's gap_follow handle it (no override)
            behavior_name = "gap_follow (no override)"
            self.get_logger().info(f"VLM says forward — navigator continues autonomously")
        elif action in behavior_map:
            behavior_name = behavior_map[action]
            # Publish behavioral command to navigator (override)
            nav_cmd = json.dumps({
                "action": "execute_behavior",
                "action_args": {"name": behavior_name},
            })
            cmd_msg = String()
            cmd_msg.data = nav_cmd
            self.command_pub.publish(cmd_msg)
            self.get_logger().info(f"VLM OVERRIDE → {behavior_name}")
        else:
            behavior_name = action

        # Also publish status for dashboard/RViz
        status_msg = String()
        status_msg.data = json.dumps({
            "provider": self.current_provider,
            "reasoning": cmd.get("reasoning", ""),
            "observation": cmd.get("observation", ""),
            "action": action,
            "behavior": behavior_name,
            "api_calls": self.api_call_count,
        })
        self.status_pub.publish(status_msg)

        self.get_logger().info(
            f"VLM [{self.current_provider}] action={action} → behavior={behavior_name} | "
            f"reason: {cmd.get('reasoning', '')[:80]}")

        # Wait for behavior to execute (shorter than raw Twist sleep)
        duration = self._clamp(
            cmd.get("duration", 1.0), config.MIN_DURATION, config.MAX_DURATION
        )
        elapsed = 0.0
        while elapsed < duration:
            if self.paused:
                return
            time.sleep(config.SLEEP_CHUNK)
            elapsed += config.SLEEP_CHUNK

    def _publish_stop(self):
        """Publish zero velocity."""
        self.cmd_pub.publish(Twist())

    @staticmethod
    def _clamp(value: float, min_val: float, max_val: float) -> float:
        return max(min_val, min(max_val, value))

    # ─── Shutdown ───────────────────────────────────────────────────

    def _signal_handler(self, signum, frame):
        """Publish stop on shutdown."""
        self.get_logger().info("Shutting down — stopping robot")
        self._publish_stop()
        self.paused = True

    def shutdown(self):
        """Clean shutdown."""
        self._publish_stop()
        self.paused = True

    # ─── Main Loop ──────────────────────────────────────────────────

    def set_objective(self, objective: str):
        """Set navigation objective and unpause."""
        self.objective = objective
        self.paused = False
        self.observations.clear()
        self.visited_positions.clear()
        self.odom_history.clear()
        self.cycle_count = 0
        self.get_logger().info(f"New objective: {objective}")

    def stop_navigation(self):
        """Stop the robot immediately."""
        self.paused = True
        self._publish_stop()
        self.get_logger().info("Navigation stopped")

    def run_loop(self):
        """Main perceive-reason-act loop. Call from a thread."""
        self.get_logger().info("Navigation loop started (waiting for objective...)")

        last_vlm_time = 0.0

        while rclpy.ok():
            # Wait if paused
            if self.paused or not self.objective:
                time.sleep(0.5)
                continue

            # Wait for frame
            frame_b64 = self.get_frame_b64()
            if frame_b64 is None:
                self.get_logger().info("Waiting for camera frame...")
                time.sleep(1.0)
                continue

            self.get_logger().info("Got frame, preparing VLM call...")

            # Rate limit
            now = time.time()
            wait = config.MIN_VLM_INTERVAL - (now - last_vlm_time)
            if wait > 0:
                time.sleep(wait)

            # Record odom for stuck detection
            odom = self.get_odom()
            self.odom_history.append(odom)

            # Record position for revisit detection
            self.cycle_count += 1
            if self.cycle_count % config.REVISIT_RECORD_INTERVAL == 0:
                self.record_position()

            # Call VLM
            self.get_logger().info(f"Calling VLM API ({self.current_provider})...")
            last_vlm_time = time.time()
            cmd = self.call_vlm(frame_b64)
            elapsed = time.time() - last_vlm_time
            self.get_logger().info(f"VLM response in {elapsed:.1f}s: {cmd is not None}")

            if cmd is None:
                # Both providers failed — stop and wait
                self._publish_stop()
                if self.on_reasoning:
                    self.on_reasoning({
                        "reasoning": "VLM API unavailable — robot stopped",
                        "action": "stop",
                        "api_calls": self.api_call_count,
                        "provider": "none",
                    })
                time.sleep(5.0)
                continue

            # Update observation memory
            obs = cmd.get("observation", "")
            if obs:
                self.observations.append(obs)

            # Broadcast reasoning to dashboard
            if self.on_reasoning:
                self.on_reasoning({
                    "reasoning": cmd.get("reasoning", ""),
                    "observation": obs,
                    "action": cmd.get("action", "stop"),
                    "speed": cmd.get("speed", 0),
                    "turn_angle": cmd.get("turn_angle", 0),
                    "duration": cmd.get("duration", 1),
                    "objective_complete": cmd.get("objective_complete", False),
                    "status": "stuck" if self.check_stuck() else "normal",
                    "api_calls": self.api_call_count,
                    "provider": self.current_provider,
                    "odom": {"x": odom[0], "y": odom[1], "theta": odom[2]},
                })

            # Check objective complete
            if cmd.get("objective_complete", False):
                self._publish_stop()
                self.get_logger().info("Objective complete!")
                if self.on_reasoning:
                    self.on_reasoning({
                        "reasoning": "Objective achieved!",
                        "action": "stop",
                        "objective_complete": True,
                        "api_calls": self.api_call_count,
                        "provider": self.current_provider,
                    })
                self.paused = True
                continue

            # Execute command
            self.execute_command(cmd)


def main():
    import sys

    rclpy.init()
    brain = NavigationBrain()

    # Spin up hybrid modules if available and enabled
    executor = MultiThreadedExecutor()
    executor.add_node(brain)

    if _vlmap_available and VLMAP_ENABLED:
        brain.vlmap = VLMapBuilder()
        executor.add_node(brain.vlmap)
        brain.get_logger().info("VLMap spatial memory ENABLED")
    else:
        brain.get_logger().info("VLMap spatial memory disabled")

    if _sign_detector_available and SIGN_DETECTOR_ENABLED:
        brain.sign_detector = SignDetector()
        executor.add_node(brain.sign_detector)
        brain.get_logger().info("Sign detector ENABLED")
    else:
        brain.get_logger().info("Sign detector disabled")

    # Spin ROS2 in a thread
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # Auto-set objective for maze navigation (always active)
    brain.set_objective("Navigate the maze: follow directional arrow signs, detect all 4 ArUco markers, avoid obstacles, and reach the goal zone.")

    try:
        brain.run_loop()
    except KeyboardInterrupt:
        pass
    finally:
        brain.shutdown()
        brain.destroy_node()
        if brain.vlmap:
            brain.vlmap.destroy_node()
        if brain.sign_detector:
            brain.sign_detector.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
