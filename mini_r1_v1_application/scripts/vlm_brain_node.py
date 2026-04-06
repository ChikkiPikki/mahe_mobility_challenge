#!/usr/bin/env python3
"""VLM Navigation Brain — Supervisory intelligence with full tool context.

Subscribes to ALL sensor topics, builds rich context from VLMToolkit,
calls VLM API with current + past frames, logs everything.
Publishes behavioral commands to the navigator state machine.
"""

import base64
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
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray
from cv_bridge import CvBridge
from openai import OpenAI

import os, sys
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
import vlm_config as config
from vlm_tools import VLMToolkit


class NavigationBrain(Node):
    def __init__(self):
        super().__init__("navigation_brain")

        self.bridge = CvBridge()
        self.toolkit = VLMToolkit()

        # ── Subscribers (ALL sensor topics) ─────────────────────────────
        self.create_subscription(Image, config.CAMERA_TOPIC, self._camera_cb, 1)
        self.create_subscription(Odometry, config.ODOM_TOPIC, self._odom_cb, 1)
        self.create_subscription(LaserScan, '/r1_mini/lidar', self._lidar_cb, 10)
        self.create_subscription(OccupancyGrid, '/local_costmap/costmap', self._costmap_cb, 10)
        self.create_subscription(String, '/mini_r1/sign_detections', self._sign_cb, 10)
        self.create_subscription(String, '/mini_r1/navigator/status', self._nav_status_cb, 10)
        self.create_subscription(MarkerArray, '/mini_r1/mission_control/detected_objects', self._marker_cb, 10)

        # ── Publishers ──────────────────────────────────────────────────
        self.cmd_pub = self.create_publisher(Twist, config.CMD_VEL_TOPIC, 1)
        self.command_pub = self.create_publisher(String, '/vlm_brain/command', 10)
        self.status_pub = self.create_publisher(String, '/vlm_brain/status', 10)

        # ── Thread-safe state ───────────────────────────────────────────
        self._lock = threading.Lock()
        self._latest_frame: bytes | None = None
        self._frame_time: float = 0.0
        self._odom_x: float = 0.0
        self._odom_y: float = 0.0
        self._odom_theta: float = 0.0

        # ── Frame history (last 2 frames for temporal awareness) ────────
        self._frame_history: deque = deque(maxlen=2)

        # ── Decision history (last 5 decisions) ────────────────────────
        self._decision_history: deque = deque(maxlen=5)

        # ── Navigation state ───────────────────────────────────────────
        self.objective: str = ""
        self.paused: bool = True
        self.observations: deque = deque(maxlen=config.MAX_OBSERVATIONS)
        self.odom_history: deque = deque(maxlen=config.STUCK_CYCLE_COUNT)
        self.visited_positions: list = []
        self.cycle_count: int = 0
        self.api_call_count: int = 0
        self.current_provider: str = "nvidia"

        # ── VLM clients ────────────────────────────────────────────────
        self._local_client = None
        self._nvidia_client = None
        self._openrouter_client = None
        if config.LOCAL_VLM_ENABLED:
            self._local_client = OpenAI(
                base_url=config.LOCAL_VLM_BASE_URL,
                api_key=config.LOCAL_VLM_API_KEY)
            self.current_provider = "local"
        if config.NVIDIA_API_KEY:
            self._nvidia_client = OpenAI(
                base_url=config.NVIDIA_BASE_URL,
                api_key=config.NVIDIA_API_KEY)
        if config.OPENROUTER_API_KEY:
            self._openrouter_client = OpenAI(
                base_url=config.OPENROUTER_BASE_URL,
                api_key=config.OPENROUTER_API_KEY)

        # ── Load prompt template ───────────────────────────────────────
        with open(config.PROMPT_FILE) as f:
            self.prompt_template = f.read()

        # ── Dashboard callback ─────────────────────────────────────────
        self.on_reasoning: callable | None = None

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        self.get_logger().info("Navigation Brain initialized (with VLMToolkit)")

    # ═══════════════════════════════════════════════════════════════════
    #  ROS2 Callbacks — feed VLMToolkit
    # ═══════════════════════════════════════════════════════════════════

    def _camera_cb(self, msg: Image):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        cv_image = cv2.resize(cv_image, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
        _, jpeg_buf = cv2.imencode(".jpg", cv_image,
                                    [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
        with self._lock:
            self._latest_frame = jpeg_buf.tobytes()
            self._frame_time = time.time()

    def _odom_cb(self, msg: Odometry):
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny, cosy)
        with self._lock:
            self._odom_x = msg.pose.pose.position.x
            self._odom_y = msg.pose.pose.position.y
            self._odom_theta = math.degrees(yaw)
        self.toolkit.record_odom(self._odom_x, self._odom_y, yaw, time.time())

    def _lidar_cb(self, msg: LaserScan):
        self.toolkit.lidar_ranges = np.array(msg.ranges, dtype=np.float32)
        self.toolkit.lidar_angle_min = msg.angle_min
        self.toolkit.lidar_angle_increment = msg.angle_increment

    def _costmap_cb(self, msg: OccupancyGrid):
        self.toolkit.costmap_data = np.array(msg.data, dtype=np.int16)
        self.toolkit.costmap_width = msg.info.width
        self.toolkit.costmap_height = msg.info.height
        self.toolkit.costmap_resolution = msg.info.resolution
        self.toolkit.costmap_origin_x = msg.info.origin.position.x
        self.toolkit.costmap_origin_y = msg.info.origin.position.y

    def _sign_cb(self, msg: String):
        self.toolkit.record_sign(msg.data, time.time())

    def _nav_status_cb(self, msg: String):
        self.toolkit.navigator_status_json = msg.data

    def _marker_cb(self, msg: MarkerArray):
        for m in msg.markers:
            if m.ns == 'aruco':
                self.toolkit.aruco_ids.add(m.id)

    # ═══════════════════════════════════════════════════════════════════
    #  Frame + Odom Access
    # ═══════════════════════════════════════════════════════════════════

    def get_frame_b64(self) -> str | None:
        with self._lock:
            if self._latest_frame is None:
                return None
            return base64.b64encode(self._latest_frame).decode("utf-8")

    def _save_frame_to_history(self, frame_b64: str):
        self._frame_history.append(frame_b64)

    def _build_composite_frame(self, current_b64: str) -> str:
        """Stitch current + past frames into one labeled image.
        Layout: [PAST 2 | PAST 1 | CURRENT] side-by-side with labels."""
        def b64_to_cv(b64_str):
            data = base64.b64decode(b64_str)
            arr = np.frombuffer(data, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)

        current_img = b64_to_cv(current_b64)
        h, w = current_img.shape[:2]

        # Label current frame
        cv2.putText(current_img, "NOW", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        frames = []
        past = list(self._frame_history)
        for i, past_b64 in enumerate(past):
            img = b64_to_cv(past_b64)
            img = cv2.resize(img, (w, h))
            label = f"-{(len(past)-i)*3}s"
            cv2.putText(img, label, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            frames.append(img)

        frames.append(current_img)

        if len(frames) == 1:
            composite = frames[0]
        else:
            composite = np.hstack(frames)

        _, buf = cv2.imencode('.jpg', composite,
                              [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
        return base64.b64encode(buf).decode('utf-8')

    def get_odom(self):
        with self._lock:
            return self._odom_x, self._odom_y, self._odom_theta

    # ═══════════════════════════════════════════════════════════════════
    #  Build Rich Context from ALL Tools
    # ═══════════════════════════════════════════════════════════════════

    def _build_full_context(self) -> str:
        """Build rich, natural-language context from ALL 8 tools."""
        lines = []

        # ── LiDAR Summary ──
        lidar = self.toolkit.tool_get_lidar_summary()
        if "error" not in lidar:
            front = lidar.get("front_min_m", 99)
            left = lidar.get("left_min_m", 99)
            right = lidar.get("right_min_m", 99)
            nearest = lidar.get("nearest_obstacle_m", 99)

            if front < 0.6:
                lines.append(f"⚠️ WALL {front:.1f}m AHEAD — DO NOT go forward!")
            elif front < 1.5:
                lines.append(f"Wall approaching ahead ({front:.1f}m). Consider turning soon.")
            else:
                lines.append(f"Path ahead clear ({front:.1f}m).")

            if left > 1.5:
                lines.append(f"Left side OPEN ({left:.1f}m clearance).")
            else:
                lines.append(f"Left side: wall at {left:.1f}m.")

            if right > 1.5:
                lines.append(f"Right side OPEN ({right:.1f}m clearance).")
            else:
                lines.append(f"Right side: wall at {right:.1f}m.")
        else:
            lines.append("LiDAR: no data yet.")

        # ── Costmap Summary ──
        costmap = self.toolkit.tool_get_costmap_summary()
        if "error" not in costmap:
            if costmap.get("dead_end"):
                lines.append("⚠️ DEAD END detected! Must turn around.")
            dirs = []
            if costmap.get("forward_free"): dirs.append("forward")
            if costmap.get("left_free"): dirs.append("left")
            if costmap.get("right_free"): dirs.append("right")
            if dirs:
                lines.append(f"Costmap free directions: {', '.join(dirs)}")
            else:
                lines.append("Costmap: ALL directions blocked!")

        # ── Sign Detections (from FastSAM — AUTHORITATIVE) ──
        signs = self.toolkit.tool_get_sign_detections()
        if signs.get("count", 0) > 0:
            for s in signs["signs"]:
                lines.append(f"🔶 SIGN DETECTED: arrow pointing {s['direction']} ({s['age_s']:.0f}s ago) — from FastSAM detector, TRUST THIS.")
        else:
            lines.append("No signs detected by FastSAM. Do NOT invent signs from the image.")

        # ── ArUco Markers ──
        aruco = self.toolkit.tool_get_aruco_markers()
        if aruco["count"] > 0:
            lines.append(f"ArUco markers found: {aruco['detected']} ({aruco['count']}/4 total)")
        else:
            lines.append("No ArUco markers seen yet (goal: find all 4).")

        # ── Stuck Check ──
        stuck = self.toolkit.tool_check_stuck()
        if stuck.get("is_stuck"):
            lines.append(f"⚠️ STUCK! Only moved {stuck['displacement_m']:.3f}m in {stuck['time_window_s']:.0f}s. "
                         f"Stuck count: {stuck['consecutive_stuck_count']}. TRY DIFFERENT DIRECTION!")

        # ── Loop Check ──
        loop = self.toolkit.tool_check_loop()
        if loop.get("is_loop"):
            lines.append(f"⚠️ LOOP DETECTED! You were here {loop['time_since_visit_s']:.0f}s ago. "
                         f"Looped {loop['loop_count']} times. GO A DIFFERENT WAY!")

        # ── Navigator Status ──
        nav = self.toolkit.tool_get_navigator_status()
        if nav.get("state") != "UNKNOWN":
            lines.append(f"Navigator: state={nav.get('state')}, behavior={nav.get('behavior')}")

        # ── Position Trail (last 5) ──
        trail = self.toolkit.position_log[-5:]
        if len(trail) >= 3:
            trail_str = " → ".join(f"({x:.1f},{y:.1f})" for x, y, _ in trail)
            lines.append(f"Position trail: {trail_str}")

        # ── Decision History ──
        if self._decision_history:
            lines.append("\nYour recent decisions:")
            now = time.time()
            for i, d in enumerate(reversed(list(self._decision_history))):
                age = now - d.get("time", now)
                lines.append(f"  {i+1}. ({age:.0f}s ago) action={d['action']}, "
                             f"reason=\"{d.get('reasoning', '')[:60]}\"")

        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════════════
    #  VLM Call
    # ═══════════════════════════════════════════════════════════════════

    def call_vlm(self, frame_b64: str) -> dict | None:
        x, y, theta = self.get_odom()

        # Build status
        status = "normal"
        if self.toolkit.tool_check_stuck().get("is_stuck"):
            status = "⚠️ STUCK — robot not moving, try backward or turn"
        elif self.toolkit.tool_check_loop().get("is_loop"):
            status = "⚠️ LOOPING — revisiting same area, go a different direction"

        # Build observations text
        obs_text = "\n".join(
            f"  {i+1}. {obs}" for i, obs in enumerate(self.observations)
        ) or "  (none yet)"

        # Build FULL tool context
        tool_context = self._build_full_context()

        # Build prompt
        try:
            prompt = self.prompt_template.format(
                objective=self.objective,
                observations=obs_text,
                x=x, y=y, theta=theta,
                status=status,
                sign_detections="(see tool data below)",
                spatial_memory="(not available)",
            )
        except KeyError:
            prompt = self.prompt_template

        prompt += f"\n\n══ SENSOR & TOOL DATA (use this, not your image interpretation) ══\n{tool_context}"

        # ── Log full context ──
        self.get_logger().info(f"═══ VLM CONTEXT ({len(prompt)} chars) ═══")
        for line in tool_context.split("\n"):
            if line.strip():
                self.get_logger().info(f"  CTX: {line}")

        # Build single composite image: stitch current + past frames side-by-side
        # (NVIDIA NIM only allows 1 image per request)
        composite_b64 = self._build_composite_frame(frame_b64)
        self._save_frame_to_history(frame_b64)

        image_content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{composite_b64}"}},
        ]

        # ── Call providers ──
        providers = []
        if self._local_client:
            providers.append(("local", self._local_client, config.LOCAL_VLM_MODEL))
        if self._nvidia_client:
            providers.append(("nvidia", self._nvidia_client, config.NVIDIA_MODEL))
        if self._openrouter_client:
            providers.append(("openrouter", self._openrouter_client, config.OPENROUTER_MODEL))

        for provider_name, client, model in providers:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": image_content}],
                    max_tokens=300,
                    temperature=0.1,
                )
                raw = response.choices[0].message.content.strip()
                self.get_logger().info(f"[{provider_name}] RAW VLM: {raw[:300]}")
                parsed = self._parse_vlm_response(raw)
                if parsed:
                    self.api_call_count += 1
                    self.current_provider = provider_name
                    return parsed
                self.get_logger().warn(f"[{provider_name}] Failed to parse JSON")
            except Exception as e:
                self.get_logger().error(f"[{provider_name}] API error: {e}")
                continue

        self.get_logger().error("All VLM providers failed")
        return None

    def _parse_vlm_response(self, raw: str) -> dict | None:
        # Strategy 1: clean markdown fences
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
        try:
            data = json.loads(cleaned)
            if "action" in data:
                if data["action"] not in config.VALID_ACTIONS:
                    data["action"] = "stop"
                return data
        except json.JSONDecodeError:
            pass
        # Strategy 2: find JSON in text
        match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if "action" in data:
                    if data["action"] not in config.VALID_ACTIONS:
                        data["action"] = "stop"
                    return data
            except json.JSONDecodeError:
                pass
        return None

    # ═══════════════════════════════════════════════════════════════════
    #  Command Execution
    # ═══════════════════════════════════════════════════════════════════

    def execute_command(self, cmd: dict):
        action = cmd.get("action", "stop")

        behavior_map = {
            "left": "turn_left_90",
            "right": "turn_right_90",
            "backward": "reverse",
            "stop": "stop",
        }

        if action == "forward":
            behavior_name = "gap_follow (no override)"
            self.get_logger().info(f"VLM → forward (navigator continues)")
        elif action in behavior_map:
            behavior_name = behavior_map[action]
            nav_cmd = json.dumps({
                "action": "execute_behavior",
                "action_args": {"name": behavior_name},
            })
            msg = String()
            msg.data = nav_cmd
            self.command_pub.publish(msg)
            self.get_logger().info(f"VLM OVERRIDE → {behavior_name}")
        else:
            behavior_name = action

        # Record decision in history
        self._decision_history.append({
            "action": action,
            "behavior": behavior_name,
            "reasoning": cmd.get("reasoning", ""),
            "observation": cmd.get("observation", ""),
            "time": time.time(),
        })

        # Publish status
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
            f"VLM [{self.current_provider}] action={action} → {behavior_name} | "
            f"reason: {cmd.get('reasoning', '')[:80]}")

        # Wait for behavior
        duration = max(0.5, min(2.0, cmd.get("duration", 1.0)))
        elapsed = 0.0
        while elapsed < duration:
            if self.paused:
                return
            time.sleep(0.1)
            elapsed += 0.1

    def _publish_stop(self):
        self.cmd_pub.publish(Twist())

    # ═══════════════════════════════════════════════════════════════════
    #  Lifecycle
    # ═══════════════════════════════════════════════════════════════════

    def _signal_handler(self, signum, frame):
        self._publish_stop()
        self.paused = True

    def shutdown(self):
        self._publish_stop()
        self.paused = True

    def set_objective(self, objective):
        self.objective = objective
        self.paused = False
        self.observations.clear()
        self.visited_positions.clear()
        self.odom_history.clear()
        self._decision_history.clear()
        self._frame_history.clear()
        self.cycle_count = 0
        self.get_logger().info(f"New objective: {objective}")

    def stop_navigation(self):
        self.paused = True
        self._publish_stop()

    def run_loop(self):
        self.get_logger().info("Navigation loop started")
        last_vlm_time = 0.0

        while rclpy.ok():
            if self.paused or not self.objective:
                time.sleep(0.5)
                continue

            frame_b64 = self.get_frame_b64()
            if frame_b64 is None:
                self.get_logger().info("Waiting for camera frame...")
                time.sleep(1.0)
                continue

            # Rate limit
            now = time.time()
            wait = config.MIN_VLM_INTERVAL - (now - last_vlm_time)
            if wait > 0:
                time.sleep(wait)

            # Record odom
            odom = self.get_odom()
            self.odom_history.append(odom)
            self.cycle_count += 1
            if self.cycle_count % config.REVISIT_RECORD_INTERVAL == 0:
                self.toolkit.record_position(self._odom_x, self._odom_y, time.time())

            # Call VLM
            last_vlm_time = time.time()
            cmd = self.call_vlm(frame_b64)
            elapsed = time.time() - last_vlm_time

            if cmd is None:
                self._publish_stop()
                self.get_logger().warn(f"VLM failed ({elapsed:.1f}s). Waiting 5s.")
                time.sleep(5.0)
                continue

            self.get_logger().info(f"VLM response in {elapsed:.1f}s")

            # Record observation
            obs = cmd.get("observation", "")
            if obs:
                self.observations.append(obs)

            # Broadcast to dashboard
            if self.on_reasoning:
                self.on_reasoning({
                    "reasoning": cmd.get("reasoning", ""),
                    "observation": obs,
                    "action": cmd.get("action", "stop"),
                    "api_calls": self.api_call_count,
                    "provider": self.current_provider,
                    "odom": {"x": odom[0], "y": odom[1], "theta": odom[2]},
                })

            if cmd.get("objective_complete", False):
                self._publish_stop()
                self.get_logger().info("Objective complete!")
                self.paused = True
                continue

            self.execute_command(cmd)


def main():
    rclpy.init()
    brain = NavigationBrain()
    executor = MultiThreadedExecutor()
    executor.add_node(brain)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    brain.set_objective(
        "Navigate the maze: follow directional arrow signs, "
        "detect all 4 ArUco markers, avoid obstacles, reach the goal zone.")

    try:
        brain.run_loop()
    except KeyboardInterrupt:
        pass
    finally:
        brain.shutdown()
        brain.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
