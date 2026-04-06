#!/usr/bin/env python3
"""
VLM Brain Node — supervisory intelligence for maze navigation.
Calls a VLM (Vision Language Model) every ~3s with the camera frame + tool context.
Publishes high-level behavioral commands to the navigator state machine.
"""
import os
import sys
import math
import json
import base64
import re
import threading

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray
import tf2_ros
from tf2_ros import TransformException
from cv_bridge import CvBridge

# Ensure local imports work
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from vlm_tools import VLMToolkit
import vlm_config as cfg

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class VLMBrainNode(Node):
    def __init__(self):
        super().__init__('vlm_brain_node')

        self.bridge = CvBridge()
        self.toolkit = VLMToolkit()
        self._lock = threading.Lock()

        # Frame storage
        self._latest_frame_b64 = None
        self._frame_stamp = 0.0

        # Observation memory
        self._observations = []
        self._objective = "Navigate the maze: follow directional arrow signs, detect all 4 ArUco markers, and reach the goal zone."

        # Pending tool results from previous cycle
        self._pending_tool_results = {}

        # VLM provider tracking
        self._current_provider = "none"

        # Load prompt template
        prompt_path = cfg.PROMPT_FILE
        if os.path.exists(prompt_path):
            with open(prompt_path) as f:
                self._prompt_template = f.read()
            self.get_logger().info(f"Loaded prompt from {prompt_path}")
        else:
            self._prompt_template = "You are a robot navigation controller. Output JSON: {{\"action\":\"continue\",\"reasoning\":\"no prompt loaded\"}}"
            self.get_logger().warn(f"Prompt file not found: {prompt_path}")

        # Build VLM clients (fallback chain: local → NVIDIA → OpenRouter)
        self._clients = []
        if OpenAI is None:
            self.get_logger().error("openai package not installed. pip install openai")
        else:
            if cfg.LOCAL_VLM_ENABLED:
                self._clients.append(("local", OpenAI(
                    base_url=cfg.LOCAL_VLM_BASE_URL,
                    api_key=cfg.LOCAL_VLM_API_KEY), cfg.LOCAL_VLM_MODEL))
            if cfg.NVIDIA_API_KEY:
                self._clients.append(("nvidia", OpenAI(
                    base_url=cfg.NVIDIA_BASE_URL,
                    api_key=cfg.NVIDIA_API_KEY), cfg.NVIDIA_MODEL))
            if cfg.OPENROUTER_API_KEY:
                self._clients.append(("openrouter", OpenAI(
                    base_url=cfg.OPENROUTER_BASE_URL,
                    api_key=cfg.OPENROUTER_API_KEY), cfg.OPENROUTER_MODEL))

        if not self._clients:
            self.get_logger().warn("No VLM providers configured. Brain will be passive.")

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscribers
        self.create_subscription(Image, '/r1_mini/camera/image_raw', self._camera_cb, 10)
        self.create_subscription(LaserScan, '/r1_mini/lidar', self._lidar_cb, 10)
        self.create_subscription(OccupancyGrid, '/local_costmap/costmap', self._costmap_cb, 10)
        self.create_subscription(String, '/mini_r1/sign_detections', self._sign_cb, 10)
        self.create_subscription(String, '/mini_r1/navigator/status', self._nav_status_cb, 10)
        self.create_subscription(String, '/mini_r1/mission_control/mission_status', self._mission_cb, 10)
        self.create_subscription(MarkerArray, '/mini_r1/mission_control/detected_objects', self._marker_cb, 10)

        # Publishers
        self.command_pub = self.create_publisher(String, '/vlm_brain/command', 10)
        self.status_pub = self.create_publisher(String, '/vlm_brain/status', 10)

        # Timer: VLM cycle
        self.create_timer(cfg.VLM_CYCLE_INTERVAL_S, self._vlm_cycle)

        # Position logging timer (2s)
        self.create_timer(2.0, self._log_position)

        # Odom recording timer (0.5s)
        self.create_timer(0.5, self._record_odom)

        self.get_logger().info(
            f"VLM Brain started. Providers: {[c[0] for c in self._clients]}. "
            f"Cycle interval: {cfg.VLM_CYCLE_INTERVAL_S}s")

    def _now(self):
        return self.get_clock().now().nanoseconds / 1e9

    # ── Subscribers ─────────────────────────────────────────────────────

    def _camera_cb(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            cv_img = cv2.resize(cv_img, (cfg.FRAME_WIDTH, cfg.FRAME_HEIGHT))
            _, buf = cv2.imencode('.jpg', cv_img,
                                  [cv2.IMWRITE_JPEG_QUALITY, cfg.JPEG_QUALITY])
            with self._lock:
                self._latest_frame_b64 = base64.b64encode(buf).decode('utf-8')
                self._frame_stamp = self._now()
        except Exception as e:
            self.get_logger().error(f"Camera convert error: {e}", throttle_duration_sec=5.0)

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
        now = self._now()
        self.toolkit.record_sign(msg.data, now)

    def _nav_status_cb(self, msg: String):
        self.toolkit.navigator_status_json = msg.data

    def _mission_cb(self, msg: String):
        if msg.data == "MISSION_COMPLETE":
            self.toolkit.mission_complete = True

    def _marker_cb(self, msg: MarkerArray):
        for m in msg.markers:
            if m.ns == 'aruco':
                self.toolkit.aruco_ids.add(m.id)

    def _record_odom(self):
        try:
            trans = self.tf_buffer.lookup_transform('odom', 'base_link', rclpy.time.Time())
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            q = trans.transform.rotation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                             1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            self.toolkit.record_odom(x, y, yaw, self._now())
        except TransformException:
            pass

    def _log_position(self):
        if self.toolkit.pose_x != 0.0 or self.toolkit.pose_y != 0.0:
            self.toolkit.record_position(
                self.toolkit.pose_x, self.toolkit.pose_y, self._now())

    # ── VLM Cycle ───────────────────────────────────────────────────────

    def _vlm_cycle(self):
        """Main VLM decision cycle — called every ~3 seconds."""
        if self.toolkit.mission_complete:
            return

        with self._lock:
            frame_b64 = self._latest_frame_b64

        if frame_b64 is None:
            return  # no frame yet

        if not self._clients:
            return  # no VLM configured

        # Build prompt
        prompt = self._build_prompt()

        # Call VLM
        response = self._call_vlm(frame_b64, prompt)
        if response is None:
            self._publish_status("VLM call failed", {})
            return

        # Parse response
        cmd = self._parse_response(response)
        if cmd is None:
            self._publish_status("VLM parse failed", {"raw": response[:200]})
            return

        # Record observation
        obs = cmd.get('observation', '')
        if obs:
            self._observations.append(obs)
            if len(self._observations) > cfg.MAX_OBSERVATIONS:
                self._observations = self._observations[-cfg.MAX_OBSERVATIONS:]

        # Execute action
        action = cmd.get('action', 'continue')
        if action in cfg.VALID_ACTIONS and action != 'continue':
            msg = String()
            msg.data = json.dumps({
                'action': action,
                'action_args': cmd.get('action_args', {}),
            })
            self.command_pub.publish(msg)

        # Execute requested tool calls for next cycle
        tool_calls = cmd.get('tool_calls', [])
        if isinstance(tool_calls, list) and tool_calls:
            self._pending_tool_results = self.toolkit.execute_tools(tool_calls)
        else:
            self._pending_tool_results = {}

        # Publish status
        self._publish_status(cmd.get('reasoning', ''), cmd)

        self.get_logger().info(
            f"VLM [{self._current_provider}]: action={action} "
            f"reason={cmd.get('reasoning', '')[:60]}",
            throttle_duration_sec=1.0)

    def _build_prompt(self) -> str:
        # Get navigator status
        nav = {}
        try:
            nav = json.loads(self.toolkit.navigator_status_json)
        except (json.JSONDecodeError, TypeError):
            nav = {"state": "?", "behavior": "?"}

        # Format tool results from previous cycle
        tool_results_text = ""
        if self._pending_tool_results:
            lines = ["Tool results from previous cycle:"]
            for name, result in self._pending_tool_results.items():
                lines.append(f"  {name}: {json.dumps(result)}")
            tool_results_text = "\n".join(lines)

        obs_text = "\n".join(f"  - {o}" for o in self._observations[-5:]) if self._observations else "  (none yet)"

        return self._prompt_template.format(
            tool_descriptions=self.toolkit.get_tool_descriptions_text(),
            objective=self._objective,
            x=self.toolkit.pose_x,
            y=self.toolkit.pose_y,
            theta=math.degrees(self.toolkit.pose_yaw),
            nav_state=nav.get('state', '?'),
            nav_behavior=nav.get('behavior', '?'),
            observations=obs_text,
            tool_results=tool_results_text,
        )

    def _call_vlm(self, frame_b64: str, prompt: str) -> str | None:
        """Call VLM API with fallback chain. Returns raw text or None."""
        for provider_name, client, model in self._clients:
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/jpeg;base64,{frame_b64}"
                            }}
                        ]
                    }],
                    max_tokens=cfg.VLM_MAX_TOKENS,
                    temperature=cfg.VLM_TEMPERATURE,
                    timeout=cfg.VLM_TIMEOUT_S,
                )
                text = resp.choices[0].message.content.strip()
                self._current_provider = provider_name
                return text
            except Exception as e:
                self.get_logger().warn(
                    f"VLM {provider_name} failed: {e}",
                    throttle_duration_sec=10.0)
                continue
        return None

    def _parse_response(self, raw: str) -> dict | None:
        """Parse VLM JSON response with multi-strategy fallback."""
        # Strategy 1: direct JSON parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Strategy 2: strip markdown fences
        cleaned = re.sub(r'```(?:json)?\s*', '', raw)
        cleaned = re.sub(r'```\s*$', '', cleaned).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Strategy 3: find JSON object in text
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        self.get_logger().warn(f"Failed to parse VLM response: {raw[:100]}")
        return None

    def _publish_status(self, reasoning: str, cmd: dict):
        status = {
            "provider": self._current_provider,
            "reasoning": reasoning,
            "action": cmd.get("action", "none"),
            "action_args": cmd.get("action_args", {}),
            "observation": cmd.get("observation", ""),
            "tool_calls": cmd.get("tool_calls", []),
            "tool_results": {k: str(v)[:100] for k, v in self._pending_tool_results.items()},
            "objective_complete": cmd.get("objective_complete", False),
            "position": self.toolkit.tool_get_position(),
            "arucos": sorted(self.toolkit.aruco_ids),
        }
        msg = String()
        msg.data = json.dumps(status)
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = VLMBrainNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
