"""
VLM Tool implementations — callable by the VLM brain.
Each tool reads from cached ROS2 data (instant, no blocking).
"""
import math
import json
import numpy as np


class VLMToolkit:
    """Registry of tools the VLM can call. All methods return JSON-serializable dicts."""

    def __init__(self):
        # Cached sensor data — updated by vlm_brain_node subscribers
        self.lidar_ranges = None
        self.lidar_angle_min = 0.0
        self.lidar_angle_increment = 0.0
        self.costmap_data = None
        self.costmap_width = 0
        self.costmap_height = 0
        self.costmap_resolution = 0.05
        self.costmap_origin_x = 0.0
        self.costmap_origin_y = 0.0
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.pose_yaw = 0.0
        self.last_sign = ""
        self.sign_stamp = 0.0
        self.sign_history = []       # [{direction, stamp}]
        self.aruco_ids = set()
        self.navigator_status_json = ""
        self.mission_complete = False

        # Odom history for stuck/loop detection
        self.odom_history = []       # [(x, y, yaw, stamp)]
        self.position_log = []       # [(x, y, stamp)] for loop detection
        self.stuck_count = 0

        # Available behaviors/recoveries (loaded from config)
        self.available_behaviors = []
        self.available_recoveries = []
        self.available_speed_profiles = []

    # ── Tool registry ───────────────────────────────────────────────────
    TOOL_DESCRIPTIONS = {
        "get_position": "Returns robot position {x, y, yaw_deg} in odom frame",
        "get_lidar_summary": "Returns min ranges in front/left/right/back quadrants and nearest obstacle",
        "get_costmap_summary": "Returns free/blocked status in each direction and corridor width",
        "get_sign_detections": "Returns recently detected arrow signs with directions and ages",
        "get_aruco_markers": "Returns list of detected ArUco marker IDs (goal: find all 4)",
        "check_stuck": "Returns whether robot hasn't moved recently",
        "check_loop": "Returns whether robot is revisiting a previous location",
        "get_navigator_status": "Returns current state machine state, behavior, and detector status",
    }

    def get_tool_descriptions_text(self) -> str:
        lines = []
        for name, desc in self.TOOL_DESCRIPTIONS.items():
            lines.append(f"- {name}: {desc}")
        return "\n".join(lines)

    def execute_tool(self, name: str) -> dict:
        method = getattr(self, f"tool_{name}", None)
        if method is None:
            return {"error": f"Unknown tool: {name}"}
        try:
            return method()
        except Exception as e:
            return {"error": str(e)}

    def execute_tools(self, names: list) -> dict:
        results = {}
        for name in names:
            results[name] = self.execute_tool(name)
        return results

    # ── Perception Tools ────────────────────────────────────────────────

    def tool_get_position(self) -> dict:
        return {
            "x": round(self.pose_x, 2),
            "y": round(self.pose_y, 2),
            "yaw_deg": round(math.degrees(self.pose_yaw), 1),
        }

    def tool_get_lidar_summary(self) -> dict:
        if self.lidar_ranges is None or len(self.lidar_ranges) == 0:
            return {"error": "No LiDAR data"}

        ranges = np.array(self.lidar_ranges, dtype=np.float32)
        n = len(ranges)
        angle_min = self.lidar_angle_min
        angle_inc = self.lidar_angle_increment
        if angle_inc == 0:
            return {"error": "Invalid LiDAR config"}

        # Compute angle for each ray
        angles = angle_min + np.arange(n) * angle_inc

        def min_in_arc(center_rad, half_width_rad):
            diff = np.abs(np.arctan2(np.sin(angles - center_rad),
                                      np.cos(angles - center_rad)))
            mask = (diff < half_width_rad) & np.isfinite(ranges) & (ranges > 0.12)
            if not np.any(mask):
                return 30.0
            return float(np.min(ranges[mask]))

        def mean_in_arc(center_rad, half_width_rad):
            diff = np.abs(np.arctan2(np.sin(angles - center_rad),
                                      np.cos(angles - center_rad)))
            mask = (diff < half_width_rad) & np.isfinite(ranges) & (ranges > 0.12)
            if not np.any(mask):
                return 30.0
            return float(np.mean(ranges[mask]))

        valid = ranges[np.isfinite(ranges) & (ranges > 0.12)]
        nearest_idx = int(np.argmin(valid)) if len(valid) > 0 else 0
        nearest_m = float(valid[nearest_idx]) if len(valid) > 0 else 30.0

        return {
            "front_min_m": round(min_in_arc(0.0, math.radians(30)), 2),
            "left_min_m": round(min_in_arc(math.pi / 2, math.radians(30)), 2),
            "right_min_m": round(min_in_arc(-math.pi / 2, math.radians(30)), 2),
            "back_min_m": round(min_in_arc(math.pi, math.radians(30)), 2),
            "front_clear_m": round(mean_in_arc(0.0, math.radians(10)), 2),
            "nearest_obstacle_m": round(nearest_m, 2),
        }

    def tool_get_costmap_summary(self) -> dict:
        if self.costmap_data is None:
            return {"error": "No costmap data"}

        res = self.costmap_resolution

        def check_arc_free(center_rad, half_deg, check_dist_m):
            half_rad = math.radians(half_deg)
            total = 0
            free = 0
            for a in np.linspace(center_rad - half_rad, center_rad + half_rad, 15):
                for d in np.arange(0.2, check_dist_m, res * 2):
                    wx = self.pose_x + d * math.cos(self.pose_yaw + a)
                    wy = self.pose_y + d * math.sin(self.pose_yaw + a)
                    gx = int((wx - self.costmap_origin_x) / res)
                    gy = int((wy - self.costmap_origin_y) / res)
                    if 0 <= gx < self.costmap_width and 0 <= gy < self.costmap_height:
                        cost = int(self.costmap_data[gy * self.costmap_width + gx])
                        total += 1
                        if cost < 65:
                            free += 1
                    else:
                        total += 1
                        free += 1  # out-of-bounds = unexplored = free
            return (free / total) > 0.5 if total > 0 else True

        left_min = self.tool_get_lidar_summary().get("left_min_m", 5.0)
        right_min = self.tool_get_lidar_summary().get("right_min_m", 5.0)

        return {
            "forward_free": check_arc_free(0.0, 30, 0.6),
            "left_free": check_arc_free(math.pi / 2, 30, 0.8),
            "right_free": check_arc_free(-math.pi / 2, 30, 0.8),
            "corridor_width_m": round(left_min + right_min, 2),
            "dead_end": not check_arc_free(0.0, 60, 0.5),
        }

    def tool_get_sign_detections(self) -> dict:
        now = max(self.sign_stamp, 0.001)
        signs = []
        for s in self.sign_history[-5:]:
            age = now - s.get("stamp", 0)
            signs.append({
                "direction": s.get("direction", "?"),
                "age_s": round(age, 1),
            })
        return {"signs": signs, "count": len(signs)}

    def tool_get_aruco_markers(self) -> dict:
        return {
            "detected": sorted(self.aruco_ids),
            "total_in_maze": 4,
            "count": len(self.aruco_ids),
        }

    def tool_check_stuck(self) -> dict:
        if len(self.odom_history) < 3:
            return {"is_stuck": False, "displacement_m": 999, "time_window_s": 0}

        recent = self.odom_history[-1]
        oldest = self.odom_history[0]
        dx = recent[0] - oldest[0]
        dy = recent[1] - oldest[1]
        displacement = math.sqrt(dx * dx + dy * dy)
        window = recent[3] - oldest[3]

        is_stuck = displacement < 0.05 and window > 3.0
        if is_stuck:
            self.stuck_count += 1
        else:
            self.stuck_count = 0

        return {
            "is_stuck": is_stuck,
            "displacement_m": round(displacement, 3),
            "time_window_s": round(window, 1),
            "consecutive_stuck_count": self.stuck_count,
        }

    def tool_check_loop(self) -> dict:
        if len(self.position_log) < 5:
            return {"is_loop": False, "revisit_distance_m": 999, "loop_count": 0}

        cx, cy = self.pose_x, self.pose_y
        now = self.position_log[-1][2] if self.position_log else 0

        min_dist = 999.0
        min_time = 0.0
        loop_count = 0
        for (hx, hy, ht) in self.position_log[:-5]:
            dx = cx - hx
            dy = cy - hy
            dist = math.sqrt(dx * dx + dy * dy)
            gap = now - ht
            if dist < 0.5 and gap > 20.0:
                loop_count += 1
                if dist < min_dist:
                    min_dist = dist
                    min_time = gap

        return {
            "is_loop": loop_count > 0,
            "revisit_distance_m": round(min_dist, 2),
            "time_since_visit_s": round(min_time, 1),
            "loop_count": loop_count,
        }

    def tool_get_navigator_status(self) -> dict:
        if self.navigator_status_json:
            try:
                return json.loads(self.navigator_status_json)
            except json.JSONDecodeError:
                pass
        return {"state": "UNKNOWN", "behavior": "unknown"}

    # ── Helpers ─────────────────────────────────────────────────────────

    def record_odom(self, x, y, yaw, stamp):
        self.pose_x = x
        self.pose_y = y
        self.pose_yaw = yaw
        self.odom_history.append((x, y, yaw, stamp))
        if len(self.odom_history) > 60:
            self.odom_history = self.odom_history[-60:]

    def record_position(self, x, y, stamp):
        self.position_log.append((x, y, stamp))
        if len(self.position_log) > 200:
            self.position_log = self.position_log[-200:]

    def record_sign(self, direction, stamp):
        self.sign_history.append({"direction": direction, "stamp": stamp})
        if len(self.sign_history) > 20:
            self.sign_history = self.sign_history[-20:]
        self.last_sign = direction
        self.sign_stamp = stamp
