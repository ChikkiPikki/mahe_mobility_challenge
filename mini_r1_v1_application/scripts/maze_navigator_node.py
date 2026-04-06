#!/usr/bin/env python3
"""
Maze Navigator Node — config-driven autonomous navigation.
Reads behavior_config.yaml and executes a state machine that follows
corridors, obeys directional signs, and recovers from dead-ends/stuck/loops.
"""
import os
import sys
import math
import numpy as np

# Ensure nav_lib is importable (installed alongside this script)
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray
import tf2_ros
from tf2_ros import TransformException

from nav_lib.sensor_state import SensorState
from nav_lib.config_loader import load_config, build_detectors, build_behaviors
from nav_lib.state_machine import ConfigStateMachine


class MazeNavigatorNode(Node):
    def __init__(self):
        super().__init__('maze_navigator_node')

        # Load behavior config
        config_path = self.declare_parameter(
            'behavior_config', '').value
        if not config_path or not os.path.exists(config_path):
            from ament_index_python.packages import get_package_share_directory
            pkg = get_package_share_directory('mini_r1_v1_application')
            config_path = os.path.join(pkg, 'config', 'behavior_config.yaml')

        self.get_logger().info(f"Loading behavior config: {config_path}")
        self.cfg = load_config(config_path)
        self.thresholds = self.cfg.get('thresholds', {})

        # Build detectors and behaviors from config
        detectors = build_detectors(self.cfg)
        behaviors = build_behaviors(self.cfg)
        self.sm = ConfigStateMachine(self.cfg, detectors, behaviors, self.get_logger())
        self.ss = SensorState()

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscribers
        self.create_subscription(
            OccupancyGrid, '/local_costmap/costmap',
            self.costmap_cb, 10)
        self.create_subscription(
            String, '/mini_r1/sign_detections',
            self.sign_cb, 10)
        self.create_subscription(
            String, '/mini_r1/mission_control/mission_status',
            self.mission_cb, 10)
        self.create_subscription(
            LaserScan, '/r1_mini/lidar',
            self.lidar_cb, 10)
        self.create_subscription(
            MarkerArray, '/mini_r1/mission_control/detected_objects',
            self.marker_cb, 10)

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/mini_r1/navigator/status', 10)

        # Timer for main loop
        rate_hz = self.thresholds.get('command_rate_hz', 20)
        self.create_timer(1.0 / rate_hz, self.tick)

        # Position logging timer
        log_interval = self.thresholds.get('position_log_interval_s', 2.0)
        self.create_timer(log_interval, self.log_position)

        # Status publishing timer (2 Hz)
        self.create_timer(0.5, self.publish_status)

        self.get_logger().info(
            f"MazeNavigator started. States: {list(self.cfg['states'].keys())}, "
            f"Behaviors: {list(self.cfg['behaviors'].keys())}, "
            f"Detectors: {list(self.cfg['detectors'].keys())}")

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    # ── Callbacks ────────────────────────────────────────────────────────
    def costmap_cb(self, msg: OccupancyGrid):
        self.ss.costmap_data = np.array(msg.data, dtype=np.int16)  # int8 wraps >127
        self.ss.costmap_width = msg.info.width
        self.ss.costmap_height = msg.info.height
        self.ss.costmap_resolution = msg.info.resolution
        self.ss.costmap_origin_x = msg.info.origin.position.x
        self.ss.costmap_origin_y = msg.info.origin.position.y
        self.ss.costmap_stamp = self._now_sec()

    def sign_cb(self, msg: String):
        self.ss.last_sign = msg.data
        self.ss.sign_stamp = self._now_sec()

    def mission_cb(self, msg: String):
        if msg.data == "MISSION_COMPLETE":
            self.ss.mission_complete = True

    def lidar_cb(self, msg: LaserScan):
        self.ss.lidar_ranges = np.array(msg.ranges, dtype=np.float32)
        self.ss.lidar_angle_min = msg.angle_min
        self.ss.lidar_angle_increment = msg.angle_increment
        self.ss.lidar_stamp = self._now_sec()

    def marker_cb(self, msg: MarkerArray):
        for m in msg.markers:
            if m.ns == 'aruco':
                self.sm.arucos_seen.add(m.id)

    # ── Update robot pose from TF ───────────────────────────────────────
    def update_odom(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                'odom', 'base_link', rclpy.time.Time())
            self.ss.x = trans.transform.translation.x
            self.ss.y = trans.transform.translation.y
            q = trans.transform.rotation
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self.ss.yaw = math.atan2(siny, cosy)
            self.ss.odom_stamp = self._now_sec()
        except TransformException:
            pass

    def log_position(self):
        if self.ss.odom_stamp > 0:
            max_hist = self.thresholds.get('loop_history_size', 200)
            self.ss.position_history.append(
                (self.ss.x, self.ss.y, self.ss.odom_stamp))
            if len(self.ss.position_history) > max_hist:
                self.ss.position_history = self.ss.position_history[-max_hist:]

    # ── Main tick ────────────────────────────────────────────────────────
    def tick(self):
        self.update_odom()
        now = self._now_sec()

        # Safety: don't drive if we've never received odom
        if self.ss.odom_stamp == 0:
            self.cmd_pub.publish(Twist())
            return

        # Run state machine
        twist = self.sm.tick(self.ss)
        self.cmd_pub.publish(twist)

    # ── Status publishing ────────────────────────────────────────────────
    def publish_status(self):
        msg = String()
        msg.data = self.sm.get_status_json(self.ss)
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MazeNavigatorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
