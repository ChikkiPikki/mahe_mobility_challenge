#!/usr/bin/env python3
"""
Mission Zone Node
Publishes translucent start/goal cylinder markers in RViz and checks
whether the robot has entered the goal zone, publishing a mission status
message when it does. Goal checking uses absolute world coordinates.
"""
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String
import tf2_ros
from tf2_ros import TransformException


class MissionZoneNode(Node):
    def __init__(self):
        super().__init__('mission_zone_node')

        # World coordinates passed from launch
        self.spawn_x = self.declare_parameter('spawn_x', 0.0).value
        self.spawn_y = self.declare_parameter('spawn_y', 0.0).value
        self.goal_x = self.declare_parameter('goal_x', 0.0).value
        self.goal_y = self.declare_parameter('goal_y', 0.0).value
        self.goal_radius = 1.0  # metres

        # Odom-relative positions for RViz markers (odom origin = spawn)
        self.start_odom_x = 0.0
        self.start_odom_y = 0.0
        self.goal_odom_x = self.goal_x - self.spawn_x
        self.goal_odom_y = self.goal_y - self.spawn_y

        self.get_logger().info(
            f"Spawn world=({self.spawn_x:.2f}, {self.spawn_y:.2f})  "
            f"Goal world=({self.goal_x:.2f}, {self.goal_y:.2f})  "
            f"Goal odom=({self.goal_odom_x:.2f}, {self.goal_odom_y:.2f})")

        # TF for robot position
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Publishers
        self.viz_pub = self.create_publisher(
            MarkerArray, '/mini_r1/mission_control/viz_markers', 10)
        self.mission_status_pub = self.create_publisher(
            String, '/mini_r1/mission_control/mission_status', 10)

        self.mission_complete = False
        self.create_timer(1.0, self.timer_callback)
        self.get_logger().info("MissionZoneNode started.")

    def _make_cylinder(self, marker_id, x, y, r, g, b, ns, stamp):
        m = Marker()
        m.header.frame_id = "odom"
        m.header.stamp = stamp
        m.ns = ns
        m.id = marker_id
        m.type = Marker.CYLINDER
        m.action = Marker.ADD
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = 1.0  # center of 2m cylinder
        m.pose.orientation.w = 1.0
        m.scale.x = 2.0  # diameter = 2 * 1m radius
        m.scale.y = 2.0
        m.scale.z = 2.0  # height
        m.color.r = float(r)
        m.color.g = float(g)
        m.color.b = float(b)
        m.color.a = 0.6
        m.lifetime.sec = 0
        return m

    def _make_label(self, x, y, text, r, g, b, ns, stamp):
        m = Marker()
        m.header.frame_id = "odom"
        m.header.stamp = stamp
        m.ns = ns
        m.id = 0
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = 2.3
        m.pose.orientation.w = 1.0
        m.scale.z = 0.3
        m.color.r = float(r)
        m.color.g = float(g)
        m.color.b = float(b)
        m.color.a = 1.0
        m.text = text
        m.lifetime.sec = 0
        return m

    def timer_callback(self):
        stamp = self.get_clock().now().to_msg()
        viz = MarkerArray()

        # Start zone (orange)
        viz.markers.append(self._make_cylinder(
            0, self.start_odom_x, self.start_odom_y,
            1.0, 0.5, 0.0, "zone_start", stamp))
        viz.markers.append(self._make_label(
            self.start_odom_x, self.start_odom_y,
            "START", 1.0, 0.5, 0.0, "zone_start_text", stamp))

        # Goal zone (cyan)
        viz.markers.append(self._make_cylinder(
            0, self.goal_odom_x, self.goal_odom_y,
            0.0, 1.0, 1.0, "zone_goal", stamp))
        viz.markers.append(self._make_label(
            self.goal_odom_x, self.goal_odom_y,
            "GOAL", 0.0, 1.0, 1.0, "zone_goal_text", stamp))

        self.viz_pub.publish(viz)

        # Goal proximity check in absolute world coordinates
        if not self.mission_complete:
            try:
                trans = self.tf_buffer.lookup_transform(
                    "odom", "base_link", rclpy.time.Time())
                # Robot odom position → world position
                robot_world_x = trans.transform.translation.x + self.spawn_x
                robot_world_y = trans.transform.translation.y + self.spawn_y
                dx = robot_world_x - self.goal_x
                dy = robot_world_y - self.goal_y
                dist = (dx * dx + dy * dy) ** 0.5
                if dist <= self.goal_radius:
                    self.mission_complete = True
                    msg = String()
                    msg.data = "MISSION_COMPLETE"
                    self.mission_status_pub.publish(msg)
                    self.get_logger().info(
                        f"MISSION COMPLETE! Robot at world "
                        f"({robot_world_x:.2f}, {robot_world_y:.2f}), "
                        f"dist to goal={dist:.2f}m")
            except TransformException:
                pass


def main(args=None):
    rclpy.init(args=args)
    node = MissionZoneNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
