#!/usr/bin/env python3
"""
ArUco Marker & Sign Detector Node
Subscribes to RGB-D camera, detects ArUco markers, computes 3D position,
and publishes MarkerArray for RViz visualization + Mission Control Panel.
"""
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
import message_filters
import tf2_ros
from tf2_ros import TransformException
from scipy.spatial.transform import Rotation as ScipyRotation


class MarkerDetectorNode(Node):
    def __init__(self):
        super().__init__('marker_detector_node')

        self.bridge = CvBridge()
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

        # Stricter detection parameters to reduce false positives
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        self.aruco_params.adaptiveThreshConstant = 7
        self.aruco_params.minMarkerPerimeterRate = 0.03
        self.aruco_params.maxMarkerPerimeterRate = 4.0
        self.aruco_params.polygonalApproxAccuracyRate = 0.05
        self.aruco_params.minCornerDistanceRate = 0.05
        self.aruco_params.minDistanceToBorder = 3
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.aruco_params.cornerRefinementWinSize = 5
        self.aruco_params.errorCorrectionRate = 0.6  # stricter error correction

        # TF2 Setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Spawn offset (world coords of the odom origin)
        self.spawn_x = self.declare_parameter('spawn_x', 0.0).value
        self.spawn_y = self.declare_parameter('spawn_y', 0.0).value
        self.get_logger().info(f"Spawn offset: ({self.spawn_x:.2f}, {self.spawn_y:.2f})")

        # Depth filtering
        self.min_depth = 0.1   # metres
        self.max_depth = self.declare_parameter('max_marker_dist', 3.5).value  # metres
        self.min_marker_area_px = 200  # minimum pixel area to accept a detection

        # Message Filters Synchronizer
        self.sub_rgb = message_filters.Subscriber(
            self, Image, '/r1_mini/camera/image_raw', qos_profile=10)
        self.sub_depth = message_filters.Subscriber(
            self, Image, '/r1_mini/camera/depth_image', qos_profile=10)
        self.sub_info = message_filters.Subscriber(
            self, CameraInfo, '/r1_mini/camera/camera_info', qos_profile=10)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_rgb, self.sub_depth, self.sub_info],
            queue_size=10, slop=0.5)
        self.ts.registerCallback(self.sync_callback)

        # Publisher for Mission Control Panel (world coordinates)
        self.marker_pub = self.create_publisher(
            MarkerArray, '/mini_r1/mission_control/detected_objects', 10)
        # Publisher for RViz 3D visualization (odom-relative coordinates)
        self.viz_pub = self.create_publisher(
            MarkerArray, '/mini_r1/mission_control/viz_markers', 10)

        self.locked_panel_markers = {}
        self.locked_viz_markers = {}

        self.get_logger().info("MarkerDetectorNode initialized. Waiting for RGB-D streams...")

    def sync_callback(self, msg_rgb, msg_depth, msg_info):
        try:
            cv_rgb = self.bridge.imgmsg_to_cv2(msg_rgb, desired_encoding="passthrough")
            cv_depth = self.bridge.imgmsg_to_cv2(msg_depth, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")
            return

        if len(cv_rgb.shape) == 3:
            cv_gray = cv2.cvtColor(cv_rgb, cv2.COLOR_RGB2GRAY)
        else:
            cv_gray = cv_rgb

        fx = msg_info.k[0]
        cx = msg_info.k[2]
        fy = msg_info.k[4]
        cy = msg_info.k[5]

        if fx == 0.0 or fy == 0.0:
            return

        camera_frame = msg_rgb.header.frame_id

        try:
            trans = self.tf_buffer.lookup_transform(
                "odom", camera_frame, rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().warn(
                f"TF lookup failed: {ex}", throttle_duration_sec=5.0)
            return

        q = trans.transform.rotation
        t = trans.transform.translation
        x, y, z, w = q.x, q.y, q.z, q.w
        rot = np.array([
            [1-2*y*y-2*z*z,  2*x*y-2*z*w,    2*x*z+2*y*w],
            [2*x*y+2*z*w,    1-2*x*x-2*z*z,  2*y*z-2*x*w],
            [2*x*z-2*y*w,    2*y*z+2*x*w,    1-2*x*x-2*y*y]
        ])
        t_vec_tf = np.array([t.x, t.y, t.z])

        panel_markers = MarkerArray()  # world coords for panel
        viz_markers = MarkerArray()    # odom coords for RViz 3D
        self.detect_aruco(cv_gray, cv_depth, fx, fy, cx, cy, rot, t_vec_tf, panel_markers, viz_markers)
        self.detect_signs(cv_rgb, cv_depth, fx, fy, cx, cy, rot, t_vec_tf, panel_markers, viz_markers)

        if panel_markers.markers:
            self.marker_pub.publish(panel_markers)
        if viz_markers.markers:
            self.viz_pub.publish(viz_markers)

    def detect_signs(self, cv_rgb, cv_depth, fx, fy, cx, cy, rot, t_vec, panel_markers, viz_markers):
        """Stub for sign classification/detection"""
        pass

    def deproject_pixel(self, u, v, depth, fx, fy, cx, cy):
        """Deproject pixel to 3D in camera LINK frame (X-fwd, Y-left, Z-up)."""
        opt_x = (u - cx) * depth / fx
        opt_y = (v - cy) * depth / fy
        opt_z = depth
        return np.array([opt_z, -opt_x, -opt_y])

    def _make_marker(self, marker_id, stamp, x, y, z, r, g, b, qx=0.0, qy=0.0, qz=0.0, qw=1.0, lifetime=5, mtype=Marker.CUBE, text="", ns="aruco"):
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = stamp
        marker.ns = ns
        marker.id = marker_id
        marker.type = mtype
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.pose.orientation.x = qx
        marker.pose.orientation.y = qy
        marker.pose.orientation.z = qz
        marker.pose.orientation.w = qw
        
        if mtype == Marker.CUBE:
            # 0.5m box to exactly match the Gazebo physical object
            marker.scale.x = 0.5
            marker.scale.y = 0.5
            marker.scale.z = 0.5
        elif mtype == Marker.TEXT_VIEW_FACING:
            marker.scale.z = 0.2  # Text height
            marker.text = text
            
        marker.color.a = 0.8
        marker.color.r = float(r)
        marker.color.g = float(g)
        marker.color.b = float(b)
        marker.lifetime.sec = lifetime
        return marker

    def detect_aruco(self, cv_gray, cv_depth, fx, fy, cx, cy, rot, t_vec, panel_markers, viz_markers):
        corners, ids, _ = cv2.aruco.detectMarkers(
            cv_gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is None:
            return

        stamp = self.get_clock().now().to_msg()
        
        camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        # Size of the marker in meters (simulated marker is 0.5m side length)
        marker_length = 0.5
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_length, camera_matrix, dist_coeffs)

        if rvecs is None or tvecs is None:
            return

        for i, marker_id_arr in enumerate(ids):
            marker_id = int(marker_id_arr[0])

            # Only accept marker IDs in the expected range (e.g. max 10 markers)
            if marker_id < 0 or marker_id > 10:
                continue

            # Lock the marker to its very first observed pose. Skip re-calculation.
            if marker_id in self.locked_viz_markers:
                continue

            c = corners[i][0]
            area = cv2.contourArea(c)
            if area < self.min_marker_area_px:
                continue

            # Extract center from corners
            center_u = int(c[:, 0].mean())
            center_v = int(c[:, 1].mean())

            h_depth, w_depth = cv_depth.shape[:2]
            if center_u < 0 or center_u >= w_depth or center_v < 0 or center_v >= h_depth:
                continue

            # Extract accurate depth from the depth camera patch
            patch_r = 3
            u_lo, u_hi = max(0, center_u-patch_r), min(w_depth, center_u+patch_r+1)
            v_lo, v_hi = max(0, center_v-patch_r), min(h_depth, center_v+patch_r+1)
            depth_patch = cv_depth[v_lo:v_hi, u_lo:u_hi].astype(np.float64)
            valid_depths = depth_patch[(depth_patch > self.min_depth) &
                                       (depth_patch < self.max_depth) &
                                       np.isfinite(depth_patch)]
            if len(valid_depths) == 0:
                continue
            
            # The depth patch gives the exact metric distance to the flat textured face
            depth_m_real = float(np.median(valid_depths))

            # Pose orientation in camera optical frame from OpenCV
            rvec = rvecs[i][0]
            
            # Convert rvec to rotation matrix
            rmat, _ = cv2.Rodrigues(rvec)

            # Optical -> Link conversion
            R_opt2link = np.array([
                [0,  0, 1],
                [-1, 0, 0],
                [0, -1, 0]
            ], dtype=np.float64)

            # Deproject pixel explicitly with true depth to fix OpenCV tracking bias
            p_cam_face = self.deproject_pixel(center_u, center_v, depth_m_real, fx, fy, cx, cy)
            
            # rmat defines marker to optical frame. R_marker2cam is marker to link frame.
            R_marker2cam = R_opt2link @ rmat

            # Shift from face center to volumetric center of the 0.5m box
            # local Z axis points OUT of the face, so push IN by 0.25 (negative Z)
            p_cam = p_cam_face + R_marker2cam @ np.array([0.0, 0.0, -0.25])

            # Transform from Link to Odom
            p_odom = rot @ p_cam + t_vec
            R_marker2odom = rot @ R_marker2cam

            # Convert rotation matrix to quaternion
            q = ScipyRotation.from_matrix(R_marker2odom).as_quat()  # [x, y, z, w]

            # World coordinates for panel
            world_x = float(p_odom[0]) + self.spawn_x
            world_y = float(p_odom[1]) + self.spawn_y
            world_z = float(p_odom[2])
            
            # Generate deterministic random color for this marker ID
            np.random.seed(marker_id)
            r, g, b = np.random.rand(3)

            # Panel marker (world coords in position)
            new_panel = self._make_marker(marker_id, stamp, world_x, world_y, world_z, r, g, b,
                                          qx=q[0], qy=q[1], qz=q[2], qw=q[3], lifetime=5)

            # Viz marker (odom-relative coords for correct RViz placement)
            new_viz = self._make_marker(marker_id, stamp,
                                        float(p_odom[0]), float(p_odom[1]), float(p_odom[2]), 
                                        r, g, b, qx=q[0], qy=q[1], qz=q[2], qw=q[3], lifetime=0)
                                  
            # Text marker (fixed ID and custom namespace so it never conflicts)
            new_text = self._make_marker(marker_id, stamp,
                                         float(p_odom[0]), float(p_odom[1]), float(p_odom[2]) + 0.40, 
                                         1.0, 1.0, 1.0, qx=q[0], qy=q[1], qz=q[2], qw=q[3], 
                                         lifetime=0, mtype=Marker.TEXT_VIEW_FACING, text=f"ID: {marker_id}", ns="aruco_text")

            self.locked_panel_markers[marker_id] = new_panel
            self.locked_viz_markers[marker_id] = [new_viz, new_text]

            self.get_logger().info(
                f"ArUco#{marker_id} depth={depth_m_real:.2f}m "
                f"LOCKED at world=({world_x:.2f},{world_y:.2f},{world_z:.2f})")

        # Append all historically locked markers to the current publishing arrays
        for pm in self.locked_panel_markers.values():
            panel_markers.markers.append(pm)
        for vm_list in self.locked_viz_markers.values():
            viz_markers.markers.extend(vm_list)


def main(args=None):
    rclpy.init(args=args)
    node = MarkerDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
