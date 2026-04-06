#!/usr/bin/env python3
"""
ArUco Marker & Sign Detector Node
Subscribes to RGB-D camera, detects ArUco markers and directional arrow signs,
computes 3D position, and publishes MarkerArray for RViz visualization +
Mission Control Panel.
"""
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String
from geometry_msgs.msg import Point
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
        self.aruco_params.errorCorrectionRate = 0.6

        # TF2 Setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── Parameters from config YAML ──────────────────────────────────
        # General
        self.spawn_x = self.declare_parameter('spawn_x', 0.0).value
        self.spawn_y = self.declare_parameter('spawn_y', 0.0).value
        self.get_logger().info(f"Spawn offset: ({self.spawn_x:.2f}, {self.spawn_y:.2f})")

        self.min_depth = 0.1
        self.max_depth = self.declare_parameter('max_marker_dist', 3.5).value
        self.min_marker_area_px = int(self.declare_parameter('min_marker_area_px', 200).value)
        self.max_aruco_id = int(self.declare_parameter('max_aruco_id', 10).value)

        # ArUco physical
        self.aruco_marker_length = self.declare_parameter('aruco_marker_length', 0.5).value
        self.aruco_box_size = self.declare_parameter('aruco_box_size', 0.5).value

        # Sign detection — blue-first approach
        self.sign_blue_h_low = int(self.declare_parameter('sign_blue_h_low', 100).value)
        self.sign_blue_h_high = int(self.declare_parameter('sign_blue_h_high', 130).value)
        self.sign_blue_s_min = int(self.declare_parameter('sign_blue_s_min', 100).value)
        self.sign_blue_v_min = int(self.declare_parameter('sign_blue_v_min', 80).value)
        self.sign_min_blue_area = int(self.declare_parameter('sign_min_blue_area', 400).value)
        self.sign_curved_max_convexity = self.declare_parameter('sign_curved_max_convexity', 0.65).value

        # ── Message Filters Synchronizer ─────────────────────────────────
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

        # Publishers
        self.marker_pub = self.create_publisher(
            MarkerArray, '/mini_r1/mission_control/detected_objects', 10)
        self.viz_pub = self.create_publisher(
            MarkerArray, '/mini_r1/mission_control/viz_markers', 10)
        self.sign_detection_pub = self.create_publisher(
            String, '/mini_r1/sign_detections', 10)
        # Locked markers (first-detection only)
        self.locked_panel_markers = {}
        self.locked_viz_markers = {}
        # Signs use separate dicts keyed by "sign_<kind>_<counter>"
        self.locked_panel_signs = {}
        self.locked_viz_signs = {}
        self.sign_counter = 0

        self.get_logger().info("MarkerDetectorNode initialized. Waiting for RGB-D streams...")

    # ══════════════════════════════════════════════════════════════════════
    #  Sync callback
    # ══════════════════════════════════════════════════════════════════════
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
        t_vec = np.array([t.x, t.y, t.z])

        panel_markers = MarkerArray()
        viz_markers = MarkerArray()
        self.detect_aruco(cv_gray, cv_depth, fx, fy, cx, cy, rot, t_vec, panel_markers, viz_markers)
        self.detect_signs(cv_rgb, cv_depth, fx, fy, cx, cy, rot, t_vec, panel_markers, viz_markers)

        if panel_markers.markers:
            self.marker_pub.publish(panel_markers)
        if viz_markers.markers:
            self.viz_pub.publish(viz_markers)

    # ══════════════════════════════════════════════════════════════════════
    #  Helpers
    # ══════════════════════════════════════════════════════════════════════
    def deproject_pixel(self, u, v, depth, fx, fy, cx, cy):
        """Deproject pixel to 3D in camera LINK frame (X-fwd, Y-left, Z-up)."""
        opt_x = (u - cx) * depth / fx
        opt_y = (v - cy) * depth / fy
        opt_z = depth
        return np.array([opt_z, -opt_x, -opt_y])

    def _make_marker(self, marker_id, stamp, x, y, z, r, g, b,
                     qx=0.0, qy=0.0, qz=0.0, qw=1.0,
                     lifetime=5, mtype=Marker.CUBE, text="", ns="aruco",
                     sx=0.5, sy=0.5, sz=0.5):
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
            marker.scale.x = sx
            marker.scale.y = sy
            marker.scale.z = sz
        elif mtype == Marker.TEXT_VIEW_FACING:
            marker.scale.z = 0.2
            marker.text = text

        marker.color.a = 0.8
        marker.color.r = float(r)
        marker.color.g = float(g)
        marker.color.b = float(b)
        marker.lifetime.sec = lifetime
        return marker

    # ══════════════════════════════════════════════════════════════════════
    #  ArUco Detection
    # ══════════════════════════════════════════════════════════════════════
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

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, self.aruco_marker_length, camera_matrix, dist_coeffs)

        if rvecs is None or tvecs is None:
            return

        for i, marker_id_arr in enumerate(ids):
            marker_id = int(marker_id_arr[0])

            if marker_id < 0 or marker_id > self.max_aruco_id:
                continue

            # Lock to first detection
            if marker_id in self.locked_viz_markers:
                continue

            c = corners[i][0]
            area = cv2.contourArea(c)
            if area < self.min_marker_area_px:
                continue

            center_u = int(c[:, 0].mean())
            center_v = int(c[:, 1].mean())

            h_depth, w_depth = cv_depth.shape[:2]
            if center_u < 0 or center_u >= w_depth or center_v < 0 or center_v >= h_depth:
                continue

            patch_r = 3
            u_lo, u_hi = max(0, center_u-patch_r), min(w_depth, center_u+patch_r+1)
            v_lo, v_hi = max(0, center_v-patch_r), min(h_depth, center_v+patch_r+1)
            depth_patch = cv_depth[v_lo:v_hi, u_lo:u_hi].astype(np.float64)
            valid_depths = depth_patch[(depth_patch > self.min_depth) &
                                       (depth_patch < self.max_depth) &
                                       np.isfinite(depth_patch)]
            if len(valid_depths) == 0:
                continue

            depth_m_real = float(np.median(valid_depths))

            rvec = rvecs[i][0]
            rmat, _ = cv2.Rodrigues(rvec)

            R_opt2link = np.array([
                [0,  0, 1],
                [-1, 0, 0],
                [0, -1, 0]
            ], dtype=np.float64)

            p_cam_face = self.deproject_pixel(center_u, center_v, depth_m_real, fx, fy, cx, cy)
            R_marker2cam = R_opt2link @ rmat
            half_box = self.aruco_box_size / 2.0
            p_cam = p_cam_face + R_marker2cam @ np.array([0.0, 0.0, -half_box])

            p_odom = rot @ p_cam + t_vec
            R_marker2odom = rot @ R_marker2cam
            q = ScipyRotation.from_matrix(R_marker2odom).as_quat()

            world_x = float(p_odom[0]) + self.spawn_x
            world_y = float(p_odom[1]) + self.spawn_y
            world_z = float(p_odom[2])

            np.random.seed(marker_id)
            r, g, b = np.random.rand(3)

            bs = self.aruco_box_size
            new_panel = self._make_marker(marker_id, stamp, world_x, world_y, world_z, r, g, b,
                                          qx=q[0], qy=q[1], qz=q[2], qw=q[3], lifetime=5,
                                          sx=bs, sy=bs, sz=bs)

            new_viz = self._make_marker(marker_id, stamp,
                                        float(p_odom[0]), float(p_odom[1]), float(p_odom[2]),
                                        r, g, b, qx=q[0], qy=q[1], qz=q[2], qw=q[3], lifetime=0,
                                        sx=bs, sy=bs, sz=bs)

            new_text = self._make_marker(marker_id, stamp,
                                         float(p_odom[0]), float(p_odom[1]), float(p_odom[2]) + 0.40,
                                         1.0, 1.0, 1.0, qx=q[0], qy=q[1], qz=q[2], qw=q[3],
                                         lifetime=0, mtype=Marker.TEXT_VIEW_FACING,
                                         text=f"ID: {marker_id}", ns="aruco_text")

            self.locked_panel_markers[marker_id] = new_panel
            self.locked_viz_markers[marker_id] = [new_viz, new_text]

            self.get_logger().info(
                f"ArUco#{marker_id} depth={depth_m_real:.2f}m "
                f"LOCKED at world=({world_x:.2f},{world_y:.2f},{world_z:.2f})")

        # Re-publish all locked markers every frame
        for pm in self.locked_panel_markers.values():
            panel_markers.markers.append(pm)
        for vm_list in self.locked_viz_markers.values():
            viz_markers.markers.extend(vm_list)

    # ══════════════════════════════════════════════════════════════════════
    #  Arrow / Sign Detection  (blue-first approach)
    # ══════════════════════════════════════════════════════════════════════
    def _has_white_surround(self, hsv_img, contour, margin=15):
        """Check if a blue contour is surrounded by white (the sign panel)."""
        x, y, w, h = cv2.boundingRect(contour)
        img_h, img_w = hsv_img.shape[:2]
        # Expand bounding box by margin
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(img_w, x + w + margin)
        y2 = min(img_h, y + h + margin)

        roi = hsv_img[y1:y2, x1:x2]
        # White: low saturation, high value
        white_mask = cv2.inRange(roi, np.array([0, 0, 170]), np.array([180, 50, 255]))
        white_ratio = np.count_nonzero(white_mask) / (roi.shape[0] * roi.shape[1] + 1)
        return white_ratio > 0.15  # at least 15% white surround

    def _classify_arrow_direction(self, contour):
        """Classify an arrow contour's direction in image space.
        Returns ('forward', 'left', 'right', 'rotate_180') or None.
        """
        hull_pts = cv2.convexHull(contour, returnPoints=True)
        hull_area = cv2.contourArea(hull_pts)
        cnt_area = cv2.contourArea(contour)
        convexity = cnt_area / hull_area if hull_area > 0 else 1.0

        if convexity < self.sign_curved_max_convexity:
            return 'rotate_180'

        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        hull_pts_sq = hull_pts.squeeze()
        if hull_pts_sq.ndim != 2:
            return None
        dists = np.sqrt((hull_pts_sq[:, 0] - cx)**2 + (hull_pts_sq[:, 1] - cy)**2)
        tip_idx = np.argmax(dists)
        tip = hull_pts_sq[tip_idx]

        dx = tip[0] - cx
        dy = tip[1] - cy
        length = np.sqrt(dx*dx + dy*dy)
        if length < 5:
            return None

        angle_deg = np.degrees(np.arctan2(dy, dx))

        if -135 < angle_deg <= -45:
            return 'forward'
        elif -45 < angle_deg <= 45:
            return 'right'
        elif 45 < angle_deg <= 135:
            return None  # pointing down = seeing sign from behind
        else:
            return 'left'

    def detect_signs(self, cv_rgb, cv_depth, fx, fy, cx, cy, rot, t_vec, panel_markers, viz_markers):
        """Detect arrow signs by finding blue regions on white panels."""

        stamp = self.get_clock().now().to_msg()
        h_depth, w_depth = cv_depth.shape[:2]

        # ── Step 1: Find blue regions in HSV ──
        hsv = cv2.cvtColor(cv_rgb, cv2.COLOR_RGB2HSV)
        lower = np.array([self.sign_blue_h_low, self.sign_blue_s_min, self.sign_blue_v_min])
        upper = np.array([self.sign_blue_h_high, 255, 255])
        blue_mask = cv2.inRange(hsv, lower, upper)

        k = np.ones((5, 5), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, k, iterations=2)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, k, iterations=1)

        # ── Step 2: Find contours ──
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.sign_min_blue_area:
                continue

            # ── Step 3: Validate — blue must be on a white sign panel ──
            if not self._has_white_surround(hsv, cnt):
                continue

            # ── Step 4: Shape validation — reject floor tiles / large blobs ──
            x, y, w, h = cv2.boundingRect(cnt)
            bbox_area = w * h
            # Arrow should fill 20-85% of its bbox (not a solid rectangle)
            fill_ratio = area / bbox_area if bbox_area > 0 else 0
            if fill_ratio > 0.85 or fill_ratio < 0.15:
                continue
            # Reject very large blobs (floor tiles) — sign is small in the image
            if bbox_area > (h_depth * w_depth * 0.15):
                continue

            # ── Step 5: Classify arrow direction ──
            direction = self._classify_arrow_direction(cnt)
            if direction is None:
                continue

            # ── Step 4: Get centroid and depth ──
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            centroid_u = int(M["m10"] / M["m00"])
            centroid_v = int(M["m01"] / M["m00"])

            if centroid_u < 0 or centroid_u >= w_depth or centroid_v < 0 or centroid_v >= h_depth:
                continue

            patch_r = 5
            u_lo = max(0, centroid_u - patch_r)
            u_hi = min(w_depth, centroid_u + patch_r + 1)
            v_lo = max(0, centroid_v - patch_r)
            v_hi = min(h_depth, centroid_v + patch_r + 1)
            depth_patch = cv_depth[v_lo:v_hi, u_lo:u_hi].astype(np.float64)
            valid = depth_patch[(depth_patch > self.min_depth) &
                                (depth_patch < self.max_depth) &
                                np.isfinite(depth_patch)]
            if len(valid) == 0:
                continue

            depth_m = float(np.median(valid))

            # ── Step 5: 3D projection ──
            p_cam = self.deproject_pixel(centroid_u, centroid_v, depth_m, fx, fy, cx, cy)
            p_odom = rot @ p_cam + t_vec

            # ── Step 6: Check if already locked nearby ──
            already_locked = False
            world_x = float(p_odom[0]) + self.spawn_x
            world_y = float(p_odom[1]) + self.spawn_y
            world_z = float(p_odom[2])
            for key in self.locked_panel_signs:
                pm = self.locked_panel_signs[key]
                dx = pm.pose.position.x - world_x
                dy = pm.pose.position.y - world_y
                if (dx*dx + dy*dy) < 1.0:
                    already_locked = True
                    break
            if already_locked:
                continue

            # ── Step 7: Create markers and publish ──
            self.sign_counter += 1
            sign_id = self.sign_counter

            # Color by direction
            color_map = {
                'forward': (0.0, 1.0, 0.0),
                'left':    (1.0, 1.0, 0.0),
                'right':   (0.0, 1.0, 1.0),
                'rotate_180': (1.0, 0.0, 1.0),
            }
            cr, cg, cb = color_map.get(direction, (1.0, 1.0, 1.0))

            # Text label in RViz
            text_marker = self._make_marker(
                sign_id, stamp,
                float(p_odom[0]), float(p_odom[1]), float(p_odom[2]) + 0.25,
                1.0, 1.0, 1.0, lifetime=0,
                mtype=Marker.TEXT_VIEW_FACING,
                text=f"Sign: {direction}", ns="sign_text")

            # Cube marker in RViz
            viz_cube = self._make_marker(
                sign_id, stamp,
                float(p_odom[0]), float(p_odom[1]), float(p_odom[2]),
                cr, cg, cb, lifetime=0, ns="sign",
                sx=0.3, sy=0.3, sz=0.3)

            # Panel marker for mission control (world coords)
            panel_m = self._make_marker(
                sign_id, stamp, world_x, world_y, world_z,
                cr, cg, cb, lifetime=5, ns="sign",
                sx=0.3, sy=0.3, sz=0.3)

            self.locked_panel_signs[sign_id] = panel_m
            self.locked_viz_signs[sign_id] = [viz_cube, text_marker]

            # Publish direction for navigator
            msg = String()
            msg.data = direction
            self.sign_detection_pub.publish(msg)

            self.get_logger().info(
                f"Sign[{direction}] #{sign_id} depth={depth_m:.2f}m "
                f"LOCKED at world=({world_x:.2f},{world_y:.2f},{world_z:.2f})")

        # Re-publish all locked signs every frame
        for pm in self.locked_panel_signs.values():
            panel_markers.markers.append(pm)
        for vm_list in self.locked_viz_signs.values():
            viz_markers.markers.extend(vm_list)

def main(args=None):
    rclpy.init(args=args)
    node = MarkerDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
