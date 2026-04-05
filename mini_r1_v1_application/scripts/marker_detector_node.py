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

        # Sign detection preprocessing
        self.sign_border_crop = int(self.declare_parameter('sign_border_crop', 20).value)
        self.sign_blur_kernel = int(self.declare_parameter('sign_blur_kernel', 5).value)
        self.sign_blur_sigma = int(self.declare_parameter('sign_blur_sigma', 1).value)
        self.sign_canny_low = int(self.declare_parameter('sign_canny_low', 50).value)
        self.sign_canny_high = int(self.declare_parameter('sign_canny_high', 50).value)
        self.sign_dilate_iter = int(self.declare_parameter('sign_dilate_iter', 2).value)
        self.sign_erode_iter = int(self.declare_parameter('sign_erode_iter', 1).value)

        # Sign contour filtering
        self.sign_min_area = int(self.declare_parameter('sign_min_area', 5000).value)

        # Straight arrow
        self.sign_approx_epsilon = self.declare_parameter('sign_approx_epsilon', 0.015).value
        self.sign_hull_min_sides = int(self.declare_parameter('sign_hull_min_sides', 4).value)
        self.sign_hull_max_sides = int(self.declare_parameter('sign_hull_max_sides', 6).value)

        # Curved arrow
        self.sign_curved_max_convexity = self.declare_parameter('sign_curved_max_convexity', 0.6).value
        self.sign_curved_tip_max_angle = self.declare_parameter('sign_curved_tip_max_angle', 60.0).value

        # Sign panel physical size
        self.sign_panel_width = self.declare_parameter('sign_panel_width', 0.3).value
        self.sign_panel_height = self.declare_parameter('sign_panel_height', 0.3).value
        self.sign_panel_thickness = self.declare_parameter('sign_panel_thickness', 0.015).value

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
    #  Arrow / Sign Detection
    # ══════════════════════════════════════════════════════════════════════
    @staticmethod
    def _angle_at_vertex(pts, i):
        """Interior angle (degrees) at vertex i of a polygon."""
        n = len(pts)
        a = pts[(i - 1) % n].astype(float)
        b = pts[i].astype(float)
        c = pts[(i + 1) % n].astype(float)
        ba, bc = a - b, c - b
        cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

    def _find_tip_straight(self, points, convex_hull):
        """Find tip of a straight arrow via hull defect vertices."""
        length = len(points)
        indices = np.setdiff1d(range(length), convex_hull)
        if len(indices) != 2:
            return None
        for i in range(2):
            j = (indices[i] + 2) % length
            if np.all(points[j] == points[indices[i - 1] - 2]):
                return tuple(points[j])
        return None

    def _find_tip_curved(self, approx_pts):
        """Find tip of a curved arrow as the sharpest interior angle vertex."""
        pts = approx_pts[:, 0, :]
        angles = [self._angle_at_vertex(pts, i) for i in range(len(pts))]
        min_angle = min(angles)
        if min_angle > self.sign_curved_tip_max_angle:
            return None, None
        return tuple(pts[int(np.argmin(angles))]), min_angle

    def _detect_arrow_in_crop(self, crop_bgr):
        """
        Run arrow detection on a SINGLE isolated crop of a sign panel.
        Returns (kind, contour, tip) or None.
        crop_bgr should be in BGR format.
        """
        bc = self.sign_border_crop
        h, w = crop_bgr.shape[:2]
        if h <= 2 * bc or w <= 2 * bc:
            return None

        inner = crop_bgr[bc:h - bc, bc:w - bc]
        gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
        k_size = self.sign_blur_kernel
        blur = cv2.GaussianBlur(gray, (k_size, k_size), self.sign_blur_sigma)
        edges = cv2.Canny(blur, self.sign_canny_low, self.sign_canny_high)
        k = np.ones((3, 3))
        processed = cv2.erode(
            cv2.dilate(edges, k, iterations=self.sign_dilate_iter),
            k, iterations=self.sign_erode_iter)

        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Relative to crop: arrow should fill at least 10% of the crop area
            crop_area = inner.shape[0] * inner.shape[1]
            if area < crop_area * 0.10:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, self.sign_approx_epsilon * peri, True)
            hull = cv2.convexHull(approx, returnPoints=False)
            sides = len(hull)

            hull_pts = cv2.convexHull(cnt, returnPoints=True)
            hull_area = cv2.contourArea(hull_pts)
            convexity = cv2.contourArea(cnt) / hull_area if hull_area > 0 else 1.0

            # Shift contour back to crop-local coordinates (add border crop offset)
            cnt_local = cnt + bc

            # Straight arrow
            if self.sign_hull_max_sides > sides > self.sign_hull_min_sides and sides + 2 == len(approx):
                hull_sq = hull.squeeze()
                if hull_sq.ndim > 0:
                    tip = self._find_tip_straight(approx[:, 0, :], hull_sq)
                    if tip:
                        tip_local = (tip[0] + bc, tip[1] + bc)
                        return ('straight', cnt_local, tip_local)

            # Curved arrow
            if convexity < self.sign_curved_max_convexity:
                tip, angle = self._find_tip_curved(approx)
                if tip:
                    tip_local = (tip[0] + bc, tip[1] + bc)
                    return ('curved', cnt_local, tip_local)

        return None

    def _find_sign_panels(self, cv_rgb):
        """
        Find white rectangular sign panels in the camera image.
        Returns list of (x, y, w, h) bounding boxes in image coordinates.
        """
        # Convert RGB to HSV
        hsv = cv2.cvtColor(cv_rgb, cv2.COLOR_RGB2HSV)

        # White regions: low saturation, high value
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 60, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)

        # Clean up the mask
        k = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 400:  # too small
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect = float(w) / h if h > 0 else 0

            # Sign panels are roughly square (0.3m x 0.3m)
            if aspect < 0.4 or aspect > 2.5:
                continue

            # Check that the bounding rect fills reasonably (not a weird L-shape)
            rect_area = w * h
            fill_ratio = area / rect_area if rect_area > 0 else 0
            if fill_ratio < 0.5:
                continue

            candidates.append((x, y, w, h))

        return candidates

    def detect_signs(self, cv_rgb, cv_depth, fx, fy, cx, cy, rot, t_vec, panel_markers, viz_markers):
        """Detect arrow signs by first isolating white panels, then running arrow detection in each crop."""

        # Find white panel candidates
        panels = self._find_sign_panels(cv_rgb)

        stamp = self.get_clock().now().to_msg()
        h_depth, w_depth = cv_depth.shape[:2]

        for (px, py, pw, ph) in panels:
            # Add some margin around the detected white panel
            margin = 10
            x1 = max(0, px - margin)
            y1 = max(0, py - margin)
            x2 = min(cv_rgb.shape[1], px + pw + margin)
            y2 = min(cv_rgb.shape[0], py + ph + margin)

            crop_rgb = cv_rgb[y1:y2, x1:x2]
            # Convert RGB→BGR for the arrow detection (originally designed for BGR)
            crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)

            result = self._detect_arrow_in_crop(crop_bgr)
            if result is None:
                continue

            kind, cnt_local, tip_local = result

            # Convert contour & tip from crop-local to full-image coordinates
            cnt_full = cnt_local.copy()
            cnt_full[:, :, 0] += x1
            cnt_full[:, :, 1] += y1
            tip_full = (tip_local[0] + x1, tip_local[1] + y1)

            # Compute centroid of the contour in image coords
            M = cv2.moments(cnt_local)
            if M["m00"] == 0:
                continue
            centroid_u = int(M["m10"] / M["m00"]) + x1
            centroid_v = int(M["m01"] / M["m00"]) + y1

            # Get depth at centroid
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

            # 3D centroid in camera link frame
            p_cam_centroid = self.deproject_pixel(centroid_u, centroid_v, depth_m, fx, fy, cx, cy)
            p_odom_centroid = rot @ p_cam_centroid + t_vec

            # Check if already locked nearby
            already_locked = False
            for key, vm_list in self.locked_viz_signs.items():
                existing = vm_list[0]  # first marker in group
                # If LINE_STRIP is first, position is unreliable? Actually LINE_STRIP doesn't use pose.position.
                # Oh wait, we need to check the TEXT marker or Panel marker position.
                bz_x = self.locked_panel_signs[key].pose.position.x
                bz_y = self.locked_panel_signs[key].pose.position.y
                bz_z = self.locked_panel_signs[key].pose.position.z
                
                # Transform centroid world coords to check against panel marker coords
                world_centroid_x = float(p_odom_centroid[0]) + self.spawn_x
                world_centroid_y = float(p_odom_centroid[1]) + self.spawn_y
                world_centroid_z = float(p_odom_centroid[2])

                dx = bz_x - world_centroid_x
                dy = bz_y - world_centroid_y
                dz = bz_z - world_centroid_z
                if (dx*dx + dy*dy + dz*dz) < 1.0:
                    already_locked = True
                    break
            if already_locked:
                continue

            # ── Build 3D contour (LINE_STRIP) from contour pixels + depth ──
            # Subsample the contour to ~80 points max for performance
            cnt_pts = cnt_full[:, 0, :]  # shape (N, 2)
            n_pts = len(cnt_pts)
            step = max(1, n_pts // 80)
            sampled_indices = list(range(0, n_pts, step))
            # Close the loop
            if sampled_indices[-1] != 0:
                sampled_indices.append(0)

            odom_3d_points = []
            for idx in sampled_indices:
                su, sv = int(cnt_pts[idx][0]), int(cnt_pts[idx][1])
                if su < 0 or su >= w_depth or sv < 0 or sv >= h_depth:
                    continue
                d = float(cv_depth[sv, su])
                if d < self.min_depth or d > self.max_depth or not np.isfinite(d):
                    d = depth_m  # fallback to centroid depth
                p3d = self.deproject_pixel(su, sv, d, fx, fy, cx, cy)
                p3d_odom = rot @ p3d + t_vec
                odom_3d_points.append(p3d_odom)

            if len(odom_3d_points) < 5:
                continue

            # ── Compute arrow direction in 3D (centroid → tip) ──
            tip_u, tip_v = int(tip_full[0]), int(tip_full[1])
            if 0 <= tip_u < w_depth and 0 <= tip_v < h_depth:
                d_tip = float(cv_depth[tip_v, tip_u])
                if d_tip < self.min_depth or d_tip > self.max_depth or not np.isfinite(d_tip):
                    d_tip = depth_m
            else:
                d_tip = depth_m
            p_cam_tip = self.deproject_pixel(tip_u, tip_v, d_tip, fx, fy, cx, cy)
            p_odom_tip = rot @ p_cam_tip + t_vec

            # ── Assign unique sign ID ──
            self.sign_counter += 1
            sign_id = self.sign_counter

            # Color: cyan for straight, magenta for curved
            if kind == 'straight':
                cr, cg, cb = 0.0, 0.9, 0.9
            else:
                cr, cg, cb = 0.9, 0.0, 0.9

            # World coordinates for panel
            world_x = float(p_odom_centroid[0]) + self.spawn_x
            world_y = float(p_odom_centroid[1]) + self.spawn_y
            world_z = float(p_odom_centroid[2])

            # ── Create LINE_STRIP marker for the arrow contour shape ──
            from geometry_msgs.msg import Point
            contour_marker = Marker()
            contour_marker.header.frame_id = "odom"
            contour_marker.header.stamp = stamp
            contour_marker.ns = "sign_contour"
            contour_marker.id = sign_id
            contour_marker.type = Marker.LINE_STRIP
            contour_marker.action = Marker.ADD
            contour_marker.scale.x = 0.008  # line thickness
            contour_marker.color.r = float(cr)
            contour_marker.color.g = float(cg)
            contour_marker.color.b = float(cb)
            contour_marker.color.a = 1.0
            contour_marker.lifetime.sec = 0
            for p3d in odom_3d_points:
                pt = Point()
                pt.x = float(p3d[0])
                pt.y = float(p3d[1])
                pt.z = float(p3d[2])
                contour_marker.points.append(pt)

            # ── Create ARROW marker showing direction (centroid → tip) ──
            arrow_marker = Marker()
            arrow_marker.header.frame_id = "odom"
            arrow_marker.header.stamp = stamp
            arrow_marker.ns = "sign_direction"
            arrow_marker.id = sign_id
            arrow_marker.type = Marker.ARROW
            arrow_marker.action = Marker.ADD
            arrow_marker.scale.x = 0.02   # shaft diameter
            arrow_marker.scale.y = 0.04   # head diameter
            arrow_marker.scale.z = 0.03   # head length
            arrow_marker.color.r = 1.0
            arrow_marker.color.g = 1.0
            arrow_marker.color.b = 0.0
            arrow_marker.color.a = 1.0
            arrow_marker.lifetime.sec = 0

            pt_start = Point()
            pt_start.x = float(p_odom_centroid[0])
            pt_start.y = float(p_odom_centroid[1])
            pt_start.z = float(p_odom_centroid[2])
            pt_end = Point()
            pt_end.x = float(p_odom_tip[0])
            pt_end.y = float(p_odom_tip[1])
            pt_end.z = float(p_odom_tip[2])
            arrow_marker.points.append(pt_start)
            arrow_marker.points.append(pt_end)

            # ── Text label ──
            text_marker = self._make_marker(
                sign_id, stamp,
                float(p_odom_centroid[0]),
                float(p_odom_centroid[1]),
                float(p_odom_centroid[2]) + 0.25,
                1.0, 1.0, 1.0, lifetime=0,
                mtype=Marker.TEXT_VIEW_FACING,
                text=f"Sign: {kind}", ns="sign_text")

            # ── Panel marker for Mission Control ──
            new_panel = self._make_marker(
                sign_id, stamp, world_x, world_y, world_z,
                cr, cg, cb, lifetime=5, ns="sign")

            # Lock
            self.locked_panel_signs[sign_id] = new_panel
            self.locked_viz_signs[sign_id] = [contour_marker, arrow_marker, text_marker]

            self.get_logger().info(
                f"Sign[{kind}] #{sign_id} depth={depth_m:.2f}m "
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
