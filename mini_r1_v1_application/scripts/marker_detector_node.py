#!/usr/bin/env python3
"""
ArUco Marker & Sign Detector Node
Subscribes to RGB-D camera, detects ArUco markers and directional arrow signs
(via MobileSAM), computes 3D position, and publishes MarkerArray for RViz
visualization + Mission Control Panel.
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

# FastSAM via ultralytics (loaded in __init__ to avoid import-time GPU allocation)


class MarkerDetectorNode(Node):
    def __init__(self):
        super().__init__('marker_detector_node')

        self.bridge = CvBridge()

        # ArUco setup — handle both old API (OpenCV <4.8) and new API (4.8+)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_params.adaptiveThreshConstant = 7
        self.aruco_params.minMarkerPerimeterRate = 0.03
        self.aruco_params.maxMarkerPerimeterRate = 4.0
        self.aruco_params.polygonalApproxAccuracyRate = 0.05
        self.aruco_params.minCornerDistanceRate = 0.05
        self.aruco_params.minDistanceToBorder = 3
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.aruco_params.cornerRefinementWinSize = 5
        self.aruco_params.errorCorrectionRate = 0.6
        # New API uses ArucoDetector class
        self._use_new_aruco = hasattr(cv2.aruco, 'ArucoDetector')
        if self._use_new_aruco:
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # TF2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── Parameters ───────────────────────────────────────────────────
        self.spawn_x = self.declare_parameter('spawn_x', 0.0).value
        self.spawn_y = self.declare_parameter('spawn_y', 0.0).value
        self.get_logger().info(f"Spawn offset: ({self.spawn_x:.2f}, {self.spawn_y:.2f})")

        self.min_depth = 0.1
        self.max_depth = self.declare_parameter('max_marker_dist', 3.5).value
        self.min_marker_area_px = int(self.declare_parameter('min_marker_area_px', 200).value)
        self.max_aruco_id = int(self.declare_parameter('max_aruco_id', 10).value)

        self.aruco_marker_length = self.declare_parameter('aruco_marker_length', 0.5).value
        self.aruco_box_size = self.declare_parameter('aruco_box_size', 0.5).value

        # Sign detection params (for SAM mask filtering — orange arrows)
        self.sign_arrow_h_low = int(self.declare_parameter('sign_arrow_h_low', 0).value)
        self.sign_arrow_h_high = int(self.declare_parameter('sign_arrow_h_high', 25).value)
        self.sign_arrow_s_min = int(self.declare_parameter('sign_arrow_s_min', 150).value)
        self.sign_arrow_v_min = int(self.declare_parameter('sign_arrow_v_min', 100).value)
        self.sign_min_mask_area = int(self.declare_parameter('sign_min_mask_area', 300).value)
        self.sign_arrow_ratio_min = self.declare_parameter('sign_arrow_ratio_min', 0.30).value
        self.sign_curved_max_convexity = self.declare_parameter('sign_curved_max_convexity', 0.65).value

        # ── FastSAM (YOLO-based, much faster than MobileSAM) ──
        import torch
        import os
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.get_logger().info("Loading FastSAM model...")
        from ultralytics import FastSAM
        self.sam_model = FastSAM("FastSAM-s.pt")  # small variant, ~23MB
        self.get_logger().info("FastSAM loaded.")
        self.frame_counter = 0
        self.sam_process_interval = int(self.declare_parameter('sam_process_interval', 2).value)

        # ── Subscribers ──────────────────────────────────────────────────
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

        # ── Publishers ───────────────────────────────────────────────────
        self.marker_pub = self.create_publisher(
            MarkerArray, '/mini_r1/mission_control/detected_objects', 10)
        self.viz_pub = self.create_publisher(
            MarkerArray, '/mini_r1/mission_control/viz_markers', 10)
        self.sign_detection_pub = self.create_publisher(
            String, '/mini_r1/sign_detections', 10)
        self.annotated_img_pub = self.create_publisher(
            Image, '/mini_r1/sign_detections/image', 10)

        # Locked markers
        self.locked_panel_markers = {}
        self.locked_viz_markers = {}
        self.locked_panel_signs = {}
        self.locked_viz_signs = {}
        self.sign_counter = 0

        self.get_logger().info("MarkerDetectorNode initialized (MobileSAM). Waiting for RGB-D streams...")

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

        # Run SAM every N frames to manage GPU load
        self.frame_counter += 1
        if self.frame_counter % self.sam_process_interval == 0:
            self.detect_signs_sam(cv_rgb, cv_depth, fx, fy, cx, cy, rot, t_vec,
                                 panel_markers, viz_markers)
        else:
            # Still republish locked signs
            for pm in self.locked_panel_signs.values():
                panel_markers.markers.append(pm)
            for vm_list in self.locked_viz_signs.values():
                viz_markers.markers.extend(vm_list)

        if panel_markers.markers:
            self.marker_pub.publish(panel_markers)
        if viz_markers.markers:
            self.viz_pub.publish(viz_markers)

    # ══════════════════════════════════════════════════════════════════════
    #  Helpers
    # ══════════════════════════════════════════════════════════════════════
    def deproject_pixel(self, u, v, depth, fx, fy, cx, cy):
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
    #  ArUco Detection (unchanged)
    # ══════════════════════════════════════════════════════════════════════
    def detect_aruco(self, cv_gray, cv_depth, fx, fy, cx, cy, rot, t_vec, panel_markers, viz_markers):
        if self._use_new_aruco:
            corners, ids, _ = self.aruco_detector.detectMarkers(cv_gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                cv_gray, self.aruco_dict, parameters=self.aruco_params)
        if ids is None:
            return

        stamp = self.get_clock().now().to_msg()
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        # estimatePoseSingleMarkers removed in OpenCV 4.8+, use solvePnP per marker
        half_len = self.aruco_marker_length / 2.0
        obj_points = np.array([
            [-half_len,  half_len, 0],
            [ half_len,  half_len, 0],
            [ half_len, -half_len, 0],
            [-half_len, -half_len, 0],
        ], dtype=np.float32)

        for i, marker_id_arr in enumerate(ids):
            marker_id = int(marker_id_arr[0])
            if marker_id < 0 or marker_id > self.max_aruco_id:
                continue
            if marker_id in self.locked_viz_markers:
                continue

            c = corners[i][0]
            area = cv2.contourArea(c)
            if area < self.min_marker_area_px:
                continue

            success, rvec, tvec = cv2.solvePnP(
                obj_points, c, camera_matrix, dist_coeffs)
            if not success:
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
            rvec_flat = rvec.flatten()
            rmat, _ = cv2.Rodrigues(rvec_flat)
            R_opt2link = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)
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

        for pm in self.locked_panel_markers.values():
            panel_markers.markers.append(pm)
        for vm_list in self.locked_viz_markers.values():
            viz_markers.markers.extend(vm_list)

    # ══════════════════════════════════════════════════════════════════════
    #  MobileSAM Sign Detection
    # ══════════════════════════════════════════════════════════════════════
    def _is_arrow_color_mask(self, hsv_img, mask):
        """Check if >arrow_ratio_min of mask pixels are orange (arrow color)."""
        lower = np.array([self.sign_arrow_h_low, self.sign_arrow_s_min, self.sign_arrow_v_min])
        upper = np.array([self.sign_arrow_h_high, 255, 255])
        color_in_hsv = cv2.inRange(hsv_img, lower, upper)
        color_pixels = np.count_nonzero(color_in_hsv & mask)
        mask_pixels = np.count_nonzero(mask)
        if mask_pixels == 0:
            return False
        return (color_pixels / mask_pixels) > self.sign_arrow_ratio_min

    def _has_white_surround(self, hsv_img, mask, margin=20):
        """Check if area around mask has white pixels (the sign panel)."""
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            return False
        x1 = max(0, int(xs.min()) - margin)
        y1 = max(0, int(ys.min()) - margin)
        x2 = min(hsv_img.shape[1], int(xs.max()) + margin)
        y2 = min(hsv_img.shape[0], int(ys.max()) + margin)
        roi = hsv_img[y1:y2, x1:x2]
        white_mask = cv2.inRange(roi, np.array([0, 0, 170]), np.array([180, 50, 255]))
        white_ratio = np.count_nonzero(white_mask) / (roi.shape[0] * roi.shape[1] + 1)
        return white_ratio > 0.10

    def _classify_direction_from_mask(self, mask):
        """Classify arrow direction from a binary mask.
        Returns ('forward', 'left', 'right', 'rotate_180') or None.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None, None
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < self.sign_min_mask_area:
            return None, None

        # Convexity check for curved arrows
        hull_pts = cv2.convexHull(cnt, returnPoints=True)
        hull_area = cv2.contourArea(hull_pts)
        convexity = area / hull_area if hull_area > 0 else 1.0
        if convexity < self.sign_curved_max_convexity:
            return 'rotate_180', cnt

        # Straight arrows: centroid-to-tip
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            return None, None
        cx_m = M["m10"] / M["m00"]
        cy_m = M["m01"] / M["m00"]

        hull_sq = hull_pts.squeeze()
        if hull_sq.ndim != 2:
            return None, None
        dists = np.sqrt((hull_sq[:, 0] - cx_m)**2 + (hull_sq[:, 1] - cy_m)**2)
        tip = hull_sq[np.argmax(dists)]

        dx = tip[0] - cx_m
        dy = tip[1] - cy_m
        if np.sqrt(dx*dx + dy*dy) < 5:
            return None, None

        angle = np.degrees(np.arctan2(dy, dx))
        # Camera sees the sign face-on, so image-left = sign's right
        if -135 < angle <= -45:
            return 'forward', cnt
        elif -45 < angle <= 45:
            return 'left', cnt    # image-right = sign points left
        elif 45 < angle <= 135:
            return None, None     # pointing down = back of sign
        else:
            return 'right', cnt   # image-left = sign points right

    def detect_signs_sam(self, cv_rgb, cv_depth, fx, fy, cx, cy, rot, t_vec,
                         panel_markers, viz_markers):
        """Detect arrow signs using MobileSAM segment-everything."""
        h_img, w_img = cv_rgb.shape[:2]
        h_depth, w_depth = cv_depth.shape[:2]
        stamp = self.get_clock().now().to_msg()

        cv_bgr = cv2.cvtColor(cv_rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(cv_rgb, cv2.COLOR_RGB2HSV)

        import torch

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            results = self.sam_model(cv_bgr, imgsz=480, conf=0.3,
                                     device="cuda" if torch.cuda.is_available() else "cpu",
                                     verbose=False)
        except Exception as e:
            self.get_logger().error(f"SAM inference error: {e}", throttle_duration_sec=10.0)
            return

        # Annotated image for RViz
        annotated = cv_bgr.copy()

        if results and results[0].masks is not None:
            masks_data = results[0].masks.data.cpu().numpy()
            n_total = masks_data.shape[0]
            n_orange = 0
            n_white = 0
            n_classified = 0

            for idx in range(n_total):
                raw_mask = masks_data[idx].astype(np.uint8)
                if raw_mask.shape[:2] != (h_img, w_img):
                    mask = cv2.resize(raw_mask, (w_img, h_img),
                                      interpolation=cv2.INTER_NEAREST)
                else:
                    mask = raw_mask
                mask_u8 = (mask > 0.5).astype(np.uint8) * 255

                mask_area = np.count_nonzero(mask_u8)
                if mask_area < self.sign_min_mask_area:
                    continue
                # Reject very large masks (floor, walls)
                if mask_area > (h_img * w_img * 0.1):
                    continue

                # Filter: must be orange (arrow color)
                if not self._is_arrow_color_mask(hsv, mask_u8):
                    continue
                n_orange += 1

                # Filter: must have white surround (sign panel)
                if not self._has_white_surround(hsv, mask_u8):
                    continue
                n_white += 1

                # Classify direction
                direction, cnt = self._classify_direction_from_mask(mask_u8)
                if direction is None or cnt is None:
                    continue
                n_classified += 1

                # Get centroid
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                centroid_u = int(M["m10"] / M["m00"])
                centroid_v = int(M["m01"] / M["m00"])

                if centroid_u < 0 or centroid_u >= w_depth or centroid_v < 0 or centroid_v >= h_depth:
                    continue

                # Depth at centroid
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

                # 3D projection
                p_cam = self.deproject_pixel(centroid_u, centroid_v, depth_m, fx, fy, cx, cy)
                p_odom = rot @ p_cam + t_vec

                world_x = float(p_odom[0]) + self.spawn_x
                world_y = float(p_odom[1]) + self.spawn_y
                world_z = float(p_odom[2])

                # Dedup: skip if ANY sign within 0.5m (prevents duplicates)
                already_locked = False
                for key in self.locked_panel_signs:
                    pm = self.locked_panel_signs[key]
                    ddx = pm.pose.position.x - world_x
                    ddy = pm.pose.position.y - world_y
                    if (ddx*ddx + ddy*ddy) < 0.25:  # 0.5m radius
                        already_locked = True
                        break
                if already_locked:
                    continue

                # ── Create markers ──
                self.sign_counter += 1
                sign_id = self.sign_counter

                color_map = {
                    'forward':    (0.0, 1.0, 0.0),
                    'left':       (1.0, 1.0, 0.0),
                    'right':      (0.0, 1.0, 1.0),
                    'rotate_180': (1.0, 0.0, 1.0),
                }
                cr, cg, cb = color_map.get(direction, (1.0, 1.0, 1.0))

                # 3D contour LINE_STRIP from mask
                cnt_pts = cnt[:, 0, :]
                n_pts = len(cnt_pts)
                step = max(1, n_pts // 60)
                sampled = list(range(0, n_pts, step))
                if sampled[-1] != 0:
                    sampled.append(0)

                odom_3d = []
                for si in sampled:
                    su, sv = int(cnt_pts[si][0]), int(cnt_pts[si][1])
                    if 0 <= su < w_depth and 0 <= sv < h_depth:
                        d = float(cv_depth[sv, su])
                        if d < self.min_depth or d > self.max_depth or not np.isfinite(d):
                            d = depth_m
                        p3 = self.deproject_pixel(su, sv, d, fx, fy, cx, cy)
                        odom_3d.append(rot @ p3 + t_vec)

                if len(odom_3d) < 5:
                    continue

                # LINE_STRIP marker (3D arrow shape)
                contour_marker = Marker()
                contour_marker.header.frame_id = "odom"
                contour_marker.header.stamp = stamp
                contour_marker.ns = "sign_contour"
                contour_marker.id = sign_id
                contour_marker.type = Marker.LINE_STRIP
                contour_marker.action = Marker.ADD
                contour_marker.scale.x = 0.008
                contour_marker.color.r = float(cr)
                contour_marker.color.g = float(cg)
                contour_marker.color.b = float(cb)
                contour_marker.color.a = 1.0
                contour_marker.lifetime.sec = 0
                for p3d in odom_3d:
                    pt = Point()
                    pt.x = float(p3d[0])
                    pt.y = float(p3d[1])
                    pt.z = float(p3d[2])
                    contour_marker.points.append(pt)

                # Text label
                text_marker = self._make_marker(
                    sign_id, stamp,
                    float(p_odom[0]), float(p_odom[1]), float(p_odom[2]) + 0.25,
                    1.0, 1.0, 1.0, lifetime=0,
                    mtype=Marker.TEXT_VIEW_FACING,
                    text=f"Sign: {direction}", ns="sign_text")

                # Cube marker at sign position (translucent so sign is visible inside)
                viz_cube = self._make_marker(
                    sign_id, stamp,
                    float(p_odom[0]), float(p_odom[1]), float(p_odom[2]),
                    cr, cg, cb, lifetime=0, ns="sign",
                    sx=0.3, sy=0.3, sz=0.3)
                viz_cube.color.a = 0.3

                # Panel marker for mission control (world coords)
                panel_m = self._make_marker(
                    sign_id, stamp, world_x, world_y, world_z,
                    cr, cg, cb, lifetime=5, ns="sign",
                    sx=0.3, sy=0.3, sz=0.3)

                self.locked_panel_signs[sign_id] = panel_m
                self.locked_viz_signs[sign_id] = [contour_marker, viz_cube, text_marker]

                # Publish direction
                msg = String()
                msg.data = direction
                self.sign_detection_pub.publish(msg)

                self.get_logger().info(
                    f"Sign[{direction}] #{sign_id} depth={depth_m:.2f}m "
                    f"LOCKED at world=({world_x:.2f},{world_y:.2f},{world_z:.2f})")

                # Draw on annotated image
                color_bgr = (int(cb*255), int(cg*255), int(cr*255))
                overlay = annotated.copy()
                overlay[mask_u8 > 0] = color_bgr
                cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0, annotated)
                cv2.drawContours(annotated, [cnt], -1, color_bgr, 2)
                bx, by, bw, bh = cv2.boundingRect(cnt)
                cv2.putText(annotated, direction, (bx, by - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)

        if n_total > 0:
            self.get_logger().info(
                f"SAM: {n_total} masks → {n_orange} orange → {n_white} white-surround → {n_classified} classified | locked={len(self.locked_panel_signs)}",
                throttle_duration_sec=3.0)

        # Republish all locked signs
        for pm in self.locked_panel_signs.values():
            panel_markers.markers.append(pm)
        for vm_list in self.locked_viz_signs.values():
            viz_markers.markers.extend(vm_list)

        # Publish annotated image
        try:
            ann_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            ann_msg.header.stamp = stamp
            self.annotated_img_pub.publish(ann_msg)
        except Exception as e:
            self.get_logger().error(f"Annotated image publish error: {e}",
                                   throttle_duration_sec=5.0)


def main(args=None):
    rclpy.init(args=args)
    node = MarkerDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
