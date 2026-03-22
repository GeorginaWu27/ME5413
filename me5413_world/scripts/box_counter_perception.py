#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import json
import yaml
import numpy as np
import cv2
import easyocr
import rospy
import tf

from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, PointStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from std_msgs.msg import String, Bool

class BoxCounterPerception:
    def __init__(self):
        # =========================
        # Parameters
        # =========================
        self.rate = rospy.get_param("~rate", 10)

        self.image_topic = rospy.get_param("~image_topic", "/front/image_raw")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/front/camera_info")
        self.scan_topic = rospy.get_param("~scan_topic", "/front/scan")
        self.odom_topic = rospy.get_param("~odom_topic", "/final_slam/odom")

        self.camera_frame_override = rospy.get_param("~camera_frame", "")
        self.lidar_frame = rospy.get_param("~lidar_frame", "front_laser")
        self.map_frame = rospy.get_param("~map_frame", "map")

        self.use_gpu = rospy.get_param("~use_gpu", True)

        # OCR thresholds
        self.ocr_conf_thresh = rospy.get_param("~ocr_conf_thresh", 0.90)
        self.min_diag_len = rospy.get_param("~min_diag_len", 45.0)
        self.max_text_len = rospy.get_param("~max_text_len", 1)

        # Recognition stability
        self.required_stable_hits = rospy.get_param("~required_stable_hits", 3)
        self.same_obs_time_window = rospy.get_param("~same_obs_time_window", 1.0)
        self.min_digit_votes = rospy.get_param("~min_digit_votes", 2)

        # LiDAR matching
        self.front_range_min = rospy.get_param("~front_range_min", 0.15)
        self.front_range_max = rospy.get_param("~front_range_max", 8.0)
        self.search_half_window = rospy.get_param("~search_half_window", 2)
        self.range_offset = rospy.get_param("~range_offset", 0.0)

        # Dedup / track association
        self.pending_match_radius = rospy.get_param("~pending_match_radius", 0.35)
        self.track_match_radius = rospy.get_param("~track_match_radius", 0.6)
        self.confirmed_reuse_radius = rospy.get_param("~confirmed_reuse_radius", 0.8)
        self.confirmed_lock_radius = rospy.get_param("~confirmed_lock_radius", 1.1)

        # =========================
        # Two-stage box counting
        # Stage 1: mapping / lidar-only box slot collection
        # Stage 2: camera counting only after external enable signal
        # =========================
        self.enable_counting_topic = rospy.get_param("~enable_counting_topic", "/percep/enable_counting")
        self.counting_enabled = rospy.get_param("~counting_enabled_initial", False)

        # lidar-only box-slot extraction
        self.box_slot_cluster_radius = rospy.get_param("~box_slot_cluster_radius", 0.45)
        self.box_slot_merge_radius = rospy.get_param("~box_slot_merge_radius", 0.60)
        self.box_slot_confirm_hits = rospy.get_param("~box_slot_confirm_hits", 6)

        # rough physical size filter for lidar clusters (meters)
        self.box_size_min = rospy.get_param("~box_size_min", 0.2)
        self.box_size_max = rospy.get_param("~box_size_max", 1.0)

        # when counting is enabled, OCR observation must match existing lidar slot
        self.slot_assign_radius = rospy.get_param("~slot_assign_radius", 1.0)

        # Cone trigger
        self.enable_cone_trigger = rospy.get_param("~enable_cone_trigger", True)
        self.cone_trigger_topic = rospy.get_param("~cone_trigger_topic", "/cmd_open_cone")
        self.cone_trigger_distance = rospy.get_param("~cone_trigger_distance", 1.0)
        self.cone_trigger_cooldown = rospy.get_param("~cone_trigger_cooldown", 2.0)

        # HSV threshold for orange cone
        self.cone_h_low = rospy.get_param("~cone_h_low", 3)
        self.cone_h_high = rospy.get_param("~cone_h_high", 28)
        self.cone_s_low = rospy.get_param("~cone_s_low", 60)
        self.cone_s_high = rospy.get_param("~cone_s_high", 255)
        self.cone_v_low = rospy.get_param("~cone_v_low", 40)
        self.cone_v_high = rospy.get_param("~cone_v_high", 255)

        self.cone_min_area = rospy.get_param("~cone_min_area", 500)
        self.cone_min_aspect = rospy.get_param("~cone_min_aspect", 0.35)
        self.cone_max_aspect = rospy.get_param("~cone_max_aspect", 3.0)

        self.last_cone_trigger_time = -1e9

        # Optional floor1 bounding box filter
        self.use_floor_filter = rospy.get_param("~use_floor_filter", False)
        self.floor_x_min = rospy.get_param("~floor_x_min", -1e9)
        self.floor_x_max = rospy.get_param("~floor_x_max", 1e9)
        self.floor_y_min = rospy.get_param("~floor_y_min", -1e9)
        self.floor_y_max = rospy.get_param("~floor_y_max", 1e9)

        # Optional central ROI to reduce false positives
        self.use_center_roi = rospy.get_param("~use_center_roi", False)
        self.center_roi_w_ratio = rospy.get_param("~center_roi_w_ratio", 0.8)
        self.center_roi_h_ratio = rospy.get_param("~center_roi_h_ratio", 0.8)

        # Visualization
        self.debug_view = rospy.get_param("~debug_view", True)
        self.debug_image_topic = rospy.get_param("~debug_image_topic", "/percep/debug_image")

        # Output
        self.output_yaml = rospy.get_param(
            "~output_yaml",
            os.path.expanduser("~/5413/ME5413_Final_Project/src/me5413_world/records/box_counts.yaml")
        )
        self.records_topic = rospy.get_param("~records_topic", "/percep/numbers")
        self.target_pose_topic = rospy.get_param("~target_pose_topic", "/percep/pose")

        # =========================
        # Internal State
        # =========================
        self.bridge = CvBridge()

        self.img_curr = None
        self.curr_odom = None
        self.scan_curr = None
        self.scan_msg_curr = None
        self.scan_params_curr = None

        # 0-9 是否出现过
        self.num_detect_result = [0] * 10

        # 最终稳定的箱子 track
        # each:
        # {
        #   "id": int,
        #   "x": float,
        #   "y": float,
        #   "seen_count": int,
        #   "score": float,
        #   "votes": {0:0,1:0,...,9:0},
        #   "assigned_digit": int,
        #   "assigned_votes": int,
        #   "confirmed": bool
        # }
        self.box_tracks = []
        self.next_track_id = 0

        # 临时观测缓存，用于时间稳定性
        # each:
        # {
        #   "x": float,
        #   "y": float,
        #   "score": float,
        #   "hits": int,
        #   "t_last": float,
        #   "votes": {0:0,...,9:0}
        # }
        self.pending_observations = []

        # Prebuilt lidar box slots in map frame
        # each:
        # {
        #   "id": int,
        #   "x": float,
        #   "y": float,
        #   "hits": int,
        #   "confirmed": bool,
        #   "digit_votes": {0:0,...,9:0},
        #   "assigned_digit": None or int,
        #   "assigned_votes": int,
        #   "counted_once": bool
        # }
        self.box_slots = []
        self.next_box_slot_id = 0

        self.counts = {i: 0 for i in range(10)}

        # 读取次数统计：每个数字被“有效读取”了多少次
        self.read_counts = {i: 0 for i in range(10)}

        # 当前读取次数最多的数字
        self.most_read_digit = None
        self.most_read_count = 0

        # 同一个 record 的重复计数冷却时间，避免停在同一箱子前面疯狂累加
        self.read_event_cooldown = rospy.get_param("~read_event_cooldown", 0.5)

        # Camera info
        rospy.loginfo("Waiting for camera info on %s ...", self.camera_info_topic)
        camera_info = rospy.wait_for_message(self.camera_info_topic, CameraInfo)
        self.intrinsic = np.array(camera_info.K).reshape(3, 3)
        self.projection = np.array(camera_info.P).reshape(3, 4)
        self.distortion = np.array(camera_info.D)

        if self.camera_frame_override:
            self.img_frame = self.camera_frame_override
        else:
            self.img_frame = camera_info.header.frame_id

        rospy.loginfo("Camera frame: %s", self.img_frame)
        rospy.loginfo("Lidar frame: %s", self.lidar_frame)
        rospy.loginfo("Map frame: %s", self.map_frame)

        # OCR
        self.ocr_detector = easyocr.Reader(["en"], gpu=self.use_gpu)

        # TF
        self.tf_listener = tf.TransformListener()

        # Publishers
        self.records_pub = rospy.Publisher(self.records_topic, String, queue_size=1)
        self.target_pose_pub = rospy.Publisher(self.target_pose_topic, PoseStamped, queue_size=1)

        self.records_pub = rospy.Publisher(self.records_topic, String, queue_size=1)
        self.target_pose_pub = rospy.Publisher(self.target_pose_topic, PoseStamped, queue_size=1)
        self.debug_image_pub = rospy.Publisher(self.debug_image_topic, Image, queue_size=1)
        
        self.cone_trigger_pub = rospy.Publisher(self.cone_trigger_topic, Bool, queue_size=1)

        # Subscribers
        self.img_sub = rospy.Subscriber(self.image_topic, Image, self.img_callback, queue_size=1)
        self.scan_sub = rospy.Subscriber(self.scan_topic, LaserScan, self.scan_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=1)
        self.enable_counting_sub = rospy.Subscriber(
            self.enable_counting_topic, Bool, self.enable_counting_callback, queue_size=1
        )

        rospy.on_shutdown(self.on_shutdown)
        rospy.loginfo("BoxCounterPerception initialized.")

    # ==================================
    # Callbacks
    # ==================================
    def odom_callback(self, msg):
        self.curr_odom = msg

    def img_callback(self, msg):
        try:
            self.img_curr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logwarn_throttle(2.0, "cv_bridge failed: %s", str(e))
            self.img_curr = None

    def scan_callback(self, msg):
        self.scan_curr = msg.ranges
        self.scan_msg_curr = msg
        self.scan_params_curr = {
            "angle_min": msg.angle_min,
            "angle_max": msg.angle_max,
            "angle_increment": msg.angle_increment
        }

    def enable_counting_callback(self, msg):
        self.counting_enabled = bool(msg.data)
        rospy.loginfo("Counting enabled set to: %s", self.counting_enabled)

    # ==================================
    # Main Loop
    # ==================================
    def run(self):
        rate = rospy.Rate(self.rate)

        while not rospy.is_shutdown():
            if self.img_curr is None or self.scan_curr is None or self.scan_params_curr is None or self.scan_msg_curr is None:
                rate.sleep()
                continue

            frame = self.img_curr.copy()
            vis_frame = frame.copy()

            # Stage 1: always keep collecting lidar box slots in map frame
            self.update_box_slots_from_lidar()

            # Only start OCR-based counting after external enable signal
            detections = []
            if self.counting_enabled:
                detections, vis_frame = self.detect_digits(frame)

            if self.enable_cone_trigger and self.counting_enabled:
                cone_detections, cone_mask = self.detect_cones(frame)
                if len(cone_detections) > 0:
                    # 画出所有候选
                    if self.debug_view:
                        for det in cone_detections:
                            x1, y1, x2, y2 = det["bbox"]
                            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 140, 255), 2)
                            cv2.putText(
                                vis_frame,
                                f"cone area={det['area']:.0f}",
                                (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 140, 255),
                                1,
                                cv2.LINE_AA,
                            )

                    self.maybe_trigger_cone_open(cone_detections[0], vis_frame)

            for det in detections:
                digit = det["digit"]
                score = det["score"]
                bbox = det["bbox"]

                x1, y1, x2, y2 = bbox
                center_u = 0.5 * (x1 + x2)
                center_v = y2 - 0.1 * (y2 - y1)

                yaw_lidar = self.compute_bearing_in_lidar(center_u, center_v)
                if yaw_lidar is None:
                    continue

                distance, beam_idx = self.get_scan_range_by_yaw(yaw_lidar)
                if distance is None:
                    continue

                distance = distance - self.range_offset
                if distance < self.front_range_min or distance > self.front_range_max:
                    continue

                map_x, map_y = self.project_detection_to_map(distance, yaw_lidar)
                if map_x is None:
                    continue

                if self.use_floor_filter and not self.is_valid_floor_point(map_x, map_y):
                    continue

                if not self.counting_enabled:
                    continue

                stable_obs = self.update_pending_observation(digit, map_x, map_y, score)
                if stable_obs is None:
                    continue

                slot = self.assign_digit_to_box_slot(stable_obs)

                if slot is not None and slot["assigned_digit"] is not None:
                    goal_p = PoseStamped()
                    goal_p.header.frame_id = self.map_frame
                    goal_p.header.stamp = rospy.Time.now()
                    goal_p.pose.position.x = slot["x"]
                    goal_p.pose.position.y = slot["y"]
                    goal_p.pose.position.z = 0.0
                    if self.curr_odom is not None:
                        goal_p.pose.orientation = self.curr_odom.pose.pose.orientation
                    else:
                        goal_p.pose.orientation.w = 1.0
                    self.target_pose_pub.publish(goal_p)

                if self.debug_view:
                    x1, y1, x2, y2 = bbox
                    cv2.putText(
                        vis_frame,
                        f"beam={beam_idx} range={distance:.2f} map=({map_x:.2f},{map_y:.2f})",
                        (x1, max(20, y1 - 35)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

            self.recompute_counts()
            self.publish_summary()
            self.save_results()

            if self.debug_view:
                self.draw_tracks(vis_frame)
                self.publish_debug_image(vis_frame)

            rate.sleep()

    # ==================================
    # OCR / Detection
    # ==================================
    def detect_digits(self, image):
        """
        returns:
            detections = [
                {
                    "digit": int,
                    "score": float,
                    "bbox": [x1, y1, x2, y2]
                }, ...
            ]
        """
        draw = image.copy()

        if self.use_center_roi:
            h, w = image.shape[:2]
            roi_w = int(w * self.center_roi_w_ratio)
            roi_h = int(h * self.center_roi_h_ratio)
            x0 = (w - roi_w) // 2
            y0 = (h - roi_h) // 2
            crop = image[y0:y0 + roi_h, x0:x0 + roi_w]
            offset_x, offset_y = x0, y0
            cv2.rectangle(draw, (x0, y0), (x0 + roi_w, y0 + roi_h), (255, 0, 0), 2)
        else:
            crop = image
            offset_x, offset_y = 0, 0

        results = self.ocr_detector.readtext(
            crop,
            batch_size=2,
            allowlist="0123456789"
        )

        detections = []

        for detection in results:
            pts = detection[0]
            text = detection[1]
            conf = float(detection[2])

            if len(text) > self.max_text_len:
                continue
            if len(text) != 1 or (not text.isdigit()):
                continue
            if conf < self.ocr_conf_thresh:
                continue

            diag_vec = np.array(pts[2]) - np.array(pts[0])
            diag_len = np.linalg.norm(diag_vec)
            if diag_len < self.min_diag_len:
                continue

            x1 = int(pts[0][0]) + offset_x
            y1 = int(pts[0][1]) + offset_y
            x2 = int(pts[2][0]) + offset_x
            y2 = int(pts[2][1]) + offset_y

            det = {
                "digit": int(text),
                "score": conf,
                "bbox": [x1, y1, x2, y2]
            }
            detections.append(det)

            if self.debug_view:
                cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    draw,
                    f"{text} {conf:.2f}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    draw,
                    f"diag={diag_len:.1f}",
                    (x1, max(40, y1 - 32)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

        return detections, draw

    def get_bbox_center(self, bbox):
        x1, y1, x2, y2 = bbox
        u = 0.5 * (x1 + x2)
        v = 0.5 * (y1 + y2)
        return u, v

    # ==================================
    # Geometry: camera -> lidar bearing
    # ==================================
    def compute_bearing_in_lidar(self, u, v):
        direction = np.array([[u], [v], [1.0]], dtype=np.float64)
        try:
            direction = np.dot(np.linalg.inv(self.intrinsic), direction)
        except np.linalg.LinAlgError:
            rospy.logwarn_throttle(2.0, "Intrinsic matrix inversion failed.")
            return None

        p_in_cam = PoseStamped()
        p_in_cam.header.frame_id = self.img_frame
        p_in_cam.header.stamp = rospy.Time(0)
        p_in_cam.pose.position.x = float(direction[0].item())
        p_in_cam.pose.position.y = float(direction[1].item())
        p_in_cam.pose.position.z = float(direction[2].item())
        p_in_cam.pose.orientation.w = 1.0

        try:
            self.tf_listener.waitForTransform(
                self.lidar_frame,
                self.img_frame,
                rospy.Time(0),
                rospy.Duration(0.5)
            )
            transformed = self.tf_listener.transformPose(self.lidar_frame, p_in_cam)
        except Exception as e:
            rospy.logwarn_throttle(2.0, "Transform camera->lidar failed: %s", str(e))
            return None

        yaw = math.atan2(transformed.pose.position.y, transformed.pose.position.x)
        return yaw

    # ==================================
    # LiDAR matching
    # ==================================
    def get_scan_range_by_yaw(self, yaw):
        angle_min = self.scan_params_curr["angle_min"]
        angle_max = self.scan_params_curr["angle_max"]
        angle_inc = self.scan_params_curr["angle_increment"]

        if yaw < angle_min or yaw > angle_max:
            return None, None

        idx_center = int(round((yaw - angle_min) / angle_inc))
        if idx_center < 0 or idx_center >= len(self.scan_curr):
            return None, None

        best_r = None
        best_idx = None

        start_idx = max(0, idx_center - self.search_half_window)
        end_idx = min(len(self.scan_curr) - 1, idx_center + self.search_half_window)

        for idx in range(start_idx, end_idx + 1):
            r = self.scan_curr[idx]
            if not np.isfinite(r):
                continue
            if r < self.front_range_min or r > self.front_range_max:
                continue
            if best_r is None or r < best_r:
                best_r = r
                best_idx = idx

        if best_r is None:
            return None, None

        return float(best_r), int(best_idx)

    # ==================================
    # LiDAR point -> map
    # ==================================
    def project_detection_to_map(self, distance, yaw):
        x_l = distance * math.cos(yaw)
        y_l = distance * math.sin(yaw)

        p_in_lidar = PoseStamped()
        p_in_lidar.header.frame_id = self.lidar_frame
        p_in_lidar.header.stamp = rospy.Time(0)
        p_in_lidar.pose.position.x = x_l
        p_in_lidar.pose.position.y = y_l
        p_in_lidar.pose.position.z = 0.0
        p_in_lidar.pose.orientation.w = 1.0

        try:
            self.tf_listener.waitForTransform(
                self.map_frame,
                self.lidar_frame,
                rospy.Time(0),
                rospy.Duration(0.5)
            )
            p_in_map = self.tf_listener.transformPose(self.map_frame, p_in_lidar)
        except Exception as e:
            rospy.logwarn_throttle(2.0, "Transform lidar->map failed: %s", str(e))
            return None, None

        x = p_in_map.pose.position.x
        y = p_in_map.pose.position.y

        if not np.isfinite(x) or not np.isfinite(y):
            return None, None

        return float(x), float(y)
    
    def project_lidar_point_to_map(self, x_l, y_l):
        p_in_lidar = PointStamped()
        p_in_lidar.header.frame_id = self.lidar_frame
        p_in_lidar.header.stamp = rospy.Time(0)
        p_in_lidar.point.x = x_l
        p_in_lidar.point.y = y_l
        p_in_lidar.point.z = 0.0

        try:
            self.tf_listener.waitForTransform(
                self.map_frame,
                self.lidar_frame,
                rospy.Time(0),
                rospy.Duration(0.5)
            )
            p_in_map = self.tf_listener.transformPoint(self.map_frame, p_in_lidar)
        except Exception as e:
            rospy.logwarn_throttle(2.0, "Transform lidar-point->map failed: %s", str(e))
            return None, None

        x = p_in_map.point.x
        y = p_in_map.point.y
        if not np.isfinite(x) or not np.isfinite(y):
            return None, None

        return float(x), float(y)
    
    def detect_cones(self, image):
        """
        Simple orange cone detector by HSV color + contour filtering.
        returns:
            detections = [
                {
                    "bbox": [x1, y1, x2, y2],
                    "area": float
                }, ...
            ]
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower = np.array([self.cone_h_low, self.cone_s_low, self.cone_v_low], dtype=np.uint8)
        upper = np.array([self.cone_h_high, self.cone_s_high, self.cone_v_high], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower, upper)

        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 31))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        H, W = image.shape[:2]

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.cone_min_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if h <= 0:
                continue

            aspect = float(w) / float(h)
            if aspect < self.cone_min_aspect or aspect > self.cone_max_aspect:
                continue

            # 地面物体先验：至少要在图像下半区附近
            if y + h < 0.45 * H:
                continue

            detections.append({
                "bbox": [x, y, x + w, y + h],
                "area": float(area),
                "bottom": float(y + h),
                "height": float(h),
            })

        # 优先更靠下、更高的候选，而不是只看 area
        detections.sort(
            key=lambda d: (d["bottom"], d["bbox"], d["area"]),
            reverse=True
        )
        return detections, mask
    
    def update_box_slots_from_lidar(self):
        """
        Use current 2D lidar scan to extract obstacle clusters, project them to map,
        and maintain persistent box slots in map frame.
        Camera is NOT involved here.
        """
        msg = self.scan_msg_curr
        if msg is None:
            return

        # 1) convert scan to lidar-frame points
        points = []
        angle = msg.angle_min
        for r in msg.ranges:
            if np.isfinite(r) and self.front_range_min <= r <= self.front_range_max:
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                points.append((x, y))
            angle += msg.angle_increment

        if len(points) < 3:
            return

        # 2) simple sequential clustering in scan order
        clusters = []
        curr = [points[0]]
        for i in range(1, len(points)):
            px, py = points[i - 1]
            qx, qy = points[i]
            d = math.hypot(qx - px, qy - py)
            if d < self.box_slot_cluster_radius:
                curr.append((qx, qy))
            else:
                if len(curr) >= 3:
                    clusters.append(curr)
                curr = [(qx, qy)]
        if len(curr) >= 3:
            clusters.append(curr)

        # 3) keep only roughly "box-sized" clusters
        for cluster in clusters:
            xs = [p[0] for p in cluster]
            ys = [p[1] for p in cluster]
            width = max(xs) - min(xs)
            height = max(ys) - min(ys)
            size = max(width, height)

            if size < self.box_size_min or size > self.box_size_max:
                continue

            cx = float(np.mean(xs))
            cy = float(np.mean(ys))

            mx, my = self.project_lidar_point_to_map(cx, cy)
            if mx is None:
                continue

            if self.use_floor_filter and not self.is_valid_floor_point(mx, my):
                continue

            self.insert_or_update_box_slot(mx, my)

    # ==================================
    # Floor filter
    # ==================================
    def is_valid_floor_point(self, x, y):
        if x < self.floor_x_min or x > self.floor_x_max:
            return False
        if y < self.floor_y_min or y > self.floor_y_max:
            return False
        return True
    
    def insert_or_update_box_slot(self, x, y):
        best_slot = None
        best_dist = 1e9

        for slot in self.box_slots:
            dist = math.hypot(slot["x"] - x, slot["y"] - y)
            if dist < self.box_slot_merge_radius and dist < best_dist:
                best_slot = slot
                best_dist = dist

        if best_slot is None:
            slot = {
                "id": self.next_box_slot_id,
                "x": float(x),
                "y": float(y),
                "hits": 1,
                "confirmed": False,
                "digit_votes": self.empty_votes(),
                "assigned_digit": None,
                "assigned_votes": 0,
                "counted_once": False,
            }
            self.next_box_slot_id += 1
            self.box_slots.append(slot)
            return slot

        best_slot["x"] = 0.9 * best_slot["x"] + 0.1 * x
        best_slot["y"] = 0.9 * best_slot["y"] + 0.1 * y
        best_slot["hits"] += 1
        best_slot["confirmed"] = best_slot["hits"] >= self.box_slot_confirm_hits
        return best_slot

    # ==================================
    # Voting helpers
    # ==================================
    def empty_votes(self):
        return {i: 0 for i in range(10)}

    def best_digit_from_votes(self, votes):
        best_digit = 0
        best_votes = -1
        for d in range(10):
            if votes[d] > best_votes:
                best_digit = d
                best_votes = votes[d]
        return best_digit, best_votes

    # ==================================
    # Stability gating
    # ==================================
    def update_pending_observation(self, digit, x, y, score):
        """
        不再按 digit+位置 匹配 pending。
        只按“位置”匹配一个候选箱子，并对 0-9 做投票。
        """
        now = rospy.Time.now().to_sec()

        # prune old pending items
        new_pending = []
        for item in self.pending_observations:
            if now - item["t_last"] <= self.same_obs_time_window:
                new_pending.append(item)
        self.pending_observations = new_pending

        best_item = None
        best_dist = 1e9

        for item in self.pending_observations:
            dist = math.hypot(item["x"] - x, item["y"] - y)
            if dist < self.pending_match_radius and dist < best_dist:
                best_item = item
                best_dist = dist

        if best_item is None:
            item = {
                "x": float(x),
                "y": float(y),
                "score": float(score),
                "hits": 1,
                "t_last": now,
                "votes": self.empty_votes()
            }
            item["votes"][digit] += 1
            self.pending_observations.append(item)
            return None

        best_item["x"] = 0.8 * best_item["x"] + 0.2 * x
        best_item["y"] = 0.8 * best_item["y"] + 0.2 * y
        best_item["score"] = max(best_item["score"], score)
        best_item["hits"] += 1
        best_item["t_last"] = now
        best_item["votes"][digit] += 1

        assigned_digit, assigned_votes = self.best_digit_from_votes(best_item["votes"])

        if best_item["hits"] >= self.required_stable_hits and assigned_votes >= self.min_digit_votes:
            return {
                "x": float(best_item["x"]),
                "y": float(best_item["y"]),
                "score": float(best_item["score"]),
                "digit": int(assigned_digit),
                "votes": dict(best_item["votes"]),
                "hits": int(best_item["hits"])
            }

        return None

    # ==================================
    # Registry / Counting
    # ==================================
    def maybe_trigger_cone_open(self, det, vis_frame=None):
        now = rospy.Time.now().to_sec()
        if now - self.last_cone_trigger_time < self.cone_trigger_cooldown:
            return False

        bbox = det["bbox"]
        x1, y1, x2, y2 = bbox
        center_u = 0.5 * (x1 + x2)
        center_v = y2 - 0.1 * (y2 - y1)

        yaw_lidar = self.compute_bearing_in_lidar(center_u, center_v)
        if yaw_lidar is None:
            return False

        distance, beam_idx = self.get_scan_range_by_yaw(yaw_lidar)
        if distance is None:
            return False

        if distance > self.cone_trigger_distance:
            return False

        self.cone_trigger_pub.publish(Bool(data=True))
        self.last_cone_trigger_time = now

        if vis_frame is not None and self.debug_view:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 140, 255), 2)
            cv2.putText(
                vis_frame,
                f"CONE OPEN sent, r={distance:.2f}m beam={beam_idx}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 140, 255),
                2,
                cv2.LINE_AA,
            )

        rospy.loginfo_throttle(1.0, "Cone trigger sent: distance=%.2f m", distance)
        return True

    def update_most_read_digit(self):
        max_count = max(self.read_counts.values()) if len(self.read_counts) > 0 else 0
        if max_count <= 0:
            self.most_read_digit = None
            self.most_read_count = 0
            return

        # 若并列，取数字更小的那个，保持确定性
        candidates = [d for d, c in self.read_counts.items() if c == max_count]
        self.most_read_digit = min(candidates)
        self.most_read_count = max_count


    def register_read_event(self, track):
        digit = track["assigned_digit"]
        if digit is None:
            return False

        if track.get("counted_once", False):
            return False

        self.read_counts[digit] += 1
        track["counted_once"] = True
        self.update_most_read_digit()
        return True
    
    def assign_digit_to_box_slot(self, obs):
        """
        During counting stage, OCR observation must match an existing confirmed lidar slot.
        No new slot / no free track creation is allowed here.
        """
        x = obs["x"]
        y = obs["y"]
        obs_votes = obs["votes"]

        best_slot = None
        best_dist = 1e9

        for slot in self.box_slots:
            if not slot["confirmed"]:
                continue
            dist = math.hypot(slot["x"] - x, slot["y"] - y)
            if dist < self.slot_assign_radius and dist < best_dist:
                best_slot = slot
                best_dist = dist

        if best_slot is None:
            return None

        for d in range(10):
            best_slot["digit_votes"][d] += obs_votes.get(d, 0)

        assigned_digit, assigned_votes = self.best_digit_from_votes(best_slot["digit_votes"])
        best_slot["assigned_digit"] = int(assigned_digit)
        best_slot["assigned_votes"] = int(assigned_votes)

        if (not best_slot["counted_once"]) and assigned_votes >= self.min_digit_votes:
            self.read_counts[assigned_digit] += 1
            best_slot["counted_once"] = True
            self.update_most_read_digit()

        return best_slot

    def insert_or_update_track(self, obs):
        x = obs["x"]
        y = obs["y"]
        score = obs["score"]
        obs_votes = obs["votes"]

        # 当前观测最可能的数字
        obs_digit, obs_digit_votes = self.best_digit_from_votes(obs_votes)

        best_track = None
        best_dist = 1e9

        # --------------------------------------------------
        # Step 1: 先查 confirmed 排他区（同数字 + 近距离 => 强制复用）
        # --------------------------------------------------
        if obs_digit is not None:
            for tr in self.box_tracks:
                if not tr["confirmed"]:
                    continue
                if tr["assigned_digit"] is None:
                    continue
                if int(tr["assigned_digit"]) != int(obs_digit):
                    continue

                dist = math.hypot(tr["x"] - x, tr["y"] - y)
                if dist < self.confirmed_lock_radius and dist < best_dist:
                    best_track = tr
                    best_dist = dist

        # --------------------------------------------------
        # Step 2: 如果没落入排他区，再做普通最近邻匹配
        # --------------------------------------------------
        if best_track is None:
            for tr in self.box_tracks:
                dist = math.hypot(tr["x"] - x, tr["y"] - y)
                match_radius = (
                    self.confirmed_reuse_radius if tr["confirmed"]
                    else self.track_match_radius
                )

                if dist < match_radius and dist < best_dist:
                    best_track = tr
                    best_dist = dist

        # --------------------------------------------------
        # Step 3: 没匹配到就新建 track
        # --------------------------------------------------
        if best_track is None:
            tr = {
                "id": self.next_track_id,
                "x": float(x),
                "y": float(y),
                "seen_count": int(obs["hits"]),
                "score": float(score),
                "votes": self.empty_votes(),
                "assigned_digit": None,
                "assigned_votes": 0,
                "confirmed": False,
                "last_read_count_time": -1e9,
                "counted_once": False,
            }
            for d in range(10):
                tr["votes"][d] += obs_votes.get(d, 0)

            self.next_track_id += 1
            self.box_tracks.append(tr)
            best_track = tr

        # --------------------------------------------------
        # Step 4: 匹配到旧 track，则更新
        # --------------------------------------------------
        else:
            # 未确认 track 可以正常更新位置
            if not best_track["confirmed"]:
                best_track["x"] = 0.9 * best_track["x"] + 0.1 * x
                best_track["y"] = 0.9 * best_track["y"] + 0.1 * y

            # confirmed track 建议位置冻结；若你想轻微更新，可改成 0.98/0.02
            # else:
            #     best_track["x"] = 0.98 * best_track["x"] + 0.02 * x
            #     best_track["y"] = 0.98 * best_track["y"] + 0.02 * y

            best_track["seen_count"] += 1
            best_track["score"] = max(best_track["score"], score)
            for d in range(10):
                best_track["votes"][d] += obs_votes.get(d, 0)

        # --------------------------------------------------
        # Step 5: 更新 digit / confirmed 状态
        # --------------------------------------------------
        assigned_digit, assigned_votes = self.best_digit_from_votes(best_track["votes"])

        best_track["assigned_digit"] = int(assigned_digit) if assigned_digit is not None else None
        best_track["assigned_votes"] = int(assigned_votes)
        best_track["confirmed"] = (
            best_track["seen_count"] >= self.required_stable_hits and
            assigned_votes >= self.min_digit_votes
        )

        # confirmed 后只计一次
        if best_track["confirmed"]:
            self.register_read_event(best_track)

        return best_track
    
    def recompute_counts(self):
        """
        Count strictly from confirmed lidar box slots that have assigned digits.
        One slot = one physical box candidate.
        """
        self.counts = {i: 0 for i in range(10)}

        for slot in self.box_slots:
            if not slot["confirmed"]:
                continue
            if slot["assigned_digit"] is None:
                continue

            digit = int(slot["assigned_digit"])
            if 0 <= digit <= 9:
                self.counts[digit] += 1

        self.num_detect_result = [1 if self.counts[i] > 0 else 0 for i in range(10)]


    def publish_summary(self):
        msg = {
            "counts": self.counts,
            "read_counts": self.read_counts,
            "most_read_digit": self.most_read_digit,
            "most_read_count": self.most_read_count,
            "box_slots": self.box_slots,
        }
        self.records_pub.publish(String(data=json.dumps(msg, ensure_ascii=False)))

    def publish_debug_image(self, image_bgr):
        if image_bgr is None:
            return

        try:
            msg = self.bridge.cv2_to_imgmsg(image_bgr, encoding="bgr8")
            msg.header.stamp = rospy.Time.now()
            self.debug_image_pub.publish(msg)
        except Exception as e:
            rospy.logwarn_throttle(2.0, "Failed to publish debug image: %s", str(e))


    def save_results(self):
        os.makedirs(os.path.dirname(self.output_yaml), exist_ok=True)
        data = {
            "counts": self.counts,
            "read_counts": self.read_counts,
            "most_read_digit": self.most_read_digit,
            "most_read_count": self.most_read_count,
            "box_slots": self.box_slots,
            "num_detect_result": self.num_detect_result,
        }
        with open(self.output_yaml, "w") as f:
            yaml.safe_dump(data, f, sort_keys=True)

    def draw_tracks(self, image):
        y0 = 25
        cv2.putText(
            image,
            f"counting_enabled={self.counting_enabled} counts={self.counts}",
            (20, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        y = y0 + 30
        for slot in self.box_slots[:12]:
            color = (0, 255, 0) if slot["confirmed"] else (0, 165, 255)
            text = (
                f"slot={slot['id']} "
                f"digit={slot['assigned_digit']} "
                f"votes={slot['assigned_votes']} "
                f"hits={slot['hits']} "
                f"pos=({slot['x']:.2f},{slot['y']:.2f})"
            )
            cv2.putText(
                image,
                text,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA
            )
            y += 22

    def on_shutdown(self):
        self.recompute_counts()
        self.save_results()


if __name__ == "__main__":
    rospy.init_node("box_counter_perception")
    node = BoxCounterPerception()
    node.run()