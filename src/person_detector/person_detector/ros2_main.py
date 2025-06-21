#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
import cv2
import numpy as np
import pyrealsense2 as rs
from ament_index_python.packages import get_package_share_directory
import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header # For the header in custom message

# 假设你的自定义消息包和消息如上定义
from person_detector_msgs.msg import PersonDetection # 导入自定义消息

from .model.yolo_detector import YOLOv11Detecter
from .utils.remove_bg import remove_background
from .utils.motion_tracker import MultiObjectTracker
from .utils.posture_classification import action_classification
from .utils.photo_judge import judge_realperson_or_photo
from .utils.robust_3d_estimation import robust_3d_estimation_bbox


def draw_label_centered(img, text_str, box, color, font_scale=0.5, thickness=2):
    x1, y1, x2, y2 = map(int, box) #确保坐标是整数
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    text_size, _ = cv2.getTextSize(text_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    tw, th = text_size
    org_x = cx - tw // 2
    org_y = cy + th // 2
    cv2.putText(img, text_str, (org_x, org_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


class PersonDetectorNode(Node):
    def __init__(self, model_path, bg_thresh, margin_ratio, sample_step, show_cv_window=True):
        super().__init__('person_detector_node')
        self.get_logger().info("Person Detector Node initializing...")

        self.model_path = model_path
        self.bg_thresh = bg_thresh
        self.margin_ratio = margin_ratio
        self.sample_step = sample_step
        self.show_cv_window = show_cv_window

        # ROS Publisher for custom detection data
        self.detection_publisher = self.create_publisher(PersonDetection, 'person_detections', 10)

        # Initialize RealSense
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        self.profile = None
        try:
            # Check for connected devices first (optional but good practice)
            ctx = rs.context()
            if not ctx.query_devices():
                self.get_logger().error("No RealSense devices found.")
                raise RuntimeError("No RealSense devices found.")
            self.profile = self.pipeline.start(config)
        except RuntimeError as e:
            self.get_logger().error(f"Failed to start RealSense pipeline: {str(e)}")
            # Consider shutting down the node or rclpy if pipeline is critical
            rclpy.shutdown()
            raise # Re-raise to stop node creation

        self.align = rs.align(rs.stream.color)
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self.get_logger().info(f"RealSense depth_scale: {self.depth_scale}")

        if self.show_cv_window:
            cv2.namedWindow("ROS2 Person Detector", cv2.WINDOW_AUTOSIZE)

        # Initialize detection components
        self.yolo = YOLOv11Detecter(self.model_path)
        self.mot = MultiObjectTracker(max_missing_time=2.0)

        # Create timer for processing loop
        self.timer_period = 0.1  # seconds (for ~10Hz)
        self.timer = self.create_timer(self.timer_period, self.process_frame_callback)
        self.get_logger().info("Person Detector Node initialized and timer started.")

    def process_frame_callback(self):
        try:
            start_time = time.time()
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                self.get_logger().warn("Could not get color or depth frame.")
                return

            color_img = np.asanyarray(color_frame.get_data())
            depth_img = np.asanyarray(depth_frame.get_data())
            depth_img = cv2.medianBlur(depth_img, 5)
            h_img, w_img = depth_img.shape

            depth_bg = remove_background(depth_img, self.depth_scale, self.bg_thresh)
            yolo_detections = self.yolo.detect_person(color_img, conf_thres=0.75)

            current_ros_time = self.get_clock().now().to_msg()
            detected_persons_for_mot = [] # For MultiObjectTracker

            for (cls_id, conf, (x1, y1, x2, y2)) in yolo_detections:
                if cls_id != 0: continue # Only person class

                w_box, h_box = (x2 - x1), (y2 - y1)
                if w_box < 50 or h_box < 50: continue

                mx, my = int(w_box * self.margin_ratio), int(h_box * self.margin_ratio)
                nx1, ny1 = max(0, x1 - mx), max(0, y1 - my)
                nx2, ny2 = min(w_img - 1, x2 + mx), min(h_img - 1, y2 + my)

                # Prepare a base PersonDetection message
                person_msg = PersonDetection()
                person_msg.header.stamp = current_ros_time
                person_msg.header.frame_id = "camera_color_optical_frame" # Or your preferred frame
                person_msg.nx1, person_msg.ny1 = int(nx1), int(ny1)
                person_msg.nx2, person_msg.ny2 = int(nx2), int(ny2)
                person_msg.is_3d_valid = False
                person_msg.x_3d, person_msg.y_3d, person_msg.z_3d = 0.0, 0.0, 0.0
                person_msg.track_id = -1 # Default, will be updated by tracker
                person_msg.is_real_person = False # Default
                person_msg.action = ""


                (X, Y, Z), valid_3d, inliers_3d = robust_3d_estimation_bbox(
                    depth_bg, nx1, ny1, nx2, ny2,
                    self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics(),
                    self.depth_scale, self.sample_step)

                if not valid_3d or inliers_3d is None:
                    if self.show_cv_window:
                        cv2.rectangle(color_img, (nx1, ny1), (nx2, ny2), (0, 0, 0), 2)
                        draw_label_centered(color_img, "Photo (Est)", (nx1, ny1, nx2, ny2), (0,0,0))
                    person_msg.is_real_person = False
                    # Even if not valid for MOT, we can publish the 2D detection
                    self.detection_publisher.publish(person_msg)
                    continue

                label_judge = judge_realperson_or_photo(inliers_3d)
                if label_judge == "photo":
                    if self.show_cv_window:
                        cv2.rectangle(color_img, (nx1, ny1), (nx2, ny2), (0, 0, 0), 2)
                        draw_label_centered(color_img, "Photo (Judge)", (nx1,ny1,nx2,ny2), (0,0,0))
                    person_msg.is_real_person = False
                    self.detection_publisher.publish(person_msg) # Publish photo detection
                else:
                    person_msg.is_real_person = True
                    person_msg.is_3d_valid = True
                    person_msg.x_3d, person_msg.y_3d, person_msg.z_3d = float(X), float(Y), float(Z)
                    # This detection will be passed to MOT.
                    # MOT will handle publishing with track_id later, or we can publish a preliminary one now.
                    # For simplicity, we'll let MOT loop publish the final tracked info.
                    detected_persons_for_mot.append(((X, Y, Z), (nx1, ny1, nx2, ny2), False)) # False indicates not a photo for MOT

            # Multi-Object Tracking
            now_t = time.time()
            self.mot.predict_all(now_t)
            self.mot.update_tracks(detected_persons_for_mot, now_t)
            self.mot.remove_lost_tracks(now_t)

            for tid, trk in self.mot.tracks.items():
                pos_trk, vel_trk, bbox_trk, is_photo_trk = trk.get_state()
                bx1, by1, bx2, by2 = map(int, bbox_trk)

                # Create and publish message for each tracked object
                tracked_person_msg = PersonDetection()
                tracked_person_msg.header.stamp = current_ros_time
                tracked_person_msg.header.frame_id = "camera_color_optical_frame"
                tracked_person_msg.nx1, tracked_person_msg.ny1 = bx1, by1
                tracked_person_msg.nx2, tracked_person_msg.ny2 = bx2, by2
                tracked_person_msg.track_id = int(tid)

                if is_photo_trk: # Should not happen if only real persons are passed to MOT update
                    tracked_person_msg.is_real_person = False
                    if self.show_cv_window:
                        cv2.rectangle(color_img, (bx1, by1), (bx2, by2), (0,0,0), 2)
                        draw_label_centered(color_img, f"T{tid}: Photo", (bx1,by1,bx2,by2), (0,0,0))
                else:
                    tracked_person_msg.is_real_person = True
                    tracked_person_msg.is_3d_valid = True # Assuming tracked means 3D was valid
                    tracked_person_msg.x_3d, tracked_person_msg.y_3d, tracked_person_msg.z_3d = \
                        float(pos_trk[0]), float(pos_trk[1]), float(pos_trk[2])

                    dist_3d = np.sqrt(pos_trk[0]**2 + pos_trk[1]**2 + pos_trk[2]**2)
                    speed_3d = np.sqrt(vel_trk[0]**2 + vel_trk[1]**2 + vel_trk[2]**2)
                    act_str = action_classification(bbox_trk, speed_3d)
                    tracked_person_msg.action = act_str

                    if self.show_cv_window:
                        text_str = f"T{tid} D={dist_3d:.2f} S={speed_3d:.2f} {act_str}"
                        cv2.rectangle(color_img, (bx1, by1), (bx2, by2), (0,0,255), 2)
                        draw_label_centered(color_img, text_str, (bx1,by1,bx2,by2), (0,0,255))
                
                self.detection_publisher.publish(tracked_person_msg)


            if self.show_cv_window:
                end_time = time.time()
                fps = 1.0 / (end_time - start_time) if (end_time - start_time) > 0 else 0
                cv2.putText(color_img, f"FPS: {fps:.2f}", (w_img - 100, h_img - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("ROS2 Person Detector", color_img)
                if cv2.waitKey(1) & 0xFF == 27: # ESC
                    self.get_logger().info("ESC pressed, shutting down...")
                    rclpy.shutdown() # This will stop the spin

        except Exception as e:
            self.get_logger().error(f"Error in process_frame_callback: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            # Consider a more robust error handling, e.g., trying to restart RealSense

    def destroy_node(self):
        self.get_logger().info("Shutting down Person Detector Node...")
        if self.timer:
            self.timer.cancel()
        if hasattr(self, 'pipeline') and self.pipeline and self.profile: # Check if pipeline was started
            self.get_logger().info("Stopping RealSense pipeline.")
            self.pipeline.stop()
        if self.show_cv_window:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser(description="ROS2 RealSense Person Detection and Tracking")
    
    
    package_share_directory = get_package_share_directory('person_detector')
    default_model_path = os.path.join(package_share_directory, 'model', 'person_detection.engine') # 或者 .pt

    parser.add_argument("--model_path", type=str, default="model/person_detection.engine", help="Path to person_detection model")
    parser.add_argument("--bg_thresh", type=float, default=3.0, help="Background threshold (meters)") # main.py had 30.0, Realsense default is usually meters
    parser.add_argument("--margin_ratio", type=float, default=0.05, help="Margin ratio for bounding box")
    parser.add_argument("--sample_step", type=int, default=3, help="Sample step for 3D estimation")
    parser.add_argument("--no_cv_window", action="store_true", help="Disable OpenCV window display")

    # Use parse_known_args to allow ROS-specific arguments to pass through
    parsed_args, _ = parser.parse_known_args(args=rclpy.utilities.remove_ros_args(args=args if args is not None else []))

    # 检查用户提供的路径是否存在，如果不存在则使用默认路径
    if not os.path.exists(parsed_args.model_path):
        print(f"Warning: Provided model path '{parsed_args.model_path}' not found. Using default: '{default_model_path}'")
        parsed_args.model_path = default_model_path

    node = None
    try:
        node = PersonDetectorNode(
            model_path=parsed_args.model_path,
            bg_thresh=parsed_args.bg_thresh,
            margin_ratio=parsed_args.margin_ratio,
            sample_step=parsed_args.sample_step,
            show_cv_window=not parsed_args.no_cv_window
        )
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt, shutting down.")
    except RuntimeError as e: # Catch RuntimeError from pipeline start
        if node: # If node was partially created
             node.get_logger().error(f"Runtime error during node execution: {e}")
        else: # Error before node object was assigned
             print(f"Runtime error before node fully initialized: {e}")
    except Exception as e:
        if node:
             node.get_logger().fatal(f"Unhandled exception: {e}")
        else:
             print(f"Unhandled exception before node initialized: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if node and rclpy.ok(): # Check if node exists and rclpy is still up
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("ROS2 shutdown complete.")

if __name__ == '__main__':
    main()