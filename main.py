# -*- coding: utf-8 -*-
"""
main.py

功能：
  - 程序入口文件，综合调用:
    1) remove_bg.remove_background: 背景滤除
    2) yolo_detector.YOLOv11Detecter: YOLOv11模型进行人体目标检测
    3) motion_tracker.MultiObjectTracker: 多目标跟踪
    4) posture_classification.action_classification: 姿态分类
    5) photo_judge.judge_realperson_or_photo: 判断是真人还是照片
    6) robust_3d_estimation_bbox: 3D位置估计

  - 在此可对一些超参数进行设置，如:
    BG_THRESH: 背景阈值，默认3.0
    MARGIN_RATIO: 边框扩展比例，默认0.1
    SAMPLE_STEP: 3D估计采样步长，默认3

算法流程：
  - 读取RealSense帧 => 对齐 => 背景滤除
  - YOLO检测出人体目标 => 对bbox做点云投影 => RANSAC + distribution => 判断是真人还是照片
  - 若为照片 => 则黑色矩形框 + Photo标签
  - 若为真人 => 进入多目标跟踪 => predict+update => 得到距离和移动速度 => 做姿态分类 => 输出红色矩形框 + Person标签+距离+速度+姿态

使用方法：
  - python main.py --model_path <模型路径> --bg_thresh <背景阈值> --margin_ratio <边框扩展比例> --sample_step <采样步长>
"""

import argparse
import time
import cv2
import numpy as np
import pyrealsense2 as rs
from utils.remove_bg import remove_background
from model.yolo_detector import YOLOv11Detecter
from utils.motion_tracker import MultiObjectTracker
from utils.posture_classification import action_classification
from utils.photo_judge import judge_realperson_or_photo
from utils.robust_3d_estimation import robust_3d_estimation_bbox


def draw_label_centered(img, text_str, box, color, font_scale=0.5, thickness=2):
    """
    在图像上绘制居中的标签文本
    :param img: 输入图像
    :param text_str: 要绘制的文本
    :param box: 文本框 (x1, y1, x2, y2)
    :param color: 文本颜色
    :param font_scale: 字体大小比例
    :param thickness: 文本厚度
    """
    x1, y1, x2, y2 = box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    text_size, _ = cv2.getTextSize(text_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    tw, th = text_size
    org_x = cx - tw // 2
    org_y = cy + th // 2

    cv2.putText(img, text_str, (org_x, org_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


def main(model_path, bg_thresh, margin_ratio, sample_step):
    """
    主函数
    :param model_path: YOLO模型路径
    :param bg_thresh: 背景阈值
    :param margin_ratio: 边框扩展比例
    :param sample_step: 3D估计采样步长
    """
    # First check if any RealSense devices are connected
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("[ERROR] No RealSense devices found. Please connect a RealSense camera.")
        print("[INFO] Connected USB devices can be checked with: lsusb")
        return

    pipeline = rs.pipeline()  # 创建RealSense管道
    config = rs.config()  # 创建配置对象
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 启用彩色流
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 启用深度流
    
    try:
        profile = pipeline.start(config)  # 启动管道并获取配置文件
    except RuntimeError as e:
        print("[ERROR] Failed to start pipeline:", str(e))
        print("[TROUBLESHOOTING] Possible solutions:")
        print("1. Ensure camera is properly connected via USB 3.0")
        print("2. Check if RealSense drivers are installed")
        print("3. Verify user has permission to access the device")
        print("4. Try a different USB port")
        return
    align = rs.align(rs.stream.color)  # 创建对齐对象，用于对齐深度帧和彩色帧

    depth_sensor = profile.get_device().first_depth_sensor()  # 获取深度传感器
    depth_scale = depth_sensor.get_depth_scale()  # 获取深度比例
    print("[INFO] depth_scale=", depth_scale)  # 打印深度比例

    cv2.namedWindow("RealSense-D455-YOLOv11-Person-Master", cv2.WINDOW_AUTOSIZE)  # 创建显示窗口

    yolo = YOLOv11Detecter(model_path)  # 初始化YOLO检测器
    mot = MultiObjectTracker(max_missing_time=2.0)  # 初始化多目标跟踪器, 最大丢失时间为2.0秒

    try:
        while True:
            start_time = time.time()  # 记录开始时间
            frames = pipeline.wait_for_frames()  # 等待获取帧
            aligned_frames = align.process(frames)  # 对齐帧
            color_frame = aligned_frames.get_color_frame()  # 获取彩色帧
            depth_frame = aligned_frames.get_depth_frame()  # 获取深度帧
            if not color_frame or not depth_frame:
                continue  # 如果没有获取到帧，则继续循环

            color_img = np.asanyarray(color_frame.get_data())  # 将彩色帧转换为numpy数组
            depth_img = np.asanyarray(depth_frame.get_data())  # 将深度帧转换为numpy数组

            # 对深度图像做预处理，使用中值滤波减少反光和噪声
            depth_img = cv2.medianBlur(depth_img, 5)
            h_img, w_img = depth_img.shape  # 获取深度图像的高度和宽度

            # 背景滤除
            depth_bg = remove_background(depth_img, depth_scale, bg_thresh)

            # YOLO检测
            dets = yolo.detect_person(color_img, conf_thres=0.75) # 可调整置信度阈值

            real_dets = []
            for (cls_id, conf, (x1, y1, x2, y2)) in dets:
                if cls_id != 0:
                    continue  # 如果检测到的不是人，则继续循环
                w_box = (x2 - x1)  # 计算边框宽度
                h_box = (y2 - y1)  # 计算边框高度
                if w_box < 50 or h_box < 50:  # 人物远近判断，小于50像素则认为是背景,可调整
                    continue  # 如果边框太小，则继续循环
                mx = int(w_box * margin_ratio)  # 计算边框扩展宽度
                my = int(h_box * margin_ratio)  # 计算边框扩展高度
                nx1 = max(0, x1 - mx)  # 计算扩展后的边框左上角x坐标
                ny1 = max(0, y1 - my)  # 计算扩展后的边框左上角y坐标
                nx2 = min(w_img - 1, x2 + mx)  # 计算扩展后的边框右下角x坐标
                ny2 = min(h_img - 1, y2 + my)  # 计算扩展后的边框右下角y坐标

                # 3D位置估计
                (X, Y, Z), valid, inliers_3d = robust_3d_estimation_bbox(
                    depth_bg, nx1, ny1, nx2, ny2,
                    profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics(),
                    depth_scale, sample_step)
                if not valid or inliers_3d is None:
                    cv2.rectangle(color_img, (nx1, ny1), (nx2, ny2), (0, 0, 0), 2)  # 绘制黑色矩形框
                    draw_label_centered(color_img, "Photo", (nx1, ny1, nx2, ny2),
                                        color=(0, 0, 0), font_scale=0.5, thickness=2)  # 绘制标签
                    continue  # 如果3D估计无效，则继续循环

                # 判断是真人还是照片
                label_judge = judge_realperson_or_photo(inliers_3d)
                if label_judge == "photo":
                    cv2.rectangle(color_img, (nx1, ny1), (nx2, ny2), (0, 0, 0), 2)  # 绘制黑色矩形框
                    draw_label_centered(color_img, "Photo", (nx1, ny1, nx2, ny2),
                                        color=(0, 0, 0), font_scale=0.5, thickness=2)  # 绘制标签
                else:
                    real_dets.append(((X, Y, Z), (nx1, ny1, nx2, ny2), False))  # 添加真实检测结果

            now_t = time.time()  # 获取当前时间
            mot.predict_all(now_t)  # 预测所有目标
            mot.update_tracks(real_dets, now_t)  # 更新跟踪目标
            mot.remove_lost_tracks(now_t)  # 移除丢失的目标

            for tid, trk in mot.tracks.items():
                pos, vel, bbox, is_photo = trk.get_state()  # 获取跟踪目标的状态
                bx1, by1, bx2, by2 = bbox
                if is_photo:
                    cv2.rectangle(color_img, (bx1, by1), (bx2, by2), (0, 0, 0), 2)  # 绘制黑色矩形框
                    draw_label_centered(color_img, "Photo", (bx1, by1, bx2, by2),
                                        color=(0, 0, 0), font_scale=0.5, thickness=2)  # 绘制标签
                else:
                    dist_3d = np.sqrt(pos[0] ** 2 + pos[1] ** 2 + pos[2] ** 2)  # 计算3D距离
                    speed_3d = np.sqrt(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2)  # 计算3D速度
                    act_str = action_classification((bx1, by1, bx2, by2), speed_3d)  # 姿态分类
                    text_str = f"Person Dist = {dist_3d:.2f}m Spd = {speed_3d:.2f}m/s Act : {act_str}"  # 构建标签文本
                    cv2.rectangle(color_img, (bx1, by1), (bx2, by2), (0, 0, 255), 2)  # 绘制红色矩形框
                    draw_label_centered(color_img, text_str, (bx1, by1, bx2, by2),
                                        color=(0, 0, 255), font_scale=0.5, thickness=2)  # 绘制标签

            end_time = time.time()  # 记录结束时间
            fps = 1.0 / (end_time - start_time)  # 计算帧率
            cv2.putText(color_img, f"FPS: {fps:.2f}", (w_img - 120, h_img - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)  # 绘制帧率标签
            cv2.imshow("RealSense-D455-YOLOv11-Person-Master", color_img)  # 显示图像
            if cv2.waitKey(1) & 0xFF == 27:
                break  # 按下ESC键退出循环

    finally:
        pipeline.stop()  # 停止管道
        cv2.destroyAllWindows()  # 销毁所有窗口


if __name__ == "__main__":
    # 创建ArgumentParser对象，用于解析命令行参数
    parser = argparse.ArgumentParser(description="RealSense D455 YOLOv11 Person Detection")  # 添加描述信息
    parser.add_argument("--model_path", type=str, default="model/yolo11n.pt", help="Path to YOLO model")  # 添加模型路径参数
    parser.add_argument("--bg_thresh", type=float, default=30.0, help="Background threshold")  # 添加背景阈值参数
    parser.add_argument("--margin_ratio", type=float, default=0.05, help="Margin ratio for bounding box")  # 添加边框扩展比例参数
    parser.add_argument("--sample_step", type=int, default=3, help="Sample step for 3D estimation")  # 添加3D估计采样步长参数

    args = parser.parse_args()  # 解析命令行参数
    main(args.model_path, args.bg_thresh, args.margin_ratio, args.sample_step)  # 调用主函数并传递参数
