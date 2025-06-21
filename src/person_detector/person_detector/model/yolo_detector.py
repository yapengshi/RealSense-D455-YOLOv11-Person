# -*- coding: utf-8 -*-
"""
yolo_detector.py

功能：
  - 提供 YOLOv11Detecter 类，加载 yolov11n.pt 等模型，并对输入图像进行 person 检测。
  - 返回 (cls_id, conf, (x1, y1, x2, y2)) 格式列表。

算法细节：
  - 使用 ultralytics 库加载 YOLO 模型；
  - 检测后只关注 (cls_id, conf, bbox) 三元组，并可根据 conf_thres 过滤结果。

使用方法：
  - from yolo_detector import YOLOv11Detecter
  - detector = YOLOv11Detecter("yolo11n.pt")
  - detections = detector.detect_person(img, conf_thres=0.6)
"""

from ultralytics import YOLO


class YOLOv11Detecter:
    def __init__(self, model_path="yolo11n.pt"):
        """
        初始化 YOLOv11Detecter 类，加载指定路径的 YOLO 模型。

        :param model_path: 模型文件路径，默认值为 "yolo11n.pt"
        """
        self.model = YOLO(model_path)  # 加载 YOLO 模型

    def detect_person(self, image, conf_thres=0.6):
        """
        对输入图像进行 person 检测。

        :param image: 输入图像 (numpy array)
        :param conf_thres: 置信度阈值，默认值为 0.6
        :return: 检测结果列表，每个元素为 (cls_id, conf, (x1, y1, x2, y2))
        """
        results = self.model(image, conf=conf_thres)  # 使用模型进行检测
        boxes = results[0].boxes  # 获取检测到的边框
        out_list = []
        for b in boxes:
            xyxy = b.xyxy[0].cpu().numpy()  # 获取边框坐标
            conf_ = float(b.conf.item())  # 获取置信度
            cls_id = int(b.cls.item())  # 获取类别 ID
            x1, y1, x2, y2 = map(int, xyxy)  # 将坐标转换为整数
            out_list.append((cls_id, conf_, (x1, y1, x2, y2)))  # 添加到输出列表
        return out_list  # 返回检测结果列表

