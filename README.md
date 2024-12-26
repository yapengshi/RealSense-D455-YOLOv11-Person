# 基于D455相机和YOLOv11的人体目标检测与识别

## 项目简介
本项目基于 Intel RealSense D455 深度相机和 YOLOv11 模型，实现了对人体目标的检测和识别，能够实时检测人体目标并进行多目标跟踪，同时判断目标是真人还是照片，输出真人目标的距离、速度和姿态。

YOLOv11 官方代码网址: [https://github.com/ultralytics/ultralytics/tree/main?tab=readme-ov-file](https://github.com/ultralytics/ultralytics/tree/main?tab=readme-ov-file)

## 算法操作流程
1. 读取 RealSense 帧并对齐。
2. 背景滤除。
3. 使用 YOLOv11 模型检测人体目标。
4. 对检测到的边框进行 3D 点云投影。
5. 使用 RANSAC 和3D分布判断是真人还是照片。
6. 若为照片，绘制黑色矩形框并标记为 "Photo"。
7. 若为真人，进行多目标跟踪，预测和更新目标位置，计算距离和移动速度，进行姿态分类，输出红色矩形框并标记为 "Person"、距离、速度和姿态。

## 硬件配置
- Intel RealSense D455 深度相机

## 工程结构
```
project/
│
├── main.py
├── model/
│   └── yolo_detector.py
├── utils/
│   ├── remove_bg.py
│   ├── motion_tracker.py
│   ├── posture_classification.py
│   ├── photo_judge.py
│   └── robust_3d_estimation.py
├── README.md
└── requirements.txt

```

## 各模块文件的作用和使用方法

### `main.py`
程序入口文件，综合调用各个模块实现完整的算法流程。可以通过命令行参数设置一些超参数，如模型路径、背景阈值、边框扩展比例和 3D 估计采样步长。

使用方法：
```bash
python main.py --model_path <模型路径> --bg_thresh <背景阈值> --margin_ratio <边框扩展比例> --sample_step <采样步长>
```

### `model/yolo_detector.py`
包含 YOLOv11 检测器类，用于对输入图像进行人体目标检测。

### `utils/remove_bg.py`
提供 `remove_background` 函数，用于背景滤除。

### `utils/motion_tracker.py`
包含多目标跟踪器类 `MultiObjectTracker`，用于对检测到的目标进行跟踪。

### `utils/posture_classification.py`
提供 `action_classification` 函数，用于姿态分类。

### `utils/photo_judge.py`
提供 `judge_realperson_or_photo` 函数，用于判断检测到的目标是真人还是照片。

### `utils/robust_3d_estimation.py`
提供 `robust_3d_estimation_bbox` 函数，根据深度图和边框坐标进行 3D 点云估计，并去除离群点。

## 运行示例
```bash
python main.py --model_path model/yolo11n.pt --bg_thresh 3.0 --margin_ratio 0.1 --sample_step 3
```

## 注意事项
- 确保电脑已安装相关依赖库。若未安装，可通过以下命令安装：
```bash
pip install -r requirements.txt
```
- 确保相机已正确连接并配置。