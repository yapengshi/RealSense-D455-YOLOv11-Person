# yolo环境配置
[B站保姆级视频教程：Jetson配置YOLOv11环境](https://blog.csdn.net/python_yjys/category_12885034.html)

## realsense 安装
lsusb # 查看usb设备


```bash
pip install numpy==1.26.4 -i https://pypi.tuna.tsinghua.edu.cn/simple # 指定numpy版本
python -c "import numpy; print(numpy.__version__)"  # 应输出 1.26.4
```


## yolo 使用
```bash
yolo predict model=yolo11n.pt source='https://ultralytics.com/images/bus.jpg'
yolo predict task=detect model=yolo11n.engine imgsz=640 source='https://ultralytics.com/images/bus.jpg'
yolo predict task=detect model=yolo11n.pt imgsz=640 source=videos/街道.mp4                       # 原始pytrch模型
```

## 人脸识别
```bash
python main.py --model_path model/yolo11n.pt --bg_thresh 3.0 --margin_ratio 0.1 --sample_step 3    # 原始pytorch模型
# python main.py --model_path model/yolo11n.engine --bg_thresh 3.0 --margin_ratio 0.1 --sample_step 3 # engine加速模型
``` 