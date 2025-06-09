# yolo环境配置
[B站保姆级视频教程：Jetson配置YOLOv11环境](https://blog.csdn.net/python_yjys/category_12885034.html)

## realsense 安装
lsusb # 查看usb设备



```bash
pip install numpy==1.26.4 -i https://pypi.tuna.tsinghua.edu.cn/simple # 指定numpy版本
python -c "import numpy; print(numpy.__version__)"  # 应输出 1.26.4
```

## Booster T1摄像头连接确认
```bash
lsusb -t # 以下输出正确(Driver=) 而非 `Class=Video, Driver=usbfs, 480M`
```
```
Bus 02.Port 1: Dev 1, Class=root_hub, Driver=tegra-xusb/4p, 10000M
    |__ Port 3: Dev 2, If 0, Class=Video, Driver=, 5000M  # 深度流
    |__ Port 3: Dev 2, If 1, Class=Video, Driver=, 5000M  # 红外流 1
    |__ Port 3: Dev 2, If 2, Class=Video, Driver=, 5000M  # 红外流 2
    |__ Port 3: Dev 2, If 3, Class=Video, Driver=, 5000M  # RGB 流
    |__ Port 3: Dev 2, If 4, Class=Video, Driver=, 5000M  # 备用接口
    |__ Port 3: Dev 2, If 5, Class=Human Interface Device, Driver=usbhid, 5000M  # IMU 单元
```

如果输出有问题，或`realsense-view`无法识别相机，则重启相机服务
```bash
sudo systemctl restart booster-daemon-perception # 可能需要[tab]后缀 `.service`
```


## yolo 使用
```bash
# conda activate yolo
# cd /home/booster/Workspace/code/RealSense-D455-YOLOv11-Person-Master
yolo predict model=yolo11n.pt source='https://ultralytics.com/images/bus.jpg'
yolo predict task=detect model=yolo11n.engine imgsz=640 source='https://ultralytics.com/images/bus.jpg'
yolo predict task=detect model=yolo11n.pt imgsz=640 source=videos/街道.mp4                       # 原始pytrch模型
```

## 人脸识别
```bash
python main.py --model_path model/yolo11n.pt --bg_thresh 3.0 --margin_ratio 0.1 --sample_step 3    # 原始pytorch模型
# python main.py --model_path model/yolo11n.engine --bg_thresh 3.0 --margin_ratio 0.1 --sample_step 3 # engine加速模型
``` 

## ros2 版本定制化输出
```bash
colcon build --packages-select person_detector_msgs
source install/setup.bash
python ros2_main.py --model_path model/yolo11n.pt # ros2版本
```