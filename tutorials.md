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

# Person_Detection 文件重构
核心思路是将您的 Python 源代码编译成二进制的动态链接库（.so 文件），这样其他人只能调用而无法直接查看源码。对于 ROS 2 的 Python 包，我们可以使用 Cython 来实现这个目标。

Cython 可以将 Python 代码（.py）转换成 C 代码（.c），然后编译成机器码（.so）。这不仅能保护源码，还能在一定程度上提升性能。
## 目标：
我们将创建一个可分发的包，其结构大致如下：
```bash
Person_Detection_Distribution/
├── install/                  # 编译好的、可运行的 ROS 工作空间
│   ├── person_detector/      # 你的核心逻辑包
│   │   ├── lib/
│   │   │   └── person_detector/
│   │   │       ├── __init__.so # 编译后的文件
│   │   │       ├── ros2_main.so  # 你的主程序，已编译
│   │   │       ├── yolo_detector.so # 你的模型加载器，已编译
│   │   │       └── utils/
│   │   │           ├── kalman_3d.so # 所有 utils 文件都已编译
│   │   │           ├── motion_tracker.so
│   │   │           └── ...
│   │   ├── share/
│   │   │   └── person_detector/
│   │   │       └── model/
│   │   │           └── yolo11n.engine # 模型文件
│   │   └── ...                 # 其他 ROS 生成的文件
│   ├── person_detector_msgs/ # 消息包不受影响
│   └── setup.bash            # 用户需要 source 的入口
├── models/
│   └── yolo11n.engine          # 将模型文件单独存放，方便管理
├── requirements.txt          # 依赖项
└── README_FOR_USER.md        # 给用户的说明文档
```
## 修改步骤
1. 在 src 目录下创建一个新的包，我们称之为 person_detector;
2. 将您的核心逻辑 Python 文件移动到这个新包中。
这是修改后的 src 目录结构：
```bash
.
└── src/
    ├── person_detector/
    │   ├── package.xml             # 新建：包信息
    │   ├── setup.py                # 新建：编译和安装配置
    │   ├── setup.cfg               # 新建：打包配置
    │   ├── person_detector/        # Python 模块目录
    │   │   ├── __init__.py         # Python 包标识
    │   │   ├── ros2_main.py        # 你的主程序
    │   │   ├── model/              # 模型加载器模块
    │   │   │   ├── __init__.py
    │   │   │   └── yolo_detector.py
    │   │   └── utils/              # 工具函数模块
    │   │       ├── __init__.py
    │   │       ├── kalman_3d.py
    │   │       ├── motion_tracker.py
    │   │       ├── photo_judge.py
    │   │       ├── posture_classification.py
    │   │       ├── remove_bg.py
    │   │       └── robust_3d_estimation.py
    │   └── resource/
    │       └── person_detector     # ament 索引所需
    │
    └── person_detector_msgs/
        ├── CMakeLists.txt
        ├── msg/
        │   └── PersonDetection.msg
        └── package.xml
```
**注意：**
- 原来的 ros2_main.py 被移动到了 src/person_detector/person_detector/。
- 原来的 model/ 和 utils/ 文件夹也被移动到了 src/person_detector/person_detector/ 下，并添加 __init__.py 使它们成为 Python 子模块。
- 模型文件（.pt, .engine）不要放在这里，我们稍后会通过 setup.py 处理。

修改 ros2_main.py 的导入路径
```python
from utils.remove_bg import remove_background
from model.yolo_detector import YOLOv11Detecter
from utils.motion_tracker import MultiObjectTracker
# ... etc
修改为相对导入：
# ros2_main.py
from .model.yolo_detector import YOLOv11Detecter
from .utils.remove_bg import remove_background
from .utils.motion_tracker import MultiObjectTracker
from .utils.posture_classification import action_classification
from .utils.photo_judge import judge_realperson_or_photo
from .utils.robust_3d_estimation import robust_3d_estimation_bbox
# ... 其他导入保持不变
```
修改src/person_detector/person_detector/utils/motion_tracker.py的导入路径
```python
from .kalman_3d import KalmanFilter3D
修改为相对导入：
from .kalman_3d import KalmanFilter3D
```
第 5 步：修改 ros2_main.py 中模型路径的加载方式
```python
# 在 main 函数中修改
import os
from ament_index_python.packages import get_package_share_directory

def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser(...)
    
    # 获取包的 share 目录路径
    package_share_directory = get_package_share_directory('person_detector')
    # 默认模型路径
    default_model_path = os.path.join(package_share_directory, 'model', 'person_detector.engine')

    parser.add_argument("--model_path", type=str, default=default_model_path, help="Path to YOLO model")
    # ... 其他参数 ...

    parsed_args, _ = parser.parse_known_args(...)
    
    # 检查用户提供的路径是否存在，如果不存在则使用默认路径
    if not os.path.exists(parsed_args.model_path):
        print(f"Warning: Provided model path '{parsed_args.model_path}' not found. Using default: '{default_model_path}'")
        parsed_args.model_path = default_model_path

    # ... 后续代码 ...
    node = PersonDetectorNode(
        model_path=parsed_args.model_path,
        # ...
    )
    # ...
```


## Person_Detection 使用
```bash
conda activate person_env
cd /home/booster/Workspace/Person_Detection
colcon build --cmake-args -DPython3_EXECUTABLE=$(which python)
source install/setup.bash
ros2 run person_detector person_detector_node
```

Note:
1. 修改`install/person_detector/lib/person_detector/person_detector_node`首行：
```
#!/home/booster/Workspace/miniconda3/envs/person_env/bin/python # 修改为conda环境的python路径`which python`
```
2. 在主目录下创建`models`文件夹，并将`yolo11n.engine`文件放入其中更名为`person_detector.engine`。