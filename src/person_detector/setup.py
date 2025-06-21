# src/person_detector/setup.py

import os
from glob import glob
from setuptools import setup, Extension
# Cython.Build 可能会提前打印信息，我们尝试延后导入
# from Cython.Build import cythonize
import sys

package_name = 'person_detector'

# 检查当前 setup.py 的执行上下文
# 如果是 colcon 在探测信息，我们就不进行 cythonize
# build, install, bdist_wheel 这些命令才是真正需要编译的时候
is_building = any(arg in sys.argv for arg in ['build', 'install', 'bdist_wheel'])

# 列出所有需要编译成 .so 的 Python 文件
extensions = [
    Extension(
        "person_detector.ros2_main",
        ["person_detector/ros2_main.py"]
    ),
    Extension(
        "person_detector.model.yolo_detector",
        ["person_detector/model/yolo_detector.py"]
    ),
    # 使用通配符编译 utils 目录下的所有 .py 文件
    *([Extension(f"person_detector.utils.{os.path.splitext(os.path.basename(p))[0]}", [p])
       for p in glob('person_detector/utils/*.py') if '__init__' not in p])
]

# 只有在真正构建时，才引入并执行 cythonize
if is_building:
    try:
        from Cython.Build import cythonize
        ext_modules = cythonize(
            extensions,
            compiler_directives={'language_level': "3"}
        )
    except ImportError:
        print("Cython not found. Please install it with 'pip install cython'")
        ext_modules = []
else:
    # 如果只是获取信息，就提供一个空的 ext_modules
    ext_modules = []


setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name, f"{package_name}.utils", f"{package_name}.model"],
    
    # 使用我们上面逻辑判断后的 ext_modules
    ext_modules=ext_modules,

    # 将模型文件等非代码文件安装到 share 目录
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 将模型文件夹安装到 share/person_detector/model
        # 注意路径是相对于 setup.py 的
        (os.path.join('share', package_name, 'model'), glob('../model/*.*')),
    ],
    install_requires=['setuptools', 'cython'], # 确保 cython 作为依赖
    zip_safe=False,
    maintainer='Your Name',
    maintainer_email='your_email@example.com',
    description='Person detector node with code protection.',
    license='Proprietary',
    tests_require=['pytest'],
    
    # 定义 ROS 2 节点入口点
    entry_points={
        'console_scripts': [
            'person_detector_node = person_detector.ros2_main:main'
        ],
    },
)