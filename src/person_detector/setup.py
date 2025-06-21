# src/person_detector/setup.py

import os
from glob import glob
from setuptools import setup, Extension
from setuptools.command.build_py import build_py
from Cython.Build import cythonize

package_name = 'person_detector'

# --- 步骤 1: 定义所有需要被 Cython 编译的 .py 文件 ---
# 我们将把这些文件的 .py 源码从最终的安装包中排除
extensions = [
    Extension(
        "person_detector.ros2_main",
        ["person_detector/ros2_main.py"]
    ),
    Extension(
        "person_detector.model.yolo_detector",
        ["person_detector/model/yolo_detector.py"]
    ),
    # 使用通配符编译 utils 目录下的所有 .py 文件 (除了 __init__.py)
    *([Extension(f"person_detector.utils.{os.path.splitext(os.path.basename(p))[0]}", [p])
       for p in glob('person_detector/utils/*.py') if '__init__' not in os.path.basename(p)])
]

# 获取所有被编译模块的名称，方便后续检查
cython_module_names = {ext.name for ext in extensions}


# --- 步骤 2: 自定义 build_py 命令 ---
# 这个类的作用是在 setuptools 复制 .py 文件前进行拦截，
# 并移除那些已经被我们编译成 .so 的文件。
class CustomBuildPy(build_py):
    def find_package_modules(self, package, package_dir):
        # 首先，调用父类的方法获取所有应该被打包的模块
        modules = super().find_package_modules(package, package_dir)
        
        # 创建一个新的列表，只包含我们不想编译的模块 (比如 __init__.py)
        filtered_modules = []
        for (pkg, module, filepath) in modules:
            # 构造完整的模块名，例如 'person_detector.utils.kalman_3d'
            full_module_name = f"{pkg}.{module}"
            
            # 如果这个模块不在我们编译的列表中，就保留它
            if full_module_name not in cython_module_names:
                filtered_modules.append((pkg, module, filepath))
            else:
                # 如果在列表中，就打印信息并跳过，不将其 .py 文件打包
                print(f"Excluding source file [{filepath}] (compiled by Cython).")

        return filtered_modules


# --- 步骤 3: 配置 setup() 函数 ---
setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name, f"{package_name}.utils", f"{package_name}.model"],
    
    # 使用 cythonize 处理 extensions
    ext_modules=cythonize(
        extensions,
        compiler_directives={'language_level': "3"} # 指定 Python 3
    ),
    
    # **关键**: 使用我们自定义的 build_py 命令替换默认命令
    cmdclass={'build_py': CustomBuildPy},

    # 将模型文件等非代码文件安装到 share 目录
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 将模型文件夹安装到 share/person_detector/model
        # (os.path.join('share', package_name, 'model'), glob('model/*')),
        (os.path.join('share', package_name, 'model'), glob('../../model/*')),
    ],
    install_requires=['setuptools', 'cython'],
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