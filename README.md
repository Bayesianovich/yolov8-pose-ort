# YOLOv8-Pose: 高性能人体姿态检测

基于YOLOv8-Pose模型和ONNX Runtime的轻量级人体姿态检测系统，专注于高效的实时推理。

## 🌟 主要特性

- **高效推理**：优化的C++实现，去除冗余检查，专注性能
- **实时处理**：支持图像和视频姿态检测
- **COCO标准**：采用COCO 17关键点标准
- **轻量设计**：简化的代码结构，更少的内存占用

## 🚀 快速开始

### 系统要求

- **编译器**：支持C++17的编译器（GCC 7+, MSVC 2017+）
- **CMake**：版本3.16或更高
- **OpenCV**：版本4.0或更高
- **ONNX Runtime**：版本1.19.0

### 依赖安装

#### Ubuntu/Debian
```bash
# 安装基础依赖
sudo apt update
sudo apt install cmake build-essential

# 安装OpenCV
sudo apt install libopencv-dev

# 下载ONNX Runtime,注意与cuda版本适配
wget https://github.com/microsoft/onnxruntime/releases/download/v1.19.0/onnxruntime-linux-x64-1.19.0.tgz
tar -xzf onnxruntime-linux-x64-1.19.0.tgz
```



#### Windows
1. 安装Visual Studio Code
2. 下载并安装CMake
3. 下载预编译的OpenCV和ONNX Runtime
4. 设置环境变量或修改CMakeLists.txt中的路径

### 编译构建

```bash
# 克隆仓库
git clone https://github.com/Bayesianovich/yolov8-pose-ort.git
cd yolov8-pose-ort

# 创建构建目录
mkdir build && cd build

# 配置CMake（需要根据实际路径调整）
cmake .. \
  -DOPENCV_ROOT=/path/to/opencv \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime

# 编译
make -j4
```

### 模型准备

1. 下载YOLOv8-Pose ONNX模型：
```bash
# 从官方或其他源下载yolov8n-pose.onnx
wget https://huggingface.co/Xenova/yolov8-pose-onnx/blob/main/yolov8n-pose.onnx
```

2. 将模型文件放置在项目根目录或更新代码中的`MODEL_PATH`

## 📖 使用方法

### 基本用法

```bash
# 默认模式（使用内置测试图片）
./yolov8_pose_ort

# 处理单张图片
./yolov8_pose_ort -i /path/to/image.jpg

# 处理视频文件
./yolov8_pose_ort -v /path/to/video.mp4
```

### 配置参数

在`src/yolov8_ort_pose.cpp`中的核心参数：

```cpp
const float CONFIDENCE_THRESHOLD = 0.5f;  // 置信度阈值
const float NMS_THRESHOLD = 0.45f;         // NMS阈值
const int INPUT_SIZE = 640;                // 模型输入尺寸（固定）
```



## 🏗️ 架构设计

### 核心组件

- **YOLOv8PoseDetector**：主检测器类，封装完整的推理流程
- **预处理模块**：Letterbox算法、像素归一化、格式转换
- **推理引擎**：ONNX Runtime会话管理和模型执行
- **后处理模块**：优化的结果解析、NMS去重、坐标还原（无冗余检查）
- **可视化模块**：关键点绘制、骨架连接

### 特性
- **简化错误检查**：专注人体关键点检测，减少不必要判断
- **自动维度处理**：智能处理[56,8400]和[8400,56]格式转换

### 技术栈

| 组件 | 版本 | 作用 |
|------|------|------|
| YOLOv8-Pose | - | 人体姿态检测模型 |
| ONNX Runtime | 1.19+ | 跨平台推理引擎 |
| OpenCV | 4.0+ | 计算机视觉处理 |
| CMake | 3.16+ | 构建系统 |
| C++17 | - | 编程语言标准 |


## 🎯 COCO 17关键点

本项目使用标准的COCO人体17关键点定义：

```
0: 鼻子    1: 左眼    2: 右眼    3: 左耳    4: 右耳
5: 左肩    6: 右肩    7: 左肘    8: 右肘    9: 左腕    10: 右腕
11: 左髋   12: 右髋   13: 左膝   14: 右膝   15: 左踝   16: 右踝
```

### 骨架连接

系统自动绘制解剖学正确的骨架连接：
- **面部**：鼻子-眼睛-耳朵
- **躯干**：肩膀-髋部连接
- **四肢**：上臂-前臂-手腕，大腿-小腿-脚踝


### CMake构建选项

```bash
# Release模式（推荐用于生产环境）
cmake .. -DCMAKE_BUILD_TYPE=Release

# Debug构建
cmake .. -DCMAKE_BUILD_TYPE=Debug

# 启用更多编译器警告
cmake .. -DCMAKE_CXX_FLAGS="-Wall -Wextra -O3"
```

### 运行时优化

```cpp
// 在初始化时设置更多线程
session_options.SetIntraOpNumThreads(4);

// 启用图优化
session_options.SetGraphOptimizationLevel(
    GraphOptimizationLevel::ORT_ENABLE_ALL);

// GPU推理（需要对应的ONNX Runtime版本）
session_options.AppendExecutionProvider_CUDA(0);
```

## 🐛 常见问题

### 编译错误

**问题**：找不到OpenCV或ONNX Runtime
```bash
CMake Error: Could not find OpenCV
```

**解决**：
1. 确保正确安装依赖
2. 检查CMakeLists.txt中的路径设置
3. 设置环境变量`OpenCV_DIR`和`ONNXRUNTIME_ROOT`

### 运行时错误

**问题**：模型加载失败
```
Error initializing ONNX session: No such file
```

**解决**：
1. 检查模型文件路径是否正确
2. 确保模型文件完整下载
3. 验证ONNX Runtime版本兼容性

### 性能问题

**问题**：处理速度过慢

**解决**：
1. 调整置信度阈值以减少检测数量
2. 启用多线程或GPU推理
3. 优化输入图像分辨率

## 🤝 贡献指南

欢迎提交问题和改进建议！

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建Pull Request

### 开发规范

- 遵循C++17标准
- 使用一致的代码格式（建议使用clang-format）
- 添加适当的注释和文档
- 确保跨平台兼容性

## 📝 许可证

本项目采用MIT许可证 - 查看[LICENSE](LICENSE)文件了解详细信息。

## 🙏 致谢

- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLOv8模型实现
- [Microsoft ONNX Runtime](https://github.com/microsoft/onnxruntime) - 高性能推理引擎
- [OpenCV](https://opencv.org/) - 计算机视觉库

## 📞 联系方式

- **问题反馈**：[GitHub Issues](https://github.com/Bayesianovich/yolov8-pose-ort/issues)
- **功能请求**：[GitHub Discussions](https://github.com/Bayesianovich/yolov8-pose-ort/discussions)

-------------------------------------------------

⭐ 如果这个项目对您有帮助，请给它一个星标！