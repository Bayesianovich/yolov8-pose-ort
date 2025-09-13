cmake_minimum_required(VERSION 3.16)
project(yolov8_pose_ort)

# 设置C++标准和编译类型
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# 设置库路径
set(OPENCV_ROOT "E:/opencv_cuda/install")
set(ONNXRUNTIME_ROOT "E:/onnxruntime-win-x64-gpu-1.21.0")

# 手动配置OpenCV（适用于预编译版本）
set(OpenCV_INCLUDE_DIRS "${OPENCV_ROOT}/include")
# 使用opencv_world库（所有模块的统一库）
set(OpenCV_LIBS 
    "${OPENCV_ROOT}/x64/vc17/lib/opencv_world4130.lib"
    "${OPENCV_ROOT}/x64/vc17/lib/opencv_img_hash4130.lib"
)

# 创建可执行文件
add_executable(${PROJECT_NAME}
    src/yolov8_ort_pose.cpp
    main.cpp
)

# 设置包含目录和链接库
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_ROOT}/include
)

target_link_directories(${PROJECT_NAME} PRIVATE
    ${OPENCV_ROOT}/x64/vc17/lib
    ${ONNXRUNTIME_ROOT}/lib
)

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    onnxruntime
)

# Windows设置：复制必要的DLL
if(WIN32)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
        "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_cuda.dll"
        "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_shared.dll"
        $<TARGET_FILE_DIR:${PROJECT_NAME}>
    )
    
    # MSVC编译优化
    target_compile_definitions(${PROJECT_NAME} PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(${PROJECT_NAME} PRIVATE /MP)
endif()

# 创建输出目录
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/output)
