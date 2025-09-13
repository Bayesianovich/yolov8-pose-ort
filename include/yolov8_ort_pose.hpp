#ifndef YOLOV11_ORT_POSE_HPP
#define YOLOV11_ORT_POSE_HPP

#include <opencv2/opencv.hpp>
#include "onnxruntime_cxx_api.h"
#include <vector>
#include <string>

// 配置参数常量声明
extern const std::string LABEL_PATH;
extern const std::string IMAGE_PATH;
extern const std::string MODEL_PATH;
extern const float CONFIDENCE_THRESHOLD; 
extern const float NMS_THRESHOLD;
extern const int INPUT_SIZE;

/**
 * @brief 读取类别标签文件
 * @return 包含类别名称的字符串向量
 */
std::vector<std::string> readClassNames();

/**  
 * @brief 预处理输入图像，将图像转换为模型需要的格式
 * @param frame 输入的原始图像
 * @param blob 输出的blob数据，用于模型推理
 * @param x_factor 输出的x方向缩放因子
 * @param y_factor 输出的y方向缩放因子
 */
void preprocessImage(const cv::Mat& frame, cv::Mat& blob, float& x_factor, float& y_factor);

/**
 * @brief 创建关键点变换矩阵
 * @param x_factor x方向缩放因子
 * @param y_factor y方向缩放因子
 * @return 关键点变换矩阵
 */
cv::Mat createKeypointTransformMatrix(float x_factor, float y_factor);

/**
 * @brief 后处理模型输出结果，提取边界框、置信度和关键点
 * @param pdata 模型输出数据指针
 * @param out_feat 输出特征数量
 * @param out_box 输出边界框数量
 * @param x_factor x方向缩放因子
 * @param y_factor y方向缩放因子
 * @param boxes 输出的边界框向量
 * @param confidences 输出的置信度向量
 * @param keypoints 输出的关键点数据向量
 */
void postprocessResults(const float* pdata,
                        int out_feat,
                        int out_box,
                        float x_factor,
                        float y_factor,
                        std::vector<cv::Rect>& boxes,
                        std::vector<float>& confidences,
                        std::vector<cv::Mat>& keypoints,
                        int original_width,
                        int original_height);

/**
 * @brief 在图像上绘制姿态关键点和连接线
 * @param frame 要绘制的图像
 * @param keypoints 关键点数据
 * @param colors_tables 颜色表
 */
void drawPoseConnections(cv::Mat& frame,
                        const cv::Mat& keypoints,
                        const std::vector<cv::Scalar>& colors_tables);

#endif // YOLOV11_ORT_POSE_HPP