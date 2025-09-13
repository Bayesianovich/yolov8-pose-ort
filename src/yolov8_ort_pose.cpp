#include "yolov8_ort_pose.hpp"
#include <fstream>
#include <iostream>

//配置参数定义
const std::string LABEL_PATH = "G:\\yolov8-ort-pose\\classes.txt"; //类别标签文件
const std::string IMAGE_PATH = "G:\\yolov8-ort-pose\\dance.png"; //测试图片路径
const std::string MODEL_PATH = "G:\\yolov8-ort-pose\\yolov8n-pose.onnx"; //模型文件路径 
const float CONFIDENCE_THRESHOLD = 0.5f; //置信度阈值 - 设置更低用于调试
const float NMS_THRESHOLD = 0.45f; //NMS阈值
const int INPUT_SIZE = 640; //模型输入尺寸

std::vector<std::string> readClassNames()
{
    std::vector<std::string> class_names;
    std::ifstream file(LABEL_PATH);

    if (!file.is_open()) 
    {
        std::cerr << "Failed to open class names file: " << LABEL_PATH << std::endl;
        return class_names;
    }

    std::string name;
    while (std::getline(file, name))
    {
        if (!name.empty())
            class_names.push_back(name);

    }
    return class_names;
}

void preprocessImage(const cv::Mat& frame, cv::Mat& blob, float& x_factor, float& y_factor)
{
    int original_width = frame.cols;
    int original_height = frame.rows;
    
    // 计算缩放因子，保持宽高比
    float scale = std::min(static_cast<float>(INPUT_SIZE) / original_width, 
                          static_cast<float>(INPUT_SIZE) / original_height);
    
    int new_width = static_cast<int>(original_width * scale);
    int new_height = static_cast<int>(original_height * scale);
    
    // 直接调整到计算出的尺寸
    cv::Mat scaled_image;
    cv::resize(frame, scaled_image, cv::Size(new_width, new_height));
    
    // 创建640x640的黑色背景图像，左上角对齐
    cv::Mat resized_image = cv::Mat::zeros(cv::Size(INPUT_SIZE, INPUT_SIZE), CV_8UC3);
    cv::Rect roi(0, 0, new_width, new_height);  // 左上角对齐
    scaled_image.copyTo(resized_image(roi));

    // 简化的缩放因子计算
    x_factor = 1.0f / scale;
    y_factor = 1.0f / scale;
    
    // 创建blob用于模型推理
    blob = cv::dnn::blobFromImage(resized_image, 
                                1.0 / 255.0, 
                                cv::Size(INPUT_SIZE, INPUT_SIZE), 
                                cv::Scalar(), 
                                true, 
                                false
                            );
}

void postprocessResults(const float* pdata,
                        int out_feat,
                        int out_box,
                        float x_factor,
                        float y_factor,
                        std::vector<cv::Rect>& boxes,
                        std::vector<float>& confidences,
                        std::vector<cv::Mat>& keypoints,
                        int original_width,
                        int original_height)
{
    cv::Mat detection_output = cv::Mat(out_box, out_feat, CV_32F, (float*)pdata);
    
    // YOLOv8-pose输出固定为56特征，如果维度不对则转置
    if (out_box == 56) {
        detection_output = detection_output.t();
        std::swap(out_box, out_feat);
    }
    
    for(int i = 0; i < detection_output.rows; ++i)
    {
        float conf = detection_output.at<float>(i, 4);
        if(conf >= CONFIDENCE_THRESHOLD)
        {
            // 提取边界框坐标
            float cx = detection_output.at<float>(i, 0);
            float cy = detection_output.at<float>(i, 1);
            float bw = detection_output.at<float>(i, 2);
            float bh = detection_output.at<float>(i, 3);

            // 转换为左上角坐标
            int left = static_cast<int>((cx - 0.5 * bw) * x_factor);
            int top = static_cast<int>((cy - 0.5 * bh) * y_factor);
            int width = static_cast<int>(bw * x_factor);
            int height = static_cast<int>(bh * y_factor);
            
            boxes.emplace_back(left, top, width, height);
            confidences.push_back(conf);

            // 提取关键点数据（固定51个值：17个关键点×3）
            cv::Mat keypoint_data = detection_output.row(i).colRange(5, 56).clone();
            keypoints.push_back(keypoint_data);
        }
    }
}

void drawPoseConnections(cv::Mat& frame,
                        const cv::Mat& keypoints,
                        const std::vector<cv::Scalar>& colors_tables)
{
    // COCO人体关键点标准连接关系
    // 0:鼻子 1:左眼 2:右眼 3:左耳 4:右耳 5:左肩 6:右肩 7:左肘 8:右肘 
    // 9:左腕 10:右腕 11:左髋 12:右髋 13:左膝 14:右膝 15:左踝 16:右踝
    std::vector<std::pair<int, int>> connections_pairs = {
        // 头部连接
        {0, 1}, {0, 2}, {1, 3}, {2, 4},
        // 躯干连接
        {5, 6}, {5, 11}, {6, 12}, {11, 12},
        // 左臂连接
        {5, 7}, {7, 9},
        // 右臂连接  
        {6, 8}, {8, 10},
        // 左腿连接
        {11, 13}, {13, 15},
        // 右腿连接
        {12, 14}, {14, 16}
    };
    
    // 可见性阈值
    const float visibility_threshold = 0.3f;
    
    //绘制连接线
    for (size_t i = 0; i < connections_pairs.size(); ++i)
    {
        int pt1_idx = connections_pairs[i].first;
        int pt2_idx = connections_pairs[i].second;
        
        // 检查关键点可见性
        float vis1 = keypoints.at<float>(pt1_idx, 2);
        float vis2 = keypoints.at<float>(pt2_idx, 2);
        
        if (vis1 > visibility_threshold && vis2 > visibility_threshold)
        {
            cv::Point pt1(static_cast<int>(keypoints.at<float>(pt1_idx, 0)), 
                         static_cast<int>(keypoints.at<float>(pt1_idx, 1)));
            cv::Point pt2(static_cast<int>(keypoints.at<float>(pt2_idx, 0)), 
                         static_cast<int>(keypoints.at<float>(pt2_idx, 1)));

            cv::line(frame, pt1, pt2, colors_tables[i % colors_tables.size()], 2, 8, 0);
        }
    }

    //绘制关键点
    for (int i = 0; i < keypoints.rows; ++i)
    {
        float visibility = keypoints.at<float>(i, 2);
        if (visibility > visibility_threshold)
        {
            cv::Point pt(static_cast<int>(keypoints.at<float>(i, 0)), 
                        static_cast<int>(keypoints.at<float>(i, 1)));
            cv::circle(frame, pt, 4, colors_tables[i % colors_tables.size()], -1);
        }
    }
}

