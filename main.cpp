#include "yolov8_ort_pose.hpp"
#include <iostream>
#include <filesystem>
#include <chrono>
#include <iomanip>

namespace fs = std::filesystem;

class YOLOv8PoseDetector {
private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;
    
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    std::vector<std::int64_t> input_dims;
    std::vector<std::int64_t> output_dims;
    
    std::vector<std::string> class_names;
    std::vector<cv::Scalar> colors_table;
    
public:
    YOLOv8PoseDetector() : env(ORT_LOGGING_LEVEL_WARNING, "YOLOv8-Pose") {
        // 初始化颜色表
        initializeColors();
        
        // 加载类别名称
        class_names = readClassNames();
        
        // 初始化ONNX Runtime会话
        initializeSession();
    }
    
    ~YOLOv8PoseDetector() {
        // 释放动态分配的字符串
        for (auto* name : input_names) {
            free(const_cast<char*>(name));
        }
        for (auto* name : output_names) {
            free(const_cast<char*>(name));
        }
    }
    
private:
    void initializeColors() {
        colors_table = {
            cv::Scalar(255, 0, 0),     // 红色
            cv::Scalar(0, 255, 0),     // 绿色
            cv::Scalar(0, 0, 255),     // 蓝色
            cv::Scalar(255, 255, 0),   // 黄色
            cv::Scalar(255, 0, 255),   // 洋红色
            cv::Scalar(0, 255, 255),   // 青色
            cv::Scalar(128, 0, 128),   // 紫色
            cv::Scalar(255, 165, 0),   // 橙色
            cv::Scalar(255, 192, 203), // 粉色
            cv::Scalar(0, 128, 0),     // 深绿色
            cv::Scalar(128, 128, 0),   // 橄榄色
            cv::Scalar(0, 0, 128),     // 深蓝色
            cv::Scalar(128, 0, 0),     // 深红色
            cv::Scalar(192, 192, 192), // 银色
            cv::Scalar(255, 215, 0),   // 金色
            cv::Scalar(220, 20, 60),   // 深红色
            cv::Scalar(75, 0, 130)     // 靛蓝色
        };
    }
    
    void initializeSession() {
        try {
            // 设置会话选项
            session_options.SetIntraOpNumThreads(1);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
            
            // 创建会话
            std::wstring model_path_w(MODEL_PATH.begin(), MODEL_PATH.end());
            session = std::make_unique<Ort::Session>(env, model_path_w.c_str(), session_options);
            
            // 获取输入输出信息
            getModelInfo();
            
            std::cout << "YOLOv8 Pose model loaded successfully!" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error initializing ONNX session: " << e.what() << std::endl;
            throw;
        }
    }
    
    void getModelInfo() {
        // 获取输入信息
        size_t num_input_nodes = session->GetInputCount();
        input_names.resize(num_input_nodes);
        
        for (size_t i = 0; i < num_input_nodes; i++) {
            auto input_name = session->GetInputNameAllocated(i, allocator);
            // 创建持久化的字符串副本
            std::string name_str(input_name.get());
            input_names[i] = strdup(name_str.c_str());
            
            Ort::TypeInfo input_type_info = session->GetInputTypeInfo(i);
            auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
            input_dims = input_tensor_info.GetShape();
        }
        
        // 获取输出信息
        size_t num_output_nodes = session->GetOutputCount();
        output_names.resize(num_output_nodes);
        
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto output_name = session->GetOutputNameAllocated(i, allocator);
            // 创建持久化的字符串副本
            std::string name_str(output_name.get());
            output_names[i] = strdup(name_str.c_str());
            
            Ort::TypeInfo output_type_info = session->GetOutputTypeInfo(i);
            auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
            output_dims = output_tensor_info.GetShape();
        }
        
        std::cout << "Model info - Input: " << input_names[0] << ", Output: " << output_names[0] << std::endl;
    }
    
public:
    cv::Mat detectPose(const cv::Mat& frame) {
        cv::Mat result_frame = frame.clone();
        
        try {
            // 预处理
            cv::Mat blob;
            float x_factor, y_factor;
            preprocessImage(frame, blob, x_factor, y_factor);
            
            // 创建输入tensor
            std::vector<int64_t> input_shape = {1, 3, INPUT_SIZE, INPUT_SIZE};
            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, 
                                                                     (float*)blob.data, 
                                                                     blob.total(), 
                                                                     input_shape.data(), 
                                                                     input_shape.size());
            
            // 运行推理
            auto output_tensors = session->Run(Ort::RunOptions{nullptr}, 
                                             input_names.data(), 
                                             &input_tensor, 
                                             input_names.size(), 
                                             output_names.data(), 
                                             output_names.size());
            
            // 后处理
            float* pdata = output_tensors[0].GetTensorMutableData<float>();
            int out_feat = static_cast<int>(output_dims[2]);  // 特征数量
            int out_box = static_cast<int>(output_dims[1]);   // 检测框数量
            
            std::vector<cv::Rect> boxes;
            std::vector<float> confidences;
            std::vector<cv::Mat> keypoints_data;
            
            postprocessResults(pdata, out_feat, out_box, x_factor, y_factor, 
                             boxes, confidences, keypoints_data, frame.cols, frame.rows);
            
            // NMS处理
            std::vector<int> indices;
            cv::dnn::NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, indices);
            
            // 绘制结果
            for (int i : indices) {
                // 绘制边界框
                cv::rectangle(result_frame, boxes[i], cv::Scalar(0, 255, 0), 2);
                
                // 绘制置信度
                std::string label = "Person: " + std::to_string(confidences[i]).substr(0, 4);
                cv::putText(result_frame, label, 
                           cv::Point(boxes[i].x, boxes[i].y - 10), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                           cv::Scalar(0, 255, 0), 2);
                
                // 关键点数据固定为51个元素 (17个关键点 x 3个值)
                cv::Mat keypoints_raw = keypoints_data[i];
                cv::Mat keypoints = keypoints_raw.reshape(1, 17);
                cv::Mat scaled_keypoints(17, 3, CV_32F);
                
                // 关键点坐标变换到原图尺寸
                for (int j = 0; j < 17; ++j) {
                    scaled_keypoints.at<float>(j, 0) = keypoints.at<float>(j, 0) * x_factor;
                    scaled_keypoints.at<float>(j, 1) = keypoints.at<float>(j, 1) * y_factor;
                    scaled_keypoints.at<float>(j, 2) = keypoints.at<float>(j, 2);
                }
                
                // 绘制姿态连接
                drawPoseConnections(result_frame, scaled_keypoints, colors_table);
            }
            
        } catch (const std::exception& e) {
            std::cerr << "Error during pose detection: " << e.what() << std::endl;
        }
        
        return result_frame;
    }
    
    void processImage(const std::string& input_path, const std::string& output_path) {
        std::cout << "Processing image: " << input_path << std::endl;
        
        cv::Mat frame = cv::imread(input_path);
        if (frame.empty()) {
            std::cerr << "Error: Could not read image " << input_path << std::endl;
            return;
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        cv::Mat result = detectPose(frame);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Processing time: " << duration.count() << " ms" << std::endl;
        
        // 保存结果
        if (cv::imwrite(output_path, result)) {
            std::cout << "Result saved to: " << output_path << std::endl;
        } else {
            std::cerr << "Error: Could not save result to " << output_path << std::endl;
        }
    }
    
    void processVideo(const std::string& input_path, const std::string& output_path) {
        std::cout << "Processing video: " << input_path << std::endl;
        
        cv::VideoCapture cap(input_path);
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open video " << input_path << std::endl;
            return;
        }
        
        // 获取视频属性
        int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        double fps = cap.get(cv::CAP_PROP_FPS);
        int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        
        std::cout << "Video info: " << frame_width << "x" << frame_height 
                  << " @ " << fps << " fps, " << total_frames << " frames" << std::endl;
        
        // 创建视频写入器
        cv::VideoWriter writer(output_path, 
                              cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 
                              fps, 
                              cv::Size(frame_width, frame_height));
        
        if (!writer.isOpened()) {
            std::cerr << "Error: Could not create video writer for " << output_path << std::endl;
            return;
        }
        
        cv::Mat frame;
        int frame_count = 0;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        while (cap.read(frame)) {
            frame_count++;
            
            auto frame_start = std::chrono::high_resolution_clock::now();
            cv::Mat result = detectPose(frame);
            auto frame_end = std::chrono::high_resolution_clock::now();
            
            auto frame_duration = std::chrono::duration_cast<std::chrono::milliseconds>(frame_end - frame_start);
            
            // 在图像上显示帧数和处理时间
            std::string info = "Frame: " + std::to_string(frame_count) + "/" + std::to_string(total_frames) 
                              + " | Time: " + std::to_string(frame_duration.count()) + "ms";
            cv::putText(result, info, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
            
            writer.write(result);
            
            // 显示进度
            if (frame_count % 30 == 0) {
                float progress = static_cast<float>(frame_count) / total_frames * 100.0f;
                std::cout << "Progress: " << std::fixed << std::setprecision(1) 
                          << progress << "% (" << frame_count << "/" << total_frames << ")" << std::endl;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
        
        cap.release();
        writer.release();
        
        std::cout << "Video processing completed!" << std::endl;
        std::cout << "Total time: " << total_duration.count() << " seconds" << std::endl;
        std::cout << "Average FPS: " << frame_count / total_duration.count() << std::endl;
        std::cout << "Result saved to: " << output_path << std::endl;
    }
};

void printUsage() {
    std::cout << "Usage:" << std::endl;
    std::cout << "  For image: main.exe -i <input_image_path>" << std::endl;
    std::cout << "  For video: main.exe -v <input_video_path>" << std::endl;
    std::cout << "  Default test: main.exe (uses default image)" << std::endl;
    std::cout << std::endl;
    std::cout << "Output will be saved to: H:\\study\\onnxruntime_learning\\yolov8-pose-ort\\output\\" << std::endl;
}

std::string getTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
    ss << "_" << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

int main(int argc, char* argv[]) {
    std::cout << "YOLOv8 Pose Detection Application" << std::endl;
    std::cout << "=================================" << std::endl;
    
    try {
        // 初始化检测器
        YOLOv8PoseDetector detector;

        std::string output_dir = "G:\\yolov8-ort-pose\\output\\";
        std::string timestamp = getTimestamp();
        
        if (argc == 1) {
            // 默认测试模式，使用默认图像
            std::cout << "Running in default test mode..." << std::endl;
            std::string output_path = output_dir + "result_" + timestamp + ".jpg";
            detector.processImage(IMAGE_PATH, output_path);
            
        } else if (argc == 3) {
            std::string mode = argv[1];
            std::string input_path = argv[2];
            
            if (mode == "-i") {
                // 图像模式
                std::string extension = fs::path(input_path).extension().string();
                std::string output_path = output_dir + "result_" + timestamp + extension;
                detector.processImage(input_path, output_path);
                
            } else if (mode == "-v") {
                // 视频模式
                std::string output_path = output_dir + "result_" + timestamp + ".mp4";
                detector.processVideo(input_path, output_path);
                
            } else {
                std::cerr << "Error: Invalid mode. Use -i for image or -v for video." << std::endl;
                printUsage();
                return -1;
            }
            
        } else {
            std::cerr << "Error: Invalid number of arguments." << std::endl;
            printUsage();
            return -1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    std::cout << "Application completed successfully!" << std::endl;
    return 0;
}