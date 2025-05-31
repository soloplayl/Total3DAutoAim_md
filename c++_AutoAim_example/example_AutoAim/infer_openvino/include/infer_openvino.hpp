#ifndef INFER_OPENVINO_HPP
#define INFER_OPENVINO_HPP

#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <string>
#include <random>

using namespace cv;
using namespace std;

struct Detection
{
    int class_id{0};
    std::string className{};
    float confidence{0.0};

    cv::Rect box{};
    std::vector<cv::Point2f> key_points{};
};
class infer_vino
{
public:
    ov::CompiledModel compiled_model; // 添加成员变量
    infer_vino(std::string model_xml, std::string device);
    std::vector<Detection> image_detect(cv::Mat &frame);
    std::string model_xml, mode;
    ov::InferRequest request;
    float x_factor = 3.375; //_max = 1440 , 1440/640=2.25 1080/640=1.6875 1080/320=3.375
    float y_factor = 3.375;
    float keypoints_nums = 0; // 关键点个数要修改
    float conf_threshold = 0.50;
    float nms_threshold = 0.50;
    // std::string _mode="Pose";
    std::vector<std::string> classes{"b_blue_1", "s_blue_3"};                                                                      // 要修改
    std::vector<std::vector<float>> vector_pred(const std::vector<std::vector<float>> &input_data, int input_step, int input_dim); // 新增
};
#endif // INFER_OPENVINO_HPP