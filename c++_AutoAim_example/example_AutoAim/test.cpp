#include <iostream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "infer_openvino.hpp"
// 该模型使用的是mm坐标系，如果是m请记得×1000作为预处理的补充
std::pair<std::vector<std::vector<float>>, std::vector<float>> create_dataset(const std::vector<std::vector<double>> &centers, int input_step, int input_dim);
bool panlian(const vector<float> &xs, float threshold);
// 输入长度
int input_step = 20;
int input_dim = 3;
// 输出长度
int offset_step = 36;
int predict_step = 10;

int main()
{
    // 模型初始化
    infer_vino predictor("../ir_pred_model/pred_model.xml", "AUTO");
    // 比如你得到的是cv::Mat类型的xyz数据
    vector<vector<double>> aim_centers; // xyz

    while (true)
    {
        // 初始化xyz
        cv::Mat xyz = (Mat_<double>(3, 1) << 0, 0, 0);
        // 得到的xyz
        xyz = (Mat_<double>(3, 1) << 3296.081385, 800.000000, 395.0); // 该模型使用的是mm坐标系，如果是m请记得×1000作为预处理的补充
        if (xyz.at<double>(0) != 0.0)
        {
            vector<double> new_entry = {
                xyz.at<double>(0),
                xyz.at<double>(1),
                xyz.at<double>(2)};

            // 维护最多input_step个最新数据
            if (aim_centers.size() >= input_step)
            {
                aim_centers.erase(aim_centers.begin());
            }

            aim_centers.push_back(new_entry);
        }

        if (aim_centers.size() == input_step)
        {
            // 提取x坐标用于连续性检测
            vector<float> y_coords;
            for (auto &c : aim_centers)
            {
                y_coords.push_back(c[1]);
            }

            if (panlian(y_coords, 40.0)) // 使用40mm阈值判断是否连续，你也可以不判断连续，根据自己需求仅是一个示例展示。
            {

                // 创建数据集
                std::vector<std::vector<float>> diff_features;
                std::vector<float> initial_values;
                std::vector<std::vector<float>> pred_coords(predict_step, std::vector<float>(3, 0));
                // std::vector<float> mean_coord{0, 0, 0};

                auto input_data = create_dataset(aim_centers, input_step, input_dim);

                diff_features = input_data.first;
                initial_values = input_data.second;

                std::vector<std::vector<float>> output = predictor.vector_pred(diff_features, input_step, input_dim);

                // 后处理还原值
                // 如果想要哪一个预测时间对应的预测值那就是dt*offset~dt*(offset+predict_step)，步长是dt 预测长度是predict_step
                // 也就是说输出的是[3,predict_step]的std::vector<std::vector<float>> 二维向量
                // 例如需要最后一个点假设dt=0.0125 那么最后一个点对应的预测时间是0.0125*46=0.575s
                // 其实可以线性差值把这个补全比如你要预测的是t=0.556s 那么线性插值最后一段数据即可

                for (int k = 0; k < 3; k++)
                {
                    pred_coords[predict_step - 1][k] = output[predict_step - 1][k] + initial_values[k];
                }
                std::cout << "=------------------------------------------------------=" << std::endl;
                std::cout << "预测坐标值x为:" << pred_coords[predict_step - 1][0] << std::endl;
                std::cout << "预测坐标值y为:" << pred_coords[predict_step - 1][1] << std::endl;
                std::cout << "预测坐标值z为:" << pred_coords[predict_step - 1][2] << std::endl;
            }
        }
    }

    return 0;
}

std::pair<std::vector<std::vector<float>>, std::vector<float>> create_dataset(const std::vector<std::vector<double>> &centers, int input_step = 10, int input_dim = 3)
{
    std::vector<std::vector<double>> features(input_step, std::vector<double>(input_dim, 0));
    std::vector<double> initial_values(input_dim);
    // 填充数据
    for (int i = 0; i < input_step; ++i)
    {
        for (int k = 0; k < input_dim; ++k)
        {
            if (i == 0)
            {
                initial_values[k] = centers[i][k];
            }
            features[i][k] = centers[i][k] - initial_values[k];

            // std::cout << features[i][k] << " ";
        }
        // std::cout << std::endl;
    }

    // 转换为 float
    std::vector<std::vector<float>> float_features;
    std::vector<float> float_initial_values;

    // 转换 features
    for (const auto &feature_row : features)
    {
        std::vector<float> float_row;
        for (const auto &feature : feature_row)
        {
            float_row.push_back(static_cast<float>(feature));
        }
        float_features.push_back(float_row);
    }

    // 转换 initial_values
    for (const auto &value : initial_values)
    {
        float_initial_values.push_back(static_cast<float>(value));
    }

    return std::make_pair(float_features, float_initial_values);
}

bool panlian(const vector<float> &xs, float threshold)
{
    for (size_t i = 0; i < xs.size() - 1; ++i)
    {
        if (abs(xs[i] - xs[i + 1]) > threshold)
            return false;
    }
    return true;
}