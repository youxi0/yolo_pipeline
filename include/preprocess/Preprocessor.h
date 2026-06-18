#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

/*
    PreprocessResult 用来保存预处理后的结果。
    为什么要保存 scale、pad_x、pad_y？
    因为 YOLO 输出的检测框和 mask 坐标是基于 640×640 图像的。
    但是我们最后要把结果画回原图。
    所以必须记住：
    1. 原图被缩放了多少
    2. 左右填充了多少
    3. 上下填充了多少
    后处理时要用这些信息把坐标还原回原图。
*/
struct PreprocessResult {
    cv::Mat letterbox_image;   // 经过 LetterBox 后的 640×640 图像
    cv::Mat blob;              // 输入 YOLO 模型的 blob

    float scale = 1.0f;        // 原图缩放比例
    int pad_x = 0;             // 左右方向填充
    int pad_y = 0;             // 上下方向填充

    int original_width = 0;    // 原图宽度
    int original_height = 0;   // 原图高度
};

/*
    Preprocessor 类：
    专门负责图像预处理。
    输入：
        原始 cv::Mat 图像
    输出：
        YOLO 可以使用的 blob
*/
class Preprocessor {
public:

    Preprocessor(int input_width = 640, int input_height = 640);

    PreprocessResult process(const cv::Mat& image);

private:
    int input_width_;
    int input_height_;

    uint8_t* imageDevice_ = nullptr;
    size_t imageDeviceBytes_ = 0;
};