#include "preprocess/Preprocessor.h"

#include <iostream>

/*
    构造函数：
    保存模型输入尺寸。
*/
Preprocessor::Preprocessor(int input_width, int input_height)
    : input_width_(input_width),
      input_height_(input_height) {}

/*
    process()：
    图像预处理主函数。

    这个函数后面会在 main.cpp 中这样使用：

        Preprocessor preprocessor(640, 640);
        PreprocessResult prep = preprocessor.process(frame.image);

    然后把 prep.blob 输入给 YOLO 模型。
*/
PreprocessResult Preprocessor::process(const cv::Mat& image) {
    PreprocessResult result;

    if (image.empty()) {
        std::cerr << "Preprocessor input image is empty." << std::endl;
        return result;
    }

    // 2. 记录原图尺寸。
    // image.cols：图像宽度，也就是列数 image.rows：图像高度，也就是行数
    int src_w = image.cols;
    int src_h = image.rows;

    result.original_width = src_w;
    result.original_height = src_h;
    /*
        3. 计算等比例缩放比例。
        为什么取 min？
        因为我们要保证原图完整放进 640×640 里面，
        不能裁掉图像。
        假设原图是 1920×1080：
            640 / 1920 = 0.333
            640 / 1080 = 0.592
        取较小的 0.333，
        缩放后图像变成：
            640 × 360
        剩下的高度用灰边填充。
    */
    float scale = std::min(
        static_cast<float>(input_width_) / static_cast<float>(src_w),
        static_cast<float>(input_height_) / static_cast<float>(src_h)
    );

    result.scale = scale;

    /*
        4. 计算缩放后的图像尺寸。
    */
    int new_w = static_cast<int>(src_w * scale);
    int new_h = static_cast<int>(src_h * scale);

    /*
        5. 使用 OpenCV resize 缩放图像。
        cv::resize() 是 OpenCV 的图像缩放函数。
        参数说明：image：输入图像,resized：输出图像
        cv::Size(new_w, new_h)：缩放后的目标尺寸
    */
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_w, new_h));

    /*
        6. 计算需要填充多少灰边。
        pad_x：
            左右方向的填充量
        pad_y：
            上下方向的填充量
        例如：
            缩放后是 640×360
            目标是 640×640
        那么：
            pad_x = 0
            pad_y = (640 - 360) / 2 = 140
    */
    int pad_x = (input_width_ - new_w) / 2;
    int pad_y = (input_height_ - new_h) / 2;

    result.pad_x = pad_x;
    result.pad_y = pad_y;

    /*
        7. 创建一张 640×640 的灰色背景图。
        cv::Mat letterbox_image(...) 创建一张新图像。
        参数说明：input_height_：图像高度,input_width_：图像宽度
        image.type()：和原图保持相同类型，例如 CV_8UC3          
        cv::Scalar(114, 114, 114)：填充值。 
        YOLO 常用 114 作为灰边颜色。                             
        注意：
            OpenCV 默认是 BGR 通道顺序。
            所以 Scalar(114,114,114) 表示 B=114,G=114,R=114。
    */
    cv::Mat letterbox_image(
        input_height_,
        input_width_,
        image.type(),
        cv::Scalar(114, 114, 114)
    );

    /*
        8. 把缩放后的图像复制到灰色背景中间。
        cv::Rect(pad_x, pad_y, new_w, new_h)：
            表示在 letterbox_image 上取一个矩形区域。
        这个区域的位置是：
            左上角 x = pad_x
            左上角 y = pad_y
            宽度 = new_w
            高度 = new_h
        resized.copyTo(...)：
            把 resized 图像复制到这个区域里面。
    */
    resized.copyTo(
        letterbox_image(cv::Rect(pad_x, pad_y, new_w, new_h))
    );

    result.letterbox_image = letterbox_image;

    /*
        9. 使用 blobFromImage 构造模型输入 blob。
        cv::dnn::blobFromImage() 是 OpenCV DNN 模块的函数。
        它的作用：
            把普通图像 cv::Mat 转成神经网络需要的 4 维输入。
        普通图像格式：
            H × W × C
            也就是 高 × 宽 × 通道
        blob 格式：
            N × C × H × W
            也就是 batch × 通道 × 高 × 宽
        YOLO 模型需要的就是这种格式。
    */
    result.blob = cv::dnn::blobFromImage(
        letterbox_image,                         // 输入图像
        1.0 / 255.0,                             // 像素归一化，0~255 变成 0~1
        cv::Size(input_width_, input_height_),   // 模型输入尺寸
        cv::Scalar(),                            // 不减均值
        true,                                    // swapRB=true，BGR 转 RGB
        false                                    // crop=false，不裁剪
    );

    return result;
}