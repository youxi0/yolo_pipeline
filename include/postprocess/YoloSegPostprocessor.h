#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "common/DetectionResult.h"
#include "preprocess/Preprocessor.h"

//模块:YoloSegPostprocessor负责解析YOLOv8-seg的TensorRT输出
class YoloSegPostprocessor{
public:
    YoloSegPostprocessor(float confThreshold, float nmsThreshold, float maskThreshold);

    std::vector<DetectionResult> process(
        const std::vector<cv::Mat>& outputs,
        const PreprocessResult& prep,
        const std::vector<std::string>& classNames
    );

private:
    cv::Mat buildMask(
        const cv::Mat& proto,
        const std::array<float, 32>& maskCoeff,
        const cv::Rect& box,
        const PreprocessResult& prep
    );

    float sigmoid(float x) const;

private:
    //去掉置信度低的
    float confThreshold_ = 0.25f; 
    //两个框的IOU交并比大于这个值去掉
    float nmsThreshold_ = 0.45f;
    //mask概率图，概率大于这个为白255，小于为0黑  
    float maskThreshold_ = 0.5f;
};