#include "postprocess/YoloSegPostprocessor.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <array>

//模块:YoloSegPostprocessor构造函数,设置置信度阈值、NMS阈值和mask二值化阈值
YoloSegPostprocessor::YoloSegPostprocessor(float confThreshold, float nmsThreshold, float maskThreshold)
    : confThreshold_(confThreshold),
      nmsThreshold_(nmsThreshold),
      maskThreshold_(maskThreshold){}

//模块:sigmoid把mask预测值转换为0到1之间的概率
float YoloSegPostprocessor::sigmoid(float x) const
{
    return 1.0f / (1.0f + std::exp(-x));
}

//模块:process负责把TensorRT原始输出解析成检测框、类别、置信度和实例mask
std::vector<DetectionResult> YoloSegPostprocessor::process(
    const std::vector<cv::Mat>& outputs,
    const PreprocessResult& prep,
    const std::vector<std::string>& classNames
){
    std::vector<DetectionResult> results;

    if (outputs.size() < 2)
    {
        std::cerr << "YOLOv8-seg needs 2 outputs." << std::endl;
        return results;
    }

    const cv::Mat* protoOutput = nullptr;
    const cv::Mat* detectOutput = nullptr;

    //模块:根据输出维度自动判断哪个是proto输出,哪个是检测输出
    for (const auto& output : outputs)
    {
        if (output.dims == 4)
        {
            protoOutput = &output;
        }
        else if (output.dims == 3)
        {
            detectOutput = &output;
        }
    }

    if (protoOutput == nullptr || detectOutput == nullptr)
    {
        std::cerr << "Invalid YOLOv8-seg output format." << std::endl;
        return results;
    }

    
    // detect:Output[1] shape: 1 x 40(4 + 4 + maskdim) x 8400
    int channels = detectOutput->size[1];
    int numPreds = detectOutput->size[2];
    int maskDim = protoOutput->size[1];
    int numClasses = channels - 4 - maskDim;

    if (numClasses <= 0)
    {
        std::cerr << "Invalid numClasses: " << numClasses << std::endl;
        return results;
    }


    //Mat用的是uint8_t/uchar储存的，但是yolo输出是float
    const float* detectData = reinterpret_cast<const float*>(detectOutput->data);

    //Rect为矩形框，装好要输出的boxes，scores,classIds,maskCoeffs，后面nms过滤以后能下标对齐
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> classIds;
    std::vector<std::array<float, 32>> maskCoeffs;

    //模块:遍历8400个候选框,解析cx cy w h、类别分数和mask系数
    // cx_0, cx_1, cx_2, ..., cx_8399,
    // cy_0, cy_1, cy_2, ..., cy_8399,
    // w_0,  w_1,  w_2,  ..., w_8399,
    // h_0,  h_1,  h_2,  ..., h_8399,
    // class0_0, class0_1, ..., class0_8399,
    // class1_0, class1_1, ..., class1_8399,
    // ...
    for (int i = 0; i < numPreds; ++i)
    {
        float cx = detectData[0 * numPreds + i];
        float cy = detectData[1 * numPreds + i];
        float w = detectData[2 * numPreds + i];
        float h = detectData[3 * numPreds + i];

        int bestClassId = -1;
        float bestScore = 0.0f;

        //模块:从类别分数中找到置信度最高的类别
        for (int c = 0; c < numClasses; ++c)
        {
            float score = detectData[(4 + c) * numPreds + i];

            if (score > bestScore)
            {
                bestScore = score;
                bestClassId = c;
            }
        }

        if (bestScore < confThreshold_)
        {
            continue;
        }

        //模块:把YOLO输出的中心点宽高格式转换成输入图上的左上右下坐标
        float x1Input = cx - w * 0.5f;
        float y1Input = cy - h * 0.5f;
        float x2Input = cx + w * 0.5f;
        float y2Input = cy + h * 0.5f;

        //模块:根据letterbox的缩放比例和padding把坐标还原到原图
        float x1 = (x1Input - prep.pad_x) / prep.scale;
        float y1 = (y1Input - prep.pad_y) / prep.scale;
        float x2 = (x2Input - prep.pad_x) / prep.scale;
        float y2 = (y2Input - prep.pad_y) / prep.scale;
        
        //把出界的框卡在0 ~ original_width - 1
        x1 = std::max(0.0f, std::min(x1, static_cast<float>(prep.original_width - 1)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(prep.original_height - 1)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(prep.original_width - 1)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(prep.original_height - 1)));

        int boxX = static_cast<int>(x1);
        int boxY = static_cast<int>(y1);
        int boxW = static_cast<int>(x2 - x1);
        int boxH = static_cast<int>(y2 - y1);

        if (boxW <= 0 || boxH <= 0)
        {
            continue;
        }

        boxes.emplace_back(boxX, boxY, boxW, boxH);
        scores.push_back(bestScore);
        classIds.push_back(bestClassId);

        std::array<float, 32> coeff{};

        //模块:读取当前候选框对应的32维mask系数
        // mask = coeff0 * proto0
        //  + coeff1 * proto1
        //  + coeff2 * proto2
        //  + ...
        //  + coeff31 * proto31
        for (int m = 0; m < maskDim; ++m)
        {
            coeff[m] = detectData[(4 + numClasses + m) * numPreds + i];
        }

        maskCoeffs.push_back(coeff);
    }


    std::vector<int> indices;

    //模块:cv::dnn::NMSBoxes执行非极大值抑制,去掉重复检测框， indices最终保留下来的框的下标
    cv::dnn::NMSBoxes(boxes, scores, confThreshold_, nmsThreshold_, indices);

    //模块:根据NMS保留的索引生成最终检测结果
    for (int index : indices)
    {
        DetectionResult result;

        result.classId = classIds[index];
        result.confidence = scores[index];
        result.box = boxes[index];

        if (result.classId >= 0 && result.classId < static_cast<int>(classNames.size()))
        {
            result.className = classNames[result.classId];
        }
        else
        {
            result.className = std::to_string(result.classId);
        }

        result.mask = buildMask(*protoOutput, maskCoeffs[index], result.box, prep);

        results.push_back(result);
    }

    return results;
}

//模块:buildMask负责用prototype和mask系数生成单个实例的二值mask
//只在目标框对应的proto ROI区域内生成mask,避免每个目标都生成整张原图大小的mask
cv::Mat YoloSegPostprocessor::buildMask(
    const cv::Mat& proto,
    const std::array<float, 32>& maskCoeff,
    const cv::Rect& box,
    const PreprocessResult& prep
)
{
    // proto: 1 x maskDim x maskH x maskW
    int maskDim = proto.size[1];
    int maskH = proto.size[2];
    int maskW = proto.size[3];

    if (maskDim <= 0 || maskH <= 0 || maskW <= 0) {
        return cv::Mat::zeros(prep.original_height, prep.original_width, CV_8UC1);
    }

    if (static_cast<int>(maskCoeff.size()) < maskDim) {
        std::cerr << "Invalid mask coeff size." << std::endl;
        return cv::Mat::zeros(prep.original_height, prep.original_width, CV_8UC1);
    }



    //模块:最终仍然返回一张原图大小的二值mask,这样Visualizer原来的逻辑不用改
    // cv::Mat croppedMask = cv::Mat::zeros(
    //     prep.original_height,
    //     prep.original_width,
    //     CV_8UC1
    // );

    //模块:防止检测框越界
    cv::Rect imageRect(0, 0, prep.original_width, prep.original_height);
    cv::Rect safeBox = box & imageRect;

    if (safeBox.width <= 0 || safeBox.height <= 0) {
        return cv::Mat();
    }

    //模块:把原图上的box映射回640x640 letterbox输入图坐标
    float x1Input = safeBox.x * prep.scale + prep.pad_x;
    float y1Input = safeBox.y * prep.scale + prep.pad_y;
    float x2Input = (safeBox.x + safeBox.width) * prep.scale + prep.pad_x;
    float y2Input = (safeBox.y + safeBox.height) * prep.scale + prep.pad_y;

    x1Input = std::max(0.0f, std::min(x1Input, 640.0f));
    y1Input = std::max(0.0f, std::min(y1Input, 640.0f));
    x2Input = std::max(0.0f, std::min(x2Input, 640.0f));
    y2Input = std::max(0.0f, std::min(y2Input, 640.0f));

    if (x2Input <= x1Input || y2Input <= y1Input) {
        return cv::Mat();
    }

    //模块:把640x640输入图坐标映射到proto的160x160坐标
    float protoScaleX = static_cast<float>(maskW) / 640.0f;
    float protoScaleY = static_cast<float>(maskH) / 640.0f;

    int px1 = static_cast<int>(std::floor(x1Input * protoScaleX));
    int py1 = static_cast<int>(std::floor(y1Input * protoScaleY));
    int px2 = static_cast<int>(std::ceil(x2Input * protoScaleX));
    int py2 = static_cast<int>(std::ceil(y2Input * protoScaleY));

    px1 = std::max(0, std::min(px1, maskW - 1));
    py1 = std::max(0, std::min(py1, maskH - 1));
    px2 = std::max(0, std::min(px2, maskW));
    py2 = std::max(0, std::min(py2, maskH));

    int roiW = px2 - px1;
    int roiH = py2 - py1;

    if (roiW <= 0 || roiH <= 0) {
        return cv::Mat();
    }

    //模块:只创建proto ROI大小的float mask,不再创建完整160x160 mask
    cv::Mat maskRoi = cv::Mat::zeros(roiH, roiW, CV_32F);

    const float* protoData = reinterpret_cast<const float*>(proto.data);
    int protoArea = maskH * maskW;
    
    //模块:prototype和mask系数做线性组合,但只计算目标框对应的proto ROI区域
    for (int m = 0; m < maskDim; ++m)
    {
        float coeff = maskCoeff[m];
        const float* protoChannel = protoData + m * protoArea;

        for (int y = 0; y < roiH; ++y)
        {
            const float* protoRow = protoChannel + (py1 + y) * maskW + px1;
            float* dstRow = maskRoi.ptr<float>(y);

            for (int x = 0; x < roiW; ++x)
            {
                dstRow[x] += coeff * protoRow[x];
            }
        }
    }

    //模块:对ROI区域执行sigmoid
    for (int y = 0; y < maskRoi.rows; ++y)
    {
        float* rowPtr = maskRoi.ptr<float>(y);

        for (int x = 0; x < maskRoi.cols; ++x)
        {
            rowPtr[x] = sigmoid(rowPtr[x]);
        }
    }

    cv::Mat maskBoxFloat;

    //模块:把proto ROI mask直接resize到原图box大小
    //原来是160x160 -> 640x640 -> 原图大小 -> 裁box
    //现在是proto ROI -> box大小
    cv::resize(
        maskRoi,
        maskBoxFloat,
        cv::Size(safeBox.width, safeBox.height),
        0,
        0,
        cv::INTER_LINEAR
    );

    cv::Mat maskBoxBinary;

    //模块:把0到1概率mask转成0或255的二值mask
    cv::threshold(
        maskBoxFloat,
        maskBoxBinary,
        maskThreshold_,
        255,
        cv::THRESH_BINARY
    );

    maskBoxBinary.convertTo(maskBoxBinary, CV_8UC1);

    return maskBoxBinary;
}