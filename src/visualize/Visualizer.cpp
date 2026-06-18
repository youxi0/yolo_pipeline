#include "visualize/Visualizer.h"

#include <algorithm>
#include <iomanip>
#include <sstream>

//模块:draw负责绘制实例分割结果
cv::Mat Visualizer::draw(const cv::Mat& image, const std::vector<DetectionResult>& results)
{
    //模块:clone复制原图,避免直接修改输入图像
    cv::Mat vis = image.clone();

    if (vis.empty()) {
        return vis;
    }

    cv::Rect imageRect(0, 0, vis.cols, vis.rows);

    for (const auto& result : results)
    {
        cv::Scalar color(0, 255, 0);

        //模块:防止检测框越界
        cv::Rect safeBox = result.box & imageRect;

        if (safeBox.width <= 0 || safeBox.height <= 0) {
            continue;
        }

        //模块:只在box区域内绘制mask,避免整图alpha混合
        if (!result.mask.empty())
        {
            const cv::Mat* maskPtr = &result.mask;
            cv::Mat resizedMask;

            //模块:如果mask尺寸和box尺寸不一致,先缩放到box大小
            if (result.mask.size() != safeBox.size())
            {
                cv::resize(
                    result.mask,
                    resizedMask,
                    safeBox.size(),
                    0,
                    0,
                    cv::INTER_NEAREST
                );

                maskPtr = &resizedMask;
            }

            //注意:这里必须是vis(safeBox),不能是image(safeBox)
            cv::Mat imageRoi = vis(safeBox);
            const cv::Mat& maskRoi = *maskPtr;

            constexpr float keepAlpha = 0.65f;
            constexpr float maskAlpha = 0.35f;

            for (int y = 0; y < safeBox.height; ++y)
            {
                const uchar* maskRow = maskRoi.ptr<uchar>(y);
                cv::Vec3b* imageRow = imageRoi.ptr<cv::Vec3b>(y);

                for (int x = 0; x < safeBox.width; ++x)
                {
                    if (maskRow[x] == 0) {
                        continue;
                    }

                    imageRow[x][0] = static_cast<uchar>(
                        imageRow[x][0] * keepAlpha + color[0] * maskAlpha
                    );

                    imageRow[x][1] = static_cast<uchar>(
                        imageRow[x][1] * keepAlpha + color[1] * maskAlpha
                    );

                    imageRow[x][2] = static_cast<uchar>(
                        imageRow[x][2] * keepAlpha + color[2] * maskAlpha
                    );
                }
            }
        }

        //模块:绘制检测框
        cv::rectangle(vis, safeBox, color, 2);

        std::ostringstream oss;
        oss << result.className << " "
            << std::fixed << std::setprecision(2)
            << result.confidence;

        std::string text = oss.str();

        int baseLine = 0;

        cv::Size textSize = cv::getTextSize(
            text,
            cv::FONT_HERSHEY_SIMPLEX,
            0.6,
            1,
            &baseLine
        );

        int textX = safeBox.x;
        int textY = std::max(0, safeBox.y - textSize.height - 6);

        cv::Rect textBg(
            textX,
            textY,
            textSize.width + 6,
            textSize.height + baseLine + 6
        );

        textBg = textBg & imageRect;

        //模块:绘制文字背景
        cv::rectangle(vis, textBg, color, -1);

        //模块:绘制类别名和置信度
        cv::putText(
            vis,
            text,
            cv::Point(textX + 3, textY + textSize.height + 2),
            cv::FONT_HERSHEY_SIMPLEX,
            0.6,
            cv::Scalar(0, 0, 0),
            1
        );
    }

    return vis;
}