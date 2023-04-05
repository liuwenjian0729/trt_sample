#pragma once
#include<iostream>
#include<algorithm>
#include<vector>
#include<string.h>
#include<map>
#include <opencv2/opencv.hpp>
#include"yolov5_layer.h"

namespace common {

static bool cmpDD(const Yolo::Detection& a, const Yolo::Detection& b)
{
    return a.conf > b.conf;
}

static float iou(float *lbox, float *rbox)
{
    float interBox[] = {
        (std::max)(lbox[0] - lbox[2] * 0.5f , rbox[0] - rbox[2] * 0.5f), //left
        (std::min)(lbox[0] + lbox[2] * 0.5f , rbox[0] + rbox[2] * 0.5f), //right
        (std::max)(lbox[1] - lbox[3] * 0.5f , rbox[1] - rbox[3] * 0.5f), //top
        (std::min)(lbox[1] + lbox[3] * 0.5f , rbox[1] + rbox[3] * 0.5f), //bottom
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0])*(interBox[3] - interBox[2]);
    return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}

static void nms(std::vector<Yolo::Detection>& res, float* output, float conf_thresh, float nms_thresh)
{
    int det_size = sizeof(Yolo::Detection) / sizeof(float);
    std::map<float, std::vector<Yolo::Detection>> m;
    for (int i = 0; i < output[0] && i < Yolo::MAX_OUTPUT_BBOX_COUNT; ++i)
    {
        if (output[1 + det_size * i + 4] <= conf_thresh)
            continue;
        Yolo::Detection det;
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        if (m.count(det.class_id) == 0)
        {
            m.emplace(det.class_id, std::vector<Yolo::Detection>());
        }
        m[det.class_id].push_back(det);
    }

    for (auto it = m.begin(); it != m.end(); it++)
    {
        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), cmpDD);
        for (size_t m = 0; m < dets.size(); ++m)
        {
            auto& item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n)
            {
                if (iou(item.bbox, dets[n].bbox) > nms_thresh)
                {
                    dets.erase(dets.begin() + n);
                    --n;
                }
            }
        }
    }
}

static cv::Rect get_rect(const cv::Mat& img, float bbox[4])
{
    int l, r, t, b;
    float r_w = Yolo::INPUT_W / (img.cols * 1.0);
    float r_h = Yolo::INPUT_H / (img.rows * 1.0);
    if (r_h > r_w) {
        l = bbox[0] - bbox[2] * 0.5;
        r = bbox[0] + bbox[2] * 0.5;
        // t = bbox[1] - bbox[3] * 0.5 - (Yolo::INPUT_H - r_w * img.rows) * 0.5;
        // b = bbox[1] + bbox[3] * 0.5 - (Yolo::INPUT_H - r_w * img.rows) * 0.5;
        t = bbox[1] - bbox[3] * 0.5f;
        b = bbox[1] + bbox[3] * 0.5f;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    }
    else
    {
        l = bbox[0] - bbox[2] * 0.5 - (Yolo::INPUT_W - r_h * img.cols) * 0.5;
        r = bbox[0] + bbox[2] * 0.5 - (Yolo::INPUT_W - r_h * img.cols) * 0.5;
        t = bbox[1] - bbox[3] * 0.5f;
        b = bbox[1] + bbox[3] * 0.5f;
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    return cv::Rect(l, t, r - l, b - t);
}

}
