#pragma once

#include<iostream>
#include<chrono>
#include<fstream>
#include<assert.h>
#include<vector>
#include<NvInfer.h>
#include<opencv2/opencv.hpp>
#include"data_type.h"

struct TrtDetectorOptions
{
    std::string engine_name;
    float conf_thr;
    int max_batch_size = 1;
};

class TrtDetector
{
public:
    TrtDetector() = default;
    ~TrtDetector();

    /**************************************************************
     * 
     * @brief initialize a trt detector
     * 
     * @param [in] options parameters of detector
     * 
     * @return SUCCESS if success
     * 
    **************************************************************/
    int init(const TrtDetectorOptions& options);

    /**************************************************************
     * 
     * @brief object detect for multi frames
     * 
     * @param [in] frames input frames
     * @param [out] outputs detected object from frames
     * 
     * @return SUCCESS if success
     * 
    **************************************************************/
    int detect(const std::vector<cv::Mat>& frames, std::vector<std::vector<base::DetectBox>>& outputs);

    /**************************************************************
     * 
     * @brief object detect for single frame
     * 
     * @param [in] frame input frame
     * @param [out] output detected object from frame
     * @param [in] batch_size batch size with default value
     * 
     * @return SUCCESS if success
     * 
    **************************************************************/
    int detect(const cv::Mat& frames, std::vector<base::DetectBox>& output, int batch_size=1);

private:
    TrtDetector(const TrtDetector&) = delete;
    TrtDetector& operator=(const TrtDetector&) = delete;

    /**************************************************************
     * 
     * @brief load weight to memory.
     * 
     * @param [in] engine_name engine file
     * 
     * @return true if success
     * 
    **************************************************************/
    bool loadEngine(const std::string& engine_name);

    /**************************************************************
     * 
     * @brief parse multi frames prob.
     * 
     * @param [in] frame input frames
     * @param [out] result object detection boxes
     * @param [in] batch_size size of frames
     * 
     * @return void
     * 
    **************************************************************/
    void probParse(const std::vector<cv::Mat>& frames, std::vector<std::vector<base::DetectBox>>& results, int batch_size);

    /**************************************************************
     * 
     * @brief parse single frame prob.
     * 
     * @param [in] frame input frame
     * @param [out] result object detection box
     * @param [in] batch_size default value 1
     * 
     * @return void
     * 
    **************************************************************/
    void probParse(const cv::Mat& frame, std::vector<base::DetectBox>& result, int batch_size=1);

    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    cudaStream_t stream_;

    void* buffers_[2];      // memory melloc from device
    int inputIndex_;        // memory index of input
    int outputIndex_;       // memory index of output

    float* blob_ = nullptr; // memory melloc from host
    float* prob_ = nullptr; // memory melloc from host

    uint8_t* img_host_ = nullptr;   // pinned memory melloc from host
    uint8_t* img_device_ = nullptr; // memory malloc from device

    float conf_thresh_;
};
