#ifndef _BASE_DETECTOR_H
#define _BASE_DETECTOR_H

#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include"data_type.h"

typedef struct DetectorOptions
{
    std::string weight_path;
};

/***********************************************
 * 
 * @brief abstract class for detector
 * 
***********************************************/
class BaseDetector
{
public:
    BaseDetector() = default;
    virtual ~BaseDetector() = default;

    /****************************************************
     * 
     * @brief initialize a detector
     * 
     * @param [in] options initial params for detector
     * 
     * @return true if success
     * 
    ****************************************************/
    virtual bool Initialize(const DetectorOptions& options) = 0;

    /****************************************************
     * 
     * @brief process a detect request
     * 
     * @param [in] frame input frame
     * @param [out] output detect box for object
     * 
     * @return SUCCESS if success
     * 
    ****************************************************/
    virtual int ProcessRequest(cv::Mat& frame, std::vector<base::DetectBox>& output) = 0;

    /****************************************************
     * 
     * @brief flush source of detector
     * 
     * @return void
     * 
    ****************************************************/
    virtual void Flush() = 0;

    BaseDetector(const BaseDetector&) = delete;
    BaseDetector& operator=(const BaseDetector&) = delete;
};

#endif