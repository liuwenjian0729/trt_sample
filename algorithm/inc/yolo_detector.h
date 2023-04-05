#ifndef _YOLO_DETECTOR_H_
#define _YOLO_DETECTOR_H_
#include"base_detector.h"

class YoloDetector: public BaseDetector
{
public:
    YoloDetector();
    ~YoloDetector();

    /*********************************************************
     * 
     * @brief initialize a yolo detector.
     * 
     * @param [in] options intial params
     * 
     * @return true if success
     * 
    *********************************************************/
    bool Initialize(const DetectorOptions& options) override;

    /*********************************************************
     * 
     * @brief detect object on frame
     * 
     * @param [in] frame input frame
     * @param [out] output detected objects
     * 
     * @return return SUCCESS if success
     * 
    *********************************************************/
    int ProcessRequest(cv::Mat& frame, std::vector<base::DetectBox>& output) override;

    /*********************************************************
     * 
     * @brief clear detector source
     * 
     * @return void
     * 
    *********************************************************/
    void Flush() override;

private:
    void* pHandle_;
};

#endif