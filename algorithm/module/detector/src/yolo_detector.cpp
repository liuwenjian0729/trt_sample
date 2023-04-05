#include"trt_detector.h"
#include"yolo_detector.h"
#include"timer.h"
#include"logging.h"

////////////////////////////////////////////////////////////////
// YoloDetector::YoloDetector
////////////////////////////////////////////////////////////////
YoloDetector::YoloDetector():
    pHandle_(nullptr)
{

}

////////////////////////////////////////////////////////////////
// YoloDetector::~YoloDetector
////////////////////////////////////////////////////////////////
YoloDetector::~YoloDetector()
{
    this->Flush();
}

////////////////////////////////////////////////////////////////
// YoloDetector::Initialize
////////////////////////////////////////////////////////////////
bool YoloDetector::Initialize(const DetectorOptions& options)
{
    TrtDetector* pDetector = new TrtDetector();
    
    TrtDetectorOptions opts;
    opts.engine_name = options.weight_path;
    opts.max_batch_size = 5;
    opts.conf_thr = 0.45;
    if(1 != pDetector->init(opts))
        return false;
    
    pHandle_ = reinterpret_cast<void*>(pDetector);
    return true;
}

////////////////////////////////////////////////////////////////
// YoloDetector::ProcessRequest
////////////////////////////////////////////////////////////////
int YoloDetector::ProcessRequest(cv::Mat& frame, std::vector<base::DetectBox>& output)
{
    int result = 0;
    if(pHandle_ == nullptr)
        result = -1;
    
    if(result == 0)
    {
        TrtDetector* pDetector = reinterpret_cast<TrtDetector*>(pHandle_);
        auto start = Timer::Now();
        int result = pDetector->detect(frame, output);
        auto end = Timer::Now();
        ALOGI("taking %lums", Timer::Duration(start, end));
    }

    return result;
}

////////////////////////////////////////////////////////////////
// YoloDetector::Flush
////////////////////////////////////////////////////////////////
void YoloDetector::Flush()
{
    if(nullptr != pHandle_)
    {
        delete pHandle_;
        pHandle_ = nullptr;
    }
}