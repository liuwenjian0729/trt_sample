#include<vector>
#include<cuda_runtime.h>
#include"yolov5_layer.h"
#include"trt_detector.h"
#include"cuda_utils.h"
#include"preprocess_imgs.h"
#include"detector_utils.h"
#include"logging.h"

static Logger gLogger;

static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;
static const char* INPUT_BLOB_NAME = "data";
static const char* OUTPUT_BLOB_NAME = "prob";
static const float NMS_THRESH = 0.5;

//////////////////////////////////////////////////////////
// TrtDetector::~TrtDetector()
//////////////////////////////////////////////////////////
TrtDetector::~TrtDetector()
{
    // recycling resources
    cudaStreamDestroy(stream_);

    CUDA_CHECK(cudaFree(buffers_[inputIndex_]));
    CUDA_CHECK(cudaFree(buffers_[outputIndex_]));

    if (blob_ != nullptr)
    {
        delete[] blob_;
        blob_ = nullptr;
    }

    if (prob_ != nullptr)
    {
        delete[] blob_;
        prob_ = nullptr;
    }

    if (context_ != nullptr) {
        context_->destroy();
        context_ = nullptr;
    }
    if (engine_ != nullptr)
    {
        engine_->destroy();
        engine_ = nullptr;
    }
    if (runtime_ != nullptr) {
        runtime_->destroy();
        runtime_ = nullptr;
    }
}

//////////////////////////////////////////////////////////
// TrtDetector::init()
//////////////////////////////////////////////////////////
int TrtDetector::init(const TrtDetectorOptions& options)
{
    // create tensorrt runtime
    runtime_ = nvinfer1::createInferRuntime(gLogger);
    if(nullptr == runtime_)
    {
        ALOGE("fail to create InferRuntime");
        return -1;
    }

    // create tensorrt engine
    if(!loadEngine(options.engine_name))
    {
        ALOGE("fail to create CudaEngine");
        return -1;   
    }

    // create context
    context_ = engine_->createExecutionContext();
    if(nullptr == context_)
    {
        ALOGE("fail to create ExecutionContext");
        return -1;
    }

    conf_thresh_ = options.conf_thr;

    // malloc memory from device or host
    int MaxBatchSize = options.max_batch_size;

    auto out_dims = engine_->getBindingDimensions(1);

    blob_ = new float[MaxBatchSize * 3 * INPUT_H * INPUT_W];
    prob_ = new float[MaxBatchSize * OUTPUT_SIZE];

    assert(engine_->getNbBindings() == 2);

    inputIndex_ = engine_->getBindingIndex(INPUT_BLOB_NAME);
    assert(mEngine->getBindingDataType(inputIndex_) == nvinfer1::DataType::kFLOAT);

    outputIndex_ = engine_->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(mEngine->getBindingDataType(outputIndex_) == nvinfer1::DataType::kFLOAT);

    // melloc memory from device
    CUDA_CHECK(cudaMalloc(&buffers_[inputIndex_], MaxBatchSize * INPUT_H * INPUT_W * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers_[outputIndex_], MaxBatchSize * OUTPUT_SIZE * sizeof(float)));

    // melloc memory from host
    CUDA_CHECK(cudaMallocHost((void**)&img_host_, MaxBatchSize * INPUT_H * INPUT_W * 3));
    CUDA_CHECK(cudaMalloc((void**)&img_device_, MaxBatchSize * INPUT_H * INPUT_W * 3));

    // create CUDA stream
    CUDA_CHECK(cudaStreamCreate(&stream_));

    return 1;
}

//////////////////////////////////////////////////////////
// TrtDetector::loadEngine()
//////////////////////////////////////////////////////////
bool TrtDetector::loadEngine(const std::string& engine_name)
{
    char *trtModelStream{nullptr};
    size_t size = 0;

    std::ifstream file(engine_name, std::ios::binary);
    if(file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);

        trtModelStream = new char[size];
        assert(trtModelStream != nullptr);
        file.read(trtModelStream, size);
        file.close();
    }
    else
    {
        ALOGE("fail to load engine from [%s], please check engine file", engine_name.c_str());
        return false;
    }

    engine_ = runtime_->deserializeCudaEngine(trtModelStream, size);
    if(nullptr == engine_)
    {
        ALOGE("fail to create trt engine");
        delete[] trtModelStream;
        trtModelStream = nullptr;

        return false;
    }

    delete[] trtModelStream;
    trtModelStream = nullptr;

    return true;
}


//////////////////////////////////////////////////////////
// TrtDetector::detect()
//////////////////////////////////////////////////////////
int TrtDetector::detect(const std::vector<cv::Mat>& frames, std::vector<std::vector<base::DetectBox>>& outputs)
{
    int result = 0;
    // preprocess
    int batch_size = frames.size();

    float* idx = (float*)buffers_[inputIndex_];  // memory on device
    for(int i=0; i<batch_size; i++)
    {
        // get current frame size
        size_t src_size = frames[i].cols * frames[i].rows * 3;
        size_t dst_size = INPUT_W * INPUT_H * 3;

        // read frame to host pinned memory
        memcpy(img_host_, frames[i].data, src_size);

        // memory copy from host to device
        CUDA_CHECK(cudaMemcpyAsync(img_device_, img_host_, src_size, cudaMemcpyHostToDevice, stream_));

        // preprocess on device
        preprocess_kernel_img(img_device_, frames[i].cols, frames[i].rows, idx, INPUT_W, INPUT_H, stream_);
        idx += dst_size;

        // execute cuda stream
        cudaStreamSynchronize(stream_);
    }

    // do inference
    context_->enqueue(batch_size, buffers_, stream_, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(prob_, buffers_[outputIndex_], batch_size * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream_));
    cudaStreamSynchronize(stream_);

    // parse result from prob_
    this->probParse(frames, outputs, batch_size);

    return result;
}

//////////////////////////////////////////////////////////
// TrtDetector::detect()
//////////////////////////////////////////////////////////
int TrtDetector::detect(const cv::Mat& frame, std::vector<base::DetectBox>& output, int batch_size)
{
    int result = 0;
    float* idx = (float*)buffers_[inputIndex_];

    // memory copy and image preprocess
    size_t src_size = frame.cols * frame.rows * 3;
    memcpy(img_host_, frame.data, src_size);
    CUDA_CHECK(cudaMemcpyAsync(img_device_, img_host_, src_size, cudaMemcpyHostToDevice, stream_));
    preprocess_kernel_img(img_device_, frame.cols, frame.rows, idx, INPUT_W, INPUT_H, stream_);

    // do inference
    context_->enqueue(batch_size, buffers_, stream_, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(prob_, buffers_[outputIndex_], batch_size * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream_));
    cudaStreamSynchronize(stream_);

    // parse result from prob_
    this->probParse(frame, output, batch_size);

    return result;
}

//////////////////////////////////////////////////////////
// TrtDetector::probParse()
//////////////////////////////////////////////////////////
void TrtDetector::probParse(const cv::Mat& frame, std::vector<base::DetectBox>& result, int batch_size)
{
    if(nullptr == prob_)
    {
        ALOGE("nothing in prob");
        return;
    }

    std::vector<std::vector<Yolo::Detection>> batch_res(batch_size);

    auto& res = batch_res[0];
    common::nms(res, &prob_[0], conf_thresh_, NMS_THRESH);
    unsigned int nums = res.size();

    result.resize(nums);
    for(int i=0; i<nums; i++)
    {
        cv::Rect rect = common::get_rect(frame, res[i].bbox);

        base::DetectBox box(rect.x, rect.y, rect.x + rect.width, rect.y + rect.height, (float)res[i].conf, (int)res[i].class_id);
        result.push_back(box);
    }
}

//////////////////////////////////////////////////////////
// TrtDetector::probParse()
//////////////////////////////////////////////////////////
void TrtDetector::probParse(const std::vector<cv::Mat>& frames, std::vector<std::vector<base::DetectBox>>& results, int batch_size)
{
    if(nullptr == prob_)
    {
        ALOGE("nothing in prob");
        return;
    }

    std::vector<std::vector<Yolo::Detection>> batch_res(batch_size);
    for(int i = 0; i < batch_size; i++)
    {
        auto& res = batch_res[i];
        common::nms(res, &prob_[i * OUTPUT_SIZE], conf_thresh_, NMS_THRESH);
        unsigned int nums = res.size();

        std::vector<base::DetectBox> result;
        result.resize(nums);
        for(int j=0; j<nums; j++)
        {
            cv::Rect rect = common::get_rect(frames[i], res[j].bbox);

            base::DetectBox box(rect.x, rect.y, rect.x + rect.width, rect.y + rect.height, (float)res[j].conf, (int)res[j].class_id);
            result.push_back(box);
        }
        results.push_back(result);
    }
}

