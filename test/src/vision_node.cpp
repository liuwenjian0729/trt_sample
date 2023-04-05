#include"vision_node.h"
#include"defines.h"

static const std::string ENGINE = "./algorithm/module/detector/model/yolov5s.engine";
static const std::string VID_OUT = "./video/out.mp4"; 

static const std::string CV_WINDOW = "display";

////////////////////////////////////////////////////////////////////////
// VisionPerceptionNode::VisionPerceptionNode()
////////////////////////////////////////////////////////////////////////
VisionPerceptionNode::VisionPerceptionNode():
    m_init(false),
    m_stop(false)
{
    // cv::namedWindow(CV_WINDOW);
    cv::Size size = cv::Size(1280, 720);
    writer_.open(VID_OUT, CV_FOURCC('P', 'I', 'M', '1'), 25.0, size, true);
};

////////////////////////////////////////////////////////////////////////
// VisionPerceptionNode::~VisionPerceptionNode()
////////////////////////////////////////////////////////////////////////
VisionPerceptionNode::~VisionPerceptionNode()
{
    this->shutdown();
    // cv::destroyWindow(CV_WINDOW);
    writer_.release();
};

////////////////////////////////////////////////////////////////////////
// VisionPerceptionNode::shutdown()
////////////////////////////////////////////////////////////////////////
void VisionPerceptionNode::shutdown()
{
    std::unique_lock<std::mutex> lock(m_mutex);
    m_stop = true;
    m_cond.notify_one();
    lock.unlock();
    if (m_thread.joinable())
    {
        m_thread.join();
    }
}

////////////////////////////////////////////////////////////////////////
// VisionPerceptionNode::init()
////////////////////////////////////////////////////////////////////////
bool VisionPerceptionNode::init()
{
    {
        detector_.reset(new YoloDetector());

        DetectorOptions options;
        options.weight_path = ENGINE;

        if(!detector_->Initialize(options))
        {
            ALOGE("init detector failed");
            return false;
        }
        else
        {
            ALOGI("init detector success");
        }
    }

    // init tracker
    {
        tracker_.reset(new OmtTracker());

        TrackerOptions options;
        options.fps = 30;
        options.buf_count = 30;

        if(!tracker_->Initialize(options))
        {
            ALOGE("init tracker failed");
            return false;
        }
        else
        {
            ALOGI("init tracker success");
        }
    }

    m_thread = std::thread(ThreadWorker(this));
    m_init = true;
    return true;
}

////////////////////////////////////////////////////////////////////////
// VisionPerceptionNode::addTask()
////////////////////////////////////////////////////////////////////////
void VisionPerceptionNode::addTask(const cv::Mat& frame)
{
    if(m_init)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_tasks.push(frame);
        m_cond.notify_one();
    }
}

////////////////////////////////////////////////////////////////////////
// VisionPerceptionNode::process()
////////////////////////////////////////////////////////////////////////
void VisionPerceptionNode::process(cv::Mat& frame)
{
    std::vector<base::DetectBox> outputs;
    if(0 == detector_->ProcessRequest(frame, outputs))
    {
        // tracker
        std::vector<base::Object> objects;
        if(0 == tracker_->ProcessRequest(outputs, objects))
        {
            ALOGI("detections nums: %d, tracked nums: %d", outputs.size(), objects.size());
            for(int i=0; i<objects.size(); i++)
            {
                draw_rect_on_frame(frame, objects[i]);
            }
            writer_.write(frame);
        }

        // cv::imshow(CV_WINDOW, frames[i]);
        // cv::waitKey(3);
    }
    else
    {
        ALOGE("fail to process request");
    }
}

////////////////////////////////////////////////////////////////////////
// VisionPerceptionNode::draw_rect_on_frame()
////////////////////////////////////////////////////////////////////////
void VisionPerceptionNode::draw_rect_on_frame(cv::Mat& frame, const base::Object& object)
{
    auto font_face = cv::FONT_HERSHEY_COMPLEX;   // font type
    auto font_scale = 1.0;  // font scale
    int thickness = 2;      // line type
    int baseline = 0;       // unkown

    std::stringstream text; // mark 
    text<<"ClassID: "<<object.track_id;

    cv::Scalar color = get_color(object.track_id);
    cv::Rect box{object.box.xmin, object.box.ymin, object.box.xmax - object.box.xmin, object.box.ymax - object.box.ymin};
    cv::putText(frame, text.str(), cv::Point(object.box.xmin, std::max(int(object.box.ymin) - 5, 0)), font_face, font_scale, color, thickness);
    cv::rectangle(frame, box, color, 2);
}

////////////////////////////////////////////////////////////////////////
// VisionPerceptionNode::get_color()
////////////////////////////////////////////////////////////////////////
cv::Scalar VisionPerceptionNode::get_color(int idx)
{
	idx += 3;
	return cv::Scalar(37 * idx % 255, 17 * idx % 255, 29 * idx % 255);
}
