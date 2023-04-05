#include"omt_tracker.h"
#include"byte_tracker.h"

using namespace byte_tracker;

////////////////////////////////////////////////////////////////
// OmtTracker::OmtTracker()
////////////////////////////////////////////////////////////////
OmtTracker::OmtTracker():
    pHandle_(nullptr)
{

}

////////////////////////////////////////////////////////////////
// OmtTracker::OmtTracker()
////////////////////////////////////////////////////////////////
OmtTracker::~OmtTracker()
{
    this->Flush();
}

////////////////////////////////////////////////////////////////
// OmtTracker::Initialize()
////////////////////////////////////////////////////////////////
bool OmtTracker::Initialize(const TrackerOptions& options)
{
    ByteTracker* pTracker = new ByteTracker(options.fps, options.buf_count);
    if(nullptr == pTracker)
    {
        // no memory
        return false;
    }

    pHandle_ = reinterpret_cast<void*>(pTracker);
    return true;
}

////////////////////////////////////////////////////////////////
// OmtTracker::ProcessRequest()
////////////////////////////////////////////////////////////////
int OmtTracker::ProcessRequest(const std::vector<base::DetectBox>& detections, std::vector<base::Object>& objects)
{
    int result = 0;
    if(nullptr == pHandle_)
        result = -1;

    if(0 == result)
    {
        ByteTracker* pTracker = reinterpret_cast<ByteTracker*>(pHandle_);
        std::vector<ByteTarget> targets = pTracker->update(detections);

        for(int i=0; i<targets.size(); i++)
        {
            base::Object object;
            object.track_id = targets[i].track_id;            
            object.box.xmin = targets[i].tlbr[0];
            object.box.ymin = targets[i].tlbr[1];
            object.box.xmax = targets[i].tlbr[2];
            object.box.ymax = targets[i].tlbr[3];
            objects.push_back(std::move(object));
        }
    }

    return result;
}

////////////////////////////////////////////////////////////////
// OmtTracker::Flush()
////////////////////////////////////////////////////////////////
void OmtTracker::Flush()
{
    if(nullptr != pHandle_)
    {
        delete pHandle_;
        pHandle_ = nullptr;
    }
}