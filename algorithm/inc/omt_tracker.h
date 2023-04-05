#ifndef _OMT_TRACKER_H
#define _OMT_TRACKER_H
#include"base_tarcker.h"

class OmtTracker: public BaseTracker
{
public:
    OmtTracker();
    ~OmtTracker();

    /**********************************************************
     * 
     * @brief initialize a tracker by params
     * 
     * @param [in] options initial params
     * 
     * @return true if success 
     * 
    **********************************************************/
    bool Initialize(const TrackerOptions& options) override;

    /**********************************************************
     * 
     * @brief process a track request
     * 
     * @param [in] detections result of detector
     * @param [out] output tracked result.
     * 
     * @return SUCCESS if success 
     * 
    **********************************************************/
    int ProcessRequest(const std::vector<base::DetectBox>& detections, std::vector<base::Object>& output) override;

    /****************************************************
     * 
     * @brief flush source of detector
     * 
     * @return void
     * 
    ****************************************************/
    void Flush() override;

private:
    void* pHandle_;
};

#endif