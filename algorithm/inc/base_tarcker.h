#ifndef _BASE_TRACKER_H
#define _BASE_TRACKER_H

#include<iostream>
#include<string>
#include<vector>
#include<opencv2/opencv.hpp>
#include"data_type.h"

typedef struct TrackerOptions
{
    int fps;
    int buf_count;
};

/***********************************************
 * 
 * @brief abstract class for tracker
 * 
***********************************************/
class BaseTracker
{
public:
    BaseTracker() = default;
    virtual ~BaseTracker() = default;

    /**********************************************************
     * 
     * @brief initialize a tracker by params
     * 
     * @param [in] options initial params
     * 
     * @return true if success 
     * 
    **********************************************************/
    virtual bool Initialize(const TrackerOptions& options) = 0;

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
    virtual int ProcessRequest(const std::vector<base::DetectBox>& detections, std::vector<base::Object>& output) = 0;

    /****************************************************
     * 
     * @brief flush source of detector
     * 
     * @return void
     * 
    ****************************************************/
    virtual void Flush() = 0;

    BaseTracker(const BaseTracker&) = delete;
    BaseTracker& operator=(const BaseTracker&) = delete;    
};

#endif