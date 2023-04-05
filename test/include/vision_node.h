#pragma once
#include<iostream>
#include<algorithm>
#include<thread>
#include<mutex>
#include<queue>
#include<condition_variable>
#include<opencv2/opencv.hpp>
#include"yolo_detector.h"
#include"omt_tracker.h"

class VisionPerceptionNode
{
public:
    VisionPerceptionNode();
    ~VisionPerceptionNode();

    void shutdown();
 
    bool init();

    void addTask(const cv::Mat& frame);

    void process(cv::Mat& frame);
private:
    class ThreadWorker
    {
    public:
        ThreadWorker(VisionPerceptionNode* pHandle):
            handle_(pHandle)
        {}

        void operator()()
        {
            while(!handle_->m_stop)
            {
                std::unique_lock<std::mutex> lock(handle_->m_mutex);
                handle_->m_cond.wait(lock, [&](){return handle_->m_stop or !handle_->m_tasks.empty();});
                if(handle_->m_stop)
                    break;
                
                cv::Mat image = handle_->m_tasks.front();
                handle_->m_tasks.pop();
                lock.unlock();

                handle_->process(image);
            }
        }

    private:
        VisionPerceptionNode* handle_;
    };

    /****************************************************************
     * 
     * @brief draw rectangle on frame
     * 
     * @param [in] frame input frame
     * @param [in] object object info
     * 
     * @return null
     * 
    ****************************************************************/
    void draw_rect_on_frame(cv::Mat& frame, const base::Object& object);

    cv::Scalar get_color(int idx);

    bool m_stop;
    bool m_init;
    std::mutex m_mutex;
    std::thread m_thread;
    std::condition_variable m_cond;
    std::queue<cv::Mat> m_tasks;
    std::unique_ptr<BaseDetector> detector_;
    std::unique_ptr<BaseTracker> tracker_;
    cv::VideoWriter writer_;
};