#include<iostream>
#include"vision_node.h"

int main()
{
    cv::VideoCapture video("./video/video.mp4");
    if(!video.isOpened())
    {
        std::cerr<<"fail to open video"<<std::endl;
        return -1;
    }
    else
    {
        std::cout<<"total: "<<video.get(cv::CAP_PROP_FRAME_COUNT)<<std::endl;
    }

    VisionPerceptionNode entity;
    if(!entity.init())
    {
        std::cout<<"init fail"<<std::endl;
        return -1;
    }

    int count = 0;
    while(count<video.get(cv::CAP_PROP_FRAME_COUNT))
    {
        count++;
        cv::Mat frame;
        video.read(frame);
        if(!frame.empty())
            entity.addTask(frame);
        
        cv::waitKey(30);
    }

    while (true)
    {
        char cmd = std::cin.get();
        if(cmd == 'q')
        {
            break;
        }
    }
    
    return 0;
}
