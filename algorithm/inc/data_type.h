#ifndef _DATA_TYPE_H
#define _DATA_TYPE_H

#include<iostream>

namespace base {

typedef struct BBox2D
{
    float xmin;
    float ymin;
    float xmax;
    float ymax;
};

typedef struct DetectBox {
    DetectBox(float x1=0, float y1=0, float x2=0, float y2=0, 
        float confidence=0, float classID=-1, float trackID=-1) {
        this->xmin = x1;
        this->ymin = y1;
        this->xmax = x2;
        this->ymax = y2;
        this->confidence = confidence;
        this->classID = classID;
        this->trackID = trackID;
    }
    float xmin, ymin, xmax, ymax;   // <tlbr
    float confidence;
    float classID;
    float trackID;
} DetectBox;

typedef struct Object
{
    uint64_t time_stamp;
    uint32_t class_id;
    int track_id;
    std::string uuid;
    float   confidence;
    BBox2D  box;   
};

}   // namespace base
#endif