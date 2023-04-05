#pragma once

#include<iostream>
#include<sys/syscall.h>
#include<stdio.h>
#include<unistd.h>
#include<NvInfer.h>

#define _GREEN_ "\033[1;32m"
#define _RED_ "\033[0;31m"
#define _END_COLOR_ "\033[0m"

#define TAG "Trt"
#define RM_PREFIX(x) strrchr(x,'/')?strrchr(x,'/')+1:x
#define ALOGI(format, argc...)  \
do {  \
    printf(_GREEN_);    \
    printf("%d %d I %s: %s %s %d " format"\n", (int)getpid(), (int)syscall(SYS_gettid), TAG, RM_PREFIX(__FILE__), __FUNCTION__, __LINE__, ##argc);   \
    printf(_END_COLOR_);    \
}while(0)

#define ALOGE(format, argc...)  \
do {  \
    printf(_RED_);    \
    printf("%d %d E %s: %s %s %d " format"\n", (int)getpid(), (int)syscall(SYS_gettid), TAG, RM_PREFIX(__FILE__), __FUNCTION__, __LINE__, ##argc);   \
    printf(_END_COLOR_);    \
}while(0)


class Logger : public nvinfer1::ILogger
{
    virtual void log(Severity severity, const char* msg)  noexcept
    {
        if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR))
        {}
    }
};

