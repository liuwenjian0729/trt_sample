#ifndef _CUDA_UTILS_H
#define _CUDA_UTILS_H
#include<cuda_runtime_api.h>

#define CUDA_CHECK(condition)                                                              \
    do {                                                                                   \
        cudaError_t ecode = condition;                                                     \
        if (ecode != cudaSuccess) {                                                        \
            std::cerr << "CUDA error " << ecode << " at " << __FILE__ << ":" << __LINE__;  \
            assert(0);                                                                     \
        }                                                                                  \
    } while (0)

#endif