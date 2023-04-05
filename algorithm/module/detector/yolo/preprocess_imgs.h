#ifndef __PREPROCESS_IMGS_H
#define __PREPROCESS_IMGS_H

#include <cuda_runtime.h>
#include <cstdint>


struct AffineMatrix{
    float value[6];
};


void preprocess_kernel_img(uint8_t* src, int src_width, int src_height,
                           float* dst, int dst_width, int dst_height,
                           cudaStream_t stream);
#endif  // __PREPROCESS_IMGS_H