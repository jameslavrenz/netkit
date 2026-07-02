#pragma once
#include "kernel_activation.hpp"
#include "tensor_access.hpp"
#include <cstdint>

struct Conv2D
{
    int kernel_size = 3;
    int stride = 1;
    int pad_h = 0;
    int pad_w = 0;
    int in_channels;
    int out_channels;

    float* weights; // [out][kh][kw][in]
    float* bias;    // [out]

    bool forward(const Tensor& input,
                 Tensor& output,
                 NetkitKernelActivation fuse_activation = NetkitKernelActivation::None);
};
