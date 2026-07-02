#pragma once
#include "netkit_backend.h"
#include "tensor_access.hpp"
#include <cstdint>

struct Conv2D
{
    int kernel_size = 3;
    int stride = 1;
    int in_channels;
    int out_channels;

    float* weights; // [out][kh][kw][in]
    float* bias;    // [out]

    // Returns true when CMSIS-NN handled the forward pass.
    bool forward(const Tensor& input,
                 Tensor& output,
                 NetkitBackendActivation fuse_activation = NETKIT_BACKEND_ACT_NONE);
};
