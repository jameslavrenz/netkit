#pragma once

#include "tensor.hpp"
#include <cstdint>

// ResNet BasicBlock (ResNet-18/34): conv3x3-BN-ReLU-conv3x3-BN + residual + ReLU.
struct ResNetBasicBlock
{
    int in_channels = 0;
    int out_channels = 0;
    int stride = 1;

    float* conv1_weights = nullptr; // [out, 3, 3, in]
    float* conv1_bias = nullptr;
    float* bn1_scale = nullptr;
    float* bn1_bias = nullptr;

    float* conv2_weights = nullptr; // [out, 3, 3, out]
    float* conv2_bias = nullptr;
    float* bn2_scale = nullptr;
    float* bn2_bias = nullptr;

    float* shortcut_weights = nullptr; // [out, 1, 1, in] when projection shortcut
    float* shortcut_bias = nullptr;
    float* shortcut_bn_scale = nullptr;
    float* shortcut_bn_bias = nullptr;

    float* scratch = nullptr;
    uint32_t scratch_elems = 0;

    bool has_identity_shortcut() const;
    void output_spatial(uint32_t in_h, uint32_t in_w, uint32_t& out_h, uint32_t& out_w) const;
    void forward(const Tensor& input, Tensor& output);
};
