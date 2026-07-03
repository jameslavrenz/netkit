#pragma once

#include "tensor.hpp"
#include <cstdint>

// ConvNeXt V2 block (Atto-compatible): 7x7 depthwise, LayerNorm NHWC, 4x MLP, GELU, GRN, residual.
struct ConvNeXtV2Block
{
    static constexpr int kDwKernel = 7;
    static constexpr int kDwPad = 3;
    static constexpr int kMlpRatio = 4;

    int channels = 0;
    float eps = 1e-6f;

    float* dw_weights = nullptr; // [C, 7, 7]
    float* dw_bias = nullptr;    // [C]
    float* ln_weight = nullptr;  // [C]
    float* ln_bias = nullptr;      // [C]
    float* pw1_weight = nullptr; // [4C, C]
    float* pw1_bias = nullptr;     // [4C]
    float* grn_gamma = nullptr;  // [4C]
    float* grn_beta = nullptr;     // [4C]
    float* pw2_weight = nullptr;   // [C, 4C]
    float* pw2_bias = nullptr;     // [C]

    float* scratch = nullptr; // caller-owned; at least H*W*4*C floats
    uint32_t scratch_elems = 0;

    void forward(const Tensor& input, Tensor& output);
};
