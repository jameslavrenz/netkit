#pragma once

#include "tensor.hpp"
#include <cstdint>

// MobileNetV4 Universal Inverted Bottleneck (UIB) — building block of MNv4-Conv-Small.
struct MobileNetV4Uib
{
    int in_channels = 0;
    int out_channels = 0;
    int start_dw_kernel = 0;  // 0 = absent, else 3 or 5
    int middle_dw_kernel = 0; // 0 = absent, else 3 or 5
    int stride = 1;
    bool middle_dw_downsample = true;
    float expand_ratio = 1.0f;

    float* start_dw_weights = nullptr;
    float* start_dw_bias = nullptr;
    float* start_bn_scale = nullptr;
    float* start_bn_bias = nullptr;

    float* expand_weights = nullptr;
    float* expand_bias = nullptr;
    float* expand_bn_scale = nullptr;
    float* expand_bn_bias = nullptr;

    float* middle_dw_weights = nullptr;
    float* middle_dw_bias = nullptr;
    float* middle_bn_scale = nullptr;
    float* middle_bn_bias = nullptr;

    float* proj_weights = nullptr;
    float* proj_bias = nullptr;
    float* proj_bn_scale = nullptr;
    float* proj_bn_bias = nullptr;

    float* scratch = nullptr;
    uint32_t scratch_elems = 0;

    static uint32_t MakeDivisible(float value, uint32_t divisor = 8);
    uint32_t expanded_channels() const;
    uint32_t start_dw_stride() const;
    uint32_t middle_dw_stride() const;
    void output_spatial(uint32_t in_h, uint32_t in_w, uint32_t& out_h, uint32_t& out_w) const;
    bool has_residual() const;

    void forward(const Tensor& input, Tensor& output);
};
