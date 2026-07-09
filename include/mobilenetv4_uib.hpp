#pragma once

#include "nk_format.hpp"
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

    // Quantized UIB path (BN pre-folded at export; int8 weights + per-sub-op quant).
    bool quant_enabled = false;
    int8_t* start_dw_weights_q = nullptr;
    int32_t* start_dw_bias_q = nullptr;
    int8_t* expand_weights_q = nullptr;
    int32_t* expand_bias_q = nullptr;
    int8_t* middle_dw_weights_q = nullptr;
    int32_t* middle_dw_bias_q = nullptr;
    int8_t* proj_weights_q = nullptr;
    int32_t* proj_bias_q = nullptr;
    NkFormat::MlpLayerQuantDesc start_dw_quant{};
    NkFormat::MlpLayerQuantDesc expand_quant{};
    NkFormat::MlpLayerQuantDesc middle_dw_quant{};
    NkFormat::MlpLayerQuantDesc proj_quant{};
    float block_input_scale = 1.0f;
    int32_t block_input_zero_point = 0;
    int8_t* scratch_i8 = nullptr;
    uint32_t scratch_i8_bytes = 0;

    // Set once forward() has folded each BatchNorm into its preceding conv's
    // weights/bias (see FoldBatchNorm). Guards against re-folding on later calls.
    bool bn_folded = false;

    static uint32_t MakeDivisible(float value, uint32_t divisor = 8);
    uint32_t expanded_channels() const;
    uint32_t start_dw_stride() const;
    uint32_t middle_dw_stride() const;
    void output_spatial(uint32_t in_h, uint32_t in_w, uint32_t& out_h, uint32_t& out_w) const;
    bool has_residual() const;

    // Fold every BatchNorm (out = in*scale + beta) into the weights/bias of the
    // conv/depthwise that produces it, so forward() can drop the standalone BN
    // pass and fuse the following ReLU into the conv epilogue (matches how a
    // TFLite converter folds BN + fused activation). Idempotent via bn_folded.
    void FoldBatchNorm();

    void forward(const Tensor& input, Tensor& output);

    void forward_quant(const int8_t* input, int8_t* output, uint32_t in_h, uint32_t in_w) const;
};
