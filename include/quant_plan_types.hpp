#pragma once

#include "cmsis_hoisted_plan.hpp"
#include "quant_integer.hpp"
#include "quant_output.hpp"
#include "xnnpack_hoisted_plan.hpp"

#include <cstdint>

namespace CmsisQuantPlan
{

constexpr uint32_t kMaxPerChannel = 2048;

enum class LayerKind : uint8_t
{
    Conv2D,
    DepthwiseConv2D,
    MaxPool2D,
    AvgPool2D,
    FlattenView,
    Dense,
    DenseSoftmax,
    MobilenetV4Uib,
};

struct Conv2DPlan
{
    int32_t input_offset = 0;   // -input_zero_point (CMSIS-NN / TFLM)
    int32_t output_offset = 0;  // +output_zero_point (added after requant; TFLM convention)
    int32_t stride = 1;
    int32_t pad_h = 0;
    int32_t pad_w = 0;
    QuantInteger::QuantClamp clamp = QuantInteger::QuantClamp::None;
    // Float scales retained for XNNPACK qs8 (CMSIS uses multipliers/shifts only).
    float input_scale = 1.0f;
    float weight_scale = 1.0f;
    float output_scale = 1.0f;
    // Optional TFLite-style per-output-channel weight scales (points into arena).
    const float* weight_channel_scales = nullptr;
    uint32_t num_weight_channel_scales = 0;
    int32_t in_h = 0;
    int32_t in_w = 0;
    int32_t in_c = 0;
    int32_t out_h = 0;
    int32_t out_w = 0;
    int32_t out_c = 0;
    int32_t kernel_size = 0;
    int32_t workspace_bytes = 0;
    int32_t* multipliers = nullptr;
    int32_t* shifts = nullptr;
    // Baked output clamp (TF Lite quantized_activation_min/max).
    int32_t act_min = -128;
    int32_t act_max = 127;
    // Prepare-time bias' = bias + input_offset * sum(w) when weight_zp==0 and
    // input_offset!=0. Used by scalar ref interior / 1x1 paths (bit-exact).
    int32_t* bias_folded = nullptr;
#if NETKIT_CMSIS_PLAN_HOIST
    Conv2DCmsisHoist cmsis{};
#endif
#if NETKIT_XNNPACK_PLAN_HOIST
    XnnpackOpHoist xnn{};
#endif
    bool ready = false;
};

struct DepthwiseConv2DPlan
{
    int32_t input_offset = 0;   // -input_zero_point
    int32_t output_offset = 0;  // +output_zero_point (TFLM / CMSIS-NN)
    int32_t stride = 1;
    int32_t pad_h = 0;
    int32_t pad_w = 0;
    QuantInteger::QuantClamp clamp = QuantInteger::QuantClamp::None;
    float input_scale = 1.0f;
    float weight_scale = 1.0f;
    float output_scale = 1.0f;
    // Optional TFLite-style per-output-channel weight scales (points into arena).
    const float* weight_channel_scales = nullptr;
    uint32_t num_weight_channel_scales = 0;
    int32_t in_h = 0;
    int32_t in_w = 0;
    int32_t channels = 0;
    int32_t out_h = 0;
    int32_t out_w = 0;
    int32_t kernel_h = 0;
    int32_t kernel_w = 0;
    int32_t workspace_bytes = 0;
    int32_t* multipliers = nullptr;
    int32_t* shifts = nullptr;
    int32_t act_min = -128;
    int32_t act_max = 127;
    int8_t* weights_hwc = nullptr;
    // Prepare-time folded bias (see Conv2DPlan::bias_folded).
    int32_t* bias_folded = nullptr;
#if NETKIT_CMSIS_PLAN_HOIST
    DepthwiseConv2DCmsisHoist cmsis{};
#endif
#if NETKIT_XNNPACK_PLAN_HOIST
    XnnpackOpHoist xnn{};
#endif
    bool ready = false;
};

struct Pool2DPlan
{
    int32_t stride = 1;
    int32_t pad_h = 0;
    int32_t pad_w = 0;
    int32_t pool_h = 0;
    int32_t pool_w = 0;
    int32_t in_h = 0;
    int32_t in_w = 0;
    int32_t in_c = 0;
    int32_t out_h = 0;
    int32_t out_w = 0;
    float input_scale = 1.0f;
    int32_t input_zero_point = 0;
    float output_scale = 1.0f;
    int32_t output_zero_point = 0;
    QuantInteger::QuantClamp clamp = QuantInteger::QuantClamp::None;
#if NETKIT_CMSIS_PLAN_HOIST
    Pool2DCmsisHoist cmsis{};
#endif
#if NETKIT_XNNPACK_PLAN_HOIST
    XnnpackOpHoist xnn{};
#endif
    bool ready = false;
};

struct ElementwiseAddPlan
{
    // TF Lite ArithmeticParams-style (offsets are -zero_point).
    int32_t input1_offset = 0;
    int32_t input2_offset = 0;
    int32_t output_offset = 0;
    int32_t input1_mult = 0;
    int32_t input1_shift = 0;
    int32_t input2_mult = 0;
    int32_t input2_shift = 0;
    int32_t left_shift = 20;
    int32_t output_mult = 0;
    int32_t output_shift = 0;
    int32_t block_size = 0;
    int32_t act_min = -128;
    int32_t act_max = 127;
    bool ready = false;
};

struct MobilenetV4UibPlan
{
    DepthwiseConv2DPlan start_dw{};
    Conv2DPlan expand{};
    DepthwiseConv2DPlan middle_dw{};
    Conv2DPlan proj{};
    ElementwiseAddPlan add{};
    bool has_start_dw = false;
    bool has_middle_dw = false;
    bool has_residual = false;
    int32_t scratch_bytes = 0;
    int8_t* scratch = nullptr;
    int32_t in_h = 0;
    int32_t in_w = 0;
    int32_t in_c = 0;
    int32_t out_h = 0;
    int32_t out_w = 0;
    int32_t out_c = 0;
    // Optional fused XNNPACK subgraph for the UIB body (residual via xnn_binary_add).
    void* xnn_subgraph = nullptr;  // transient; deleted after runtime create
    void* xnn_runtime = nullptr;   // xnn_runtime_t
    uint32_t xnn_ext_input_id = 0;
    uint32_t xnn_ext_output_id = 1;
    bool xnn_subgraph_ready = false;
    bool xnn_subgraph_includes_residual = false;
    // Skip xnn_setup_runtime_v2 when I/O addresses are unchanged.
    const int8_t* xnn_bound_input = nullptr;
    int8_t* xnn_bound_output = nullptr;
    // Per-channel bias scales for subgraph tensors (heap; freed in DestroyUibSubgraph).
    float* xnn_start_dw_bias_scales = nullptr;
    float* xnn_expand_bias_scales = nullptr;
    float* xnn_middle_dw_bias_scales = nullptr;
    float* xnn_proj_bias_scales = nullptr;
    bool ready = false;
};

struct FcPlan
{
    int32_t input_offset = 0;   // -input_zero_point
    int32_t filter_offset = 0;  // -weight_zero_point
    int32_t output_offset = 0;  // +output_zero_point (TFLM / CMSIS-NN)
    QuantInteger::QuantClamp clamp = QuantInteger::QuantClamp::None;
    float input_scale = 1.0f;
    float weight_scale = 1.0f;
    float output_scale = 1.0f;
    const float* weight_channel_scales = nullptr;
    uint32_t num_weight_channel_scales = 0;
    int32_t in_features = 0;
    int32_t out_features = 0;
    // Per-tensor fallback (also first channel when per-channel arrays are set).
    int32_t multiplier = 0;
    int32_t shift = 0;
    // Per-output-channel requant (TFLite-style); length == out_features when set.
    int32_t* multipliers = nullptr;
    int32_t* shifts = nullptr;
    int32_t act_min = -128;
    int32_t act_max = 127;
    int32_t workspace_bytes = 0;
    int32_t* kernel_sums = nullptr;
    // Prepare-time folded bias (see Conv2DPlan::bias_folded).
    int32_t* bias_folded = nullptr;
#if NETKIT_CMSIS_PLAN_HOIST
    FcCmsisHoist cmsis{};
#endif
#if NETKIT_XNNPACK_PLAN_HOIST
    XnnpackOpHoist xnn{};
#endif
    bool ready = false;
};

struct SoftmaxPlan
{
    QuantOps::SoftmaxS8Params params{};
    int32_t row_size = 0;
    bool ready = false;
};

struct LayerPlan
{
    LayerKind kind = LayerKind::Conv2D;
    uint32_t output_elements = 0;
    Conv2DPlan conv{};
    DepthwiseConv2DPlan depthwise{};
    Pool2DPlan pool{};
    MobilenetV4UibPlan uib{};
    FcPlan fc{};
    SoftmaxPlan softmax{};
};

}  // namespace CmsisQuantPlan
