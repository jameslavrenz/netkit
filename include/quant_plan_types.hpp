#pragma once

#include "cmsis_hoisted_plan.hpp"
#include "quant_output.hpp"

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
    int32_t input_offset = 0;
    int32_t output_offset = 0;
    int32_t stride = 1;
    int32_t pad_h = 0;
    int32_t pad_w = 0;
    bool apply_relu = false;
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
#if NETKIT_CMSIS_PLAN_HOIST
    Conv2DCmsisHoist cmsis{};
#endif
    bool ready = false;
};

struct DepthwiseConv2DPlan
{
    int32_t input_offset = 0;
    int32_t output_offset = 0;
    int32_t stride = 1;
    int32_t pad_h = 0;
    int32_t pad_w = 0;
    bool apply_relu = false;
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
    int8_t* weights_hwc = nullptr;
#if NETKIT_CMSIS_PLAN_HOIST
    DepthwiseConv2DCmsisHoist cmsis{};
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
#if NETKIT_CMSIS_PLAN_HOIST
    Pool2DCmsisHoist cmsis{};
#endif
    bool ready = false;
};

struct ElementwiseAddPlan
{
    int32_t input1_offset = 0;
    int32_t input2_offset = 0;
    int32_t output_offset = 0;
    int32_t input1_mult = 0;
    int32_t input1_shift = 0;
    int32_t input2_mult = 0;
    int32_t input2_shift = 0;
    int32_t left_shift = 0;
    int32_t output_mult = 0;
    int32_t output_shift = 0;
    int32_t block_size = 0;
    bool ready = false;
};

struct MobilenetV4UibPlan
{
    Conv2DPlan start_dw{};
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
    bool ready = false;
};

struct FcPlan
{
    int32_t input_offset = 0;
    int32_t filter_offset = 0;
    int32_t output_offset = 0;
    bool apply_relu = false;
    int32_t in_features = 0;
    int32_t out_features = 0;
    int32_t multiplier = 0;
    int32_t shift = 0;
    int32_t workspace_bytes = 0;
    int32_t* kernel_sums = nullptr;
#if NETKIT_CMSIS_PLAN_HOIST
    FcCmsisHoist cmsis{};
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
