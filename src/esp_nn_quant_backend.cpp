/*
 * ESP-NN int8 adapters for netkit quantized conv / pool / FC (TFLite-style).
 * Mirrors src/cmsis_nn_quant_backend.cpp; always linked (stubs when disabled).
 */
#include "esp_nn_quant.hpp"
#include "arena.hpp"
#include "kernel_workspace.hpp"
#include "nk_op_detail.hpp"
#include "netkit_config.h"
#include "quant_integer.hpp"
#include "quant_trace.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(NETKIT_USE_ESP_NN) && NETKIT_USE_ESP_NN && NETKIT_ESP_NN_ALLOWED

#include <esp_nn.h>

namespace
{
    void QuantizeMultiplier(double double_multiplier, int32_t* multiplier, int32_t* shift)
    {
        if (double_multiplier <= 0.0)
        {
            *multiplier = 0;
            *shift = 0;
            return;
        }

        int shift_val = 0;
        double q = std::frexp(double_multiplier, &shift_val);
        int64_t q_fixed = static_cast<int64_t>(std::llround(q * (1LL << 31)));
        if (q_fixed == (1LL << 31))
        {
            q_fixed /= 2;
            ++shift_val;
        }

        if (shift_val < -31)
        {
            shift_val = 0;
            q_fixed = 0;
        }
        if (shift_val > 30)
        {
            shift_val = 30;
            q_fixed = (1LL << 31) - 1;
        }

        *multiplier = static_cast<int32_t>(q_fixed);
        *shift = shift_val;
    }

    void ActivationClamp(QuantInteger::QuantClamp clamp,
                         float output_scale,
                         int32_t output_zero_point,
                         int32_t* act_min,
                         int32_t* act_max)
    {
        *act_min = -128;
        *act_max = 127;
        QuantInteger::QuantClampRange(clamp, output_scale, output_zero_point, act_min, act_max);
    }

    bool FillPerChannelQuant(const NkFormat::MlpLayerQuantDesc& quant,
                             int out_channels,
                             int32_t* multipliers,
                             int32_t* shifts)
    {
        if (out_channels <= 0 || quant.output_scale <= 0.0f)
            return false;

        const bool per_channel =
            quant.weight_channel_scales != nullptr &&
            quant.num_weight_channel_scales == static_cast<uint32_t>(out_channels);

        if (!per_channel)
        {
            const double effective =
                static_cast<double>(quant.input_scale) * static_cast<double>(quant.weight_scale) /
                static_cast<double>(quant.output_scale);
            int32_t multiplier = 0;
            int32_t shift = 0;
            QuantizeMultiplier(effective, &multiplier, &shift);
            for (int oc = 0; oc < out_channels; ++oc)
            {
                multipliers[oc] = multiplier;
                shifts[oc] = shift;
            }
            return true;
        }

        for (int oc = 0; oc < out_channels; ++oc)
        {
            const double effective =
                static_cast<double>(quant.input_scale) *
                static_cast<double>(quant.weight_channel_scales[oc]) /
                static_cast<double>(quant.output_scale);
            QuantizeMultiplier(effective, &multipliers[oc], &shifts[oc]);
        }
        return true;
    }

    bool BindScratch(int32_t required_bytes)
    {
        void* buf = nullptr;
        int32_t size = 0;
        if (!BindCmsisWorkspace(buf, size, required_bytes))
            return false;
        if (required_bytes > 0 && buf != nullptr)
        {
            esp_nn_set_conv_scratch_buf(buf);
            esp_nn_set_depthwise_conv_scratch_buf(buf);
        }
        return true;
    }

    uint32_t ActiveWorkspaceBytes()
    {
        KernelWorkspace* ws = GetActiveKernelWorkspace();
        return ws ? static_cast<uint32_t>(ws->size_bytes) : 0;
    }
}  // namespace

namespace EspNnQuant
{

void FinalizeConv2DPlan(CmsisQuantPlan::Conv2DPlan& plan)
{
    if (!plan.ready || plan.in_h <= 0 || plan.in_w <= 0 || plan.in_c <= 0 || plan.out_c <= 0 ||
        plan.kernel_size <= 0)
        return;

    const data_dims_t input_dims = {
        .width = plan.in_w, .height = plan.in_h, .channels = plan.in_c, .extra = 1};
    const data_dims_t filter_dims = {
        .width = plan.kernel_size,
        .height = plan.kernel_size,
        .channels = plan.in_c,
        .extra = plan.out_c};
    const data_dims_t output_dims = {
        .width = plan.out_w, .height = plan.out_h, .channels = plan.out_c, .extra = 1};

    int32_t act_min = -128;
    int32_t act_max = 127;
    ActivationClamp(plan.clamp, plan.output_scale, plan.output_offset, &act_min, &act_max);

    const conv_params_t conv_params = {
        .in_offset = plan.input_offset,
        .out_offset = plan.output_offset,
        .stride = {.width = plan.stride, .height = plan.stride},
        .padding = {.width = plan.pad_w, .height = plan.pad_h},
        .dilation = {.width = 1, .height = 1},
        .activation = {.min = act_min, .max = act_max},
    };

    const int scratch = esp_nn_get_conv_scratch_size(&input_dims, &filter_dims, &output_dims,
                                                     &conv_params);
    if (scratch > plan.workspace_bytes)
        plan.workspace_bytes = scratch;
}

void FinalizeDepthwiseConv2DPlan(CmsisQuantPlan::DepthwiseConv2DPlan& plan)
{
    if (!plan.ready || plan.in_h <= 0 || plan.in_w <= 0 || plan.channels <= 0 ||
        plan.kernel_h <= 0 || plan.kernel_w <= 0)
        return;

    const data_dims_t input_dims = {
        .width = plan.in_w, .height = plan.in_h, .channels = plan.channels, .extra = 1};
    const data_dims_t filter_dims = {
        .width = plan.kernel_w,
        .height = plan.kernel_h,
        .channels = plan.channels,
        .extra = 1};
    const data_dims_t output_dims = {
        .width = plan.out_w, .height = plan.out_h, .channels = plan.channels, .extra = 1};

    int32_t act_min = -128;
    int32_t act_max = 127;
    ActivationClamp(plan.clamp, plan.output_scale, plan.output_offset, &act_min, &act_max);

    const dw_conv_params_t dw_params = {
        .in_offset = plan.input_offset,
        .out_offset = plan.output_offset,
        .ch_mult = 1,
        .stride = {.width = plan.stride, .height = plan.stride},
        .padding = {.width = plan.pad_w, .height = plan.pad_h},
        .dilation = {.width = 1, .height = 1},
        .activation = {.min = act_min, .max = act_max},
    };

    int scratch = esp_nn_get_depthwise_conv_scratch_size(&input_dims, &filter_dims, &output_dims,
                                                         &dw_params);
    // Runtime path may still need CHW→HWC pack space when weights_hwc is unset.
    if (!plan.weights_hwc)
    {
        scratch += plan.kernel_h * plan.kernel_w * plan.channels;
    }
    if (scratch > plan.workspace_bytes)
        plan.workspace_bytes = scratch;
}

void FinalizePool2DPlan(CmsisQuantPlan::Pool2DPlan& /*plan*/) {}

bool FinalizeFcPlan(CmsisQuantPlan::FcPlan& plan,
                    const int8_t* /*weights*/,
                    const int32_t* /*bias*/,
                    Arena& /*arena*/)
{
    return plan.ready;
}

bool TryConv2dNhwcQuantPlan(const CmsisQuantPlan::Conv2DPlan& plan,
                            const int8_t* input,
                            const int8_t* weights,
                            const int32_t* bias,
                            int8_t* output)
{
    if (!plan.ready || !input || !weights || !bias || !output || !plan.multipliers || !plan.shifts)
        return false;

    const data_dims_t input_dims = {
        .width = plan.in_w, .height = plan.in_h, .channels = plan.in_c, .extra = 1};
    const data_dims_t filter_dims = {
        .width = plan.kernel_size,
        .height = plan.kernel_size,
        .channels = plan.in_c,
        .extra = plan.out_c};
    const data_dims_t output_dims = {
        .width = plan.out_w, .height = plan.out_h, .channels = plan.out_c, .extra = 1};

    int32_t act_min = -128;
    int32_t act_max = 127;
    ActivationClamp(plan.clamp, plan.output_scale, plan.output_offset, &act_min, &act_max);

    const conv_params_t conv_params = {
        .in_offset = plan.input_offset,
        .out_offset = plan.output_offset,
        .stride = {.width = plan.stride, .height = plan.stride},
        .padding = {.width = plan.pad_w, .height = plan.pad_h},
        .dilation = {.width = 1, .height = 1},
        .activation = {.min = act_min, .max = act_max},
    };
    const quant_data_t quant_data = {.shift = plan.shifts, .mult = plan.multipliers};

    const int scratch = esp_nn_get_conv_scratch_size(&input_dims, &filter_dims, &output_dims,
                                                     &conv_params);
    if (scratch > 0 && !BindScratch(scratch))
        return false;

    esp_nn_conv_s8(&input_dims, input, &filter_dims, weights, bias, &output_dims, output,
                   &conv_params, &quant_data);
    QuantTrace::RecordConv2dCmsisOk();
    return true;
}

bool TryDepthwiseConv2dNhwcQuantPlan(const CmsisQuantPlan::DepthwiseConv2DPlan& plan,
                                     const int8_t* input,
                                     const int8_t* weights,
                                     const int32_t* bias,
                                     int8_t* output)
{
    if (!plan.ready || !input || !weights || !bias || !output || !plan.multipliers ||
        !plan.shifts)
        return false;

    const data_dims_t input_dims = {
        .width = plan.in_w, .height = plan.in_h, .channels = plan.channels, .extra = 1};
    const data_dims_t filter_dims = {
        .width = plan.kernel_w,
        .height = plan.kernel_h,
        .channels = plan.channels,
        .extra = 1};
    const data_dims_t output_dims = {
        .width = plan.out_w, .height = plan.out_h, .channels = plan.channels, .extra = 1};

    int32_t act_min = -128;
    int32_t act_max = 127;
    ActivationClamp(plan.clamp, plan.output_scale, plan.output_offset, &act_min, &act_max);

    const dw_conv_params_t dw_params = {
        .in_offset = plan.input_offset,
        .out_offset = plan.output_offset,
        .ch_mult = 1,
        .stride = {.width = plan.stride, .height = plan.stride},
        .padding = {.width = plan.pad_w, .height = plan.pad_h},
        .dilation = {.width = 1, .height = 1},
        .activation = {.min = act_min, .max = act_max},
    };
    const quant_data_t quant_data = {.shift = plan.shifts, .mult = plan.multipliers};

    // AOT lowered plans pass HWC (plan.weights_hwc / kW*Hwc). Runtime QuantOps passes
    // CHW and leaves plan.weights_hwc null — repack into workspace in that case.
    const bool weights_are_hwc = (plan.weights_hwc != nullptr);
    const std::size_t kernel_area = static_cast<std::size_t>(plan.kernel_h) *
                                    static_cast<std::size_t>(plan.kernel_w) *
                                    static_cast<std::size_t>(plan.channels);
    const int scratch = esp_nn_get_depthwise_conv_scratch_size(&input_dims, &filter_dims,
                                                               &output_dims, &dw_params);
    const int32_t total_need =
        scratch + (weights_are_hwc ? 0 : static_cast<int32_t>(kernel_area));

    void* buf = nullptr;
    int32_t size = 0;
    if (total_need > 0)
    {
        if (!BindCmsisWorkspace(buf, size, total_need) || !buf)
            return false;
        if (scratch > 0)
            esp_nn_set_depthwise_conv_scratch_buf(buf);
    }

    const int8_t* weights_hwc = weights;
    if (!weights_are_hwc)
    {
        int8_t* packed = static_cast<int8_t*>(buf) + scratch;
        for (int c = 0; c < plan.channels; ++c)
        {
            for (int y = 0; y < plan.kernel_h; ++y)
            {
                for (int x = 0; x < plan.kernel_w; ++x)
                {
                    packed[(y * plan.kernel_w + x) * plan.channels + c] =
                        weights[(c * plan.kernel_h + y) * plan.kernel_w + x];
                }
            }
        }
        weights_hwc = packed;
    }

    esp_nn_depthwise_conv_s8(&input_dims, input, &filter_dims, weights_hwc, bias, &output_dims,
                             output, &dw_params, &quant_data);
    QuantTrace::RecordConv2dCmsisOk();
    return true;
}

bool TryMaxPool2dNhwcQuantPlan(const CmsisQuantPlan::Pool2DPlan& plan,
                               const int8_t* input,
                               int8_t* output)
{
    if (!plan.ready || !input || !output)
        return false;

    int32_t act_min = -128;
    int32_t act_max = 127;
    ActivationClamp(plan.clamp, plan.output_scale, plan.output_zero_point, &act_min, &act_max);

    esp_nn_max_pool_s8(input,
                       static_cast<uint16_t>(plan.in_w),
                       static_cast<uint16_t>(plan.in_h),
                       output,
                       static_cast<uint16_t>(plan.out_w),
                       static_cast<uint16_t>(plan.out_h),
                       static_cast<uint16_t>(plan.stride),
                       static_cast<uint16_t>(plan.stride),
                       static_cast<uint16_t>(plan.pool_w),
                       static_cast<uint16_t>(plan.pool_h),
                       static_cast<uint16_t>(plan.pad_w),
                       static_cast<uint16_t>(plan.pad_h),
                       act_min,
                       act_max,
                       static_cast<uint16_t>(plan.in_c));
    return true;
}

bool TryFullyConnectedQuantPlan(const CmsisQuantPlan::FcPlan& plan,
                                const int8_t* input,
                                const int8_t* weights,
                                const int32_t* bias,
                                int8_t* output_int8)
{
    if (!plan.ready || !input || !weights || !bias || !output_int8 || !plan.multipliers ||
        !plan.shifts)
        return false;

    int32_t act_min = -128;
    int32_t act_max = 127;
    ActivationClamp(plan.clamp, plan.output_scale, plan.output_offset, &act_min, &act_max);

    const bool per_channel =
        plan.weight_channel_scales != nullptr &&
        plan.num_weight_channel_scales == static_cast<uint32_t>(plan.out_features);

    if (per_channel)
    {
        esp_nn_fully_connected_per_ch_s8(input,
                                         plan.input_offset,
                                         static_cast<uint16_t>(plan.in_features),
                                         weights,
                                         plan.filter_offset,
                                         bias,
                                         output_int8,
                                         static_cast<uint16_t>(plan.out_features),
                                         plan.output_offset,
                                         plan.shifts,
                                         plan.multipliers,
                                         act_min,
                                         act_max);
    }
    else
    {
        esp_nn_fully_connected_s8(input,
                                  plan.input_offset,
                                  static_cast<uint16_t>(plan.in_features),
                                  weights,
                                  plan.filter_offset,
                                  bias,
                                  output_int8,
                                  static_cast<uint16_t>(plan.out_features),
                                  plan.output_offset,
                                  plan.shifts[0],
                                  plan.multipliers[0],
                                  act_min,
                                  act_max);
    }
    QuantTrace::RecordFcCmsisOk();
    return true;
}

bool TrySoftmaxS8Plan(const CmsisQuantPlan::SoftmaxPlan& plan, const int8_t* input, int8_t* output)
{
    if (!plan.ready || !input || !output || plan.row_size <= 0)
        return false;

    const int32_t scratch = esp_nn_get_softmax_scratch_size(plan.row_size, 1);
    if (scratch > 0)
    {
        void* buf = nullptr;
        int32_t size = 0;
        if (!BindCmsisWorkspace(buf, size, scratch) || !buf)
            return false;
        esp_nn_set_softmax_scratch_buf(buf);
    }

    esp_nn_softmax_s8(input, 1, plan.row_size, plan.params.mult, plan.params.shift,
                      plan.params.diff_min, output);
    QuantTrace::RecordSoftmaxCmsisOk();
    return true;
}

bool TryConv2dNhwcQuant(const int8_t* input,
                        uint32_t in_h,
                        uint32_t in_w,
                        uint32_t in_c,
                        const int8_t* weights,
                        const int32_t* bias,
                        int kernel_size,
                        int stride,
                        int pad_h,
                        int pad_w,
                        int pad_h_end,
                        int pad_w_end,
                        int out_channels,
                        const NkFormat::MlpLayerQuantDesc& quant,
                        bool apply_relu,
                        int8_t* output)
{
    const uint32_t workspace_bytes = ActiveWorkspaceBytes();

    if (!input || !weights || !bias || !output || out_channels <= 0)
    {
        QuantTrace::RecordConv2dCmsisFail(QuantTrace::Conv2dFail::NullPtr, 0, workspace_bytes);
        return false;
    }
    if (pad_h != pad_h_end || pad_w != pad_w_end)
    {
        QuantTrace::RecordConv2dCmsisFail(QuantTrace::Conv2dFail::AsymmetricPad, 0, workspace_bytes);
        return false;
    }
    if (out_channels > 512)
    {
        QuantTrace::RecordConv2dCmsisFail(QuantTrace::Conv2dFail::TooManyChannels, 0, workspace_bytes);
        return false;
    }

    const uint32_t out_h =
        nk_op_detail::CalcOutputDimAsymmetric(in_h, kernel_size, stride, pad_h, pad_h_end);
    const uint32_t out_w =
        nk_op_detail::CalcOutputDimAsymmetric(in_w, kernel_size, stride, pad_w, pad_w_end);

    int32_t multipliers[512];
    int32_t shifts[512];
    if (!FillPerChannelQuant(quant, out_channels, multipliers, shifts))
    {
        QuantTrace::RecordConv2dCmsisFail(QuantTrace::Conv2dFail::BadQuant, 0, workspace_bytes);
        return false;
    }

    CmsisQuantPlan::Conv2DPlan plan{};
    plan.ready = true;
    plan.in_h = static_cast<int32_t>(in_h);
    plan.in_w = static_cast<int32_t>(in_w);
    plan.in_c = static_cast<int32_t>(in_c);
    plan.out_h = static_cast<int32_t>(out_h);
    plan.out_w = static_cast<int32_t>(out_w);
    plan.out_c = out_channels;
    plan.kernel_size = kernel_size;
    plan.stride = stride;
    plan.pad_h = pad_h;
    plan.pad_w = pad_w;
    plan.input_offset = -quant.input_zero_point;
    plan.output_offset = quant.output_zero_point;
    plan.output_scale = quant.output_scale;
    plan.clamp = apply_relu ? QuantInteger::QuantClamp::ReLU : QuantInteger::QuantClamp::None;
    plan.multipliers = multipliers;
    plan.shifts = shifts;
    (void)pad_h_end;
    (void)pad_w_end;

    return TryConv2dNhwcQuantPlan(plan, input, weights, bias, output);
}

bool TryDepthwiseConv2dNhwcQuant(const int8_t* input,
                                 uint32_t in_h,
                                 uint32_t in_w,
                                 uint32_t channels,
                                 const int8_t* weights,
                                 const int32_t* bias,
                                 int kernel_h,
                                 int kernel_w,
                                 int stride,
                                 int pad_h,
                                 int pad_w,
                                 int pad_h_end,
                                 int pad_w_end,
                                 const NkFormat::MlpLayerQuantDesc& quant,
                                 bool apply_relu,
                                 int8_t* output)
{
    if (!input || !weights || !bias || !output || channels == 0 || channels > 512)
        return false;
    if (pad_h != pad_h_end || pad_w != pad_w_end)
        return false;

    const uint32_t out_h =
        nk_op_detail::CalcOutputDimAsymmetric(in_h, kernel_h, stride, pad_h, pad_h_end);
    const uint32_t out_w =
        nk_op_detail::CalcOutputDimAsymmetric(in_w, kernel_w, stride, pad_w, pad_w_end);

    int32_t multipliers[512];
    int32_t shifts[512];
    if (!FillPerChannelQuant(quant, static_cast<int>(channels), multipliers, shifts))
        return false;

    CmsisQuantPlan::DepthwiseConv2DPlan plan{};
    plan.ready = true;
    plan.in_h = static_cast<int32_t>(in_h);
    plan.in_w = static_cast<int32_t>(in_w);
    plan.channels = static_cast<int32_t>(channels);
    plan.out_h = static_cast<int32_t>(out_h);
    plan.out_w = static_cast<int32_t>(out_w);
    plan.kernel_h = kernel_h;
    plan.kernel_w = kernel_w;
    plan.stride = stride;
    plan.pad_h = pad_h;
    plan.pad_w = pad_w;
    plan.input_offset = -quant.input_zero_point;
    plan.output_offset = quant.output_zero_point;
    plan.output_scale = quant.output_scale;
    plan.clamp = apply_relu ? QuantInteger::QuantClamp::ReLU : QuantInteger::QuantClamp::None;
    plan.multipliers = multipliers;
    plan.shifts = shifts;
    (void)pad_h_end;
    (void)pad_w_end;

    return TryDepthwiseConv2dNhwcQuantPlan(plan, input, weights, bias, output);
}

bool TryMaxPool2dNhwcQuant(const int8_t* input,
                           uint32_t in_h,
                           uint32_t in_w,
                           uint32_t in_c,
                           int pool_h,
                           int pool_w,
                           int stride,
                           int pad_h,
                           int pad_w,
                           int pad_h_end,
                           int pad_w_end,
                           int8_t* output)
{
    if (!input || !output || pad_h != pad_h_end || pad_w != pad_w_end)
        return false;

    const uint32_t out_h =
        nk_op_detail::CalcOutputDimAsymmetric(in_h, pool_h, stride, pad_h, pad_h_end);
    const uint32_t out_w =
        nk_op_detail::CalcOutputDimAsymmetric(in_w, pool_w, stride, pad_w, pad_w_end);

    CmsisQuantPlan::Pool2DPlan plan{};
    plan.ready = true;
    plan.in_h = static_cast<int32_t>(in_h);
    plan.in_w = static_cast<int32_t>(in_w);
    plan.in_c = static_cast<int32_t>(in_c);
    plan.out_h = static_cast<int32_t>(out_h);
    plan.out_w = static_cast<int32_t>(out_w);
    plan.pool_h = pool_h;
    plan.pool_w = pool_w;
    plan.stride = stride;
    plan.pad_h = pad_h;
    plan.pad_w = pad_w;
    plan.clamp = QuantInteger::QuantClamp::None;
    plan.output_scale = 1.0f;
    plan.output_zero_point = 0;
    (void)pad_h_end;
    (void)pad_w_end;

    return TryMaxPool2dNhwcQuantPlan(plan, input, output);
}

bool TryFullyConnectedQuant(const int8_t* input,
                            uint32_t batch,
                            uint32_t in_features,
                            const int8_t* weights,
                            const int32_t* bias,
                            uint32_t out_features,
                            const NkFormat::MlpLayerQuantDesc& quant,
                            bool apply_relu,
                            int8_t* output_int8)
{
    if (!input || !weights || !bias || !output_int8 || batch != 1 || out_features == 0 ||
        out_features > 512)
        return false;

    int32_t multipliers[512];
    int32_t shifts[512];
    if (!FillPerChannelQuant(quant, static_cast<int>(out_features), multipliers, shifts))
        return false;

    CmsisQuantPlan::FcPlan plan{};
    plan.ready = true;
    plan.in_features = static_cast<int32_t>(in_features);
    (void)batch;
    plan.out_features = static_cast<int32_t>(out_features);
    plan.input_offset = -quant.input_zero_point;
    plan.filter_offset = 0;
    plan.output_offset = quant.output_zero_point;
    plan.output_scale = quant.output_scale;
    plan.clamp = apply_relu ? QuantInteger::QuantClamp::ReLU : QuantInteger::QuantClamp::None;
    plan.multipliers = multipliers;
    plan.shifts = shifts;
    plan.weight_channel_scales = quant.weight_channel_scales;
    plan.num_weight_channel_scales = quant.num_weight_channel_scales;

    return TryFullyConnectedQuantPlan(plan, input, weights, bias, output_int8);
}

bool TrySoftmaxS8(const int8_t* input,
                  uint32_t num_rows,
                  uint32_t row_size,
                  float logit_scale,
                  int8_t* output)
{
    if (!input || !output || num_rows == 0 || row_size == 0 || logit_scale <= 0.0f)
        return false;

    // Match CMSIS/TFLM softmax prep: scale → mult/shift with diff_min.
    int32_t mult = 0;
    int32_t shift = 0;
    QuantizeMultiplier(static_cast<double>(logit_scale) * (1.0 / 256.0), &mult, &shift);
    const int32_t diff_min = -128;

    for (uint32_t r = 0; r < num_rows; ++r)
    {
        const int8_t* row_in = input + r * row_size;
        int8_t* row_out = output + r * row_size;
        const int32_t scratch = esp_nn_get_softmax_scratch_size(static_cast<int32_t>(row_size), 1);
        if (scratch > 0)
        {
            void* buf = nullptr;
            int32_t size = 0;
            if (!BindCmsisWorkspace(buf, size, scratch) || !buf)
                return false;
            esp_nn_set_softmax_scratch_buf(buf);
        }
        esp_nn_softmax_s8(row_in, 1, static_cast<int32_t>(row_size), mult, shift, diff_min,
                          row_out);
    }
    QuantTrace::RecordSoftmaxCmsisOk();
    return true;
}

bool TryElementwiseAddS8(const int8_t* input1,
                         const int8_t* input2,
                         uint32_t count,
                         float input1_scale,
                         int32_t input1_zero_point,
                         float input2_scale,
                         int32_t input2_zero_point,
                         float output_scale,
                         int32_t output_zero_point,
                         int8_t* output)
{
    if (!input1 || !input2 || !output || count == 0 || output_scale <= 0.0f)
        return false;

    // Match TFLite PrepareAdd / CMSIS-NN / ESP-NN TFLM prep.
    constexpr int32_t kLeftShift = 20;
    const double twice_max_input_scale =
        2.0 * std::max(static_cast<double>(input1_scale), static_cast<double>(input2_scale));
    if (twice_max_input_scale <= 0.0)
        return false;

    int32_t input1_mult = 0;
    int32_t input1_shift = 0;
    int32_t input2_mult = 0;
    int32_t input2_shift = 0;
    int32_t output_mult = 0;
    int32_t output_shift = 0;
    QuantizeMultiplier(static_cast<double>(input1_scale) / twice_max_input_scale, &input1_mult,
                       &input1_shift);
    QuantizeMultiplier(static_cast<double>(input2_scale) / twice_max_input_scale, &input2_mult,
                       &input2_shift);
    QuantizeMultiplier(
        twice_max_input_scale / ((1LL << kLeftShift) * static_cast<double>(output_scale)),
        &output_mult, &output_shift);

    esp_nn_add_elementwise_s8(input1,
                              input2,
                              -input1_zero_point,
                              -input2_zero_point,
                              input1_mult,
                              input2_mult,
                              input1_shift,
                              input2_shift,
                              kLeftShift,
                              output,
                              output_zero_point,
                              output_mult,
                              output_shift,
                              -128,
                              127,
                              static_cast<int32_t>(count));
    return true;
}

}  // namespace EspNnQuant

#else

#if NETKIT_REFERENCE_QUANT_LOOPS
#include "quant_ops.hpp"
#endif

namespace EspNnQuant
{

void FinalizeConv2DPlan(CmsisQuantPlan::Conv2DPlan& /*plan*/) {}
void FinalizeDepthwiseConv2DPlan(CmsisQuantPlan::DepthwiseConv2DPlan& /*plan*/) {}
void FinalizePool2DPlan(CmsisQuantPlan::Pool2DPlan& /*plan*/) {}
bool FinalizeFcPlan(CmsisQuantPlan::FcPlan& plan,
                    const int8_t* /*weights*/,
                    const int32_t* /*bias*/,
                    Arena& /*arena*/)
{
    return plan.ready;
}

#if NETKIT_REFERENCE_QUANT_LOOPS

namespace
{
NkFormat::MlpLayerQuantDesc QuantDescFromConv(const CmsisQuantPlan::Conv2DPlan& plan)
{
    NkFormat::MlpLayerQuantDesc q{};
    q.input_scale = plan.input_scale;
    q.input_zero_point = -plan.input_offset;
    q.weight_scale = plan.weight_scale;
    q.weight_zero_point = 0;
    q.output_scale = plan.output_scale;
    q.output_zero_point = plan.output_offset;
    q.weight_channel_scales = plan.weight_channel_scales;
    q.num_weight_channel_scales = plan.num_weight_channel_scales;
    return q;
}

NkFormat::MlpLayerQuantDesc QuantDescFromDw(const CmsisQuantPlan::DepthwiseConv2DPlan& plan)
{
    NkFormat::MlpLayerQuantDesc q{};
    q.input_scale = plan.input_scale;
    q.input_zero_point = -plan.input_offset;
    q.weight_scale = plan.weight_scale;
    q.weight_zero_point = 0;
    q.output_scale = plan.output_scale;
    q.output_zero_point = plan.output_offset;
    q.weight_channel_scales = plan.weight_channel_scales;
    q.num_weight_channel_scales = plan.num_weight_channel_scales;
    return q;
}

NkFormat::MlpLayerQuantDesc QuantDescFromFc(const CmsisQuantPlan::FcPlan& plan)
{
    NkFormat::MlpLayerQuantDesc q{};
    q.input_scale = plan.input_scale;
    q.input_zero_point = -plan.input_offset;
    q.weight_scale = plan.weight_scale;
    q.weight_zero_point = -plan.filter_offset;
    q.output_scale = plan.output_scale;
    q.output_zero_point = plan.output_offset;
    q.weight_channel_scales = plan.weight_channel_scales;
    q.num_weight_channel_scales = plan.num_weight_channel_scales;
    return q;
}
}  // namespace

bool TryConv2dNhwcQuantPlan(const CmsisQuantPlan::Conv2DPlan& plan,
                            const int8_t* input,
                            const int8_t* weights,
                            const int32_t* bias,
                            int8_t* output)
{
    if (!plan.ready || !input || !weights || !bias || !output || !plan.multipliers || !plan.shifts)
        return false;
    const NkFormat::MlpLayerQuantDesc quant = QuantDescFromConv(plan);
    // AOT plans often leave act_min/act_max at defaults; let QuantOps bake from clamp.
    QuantOps::Conv2dNhwcQuant(input,
                              static_cast<uint32_t>(plan.in_h),
                              static_cast<uint32_t>(plan.in_w),
                              static_cast<uint32_t>(plan.in_c),
                              weights,
                              bias,
                              plan.kernel_size,
                              plan.stride,
                              plan.pad_h,
                              plan.pad_w,
                              plan.pad_h,
                              plan.pad_w,
                              plan.out_c,
                              quant,
                              plan.clamp,
                              output,
                              nullptr,
                              plan.multipliers,
                              plan.shifts,
                              nullptr,
                              nullptr,
                              plan.bias_folded);
    QuantTrace::RecordConv2dReference();
    return true;
}

bool TryDepthwiseConv2dNhwcQuantPlan(const CmsisQuantPlan::DepthwiseConv2DPlan& plan,
                                     const int8_t* input,
                                     const int8_t* weights,
                                     const int32_t* bias,
                                     int8_t* output)
{
    if (!plan.ready || !input || !weights || !bias || !output || !plan.multipliers || !plan.shifts)
        return false;

    // Lowered AOT passes HWC when plan.weights_hwc is set; QuantOps expects CHW.
    const int8_t* weights_chw = weights;
    if (plan.weights_hwc != nullptr)
    {
        const int32_t n = plan.channels * plan.kernel_h * plan.kernel_w;
        void* buf = nullptr;
        int32_t size = 0;
        if (!BindCmsisWorkspace(buf, size, n) || !buf)
            return false;
        int8_t* packed = static_cast<int8_t*>(buf);
        for (int c = 0; c < plan.channels; ++c)
        {
            for (int y = 0; y < plan.kernel_h; ++y)
            {
                for (int x = 0; x < plan.kernel_w; ++x)
                {
                    const int hwc = (y * plan.kernel_w + x) * plan.channels + c;
                    const int chw = c * plan.kernel_h * plan.kernel_w + y * plan.kernel_w + x;
                    packed[chw] = weights[hwc];
                }
            }
        }
        weights_chw = packed;
    }

    const NkFormat::MlpLayerQuantDesc quant = QuantDescFromDw(plan);
    QuantOps::DepthwiseConv2dNhwcQuant(input,
                                       static_cast<uint32_t>(plan.in_h),
                                       static_cast<uint32_t>(plan.in_w),
                                       static_cast<uint32_t>(plan.channels),
                                       weights_chw,
                                       bias,
                                       plan.kernel_h,
                                       plan.kernel_w,
                                       plan.stride,
                                       plan.pad_h,
                                       plan.pad_w,
                                       plan.pad_h,
                                       plan.pad_w,
                                       quant,
                                       plan.clamp,
                                       output,
                                       plan.multipliers,
                                       plan.shifts,
                                       nullptr,
                                       nullptr,
                                       plan.bias_folded);
    QuantTrace::RecordConv2dReference();
    return true;
}

bool TryMaxPool2dNhwcQuantPlan(const CmsisQuantPlan::Pool2DPlan& plan,
                               const int8_t* input,
                               int8_t* output)
{
    if (!plan.ready || !input || !output)
        return false;
    QuantOps::MaxPool2dNhwcQuant(input,
                                 static_cast<uint32_t>(plan.in_h),
                                 static_cast<uint32_t>(plan.in_w),
                                 static_cast<uint32_t>(plan.in_c),
                                 plan.pool_h,
                                 plan.pool_w,
                                 plan.stride,
                                 plan.pad_h,
                                 plan.pad_w,
                                 plan.pad_h,
                                 plan.pad_w,
                                 output);
    return true;
}

bool TryFullyConnectedQuantPlan(const CmsisQuantPlan::FcPlan& plan,
                                const int8_t* input,
                                const int8_t* weights,
                                const int32_t* bias,
                                int8_t* output_int8)
{
    if (!plan.ready || !input || !weights || !bias || !output_int8 || !plan.multipliers ||
        !plan.shifts)
        return false;
    const NkFormat::MlpLayerQuantDesc quant = QuantDescFromFc(plan);

    // AOT per-tensor FC stores length-1 mult/shift; QuantOps indexes per out-channel.
    const bool per_channel =
        plan.weight_channel_scales != nullptr &&
        plan.num_weight_channel_scales == static_cast<uint32_t>(plan.out_features);
    const int32_t* multipliers = plan.multipliers;
    const int32_t* shifts = plan.shifts;
    int32_t* broadcast = nullptr;
    if (!per_channel && plan.out_features > 1)
    {
        const int32_t bytes = plan.out_features * static_cast<int32_t>(sizeof(int32_t)) * 2;
        void* buf = nullptr;
        int32_t size = 0;
        if (!BindCmsisWorkspace(buf, size, bytes) || !buf)
            return false;
        broadcast = static_cast<int32_t*>(buf);
        for (int32_t i = 0; i < plan.out_features; ++i)
        {
            broadcast[i] = plan.multipliers[0];
            broadcast[plan.out_features + i] = plan.shifts[0];
        }
        multipliers = broadcast;
        shifts = broadcast + plan.out_features;
    }

    QuantOps::FullyConnectedQuant(input,
                                  1u,
                                  static_cast<uint32_t>(plan.in_features),
                                  weights,
                                  bias,
                                  static_cast<uint32_t>(plan.out_features),
                                  quant,
                                  plan.clamp,
                                  output_int8,
                                  multipliers,
                                  shifts,
                                  nullptr,
                                  nullptr,
                                  plan.bias_folded);
    return true;
}

bool TrySoftmaxS8Plan(const CmsisQuantPlan::SoftmaxPlan& /*plan*/,
                      const int8_t* /*input*/,
                      int8_t* /*output*/)
{
    // SoftmaxPlan carries prepared mult/shift only; public QuantOps::SoftmaxS8 needs
    // logit_scale. Classification AOT uses --omit-final-softmax; keep stub miss.
    return false;
}

#else  // !NETKIT_REFERENCE_QUANT_LOOPS

bool TryConv2dNhwcQuantPlan(const CmsisQuantPlan::Conv2DPlan&,
                            const int8_t*,
                            const int8_t*,
                            const int32_t*,
                            int8_t*)
{
    return false;
}

bool TryDepthwiseConv2dNhwcQuantPlan(const CmsisQuantPlan::DepthwiseConv2DPlan&,
                                     const int8_t*,
                                     const int8_t*,
                                     const int32_t*,
                                     int8_t*)
{
    return false;
}

bool TryMaxPool2dNhwcQuantPlan(const CmsisQuantPlan::Pool2DPlan&, const int8_t*, int8_t*)
{
    return false;
}

bool TryFullyConnectedQuantPlan(const CmsisQuantPlan::FcPlan&,
                                const int8_t*,
                                const int8_t*,
                                const int32_t*,
                                int8_t*)
{
    return false;
}

bool TrySoftmaxS8Plan(const CmsisQuantPlan::SoftmaxPlan&, const int8_t*, int8_t*)
{
    return false;
}

#endif  // NETKIT_REFERENCE_QUANT_LOOPS

bool TryConv2dNhwcQuant(const int8_t*,
                        uint32_t,
                        uint32_t,
                        uint32_t,
                        const int8_t*,
                        const int32_t*,
                        int,
                        int,
                        int,
                        int,
                        int,
                        int,
                        int,
                        const NkFormat::MlpLayerQuantDesc&,
                        bool,
                        int8_t*)
{
    QuantTrace::RecordConv2dCmsisFail(QuantTrace::Conv2dFail::Disabled, 0, 0);
    return false;
}

bool TryDepthwiseConv2dNhwcQuant(const int8_t*,
                                 uint32_t,
                                 uint32_t,
                                 uint32_t,
                                 const int8_t*,
                                 const int32_t*,
                                 int,
                                 int,
                                 int,
                                 int,
                                 int,
                                 int,
                                 int,
                                 const NkFormat::MlpLayerQuantDesc&,
                                 bool,
                                 int8_t*)
{
    QuantTrace::RecordConv2dCmsisFail(QuantTrace::Conv2dFail::Disabled, 0, 0);
    return false;
}

bool TryMaxPool2dNhwcQuant(const int8_t*,
                           uint32_t,
                           uint32_t,
                           uint32_t,
                           int,
                           int,
                           int,
                           int,
                           int,
                           int,
                           int,
                           int8_t*)
{
    return false;
}

bool TryFullyConnectedQuant(const int8_t*,
                            uint32_t,
                            uint32_t,
                            const int8_t*,
                            const int32_t*,
                            uint32_t,
                            const NkFormat::MlpLayerQuantDesc&,
                            bool,
                            int8_t*)
{
    return false;
}

bool TrySoftmaxS8(const int8_t*, uint32_t, uint32_t, float, int8_t*)
{
    return false;
}

bool TryElementwiseAddS8(const int8_t*,
                         const int8_t*,
                         uint32_t,
                         float,
                         int32_t,
                         float,
                         int32_t,
                         float,
                         int32_t,
                         int8_t*)
{
    return false;
}

}  // namespace EspNnQuant

#endif
