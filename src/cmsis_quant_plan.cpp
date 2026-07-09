#include "cmsis_quant_plan.hpp"

#include "arena.hpp"
#include "cnn.hpp"
#include "cmsis_buffer_size.hpp"
#include "cmsis_nn_quant.hpp"
#include "kernel_workspace.hpp"
#include "netkit_config.h"
#include "nk_op_detail.hpp"
#include "quant_ops.hpp"
#include "quant_output.hpp"
#include "quant_trace.hpp"
#include "tensor_factory.hpp"
#include "xnnpack_quant.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace
{
    using namespace TensorFactory;

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

    uint32_t PlanOutputShape(CnnBlock& block, uint32_t& h, uint32_t& w, uint32_t& c)
    {
        switch (block.type)
        {
            case CnnBlockType::Conv2D:
            {
                const Conv2D& conv = block.conv.conv;
                h = nk_op_detail::CalcOutputDimAsymmetric(
                    h, conv.kernel_size, conv.stride, conv.pad_h, conv.pad_h_end);
                w = nk_op_detail::CalcOutputDimAsymmetric(
                    w, conv.kernel_size, conv.stride, conv.pad_w, conv.pad_w_end);
                c = static_cast<uint32_t>(conv.out_channels);
                break;
            }
            case CnnBlockType::DepthwiseConv2D:
            {
                const DepthwiseConv2D& dw = block.depthwise_conv.depthwise;
                h = nk_op_detail::CalcOutputDimAsymmetric(
                    h, dw.kernel_h, dw.stride, dw.pad_h, dw.pad_h_end);
                w = nk_op_detail::CalcOutputDimAsymmetric(
                    w, dw.kernel_w, dw.stride, dw.pad_w, dw.pad_w_end);
                break;
            }
            case CnnBlockType::MaxPool2D:
            {
                const MaxPool2DLayer& pool = block.pool;
                h = nk_op_detail::CalcOutputDimAsymmetric(
                    h, pool.pool_h, pool.stride, pool.pad_h, pool.pad_h_end);
                w = nk_op_detail::CalcOutputDimAsymmetric(
                    w, pool.pool_w, pool.stride, pool.pad_w, pool.pad_w_end);
                break;
            }
            case CnnBlockType::AvgPool2D:
            {
                const AvgPool2DLayer& pool = block.avg_pool;
                h = nk_op_detail::CalcOutputDimAsymmetric(
                    h, pool.pool_h, pool.stride, pool.pad_h, pool.pad_h_end);
                w = nk_op_detail::CalcOutputDimAsymmetric(
                    w, pool.pool_w, pool.stride, pool.pad_w, pool.pad_w_end);
                break;
            }
            case CnnBlockType::MobilenetV4Uib:
            {
                const MobileNetV4Uib& uib = block.mobilenetv4_uib.block;
                uib.output_spatial(h, w, h, w);
                c = static_cast<uint32_t>(uib.out_channels);
                break;
            }
            case CnnBlockType::Flatten:
            {
                const uint32_t features = h * w * c;
                h = 1;
                w = features;
                c = 1;
                break;
            }
            case CnnBlockType::Dense:
            {
                const uint32_t units = block.dense.weights.shape[0];
                h = 1;
                w = units;
                c = 1;
                break;
            }
            default:
                return 0;
        }

        return h * w * c;
    }


    QuantInteger::QuantClamp ClampFromConvActivation(ConvActivationType activation)
    {
        if (activation == ConvActivationType::ReLU)
            return QuantInteger::QuantClamp::ReLU;
        if (activation == ConvActivationType::ReLU6)
            return QuantInteger::QuantClamp::ReLU6;
        return QuantInteger::QuantClamp::None;
    }

    QuantInteger::QuantClamp ClampFromMlpActivation(ActivationType activation)
    {
        if (activation == ActivationType::ReLU)
            return QuantInteger::QuantClamp::ReLU;
        if (activation == ActivationType::ReLU6)
            return QuantInteger::QuantClamp::ReLU6;
        return QuantInteger::QuantClamp::None;
    }

    bool BuildConvPlan(CmsisQuantPlan::Conv2DPlan& plan,
                       const Conv2D& conv,
                       const NkFormat::MlpLayerQuantDesc& quant,
                       QuantInteger::QuantClamp clamp,
                       uint32_t in_h,
                       uint32_t in_w,
                       uint32_t in_c,
                       Arena& arena)
    {
        if (conv.pad_h != conv.pad_h_end || conv.pad_w != conv.pad_w_end)
            return false;
        if (conv.out_channels <= 0 ||
            static_cast<uint32_t>(conv.out_channels) > CmsisQuantPlan::kMaxPerChannel)
            return false;
        if (quant.output_scale <= 0.0f)
            return false;

        const uint32_t out_h = nk_op_detail::CalcOutputDimAsymmetric(
            in_h, conv.kernel_size, conv.stride, conv.pad_h, conv.pad_h_end);
        const uint32_t out_w = nk_op_detail::CalcOutputDimAsymmetric(
            in_w, conv.kernel_size, conv.stride, conv.pad_w, conv.pad_w_end);

        plan.input_offset = -quant.input_zero_point;
        plan.output_offset = -quant.output_zero_point;
        plan.stride = conv.stride;
        plan.pad_h = conv.pad_h;
        plan.pad_w = conv.pad_w;
        plan.clamp = clamp;
        plan.input_scale = quant.input_scale;
        plan.weight_scale = quant.weight_scale;
        plan.output_scale = quant.output_scale;
        plan.weight_channel_scales = quant.weight_channel_scales;
        plan.num_weight_channel_scales = quant.num_weight_channel_scales;
        plan.in_h = static_cast<int32_t>(in_h);
        plan.in_w = static_cast<int32_t>(in_w);
        plan.in_c = static_cast<int32_t>(in_c);
        plan.out_h = static_cast<int32_t>(out_h);
        plan.out_w = static_cast<int32_t>(out_w);
        plan.out_c = conv.out_channels;
        plan.kernel_size = conv.kernel_size;

        const int32_t channels = conv.out_channels;
        plan.multipliers = static_cast<int32_t*>(arena.alloc(
            static_cast<std::size_t>(channels) * sizeof(int32_t) * 2, alignof(int32_t)));
        if (!plan.multipliers)
            return false;
        plan.shifts = plan.multipliers + channels;

        const bool per_channel =
            quant.weight_channel_scales != nullptr &&
            quant.num_weight_channel_scales == static_cast<uint32_t>(channels);
        if (!per_channel)
        {
            const double effective = static_cast<double>(quant.input_scale) *
                                     static_cast<double>(quant.weight_scale) /
                                     static_cast<double>(quant.output_scale);
            int32_t multiplier = 0;
            int32_t shift = 0;
            QuantizeMultiplier(effective, &multiplier, &shift);
            for (int32_t oc = 0; oc < channels; ++oc)
            {
                plan.multipliers[oc] = multiplier;
                plan.shifts[oc] = shift;
            }
        }
        else
        {
            for (int32_t oc = 0; oc < channels; ++oc)
            {
                const double effective = static_cast<double>(quant.input_scale) *
                                         static_cast<double>(quant.weight_channel_scales[oc]) /
                                         static_cast<double>(quant.output_scale);
                QuantizeMultiplier(effective, &plan.multipliers[oc], &plan.shifts[oc]);
            }
        }

#if defined(NETKIT_USE_CMSIS_NN) && NETKIT_USE_CMSIS_NN && NETKIT_CMSIS_NN_ALLOWED
        plan.workspace_bytes = static_cast<int32_t>(CmsisConv2dS8WorkspaceBytes(
            static_cast<uint32_t>(in_h),
            static_cast<uint32_t>(in_w),
            static_cast<uint32_t>(conv.kernel_size),
            conv.stride,
            conv.pad_h,
            conv.pad_w,
            static_cast<uint32_t>(in_c),
            static_cast<uint32_t>(conv.out_channels)));
#else
        plan.workspace_bytes = 0;
#endif
        plan.ready = true;
        CmsisNnQuant::FinalizeConv2DPlan(plan);
        return true;
    }

    void RepackDepthwiseChwToHwc(const int8_t* chw,
                                 int8_t* hwc,
                                 int kernel_h,
                                 int kernel_w,
                                 int channels)
    {
        const int kh = kernel_h;
        const int kw = kernel_w;
        const int ch = channels;
        for (int c = 0; c < ch; ++c)
        {
            for (int y = 0; y < kh; ++y)
            {
                for (int x = 0; x < kw; ++x)
                {
                    hwc[(y * kw + x) * ch + c] =
                        chw[(c * kh + y) * kw + x];
                }
            }
        }
    }

    bool BuildDepthwisePlan(CmsisQuantPlan::DepthwiseConv2DPlan& plan,
                            const DepthwiseConv2D& dw,
                            const NkFormat::MlpLayerQuantDesc& quant,
                            QuantInteger::QuantClamp clamp,
                            uint32_t in_h,
                            uint32_t in_w,
                            uint32_t in_c,
                            const int8_t* weights_chw,
                            Arena& arena)
    {
        if (dw.pad_h != dw.pad_h_end || dw.pad_w != dw.pad_w_end)
            return false;
        if (dw.channels <= 0 || static_cast<uint32_t>(dw.channels) != in_c ||
            static_cast<uint32_t>(dw.channels) > CmsisQuantPlan::kMaxPerChannel)
            return false;
        if (quant.output_scale <= 0.0f)
            return false;

        const uint32_t out_h = nk_op_detail::CalcOutputDimAsymmetric(
            in_h, dw.kernel_h, dw.stride, dw.pad_h, dw.pad_h_end);
        const uint32_t out_w = nk_op_detail::CalcOutputDimAsymmetric(
            in_w, dw.kernel_w, dw.stride, dw.pad_w, dw.pad_w_end);

        plan.input_offset = -quant.input_zero_point;
        plan.output_offset = -quant.output_zero_point;
        plan.stride = dw.stride;
        plan.pad_h = dw.pad_h;
        plan.pad_w = dw.pad_w;
        plan.clamp = clamp;
        plan.input_scale = quant.input_scale;
        plan.weight_scale = quant.weight_scale;
        plan.output_scale = quant.output_scale;
        plan.weight_channel_scales = quant.weight_channel_scales;
        plan.num_weight_channel_scales = quant.num_weight_channel_scales;
        plan.in_h = static_cast<int32_t>(in_h);
        plan.in_w = static_cast<int32_t>(in_w);
        plan.channels = dw.channels;
        plan.out_h = static_cast<int32_t>(out_h);
        plan.out_w = static_cast<int32_t>(out_w);
        plan.kernel_h = dw.kernel_h;
        plan.kernel_w = dw.kernel_w;

        const int32_t channels = dw.channels;
        plan.multipliers = static_cast<int32_t*>(arena.alloc(
            static_cast<std::size_t>(channels) * sizeof(int32_t) * 2, alignof(int32_t)));
        if (!plan.multipliers)
            return false;
        plan.shifts = plan.multipliers + channels;

        const bool per_channel =
            quant.weight_channel_scales != nullptr &&
            quant.num_weight_channel_scales == static_cast<uint32_t>(channels);
        if (!per_channel)
        {
            const double effective = static_cast<double>(quant.input_scale) *
                                     static_cast<double>(quant.weight_scale) /
                                     static_cast<double>(quant.output_scale);
            int32_t multiplier = 0;
            int32_t shift = 0;
            QuantizeMultiplier(effective, &multiplier, &shift);
            for (int32_t c = 0; c < channels; ++c)
            {
                plan.multipliers[c] = multiplier;
                plan.shifts[c] = shift;
            }
        }
        else
        {
            for (int32_t c = 0; c < channels; ++c)
            {
                const double effective = static_cast<double>(quant.input_scale) *
                                         static_cast<double>(quant.weight_channel_scales[c]) /
                                         static_cast<double>(quant.output_scale);
                QuantizeMultiplier(effective, &plan.multipliers[c], &plan.shifts[c]);
            }
        }

#if defined(NETKIT_USE_CMSIS_NN) && NETKIT_USE_CMSIS_NN && NETKIT_CMSIS_NN_ALLOWED
        const std::size_t kernel_area =
            static_cast<std::size_t>(dw.kernel_h) * static_cast<std::size_t>(dw.kernel_w) *
            static_cast<std::size_t>(dw.channels);
        plan.weights_hwc = static_cast<int8_t*>(arena.alloc(kernel_area, alignof(int8_t)));
        if (!plan.weights_hwc || !weights_chw)
            return false;
        RepackDepthwiseChwToHwc(weights_chw, plan.weights_hwc, dw.kernel_h, dw.kernel_w, dw.channels);

        plan.workspace_bytes = static_cast<int32_t>(CmsisDepthwiseConv2dS8WorkspaceBytes(
            in_h,
            in_w,
            dw.kernel_h,
            dw.kernel_w,
            dw.stride,
            dw.pad_h,
            dw.pad_w,
            dw.channels));
#else
        plan.weights_hwc = nullptr;
        plan.workspace_bytes = 0;
#endif
        plan.ready = true;
        CmsisNnQuant::FinalizeDepthwiseConv2DPlan(plan);
        return true;
    }

    bool BuildPoolPlan(CmsisQuantPlan::Pool2DPlan& plan,
                       const MaxPool2DLayer& pool,
                       uint32_t in_h,
                       uint32_t in_w,
                       uint32_t in_c)
    {
        if (pool.pad_h != pool.pad_h_end || pool.pad_w != pool.pad_w_end)
            return false;

        plan.stride = pool.stride;
        plan.pad_h = pool.pad_h;
        plan.pad_w = pool.pad_w;
        plan.pool_h = pool.pool_h;
        plan.pool_w = pool.pool_w;
        plan.in_h = static_cast<int32_t>(in_h);
        plan.in_w = static_cast<int32_t>(in_w);
        plan.in_c = static_cast<int32_t>(in_c);
        plan.out_h = static_cast<int32_t>(nk_op_detail::CalcOutputDimAsymmetric(
            in_h, pool.pool_h, pool.stride, pool.pad_h, pool.pad_h_end));
        plan.out_w = static_cast<int32_t>(nk_op_detail::CalcOutputDimAsymmetric(
            in_w, pool.pool_w, pool.stride, pool.pad_w, pool.pad_w_end));
        plan.clamp = ClampFromConvActivation(pool.activation);
        plan.ready = true;
        CmsisNnQuant::FinalizePool2DPlan(plan);
        return true;
    }

    bool BuildAvgPoolPlan(CmsisQuantPlan::Pool2DPlan& plan,
                          const AvgPool2DLayer& pool,
                          uint32_t in_h,
                          uint32_t in_w,
                          uint32_t in_c,
                          float input_scale,
                          int32_t input_zero_point)
    {
        if (pool.pad_h != pool.pad_h_end || pool.pad_w != pool.pad_w_end)
            return false;

        plan.stride = pool.stride;
        plan.pad_h = pool.pad_h;
        plan.pad_w = pool.pad_w;
        plan.pool_h = pool.pool_h;
        plan.pool_w = pool.pool_w;
        plan.in_h = static_cast<int32_t>(in_h);
        plan.in_w = static_cast<int32_t>(in_w);
        plan.in_c = static_cast<int32_t>(in_c);
        plan.out_h = static_cast<int32_t>(nk_op_detail::CalcOutputDimAsymmetric(
            in_h, pool.pool_h, pool.stride, pool.pad_h, pool.pad_h_end));
        plan.out_w = static_cast<int32_t>(nk_op_detail::CalcOutputDimAsymmetric(
            in_w, pool.pool_w, pool.stride, pool.pad_w, pool.pad_w_end));
        plan.input_scale = input_scale;
        plan.input_zero_point = input_zero_point;
        plan.output_scale = input_scale;
        plan.output_zero_point = input_zero_point;
        plan.ready = true;
        return true;
    }

    bool BuildFcPlan(CmsisQuantPlan::FcPlan& plan,
                     const NkFormat::MlpLayerQuantDesc& quant,
                     QuantInteger::QuantClamp clamp,
                     uint32_t in_features,
                     uint32_t out_features,
                     Arena& arena)
    {
        if (in_features == 0 || out_features == 0 || quant.output_scale <= 0.0f)
            return false;

        plan.input_offset = -quant.input_zero_point;
        plan.filter_offset = -quant.weight_zero_point;
        plan.output_offset = -quant.output_zero_point;
        plan.clamp = clamp;
        plan.input_scale = quant.input_scale;
        plan.weight_scale = quant.weight_scale;
        plan.output_scale = quant.output_scale;
        plan.weight_channel_scales = quant.weight_channel_scales;
        plan.num_weight_channel_scales = quant.num_weight_channel_scales;
        plan.in_features = static_cast<int32_t>(in_features);
        plan.out_features = static_cast<int32_t>(out_features);

        const int32_t channels = static_cast<int32_t>(out_features);
        plan.multipliers = static_cast<int32_t*>(arena.alloc(
            static_cast<std::size_t>(channels) * sizeof(int32_t) * 2, alignof(int32_t)));
        if (!plan.multipliers)
            return false;
        plan.shifts = plan.multipliers + channels;

        const bool per_channel =
            quant.weight_channel_scales != nullptr &&
            quant.num_weight_channel_scales == static_cast<uint32_t>(channels);
        if (!per_channel)
        {
            const double effective = static_cast<double>(quant.input_scale) *
                                     static_cast<double>(quant.weight_scale) /
                                     static_cast<double>(quant.output_scale);
            QuantizeMultiplier(effective, &plan.multiplier, &plan.shift);
            for (int32_t oc = 0; oc < channels; ++oc)
            {
                plan.multipliers[oc] = plan.multiplier;
                plan.shifts[oc] = plan.shift;
            }
        }
        else
        {
            for (int32_t oc = 0; oc < channels; ++oc)
            {
                const double effective = static_cast<double>(quant.input_scale) *
                                         static_cast<double>(quant.weight_channel_scales[oc]) /
                                         static_cast<double>(quant.output_scale);
                QuantizeMultiplier(effective, &plan.multipliers[oc], &plan.shifts[oc]);
            }
            plan.multiplier = plan.multipliers[0];
            plan.shift = plan.shifts[0];
        }

#if defined(NETKIT_USE_CMSIS_NN) && NETKIT_USE_CMSIS_NN && NETKIT_CMSIS_NN_ALLOWED
        plan.workspace_bytes = static_cast<int32_t>(
            CmsisFullyConnectedS8WorkspaceBytes(in_features, out_features));
#else
        plan.workspace_bytes = 0;
#endif
        plan.ready = true;
        return true;
    }

    bool BuildSoftmaxPlan(CmsisQuantPlan::SoftmaxPlan& plan, float logit_scale, int32_t row_size)
    {
        plan.params = QuantOps::ComputeSoftmaxS8Params(logit_scale);
        plan.row_size = row_size;
        plan.ready = true;
        return true;
    }
}

namespace CmsisQuantPlan
{

bool BuildRuntime(CNNNetwork& network, Arena& arena, uint32_t in_h, uint32_t in_w, uint32_t in_c)
{
    const uint32_t n = network.layer_count();
    if (n == 0)
        return false;

    Runtime* runtime = static_cast<Runtime*>(arena.alloc(sizeof(Runtime), alignof(Runtime)));
    if (!runtime)
        return false;
    std::memset(runtime, 0, sizeof(Runtime));

    runtime->layers = static_cast<LayerPlan*>(arena.alloc(sizeof(LayerPlan) * n, alignof(LayerPlan)));
    if (!runtime->layers)
        return false;
    std::memset(runtime->layers, 0, sizeof(LayerPlan) * n);
    runtime->num_layers = n;

    uint32_t h = in_h;
    uint32_t w = in_w;
    uint32_t c = in_c;
    const uint32_t input_quant_elements = in_h * in_w * in_c;
    uint32_t even_max = 0;
    uint32_t odd_max = 0;
    std::size_t workspace_bytes = 0;
    uint32_t logits_elements = 0;
    float activation_scale = 1.0f;
    int32_t activation_zero_point = 0;

    const CnnBlock& first = network.GetBlock(0);
    if (first.type == CnnBlockType::Conv2D && first.conv.quant.enabled)
    {
        runtime->input_scale = first.conv.quant.params.input_scale;
        runtime->input_zero_point = first.conv.quant.params.input_zero_point;
        activation_scale = runtime->input_scale;
        activation_zero_point = runtime->input_zero_point;
    }
    else if (first.type == CnnBlockType::DepthwiseConv2D && first.depthwise_conv.quant.enabled)
    {
        runtime->input_scale = first.depthwise_conv.quant.params.input_scale;
        runtime->input_zero_point = first.depthwise_conv.quant.params.input_zero_point;
        activation_scale = runtime->input_scale;
        activation_zero_point = runtime->input_zero_point;
    }
    else if (first.type == CnnBlockType::MobilenetV4Uib &&
             first.mobilenetv4_uib.block.quant_enabled)
    {
        runtime->input_scale = first.mobilenetv4_uib.block.block_input_scale;
        runtime->input_zero_point = first.mobilenetv4_uib.block.block_input_zero_point;
        activation_scale = runtime->input_scale;
        activation_zero_point = runtime->input_zero_point;
    }

    for (uint32_t i = 0; i < n; ++i)
    {
        LayerPlan& lp = runtime->layers[i];
        CnnBlock& block = network.GetBlock(i);
        const uint32_t in_h_layer = h;
        const uint32_t in_w_layer = w;
        const uint32_t in_c_layer = c;
        const uint32_t elements = PlanOutputShape(block, h, w, c);
        if (elements == 0)
        {
            std::fprintf(stderr,
                         "BuildRuntime: zero output elements at layer %u type=%d h=%u w=%u c=%u\n",
                         i,
                         static_cast<int>(block.type),
                         h,
                         w,
                         c);
            return false;
        }

        lp.output_elements = elements;
        if (i % 2 == 0)
            even_max = std::max(even_max, elements);
        else
            odd_max = std::max(odd_max, elements);

        switch (block.type)
        {
            case CnnBlockType::Conv2D:
            {
                lp.kind = LayerKind::Conv2D;
                const QuantInteger::QuantClamp clamp =
                    ClampFromConvActivation(block.conv.activation);
                if (!BuildConvPlan(lp.conv,
                                   block.conv.conv,
                                   block.conv.quant.params,
                                   clamp,
                                   in_h_layer,
                                   in_w_layer,
                                   in_c_layer,
                                   arena))
                {
                    std::fprintf(stderr, "BuildRuntime: BuildConvPlan failed layer %u\n", i);
                    return false;
                }
                workspace_bytes = std::max(workspace_bytes,
                                           static_cast<std::size_t>(lp.conv.workspace_bytes));
                activation_scale = block.conv.quant.params.output_scale;
                activation_zero_point = block.conv.quant.params.output_zero_point;
                break;
            }
            case CnnBlockType::DepthwiseConv2D:
            {
                lp.kind = LayerKind::DepthwiseConv2D;
                const QuantInteger::QuantClamp clamp =
                    ClampFromConvActivation(block.depthwise_conv.activation);
                if (!BuildDepthwisePlan(lp.depthwise,
                                        block.depthwise_conv.depthwise,
                                        block.depthwise_conv.quant.params,
                                        clamp,
                                        in_h_layer,
                                        in_w_layer,
                                        in_c_layer,
                                        block.depthwise_conv.depthwise.weights_q,
                                        arena))
                {
                    std::fprintf(stderr, "BuildRuntime: BuildDepthwisePlan failed layer %u\n", i);
                    return false;
                }
                workspace_bytes = std::max(workspace_bytes,
                                           static_cast<std::size_t>(lp.depthwise.workspace_bytes));
                activation_scale = block.depthwise_conv.quant.params.output_scale;
                activation_zero_point = block.depthwise_conv.quant.params.output_zero_point;
                break;
            }
            case CnnBlockType::MaxPool2D:
            {
                lp.kind = LayerKind::MaxPool2D;
                if (!BuildPoolPlan(lp.pool, block.pool, in_h_layer, in_w_layer, in_c_layer))
                    return false;
                break;
            }
            case CnnBlockType::AvgPool2D:
            {
                lp.kind = LayerKind::AvgPool2D;
                if (!BuildAvgPoolPlan(lp.pool,
                                      block.avg_pool,
                                      in_h_layer,
                                      in_w_layer,
                                      in_c_layer,
                                      activation_scale,
                                      activation_zero_point))
                    return false;
                activation_scale = lp.pool.output_scale;
                activation_zero_point = lp.pool.output_zero_point;
                break;
            }
            case CnnBlockType::MobilenetV4Uib:
            {
                lp.kind = LayerKind::MobilenetV4Uib;
                MobileNetV4Uib& uib = block.mobilenetv4_uib.block;
                if (!uib.quant_enabled)
                    return false;
                MobilenetV4UibPlan& up = lp.uib;
                up.in_h = static_cast<int32_t>(in_h_layer);
                up.in_w = static_cast<int32_t>(in_w_layer);
                up.in_c = static_cast<int32_t>(in_c_layer);
                up.out_h = static_cast<int32_t>(h);
                up.out_w = static_cast<int32_t>(w);
                up.out_c = static_cast<int32_t>(c);
                up.has_start_dw = uib.start_dw_kernel > 0;
                up.has_middle_dw = uib.middle_dw_kernel > 0;
                up.has_residual = uib.has_residual();
                up.scratch = uib.scratch_i8;
                up.scratch_bytes = static_cast<int32_t>(uib.scratch_i8_bytes);
                up.ready = true;
                activation_scale = uib.proj_quant.output_scale;
                activation_zero_point = uib.proj_quant.output_zero_point;
                break;
            }
            case CnnBlockType::Flatten:
                lp.kind = LayerKind::FlattenView;
                break;
            case CnnBlockType::Dense:
            {
                const QuantInteger::QuantClamp clamp =
                    ClampFromMlpActivation(block.dense.activation);
                const bool apply_softmax = block.dense.activation == ActivationType::Softmax;
                const uint32_t in_features = in_h_layer * in_w_layer * in_c_layer;
                const uint32_t out_features = block.dense.weights.shape[0];
                if (apply_softmax)
                {
                    lp.kind = LayerKind::DenseSoftmax;
                    logits_elements = out_features;
                    if (!BuildFcPlan(lp.fc,
                                     block.dense.quant.params,
                                     QuantInteger::QuantClamp::None,
                                     in_features,
                                     out_features,
                                     arena))
                    {
                        std::fprintf(stderr, "BuildRuntime: BuildFcPlan(softmax) failed layer %u\n", i);
                        return false;
                    }
                    {
                        const int8_t* weights =
                            static_cast<const int8_t*>(block.dense.weights.data);
                        const int32_t* bias = static_cast<const int32_t*>(block.dense.bias.data);
                        if (!CmsisNnQuant::FinalizeFcPlan(lp.fc, weights, bias, arena))
                        {
                            std::fprintf(stderr, "BuildRuntime: FinalizeFcPlan(softmax) failed layer %u\n", i);
                            return false;
                        }
                    }
                    if (!BuildSoftmaxPlan(lp.softmax,
                                          block.dense.quant.params.output_scale > 0.0f
                                              ? block.dense.quant.params.output_scale
                                              : 1.0f,
                                          static_cast<int32_t>(out_features)))
                    {
                        std::fprintf(stderr, "BuildRuntime: BuildSoftmaxPlan failed layer %u\n", i);
                        return false;
                    }
                }
                else
                {
                    lp.kind = LayerKind::Dense;
                    if (!BuildFcPlan(lp.fc,
                                     block.dense.quant.params,
                                     clamp,
                                     in_features,
                                     out_features,
                                     arena))
                    {
                        std::fprintf(stderr, "BuildRuntime: BuildFcPlan failed layer %u\n", i);
                        return false;
                    }
                    {
                        const int8_t* weights =
                            static_cast<const int8_t*>(block.dense.weights.data);
                        const int32_t* bias = static_cast<const int32_t*>(block.dense.bias.data);
                        if (!CmsisNnQuant::FinalizeFcPlan(lp.fc, weights, bias, arena))
                        {
                            std::fprintf(stderr, "BuildRuntime: FinalizeFcPlan failed layer %u\n", i);
                            return false;
                        }
                    }
                    activation_scale = block.dense.quant.params.output_scale;
                    activation_zero_point = block.dense.quant.params.output_zero_point;
                }
                workspace_bytes = std::max(workspace_bytes,
                                           static_cast<std::size_t>(lp.fc.workspace_bytes));
                break;
            }
            default:
                std::fprintf(stderr, "BuildRuntime: unsupported layer type %d at %u\n",
                             static_cast<int>(block.type), i);
                return false;
        }
    }

    runtime->input_quant_elements = input_quant_elements;

    runtime->act_a_bytes = even_max;
    runtime->act_b_bytes = odd_max;
    if (runtime->act_a_bytes > 0)
    {
        runtime->act_a = static_cast<int8_t*>(
            arena.alloc(static_cast<std::size_t>(runtime->act_a_bytes) * sizeof(int8_t),
                        alignof(int8_t)));
        if (!runtime->act_a)
        {
            std::fprintf(stderr, "BuildRuntime: act_a alloc failed bytes=%u rem=%zu\n",
                         runtime->act_a_bytes, arena.remaining());
            return false;
        }
    }
    if (runtime->act_b_bytes > 0)
    {
        runtime->act_b = static_cast<int8_t*>(
            arena.alloc(static_cast<std::size_t>(runtime->act_b_bytes) * sizeof(int8_t),
                        alignof(int8_t)));
        if (!runtime->act_b)
        {
            std::fprintf(stderr, "BuildRuntime: act_b alloc failed bytes=%u rem=%zu\n",
                         runtime->act_b_bytes, arena.remaining());
            return false;
        }
    }

    if (logits_elements > 0)
    {
        runtime->logits_elements = logits_elements;
        runtime->logits = static_cast<int8_t*>(
            arena.alloc(static_cast<std::size_t>(logits_elements) * sizeof(int8_t), alignof(int8_t)));
        if (!runtime->logits)
            return false;
    }

    if (workspace_bytes > 0)
    {
        runtime->workspace = static_cast<uint8_t*>(
            arena.alloc(workspace_bytes, alignof(std::max_align_t)));
        if (!runtime->workspace)
            return false;
        runtime->workspace_bytes = workspace_bytes;
    }

    network.SetQuantRuntime(runtime);
    QuantTrace::RecordKernelPlan(static_cast<uint32_t>(workspace_bytes),
                                 static_cast<uint32_t>(runtime->act_a_bytes + runtime->act_b_bytes));
    return true;
}

void DestroyRuntime(Runtime& runtime)
{
    runtime = {};
}

namespace
{
    bool ForwardLayers(Runtime& runtime,
                       CNNNetwork& network,
                       int8_t* current,
                       QuantOutputFormat output_format,
                       Tensor& output_cache,
                       int8_t* output_dest)
    {
        if (!runtime.layers || runtime.num_layers == 0 || !current)
            return false;

        KernelWorkspace workspace{runtime.workspace, runtime.workspace_bytes};
        KernelWorkspaceScope workspace_scope(&workspace);

        int8_t* slot_a = runtime.act_a;
        int8_t* slot_b = runtime.act_b;
        bool use_a = true;

        for (uint32_t i = 0; i < runtime.num_layers; ++i)
        {
            const LayerPlan& lp = runtime.layers[i];
            const bool is_last = i + 1 == runtime.num_layers;
            int8_t* out = nullptr;

            if (lp.kind == LayerKind::FlattenView)
            {
                continue;
            }

            if (use_a)
                out = slot_a;
            else
                out = slot_b;

            if (!out && lp.kind != LayerKind::FlattenView)
                return false;

            CnnBlock& block = network.GetBlock(i);
            switch (lp.kind)
            {
                case LayerKind::Conv2D:
                {
                    const Conv2D& conv = block.conv.conv;
                    if (XnnpackQuant::TryConv2dNhwcQuantPlan(
                            lp.conv, current, conv.weights_q, conv.bias_q, out))
                    {
                        break;
                    }
                    if (CmsisNnQuant::TryConv2dNhwcQuantPlan(
                            lp.conv, current, conv.weights_q, conv.bias_q, out))
                    {
                        break;
                    }
#if defined(NETKIT_USE_CMSIS_NN) && NETKIT_USE_CMSIS_NN && NETKIT_CMSIS_NN_ALLOWED
                    return false;
#else
                    QuantOps::Conv2dNhwcQuant(current,
                                              static_cast<uint32_t>(lp.conv.in_h),
                                              static_cast<uint32_t>(lp.conv.in_w),
                                              static_cast<uint32_t>(lp.conv.in_c),
                                              conv.weights_q,
                                              conv.bias_q,
                                              conv.kernel_size,
                                              conv.stride,
                                              conv.pad_h,
                                              conv.pad_w,
                                              conv.pad_h_end,
                                              conv.pad_w_end,
                                              conv.out_channels,
                                              block.conv.quant.params,
                                              lp.conv.clamp,
                                              out);
                    break;
#endif
                }
                case LayerKind::DepthwiseConv2D:
                {
                    const DepthwiseConv2D& dw = block.depthwise_conv.depthwise;
                    const int8_t* weights = lp.depthwise.weights_hwc ? lp.depthwise.weights_hwc
                                                                     : dw.weights_q;
                    if (XnnpackQuant::TryDepthwiseConv2dNhwcQuantPlan(
                            lp.depthwise, current, dw.weights_q, dw.bias_q, out))
                    {
                        break;
                    }
                    if (CmsisNnQuant::TryDepthwiseConv2dNhwcQuantPlan(
                            lp.depthwise, current, weights, dw.bias_q, out))
                    {
                        break;
                    }
#if defined(NETKIT_USE_CMSIS_NN) && NETKIT_USE_CMSIS_NN && NETKIT_CMSIS_NN_ALLOWED
                    return false;
#else
                    QuantOps::DepthwiseConv2dNhwcQuant(current,
                                                       static_cast<uint32_t>(lp.depthwise.in_h),
                                                       static_cast<uint32_t>(lp.depthwise.in_w),
                                                       static_cast<uint32_t>(lp.depthwise.channels),
                                                       dw.weights_q,
                                                       dw.bias_q,
                                                       lp.depthwise.kernel_h,
                                                       lp.depthwise.kernel_w,
                                                       lp.depthwise.stride,
                                                       lp.depthwise.pad_h,
                                                       lp.depthwise.pad_w,
                                                       dw.pad_h_end,
                                                       dw.pad_w_end,
                                                       block.depthwise_conv.quant.params,
                                                       lp.depthwise.clamp,
                                                       out);
                    break;
#endif
                }
                case LayerKind::MaxPool2D:
                {
                    if (XnnpackQuant::TryMaxPool2dNhwcQuantPlan(lp.pool, current, out))
                        break;
                    if (CmsisNnQuant::TryMaxPool2dNhwcQuantPlan(lp.pool, current, out))
                        break;
#if defined(NETKIT_USE_CMSIS_NN) && NETKIT_USE_CMSIS_NN && NETKIT_CMSIS_NN_ALLOWED
                    return false;
#else
                    const MaxPool2DLayer& pool = block.pool;
                    QuantOps::MaxPool2dNhwcQuant(current,
                                                   static_cast<uint32_t>(lp.pool.in_h),
                                                   static_cast<uint32_t>(lp.pool.in_w),
                                                   static_cast<uint32_t>(lp.pool.in_c),
                                                   pool.pool_h,
                                                   pool.pool_w,
                                                   pool.stride,
                                                   pool.pad_h,
                                                   pool.pad_w,
                                                   pool.pad_h_end,
                                                   pool.pad_w_end,
                                                   out);
                    break;
#endif
                }
                case LayerKind::AvgPool2D:
                {
                    const AvgPool2DLayer& pool = block.avg_pool;
                    QuantOps::AvgPool2dNhwcQuant(current,
                                                  static_cast<uint32_t>(lp.pool.in_h),
                                                  static_cast<uint32_t>(lp.pool.in_w),
                                                  static_cast<uint32_t>(lp.pool.in_c),
                                                  pool.pool_h,
                                                  pool.pool_w,
                                                  pool.stride,
                                                  pool.pad_h,
                                                  pool.pad_w,
                                                  pool.pad_h_end,
                                                  pool.pad_w_end,
                                                  lp.pool.input_scale,
                                                  lp.pool.input_zero_point,
                                                  lp.pool.output_scale,
                                                  lp.pool.output_zero_point,
                                                  out);
                    break;
                }
                case LayerKind::MobilenetV4Uib:
                {
                    MobileNetV4Uib& uib = block.mobilenetv4_uib.block;
                    uib.forward_quant(current,
                                      out,
                                      static_cast<uint32_t>(lp.uib.in_h),
                                      static_cast<uint32_t>(lp.uib.in_w));
                    break;
                }
                case LayerKind::Dense:
                {
                    const int8_t* weights = static_cast<const int8_t*>(block.dense.weights.data);
                    const int32_t* bias = static_cast<const int32_t*>(block.dense.bias.data);
                    int8_t* dense_out = (is_last && output_dest != nullptr) ? output_dest : out;
                    if (XnnpackQuant::TryFullyConnectedQuantPlan(
                            lp.fc, current, weights, bias, dense_out) ||
                        CmsisNnQuant::TryFullyConnectedQuantPlan(
                            lp.fc, current, weights, bias, dense_out))
                    {
                        if (dense_out == output_dest)
                            out = output_dest;
                        break;
                    }
#if defined(NETKIT_USE_CMSIS_NN) && NETKIT_USE_CMSIS_NN && NETKIT_CMSIS_NN_ALLOWED
                    return false;
#else
                    QuantOps::FullyConnectedQuant(current,
                                                  1u,
                                                  static_cast<uint32_t>(lp.fc.in_features),
                                                  weights,
                                                  bias,
                                                  static_cast<uint32_t>(lp.fc.out_features),
                                                  block.dense.quant.params,
                                                  lp.fc.clamp,
                                                  dense_out);
                    if (dense_out == output_dest)
                        out = output_dest;
                    break;
#endif
                }
                case LayerKind::DenseSoftmax:
                {
                    const int8_t* weights = static_cast<const int8_t*>(block.dense.weights.data);
                    const int32_t* bias = static_cast<const int32_t*>(block.dense.bias.data);
                    int8_t* softmax_out = (output_dest != nullptr) ? output_dest : out;
                    // Classification path: write logits directly (argmax-equivalent to Softmax).
                    int8_t* fc_dest =
                        runtime.omit_final_softmax ? softmax_out : runtime.logits;
                    if (!XnnpackQuant::TryFullyConnectedQuantPlan(
                            lp.fc, current, weights, bias, fc_dest) &&
                        !CmsisNnQuant::TryFullyConnectedQuantPlan(
                            lp.fc, current, weights, bias, fc_dest))
                    {
#if defined(NETKIT_USE_CMSIS_NN) && NETKIT_USE_CMSIS_NN && NETKIT_CMSIS_NN_ALLOWED
                        return false;
#else
                        QuantOps::FullyConnectedQuant(current,
                                                      1u,
                                                      static_cast<uint32_t>(lp.fc.in_features),
                                                      weights,
                                                      bias,
                                                      static_cast<uint32_t>(lp.fc.out_features),
                                                      block.dense.quant.params,
                                                      lp.fc.clamp,
                                                      fc_dest);
#endif
                    }
                    if (runtime.omit_final_softmax)
                    {
                        out = softmax_out;
                        (void)output_format;
                        break;
                    }
                    if (!CmsisNnQuant::TrySoftmaxS8Plan(lp.softmax, runtime.logits, softmax_out))
                    {
#if defined(NETKIT_USE_CMSIS_NN) && NETKIT_USE_CMSIS_NN && NETKIT_CMSIS_NN_ALLOWED
                        return false;
#else
                        QuantOps::SoftmaxS8(runtime.logits,
                                            static_cast<uint32_t>(lp.softmax.row_size),
                                            block.dense.quant.params.output_scale,
                                            softmax_out);
#endif
                    }
                    out = softmax_out;
                    (void)output_format;
                    break;
                }
                default:
                    break;
            }

            if (lp.kind != LayerKind::FlattenView)
            {
                current = out;
                if (out != output_dest)
                    use_a = !use_a;
            }
        }

        if (output_dest != nullptr)
            return true;

        (void)output_format;
        const LayerPlan& last = runtime.layers[runtime.num_layers - 1];
        output_cache = View2DInt8(current, 1, last.output_elements);
        return true;
    }
}  // namespace

bool Forward(Runtime& runtime,
             CNNNetwork& network,
             const Tensor& input,
             QuantOutputFormat output_format,
             Tensor& output_cache)
{
    // Int8 I/O only — float↔int8 belongs in Python (export / offline), not C++.
    if (input.type != DataType::Int8)
        return false;

    return ForwardInt8(runtime,
                       network,
                       static_cast<const int8_t*>(input.data),
                       input.num_elements,
                       output_format,
                       output_cache);
}

bool ForwardInt8(Runtime& runtime,
                 CNNNetwork& network,
                 const int8_t* input,
                 uint32_t input_elements,
                 QuantOutputFormat output_format,
                 Tensor& output_cache)
{
    if (!input || input_elements != runtime.input_quant_elements)
        return false;

    return ForwardLayers(runtime,
                       network,
                       const_cast<int8_t*>(input),
                       output_format,
                       output_cache,
                       nullptr);
}

bool ForwardInt8ToBuffer(Runtime& runtime,
                         CNNNetwork& network,
                         const int8_t* input,
                         int8_t* output,
                         uint32_t output_elements)
{
    if (!input || !output || runtime.num_layers == 0 || !runtime.layers)
        return false;
    if (runtime.input_quant_elements == 0 ||
        output_elements != runtime.layers[runtime.num_layers - 1].output_elements)
        return false;

    Tensor unused_cache{};
    return ForwardLayers(runtime,
                       network,
                       const_cast<int8_t*>(input),
                       QuantOutputFormat::Int8,
                       unused_cache,
                       output);
}

}  // namespace CmsisQuantPlan
