#include "cmsis_quant_plan.hpp"

#include "arena.hpp"
#include "cnn.hpp"
#include "cmsis_buffer_size.hpp"
#include "cmsis_nn_quant.hpp"
#include "esp_nn_quant.hpp"
#include "nmsis_nn_quant.hpp"
#include "im2col_quant.hpp"
#include "kernel_workspace.hpp"
#include "netkit_config.h"
#include "nk_op_detail.hpp"
#include "quant_ops.hpp"
#include "quant_output.hpp"
#include "quant_trace.hpp"
#include "tensor_factory.hpp"
#include "xnnpack_quant.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <new>

#if NETKIT_XNNPACK_PLAN_HOIST
#include <xnnpack.h>
#endif

// Optional per-layer stage timing for latency investigation.
// Default OFF (production). Enable only for debug builds:
//   -DNETKIT_STAGE_TIMING=1
// On MCU also set -DNETKIT_STAGE_TIMING_UART=1 (board Makefile does this when
// NETKIT_STAGE_TIMING=1) so summaries go over UART and timers use DWT.
// Leave unset/0 in production — zero cost when disabled.
#ifndef NETKIT_STAGE_TIMING
#define NETKIT_STAGE_TIMING 0
#endif

#if NETKIT_STAGE_TIMING && defined(NETKIT_STAGE_TIMING_UART) && NETKIT_STAGE_TIMING_UART
#include "dwt_time.h"
#include "uart.h"
#endif

namespace
{
    using namespace TensorFactory;

#if NETKIT_STAGE_TIMING
    struct StageTiming
    {
        // Pre-UIB / MNIST-style Conv2Ds: first two split out; further stem convs in stem_extra.
        double conv1_us = 0;
        double conv2_us = 0;
        double stem_extra_us = 0;
        // Time inside arm_convolve_wrapper_s8 only (excludes BindContext / plan glue).
        double cmsis_conv_kernel_us = 0;
        double maxpool_us = 0;
        double uib_start_dw_us = 0;
        double uib_expand_us = 0;
        double uib_middle_dw_us = 0;
        double uib_proj_us = 0;
        double uib_residual_memcpy_us = 0;
        double uib_residual_add_us = 0;
        double uib_fallback_us = 0;
        double avgpool_us = 0;
        double head_conv_us = 0;
        double dense_us = 0;
        double other_us = 0;
        uint64_t invokes = 0;
        uint64_t uib_plan_ok = 0;
        uint64_t uib_fallback = 0;
    };

    StageTiming g_stage_timing{};

    struct ScopedStageTimer
    {
        double* bucket;
#if defined(NETKIT_STAGE_TIMING_UART) && NETKIT_STAGE_TIMING_UART
        uint32_t t0_cycles;
        explicit ScopedStageTimer(double* b) : bucket(b), t0_cycles(dwt_cycles()) {}
        ~ScopedStageTimer()
        {
            *bucket += dwt_cycles_to_us(dwt_cycles() - t0_cycles);
        }
#else
        std::chrono::steady_clock::time_point t0;
        explicit ScopedStageTimer(double* b)
            : bucket(b), t0(std::chrono::steady_clock::now())
        {
        }
        ~ScopedStageTimer()
        {
            *bucket += std::chrono::duration<double, std::micro>(
                           std::chrono::steady_clock::now() - t0)
                           .count();
        }
#endif
    };

    double StageElapsedUs(
#if defined(NETKIT_STAGE_TIMING_UART) && NETKIT_STAGE_TIMING_UART
        uint32_t t0_cycles
#else
        std::chrono::steady_clock::time_point t0
#endif
    )
    {
#if defined(NETKIT_STAGE_TIMING_UART) && NETKIT_STAGE_TIMING_UART
        return dwt_cycles_to_us(dwt_cycles() - t0_cycles);
#else
        return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0)
            .count();
#endif
    }

    void FormatStageTimingLine(char* buf, std::size_t capacity)
    {
        const double total = g_stage_timing.conv1_us + g_stage_timing.conv2_us +
                             g_stage_timing.stem_extra_us + g_stage_timing.maxpool_us +
                             g_stage_timing.uib_start_dw_us + g_stage_timing.uib_expand_us +
                             g_stage_timing.uib_middle_dw_us + g_stage_timing.uib_proj_us +
                             g_stage_timing.uib_residual_memcpy_us +
                             g_stage_timing.uib_residual_add_us + g_stage_timing.uib_fallback_us +
                             g_stage_timing.avgpool_us + g_stage_timing.head_conv_us +
                             g_stage_timing.dense_us + g_stage_timing.other_us;
        const double n = static_cast<double>(std::max<uint64_t>(g_stage_timing.invokes, 1));
        auto avg = [n](double v) { return v / n; };
        // Two lines so MCU uart_printf-sized buffers are not required; uart_write is used.
        std::snprintf(
            buf,
            capacity,
            "STAGE_TIMING invokes=%llu plan_uib=%llu fallback_uib=%llu "
            "avg_us: conv1=%.1f conv2=%.1f stem_extra=%.1f cmsis_conv=%.1f "
            "conv_glue=%.1f maxpool=%.1f "
            "uib_sdw=%.1f uib_exp=%.1f uib_mdw=%.1f uib_proj=%.1f "
            "res_memcpy=%.1f res_add=%.1f uib_fb=%.1f avgpool=%.1f head=%.1f dense=%.1f "
            "other=%.1f total=%.1f",
            static_cast<unsigned long long>(g_stage_timing.invokes),
            static_cast<unsigned long long>(g_stage_timing.uib_plan_ok),
            static_cast<unsigned long long>(g_stage_timing.uib_fallback),
            avg(g_stage_timing.conv1_us),
            avg(g_stage_timing.conv2_us),
            avg(g_stage_timing.stem_extra_us),
            avg(g_stage_timing.cmsis_conv_kernel_us),
            avg(g_stage_timing.conv1_us + g_stage_timing.conv2_us + g_stage_timing.stem_extra_us +
                g_stage_timing.head_conv_us - g_stage_timing.cmsis_conv_kernel_us),
            avg(g_stage_timing.maxpool_us),
            avg(g_stage_timing.uib_start_dw_us),
            avg(g_stage_timing.uib_expand_us),
            avg(g_stage_timing.uib_middle_dw_us),
            avg(g_stage_timing.uib_proj_us),
            avg(g_stage_timing.uib_residual_memcpy_us),
            avg(g_stage_timing.uib_residual_add_us),
            avg(g_stage_timing.uib_fallback_us),
            avg(g_stage_timing.avgpool_us),
            avg(g_stage_timing.head_conv_us),
            avg(g_stage_timing.dense_us),
            avg(g_stage_timing.other_us),
            avg(total));
    }

    void PrintStageTimingIfDue()
    {
        // Host only: periodic stderr. MCU prints once via PrintStageTimingSummary
        // after the timed bench so UART I/O does not inflate invoke latency.
#if !(defined(NETKIT_STAGE_TIMING_UART) && NETKIT_STAGE_TIMING_UART)
        if (g_stage_timing.invokes > 3 && (g_stage_timing.invokes % 10) != 0)
            return;
        char line[512];
        FormatStageTimingLine(line, sizeof(line));
        std::fprintf(stderr, "%s\n", line);
#endif
    }
#endif

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
        // CMSIS-NN / TFLM: output_offset is the output zero-point (added after requant).
        // input_offset is -input_zp. Do not negate output_zp (CMSIS header wording is misleading).
        plan.output_offset = quant.output_zero_point;
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

        QuantInteger::QuantClampRange(
            clamp, quant.output_scale, quant.output_zero_point, &plan.act_min, &plan.act_max);

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
        // QuantOps im2col scratch only when CMSIS-NN is off (otherwise CMSIS owns
        // the workspace and a full-im2col bump can OOM a 64 KiB MCU arena).
        {
            const std::size_t im2col_bytes = Conv2dQuantIm2ColWorkspaceBytes(
                out_h,
                out_w,
                static_cast<uint32_t>(conv.kernel_size),
                static_cast<uint32_t>(conv.kernel_size),
                static_cast<uint32_t>(in_c),
                conv.stride);
            if (im2col_bytes > static_cast<std::size_t>(plan.workspace_bytes))
                plan.workspace_bytes = static_cast<int32_t>(im2col_bytes);
        }
#endif
        // Prepare-time input_offset fold for scalar ref (weight_zp must be 0).
        plan.bias_folded = nullptr;
        if (plan.input_offset != 0 && quant.weight_zero_point == 0 && conv.weights_q &&
            conv.bias_q)
        {
            const uint32_t filter_elems =
                static_cast<uint32_t>(conv.kernel_size) * static_cast<uint32_t>(conv.kernel_size) *
                in_c;
            plan.bias_folded = static_cast<int32_t*>(
                arena.alloc(static_cast<std::size_t>(channels) * sizeof(int32_t), alignof(int32_t)));
            if (!plan.bias_folded)
                return false;
            for (int32_t oc = 0; oc < channels; ++oc)
            {
                const int8_t* filter =
                    conv.weights_q + static_cast<std::size_t>(oc) * filter_elems;
                int32_t sum = 0;
                for (uint32_t i = 0; i < filter_elems; ++i)
                    sum += static_cast<int32_t>(filter[i]);
                plan.bias_folded[oc] = conv.bias_q[oc] + plan.input_offset * sum;
            }
        }
        plan.ready = true;
        CmsisNnQuant::FinalizeConv2DPlan(plan);
        NmsisNnQuant::FinalizeConv2DPlan(plan);
        EspNnQuant::FinalizeConv2DPlan(plan);
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
        plan.output_offset = quant.output_zero_point;
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

        QuantInteger::QuantClampRange(
            clamp, quant.output_scale, quant.output_zero_point, &plan.act_min, &plan.act_max);

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
        plan.workspace_bytes = 0;
#endif
        // Always repack CHW → HWC for XNNPACK (operator DW flag + subgraph [1,Kh,Kw,C]).
        {
            const std::size_t kernel_area =
                static_cast<std::size_t>(dw.kernel_h) * static_cast<std::size_t>(dw.kernel_w) *
                static_cast<std::size_t>(dw.channels);
            plan.weights_hwc = static_cast<int8_t*>(arena.alloc(kernel_area, alignof(int8_t)));
            if (!plan.weights_hwc || !weights_chw)
                return false;
            RepackDepthwiseChwToHwc(
                weights_chw, plan.weights_hwc, dw.kernel_h, dw.kernel_w, dw.channels);
        }
        plan.bias_folded = nullptr;
        if (plan.input_offset != 0 && quant.weight_zero_point == 0 && weights_chw && dw.bias_q)
        {
            const uint32_t kernel_area =
                static_cast<uint32_t>(dw.kernel_h) * static_cast<uint32_t>(dw.kernel_w);
            plan.bias_folded = static_cast<int32_t*>(
                arena.alloc(static_cast<std::size_t>(channels) * sizeof(int32_t), alignof(int32_t)));
            if (!plan.bias_folded)
                return false;
            for (int32_t c = 0; c < channels; ++c)
            {
                const int8_t* filter =
                    weights_chw + static_cast<std::size_t>(c) * kernel_area;
                int32_t sum = 0;
                for (uint32_t i = 0; i < kernel_area; ++i)
                    sum += static_cast<int32_t>(filter[i]);
                plan.bias_folded[c] = dw.bias_q[c] + plan.input_offset * sum;
            }
        }
        plan.ready = true;
        CmsisNnQuant::FinalizeDepthwiseConv2DPlan(plan);
        NmsisNnQuant::FinalizeDepthwiseConv2DPlan(plan);
        EspNnQuant::FinalizeDepthwiseConv2DPlan(plan);
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
        NmsisNnQuant::FinalizePool2DPlan(plan);
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
                     Arena& arena,
                     const int8_t* weights = nullptr,
                     const int32_t* bias = nullptr)
    {
        if (in_features == 0 || out_features == 0 || quant.output_scale <= 0.0f)
            return false;

        plan.input_offset = -quant.input_zero_point;
        plan.filter_offset = -quant.weight_zero_point;
        plan.output_offset = quant.output_zero_point;
        plan.clamp = clamp;
        plan.input_scale = quant.input_scale;
        plan.weight_scale = quant.weight_scale;
        plan.output_scale = quant.output_scale;
        plan.weight_channel_scales = quant.weight_channel_scales;
        plan.num_weight_channel_scales = quant.num_weight_channel_scales;
        plan.in_features = static_cast<int32_t>(in_features);
        plan.out_features = static_cast<int32_t>(out_features);

        QuantInteger::QuantClampRange(
            clamp, quant.output_scale, quant.output_zero_point, &plan.act_min, &plan.act_max);

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
        plan.bias_folded = nullptr;
        if (plan.input_offset != 0 && plan.filter_offset == 0 && weights && bias)
        {
            plan.bias_folded = static_cast<int32_t*>(
                arena.alloc(static_cast<std::size_t>(channels) * sizeof(int32_t), alignof(int32_t)));
            if (!plan.bias_folded)
                return false;
            for (int32_t oc = 0; oc < channels; ++oc)
            {
                const int8_t* row =
                    weights + static_cast<std::size_t>(oc) * in_features;
                int32_t sum = 0;
                for (uint32_t i = 0; i < in_features; ++i)
                    sum += static_cast<int32_t>(row[i]);
                plan.bias_folded[oc] = bias[oc] + plan.input_offset * sum;
            }
        }
        plan.ready = true;
        return true;
    }

    bool BuildElementwiseAddPlan(CmsisQuantPlan::ElementwiseAddPlan& plan,
                                 float input1_scale,
                                 int32_t input1_zero_point,
                                 float input2_scale,
                                 int32_t input2_zero_point,
                                 float output_scale,
                                 int32_t output_zero_point,
                                 int32_t block_size)
    {
        constexpr int32_t kLeftShift = 20;
        const double twice_max_input_scale =
            2.0 * std::max(static_cast<double>(input1_scale), static_cast<double>(input2_scale));
        if (twice_max_input_scale <= 0.0 || output_scale <= 0.0f)
            return false;

        plan.input1_offset = -input1_zero_point;
        plan.input2_offset = -input2_zero_point;
        plan.output_offset = output_zero_point;
        plan.left_shift = kLeftShift;
        plan.block_size = block_size;
        plan.act_min = -128;
        plan.act_max = 127;
        QuantizeMultiplier(static_cast<double>(input1_scale) / twice_max_input_scale,
                           &plan.input1_mult,
                           &plan.input1_shift);
        QuantizeMultiplier(static_cast<double>(input2_scale) / twice_max_input_scale,
                           &plan.input2_mult,
                           &plan.input2_shift);
        QuantizeMultiplier(
            twice_max_input_scale / ((1LL << kLeftShift) * static_cast<double>(output_scale)),
            &plan.output_mult,
            &plan.output_shift);
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
    new (runtime) Runtime{};

    runtime->layers = static_cast<LayerPlan*>(arena.alloc(sizeof(LayerPlan) * n, alignof(LayerPlan)));
    if (!runtime->layers)
        return false;
    for (uint32_t i = 0; i < n; ++i)
        new (runtime->layers + i) LayerPlan{};
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
                // MaxPool preserves activation quant params (TF Lite / XNNPACK qs8).
                lp.pool.input_scale = activation_scale;
                lp.pool.input_zero_point = activation_zero_point;
                lp.pool.output_scale = activation_scale;
                lp.pool.output_zero_point = activation_zero_point;
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

                uint32_t cur_h = in_h_layer;
                uint32_t cur_w = in_w_layer;
                uint32_t cur_c = in_c_layer;
                const uint32_t expand_c = uib.expanded_channels();

                if (up.has_start_dw)
                {
                    const int pad = (uib.start_dw_kernel - 1) / 2;
                    DepthwiseConv2D dw{};
                    dw.kernel_h = uib.start_dw_kernel;
                    dw.kernel_w = uib.start_dw_kernel;
                    dw.stride = static_cast<int>(uib.start_dw_stride());
                    dw.pad_h = pad;
                    dw.pad_w = pad;
                    dw.pad_h_end = pad;
                    dw.pad_w_end = pad;
                    dw.channels = static_cast<int>(cur_c);
                    dw.weights_q = uib.start_dw_weights_q;
                    dw.bias_q = uib.start_dw_bias_q;
                    if (!BuildDepthwisePlan(up.start_dw,
                                            dw,
                                            uib.start_dw_quant,
                                            QuantInteger::QuantClamp::None,
                                            cur_h,
                                            cur_w,
                                            cur_c,
                                            uib.start_dw_weights_q,
                                            arena))
                    {
                        std::fprintf(stderr,
                                     "BuildRuntime: UIB start_dw plan failed layer %u\n",
                                     i);
                        return false;
                    }
                    cur_h = static_cast<uint32_t>(up.start_dw.out_h);
                    cur_w = static_cast<uint32_t>(up.start_dw.out_w);
                }

                {
                    Conv2D expand{};
                    expand.kernel_size = 1;
                    expand.stride = 1;
                    expand.pad_h = 0;
                    expand.pad_w = 0;
                    expand.pad_h_end = 0;
                    expand.pad_w_end = 0;
                    expand.in_channels = static_cast<int>(cur_c);
                    expand.out_channels = static_cast<int>(expand_c);
                    expand.weights_q = uib.expand_weights_q;
                    expand.bias_q = uib.expand_bias_q;
                    if (!BuildConvPlan(up.expand,
                                       expand,
                                       uib.expand_quant,
                                       QuantInteger::QuantClamp::ReLU,
                                       cur_h,
                                       cur_w,
                                       cur_c,
                                       arena))
                    {
                        std::fprintf(stderr,
                                     "BuildRuntime: UIB expand plan failed layer %u\n",
                                     i);
                        return false;
                    }
                    cur_c = expand_c;
                }

                if (up.has_middle_dw)
                {
                    const int pad = (uib.middle_dw_kernel - 1) / 2;
                    DepthwiseConv2D dw{};
                    dw.kernel_h = uib.middle_dw_kernel;
                    dw.kernel_w = uib.middle_dw_kernel;
                    dw.stride = static_cast<int>(uib.middle_dw_stride());
                    dw.pad_h = pad;
                    dw.pad_w = pad;
                    dw.pad_h_end = pad;
                    dw.pad_w_end = pad;
                    dw.channels = static_cast<int>(cur_c);
                    dw.weights_q = uib.middle_dw_weights_q;
                    dw.bias_q = uib.middle_dw_bias_q;
                    if (!BuildDepthwisePlan(up.middle_dw,
                                            dw,
                                            uib.middle_dw_quant,
                                            QuantInteger::QuantClamp::ReLU,
                                            cur_h,
                                            cur_w,
                                            cur_c,
                                            uib.middle_dw_weights_q,
                                            arena))
                    {
                        std::fprintf(stderr,
                                     "BuildRuntime: UIB middle_dw plan failed layer %u\n",
                                     i);
                        return false;
                    }
                    cur_h = static_cast<uint32_t>(up.middle_dw.out_h);
                    cur_w = static_cast<uint32_t>(up.middle_dw.out_w);
                }

                {
                    Conv2D proj{};
                    proj.kernel_size = 1;
                    proj.stride = 1;
                    proj.pad_h = 0;
                    proj.pad_w = 0;
                    proj.pad_h_end = 0;
                    proj.pad_w_end = 0;
                    proj.in_channels = static_cast<int>(cur_c);
                    proj.out_channels = uib.out_channels;
                    proj.weights_q = uib.proj_weights_q;
                    proj.bias_q = uib.proj_bias_q;
                    if (!BuildConvPlan(up.proj,
                                       proj,
                                       uib.proj_quant,
                                       QuantInteger::QuantClamp::None,
                                       cur_h,
                                       cur_w,
                                       cur_c,
                                       arena))
                    {
                        std::fprintf(stderr,
                                     "BuildRuntime: UIB proj plan failed layer %u\n",
                                     i);
                        return false;
                    }
                }

                if (up.has_residual)
                {
                    const int32_t block_size = static_cast<int32_t>(up.out_h) *
                                              static_cast<int32_t>(up.out_w) *
                                              static_cast<int32_t>(up.out_c);
                    if (!BuildElementwiseAddPlan(up.add,
                                                 uib.proj_quant.output_scale,
                                                 uib.proj_quant.output_zero_point,
                                                 uib.block_input_scale,
                                                 uib.block_input_zero_point,
                                                 uib.proj_quant.output_scale,
                                                 uib.proj_quant.output_zero_point,
                                                 block_size))
                    {
                        std::fprintf(stderr,
                                     "BuildRuntime: UIB residual add plan failed layer %u\n",
                                     i);
                        return false;
                    }
                }

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
                                     arena,
                                     static_cast<const int8_t*>(block.dense.weights.data),
                                     static_cast<const int32_t*>(block.dense.bias.data)))
                    {
                        std::fprintf(stderr, "BuildRuntime: BuildFcPlan(softmax) failed layer %u\n", i);
                        return false;
                    }
                    {
                        const int8_t* weights =
                            static_cast<const int8_t*>(block.dense.weights.data);
                        const int32_t* bias = static_cast<const int32_t*>(block.dense.bias.data);
                        (void)NmsisNnQuant::FinalizeFcPlan(lp.fc, weights, bias, arena);
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
                                     arena,
                                     static_cast<const int8_t*>(block.dense.weights.data),
                                     static_cast<const int32_t*>(block.dense.bias.data)))
                    {
                        std::fprintf(stderr, "BuildRuntime: BuildFcPlan failed layer %u\n", i);
                        return false;
                    }
                    {
                        const int8_t* weights =
                            static_cast<const int8_t*>(block.dense.weights.data);
                        const int32_t* bias = static_cast<const int32_t*>(block.dense.bias.data);
                        (void)NmsisNnQuant::FinalizeFcPlan(lp.fc, weights, bias, arena);
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
        // ESP-NN S3 asm wants ≥16-byte alignment for scratch.
        constexpr std::size_t kWsAlign = 16;
        runtime->workspace = static_cast<uint8_t*>(arena.alloc(workspace_bytes, kWsAlign));
        if (!runtime->workspace)
        {
#if defined(NETKIT_USE_CMSIS_NN) && NETKIT_USE_CMSIS_NN && NETKIT_CMSIS_NN_ALLOWED
            return false;
#else
            // QuantOps im2col: take remaining arena so partial patches can still run.
            const std::size_t rem = arena.remaining();
            if (rem > 0)
            {
                runtime->workspace = static_cast<uint8_t*>(arena.alloc(rem, kWsAlign));
                runtime->workspace_bytes = runtime->workspace ? rem : 0;
            }
#endif
        }
        else
        {
            runtime->workspace_bytes = workspace_bytes;
        }
    }

#if NETKIT_XNNPACK_PLAN_HOIST
    // Create persistent XNNPACK ops after activation/workspace arena allocs so
    // qs8 workspaces use remaining arena capacity (else heap via Create*).
    // Shared weights cache packs kernels once and reuses across operators.
    // Shared xnn_workspace matches TF Lite XNNPACK delegate (one workspace per
    // delegate / runtime group).
    if (xnn_initialize(/*allocator=*/nullptr) == xnn_status_success)
    {
        // Prefer a large initial capacity so packing for ImageNet-scale models
        // does not grow/relocate the cache during create (helps cold BuildRuntime).
        // XNNPACK's public API is in-memory only; TF Lite's file-backed mmap
        // weight cache lives in the LiteRT delegate, not in libXNNPACK.
        constexpr size_t kWeightsCacheBytes = 16u * 1024u * 1024u;
        xnn_weights_cache_t cache = nullptr;
        if (xnn_create_weights_cache_with_size(kWeightsCacheBytes, &cache) ==
                xnn_status_success ||
            xnn_create_weights_cache(&cache) == xnn_status_success)
            runtime->xnn_weights_cache = cache;
        xnn_workspace_t ws = nullptr;
        if (xnn_create_workspace(&ws) == xnn_status_success)
            runtime->xnn_workspace = ws;
    }
    void* weights_cache = runtime->xnn_weights_cache;
    void* xnn_ws = runtime->xnn_workspace;

    // Prefer a single full-network qs8 subgraph (MobileNetV4 ImageNet, etc.).
    const bool network_created =
        XnnpackQuant::CreateNetworkSubgraph(*runtime, network, weights_cache);

    if (!runtime->xnn_network_runtime)
    {
        for (uint32_t i = 0; i < n; ++i)
        {
            LayerPlan& lp = runtime->layers[i];
            CnnBlock& block = network.GetBlock(i);
            switch (lp.kind)
            {
                case LayerKind::Conv2D:
                    (void)XnnpackQuant::CreateConv2dNhwcQuantPlan(
                        lp.conv, block.conv.conv.weights_q, block.conv.conv.bias_q, &arena,
                        weights_cache);
                    break;
                case LayerKind::DepthwiseConv2D:
                    (void)XnnpackQuant::CreateDepthwiseConv2dNhwcQuantPlan(
                        lp.depthwise,
                        block.depthwise_conv.depthwise.weights_q,
                        block.depthwise_conv.depthwise.bias_q,
                        &arena,
                        weights_cache);
                    break;
                case LayerKind::MaxPool2D:
                    (void)XnnpackQuant::CreateMaxPool2dNhwcQuantPlan(lp.pool, &arena);
                    break;
                case LayerKind::Dense:
                case LayerKind::DenseSoftmax:
                {
                    const int8_t* weights =
                        static_cast<const int8_t*>(block.dense.weights.data);
                    const int32_t* bias = static_cast<const int32_t*>(block.dense.bias.data);
                    (void)XnnpackQuant::CreateFullyConnectedQuantPlan(
                        lp.fc, weights, bias, &arena, weights_cache);
                    break;
                }
                case LayerKind::MobilenetV4Uib:
                {
                    MobileNetV4Uib& uib = block.mobilenetv4_uib.block;
                    MobilenetV4UibPlan& up = lp.uib;
                    // Prefer fused UIB subgraph; fall back to per-op hoists on failure.
                    if (XnnpackQuant::CreateUibSubgraph(up, uib, weights_cache, xnn_ws))
                        break;
                    if (up.has_start_dw)
                        (void)XnnpackQuant::CreateDepthwiseConv2dNhwcQuantPlan(
                            up.start_dw, uib.start_dw_weights_q, uib.start_dw_bias_q, &arena,
                            weights_cache);
                    (void)XnnpackQuant::CreateConv2dNhwcQuantPlan(
                        up.expand, uib.expand_weights_q, uib.expand_bias_q, &arena,
                        weights_cache);
                    if (up.has_middle_dw)
                        (void)XnnpackQuant::CreateDepthwiseConv2dNhwcQuantPlan(
                            up.middle_dw, uib.middle_dw_weights_q, uib.middle_dw_bias_q, &arena,
                            weights_cache);
                    (void)XnnpackQuant::CreateConv2dNhwcQuantPlan(
                        up.proj, uib.proj_weights_q, uib.proj_bias_q, &arena, weights_cache);
                    break;
                }
                default:
                    break;
            }
        }
    }

    if (runtime->xnn_weights_cache)
    {
        auto* cache = static_cast<xnn_weights_cache_t>(runtime->xnn_weights_cache);
        // Hard finalize: trim to page boundary and mark read-only. Soft leaves
        // spare capacity for late inserts; we never insert after BuildRuntime.
        if (xnn_finalize_weights_cache(cache, xnn_weights_cache_finalization_kind_hard) !=
            xnn_status_success)
        {
            (void)xnn_finalize_weights_cache(cache, xnn_weights_cache_finalization_kind_soft);
        }
    }

    if (runtime->xnn_network_runtime)
    {
        XnnpackQuant::FinishNetworkAfterWeightsCache(*runtime);
        if (runtime->xnn_network_ready)
        {
            std::fprintf(stderr, "BuildRuntime: network xnn_subgraph ready\n");
        }
        else
        {
            std::fprintf(stderr,
                         "BuildRuntime: network xnn_subgraph failed, using per-layer path\n");
            XnnpackQuant::DestroyNetworkSubgraph(*runtime);
            // Fall through to per-layer create below (cache already finalized — create
            // without cache / with nullptr so reshape is immediate).
            for (uint32_t i = 0; i < n; ++i)
            {
                LayerPlan& lp = runtime->layers[i];
                CnnBlock& block = network.GetBlock(i);
                switch (lp.kind)
                {
                    case LayerKind::Conv2D:
                        (void)XnnpackQuant::CreateConv2dNhwcQuantPlan(
                            lp.conv,
                            block.conv.conv.weights_q,
                            block.conv.conv.bias_q,
                            &arena,
                            /*weights_cache=*/nullptr);
                        break;
                    case LayerKind::DepthwiseConv2D:
                        (void)XnnpackQuant::CreateDepthwiseConv2dNhwcQuantPlan(
                            lp.depthwise,
                            block.depthwise_conv.depthwise.weights_q,
                            block.depthwise_conv.depthwise.bias_q,
                            &arena,
                            /*weights_cache=*/nullptr);
                        break;
                    case LayerKind::MaxPool2D:
                        (void)XnnpackQuant::CreateMaxPool2dNhwcQuantPlan(lp.pool, &arena);
                        break;
                    case LayerKind::Dense:
                    case LayerKind::DenseSoftmax:
                    {
                        const int8_t* weights =
                            static_cast<const int8_t*>(block.dense.weights.data);
                        const int32_t* bias =
                            static_cast<const int32_t*>(block.dense.bias.data);
                        (void)XnnpackQuant::CreateFullyConnectedQuantPlan(
                            lp.fc, weights, bias, &arena, /*weights_cache=*/nullptr);
                        break;
                    }
                    case LayerKind::MobilenetV4Uib:
                    {
                        MobileNetV4Uib& uib = block.mobilenetv4_uib.block;
                        MobilenetV4UibPlan& up = lp.uib;
                        if (XnnpackQuant::CreateUibSubgraph(up, uib, /*weights_cache=*/nullptr,
                                                            xnn_ws))
                            break;
                        if (up.has_start_dw)
                            (void)XnnpackQuant::CreateDepthwiseConv2dNhwcQuantPlan(
                                up.start_dw,
                                uib.start_dw_weights_q,
                                uib.start_dw_bias_q,
                                &arena,
                                /*weights_cache=*/nullptr);
                        (void)XnnpackQuant::CreateConv2dNhwcQuantPlan(
                            up.expand,
                            uib.expand_weights_q,
                            uib.expand_bias_q,
                            &arena,
                            /*weights_cache=*/nullptr);
                        if (up.has_middle_dw)
                            (void)XnnpackQuant::CreateDepthwiseConv2dNhwcQuantPlan(
                                up.middle_dw,
                                uib.middle_dw_weights_q,
                                uib.middle_dw_bias_q,
                                &arena,
                                /*weights_cache=*/nullptr);
                        (void)XnnpackQuant::CreateConv2dNhwcQuantPlan(
                            up.proj,
                            uib.proj_weights_q,
                            uib.proj_bias_q,
                            &arena,
                            /*weights_cache=*/nullptr);
                        break;
                    }
                    default:
                        break;
                }
            }
        }
    }
    else
    {
        (void)network_created;
        std::fprintf(stderr,
                     "BuildRuntime: network xnn_subgraph failed, using per-layer path\n");

        // Reshape after finalize so packed-weight absolute pointers are stable.
        uint32_t uib_subgraph_ok = 0;
        uint32_t uib_total = 0;
        for (uint32_t i = 0; i < n; ++i)
        {
            LayerPlan& lp = runtime->layers[i];
            switch (lp.kind)
            {
                case LayerKind::Conv2D:
                    (void)XnnpackQuant::FinishConvAfterWeightsCache(lp.conv.xnn, &arena);
                    break;
                case LayerKind::DepthwiseConv2D:
                    (void)XnnpackQuant::FinishConvAfterWeightsCache(lp.depthwise.xnn, &arena);
                    break;
                case LayerKind::Dense:
                case LayerKind::DenseSoftmax:
                    (void)XnnpackQuant::FinishFullyConnectedAfterWeightsCache(lp.fc.xnn);
                    break;
                case LayerKind::MobilenetV4Uib:
                {
                    ++uib_total;
                    MobilenetV4UibPlan& up = lp.uib;
                    CnnBlock& block = network.GetBlock(i);
                    MobileNetV4Uib& uib = block.mobilenetv4_uib.block;
                    if (up.xnn_runtime)
                    {
                        if (XnnpackQuant::FinishUibAfterWeightsCache(up))
                        {
                            ++uib_subgraph_ok;
                            break;
                        }
                    }
                    if (!up.xnn_subgraph_ready)
                    {
                        if (up.has_start_dw)
                            (void)XnnpackQuant::CreateDepthwiseConv2dNhwcQuantPlan(
                                up.start_dw,
                                uib.start_dw_weights_q,
                                uib.start_dw_bias_q,
                                &arena,
                                /*weights_cache=*/nullptr);
                        (void)XnnpackQuant::CreateConv2dNhwcQuantPlan(
                            up.expand,
                            uib.expand_weights_q,
                            uib.expand_bias_q,
                            &arena,
                            /*weights_cache=*/nullptr);
                        if (up.has_middle_dw)
                            (void)XnnpackQuant::CreateDepthwiseConv2dNhwcQuantPlan(
                                up.middle_dw,
                                uib.middle_dw_weights_q,
                                uib.middle_dw_bias_q,
                                &arena,
                                /*weights_cache=*/nullptr);
                        (void)XnnpackQuant::CreateConv2dNhwcQuantPlan(
                            up.proj,
                            uib.proj_weights_q,
                            uib.proj_bias_q,
                            &arena,
                            /*weights_cache=*/nullptr);
                    }
                    break;
                }
                default:
                    break;
            }
        }
        if (uib_total > 0)
        {
            std::fprintf(stderr,
                         "BuildRuntime: UIB xnn_subgraph ready %u/%u\n",
                         uib_subgraph_ok,
                         uib_total);
        }
    }
#endif

    network.SetQuantRuntime(runtime);
    QuantTrace::RecordKernelPlan(static_cast<uint32_t>(workspace_bytes),
                                 static_cast<uint32_t>(runtime->act_a_bytes + runtime->act_b_bytes));
    return true;
}

void DestroyRuntime(Runtime& runtime)
{
#if NETKIT_XNNPACK_PLAN_HOIST
    XnnpackQuant::DestroyNetworkSubgraph(runtime);
    if (runtime.layers)
    {
        for (uint32_t i = 0; i < runtime.num_layers; ++i)
        {
            LayerPlan& lp = runtime.layers[i];
            switch (lp.kind)
            {
                case LayerKind::Conv2D:
                    XnnpackQuant::DestroyXnnpackOp(lp.conv.xnn);
                    break;
                case LayerKind::DepthwiseConv2D:
                    XnnpackQuant::DestroyXnnpackOp(lp.depthwise.xnn);
                    break;
                case LayerKind::MaxPool2D:
                    XnnpackQuant::DestroyXnnpackOp(lp.pool.xnn);
                    break;
                case LayerKind::Dense:
                case LayerKind::DenseSoftmax:
                    XnnpackQuant::DestroyXnnpackOp(lp.fc.xnn);
                    break;
                case LayerKind::MobilenetV4Uib:
                    XnnpackQuant::DestroyUibSubgraph(lp.uib);
                    XnnpackQuant::DestroyXnnpackOp(lp.uib.start_dw.xnn);
                    XnnpackQuant::DestroyXnnpackOp(lp.uib.expand.xnn);
                    XnnpackQuant::DestroyXnnpackOp(lp.uib.middle_dw.xnn);
                    XnnpackQuant::DestroyXnnpackOp(lp.uib.proj.xnn);
                    break;
                default:
                    break;
            }
        }
    }
    // Ops must be deleted before the shared weights cache / workspace.
    if (runtime.xnn_weights_cache)
    {
        (void)xnn_delete_weights_cache(static_cast<xnn_weights_cache_t>(runtime.xnn_weights_cache));
        runtime.xnn_weights_cache = nullptr;
    }
    if (runtime.xnn_workspace)
    {
        (void)xnn_release_workspace(static_cast<xnn_workspace_t>(runtime.xnn_workspace));
        runtime.xnn_workspace = nullptr;
    }
#endif
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

#if NETKIT_XNNPACK_PLAN_HOIST
        if (runtime.xnn_network_ready)
        {
            const LayerPlan& last = runtime.layers[runtime.num_layers - 1];
            int8_t* dest = output_dest;
            if (!dest)
            {
                if (runtime.act_a && runtime.act_a_bytes >= last.output_elements)
                    dest = runtime.act_a;
                else if (runtime.act_b && runtime.act_b_bytes >= last.output_elements)
                    dest = runtime.act_b;
                else if (runtime.logits && runtime.logits_elements >= last.output_elements)
                    dest = runtime.logits;
                else
                    return false;
            }
            if (!XnnpackQuant::InvokeNetworkSubgraph(runtime, current, dest))
                return false;
            if (output_dest != nullptr)
                return true;
            (void)output_format;
            output_cache = View2DInt8(dest, 1, last.output_elements);
            return true;
        }
#endif

        KernelWorkspace workspace{runtime.workspace, runtime.workspace_bytes};
        KernelWorkspaceScope workspace_scope(&workspace);

        int8_t* slot_a = runtime.act_a;
        int8_t* slot_b = runtime.act_b;
        bool use_a = true;
#if NETKIT_STAGE_TIMING
        bool seen_uib = false;
        uint32_t stem_conv_idx = 0;
#endif

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
#if NETKIT_STAGE_TIMING
            const auto layer_t0 =
#if defined(NETKIT_STAGE_TIMING_UART) && NETKIT_STAGE_TIMING_UART
                dwt_cycles();
#else
                std::chrono::steady_clock::now();
#endif
            double* layer_bucket = &g_stage_timing.other_us;
            if (lp.kind == LayerKind::Conv2D)
            {
                if (seen_uib)
                {
                    layer_bucket = &g_stage_timing.head_conv_us;
                }
                else if (stem_conv_idx == 0)
                {
                    layer_bucket = &g_stage_timing.conv1_us;
                    ++stem_conv_idx;
                }
                else if (stem_conv_idx == 1)
                {
                    layer_bucket = &g_stage_timing.conv2_us;
                    ++stem_conv_idx;
                }
                else
                {
                    layer_bucket = &g_stage_timing.stem_extra_us;
                    ++stem_conv_idx;
                }
            }
            else if (lp.kind == LayerKind::MaxPool2D)
                layer_bucket = &g_stage_timing.maxpool_us;
            else if (lp.kind == LayerKind::AvgPool2D)
                layer_bucket = &g_stage_timing.avgpool_us;
            else if (lp.kind == LayerKind::Dense || lp.kind == LayerKind::DenseSoftmax)
                layer_bucket = &g_stage_timing.dense_us;
            // UIB timed in sub-op buckets below; layer_bucket unused for UIB.
#endif
            switch (lp.kind)
            {
                case LayerKind::Conv2D:
                {
                    const Conv2D& conv = block.conv.conv;
                    if constexpr (XnnpackQuant::kEnabled)
                    {
                        if (XnnpackQuant::TryConv2dNhwcQuantPlan(
                                lp.conv, current, conv.weights_q, conv.bias_q, out))
                            break;
                    }
                    if constexpr (EspNnQuant::kEnabled)
                    {
                        if (EspNnQuant::TryConv2dNhwcQuantPlan(
                                lp.conv, current, conv.weights_q, conv.bias_q, out))
                            break;
                    }

                    if constexpr (NmsisNnQuant::kEnabled)
                    {
                        if (NmsisNnQuant::TryConv2dNhwcQuantPlan(
                                lp.conv, current, conv.weights_q, conv.bias_q, out))
                            break;
                    }
                    if constexpr (CmsisNnQuant::kEnabled)
                    {
                        if (CmsisNnQuant::TryConv2dNhwcQuantPlan(
                                lp.conv, current, conv.weights_q, conv.bias_q, out))
                            break;
                    }
#if NETKIT_MCU_CMSIS_ONLY
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
                                              out,
                                              nullptr,
                                              lp.conv.multipliers,
                                              lp.conv.shifts,
                                              &lp.conv.act_min,
                                              &lp.conv.act_max,
                                              lp.conv.bias_folded);
                    break;
#endif
                }
                case LayerKind::DepthwiseConv2D:
                {
                    const DepthwiseConv2D& dw = block.depthwise_conv.depthwise;
                    const int8_t* weights = lp.depthwise.weights_hwc ? lp.depthwise.weights_hwc
                                                                     : dw.weights_q;
                    if constexpr (XnnpackQuant::kEnabled)
                    {
                        if (XnnpackQuant::TryDepthwiseConv2dNhwcQuantPlan(
                                lp.depthwise, current, dw.weights_q, dw.bias_q, out))
                            break;
                    }
                    if constexpr (EspNnQuant::kEnabled)
                    {
                        if (EspNnQuant::TryDepthwiseConv2dNhwcQuantPlan(
                                lp.depthwise, current, weights, dw.bias_q, out))
                            break;
                    }

                    if constexpr (NmsisNnQuant::kEnabled)
                    {
                        if (NmsisNnQuant::TryDepthwiseConv2dNhwcQuantPlan(
                                lp.depthwise, current, weights, dw.bias_q, out))
                            break;
                    }
                    if constexpr (CmsisNnQuant::kEnabled)
                    {
                        if (CmsisNnQuant::TryDepthwiseConv2dNhwcQuantPlan(
                                lp.depthwise, current, weights, dw.bias_q, out))
                            break;
                    }
#if NETKIT_MCU_CMSIS_ONLY
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
                                                       out,
                                                       lp.depthwise.multipliers,
                                                       lp.depthwise.shifts,
                                                       &lp.depthwise.act_min,
                                                       &lp.depthwise.act_max,
                                                       lp.depthwise.bias_folded);
                    break;
#endif
                }
                case LayerKind::MaxPool2D:
                {
                    if constexpr (XnnpackQuant::kEnabled)
                    {
                        if (XnnpackQuant::TryMaxPool2dNhwcQuantPlan(lp.pool, current, out))
                            break;
                    }
                    if constexpr (EspNnQuant::kEnabled)
                    {
                        if (EspNnQuant::TryMaxPool2dNhwcQuantPlan(lp.pool, current, out))
                            break;
                    }

                    if constexpr (NmsisNnQuant::kEnabled)
                    {
                        if (NmsisNnQuant::TryMaxPool2dNhwcQuantPlan(lp.pool, current, out))
                            break;
                    }
                    if constexpr (CmsisNnQuant::kEnabled)
                    {
                        if (CmsisNnQuant::TryMaxPool2dNhwcQuantPlan(lp.pool, current, out))
                            break;
                    }
#if NETKIT_MCU_CMSIS_ONLY
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
                    MobilenetV4UibPlan& up = runtime.layers[i].uib;
                    bool ran_plans = false;

#if NETKIT_XNNPACK_PLAN_HOIST
                    // Fused UIB subgraph only exists when XNNPACK plan hoist is on (CI forces
                    // NETKIT_XNNPACK=0 and compiles reference / Try* fallbacks instead).
                    if (up.xnn_subgraph_ready)
                    {
                        // Residual aliases `current` when the subgraph omits the add —
                        // UIB body does not overwrite the block input buffer.
                        if (XnnpackQuant::kEnabled &&
                            XnnpackQuant::InvokeUibSubgraph(up, current, out))
                        {
                            if (up.has_residual && !up.xnn_subgraph_includes_residual)
                            {
                                const uint32_t count = static_cast<uint32_t>(up.out_h) *
                                                       static_cast<uint32_t>(up.out_w) *
                                                       static_cast<uint32_t>(up.out_c);
                                QuantOps::ElementwiseAddS8(out,
                                                           current,
                                                           count,
                                                           up.add,
                                                           out);
                            }
                            ran_plans = true;
                        }
                    }
#endif

                    if (!ran_plans && up.ready && up.scratch)
                    {
                        const uint32_t in_h = static_cast<uint32_t>(up.in_h);
                        const uint32_t in_w = static_cast<uint32_t>(up.in_w);
                        const uint32_t in_c = static_cast<uint32_t>(up.in_c);
                        const uint32_t expand_c = uib.expanded_channels();
                        const uint32_t max_spatial = in_h * in_w;
                        int8_t* work_a = up.scratch;
                        int8_t* work_b =
                            up.scratch + static_cast<std::size_t>(max_spatial) * expand_c;
                        // Residual aliases block input; body writes only to scratch/`out`.
                        const int8_t* residual_src = up.has_residual ? current : nullptr;

                        const int8_t* cur_data = current;
                        int8_t* next_data = work_a;
                        bool cur_in_work = false;
                        bool ok = true;
                        bool up_residual_done = false;

                        if (up.has_start_dw)
                        {
#if NETKIT_STAGE_TIMING
                            ScopedStageTimer _t_sdw(&g_stage_timing.uib_start_dw_us);
#endif
                            ok = false;
                            if constexpr (XnnpackQuant::kEnabled)
                            {
                                ok = XnnpackQuant::TryDepthwiseConv2dNhwcQuantPlan(
                                    up.start_dw,
                                    cur_data,
                                    uib.start_dw_weights_q,
                                    uib.start_dw_bias_q,
                                    next_data);
                            }
                            if constexpr (EspNnQuant::kEnabled)
                            {
                                ok = ok || EspNnQuant::TryDepthwiseConv2dNhwcQuantPlan(
                                               up.start_dw,
                                               cur_data,
                                               up.start_dw.weights_hwc ? up.start_dw.weights_hwc
                                                                       : uib.start_dw_weights_q,
                                               uib.start_dw_bias_q,
                                               next_data);
                            }

                            if constexpr (NmsisNnQuant::kEnabled)
                            {
                                ok = ok || NmsisNnQuant::TryDepthwiseConv2dNhwcQuantPlan(
                                               up.start_dw,
                                               cur_data,
                                               up.start_dw.weights_hwc ? up.start_dw.weights_hwc
                                                                       : uib.start_dw_weights_q,
                                               uib.start_dw_bias_q,
                                               next_data);
                            }
                            if constexpr (CmsisNnQuant::kEnabled)
                            {
                                ok = ok || CmsisNnQuant::TryDepthwiseConv2dNhwcQuantPlan(
                                               up.start_dw,
                                               cur_data,
                                               up.start_dw.weights_hwc ? up.start_dw.weights_hwc
                                                                       : uib.start_dw_weights_q,
                                               uib.start_dw_bias_q,
                                               next_data);
                            }
#if !NETKIT_MCU_CMSIS_ONLY
                            if (!ok)
                            {
                                QuantOps::DepthwiseConv2dNhwcQuant(
                                    cur_data,
                                    in_h,
                                    in_w,
                                    in_c,
                                    uib.start_dw_weights_q,
                                    uib.start_dw_bias_q,
                                    up.start_dw.kernel_h,
                                    up.start_dw.kernel_w,
                                    up.start_dw.stride,
                                    up.start_dw.pad_h,
                                    up.start_dw.pad_w,
                                    up.start_dw.pad_h,
                                    up.start_dw.pad_w,
                                    uib.start_dw_quant,
                                    up.start_dw.clamp,
                                    next_data,
                                    up.start_dw.multipliers,
                                    up.start_dw.shifts,
                                    &up.start_dw.act_min,
                                    &up.start_dw.act_max,
                                    up.start_dw.bias_folded);
                                ok = true;
                            }
#endif
                            if (ok)
                            {
                                cur_data = next_data;
                                cur_in_work = true;
                            }
                        }

                        if (ok)
                        {
#if NETKIT_STAGE_TIMING
                            ScopedStageTimer _t_exp(&g_stage_timing.uib_expand_us);
#endif
                            int8_t* expand_out = cur_in_work ? work_b : work_a;
                            ok = false;
                            if constexpr (XnnpackQuant::kEnabled)
                            {
                                ok = XnnpackQuant::TryConv2dNhwcQuantPlan(
                                    up.expand,
                                    cur_data,
                                    uib.expand_weights_q,
                                    uib.expand_bias_q,
                                    expand_out);
                            }
                            if constexpr (EspNnQuant::kEnabled)
                            {
                                ok = ok || EspNnQuant::TryConv2dNhwcQuantPlan(
                                               up.expand,
                                               cur_data,
                                               uib.expand_weights_q,
                                               uib.expand_bias_q,
                                               expand_out);
                            }

                            if constexpr (NmsisNnQuant::kEnabled)
                            {
                                ok = ok || NmsisNnQuant::TryConv2dNhwcQuantPlan(
                                               up.expand,
                                               cur_data,
                                               uib.expand_weights_q,
                                               uib.expand_bias_q,
                                               expand_out);
                            }
                            if constexpr (CmsisNnQuant::kEnabled)
                            {
                                ok = ok || CmsisNnQuant::TryConv2dNhwcQuantPlan(
                                               up.expand,
                                               cur_data,
                                               uib.expand_weights_q,
                                               uib.expand_bias_q,
                                               expand_out);
                            }
#if !NETKIT_MCU_CMSIS_ONLY
                            if (!ok)
                            {
                                const uint32_t cur_h = static_cast<uint32_t>(
                                    up.has_start_dw ? up.start_dw.out_h : up.in_h);
                                const uint32_t cur_w = static_cast<uint32_t>(
                                    up.has_start_dw ? up.start_dw.out_w : up.in_w);
                                QuantOps::Conv2dNhwcQuant(cur_data,
                                                          cur_h,
                                                          cur_w,
                                                          in_c,
                                                          uib.expand_weights_q,
                                                          uib.expand_bias_q,
                                                          up.expand.kernel_size,
                                                          up.expand.stride,
                                                          up.expand.pad_h,
                                                          up.expand.pad_w,
                                                          up.expand.pad_h,
                                                          up.expand.pad_w,
                                                          static_cast<int>(expand_c),
                                                          uib.expand_quant,
                                                          up.expand.clamp,
                                                          expand_out,
                                                          nullptr,
                                                          up.expand.multipliers,
                                                          up.expand.shifts,
                                                          &up.expand.act_min,
                                                          &up.expand.act_max,
                                                          up.expand.bias_folded);
                                ok = true;
                            }
#endif
                            if (ok)
                            {
                                cur_data = expand_out;
                                cur_in_work = expand_out == work_a;
                            }
                        }

                        if (ok && up.has_middle_dw)
                        {
#if NETKIT_STAGE_TIMING
                            ScopedStageTimer _t_mdw(&g_stage_timing.uib_middle_dw_us);
#endif
                            int8_t* middle_out = cur_in_work ? work_b : work_a;
                            ok = false;
                            if constexpr (XnnpackQuant::kEnabled)
                            {
                                ok = XnnpackQuant::TryDepthwiseConv2dNhwcQuantPlan(
                                    up.middle_dw,
                                    cur_data,
                                    uib.middle_dw_weights_q,
                                    uib.middle_dw_bias_q,
                                    middle_out);
                            }
                            if constexpr (EspNnQuant::kEnabled)
                            {
                                ok = ok || EspNnQuant::TryDepthwiseConv2dNhwcQuantPlan(
                                               up.middle_dw,
                                               cur_data,
                                               up.middle_dw.weights_hwc ? up.middle_dw.weights_hwc
                                                                        : uib.middle_dw_weights_q,
                                               uib.middle_dw_bias_q,
                                               middle_out);
                            }

                            if constexpr (NmsisNnQuant::kEnabled)
                            {
                                ok = ok || NmsisNnQuant::TryDepthwiseConv2dNhwcQuantPlan(
                                               up.middle_dw,
                                               cur_data,
                                               up.middle_dw.weights_hwc ? up.middle_dw.weights_hwc
                                                                        : uib.middle_dw_weights_q,
                                               uib.middle_dw_bias_q,
                                               middle_out);
                            }
                            if constexpr (CmsisNnQuant::kEnabled)
                            {
                                ok = ok || CmsisNnQuant::TryDepthwiseConv2dNhwcQuantPlan(
                                               up.middle_dw,
                                               cur_data,
                                               up.middle_dw.weights_hwc ? up.middle_dw.weights_hwc
                                                                        : uib.middle_dw_weights_q,
                                               uib.middle_dw_bias_q,
                                               middle_out);
                            }
#if !NETKIT_MCU_CMSIS_ONLY
                            if (!ok)
                            {
                                QuantOps::DepthwiseConv2dNhwcQuant(
                                    cur_data,
                                    static_cast<uint32_t>(up.expand.out_h),
                                    static_cast<uint32_t>(up.expand.out_w),
                                    expand_c,
                                    uib.middle_dw_weights_q,
                                    uib.middle_dw_bias_q,
                                    up.middle_dw.kernel_h,
                                    up.middle_dw.kernel_w,
                                    up.middle_dw.stride,
                                    up.middle_dw.pad_h,
                                    up.middle_dw.pad_w,
                                    up.middle_dw.pad_h,
                                    up.middle_dw.pad_w,
                                    uib.middle_dw_quant,
                                    up.middle_dw.clamp,
                                    middle_out,
                                    up.middle_dw.multipliers,
                                    up.middle_dw.shifts,
                                    &up.middle_dw.act_min,
                                    &up.middle_dw.act_max,
                                    up.middle_dw.bias_folded);
                                ok = true;
                            }
#endif
                            if (ok)
                                cur_data = middle_out;
                        }

                        if (ok)
                        {
#if NETKIT_STAGE_TIMING
                            ScopedStageTimer _t_proj(&g_stage_timing.uib_proj_us);
#endif
                            ok = false;
                            if constexpr (XnnpackQuant::kEnabled)
                            {
                                ok = XnnpackQuant::TryConv2dNhwcQuantPlan(
                                    up.proj, cur_data, uib.proj_weights_q, uib.proj_bias_q, out);
                            }
                            if constexpr (EspNnQuant::kEnabled)
                            {
                                ok = ok || EspNnQuant::TryConv2dNhwcQuantPlan(
                                               up.proj,
                                               cur_data,
                                               uib.proj_weights_q,
                                               uib.proj_bias_q,
                                               out);
                            }

                            if constexpr (NmsisNnQuant::kEnabled)
                            {
                                ok = ok || NmsisNnQuant::TryConv2dNhwcQuantPlan(
                                               up.proj,
                                               cur_data,
                                               uib.proj_weights_q,
                                               uib.proj_bias_q,
                                               out);
                            }
                            if constexpr (CmsisNnQuant::kEnabled)
                            {
                                ok = ok || CmsisNnQuant::TryConv2dNhwcQuantPlan(
                                               up.proj,
                                               cur_data,
                                               uib.proj_weights_q,
                                               uib.proj_bias_q,
                                               out);
                            }
#if !NETKIT_MCU_CMSIS_ONLY
                            if (!ok)
                            {
                                const uint32_t proj_in_h = static_cast<uint32_t>(
                                    up.has_middle_dw ? up.middle_dw.out_h : up.expand.out_h);
                                const uint32_t proj_in_w = static_cast<uint32_t>(
                                    up.has_middle_dw ? up.middle_dw.out_w : up.expand.out_w);
                                QuantOps::ResidualAddS8 residual{};
                                const QuantOps::ResidualAddS8* residual_ptr = nullptr;
                                if (up.has_residual)
                                {
                                    residual.data = residual_src;
                                    residual.scale = uib.block_input_scale;
                                    residual.zero_point = uib.block_input_zero_point;
                                    residual.add_plan = up.add.ready ? &up.add : nullptr;
                                    residual_ptr = &residual;
                                }
                                QuantOps::Conv2dNhwcQuant(cur_data,
                                                          proj_in_h,
                                                          proj_in_w,
                                                          expand_c,
                                                          uib.proj_weights_q,
                                                          uib.proj_bias_q,
                                                          up.proj.kernel_size,
                                                          up.proj.stride,
                                                          up.proj.pad_h,
                                                          up.proj.pad_w,
                                                          up.proj.pad_h,
                                                          up.proj.pad_w,
                                                          static_cast<int>(up.out_c),
                                                          uib.proj_quant,
                                                          up.proj.clamp,
                                                          out,
                                                          residual_ptr,
                                                          up.proj.multipliers,
                                                          up.proj.shifts,
                                                          &up.proj.act_min,
                                                          &up.proj.act_max,
                                                          up.proj.bias_folded);
                                ok = true;
                                // Residual already fused in Conv2dNhwcQuant epilogue.
                                up_residual_done = true;
                            }
#endif
                        }

                        if (ok && up.has_residual && !up_residual_done)
                        {
#if NETKIT_STAGE_TIMING
                            ScopedStageTimer _t_add(&g_stage_timing.uib_residual_add_us);
#endif
                            const uint32_t count = static_cast<uint32_t>(up.out_h) *
                                                   static_cast<uint32_t>(up.out_w) *
                                                   static_cast<uint32_t>(up.out_c);
                            if (up.add.ready)
                            {
                                QuantOps::ElementwiseAddS8(
                                    out, residual_src, count, up.add, out);
                            }
                            else
                            {
                                QuantOps::ElementwiseAddS8(out,
                                                           residual_src,
                                                           count,
                                                           uib.proj_quant.output_scale,
                                                           uib.proj_quant.output_zero_point,
                                                           uib.block_input_scale,
                                                           uib.block_input_zero_point,
                                                           uib.proj_quant.output_scale,
                                                           uib.proj_quant.output_zero_point,
                                                           out);
                            }
                        }

                        ran_plans = ok;
#if NETKIT_STAGE_TIMING
                        if (ok)
                            ++g_stage_timing.uib_plan_ok;
#endif
                    }
                    if (!ran_plans)
                    {
#if NETKIT_STAGE_TIMING
                        ScopedStageTimer _t_fb(&g_stage_timing.uib_fallback_us);
                        ++g_stage_timing.uib_fallback;
#endif
                        uib.forward_quant(current,
                                          out,
                                          static_cast<uint32_t>(lp.uib.in_h),
                                          static_cast<uint32_t>(lp.uib.in_w));
                    }
#if NETKIT_STAGE_TIMING
                    seen_uib = true;
#endif
                    break;
                }
                case LayerKind::Dense:
                {
                    const int8_t* weights = static_cast<const int8_t*>(block.dense.weights.data);
                    const int32_t* bias = static_cast<const int32_t*>(block.dense.bias.data);
                    int8_t* dense_out = (is_last && output_dest != nullptr) ? output_dest : out;
                    {
                        bool ran = false;
                        if constexpr (XnnpackQuant::kEnabled)
                        {
                            ran = XnnpackQuant::TryFullyConnectedQuantPlan(
                                lp.fc, current, weights, bias, dense_out);
                        }
                        if constexpr (EspNnQuant::kEnabled)
                        {
                            ran = ran || EspNnQuant::TryFullyConnectedQuantPlan(
                                             lp.fc, current, weights, bias, dense_out);
                        }

                        if constexpr (NmsisNnQuant::kEnabled)
                        {
                            ran = ran || NmsisNnQuant::TryFullyConnectedQuantPlan(
                                             lp.fc, current, weights, bias, dense_out);
                        }
                        if constexpr (CmsisNnQuant::kEnabled)
                        {
                            ran = ran || CmsisNnQuant::TryFullyConnectedQuantPlan(
                                             lp.fc, current, weights, bias, dense_out);
                        }
                        if (ran)
                        {
                            if (dense_out == output_dest)
                                out = output_dest;
                            break;
                        }
                    }
#if NETKIT_MCU_CMSIS_ONLY
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
                                                  dense_out,
                                                  lp.fc.multipliers,
                                                  lp.fc.shifts,
                                                  &lp.fc.act_min,
                                                  &lp.fc.act_max,
                                                  lp.fc.bias_folded);
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
                    {
                        bool ran = false;
                        if constexpr (XnnpackQuant::kEnabled)
                        {
                            ran = XnnpackQuant::TryFullyConnectedQuantPlan(
                                lp.fc, current, weights, bias, fc_dest);
                        }
                        if constexpr (EspNnQuant::kEnabled)
                        {
                            ran = ran || EspNnQuant::TryFullyConnectedQuantPlan(
                                             lp.fc, current, weights, bias, fc_dest);
                        }

                        if constexpr (NmsisNnQuant::kEnabled)
                        {
                            ran = ran || NmsisNnQuant::TryFullyConnectedQuantPlan(
                                             lp.fc, current, weights, bias, fc_dest);
                        }
                        if constexpr (CmsisNnQuant::kEnabled)
                        {
                            ran = ran || CmsisNnQuant::TryFullyConnectedQuantPlan(
                                             lp.fc, current, weights, bias, fc_dest);
                        }
                        if (!ran)
                        {
#if NETKIT_MCU_CMSIS_ONLY
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
                                                      fc_dest,
                                                      lp.fc.multipliers,
                                                      lp.fc.shifts,
                                                      &lp.fc.act_min,
                                                      &lp.fc.act_max,
                                                      lp.fc.bias_folded);
#endif
                        }
                    }
                    if (runtime.omit_final_softmax)
                    {
                        out = softmax_out;
                        (void)output_format;
                        break;
                    }
                    {
                        bool ran = false;
                        if constexpr (EspNnQuant::kEnabled)
                        {
                            ran = EspNnQuant::TrySoftmaxS8Plan(
                                lp.softmax, runtime.logits, softmax_out);
                        }

                        if constexpr (NmsisNnQuant::kEnabled)
                        {
                            ran = NmsisNnQuant::TrySoftmaxS8Plan(
                                lp.softmax, runtime.logits, softmax_out);
                        }
                        if constexpr (CmsisNnQuant::kEnabled)
                        {
                            ran = CmsisNnQuant::TrySoftmaxS8Plan(
                                lp.softmax, runtime.logits, softmax_out);
                        }
                        if (!ran)
                        {
#if NETKIT_MCU_CMSIS_ONLY
                        return false;
#else
                        QuantOps::SoftmaxS8(runtime.logits,
                                            static_cast<uint32_t>(lp.softmax.row_size),
                                            block.dense.quant.params.output_scale,
                                            softmax_out);
#endif
                        }
                    }
                    out = softmax_out;
                    (void)output_format;
                    break;
                }
                default:
                    break;
            }

#if NETKIT_STAGE_TIMING
            if (lp.kind != LayerKind::MobilenetV4Uib && lp.kind != LayerKind::FlattenView)
            {
                *layer_bucket += StageElapsedUs(layer_t0);
            }
#endif

            if (lp.kind != LayerKind::FlattenView)
            {
                current = out;
                if (out != output_dest)
                    use_a = !use_a;
            }
        }

#if NETKIT_STAGE_TIMING
        ++g_stage_timing.invokes;
        PrintStageTimingIfDue();
#endif

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

#if NETKIT_STAGE_TIMING
void ResetStageTiming()
{
    g_stage_timing = {};
}

void PrintStageTimingSummary()
{
    char line[640];
    FormatStageTimingLine(line, sizeof(line));
#if defined(NETKIT_STAGE_TIMING_UART) && NETKIT_STAGE_TIMING_UART
    // uart_printf() uses a 192-byte stack buffer; write the full line directly.
    uart_write(line);
    uart_write("\r\n");
#else
    std::fprintf(stderr, "%s\n", line);
#endif
}

void RecordCmsisConvKernelUs(double us)
{
    g_stage_timing.cmsis_conv_kernel_us += us;
}
#endif

}  // namespace CmsisQuantPlan
