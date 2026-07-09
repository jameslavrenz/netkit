/*
 * CMSIS-NN int8 adapters for netkit quantized conv / pool / FC (TFLite-style).
 */
#include "cmsis_nn_quant.hpp"
#include "arena.hpp"
#include "cmsis_buffer_size.hpp"
#include "kernel_workspace.hpp"
#include "nk_op_detail.hpp"
#include "netkit_config.h"
#include "quant_integer.hpp"
#include "quant_trace.hpp"

#if defined(NETKIT_USE_CMSIS_NN) && NETKIT_USE_CMSIS_NN && NETKIT_CMSIS_NN_ALLOWED

#include <arm_nnfunctions.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace
{
    bool cmsis_status_ok(arm_cmsis_nn_status status)
    {
        return status == ARM_CMSIS_NN_SUCCESS;
    }

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

    cmsis_nn_activation activation_clamp(QuantInteger::QuantClamp clamp,
                                         float output_scale,
                                         int32_t output_zero_point)
    {
        int32_t act_min = -128;
        int32_t act_max = 127;
        QuantInteger::QuantClampRange(clamp, output_scale, output_zero_point, &act_min, &act_max);
        return {.min = act_min, .max = act_max};
    }

    // Legacy helper: ReLU clamp only (identity otherwise).
    cmsis_nn_activation activation_relu(bool apply_relu)
    {
        return activation_clamp(apply_relu ? QuantInteger::QuantClamp::ReLU
                                           : QuantInteger::QuantClamp::None,
                                1.0f,
                                0);
    }

    bool FillPerChannelQuant(const NkFormat::MlpLayerQuantDesc& quant,
                             int out_channels,
                             int32_t* multipliers,
                             int32_t* shifts)
    {
        if (out_channels <= 0 || quant.output_scale <= 0.0f)
            return false;

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

    bool BindContext(cmsis_nn_context& ctx, int32_t required_bytes)
    {
        if (required_bytes <= 0)
        {
            ctx.buf = nullptr;
            ctx.size = 0;
            return true;
        }
        return BindCmsisWorkspace(ctx.buf, ctx.size, required_bytes);
    }

    uint32_t ActiveWorkspaceBytes()
    {
        const KernelWorkspace* ws = GetActiveKernelWorkspace();
        return ws ? static_cast<uint32_t>(ws->size_bytes) : 0u;
    }
}

namespace CmsisNnQuant
{

void FinalizeConv2DPlan(CmsisQuantPlan::Conv2DPlan& plan)
{
#if NETKIT_CMSIS_PLAN_HOIST
    if (!plan.ready || !plan.multipliers || !plan.shifts)
        return;

    plan.cmsis.conv = {
        .input_offset = plan.input_offset,
        .output_offset = plan.output_offset,
        .stride = {.w = plan.stride, .h = plan.stride},
        .padding = {.w = plan.pad_w, .h = plan.pad_h},
        .dilation = {.w = 1, .h = 1},
        .activation = activation_clamp(plan.clamp, plan.output_scale, -plan.output_offset),
    };
    plan.cmsis.quant = {
        .multiplier = plan.multipliers,
        .shift = plan.shifts,
    };
    plan.cmsis.input = {.n = 1, .h = plan.in_h, .w = plan.in_w, .c = plan.in_c};
    plan.cmsis.filter = {
        .n = plan.out_c, .h = plan.kernel_size, .w = plan.kernel_size, .c = plan.in_c,
    };
    plan.cmsis.bias = {.n = 1, .h = 1, .w = 1, .c = plan.out_c};
    plan.cmsis.output = {.n = 1, .h = plan.out_h, .w = plan.out_w, .c = plan.out_c};
    plan.cmsis.ready = true;
#else
    (void)plan;
#endif
}

void FinalizeDepthwiseConv2DPlan(CmsisQuantPlan::DepthwiseConv2DPlan& plan)
{
#if NETKIT_CMSIS_PLAN_HOIST
    if (!plan.ready || !plan.multipliers || !plan.shifts)
        return;

    plan.cmsis.dw_conv = {
        .input_offset = plan.input_offset,
        .output_offset = plan.output_offset,
        .ch_mult = 1,
        .stride = {.w = plan.stride, .h = plan.stride},
        .padding = {.w = plan.pad_w, .h = plan.pad_h},
        .dilation = {.w = 1, .h = 1},
        .activation = activation_clamp(plan.clamp, plan.output_scale, -plan.output_offset),
    };
    plan.cmsis.quant = {
        .multiplier = plan.multipliers,
        .shift = plan.shifts,
    };
    plan.cmsis.input = {.n = 1, .h = plan.in_h, .w = plan.in_w, .c = plan.channels};
    plan.cmsis.filter = {.n = 1, .h = plan.kernel_h, .w = plan.kernel_w, .c = plan.channels};
    plan.cmsis.bias = {.n = 1, .h = 1, .w = 1, .c = plan.channels};
    plan.cmsis.output = {.n = 1, .h = plan.out_h, .w = plan.out_w, .c = plan.channels};
    plan.cmsis.ready = true;
#else
    (void)plan;
#endif
}

void FinalizePool2DPlan(CmsisQuantPlan::Pool2DPlan& plan)
{
#if NETKIT_CMSIS_PLAN_HOIST
    if (!plan.ready)
        return;

    plan.cmsis.pool = {
        .stride = {.w = plan.stride, .h = plan.stride},
        .padding = {.w = plan.pad_w, .h = plan.pad_h},
        .activation = activation_clamp(plan.clamp, plan.output_scale, plan.output_zero_point),
    };
    plan.cmsis.input = {.n = 1, .h = plan.in_h, .w = plan.in_w, .c = plan.in_c};
    plan.cmsis.filter = {.n = 1, .h = plan.pool_h, .w = plan.pool_w, .c = 1};
    plan.cmsis.output = {.n = 1, .h = plan.out_h, .w = plan.out_w, .c = plan.in_c};
    plan.cmsis.ready = true;
#else
    (void)plan;
#endif
}

bool FinalizeFcPlan(CmsisQuantPlan::FcPlan& plan,
                    const int8_t* weights,
                    const int32_t* bias,
                    Arena& arena)
{
#if NETKIT_CMSIS_PLAN_HOIST
    if (!plan.ready)
        return false;

    plan.cmsis.fc = {
        .input_offset = plan.input_offset,
        .filter_offset = plan.filter_offset,
        .output_offset = plan.output_offset,
        .activation = activation_clamp(plan.clamp, plan.output_scale, -plan.output_offset),
    };
    plan.cmsis.quant = {
        .multiplier = plan.multiplier,
        .shift = plan.shift,
    };
    plan.cmsis.input = {.n = 1, .h = 1, .w = 1, .c = plan.in_features};
    plan.cmsis.filter = {
        .n = plan.in_features, .h = 1, .w = 1, .c = plan.out_features,
    };
    plan.cmsis.bias = {.n = 1, .h = 1, .w = 1, .c = plan.out_features};
    plan.cmsis.output = {.n = 1, .h = 1, .w = 1, .c = plan.out_features};
    plan.cmsis.ready = true;

#if defined(NETKIT_KERNELS_OPTIMIZED_FOR_SPEED) && NETKIT_KERNELS_OPTIMIZED_FOR_SPEED
    if (weights && bias && plan.workspace_bytes > 0 && !plan.kernel_sums)
    {
        plan.kernel_sums = static_cast<int32_t*>(arena.alloc(
            static_cast<std::size_t>(plan.workspace_bytes), alignof(int32_t)));
        if (!plan.kernel_sums)
            return false;

        if (arm_vector_sum_s8(plan.kernel_sums,
                              plan.in_features,
                              plan.out_features,
                              weights,
                              plan.input_offset,
                              plan.filter_offset,
                              bias) != ARM_CMSIS_NN_SUCCESS)
        {
            return false;
        }
    }
#else
    (void)weights;
    (void)bias;
    (void)arena;
#endif
    return true;
#else
    (void)plan;
    (void)weights;
    (void)bias;
    (void)arena;
    return plan.ready;
#endif
}

bool TryConv2dNhwcQuantPlan(const CmsisQuantPlan::Conv2DPlan& plan,
                            const int8_t* input,
                            const int8_t* weights,
                            const int32_t* bias,
                            int8_t* output)
{
    const uint32_t workspace_bytes = ActiveWorkspaceBytes();

    if (!plan.ready || !input || !weights || !bias || !output || !plan.multipliers || !plan.shifts)
    {
        QuantTrace::RecordConv2dCmsisFail(QuantTrace::Conv2dFail::NullPtr, 0, workspace_bytes);
        return false;
    }

#if NETKIT_CMSIS_PLAN_HOIST
    if (!plan.cmsis.ready)
    {
        QuantTrace::RecordConv2dCmsisFail(QuantTrace::Conv2dFail::NullPtr, 0, workspace_bytes);
        return false;
    }
#endif

    cmsis_nn_context ctx = {0};
    if (!BindContext(ctx, plan.workspace_bytes))
    {
        QuantTrace::RecordConv2dCmsisFail(
            QuantTrace::Conv2dFail::BindContext, plan.workspace_bytes, workspace_bytes);
        return false;
    }

#if NETKIT_CMSIS_PLAN_HOIST
    if (!cmsis_status_ok(arm_convolve_wrapper_s8(&ctx,
                                                 &plan.cmsis.conv,
                                                 &plan.cmsis.quant,
                                                 &plan.cmsis.input,
                                                 input,
                                                 &plan.cmsis.filter,
                                                 weights,
                                                 &plan.cmsis.bias,
                                                 bias,
                                                 &plan.cmsis.output,
                                                 output)))
#else
    const cmsis_nn_conv_params conv_params = {
        .input_offset = plan.input_offset,
        .output_offset = plan.output_offset,
        .stride = {.w = plan.stride, .h = plan.stride},
        .padding = {.w = plan.pad_w, .h = plan.pad_h},
        .dilation = {.w = 1, .h = 1},
        .activation = activation_clamp(plan.clamp, plan.output_scale, -plan.output_offset),
    };

    const cmsis_nn_per_channel_quant_params quant_params = {
        .multiplier = plan.multipliers,
        .shift = plan.shifts,
    };

    const cmsis_nn_dims input_dims = {
        .n = 1, .h = plan.in_h, .w = plan.in_w, .c = plan.in_c,
    };
    const cmsis_nn_dims filter_dims = {
        .n = plan.out_c, .h = plan.kernel_size, .w = plan.kernel_size, .c = plan.in_c,
    };
    const cmsis_nn_dims bias_dims = {.n = 1, .h = 1, .w = 1, .c = plan.out_c};
    const cmsis_nn_dims output_dims = {
        .n = 1, .h = plan.out_h, .w = plan.out_w, .c = plan.out_c,
    };

    if (!cmsis_status_ok(arm_convolve_wrapper_s8(&ctx,
                                                 &conv_params,
                                                 &quant_params,
                                                 &input_dims,
                                                 input,
                                                 &filter_dims,
                                                 weights,
                                                 &bias_dims,
                                                 bias,
                                                 &output_dims,
                                                 output)))
#endif
    {
        QuantTrace::RecordConv2dCmsisFail(
            QuantTrace::Conv2dFail::CmsisStatus, plan.workspace_bytes, workspace_bytes);
        return false;
    }

    QuantTrace::RecordConv2dCmsisOk();
    return true;
}

bool TryDepthwiseConv2dNhwcQuantPlan(const CmsisQuantPlan::DepthwiseConv2DPlan& plan,
                                     const int8_t* input,
                                     const int8_t* weights,
                                     const int32_t* bias,
                                     int8_t* output)
{
    const uint32_t workspace_bytes = ActiveWorkspaceBytes();

    if (!plan.ready || !input || !weights || !bias || !output || !plan.multipliers || !plan.shifts)
    {
        QuantTrace::RecordConv2dCmsisFail(QuantTrace::Conv2dFail::NullPtr, 0, workspace_bytes);
        return false;
    }

#if NETKIT_CMSIS_PLAN_HOIST
    if (!plan.cmsis.ready)
    {
        QuantTrace::RecordConv2dCmsisFail(QuantTrace::Conv2dFail::NullPtr, 0, workspace_bytes);
        return false;
    }
#endif

    cmsis_nn_context ctx = {0};
    if (!BindContext(ctx, plan.workspace_bytes))
    {
        QuantTrace::RecordConv2dCmsisFail(
            QuantTrace::Conv2dFail::BindContext, plan.workspace_bytes, workspace_bytes);
        return false;
    }

#if NETKIT_CMSIS_PLAN_HOIST
    if (!cmsis_status_ok(arm_depthwise_conv_wrapper_s8(&ctx,
                                                       &plan.cmsis.dw_conv,
                                                       &plan.cmsis.quant,
                                                       &plan.cmsis.input,
                                                       input,
                                                       &plan.cmsis.filter,
                                                       weights,
                                                       &plan.cmsis.bias,
                                                       bias,
                                                       &plan.cmsis.output,
                                                       output)))
#else
    const cmsis_nn_dw_conv_params dw_conv_params = {
        .input_offset = plan.input_offset,
        .output_offset = plan.output_offset,
        .ch_mult = 1,
        .stride = {.w = plan.stride, .h = plan.stride},
        .padding = {.w = plan.pad_w, .h = plan.pad_h},
        .dilation = {.w = 1, .h = 1},
        .activation = activation_clamp(plan.clamp, plan.output_scale, -plan.output_offset),
    };

    const cmsis_nn_per_channel_quant_params quant_params = {
        .multiplier = plan.multipliers,
        .shift = plan.shifts,
    };

    const cmsis_nn_dims input_dims = {
        .n = 1, .h = plan.in_h, .w = plan.in_w, .c = plan.channels,
    };
    const cmsis_nn_dims filter_dims = {
        .n = 1, .h = plan.kernel_h, .w = plan.kernel_w, .c = plan.channels,
    };
    const cmsis_nn_dims bias_dims = {.n = 1, .h = 1, .w = 1, .c = plan.channels};
    const cmsis_nn_dims output_dims = {
        .n = 1, .h = plan.out_h, .w = plan.out_w, .c = plan.channels,
    };

    if (!cmsis_status_ok(arm_depthwise_conv_wrapper_s8(&ctx,
                                                       &dw_conv_params,
                                                       &quant_params,
                                                       &input_dims,
                                                       input,
                                                       &filter_dims,
                                                       weights,
                                                       &bias_dims,
                                                       bias,
                                                       &output_dims,
                                                       output)))
#endif
    {
        QuantTrace::RecordConv2dCmsisFail(
            QuantTrace::Conv2dFail::CmsisStatus, plan.workspace_bytes, workspace_bytes);
        return false;
    }

    QuantTrace::RecordConv2dCmsisOk();
    return true;
}

bool TryMaxPool2dNhwcQuantPlan(const CmsisQuantPlan::Pool2DPlan& plan,
                               const int8_t* input,
                               int8_t* output)
{
    if (!plan.ready || !input || !output)
        return false;

#if NETKIT_CMSIS_PLAN_HOIST
    if (!plan.cmsis.ready)
        return false;
#endif

    cmsis_nn_context ctx = {0};
#if NETKIT_CMSIS_PLAN_HOIST
    if (!cmsis_status_ok(arm_max_pool_s8(&ctx,
                                         &plan.cmsis.pool,
                                         &plan.cmsis.input,
                                         input,
                                         &plan.cmsis.filter,
                                         &plan.cmsis.output,
                                         output)))
#else
    const cmsis_nn_pool_params pool_params = {
        .stride = {.w = plan.stride, .h = plan.stride},
        .padding = {.w = plan.pad_w, .h = plan.pad_h},
        .activation = activation_clamp(plan.clamp, plan.output_scale, plan.output_zero_point),
    };

    const cmsis_nn_dims input_dims = {
        .n = 1, .h = plan.in_h, .w = plan.in_w, .c = plan.in_c,
    };
    const cmsis_nn_dims filter_dims = {
        .n = 1, .h = plan.pool_h, .w = plan.pool_w, .c = 1,
    };
    const cmsis_nn_dims output_dims = {
        .n = 1, .h = plan.out_h, .w = plan.out_w, .c = plan.in_c,
    };

    if (!cmsis_status_ok(
            arm_max_pool_s8(&ctx, &pool_params, &input_dims, input, &filter_dims, &output_dims, output)))
#endif
    {
        return false;
    }

    QuantTrace::RecordPoolCmsisOk();
    return true;
}

bool TryFullyConnectedQuantPlan(const CmsisQuantPlan::FcPlan& plan,
                                const int8_t* input,
                                const int8_t* weights,
                                const int32_t* bias,
                                int8_t* output_int8)
{
    const uint32_t workspace_bytes = ActiveWorkspaceBytes();

    if (!plan.ready || !input || !weights || !bias || !output_int8)
    {
        QuantTrace::RecordFcCmsisFail(QuantTrace::FcFail::NullPtr, 0, workspace_bytes);
        return false;
    }

#if NETKIT_CMSIS_PLAN_HOIST
    if (!plan.cmsis.ready)
    {
        QuantTrace::RecordFcCmsisFail(QuantTrace::FcFail::NullPtr, 0, workspace_bytes);
        return false;
    }
#endif

    cmsis_nn_context ctx = {0};
    if (plan.kernel_sums != nullptr)
    {
        ctx.buf = plan.kernel_sums;
        ctx.size = plan.workspace_bytes;
    }
    else if (!BindContext(ctx, plan.workspace_bytes))
    {
        QuantTrace::RecordFcCmsisFail(QuantTrace::FcFail::BindContext, plan.workspace_bytes, workspace_bytes);
        return false;
    }
#if defined(NETKIT_KERNELS_OPTIMIZED_FOR_SPEED) && NETKIT_KERNELS_OPTIMIZED_FOR_SPEED
    else if (ctx.buf != nullptr && plan.workspace_bytes > 0)
    {
        if (arm_vector_sum_s8(static_cast<int32_t*>(ctx.buf),
                              plan.in_features,
                              plan.out_features,
                              weights,
                              plan.input_offset,
                              plan.filter_offset,
                              bias) != ARM_CMSIS_NN_SUCCESS)
        {
            QuantTrace::RecordFcCmsisFail(QuantTrace::FcFail::CmsisStatus, plan.workspace_bytes, workspace_bytes);
            return false;
        }
    }
#endif

    int32_t multiplier = plan.multiplier;
    int32_t shift = plan.shift;
    const cmsis_nn_quant_params wrapper_quant = {
        .multiplier = &multiplier,
        .shift = &shift,
        .is_per_channel = 0,
    };

#if NETKIT_CMSIS_PLAN_HOIST
    if (!cmsis_status_ok(arm_fully_connected_wrapper_s8(&ctx,
                                                        &plan.cmsis.fc,
                                                        &wrapper_quant,
                                                        &plan.cmsis.input,
                                                        input,
                                                        &plan.cmsis.filter,
                                                        weights,
                                                        &plan.cmsis.bias,
                                                        bias,
                                                        &plan.cmsis.output,
                                                        output_int8)))
#else
    const cmsis_nn_per_tensor_quant_params quant_params = {
        .multiplier = plan.multiplier,
        .shift = plan.shift,
    };

    const cmsis_nn_fc_params fc_params = {
        .input_offset = plan.input_offset,
        .filter_offset = plan.filter_offset,
        .output_offset = plan.output_offset,
        .activation = activation_clamp(plan.clamp, plan.output_scale, -plan.output_offset),
    };

    const cmsis_nn_dims input_dims = {.n = 1, .h = 1, .w = 1, .c = plan.in_features};
    const cmsis_nn_dims filter_dims = {
        .n = plan.in_features, .h = 1, .w = 1, .c = plan.out_features,
    };
    const cmsis_nn_dims bias_dims = {.n = 1, .h = 1, .w = 1, .c = plan.out_features};
    const cmsis_nn_dims output_dims = {.n = 1, .h = 1, .w = 1, .c = plan.out_features};

    if (!cmsis_status_ok(arm_fully_connected_wrapper_s8(&ctx,
                                                        &fc_params,
                                                        &wrapper_quant,
                                                        &input_dims,
                                                        input,
                                                        &filter_dims,
                                                        weights,
                                                        &bias_dims,
                                                        bias,
                                                        &output_dims,
                                                        output_int8)))
#endif
    {
        QuantTrace::RecordFcCmsisFail(QuantTrace::FcFail::CmsisStatus, plan.workspace_bytes, workspace_bytes);
        return false;
    }

    QuantTrace::RecordFcCmsisOk();
    return true;
}

bool TrySoftmaxS8Plan(const CmsisQuantPlan::SoftmaxPlan& plan, const int8_t* input, int8_t* output)
{
    if (!plan.ready || !input || !output || plan.row_size <= 0)
        return false;

    arm_softmax_s8(input,
                   1,
                   plan.row_size,
                   plan.params.mult,
                   plan.params.shift,
                   plan.params.diff_min,
                   output);
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

    const uint32_t out_h = nk_op_detail::CalcOutputDimAsymmetric(
        in_h, kernel_size, stride, pad_h, pad_h_end);
    const uint32_t out_w = nk_op_detail::CalcOutputDimAsymmetric(
        in_w, kernel_size, stride, pad_w, pad_w_end);

    if (out_channels > 512)
    {
        QuantTrace::RecordConv2dCmsisFail(QuantTrace::Conv2dFail::TooManyChannels, 0, workspace_bytes);
        return false;
    }

    int32_t multipliers[512];
    int32_t shifts[512];
    if (!FillPerChannelQuant(quant, out_channels, multipliers, shifts))
    {
        QuantTrace::RecordConv2dCmsisFail(QuantTrace::Conv2dFail::BadQuant, 0, workspace_bytes);
        return false;
    }

    const cmsis_nn_conv_params conv_params = {
        .input_offset = -quant.input_zero_point,
        .output_offset = -quant.output_zero_point,
        .stride = {.w = stride, .h = stride},
        .padding = {.w = pad_w, .h = pad_h},
        .dilation = {.w = 1, .h = 1},
        .activation = activation_relu(apply_relu),
    };

    const cmsis_nn_per_channel_quant_params quant_params = {
        .multiplier = multipliers,
        .shift = shifts,
    };

    const cmsis_nn_dims input_dims = {
        .n = 1,
        .h = static_cast<int32_t>(in_h),
        .w = static_cast<int32_t>(in_w),
        .c = static_cast<int32_t>(in_c),
    };
    const cmsis_nn_dims filter_dims = {
        .n = out_channels,
        .h = kernel_size,
        .w = kernel_size,
        .c = static_cast<int32_t>(in_c),
    };
    const cmsis_nn_dims bias_dims = {.n = 1, .h = 1, .w = 1, .c = out_channels};
    const cmsis_nn_dims output_dims = {
        .n = 1,
        .h = static_cast<int32_t>(out_h),
        .w = static_cast<int32_t>(out_w),
        .c = out_channels,
    };

    const int32_t buf_size = static_cast<int32_t>(CmsisConv2dS8WorkspaceBytes(
        in_h, in_w, kernel_size, stride, pad_h, pad_w, in_c, out_channels));
    cmsis_nn_context ctx = {0};
    if (!BindContext(ctx, buf_size))
    {
        QuantTrace::RecordConv2dCmsisFail(
            QuantTrace::Conv2dFail::BindContext, buf_size, workspace_bytes);
        return false;
    }

    if (!cmsis_status_ok(arm_convolve_wrapper_s8(&ctx,
                                                 &conv_params,
                                                 &quant_params,
                                                 &input_dims,
                                                 input,
                                                 &filter_dims,
                                                 weights,
                                                 &bias_dims,
                                                 bias,
                                                 &output_dims,
                                                 output)))
    {
        QuantTrace::RecordConv2dCmsisFail(
            QuantTrace::Conv2dFail::CmsisStatus, buf_size, workspace_bytes);
        return false;
    }

    QuantTrace::RecordConv2dCmsisOk();
    return true;
}

bool TryDepthwiseConv2dNhwcQuant(const int8_t* input,
                                 uint32_t in_h,
                                 uint32_t in_w,
                                 uint32_t channels,
                                 const int8_t* weights_chw,
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
    const uint32_t workspace_bytes = ActiveWorkspaceBytes();

    if (!input || !weights_chw || !bias || !output || channels == 0)
    {
        QuantTrace::RecordConv2dCmsisFail(QuantTrace::Conv2dFail::NullPtr, 0, workspace_bytes);
        return false;
    }
    if (pad_h != pad_h_end || pad_w != pad_w_end)
    {
        QuantTrace::RecordConv2dCmsisFail(QuantTrace::Conv2dFail::AsymmetricPad, 0, workspace_bytes);
        return false;
    }
    if (channels > 512 || quant.output_scale <= 0.0f)
    {
        QuantTrace::RecordConv2dCmsisFail(QuantTrace::Conv2dFail::BadQuant, 0, workspace_bytes);
        return false;
    }

    const uint32_t out_h =
        nk_op_detail::CalcOutputDimAsymmetric(in_h, kernel_h, stride, pad_h, pad_h_end);
    const uint32_t out_w =
        nk_op_detail::CalcOutputDimAsymmetric(in_w, kernel_w, stride, pad_w, pad_w_end);

    int32_t multipliers[512];
    int32_t shifts[512];
    if (!FillPerChannelQuant(quant, static_cast<int>(channels), multipliers, shifts))
    {
        QuantTrace::RecordConv2dCmsisFail(QuantTrace::Conv2dFail::BadQuant, 0, workspace_bytes);
        return false;
    }

    const std::size_t kernel_area =
        static_cast<std::size_t>(kernel_h) * static_cast<std::size_t>(kernel_w) *
        static_cast<std::size_t>(channels);
    const int32_t dw_ws = static_cast<int32_t>(CmsisDepthwiseConv2dS8WorkspaceBytes(
        in_h, in_w, kernel_h, kernel_w, stride, pad_h, pad_w, channels));
    const int32_t total_need = dw_ws + static_cast<int32_t>(kernel_area);
    cmsis_nn_context ctx = {0};
    if (!BindContext(ctx, total_need))
    {
        QuantTrace::RecordConv2dCmsisFail(
            QuantTrace::Conv2dFail::BindContext, total_need, workspace_bytes);
        return false;
    }

    // Repack CHW → HWC into the tail of the CMSIS workspace buffer.
    int8_t* weights_hwc = nullptr;
    if (ctx.buf && ctx.size >= total_need)
    {
        weights_hwc = static_cast<int8_t*>(ctx.buf) + dw_ws;
        ctx.size = dw_ws;
    }
    else
    {
        QuantTrace::RecordConv2dCmsisFail(
            QuantTrace::Conv2dFail::BindContext, total_need, workspace_bytes);
        return false;
    }

    for (uint32_t c = 0; c < channels; ++c)
    {
        for (int y = 0; y < kernel_h; ++y)
        {
            for (int x = 0; x < kernel_w; ++x)
            {
                weights_hwc[(y * kernel_w + x) * channels + c] =
                    weights_chw[(c * kernel_h + y) * kernel_w + x];
            }
        }
    }

    const cmsis_nn_dw_conv_params dw_conv_params = {
        .input_offset = -quant.input_zero_point,
        .output_offset = -quant.output_zero_point,
        .ch_mult = 1,
        .stride = {.w = stride, .h = stride},
        .padding = {.w = pad_w, .h = pad_h},
        .dilation = {.w = 1, .h = 1},
        .activation = activation_relu(apply_relu),
    };
    const cmsis_nn_per_channel_quant_params quant_params = {
        .multiplier = multipliers,
        .shift = shifts,
    };
    const cmsis_nn_dims input_dims = {
        .n = 1,
        .h = static_cast<int32_t>(in_h),
        .w = static_cast<int32_t>(in_w),
        .c = static_cast<int32_t>(channels),
    };
    const cmsis_nn_dims filter_dims = {
        .n = 1,
        .h = kernel_h,
        .w = kernel_w,
        .c = static_cast<int32_t>(channels),
    };
    const cmsis_nn_dims bias_dims = {.n = 1, .h = 1, .w = 1, .c = static_cast<int32_t>(channels)};
    const cmsis_nn_dims output_dims = {
        .n = 1,
        .h = static_cast<int32_t>(out_h),
        .w = static_cast<int32_t>(out_w),
        .c = static_cast<int32_t>(channels),
    };

    if (!cmsis_status_ok(arm_depthwise_conv_wrapper_s8(&ctx,
                                                       &dw_conv_params,
                                                       &quant_params,
                                                       &input_dims,
                                                       input,
                                                       &filter_dims,
                                                       weights_hwc,
                                                       &bias_dims,
                                                       bias,
                                                       &output_dims,
                                                       output)))
    {
        QuantTrace::RecordConv2dCmsisFail(
            QuantTrace::Conv2dFail::CmsisStatus, total_need, workspace_bytes);
        return false;
    }

    QuantTrace::RecordConv2dCmsisOk();
    return true;
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
    if (!input || !output)
        return false;

    if (pad_h != pad_h_end || pad_w != pad_w_end)
        return false;

    const uint32_t out_h = nk_op_detail::CalcOutputDimAsymmetric(
        in_h, pool_h, stride, pad_h, pad_h_end);
    const uint32_t out_w = nk_op_detail::CalcOutputDimAsymmetric(
        in_w, pool_w, stride, pad_w, pad_w_end);

    const cmsis_nn_pool_params pool_params = {
        .stride = {.w = stride, .h = stride},
        .padding = {.w = pad_w, .h = pad_h},
        .activation = {.min = -128, .max = 127},
    };

    const cmsis_nn_dims input_dims = {
        .n = 1,
        .h = static_cast<int32_t>(in_h),
        .w = static_cast<int32_t>(in_w),
        .c = static_cast<int32_t>(in_c),
    };
    const cmsis_nn_dims filter_dims = {
        .n = 1,
        .h = pool_h,
        .w = pool_w,
        .c = 1,
    };
    const cmsis_nn_dims output_dims = {
        .n = 1,
        .h = static_cast<int32_t>(out_h),
        .w = static_cast<int32_t>(out_w),
        .c = static_cast<int32_t>(in_c),
    };

    cmsis_nn_context ctx = {0};
    if (!cmsis_status_ok(
            arm_max_pool_s8(&ctx, &pool_params, &input_dims, input, &filter_dims, &output_dims, output)))
    {
        return false;
    }

    QuantTrace::RecordPoolCmsisOk();
    return true;
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
    const uint32_t workspace_bytes = ActiveWorkspaceBytes();

    if (!input || !weights || !bias || batch != 1)
    {
        QuantTrace::RecordFcCmsisFail(QuantTrace::FcFail::NullPtr, 0, workspace_bytes);
        return false;
    }

    if (!output_int8 || out_features == 0 || in_features == 0 || quant.output_scale <= 0.0f)
    {
        QuantTrace::RecordFcCmsisFail(QuantTrace::FcFail::BadShape, 0, workspace_bytes);
        return false;
    }

    int32_t multiplier = 0;
    int32_t shift = 0;
    const double effective =
        static_cast<double>(quant.input_scale) * static_cast<double>(quant.weight_scale) /
        static_cast<double>(quant.output_scale);
    QuantizeMultiplier(effective, &multiplier, &shift);

    const cmsis_nn_per_tensor_quant_params quant_params = {
        .multiplier = multiplier,
        .shift = shift,
    };

    const cmsis_nn_fc_params fc_params = {
        .input_offset = -quant.input_zero_point,
        .filter_offset = -quant.weight_zero_point,
        .output_offset = -quant.output_zero_point,
        .activation = activation_relu(apply_relu),
    };

    const cmsis_nn_dims input_dims = {
        .n = static_cast<int32_t>(batch),
        .h = 1,
        .w = 1,
        .c = static_cast<int32_t>(in_features),
    };
    const cmsis_nn_dims filter_dims = {
        .n = static_cast<int32_t>(in_features),
        .h = 1,
        .w = 1,
        .c = static_cast<int32_t>(out_features),
    };
    const cmsis_nn_dims bias_dims = {
        .n = 1,
        .h = 1,
        .w = 1,
        .c = static_cast<int32_t>(out_features),
    };
    const cmsis_nn_dims output_dims = {
        .n = static_cast<int32_t>(batch),
        .h = 1,
        .w = 1,
        .c = static_cast<int32_t>(out_features),
    };

    const int32_t buf_size = arm_fully_connected_s8_get_buffer_size(&filter_dims);
    cmsis_nn_context ctx = {0};
    if (!BindContext(ctx, buf_size))
    {
        QuantTrace::RecordFcCmsisFail(QuantTrace::FcFail::BindContext, buf_size, workspace_bytes);
        return false;
    }

    if (!cmsis_status_ok(arm_fully_connected_s8(&ctx,
                                                &fc_params,
                                                &quant_params,
                                                &input_dims,
                                                input,
                                                &filter_dims,
                                                weights,
                                                &bias_dims,
                                                bias,
                                                &output_dims,
                                                output_int8)))
    {
        QuantTrace::RecordFcCmsisFail(QuantTrace::FcFail::CmsisStatus, buf_size, workspace_bytes);
        return false;
    }

    QuantTrace::RecordFcCmsisOk();
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

    const double max_input_scale = std::max(static_cast<double>(input1_scale),
                                            static_cast<double>(input2_scale));
    if (max_input_scale <= 0.0)
        return false;

    int32_t input1_mult = 0;
    int32_t input1_shift = 0;
    int32_t input2_mult = 0;
    int32_t input2_shift = 0;
    int32_t output_mult = 0;
    int32_t output_shift = 0;
    QuantizeMultiplier(static_cast<double>(input1_scale) / max_input_scale, &input1_mult, &input1_shift);
    QuantizeMultiplier(static_cast<double>(input2_scale) / max_input_scale, &input2_mult, &input2_shift);
    QuantizeMultiplier(max_input_scale / static_cast<double>(output_scale), &output_mult, &output_shift);

    constexpr int32_t kLeftShift = 20;
    const arm_cmsis_nn_status status = arm_elementwise_add_s8(
        input1,
        input2,
        -input1_zero_point,
        input1_mult,
        input1_shift,
        -input2_zero_point,
        input2_mult,
        input2_shift,
        kLeftShift,
        output,
        -output_zero_point,
        output_mult,
        output_shift,
        -128,
        127,
        static_cast<int32_t>(count));
    return cmsis_status_ok(status);
}

}  // namespace CmsisNnQuant

#else

namespace CmsisNnQuant
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

bool TryConv2dNhwcQuantPlan(const CmsisQuantPlan::Conv2DPlan& /*plan*/,
                            const int8_t* /*input*/,
                            const int8_t* /*weights*/,
                            const int32_t* /*bias*/,
                            int8_t* /*output*/)
{
    return false;
}

bool TryDepthwiseConv2dNhwcQuantPlan(const CmsisQuantPlan::DepthwiseConv2DPlan& /*plan*/,
                                     const int8_t* /*input*/,
                                     const int8_t* /*weights*/,
                                     const int32_t* /*bias*/,
                                     int8_t* /*output*/)
{
    return false;
}

bool TryMaxPool2dNhwcQuantPlan(const CmsisQuantPlan::Pool2DPlan& /*plan*/,
                               const int8_t* /*input*/,
                               int8_t* /*output*/)
{
    return false;
}

bool TryFullyConnectedQuantPlan(const CmsisQuantPlan::FcPlan& /*plan*/,
                                const int8_t* /*input*/,
                                const int8_t* /*weights*/,
                                const int32_t* /*bias*/,
                                int8_t* /*output_int8*/)
{
    return false;
}

bool TrySoftmaxS8Plan(const CmsisQuantPlan::SoftmaxPlan& /*plan*/,
                      const int8_t* /*input*/,
                      int8_t* /*output*/)
{
    return false;
}

bool TryConv2dNhwcQuant(const int8_t* /*input*/,
                        uint32_t /*in_h*/,
                        uint32_t /*in_w*/,
                        uint32_t /*in_c*/,
                        const int8_t* /*weights*/,
                        const int32_t* /*bias*/,
                        int /*kernel_size*/,
                        int /*stride*/,
                        int /*pad_h*/,
                        int /*pad_w*/,
                        int /*pad_h_end*/,
                        int /*pad_w_end*/,
                        int /*out_channels*/,
                        const NkFormat::MlpLayerQuantDesc& /*quant*/,
                        bool /*apply_relu*/,
                        int8_t* /*output*/)
{
    QuantTrace::RecordConv2dCmsisFail(QuantTrace::Conv2dFail::Disabled, 0, 0);
    return false;
}

bool TryDepthwiseConv2dNhwcQuant(const int8_t* /*input*/,
                                 uint32_t /*in_h*/,
                                 uint32_t /*in_w*/,
                                 uint32_t /*channels*/,
                                 const int8_t* /*weights*/,
                                 const int32_t* /*bias*/,
                                 int /*kernel_h*/,
                                 int /*kernel_w*/,
                                 int /*stride*/,
                                 int /*pad_h*/,
                                 int /*pad_w*/,
                                 int /*pad_h_end*/,
                                 int /*pad_w_end*/,
                                 const NkFormat::MlpLayerQuantDesc& /*quant*/,
                                 bool /*apply_relu*/,
                                 int8_t* /*output*/)
{
    QuantTrace::RecordConv2dCmsisFail(QuantTrace::Conv2dFail::Disabled, 0, 0);
    return false;
}

bool TryMaxPool2dNhwcQuant(const int8_t* /*input*/,
                           uint32_t /*in_h*/,
                           uint32_t /*in_w*/,
                           uint32_t /*in_c*/,
                           int /*pool_h*/,
                           int /*pool_w*/,
                           int /*stride*/,
                           int /*pad_h*/,
                           int /*pad_w*/,
                           int /*pad_h_end*/,
                           int /*pad_w_end*/,
                           int8_t* /*output*/)
{
    return false;
}

bool TryFullyConnectedQuant(const int8_t* /*input*/,
                            uint32_t /*batch*/,
                            uint32_t /*in_features*/,
                            const int8_t* /*weights*/,
                            const int32_t* /*bias*/,
                            uint32_t /*out_features*/,
                            const NkFormat::MlpLayerQuantDesc& /*quant*/,
                            bool /*apply_relu*/,
                            int8_t* /*output_int8*/)
{
    return false;
}

bool TryElementwiseAddS8(const int8_t* /*input1*/,
                         const int8_t* /*input2*/,
                         uint32_t /*count*/,
                         float /*input1_scale*/,
                         int32_t /*input1_zero_point*/,
                         float /*input2_scale*/,
                         int32_t /*input2_zero_point*/,
                         float /*output_scale*/,
                         int32_t /*output_zero_point*/,
                         int8_t* /*output*/)
{
    return false;
}

}  // namespace CmsisNnQuant

#endif
