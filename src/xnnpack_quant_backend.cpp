/*
 * XNNPACK int8 (qs8) adapters for netkit quantized conv / depthwise / pool / FC.
 *
 * Enabled on cpu/mpu when NETKIT_XNNPACK=1 (same flag as float32 LayerFast).
 * XNNPACK is BSD-3 — see third_party/XNNPACK/LICENSE.
 *
 * Persistent path: Create* at BuildRuntime (create+reshape+workspace), then
 * Try* does setup+run only when plan.xnn.ready. Ephemeral create/run/delete
 * remains as fallback when hoist is not ready.
 */
#include "xnnpack_quant.hpp"
#include "arena.hpp"
#include "cmsis_quant_plan.hpp"
#include "cnn.hpp"
#include "mlp.hpp"
#include "mobilenetv4_uib.hpp"
#include "netkit_config.h"
#include "nk_op_detail.hpp"
#include "quant_integer.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <mutex>
#include <new>
#include <vector>

#if defined(NETKIT_USE_XNNPACK) && NETKIT_USE_XNNPACK && NETKIT_XNNPACK_ALLOWED

#include <xnnpack.h>

namespace
{
    std::once_flag g_xnn_init_flag;
    bool g_xnn_ready = false;

    bool EnsureXnnInitialized()
    {
        std::call_once(g_xnn_init_flag, []() {
            g_xnn_ready = (xnn_initialize(/*allocator=*/nullptr) == xnn_status_success);
        });
        return g_xnn_ready;
    }

    struct ScopedOp
    {
        xnn_operator_t op = nullptr;
        ~ScopedOp()
        {
            if (op)
                xnn_delete_operator(op);
        }
        xnn_operator_t release()
        {
            xnn_operator_t tmp = op;
            op = nullptr;
            return tmp;
        }
    };

    void ActivationClampS8(QuantInteger::QuantClamp clamp,
                           float output_scale,
                           int32_t output_zero_point,
                           int8_t& out_min,
                           int8_t& out_max)
    {
        int32_t act_min = -128;
        int32_t act_max = 127;
        QuantInteger::QuantClampRange(clamp, output_scale, output_zero_point, &act_min, &act_max);
        out_min = static_cast<int8_t>(act_min);
        out_max = static_cast<int8_t>(act_max);
    }

    bool AllocWorkspace(CmsisQuantPlan::XnnpackOpHoist& hoist, size_t bytes, Arena* arena)
    {
        hoist.workspace = nullptr;
        hoist.workspace_bytes = bytes;
        hoist.workspace_heap = false;
        if (bytes == 0)
            return true;
        if (arena)
        {
            hoist.workspace =
                static_cast<uint8_t*>(arena->alloc(bytes, alignof(std::max_align_t)));
            if (hoist.workspace)
                return true;
        }
        hoist.workspace = new (std::nothrow) uint8_t[bytes];
        if (!hoist.workspace)
        {
            hoist.workspace_bytes = 0;
            return false;
        }
        hoist.workspace_heap = true;
        return true;
    }

    void ActivationClampFloat(QuantInteger::QuantClamp clamp,
                              float output_scale,
                              int32_t output_zero_point,
                              float& out_min,
                              float& out_max)
    {
        // Subgraph conv nodes clip in real (dequantized) space.
        out_min = static_cast<float>(-128 - output_zero_point) * output_scale;
        out_max = static_cast<float>(127 - output_zero_point) * output_scale;
        if (clamp == QuantInteger::QuantClamp::None)
            return;
        if (clamp == QuantInteger::QuantClamp::ReLU)
        {
            out_min = 0.0f;
            return;
        }
        // ReLU6
        out_min = 0.0f;
        out_max = std::min(out_max, 6.0f);
    }

    bool DefineActQint8(xnn_subgraph_t subgraph,
                        int32_t zero_point,
                        float scale,
                        size_t h,
                        size_t w,
                        size_t c,
                        uint32_t external_id,
                        uint32_t flags,
                        uint32_t* id_out)
    {
        const size_t dims[4] = {1, h, w, c};
        return xnn_define_quantized_tensor_value(subgraph,
                                                 xnn_datatype_qint8,
                                                 zero_point,
                                                 scale,
                                                 /*num_dims=*/4,
                                                 dims,
                                                 /*data=*/nullptr,
                                                 external_id,
                                                 flags,
                                                 id_out) == xnn_status_success;
    }

    bool DefineFilterAndBiasConv(xnn_subgraph_t subgraph,
                                 const CmsisQuantPlan::Conv2DPlan& plan,
                                 const int8_t* weights,
                                 const int32_t* bias,
                                 float** bias_scales_out,
                                 uint32_t* filter_id,
                                 uint32_t* bias_id)
    {
        if (!weights || !bias || plan.kernel_size <= 0 || plan.in_c <= 0 || plan.out_c <= 0)
            return false;

        const size_t kh = static_cast<size_t>(plan.kernel_size);
        const size_t kw = static_cast<size_t>(plan.kernel_size);
        const size_t ic = static_cast<size_t>(plan.in_c);
        const size_t oc = static_cast<size_t>(plan.out_c);
        const size_t filter_dims[4] = {oc, kh, kw, ic};
        const size_t bias_dims[1] = {oc};

        const bool per_channel =
            plan.weight_channel_scales != nullptr &&
            plan.num_weight_channel_scales == static_cast<uint32_t>(plan.out_c);

        if (per_channel)
        {
            if (xnn_define_channelwise_quantized_tensor_value(
                    subgraph,
                    xnn_datatype_qcint8,
                    plan.weight_channel_scales,
                    /*num_dims=*/4,
                    /*channel_dim=*/0,
                    filter_dims,
                    weights,
                    XNN_INVALID_VALUE_ID,
                    /*flags=*/0,
                    filter_id) != xnn_status_success)
                return false;

            // Bias scale[c] = input_scale * weight_scale[c]; must outlive runtime.
            float* bias_scales = new (std::nothrow) float[oc];
            if (!bias_scales)
                return false;
            for (size_t i = 0; i < oc; ++i)
                bias_scales[i] = plan.input_scale * plan.weight_channel_scales[i];
            *bias_scales_out = bias_scales;
            if (xnn_define_channelwise_quantized_tensor_value(
                    subgraph,
                    xnn_datatype_qcint32,
                    bias_scales,
                    /*num_dims=*/1,
                    /*channel_dim=*/0,
                    bias_dims,
                    bias,
                    XNN_INVALID_VALUE_ID,
                    /*flags=*/0,
                    bias_id) != xnn_status_success)
                return false;
        }
        else
        {
            if (plan.weight_scale <= 0.0f)
                return false;
            if (xnn_define_quantized_tensor_value(subgraph,
                                                 xnn_datatype_qint8,
                                                 /*zero_point=*/0,
                                                 plan.weight_scale,
                                                 /*num_dims=*/4,
                                                 filter_dims,
                                                 weights,
                                                 XNN_INVALID_VALUE_ID,
                                                 /*flags=*/0,
                                                 filter_id) != xnn_status_success)
                return false;
            const float bias_scale = plan.input_scale * plan.weight_scale;
            if (xnn_define_quantized_tensor_value(subgraph,
                                                 xnn_datatype_qint32,
                                                 /*zero_point=*/0,
                                                 bias_scale,
                                                 /*num_dims=*/1,
                                                 bias_dims,
                                                 bias,
                                                 XNN_INVALID_VALUE_ID,
                                                 /*flags=*/0,
                                                 bias_id) != xnn_status_success)
                return false;
        }
        return true;
    }

    bool DefineFilterAndBiasDw(xnn_subgraph_t subgraph,
                               const CmsisQuantPlan::DepthwiseConv2DPlan& plan,
                               const int8_t* weights_hwc,
                               const int32_t* bias,
                               float** bias_scales_out,
                               uint32_t* filter_id,
                               uint32_t* bias_id)
    {
        if (!weights_hwc || !bias || plan.kernel_h <= 0 || plan.kernel_w <= 0 ||
            plan.channels <= 0)
            return false;

        const size_t kh = static_cast<size_t>(plan.kernel_h);
        const size_t kw = static_cast<size_t>(plan.kernel_w);
        const size_t c = static_cast<size_t>(plan.channels);
        // Depthwise filter layout for subgraph: [1, Kh, Kw, C].
        const size_t filter_dims[4] = {1, kh, kw, c};
        const size_t bias_dims[1] = {c};

        const bool per_channel =
            plan.weight_channel_scales != nullptr &&
            plan.num_weight_channel_scales == static_cast<uint32_t>(plan.channels);

        if (per_channel)
        {
            if (xnn_define_channelwise_quantized_tensor_value(
                    subgraph,
                    xnn_datatype_qcint8,
                    plan.weight_channel_scales,
                    /*num_dims=*/4,
                    /*channel_dim=*/3,
                    filter_dims,
                    weights_hwc,
                    XNN_INVALID_VALUE_ID,
                    /*flags=*/0,
                    filter_id) != xnn_status_success)
                return false;

            float* bias_scales = new (std::nothrow) float[c];
            if (!bias_scales)
                return false;
            for (size_t i = 0; i < c; ++i)
                bias_scales[i] = plan.input_scale * plan.weight_channel_scales[i];
            *bias_scales_out = bias_scales;
            if (xnn_define_channelwise_quantized_tensor_value(
                    subgraph,
                    xnn_datatype_qcint32,
                    bias_scales,
                    /*num_dims=*/1,
                    /*channel_dim=*/0,
                    bias_dims,
                    bias,
                    XNN_INVALID_VALUE_ID,
                    /*flags=*/0,
                    bias_id) != xnn_status_success)
                return false;
        }
        else
        {
            if (plan.weight_scale <= 0.0f)
                return false;
            if (xnn_define_quantized_tensor_value(subgraph,
                                                 xnn_datatype_qint8,
                                                 /*zero_point=*/0,
                                                 plan.weight_scale,
                                                 /*num_dims=*/4,
                                                 filter_dims,
                                                 weights_hwc,
                                                 XNN_INVALID_VALUE_ID,
                                                 /*flags=*/0,
                                                 filter_id) != xnn_status_success)
                return false;
            const float bias_scale = plan.input_scale * plan.weight_scale;
            if (xnn_define_quantized_tensor_value(subgraph,
                                                 xnn_datatype_qint32,
                                                 /*zero_point=*/0,
                                                 bias_scale,
                                                 /*num_dims=*/1,
                                                 bias_dims,
                                                 bias,
                                                 XNN_INVALID_VALUE_ID,
                                                 /*flags=*/0,
                                                 bias_id) != xnn_status_success)
                return false;
        }
        return true;
    }

    bool DefineDwNode(xnn_subgraph_t subgraph,
                      const CmsisQuantPlan::DepthwiseConv2DPlan& plan,
                      uint32_t input_id,
                      uint32_t filter_id,
                      uint32_t bias_id,
                      uint32_t output_id)
    {
        float out_min = 0.0f;
        float out_max = 0.0f;
        ActivationClampFloat(plan.clamp, plan.output_scale, plan.output_offset, out_min, out_max);
        return xnn_define_depthwise_convolution_2d(
                   subgraph,
                   static_cast<uint32_t>(plan.pad_h),
                   static_cast<uint32_t>(plan.pad_w),
                   static_cast<uint32_t>(plan.pad_h),
                   static_cast<uint32_t>(plan.pad_w),
                   static_cast<uint32_t>(plan.kernel_h),
                   static_cast<uint32_t>(plan.kernel_w),
                   static_cast<uint32_t>(plan.stride),
                   static_cast<uint32_t>(plan.stride),
                   /*dilation_height=*/1,
                   /*dilation_width=*/1,
                   /*depth_multiplier=*/1,
                   static_cast<size_t>(plan.channels),
                   out_min,
                   out_max,
                   input_id,
                   filter_id,
                   bias_id,
                   output_id,
                   /*flags=*/0) == xnn_status_success;
    }

    bool DefineConvNode(xnn_subgraph_t subgraph,
                        const CmsisQuantPlan::Conv2DPlan& plan,
                        uint32_t input_id,
                        uint32_t filter_id,
                        uint32_t bias_id,
                        uint32_t output_id)
    {
        float out_min = 0.0f;
        float out_max = 0.0f;
        ActivationClampFloat(plan.clamp, plan.output_scale, plan.output_offset, out_min, out_max);
        return xnn_define_convolution_2d(subgraph,
                                         static_cast<uint32_t>(plan.pad_h),
                                         static_cast<uint32_t>(plan.pad_w),
                                         static_cast<uint32_t>(plan.pad_h),
                                         static_cast<uint32_t>(plan.pad_w),
                                         static_cast<uint32_t>(plan.kernel_size),
                                         static_cast<uint32_t>(plan.kernel_size),
                                         static_cast<uint32_t>(plan.stride),
                                         static_cast<uint32_t>(plan.stride),
                                         /*dilation_height=*/1,
                                         /*dilation_width=*/1,
                                         /*groups=*/1,
                                         static_cast<size_t>(plan.in_c),
                                         static_cast<size_t>(plan.out_c),
                                         out_min,
                                         out_max,
                                         input_id,
                                         filter_id,
                                         bias_id,
                                         output_id,
                                         /*flags=*/0) == xnn_status_success;
    }

    const char* XnnStatusName(enum xnn_status st)
    {
        switch (st)
        {
            case xnn_status_success:
                return "success";
            case xnn_status_uninitialized:
                return "uninitialized";
            case xnn_status_invalid_parameter:
                return "invalid_parameter";
            case xnn_status_invalid_state:
                return "invalid_state";
            case xnn_status_unsupported_parameter:
                return "unsupported_parameter";
            case xnn_status_unsupported_hardware:
                return "unsupported_hardware";
            case xnn_status_out_of_memory:
                return "out_of_memory";
            case xnn_status_reallocation_required:
                return "reallocation_required";
            case xnn_status_deprecated:
                return "deprecated";
            default:
                return "unknown";
        }
    }

    bool PushBiasScales(CmsisQuantPlan::Runtime& runtime, float* scales)
    {
        if (!scales)
            return true;
        float** grown = new (std::nothrow) float*[runtime.xnn_net_bias_scales_count + 1];
        if (!grown)
        {
            delete[] scales;
            return false;
        }
        for (uint32_t i = 0; i < runtime.xnn_net_bias_scales_count; ++i)
            grown[i] = runtime.xnn_net_bias_scales[i];
        grown[runtime.xnn_net_bias_scales_count] = scales;
        delete[] runtime.xnn_net_bias_scales;
        runtime.xnn_net_bias_scales = grown;
        ++runtime.xnn_net_bias_scales_count;
        return true;
    }

    bool DefineFilterAndBiasFc(xnn_subgraph_t subgraph,
                               const CmsisQuantPlan::FcPlan& plan,
                               const int8_t* weights,
                               const int32_t* bias,
                               float** bias_scales_out,
                               uint32_t* filter_id,
                               uint32_t* bias_id)
    {
        if (!weights || !bias || plan.in_features <= 0 || plan.out_features <= 0)
            return false;
        if (plan.filter_offset != 0)
            return false;

        const size_t ic = static_cast<size_t>(plan.in_features);
        const size_t oc = static_cast<size_t>(plan.out_features);
        const size_t filter_dims[2] = {oc, ic};
        const size_t bias_dims[1] = {oc};

        const bool per_channel =
            plan.weight_channel_scales != nullptr &&
            plan.num_weight_channel_scales == static_cast<uint32_t>(plan.out_features);

        if (per_channel)
        {
            if (xnn_define_channelwise_quantized_tensor_value(
                    subgraph,
                    xnn_datatype_qcint8,
                    plan.weight_channel_scales,
                    /*num_dims=*/2,
                    /*channel_dim=*/0,
                    filter_dims,
                    weights,
                    XNN_INVALID_VALUE_ID,
                    /*flags=*/0,
                    filter_id) != xnn_status_success)
                return false;

            float* bias_scales = new (std::nothrow) float[oc];
            if (!bias_scales)
                return false;
            for (size_t i = 0; i < oc; ++i)
                bias_scales[i] = plan.input_scale * plan.weight_channel_scales[i];
            *bias_scales_out = bias_scales;
            if (xnn_define_channelwise_quantized_tensor_value(
                    subgraph,
                    xnn_datatype_qcint32,
                    bias_scales,
                    /*num_dims=*/1,
                    /*channel_dim=*/0,
                    bias_dims,
                    bias,
                    XNN_INVALID_VALUE_ID,
                    /*flags=*/0,
                    bias_id) != xnn_status_success)
                return false;
        }
        else
        {
            if (plan.weight_scale <= 0.0f)
                return false;
            if (xnn_define_quantized_tensor_value(subgraph,
                                                 xnn_datatype_qint8,
                                                 /*zero_point=*/0,
                                                 plan.weight_scale,
                                                 /*num_dims=*/2,
                                                 filter_dims,
                                                 weights,
                                                 XNN_INVALID_VALUE_ID,
                                                 /*flags=*/0,
                                                 filter_id) != xnn_status_success)
                return false;
            const float bias_scale = plan.input_scale * plan.weight_scale;
            if (xnn_define_quantized_tensor_value(subgraph,
                                                 xnn_datatype_qint32,
                                                 /*zero_point=*/0,
                                                 bias_scale,
                                                 /*num_dims=*/1,
                                                 bias_dims,
                                                 bias,
                                                 XNN_INVALID_VALUE_ID,
                                                 /*flags=*/0,
                                                 bias_id) != xnn_status_success)
                return false;
        }
        return true;
    }

    // Append UIB body (start_dw? → expand → middle_dw? → proj [→ residual add]).
    // When external_output_id != XNN_INVALID_VALUE_ID, the block output uses that
    // external id (UIB-only subgraph). Otherwise an internal value is allocated.
    bool AppendUibBody(xnn_subgraph_t subgraph,
                       CmsisQuantPlan::MobilenetV4UibPlan& plan,
                       const MobileNetV4Uib& uib,
                       uint32_t input_id,
                       uint32_t external_output_id,
                       uint32_t* output_id_out,
                       float** start_dw_bias_scales,
                       float** expand_bias_scales,
                       float** middle_dw_bias_scales,
                       float** proj_bias_scales,
                       bool* includes_residual_out)
    {
        if (!plan.ready || !plan.expand.ready || !plan.proj.ready)
            return false;
        if (plan.has_start_dw && !plan.start_dw.ready)
            return false;
        if (plan.has_middle_dw && !plan.middle_dw.ready)
            return false;

        const uint32_t block_in_id = input_id;
        uint32_t cur_id = input_id;
        *includes_residual_out = false;

        if (plan.has_start_dw)
        {
            if (!plan.start_dw.weights_hwc)
                return false;
            uint32_t filter_id = 0;
            uint32_t bias_id = 0;
            uint32_t out_id = 0;
            if (!DefineFilterAndBiasDw(subgraph,
                                       plan.start_dw,
                                       plan.start_dw.weights_hwc,
                                       uib.start_dw_bias_q,
                                       start_dw_bias_scales,
                                       &filter_id,
                                       &bias_id))
                return false;
            if (!DefineActQint8(subgraph,
                                plan.start_dw.output_offset,
                                plan.start_dw.output_scale,
                                static_cast<size_t>(plan.start_dw.out_h),
                                static_cast<size_t>(plan.start_dw.out_w),
                                static_cast<size_t>(plan.start_dw.channels),
                                XNN_INVALID_VALUE_ID,
                                /*flags=*/0,
                                &out_id))
                return false;
            if (!DefineDwNode(subgraph, plan.start_dw, cur_id, filter_id, bias_id, out_id))
                return false;
            cur_id = out_id;
        }

        {
            uint32_t filter_id = 0;
            uint32_t bias_id = 0;
            uint32_t out_id = 0;
            if (!DefineFilterAndBiasConv(subgraph,
                                         plan.expand,
                                         uib.expand_weights_q,
                                         uib.expand_bias_q,
                                         expand_bias_scales,
                                         &filter_id,
                                         &bias_id))
                return false;
            if (!DefineActQint8(subgraph,
                                plan.expand.output_offset,
                                plan.expand.output_scale,
                                static_cast<size_t>(plan.expand.out_h),
                                static_cast<size_t>(plan.expand.out_w),
                                static_cast<size_t>(plan.expand.out_c),
                                XNN_INVALID_VALUE_ID,
                                /*flags=*/0,
                                &out_id))
                return false;
            if (!DefineConvNode(subgraph, plan.expand, cur_id, filter_id, bias_id, out_id))
                return false;
            cur_id = out_id;
        }

        if (plan.has_middle_dw)
        {
            if (!plan.middle_dw.weights_hwc)
                return false;
            uint32_t filter_id = 0;
            uint32_t bias_id = 0;
            uint32_t out_id = 0;
            if (!DefineFilterAndBiasDw(subgraph,
                                       plan.middle_dw,
                                       plan.middle_dw.weights_hwc,
                                       uib.middle_dw_bias_q,
                                       middle_dw_bias_scales,
                                       &filter_id,
                                       &bias_id))
                return false;
            if (!DefineActQint8(subgraph,
                                plan.middle_dw.output_offset,
                                plan.middle_dw.output_scale,
                                static_cast<size_t>(plan.middle_dw.out_h),
                                static_cast<size_t>(plan.middle_dw.out_w),
                                static_cast<size_t>(plan.middle_dw.channels),
                                XNN_INVALID_VALUE_ID,
                                /*flags=*/0,
                                &out_id))
                return false;
            if (!DefineDwNode(subgraph, plan.middle_dw, cur_id, filter_id, bias_id, out_id))
                return false;
            cur_id = out_id;
        }

        {
            uint32_t filter_id = 0;
            uint32_t bias_id = 0;
            uint32_t proj_out_id = 0;
            if (!DefineFilterAndBiasConv(subgraph,
                                         plan.proj,
                                         uib.proj_weights_q,
                                         uib.proj_bias_q,
                                         proj_bias_scales,
                                         &filter_id,
                                         &bias_id))
                return false;

            if (plan.has_residual)
            {
                // Proj writes an internal tensor; residual add produces the block output.
                if (!DefineActQint8(subgraph,
                                    plan.proj.output_offset,
                                    plan.proj.output_scale,
                                    static_cast<size_t>(plan.out_h),
                                    static_cast<size_t>(plan.out_w),
                                    static_cast<size_t>(plan.out_c),
                                    XNN_INVALID_VALUE_ID,
                                    /*flags=*/0,
                                    &proj_out_id))
                    return false;
                if (!DefineConvNode(subgraph, plan.proj, cur_id, filter_id, bias_id, proj_out_id))
                    return false;

                uint32_t out_id = 0;
                const uint32_t out_flags =
                    (external_output_id != XNN_INVALID_VALUE_ID) ? XNN_VALUE_FLAG_EXTERNAL_OUTPUT
                                                                : 0;
                if (!DefineActQint8(subgraph,
                                    plan.proj.output_offset,
                                    plan.proj.output_scale,
                                    static_cast<size_t>(plan.out_h),
                                    static_cast<size_t>(plan.out_w),
                                    static_cast<size_t>(plan.out_c),
                                    external_output_id,
                                    out_flags,
                                    &out_id))
                    return false;
                if (xnn_define_binary(subgraph,
                                      xnn_binary_add,
                                      /*params=*/nullptr,
                                      proj_out_id,
                                      block_in_id,
                                      out_id,
                                      /*flags=*/0) != xnn_status_success)
                    return false;
                *output_id_out = out_id;
                *includes_residual_out = true;
            }
            else
            {
                const uint32_t out_flags =
                    (external_output_id != XNN_INVALID_VALUE_ID) ? XNN_VALUE_FLAG_EXTERNAL_OUTPUT
                                                                : 0;
                if (!DefineActQint8(subgraph,
                                    plan.proj.output_offset,
                                    plan.proj.output_scale,
                                    static_cast<size_t>(plan.out_h),
                                    static_cast<size_t>(plan.out_w),
                                    static_cast<size_t>(plan.out_c),
                                    external_output_id,
                                    out_flags,
                                    &proj_out_id))
                    return false;
                if (!DefineConvNode(subgraph, plan.proj, cur_id, filter_id, bias_id, proj_out_id))
                    return false;
                *output_id_out = proj_out_id;
            }
        }
        return true;
    }

    bool ReshapeQs8Conv(CmsisQuantPlan::XnnpackOpHoist& hoist, Arena* arena)
    {
        if (hoist.ready)
            return true;
        if (!hoist.op)
            return false;

        auto* op = static_cast<xnn_operator_t>(hoist.op);
        size_t workspace_size = 0;
        size_t out_h = 0;
        size_t out_w = 0;

        if (hoist.per_channel)
        {
            if (xnn_reshape_convolution2d_nhwc_qs8_qc8w(op,
                                                        /*batch_size=*/1,
                                                        hoist.reshape_in_h,
                                                        hoist.reshape_in_w,
                                                        &workspace_size,
                                                        &out_h,
                                                        &out_w,
                                                        /*threadpool=*/nullptr) !=
                xnn_status_success)
                return false;
        }
        else
        {
            if (xnn_reshape_convolution2d_nhwc_qs8(op,
                                                   /*batch_size=*/1,
                                                   hoist.reshape_in_h,
                                                   hoist.reshape_in_w,
                                                   &workspace_size,
                                                   &out_h,
                                                   &out_w,
                                                   /*threadpool=*/nullptr) != xnn_status_success)
                return false;
        }

        if (out_h != hoist.reshape_out_h || out_w != hoist.reshape_out_w)
            return false;
        if (!AllocWorkspace(hoist, workspace_size, arena))
            return false;

        hoist.ready = true;
        return true;
    }

    bool CreateQs8Conv(CmsisQuantPlan::XnnpackOpHoist& hoist,
                       uint32_t pad_h,
                       uint32_t pad_w,
                       uint32_t kernel_h,
                       uint32_t kernel_w,
                       uint32_t stride,
                       uint32_t groups,
                       size_t group_input_channels,
                       size_t group_output_channels,
                       size_t input_channel_stride,
                       size_t output_channel_stride,
                       int8_t input_zp,
                       float input_scale,
                       float kernel_scale,
                       const float* kernel_scales,
                       size_t kernel_scale_count,
                       const int8_t* kernel,
                       const int32_t* bias,
                       int8_t output_zp,
                       float output_scale,
                       int8_t output_min,
                       int8_t output_max,
                       uint32_t flags,
                       size_t in_h,
                       size_t in_w,
                       size_t expect_out_h,
                       size_t expect_out_w,
                       Arena* arena,
                       xnn_weights_cache_t weights_cache)
    {
        if (hoist.ready)
            return true;
        if (!EnsureXnnInitialized() || !kernel || !bias)
            return false;
        if (input_scale <= 0.0f || output_scale <= 0.0f)
            return false;

        const size_t expected_scales = groups * group_output_channels;
        const bool per_channel =
            kernel_scales != nullptr && kernel_scale_count == expected_scales;

        ScopedOp scoped;

        if (per_channel)
        {
            if (xnn_create_convolution2d_nhwc_qs8_qc8w(
                    pad_h,
                    pad_w,
                    pad_h,
                    pad_w,
                    kernel_h,
                    kernel_w,
                    stride,
                    stride,
                    /*dilation_height=*/1,
                    /*dilation_width=*/1,
                    groups,
                    group_input_channels,
                    group_output_channels,
                    input_channel_stride,
                    output_channel_stride,
                    input_zp,
                    input_scale,
                    kernel_scales,
                    kernel,
                    bias,
                    output_zp,
                    output_scale,
                    output_min,
                    output_max,
                    flags,
                    weights_cache,
                    &scoped.op) != xnn_status_success)
                return false;
        }
        else
        {
            if (kernel_scale <= 0.0f)
                return false;
            if (xnn_create_convolution2d_nhwc_qs8(
                    pad_h,
                    pad_w,
                    pad_h,
                    pad_w,
                    kernel_h,
                    kernel_w,
                    stride,
                    stride,
                    /*dilation_height=*/1,
                    /*dilation_width=*/1,
                    groups,
                    group_input_channels,
                    group_output_channels,
                    input_channel_stride,
                    output_channel_stride,
                    input_zp,
                    input_scale,
                    kernel_scale,
                    kernel,
                    bias,
                    output_zp,
                    output_scale,
                    output_min,
                    output_max,
                    flags,
                    weights_cache,
                    &scoped.op) != xnn_status_success)
                return false;
        }

        hoist.op = scoped.release();
        hoist.per_channel = per_channel;
        hoist.reshape_in_h = in_h;
        hoist.reshape_in_w = in_w;
        hoist.reshape_out_h = expect_out_h;
        hoist.reshape_out_w = expect_out_w;

        // With a shared weights_cache, reshape must wait until after
        // xnn_finalize_weights_cache (buffer may relocate during create growth).
        if (weights_cache)
        {
            hoist.ready = false;
            return true;
        }

        return ReshapeQs8Conv(hoist, arena);
    }

    bool RunQs8ConvCached(CmsisQuantPlan::XnnpackOpHoist& hoist,
                          const int8_t* input,
                          int8_t* output)
    {
        if (!hoist.ready || !hoist.op || !input || !output)
            return false;
        auto* op = static_cast<xnn_operator_t>(hoist.op);
        void* workspace = hoist.workspace_bytes > 0 ? hoist.workspace : nullptr;
        if (hoist.per_channel)
        {
            if (xnn_setup_convolution2d_nhwc_qs8_qc8w(op, workspace, input, output) !=
                xnn_status_success)
                return false;
        }
        else
        {
            if (xnn_setup_convolution2d_nhwc_qs8(op, workspace, input, output) !=
                xnn_status_success)
                return false;
        }
        return xnn_run_operator(op, /*threadpool=*/nullptr) == xnn_status_success;
    }

    bool RunQs8Conv(uint32_t pad_h,
                    uint32_t pad_w,
                    uint32_t kernel_h,
                    uint32_t kernel_w,
                    uint32_t stride,
                    uint32_t groups,
                    size_t group_input_channels,
                    size_t group_output_channels,
                    size_t input_channel_stride,
                    size_t output_channel_stride,
                    int8_t input_zp,
                    float input_scale,
                    float kernel_scale,
                    const float* kernel_scales,
                    size_t kernel_scale_count,
                    const int8_t* kernel,
                    const int32_t* bias,
                    int8_t output_zp,
                    float output_scale,
                    int8_t output_min,
                    int8_t output_max,
                    uint32_t flags,
                    size_t in_h,
                    size_t in_w,
                    size_t expect_out_h,
                    size_t expect_out_w,
                    const int8_t* input,
                    int8_t* output)
    {
        if (!EnsureXnnInitialized() || !kernel || !bias || !input || !output)
            return false;
        if (input_scale <= 0.0f || output_scale <= 0.0f)
            return false;

        const size_t expected_scales = groups * group_output_channels;
        const bool per_channel =
            kernel_scales != nullptr && kernel_scale_count == expected_scales;

        ScopedOp scoped;
        if (per_channel)
        {
            if (xnn_create_convolution2d_nhwc_qs8_qc8w(
                    pad_h,
                    pad_w,
                    pad_h,
                    pad_w,
                    kernel_h,
                    kernel_w,
                    stride,
                    stride,
                    /*dilation_height=*/1,
                    /*dilation_width=*/1,
                    groups,
                    group_input_channels,
                    group_output_channels,
                    input_channel_stride,
                    output_channel_stride,
                    input_zp,
                    input_scale,
                    kernel_scales,
                    kernel,
                    bias,
                    output_zp,
                    output_scale,
                    output_min,
                    output_max,
                    flags,
                    /*weights_cache=*/nullptr,
                    &scoped.op) != xnn_status_success)
                return false;

            size_t workspace_size = 0;
            size_t out_h = 0;
            size_t out_w = 0;
            if (xnn_reshape_convolution2d_nhwc_qs8_qc8w(scoped.op,
                                                        /*batch_size=*/1,
                                                        in_h,
                                                        in_w,
                                                        &workspace_size,
                                                        &out_h,
                                                        &out_w,
                                                        /*threadpool=*/nullptr) !=
                xnn_status_success)
                return false;
            if (out_h != expect_out_h || out_w != expect_out_w)
                return false;

            std::vector<uint8_t> workspace_storage(workspace_size);
            void* workspace = workspace_size > 0 ? workspace_storage.data() : nullptr;
            if (xnn_setup_convolution2d_nhwc_qs8_qc8w(scoped.op, workspace, input, output) !=
                xnn_status_success)
                return false;
            return xnn_run_operator(scoped.op, /*threadpool=*/nullptr) == xnn_status_success;
        }

        if (kernel_scale <= 0.0f)
            return false;
        if (xnn_create_convolution2d_nhwc_qs8(
                pad_h,
                pad_w,
                pad_h,
                pad_w,
                kernel_h,
                kernel_w,
                stride,
                stride,
                /*dilation_height=*/1,
                /*dilation_width=*/1,
                groups,
                group_input_channels,
                group_output_channels,
                input_channel_stride,
                output_channel_stride,
                input_zp,
                input_scale,
                kernel_scale,
                kernel,
                bias,
                output_zp,
                output_scale,
                output_min,
                output_max,
                flags,
                /*weights_cache=*/nullptr,
                &scoped.op) != xnn_status_success)
            return false;

        size_t workspace_size = 0;
        size_t out_h = 0;
        size_t out_w = 0;
        if (xnn_reshape_convolution2d_nhwc_qs8(scoped.op,
                                               /*batch_size=*/1,
                                               in_h,
                                               in_w,
                                               &workspace_size,
                                               &out_h,
                                               &out_w,
                                               /*threadpool=*/nullptr) != xnn_status_success)
            return false;
        if (out_h != expect_out_h || out_w != expect_out_w)
            return false;

        std::vector<uint8_t> workspace_storage(workspace_size);
        void* workspace = workspace_size > 0 ? workspace_storage.data() : nullptr;
        if (xnn_setup_convolution2d_nhwc_qs8(scoped.op, workspace, input, output) !=
            xnn_status_success)
            return false;
        return xnn_run_operator(scoped.op, /*threadpool=*/nullptr) == xnn_status_success;
    }
}  // namespace

namespace XnnpackQuant
{

void DestroyXnnpackOp(CmsisQuantPlan::XnnpackOpHoist& hoist)
{
    if (hoist.op)
    {
        xnn_delete_operator(static_cast<xnn_operator_t>(hoist.op));
        hoist.op = nullptr;
    }
    if (hoist.workspace_heap && hoist.workspace)
        delete[] hoist.workspace;
    hoist.workspace = nullptr;
    hoist.workspace_bytes = 0;
    hoist.workspace_heap = false;
    hoist.per_channel = false;
    hoist.ready = false;
    hoist.reshape_in_h = 0;
    hoist.reshape_in_w = 0;
    hoist.reshape_out_h = 0;
    hoist.reshape_out_w = 0;
}

bool FinishConvAfterWeightsCache(CmsisQuantPlan::XnnpackOpHoist& hoist, Arena* arena)
{
    return ReshapeQs8Conv(hoist, arena);
}

bool FinishFullyConnectedAfterWeightsCache(CmsisQuantPlan::XnnpackOpHoist& hoist)
{
    if (hoist.ready)
        return true;
    if (!hoist.op)
        return false;
    auto* op = static_cast<xnn_operator_t>(hoist.op);
    if (hoist.per_channel)
    {
        if (xnn_reshape_fully_connected_nc_qs8_qc8w(op, /*batch_size=*/1, nullptr) !=
            xnn_status_success)
            return false;
    }
    else
    {
        if (xnn_reshape_fully_connected_nc_qs8(op, /*batch_size=*/1, nullptr) !=
            xnn_status_success)
            return false;
    }
    hoist.ready = true;
    return true;
}

bool CreateConv2dNhwcQuantPlan(CmsisQuantPlan::Conv2DPlan& plan,
                               const int8_t* weights,
                               const int32_t* bias,
                               Arena* arena,
                               void* weights_cache)
{
    if (!plan.ready || !weights || !bias)
        return false;
    if (plan.xnn.ready)
        return true;

    int8_t out_min = -128;
    int8_t out_max = 127;
    ActivationClampS8(plan.clamp, plan.output_scale, plan.output_offset, out_min, out_max);

    return CreateQs8Conv(plan.xnn,
                         static_cast<uint32_t>(plan.pad_h),
                         static_cast<uint32_t>(plan.pad_w),
                         static_cast<uint32_t>(plan.kernel_size),
                         static_cast<uint32_t>(plan.kernel_size),
                         static_cast<uint32_t>(plan.stride),
                         /*groups=*/1,
                         static_cast<size_t>(plan.in_c),
                         static_cast<size_t>(plan.out_c),
                         static_cast<size_t>(plan.in_c),
                         static_cast<size_t>(plan.out_c),
                         static_cast<int8_t>(-plan.input_offset),
                         plan.input_scale,
                         plan.weight_scale,
                         plan.weight_channel_scales,
                         plan.num_weight_channel_scales,
                         weights,
                         bias,
                         static_cast<int8_t>(plan.output_offset),
                         plan.output_scale,
                         out_min,
                         out_max,
                         /*flags=*/0,
                         static_cast<size_t>(plan.in_h),
                         static_cast<size_t>(plan.in_w),
                         static_cast<size_t>(plan.out_h),
                         static_cast<size_t>(plan.out_w),
                         arena,
                         static_cast<xnn_weights_cache_t>(weights_cache));
}

bool CreateDepthwiseConv2dNhwcQuantPlan(CmsisQuantPlan::DepthwiseConv2DPlan& plan,
                                        const int8_t* weights_chw,
                                        const int32_t* bias,
                                        Arena* arena,
                                        void* weights_cache)
{
    if (!plan.ready || !weights_chw || !bias)
        return false;
    if (plan.xnn.ready)
        return true;

    const int8_t* kernel = plan.weights_hwc ? plan.weights_hwc : weights_chw;
    const uint32_t flags = plan.weights_hwc ? XNN_FLAG_DEPTHWISE_CONVOLUTION : 0;

    int8_t out_min = -128;
    int8_t out_max = 127;
    ActivationClampS8(plan.clamp, plan.output_scale, plan.output_offset, out_min, out_max);

    const size_t channels = static_cast<size_t>(plan.channels);
    return CreateQs8Conv(plan.xnn,
                         static_cast<uint32_t>(plan.pad_h),
                         static_cast<uint32_t>(plan.pad_w),
                         static_cast<uint32_t>(plan.kernel_h),
                         static_cast<uint32_t>(plan.kernel_w),
                         static_cast<uint32_t>(plan.stride),
                         static_cast<uint32_t>(plan.channels),
                         /*group_input_channels=*/1,
                         /*group_output_channels=*/1,
                         channels,
                         channels,
                         static_cast<int8_t>(-plan.input_offset),
                         plan.input_scale,
                         plan.weight_scale,
                         plan.weight_channel_scales,
                         plan.num_weight_channel_scales,
                         kernel,
                         bias,
                         static_cast<int8_t>(plan.output_offset),
                         plan.output_scale,
                         out_min,
                         out_max,
                         flags,
                         static_cast<size_t>(plan.in_h),
                         static_cast<size_t>(plan.in_w),
                         static_cast<size_t>(plan.out_h),
                         static_cast<size_t>(plan.out_w),
                         arena,
                         static_cast<xnn_weights_cache_t>(weights_cache));
}

bool CreateMaxPool2dNhwcQuantPlan(CmsisQuantPlan::Pool2DPlan& plan, Arena* arena)
{
    (void)arena;
    if (!plan.ready || plan.pool_h != plan.pool_w)
        return false;
    if (plan.xnn.ready)
        return true;
    if (!EnsureXnnInitialized())
        return false;

    int8_t out_min = -128;
    int8_t out_max = 127;
    ActivationClampS8(plan.clamp, plan.output_scale, plan.output_zero_point, out_min, out_max);

    ScopedOp scoped;
    if (xnn_create_max_pooling2d_nhwc_s8(
            static_cast<uint32_t>(plan.pad_h),
            static_cast<uint32_t>(plan.pad_w),
            static_cast<uint32_t>(plan.pad_h),
            static_cast<uint32_t>(plan.pad_w),
            static_cast<uint32_t>(plan.pool_h),
            static_cast<uint32_t>(plan.pool_w),
            static_cast<uint32_t>(plan.stride),
            static_cast<uint32_t>(plan.stride),
            /*dilation_height=*/1,
            /*dilation_width=*/1,
            out_min,
            out_max,
            /*flags=*/0,
            &scoped.op) != xnn_status_success)
        return false;

    size_t out_h = 0;
    size_t out_w = 0;
    const size_t channels = static_cast<size_t>(plan.in_c);
    if (xnn_reshape_max_pooling2d_nhwc_s8(scoped.op,
                                          1,
                                          static_cast<size_t>(plan.in_h),
                                          static_cast<size_t>(plan.in_w),
                                          channels,
                                          channels,
                                          channels,
                                          &out_h,
                                          &out_w,
                                          nullptr) != xnn_status_success)
        return false;
    if (out_h != static_cast<size_t>(plan.out_h) || out_w != static_cast<size_t>(plan.out_w))
        return false;

    plan.xnn.op = scoped.release();
    plan.xnn.per_channel = false;
    plan.xnn.ready = true;
    return true;
}

bool CreateFullyConnectedQuantPlan(CmsisQuantPlan::FcPlan& plan,
                                   const int8_t* weights,
                                   const int32_t* bias,
                                   Arena* arena,
                                   void* weights_cache)
{
    (void)arena;
    if (!plan.ready || !weights || !bias)
        return false;
    if (plan.xnn.ready)
        return true;
    if (plan.input_scale <= 0.0f || plan.output_scale <= 0.0f)
        return false;
    if (plan.filter_offset != 0)
        return false;
    if (!EnsureXnnInitialized())
        return false;

    int8_t out_min = -128;
    int8_t out_max = 127;
    ActivationClampS8(plan.clamp, plan.output_scale, plan.output_offset, out_min, out_max);

    const size_t in_channels = static_cast<size_t>(plan.in_features);
    const size_t out_channels = static_cast<size_t>(plan.out_features);
    const bool per_channel =
        plan.weight_channel_scales != nullptr &&
        plan.num_weight_channel_scales == static_cast<uint32_t>(plan.out_features);
    auto* cache = static_cast<xnn_weights_cache_t>(weights_cache);

    ScopedOp scoped;
    if (per_channel)
    {
        if (xnn_create_fully_connected_nc_qs8_qc8w(
                in_channels,
                out_channels,
                /*input_stride=*/in_channels,
                /*output_stride=*/out_channels,
                /*input_zero_point=*/static_cast<int8_t>(-plan.input_offset),
                plan.input_scale,
                plan.weight_channel_scales,
                weights,
                bias,
                /*output_zero_point=*/static_cast<int8_t>(plan.output_offset),
                plan.output_scale,
                out_min,
                out_max,
                /*flags=*/0,
                cache,
                &scoped.op) != xnn_status_success)
            return false;
    }
    else
    {
        if (plan.weight_scale <= 0.0f)
            return false;
        if (xnn_create_fully_connected_nc_qs8(
                in_channels,
                out_channels,
                /*input_stride=*/in_channels,
                /*output_stride=*/out_channels,
                /*input_zero_point=*/static_cast<int8_t>(-plan.input_offset),
                plan.input_scale,
                plan.weight_scale,
                weights,
                bias,
                /*output_zero_point=*/static_cast<int8_t>(plan.output_offset),
                plan.output_scale,
                out_min,
                out_max,
                /*flags=*/0,
                cache,
                &scoped.op) != xnn_status_success)
            return false;
    }

    plan.xnn.op = scoped.release();
    plan.xnn.per_channel = per_channel;

    if (cache)
    {
        plan.xnn.ready = false;
        return true;
    }

    return FinishFullyConnectedAfterWeightsCache(plan.xnn);
}

void DestroyUibSubgraph(CmsisQuantPlan::MobilenetV4UibPlan& plan)
{
    if (plan.xnn_runtime)
    {
        (void)xnn_delete_runtime(static_cast<xnn_runtime_t>(plan.xnn_runtime));
        plan.xnn_runtime = nullptr;
    }
    if (plan.xnn_subgraph)
    {
        (void)xnn_delete_subgraph(static_cast<xnn_subgraph_t>(plan.xnn_subgraph));
        plan.xnn_subgraph = nullptr;
    }
    delete[] plan.xnn_start_dw_bias_scales;
    plan.xnn_start_dw_bias_scales = nullptr;
    delete[] plan.xnn_expand_bias_scales;
    plan.xnn_expand_bias_scales = nullptr;
    delete[] plan.xnn_middle_dw_bias_scales;
    plan.xnn_middle_dw_bias_scales = nullptr;
    delete[] plan.xnn_proj_bias_scales;
    plan.xnn_proj_bias_scales = nullptr;
    plan.xnn_subgraph_ready = false;
    plan.xnn_subgraph_includes_residual = false;
    plan.xnn_bound_input = nullptr;
    plan.xnn_bound_output = nullptr;
}

bool CreateUibSubgraph(CmsisQuantPlan::MobilenetV4UibPlan& plan,
                       const MobileNetV4Uib& uib,
                       void* weights_cache,
                       void* workspace)
{
    DestroyUibSubgraph(plan);
    if (!plan.ready || !EnsureXnnInitialized())
        return false;
    if (!plan.expand.ready || !plan.proj.ready)
        return false;
    if (plan.has_start_dw && !plan.start_dw.ready)
        return false;
    if (plan.has_middle_dw && !plan.middle_dw.ready)
        return false;

    plan.xnn_ext_input_id = 0;
    plan.xnn_ext_output_id = 1;
    plan.xnn_subgraph_includes_residual = false;

    xnn_subgraph_t subgraph = nullptr;
    if (xnn_create_subgraph(/*external_value_ids=*/2, /*flags=*/0, &subgraph) !=
        xnn_status_success)
        return false;

    const int32_t block_in_zp =
        plan.has_start_dw ? -plan.start_dw.input_offset : -plan.expand.input_offset;
    const float block_in_scale =
        plan.has_start_dw ? plan.start_dw.input_scale : plan.expand.input_scale;

    uint32_t cur_id = 0;
    if (!DefineActQint8(subgraph,
                        block_in_zp,
                        block_in_scale,
                        static_cast<size_t>(plan.in_h),
                        static_cast<size_t>(plan.in_w),
                        static_cast<size_t>(plan.in_c),
                        plan.xnn_ext_input_id,
                        XNN_VALUE_FLAG_EXTERNAL_INPUT,
                        &cur_id))
    {
        (void)xnn_delete_subgraph(subgraph);
        return false;
    }

    auto fail = [&]() {
        (void)xnn_delete_subgraph(subgraph);
        DestroyUibSubgraph(plan);
        return false;
    };

    uint32_t out_id = 0;
    bool includes_residual = false;
    if (!AppendUibBody(subgraph,
                       plan,
                       uib,
                       cur_id,
                       plan.xnn_ext_output_id,
                       &out_id,
                       &plan.xnn_start_dw_bias_scales,
                       &plan.xnn_expand_bias_scales,
                       &plan.xnn_middle_dw_bias_scales,
                       &plan.xnn_proj_bias_scales,
                       &includes_residual))
        return fail();
    plan.xnn_subgraph_includes_residual = includes_residual;
    (void)out_id;

    xnn_runtime_t runtime = nullptr;
    // Match TF Lite XNNPACK delegate: always XNN_FLAG_DONT_SPIN_WORKERS; shared
    // workspace when provided; null threadpool when num_threads <= 1.
    const xnn_status create_st =
        xnn_create_runtime_v4(subgraph,
                              static_cast<xnn_weights_cache_t>(weights_cache),
                              static_cast<xnn_workspace_t>(workspace),
                              /*threadpool=*/nullptr,
                              XNN_FLAG_DONT_SPIN_WORKERS,
                              &runtime);
    // Runtime owns the execution plan; subgraph can be deleted either way.
    (void)xnn_delete_subgraph(subgraph);
    subgraph = nullptr;
    if (create_st != xnn_status_success || !runtime)
    {
        DestroyUibSubgraph(plan);
        return false;
    }

    plan.xnn_runtime = runtime;
    plan.xnn_subgraph = nullptr;

    if (weights_cache)
    {
        // Reshape after weights_cache finalize (absolute packed pointers).
        plan.xnn_subgraph_ready = false;
        return true;
    }

    return FinishUibAfterWeightsCache(plan);
}

bool FinishUibAfterWeightsCache(CmsisQuantPlan::MobilenetV4UibPlan& plan)
{
    if (plan.xnn_subgraph_ready)
        return true;
    if (!plan.xnn_runtime)
        return false;

    auto* runtime = static_cast<xnn_runtime_t>(plan.xnn_runtime);
    const size_t in_dims[4] = {1,
                               static_cast<size_t>(plan.in_h),
                               static_cast<size_t>(plan.in_w),
                               static_cast<size_t>(plan.in_c)};
    if (xnn_reshape_external_value(runtime, plan.xnn_ext_input_id, 4, in_dims) !=
        xnn_status_success)
    {
        DestroyUibSubgraph(plan);
        return false;
    }
    if (xnn_reshape_runtime(runtime) != xnn_status_success)
    {
        DestroyUibSubgraph(plan);
        return false;
    }
    plan.xnn_subgraph_ready = true;
    return true;
}

bool InvokeUibSubgraph(CmsisQuantPlan::MobilenetV4UibPlan& plan,
                       const int8_t* input,
                       int8_t* output)
{
    if (!plan.xnn_subgraph_ready || !plan.xnn_runtime || !input || !output)
        return false;

    auto* runtime = static_cast<xnn_runtime_t>(plan.xnn_runtime);
    if (input != plan.xnn_bound_input || output != plan.xnn_bound_output)
    {
        const xnn_external_value externals[2] = {
            {plan.xnn_ext_input_id, const_cast<int8_t*>(input)},
            {plan.xnn_ext_output_id, output},
        };
        if (xnn_setup_runtime_v2(runtime, 2, externals) != xnn_status_success)
            return false;
        plan.xnn_bound_input = input;
        plan.xnn_bound_output = output;
    }
    return xnn_invoke_runtime(runtime) == xnn_status_success;
}

void DestroyNetworkSubgraph(CmsisQuantPlan::Runtime& runtime)
{
    if (runtime.xnn_network_runtime)
    {
        (void)xnn_delete_runtime(static_cast<xnn_runtime_t>(runtime.xnn_network_runtime));
        runtime.xnn_network_runtime = nullptr;
    }
    if (runtime.xnn_net_bias_scales)
    {
        for (uint32_t i = 0; i < runtime.xnn_net_bias_scales_count; ++i)
            delete[] runtime.xnn_net_bias_scales[i];
        delete[] runtime.xnn_net_bias_scales;
        runtime.xnn_net_bias_scales = nullptr;
        runtime.xnn_net_bias_scales_count = 0;
    }
    runtime.xnn_network_ready = false;
    runtime.xnn_net_ext_in = 0;
    runtime.xnn_net_ext_out = 1;
    runtime.bound_input = nullptr;
    runtime.bound_output = nullptr;
}

bool CreateNetworkSubgraph(CmsisQuantPlan::Runtime& runtime,
                           CNNNetwork& network,
                           void* weights_cache)
{
    DestroyNetworkSubgraph(runtime);
    if (!runtime.layers || runtime.num_layers == 0 || !EnsureXnnInitialized())
        return false;

    runtime.xnn_net_ext_in = 0;
    runtime.xnn_net_ext_out = 1;

    xnn_subgraph_t subgraph = nullptr;
    if (xnn_create_subgraph(/*external_value_ids=*/2, /*flags=*/0, &subgraph) !=
        xnn_status_success)
    {
        std::fprintf(stderr, "CreateNetworkSubgraph: xnn_create_subgraph failed\n");
        return false;
    }

    auto fail = [&](const char* reason) {
        std::fprintf(stderr, "CreateNetworkSubgraph: %s\n", reason);
        (void)xnn_delete_subgraph(subgraph);
        DestroyNetworkSubgraph(runtime);
        return false;
    };

    const CmsisQuantPlan::LayerPlan& first = runtime.layers[0];
    int32_t in_h = 0;
    int32_t in_w = 0;
    int32_t in_c = 0;
    int32_t in_zp = 0;
    float in_scale = 0.0f;
    if (first.kind == CmsisQuantPlan::LayerKind::Conv2D && first.conv.ready)
    {
        in_h = first.conv.in_h;
        in_w = first.conv.in_w;
        in_c = first.conv.in_c;
        in_zp = -first.conv.input_offset;
        in_scale = first.conv.input_scale;
    }
    else if (first.kind == CmsisQuantPlan::LayerKind::MobilenetV4Uib && first.uib.ready)
    {
        in_h = first.uib.in_h;
        in_w = first.uib.in_w;
        in_c = first.uib.in_c;
        in_zp = first.uib.has_start_dw ? -first.uib.start_dw.input_offset
                                       : -first.uib.expand.input_offset;
        in_scale = first.uib.has_start_dw ? first.uib.start_dw.input_scale
                                          : first.uib.expand.input_scale;
    }
    else
    {
        return fail("unsupported first layer kind");
    }
    if (in_h <= 0 || in_w <= 0 || in_c <= 0 || in_scale <= 0.0f)
        return fail("invalid network input geometry/scale");

    uint32_t cur_id = 0;
    if (!DefineActQint8(subgraph,
                        in_zp,
                        in_scale,
                        static_cast<size_t>(in_h),
                        static_cast<size_t>(in_w),
                        static_cast<size_t>(in_c),
                        runtime.xnn_net_ext_in,
                        XNN_VALUE_FLAG_EXTERNAL_INPUT,
                        &cur_id))
        return fail("define external input failed");

    // Track NHWC geometry + quant params for Flatten reshape and MaxPool.
    uint32_t cur_h = static_cast<uint32_t>(in_h);
    uint32_t cur_w = static_cast<uint32_t>(in_w);
    uint32_t cur_c = static_cast<uint32_t>(in_c);
    int32_t cur_zp = in_zp;
    float cur_scale = in_scale;

    for (uint32_t i = 0; i < runtime.num_layers; ++i)
    {
        CmsisQuantPlan::LayerPlan& lp = runtime.layers[i];
        CnnBlock& block = network.GetBlock(i);
        const bool is_last = i + 1 == runtime.num_layers;
        char layer_err[96];

        switch (lp.kind)
        {
            case CmsisQuantPlan::LayerKind::FlattenView:
            {
                if (cur_h == 1 && cur_w == 1)
                    break;
                const uint32_t features = cur_h * cur_w * cur_c;
                uint32_t out_id = 0;
                const uint32_t out_ext =
                    is_last ? runtime.xnn_net_ext_out : XNN_INVALID_VALUE_ID;
                const uint32_t out_flags = is_last ? XNN_VALUE_FLAG_EXTERNAL_OUTPUT : 0;
                if (!DefineActQint8(subgraph,
                                    cur_zp,
                                    cur_scale,
                                    /*h=*/1,
                                    /*w=*/1,
                                    static_cast<size_t>(features),
                                    out_ext,
                                    out_flags,
                                    &out_id))
                {
                    std::snprintf(layer_err, sizeof(layer_err),
                                  "flatten act failed layer %u", i);
                    return fail(layer_err);
                }
                const size_t new_shape[4] = {1, 1, 1, static_cast<size_t>(features)};
                if (xnn_define_static_reshape(subgraph,
                                              /*num_dims=*/4,
                                              new_shape,
                                              cur_id,
                                              out_id,
                                              /*flags=*/0) != xnn_status_success)
                {
                    std::snprintf(layer_err, sizeof(layer_err),
                                  "flatten reshape failed layer %u", i);
                    return fail(layer_err);
                }
                cur_id = out_id;
                cur_h = 1;
                cur_w = 1;
                cur_c = features;
                break;
            }

            case CmsisQuantPlan::LayerKind::Conv2D:
            {
                if (!lp.conv.ready)
                    return fail("conv plan not ready");
                uint32_t filter_id = 0;
                uint32_t bias_id = 0;
                float* bias_scales = nullptr;
                if (!DefineFilterAndBiasConv(subgraph,
                                             lp.conv,
                                             block.conv.conv.weights_q,
                                             block.conv.conv.bias_q,
                                             &bias_scales,
                                             &filter_id,
                                             &bias_id))
                {
                    delete[] bias_scales;
                    std::snprintf(layer_err, sizeof(layer_err),
                                  "conv filter/bias failed layer %u", i);
                    return fail(layer_err);
                }
                if (!PushBiasScales(runtime, bias_scales))
                    return fail("conv bias scales alloc failed");

                uint32_t out_id = 0;
                const uint32_t out_ext =
                    is_last ? runtime.xnn_net_ext_out : XNN_INVALID_VALUE_ID;
                const uint32_t out_flags = is_last ? XNN_VALUE_FLAG_EXTERNAL_OUTPUT : 0;
                if (!DefineActQint8(subgraph,
                                    lp.conv.output_offset,
                                    lp.conv.output_scale,
                                    static_cast<size_t>(lp.conv.out_h),
                                    static_cast<size_t>(lp.conv.out_w),
                                    static_cast<size_t>(lp.conv.out_c),
                                    out_ext,
                                    out_flags,
                                    &out_id))
                {
                    std::snprintf(layer_err, sizeof(layer_err),
                                  "conv act failed layer %u", i);
                    return fail(layer_err);
                }
                if (!DefineConvNode(subgraph, lp.conv, cur_id, filter_id, bias_id, out_id))
                {
                    std::snprintf(layer_err, sizeof(layer_err),
                                  "conv node failed layer %u", i);
                    return fail(layer_err);
                }
                cur_id = out_id;
                cur_h = static_cast<uint32_t>(lp.conv.out_h);
                cur_w = static_cast<uint32_t>(lp.conv.out_w);
                cur_c = static_cast<uint32_t>(lp.conv.out_c);
                cur_zp = lp.conv.output_offset;
                cur_scale = lp.conv.output_scale;
                break;
            }

            case CmsisQuantPlan::LayerKind::DepthwiseConv2D:
            {
                if (!lp.depthwise.ready || !lp.depthwise.weights_hwc)
                    return fail("depthwise plan/weights not ready");
                uint32_t filter_id = 0;
                uint32_t bias_id = 0;
                float* bias_scales = nullptr;
                if (!DefineFilterAndBiasDw(subgraph,
                                           lp.depthwise,
                                           lp.depthwise.weights_hwc,
                                           block.depthwise_conv.depthwise.bias_q,
                                           &bias_scales,
                                           &filter_id,
                                           &bias_id))
                {
                    delete[] bias_scales;
                    std::snprintf(layer_err, sizeof(layer_err),
                                  "dw filter/bias failed layer %u", i);
                    return fail(layer_err);
                }
                if (!PushBiasScales(runtime, bias_scales))
                    return fail("dw bias scales alloc failed");

                uint32_t out_id = 0;
                const uint32_t out_ext =
                    is_last ? runtime.xnn_net_ext_out : XNN_INVALID_VALUE_ID;
                const uint32_t out_flags = is_last ? XNN_VALUE_FLAG_EXTERNAL_OUTPUT : 0;
                if (!DefineActQint8(subgraph,
                                    lp.depthwise.output_offset,
                                    lp.depthwise.output_scale,
                                    static_cast<size_t>(lp.depthwise.out_h),
                                    static_cast<size_t>(lp.depthwise.out_w),
                                    static_cast<size_t>(lp.depthwise.channels),
                                    out_ext,
                                    out_flags,
                                    &out_id))
                {
                    std::snprintf(layer_err, sizeof(layer_err),
                                  "dw act failed layer %u", i);
                    return fail(layer_err);
                }
                if (!DefineDwNode(
                        subgraph, lp.depthwise, cur_id, filter_id, bias_id, out_id))
                {
                    std::snprintf(layer_err, sizeof(layer_err),
                                  "dw node failed layer %u", i);
                    return fail(layer_err);
                }
                cur_id = out_id;
                cur_h = static_cast<uint32_t>(lp.depthwise.out_h);
                cur_w = static_cast<uint32_t>(lp.depthwise.out_w);
                cur_c = static_cast<uint32_t>(lp.depthwise.channels);
                cur_zp = lp.depthwise.output_offset;
                cur_scale = lp.depthwise.output_scale;
                break;
            }

            case CmsisQuantPlan::LayerKind::MobilenetV4Uib:
            {
                MobileNetV4Uib& uib = block.mobilenetv4_uib.block;
                float* start_scales = nullptr;
                float* expand_scales = nullptr;
                float* middle_scales = nullptr;
                float* proj_scales = nullptr;
                uint32_t out_id = 0;
                bool includes_residual = false;
                const uint32_t out_ext =
                    is_last ? runtime.xnn_net_ext_out : XNN_INVALID_VALUE_ID;
                if (!AppendUibBody(subgraph,
                                   lp.uib,
                                   uib,
                                   cur_id,
                                   out_ext,
                                   &out_id,
                                   &start_scales,
                                   &expand_scales,
                                   &middle_scales,
                                   &proj_scales,
                                   &includes_residual))
                {
                    delete[] start_scales;
                    delete[] expand_scales;
                    delete[] middle_scales;
                    delete[] proj_scales;
                    std::snprintf(layer_err, sizeof(layer_err),
                                  "UIB body failed layer %u", i);
                    return fail(layer_err);
                }
                if (!PushBiasScales(runtime, start_scales))
                {
                    delete[] expand_scales;
                    delete[] middle_scales;
                    delete[] proj_scales;
                    return fail("UIB bias scales alloc failed");
                }
                start_scales = nullptr;
                if (!PushBiasScales(runtime, expand_scales))
                {
                    delete[] middle_scales;
                    delete[] proj_scales;
                    return fail("UIB bias scales alloc failed");
                }
                expand_scales = nullptr;
                if (!PushBiasScales(runtime, middle_scales))
                {
                    delete[] proj_scales;
                    return fail("UIB bias scales alloc failed");
                }
                middle_scales = nullptr;
                if (!PushBiasScales(runtime, proj_scales))
                    return fail("UIB bias scales alloc failed");
                lp.uib.xnn_subgraph_includes_residual = includes_residual;
                cur_id = out_id;
                cur_h = static_cast<uint32_t>(lp.uib.out_h);
                cur_w = static_cast<uint32_t>(lp.uib.out_w);
                cur_c = static_cast<uint32_t>(lp.uib.out_c);
                cur_zp = lp.uib.proj.output_offset;
                cur_scale = lp.uib.proj.output_scale;
                break;
            }

            case CmsisQuantPlan::LayerKind::AvgPool2D:
            {
                if (!lp.pool.ready)
                    return fail("avgpool plan not ready");
                // Match TF Lite int8 head: MEAN → xnn_define_static_reduce_v2
                // (AVERAGE_POOL_2D in the XNNPACK delegate is float32-only;
                //  xnn_define_average_pooling_2d has no qs8 path).
                // KEEP_DIMS → [1,1,1,C] so a following 1×1 Conv2D still sees NHWC.
                const float out_scale =
                    lp.pool.output_scale > 0.0f ? lp.pool.output_scale : lp.pool.input_scale;
                const int32_t out_zp = lp.pool.output_scale > 0.0f
                                          ? lp.pool.output_zero_point
                                          : lp.pool.input_zero_point;
                uint32_t out_id = 0;
                const uint32_t out_ext =
                    is_last ? runtime.xnn_net_ext_out : XNN_INVALID_VALUE_ID;
                const uint32_t out_flags = is_last ? XNN_VALUE_FLAG_EXTERNAL_OUTPUT : 0;
                if (!DefineActQint8(subgraph,
                                    out_zp,
                                    out_scale,
                                    /*h=*/1,
                                    /*w=*/1,
                                    static_cast<size_t>(lp.pool.in_c),
                                    out_ext,
                                    out_flags,
                                    &out_id))
                {
                    std::snprintf(layer_err, sizeof(layer_err),
                                  "avgpool act failed layer %u", i);
                    return fail(layer_err);
                }
                const int64_t axes[2] = {1, 2};
                if (xnn_define_static_reduce_v2(subgraph,
                                                xnn_reduce_mean,
                                                /*num_reduction_axes=*/2,
                                                axes,
                                                cur_id,
                                                out_id,
                                                XNN_FLAG_KEEP_DIMS) != xnn_status_success)
                {
                    std::snprintf(layer_err, sizeof(layer_err),
                                  "avgpool reduce_mean failed layer %u", i);
                    return fail(layer_err);
                }
                cur_id = out_id;
                cur_h = 1;
                cur_w = 1;
                cur_c = static_cast<uint32_t>(lp.pool.in_c);
                cur_zp = out_zp;
                cur_scale = out_scale;
                break;
            }

            case CmsisQuantPlan::LayerKind::MaxPool2D:
            {
                if (!lp.pool.ready)
                    return fail("maxpool plan not ready");
                if (lp.pool.pad_h < 0 || lp.pool.pad_w < 0)
                    return fail("maxpool negative pad unsupported");
                const float out_scale =
                    lp.pool.output_scale > 0.0f ? lp.pool.output_scale : cur_scale;
                const int32_t out_zp = lp.pool.output_scale > 0.0f
                                          ? lp.pool.output_zero_point
                                          : cur_zp;
                float out_min = 0.0f;
                float out_max = 0.0f;
                ActivationClampFloat(lp.pool.clamp, out_scale, out_zp, out_min, out_max);

                uint32_t out_id = 0;
                const uint32_t out_ext =
                    is_last ? runtime.xnn_net_ext_out : XNN_INVALID_VALUE_ID;
                const uint32_t out_flags = is_last ? XNN_VALUE_FLAG_EXTERNAL_OUTPUT : 0;
                if (!DefineActQint8(subgraph,
                                    out_zp,
                                    out_scale,
                                    static_cast<size_t>(lp.pool.out_h),
                                    static_cast<size_t>(lp.pool.out_w),
                                    static_cast<size_t>(lp.pool.in_c),
                                    out_ext,
                                    out_flags,
                                    &out_id))
                {
                    std::snprintf(layer_err, sizeof(layer_err),
                                  "maxpool act failed layer %u", i);
                    return fail(layer_err);
                }
                if (xnn_define_max_pooling_2d(
                        subgraph,
                        static_cast<uint32_t>(lp.pool.pad_h),
                        static_cast<uint32_t>(lp.pool.pad_w),
                        static_cast<uint32_t>(lp.pool.pad_h),
                        static_cast<uint32_t>(lp.pool.pad_w),
                        static_cast<uint32_t>(lp.pool.pool_h),
                        static_cast<uint32_t>(lp.pool.pool_w),
                        static_cast<uint32_t>(lp.pool.stride),
                        static_cast<uint32_t>(lp.pool.stride),
                        /*dilation_height=*/1,
                        /*dilation_width=*/1,
                        out_min,
                        out_max,
                        cur_id,
                        out_id,
                        /*flags=*/0) != xnn_status_success)
                {
                    std::snprintf(layer_err, sizeof(layer_err),
                                  "maxpool node failed layer %u", i);
                    return fail(layer_err);
                }
                cur_id = out_id;
                cur_h = static_cast<uint32_t>(lp.pool.out_h);
                cur_w = static_cast<uint32_t>(lp.pool.out_w);
                cur_c = static_cast<uint32_t>(lp.pool.in_c);
                cur_zp = out_zp;
                cur_scale = out_scale;
                break;
            }

            case CmsisQuantPlan::LayerKind::Dense:
            case CmsisQuantPlan::LayerKind::DenseSoftmax:
            {
                if (!lp.fc.ready)
                    return fail("fc plan not ready");
                // Softmax is omitted for classification (logits); DenseSoftmax uses None clamp.
                const int8_t* weights =
                    static_cast<const int8_t*>(block.dense.weights.data);
                const int32_t* bias = static_cast<const int32_t*>(block.dense.bias.data);
                uint32_t filter_id = 0;
                uint32_t bias_id = 0;
                float* bias_scales = nullptr;
                if (!DefineFilterAndBiasFc(subgraph,
                                           lp.fc,
                                           weights,
                                           bias,
                                           &bias_scales,
                                           &filter_id,
                                           &bias_id))
                {
                    delete[] bias_scales;
                    std::snprintf(layer_err, sizeof(layer_err),
                                  "fc filter/bias failed layer %u", i);
                    return fail(layer_err);
                }
                if (!PushBiasScales(runtime, bias_scales))
                    return fail("fc bias scales alloc failed");

                float out_min = 0.0f;
                float out_max = 0.0f;
                ActivationClampFloat(
                    lp.fc.clamp, lp.fc.output_scale, lp.fc.output_offset, out_min, out_max);

                // Flatten is a no-op. Input is NHWC 1×1×C from the head conv; FC allows
                // N-D when the last dim matches in_features. Match output rank to input
                // (no XNN_FLAG_TENSORFLOW_RESHAPE_2D in this XNNPACK).
                uint32_t out_id = 0;
                const uint32_t out_ext =
                    is_last ? runtime.xnn_net_ext_out : XNN_INVALID_VALUE_ID;
                const uint32_t out_flags = is_last ? XNN_VALUE_FLAG_EXTERNAL_OUTPUT : 0;
                const size_t out_dims[4] = {
                    1, 1, 1, static_cast<size_t>(lp.fc.out_features)};
                if (xnn_define_quantized_tensor_value(subgraph,
                                                     xnn_datatype_qint8,
                                                     lp.fc.output_offset,
                                                     lp.fc.output_scale,
                                                     /*num_dims=*/4,
                                                     out_dims,
                                                     /*data=*/nullptr,
                                                     out_ext,
                                                     out_flags,
                                                     &out_id) != xnn_status_success)
                {
                    std::snprintf(layer_err, sizeof(layer_err),
                                  "fc act failed layer %u", i);
                    return fail(layer_err);
                }
                if (xnn_define_fully_connected(subgraph,
                                               out_min,
                                               out_max,
                                               cur_id,
                                               filter_id,
                                               bias_id,
                                               out_id,
                                               /*flags=*/0) != xnn_status_success)
                {
                    std::snprintf(layer_err, sizeof(layer_err),
                                  "fc node failed layer %u", i);
                    return fail(layer_err);
                }
                cur_id = out_id;
                cur_h = 1;
                cur_w = 1;
                cur_c = static_cast<uint32_t>(lp.fc.out_features);
                cur_zp = lp.fc.output_offset;
                cur_scale = lp.fc.output_scale;
                break;
            }

            default:
                std::snprintf(layer_err, sizeof(layer_err),
                              "unsupported layer kind %d at %u",
                              static_cast<int>(lp.kind),
                              i);
                return fail(layer_err);
        }
    }

    xnn_runtime_t xnn_rt = nullptr;
    // Match TF Lite XNNPACK delegate defaults for 1-thread peer benches:
    //   flags = XNN_FLAG_DONT_SPIN_WORKERS
    //   shared xnn_workspace (created with the delegate)
    //   threadpool = nullptr when num_threads <= 1
    const xnn_status create_st =
        xnn_create_runtime_v4(subgraph,
                              static_cast<xnn_weights_cache_t>(weights_cache),
                              static_cast<xnn_workspace_t>(runtime.xnn_workspace),
                              /*threadpool=*/nullptr,
                              XNN_FLAG_DONT_SPIN_WORKERS,
                              &xnn_rt);
    (void)xnn_delete_subgraph(subgraph);
    subgraph = nullptr;
    if (create_st != xnn_status_success || !xnn_rt)
    {
        std::fprintf(stderr,
                     "CreateNetworkSubgraph: xnn_create_runtime_v4 failed (%s)\n",
                     XnnStatusName(create_st));
        DestroyNetworkSubgraph(runtime);
        return false;
    }

    runtime.xnn_network_runtime = xnn_rt;
    if (weights_cache)
    {
        runtime.xnn_network_ready = false;
        return true;
    }
    FinishNetworkAfterWeightsCache(runtime);
    return runtime.xnn_network_ready;
}

void FinishNetworkAfterWeightsCache(CmsisQuantPlan::Runtime& runtime)
{
    if (runtime.xnn_network_ready)
        return;
    if (!runtime.xnn_network_runtime || !runtime.layers || runtime.num_layers == 0)
        return;

    const CmsisQuantPlan::LayerPlan& first = runtime.layers[0];
    size_t in_h = 0;
    size_t in_w = 0;
    size_t in_c = 0;
    if (first.kind == CmsisQuantPlan::LayerKind::Conv2D)
    {
        in_h = static_cast<size_t>(first.conv.in_h);
        in_w = static_cast<size_t>(first.conv.in_w);
        in_c = static_cast<size_t>(first.conv.in_c);
    }
    else if (first.kind == CmsisQuantPlan::LayerKind::MobilenetV4Uib)
    {
        in_h = static_cast<size_t>(first.uib.in_h);
        in_w = static_cast<size_t>(first.uib.in_w);
        in_c = static_cast<size_t>(first.uib.in_c);
    }
    else
    {
        DestroyNetworkSubgraph(runtime);
        return;
    }

    auto* xnn_rt = static_cast<xnn_runtime_t>(runtime.xnn_network_runtime);
    const size_t in_dims[4] = {1, in_h, in_w, in_c};
    if (xnn_reshape_external_value(xnn_rt, runtime.xnn_net_ext_in, 4, in_dims) !=
        xnn_status_success)
    {
        std::fprintf(stderr, "FinishNetworkAfterWeightsCache: reshape_external_value failed\n");
        DestroyNetworkSubgraph(runtime);
        return;
    }
    if (xnn_reshape_runtime(xnn_rt) != xnn_status_success)
    {
        std::fprintf(stderr, "FinishNetworkAfterWeightsCache: reshape_runtime failed\n");
        DestroyNetworkSubgraph(runtime);
        return;
    }
    runtime.xnn_network_ready = true;
}

bool InvokeNetworkSubgraph(CmsisQuantPlan::Runtime& runtime,
                           const int8_t* input,
                           int8_t* output)
{
    if (!runtime.xnn_network_ready || !runtime.xnn_network_runtime || !input || !output)
        return false;

    auto* xnn_rt = static_cast<xnn_runtime_t>(runtime.xnn_network_runtime);
    if (input != runtime.bound_input || output != runtime.bound_output)
    {
        const xnn_external_value externals[2] = {
            {runtime.xnn_net_ext_in, const_cast<int8_t*>(input)},
            {runtime.xnn_net_ext_out, output},
        };
        if (xnn_setup_runtime_v2(xnn_rt, 2, externals) != xnn_status_success)
            return false;
        runtime.bound_input = input;
        runtime.bound_output = output;
    }
    return xnn_invoke_runtime(xnn_rt) == xnn_status_success;
}

void DestroyMlpRuntime(MlpRuntime& runtime)
{
    if (runtime.xnn_network_runtime)
    {
        (void)xnn_delete_runtime(static_cast<xnn_runtime_t>(runtime.xnn_network_runtime));
        runtime.xnn_network_runtime = nullptr;
    }
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
    if (runtime.bias_scales)
    {
        for (uint32_t i = 0; i < runtime.bias_scales_count; ++i)
            delete[] runtime.bias_scales[i];
        delete[] runtime.bias_scales;
        runtime.bias_scales = nullptr;
        runtime.bias_scales_count = 0;
    }
    runtime.ready = false;
    runtime.ext_in = 0;
    runtime.ext_out = 1;
    runtime.in_features = 0;
    runtime.out_features = 0;
    runtime.bound_input = nullptr;
    runtime.bound_output = nullptr;
}

bool BuildMlpNetworkSubgraph(MLPNetwork& network,
                             Arena& arena,
                             uint32_t in_features,
                             MlpRuntime*& out_runtime)
{
    (void)arena;
    out_runtime = nullptr;
    if (!network.IsValid() || network.layer_count() == 0 || in_features == 0 ||
        !EnsureXnnInitialized())
        return false;

    MlpRuntime* runtime = new (std::nothrow) MlpRuntime{};
    if (!runtime)
        return false;

    auto fail = [&](const char* reason) {
        std::fprintf(stderr, "BuildMlpNetworkSubgraph: %s\n", reason);
        DestroyMlpRuntime(*runtime);
        delete runtime;
        return false;
    };

    constexpr size_t kWeightsCacheBytes = 16u * 1024u * 1024u;
    xnn_weights_cache_t cache = nullptr;
    if (xnn_create_weights_cache_with_size(kWeightsCacheBytes, &cache) == xnn_status_success ||
        xnn_create_weights_cache(&cache) == xnn_status_success)
        runtime->xnn_weights_cache = cache;
    xnn_workspace_t ws = nullptr;
    if (xnn_create_workspace(&ws) == xnn_status_success)
        runtime->xnn_workspace = ws;

    runtime->ext_in = 0;
    runtime->ext_out = 1;
    runtime->in_features = in_features;

    xnn_subgraph_t subgraph = nullptr;
    if (xnn_create_subgraph(/*external_value_ids=*/2, /*flags=*/0, &subgraph) !=
        xnn_status_success)
        return fail("xnn_create_subgraph failed");

    auto fail_sg = [&](const char* reason) {
        (void)xnn_delete_subgraph(subgraph);
        return fail(reason);
    };

    MLPLayer& first = network.GetLayer(0);
    if (!first.quant.enabled || first.quant.params.input_scale <= 0.0f)
        return fail_sg("first layer missing quant params");

    uint32_t cur_id = 0;
    const size_t in_dims[2] = {1, static_cast<size_t>(in_features)};
    if (xnn_define_quantized_tensor_value(subgraph,
                                         xnn_datatype_qint8,
                                         first.quant.params.input_zero_point,
                                         first.quant.params.input_scale,
                                         /*num_dims=*/2,
                                         in_dims,
                                         /*data=*/nullptr,
                                         runtime->ext_in,
                                         XNN_VALUE_FLAG_EXTERNAL_INPUT,
                                         &cur_id) != xnn_status_success)
        return fail_sg("define external input failed");

    uint32_t cur_features = in_features;
    const uint32_t n = network.layer_count();
    for (uint32_t i = 0; i < n; ++i)
    {
        MLPLayer& layer = network.GetLayer(i);
        const bool is_last = i + 1 == n;
        char layer_err[96];

        if (!layer.quant.enabled || !layer.weights.data || layer.weights.rank != 2)
            return fail_sg("dense quant/weights missing");

        const auto& q = layer.quant.params;
        CmsisQuantPlan::FcPlan plan{};
        plan.input_offset = -q.input_zero_point;
        plan.filter_offset = -q.weight_zero_point;
        plan.output_offset = q.output_zero_point;
        plan.clamp = (layer.activation == ActivationType::ReLU)
                         ? QuantInteger::QuantClamp::ReLU
                         : (layer.activation == ActivationType::ReLU6)
                               ? QuantInteger::QuantClamp::ReLU6
                               : QuantInteger::QuantClamp::None;
        // Softmax omitted in subgraph (logits), matching CNN DenseSoftmax path.
        if (layer.activation == ActivationType::Softmax)
            plan.clamp = QuantInteger::QuantClamp::None;
        plan.input_scale = q.input_scale;
        plan.weight_scale = q.weight_scale;
        plan.output_scale = q.output_scale;
        plan.weight_channel_scales = q.weight_channel_scales;
        plan.num_weight_channel_scales = q.num_weight_channel_scales;
        plan.in_features = static_cast<int32_t>(layer.weights.shape[1]);
        plan.out_features = static_cast<int32_t>(layer.weights.shape[0]);
        plan.ready = true;

        if (static_cast<uint32_t>(plan.in_features) != cur_features)
        {
            std::snprintf(layer_err, sizeof(layer_err),
                          "dense in_features mismatch layer %u", i);
            return fail_sg(layer_err);
        }

        uint32_t filter_id = 0;
        uint32_t bias_id = 0;
        float* bias_scales = nullptr;
        if (!DefineFilterAndBiasFc(subgraph,
                                   plan,
                                   static_cast<const int8_t*>(layer.weights.data),
                                   static_cast<const int32_t*>(layer.bias.data),
                                   &bias_scales,
                                   &filter_id,
                                   &bias_id))
        {
            delete[] bias_scales;
            std::snprintf(layer_err, sizeof(layer_err),
                          "fc filter/bias failed layer %u", i);
            return fail_sg(layer_err);
        }
        if (bias_scales)
        {
            float** next =
                new (std::nothrow) float*[runtime->bias_scales_count + 1];
            if (!next)
            {
                delete[] bias_scales;
                return fail_sg("bias scales alloc failed");
            }
            for (uint32_t j = 0; j < runtime->bias_scales_count; ++j)
                next[j] = runtime->bias_scales[j];
            next[runtime->bias_scales_count] = bias_scales;
            delete[] runtime->bias_scales;
            runtime->bias_scales = next;
            ++runtime->bias_scales_count;
        }

        float out_min = 0.0f;
        float out_max = 0.0f;
        ActivationClampFloat(
            plan.clamp, plan.output_scale, plan.output_offset, out_min, out_max);

        uint32_t out_id = 0;
        const uint32_t out_ext = is_last ? runtime->ext_out : XNN_INVALID_VALUE_ID;
        const uint32_t out_flags = is_last ? XNN_VALUE_FLAG_EXTERNAL_OUTPUT : 0;
        const size_t out_dims[2] = {1, static_cast<size_t>(plan.out_features)};
        if (xnn_define_quantized_tensor_value(subgraph,
                                             xnn_datatype_qint8,
                                             plan.output_offset,
                                             plan.output_scale,
                                             /*num_dims=*/2,
                                             out_dims,
                                             /*data=*/nullptr,
                                             out_ext,
                                             out_flags,
                                             &out_id) != xnn_status_success)
        {
            std::snprintf(layer_err, sizeof(layer_err), "fc act failed layer %u", i);
            return fail_sg(layer_err);
        }
        if (xnn_define_fully_connected(subgraph,
                                       out_min,
                                       out_max,
                                       cur_id,
                                       filter_id,
                                       bias_id,
                                       out_id,
                                       /*flags=*/0) != xnn_status_success)
        {
            std::snprintf(layer_err, sizeof(layer_err), "fc node failed layer %u", i);
            return fail_sg(layer_err);
        }
        cur_id = out_id;
        cur_features = static_cast<uint32_t>(plan.out_features);
        if (is_last)
            runtime->out_features = cur_features;
    }

    xnn_runtime_t xnn_rt = nullptr;
    const xnn_status create_st =
        xnn_create_runtime_v4(subgraph,
                              static_cast<xnn_weights_cache_t>(runtime->xnn_weights_cache),
                              static_cast<xnn_workspace_t>(runtime->xnn_workspace),
                              /*threadpool=*/nullptr,
                              XNN_FLAG_DONT_SPIN_WORKERS,
                              &xnn_rt);
    (void)xnn_delete_subgraph(subgraph);
    subgraph = nullptr;
    if (create_st != xnn_status_success || !xnn_rt)
        return fail("xnn_create_runtime_v4 failed");
    runtime->xnn_network_runtime = xnn_rt;

    if (runtime->xnn_weights_cache)
    {
        auto* cache_ptr = static_cast<xnn_weights_cache_t>(runtime->xnn_weights_cache);
        if (xnn_finalize_weights_cache(cache_ptr, xnn_weights_cache_finalization_kind_hard) !=
            xnn_status_success)
        {
            (void)xnn_finalize_weights_cache(cache_ptr,
                                             xnn_weights_cache_finalization_kind_soft);
        }
    }

    const size_t reshape_in[2] = {1, static_cast<size_t>(in_features)};
    if (xnn_reshape_external_value(xnn_rt, runtime->ext_in, 2, reshape_in) !=
        xnn_status_success)
        return fail("reshape input failed");
    const size_t reshape_out[2] = {1, static_cast<size_t>(runtime->out_features)};
    if (xnn_reshape_external_value(xnn_rt, runtime->ext_out, 2, reshape_out) !=
        xnn_status_success)
        return fail("reshape output failed");
    if (xnn_reshape_runtime(xnn_rt) != xnn_status_success)
        return fail("reshape_runtime failed");

    runtime->ready = true;
    std::fprintf(stderr, "BuildMlpNetworkSubgraph: MLP xnn_subgraph ready\n");
    out_runtime = runtime;
    return true;
}

bool InvokeMlpNetworkSubgraph(MlpRuntime& runtime, const int8_t* input, int8_t* output)
{
    if (!runtime.ready || !runtime.xnn_network_runtime || !input || !output)
        return false;
    auto* xnn_rt = static_cast<xnn_runtime_t>(runtime.xnn_network_runtime);
    if (input != runtime.bound_input || output != runtime.bound_output)
    {
        const xnn_external_value externals[2] = {
            {runtime.ext_in, const_cast<int8_t*>(input)},
            {runtime.ext_out, output},
        };
        if (xnn_setup_runtime_v2(xnn_rt, 2, externals) != xnn_status_success)
            return false;
        runtime.bound_input = input;
        runtime.bound_output = output;
    }
    return xnn_invoke_runtime(xnn_rt) == xnn_status_success;
}

bool TryConv2dNhwcQuantPlan(const CmsisQuantPlan::Conv2DPlan& plan,
                            const int8_t* input,
                            const int8_t* weights,
                            const int32_t* bias,
                            int8_t* output)
{
    if (!plan.ready || !input || !weights || !bias || !output)
        return false;

    if (plan.xnn.ready)
    {
        return RunQs8ConvCached(const_cast<CmsisQuantPlan::XnnpackOpHoist&>(plan.xnn),
                                input,
                                output);
    }

    int8_t out_min = -128;
    int8_t out_max = 127;
    ActivationClampS8(plan.clamp, plan.output_scale, plan.output_offset, out_min, out_max);

    return RunQs8Conv(/*pad_h=*/static_cast<uint32_t>(plan.pad_h),
                      /*pad_w=*/static_cast<uint32_t>(plan.pad_w),
                      /*kernel_h=*/static_cast<uint32_t>(plan.kernel_size),
                      /*kernel_w=*/static_cast<uint32_t>(plan.kernel_size),
                      /*stride=*/static_cast<uint32_t>(plan.stride),
                      /*groups=*/1,
                      /*group_input_channels=*/static_cast<size_t>(plan.in_c),
                      /*group_output_channels=*/static_cast<size_t>(plan.out_c),
                      /*input_channel_stride=*/static_cast<size_t>(plan.in_c),
                      /*output_channel_stride=*/static_cast<size_t>(plan.out_c),
                      /*input_zp=*/static_cast<int8_t>(-plan.input_offset),
                      plan.input_scale,
                      plan.weight_scale,
                      plan.weight_channel_scales,
                      plan.num_weight_channel_scales,
                      weights,
                      bias,
                      /*output_zp=*/static_cast<int8_t>(plan.output_offset),
                      plan.output_scale,
                      out_min,
                      out_max,
                      /*flags=*/0,
                      static_cast<size_t>(plan.in_h),
                      static_cast<size_t>(plan.in_w),
                      static_cast<size_t>(plan.out_h),
                      static_cast<size_t>(plan.out_w),
                      input,
                      output);
}

bool TryDepthwiseConv2dNhwcQuantPlan(const CmsisQuantPlan::DepthwiseConv2DPlan& plan,
                                     const int8_t* input,
                                     const int8_t* weights_chw,
                                     const int32_t* bias,
                                     int8_t* output)
{
    if (!plan.ready || !input || !weights_chw || !bias || !output)
        return false;

    if (plan.xnn.ready)
    {
        return RunQs8ConvCached(const_cast<CmsisQuantPlan::XnnpackOpHoist&>(plan.xnn),
                                input,
                                output);
    }

    const int8_t* kernel = plan.weights_hwc ? plan.weights_hwc : weights_chw;
    const uint32_t flags = plan.weights_hwc ? XNN_FLAG_DEPTHWISE_CONVOLUTION : 0;

    int8_t out_min = -128;
    int8_t out_max = 127;
    ActivationClampS8(plan.clamp, plan.output_scale, plan.output_offset, out_min, out_max);

    const size_t channels = static_cast<size_t>(plan.channels);
    return RunQs8Conv(/*pad_h=*/static_cast<uint32_t>(plan.pad_h),
                      /*pad_w=*/static_cast<uint32_t>(plan.pad_w),
                      /*kernel_h=*/static_cast<uint32_t>(plan.kernel_h),
                      /*kernel_w=*/static_cast<uint32_t>(plan.kernel_w),
                      /*stride=*/static_cast<uint32_t>(plan.stride),
                      /*groups=*/static_cast<uint32_t>(plan.channels),
                      /*group_input_channels=*/1,
                      /*group_output_channels=*/1,
                      /*input_channel_stride=*/channels,
                      /*output_channel_stride=*/channels,
                      /*input_zp=*/static_cast<int8_t>(-plan.input_offset),
                      plan.input_scale,
                      plan.weight_scale,
                      plan.weight_channel_scales,
                      plan.num_weight_channel_scales,
                      kernel,
                      bias,
                      /*output_zp=*/static_cast<int8_t>(plan.output_offset),
                      plan.output_scale,
                      out_min,
                      out_max,
                      flags,
                      static_cast<size_t>(plan.in_h),
                      static_cast<size_t>(plan.in_w),
                      static_cast<size_t>(plan.out_h),
                      static_cast<size_t>(plan.out_w),
                      input,
                      output);
}

bool TryMaxPool2dNhwcQuantPlan(const CmsisQuantPlan::Pool2DPlan& plan,
                               const int8_t* input,
                               int8_t* output)
{
    if (!plan.ready || !input || !output || plan.pool_h != plan.pool_w)
        return false;

    if (plan.xnn.ready && plan.xnn.op)
    {
        auto* op = static_cast<xnn_operator_t>(plan.xnn.op);
        if (xnn_setup_max_pooling2d_nhwc_s8(op, input, output) != xnn_status_success)
            return false;
        return xnn_run_operator(op, nullptr) == xnn_status_success;
    }

    if (!EnsureXnnInitialized())
        return false;

    int8_t out_min = -128;
    int8_t out_max = 127;
    ActivationClampS8(plan.clamp, plan.output_scale, plan.output_zero_point, out_min, out_max);

    ScopedOp scoped;
    if (xnn_create_max_pooling2d_nhwc_s8(
            static_cast<uint32_t>(plan.pad_h),
            static_cast<uint32_t>(plan.pad_w),
            static_cast<uint32_t>(plan.pad_h),
            static_cast<uint32_t>(plan.pad_w),
            static_cast<uint32_t>(plan.pool_h),
            static_cast<uint32_t>(plan.pool_w),
            static_cast<uint32_t>(plan.stride),
            static_cast<uint32_t>(plan.stride),
            /*dilation_height=*/1,
            /*dilation_width=*/1,
            out_min,
            out_max,
            /*flags=*/0,
            &scoped.op) != xnn_status_success)
        return false;

    size_t out_h = 0;
    size_t out_w = 0;
    const size_t channels = static_cast<size_t>(plan.in_c);
    if (xnn_reshape_max_pooling2d_nhwc_s8(scoped.op,
                                          1,
                                          static_cast<size_t>(plan.in_h),
                                          static_cast<size_t>(plan.in_w),
                                          channels,
                                          channels,
                                          channels,
                                          &out_h,
                                          &out_w,
                                          nullptr) != xnn_status_success)
        return false;
    if (out_h != static_cast<size_t>(plan.out_h) || out_w != static_cast<size_t>(plan.out_w))
        return false;
    if (xnn_setup_max_pooling2d_nhwc_s8(scoped.op, input, output) != xnn_status_success)
        return false;
    return xnn_run_operator(scoped.op, nullptr) == xnn_status_success;
}

bool TryFullyConnectedQuantPlan(const CmsisQuantPlan::FcPlan& plan,
                                const int8_t* input,
                                const int8_t* weights,
                                const int32_t* bias,
                                int8_t* output_int8)
{
    if (!plan.ready || !input || !weights || !bias || !output_int8)
        return false;
    if (plan.input_scale <= 0.0f || plan.output_scale <= 0.0f)
        return false;
    if (plan.filter_offset != 0)
        return false;

    if (plan.xnn.ready && plan.xnn.op)
    {
        auto* op = static_cast<xnn_operator_t>(plan.xnn.op);
        if (plan.xnn.per_channel)
        {
            if (xnn_setup_fully_connected_nc_qs8_qc8w(op, input, output_int8) !=
                xnn_status_success)
                return false;
        }
        else
        {
            if (xnn_setup_fully_connected_nc_qs8(op, input, output_int8) != xnn_status_success)
                return false;
        }
        return xnn_run_operator(op, nullptr) == xnn_status_success;
    }

    if (!EnsureXnnInitialized())
        return false;

    int8_t out_min = -128;
    int8_t out_max = 127;
    ActivationClampS8(plan.clamp, plan.output_scale, plan.output_offset, out_min, out_max);

    const size_t in_channels = static_cast<size_t>(plan.in_features);
    const size_t out_channels = static_cast<size_t>(plan.out_features);
    const bool per_channel =
        plan.weight_channel_scales != nullptr &&
        plan.num_weight_channel_scales == static_cast<uint32_t>(plan.out_features);

    ScopedOp scoped;
    if (per_channel)
    {
        if (xnn_create_fully_connected_nc_qs8_qc8w(
                in_channels,
                out_channels,
                /*input_stride=*/in_channels,
                /*output_stride=*/out_channels,
                /*input_zero_point=*/static_cast<int8_t>(-plan.input_offset),
                plan.input_scale,
                plan.weight_channel_scales,
                weights,
                bias,
                /*output_zero_point=*/static_cast<int8_t>(plan.output_offset),
                plan.output_scale,
                out_min,
                out_max,
                /*flags=*/0,
                /*weights_cache=*/nullptr,
                &scoped.op) != xnn_status_success)
            return false;
        if (xnn_reshape_fully_connected_nc_qs8_qc8w(scoped.op, /*batch_size=*/1, nullptr) !=
            xnn_status_success)
            return false;
        if (xnn_setup_fully_connected_nc_qs8_qc8w(scoped.op, input, output_int8) !=
            xnn_status_success)
            return false;
        return xnn_run_operator(scoped.op, nullptr) == xnn_status_success;
    }

    if (plan.weight_scale <= 0.0f)
        return false;
    if (xnn_create_fully_connected_nc_qs8(
            in_channels,
            out_channels,
            /*input_stride=*/in_channels,
            /*output_stride=*/out_channels,
            /*input_zero_point=*/static_cast<int8_t>(-plan.input_offset),
            plan.input_scale,
            plan.weight_scale,
            weights,
            bias,
            /*output_zero_point=*/static_cast<int8_t>(plan.output_offset),
            plan.output_scale,
            out_min,
            out_max,
            /*flags=*/0,
            /*weights_cache=*/nullptr,
            &scoped.op) != xnn_status_success)
        return false;

    if (xnn_reshape_fully_connected_nc_qs8(scoped.op, /*batch_size=*/1, nullptr) !=
        xnn_status_success)
        return false;
    if (xnn_setup_fully_connected_nc_qs8(scoped.op, input, output_int8) != xnn_status_success)
        return false;
    return xnn_run_operator(scoped.op, nullptr) == xnn_status_success;
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
    if (pad_h != pad_h_end || pad_w != pad_w_end || !weights || !bias || !input || !output)
        return false;
    if (quant.input_scale <= 0.0f || quant.weight_scale <= 0.0f || quant.output_scale <= 0.0f)
        return false;

    const uint32_t out_h =
        nk_op_detail::CalcOutputDimAsymmetric(in_h, kernel_size, stride, pad_h, pad_h_end);
    const uint32_t out_w =
        nk_op_detail::CalcOutputDimAsymmetric(in_w, kernel_size, stride, pad_w, pad_w_end);

    const QuantInteger::QuantClamp clamp =
        apply_relu ? QuantInteger::QuantClamp::ReLU : QuantInteger::QuantClamp::None;
    int8_t out_min = -128;
    int8_t out_max = 127;
    ActivationClampS8(clamp, quant.output_scale, quant.output_zero_point, out_min, out_max);

    return RunQs8Conv(static_cast<uint32_t>(pad_h),
                      static_cast<uint32_t>(pad_w),
                      static_cast<uint32_t>(kernel_size),
                      static_cast<uint32_t>(kernel_size),
                      static_cast<uint32_t>(stride),
                      1,
                      in_c,
                      static_cast<size_t>(out_channels),
                      in_c,
                      static_cast<size_t>(out_channels),
                      static_cast<int8_t>(quant.input_zero_point),
                      quant.input_scale,
                      quant.weight_scale,
                      quant.weight_channel_scales,
                      quant.num_weight_channel_scales,
                      weights,
                      bias,
                      static_cast<int8_t>(quant.output_zero_point),
                      quant.output_scale,
                      out_min,
                      out_max,
                      0,
                      in_h,
                      in_w,
                      out_h,
                      out_w,
                      input,
                      output);
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
    if (pad_h != pad_h_end || pad_w != pad_w_end || !weights_chw || !bias || !input || !output)
        return false;
    if (quant.input_scale <= 0.0f || quant.weight_scale <= 0.0f || quant.output_scale <= 0.0f)
        return false;

    const uint32_t out_h =
        nk_op_detail::CalcOutputDimAsymmetric(in_h, kernel_h, stride, pad_h, pad_h_end);
    const uint32_t out_w =
        nk_op_detail::CalcOutputDimAsymmetric(in_w, kernel_w, stride, pad_w, pad_w_end);

    const QuantInteger::QuantClamp clamp =
        apply_relu ? QuantInteger::QuantClamp::ReLU : QuantInteger::QuantClamp::None;
    int8_t out_min = -128;
    int8_t out_max = 127;
    ActivationClampS8(clamp, quant.output_scale, quant.output_zero_point, out_min, out_max);

    return RunQs8Conv(static_cast<uint32_t>(pad_h),
                      static_cast<uint32_t>(pad_w),
                      static_cast<uint32_t>(kernel_h),
                      static_cast<uint32_t>(kernel_w),
                      static_cast<uint32_t>(stride),
                      channels,
                      1,
                      1,
                      channels,
                      channels,
                      static_cast<int8_t>(quant.input_zero_point),
                      quant.input_scale,
                      quant.weight_scale,
                      quant.weight_channel_scales,
                      quant.num_weight_channel_scales,
                      weights_chw,
                      bias,
                      static_cast<int8_t>(quant.output_zero_point),
                      quant.output_scale,
                      out_min,
                      out_max,
                      /*flags=*/0,
                      in_h,
                      in_w,
                      out_h,
                      out_w,
                      input,
                      output);
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
    if (pad_h != pad_h_end || pad_w != pad_w_end || pool_h != pool_w || !input || !output)
        return false;
    if (!EnsureXnnInitialized())
        return false;

    const uint32_t out_h =
        nk_op_detail::CalcOutputDimAsymmetric(in_h, pool_h, stride, pad_h, pad_h_end);
    const uint32_t out_w =
        nk_op_detail::CalcOutputDimAsymmetric(in_w, pool_w, stride, pad_w, pad_w_end);

    ScopedOp scoped;
    if (xnn_create_max_pooling2d_nhwc_s8(static_cast<uint32_t>(pad_h),
                                         static_cast<uint32_t>(pad_w),
                                         static_cast<uint32_t>(pad_h),
                                         static_cast<uint32_t>(pad_w),
                                         static_cast<uint32_t>(pool_h),
                                         static_cast<uint32_t>(pool_w),
                                         static_cast<uint32_t>(stride),
                                         static_cast<uint32_t>(stride),
                                         1,
                                         1,
                                         /*output_min=*/-128,
                                         /*output_max=*/127,
                                         0,
                                         &scoped.op) != xnn_status_success)
        return false;

    size_t got_h = 0;
    size_t got_w = 0;
    if (xnn_reshape_max_pooling2d_nhwc_s8(scoped.op,
                                          1,
                                          in_h,
                                          in_w,
                                          in_c,
                                          in_c,
                                          in_c,
                                          &got_h,
                                          &got_w,
                                          nullptr) != xnn_status_success)
        return false;
    if (got_h != out_h || got_w != out_w)
        return false;
    if (xnn_setup_max_pooling2d_nhwc_s8(scoped.op, input, output) != xnn_status_success)
        return false;
    return xnn_run_operator(scoped.op, nullptr) == xnn_status_success;
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
    if (!input || !weights || !bias || !output_int8 || batch == 0)
        return false;
    if (quant.input_scale <= 0.0f || quant.output_scale <= 0.0f)
        return false;
    if (quant.weight_zero_point != 0)
        return false;
    if (!EnsureXnnInitialized())
        return false;

    const QuantInteger::QuantClamp clamp =
        apply_relu ? QuantInteger::QuantClamp::ReLU : QuantInteger::QuantClamp::None;
    int8_t out_min = -128;
    int8_t out_max = 127;
    ActivationClampS8(clamp, quant.output_scale, quant.output_zero_point, out_min, out_max);

    const bool per_channel =
        quant.weight_channel_scales != nullptr &&
        quant.num_weight_channel_scales == out_features;

    ScopedOp scoped;
    if (per_channel)
    {
        if (xnn_create_fully_connected_nc_qs8_qc8w(
                in_features,
                out_features,
                in_features,
                out_features,
                static_cast<int8_t>(quant.input_zero_point),
                quant.input_scale,
                quant.weight_channel_scales,
                weights,
                bias,
                static_cast<int8_t>(quant.output_zero_point),
                quant.output_scale,
                out_min,
                out_max,
                0,
                nullptr,
                &scoped.op) != xnn_status_success)
            return false;
        if (xnn_reshape_fully_connected_nc_qs8_qc8w(scoped.op, batch, nullptr) !=
            xnn_status_success)
            return false;
        if (xnn_setup_fully_connected_nc_qs8_qc8w(scoped.op, input, output_int8) !=
            xnn_status_success)
            return false;
        return xnn_run_operator(scoped.op, nullptr) == xnn_status_success;
    }

    if (quant.weight_scale <= 0.0f)
        return false;
    if (xnn_create_fully_connected_nc_qs8(in_features,
                                          out_features,
                                          in_features,
                                          out_features,
                                          static_cast<int8_t>(quant.input_zero_point),
                                          quant.input_scale,
                                          quant.weight_scale,
                                          weights,
                                          bias,
                                          static_cast<int8_t>(quant.output_zero_point),
                                          quant.output_scale,
                                          out_min,
                                          out_max,
                                          0,
                                          nullptr,
                                          &scoped.op) != xnn_status_success)
        return false;
    if (xnn_reshape_fully_connected_nc_qs8(scoped.op, batch, nullptr) != xnn_status_success)
        return false;
    if (xnn_setup_fully_connected_nc_qs8(scoped.op, input, output_int8) != xnn_status_success)
        return false;
    return xnn_run_operator(scoped.op, nullptr) == xnn_status_success;
}

}  // namespace XnnpackQuant

#else  // !NETKIT_USE_XNNPACK

namespace XnnpackQuant
{

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

}  // namespace XnnpackQuant

#endif
