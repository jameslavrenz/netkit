/*
 * XNNPACK int8 (qs8) adapters for netkit quantized conv / depthwise / pool / FC.
 *
 * Enabled on cpu/mpu when NETKIT_XNNPACK=1 (same flag as float32 LayerFast).
 * XNNPACK is BSD-3 — see third_party/XNNPACK/LICENSE.
 */
#include "xnnpack_quant.hpp"
#include "netkit_config.h"
#include "nk_op_detail.hpp"
#include "quant_integer.hpp"

#include <cstdint>
#include <mutex>
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
        if (input_scale <= 0.0f || kernel_scale <= 0.0f || output_scale <= 0.0f)
            return false;

        ScopedOp scoped;
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

bool TryConv2dNhwcQuantPlan(const CmsisQuantPlan::Conv2DPlan& plan,
                            const int8_t* input,
                            const int8_t* weights,
                            const int32_t* bias,
                            int8_t* output)
{
    if (!plan.ready || !input || !weights || !bias || !output)
        return false;

    int8_t out_min = -128;
    int8_t out_max = 127;
    ActivationClampS8(plan.clamp, plan.output_scale, -plan.output_offset, out_min, out_max);

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
                      weights,
                      bias,
                      /*output_zp=*/static_cast<int8_t>(-plan.output_offset),
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

    // Prefer CMSIS HWC repack when present; else use netkit CHW [C,Kh,Kw] with
    // flags=0 so XNNPACK packs via pack_dwconv_ghw_w (same as float depthwise).
    const int8_t* kernel = plan.weights_hwc ? plan.weights_hwc : weights_chw;
    const uint32_t flags = plan.weights_hwc ? XNN_FLAG_DEPTHWISE_CONVOLUTION : 0;

    int8_t out_min = -128;
    int8_t out_max = 127;
    ActivationClampS8(plan.clamp, plan.output_scale, -plan.output_offset, out_min, out_max);

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
                      kernel,
                      bias,
                      /*output_zp=*/static_cast<int8_t>(-plan.output_offset),
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
    if (plan.input_scale <= 0.0f || plan.weight_scale <= 0.0f || plan.output_scale <= 0.0f)
        return false;
    // XNNPACK qs8 FC expects filter zero-point 0 (symmetric weights).
    if (plan.filter_offset != 0)
        return false;
    if (!EnsureXnnInitialized())
        return false;

    int8_t out_min = -128;
    int8_t out_max = 127;
    ActivationClampS8(plan.clamp, plan.output_scale, -plan.output_offset, out_min, out_max);

    const size_t in_channels = static_cast<size_t>(plan.in_features);
    const size_t out_channels = static_cast<size_t>(plan.out_features);

    ScopedOp scoped;
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
            /*output_zero_point=*/static_cast<int8_t>(-plan.output_offset),
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

    // netkit depthwise weights are CHW [C,Kh,Kw] — use GHW packing (flags=0).
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
    if (quant.input_scale <= 0.0f || quant.weight_scale <= 0.0f || quant.output_scale <= 0.0f)
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

    ScopedOp scoped;
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
