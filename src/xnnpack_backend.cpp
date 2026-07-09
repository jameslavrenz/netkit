/*
 * XNNPACK adapter for netkit float32 conv / depthwise / pool / FC (NHWC).
 *
 * XNNPACK is BSD-3 — see third_party/XNNPACK/LICENSE.
 * Enabled on cpu/mpu when NETKIT_XNNPACK=1; MCU leaves this off.
 */
#include "xnnpack_kernel.hpp"
#include "netkit_config.h"

#if defined(NETKIT_USE_XNNPACK) && NETKIT_USE_XNNPACK && NETKIT_XNNPACK_ALLOWED

#include <xnnpack.h>

#include <cfloat>
#include <cstdint>
#include <mutex>
#include <vector>

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

    bool ActivationClamp(NetkitKernelActivation activation, float& out_min, float& out_max)
    {
        out_min = -FLT_MAX;
        out_max = FLT_MAX;
        switch (activation)
        {
            case NetkitKernelActivation::None:
                return true;
            case NetkitKernelActivation::ReLU:
                out_min = 0.0f;
                return true;
            case NetkitKernelActivation::ReLU6:
                out_min = 0.0f;
                out_max = 6.0f;
                return true;
            default:
                // Sigmoid/Tanh/LeakyReLU/GELU: let ComposedKernel fall back.
                return false;
        }
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
}

bool XnnpackKernel::TryConv2dForward(const Tensor& input,
                                     float* weights,
                                     float* bias,
                                     int kernel_size,
                                     int stride,
                                     int pad_h,
                                     int pad_w,
                                     int in_channels,
                                     int out_channels,
                                     NetkitKernelActivation fuse_activation,
                                     Tensor& output)
{
    if (!EnsureXnnInitialized() || !weights || input.rank != 3 || output.rank != 3)
        return false;
    if (static_cast<int>(input.shape[2]) != in_channels ||
        static_cast<int>(output.shape[2]) != out_channels)
        return false;

    float out_min = -FLT_MAX;
    float out_max = FLT_MAX;
    if (!ActivationClamp(fuse_activation, out_min, out_max))
        return false;

    // netkit conv weights are already [out, kh, kw, in] (OHWI), matching
    // XNNPACK's NHWC filter layout:
    //   [groups * group_output_channels, kh, kw, group_input_channels]

    ScopedOp scoped;
    if (xnn_create_convolution2d_nhwc_f32(
            /*input_padding_top=*/static_cast<uint32_t>(pad_h),
            /*input_padding_right=*/static_cast<uint32_t>(pad_w),
            /*input_padding_bottom=*/static_cast<uint32_t>(pad_h),
            /*input_padding_left=*/static_cast<uint32_t>(pad_w),
            /*kernel_height=*/static_cast<uint32_t>(kernel_size),
            /*kernel_width=*/static_cast<uint32_t>(kernel_size),
            /*subsampling_height=*/static_cast<uint32_t>(stride),
            /*subsampling_width=*/static_cast<uint32_t>(stride),
            /*dilation_height=*/1,
            /*dilation_width=*/1,
            /*groups=*/1,
            /*group_input_channels=*/static_cast<size_t>(in_channels),
            /*group_output_channels=*/static_cast<size_t>(out_channels),
            /*input_channel_stride=*/static_cast<size_t>(in_channels),
            /*output_channel_stride=*/static_cast<size_t>(out_channels),
            weights,
            bias,
            out_min,
            out_max,
            /*flags=*/0,
            /*weights_cache=*/nullptr,
            &scoped.op) != xnn_status_success)
        return false;

    size_t workspace_size = 0;
    size_t out_h = 0;
    size_t out_w = 0;
    if (xnn_reshape_convolution2d_nhwc_f32(scoped.op,
                                           /*batch_size=*/1,
                                           input.shape[0],
                                           input.shape[1],
                                           &workspace_size,
                                           &out_h,
                                           &out_w,
                                           /*threadpool=*/nullptr) != xnn_status_success)
        return false;

    if (out_h != output.shape[0] || out_w != output.shape[1])
        return false;

    std::vector<uint8_t> workspace_storage(workspace_size);
    void* workspace = workspace_size > 0 ? workspace_storage.data() : nullptr;

    if (xnn_setup_convolution2d_nhwc_f32(scoped.op,
                                         workspace,
                                         static_cast<const float*>(input.data),
                                         static_cast<float*>(output.data)) != xnn_status_success)
        return false;

    return xnn_run_operator(scoped.op, /*threadpool=*/nullptr) == xnn_status_success;
}

bool XnnpackKernel::TryDepthwiseConv2dForward(const Tensor& input,
                                              float* weights,
                                              float* bias,
                                              int kernel_h,
                                              int kernel_w,
                                              int stride,
                                              int pad_h,
                                              int pad_w,
                                              int channels,
                                              NetkitKernelActivation fuse_activation,
                                              Tensor& output)
{
    if (!EnsureXnnInitialized() || !weights || input.rank != 3 || output.rank != 3)
        return false;
    if (static_cast<int>(input.shape[2]) != channels || static_cast<int>(output.shape[2]) != channels)
        return false;

    float out_min = -FLT_MAX;
    float out_max = FLT_MAX;
    if (!ActivationClamp(fuse_activation, out_min, out_max))
        return false;

    // Depthwise: groups == channels, gic == 1, goc == 1.
    // netkit stores [C, Kh, Kw] (GHW). XNNPACK's dwconv path packs with
    // pack_dwconv_ghw_w when XNN_FLAG_DEPTHWISE_CONVOLUTION is unset; that flag
    // selects pack_dwconv_hwg_w ([Kh, Kw, C]) instead and would scramble weights.
    ScopedOp scoped;
    if (xnn_create_convolution2d_nhwc_f32(
            static_cast<uint32_t>(pad_h),
            static_cast<uint32_t>(pad_w),
            static_cast<uint32_t>(pad_h),
            static_cast<uint32_t>(pad_w),
            static_cast<uint32_t>(kernel_h),
            static_cast<uint32_t>(kernel_w),
            static_cast<uint32_t>(stride),
            static_cast<uint32_t>(stride),
            1,
            1,
            /*groups=*/static_cast<uint32_t>(channels),
            /*group_input_channels=*/1,
            /*group_output_channels=*/1,
            /*input_channel_stride=*/static_cast<size_t>(channels),
            /*output_channel_stride=*/static_cast<size_t>(channels),
            weights,
            bias,
            out_min,
            out_max,
            /*flags=*/0,
            /*weights_cache=*/nullptr,
            &scoped.op) != xnn_status_success)
        return false;

    size_t workspace_size = 0;
    size_t out_h = 0;
    size_t out_w = 0;
    if (xnn_reshape_convolution2d_nhwc_f32(scoped.op,
                                           1,
                                           input.shape[0],
                                           input.shape[1],
                                           &workspace_size,
                                           &out_h,
                                           &out_w,
                                           nullptr) != xnn_status_success)
        return false;
    if (out_h != output.shape[0] || out_w != output.shape[1])
        return false;

    std::vector<uint8_t> workspace_storage(workspace_size);
    void* workspace = workspace_size > 0 ? workspace_storage.data() : nullptr;

    if (xnn_setup_convolution2d_nhwc_f32(scoped.op,
                                         workspace,
                                         static_cast<const float*>(input.data),
                                         static_cast<float*>(output.data)) != xnn_status_success)
        return false;
    return xnn_run_operator(scoped.op, nullptr) == xnn_status_success;
}

bool XnnpackKernel::TryMaxPool2dForward(const Tensor& input,
                                        int pool_size,
                                        int stride,
                                        int pad_h,
                                        int pad_w,
                                        NetkitKernelActivation fuse_activation,
                                        Tensor& output)
{
    if (!EnsureXnnInitialized() || input.rank != 3 || output.rank != 3)
        return false;

    float out_min = -FLT_MAX;
    float out_max = FLT_MAX;
    if (!ActivationClamp(fuse_activation, out_min, out_max))
        return false;

    const int channels = static_cast<int>(input.shape[2]);
    ScopedOp scoped;
    if (xnn_create_max_pooling2d_nhwc_f32(
            static_cast<uint32_t>(pad_h),
            static_cast<uint32_t>(pad_w),
            static_cast<uint32_t>(pad_h),
            static_cast<uint32_t>(pad_w),
            static_cast<uint32_t>(pool_size),
            static_cast<uint32_t>(pool_size),
            static_cast<uint32_t>(stride),
            static_cast<uint32_t>(stride),
            1,
            1,
            out_min,
            out_max,
            /*flags=*/0,
            &scoped.op) != xnn_status_success)
        return false;

    size_t out_h = 0;
    size_t out_w = 0;
    if (xnn_reshape_max_pooling2d_nhwc_f32(scoped.op,
                                           1,
                                           input.shape[0],
                                           input.shape[1],
                                           static_cast<size_t>(channels),
                                           static_cast<size_t>(channels),
                                           static_cast<size_t>(channels),
                                           &out_h,
                                           &out_w,
                                           nullptr) != xnn_status_success)
        return false;
    if (out_h != output.shape[0] || out_w != output.shape[1])
        return false;

    if (xnn_setup_max_pooling2d_nhwc_f32(scoped.op,
                                         static_cast<const float*>(input.data),
                                         static_cast<float*>(output.data)) != xnn_status_success)
        return false;
    return xnn_run_operator(scoped.op, nullptr) == xnn_status_success;
}

bool XnnpackKernel::TryAvgPool2dForward(const Tensor& input,
                                        int pool_size,
                                        int stride,
                                        int pad_h,
                                        int pad_w,
                                        Tensor& output)
{
    if (!EnsureXnnInitialized() || input.rank != 3 || output.rank != 3)
        return false;

    const int channels = static_cast<int>(input.shape[2]);
    ScopedOp scoped;
    if (xnn_create_average_pooling2d_nhwc_f32(
            static_cast<uint32_t>(pad_h),
            static_cast<uint32_t>(pad_w),
            static_cast<uint32_t>(pad_h),
            static_cast<uint32_t>(pad_w),
            static_cast<uint32_t>(pool_size),
            static_cast<uint32_t>(pool_size),
            static_cast<uint32_t>(stride),
            static_cast<uint32_t>(stride),
            -FLT_MAX,
            FLT_MAX,
            /*flags=*/0,
            &scoped.op) != xnn_status_success)
        return false;

    size_t out_h = 0;
    size_t out_w = 0;
    if (xnn_reshape_average_pooling2d_nhwc_f32(scoped.op,
                                               1,
                                               input.shape[0],
                                               input.shape[1],
                                               static_cast<size_t>(channels),
                                               static_cast<size_t>(channels),
                                               static_cast<size_t>(channels),
                                               &out_h,
                                               &out_w,
                                               nullptr) != xnn_status_success)
        return false;
    if (out_h != output.shape[0] || out_w != output.shape[1])
        return false;

    if (xnn_setup_average_pooling2d_nhwc_f32(scoped.op,
                                             static_cast<const float*>(input.data),
                                             static_cast<float*>(output.data)) != xnn_status_success)
        return false;
    return xnn_run_operator(scoped.op, nullptr) == xnn_status_success;
}

bool XnnpackKernel::TryFullyConnectedWithBias(const Tensor& input,
                                              const Tensor& weights,
                                              const Tensor& bias,
                                              NetkitKernelActivation fuse_activation,
                                              Tensor& output)
{
    if (!EnsureXnnInitialized() || !input.data || !weights.data || !bias.data || !output.data)
        return false;

    float out_min = -FLT_MAX;
    float out_max = FLT_MAX;
    if (!ActivationClamp(fuse_activation, out_min, out_max))
        return false;

    // Dense layout: weights [out, in], bias [out] or [1, out], input [batch, in].
    if (input.rank != 2 || weights.rank != 2 || output.rank != 2 || !bias.data)
        return false;

    const size_t batch = input.shape[0];
    const size_t in_channels = weights.shape[1];
    const size_t out_channels = weights.shape[0];
    if (input.shape[1] != in_channels || output.shape[0] != batch ||
        output.shape[1] != out_channels)
        return false;

    const float* bias_ptr = static_cast<const float*>(bias.data);
    if (bias.rank == 1)
    {
        if (bias.shape[0] != out_channels)
            return false;
    }
    else if (bias.rank == 2)
    {
        // netkit packs dense bias as [1, out] (matches CMSIS-NN / reference).
        if (bias.shape[0] != 1 || bias.shape[1] != out_channels)
            return false;
    }
    else
    {
        return false;
    }

    ScopedOp scoped;
    if (xnn_create_fully_connected_nc_f32(in_channels,
                                          out_channels,
                                          /*input_stride=*/in_channels,
                                          /*output_stride=*/out_channels,
                                          static_cast<const float*>(weights.data),
                                          bias_ptr,
                                          out_min,
                                          out_max,
                                          /*flags=*/0,
                                          /*weights_cache=*/nullptr,
                                          &scoped.op) != xnn_status_success)
        return false;

    if (xnn_reshape_fully_connected_nc_f32(scoped.op, batch, nullptr) != xnn_status_success)
        return false;
    if (xnn_setup_fully_connected_nc_f32(scoped.op,
                                         static_cast<const float*>(input.data),
                                         static_cast<float*>(output.data)) != xnn_status_success)
        return false;
    return xnn_run_operator(scoped.op, nullptr) == xnn_status_success;
}

#else  // !NETKIT_USE_XNNPACK

bool XnnpackKernel::TryConv2dForward(const Tensor&,
                                     float*,
                                     float*,
                                     int,
                                     int,
                                     int,
                                     int,
                                     int,
                                     int,
                                     NetkitKernelActivation,
                                     Tensor&)
{
    return false;
}

bool XnnpackKernel::TryDepthwiseConv2dForward(const Tensor&,
                                              float*,
                                              float*,
                                              int,
                                              int,
                                              int,
                                              int,
                                              int,
                                              int,
                                              NetkitKernelActivation,
                                              Tensor&)
{
    return false;
}

bool XnnpackKernel::TryMaxPool2dForward(const Tensor&,
                                        int,
                                        int,
                                        int,
                                        int,
                                        NetkitKernelActivation,
                                        Tensor&)
{
    return false;
}

bool XnnpackKernel::TryAvgPool2dForward(const Tensor&, int, int, int, int, Tensor&)
{
    return false;
}

bool XnnpackKernel::TryFullyConnectedWithBias(const Tensor&,
                                              const Tensor&,
                                              const Tensor&,
                                              NetkitKernelActivation,
                                              Tensor&)
{
    return false;
}

#endif
