#include "cmsis_buffer_size.hpp"

#include "netkit_config.h"

namespace
{
    std::size_t* g_max_kernel_workspace_bytes = nullptr;

    inline void BumpMaxKernelWorkspace(std::size_t bytes)
    {
        if (!g_max_kernel_workspace_bytes)
            return;

        if (bytes > *g_max_kernel_workspace_bytes)
            *g_max_kernel_workspace_bytes = bytes;
    }
}

#if defined(NETKIT_USE_CMSIS_NN) && NETKIT_USE_CMSIS_NN && NETKIT_CMSIS_NN_ALLOWED

#include <arm_nnfunctions.h>

#include <cfloat>

std::size_t CmsisConv2dWorkspaceBytes(uint32_t in_h,
                                      uint32_t in_w,
                                      int kernel_size,
                                      int stride,
                                      int pad_h,
                                      int pad_w,
                                      int in_channels,
                                      int out_channels)
{
    const uint32_t out_h =
        static_cast<uint32_t>((static_cast<int>(in_h) + 2 * pad_h - kernel_size) / stride + 1);
    const uint32_t out_w =
        static_cast<uint32_t>((static_cast<int>(in_w) + 2 * pad_w - kernel_size) / stride + 1);

    const cmsis_nn_conv_params_f32 conv_params = {
        .stride = {.w = stride, .h = stride},
        .padding = {.w = pad_w, .h = pad_h},
        .dilation = {.w = 1, .h = 1},
        .activation = {.min = -FLT_MAX, .max = FLT_MAX},
        .weight_format = ARM_NN_WEIGHT_FORMAT_STANDARD,
    };

    const cmsis_nn_dims input_dims = {
        .n = 1,
        .h = static_cast<int32_t>(in_h),
        .w = static_cast<int32_t>(in_w),
        .c = in_channels,
    };
    const cmsis_nn_dims filter_dims = {
        .n = out_channels,
        .h = kernel_size,
        .w = kernel_size,
        .c = in_channels,
    };
    const cmsis_nn_dims output_dims = {
        .n = 1,
        .h = static_cast<int32_t>(out_h),
        .w = static_cast<int32_t>(out_w),
        .c = out_channels,
    };

    const int32_t buf_size =
        arm_convolve_wrapper_f32_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    return buf_size > 0 ? static_cast<std::size_t>(buf_size) : 0u;
}

std::size_t CmsisDepthwiseConv2dWorkspaceBytes(uint32_t in_h,
                                               uint32_t in_w,
                                               int kernel_h,
                                               int kernel_w,
                                               int stride,
                                               int pad_h,
                                               int pad_w,
                                               int channels)
{
    const uint32_t out_h =
        static_cast<uint32_t>((static_cast<int>(in_h) + 2 * pad_h - kernel_h) / stride + 1);
    const uint32_t out_w =
        static_cast<uint32_t>((static_cast<int>(in_w) + 2 * pad_w - kernel_w) / stride + 1);

    const cmsis_nn_dw_conv_params_f32 dw_conv_params = {
        .ch_mult = 1,
        .stride = {.w = stride, .h = stride},
        .padding = {.w = pad_w, .h = pad_h},
        .dilation = {.w = 1, .h = 1},
        .activation = {.min = -FLT_MAX, .max = FLT_MAX},
    };

    const cmsis_nn_dims input_dims = {
        .n = 1,
        .h = static_cast<int32_t>(in_h),
        .w = static_cast<int32_t>(in_w),
        .c = channels,
    };
    const cmsis_nn_dims filter_dims = {
        .n = channels,
        .h = kernel_h,
        .w = kernel_w,
        .c = channels,
    };
    const cmsis_nn_dims output_dims = {
        .n = 1,
        .h = static_cast<int32_t>(out_h),
        .w = static_cast<int32_t>(out_w),
        .c = channels,
    };

    const int32_t buf_size = arm_depthwise_conv_wrapper_f32_get_buffer_size(
        &dw_conv_params, &input_dims, &filter_dims, &output_dims);
    return buf_size > 0 ? static_cast<std::size_t>(buf_size) : 0u;
}

std::size_t CmsisGeluWorkspaceBytes(uint32_t num_elements)
{
    return static_cast<std::size_t>(num_elements) * sizeof(float);
}

#else

std::size_t CmsisConv2dWorkspaceBytes(uint32_t /*in_h*/,
                                      uint32_t /*in_w*/,
                                      int /*kernel_size*/,
                                      int /*stride*/,
                                      int /*pad_h*/,
                                      int /*pad_w*/,
                                      int /*in_channels*/,
                                      int /*out_channels*/)
{
    return 0;
}

std::size_t CmsisDepthwiseConv2dWorkspaceBytes(uint32_t /*in_h*/,
                                               uint32_t /*in_w*/,
                                               int /*kernel_h*/,
                                               int /*kernel_w*/,
                                               int /*stride*/,
                                               int /*pad_h*/,
                                               int /*pad_w*/,
                                               int /*channels*/)
{
    return 0;
}

std::size_t CmsisGeluWorkspaceBytes(uint32_t /*num_elements*/)
{
    return 0;
}

#endif

void CmsisBeginKernelWorkspacePlan(std::size_t* max_out)
{
    g_max_kernel_workspace_bytes = max_out;
    if (max_out)
        *max_out = 0;
}

void CmsisEndKernelWorkspacePlan()
{
    g_max_kernel_workspace_bytes = nullptr;
}

void CmsisBumpConv2dWorkspace(uint32_t in_h,
                              uint32_t in_w,
                              int kernel_size,
                              int stride,
                              int pad_h,
                              int pad_w,
                              int in_channels,
                              int out_channels)
{
    BumpMaxKernelWorkspace(CmsisConv2dWorkspaceBytes(
        in_h, in_w, kernel_size, stride, pad_h, pad_w, in_channels, out_channels));
}

void CmsisBumpDepthwiseConv2dWorkspace(uint32_t in_h,
                                       uint32_t in_w,
                                       int kernel_h,
                                       int kernel_w,
                                       int stride,
                                       int pad_h,
                                       int pad_w,
                                       int channels)
{
    BumpMaxKernelWorkspace(CmsisDepthwiseConv2dWorkspaceBytes(
        in_h, in_w, kernel_h, kernel_w, stride, pad_h, pad_w, channels));
}

void CmsisBumpGeluWorkspace(uint32_t num_elements)
{
    BumpMaxKernelWorkspace(CmsisGeluWorkspaceBytes(num_elements));
}
