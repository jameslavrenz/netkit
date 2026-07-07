#include "im2col_partial.hpp"

#include "kernel_activation.hpp"
#include "netkit_config.h"
#include "netkit_loop_unroll.hpp"

#if defined(NETKIT_USE_CMSIS_DSP) && NETKIT_USE_CMSIS_DSP
#include <arm_math.h>
#endif

namespace
{
    float dot_contiguous_f32(const float* a, const float* b, uint32_t count)
    {
#if defined(NETKIT_USE_CMSIS_DSP) && NETKIT_USE_CMSIS_DSP
        float sum = 0.0f;
        arm_dot_prod_f32(const_cast<float*>(a), const_cast<float*>(b), count, &sum);
        return sum;
#else
        return NetkitLoopUnroll::dot_contiguous(a, b, count);
#endif
    }
    float ConvOutputValue(float sum, NetkitKernelActivation fuse_activation)
    {
        return ApplyKernelActivation(sum, fuse_activation);
    }

    void im2col_patch_nhwc(const float* in,
                           float* col,
                           uint32_t in_h,
                           uint32_t in_w,
                           uint32_t in_ch,
                           uint32_t oh,
                           uint32_t ow,
                           uint32_t kernel_h,
                           uint32_t kernel_w,
                           int stride,
                           int pad_h,
                           int pad_w)
    {
        uint32_t k_idx = 0;

        for (uint32_t kh = 0; kh < kernel_h; ++kh)
        {
            const int ih = static_cast<int>(oh) * stride + static_cast<int>(kh) - pad_h;

            for (uint32_t kw = 0; kw < kernel_w; ++kw)
            {
                const int iw = static_cast<int>(ow) * stride + static_cast<int>(kw) - pad_w;
                const bool in_bounds =
                    ih >= 0 && ih < static_cast<int>(in_h) && iw >= 0 && iw < static_cast<int>(in_w);

                for (uint32_t ic = 0; ic < in_ch; ++ic)
                {
                    float value = 0.0f;
                    if (in_bounds)
                    {
                        value = in[(static_cast<uint32_t>(ih) * in_w + static_cast<uint32_t>(iw)) * in_ch +
                                   ic];
                    }
                    col[k_idx++] = value;
                }
            }
        }
    }
}

std::size_t ConvPartialIm2ColWorkspaceBytes(uint32_t kernel_h,
                                            uint32_t kernel_w,
                                            uint32_t in_channels)
{
    const uint32_t patch = kernel_h * kernel_w * in_channels;
    if (patch == 0)
        return 0;

    return static_cast<std::size_t>(patch) * sizeof(float);
}

bool ConvPartialIm2ColForward(const float* in,
                              const float* weights_oki,
                              const float* bias,
                              float* out,
                              float* patch_workspace,
                              uint32_t in_h,
                              uint32_t in_w,
                              uint32_t in_ch,
                              uint32_t out_h,
                              uint32_t out_w,
                              uint32_t out_ch,
                              uint32_t kernel_h,
                              uint32_t kernel_w,
                              int stride,
                              int pad_h,
                              int pad_w,
                              int pad_h_end,
                              int pad_w_end,
                              int out_channels,
                              NetkitKernelActivation fuse_activation)
{
    (void)pad_h_end;
    (void)pad_w_end;

    const uint32_t patch_elems = kernel_h * kernel_w * in_ch;

    for (uint32_t oh = 0; oh < out_h; ++oh)
    {
        for (uint32_t ow = 0; ow < out_w; ++ow)
        {
            const uint32_t spatial_idx = oh * out_w + ow;

            im2col_patch_nhwc(in,
                              patch_workspace,
                              in_h,
                              in_w,
                              in_ch,
                              oh,
                              ow,
                              kernel_h,
                              kernel_w,
                              stride,
                              pad_h,
                              pad_w);

            for (int oc = 0; oc < out_channels; ++oc)
            {
                const float* wt_row = weights_oki + static_cast<uint32_t>(oc) * patch_elems;
                const float b = bias ? bias[oc] : 0.0f;
                const float sum =
                    b + dot_contiguous_f32(wt_row, patch_workspace, patch_elems);
                out[spatial_idx * out_ch + static_cast<uint32_t>(oc)] =
                    ConvOutputValue(sum, fuse_activation);
            }
        }
    }

    return kernel_activation_is_fused(fuse_activation);
}
