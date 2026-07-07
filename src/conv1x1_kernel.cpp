#include "conv1x1_kernel.hpp"

#include "cmsis_dsp_util.hpp"
#include "kernel_activation.hpp"

namespace
{
    float ConvOutputValue(float sum, NetkitKernelActivation fuse_activation)
    {
        return ApplyKernelActivation(sum, fuse_activation);
    }
}

bool Conv1x1Forward(const float* in,
                    const float* weights_oki,
                    const float* bias,
                    float* out,
                    uint32_t in_h,
                    uint32_t in_w,
                    uint32_t in_ch,
                    uint32_t out_h,
                    uint32_t out_w,
                    uint32_t out_ch,
                    int out_channels,
                    NetkitKernelActivation fuse_activation)
{
    (void)in_h;

    for (uint32_t oh = 0; oh < out_h; ++oh)
    {
        for (uint32_t ow = 0; ow < out_w; ++ow)
        {
            const uint32_t in_base = (oh * in_w + ow) * in_ch;
            const uint32_t out_spatial_base = (oh * out_w + ow) * out_ch;

            for (int oc = 0; oc < out_channels; ++oc)
            {
                const float* wt_row = weights_oki + static_cast<uint32_t>(oc) * in_ch;
                const float b = bias ? bias[oc] : 0.0f;
                const float sum = b + CmsisDspUtil::DotProductF32(in + in_base, wt_row, in_ch);
                out[out_spatial_base + static_cast<uint32_t>(oc)] =
                    ConvOutputValue(sum, fuse_activation);
            }
        }
    }

    return kernel_activation_is_fused(fuse_activation);
}
