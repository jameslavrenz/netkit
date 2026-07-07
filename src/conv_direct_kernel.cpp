#include "conv_direct_kernel.hpp"

#include "cmsis_dsp_util.hpp"
#include "kernel_activation.hpp"
#include "netkit_loop_unroll.hpp"

namespace
{
    constexpr uint32_t kInputStationaryMinOutChannels = 16u;

    float ConvOutputValue(float sum, NetkitKernelActivation fuse_activation)
    {
        return ApplyKernelActivation(sum, fuse_activation);
    }
}

bool ConvDirectForward3x3S1P0(const float* in,
                              float* weights,
                              const float* bias,
                              float* out,
                              uint32_t in_w,
                              uint32_t in_ch,
                              uint32_t out_h,
                              uint32_t out_w,
                              uint32_t out_ch,
                              int out_channels,
                              NetkitKernelActivation fuse_activation)
{
    const uint32_t filter_elems = 9u * in_ch;

    for (uint32_t oh = 0; oh < out_h; ++oh)
    {
        const uint32_t ih0 = oh;
        const uint32_t in_row0 = ih0 * in_w;
        const uint32_t in_row1 = (ih0 + 1u) * in_w;
        const uint32_t in_row2 = (ih0 + 2u) * in_w;

        for (uint32_t ow = 0; ow < out_w; ++ow)
        {
            const uint32_t iw0 = ow;
            const uint32_t out_spatial_base = (oh * out_w + ow) * out_ch;

            for (int oc = 0; oc < out_channels; ++oc)
            {
                const float* filter = weights + static_cast<uint32_t>(oc) * filter_elems;
                float sum = bias ? bias[oc] : 0.0f;

                sum += CmsisDspUtil::DotProductF32(
                    in + (in_row0 + iw0) * in_ch, filter + 0u * in_ch, in_ch);
                sum += CmsisDspUtil::DotProductF32(
                    in + (in_row0 + iw0 + 1u) * in_ch, filter + 1u * in_ch, in_ch);
                sum += CmsisDspUtil::DotProductF32(
                    in + (in_row0 + iw0 + 2u) * in_ch, filter + 2u * in_ch, in_ch);
                sum += CmsisDspUtil::DotProductF32(
                    in + (in_row1 + iw0) * in_ch, filter + 3u * in_ch, in_ch);
                sum += CmsisDspUtil::DotProductF32(
                    in + (in_row1 + iw0 + 1u) * in_ch, filter + 4u * in_ch, in_ch);
                sum += CmsisDspUtil::DotProductF32(
                    in + (in_row1 + iw0 + 2u) * in_ch, filter + 5u * in_ch, in_ch);
                sum += CmsisDspUtil::DotProductF32(
                    in + (in_row2 + iw0) * in_ch, filter + 6u * in_ch, in_ch);
                sum += CmsisDspUtil::DotProductF32(
                    in + (in_row2 + iw0 + 1u) * in_ch, filter + 7u * in_ch, in_ch);
                sum += CmsisDspUtil::DotProductF32(
                    in + (in_row2 + iw0 + 2u) * in_ch, filter + 8u * in_ch, in_ch);

                out[out_spatial_base + static_cast<uint32_t>(oc)] =
                    ConvOutputValue(sum, fuse_activation);
            }
        }
    }

    return kernel_activation_is_fused(fuse_activation);
}

bool ConvDirectForwardNoPad(const float* in,
                            float* weights,
                            const float* bias,
                            float* out,
                            uint32_t in_w,
                            uint32_t in_ch,
                            uint32_t out_h,
                            uint32_t out_w,
                            uint32_t out_ch,
                            int kernel_size,
                            int stride,
                            int out_channels,
                            NetkitKernelActivation fuse_activation)
{
    const uint32_t k_kernel = static_cast<uint32_t>(kernel_size);
    const uint32_t filter_spatial = k_kernel * k_kernel;

    for (uint32_t oh = 0; oh < out_h; ++oh)
    {
        const uint32_t ih_base = static_cast<uint32_t>(oh) * static_cast<uint32_t>(stride);

        for (uint32_t ow = 0; ow < out_w; ++ow)
        {
            const uint32_t iw_base = static_cast<uint32_t>(ow) * static_cast<uint32_t>(stride);
            const uint32_t out_spatial_base = (oh * out_w + ow) * out_ch;

            for (int oc = 0; oc < out_channels; ++oc)
            {
                float sum = bias ? bias[oc] : 0.0f;
                const uint32_t oc_u = static_cast<uint32_t>(oc);

                for (uint32_t kh = 0; kh < k_kernel; ++kh)
                {
                    const uint32_t in_row = (ih_base + kh) * in_w;

                    for (uint32_t kw = 0; kw < k_kernel; ++kw)
                    {
                        const uint32_t in_base = (in_row + iw_base + kw) * in_ch;
                        const uint32_t w_base = (oc_u * filter_spatial + kh * k_kernel + kw) * in_ch;
                        sum += CmsisDspUtil::DotProductF32(in + in_base, weights + w_base, in_ch);
                    }
                }

                out[out_spatial_base + oc_u] = ConvOutputValue(sum, fuse_activation);
            }
        }
    }

    return kernel_activation_is_fused(fuse_activation);
}

bool ConvDirectForwardPadded(const float* in,
                             float* weights,
                             const float* bias,
                             float* out,
                             uint32_t in_h,
                             uint32_t in_w,
                             uint32_t in_ch,
                             uint32_t out_h,
                             uint32_t out_w,
                             uint32_t out_ch,
                             int kernel_size,
                             int stride,
                             int pad_h,
                             int pad_w,
                             int out_channels,
                             NetkitKernelActivation fuse_activation)
{
    const int in_h_i = static_cast<int>(in_h);
    const int in_w_i = static_cast<int>(in_w);

    for (uint32_t oh = 0; oh < out_h; oh++)
    {
        for (uint32_t ow = 0; ow < out_w; ow++)
        {
            const uint32_t out_spatial_base = (oh * out_w + ow) * out_ch;

            for (int oc = 0; oc < out_channels; oc++)
            {
                float sum = bias ? bias[oc] : 0.0f;

                for (int kh = 0; kh < kernel_size; kh++)
                {
                    const int ih = static_cast<int>(oh) * stride + kh - pad_h;
                    if (ih < 0 || ih >= in_h_i)
                        continue;

                    const uint32_t in_row = static_cast<uint32_t>(ih) * in_w;

                    for (int kw = 0; kw < kernel_size; kw++)
                    {
                        const int iw = static_cast<int>(ow) * stride + kw - pad_w;
                        if (iw < 0 || iw >= in_w_i)
                            continue;

                        const uint32_t in_base = (in_row + static_cast<uint32_t>(iw)) * in_ch;
                        const uint32_t w_base =
                            ((static_cast<uint32_t>(oc) * static_cast<uint32_t>(kernel_size) +
                              static_cast<uint32_t>(kh)) *
                                 static_cast<uint32_t>(kernel_size) +
                             static_cast<uint32_t>(kw)) *
                            in_ch;

                        sum += CmsisDspUtil::DotProductF32(in + in_base, weights + w_base, in_ch);
                    }
                }

                out[out_spatial_base + static_cast<uint32_t>(oc)] = ConvOutputValue(sum, fuse_activation);
            }
        }
    }

    return kernel_activation_is_fused(fuse_activation);
}

bool ConvDirectTryInputStationaryForward(const float* in,
                                         const float* weights_hwio,
                                         const float* bias,
                                         float* out,
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

    if (!weights_hwio || out_channels < static_cast<int>(kInputStationaryMinOutChannels))
        return false;

    for (uint32_t oh = 0; oh < out_h; ++oh)
    {
        for (uint32_t ow = 0; ow < out_w; ++ow)
        {
            const uint32_t out_spatial_base = (oh * out_w + ow) * out_ch;

            for (int oc = 0; oc < out_channels; ++oc)
            {
                out[out_spatial_base + static_cast<uint32_t>(oc)] = bias ? bias[oc] : 0.0f;
            }

            for (uint32_t kh = 0; kh < kernel_h; ++kh)
            {
                const int ih = static_cast<int>(oh) * stride + static_cast<int>(kh) - pad_h;

                for (uint32_t kw = 0; kw < kernel_w; ++kw)
                {
                    const int iw = static_cast<int>(ow) * stride + static_cast<int>(kw) - pad_w;
                    if (ih < 0 || ih >= static_cast<int>(in_h) || iw < 0 ||
                        iw >= static_cast<int>(in_w))
                    {
                        continue;
                    }

                    const uint32_t in_base =
                        (static_cast<uint32_t>(ih) * in_w + static_cast<uint32_t>(iw)) * in_ch;

                    for (uint32_t ic = 0; ic < in_ch; ++ic)
                    {
                        const float value = in[in_base + ic];
                        if (value == 0.0f)
                            continue;

                        const float* w_slice =
                            weights_hwio + ((kh * kernel_w + kw) * in_ch + ic) * out_ch;

                        for (uint32_t oc = 0; oc < out_ch; ++oc)
                        {
                            out[out_spatial_base + oc] += value * w_slice[oc];
                        }
                    }
                }
            }

            for (int oc = 0; oc < out_channels; ++oc)
            {
                out[out_spatial_base + static_cast<uint32_t>(oc)] =
                    ConvOutputValue(out[out_spatial_base + static_cast<uint32_t>(oc)], fuse_activation);
            }
        }
    }

    return kernel_activation_is_fused(fuse_activation);
}
