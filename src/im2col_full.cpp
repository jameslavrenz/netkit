#include "im2col_full.hpp"

#include "active_kernel.hpp"
#include "kernel_activation.hpp"
#include "tensor.hpp"

namespace
{
    float ConvOutputValue(float sum, NetkitKernelActivation fuse_activation)
    {
        return ApplyKernelActivation(sum, fuse_activation);
    }

    void im2col_nhwc(const float* in,
                     float* col,
                     uint32_t in_h,
                     uint32_t in_w,
                     uint32_t in_ch,
                     uint32_t out_h,
                     uint32_t out_w,
                     uint32_t kernel_h,
                     uint32_t kernel_w,
                     int stride,
                     int pad_h,
                     int pad_w)
    {
        const uint32_t patch_elems = kernel_h * kernel_w * in_ch;

        for (uint32_t oh = 0; oh < out_h; ++oh)
        {
            for (uint32_t ow = 0; ow < out_w; ++ow)
            {
                const uint32_t spatial_idx = oh * out_w + ow;
                uint32_t k_idx = 0;

                for (uint32_t kh = 0; kh < kernel_h; ++kh)
                {
                    const int ih = static_cast<int>(oh) * stride + static_cast<int>(kh) - pad_h;

                    for (uint32_t kw = 0; kw < kernel_w; ++kw)
                    {
                        const int iw = static_cast<int>(ow) * stride + static_cast<int>(kw) - pad_w;
                        const bool in_bounds = ih >= 0 && ih < static_cast<int>(in_h) && iw >= 0 &&
                                               iw < static_cast<int>(in_w);

                        for (uint32_t ic = 0; ic < in_ch; ++ic)
                        {
                            float value = 0.0f;
                            if (in_bounds)
                            {
                                value =
                                    in[(static_cast<uint32_t>(ih) * in_w + static_cast<uint32_t>(iw)) *
                                           in_ch +
                                       ic];
                            }
                            col[spatial_idx * patch_elems + k_idx++] = value;
                        }
                    }
                }
            }
        }
    }
}

std::size_t ConvFullIm2ColWorkspaceBytes(uint32_t out_h,
                                         uint32_t out_w,
                                         uint32_t kernel_h,
                                         uint32_t kernel_w,
                                         uint32_t in_channels)
{
    const uint32_t patch = kernel_h * kernel_w * in_channels;
    const uint32_t spatial = out_h * out_w;
    if (patch == 0 || spatial == 0)
        return 0;

    return static_cast<std::size_t>(patch) * static_cast<std::size_t>(spatial) * sizeof(float);
}

bool ConvFullIm2ColForward(const float* in,
                           const float* weights_oki,
                           const float* bias,
                           float* out,
                           float* workspace,
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
    const uint32_t out_spatial = out_h * out_w;

    im2col_nhwc(in,
                workspace,
                in_h,
                in_w,
                in_ch,
                out_h,
                out_w,
                kernel_h,
                kernel_w,
                stride,
                pad_h,
                pad_w);

    Tensor col{};
    col.data = workspace;
    col.type = DataType::Float32;
    col.rank = 2;
    col.shape = {out_spatial, patch_elems};
    col.stride = {patch_elems, 1};
    col.num_elements = out_spatial * patch_elems;
    col.bytes = col.num_elements * sizeof(float);

    Tensor wt{};
    wt.data = const_cast<float*>(weights_oki);
    wt.type = DataType::Float32;
    wt.rank = 2;
    wt.shape = {patch_elems, static_cast<uint32_t>(out_channels)};
    wt.stride = {1, patch_elems};
    wt.num_elements = patch_elems * static_cast<uint32_t>(out_channels);
    wt.bytes = wt.num_elements * sizeof(float);

    Tensor output{};
    output.data = out;
    output.type = DataType::Float32;
    output.rank = 2;
    output.shape = {out_spatial, out_ch};
    output.stride = {out_ch, 1};
    output.num_elements = out_spatial * out_ch;
    output.bytes = output.num_elements * sizeof(float);

    Kernels::MatMulImpl(col, wt, output);

    for (uint32_t s = 0; s < out_spatial; ++s)
    {
        for (int oc = 0; oc < out_channels; ++oc)
        {
            const uint32_t idx = s * out_ch + static_cast<uint32_t>(oc);
            float sum = out[idx];
            if (bias)
            {
                sum += bias[oc];
            }
            out[idx] = ConvOutputValue(sum, fuse_activation);
        }
    }

    return kernel_activation_is_fused(fuse_activation);
}
