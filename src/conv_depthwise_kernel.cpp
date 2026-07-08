#include "conv_depthwise_kernel.hpp"

#include "kernel_activation.hpp"
#include "netkit_loop_unroll.hpp"
#include "tensor_access.hpp"

#include <cstddef>

namespace
{
    float ConvOutputValue(float sum, NetkitKernelActivation fuse_activation)
    {
        return ApplyKernelActivation(sum, fuse_activation);
    }
}

bool ConvDepthwiseForward(const Tensor& input,
                          float* weights,
                          float* bias,
                          int kernel_h,
                          int kernel_w,
                          int stride,
                          int pad_h,
                          int pad_w,
                          int /*pad_h_end*/,
                          int /*pad_w_end*/,
                          int channels,
                          NetkitKernelActivation fuse_activation,
                          Tensor& output)
{
    float* in = tensor_data_f32(const_cast<Tensor&>(input));
    float* out = tensor_data_f32(output);

    const uint32_t out_h = output.shape[0];
    const uint32_t out_w = output.shape[1];
    const int in_h = static_cast<int>(input.shape[0]);
    const int in_w = static_cast<int>(input.shape[1]);
    const uint32_t in_w_u = input.shape[1];
    const uint32_t ch_u = static_cast<uint32_t>(channels);

    const uint32_t kernel_h_u = static_cast<uint32_t>(kernel_h);
    const uint32_t kernel_w_u = static_cast<uint32_t>(kernel_w);

    for (size_t oh = 0; oh < out_h; ++oh)
    {
        // Kernel rows that land inside the input (padding clipped to a contiguous range),
        // so the inner reduction runs with no per-tap bounds branch.
        const int base_h = static_cast<int>(oh) * stride - pad_h;
        const int kh_lo = base_h < 0 ? -base_h : 0;
        int kh_hi = in_h - base_h;
        if (kh_hi > kernel_h)
            kh_hi = kernel_h;

        for (size_t ow = 0; ow < out_w; ++ow)
        {
            const int base_w = static_cast<int>(ow) * stride - pad_w;
            const int kw_lo = base_w < 0 ? -base_w : 0;
            int kw_hi = in_w - base_w;
            if (kw_hi > kernel_w)
                kw_hi = kernel_w;
            const uint32_t kw_count = (kw_hi > kw_lo) ? static_cast<uint32_t>(kw_hi - kw_lo) : 0u;

            const uint32_t out_spatial_base = (oh * out_w + ow) * ch_u;

            for (int c = 0; c < channels; ++c)
            {
                const uint32_t c_u = static_cast<uint32_t>(c);

                // Independent accumulators (rows round-robined across 4) break the
                // cross-row serial dependency; each row's tap reduction uses the
                // header-inline 4-accumulator dot_strided (input stride = channels,
                // weight stride = 1).
                float s0 = 0.0f;
                float s1 = 0.0f;
                float s2 = 0.0f;
                float s3 = 0.0f;

                if (kw_count > 0u)
                {
                    for (int kh = kh_lo; kh < kh_hi; ++kh)
                    {
                        const uint32_t in_row = static_cast<uint32_t>(base_h + kh) * in_w_u;
                        const uint32_t in_base =
                            (in_row + static_cast<uint32_t>(base_w + kw_lo)) * ch_u + c_u;
                        const uint32_t w_base =
                            (c_u * kernel_h_u + static_cast<uint32_t>(kh)) * kernel_w_u +
                            static_cast<uint32_t>(kw_lo);
                        const float rowsum = NetkitLoopUnroll::dot_strided(
                            in + in_base, ch_u, weights + w_base, 1u, kw_count);
                        switch (kh & 3)
                        {
                            case 0:
                                s0 += rowsum;
                                break;
                            case 1:
                                s1 += rowsum;
                                break;
                            case 2:
                                s2 += rowsum;
                                break;
                            default:
                                s3 += rowsum;
                                break;
                        }
                    }
                }

                const float sum = (bias ? bias[c] : 0.0f) + ((s0 + s1) + (s2 + s3));
                out[out_spatial_base + c_u] = ConvOutputValue(sum, fuse_activation);
            }
        }
    }

    return kernel_activation_is_fused(fuse_activation);
}
