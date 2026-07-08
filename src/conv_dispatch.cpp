#include "conv_dispatch.hpp"

#include "conv1x1_kernel.hpp"
#include "conv_direct_kernel.hpp"
#include "im2col_full.hpp"
#include "im2col_partial.hpp"
#include "kernel_activation.hpp"
#include "kernel_workspace.hpp"
#include "netkit_config.h"
#include "tensor.hpp"
#include "tensor_access.hpp"

namespace
{
    constexpr uint32_t kIm2ColMinPatchVolume = 2048u;
#if NETKIT_IM2COL >= 2
    constexpr uint32_t kIm2ColFullMinPatchVolume = 32768u;
#endif

    bool conv_padding_zero(int pad_h, int pad_w, int pad_h_end, int pad_w_end)
    {
        return pad_h == 0 && pad_w == 0 && pad_h_end == 0 && pad_w_end == 0;
    }

    bool conv_volume_warrants_im2col(uint32_t kernel_h,
                                     uint32_t kernel_w,
                                     uint32_t in_channels,
                                     uint32_t out_h,
                                     uint32_t out_w)
    {
        const uint32_t patch = kernel_h * kernel_w * in_channels;
        const uint32_t spatial = out_h * out_w;
        return patch > 0 && spatial > 0 && patch * spatial >= kIm2ColMinPatchVolume;
    }

    bool Conv2dRunDirect(const float* in,
                         float* weights,
                         const float* weights_hwio,
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
                         int pad_h_end,
                         int pad_w_end,
                         int out_channels,
                         NetkitKernelActivation fuse_activation)
    {
        if (kernel_size == 1 && stride == 1)
        {
            return Conv1x1Forward(in,
                                  weights,
                                  bias,
                                  out,
                                  in_h,
                                  in_w,
                                  in_ch,
                                  out_h,
                                  out_w,
                                  out_ch,
                                  out_channels,
                                  fuse_activation);
        }

        if (conv_padding_zero(pad_h, pad_w, pad_h_end, pad_w_end))
        {
            if (kernel_size == 3 && stride == 1)
            {
                return ConvDirectForward3x3S1P0(in,
                                                weights,
                                                bias,
                                                out,
                                                in_w,
                                                in_ch,
                                                out_h,
                                                out_w,
                                                out_ch,
                                                out_channels,
                                                fuse_activation);
            }

            if (ConvDirectTryInputStationaryForward(in,
                                                    weights_hwio,
                                                    bias,
                                                    out,
                                                    in_h,
                                                    in_w,
                                                    in_ch,
                                                    out_h,
                                                    out_w,
                                                    out_ch,
                                                    static_cast<uint32_t>(kernel_size),
                                                    static_cast<uint32_t>(kernel_size),
                                                    stride,
                                                    pad_h,
                                                    pad_w,
                                                    pad_h_end,
                                                    pad_w_end,
                                                    out_channels,
                                                    fuse_activation))
            {
                return kernel_activation_is_fused(fuse_activation);
            }

            return ConvDirectForwardNoPad(in,
                                          weights,
                                          bias,
                                          out,
                                          in_w,
                                          in_ch,
                                          out_h,
                                          out_w,
                                          out_ch,
                                          kernel_size,
                                          stride,
                                          out_channels,
                                          fuse_activation);
        }

        if (ConvDirectTryInputStationaryForward(in,
                                                weights_hwio,
                                                bias,
                                                out,
                                                in_h,
                                                in_w,
                                                in_ch,
                                                out_h,
                                                out_w,
                                                out_ch,
                                                static_cast<uint32_t>(kernel_size),
                                                static_cast<uint32_t>(kernel_size),
                                                stride,
                                                pad_h,
                                                pad_w,
                                                pad_h_end,
                                                pad_w_end,
                                                out_channels,
                                                fuse_activation))
        {
            return kernel_activation_is_fused(fuse_activation);
        }

        return ConvDirectForwardPadded(in,
                                       weights,
                                       bias,
                                       out,
                                       in_h,
                                       in_w,
                                       in_ch,
                                       out_h,
                                       out_w,
                                       out_ch,
                                       kernel_size,
                                       stride,
                                       pad_h,
                                       pad_w,
                                       out_channels,
                                       fuse_activation);
    }

    bool Conv2dTryPartialIm2Col(const float* in,
                                const float* weights,
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
        KernelWorkspace* workspace = GetActiveKernelWorkspace();
        if (!workspace || !workspace->data)
            return false;

        const std::size_t required = ConvPartialIm2ColWorkspaceBytes(kernel_h, kernel_w, in_ch);
        if (required == 0 || workspace->size_bytes < required)
            return false;

        ConvPartialIm2ColForward(in,
                                 weights,
                                 bias,
                                 out,
                                 reinterpret_cast<float*>(workspace->data),
                                 in_h,
                                 in_w,
                                 in_ch,
                                 out_h,
                                 out_w,
                                 out_ch,
                                 kernel_h,
                                 kernel_w,
                                 stride,
                                 pad_h,
                                 pad_w,
                                 pad_h_end,
                                 pad_w_end,
                                 out_channels,
                                 fuse_activation);
        return true;
    }

    bool Conv2dTryFullIm2Col(const float* in,
                             const float* weights,
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
        KernelWorkspace* workspace = GetActiveKernelWorkspace();
        if (!workspace || !workspace->data)
            return false;

        const std::size_t required =
            ConvFullIm2ColWorkspaceBytes(out_h, out_w, kernel_h, kernel_w, in_ch);
        if (required == 0 || workspace->size_bytes < required)
            return false;

        ConvFullIm2ColForward(in,
                              weights,
                              bias,
                              out,
                              reinterpret_cast<float*>(workspace->data),
                              in_h,
                              in_w,
                              in_ch,
                              out_h,
                              out_w,
                              out_ch,
                              kernel_h,
                              kernel_w,
                              stride,
                              pad_h,
                              pad_w,
                              pad_h_end,
                              pad_w_end,
                              out_channels,
                              fuse_activation);
        return true;
    }
}

Conv2dExecMode SelectConv2dExecMode(int kernel_h,
                                    int kernel_w,
                                    int stride,
                                    uint32_t in_channels,
                                    uint32_t out_h,
                                    uint32_t out_w)
{
    if (kernel_h == 1 && kernel_w == 1 && stride == 1)
        return Conv2dExecMode::Direct;

#if NETKIT_IM2COL >= 1
    const uint32_t kh = static_cast<uint32_t>(kernel_h);
    const uint32_t kw = static_cast<uint32_t>(kernel_w);
    const bool large_volume =
        conv_volume_warrants_im2col(kh, kw, in_channels, out_h, out_w);
#if NETKIT_IM2COL >= 2
    const uint32_t patch = kh * kw * in_channels;
    const uint32_t spatial = out_h * out_w;
    const bool full_gemm_volume =
        patch > 0 && spatial > 0 && patch * spatial >= kIm2ColFullMinPatchVolume;
#endif

    if (kernel_h == 3 && kernel_w == 3 && stride == 1)
    {
#if NETKIT_IM2COL >= 2
        if (full_gemm_volume)
            return Conv2dExecMode::FullIm2Col;
#endif
        if (large_volume)
            return Conv2dExecMode::PartialIm2Col;
        return Conv2dExecMode::Direct;
    }

    if (kernel_h >= 5 || kernel_w >= 5)
    {
#if NETKIT_IM2COL >= 2
        if (large_volume)
            return Conv2dExecMode::FullIm2Col;
#endif
        if (large_volume)
            return Conv2dExecMode::PartialIm2Col;
        return Conv2dExecMode::Direct;
    }

#if NETKIT_IM2COL >= 2
    if (large_volume)
        return Conv2dExecMode::FullIm2Col;
#endif
    if (large_volume)
        return Conv2dExecMode::PartialIm2Col;
    return Conv2dExecMode::Direct;
#else  /* NETKIT_IM2COL == 0: direct loops only */
    (void) kernel_h;
    (void) kernel_w;
    (void) stride;
    (void) in_channels;
    (void) out_h;
    (void) out_w;
    return Conv2dExecMode::Direct;
#endif
}

std::size_t Conv2dWorkspaceBytes(uint32_t out_h,
                                 uint32_t out_w,
                                 uint32_t kernel_h,
                                 uint32_t kernel_w,
                                 uint32_t in_channels,
                                 int stride)
{
    const Conv2dExecMode mode = SelectConv2dExecMode(static_cast<int>(kernel_h),
                                                     static_cast<int>(kernel_w),
                                                     stride,
                                                     in_channels,
                                                     out_h,
                                                     out_w);

    if (mode == Conv2dExecMode::Direct)
        return 0;

    if (!conv_volume_warrants_im2col(kernel_h, kernel_w, in_channels, out_h, out_w))
        return 0;

    if (mode == Conv2dExecMode::PartialIm2Col)
        return ConvPartialIm2ColWorkspaceBytes(kernel_h, kernel_w, in_channels);

    return ConvFullIm2ColWorkspaceBytes(out_h, out_w, kernel_h, kernel_w, in_channels);
}

bool Conv2dDispatchForward(const Tensor& input,
                           float* weights,
                           float* bias,
                           int kernel_size,
                           int stride,
                           int pad_h,
                           int pad_w,
                           int pad_h_end,
                           int pad_w_end,
                           int in_channels,
                           int out_channels,
                           NetkitKernelActivation fuse_activation,
                           Tensor& output,
                           const float* weights_hwio)
{
    float* in = tensor_data_f32(const_cast<Tensor&>(input));
    float* out = tensor_data_f32(output);

    const uint32_t in_h = input.shape[0];
    const uint32_t out_h = output.shape[0];
    const uint32_t out_w = output.shape[1];
    const uint32_t in_w_u = input.shape[1];
    const uint32_t in_ch = static_cast<uint32_t>(in_channels);
    const uint32_t out_ch = static_cast<uint32_t>(out_channels);
    const uint32_t k_kernel = static_cast<uint32_t>(kernel_size);

    const Conv2dExecMode mode = SelectConv2dExecMode(
        kernel_size, kernel_size, stride, in_ch, out_h, out_w);

    if (mode == Conv2dExecMode::Direct)
    {
        return Conv2dRunDirect(in,
                               weights,
                               weights_hwio,
                               bias,
                               out,
                               in_h,
                               in_w_u,
                               in_ch,
                               out_h,
                               out_w,
                               out_ch,
                               kernel_size,
                               stride,
                               pad_h,
                               pad_w,
                               pad_h_end,
                               pad_w_end,
                               out_channels,
                               fuse_activation);
    }

    if (mode == Conv2dExecMode::FullIm2Col)
    {
        if (Conv2dTryFullIm2Col(in,
                                weights,
                                bias,
                                out,
                                in_h,
                                in_w_u,
                                in_ch,
                                out_h,
                                out_w,
                                out_ch,
                                k_kernel,
                                k_kernel,
                                stride,
                                pad_h,
                                pad_w,
                                pad_h_end,
                                pad_w_end,
                                out_channels,
                                fuse_activation))
        {
            return kernel_activation_is_fused(fuse_activation);
        }

        if (Conv2dTryPartialIm2Col(in,
                                 weights,
                                 bias,
                                 out,
                                 in_h,
                                 in_w_u,
                                 in_ch,
                                 out_h,
                                 out_w,
                                 out_ch,
                                 k_kernel,
                                 k_kernel,
                                 stride,
                                 pad_h,
                                 pad_w,
                                 pad_h_end,
                                 pad_w_end,
                                 out_channels,
                                 fuse_activation))
        {
            return kernel_activation_is_fused(fuse_activation);
        }
    }
    else if (mode == Conv2dExecMode::PartialIm2Col)
    {
        if (Conv2dTryPartialIm2Col(in,
                                 weights,
                                 bias,
                                 out,
                                 in_h,
                                 in_w_u,
                                 in_ch,
                                 out_h,
                                 out_w,
                                 out_ch,
                                 k_kernel,
                                 k_kernel,
                                 stride,
                                 pad_h,
                                 pad_w,
                                 pad_h_end,
                                 pad_w_end,
                                 out_channels,
                                 fuse_activation))
        {
            return kernel_activation_is_fused(fuse_activation);
        }
    }

    return Conv2dRunDirect(in,
                           weights,
                           weights_hwio,
                           bias,
                           out,
                           in_h,
                           in_w_u,
                           in_ch,
                           out_h,
                           out_w,
                           out_ch,
                           kernel_size,
                           stride,
                           pad_h,
                           pad_w,
                           pad_h_end,
                           pad_w_end,
                           out_channels,
                           fuse_activation);
}
