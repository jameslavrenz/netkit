#include "conv2d.hpp"
#include "active_kernel.hpp"
#include "conv_dispatch.hpp"

bool Conv2D::forward(const Tensor& input, Tensor& output, NetkitKernelActivation fuse_activation)
{
    const int pad_h_end = this->pad_h_end;
    const int pad_w_end = this->pad_w_end;
    if (pad_h_end == pad_h && pad_w_end == pad_w)
    {
        if (Kernels::Conv2dForward(input,
                                   weights,
                                   bias,
                                   kernel_size,
                                   stride,
                                   pad_h,
                                   pad_w,
                                   in_channels,
                                   out_channels,
                                   fuse_activation,
                                   output))
        {
            return true;
        }
    }

    return Conv2dDispatchForward(input,
                                 weights,
                                 bias,
                                 kernel_size,
                                 stride,
                                 pad_h,
                                 pad_w,
                                 pad_h_end,
                                 pad_w_end,
                                 in_channels,
                                 out_channels,
                                 fuse_activation,
                                 output,
                                 weights_hwio);
}
