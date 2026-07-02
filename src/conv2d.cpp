#include "conv2d.hpp"
#include "active_kernel.hpp"

bool Conv2D::forward(const Tensor& input, Tensor& output, NetkitKernelActivation fuse_activation)
{
    return Kernels::Conv2dForward(input,
                                  weights,
                                  bias,
                                  kernel_size,
                                  stride,
                                  pad_h,
                                  pad_w,
                                  in_channels,
                                  out_channels,
                                  fuse_activation,
                                  output);
}
