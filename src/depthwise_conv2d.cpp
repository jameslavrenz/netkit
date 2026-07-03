#include "depthwise_conv2d.hpp"
#include "active_kernel.hpp"

bool DepthwiseConv2D::forward(const Tensor& input, Tensor& output, NetkitKernelActivation fuse_activation)
{
    return Kernels::DepthwiseConv2dForward(input,
                                           weights,
                                           bias,
                                           kernel_h,
                                           kernel_w,
                                           stride,
                                           pad_h,
                                           pad_w,
                                           channels,
                                           fuse_activation,
                                           output);
}
