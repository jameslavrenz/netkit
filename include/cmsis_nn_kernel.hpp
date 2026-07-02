#pragma once

#include "kernel_activation.hpp"
#include "tensor.hpp"

struct CmsisNnKernel
{
    static bool TryConv2dForward(const Tensor& input,
                                 float* weights,
                                 float* bias,
                                 int kernel_size,
                                 int stride,
                                 int pad_h,
                                 int pad_w,
                                 int in_channels,
                                 int out_channels,
                                 NetkitKernelActivation fuse_activation,
                                 Tensor& output);

    static bool TryMaxPool2dForward(const Tensor& input,
                                    int pool_size,
                                    int stride,
                                    int pad_h,
                                    int pad_w,
                                    NetkitKernelActivation fuse_activation,
                                    Tensor& output);

    static bool TryAvgPool2dForward(const Tensor& input,
                                    int pool_size,
                                    int stride,
                                    int pad_h,
                                    int pad_w,
                                    Tensor& output);

    static bool TryBatchNorm2dForward(const Tensor& input,
                                      const float* scale,
                                      const float* bias,
                                      int channels,
                                      Tensor& output);

    static bool TryFullyConnectedWithBias(const Tensor& input,
                                          const Tensor& weights,
                                          const Tensor& bias,
                                          NetkitKernelActivation fuse_activation,
                                          Tensor& output);

    static bool TryActivationForward(const Tensor& input,
                                     Tensor& output,
                                     NetkitKernelActivation activation,
                                     float leaky_alpha);

    static bool TrySoftmaxForward(const Tensor& input, Tensor& output);

    static bool TryMatAdd(const Tensor& a, const Tensor& b, Tensor& c);
};
