#pragma once

#include "kernel_crtp.hpp"

struct ReferenceKernel : KernelBase<ReferenceKernel>
{
    static void MulImpl(const Tensor& a, const Tensor& b, Tensor& c);
    static void MulScalarImpl(const Tensor& a, float scalar, Tensor& c);
    static void MatAddImpl(const Tensor& a, const Tensor& b, Tensor& c);
    static void MatAddNDImpl(const Tensor& a, const Tensor& b, Tensor& c);
    static void MatMulImpl(const Tensor& a, const Tensor& b, Tensor& c);
    static void MulNDImpl(const Tensor& a, const Tensor& b, Tensor& c);

    static void ReLUImpl(const Tensor& a, Tensor& c);
    static void SigmoidImpl(const Tensor& a, Tensor& c);
    static void TanhImpl(const Tensor& a, Tensor& c);
    static void LeakyReLUImpl(const Tensor& a, Tensor& c, float alpha);
    static void ReLU6Impl(const Tensor& a, Tensor& c);
    static void SoftmaxImpl(const Tensor& a, Tensor& c);

    static bool Conv2dForwardImpl(const Tensor& input,
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

    static void MaxPool2dForwardImpl(const Tensor& input,
                                     int pool_size,
                                     int stride,
                                     int pad_h,
                                     int pad_w,
                                     Tensor& output);
    static void AvgPool2dForwardImpl(const Tensor& input,
                                     int pool_size,
                                     int stride,
                                     int pad_h,
                                     int pad_w,
                                     Tensor& output);
    static void BatchNorm2dForwardImpl(const Tensor& input,
                                       const float* scale,
                                       const float* bias,
                                       int channels,
                                       Tensor& output);

    static bool FullyConnectedWithBiasImpl(const Tensor& input,
                                           const Tensor& weights,
                                           const Tensor& bias,
                                           NetkitKernelActivation fuse_activation,
                                           Tensor& output);

    static void FullyConnected(const Tensor& input, const Tensor& kernel, Tensor& output)
    {
        FullyConnectedImpl(input, kernel, output);
    }

private:
    static void FullyConnectedImpl(const Tensor& input, const Tensor& kernel, Tensor& output);
};
