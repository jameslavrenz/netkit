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
    static void GeluImpl(const Tensor& a, Tensor& c);
    static void Grn2dForwardImpl(const Tensor& input,
                                 const float* gamma,
                                 const float* beta,
                                 int channels,
                                 float eps,
                                 float* channel_norm_scratch,
                                 Tensor& output);

    static bool Conv2dForwardImpl(const Tensor& input,
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
                                  const float* weights_hwio = nullptr);

    static bool DepthwiseConv2dForwardImpl(const Tensor& input,
                                          float* weights,
                                          float* bias,
                                          int kernel_h,
                                          int kernel_w,
                                          int stride,
                                          int pad_h,
                                          int pad_w,
                                          int pad_h_end,
                                          int pad_w_end,
                                          int channels,
                                          NetkitKernelActivation fuse_activation,
                                          Tensor& output);

    static bool MaxPool2dForwardImpl(const Tensor& input,
                                     int pool_h,
                                     int pool_w,
                                     int stride,
                                     int pad_h,
                                     int pad_w,
                                     int pad_h_end,
                                     int pad_w_end,
                                     NetkitKernelActivation fuse_activation,
                                     Tensor& output);
    static void AvgPool2dForwardImpl(const Tensor& input,
                                     int pool_h,
                                     int pool_w,
                                     int stride,
                                     int pad_h,
                                     int pad_w,
                                     int pad_h_end,
                                     int pad_w_end,
                                     Tensor& output);
    static void BatchNorm2dForwardImpl(const Tensor& input,
                                       const float* scale,
                                       const float* bias,
                                       int channels,
                                       Tensor& output);
    static void LayerNorm2dForwardImpl(const Tensor& input,
                                       const float* weight,
                                       const float* bias,
                                       int channels,
                                       float eps,
                                       Tensor& output);

    static bool FullyConnectedWithBiasImpl(const Tensor& input,
                                           const Tensor& weights,
                                           const Tensor& bias,
                                           NetkitKernelActivation fuse_activation,
                                           Tensor& output);

private:
    static void FullyConnectedImpl(const Tensor& input, const Tensor& kernel, Tensor& output);
};
