#pragma once

#include "kernel_activation.hpp"
#include "tensor.hpp"

/*
 * CRTP kernel facade — static dispatch to Derived::*Impl without virtual calls.
 * Active backend is selected at compile time in active_kernel.hpp.
 */
template<typename Derived>
struct KernelBase
{
    static void Mul(const Tensor& a, const Tensor& b, Tensor& c)
    {
        Derived::MulImpl(a, b, c);
    }

    static void MulScalar(const Tensor& a, float scalar, Tensor& c)
    {
        Derived::MulScalarImpl(a, scalar, c);
    }

    static void MatAdd(const Tensor& a, const Tensor& b, Tensor& c)
    {
        Derived::MatAddImpl(a, b, c);
    }

    static void MatAddND(const Tensor& a, const Tensor& b, Tensor& c)
    {
        Derived::MatAddNDImpl(a, b, c);
    }

    static void MatMul(const Tensor& a, const Tensor& b, Tensor& c)
    {
        Derived::MatMulImpl(a, b, c);
    }

    static void MulND(const Tensor& a, const Tensor& b, Tensor& c)
    {
        Derived::MulNDImpl(a, b, c);
    }

    static void ReLU(const Tensor& a, Tensor& c)
    {
        Derived::ReLUImpl(a, c);
    }

    static void Sigmoid(const Tensor& a, Tensor& c)
    {
        Derived::SigmoidImpl(a, c);
    }

    static void Tanh(const Tensor& a, Tensor& c)
    {
        Derived::TanhImpl(a, c);
    }

    static void LeakyReLU(const Tensor& a, Tensor& c, float alpha)
    {
        Derived::LeakyReLUImpl(a, c, alpha);
    }

    static void ReLU6(const Tensor& a, Tensor& c)
    {
        Derived::ReLU6Impl(a, c);
    }

    static void Softmax(const Tensor& a, Tensor& c)
    {
        Derived::SoftmaxImpl(a, c);
    }

    static void Gelu(const Tensor& a, Tensor& c)
    {
        Derived::GeluImpl(a, c);
    }

    static void Grn2dForward(const Tensor& input,
                             const float* gamma,
                             const float* beta,
                             int channels,
                             float eps,
                             float* channel_norm_scratch,
                             Tensor& output)
    {
        Derived::Grn2dForwardImpl(input, gamma, beta, channels, eps, channel_norm_scratch, output);
    }

    static bool Conv2dForward(const Tensor& input,
                              float* weights,
                              float* bias,
                              int kernel_size,
                              int stride,
                              int pad_h,
                              int pad_w,
                              int in_channels,
                              int out_channels,
                              NetkitKernelActivation fuse_activation,
                              Tensor& output)
    {
        return Derived::Conv2dForwardImpl(input,
                                          weights,
                                          bias,
                                          kernel_size,
                                          stride,
                                          pad_h,
                                          pad_w,
                                          pad_h,
                                          pad_w,
                                          in_channels,
                                          out_channels,
                                          fuse_activation,
                                          output);
    }

    static bool DepthwiseConv2dForward(const Tensor& input,
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
                                       Tensor& output)
    {
        return Derived::DepthwiseConv2dForwardImpl(input,
                                                   weights,
                                                   bias,
                                                   kernel_h,
                                                   kernel_w,
                                                   stride,
                                                   pad_h,
                                                   pad_w,
                                                   pad_h_end,
                                                   pad_w_end,
                                                   channels,
                                                   fuse_activation,
                                                   output);
    }

    static bool MaxPool2dForward(const Tensor& input,
                                 int pool_size,
                                 int stride,
                                 int pad_h,
                                 int pad_w,
                                 NetkitKernelActivation fuse_activation,
                                 Tensor& output)
    {
        return Derived::MaxPool2dForwardImpl(input,
                                             pool_size,
                                             pool_size,
                                             stride,
                                             pad_h,
                                             pad_w,
                                             pad_h,
                                             pad_w,
                                             fuse_activation,
                                             output);
    }

    // Backward-compatible overload (no fused activation).
    static bool MaxPool2dForward(const Tensor& input,
                                 int pool_size,
                                 int stride,
                                 int pad_h,
                                 int pad_w,
                                 Tensor& output)
    {
        return MaxPool2dForward(
            input, pool_size, stride, pad_h, pad_w, NetkitKernelActivation::None, output);
    }

    static bool MaxPool2dForwardPadded(const Tensor& input,
                                       int pool_h,
                                       int pool_w,
                                       int stride,
                                       int pad_h,
                                       int pad_w,
                                       int pad_h_end,
                                       int pad_w_end,
                                       NetkitKernelActivation fuse_activation,
                                       Tensor& output)
    {
        return Derived::MaxPool2dForwardImpl(input,
                                             pool_h,
                                             pool_w,
                                             stride,
                                             pad_h,
                                             pad_w,
                                             pad_h_end,
                                             pad_w_end,
                                             fuse_activation,
                                             output);
    }

    static bool MaxPool2dForwardPadded(const Tensor& input,
                                       int pool_h,
                                       int pool_w,
                                       int stride,
                                       int pad_h,
                                       int pad_w,
                                       int pad_h_end,
                                       int pad_w_end,
                                       Tensor& output)
    {
        return MaxPool2dForwardPadded(input,
                                      pool_h,
                                      pool_w,
                                      stride,
                                      pad_h,
                                      pad_w,
                                      pad_h_end,
                                      pad_w_end,
                                      NetkitKernelActivation::None,
                                      output);
    }

    static void AvgPool2dForward(const Tensor& input,
                                 int pool_size,
                                 int stride,
                                 int pad_h,
                                 int pad_w,
                                 Tensor& output)
    {
        Derived::AvgPool2dForwardImpl(
            input, pool_size, pool_size, stride, pad_h, pad_w, pad_h, pad_w, output);
    }

    static void AvgPool2dForwardPadded(const Tensor& input,
                                       int pool_h,
                                       int pool_w,
                                       int stride,
                                       int pad_h,
                                       int pad_w,
                                       int pad_h_end,
                                       int pad_w_end,
                                       Tensor& output)
    {
        Derived::AvgPool2dForwardImpl(
            input, pool_h, pool_w, stride, pad_h, pad_w, pad_h_end, pad_w_end, output);
    }

    static void BatchNorm2dForward(const Tensor& input,
                                   const float* scale,
                                   const float* bias,
                                   int channels,
                                   Tensor& output)
    {
        Derived::BatchNorm2dForwardImpl(input, scale, bias, channels, output);
    }

    static void LayerNorm2dForward(const Tensor& input,
                                   const float* weight,
                                   const float* bias,
                                   int channels,
                                   float eps,
                                   Tensor& output)
    {
        Derived::LayerNorm2dForwardImpl(input, weight, bias, channels, eps, output);
    }

    static bool FullyConnectedWithBias(const Tensor& input,
                                       const Tensor& weights,
                                       const Tensor& bias,
                                       NetkitKernelActivation fuse_activation,
                                       Tensor& output)
    {
        return Derived::FullyConnectedWithBiasImpl(input, weights, bias, fuse_activation, output);
    }
};
