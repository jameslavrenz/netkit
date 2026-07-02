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
                                          in_channels,
                                          out_channels,
                                          fuse_activation,
                                          output);
    }

    static void MaxPool2dForward(const Tensor& input,
                                 int pool_size,
                                 int stride,
                                 int pad_h,
                                 int pad_w,
                                 Tensor& output)
    {
        Derived::MaxPool2dForwardImpl(input, pool_size, stride, pad_h, pad_w, output);
    }

    static void AvgPool2dForward(const Tensor& input,
                                 int pool_size,
                                 int stride,
                                 int pad_h,
                                 int pad_w,
                                 Tensor& output)
    {
        Derived::AvgPool2dForwardImpl(input, pool_size, stride, pad_h, pad_w, output);
    }

    static void BatchNorm2dForward(const Tensor& input,
                                   const float* scale,
                                   const float* bias,
                                   int channels,
                                   Tensor& output)
    {
        Derived::BatchNorm2dForwardImpl(input, scale, bias, channels, output);
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
