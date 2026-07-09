#pragma once

#include "active_kernel.hpp"
#include "kernel_activation.hpp"
#include "tensor.hpp"
#include "tensor_factory.hpp"

#include <array>
#include <cstdint>

namespace fused_ops
{
    inline Tensor NhwcView(float* data, uint32_t h, uint32_t w, uint32_t c)
    {
        const std::array<uint32_t, 3> shape = {h, w, c};
        return TensorFactory::ViewND(data, 3, shape);
    }

    inline void BatchNormInPlace(Tensor& tensor, int channels, const float* scale, const float* bias)
    {
        Kernels::BatchNorm2dForward(tensor, scale, bias, channels, tensor);
    }

    inline void ReluInPlace(Tensor& tensor)
    {
        Kernels::ReLU(tensor, tensor);
    }

    inline void MatAddInPlace(Tensor& accum, const Tensor& addend)
    {
        Kernels::MatAddND(accum, addend, accum);
    }

    // Residual epilogue used by ResNet BasicBlock: out = ReLU(branch + shortcut).
    // Routes through Kernels::MatAddND (CMSIS-DSP / CMSIS-NN / reference).
    inline void MatAddThenRelu(const Tensor& branch, const Tensor& shortcut, Tensor& out)
    {
        Kernels::MatAddND(branch, shortcut, out);
        ReluInPlace(out);
    }

    inline void GeluInPlace(Tensor& tensor)
    {
        Kernels::Gelu(tensor, tensor);
    }

    inline void Grn2dInPlace(Tensor& tensor,
                             int channels,
                             const float* gamma,
                             const float* beta,
                             float eps,
                             float* channel_norm_scratch)
    {
        Kernels::Grn2dForward(
            tensor, gamma, beta, channels, eps, channel_norm_scratch, tensor);
    }

    inline void FullyConnected1x1(const float* input,
                                  int in_features,
                                  int out_features,
                                  float* weights,
                                  const float* bias,
                                  float* output)
    {
        Tensor in_tensor =
            TensorFactory::View2D(const_cast<float*>(input), 1, static_cast<uint32_t>(in_features));
        Tensor weight_tensor = TensorFactory::View2D(
            weights, static_cast<uint32_t>(out_features), static_cast<uint32_t>(in_features));
        Tensor bias_tensor =
            TensorFactory::View2D(const_cast<float*>(bias), 1, static_cast<uint32_t>(out_features));
        Tensor out_tensor =
            TensorFactory::View2D(output, 1, static_cast<uint32_t>(out_features));
        Kernels::FullyConnectedWithBias(
            in_tensor, weight_tensor, bias_tensor, NetkitKernelActivation::None, out_tensor);
    }
}
