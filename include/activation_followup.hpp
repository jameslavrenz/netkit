#pragma once

#include "kernel_activation.hpp"
#include "active_kernel.hpp"
#include "cnn.hpp"
#include "mlp.hpp"
#include "tensor.hpp"

inline void ApplyFusedOutputActivation(NetkitKernelActivation activation,
                                       bool fused_in_kernel,
                                       Tensor& output,
                                       float leaky_alpha = 0.01f)
{
    switch (activation)
    {
        case NetkitKernelActivation::None:
            break;
        case NetkitKernelActivation::ReLU:
            if (fused_in_kernel && kernel_activation_is_fused(activation))
                break;
            Kernels::ReLU(output, output);
            break;
        case NetkitKernelActivation::Sigmoid:
            Kernels::Sigmoid(output, output);
            break;
        case NetkitKernelActivation::Tanh:
            Kernels::Tanh(output, output);
            break;
        case NetkitKernelActivation::LeakyReLU:
            Kernels::LeakyReLU(output, output, leaky_alpha);
            break;
        case NetkitKernelActivation::ReLU6:
            if (fused_in_kernel && kernel_activation_is_fused(activation))
                break;
            Kernels::ReLU6(output, output);
            break;
        case NetkitKernelActivation::Softmax:
            Kernels::Softmax(output, output);
            break;
    }
}

inline NetkitKernelActivation ToKernelActivation(ActivationType activation)
{
    switch (activation)
    {
        case ActivationType::None:
            return NetkitKernelActivation::None;
        case ActivationType::ReLU:
            return NetkitKernelActivation::ReLU;
        case ActivationType::Sigmoid:
            return NetkitKernelActivation::Sigmoid;
        case ActivationType::Tanh:
            return NetkitKernelActivation::Tanh;
        case ActivationType::LeakyReLU:
            return NetkitKernelActivation::LeakyReLU;
        case ActivationType::ReLU6:
            return NetkitKernelActivation::ReLU6;
        case ActivationType::Softmax:
            return NetkitKernelActivation::Softmax;
    }
    return NetkitKernelActivation::None;
}

inline NetkitKernelActivation ToKernelActivation(ConvActivationType activation)
{
    switch (activation)
    {
        case ConvActivationType::None:
            return NetkitKernelActivation::None;
        case ConvActivationType::ReLU:
            return NetkitKernelActivation::ReLU;
        case ConvActivationType::Sigmoid:
            return NetkitKernelActivation::Sigmoid;
        case ConvActivationType::Tanh:
            return NetkitKernelActivation::Tanh;
        case ConvActivationType::LeakyReLU:
            return NetkitKernelActivation::LeakyReLU;
        case ConvActivationType::ReLU6:
            return NetkitKernelActivation::ReLU6;
        case ConvActivationType::Softmax:
            return NetkitKernelActivation::Softmax;
    }
    return NetkitKernelActivation::None;
}
