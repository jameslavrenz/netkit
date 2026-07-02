#pragma once

enum class NetkitKernelActivation
{
    None = 0,
    ReLU,
    Sigmoid,
    Tanh,
    LeakyReLU,
    ReLU6,
    Softmax,
};

constexpr bool kernel_activation_is_fused(NetkitKernelActivation activation)
{
    return activation == NetkitKernelActivation::ReLU || activation == NetkitKernelActivation::ReLU6;
}
