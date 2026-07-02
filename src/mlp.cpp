#include "mlp.hpp"
#include "netkit_backend.h"
#include "ops.hpp"
#include "tensor_factory.hpp"

using namespace Ops;
using namespace TensorFactory;

namespace
{
    NetkitBackendActivation ToBackendActivation(ActivationType activation)
    {
        switch (activation)
        {
            case ActivationType::None:
                return NETKIT_BACKEND_ACT_NONE;
            case ActivationType::ReLU:
                return NETKIT_BACKEND_ACT_RELU;
            case ActivationType::Sigmoid:
                return NETKIT_BACKEND_ACT_SIGMOID;
            case ActivationType::Tanh:
                return NETKIT_BACKEND_ACT_TANH;
            case ActivationType::LeakyReLU:
                return NETKIT_BACKEND_ACT_LEAKY_RELU;
            case ActivationType::ReLU6:
                return NETKIT_BACKEND_ACT_RELU6;
            case ActivationType::Softmax:
                return NETKIT_BACKEND_ACT_SOFTMAX;
        }
        return NETKIT_BACKEND_ACT_NONE;
    }
}

// ====================================================
// MLPLayer Implementation
// ====================================================

void MLPLayer::forward(const Tensor& input, Tensor& output)
{
    const NetkitBackendActivation backend_activation = ToBackendActivation(activation);
    const bool used_cmsis_nn =
        netkit_cmsis_fully_connected_forward(&input, &weights, &bias, backend_activation, &output) != 0;

    if (!used_cmsis_nn)
    {
        if (!netkit_cmsis_dsp_fully_connected_forward(&input, &weights, &bias, &output))
        {
            FullyConnected(input, weights, output);
            MatAdd(output, bias, output);
        }
    }

    switch (activation)
    {
        case ActivationType::None:
            break;
        case ActivationType::ReLU:
            if (used_cmsis_nn && netkit_activation_is_fused(backend_activation))
                break;
            ReLU(output, output);
            break;
        case ActivationType::Sigmoid:
            Sigmoid(output, output);
            break;
        case ActivationType::Tanh:
            Tanh(output, output);
            break;
        case ActivationType::LeakyReLU:
            LeakyReLU(output, output, leaky_alpha);
            break;
        case ActivationType::ReLU6:
            if (used_cmsis_nn && netkit_activation_is_fused(backend_activation))
                break;
            ReLU6(output, output);
            break;
        case ActivationType::Softmax:
            Softmax(output, output);
            break;
    }
}

// ====================================================
// MLPNetwork Implementation
// ====================================================

MLPNetwork::MLPNetwork(uint32_t num_layers, Arena& arena)
    : layers(nullptr), num_layers(num_layers)
{
    layers = static_cast<MLPLayer*>(arena.alloc(sizeof(MLPLayer) * num_layers, alignof(MLPLayer)));
}

bool MLPNetwork::InitActivationBuffers(Arena& arena, uint32_t batch_rows)
{
    ping_a = nullptr;
    ping_b = nullptr;
    max_activation_elements = 0;

    if (num_layers <= 1 || !layers)
        return layers != nullptr;

    for (uint32_t i = 0; i < num_layers - 1; ++i)
    {
        const uint32_t cols = layers[i].weights.shape[0];
        const uint32_t elements = batch_rows * cols;
        if (elements > max_activation_elements)
            max_activation_elements = elements;
    }

    if (max_activation_elements == 0)
        return false;

    const std::size_t bytes = static_cast<std::size_t>(max_activation_elements) * sizeof(float);
    ping_a = static_cast<float*>(arena.alloc(bytes, alignof(float)));
    ping_b = static_cast<float*>(arena.alloc(bytes, alignof(float)));
    return ping_a != nullptr && ping_b != nullptr;
}

void MLPNetwork::InitLayer(uint32_t layer_idx, const Tensor& weights, const Tensor& bias,
                            ActivationType activation, float leaky_alpha)
{
    if (!layers || layer_idx >= num_layers)
        return;

    layers[layer_idx].weights = weights;
    layers[layer_idx].bias = bias;
    layers[layer_idx].activation = activation;
    layers[layer_idx].leaky_alpha = leaky_alpha;
}

void MLPNetwork::forward(const Tensor& input, Tensor& output, Arena& /*arena*/)
{
    if (!IsValid() || !HasActivationBuffers() || num_layers == 0)
        return;

    if (num_layers == 1)
    {
        layers[0].forward(input, output);
        return;
    }

    Tensor current_input = input;
    float* write_buffer = ping_a;

    for (uint32_t i = 0; i < num_layers; ++i)
    {
        if (i == num_layers - 1)
        {
            layers[i].forward(current_input, output);
            return;
        }

        const uint32_t rows = current_input.shape[0];
        const uint32_t cols = layers[i].weights.shape[0];
        Tensor layer_output = View2D(write_buffer, rows, cols);
        if (layer_output.num_elements > max_activation_elements)
            return;

        layers[i].forward(current_input, layer_output);
        current_input = layer_output;
        write_buffer = (write_buffer == ping_a) ? ping_b : ping_a;
    }
}
