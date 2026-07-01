#include "mlp.hpp"
#include "ops.hpp"
#include "tensor_factory.hpp"

using namespace Ops;
using namespace TensorFactory;

// ====================================================
// MLPLayer Implementation
// ====================================================

void MLPLayer::forward(const Tensor& input, Tensor& output)
{
    // Step 1: Linear transformation (y = x @ W)
    MatMul(input, weights, output);

    // Step 2: Add bias (y = y + b)
    MatAdd(output, bias, output);

    // Step 3: Apply activation function
    switch (activation)
    {
        case ActivationType::None:
            // No activation
            break;
        case ActivationType::ReLU:
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
        const uint32_t cols = layers[i].weights.shape[1];
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
        const uint32_t cols = layers[i].weights.shape[1];
        Tensor layer_output = View2D(write_buffer, rows, cols);
        if (layer_output.num_elements > max_activation_elements)
            return;

        layers[i].forward(current_input, layer_output);
        current_input = layer_output;
        write_buffer = (write_buffer == ping_a) ? ping_b : ping_a;
    }
}
