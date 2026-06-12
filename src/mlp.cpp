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
    : num_layers(num_layers), arena(arena)
{
    // Allocate from Arena - MCU-safe, no heap fragmentation
    layers = (MLPLayer*)arena.alloc(sizeof(MLPLayer) * num_layers);
    intermediate_outputs = (Tensor*)arena.alloc(sizeof(Tensor) * num_layers);
}

// No destructor - Arena manages all memory automatically


void MLPNetwork::InitLayer(uint32_t layer_idx, const Tensor& weights, const Tensor& bias,
                            ActivationType activation, float leaky_alpha)
{
    if (layer_idx >= num_layers)
        return;

    layers[layer_idx].weights = weights;
    layers[layer_idx].bias = bias;
    layers[layer_idx].activation = activation;
    layers[layer_idx].leaky_alpha = leaky_alpha;
}

void MLPNetwork::forward(const Tensor& input, Tensor& output, Arena& arena)
{
    if (num_layers == 0)
        return;

    Tensor current_input = input;

    for (uint32_t i = 0; i < num_layers; i++)
    {
        // Create intermediate output tensor if needed (except for last layer)
        if (i < num_layers - 1)
        {
            // Output shape: (batch_size, output_dim)
            // output_dim is determined by weights.shape[1]
            intermediate_outputs[i] = Create2D(arena, current_input.shape[0], layers[i].weights.shape[1]);
            layers[i].forward(current_input, intermediate_outputs[i]);
            current_input = intermediate_outputs[i];
        }
        else
        {
            // Last layer writes directly to output
            layers[i].forward(current_input, output);
        }
    }
}
