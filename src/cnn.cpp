#include "cnn.hpp"
#include "ops.hpp"
#include "tensor_factory.hpp"
#include "tensor_access.hpp"
#include <cstring>
#include <cmath>

using namespace Ops;
using namespace TensorFactory;

// ====================================================
// Conv2DLayer Implementation
// ====================================================

void Conv2DLayer::forward(const Tensor& input, Tensor& output)
{
    // Step 1: Conv2D forward
    conv.forward(input, output);

    // Step 2: Apply activation function
    switch (activation)
    {
        case ConvActivationType::None:
            // No activation
            break;
        case ConvActivationType::ReLU:
            ReLU(output, output);
            break;
        case ConvActivationType::Sigmoid:
            Sigmoid(output, output);
            break;
        case ConvActivationType::Tanh:
            Tanh(output, output);
            break;
        case ConvActivationType::LeakyReLU:
            LeakyReLU(output, output, leaky_alpha);
            break;
        case ConvActivationType::ReLU6:
            ReLU6(output, output);
            break;
        case ConvActivationType::Softmax:
            Softmax(output, output);
            break;
    }
}

// ====================================================
// CNNNetwork Implementation
// ====================================================

// Helper function to calculate output dimensions
static uint32_t CalcConvOutputDim(uint32_t input_dim, int kernel_size, int stride)
{
    return (input_dim - kernel_size) / stride + 1;
}

CNNNetwork::CNNNetwork(uint32_t num_layers)
    : num_layers(num_layers)
{
    layers = new Conv2DLayer[num_layers];
    intermediate_outputs = new Tensor[num_layers];
    
    // Zero-initialize
    std::memset(layers, 0, sizeof(Conv2DLayer) * num_layers);
    std::memset(intermediate_outputs, 0, sizeof(Tensor) * num_layers);
}

CNNNetwork::~CNNNetwork()
{
    delete[] layers;
    delete[] intermediate_outputs;
}

void CNNNetwork::InitLayer(uint32_t layer_idx,
                            int kernel_size,
                            int stride,
                            int in_channels,
                            int out_channels,
                            float* weights,
                            float* bias,
                            ConvActivationType activation,
                            float leaky_alpha)
{
    if (layer_idx >= num_layers)
        return;

    layers[layer_idx].conv.kernel_size = kernel_size;
    layers[layer_idx].conv.stride = stride;
    layers[layer_idx].conv.in_channels = in_channels;
    layers[layer_idx].conv.out_channels = out_channels;
    layers[layer_idx].conv.weights = weights;
    layers[layer_idx].conv.bias = bias;
    layers[layer_idx].activation = activation;
    layers[layer_idx].leaky_alpha = leaky_alpha;
}

Tensor& CNNNetwork::forward(const Tensor& input, Arena& arena)
{
    if (num_layers == 0)
        return intermediate_outputs[0];

    Tensor current_input = input;

    for (uint32_t i = 0; i < num_layers; i++)
    {
        // Calculate output dimensions for this layer
        uint32_t out_h = CalcConvOutputDim(current_input.shape[0], 
                                            layers[i].conv.kernel_size, 
                                            layers[i].conv.stride);
        uint32_t out_w = CalcConvOutputDim(current_input.shape[1], 
                                            layers[i].conv.kernel_size, 
                                            layers[i].conv.stride);
        uint32_t out_c = layers[i].conv.out_channels;

        // Create output tensor
        uint32_t shape[3] = {out_h, out_w, out_c};
        intermediate_outputs[i] = CreateND(arena, 3, shape);

        // Forward pass through this layer
        layers[i].forward(current_input, intermediate_outputs[i]);

        // Update current input for next layer
        current_input = intermediate_outputs[i];
    }

    return intermediate_outputs[num_layers - 1];
}
