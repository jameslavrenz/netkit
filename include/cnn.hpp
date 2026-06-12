#pragma once
#include "tensor.hpp"
#include "arena.hpp"
#include "conv2d.hpp"

enum class ConvActivationType
{
    None,
    ReLU,
    Sigmoid,
    Tanh,
    LeakyReLU,
    ReLU6,
    Softmax
};

struct Conv2DLayer
{
    Conv2D conv;
    ConvActivationType activation;
    float leaky_alpha;   // Only used if activation is LeakyReLU

    // Forward pass: output = activation(conv(input))
    void forward(const Tensor& input, Tensor& output);
};

class CNNNetwork
{
private:
    Conv2DLayer* layers;
    uint32_t num_layers;
    Tensor* intermediate_outputs;  // Cache for intermediate results during forward pass

public:
    CNNNetwork(uint32_t num_layers);
    ~CNNNetwork();

    // Initialize a Conv2D layer at the given index
    void InitLayer(uint32_t layer_idx, 
                   int kernel_size, 
                   int stride,
                   int in_channels,
                   int out_channels,
                   float* weights,
                   float* bias,
                   ConvActivationType activation,
                   float leaky_alpha = 0.01f);

    // Forward pass through entire network
    // input_shape: [H, W, C] (HWC format)
    // Returns output tensor reference
    Tensor& forward(const Tensor& input, Arena& arena);

    // Get a specific layer
    Conv2DLayer& GetLayer(uint32_t idx) { return layers[idx]; }

    // Get the most recent output tensor
    Tensor& GetOutput() { return intermediate_outputs[num_layers - 1]; }
};
