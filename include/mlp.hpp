#pragma once
#include "tensor.hpp"
#include "arena.hpp"

enum class ActivationType
{
    None,
    ReLU,
    Sigmoid,
    Tanh,
    LeakyReLU,
    ReLU6,
    Softmax
};

struct MLPLayer
{
    Tensor weights;      // Weight matrix (input_dim x output_dim)
    Tensor bias;         // Bias vector (1 x output_dim)
    ActivationType activation;
    float leaky_alpha;   // Only used if activation is LeakyReLU

    // Forward pass: output = activation(input @ weights + bias)
    void forward(const Tensor& input, Tensor& output);
};

class MLPNetwork
{
private:
    MLPLayer* layers;
    uint32_t num_layers;
    Tensor* intermediate_outputs;  // Cache for intermediate results during forward pass
    Arena& arena;

public:
    // Constructor allocates from Arena — no heap fragmentation
    MLPNetwork(uint32_t num_layers, Arena& arena);
    
    // No destructor needed - Arena manages all memory

    bool IsValid() const { return layers != nullptr && intermediate_outputs != nullptr; }

    // Initialize a layer at the given index
    void InitLayer(uint32_t layer_idx, const Tensor& weights, const Tensor& bias, 
                   ActivationType activation, float leaky_alpha = 0.01f);

    // Forward pass through entire network
    void forward(const Tensor& input, Tensor& output, Arena& arena);

    // Get a specific layer
    MLPLayer& GetLayer(uint32_t idx) { return layers[idx]; }
};
