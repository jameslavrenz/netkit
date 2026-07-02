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
    Tensor weights;      // Weight matrix [out_features, in_features] (CMSIS-NN / PyTorch layout)
    Tensor bias;         // Bias vector [1, out_features]
    ActivationType activation;
    float leaky_alpha;   // Only used if activation is LeakyReLU

    // Forward pass: output = activation(FC(input, weights) + bias)
    void forward(const Tensor& input, Tensor& output);
};

class MLPNetwork
{
private:
    MLPLayer* layers;
    uint32_t num_layers;
    float* ping_a{};
    float* ping_b{};
    uint32_t max_activation_elements{};

public:
    // Constructor allocates from Arena — no heap fragmentation
    MLPNetwork(uint32_t num_layers, Arena& arena);
    
    // No destructor needed - Arena manages all memory

    bool IsValid() const { return layers != nullptr; }

    bool HasActivationBuffers() const
    {
        if (num_layers <= 1)
            return true;
        return ping_a != nullptr && ping_b != nullptr;
    }

    // Preallocate two ping-pong activation buffers sized to the largest hidden layer.
    bool InitActivationBuffers(Arena& arena, uint32_t batch_rows);

    // Initialize a layer at the given index
    void InitLayer(uint32_t layer_idx, const Tensor& weights, const Tensor& bias, 
                   ActivationType activation, float leaky_alpha = 0.01f);

    // Forward pass through entire network (hidden activations reuse ping_a / ping_b)
    void forward(const Tensor& input, Tensor& output, Arena& arena);

    // Get a specific layer
    MLPLayer& GetLayer(uint32_t idx) { return layers[idx]; }
};
