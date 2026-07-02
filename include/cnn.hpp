#pragma once
#include "tensor.hpp"
#include "arena.hpp"
#include "conv2d.hpp"
#include "mlp.hpp"

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

enum class CnnBlockType
{
    Conv2D,
    MaxPool2D,
    AvgPool2D,
    BatchNorm2d,
    Flatten,
    Dense
};

struct Conv2DLayer
{
    Conv2D conv;
    ConvActivationType activation;
    float leaky_alpha = 0.01f;

    void forward(const Tensor& input, Tensor& output);
};

struct MaxPool2DLayer
{
    int pool_size = 2;
    int stride = 2;
    int pad_h = 0;
    int pad_w = 0;

    void forward(const Tensor& input, Tensor& output);
};

struct AvgPool2DLayer
{
    int pool_size = 2;
    int stride = 2;
    int pad_h = 0;
    int pad_w = 0;

    void forward(const Tensor& input, Tensor& output);
};

struct BatchNorm2DLayer
{
    int channels = 0;
    float* scale = nullptr;
    float* bias = nullptr;

    void forward(const Tensor& input, Tensor& output);
};

struct CnnBlock
{
    CnnBlockType type = CnnBlockType::Conv2D;
    Conv2DLayer conv;
    MaxPool2DLayer pool;
    AvgPool2DLayer avg_pool;
    BatchNorm2DLayer batch_norm;
    MLPLayer dense;
};

class CNNNetwork
{
private:
    CnnBlock* blocks;
    uint32_t num_layers;
    float* ping_a{};
    float* ping_b{};
    uint32_t max_activation_elements{};
    Tensor output_cache_{};

public:
    CNNNetwork(uint32_t num_layers, Arena& arena);

    bool IsValid() const { return blocks != nullptr; }

    bool HasActivationBuffers() const
    {
        if (num_layers == 0)
            return false;
        return ping_a != nullptr && ping_b != nullptr;
    }

    // Preallocate two ping-pong activation buffers sized to the largest layer output.
    bool InitActivationBuffers(Arena& arena, uint32_t in_h, uint32_t in_w, uint32_t in_c);

    void InitConvLayer(uint32_t layer_idx,
                       int kernel_size,
                       int stride,
                       int in_channels,
                       int out_channels,
                       float* weights,
                       float* bias,
                       ConvActivationType activation,
                       float leaky_alpha = 0.01f,
                       int pad_h = 0,
                       int pad_w = 0);

    void InitPoolLayer(uint32_t layer_idx, int pool_size, int stride, int pad_h = 0, int pad_w = 0);

    void InitAvgPoolLayer(uint32_t layer_idx, int pool_size, int stride, int pad_h = 0, int pad_w = 0);

    void InitBatchNormLayer(uint32_t layer_idx, int channels, float* scale, float* bias);

    void InitFlattenLayer(uint32_t layer_idx);

    void InitDenseLayer(uint32_t layer_idx,
                        const Tensor& weights,
                        const Tensor& bias,
                        ActivationType activation,
                        float leaky_alpha = 0.01f);

    Tensor& forward(const Tensor& input, Arena& arena);

    CnnBlock& GetBlock(uint32_t idx) { return blocks[idx]; }

    Tensor& GetOutput() { return output_cache_; }
};
