#include "cnn.hpp"
#include "ops.hpp"
#include "tensor_factory.hpp"
#include "tensor_access.hpp"
#include <array>
#include <cmath>
#include <cstring>

using namespace Ops;
using namespace TensorFactory;

namespace
{
    uint32_t CalcOutputDim(uint32_t input_dim, int kernel_size, int stride)
    {
        return static_cast<uint32_t>((static_cast<int>(input_dim) - kernel_size) / stride + 1);
    }

    void FlattenNhwc(const Tensor& input, Tensor& output)
    {
        const float* in = tensor_data_f32(const_cast<Tensor&>(input));
        float* out = tensor_data_f32(output);
        std::memcpy(out, in, static_cast<std::size_t>(input.num_elements) * sizeof(float));
    }

    uint32_t MaxU32(uint32_t a, uint32_t b)
    {
        return a > b ? a : b;
    }
}

void Conv2DLayer::forward(const Tensor& input, Tensor& output)
{
    conv.forward(input, output);

    switch (activation)
    {
        case ConvActivationType::None:
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

void MaxPool2DLayer::forward(const Tensor& input, Tensor& output)
{
    const float* in = tensor_data_f32(const_cast<Tensor&>(input));
    float* out = tensor_data_f32(output);

    const uint32_t channels = input.shape[2];
    const uint32_t out_h = output.shape[0];
    const uint32_t out_w = output.shape[1];

    for (uint32_t c = 0; c < channels; ++c)
    {
        for (uint32_t oh = 0; oh < out_h; ++oh)
        {
            for (uint32_t ow = 0; ow < out_w; ++ow)
            {
                float max_val = -INFINITY;
                for (int kh = 0; kh < pool_size; ++kh)
                {
                    for (int kw = 0; kw < pool_size; ++kw)
                    {
                        const uint32_t ih = oh * static_cast<uint32_t>(stride) + static_cast<uint32_t>(kh);
                        const uint32_t iw = ow * static_cast<uint32_t>(stride) + static_cast<uint32_t>(kw);
                        const uint32_t in_idx = index_nhwc(input, ih, iw, c);
                        if (in[in_idx] > max_val)
                            max_val = in[in_idx];
                    }
                }

                const uint32_t out_idx = (oh * out_w + ow) * channels + c;
                out[out_idx] = max_val;
            }
        }
    }
}

CNNNetwork::CNNNetwork(uint32_t num_layers, Arena& arena)
    : blocks(nullptr), num_layers(num_layers)
{
    blocks = static_cast<CnnBlock*>(arena.alloc(sizeof(CnnBlock) * num_layers, alignof(CnnBlock)));
}

bool CNNNetwork::InitActivationBuffers(Arena& arena, uint32_t in_h, uint32_t in_w, uint32_t in_c)
{
    ping_a = nullptr;
    ping_b = nullptr;
    max_activation_elements = 0;
    output_cache_ = {};

    if (!blocks || num_layers == 0)
        return blocks != nullptr;

    uint32_t h = in_h;
    uint32_t w = in_w;
    uint32_t channels = in_c;

    for (uint32_t i = 0; i < num_layers; ++i)
    {
        switch (blocks[i].type)
        {
            case CnnBlockType::Conv2D:
            {
                const uint32_t out_h =
                    CalcOutputDim(h, blocks[i].conv.conv.kernel_size, blocks[i].conv.conv.stride);
                const uint32_t out_w =
                    CalcOutputDim(w, blocks[i].conv.conv.kernel_size, blocks[i].conv.conv.stride);
                const uint32_t out_c = static_cast<uint32_t>(blocks[i].conv.conv.out_channels);
                max_activation_elements = MaxU32(max_activation_elements, out_h * out_w * out_c);
                h = out_h;
                w = out_w;
                channels = out_c;
                break;
            }
            case CnnBlockType::MaxPool2D:
            {
                const uint32_t out_h = CalcOutputDim(h, blocks[i].pool.pool_size, blocks[i].pool.stride);
                const uint32_t out_w = CalcOutputDim(w, blocks[i].pool.pool_size, blocks[i].pool.stride);
                max_activation_elements = MaxU32(max_activation_elements, out_h * out_w * channels);
                h = out_h;
                w = out_w;
                break;
            }
            case CnnBlockType::Flatten:
            {
                const uint32_t features = h * w * channels;
                max_activation_elements = MaxU32(max_activation_elements, features);
                h = 1;
                w = features;
                channels = 1;
                break;
            }
            case CnnBlockType::Dense:
            {
                const uint32_t out_features = blocks[i].dense.weights.shape[1];
                max_activation_elements = MaxU32(max_activation_elements, out_features);
                w = out_features;
                break;
            }
        }
    }

    if (max_activation_elements == 0)
        return false;

    const std::size_t bytes = static_cast<std::size_t>(max_activation_elements) * sizeof(float);
    ping_a = static_cast<float*>(arena.alloc(bytes, alignof(float)));
    ping_b = static_cast<float*>(arena.alloc(bytes, alignof(float)));
    return ping_a != nullptr && ping_b != nullptr;
}

void CNNNetwork::InitConvLayer(uint32_t layer_idx,
                               int kernel_size,
                               int stride,
                               int in_channels,
                               int out_channels,
                               float* weights,
                               float* bias,
                               ConvActivationType activation,
                               float leaky_alpha)
{
    if (!blocks || layer_idx >= num_layers)
        return;

    blocks[layer_idx].type = CnnBlockType::Conv2D;
    blocks[layer_idx].conv.conv.kernel_size = kernel_size;
    blocks[layer_idx].conv.conv.stride = stride;
    blocks[layer_idx].conv.conv.in_channels = in_channels;
    blocks[layer_idx].conv.conv.out_channels = out_channels;
    blocks[layer_idx].conv.conv.weights = weights;
    blocks[layer_idx].conv.conv.bias = bias;
    blocks[layer_idx].conv.activation = activation;
    blocks[layer_idx].conv.leaky_alpha = leaky_alpha;
}

void CNNNetwork::InitPoolLayer(uint32_t layer_idx, int pool_size, int stride)
{
    if (!blocks || layer_idx >= num_layers)
        return;

    blocks[layer_idx].type = CnnBlockType::MaxPool2D;
    blocks[layer_idx].pool.pool_size = pool_size;
    blocks[layer_idx].pool.stride = stride;
}

void CNNNetwork::InitFlattenLayer(uint32_t layer_idx)
{
    if (!blocks || layer_idx >= num_layers)
        return;

    blocks[layer_idx].type = CnnBlockType::Flatten;
}

void CNNNetwork::InitDenseLayer(uint32_t layer_idx,
                                const Tensor& weights,
                                const Tensor& bias,
                                ActivationType activation,
                                float leaky_alpha)
{
    if (!blocks || layer_idx >= num_layers)
        return;

    blocks[layer_idx].type = CnnBlockType::Dense;
    blocks[layer_idx].dense.weights = weights;
    blocks[layer_idx].dense.bias = bias;
    blocks[layer_idx].dense.activation = activation;
    blocks[layer_idx].dense.leaky_alpha = leaky_alpha;
}

Tensor& CNNNetwork::forward(const Tensor& input, Arena& /*arena*/)
{
    static Tensor empty{};

    if (!IsValid() || !HasActivationBuffers() || num_layers == 0)
        return empty;

    output_cache_ = {};
    Tensor current_input = input;
    float* write_buffer = ping_a;

    for (uint32_t i = 0; i < num_layers; ++i)
    {
        Tensor layer_output{};

        switch (blocks[i].type)
        {
            case CnnBlockType::Conv2D:
            {
                const uint32_t out_h =
                    CalcOutputDim(current_input.shape[0], blocks[i].conv.conv.kernel_size, blocks[i].conv.conv.stride);
                const uint32_t out_w =
                    CalcOutputDim(current_input.shape[1], blocks[i].conv.conv.kernel_size, blocks[i].conv.conv.stride);
                const uint32_t out_c = static_cast<uint32_t>(blocks[i].conv.conv.out_channels);
                const std::array<uint32_t, 3> shape = {out_h, out_w, out_c};
                layer_output = ViewND(write_buffer, 3, shape);
                break;
            }
            case CnnBlockType::MaxPool2D:
            {
                const uint32_t out_h =
                    CalcOutputDim(current_input.shape[0], blocks[i].pool.pool_size, blocks[i].pool.stride);
                const uint32_t out_w =
                    CalcOutputDim(current_input.shape[1], blocks[i].pool.pool_size, blocks[i].pool.stride);
                const uint32_t out_c = current_input.shape[2];
                const std::array<uint32_t, 3> shape = {out_h, out_w, out_c};
                layer_output = ViewND(write_buffer, 3, shape);
                break;
            }
            case CnnBlockType::Flatten:
            {
                const uint32_t features = current_input.num_elements;
                layer_output = View2D(write_buffer, 1, features);
                break;
            }
            case CnnBlockType::Dense:
            {
                const uint32_t out_features = blocks[i].dense.weights.shape[1];
                layer_output = View2D(write_buffer, 1, out_features);
                break;
            }
        }

        if (!layer_output.data || layer_output.num_elements > max_activation_elements)
            return empty;

        switch (blocks[i].type)
        {
            case CnnBlockType::Conv2D:
                blocks[i].conv.forward(current_input, layer_output);
                break;
            case CnnBlockType::MaxPool2D:
                blocks[i].pool.forward(current_input, layer_output);
                break;
            case CnnBlockType::Flatten:
                FlattenNhwc(current_input, layer_output);
                break;
            case CnnBlockType::Dense:
                blocks[i].dense.forward(current_input, layer_output);
                break;
        }

        current_input = layer_output;
        output_cache_ = layer_output;
        write_buffer = (write_buffer == ping_a) ? ping_b : ping_a;
    }

    return output_cache_;
}
