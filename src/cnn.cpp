#include "cnn.hpp"
#include "active_kernel.hpp"
#include "activation_followup.hpp"
#include "ops_resolver.hpp"
#include "tensor_factory.hpp"
#include <cstring>

using namespace TensorFactory;

const NkOpsResolver& CNNNetwork::GetOpsResolver() const
{
    if (has_custom_resolver_)
        return op_resolver_;
    return GetDefaultOpsResolver();
}

void Conv2DLayer::forward(const Tensor& input, Tensor& output)
{
    const NetkitKernelActivation kernel_activation = ToKernelActivation(activation);
    const bool fused_in_kernel = conv.forward(input, output, kernel_activation);
    ApplyFusedOutputActivation(kernel_activation, fused_in_kernel, output, leaky_alpha);
}

void DepthwiseConv2DLayer::forward(const Tensor& input, Tensor& output)
{
    const NetkitKernelActivation kernel_activation = ToKernelActivation(activation);
    const bool fused_in_kernel = depthwise.forward(input, output, kernel_activation);
    ApplyFusedOutputActivation(kernel_activation, fused_in_kernel, output, leaky_alpha);
}

void MaxPool2DLayer::forward(const Tensor& input, Tensor& output)
{
    Kernels::MaxPool2dForward(input, pool_size, stride, pad_h, pad_w, output);
}

void AvgPool2DLayer::forward(const Tensor& input, Tensor& output)
{
    Kernels::AvgPool2dForward(input, pool_size, stride, pad_h, pad_w, output);
}

void BatchNorm2DLayer::forward(const Tensor& input, Tensor& output)
{
    Kernels::BatchNorm2dForward(input, scale, bias, channels, output);
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

    const NkOpsResolver& resolver = GetOpsResolver();
    NkCnnSpatialPlan plan{in_h, in_w, in_c, &max_activation_elements};

    for (uint32_t i = 0; i < num_layers; ++i)
    {
        const NkLayerOpRegistration* registration =
            resolver.Find(static_cast<uint8_t>(ToOpCode(blocks[i].type)));
        if (!registration || !registration->plan_activation)
            return false;

        if (!registration->plan_activation(blocks[i], plan))
            return false;
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
                               float leaky_alpha,
                               int pad_h,
                               int pad_w)
{
    if (!blocks || layer_idx >= num_layers)
        return;

    blocks[layer_idx].type = CnnBlockType::Conv2D;
    blocks[layer_idx].conv.conv.kernel_size = kernel_size;
    blocks[layer_idx].conv.conv.stride = stride;
    blocks[layer_idx].conv.conv.pad_h = pad_h;
    blocks[layer_idx].conv.conv.pad_w = pad_w;
    blocks[layer_idx].conv.conv.in_channels = in_channels;
    blocks[layer_idx].conv.conv.out_channels = out_channels;
    blocks[layer_idx].conv.conv.weights = weights;
    blocks[layer_idx].conv.conv.bias = bias;
    blocks[layer_idx].conv.activation = activation;
    blocks[layer_idx].conv.leaky_alpha = leaky_alpha;
}

void CNNNetwork::InitDepthwiseConvLayer(uint32_t layer_idx,
                                        int kernel_h,
                                        int kernel_w,
                                        int stride,
                                        int channels,
                                        float* weights,
                                        float* bias,
                                        ConvActivationType activation,
                                        float leaky_alpha,
                                        int pad_h,
                                        int pad_w)
{
    if (!blocks || layer_idx >= num_layers)
        return;

    blocks[layer_idx].type = CnnBlockType::DepthwiseConv2D;
    blocks[layer_idx].depthwise_conv.depthwise.kernel_h = kernel_h;
    blocks[layer_idx].depthwise_conv.depthwise.kernel_w = kernel_w;
    blocks[layer_idx].depthwise_conv.depthwise.stride = stride;
    blocks[layer_idx].depthwise_conv.depthwise.pad_h = pad_h;
    blocks[layer_idx].depthwise_conv.depthwise.pad_w = pad_w;
    blocks[layer_idx].depthwise_conv.depthwise.channels = channels;
    blocks[layer_idx].depthwise_conv.depthwise.weights = weights;
    blocks[layer_idx].depthwise_conv.depthwise.bias = bias;
    blocks[layer_idx].depthwise_conv.activation = activation;
    blocks[layer_idx].depthwise_conv.leaky_alpha = leaky_alpha;
}

void CNNNetwork::InitPoolLayer(uint32_t layer_idx, int pool_size, int stride, int pad_h, int pad_w)
{
    if (!blocks || layer_idx >= num_layers)
        return;

    blocks[layer_idx].type = CnnBlockType::MaxPool2D;
    blocks[layer_idx].pool.pool_size = pool_size;
    blocks[layer_idx].pool.stride = stride;
    blocks[layer_idx].pool.pad_h = pad_h;
    blocks[layer_idx].pool.pad_w = pad_w;
}

void CNNNetwork::InitAvgPoolLayer(uint32_t layer_idx, int pool_size, int stride, int pad_h, int pad_w)
{
    if (!blocks || layer_idx >= num_layers)
        return;

    blocks[layer_idx].type = CnnBlockType::AvgPool2D;
    blocks[layer_idx].avg_pool.pool_size = pool_size;
    blocks[layer_idx].avg_pool.stride = stride;
    blocks[layer_idx].avg_pool.pad_h = pad_h;
    blocks[layer_idx].avg_pool.pad_w = pad_w;
}

void CNNNetwork::InitBatchNormLayer(uint32_t layer_idx, int channels, float* scale, float* bias)
{
    if (!blocks || layer_idx >= num_layers)
        return;

    blocks[layer_idx].type = CnnBlockType::BatchNorm2d;
    blocks[layer_idx].batch_norm.channels = channels;
    blocks[layer_idx].batch_norm.scale = scale;
    blocks[layer_idx].batch_norm.bias = bias;
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

    const NkOpsResolver& resolver = GetOpsResolver();
    output_cache_ = {};
    Tensor current_input = input;
    float* write_buffer = ping_a;

    for (uint32_t i = 0; i < num_layers; ++i)
    {
        const NkLayerOpRegistration* registration =
            resolver.Find(static_cast<uint8_t>(ToOpCode(blocks[i].type)));
        if (!registration || !registration->prepare_output || !registration->eval)
            return empty;

        Tensor layer_output{};
        NkCnnOpContext ctx{blocks[i], current_input, layer_output, write_buffer, max_activation_elements};
        if (!registration->prepare_output(ctx) || !layer_output.data)
            return empty;

        registration->eval(blocks[i], current_input, layer_output);

        current_input = layer_output;
        output_cache_ = layer_output;
        write_buffer = (write_buffer == ping_a) ? ping_b : ping_a;
    }

    return output_cache_;
}
