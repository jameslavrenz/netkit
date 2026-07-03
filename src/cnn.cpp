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

void LayerNorm2DLayer::forward(const Tensor& input, Tensor& output)
{
    Kernels::LayerNorm2dForward(input, weight, bias, channels, eps, output);
}

void ConvNeXtV2BlockLayer::forward(const Tensor& input, Tensor& output)
{
    block.forward(input, output);
}

void MobilenetV4UibLayer::forward(const Tensor& input, Tensor& output)
{
    block.forward(input, output);
}

void ResNetBasicBlockLayer::forward(const Tensor& input, Tensor& output)
{
    block.forward(input, output);
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

void CNNNetwork::InitLayerNormLayer(uint32_t layer_idx,
                                    int channels,
                                    float eps,
                                    float* weight,
                                    float* bias)
{
    if (!blocks || layer_idx >= num_layers)
        return;

    blocks[layer_idx].type = CnnBlockType::LayerNorm2d;
    blocks[layer_idx].layer_norm.channels = channels;
    blocks[layer_idx].layer_norm.eps = eps;
    blocks[layer_idx].layer_norm.weight = weight;
    blocks[layer_idx].layer_norm.bias = bias;
}

void CNNNetwork::InitConvNeXtV2BlockLayer(uint32_t layer_idx,
                                            Arena& arena,
                                            uint32_t spatial_h,
                                            uint32_t spatial_w,
                                            int channels,
                                            float eps,
                                            float* dw_weights,
                                            float* dw_bias,
                                            float* ln_weight,
                                            float* ln_bias,
                                            float* pw1_weight,
                                            float* pw1_bias,
                                            float* grn_gamma,
                                            float* grn_beta,
                                            float* pw2_weight,
                                            float* pw2_bias)
{
    if (!blocks || layer_idx >= num_layers)
        return;

    blocks[layer_idx].type = CnnBlockType::ConvNeXtV2Block;
    ConvNeXtV2Block& block = blocks[layer_idx].convnextv2_block.block;
    block.channels = channels;
    block.eps = eps;
    block.dw_weights = dw_weights;
    block.dw_bias = dw_bias;
    block.ln_weight = ln_weight;
    block.ln_bias = ln_bias;
    block.pw1_weight = pw1_weight;
    block.pw1_bias = pw1_bias;
    block.grn_gamma = grn_gamma;
    block.grn_beta = grn_beta;
    block.pw2_weight = pw2_weight;
    block.pw2_bias = pw2_bias;

    const uint32_t expanded =
        static_cast<uint32_t>(channels) * static_cast<uint32_t>(ConvNeXtV2Block::kMlpRatio);
    const uint32_t scratch_elems = spatial_h * spatial_w * expanded + expanded;
    block.scratch =
        static_cast<float*>(arena.alloc(static_cast<std::size_t>(scratch_elems) * sizeof(float),
                                        alignof(float)));
    block.scratch_elems = block.scratch ? scratch_elems : 0;
}

void CNNNetwork::InitMobilenetV4UibLayer(uint32_t layer_idx,
                                         Arena& arena,
                                         uint32_t spatial_h,
                                         uint32_t spatial_w,
                                         int in_channels,
                                         int out_channels,
                                         int start_dw_kernel,
                                         int middle_dw_kernel,
                                         int stride,
                                         bool middle_dw_downsample,
                                         float expand_ratio,
                                         float* start_dw_weights,
                                         float* start_dw_bias,
                                         float* start_bn_scale,
                                         float* start_bn_bias,
                                         float* expand_weights,
                                         float* expand_bias,
                                         float* expand_bn_scale,
                                         float* expand_bn_bias,
                                         float* middle_dw_weights,
                                         float* middle_dw_bias,
                                         float* middle_bn_scale,
                                         float* middle_bn_bias,
                                         float* proj_weights,
                                         float* proj_bias,
                                         float* proj_bn_scale,
                                         float* proj_bn_bias)
{
    if (!blocks || layer_idx >= num_layers)
        return;

    blocks[layer_idx].type = CnnBlockType::MobilenetV4Uib;
    MobileNetV4Uib& block = blocks[layer_idx].mobilenetv4_uib.block;
    block.in_channels = in_channels;
    block.out_channels = out_channels;
    block.start_dw_kernel = start_dw_kernel;
    block.middle_dw_kernel = middle_dw_kernel;
    block.stride = stride;
    block.middle_dw_downsample = middle_dw_downsample;
    block.expand_ratio = expand_ratio;
    block.start_dw_weights = start_dw_weights;
    block.start_dw_bias = start_dw_bias;
    block.start_bn_scale = start_bn_scale;
    block.start_bn_bias = start_bn_bias;
    block.expand_weights = expand_weights;
    block.expand_bias = expand_bias;
    block.expand_bn_scale = expand_bn_scale;
    block.expand_bn_bias = expand_bn_bias;
    block.middle_dw_weights = middle_dw_weights;
    block.middle_dw_bias = middle_dw_bias;
    block.middle_bn_scale = middle_bn_scale;
    block.middle_bn_bias = middle_bn_bias;
    block.proj_weights = proj_weights;
    block.proj_bias = proj_bias;
    block.proj_bn_scale = proj_bn_scale;
    block.proj_bn_bias = proj_bn_bias;

    const uint32_t spatial = spatial_h * spatial_w;
    const uint32_t expand_c = block.expanded_channels();
    const uint32_t residual =
        (stride == 1 && in_channels == out_channels) ? static_cast<uint32_t>(in_channels) : 0u;
    const uint32_t scratch_elems = 2u * spatial * expand_c + spatial * residual;
    block.scratch =
        static_cast<float*>(arena.alloc(static_cast<std::size_t>(scratch_elems) * sizeof(float),
                                        alignof(float)));
    block.scratch_elems = block.scratch ? scratch_elems : 0;
}

void CNNNetwork::InitResNetBasicBlockLayer(uint32_t layer_idx,
                                           Arena& arena,
                                           uint32_t spatial_h,
                                           uint32_t spatial_w,
                                           int in_channels,
                                           int out_channels,
                                           int stride,
                                           float* conv1_weights,
                                           float* conv1_bias,
                                           float* bn1_scale,
                                           float* bn1_bias,
                                           float* conv2_weights,
                                           float* conv2_bias,
                                           float* bn2_scale,
                                           float* bn2_bias,
                                           float* shortcut_weights,
                                           float* shortcut_bias,
                                           float* shortcut_bn_scale,
                                           float* shortcut_bn_bias)
{
    if (!blocks || layer_idx >= num_layers)
        return;

    blocks[layer_idx].type = CnnBlockType::ResNetBasicBlock;
    ResNetBasicBlock& block = blocks[layer_idx].resnet_basic_block.block;
    block.in_channels = in_channels;
    block.out_channels = out_channels;
    block.stride = stride;
    block.conv1_weights = conv1_weights;
    block.conv1_bias = conv1_bias;
    block.bn1_scale = bn1_scale;
    block.bn1_bias = bn1_bias;
    block.conv2_weights = conv2_weights;
    block.conv2_bias = conv2_bias;
    block.bn2_scale = bn2_scale;
    block.bn2_bias = bn2_bias;
    block.shortcut_weights = shortcut_weights;
    block.shortcut_bias = shortcut_bias;
    block.shortcut_bn_scale = shortcut_bn_scale;
    block.shortcut_bn_bias = shortcut_bn_bias;

    uint32_t out_h = spatial_h;
    uint32_t out_w = spatial_w;
    block.output_spatial(spatial_h, spatial_w, out_h, out_w);
    const uint32_t out_elems = out_h * out_w * static_cast<uint32_t>(out_channels);
    const uint32_t scratch_elems =
        2u * out_elems + (block.has_identity_shortcut() ? 0u : out_elems);
    block.scratch =
        static_cast<float*>(arena.alloc(static_cast<std::size_t>(scratch_elems) * sizeof(float),
                                        alignof(float)));
    block.scratch_elems = block.scratch ? scratch_elems : 0;
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
