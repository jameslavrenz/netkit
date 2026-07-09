#include "cnn.hpp"
#include "active_kernel.hpp"
#include "activation_followup.hpp"
#include "conv2d_layout.hpp"
#include "kernel_workspace.hpp"
#include "cmsis_buffer_size.hpp"
#include "ops_resolver.hpp"
#include "tensor_factory.hpp"
#include <array>
#include <chrono>
#include <cstring>

using namespace TensorFactory;

const NkOpsResolver& CNNNetwork::GetOpsResolver() const
{
#if defined(NETKIT_MCU_QUANT_ONLY) && NETKIT_MCU_QUANT_ONLY
    static NkOpsResolver empty;
    return empty;
#else
    if (has_custom_resolver_)
        return op_resolver_;
    return GetDefaultOpsResolver();
#endif
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
    const int pad_h_end = this->pad_h_end;
    const int pad_w_end = this->pad_w_end;
    const NetkitKernelActivation kernel_activation = ToKernelActivation(activation);
    bool fused_in_kernel = false;
    if (pool_h == pool_w && pad_h == pad_h_end && pad_w == pad_w_end)
        fused_in_kernel =
            Kernels::MaxPool2dForward(input, pool_h, stride, pad_h, pad_w, kernel_activation, output);
    else
        fused_in_kernel = Kernels::MaxPool2dForwardPadded(input,
                                                          pool_h,
                                                          pool_w,
                                                          stride,
                                                          pad_h,
                                                          pad_w,
                                                          pad_h_end,
                                                          pad_w_end,
                                                          kernel_activation,
                                                          output);
    ApplyFusedOutputActivation(kernel_activation, fused_in_kernel, output);
}

void AvgPool2DLayer::forward(const Tensor& input, Tensor& output)
{
    const int pad_h_end = this->pad_h_end;
    const int pad_w_end = this->pad_w_end;
    if (pool_h == pool_w && pad_h == pad_h_end && pad_w == pad_w_end)
        Kernels::AvgPool2dForward(input, pool_h, stride, pad_h, pad_w, output);
    else
        Kernels::AvgPool2dForwardPadded(
            input, pool_h, pool_w, stride, pad_h, pad_w, pad_h_end, pad_w_end, output);
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

void YoloxDecoupledHeadLayer::forward(const Tensor& input, Tensor& output)
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
    kernel_workspace_ = nullptr;
    kernel_workspace_bytes_ = 0;
    max_activation_elements = 0;
    layer_output_views_ = nullptr;
    output_cache_ = {};

    if (!blocks || num_layers == 0)
        return blocks != nullptr;

    for (size_t i = 0; i < num_layers; ++i)
    {
        if (blocks[i].type == CnnBlockType::Conv2D &&
            !RepackConv2dWeights(blocks[i].conv.conv, arena))
        {
            return false;
        }
    }

    const NkOpsResolver& resolver = GetOpsResolver();
    std::size_t max_kernel_workspace_bytes = 0;
    NkCnnSpatialPlan plan{in_h, in_w, in_c, &max_activation_elements};

    CmsisBeginKernelWorkspacePlan(&max_kernel_workspace_bytes);
    for (size_t i = 0; i < num_layers; ++i)
    {
        const NkLayerOpRegistration* registration =
            resolver.Find(static_cast<uint8_t>(ToOpCode(blocks[i].type)));
        if (!registration || !registration->plan_activation)
        {
            CmsisEndKernelWorkspacePlan();
            return false;
        }

        if (!registration->plan_activation(blocks[i], plan))
        {
            CmsisEndKernelWorkspacePlan();
            return false;
        }
    }
    CmsisEndKernelWorkspacePlan();

    if (max_activation_elements == 0)
        return false;

    const std::size_t gelu_workspace_bytes = CmsisGeluWorkspaceBytes(max_activation_elements);
    if (gelu_workspace_bytes > max_kernel_workspace_bytes)
        max_kernel_workspace_bytes = gelu_workspace_bytes;

    const std::size_t bytes = static_cast<std::size_t>(max_activation_elements) * sizeof(float);
    ping_a = static_cast<float*>(arena.alloc(bytes, alignof(float)));
    ping_b = static_cast<float*>(arena.alloc(bytes, alignof(float)));
    if (!ping_a || !ping_b)
        return false;

    if (max_kernel_workspace_bytes > 0)
    {
        kernel_workspace_ = static_cast<uint8_t*>(
            arena.alloc(max_kernel_workspace_bytes, alignof(std::max_align_t)));
        if (!kernel_workspace_)
            return false;
        kernel_workspace_bytes_ = max_kernel_workspace_bytes;
    }

    layer_output_views_ =
        static_cast<Tensor*>(arena.alloc(sizeof(Tensor) * num_layers, alignof(Tensor)));
    if (!layer_output_views_)
        return false;

    const std::array<uint32_t, 3> input_shape = {in_h, in_w, in_c};
    Tensor current_input = ViewND(nullptr, 3, input_shape);
    for (size_t i = 0; i < num_layers; ++i)
    {
        const NkLayerOpRegistration* registration =
            resolver.Find(static_cast<uint8_t>(ToOpCode(blocks[i].type)));
        if (!registration || !registration->prepare_output)
            return false;

        layer_output_views_[i] = {};
        NkCnnOpContext ctx{
            blocks[i], current_input, layer_output_views_[i], ping_a, max_activation_elements};
        if (!registration->prepare_output(ctx))
            return false;

        current_input = layer_output_views_[i];
        current_input.data = nullptr;
    }

    return true;
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
                               int pad_w,
                               int pad_h_end,
                               int pad_w_end)
{
    if (!blocks || layer_idx >= num_layers)
        return;

    blocks[layer_idx].type = CnnBlockType::Conv2D;
    blocks[layer_idx].conv.conv.kernel_size = kernel_size;
    blocks[layer_idx].conv.conv.stride = stride;
    blocks[layer_idx].conv.conv.pad_h = pad_h;
    blocks[layer_idx].conv.conv.pad_w = pad_w;
    blocks[layer_idx].conv.conv.pad_h_end = pad_h_end >= 0 ? pad_h_end : pad_h;
    blocks[layer_idx].conv.conv.pad_w_end = pad_w_end >= 0 ? pad_w_end : pad_w;
    blocks[layer_idx].conv.conv.in_channels = in_channels;
    blocks[layer_idx].conv.conv.out_channels = out_channels;
    blocks[layer_idx].conv.conv.weights = weights;
    blocks[layer_idx].conv.conv.weights_hwio = nullptr;
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
                                        int pad_w,
                                        int pad_h_end,
                                        int pad_w_end)
{
    if (!blocks || layer_idx >= num_layers)
        return;

    blocks[layer_idx].type = CnnBlockType::DepthwiseConv2D;
    blocks[layer_idx].depthwise_conv.depthwise.kernel_h = kernel_h;
    blocks[layer_idx].depthwise_conv.depthwise.kernel_w = kernel_w;
    blocks[layer_idx].depthwise_conv.depthwise.stride = stride;
    blocks[layer_idx].depthwise_conv.depthwise.pad_h = pad_h;
    blocks[layer_idx].depthwise_conv.depthwise.pad_w = pad_w;
    blocks[layer_idx].depthwise_conv.depthwise.pad_h_end = pad_h_end >= 0 ? pad_h_end : pad_h;
    blocks[layer_idx].depthwise_conv.depthwise.pad_w_end = pad_w_end >= 0 ? pad_w_end : pad_w;
    blocks[layer_idx].depthwise_conv.depthwise.channels = channels;
    blocks[layer_idx].depthwise_conv.depthwise.weights = weights;
    blocks[layer_idx].depthwise_conv.depthwise.bias = bias;
    blocks[layer_idx].depthwise_conv.activation = activation;
    blocks[layer_idx].depthwise_conv.leaky_alpha = leaky_alpha;
}

void CNNNetwork::InitPoolLayer(uint32_t layer_idx,
                               int pool_h,
                               int pool_w,
                               int stride,
                               int pad_h,
                               int pad_w,
                               int pad_h_end,
                               int pad_w_end)
{
    if (!blocks || layer_idx >= num_layers)
        return;

    blocks[layer_idx].type = CnnBlockType::MaxPool2D;
    blocks[layer_idx].pool.pool_h = pool_h;
    blocks[layer_idx].pool.pool_w = pool_w;
    blocks[layer_idx].pool.stride = stride;
    blocks[layer_idx].pool.pad_h = pad_h;
    blocks[layer_idx].pool.pad_w = pad_w;
    blocks[layer_idx].pool.pad_h_end = pad_h_end >= 0 ? pad_h_end : pad_h;
    blocks[layer_idx].pool.pad_w_end = pad_w_end >= 0 ? pad_w_end : pad_w;
}

void CNNNetwork::InitAvgPoolLayer(uint32_t layer_idx,
                                  int pool_h,
                                  int pool_w,
                                  int stride,
                                  int pad_h,
                                  int pad_w,
                                  int pad_h_end,
                                  int pad_w_end)
{
    if (!blocks || layer_idx >= num_layers)
        return;

    blocks[layer_idx].type = CnnBlockType::AvgPool2D;
    blocks[layer_idx].avg_pool.pool_h = pool_h;
    blocks[layer_idx].avg_pool.pool_w = pool_w;
    blocks[layer_idx].avg_pool.stride = stride;
    blocks[layer_idx].avg_pool.pad_h = pad_h;
    blocks[layer_idx].avg_pool.pad_w = pad_w;
    blocks[layer_idx].avg_pool.pad_h_end = pad_h_end >= 0 ? pad_h_end : pad_h;
    blocks[layer_idx].avg_pool.pad_w_end = pad_w_end >= 0 ? pad_w_end : pad_w;
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

void CNNNetwork::InitYoloxDecoupledHeadLayer(uint32_t layer_idx,
                                             Arena& arena,
                                             uint32_t spatial_h,
                                             uint32_t spatial_w,
                                             int in_channels,
                                             int hidden_dim,
                                             int num_classes,
                                             int num_convs,
                                             float* stem_weights,
                                             float* stem_bias,
                                             float* const* cls_conv_weights,
                                             float* const* cls_conv_bias,
                                             float* const* reg_conv_weights,
                                             float* const* reg_conv_bias,
                                             float* cls_pred_weights,
                                             float* cls_pred_bias,
                                             float* reg_pred_weights,
                                             float* reg_pred_bias,
                                             float* obj_pred_weights,
                                             float* obj_pred_bias)
{
    if (!blocks || layer_idx >= num_layers)
        return;

    blocks[layer_idx].type = CnnBlockType::YoloxDecoupledHead;
    YoloxDecoupledHead& block = blocks[layer_idx].yolox_decoupled_head.block;
    block.in_channels = in_channels;
    block.hidden_dim = hidden_dim;
    block.num_classes = num_classes;
    block.num_convs = num_convs;
    block.stem_weights = stem_weights;
    block.stem_bias = stem_bias;
    block.cls_pred_weights = cls_pred_weights;
    block.cls_pred_bias = cls_pred_bias;
    block.reg_pred_weights = reg_pred_weights;
    block.reg_pred_bias = reg_pred_bias;
    block.obj_pred_weights = obj_pred_weights;
    block.obj_pred_bias = obj_pred_bias;

    for (int i = 0; i < num_convs && i < YoloxDecoupledHead::kMaxStackedConvs; ++i)
    {
        block.cls_conv_weights[i] = cls_conv_weights[i];
        block.cls_conv_bias[i] = cls_conv_bias[i];
        block.reg_conv_weights[i] = reg_conv_weights[i];
        block.reg_conv_bias[i] = reg_conv_bias[i];
    }

    const uint32_t spatial = spatial_h * spatial_w;
    const uint32_t hidden = static_cast<uint32_t>(hidden_dim);
    const uint32_t scratch_elems = spatial * hidden * 3u;
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

    if (quantized_)
        return forward_quantized(input);

#if defined(NETKIT_MCU_QUANT_ONLY) && NETKIT_MCU_QUANT_ONLY
    return empty;
#else
    KernelWorkspace workspace{kernel_workspace_, kernel_workspace_bytes_};
    KernelWorkspaceScope workspace_scope(&workspace);

    const NkOpsResolver& resolver = GetOpsResolver();
    output_cache_ = {};
    Tensor current_input = input;
    float* write_buffer = ping_a;

    for (size_t i = 0; i < num_layers; ++i)
    {
        const NkLayerOpRegistration* registration =
            resolver.Find(static_cast<uint8_t>(ToOpCode(blocks[i].type)));
        if (!registration || !registration->eval)
            return empty;

        Tensor& layer_output = layer_output_views_[i];
        layer_output.data = write_buffer;
        if (!layer_output.data || layer_output.num_elements > max_activation_elements)
            return empty;

        registration->eval(blocks[i], current_input, layer_output);

        current_input = layer_output;
        output_cache_ = layer_output;
        write_buffer = (write_buffer == ping_a) ? ping_b : ping_a;
    }

    return output_cache_;
#endif
}

namespace {

const char* ProfileTagForBlock(const CnnBlock& block)
{
    switch (block.type)
    {
    case CnnBlockType::Conv2D:
        return "Conv2D";
    case CnnBlockType::MaxPool2D:
        return "MaxPool2D";
    case CnnBlockType::Flatten:
        return "Reshape";
    case CnnBlockType::Dense:
        return "FullyConnected";
    default:
        return "Unknown";
    }
}

}  // namespace

Tensor& CNNNetwork::forward_timed(const Tensor& input, Arena& /*arena*/, LayerTimingFn timing_fn,
                                  void* user_data)
{
    static Tensor empty{};

    if (!IsValid() || !HasActivationBuffers() || num_layers == 0)
        return empty;

    KernelWorkspace workspace{kernel_workspace_, kernel_workspace_bytes_};
    KernelWorkspaceScope workspace_scope(&workspace);

    const NkOpsResolver& resolver = GetOpsResolver();
    output_cache_ = {};
    Tensor current_input = input;
    float* write_buffer = ping_a;

    for (size_t i = 0; i < num_layers; ++i)
    {
        const NkLayerOpRegistration* registration =
            resolver.Find(static_cast<uint8_t>(ToOpCode(blocks[i].type)));
        if (!registration || !registration->eval)
            return empty;

        Tensor& layer_output = layer_output_views_[i];
        layer_output.data = write_buffer;
        if (!layer_output.data || layer_output.num_elements > max_activation_elements)
            return empty;

        const char* tag = ProfileTagForBlock(blocks[i]);
        const auto layer_start = std::chrono::steady_clock::now();
        registration->eval(blocks[i], current_input, layer_output);
        const auto layer_end = std::chrono::steady_clock::now();

        if (timing_fn)
        {
            const uint64_t duration_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(layer_end - layer_start)
                    .count());
            timing_fn(tag, duration_ns, user_data);
        }

        current_input = layer_output;
        output_cache_ = layer_output;
        write_buffer = (write_buffer == ping_a) ? ping_b : ping_a;
    }

    return output_cache_;
}
