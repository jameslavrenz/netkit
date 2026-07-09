#include "cnn.hpp"
#include "cmsis_quant_plan.hpp"
#include "netkit_config.h"
#include "tensor_factory.hpp"

#include <algorithm>

bool CNNNetwork::InitQuantizedActivationBuffers(Arena& arena, uint32_t in_h, uint32_t in_w, uint32_t in_c)
{
    ping_a = nullptr;
    ping_b = nullptr;
    ping_i8_a = nullptr;
    ping_i8_b = nullptr;
    kernel_workspace_ = nullptr;
    kernel_workspace_bytes_ = 0;
    max_activation_elements = 0;
    layer_output_views_ = nullptr;
    output_cache_ = {};
    float_output_buffer_ = nullptr;
    float_output_elements_ = 0;
    quant_runtime_ = nullptr;

    if (!blocks || num_layers == 0)
        return blocks != nullptr;

    if (!CmsisQuantPlan::BuildRuntime(*this, arena, in_h, in_w, in_c))
        return false;

    const CmsisQuantPlan::Runtime* runtime = quant_runtime_;
    if (!runtime)
        return false;

    ping_i8_a = runtime->act_a;
    ping_i8_b = runtime->act_b;
    kernel_workspace_ = runtime->workspace;
    kernel_workspace_bytes_ = runtime->workspace_bytes;
    max_activation_elements = std::max(runtime->act_a_bytes, runtime->act_b_bytes);
    return true;
}

void CNNNetwork::InitQuantizedConvLayer(uint32_t layer_idx,
                                        int kernel_size,
                                        int stride,
                                        int in_channels,
                                        int out_channels,
                                        int8_t* weights,
                                        int32_t* bias,
                                        const NkFormat::MlpLayerQuantDesc& quant,
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
    Conv2D& conv = blocks[layer_idx].conv.conv;
    conv.kernel_size = kernel_size;
    conv.stride = stride;
    conv.pad_h = pad_h;
    conv.pad_w = pad_w;
    conv.pad_h_end = pad_h_end >= 0 ? pad_h_end : pad_h;
    conv.pad_w_end = pad_w_end >= 0 ? pad_w_end : pad_w;
    conv.in_channels = in_channels;
    conv.out_channels = out_channels;
    conv.weights = nullptr;
    conv.weights_hwio = nullptr;
    conv.bias = nullptr;
    conv.weights_q = weights;
    conv.bias_q = bias;
    blocks[layer_idx].conv.activation = activation;
    blocks[layer_idx].conv.leaky_alpha = leaky_alpha;
    blocks[layer_idx].conv.quant.params = quant;
    blocks[layer_idx].conv.quant.enabled = true;
}

void CNNNetwork::InitQuantizedDepthwiseConvLayer(uint32_t layer_idx,
                                                 int kernel_h,
                                                 int kernel_w,
                                                 int stride,
                                                 int channels,
                                                 int8_t* weights,
                                                 int32_t* bias,
                                                 const NkFormat::MlpLayerQuantDesc& quant,
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
    DepthwiseConv2D& dw = blocks[layer_idx].depthwise_conv.depthwise;
    dw.kernel_h = kernel_h;
    dw.kernel_w = kernel_w;
    dw.stride = stride;
    dw.pad_h = pad_h;
    dw.pad_w = pad_w;
    dw.pad_h_end = pad_h_end >= 0 ? pad_h_end : pad_h;
    dw.pad_w_end = pad_w_end >= 0 ? pad_w_end : pad_w;
    dw.channels = channels;
    dw.weights = nullptr;
    dw.bias = nullptr;
    dw.weights_q = weights;
    dw.bias_q = bias;
    blocks[layer_idx].depthwise_conv.activation = activation;
    blocks[layer_idx].depthwise_conv.leaky_alpha = leaky_alpha;
    blocks[layer_idx].depthwise_conv.quant.params = quant;
    blocks[layer_idx].depthwise_conv.quant.enabled = true;
}

void CNNNetwork::InitQuantizedDenseLayer(uint32_t layer_idx,
                                         const Tensor& weights,
                                         const Tensor& bias,
                                         const NkFormat::MlpLayerQuantDesc& quant,
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
    blocks[layer_idx].dense.quant.params = quant;
    blocks[layer_idx].dense.quant.enabled = true;
}

void CNNNetwork::InitQuantizedMobilenetV4UibLayer(uint32_t layer_idx,
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
                                                  float block_input_scale,
                                                  int32_t block_input_zero_point,
                                                  int8_t* start_dw_weights,
                                                  int32_t* start_dw_bias,
                                                  const NkFormat::MlpLayerQuantDesc& start_dw_quant,
                                                  int8_t* expand_weights,
                                                  int32_t* expand_bias,
                                                  const NkFormat::MlpLayerQuantDesc& expand_quant,
                                                  int8_t* middle_dw_weights,
                                                  int32_t* middle_dw_bias,
                                                  const NkFormat::MlpLayerQuantDesc& middle_dw_quant,
                                                  int8_t* proj_weights,
                                                  int32_t* proj_bias,
                                                  const NkFormat::MlpLayerQuantDesc& proj_quant)
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
    block.quant_enabled = true;
    block.block_input_scale = block_input_scale;
    block.block_input_zero_point = block_input_zero_point;
    block.start_dw_weights_q = start_dw_weights;
    block.start_dw_bias_q = start_dw_bias;
    block.start_dw_quant = start_dw_quant;
    block.expand_weights_q = expand_weights;
    block.expand_bias_q = expand_bias;
    block.expand_quant = expand_quant;
    block.middle_dw_weights_q = middle_dw_weights;
    block.middle_dw_bias_q = middle_dw_bias;
    block.middle_dw_quant = middle_dw_quant;
    block.proj_weights_q = proj_weights;
    block.proj_bias_q = proj_bias;
    block.proj_quant = proj_quant;

    const uint32_t spatial = spatial_h * spatial_w;
    const uint32_t expand_c = block.expanded_channels();
    const uint32_t residual =
        (stride == 1 && in_channels == out_channels) ? static_cast<uint32_t>(in_channels) : 0u;
    const uint32_t scratch_bytes =
        (2u * spatial * expand_c + spatial * residual) * static_cast<uint32_t>(sizeof(int8_t));
    block.scratch_i8 =
        static_cast<int8_t*>(arena.alloc(static_cast<std::size_t>(scratch_bytes), alignof(int8_t)));
    block.scratch_i8_bytes = block.scratch_i8 ? scratch_bytes : 0;
}

Tensor& CNNNetwork::forward_quantized(const Tensor& input)
{
    static Tensor empty{};

    if (!IsValid() || !HasActivationBuffers() || num_layers == 0 || !quant_runtime_)
        return empty;

    if (!CmsisQuantPlan::Forward(*quant_runtime_,
                                 *this,
                                 input,
                                 quant_output_format_,
                                 output_cache_))
    {
        return empty;
    }

    return output_cache_;
}
