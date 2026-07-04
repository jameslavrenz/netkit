#pragma once
#include "tensor.hpp"
#include "arena.hpp"
#include "conv2d.hpp"
#include "convnextv2_block.hpp"
#include "depthwise_conv2d.hpp"
#include "mobilenetv4_uib.hpp"
#include "resnet_basic_block.hpp"
#include "mlp.hpp"
#include "ops_resolver.hpp"

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
    DepthwiseConv2D,
    MaxPool2D,
    AvgPool2D,
    BatchNorm2d,
    LayerNorm2d,
    Flatten,
    Dense,
    ConvNeXtV2Block,
    MobilenetV4Uib,
    ResNetBasicBlock
};

struct Conv2DLayer
{
    Conv2D conv;
    ConvActivationType activation;
    float leaky_alpha = 0.01f;

    void forward(const Tensor& input, Tensor& output);
};

struct DepthwiseConv2DLayer
{
    DepthwiseConv2D depthwise;
    ConvActivationType activation;
    float leaky_alpha = 0.01f;

    void forward(const Tensor& input, Tensor& output);
};

struct MaxPool2DLayer
{
    int pool_h = 2;
    int pool_w = 2;
    int stride = 2;
    int pad_h = 0;
    int pad_w = 0;
    int pad_h_end = 0;
    int pad_w_end = 0;

    void forward(const Tensor& input, Tensor& output);
};

struct AvgPool2DLayer
{
    int pool_h = 2;
    int pool_w = 2;
    int stride = 2;
    int pad_h = 0;
    int pad_w = 0;
    int pad_h_end = 0;
    int pad_w_end = 0;

    void forward(const Tensor& input, Tensor& output);
};

struct BatchNorm2DLayer
{
    int channels = 0;
    float* scale = nullptr;
    float* bias = nullptr;

    void forward(const Tensor& input, Tensor& output);
};

struct LayerNorm2DLayer
{
    int channels = 0;
    float eps = 1e-6f;
    float* weight = nullptr;
    float* bias = nullptr;

    void forward(const Tensor& input, Tensor& output);
};

struct ConvNeXtV2BlockLayer
{
    ConvNeXtV2Block block;

    void forward(const Tensor& input, Tensor& output);
};

struct MobilenetV4UibLayer
{
    MobileNetV4Uib block;

    void forward(const Tensor& input, Tensor& output);
};

struct ResNetBasicBlockLayer
{
    ResNetBasicBlock block;

    void forward(const Tensor& input, Tensor& output);
};

struct CnnBlock
{
    CnnBlockType type = CnnBlockType::Conv2D;
    Conv2DLayer conv;
    DepthwiseConv2DLayer depthwise_conv;
    MaxPool2DLayer pool;
    AvgPool2DLayer avg_pool;
    BatchNorm2DLayer batch_norm;
    LayerNorm2DLayer layer_norm;
    ConvNeXtV2BlockLayer convnextv2_block;
    MobilenetV4UibLayer mobilenetv4_uib;
    ResNetBasicBlockLayer resnet_basic_block;
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
    NkOpsResolver op_resolver_{};
    bool has_custom_resolver_ = false;

public:
    CNNNetwork(uint32_t num_layers, Arena& arena);

    void SetOpsResolver(const NkOpsResolver& resolver)
    {
        op_resolver_ = resolver;
        has_custom_resolver_ = true;
    }

    const NkOpsResolver& GetOpsResolver() const;

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
                       int pad_w = 0,
                       int pad_h_end = -1,
                       int pad_w_end = -1);

    void InitDepthwiseConvLayer(uint32_t layer_idx,
                                int kernel_h,
                                int kernel_w,
                                int stride,
                                int channels,
                                float* weights,
                                float* bias,
                                ConvActivationType activation,
                                float leaky_alpha = 0.01f,
                                int pad_h = 0,
                                int pad_w = 0,
                                int pad_h_end = -1,
                                int pad_w_end = -1);

    void InitPoolLayer(uint32_t layer_idx,
                       int pool_h,
                       int pool_w,
                       int stride,
                       int pad_h = 0,
                       int pad_w = 0,
                       int pad_h_end = -1,
                       int pad_w_end = -1);

    void InitAvgPoolLayer(uint32_t layer_idx,
                          int pool_h,
                          int pool_w,
                          int stride,
                          int pad_h = 0,
                          int pad_w = 0,
                          int pad_h_end = -1,
                          int pad_w_end = -1);

    void InitBatchNormLayer(uint32_t layer_idx, int channels, float* scale, float* bias);

    void InitLayerNormLayer(uint32_t layer_idx, int channels, float eps, float* weight, float* bias);

    void InitConvNeXtV2BlockLayer(uint32_t layer_idx,
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
                                        float* pw2_bias);

    void InitMobilenetV4UibLayer(uint32_t layer_idx,
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
                                 float* proj_bn_bias);

    void InitResNetBasicBlockLayer(uint32_t layer_idx,
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
                                   float* shortcut_bn_bias);

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
