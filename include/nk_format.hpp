#pragma once

#include <cstddef>
#include <cstdint>

// Binary netkit model container (.nk) — little-endian weights/biases (float32 or int8/int32).
// Spec: docs/NK_FILE_SPECIFICATION.md, docs/NK_FORMAT.md. Python writer: python/netkit/

namespace NkFormat
{
    constexpr char kMagic[4] = {'N', 'K', 'I', 'T'};
    constexpr char kTestMagic[4] = {'T', 'C', 'A', 'S'};
    constexpr char kQuantMagic[4] = {'Q', 'U', 'A', 'N'};
    constexpr uint32_t kVersion = 4;
    constexpr uint32_t kVersionMinSupported = 3;

    constexpr uint16_t kFlagHasTests = 0x0001;
    constexpr uint16_t kFlagHasQuant = 0x0002;
    // TCAS inputs are native int8 (Python-prequantized); not float[-128,127].
    constexpr uint16_t kFlagHasInt8Tests = 0x0004;
    constexpr uint32_t kMaxTestCases = 16;
    constexpr uint32_t kMaxCaseNameLen = 127;
    constexpr uint32_t kMaxCaseFloats = 16384;
    constexpr int32_t kNoLabel = -1;

    constexpr std::size_t kHeaderBytes = 48;
    constexpr std::size_t kTensorDescBytes = 24;
    constexpr std::size_t kLayerDenseBytes = 16;
    constexpr std::size_t kLayerConvBytes = 24;
    constexpr std::size_t kLayerPoolBytes = 16;
    constexpr std::size_t kLayerFlattenBytes = 4;

    constexpr uint32_t kMaxLayers = 100;
    constexpr uint32_t kMaxTensorCatalog = 128;
    constexpr uint32_t kMaxInputRank = 4;
    constexpr uint32_t kMaxTensorRank = 4;

    enum class NetworkKind : uint8_t
    {
        Mlp = 1,
        Cnn = 2
    };

    enum class LayerKind : uint8_t
    {
        Dense = 1,
        Conv2D = 2,
        MaxPool2D = 3,
        Flatten = 4,
        AvgPool2D = 5,
        BatchNorm2d = 6,
        DepthwiseConv2D = 7,
        ConvNeXtV2Block = 8,
        MobilenetV4Uib = 9,
        ResNetBasicBlock = 10,
        LayerNorm2d = 11,
        YoloxDecoupledHead = 12
    };

    enum class DType : uint8_t
    {
        Float32 = 1,
        Int8 = 2,
        Int32 = 3
    };

    struct MlpLayerQuantDesc
    {
        float input_scale = 1.0f;
        int32_t input_zero_point = 0;
        float weight_scale = 1.0f;
        int32_t weight_zero_point = 0;
        float bias_scale = 1.0f;
        int32_t bias_zero_point = 0;
        float output_scale = 1.0f;
        int32_t output_zero_point = 0;
        // Runtime-only (not in the 32-byte .nk QUAN descriptor). When non-null
        // with num_weight_channel_scales > 1, kernels use per-channel weight
        // scales (TFLite-style); otherwise weight_scale is used for all channels.
        const float* weight_channel_scales = nullptr;
        uint32_t num_weight_channel_scales = 0;
    };

    constexpr std::size_t kMlpLayerQuantBytes = 32;
    // QUAN section reserved u16 flags (after num_layers).
    constexpr uint16_t kQuanFlagPerChannelWeights = 0x0001;

    enum class Activation : uint8_t
    {
        None = 0,
        ReLU = 1,
        Sigmoid = 2,
        Tanh = 3,
        LeakyReLU = 4,
        ReLU6 = 5,
        Softmax = 6
    };

    struct FileHeader
    {
        uint32_t version = 0;
        NetworkKind network_kind = NetworkKind::Mlp;
        uint8_t input_rank = 0;
        uint16_t flags = 0;
        uint32_t input_shape[kMaxInputRank]{};
        uint32_t num_layers = 0;
        uint32_t num_weight_tensors = 0;
        uint32_t num_bias_tensors = 0;
        uint32_t weights_bytes = 0;
        uint32_t biases_bytes = 0;
    };

    struct TensorDesc
    {
        uint8_t rank = 0;
        DType dtype = DType::Float32;
        uint32_t dims[kMaxTensorRank]{};
        uint32_t num_elements = 0;
    };

    struct DenseLayerDesc
    {
        uint32_t units = 0;
        Activation activation = Activation::None;
        float alpha = 0.01f;
    };

    struct ConvLayerDesc
    {
        uint32_t kernel_size = 0; // conv2d: square kernel; depthwise: kernel_h
        uint32_t stride = 1;
        uint32_t filters = 0;
        Activation activation = Activation::None;
        uint8_t pad_h = 0;
        uint8_t pad_w = 0;
        uint8_t kernel_w = 0; // depthwise only; conv2d leaves zero
        float alpha = 0.01f;
    };

    inline uint32_t DepthwiseKernelH(const ConvLayerDesc& layer)
    {
        return layer.kernel_size;
    }

    inline uint32_t DepthwiseKernelW(const ConvLayerDesc& layer)
    {
        return layer.kernel_w;
    }

    struct BatchNormLayerDesc
    {
        uint32_t channels = 0;
        uint32_t reserved = 0;
    };

    struct PoolLayerDesc
    {
        uint32_t pool_size = 2;
        uint32_t stride = 2;
        uint8_t pad_h = 0;
        uint8_t pad_w = 0;
        uint16_t reserved = 0;
    };

    struct ConvNeXtV2BlockLayerDesc
    {
        uint32_t channels = 0;
        uint32_t reserved = 0;
        float eps = 1e-6f;
    };

    struct MobilenetV4UibLayerDesc
    {
        uint32_t in_channels = 0;
        uint32_t out_channels = 0;
        uint8_t start_dw_kernel = 0;
        uint8_t middle_dw_kernel = 0;
        uint8_t stride = 1;
        uint8_t middle_dw_downsample = 1;
        float expand_ratio = 1.0f;
        uint32_t reserved = 0;
    };

    struct ResNetBasicBlockLayerDesc
    {
        uint32_t in_channels = 0;
        uint32_t out_channels = 0;
        uint8_t stride = 1;
        uint8_t reserved[3]{};
    };

    struct LayerNormLayerDesc
    {
        uint32_t channels = 0;
        uint32_t reserved = 0;
        float eps = 1e-6f;
    };

    struct YoloxDecoupledHeadLayerDesc
    {
        uint32_t in_channels = 0;
        uint32_t hidden_dim = 256;
        uint32_t num_classes = 80;
        uint8_t num_convs = 2;
        uint8_t reserved[3]{};
    };

    struct LayerDesc
    {
        LayerKind kind = LayerKind::Dense;
        DenseLayerDesc dense{};
        ConvLayerDesc conv{};
        PoolLayerDesc pool{};
        BatchNormLayerDesc batch_norm{};
        ConvNeXtV2BlockLayerDesc convnextv2_block{};
        MobilenetV4UibLayerDesc mobilenetv4_uib{};
        ResNetBasicBlockLayerDesc resnet_basic_block{};
        LayerNormLayerDesc layernorm2d{};
        YoloxDecoupledHeadLayerDesc yolox_decoupled_head{};
    };

    const char* NetworkKindName(NetworkKind kind);
    const char* LayerKindName(LayerKind kind);
    const char* DTypeName(DType dtype);
    const char* ActivationName(Activation activation);

    inline std::size_t DTypeElementBytes(DType dtype)
    {
        switch (dtype)
        {
            case DType::Float32:
            case DType::Int32:
                return 4;
            case DType::Int8:
                return 1;
        }
        return 0;
    }

    inline bool IsSupportedVersion(uint32_t version)
    {
        return version >= kVersionMinSupported && version <= kVersion;
    }
}
