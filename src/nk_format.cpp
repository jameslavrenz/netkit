#include "nk_format.hpp"

#include <cstring>

namespace NkFormat
{
    const char* NetworkKindName(NetworkKind kind)
    {
        switch (kind)
        {
            case NetworkKind::Mlp: return "mlp";
            case NetworkKind::Cnn: return "cnn";
        }
        return "unknown";
    }

    const char* LayerKindName(LayerKind kind)
    {
        switch (kind)
        {
            case LayerKind::Dense: return "dense";
            case LayerKind::Conv2D: return "conv2d";
            case LayerKind::MaxPool2D: return "max_pool2d";
            case LayerKind::Flatten: return "flatten";
            case LayerKind::AvgPool2D: return "avg_pool2d";
            case LayerKind::BatchNorm2d: return "batch_norm2d";
            case LayerKind::DepthwiseConv2D: return "depthwise_conv2d";
            case LayerKind::ConvNeXtV2Block: return "convnextv2_block";
            case LayerKind::MobilenetV4Uib: return "mobilenetv4_uib";
            case LayerKind::ResNetBasicBlock: return "resnet_basic_block";
            case LayerKind::LayerNorm2d: return "layernorm2d";
        }
        return "unknown";
    }

    const char* DTypeName(DType dtype)
    {
        switch (dtype)
        {
            case DType::Float32: return "float32";
        }
        return "unknown";
    }

    const char* ActivationName(Activation activation)
    {
        switch (activation)
        {
            case Activation::None: return "none";
            case Activation::ReLU: return "relu";
            case Activation::Sigmoid: return "sigmoid";
            case Activation::Tanh: return "tanh";
            case Activation::LeakyReLU: return "leaky_relu";
            case Activation::ReLU6: return "relu6";
            case Activation::Softmax: return "softmax";
        }
        return "none";
    }

    Activation ParseActivationName(const char* name)
    {
        if (!name)
            return Activation::None;

        if (std::strcmp(name, "relu") == 0) return Activation::ReLU;
        if (std::strcmp(name, "sigmoid") == 0) return Activation::Sigmoid;
        if (std::strcmp(name, "tanh") == 0) return Activation::Tanh;
        if (std::strcmp(name, "leaky_relu") == 0) return Activation::LeakyReLU;
        if (std::strcmp(name, "relu6") == 0) return Activation::ReLU6;
        if (std::strcmp(name, "softmax") == 0) return Activation::Softmax;
        return Activation::None;
    }
}
