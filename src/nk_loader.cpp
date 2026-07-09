#include "nk_loader.hpp"
#include "mobilenetv4_uib.hpp"
#include "resnet_basic_block.hpp"
#include "yolox_decoupled_head.hpp"
#include "tensor_factory.hpp"
#include "nk_op_detail.hpp"
#include "netkit_config.h"
#include "nk_mmap.hpp"

#include <cstdio>
#include <cstring>
#include <new>
#ifndef NETKIT_DISABLE_IOSTREAM
#include <iostream>
#endif

namespace NkLoader
{
    namespace
    {
        char g_error[256]{};

        void FreeWeightChannelScaleBlob(ParsedModel& parsed)
        {
            delete[] parsed.weight_channel_scale_blob;
            parsed.weight_channel_scale_blob = nullptr;
            parsed.weight_channel_scale_floats = 0;
            for (uint32_t i = 0; i < parsed.num_quant_layers; ++i)
            {
                parsed.layer_quant[i].weight_channel_scales = nullptr;
                parsed.layer_quant[i].num_weight_channel_scales = 0;
            }
        }

        bool RelocateWeightChannelScalesToArena(ParsedModel& parsed, Arena& arena)
        {
            if (!parsed.weight_channel_scale_blob || parsed.weight_channel_scale_floats == 0)
                return true;
            float* arena_blob = static_cast<float*>(arena.alloc(
                parsed.weight_channel_scale_floats * sizeof(float), alignof(float)));
            if (!arena_blob)
                return false;
            std::memcpy(arena_blob,
                        parsed.weight_channel_scale_blob,
                        parsed.weight_channel_scale_floats * sizeof(float));
            const float* old_base = parsed.weight_channel_scale_blob;
            for (uint32_t i = 0; i < parsed.num_quant_layers; ++i)
            {
                if (!parsed.layer_quant[i].weight_channel_scales)
                    continue;
                const std::size_t off = static_cast<std::size_t>(
                    parsed.layer_quant[i].weight_channel_scales - old_base);
                parsed.layer_quant[i].weight_channel_scales = arena_blob + off;
            }
            delete[] parsed.weight_channel_scale_blob;
            // Arena owns the relocated blob; clear heap ownership.
            parsed.weight_channel_scale_blob = nullptr;
            parsed.weight_channel_scale_floats = 0;
            return true;
        }

        void SetError(const char* message)
        {
            std::strncpy(g_error, message, sizeof(g_error) - 1);
            g_error[sizeof(g_error) - 1] = '\0';
        }

        LoadResult Fail(LoadStatus status, const char* message)
        {
            SetError(message);
            return LoadResult{status, NetworkKind::Unknown, g_error};
        }

        NetworkKind FromNkNetwork(NkFormat::NetworkKind kind)
        {
            switch (kind)
            {
                case NkFormat::NetworkKind::Mlp: return NetworkKind::Mlp;
                case NkFormat::NetworkKind::Cnn: return NetworkKind::Cnn;
            }
            return NetworkKind::Unknown;
        }

        uint32_t CatalogWeightTensorCount(const NkFormat::LayerDesc& layer)
        {
            switch (layer.kind)
            {
                case NkFormat::LayerKind::Conv2D:
                case NkFormat::LayerKind::DepthwiseConv2D:
                case NkFormat::LayerKind::Dense:
                case NkFormat::LayerKind::BatchNorm2d:
                case NkFormat::LayerKind::LayerNorm2d:
                    return 1;
                case NkFormat::LayerKind::ConvNeXtV2Block:
                    return 5;
                case NkFormat::LayerKind::MobilenetV4Uib:
                {
                    const auto& uib = layer.mobilenetv4_uib;
                    uint32_t count = 0;
                    if (uib.start_dw_kernel > 0)
                        count += 2;
                    count += 2;
                    if (uib.middle_dw_kernel > 0)
                        count += 2;
                    count += 2;
                    return count;
                }
                case NkFormat::LayerKind::ResNetBasicBlock:
                {
                    const auto& block = layer.resnet_basic_block;
                    uint32_t count = 4;
                    if (block.stride != 1 || block.in_channels != block.out_channels)
                        count += 2;
                    return count;
                }
                case NkFormat::LayerKind::YoloxDecoupledHead:
                    return 3u + 2u * static_cast<uint32_t>(layer.yolox_decoupled_head.num_convs);
                default:
                    return 0;
            }
        }

        uint32_t ComputeOutputElements(const ParsedModel& model)
        {
            const NkFormat::FileHeader& header = model.header;
            if (header.network_kind == NkFormat::NetworkKind::Mlp)
            {
                if (header.num_layers == 0)
                    return 0;
                return header.input_shape[0] * model.layers[header.num_layers - 1].dense.units;
            }

            uint32_t h = header.input_shape[0];
            uint32_t w = header.input_shape[1];
            uint32_t c = header.input_shape[2];
            uint32_t features = h * w * c;
            bool flattened = false;
            uint32_t weight_index = 0;

            for (uint32_t i = 0; i < header.num_layers; ++i)
            {
                switch (model.layers[i].kind)
                {
                    case NkFormat::LayerKind::Conv2D:
                    {
                        const NkFormat::ConvLayerDesc& layer = model.layers[i].conv;
                        int pad_h_end = static_cast<int>(layer.pad_h);
                        int pad_w_end = static_cast<int>(layer.pad_w);
                        nk_op_detail::DecodeConvPadExtra(
                            layer.pad_h, layer.pad_w, layer.kernel_w, pad_h_end, pad_w_end);
                        h = nk_op_detail::CalcOutputDimAsymmetric(
                            h, static_cast<int>(layer.kernel_size), static_cast<int>(layer.stride),
                            static_cast<int>(layer.pad_h), pad_h_end);
                        w = nk_op_detail::CalcOutputDimAsymmetric(
                            w, static_cast<int>(layer.kernel_size), static_cast<int>(layer.stride),
                            static_cast<int>(layer.pad_w), pad_w_end);
                        c = layer.filters;
                        features = h * w * c;
                        break;
                    }
                    case NkFormat::LayerKind::DepthwiseConv2D:
                    {
                        const NkFormat::ConvLayerDesc& layer = model.layers[i].conv;
                        std::size_t weight_elems = 0;
                        if (weight_index < header.num_weight_tensors)
                            weight_elems = model.weight_tensors[weight_index].num_elements;
                        const nk_op_detail::DepthwiseMeta dw_meta = nk_op_detail::DecodeDepthwiseMeta(
                            layer, weight_elems, static_cast<std::size_t>(layer.filters));
                        h = nk_op_detail::CalcOutputDimAsymmetric(
                            h, static_cast<int>(dw_meta.kernel_h), static_cast<int>(layer.stride),
                            static_cast<int>(layer.pad_h), dw_meta.pad_h_end);
                        w = nk_op_detail::CalcOutputDimAsymmetric(
                            w, static_cast<int>(dw_meta.kernel_w), static_cast<int>(layer.stride),
                            static_cast<int>(layer.pad_w), dw_meta.pad_w_end);
                        features = h * w * c;
                        break;
                    }
                    case NkFormat::LayerKind::MaxPool2D:
                    case NkFormat::LayerKind::AvgPool2D:
                    {
                        const NkFormat::PoolLayerDesc& layer = model.layers[i].pool;
                        const nk_op_detail::PoolMeta pool_meta = nk_op_detail::DecodePoolMeta(layer);
                        h = nk_op_detail::CalcOutputDimAsymmetric(
                            h, pool_meta.pool_h, static_cast<int>(layer.stride), pool_meta.pad_h,
                            pool_meta.pad_h_end);
                        w = nk_op_detail::CalcOutputDimAsymmetric(
                            w, pool_meta.pool_w, static_cast<int>(layer.stride), pool_meta.pad_w,
                            pool_meta.pad_w_end);
                        features = h * w * c;
                        break;
                    }
                    case NkFormat::LayerKind::BatchNorm2d:
                        features = h * w * c;
                        break;
                    case NkFormat::LayerKind::LayerNorm2d:
                        features = h * w * c;
                        break;
                    case NkFormat::LayerKind::ConvNeXtV2Block:
                        features = h * w * c;
                        break;
                    case NkFormat::LayerKind::MobilenetV4Uib:
                    {
                        const NkFormat::MobilenetV4UibLayerDesc& layer = model.layers[i].mobilenetv4_uib;
                        MobileNetV4Uib shape_probe{};
                        shape_probe.in_channels = static_cast<int>(layer.in_channels);
                        shape_probe.out_channels = static_cast<int>(layer.out_channels);
                        shape_probe.start_dw_kernel = static_cast<int>(layer.start_dw_kernel);
                        shape_probe.middle_dw_kernel = static_cast<int>(layer.middle_dw_kernel);
                        shape_probe.stride = static_cast<int>(layer.stride);
                        shape_probe.middle_dw_downsample = layer.middle_dw_downsample != 0;
                        shape_probe.expand_ratio = layer.expand_ratio;
                        shape_probe.output_spatial(h, w, h, w);
                        c = layer.out_channels;
                        features = h * w * c;
                        break;
                    }
                    case NkFormat::LayerKind::ResNetBasicBlock:
                    {
                        const NkFormat::ResNetBasicBlockLayerDesc& layer =
                            model.layers[i].resnet_basic_block;
                        ResNetBasicBlock shape_probe{};
                        shape_probe.in_channels = static_cast<int>(layer.in_channels);
                        shape_probe.out_channels = static_cast<int>(layer.out_channels);
                        shape_probe.stride = static_cast<int>(layer.stride);
                        shape_probe.output_spatial(h, w, h, w);
                        c = layer.out_channels;
                        features = h * w * c;
                        break;
                    }
                    case NkFormat::LayerKind::YoloxDecoupledHead:
                    {
                        const NkFormat::YoloxDecoupledHeadLayerDesc& head =
                            model.layers[i].yolox_decoupled_head;
                        c = 4u + 1u + head.num_classes;
                        features = h * w * c;
                        break;
                    }
                    case NkFormat::LayerKind::Flatten:
                        flattened = true;
                        features = h * w * c;
                        break;
                    case NkFormat::LayerKind::Dense:
                        features = model.layers[i].dense.units;
                        break;
                    default:
                        break;
                }
                weight_index += CatalogWeightTensorCount(model.layers[i]);
            }

            if (flattened || header.num_layers == 0)
                return features;

            return h * w * c;
        }

        void ModelNameFromPath(const char* path, char* name, std::size_t capacity)
        {
            if (!path || !*path)
            {
                std::strncpy(name, "model", capacity);
                name[capacity - 1] = '\0';
                return;
            }

            const char* base = std::strrchr(path, '/');
            if (!base)
                base = std::strrchr(path, '\\');
            base = base ? base + 1 : path;

            std::strncpy(name, base, capacity - 1);
            name[capacity - 1] = '\0';

            char* dot = std::strrchr(name, '.');
            if (dot)
                *dot = '\0';
        }

        bool ReadExact(std::FILE* file, void* buffer, std::size_t bytes)
        {
            if (bytes == 0)
                return true;
            const std::size_t got = std::fread(buffer, 1, bytes, file);
            return got == bytes && !std::ferror(file);
        }

        bool ReadU8(std::FILE* file, uint8_t& value)
        {
            return ReadExact(file, &value, 1);
        }

        bool ReadU16(std::FILE* file, uint16_t& value)
        {
            return ReadExact(file, &value, sizeof(value));
        }

        bool ReadU32(std::FILE* file, uint32_t& value)
        {
            return ReadExact(file, &value, sizeof(value));
        }

        bool ReadF32(std::FILE* file, float& value)
        {
            return ReadExact(file, &value, sizeof(value));
        }

        bool ReadHeader(std::FILE* file, NkFormat::FileHeader& header)
        {
            char magic[4] = {};
            if (!ReadExact(file, magic, 4))
                return false;

            if (std::memcmp(magic, NkFormat::kMagic, 4) != 0)
            {
                SetError("Invalid .nk magic (expected NKIT)");
                return false;
            }

            if (!ReadU32(file, header.version))
                return false;

            uint8_t network_kind = 0;
            if (!ReadU8(file, network_kind) || !ReadU8(file, header.input_rank) || !ReadU16(file, header.flags))
                return false;

            header.network_kind = static_cast<NkFormat::NetworkKind>(network_kind);
            for (uint32_t i = 0; i < NkFormat::kMaxInputRank; ++i)
            {
                if (!ReadU32(file, header.input_shape[i]))
                    return false;
            }

            if (!ReadU32(file, header.num_layers) || !ReadU32(file, header.num_weight_tensors) ||
                !ReadU32(file, header.num_bias_tensors) || !ReadU32(file, header.weights_bytes) ||
                !ReadU32(file, header.biases_bytes))
                return false;

            return true;
        }

        bool ReadTensorDesc(std::FILE* file, NkFormat::TensorDesc& desc)
        {
            uint8_t dtype = 0;
            uint16_t pad = 0;
            if (!ReadU8(file, desc.rank) || !ReadU8(file, dtype) || !ReadU16(file, pad))
                return false;

            desc.dtype = static_cast<NkFormat::DType>(dtype);
            for (uint32_t i = 0; i < NkFormat::kMaxTensorRank; ++i)
            {
                if (!ReadU32(file, desc.dims[i]))
                    return false;
            }

            return ReadU32(file, desc.num_elements);
        }

        bool ReadDenseLayer(std::FILE* file, NkFormat::LayerDesc& layer)
        {
            layer.kind = NkFormat::LayerKind::Dense;
            uint8_t pad[3] = {};
            uint8_t activation = 0;
            if (!ReadU32(file, layer.dense.units) || !ReadU8(file, activation) ||
                !ReadExact(file, pad, sizeof(pad)) || !ReadF32(file, layer.dense.alpha))
                return false;

            layer.dense.activation = static_cast<NkFormat::Activation>(activation);
            return true;
        }

        bool ReadConvLayer(std::FILE* file, NkFormat::LayerDesc& layer)
        {
            layer.kind = NkFormat::LayerKind::Conv2D;
            uint8_t activation = 0;
            if (!ReadU32(file, layer.conv.kernel_size) || !ReadU32(file, layer.conv.stride) ||
                !ReadU32(file, layer.conv.filters) || !ReadU8(file, activation) ||
                !ReadU8(file, layer.conv.pad_h) || !ReadU8(file, layer.conv.pad_w) ||
                !ReadU8(file, layer.conv.kernel_w) || !ReadF32(file, layer.conv.alpha))
                return false;

            layer.conv.activation = static_cast<NkFormat::Activation>(activation);
            return true;
        }

        bool ReadDepthwiseConvLayer(std::FILE* file, NkFormat::LayerDesc& layer)
        {
            layer.kind = NkFormat::LayerKind::DepthwiseConv2D;
            uint8_t activation = 0;
            if (!ReadU32(file, layer.conv.kernel_size) || !ReadU32(file, layer.conv.stride) ||
                !ReadU32(file, layer.conv.filters) || !ReadU8(file, activation) ||
                !ReadU8(file, layer.conv.pad_h) || !ReadU8(file, layer.conv.pad_w) ||
                !ReadU8(file, layer.conv.kernel_w) || !ReadF32(file, layer.conv.alpha))
                return false;

            layer.conv.activation = static_cast<NkFormat::Activation>(activation);
            return true;
        }

        bool ReadPoolLayer(std::FILE* file, NkFormat::LayerDesc& layer)
        {
            layer.kind = NkFormat::LayerKind::MaxPool2D;
            return ReadU32(file, layer.pool.pool_size) && ReadU32(file, layer.pool.stride) &&
                   ReadU8(file, layer.pool.pad_h) && ReadU8(file, layer.pool.pad_w) &&
                   ReadU16(file, layer.pool.reserved);
        }

        bool ReadAvgPoolLayer(std::FILE* file, NkFormat::LayerDesc& layer)
        {
            layer.kind = NkFormat::LayerKind::AvgPool2D;
            return ReadU32(file, layer.pool.pool_size) && ReadU32(file, layer.pool.stride) &&
                   ReadU8(file, layer.pool.pad_h) && ReadU8(file, layer.pool.pad_w) &&
                   ReadU16(file, layer.pool.reserved);
        }

        bool ReadBatchNormLayer(std::FILE* file, NkFormat::LayerDesc& layer)
        {
            layer.kind = NkFormat::LayerKind::BatchNorm2d;
            return ReadU32(file, layer.batch_norm.channels) && ReadU32(file, layer.batch_norm.reserved);
        }

        bool ReadConvNeXtV2BlockLayer(std::FILE* file, NkFormat::LayerDesc& layer)
        {
            layer.kind = NkFormat::LayerKind::ConvNeXtV2Block;
            return ReadU32(file, layer.convnextv2_block.channels) &&
                   ReadU32(file, layer.convnextv2_block.reserved) &&
                   ReadF32(file, layer.convnextv2_block.eps);
        }

        bool ReadMobilenetV4UibLayer(std::FILE* file, NkFormat::LayerDesc& layer)
        {
            layer.kind = NkFormat::LayerKind::MobilenetV4Uib;
            return ReadU32(file, layer.mobilenetv4_uib.in_channels) &&
                   ReadU32(file, layer.mobilenetv4_uib.out_channels) &&
                   ReadU8(file, layer.mobilenetv4_uib.start_dw_kernel) &&
                   ReadU8(file, layer.mobilenetv4_uib.middle_dw_kernel) &&
                   ReadU8(file, layer.mobilenetv4_uib.stride) &&
                   ReadU8(file, layer.mobilenetv4_uib.middle_dw_downsample) &&
                   ReadF32(file, layer.mobilenetv4_uib.expand_ratio) &&
                   ReadU32(file, layer.mobilenetv4_uib.reserved);
        }

        bool ReadYoloxDecoupledHeadLayer(std::FILE* file, NkFormat::LayerDesc& layer)
        {
            layer.kind = NkFormat::LayerKind::YoloxDecoupledHead;
            return ReadU32(file, layer.yolox_decoupled_head.in_channels) &&
                   ReadU32(file, layer.yolox_decoupled_head.hidden_dim) &&
                   ReadU32(file, layer.yolox_decoupled_head.num_classes) &&
                   ReadU8(file, layer.yolox_decoupled_head.num_convs) &&
                   ReadExact(file, layer.yolox_decoupled_head.reserved,
                             sizeof(layer.yolox_decoupled_head.reserved));
        }

        bool ReadResNetBasicBlockLayer(std::FILE* file, NkFormat::LayerDesc& layer)
        {
            layer.kind = NkFormat::LayerKind::ResNetBasicBlock;
            return ReadU32(file, layer.resnet_basic_block.in_channels) &&
                   ReadU32(file, layer.resnet_basic_block.out_channels) &&
                   ReadU8(file, layer.resnet_basic_block.stride) &&
                   ReadExact(file, layer.resnet_basic_block.reserved, sizeof(layer.resnet_basic_block.reserved));
        }

        bool ReadLayerNormLayer(std::FILE* file, NkFormat::LayerDesc& layer)
        {
            layer.kind = NkFormat::LayerKind::LayerNorm2d;
            return ReadU32(file, layer.layernorm2d.channels) &&
                   ReadU32(file, layer.layernorm2d.reserved) &&
                   ReadF32(file, layer.layernorm2d.eps);
        }

        bool ReadFlattenLayer(std::FILE* file, NkFormat::LayerDesc& layer)
        {
            layer.kind = NkFormat::LayerKind::Flatten;
            (void)file;
            return true;
        }

        bool ReadLayer(std::FILE* file, NkFormat::LayerDesc& layer)
        {
            uint8_t kind = 0;
            uint8_t pad[3] = {};
            if (!ReadU8(file, kind) || !ReadExact(file, pad, sizeof(pad)))
                return false;

            switch (static_cast<NkFormat::LayerKind>(kind))
            {
                case NkFormat::LayerKind::Dense:
                    return ReadDenseLayer(file, layer);
                case NkFormat::LayerKind::Conv2D:
                    return ReadConvLayer(file, layer);
                case NkFormat::LayerKind::DepthwiseConv2D:
                    return ReadDepthwiseConvLayer(file, layer);
                case NkFormat::LayerKind::MaxPool2D:
                    return ReadPoolLayer(file, layer);
                case NkFormat::LayerKind::AvgPool2D:
                    return ReadAvgPoolLayer(file, layer);
                case NkFormat::LayerKind::BatchNorm2d:
                    return ReadBatchNormLayer(file, layer);
                case NkFormat::LayerKind::ConvNeXtV2Block:
                    return ReadConvNeXtV2BlockLayer(file, layer);
                case NkFormat::LayerKind::MobilenetV4Uib:
                    return ReadMobilenetV4UibLayer(file, layer);
                case NkFormat::LayerKind::YoloxDecoupledHead:
                    return ReadYoloxDecoupledHeadLayer(file, layer);
                case NkFormat::LayerKind::ResNetBasicBlock:
                    return ReadResNetBasicBlockLayer(file, layer);
                case NkFormat::LayerKind::LayerNorm2d:
                    return ReadLayerNormLayer(file, layer);
                case NkFormat::LayerKind::Flatten:
                    return ReadFlattenLayer(file, layer);
                default:
                    SetError("Unsupported .nk layer kind");
                    return false;
            }
        }

        struct ByteCursor
        {
            const uint8_t* data = nullptr;
            std::size_t size = 0;
            std::size_t pos = 0;
        };

        bool AdvancePayloadAlignmentPadding(ByteCursor& cursor)
        {
            const std::size_t pos_after_catalog = cursor.pos;
            const std::size_t misalign = pos_after_catalog % alignof(float);
            if (misalign == 0)
                return true;

            const std::size_t pad = alignof(float) - misalign;
            if (cursor.pos + pad > cursor.size)
                return false;

            for (std::size_t i = 0; i < pad; ++i)
            {
                if (cursor.data[cursor.pos + i] != 0)
                    return true;
            }

            cursor.pos += pad;
            return true;
        }

        bool AdvancePayloadAlignmentPadding(std::FILE* file, std::size_t& payload_offset)
        {
            const long pos = std::ftell(file);
            if (pos < 0)
                return false;

            const std::size_t misalign = static_cast<std::size_t>(pos) % alignof(float);
            if (misalign == 0)
            {
                payload_offset = static_cast<std::size_t>(pos);
                return true;
            }

            const std::size_t pad = alignof(float) - misalign;
            for (std::size_t i = 0; i < pad; ++i)
            {
                uint8_t byte = 0;
                if (!ReadU8(file, byte))
                    return false;
                if (byte != 0)
                {
                    if (std::fseek(file, pos, SEEK_SET) != 0)
                        return false;
                    payload_offset = static_cast<std::size_t>(pos);
                    return true;
                }
            }

            const long after = std::ftell(file);
            if (after < 0)
                return false;
            payload_offset = static_cast<std::size_t>(after);
            return true;
        }

        bool CursorAdvance(ByteCursor& cursor, std::size_t bytes)
        {
            if (bytes == 0)
                return true;
            if (cursor.pos + bytes > cursor.size)
                return false;
            cursor.pos += bytes;
            return true;
        }

        bool CursorReadExact(ByteCursor& cursor, void* buffer, std::size_t bytes)
        {
            if (bytes == 0)
                return true;
            if (cursor.pos + bytes > cursor.size)
                return false;
            std::memcpy(buffer, cursor.data + cursor.pos, bytes);
            cursor.pos += bytes;
            return true;
        }

        bool CursorReadU8(ByteCursor& cursor, uint8_t& value)
        {
            return CursorReadExact(cursor, &value, 1);
        }

        bool CursorReadU16(ByteCursor& cursor, uint16_t& value)
        {
            return CursorReadExact(cursor, &value, sizeof(value));
        }

        bool CursorReadU32(ByteCursor& cursor, uint32_t& value)
        {
            return CursorReadExact(cursor, &value, sizeof(value));
        }

        bool CursorReadF32(ByteCursor& cursor, float& value)
        {
            return CursorReadExact(cursor, &value, sizeof(value));
        }

        bool CursorReadI32(ByteCursor& cursor, int32_t& value)
        {
            return CursorReadExact(cursor, &value, sizeof(value));
        }

        bool ReadHeaderCursor(ByteCursor& cursor, NkFormat::FileHeader& header)
        {
            char magic[4] = {};
            if (!CursorReadExact(cursor, magic, 4))
                return false;

            if (std::memcmp(magic, NkFormat::kMagic, 4) != 0)
            {
                SetError("Invalid .nk magic (expected NKIT)");
                return false;
            }

            if (!CursorReadU32(cursor, header.version))
                return false;

            uint8_t network_kind = 0;
            if (!CursorReadU8(cursor, network_kind) || !CursorReadU8(cursor, header.input_rank) ||
                !CursorReadU16(cursor, header.flags))
                return false;

            header.network_kind = static_cast<NkFormat::NetworkKind>(network_kind);
            for (uint32_t i = 0; i < NkFormat::kMaxInputRank; ++i)
            {
                if (!CursorReadU32(cursor, header.input_shape[i]))
                    return false;
            }

            return CursorReadU32(cursor, header.num_layers) && CursorReadU32(cursor, header.num_weight_tensors) &&
                   CursorReadU32(cursor, header.num_bias_tensors) && CursorReadU32(cursor, header.weights_bytes) &&
                   CursorReadU32(cursor, header.biases_bytes);
        }

        bool ReadTensorDescCursor(ByteCursor& cursor, NkFormat::TensorDesc& desc)
        {
            uint8_t dtype = 0;
            uint16_t pad = 0;
            if (!CursorReadU8(cursor, desc.rank) || !CursorReadU8(cursor, dtype) || !CursorReadU16(cursor, pad))
                return false;

            desc.dtype = static_cast<NkFormat::DType>(dtype);
            for (uint32_t i = 0; i < NkFormat::kMaxTensorRank; ++i)
            {
                if (!CursorReadU32(cursor, desc.dims[i]))
                    return false;
            }

            return CursorReadU32(cursor, desc.num_elements);
        }

        bool ReadMlpLayerQuantCursor(ByteCursor& cursor, NkFormat::MlpLayerQuantDesc& quant)
        {
            if (!CursorReadF32(cursor, quant.input_scale) || !CursorReadI32(cursor, quant.input_zero_point) ||
                !CursorReadF32(cursor, quant.weight_scale) || !CursorReadI32(cursor, quant.weight_zero_point) ||
                !CursorReadF32(cursor, quant.bias_scale) || !CursorReadI32(cursor, quant.bias_zero_point) ||
                !CursorReadF32(cursor, quant.output_scale) || !CursorReadI32(cursor, quant.output_zero_point))
                return false;
            return true;
        }

        bool ReadQuantBlockCursor(ByteCursor& cursor, ParsedModel& out)
        {
            out.has_quant = false;
            out.num_quant_layers = 0;
            out.weight_channel_scale_blob = nullptr;
            out.weight_channel_scale_floats = 0;

            if ((out.header.flags & NkFormat::kFlagHasQuant) == 0)
                return true;

            if (out.header.version < 4)
                return false;

            char magic[4] = {};
            if (!CursorReadExact(cursor, magic, sizeof(magic)))
                return false;
            if (std::memcmp(magic, NkFormat::kQuantMagic, sizeof(magic)) != 0)
                return false;

            uint16_t num_layers = 0;
            uint16_t quan_flags = 0;
            if (!CursorReadU16(cursor, num_layers) || !CursorReadU16(cursor, quan_flags))
                return false;
            if (num_layers == 0 || num_layers > NkFormat::kMaxLayers)
                return false;

            out.has_quant = true;
            out.num_quant_layers = num_layers;
            for (uint32_t i = 0; i < num_layers; ++i)
            {
                if (!ReadMlpLayerQuantCursor(cursor, out.layer_quant[i]))
                    return false;
                out.layer_quant[i].weight_channel_scales = nullptr;
                out.layer_quant[i].num_weight_channel_scales = 0;
            }

            if ((quan_flags & NkFormat::kQuanFlagPerChannelWeights) != 0)
            {
                // Layout matches pack_quant_section: per layer u32 n_ch + float32[n_ch].
                uint32_t counts[NkFormat::kMaxLayers]{};
                std::size_t total = 0;
                for (uint32_t i = 0; i < num_layers; ++i)
                {
                    uint32_t n_ch = 0;
                    if (!CursorReadU32(cursor, n_ch))
                        return false;
                    counts[i] = n_ch;
                    total += n_ch;
                    if (!CursorAdvance(cursor, static_cast<std::size_t>(n_ch) * sizeof(float)))
                        return false;
                }
                // Rewind to the start of the per-channel blob and load into one heap array.
                std::size_t rewind = 0;
                for (uint32_t i = 0; i < num_layers; ++i)
                    rewind += sizeof(uint32_t) + static_cast<std::size_t>(counts[i]) * sizeof(float);
                if (cursor.pos < rewind)
                    return false;
                cursor.pos -= rewind;
                if (total > 0)
                {
                    // Owned for ParsedModel lifetime; Instantiate copies into Arena.
                    float* blob = new (std::nothrow) float[total];
                    if (!blob)
                        return false;
                    std::size_t offset = 0;
                    for (uint32_t i = 0; i < num_layers; ++i)
                    {
                        uint32_t n_ch = 0;
                        if (!CursorReadU32(cursor, n_ch) || n_ch != counts[i])
                        {
                            delete[] blob;
                            return false;
                        }
                        if (n_ch == 0)
                            continue;
                        if (!CursorReadExact(cursor, blob + offset, n_ch * sizeof(float)))
                        {
                            delete[] blob;
                            return false;
                        }
                        out.layer_quant[i].weight_channel_scales = blob + offset;
                        out.layer_quant[i].num_weight_channel_scales = n_ch;
                        offset += n_ch;
                    }
                    out.weight_channel_scale_blob = blob;
                    out.weight_channel_scale_floats = total;
                }
            }
            return true;
        }

        bool ReadMlpLayerQuantFile(std::FILE* file, NkFormat::MlpLayerQuantDesc& quant)
        {
            if (!ReadF32(file, quant.input_scale) || !ReadExact(file, &quant.input_zero_point, sizeof(quant.input_zero_point)) ||
                !ReadF32(file, quant.weight_scale) || !ReadExact(file, &quant.weight_zero_point, sizeof(quant.weight_zero_point)) ||
                !ReadF32(file, quant.bias_scale) || !ReadExact(file, &quant.bias_zero_point, sizeof(quant.bias_zero_point)) ||
                !ReadF32(file, quant.output_scale) || !ReadExact(file, &quant.output_zero_point, sizeof(quant.output_zero_point)))
                return false;
            return true;
        }

        bool ReadQuantBlockFile(std::FILE* file, ParsedModel& out)
        {
            out.has_quant = false;
            out.num_quant_layers = 0;
            out.weight_channel_scale_blob = nullptr;
            out.weight_channel_scale_floats = 0;

            if ((out.header.flags & NkFormat::kFlagHasQuant) == 0)
                return true;

            if (out.header.version < 4)
                return false;

            char magic[4] = {};
            if (!ReadExact(file, magic, sizeof(magic)))
                return false;
            if (std::memcmp(magic, NkFormat::kQuantMagic, sizeof(magic)) != 0)
                return false;

            uint16_t num_layers = 0;
            uint16_t quan_flags = 0;
            if (!ReadU16(file, num_layers) || !ReadU16(file, quan_flags))
                return false;
            if (num_layers == 0 || num_layers > NkFormat::kMaxLayers)
                return false;

            out.has_quant = true;
            out.num_quant_layers = num_layers;
            for (uint32_t i = 0; i < num_layers; ++i)
            {
                if (!ReadMlpLayerQuantFile(file, out.layer_quant[i]))
                    return false;
                out.layer_quant[i].weight_channel_scales = nullptr;
                out.layer_quant[i].num_weight_channel_scales = 0;
            }

            if ((quan_flags & NkFormat::kQuanFlagPerChannelWeights) != 0)
            {
                // Layout matches pack_quant_section: per layer u32 n_ch + float32[n_ch].
                uint32_t counts[NkFormat::kMaxLayers]{};
                std::size_t total = 0;
                const long save_pos = std::ftell(file);
                if (save_pos < 0)
                    return false;
                for (uint32_t i = 0; i < num_layers; ++i)
                {
                    uint32_t n_ch = 0;
                    if (!ReadU32(file, n_ch))
                        return false;
                    counts[i] = n_ch;
                    total += n_ch;
                    if (n_ch > 0 &&
                        std::fseek(file, static_cast<long>(n_ch * sizeof(float)), SEEK_CUR) != 0)
                        return false;
                }
                if (std::fseek(file, save_pos, SEEK_SET) != 0)
                    return false;
                if (total > 0)
                {
                    float* blob = new (std::nothrow) float[total];
                    if (!blob)
                        return false;
                    std::size_t offset = 0;
                    for (uint32_t i = 0; i < num_layers; ++i)
                    {
                        uint32_t n_ch = 0;
                        if (!ReadU32(file, n_ch) || n_ch != counts[i])
                        {
                            delete[] blob;
                            return false;
                        }
                        if (n_ch == 0)
                            continue;
                        if (!ReadExact(file, blob + offset, n_ch * sizeof(float)))
                        {
                            delete[] blob;
                            return false;
                        }
                        out.layer_quant[i].weight_channel_scales = blob + offset;
                        out.layer_quant[i].num_weight_channel_scales = n_ch;
                        offset += n_ch;
                    }
                    out.weight_channel_scale_blob = blob;
                    out.weight_channel_scale_floats = total;
                }
            }
            return true;
        }

        bool ModelIsQuantized(const ParsedModel& parsed)
        {
            return parsed.has_quant && (parsed.header.flags & NkFormat::kFlagHasQuant) != 0;
        }

        bool ReadDenseLayerCursor(ByteCursor& cursor, NkFormat::LayerDesc& layer)
        {
            layer.kind = NkFormat::LayerKind::Dense;
            uint8_t pad[3] = {};
            uint8_t activation = 0;
            return CursorReadU32(cursor, layer.dense.units) && CursorReadU8(cursor, activation) &&
                   CursorReadExact(cursor, pad, sizeof(pad)) && CursorReadF32(cursor, layer.dense.alpha) &&
                   (layer.dense.activation = static_cast<NkFormat::Activation>(activation), true);
        }

        bool ReadConvLayerCursor(ByteCursor& cursor, NkFormat::LayerDesc& layer)
        {
            layer.kind = NkFormat::LayerKind::Conv2D;
            uint8_t activation = 0;
            return CursorReadU32(cursor, layer.conv.kernel_size) && CursorReadU32(cursor, layer.conv.stride) &&
                   CursorReadU32(cursor, layer.conv.filters) && CursorReadU8(cursor, activation) &&
                   CursorReadU8(cursor, layer.conv.pad_h) && CursorReadU8(cursor, layer.conv.pad_w) &&
                   CursorReadU8(cursor, layer.conv.kernel_w) && CursorReadF32(cursor, layer.conv.alpha) &&
                   (layer.conv.activation = static_cast<NkFormat::Activation>(activation), true);
        }

        bool ReadDepthwiseConvLayerCursor(ByteCursor& cursor, NkFormat::LayerDesc& layer)
        {
            layer.kind = NkFormat::LayerKind::DepthwiseConv2D;
            uint8_t activation = 0;
            return CursorReadU32(cursor, layer.conv.kernel_size) && CursorReadU32(cursor, layer.conv.stride) &&
                   CursorReadU32(cursor, layer.conv.filters) && CursorReadU8(cursor, activation) &&
                   CursorReadU8(cursor, layer.conv.pad_h) && CursorReadU8(cursor, layer.conv.pad_w) &&
                   CursorReadU8(cursor, layer.conv.kernel_w) && CursorReadF32(cursor, layer.conv.alpha) &&
                   (layer.conv.activation = static_cast<NkFormat::Activation>(activation), true);
        }

        bool ReadPoolLayerCursor(ByteCursor& cursor, NkFormat::LayerDesc& layer)
        {
            layer.kind = NkFormat::LayerKind::MaxPool2D;
            return CursorReadU32(cursor, layer.pool.pool_size) && CursorReadU32(cursor, layer.pool.stride) &&
                   CursorReadU8(cursor, layer.pool.pad_h) && CursorReadU8(cursor, layer.pool.pad_w) &&
                   CursorReadU16(cursor, layer.pool.reserved);
        }

        bool ReadAvgPoolLayerCursor(ByteCursor& cursor, NkFormat::LayerDesc& layer)
        {
            layer.kind = NkFormat::LayerKind::AvgPool2D;
            return CursorReadU32(cursor, layer.pool.pool_size) && CursorReadU32(cursor, layer.pool.stride) &&
                   CursorReadU8(cursor, layer.pool.pad_h) && CursorReadU8(cursor, layer.pool.pad_w) &&
                   CursorReadU16(cursor, layer.pool.reserved);
        }

        bool ReadBatchNormLayerCursor(ByteCursor& cursor, NkFormat::LayerDesc& layer)
        {
            layer.kind = NkFormat::LayerKind::BatchNorm2d;
            return CursorReadU32(cursor, layer.batch_norm.channels) &&
                   CursorReadU32(cursor, layer.batch_norm.reserved);
        }

        bool ReadConvNeXtV2BlockLayerCursor(ByteCursor& cursor, NkFormat::LayerDesc& layer)
        {
            layer.kind = NkFormat::LayerKind::ConvNeXtV2Block;
            return CursorReadU32(cursor, layer.convnextv2_block.channels) &&
                   CursorReadU32(cursor, layer.convnextv2_block.reserved) &&
                   CursorReadF32(cursor, layer.convnextv2_block.eps);
        }

        bool ReadMobilenetV4UibLayerCursor(ByteCursor& cursor, NkFormat::LayerDesc& layer)
        {
            layer.kind = NkFormat::LayerKind::MobilenetV4Uib;
            return CursorReadU32(cursor, layer.mobilenetv4_uib.in_channels) &&
                   CursorReadU32(cursor, layer.mobilenetv4_uib.out_channels) &&
                   CursorReadU8(cursor, layer.mobilenetv4_uib.start_dw_kernel) &&
                   CursorReadU8(cursor, layer.mobilenetv4_uib.middle_dw_kernel) &&
                   CursorReadU8(cursor, layer.mobilenetv4_uib.stride) &&
                   CursorReadU8(cursor, layer.mobilenetv4_uib.middle_dw_downsample) &&
                   CursorReadF32(cursor, layer.mobilenetv4_uib.expand_ratio) &&
                   CursorReadU32(cursor, layer.mobilenetv4_uib.reserved);
        }

        bool ReadYoloxDecoupledHeadLayerCursor(ByteCursor& cursor, NkFormat::LayerDesc& layer)
        {
            layer.kind = NkFormat::LayerKind::YoloxDecoupledHead;
            return CursorReadU32(cursor, layer.yolox_decoupled_head.in_channels) &&
                   CursorReadU32(cursor, layer.yolox_decoupled_head.hidden_dim) &&
                   CursorReadU32(cursor, layer.yolox_decoupled_head.num_classes) &&
                   CursorReadU8(cursor, layer.yolox_decoupled_head.num_convs) &&
                   CursorReadExact(cursor, layer.yolox_decoupled_head.reserved,
                                   sizeof(layer.yolox_decoupled_head.reserved));
        }

        bool ReadResNetBasicBlockLayerCursor(ByteCursor& cursor, NkFormat::LayerDesc& layer)
        {
            layer.kind = NkFormat::LayerKind::ResNetBasicBlock;
            return CursorReadU32(cursor, layer.resnet_basic_block.in_channels) &&
                   CursorReadU32(cursor, layer.resnet_basic_block.out_channels) &&
                   CursorReadU8(cursor, layer.resnet_basic_block.stride) &&
                   CursorReadExact(cursor, layer.resnet_basic_block.reserved,
                                   sizeof(layer.resnet_basic_block.reserved));
        }

        bool ReadLayerNormLayerCursor(ByteCursor& cursor, NkFormat::LayerDesc& layer)
        {
            layer.kind = NkFormat::LayerKind::LayerNorm2d;
            return CursorReadU32(cursor, layer.layernorm2d.channels) &&
                   CursorReadU32(cursor, layer.layernorm2d.reserved) &&
                   CursorReadF32(cursor, layer.layernorm2d.eps);
        }

        bool ReadFlattenLayerCursor(ByteCursor& cursor, NkFormat::LayerDesc& layer)
        {
            layer.kind = NkFormat::LayerKind::Flatten;
            (void)cursor;
            return true;
        }

        bool ReadLayerCursor(ByteCursor& cursor, NkFormat::LayerDesc& layer)
        {
            uint8_t kind = 0;
            uint8_t pad[3] = {};
            if (!CursorReadU8(cursor, kind) || !CursorReadExact(cursor, pad, sizeof(pad)))
                return false;

            switch (static_cast<NkFormat::LayerKind>(kind))
            {
                case NkFormat::LayerKind::Dense:
                    return ReadDenseLayerCursor(cursor, layer);
                case NkFormat::LayerKind::Conv2D:
                    return ReadConvLayerCursor(cursor, layer);
                case NkFormat::LayerKind::DepthwiseConv2D:
                    return ReadDepthwiseConvLayerCursor(cursor, layer);
                case NkFormat::LayerKind::MaxPool2D:
                    return ReadPoolLayerCursor(cursor, layer);
                case NkFormat::LayerKind::AvgPool2D:
                    return ReadAvgPoolLayerCursor(cursor, layer);
                case NkFormat::LayerKind::BatchNorm2d:
                    return ReadBatchNormLayerCursor(cursor, layer);
                case NkFormat::LayerKind::ConvNeXtV2Block:
                    return ReadConvNeXtV2BlockLayerCursor(cursor, layer);
                case NkFormat::LayerKind::MobilenetV4Uib:
                    return ReadMobilenetV4UibLayerCursor(cursor, layer);
                case NkFormat::LayerKind::YoloxDecoupledHead:
                    return ReadYoloxDecoupledHeadLayerCursor(cursor, layer);
                case NkFormat::LayerKind::ResNetBasicBlock:
                    return ReadResNetBasicBlockLayerCursor(cursor, layer);
                case NkFormat::LayerKind::LayerNorm2d:
                    return ReadLayerNormLayerCursor(cursor, layer);
                case NkFormat::LayerKind::Flatten:
                    return ReadFlattenLayerCursor(cursor, layer);
                default:
                    SetError("Unsupported .nk layer kind");
                    return false;
            }
        }

        LoadResult ValidateParsedSize(const ParsedModel& parsed, std::size_t total_size)
        {
            const std::size_t expected_size = parsed.payload_offset +
                                              static_cast<std::size_t>(parsed.header.weights_bytes) +
                                              static_cast<std::size_t>(parsed.header.biases_bytes);

            if (total_size < expected_size)
                return Fail(LoadStatus::TruncatedFile, ".nk payload size does not match header");

            const bool has_tests = (parsed.header.flags & NkFormat::kFlagHasTests) != 0;
            if (!has_tests && total_size != expected_size)
                return Fail(LoadStatus::TruncatedFile, ".nk payload size does not match header");

            return LoadResult{LoadStatus::Ok, FromNkNetwork(parsed.header.network_kind), nullptr};
        }

        /* Optional POSIX mmap (NETKIT_USE_MMAP): arena owns the mapping.
         * Otherwise (MCU, RTOS MPU, or mmap unavailable): fread into the arena.
         * Prefer Load*FromBuffer / flash for firmware without a filesystem. */
        LoadResult ReadNkFile(const char* nk_path,
                              Arena& arena,
                              const uint8_t*& out_data,
                              std::size_t& out_size)
        {
            out_data = nullptr;
            out_size = 0;

            if (!nk_path)
                return Fail(LoadStatus::FileOpenFailed, "Missing .nk path");

#if NETKIT_USE_MMAP
            {
                NkMmap::Mapping mapping{};
                if (NkMmap::MapFile(nk_path, mapping))
                {
                    if (mapping.size > 0)
                        arena.attach_mapped_file(mapping.data, mapping.size);
                    out_data = mapping.data;
                    out_size = mapping.size;
                    return LoadResult{LoadStatus::Ok, NetworkKind::Unknown, nullptr};
                }
                // Fall through to fread if mmap failed at runtime.
            }
#endif

            std::FILE* file = std::fopen(nk_path, "rb");
            if (!file)
                return Fail(LoadStatus::FileOpenFailed, "Could not open .nk file");

            if (std::fseek(file, 0, SEEK_END) != 0)
            {
                std::fclose(file);
                return Fail(LoadStatus::ReadFailed, "Could not seek .nk file");
            }

            const long file_size = std::ftell(file);
            if (file_size < 0)
            {
                std::fclose(file);
                return Fail(LoadStatus::ReadFailed, "Could not size .nk file");
            }

            if (std::fseek(file, 0, SEEK_SET) != 0)
            {
                std::fclose(file);
                return Fail(LoadStatus::ReadFailed, "Could not rewind .nk file");
            }

            const std::size_t total_bytes = static_cast<std::size_t>(file_size);
            uint8_t* storage = nullptr;
            if (total_bytes > 0)
            {
                storage = static_cast<uint8_t*>(arena.alloc(total_bytes, alignof(std::max_align_t)));
                if (!storage)
                {
                    std::fclose(file);
                    return Fail(LoadStatus::ArenaOverflow, "Arena out of memory while loading .nk file");
                }

                if (std::fread(storage, 1, total_bytes, file) != total_bytes)
                {
                    std::fclose(file);
                    return Fail(LoadStatus::ReadFailed, "Failed to read .nk file");
                }
            }

            std::fclose(file);
            out_data = storage;
            out_size = total_bytes;
            return LoadResult{LoadStatus::Ok, NetworkKind::Unknown, nullptr};
        }

        LoadResult BindPayloadFromBlob(const ParsedModel& parsed,
                                       const uint8_t* blob,
                                       std::size_t blob_size,
                                       float*& weights,
                                       float*& biases)
        {
            const std::size_t weights_bytes = parsed.header.weights_bytes;
            const std::size_t biases_bytes = parsed.header.biases_bytes;
            const std::size_t total_bytes = weights_bytes + biases_bytes;
            const std::size_t needed = parsed.payload_offset + total_bytes;

            if (!blob || blob_size < needed)
                return Fail(LoadStatus::TruncatedFile, ".nk buffer too small for payload");

            if (total_bytes == 0)
            {
                weights = nullptr;
                biases = nullptr;
                return LoadResult{LoadStatus::Ok, FromNkNetwork(parsed.header.network_kind), nullptr};
            }

            const auto* payload = blob + parsed.payload_offset;
            const uintptr_t weights_addr = reinterpret_cast<uintptr_t>(payload);
            if (weights_addr % alignof(float) != 0)
                return Fail(LoadStatus::SizeMismatch, ".nk payload not float-aligned for flash weights");

            weights = const_cast<float*>(reinterpret_cast<const float*>(payload));
            biases = const_cast<float*>(reinterpret_cast<const float*>(payload + weights_bytes));
            return LoadResult{LoadStatus::Ok, FromNkNetwork(parsed.header.network_kind), nullptr};
        }

        LoadResult ResolvePayloadFromBuffer(const ParsedModel& parsed,
                                            const uint8_t* blob,
                                            std::size_t blob_size,
                                            Arena& arena,
                                            float*& weights,
                                            float*& biases)
        {
            (void)arena;
            return BindPayloadFromBlob(parsed, blob, blob_size, weights, biases);
        }

        ActivationType ToMlpActivation(NkFormat::Activation activation)
        {
            switch (activation)
            {
                case NkFormat::Activation::ReLU: return ActivationType::ReLU;
                case NkFormat::Activation::Sigmoid: return ActivationType::Sigmoid;
                case NkFormat::Activation::Tanh: return ActivationType::Tanh;
                case NkFormat::Activation::LeakyReLU: return ActivationType::LeakyReLU;
                case NkFormat::Activation::ReLU6: return ActivationType::ReLU6;
                case NkFormat::Activation::Softmax: return ActivationType::Softmax;
                default: return ActivationType::None;
            }
        }

        ConvActivationType ToConvActivation(NkFormat::Activation activation)
        {
            switch (activation)
            {
                case NkFormat::Activation::ReLU: return ConvActivationType::ReLU;
                case NkFormat::Activation::Sigmoid: return ConvActivationType::Sigmoid;
                case NkFormat::Activation::Tanh: return ConvActivationType::Tanh;
                case NkFormat::Activation::LeakyReLU: return ConvActivationType::LeakyReLU;
                case NkFormat::Activation::ReLU6: return ConvActivationType::ReLU6;
                case NkFormat::Activation::Softmax: return ConvActivationType::Softmax;
                default: return ConvActivationType::None;
            }
        }

        LoadResult InstantiateMLP(const ParsedModel& parsed,
                                float* weights,
                                float* biases,
                                Arena& arena,
                                MLPNetwork*& network,
                                const std::array<uint32_t, kMaxTensorRank>& input_shape,
                                uint32_t input_rank)
        {
            network = nullptr;

            void* network_mem = arena.alloc(sizeof(MLPNetwork), alignof(MLPNetwork));
            if (!network_mem)
                return Fail(LoadStatus::ArenaOverflow, "Arena out of memory while creating MLPNetwork");

            network = new (network_mem) MLPNetwork(parsed.header.num_layers, arena);
            if (!network->IsValid())
                return Fail(LoadStatus::ArenaOverflow, "Arena out of memory while allocating MLP layers");

            uint32_t weight_index = 0;
            uint32_t bias_index = 0;
            uint32_t in_features = input_shape[1];
            std::size_t weight_offset = 0;
            std::size_t bias_offset = 0;

            for (uint32_t i = 0; i < parsed.header.num_layers; ++i)
            {
                const uint32_t out_features = parsed.layers[i].dense.units;
                const NkFormat::TensorDesc& w_desc = parsed.weight_tensors[weight_index++];
                const NkFormat::TensorDesc& b_desc = parsed.bias_tensors[bias_index++];

                if (w_desc.num_elements != in_features * out_features || b_desc.num_elements != out_features)
                    return Fail(LoadStatus::SizeMismatch, "MLP tensor shape mismatch in .nk catalog");

                Tensor W = TensorFactory::View2D(weights + weight_offset, out_features, in_features);
                Tensor B = TensorFactory::View2D(biases + bias_offset, 1, out_features);
                weight_offset += w_desc.num_elements;
                bias_offset += b_desc.num_elements;

                network->InitLayer(i, W, B, ToMlpActivation(parsed.layers[i].dense.activation),
                                   parsed.layers[i].dense.alpha);
                in_features = out_features;
            }

            if (!network->InitActivationBuffers(arena, input_shape[0]))
                return Fail(LoadStatus::ArenaOverflow, "Arena out of memory while allocating MLP activation buffers");

            (void)input_rank;
            return LoadResult{LoadStatus::Ok, NetworkKind::Mlp, nullptr};
        }

        LoadResult InstantiateQuantizedMLP(ParsedModel& parsed,
                                           int8_t* weights,
                                           int32_t* biases,
                                           Arena& arena,
                                           MLPNetwork*& network,
                                           const std::array<uint32_t, kMaxTensorRank>& input_shape,
                                           uint32_t input_rank)
        {
            network = nullptr;

            if (!ModelIsQuantized(parsed))
                return Fail(LoadStatus::SizeMismatch, ".nk file is not a quantized MLP");

            if (parsed.num_quant_layers != parsed.header.num_weight_tensors)
                return Fail(LoadStatus::SizeMismatch, "Quantized MLP layer count mismatch");

            if (!RelocateWeightChannelScalesToArena(parsed, arena))
            {
                FreeWeightChannelScaleBlob(parsed);
                return Fail(LoadStatus::ArenaOverflow,
                            "Arena out of memory while copying per-channel weight scales");
            }

            void* network_mem = arena.alloc(sizeof(MLPNetwork), alignof(MLPNetwork));
            if (!network_mem)
                return Fail(LoadStatus::ArenaOverflow, "Arena out of memory while creating MLPNetwork");

            network = new (network_mem) MLPNetwork(parsed.header.num_layers, arena);
            if (!network->IsValid())
                return Fail(LoadStatus::ArenaOverflow, "Arena out of memory while allocating MLP layers");

            network->SetQuantized(true);

            uint32_t weight_index = 0;
            uint32_t bias_index = 0;
            uint32_t in_features = input_shape[1];
            std::size_t weight_offset = 0;
            std::size_t bias_offset = 0;

            for (uint32_t i = 0; i < parsed.header.num_layers; ++i)
            {
                const uint32_t out_features = parsed.layers[i].dense.units;
                const NkFormat::TensorDesc& w_desc = parsed.weight_tensors[weight_index++];
                const NkFormat::TensorDesc& b_desc = parsed.bias_tensors[bias_index++];

                if (w_desc.dtype != NkFormat::DType::Int8 || b_desc.dtype != NkFormat::DType::Int32)
                    return Fail(LoadStatus::UnsupportedLayer, "Quantized MLP expects int8 weights and int32 biases");

                if (w_desc.num_elements != in_features * out_features || b_desc.num_elements != out_features)
                    return Fail(LoadStatus::SizeMismatch, "MLP tensor shape mismatch in .nk catalog");

                Tensor W = TensorFactory::View2DInt8(weights + weight_offset, out_features, in_features);
                Tensor B = TensorFactory::View1DInt32(biases + bias_offset, out_features);
                weight_offset += w_desc.num_elements;
                bias_offset += b_desc.num_elements;

                network->InitQuantizedLayer(i,
                                            W,
                                            B,
                                            ToMlpActivation(parsed.layers[i].dense.activation),
                                            parsed.layer_quant[i],
                                            parsed.layers[i].dense.alpha);
                in_features = out_features;
            }

            if (!network->InitActivationBuffers(arena, input_shape[0]))
                return Fail(LoadStatus::ArenaOverflow, "Arena out of memory while allocating MLP activation buffers");

            (void)input_rank;
            return LoadResult{LoadStatus::Ok, NetworkKind::Mlp, nullptr};
        }

        LoadResult BindQuantizedPayloadFromBlob(const ParsedModel& parsed,
                                                const uint8_t* blob,
                                                std::size_t blob_size,
                                                int8_t*& weights,
                                                int32_t*& biases)
        {
            const std::size_t weights_bytes = parsed.header.weights_bytes;
            const std::size_t biases_bytes = parsed.header.biases_bytes;
            const std::size_t needed = parsed.payload_offset + weights_bytes + biases_bytes;

            if (!blob || blob_size < needed)
                return Fail(LoadStatus::TruncatedFile, ".nk buffer too small for payload");

            const auto* payload = blob + parsed.payload_offset;
            weights = const_cast<int8_t*>(reinterpret_cast<const int8_t*>(payload));

            const uintptr_t bias_addr =
                reinterpret_cast<uintptr_t>(payload + weights_bytes);
            if (bias_addr % alignof(int32_t) != 0)
                return Fail(LoadStatus::SizeMismatch, ".nk bias payload not int32-aligned for flash weights");

            biases = const_cast<int32_t*>(reinterpret_cast<const int32_t*>(payload + weights_bytes));
            return LoadResult{LoadStatus::Ok, NetworkKind::Mlp, nullptr};
        }

        LoadResult ResolveQuantizedPayloadFromBuffer(const ParsedModel& parsed,
                                                     const uint8_t* blob,
                                                     std::size_t blob_size,
                                                     Arena& arena,
                                                     int8_t*& weights,
                                                     int32_t*& biases)
        {
            (void)arena;
            return BindQuantizedPayloadFromBlob(parsed, blob, blob_size, weights, biases);
        }

        LoadResult InstantiateCNN(const ParsedModel& parsed,
                                  float* weights,
                                  float* biases,
                                  Arena& arena,
                                  CNNNetwork*& network,
                                  const std::array<uint32_t, kMaxTensorRank>& input_shape,
                                  uint32_t input_rank)
        {
            network = nullptr;

            void* network_mem = arena.alloc(sizeof(CNNNetwork), alignof(CNNNetwork));
            if (!network_mem)
                return Fail(LoadStatus::ArenaOverflow, "Arena out of memory while creating CNNNetwork");

            network = new (network_mem) CNNNetwork(parsed.header.num_layers, arena);
            if (!network->IsValid())
                return Fail(LoadStatus::ArenaOverflow, "Arena out of memory while allocating CNN layers");

            uint32_t weight_index = 0;
            uint32_t bias_index = 0;
            std::size_t weight_offset = 0;
            std::size_t bias_offset = 0;

            uint32_t in_channels = input_shape[2];
            uint32_t h = input_shape[0];
            uint32_t w = input_shape[1];
            uint32_t dense_in = 0;

            for (uint32_t i = 0; i < parsed.header.num_layers; ++i)
            {
                switch (parsed.layers[i].kind)
                {
                    case NkFormat::LayerKind::Conv2D:
                    {
                        const NkFormat::ConvLayerDesc& layer = parsed.layers[i].conv;
                        const NkFormat::TensorDesc& w_desc = parsed.weight_tensors[weight_index++];
                        const NkFormat::TensorDesc& b_desc = parsed.bias_tensors[bias_index++];

                        const std::size_t kernel_area =
                            static_cast<std::size_t>(layer.kernel_size) * layer.kernel_size;
                        if (kernel_area == 0 || layer.filters == 0 ||
                            w_desc.num_elements % (kernel_area * layer.filters) != 0)
                        {
                            return Fail(LoadStatus::SizeMismatch,
                                          "CNN conv tensor shape mismatch in .nk catalog");
                        }
                        const std::size_t conv_in_channels =
                            w_desc.num_elements / (kernel_area * layer.filters);
                        const std::size_t weight_elems = w_desc.num_elements;

                        if (b_desc.num_elements != layer.filters)
                            return Fail(LoadStatus::SizeMismatch, "CNN conv tensor shape mismatch in .nk catalog");

                        int pad_h_end = static_cast<int>(layer.pad_h);
                        int pad_w_end = static_cast<int>(layer.pad_w);
                        nk_op_detail::DecodeConvPadExtra(
                            layer.pad_h, layer.pad_w, layer.kernel_w, pad_h_end, pad_w_end);

                        network->InitConvLayer(i,
                                               static_cast<int>(layer.kernel_size),
                                               static_cast<int>(layer.stride),
                                               static_cast<int>(conv_in_channels),
                                               static_cast<int>(layer.filters),
                                               weights + weight_offset,
                                               biases + bias_offset,
                                               ToConvActivation(layer.activation),
                                               layer.alpha,
                                               static_cast<int>(layer.pad_h),
                                               static_cast<int>(layer.pad_w),
                                               pad_h_end,
                                               pad_w_end);
                        weight_offset += weight_elems;
                        bias_offset += layer.filters;
                        h = static_cast<int>(nk_op_detail::CalcOutputDimAsymmetric(
                            static_cast<uint32_t>(h),
                            static_cast<int>(layer.kernel_size),
                            static_cast<int>(layer.stride),
                            static_cast<int>(layer.pad_h),
                            pad_h_end));
                        w = static_cast<int>(nk_op_detail::CalcOutputDimAsymmetric(
                            static_cast<uint32_t>(w),
                            static_cast<int>(layer.kernel_size),
                            static_cast<int>(layer.stride),
                            static_cast<int>(layer.pad_w),
                            pad_w_end));
                        in_channels = layer.filters;
                        break;
                    }
                    case NkFormat::LayerKind::DepthwiseConv2D:
                    {
                        const NkFormat::ConvLayerDesc& layer = parsed.layers[i].conv;
                        const NkFormat::TensorDesc& w_desc = parsed.weight_tensors[weight_index++];
                        const NkFormat::TensorDesc& b_desc = parsed.bias_tensors[bias_index++];

                        const nk_op_detail::DepthwiseMeta dw_meta = nk_op_detail::DecodeDepthwiseMeta(
                            layer,
                            static_cast<std::size_t>(w_desc.num_elements),
                            static_cast<std::size_t>(layer.filters));
                        const uint32_t kernel_h = dw_meta.kernel_h;
                        const uint32_t kernel_w = dw_meta.kernel_w;
                        if (kernel_w == 0)
                            return Fail(LoadStatus::SizeMismatch,
                                          "Depthwise conv kernel_w must be non-zero in .nk");

                        const std::size_t weight_elems =
                            static_cast<std::size_t>(kernel_h) * kernel_w * layer.filters;

                        if (layer.filters != in_channels)
                            return Fail(LoadStatus::SizeMismatch,
                                          "Depthwise conv filters must match input channels in .nk");

                        if (w_desc.num_elements != weight_elems || b_desc.num_elements != layer.filters)
                            return Fail(LoadStatus::SizeMismatch,
                                          "CNN depthwise conv tensor shape mismatch in .nk catalog");

                        network->InitDepthwiseConvLayer(i,
                                                      static_cast<int>(kernel_h),
                                                      static_cast<int>(kernel_w),
                                                      static_cast<int>(layer.stride),
                                                      static_cast<int>(layer.filters),
                                                      weights + weight_offset,
                                                      biases + bias_offset,
                                                      ToConvActivation(layer.activation),
                                                      layer.alpha,
                                                      static_cast<int>(layer.pad_h),
                                                      static_cast<int>(layer.pad_w),
                                                      dw_meta.pad_h_end,
                                                      dw_meta.pad_w_end);
                        weight_offset += weight_elems;
                        bias_offset += layer.filters;
                        h = nk_op_detail::CalcOutputDimAsymmetric(
                            h, static_cast<int>(kernel_h), static_cast<int>(layer.stride),
                            static_cast<int>(layer.pad_h), dw_meta.pad_h_end);
                        w = nk_op_detail::CalcOutputDimAsymmetric(
                            w, static_cast<int>(kernel_w), static_cast<int>(layer.stride),
                            static_cast<int>(layer.pad_w), dw_meta.pad_w_end);
                        break;
                    }
                    case NkFormat::LayerKind::MaxPool2D:
                    {
                        const NkFormat::PoolLayerDesc& layer = parsed.layers[i].pool;
                        const nk_op_detail::PoolMeta pool_meta = nk_op_detail::DecodePoolMeta(layer);
                        network->InitPoolLayer(i,
                                               pool_meta.pool_h,
                                               pool_meta.pool_w,
                                               static_cast<int>(layer.stride),
                                               pool_meta.pad_h,
                                               pool_meta.pad_w,
                                               pool_meta.pad_h_end,
                                               pool_meta.pad_w_end);
                        h = static_cast<int>(nk_op_detail::CalcOutputDimAsymmetric(
                            static_cast<uint32_t>(h),
                            pool_meta.pool_h,
                            static_cast<int>(layer.stride),
                            pool_meta.pad_h,
                            pool_meta.pad_h_end));
                        w = static_cast<int>(nk_op_detail::CalcOutputDimAsymmetric(
                            static_cast<uint32_t>(w),
                            pool_meta.pool_w,
                            static_cast<int>(layer.stride),
                            pool_meta.pad_w,
                            pool_meta.pad_w_end));
                        break;
                    }
                    case NkFormat::LayerKind::AvgPool2D:
                    {
                        const NkFormat::PoolLayerDesc& layer = parsed.layers[i].pool;
                        const nk_op_detail::PoolMeta pool_meta = nk_op_detail::DecodePoolMeta(layer);
                        network->InitAvgPoolLayer(i,
                                                  pool_meta.pool_h,
                                                  pool_meta.pool_w,
                                                  static_cast<int>(layer.stride),
                                                  pool_meta.pad_h,
                                                  pool_meta.pad_w,
                                                  pool_meta.pad_h_end,
                                                  pool_meta.pad_w_end);
                        h = static_cast<int>(nk_op_detail::CalcOutputDimAsymmetric(
                            static_cast<uint32_t>(h),
                            pool_meta.pool_h,
                            static_cast<int>(layer.stride),
                            pool_meta.pad_h,
                            pool_meta.pad_h_end));
                        w = static_cast<int>(nk_op_detail::CalcOutputDimAsymmetric(
                            static_cast<uint32_t>(w),
                            pool_meta.pool_w,
                            static_cast<int>(layer.stride),
                            pool_meta.pad_w,
                            pool_meta.pad_w_end));
                        break;
                    }
                    case NkFormat::LayerKind::BatchNorm2d:
                    {
                        const NkFormat::BatchNormLayerDesc& layer = parsed.layers[i].batch_norm;
                        const NkFormat::TensorDesc& s_desc = parsed.weight_tensors[weight_index++];
                        const NkFormat::TensorDesc& b_desc = parsed.bias_tensors[bias_index++];

                        if (s_desc.num_elements != layer.channels || b_desc.num_elements != layer.channels)
                            return Fail(LoadStatus::SizeMismatch, "CNN batch norm tensor shape mismatch in .nk catalog");

                        network->InitBatchNormLayer(i,
                                                    static_cast<int>(layer.channels),
                                                    weights + weight_offset,
                                                    biases + bias_offset);
                        weight_offset += layer.channels;
                        bias_offset += layer.channels;
                        break;
                    }
                    case NkFormat::LayerKind::LayerNorm2d:
                    {
                        const NkFormat::LayerNormLayerDesc& layer = parsed.layers[i].layernorm2d;
                        const NkFormat::TensorDesc& w_desc = parsed.weight_tensors[weight_index++];
                        const NkFormat::TensorDesc& b_desc = parsed.bias_tensors[bias_index++];

                        if (w_desc.num_elements != layer.channels || b_desc.num_elements != layer.channels)
                            return Fail(LoadStatus::SizeMismatch, "CNN layer norm tensor shape mismatch in .nk catalog");

                        if (layer.channels != in_channels)
                            return Fail(LoadStatus::SizeMismatch,
                                          "LayerNorm2d channels must match input channels in .nk");

                        network->InitLayerNormLayer(i,
                                                    static_cast<int>(layer.channels),
                                                    layer.eps,
                                                    weights + weight_offset,
                                                    biases + bias_offset);
                        weight_offset += layer.channels;
                        bias_offset += layer.channels;
                        break;
                    }
                    case NkFormat::LayerKind::ConvNeXtV2Block:
                    {
                        const NkFormat::ConvNeXtV2BlockLayerDesc& layer = parsed.layers[i].convnextv2_block;
                        const uint32_t channels = layer.channels;
                        const uint32_t expanded = channels * 4u;
                        const std::size_t dw_w_elems = static_cast<std::size_t>(channels) * 49u;
                        const std::size_t pw1_w_elems = static_cast<std::size_t>(expanded) * channels;
                        const std::size_t pw2_w_elems = static_cast<std::size_t>(channels) * expanded;

                        auto take_pair = [&](std::size_t expected_w,
                                             std::size_t expected_b) -> std::pair<float*, float*>
                        {
                            const NkFormat::TensorDesc& w_desc = parsed.weight_tensors[weight_index++];
                            const NkFormat::TensorDesc& b_desc = parsed.bias_tensors[bias_index++];
                            if (w_desc.num_elements != expected_w || b_desc.num_elements != expected_b)
                                return {nullptr, nullptr};
                            float* w_ptr = weights + weight_offset;
                            float* b_ptr = biases + bias_offset;
                            weight_offset += expected_w;
                            bias_offset += expected_b;
                            return {w_ptr, b_ptr};
                        };

                        const auto [dw_w, dw_b] = take_pair(dw_w_elems, channels);
                        const auto [ln_w, ln_b] = take_pair(channels, channels);
                        const auto [pw1_w, pw1_b] = take_pair(pw1_w_elems, expanded);
                        const auto [grn_gamma, grn_beta] = take_pair(expanded, expanded);
                        const auto [pw2_w, pw2_b] = take_pair(pw2_w_elems, channels);

                        if (!dw_w || !ln_w || !pw1_w || !grn_gamma || !pw2_w)
                            return Fail(LoadStatus::SizeMismatch,
                                          "ConvNeXt V2 block tensor shape mismatch in .nk catalog");

                        if (channels != in_channels)
                            return Fail(LoadStatus::SizeMismatch,
                                          "ConvNeXt V2 block channels must match input channels in .nk");

                        network->InitConvNeXtV2BlockLayer(i,
                                                        arena,
                                                        h,
                                                        w,
                                                        static_cast<int>(channels),
                                                        layer.eps,
                                                        dw_w,
                                                        dw_b,
                                                        ln_w,
                                                        ln_b,
                                                        pw1_w,
                                                        pw1_b,
                                                        grn_gamma,
                                                        grn_beta,
                                                        pw2_w,
                                                        pw2_b);
                        break;
                    }
                    case NkFormat::LayerKind::MobilenetV4Uib:
                    {
                        const NkFormat::MobilenetV4UibLayerDesc& layer = parsed.layers[i].mobilenetv4_uib;
                        const uint32_t in_c = layer.in_channels;
                        const uint32_t out_c = layer.out_channels;
                        const uint32_t start_k = layer.start_dw_kernel;
                        const uint32_t middle_k = layer.middle_dw_kernel;
                        const uint32_t expand_c =
                            MobileNetV4Uib::MakeDivisible(static_cast<float>(in_c) * layer.expand_ratio, 8);

                        auto take_pair = [&](std::size_t expected_w,
                                             std::size_t expected_b) -> std::pair<float*, float*>
                        {
                            const NkFormat::TensorDesc& w_desc = parsed.weight_tensors[weight_index++];
                            const NkFormat::TensorDesc& b_desc = parsed.bias_tensors[bias_index++];
                            if (w_desc.num_elements != expected_w || b_desc.num_elements != expected_b)
                                return {nullptr, nullptr};
                            float* w_ptr = weights + weight_offset;
                            float* b_ptr = biases + bias_offset;
                            weight_offset += expected_w;
                            bias_offset += expected_b;
                            return {w_ptr, b_ptr};
                        };

                        float* start_dw_w = nullptr;
                        float* start_dw_b = nullptr;
                        float* start_bn_s = nullptr;
                        float* start_bn_b = nullptr;
                        if (start_k > 0)
                        {
                            const std::size_t dw_elems =
                                static_cast<std::size_t>(start_k) * start_k * in_c;
                            const auto [dw_w, dw_b] = take_pair(dw_elems, in_c);
                            const auto [bn_s, bn_b] = take_pair(in_c, in_c);
                            if (!dw_w || !bn_s)
                                return Fail(LoadStatus::SizeMismatch,
                                              "MobileNetV4 UIB start depthwise tensor shape mismatch in .nk catalog");
                            start_dw_w = dw_w;
                            start_dw_b = dw_b;
                            start_bn_s = bn_s;
                            start_bn_b = bn_b;
                        }

                        const auto [expand_w, expand_b] = take_pair(
                            static_cast<std::size_t>(expand_c) * in_c, expand_c);
                        const auto [expand_bn_s, expand_bn_b] = take_pair(expand_c, expand_c);

                        float* middle_dw_w = nullptr;
                        float* middle_dw_b = nullptr;
                        float* middle_bn_s = nullptr;
                        float* middle_bn_b = nullptr;
                        if (middle_k > 0)
                        {
                            const std::size_t dw_elems =
                                static_cast<std::size_t>(middle_k) * middle_k * expand_c;
                            const auto [dw_w, dw_b] = take_pair(dw_elems, expand_c);
                            const auto [bn_s, bn_b] = take_pair(expand_c, expand_c);
                            if (!dw_w || !bn_s)
                                return Fail(LoadStatus::SizeMismatch,
                                              "MobileNetV4 UIB middle depthwise tensor shape mismatch in .nk catalog");
                            middle_dw_w = dw_w;
                            middle_dw_b = dw_b;
                            middle_bn_s = bn_s;
                            middle_bn_b = bn_b;
                        }

                        const auto [proj_w, proj_b] =
                            take_pair(static_cast<std::size_t>(out_c) * expand_c, out_c);
                        const auto [proj_bn_s, proj_bn_b] = take_pair(out_c, out_c);

                        if (!expand_w || !proj_w || !proj_bn_s)
                            return Fail(LoadStatus::SizeMismatch,
                                          "MobileNetV4 UIB tensor shape mismatch in .nk catalog");

                        if (in_c != in_channels)
                            return Fail(LoadStatus::SizeMismatch,
                                          "MobileNetV4 UIB in_channels must match input channels in .nk");

                        network->InitMobilenetV4UibLayer(i,
                                                         arena,
                                                         h,
                                                         w,
                                                         static_cast<int>(in_c),
                                                         static_cast<int>(out_c),
                                                         static_cast<int>(start_k),
                                                         static_cast<int>(middle_k),
                                                         static_cast<int>(layer.stride),
                                                         layer.middle_dw_downsample != 0,
                                                         layer.expand_ratio,
                                                         start_dw_w,
                                                         start_dw_b,
                                                         start_bn_s,
                                                         start_bn_b,
                                                         expand_w,
                                                         expand_b,
                                                         expand_bn_s,
                                                         expand_bn_b,
                                                         middle_dw_w,
                                                         middle_dw_b,
                                                         middle_bn_s,
                                                         middle_bn_b,
                                                         proj_w,
                                                         proj_b,
                                                         proj_bn_s,
                                                         proj_bn_b);

                        MobileNetV4Uib shape_probe{};
                        shape_probe.in_channels = static_cast<int>(in_c);
                        shape_probe.out_channels = static_cast<int>(out_c);
                        shape_probe.start_dw_kernel = static_cast<int>(start_k);
                        shape_probe.middle_dw_kernel = static_cast<int>(middle_k);
                        shape_probe.stride = static_cast<int>(layer.stride);
                        shape_probe.middle_dw_downsample = layer.middle_dw_downsample != 0;
                        shape_probe.expand_ratio = layer.expand_ratio;
                        shape_probe.output_spatial(h, w, h, w);
                        in_channels = out_c;
                        break;
                    }
                    case NkFormat::LayerKind::ResNetBasicBlock:
                    {
                        const NkFormat::ResNetBasicBlockLayerDesc& layer = parsed.layers[i].resnet_basic_block;
                        const uint32_t in_c = layer.in_channels;
                        const uint32_t out_c = layer.out_channels;
                        const bool identity = layer.stride == 1 && in_c == out_c;

                        auto take_pair = [&](std::size_t expected_w,
                                             std::size_t expected_b) -> std::pair<float*, float*>
                        {
                            const NkFormat::TensorDesc& w_desc = parsed.weight_tensors[weight_index++];
                            const NkFormat::TensorDesc& b_desc = parsed.bias_tensors[bias_index++];
                            if (w_desc.num_elements != expected_w || b_desc.num_elements != expected_b)
                                return {nullptr, nullptr};
                            float* w_ptr = weights + weight_offset;
                            float* b_ptr = biases + bias_offset;
                            weight_offset += expected_w;
                            bias_offset += expected_b;
                            return {w_ptr, b_ptr};
                        };

                        const std::size_t conv1_w_elems =
                            static_cast<std::size_t>(out_c) * 3u * 3u * in_c;
                        const auto [conv1_w, conv1_b] = take_pair(conv1_w_elems, out_c);
                        const auto [bn1_s, bn1_b] = take_pair(out_c, out_c);
                        const std::size_t conv2_w_elems =
                            static_cast<std::size_t>(out_c) * 3u * 3u * out_c;
                        const auto [conv2_w, conv2_b] = take_pair(conv2_w_elems, out_c);
                        const auto [bn2_s, bn2_b] = take_pair(out_c, out_c);

                        float* shortcut_w = nullptr;
                        float* shortcut_b = nullptr;
                        float* shortcut_bn_s = nullptr;
                        float* shortcut_bn_b = nullptr;
                        if (!identity)
                        {
                            const std::size_t shortcut_w_elems =
                                static_cast<std::size_t>(out_c) * in_c;
                            const auto [sc_w, sc_b] = take_pair(shortcut_w_elems, out_c);
                            const auto [sc_bn_s, sc_bn_b] = take_pair(out_c, out_c);
                            if (!sc_w || !sc_bn_s)
                                return Fail(LoadStatus::SizeMismatch,
                                              "ResNet basic block shortcut tensor shape mismatch in .nk catalog");
                            shortcut_w = sc_w;
                            shortcut_b = sc_b;
                            shortcut_bn_s = sc_bn_s;
                            shortcut_bn_b = sc_bn_b;
                        }

                        if (!conv1_w || !bn1_s || !conv2_w || !bn2_s)
                            return Fail(LoadStatus::SizeMismatch,
                                          "ResNet basic block tensor shape mismatch in .nk catalog");

                        if (in_c != in_channels)
                            return Fail(LoadStatus::SizeMismatch,
                                          "ResNet basic block in_channels must match input channels in .nk");

                        network->InitResNetBasicBlockLayer(i,
                                                           arena,
                                                           h,
                                                           w,
                                                           static_cast<int>(in_c),
                                                           static_cast<int>(out_c),
                                                           static_cast<int>(layer.stride),
                                                           conv1_w,
                                                           conv1_b,
                                                           bn1_s,
                                                           bn1_b,
                                                           conv2_w,
                                                           conv2_b,
                                                           bn2_s,
                                                           bn2_b,
                                                           shortcut_w,
                                                           shortcut_b,
                                                           shortcut_bn_s,
                                                           shortcut_bn_b);

                        ResNetBasicBlock shape_probe{};
                        shape_probe.in_channels = static_cast<int>(in_c);
                        shape_probe.out_channels = static_cast<int>(out_c);
                        shape_probe.stride = static_cast<int>(layer.stride);
                        shape_probe.output_spatial(h, w, h, w);
                        in_channels = out_c;
                        break;
                    }
                    case NkFormat::LayerKind::YoloxDecoupledHead:
                    {
                        const NkFormat::YoloxDecoupledHeadLayerDesc& layer =
                            parsed.layers[i].yolox_decoupled_head;
                        const uint32_t in_c = layer.in_channels;
                        const uint32_t hidden = layer.hidden_dim;
                        const uint32_t num_classes = layer.num_classes;
                        const int num_convs = static_cast<int>(layer.num_convs);

                        auto take_pair = [&](std::size_t expected_w,
                                             std::size_t expected_b) -> std::pair<float*, float*>
                        {
                            const NkFormat::TensorDesc& w_desc = parsed.weight_tensors[weight_index++];
                            const NkFormat::TensorDesc& b_desc = parsed.bias_tensors[bias_index++];
                            if (w_desc.num_elements != expected_w || b_desc.num_elements != expected_b)
                                return {nullptr, nullptr};
                            float* w_ptr = weights + weight_offset;
                            float* b_ptr = biases + bias_offset;
                            weight_offset += expected_w;
                            bias_offset += expected_b;
                            return {w_ptr, b_ptr};
                        };

                        const auto [stem_w, stem_b] =
                            take_pair(static_cast<std::size_t>(hidden) * in_c, hidden);
                        if (!stem_w)
                            return Fail(LoadStatus::SizeMismatch,
                                          "YOLOX head stem tensor shape mismatch in .nk catalog");

                        float* cls_conv_w[YoloxDecoupledHead::kMaxStackedConvs]{};
                        float* cls_conv_b[YoloxDecoupledHead::kMaxStackedConvs]{};
                        float* reg_conv_w[YoloxDecoupledHead::kMaxStackedConvs]{};
                        float* reg_conv_b[YoloxDecoupledHead::kMaxStackedConvs]{};
                        const std::size_t branch_w_elems =
                            static_cast<std::size_t>(hidden) * hidden * 9u;

                        for (int ci = 0; ci < num_convs; ++ci)
                        {
                            const auto [cw, cb] = take_pair(branch_w_elems, hidden);
                            if (!cw)
                                return Fail(LoadStatus::SizeMismatch,
                                              "YOLOX head cls conv tensor shape mismatch in .nk catalog");
                            cls_conv_w[ci] = cw;
                            cls_conv_b[ci] = cb;
                        }

                        for (int ri = 0; ri < num_convs; ++ri)
                        {
                            const auto [rw, rb] = take_pair(branch_w_elems, hidden);
                            if (!rw)
                                return Fail(LoadStatus::SizeMismatch,
                                              "YOLOX head reg conv tensor shape mismatch in .nk catalog");
                            reg_conv_w[ri] = rw;
                            reg_conv_b[ri] = rb;
                        }

                        const auto [cls_pred_w, cls_pred_b] =
                            take_pair(static_cast<std::size_t>(num_classes) * hidden, num_classes);
                        const auto [reg_pred_w, reg_pred_b] =
                            take_pair(static_cast<std::size_t>(4u) * hidden, 4u);
                        const auto [obj_pred_w, obj_pred_b] =
                            take_pair(static_cast<std::size_t>(hidden), 1u);

                        if (!cls_pred_w || !reg_pred_w || !obj_pred_w)
                            return Fail(LoadStatus::SizeMismatch,
                                          "YOLOX head prediction tensor shape mismatch in .nk catalog");

                        if (in_c != in_channels)
                            return Fail(LoadStatus::SizeMismatch,
                                          "YOLOX head in_channels must match input channels in .nk");

                        network->InitYoloxDecoupledHeadLayer(i,
                                                             arena,
                                                             h,
                                                             w,
                                                             static_cast<int>(in_c),
                                                             static_cast<int>(hidden),
                                                             static_cast<int>(num_classes),
                                                             num_convs,
                                                             stem_w,
                                                             stem_b,
                                                             cls_conv_w,
                                                             cls_conv_b,
                                                             reg_conv_w,
                                                             reg_conv_b,
                                                             cls_pred_w,
                                                             cls_pred_b,
                                                             reg_pred_w,
                                                             reg_pred_b,
                                                             obj_pred_w,
                                                             obj_pred_b);
                        in_channels = 4 + 1 + static_cast<int>(num_classes);
                        break;
                    }
                    case NkFormat::LayerKind::Flatten:
                        dense_in = h * w * in_channels;
                        network->InitFlattenLayer(i);
                        break;
                    case NkFormat::LayerKind::Dense:
                    {
                        const NkFormat::DenseLayerDesc& layer = parsed.layers[i].dense;
                        const NkFormat::TensorDesc& w_desc = parsed.weight_tensors[weight_index++];
                        const NkFormat::TensorDesc& b_desc = parsed.bias_tensors[bias_index++];
                        const std::size_t weight_elems = static_cast<std::size_t>(dense_in) * layer.units;

                        if (w_desc.num_elements != weight_elems || b_desc.num_elements != layer.units)
                            return Fail(LoadStatus::SizeMismatch, "CNN dense tensor shape mismatch in .nk catalog");

                        Tensor W = TensorFactory::View2D(weights + weight_offset, layer.units, dense_in);
                        Tensor B = TensorFactory::View2D(biases + bias_offset, 1, layer.units);
                        network->InitDenseLayer(i, W, B, ToMlpActivation(layer.activation), layer.alpha);
                        weight_offset += weight_elems;
                        bias_offset += layer.units;
                        dense_in = layer.units;
                        break;
                    }
                    default:
                        break;
                }
            }

            if (!network->InitActivationBuffers(arena, input_shape[0], input_shape[1], input_shape[2]))
                return Fail(LoadStatus::ArenaOverflow, "Arena out of memory while allocating CNN activation buffers");

            (void)input_rank;
            return LoadResult{LoadStatus::Ok, NetworkKind::Cnn, nullptr};
        }

        LoadResult InstantiateQuantizedCNN(ParsedModel& parsed,
                                           int8_t* weights,
                                           int32_t* biases,
                                           Arena& arena,
                                           CNNNetwork*& network,
                                           const std::array<uint32_t, kMaxTensorRank>& input_shape,
                                           uint32_t input_rank)
        {
            network = nullptr;

            if (!ModelIsQuantized(parsed))
                return Fail(LoadStatus::SizeMismatch, ".nk file is not a quantized CNN");

            if (parsed.num_quant_layers != parsed.header.num_weight_tensors)
                return Fail(LoadStatus::SizeMismatch, "Quantized CNN tensor count mismatch");

            if (!RelocateWeightChannelScalesToArena(parsed, arena))
            {
                FreeWeightChannelScaleBlob(parsed);
                return Fail(LoadStatus::ArenaOverflow,
                            "Arena out of memory while copying per-channel weight scales");
            }

            void* network_mem = arena.alloc(sizeof(CNNNetwork), alignof(CNNNetwork));
            if (!network_mem)
                return Fail(LoadStatus::ArenaOverflow, "Arena out of memory while creating CNNNetwork");

            network = new (network_mem) CNNNetwork(parsed.header.num_layers, arena);
            if (!network->IsValid())
                return Fail(LoadStatus::ArenaOverflow, "Arena out of memory while allocating CNN layers");

            network->SetQuantized(true);

            uint32_t weight_index = 0;
            uint32_t bias_index = 0;
            uint32_t quant_index = 0;
            std::size_t weight_offset = 0;
            std::size_t bias_offset = 0;

            uint32_t in_channels = input_shape[2];
            uint32_t h = input_shape[0];
            uint32_t w = input_shape[1];
            uint32_t dense_in = 0;
            float activation_scale = 1.0f;
            int32_t activation_zero_point = 0;
            bool have_activation_quant = false;

            for (uint32_t i = 0; i < parsed.header.num_layers; ++i)
            {
                switch (parsed.layers[i].kind)
                {
                    case NkFormat::LayerKind::Conv2D:
                    {
                        const NkFormat::ConvLayerDesc& layer = parsed.layers[i].conv;
                        const NkFormat::TensorDesc& w_desc = parsed.weight_tensors[weight_index++];
                        const NkFormat::TensorDesc& b_desc = parsed.bias_tensors[bias_index++];

                        if (w_desc.dtype != NkFormat::DType::Int8 || b_desc.dtype != NkFormat::DType::Int32)
                            return Fail(LoadStatus::UnsupportedLayer,
                                          "Quantized CNN expects int8 weights and int32 biases");

                        const std::size_t kernel_area =
                            static_cast<std::size_t>(layer.kernel_size) * layer.kernel_size;
                        const std::size_t conv_in_channels =
                            w_desc.num_elements / (kernel_area * layer.filters);

                        if (b_desc.num_elements != layer.filters)
                            return Fail(LoadStatus::SizeMismatch, "CNN conv tensor shape mismatch in .nk catalog");

                        int pad_h_end = static_cast<int>(layer.pad_h);
                        int pad_w_end = static_cast<int>(layer.pad_w);
                        nk_op_detail::DecodeConvPadExtra(
                            layer.pad_h, layer.pad_w, layer.kernel_w, pad_h_end, pad_w_end);

                        network->InitQuantizedConvLayer(
                            i,
                            static_cast<int>(layer.kernel_size),
                            static_cast<int>(layer.stride),
                            static_cast<int>(conv_in_channels),
                            static_cast<int>(layer.filters),
                            weights + weight_offset,
                            biases + bias_offset,
                            parsed.layer_quant[quant_index++],
                            ToConvActivation(layer.activation),
                            layer.alpha,
                            static_cast<int>(layer.pad_h),
                            static_cast<int>(layer.pad_w),
                            pad_h_end,
                            pad_w_end);

                        weight_offset += w_desc.num_elements;
                        bias_offset += b_desc.num_elements;

                        if (!have_activation_quant)
                        {
                            activation_scale = parsed.layer_quant[quant_index - 1].input_scale;
                            activation_zero_point = parsed.layer_quant[quant_index - 1].input_zero_point;
                            have_activation_quant = true;
                        }
                        activation_scale = parsed.layer_quant[quant_index - 1].output_scale;
                        activation_zero_point = parsed.layer_quant[quant_index - 1].output_zero_point;

                        h = nk_op_detail::CalcOutputDimAsymmetric(
                            h, static_cast<int>(layer.kernel_size), static_cast<int>(layer.stride),
                            static_cast<int>(layer.pad_h), pad_h_end);
                        w = nk_op_detail::CalcOutputDimAsymmetric(
                            w, static_cast<int>(layer.kernel_size), static_cast<int>(layer.stride),
                            static_cast<int>(layer.pad_w), pad_w_end);
                        in_channels = layer.filters;
                        break;
                    }
                    case NkFormat::LayerKind::DepthwiseConv2D:
                    {
                        const NkFormat::ConvLayerDesc& layer = parsed.layers[i].conv;
                        const NkFormat::TensorDesc& w_desc = parsed.weight_tensors[weight_index++];
                        const NkFormat::TensorDesc& b_desc = parsed.bias_tensors[bias_index++];

                        if (w_desc.dtype != NkFormat::DType::Int8 || b_desc.dtype != NkFormat::DType::Int32)
                            return Fail(LoadStatus::UnsupportedLayer,
                                          "Quantized CNN expects int8 weights and int32 biases");

                        const nk_op_detail::DepthwiseMeta dw_meta = nk_op_detail::DecodeDepthwiseMeta(
                            layer,
                            static_cast<std::size_t>(w_desc.num_elements),
                            static_cast<std::size_t>(layer.filters));
                        const uint32_t kernel_h = dw_meta.kernel_h;
                        const uint32_t kernel_w = dw_meta.kernel_w;
                        if (kernel_w == 0)
                            return Fail(LoadStatus::SizeMismatch,
                                          "Depthwise conv kernel_w must be non-zero in .nk");

                        const std::size_t weight_elems =
                            static_cast<std::size_t>(kernel_h) * kernel_w * layer.filters;

                        if (layer.filters != in_channels)
                            return Fail(LoadStatus::SizeMismatch,
                                          "Depthwise conv filters must match input channels in .nk");

                        if (w_desc.num_elements != weight_elems || b_desc.num_elements != layer.filters)
                            return Fail(LoadStatus::SizeMismatch,
                                          "CNN depthwise conv tensor shape mismatch in .nk catalog");

                        network->InitQuantizedDepthwiseConvLayer(
                            i,
                            static_cast<int>(kernel_h),
                            static_cast<int>(kernel_w),
                            static_cast<int>(layer.stride),
                            static_cast<int>(layer.filters),
                            weights + weight_offset,
                            biases + bias_offset,
                            parsed.layer_quant[quant_index++],
                            ToConvActivation(layer.activation),
                            layer.alpha,
                            static_cast<int>(layer.pad_h),
                            static_cast<int>(layer.pad_w),
                            dw_meta.pad_h_end,
                            dw_meta.pad_w_end);

                        weight_offset += weight_elems;
                        bias_offset += layer.filters;
                        if (!have_activation_quant)
                        {
                            activation_scale = parsed.layer_quant[quant_index - 1].input_scale;
                            activation_zero_point = parsed.layer_quant[quant_index - 1].input_zero_point;
                            have_activation_quant = true;
                        }
                        activation_scale = parsed.layer_quant[quant_index - 1].output_scale;
                        activation_zero_point = parsed.layer_quant[quant_index - 1].output_zero_point;
                        h = nk_op_detail::CalcOutputDimAsymmetric(
                            h, static_cast<int>(kernel_h), static_cast<int>(layer.stride),
                            static_cast<int>(layer.pad_h), dw_meta.pad_h_end);
                        w = nk_op_detail::CalcOutputDimAsymmetric(
                            w, static_cast<int>(kernel_w), static_cast<int>(layer.stride),
                            static_cast<int>(layer.pad_w), dw_meta.pad_w_end);
                        break;
                    }
                    case NkFormat::LayerKind::MobilenetV4Uib:
                    {
                        const NkFormat::MobilenetV4UibLayerDesc& layer = parsed.layers[i].mobilenetv4_uib;
                        const uint32_t in_c = layer.in_channels;
                        const uint32_t out_c = layer.out_channels;
                        const uint32_t start_k = layer.start_dw_kernel;
                        const uint32_t middle_k = layer.middle_dw_kernel;
                        const uint32_t expand_c =
                            MobileNetV4Uib::MakeDivisible(static_cast<float>(in_c) * layer.expand_ratio, 8);

                        if (in_c != in_channels)
                            return Fail(LoadStatus::SizeMismatch,
                                          "MobileNetV4 UIB in_channels must match input channels in .nk");

                        const float block_input_scale = activation_scale;
                        const int32_t block_input_zp = activation_zero_point;

                        auto take_quant_pair = [&](std::size_t expected_w,
                                                   std::size_t expected_b)
                            -> std::tuple<int8_t*, int32_t*, NkFormat::MlpLayerQuantDesc>
                        {
                            const NkFormat::TensorDesc& w_desc = parsed.weight_tensors[weight_index++];
                            const NkFormat::TensorDesc& b_desc = parsed.bias_tensors[bias_index++];
                            if (w_desc.dtype != NkFormat::DType::Int8 ||
                                b_desc.dtype != NkFormat::DType::Int32 ||
                                w_desc.num_elements != expected_w || b_desc.num_elements != expected_b)
                                return {nullptr, nullptr, {}};
                            int8_t* w_ptr = weights + weight_offset;
                            int32_t* b_ptr = biases + bias_offset;
                            weight_offset += expected_w;
                            bias_offset += expected_b;
                            const NkFormat::MlpLayerQuantDesc quant = parsed.layer_quant[quant_index++];
                            return {w_ptr, b_ptr, quant};
                        };

                        int8_t* start_dw_w = nullptr;
                        int32_t* start_dw_b = nullptr;
                        NkFormat::MlpLayerQuantDesc start_dw_quant{};
                        if (start_k > 0)
                        {
                            const std::size_t dw_elems =
                                static_cast<std::size_t>(start_k) * start_k * in_c;
                            const auto [dw_w, dw_b, dw_q] = take_quant_pair(dw_elems, in_c);
                            if (!dw_w)
                                return Fail(LoadStatus::SizeMismatch,
                                              "Quantized MobileNetV4 UIB start depthwise tensor mismatch");
                            start_dw_w = dw_w;
                            start_dw_b = dw_b;
                            start_dw_quant = dw_q;
                        }

                        const auto [expand_w, expand_b, expand_quant] =
                            take_quant_pair(static_cast<std::size_t>(expand_c) * in_c, expand_c);
                        int8_t* middle_dw_w = nullptr;
                        int32_t* middle_dw_b = nullptr;
                        NkFormat::MlpLayerQuantDesc middle_dw_quant{};
                        if (middle_k > 0)
                        {
                            const std::size_t dw_elems =
                                static_cast<std::size_t>(middle_k) * middle_k * expand_c;
                            const auto [dw_w, dw_b, dw_q] = take_quant_pair(dw_elems, expand_c);
                            if (!dw_w)
                                return Fail(LoadStatus::SizeMismatch,
                                              "Quantized MobileNetV4 UIB middle depthwise tensor mismatch");
                            middle_dw_w = dw_w;
                            middle_dw_b = dw_b;
                            middle_dw_quant = dw_q;
                        }

                        const auto [proj_w, proj_b, proj_quant] =
                            take_quant_pair(static_cast<std::size_t>(out_c) * expand_c, out_c);
                        if (!expand_w || !proj_w)
                            return Fail(LoadStatus::SizeMismatch,
                                          "Quantized MobileNetV4 UIB tensor shape mismatch in .nk catalog");

                        if (!have_activation_quant)
                        {
                            activation_scale = expand_quant.input_scale;
                            activation_zero_point = expand_quant.input_zero_point;
                            have_activation_quant = true;
                        }

                        network->InitQuantizedMobilenetV4UibLayer(i,
                                                                  arena,
                                                                  h,
                                                                  w,
                                                                  static_cast<int>(in_c),
                                                                  static_cast<int>(out_c),
                                                                  static_cast<int>(start_k),
                                                                  static_cast<int>(middle_k),
                                                                  static_cast<int>(layer.stride),
                                                                  layer.middle_dw_downsample != 0,
                                                                  layer.expand_ratio,
                                                                  block_input_scale,
                                                                  block_input_zp,
                                                                  start_dw_w,
                                                                  start_dw_b,
                                                                  start_dw_quant,
                                                                  expand_w,
                                                                  expand_b,
                                                                  expand_quant,
                                                                  middle_dw_w,
                                                                  middle_dw_b,
                                                                  middle_dw_quant,
                                                                  proj_w,
                                                                  proj_b,
                                                                  proj_quant);

                        MobileNetV4Uib shape_probe{};
                        shape_probe.in_channels = static_cast<int>(in_c);
                        shape_probe.out_channels = static_cast<int>(out_c);
                        shape_probe.start_dw_kernel = static_cast<int>(start_k);
                        shape_probe.middle_dw_kernel = static_cast<int>(middle_k);
                        shape_probe.stride = static_cast<int>(layer.stride);
                        shape_probe.middle_dw_downsample = layer.middle_dw_downsample != 0;
                        shape_probe.expand_ratio = layer.expand_ratio;
                        shape_probe.output_spatial(h, w, h, w);
                        in_channels = out_c;
                        activation_scale = proj_quant.output_scale;
                        activation_zero_point = proj_quant.output_zero_point;
                        break;
                    }
                    case NkFormat::LayerKind::MaxPool2D:
                    {
                        const NkFormat::PoolLayerDesc& layer = parsed.layers[i].pool;
                        const nk_op_detail::PoolMeta pool_meta = nk_op_detail::DecodePoolMeta(layer);
                        network->InitPoolLayer(i,
                                               pool_meta.pool_h,
                                               pool_meta.pool_w,
                                               static_cast<int>(layer.stride),
                                               pool_meta.pad_h,
                                               pool_meta.pad_w,
                                               pool_meta.pad_h_end,
                                               pool_meta.pad_w_end);
                        h = nk_op_detail::CalcOutputDimAsymmetric(
                            h, pool_meta.pool_h, static_cast<int>(layer.stride), pool_meta.pad_h,
                            pool_meta.pad_h_end);
                        w = nk_op_detail::CalcOutputDimAsymmetric(
                            w, pool_meta.pool_w, static_cast<int>(layer.stride), pool_meta.pad_w,
                            pool_meta.pad_w_end);
                        break;
                    }
                    case NkFormat::LayerKind::AvgPool2D:
                    {
                        const NkFormat::PoolLayerDesc& layer = parsed.layers[i].pool;
                        const nk_op_detail::PoolMeta pool_meta = nk_op_detail::DecodePoolMeta(layer);
                        network->InitAvgPoolLayer(i,
                                                  pool_meta.pool_h,
                                                  pool_meta.pool_w,
                                                  static_cast<int>(layer.stride),
                                                  pool_meta.pad_h,
                                                  pool_meta.pad_w,
                                                  pool_meta.pad_h_end,
                                                  pool_meta.pad_w_end);
                        h = nk_op_detail::CalcOutputDimAsymmetric(
                            h, pool_meta.pool_h, static_cast<int>(layer.stride), pool_meta.pad_h,
                            pool_meta.pad_h_end);
                        w = nk_op_detail::CalcOutputDimAsymmetric(
                            w, pool_meta.pool_w, static_cast<int>(layer.stride), pool_meta.pad_w,
                            pool_meta.pad_w_end);
                        break;
                    }
                    case NkFormat::LayerKind::Flatten:
                    {
                        network->InitFlattenLayer(i);
                        dense_in = h * w * in_channels;
                        break;
                    }
                    case NkFormat::LayerKind::Dense:
                    {
                        const NkFormat::DenseLayerDesc& layer = parsed.layers[i].dense;
                        const NkFormat::TensorDesc& w_desc = parsed.weight_tensors[weight_index++];
                        const NkFormat::TensorDesc& b_desc = parsed.bias_tensors[bias_index++];

                        if (w_desc.dtype != NkFormat::DType::Int8 || b_desc.dtype != NkFormat::DType::Int32)
                            return Fail(LoadStatus::UnsupportedLayer,
                                          "Quantized CNN expects int8 weights and int32 biases");

                        const std::size_t weight_elems = static_cast<std::size_t>(dense_in) * layer.units;
                        if (w_desc.num_elements != weight_elems || b_desc.num_elements != layer.units)
                            return Fail(LoadStatus::SizeMismatch, "CNN dense tensor shape mismatch in .nk catalog");

                        Tensor W = TensorFactory::View2DInt8(weights + weight_offset, layer.units, dense_in);
                        Tensor B = TensorFactory::View1DInt32(biases + bias_offset, layer.units);
                        network->InitQuantizedDenseLayer(i,
                                                         W,
                                                         B,
                                                         parsed.layer_quant[quant_index++],
                                                         ToMlpActivation(layer.activation),
                                                         layer.alpha);
                        weight_offset += weight_elems;
                        bias_offset += b_desc.num_elements;
                        dense_in = layer.units;
                        break;
                    }
                    default:
                        return Fail(LoadStatus::UnsupportedLayer,
                                      "Quantized CNN does not support this layer kind yet");
                }
            }

            if (!network->InitQuantizedActivationBuffers(arena, input_shape[0], input_shape[1], input_shape[2]))
                return Fail(LoadStatus::ArenaOverflow,
                            "Arena out of memory while allocating quantized CNN activation buffers");

            (void)input_rank;
            return LoadResult{LoadStatus::Ok, NetworkKind::Cnn, nullptr};
        }
    }

    const char* StatusMessage(LoadStatus status)
    {
        switch (status)
        {
            case LoadStatus::Ok: return "ok";
            case LoadStatus::FileOpenFailed: return "file open failed";
            case LoadStatus::ReadFailed: return "read failed";
            case LoadStatus::InvalidMagic: return "invalid magic";
            case LoadStatus::UnsupportedVersion: return "unsupported version";
            case LoadStatus::TruncatedFile: return "truncated file";
            case LoadStatus::UnsupportedLayer: return "unsupported layer";
            case LoadStatus::SizeMismatch: return "size mismatch";
            case LoadStatus::ArenaOverflow: return "arena overflow";
        }
        return "unknown";
    }

    void FreeParsedModelExtras(ParsedModel& parsed)
    {
        FreeWeightChannelScaleBlob(parsed);
    }

    LoadResult ParseFile(const char* nk_path, ParsedModel& out)
    {
        out = ParsedModel();

        std::FILE* file = std::fopen(nk_path, "rb");
        if (!file)
            return Fail(LoadStatus::FileOpenFailed, "Could not open .nk file");

        if (!ReadHeader(file, out.header))
        {
            std::fclose(file);
            return Fail(LoadStatus::InvalidMagic, g_error[0] ? g_error : "Failed to read .nk header");
        }

        if (!NkFormat::IsSupportedVersion(out.header.version))
        {
            std::fclose(file);
            return Fail(LoadStatus::UnsupportedVersion, "Unsupported .nk version");
        }

        if (out.header.num_layers > NkFormat::kMaxLayers ||
            out.header.num_weight_tensors > NkFormat::kMaxTensorCatalog ||
            out.header.num_bias_tensors > NkFormat::kMaxTensorCatalog)
        {
            std::fclose(file);
            return Fail(LoadStatus::UnsupportedLayer, "Too many layers or tensors in .nk file");
        }

        for (uint32_t i = 0; i < out.header.num_layers; ++i)
        {
            if (!ReadLayer(file, out.layers[i]))
            {
                std::fclose(file);
                return Fail(LoadStatus::ReadFailed, g_error[0] ? g_error : "Failed to read layer descriptor");
            }
        }

        for (uint32_t i = 0; i < out.header.num_weight_tensors; ++i)
        {
            if (!ReadTensorDesc(file, out.weight_tensors[i]))
            {
                std::fclose(file);
                return Fail(LoadStatus::ReadFailed, "Failed to read weight tensor descriptor");
            }
        }

        for (uint32_t i = 0; i < out.header.num_bias_tensors; ++i)
        {
            if (!ReadTensorDesc(file, out.bias_tensors[i]))
            {
                std::fclose(file);
                return Fail(LoadStatus::ReadFailed, "Failed to read bias tensor descriptor");
            }
        }

        if (!ReadQuantBlockFile(file, out))
        {
            FreeWeightChannelScaleBlob(out);
            std::fclose(file);
            return Fail(LoadStatus::ReadFailed, "Failed to read .nk quantization block");
        }

        const long payload_start = std::ftell(file);
        if (payload_start < 0)
        {
            FreeWeightChannelScaleBlob(out);
            std::fclose(file);
            return Fail(LoadStatus::ReadFailed, "Could not seek .nk file");
        }

        if (!AdvancePayloadAlignmentPadding(file, out.payload_offset))
        {
            FreeWeightChannelScaleBlob(out);
            std::fclose(file);
            return Fail(LoadStatus::ReadFailed, "Could not align .nk payload offset");
        }

        if (std::fseek(file, 0, SEEK_END) != 0)
        {
            FreeWeightChannelScaleBlob(out);
            std::fclose(file);
            return Fail(LoadStatus::ReadFailed, "Could not seek .nk file end");
        }

        const long file_size = std::ftell(file);
        const std::size_t expected_size = out.payload_offset +
                                          static_cast<std::size_t>(out.header.weights_bytes) +
                                          static_cast<std::size_t>(out.header.biases_bytes);

        std::fclose(file);

        if (file_size < 0 || static_cast<std::size_t>(file_size) < expected_size)
        {
            FreeWeightChannelScaleBlob(out);
            return Fail(LoadStatus::TruncatedFile, ".nk payload size does not match header");
        }

        const LoadResult validated = ValidateParsedSize(out, static_cast<std::size_t>(file_size));
        if (validated.status != LoadStatus::Ok)
            FreeWeightChannelScaleBlob(out);
        return validated;
    }

    LoadResult ParseBuffer(const uint8_t* data, std::size_t size, ParsedModel& out)
    {
        out = ParsedModel();

        if (!data || size == 0)
            return Fail(LoadStatus::ReadFailed, "Empty .nk buffer");

        ByteCursor cursor{data, size, 0};

        if (!ReadHeaderCursor(cursor, out.header))
            return Fail(LoadStatus::InvalidMagic, g_error[0] ? g_error : "Failed to read .nk header");

        if (!NkFormat::IsSupportedVersion(out.header.version))
            return Fail(LoadStatus::UnsupportedVersion, "Unsupported .nk version");

        if (out.header.num_layers > NkFormat::kMaxLayers ||
            out.header.num_weight_tensors > NkFormat::kMaxTensorCatalog ||
            out.header.num_bias_tensors > NkFormat::kMaxTensorCatalog)
            return Fail(LoadStatus::UnsupportedLayer, "Too many layers or tensors in .nk file");

        for (uint32_t i = 0; i < out.header.num_layers; ++i)
        {
            if (!ReadLayerCursor(cursor, out.layers[i]))
                return Fail(LoadStatus::ReadFailed, g_error[0] ? g_error : "Failed to read layer descriptor");
        }

        for (uint32_t i = 0; i < out.header.num_weight_tensors; ++i)
        {
            if (!ReadTensorDescCursor(cursor, out.weight_tensors[i]))
                return Fail(LoadStatus::ReadFailed, "Failed to read weight tensor descriptor");
        }

        for (uint32_t i = 0; i < out.header.num_bias_tensors; ++i)
        {
            if (!ReadTensorDescCursor(cursor, out.bias_tensors[i]))
                return Fail(LoadStatus::ReadFailed, "Failed to read bias tensor descriptor");
        }

        if (!ReadQuantBlockCursor(cursor, out))
        {
            FreeWeightChannelScaleBlob(out);
            return Fail(LoadStatus::ReadFailed, "Failed to read .nk quantization block");
        }

        if (!AdvancePayloadAlignmentPadding(cursor))
        {
            FreeWeightChannelScaleBlob(out);
            return Fail(LoadStatus::ReadFailed, "Could not align .nk payload offset");
        }

        out.payload_offset = cursor.pos;
        const LoadResult validated = ValidateParsedSize(out, size);
        if (validated.status != LoadStatus::Ok)
            FreeWeightChannelScaleBlob(out);
        return validated;
    }

    std::size_t ModelPayloadBytes(const ParsedModel& model)
    {
        return model.payload_offset + static_cast<std::size_t>(model.header.weights_bytes) +
               static_cast<std::size_t>(model.header.biases_bytes);
    }

    LoadResult ReadTestSuite(const char* nk_path, TestSuite& out)
    {
        out = TestSuite{};

        ParsedModel parsed{};
        const LoadResult parse_result = ParseFile(nk_path, parsed);
        if (parse_result.status != LoadStatus::Ok)
            return parse_result;

        if ((parsed.header.flags & NkFormat::kFlagHasTests) == 0)
            return Fail(LoadStatus::ReadFailed, "No embedded regression tests in .nk file");

        std::FILE* file = std::fopen(nk_path, "rb");
        if (!file)
            return Fail(LoadStatus::FileOpenFailed, "Could not open .nk file");

        const std::size_t model_bytes = ModelPayloadBytes(parsed);
        if (std::fseek(file, static_cast<long>(model_bytes), SEEK_SET) != 0)
        {
            std::fclose(file);
            return Fail(LoadStatus::ReadFailed, "Could not seek to .nk test section");
        }

        char magic[4] = {};
        if (!ReadExact(file, magic, 4) || std::memcmp(magic, NkFormat::kTestMagic, 4) != 0)
        {
            std::fclose(file);
            return Fail(LoadStatus::ReadFailed, "Invalid .nk test section magic (expected TCAS)");
        }

        uint32_t num_cases = 0;
        if (!ReadU32(file, num_cases) || num_cases == 0 || num_cases > NkFormat::kMaxTestCases)
        {
            std::fclose(file);
            return Fail(LoadStatus::ReadFailed, "Invalid embedded test case count");
        }

        if (!ReadF32(file, out.tolerance))
        {
            std::fclose(file);
            return Fail(LoadStatus::ReadFailed, "Failed to read test tolerance");
        }

        for (uint32_t i = 0; i < num_cases; ++i)
        {
            TestCase& test_case = out.cases[i];

            uint8_t name_len = 0;
            if (!ReadU8(file, name_len) || name_len == 0 || name_len > NkFormat::kMaxCaseNameLen)
            {
                std::fclose(file);
                return Fail(LoadStatus::ReadFailed, "Invalid embedded test case name length");
            }

            if (!ReadExact(file, test_case.name, name_len))
            {
                std::fclose(file);
                return Fail(LoadStatus::ReadFailed, "Failed to read embedded test case name");
            }
            test_case.name[name_len] = '\0';

            const uint32_t name_pad = (4u - ((1u + name_len) % 4u)) % 4u;
            if (name_pad > 0)
            {
                char pad[3] = {};
                if (!ReadExact(file, pad, name_pad))
                {
                    std::fclose(file);
                    return Fail(LoadStatus::ReadFailed, "Failed to read test case name padding");
                }
            }

            int32_t label = NkFormat::kNoLabel;
            if (!ReadExact(file, &label, sizeof(label)))
            {
                std::fclose(file);
                return Fail(LoadStatus::ReadFailed, "Failed to read embedded test label");
            }
            test_case.label = label;

            if (!ReadU32(file, test_case.input_count) || test_case.input_count == 0 ||
                test_case.input_count > NkFormat::kMaxCaseFloats)
            {
                std::fclose(file);
                return Fail(LoadStatus::ReadFailed, "Invalid embedded test input count");
            }

            const bool int8_inputs =
                (parsed.header.flags & NkFormat::kFlagHasInt8Tests) != 0;
            if (int8_inputs)
            {
                if (!ReadExact(file,
                               test_case.input_i8,
                               static_cast<std::size_t>(test_case.input_count) * sizeof(int8_t)))
                {
                    std::fclose(file);
                    return Fail(LoadStatus::ReadFailed, "Failed to read embedded int8 test input");
                }
            }
            else if (!ReadExact(file,
                                test_case.input,
                                static_cast<std::size_t>(test_case.input_count) * sizeof(float)))
            {
                std::fclose(file);
                return Fail(LoadStatus::ReadFailed, "Failed to read embedded test input");
            }

            if (!ReadU32(file, test_case.output_count) || test_case.output_count == 0 ||
                test_case.output_count > NkFormat::kMaxCaseFloats)
            {
                std::fclose(file);
                return Fail(LoadStatus::ReadFailed, "Invalid embedded test output count");
            }

            if (!ReadExact(file, test_case.expected,
                           static_cast<std::size_t>(test_case.output_count) * sizeof(float)))
            {
                std::fclose(file);
                return Fail(LoadStatus::ReadFailed, "Failed to read embedded test expected output");
            }
        }

        out.num_cases = num_cases;
        out.inputs_are_int8 = (parsed.header.flags & NkFormat::kFlagHasInt8Tests) != 0;
        std::fclose(file);
        FreeParsedModelExtras(parsed);
        return LoadResult{LoadStatus::Ok, parse_result.kind, nullptr};
    }

    void FillArchInfo(const ParsedModel& model, ArchInfo& info)
    {
        info = ArchInfo{};
        info.version = model.header.version;
        info.kind = FromNkNetwork(model.header.network_kind);
        info.input_rank = model.header.input_rank;
        info.num_layers = model.header.num_layers;
        for (uint32_t i = 0; i < info.input_rank; ++i)
            info.input_shape[i] = model.header.input_shape[i];
        info.input_elements = InputElements(model);
        info.output_elements = OutputElements(model);
        info.weight_floats =
            (static_cast<std::size_t>(model.header.weights_bytes) + model.header.biases_bytes) /
            sizeof(float);
    }

    uint32_t InputElements(const ParsedModel& model)
    {
        uint32_t count = 1;
        for (uint32_t i = 0; i < model.header.input_rank; ++i)
            count *= model.header.input_shape[i];
        return count;
    }

    uint32_t OutputElements(const ParsedModel& model)
    {
        return ComputeOutputElements(model);
    }

    const char* NetworkKindName(NetworkKind kind)
    {
        switch (kind)
        {
            case NetworkKind::Mlp: return "MLP";
            case NetworkKind::Cnn: return "CNN";
            default: return "Unknown";
        }
    }

    void PrintNetworkSummary(const char* nk_path, const ParsedModel& model)
    {
#ifndef NETKIT_DISABLE_IOSTREAM
        char name[kMaxPathLen] = {};
        ModelNameFromPath(nk_path, name, sizeof(name));

        std::cout << "=====================================================\n";
        std::cout << "Network Summary\n";
        std::cout << "=====================================================\n\n";
        std::cout << "Name        : " << name << "\n";
        std::cout << "Type        : " << NetworkKindName(FromNkNetwork(model.header.network_kind)) << "\n";
        std::cout << "Version     : " << model.header.version << "\n\n";
        std::cout << "Input Shape : [";
        for (uint32_t i = 0; i < model.header.input_rank; ++i)
        {
            std::cout << model.header.input_shape[i];
            if (i + 1 < model.header.input_rank)
                std::cout << ", ";
        }
        std::cout << "]\n\n";
        std::cout << "Layers (" << model.header.num_layers << ")\n";
        std::cout << "-----------------------------------------------------\n";

        for (uint32_t i = 0; i < model.header.num_layers; ++i)
        {
            std::cout << "  [" << i << "] ";
            const NkFormat::LayerDesc& layer = model.layers[i];
            switch (layer.kind)
            {
                case NkFormat::LayerKind::Dense:
                    std::cout << "Dense units=" << layer.dense.units
                              << " activation=" << NkFormat::ActivationName(layer.dense.activation);
                    break;
                case NkFormat::LayerKind::Conv2D:
                    std::cout << "Conv2D kernel=" << layer.conv.kernel_size
                              << " stride=" << layer.conv.stride << " filters=" << layer.conv.filters
                              << " pad=" << static_cast<uint32_t>(layer.conv.pad_h) << ","
                              << static_cast<uint32_t>(layer.conv.pad_w)
                              << " activation=" << NkFormat::ActivationName(layer.conv.activation);
                    break;
                case NkFormat::LayerKind::DepthwiseConv2D:
                {
                    const uint32_t kernel_w = NkFormat::DepthwiseKernelW(layer.conv);
                    std::cout << "DepthwiseConv2D kernel=" << layer.conv.kernel_size << "x" << kernel_w
                              << " stride=" << layer.conv.stride << " channels=" << layer.conv.filters
                              << " pad=" << static_cast<uint32_t>(layer.conv.pad_h) << ","
                              << static_cast<uint32_t>(layer.conv.pad_w)
                              << " activation=" << NkFormat::ActivationName(layer.conv.activation);
                    break;
                }
                case NkFormat::LayerKind::MaxPool2D:
                    std::cout << "MaxPool2D pool=" << layer.pool.pool_size
                              << " stride=" << layer.pool.stride;
                    break;
                case NkFormat::LayerKind::AvgPool2D:
                    std::cout << "AvgPool2D pool=" << layer.pool.pool_size
                              << " stride=" << layer.pool.stride;
                    break;
                case NkFormat::LayerKind::BatchNorm2d:
                    std::cout << "BatchNorm2d channels=" << layer.batch_norm.channels;
                    break;
                case NkFormat::LayerKind::LayerNorm2d:
                    std::cout << "LayerNorm2d channels=" << layer.layernorm2d.channels
                              << " eps=" << layer.layernorm2d.eps;
                    break;
                case NkFormat::LayerKind::ConvNeXtV2Block:
                    std::cout << "ConvNeXtV2Block channels=" << layer.convnextv2_block.channels
                              << " eps=" << layer.convnextv2_block.eps;
                    break;
                case NkFormat::LayerKind::MobilenetV4Uib:
                    std::cout << "MobilenetV4Uib in=" << layer.mobilenetv4_uib.in_channels
                              << " out=" << layer.mobilenetv4_uib.out_channels
                              << " start_dw=" << static_cast<uint32_t>(layer.mobilenetv4_uib.start_dw_kernel)
                              << " middle_dw="
                              << static_cast<uint32_t>(layer.mobilenetv4_uib.middle_dw_kernel)
                              << " stride=" << static_cast<uint32_t>(layer.mobilenetv4_uib.stride)
                              << " expand=" << layer.mobilenetv4_uib.expand_ratio;
                    break;
                case NkFormat::LayerKind::ResNetBasicBlock:
                    std::cout << "ResNetBasicBlock in=" << layer.resnet_basic_block.in_channels
                              << " out=" << layer.resnet_basic_block.out_channels
                              << " stride=" << static_cast<uint32_t>(layer.resnet_basic_block.stride);
                    break;
                case NkFormat::LayerKind::YoloxDecoupledHead:
                    std::cout << "YoloxDecoupledHead in=" << layer.yolox_decoupled_head.in_channels
                              << " hidden=" << layer.yolox_decoupled_head.hidden_dim
                              << " classes=" << layer.yolox_decoupled_head.num_classes
                              << " convs=" << static_cast<uint32_t>(layer.yolox_decoupled_head.num_convs);
                    break;
                case NkFormat::LayerKind::Flatten:
                    std::cout << "Flatten";
                    break;
            }
            std::cout << "\n";
        }

        std::cout << "-----------------------------------------------------\n";
        std::cout << "Input elements : " << InputElements(model) << "\n";
        std::cout << "Output elements: " << OutputElements(model) << "\n";
        std::cout << "Weight floats  : " << (model.header.weights_bytes + model.header.biases_bytes) / sizeof(float)
                  << "\n";
        std::cout << "=====================================================\n";
#else
        (void)nk_path;
        (void)model;
#endif
    }

    LoadResult LoadMLP(const char* nk_path,
                       Arena& arena,
                       MLPNetwork*& network,
                       std::array<uint32_t, kMaxTensorRank>& input_shape,
                       uint32_t& input_rank)
    {
        network = nullptr;

        const uint8_t* blob = nullptr;
        std::size_t blob_size = 0;
        const LoadResult blob_result = ReadNkFile(nk_path, arena, blob, blob_size);
        if (blob_result.status != LoadStatus::Ok)
            return blob_result;
        return LoadMLPFromBuffer(blob, blob_size, arena, network, input_shape, input_rank);
    }

    LoadResult LoadMLPFromBuffer(const uint8_t* data,
                                 std::size_t size,
                                 Arena& arena,
                                 MLPNetwork*& network,
                                 std::array<uint32_t, kMaxTensorRank>& input_shape,
                                 uint32_t& input_rank)
    {
        network = nullptr;

        ParsedModel parsed{};
        LoadResult parse_result = ParseBuffer(data, size, parsed);
        if (parse_result.status != LoadStatus::Ok)
            return parse_result;

        if (parsed.header.network_kind != NkFormat::NetworkKind::Mlp)
        {
            FreeWeightChannelScaleBlob(parsed);
            return Fail(LoadStatus::UnsupportedLayer, ".nk buffer is not an MLP");
        }

        input_rank = parsed.header.input_rank;
        for (uint32_t i = 0; i < input_rank; ++i)
            input_shape[i] = parsed.header.input_shape[i];

        if (ModelIsQuantized(parsed))
        {
            int8_t* weights = nullptr;
            int32_t* biases = nullptr;
            const LoadResult copy_result =
                ResolveQuantizedPayloadFromBuffer(parsed, data, size, arena, weights, biases);
            if (copy_result.status != LoadStatus::Ok)
            {
                FreeWeightChannelScaleBlob(parsed);
                return copy_result;
            }

            return InstantiateQuantizedMLP(
                parsed, weights, biases, arena, network, input_shape, input_rank);
        }

        float* weights = nullptr;
        float* biases = nullptr;
        LoadResult copy_result = ResolvePayloadFromBuffer(parsed, data, size, arena, weights, biases);
        if (copy_result.status != LoadStatus::Ok)
            return copy_result;

        return InstantiateMLP(parsed, weights, biases, arena, network, input_shape, input_rank);
    }

    LoadResult LoadCNN(const char* nk_path,
                       Arena& arena,
                       CNNNetwork*& network,
                       std::array<uint32_t, kMaxTensorRank>& input_shape,
                       uint32_t& input_rank)
    {
        network = nullptr;

        const uint8_t* blob = nullptr;
        std::size_t blob_size = 0;
        const LoadResult blob_result = ReadNkFile(nk_path, arena, blob, blob_size);
        if (blob_result.status != LoadStatus::Ok)
            return blob_result;
        return LoadCNNFromBuffer(blob, blob_size, arena, network, input_shape, input_rank);
    }

    LoadResult LoadCNNFromBuffer(const uint8_t* data,
                                 std::size_t size,
                                 Arena& arena,
                                 CNNNetwork*& network,
                                 std::array<uint32_t, kMaxTensorRank>& input_shape,
                                 uint32_t& input_rank)
    {
        network = nullptr;

        ParsedModel parsed{};
        LoadResult parse_result = ParseBuffer(data, size, parsed);
        if (parse_result.status != LoadStatus::Ok)
            return parse_result;

        if (parsed.header.network_kind != NkFormat::NetworkKind::Cnn)
        {
            FreeWeightChannelScaleBlob(parsed);
            return Fail(LoadStatus::UnsupportedLayer, ".nk buffer is not a CNN");
        }

        input_rank = parsed.header.input_rank;
        for (uint32_t i = 0; i < input_rank; ++i)
            input_shape[i] = parsed.header.input_shape[i];

        if (ModelIsQuantized(parsed))
        {
            int8_t* weights = nullptr;
            int32_t* biases = nullptr;
            const LoadResult copy_result =
                ResolveQuantizedPayloadFromBuffer(parsed, data, size, arena, weights, biases);
            if (copy_result.status != LoadStatus::Ok)
            {
                FreeWeightChannelScaleBlob(parsed);
                return copy_result;
            }

            return InstantiateQuantizedCNN(
                parsed, weights, biases, arena, network, input_shape, input_rank);
        }

        float* weights = nullptr;
        float* biases = nullptr;
        LoadResult copy_result = ResolvePayloadFromBuffer(parsed, data, size, arena, weights, biases);
        if (copy_result.status != LoadStatus::Ok)
            return copy_result;

        return InstantiateCNN(parsed, weights, biases, arena, network, input_shape, input_rank);
    }

    LoadResult Load(const char* nk_path,
                    Arena& arena,
                    NetworkKind& kind,
                    MLPNetwork*& mlp,
                    CNNNetwork*& cnn,
                    std::array<uint32_t, kMaxTensorRank>& input_shape,
                    uint32_t& input_rank)
    {
        mlp = nullptr;
        cnn = nullptr;

        ParsedModel parsed{};
        LoadResult parse_result = ParseFile(nk_path, parsed);
        if (parse_result.status != LoadStatus::Ok)
            return parse_result;

        if (parsed.header.network_kind == NkFormat::NetworkKind::Mlp)
        {
            const LoadResult result = LoadMLP(nk_path, arena, mlp, input_shape, input_rank);
            kind = NetworkKind::Mlp;
            return result;
        }

        const LoadResult result = LoadCNN(nk_path, arena, cnn, input_shape, input_rank);
        kind = NetworkKind::Cnn;
        return result;
    }
}
