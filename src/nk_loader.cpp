#include "nk_loader.hpp"
#include "tensor_factory.hpp"

#include <cstdio>
#include <cstring>
#ifndef NETKIT_DISABLE_IOSTREAM
#include <iostream>
#endif

namespace NkLoader
{
    namespace
    {
        char g_error[256]{};

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

            for (uint32_t i = 0; i < header.num_layers; ++i)
            {
                switch (model.layers[i].kind)
                {
                    case NkFormat::LayerKind::Conv2D:
                    {
                        const NkFormat::ConvLayerDesc& layer = model.layers[i].conv;
                        h = (h + 2 * layer.pad_h - layer.kernel_size) / layer.stride + 1;
                        w = (w + 2 * layer.pad_w - layer.kernel_size) / layer.stride + 1;
                        c = layer.filters;
                        features = h * w * c;
                        break;
                    }
                    case NkFormat::LayerKind::MaxPool2D:
                    case NkFormat::LayerKind::AvgPool2D:
                    {
                        const NkFormat::PoolLayerDesc& layer = model.layers[i].pool;
                        h = (h + 2 * layer.pad_h - layer.pool_size) / layer.stride + 1;
                        w = (w + 2 * layer.pad_w - layer.pool_size) / layer.stride + 1;
                        features = h * w * c;
                        break;
                    }
                    case NkFormat::LayerKind::BatchNorm2d:
                        features = h * w * c;
                        break;
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
                !ReadU8(file, layer.conv.reserved) || !ReadF32(file, layer.conv.alpha))
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
                case NkFormat::LayerKind::MaxPool2D:
                    return ReadPoolLayer(file, layer);
                case NkFormat::LayerKind::AvgPool2D:
                    return ReadAvgPoolLayer(file, layer);
                case NkFormat::LayerKind::BatchNorm2d:
                    return ReadBatchNormLayer(file, layer);
                case NkFormat::LayerKind::Flatten:
                    return ReadFlattenLayer(file, layer);
                default:
                    SetError("Unsupported .nk layer kind");
                    return false;
            }
        }

        bool ReadPayload(std::FILE* file,
                         const ParsedModel& parsed,
                         float* weights,
                         float* biases)
        {
            if (std::fseek(file, static_cast<long>(parsed.payload_offset), SEEK_SET) != 0)
                return false;

            if (!ReadExact(file, weights, parsed.header.weights_bytes))
                return false;

            return ReadExact(file, biases, parsed.header.biases_bytes);
        }

        struct ByteCursor
        {
            const uint8_t* data = nullptr;
            std::size_t size = 0;
            std::size_t pos = 0;
        };

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
                   CursorReadU8(cursor, layer.conv.reserved) && CursorReadF32(cursor, layer.conv.alpha) &&
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
                case NkFormat::LayerKind::MaxPool2D:
                    return ReadPoolLayerCursor(cursor, layer);
                case NkFormat::LayerKind::AvgPool2D:
                    return ReadAvgPoolLayerCursor(cursor, layer);
                case NkFormat::LayerKind::BatchNorm2d:
                    return ReadBatchNormLayerCursor(cursor, layer);
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

        LoadResult CopyPayloadToArena(const ParsedModel& parsed,
                                      const uint8_t* blob,
                                      std::size_t blob_size,
                                      Arena& arena,
                                      float*& weights,
                                      float*& biases)
        {
            const std::size_t weights_bytes = parsed.header.weights_bytes;
            const std::size_t biases_bytes = parsed.header.biases_bytes;
            const std::size_t total_bytes = weights_bytes + biases_bytes;
            const std::size_t needed = parsed.payload_offset + total_bytes;

            if (!blob || blob_size < needed)
                return Fail(LoadStatus::TruncatedFile, ".nk buffer too small for payload");

            float* storage = static_cast<float*>(arena.alloc(total_bytes, alignof(float)));
            if (!storage)
                return Fail(LoadStatus::ArenaOverflow, "Arena out of memory while loading .nk weights");

            weights = storage;
            biases = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(storage) + weights_bytes);
            std::memcpy(weights, blob + parsed.payload_offset, weights_bytes);
            std::memcpy(biases, blob + parsed.payload_offset + weights_bytes, biases_bytes);
            return LoadResult{LoadStatus::Ok, FromNkNetwork(parsed.header.network_kind), nullptr};
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

                        const std::size_t kernel_elems = static_cast<std::size_t>(layer.kernel_size) *
                                                         layer.kernel_size * in_channels;
                        const std::size_t weight_elems = kernel_elems * layer.filters;

                        if (w_desc.num_elements != weight_elems || b_desc.num_elements != layer.filters)
                            return Fail(LoadStatus::SizeMismatch, "CNN conv tensor shape mismatch in .nk catalog");

                        network->InitConvLayer(i,
                                               static_cast<int>(layer.kernel_size),
                                               static_cast<int>(layer.stride),
                                               static_cast<int>(in_channels),
                                               static_cast<int>(layer.filters),
                                               weights + weight_offset,
                                               biases + bias_offset,
                                               ToConvActivation(layer.activation),
                                               layer.alpha,
                                               static_cast<int>(layer.pad_h),
                                               static_cast<int>(layer.pad_w));
                        weight_offset += weight_elems;
                        bias_offset += layer.filters;
                        h = (h + 2 * layer.pad_h - layer.kernel_size) / layer.stride + 1;
                        w = (w + 2 * layer.pad_w - layer.kernel_size) / layer.stride + 1;
                        in_channels = layer.filters;
                        break;
                    }
                    case NkFormat::LayerKind::MaxPool2D:
                    {
                        const NkFormat::PoolLayerDesc& layer = parsed.layers[i].pool;
                        network->InitPoolLayer(i,
                                               static_cast<int>(layer.pool_size),
                                               static_cast<int>(layer.stride),
                                               static_cast<int>(layer.pad_h),
                                               static_cast<int>(layer.pad_w));
                        h = (h + 2 * layer.pad_h - layer.pool_size) / layer.stride + 1;
                        w = (w + 2 * layer.pad_w - layer.pool_size) / layer.stride + 1;
                        break;
                    }
                    case NkFormat::LayerKind::AvgPool2D:
                    {
                        const NkFormat::PoolLayerDesc& layer = parsed.layers[i].pool;
                        network->InitAvgPoolLayer(i,
                                                  static_cast<int>(layer.pool_size),
                                                  static_cast<int>(layer.stride),
                                                  static_cast<int>(layer.pad_h),
                                                  static_cast<int>(layer.pad_w));
                        h = (h + 2 * layer.pad_h - layer.pool_size) / layer.stride + 1;
                        w = (w + 2 * layer.pad_w - layer.pool_size) / layer.stride + 1;
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

        void PrintTensorDesc(const char* label, const NkFormat::TensorDesc& desc)
        {
#ifndef NETKIT_DISABLE_IOSTREAM
            std::cout << "  " << label << ": rank=" << static_cast<uint32_t>(desc.rank)
                      << " dtype=" << NkFormat::DTypeName(desc.dtype) << " shape=[";
            for (uint32_t i = 0; i < desc.rank; ++i)
            {
                std::cout << desc.dims[i];
                if (i + 1 < desc.rank)
                    std::cout << ", ";
            }
            std::cout << "] elements=" << desc.num_elements << "\n";
#else
            (void)label;
            (void)desc;
#endif
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

    bool IsNkPath(const char* path)
    {
        if (!path)
            return false;

        const char* dot = std::strrchr(path, '.');
        if (!dot)
            return false;

        return std::strcmp(dot, ".nk") == 0;
    }

    LoadResult ParseFile(const char* nk_path, ParsedModel& out)
    {
        out = ParsedModel{};

        std::FILE* file = std::fopen(nk_path, "rb");
        if (!file)
            return Fail(LoadStatus::FileOpenFailed, "Could not open .nk file");

        if (!ReadHeader(file, out.header))
        {
            std::fclose(file);
            return Fail(LoadStatus::InvalidMagic, g_error[0] ? g_error : "Failed to read .nk header");
        }

        if (out.header.version != NkFormat::kVersion)
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

        const long payload_start = std::ftell(file);
        if (payload_start < 0)
        {
            std::fclose(file);
            return Fail(LoadStatus::ReadFailed, "Could not seek .nk file");
        }

        out.payload_offset = static_cast<std::size_t>(payload_start);
        if (std::fseek(file, 0, SEEK_END) != 0)
        {
            std::fclose(file);
            return Fail(LoadStatus::ReadFailed, "Could not seek .nk file end");
        }

        const long file_size = std::ftell(file);
        const std::size_t expected_size = out.payload_offset +
                                          static_cast<std::size_t>(out.header.weights_bytes) +
                                          static_cast<std::size_t>(out.header.biases_bytes);

        std::fclose(file);

        if (file_size < 0 || static_cast<std::size_t>(file_size) < expected_size)
            return Fail(LoadStatus::TruncatedFile, ".nk payload size does not match header");

        return ValidateParsedSize(out, static_cast<std::size_t>(file_size));
    }

    LoadResult ParseBuffer(const uint8_t* data, std::size_t size, ParsedModel& out)
    {
        out = ParsedModel{};

        if (!data || size == 0)
            return Fail(LoadStatus::ReadFailed, "Empty .nk buffer");

        ByteCursor cursor{data, size, 0};

        if (!ReadHeaderCursor(cursor, out.header))
            return Fail(LoadStatus::InvalidMagic, g_error[0] ? g_error : "Failed to read .nk header");

        if (out.header.version != NkFormat::kVersion)
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

        out.payload_offset = cursor.pos;
        return ValidateParsedSize(out, size);
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

            if (!ReadExact(file, test_case.input, static_cast<std::size_t>(test_case.input_count) * sizeof(float)))
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
        std::fclose(file);
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

    void PrintHeader(const char* nk_path, const ParsedModel& model)
    {
#ifndef NETKIT_DISABLE_IOSTREAM
        const NkFormat::FileHeader& header = model.header;

        std::cout << "netkit binary model (.nk)\n";
        std::cout << "  file:            " << nk_path << "\n";
        std::cout << "  format version:  " << header.version << "\n";
        std::cout << "  network:         " << NkFormat::NetworkKindName(header.network_kind) << "\n";
        std::cout << "  input rank:      " << static_cast<uint32_t>(header.input_rank) << "\n";
        std::cout << "  input shape:     [";
        for (uint32_t i = 0; i < header.input_rank; ++i)
        {
            std::cout << header.input_shape[i];
            if (i + 1 < header.input_rank)
                std::cout << ", ";
        }
        std::cout << "]\n";
        std::cout << "  layers:          " << header.num_layers << "\n";
        std::cout << "  weight tensors:  " << header.num_weight_tensors << " ("
                  << header.weights_bytes << " bytes)\n";
        std::cout << "  bias tensors:    " << header.num_bias_tensors << " (" << header.biases_bytes
                  << " bytes)\n";

        std::cout << "\nLayer stack:\n";
        for (uint32_t i = 0; i < header.num_layers; ++i)
        {
            const NkFormat::LayerDesc& layer = model.layers[i];
            std::cout << "  [" << i << "] " << NkFormat::LayerKindName(layer.kind);
            switch (layer.kind)
            {
                case NkFormat::LayerKind::Dense:
                    std::cout << " units=" << layer.dense.units
                              << " activation=" << NkFormat::ActivationName(layer.dense.activation);
                    break;
                case NkFormat::LayerKind::Conv2D:
                    std::cout << " kernel=" << layer.conv.kernel_size << " stride=" << layer.conv.stride
                              << " filters=" << layer.conv.filters
                              << " pad=" << static_cast<uint32_t>(layer.conv.pad_h) << ","
                              << static_cast<uint32_t>(layer.conv.pad_w)
                              << " activation=" << NkFormat::ActivationName(layer.conv.activation);
                    break;
                case NkFormat::LayerKind::MaxPool2D:
                    std::cout << " pool=" << layer.pool.pool_size << " stride=" << layer.pool.stride;
                    break;
                case NkFormat::LayerKind::AvgPool2D:
                    std::cout << " pool=" << layer.pool.pool_size << " stride=" << layer.pool.stride;
                    break;
                case NkFormat::LayerKind::BatchNorm2d:
                    std::cout << " channels=" << layer.batch_norm.channels;
                    break;
                case NkFormat::LayerKind::Flatten:
                    break;
            }
            std::cout << "\n";
        }

        if (header.num_weight_tensors > 0)
        {
            std::cout << "\nWeight tensor catalog:\n";
            for (uint32_t i = 0; i < header.num_weight_tensors; ++i)
            {
                char label[32];
                std::snprintf(label, sizeof(label), "weight[%u]", i);
                PrintTensorDesc(label, model.weight_tensors[i]);
            }
        }

        if (header.num_bias_tensors > 0)
        {
            std::cout << "\nBias tensor catalog:\n";
            for (uint32_t i = 0; i < header.num_bias_tensors; ++i)
            {
                char label[32];
                std::snprintf(label, sizeof(label), "bias[%u]", i);
                PrintTensorDesc(label, model.bias_tensors[i]);
            }
        }
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

        ParsedModel parsed{};
        LoadResult parse_result = ParseFile(nk_path, parsed);
        if (parse_result.status != LoadStatus::Ok)
            return parse_result;

        if (parsed.header.network_kind != NkFormat::NetworkKind::Mlp)
            return Fail(LoadStatus::UnsupportedLayer, ".nk file is not an MLP");

        input_rank = parsed.header.input_rank;
        for (uint32_t i = 0; i < input_rank; ++i)
            input_shape[i] = parsed.header.input_shape[i];

        const std::size_t weights_bytes = parsed.header.weights_bytes;
        const std::size_t biases_bytes = parsed.header.biases_bytes;
        const std::size_t total_bytes = weights_bytes + biases_bytes;

        float* storage = static_cast<float*>(arena.alloc(total_bytes, alignof(float)));
        if (!storage)
            return Fail(LoadStatus::ArenaOverflow, "Arena out of memory while loading .nk weights");

        float* weights = storage;
        float* biases = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(storage) + weights_bytes);

        std::FILE* file = std::fopen(nk_path, "rb");
        if (!file)
            return Fail(LoadStatus::FileOpenFailed, "Could not open .nk file");

        if (!ReadPayload(file, parsed, weights, biases))
        {
            std::fclose(file);
            return Fail(LoadStatus::ReadFailed, "Failed to read .nk payload");
        }

        std::fclose(file);

        return InstantiateMLP(parsed, weights, biases, arena, network, input_shape, input_rank);
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
            return Fail(LoadStatus::UnsupportedLayer, ".nk buffer is not an MLP");

        input_rank = parsed.header.input_rank;
        for (uint32_t i = 0; i < input_rank; ++i)
            input_shape[i] = parsed.header.input_shape[i];

        float* weights = nullptr;
        float* biases = nullptr;
        LoadResult copy_result = CopyPayloadToArena(parsed, data, size, arena, weights, biases);
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

        ParsedModel parsed{};
        LoadResult parse_result = ParseFile(nk_path, parsed);
        if (parse_result.status != LoadStatus::Ok)
            return parse_result;

        if (parsed.header.network_kind != NkFormat::NetworkKind::Cnn)
            return Fail(LoadStatus::UnsupportedLayer, ".nk file is not a CNN");

        input_rank = parsed.header.input_rank;
        for (uint32_t i = 0; i < input_rank; ++i)
            input_shape[i] = parsed.header.input_shape[i];

        const std::size_t weights_bytes = parsed.header.weights_bytes;
        const std::size_t biases_bytes = parsed.header.biases_bytes;
        const std::size_t total_bytes = weights_bytes + biases_bytes;

        float* storage = static_cast<float*>(arena.alloc(total_bytes, alignof(float)));
        if (!storage)
            return Fail(LoadStatus::ArenaOverflow, "Arena out of memory while loading .nk weights");

        float* weights = storage;
        float* biases = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(storage) + weights_bytes);

        std::FILE* file = std::fopen(nk_path, "rb");
        if (!file)
            return Fail(LoadStatus::FileOpenFailed, "Could not open .nk file");

        if (!ReadPayload(file, parsed, weights, biases))
        {
            std::fclose(file);
            return Fail(LoadStatus::ReadFailed, "Failed to read .nk payload");
        }

        std::fclose(file);

        return InstantiateCNN(parsed, weights, biases, arena, network, input_shape, input_rank);
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
            return Fail(LoadStatus::UnsupportedLayer, ".nk buffer is not a CNN");

        input_rank = parsed.header.input_rank;
        for (uint32_t i = 0; i < input_rank; ++i)
            input_shape[i] = parsed.header.input_shape[i];

        float* weights = nullptr;
        float* biases = nullptr;
        LoadResult copy_result = CopyPayloadToArena(parsed, data, size, arena, weights, biases);
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

    LoadResult LoadFromBuffer(const uint8_t* data,
                              std::size_t size,
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
        LoadResult parse_result = ParseBuffer(data, size, parsed);
        if (parse_result.status != LoadStatus::Ok)
            return parse_result;

        if (parsed.header.network_kind == NkFormat::NetworkKind::Mlp)
        {
            const LoadResult result = LoadMLPFromBuffer(data, size, arena, mlp, input_shape, input_rank);
            kind = NetworkKind::Mlp;
            return result;
        }

        const LoadResult result = LoadCNNFromBuffer(data, size, arena, cnn, input_shape, input_rank);
        kind = NetworkKind::Cnn;
        return result;
    }
}
