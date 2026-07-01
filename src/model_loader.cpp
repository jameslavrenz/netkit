#include "model_loader.hpp"
#include "json_parser.hpp"
#include "tensor_factory.hpp"
#include <cstdio>
#include <cstring>
#include <iostream>
#include <new>

namespace ModelLoader
{
    namespace
    {
        bool ReadFile(const char* path, char* buffer, std::size_t capacity, std::size_t& bytes_read)
        {
            std::FILE* file = std::fopen(path, "rb");
            if (!file)
                return false;

            bytes_read = std::fread(buffer, 1, capacity, file);
            const bool ok = !std::ferror(file);
            std::fclose(file);
            return ok;
        }

        bool EndsWithLiteral(const char* path, const char* suffix)
        {
            const std::size_t path_len = std::strlen(path);
            const std::size_t suffix_len = std::strlen(suffix);
            if (path_len < suffix_len)
                return false;
            return std::strcmp(path + path_len - suffix_len, suffix) == 0;
        }

        ActivationType ParseMlpActivation(const char* name)
        {
            if (std::strcmp(name, "none") == 0) return ActivationType::None;
            if (std::strcmp(name, "relu") == 0) return ActivationType::ReLU;
            if (std::strcmp(name, "sigmoid") == 0) return ActivationType::Sigmoid;
            if (std::strcmp(name, "tanh") == 0) return ActivationType::Tanh;
            if (std::strcmp(name, "leaky_relu") == 0) return ActivationType::LeakyReLU;
            if (std::strcmp(name, "relu6") == 0) return ActivationType::ReLU6;
            if (std::strcmp(name, "softmax") == 0) return ActivationType::Softmax;
            return ActivationType::None;
        }

        ConvActivationType ParseCnnActivation(const char* name)
        {
            if (std::strcmp(name, "none") == 0) return ConvActivationType::None;
            if (std::strcmp(name, "relu") == 0) return ConvActivationType::ReLU;
            if (std::strcmp(name, "sigmoid") == 0) return ConvActivationType::Sigmoid;
            if (std::strcmp(name, "tanh") == 0) return ConvActivationType::Tanh;
            if (std::strcmp(name, "leaky_relu") == 0) return ConvActivationType::LeakyReLU;
            if (std::strcmp(name, "relu6") == 0) return ConvActivationType::ReLU6;
            if (std::strcmp(name, "softmax") == 0) return ConvActivationType::Softmax;
            return ConvActivationType::None;
        }

        bool ParseInputShape(const char* json, const char* json_end,
                             std::array<uint32_t, kMaxTensorRank>& input_shape,
                             uint32_t& input_rank)
        {
            const char* value = Json::FindKeyValue(json, json_end, "input");
            if (!value || *value != '[')
                return false;

            const char* cursor = value;
            const char* elem_begin = nullptr;
            const char* elem_end = nullptr;
            input_rank = 0;

            while (input_rank < kMaxTensorRank &&
                   Json::NextArrayElement(cursor, json_end, elem_begin, elem_end))
            {
                const char* p = elem_begin;
                if (!Json::ParseUint(p, elem_end, input_shape[input_rank]))
                    return false;
                ++input_rank;
            }

            return input_rank > 0;
        }

        bool ParseNetworkKind(const char* json, const char* json_end, NetworkKind& kind)
        {
            const char* value = Json::FindKeyValue(json, json_end, "network");
            if (!value)
                return false;

            char network_name[Json::kMaxStringLen] = {};
            const char* p = value;
            if (!Json::ParseString(p, json_end, network_name, sizeof(network_name)))
                return false;

            if (std::strcmp(network_name, "mlp") == 0)
            {
                kind = NetworkKind::MLP;
                return true;
            }

            if (std::strcmp(network_name, "cnn") == 0)
            {
                kind = NetworkKind::CNN;
                return true;
            }

            return false;
        }

        bool ParseVersion(const char* json, const char* json_end)
        {
            const char* value = Json::FindKeyValue(json, json_end, "version");
            if (!value)
                return false;

            uint32_t version = 0;
            const char* p = value;
            return Json::ParseUint(p, json_end, version) && version == 1;
        }

        bool ParseLayerStringField(const char* layer_json, const char* layer_end,
                                   const char* key, char* out, std::size_t out_capacity)
        {
            const char* value = Json::FindKeyValue(layer_json, layer_end, key);
            if (!value)
                return false;

            const char* p = value;
            return Json::ParseString(p, layer_end, out, out_capacity);
        }

        bool ParseLayerUintField(const char* layer_json, const char* layer_end,
                                 const char* key, uint32_t& out)
        {
            const char* value = Json::FindKeyValue(layer_json, layer_end, key);
            if (!value)
                return false;

            const char* p = value;
            return Json::ParseUint(p, layer_end, out);
        }

        bool ParseLayerFloatField(const char* layer_json, const char* layer_end,
                                  const char* key, float& out, float default_value)
        {
            const char* value = Json::FindKeyValue(layer_json, layer_end, key);
            if (!value)
            {
                out = default_value;
                return true;
            }

            float parsed = default_value;
            const char* p = value;
            if (!Json::ParseFloat(p, layer_end, parsed))
                return false;

            out = parsed;
            return true;
        }

        LoadResult Fail(LoadStatus status, const char* message, NetworkKind kind = NetworkKind::Unknown)
        {
            return LoadResult{status, kind, message};
        }

        bool ReadJsonFile(const char* json_path, char* json_buffer, std::size_t capacity, std::size_t& json_bytes)
        {
            return ReadFile(json_path, json_buffer, capacity, json_bytes);
        }

        std::size_t ExpectedMlpWeightFloats(const ArchitectureSpec& spec)
        {
            std::size_t expected = 0;
            uint32_t in_features = spec.input_shape[1];

            for (uint32_t i = 0; i < spec.num_layers; ++i)
            {
                const uint32_t out_features = spec.dense_layers[i].units;
                expected += static_cast<std::size_t>(in_features) * out_features;
                expected += out_features;
                in_features = out_features;
            }

            return expected;
        }

        std::size_t ExpectedCnnWeightFloats(const ArchitectureSpec& spec)
        {
            std::size_t expected = 0;
            uint32_t h = spec.input_shape[0];
            uint32_t w = spec.input_shape[1];
            uint32_t channels = spec.input_shape[2];
            uint32_t dense_in = 0;
            bool spatial = true;

            for (uint32_t i = 0; i < spec.num_layers; ++i)
            {
                switch (spec.cnn_layer_kinds[i])
                {
                    case CnnLayerKind::Conv2D:
                    {
                        const ConvLayerConfig& layer = spec.conv_layers[i];
                        const std::size_t kernel_elems = static_cast<std::size_t>(layer.kernel_size) *
                                                         layer.kernel_size *
                                                         channels;
                        expected += kernel_elems * layer.filters;
                        expected += layer.filters;
                        h = (h - layer.kernel_size) / layer.stride + 1;
                        w = (w - layer.kernel_size) / layer.stride + 1;
                        channels = layer.filters;
                        break;
                    }
                    case CnnLayerKind::MaxPool2D:
                    {
                        const PoolLayerConfig& layer = spec.pool_layers[i];
                        h = (h - layer.pool_size) / layer.stride + 1;
                        w = (w - layer.pool_size) / layer.stride + 1;
                        break;
                    }
                    case CnnLayerKind::Flatten:
                        dense_in = h * w * channels;
                        spatial = false;
                        break;
                    case CnnLayerKind::Dense:
                    {
                        const DenseLayerConfig& layer = spec.cnn_dense_layers[i];
                        expected += static_cast<std::size_t>(dense_in) * layer.units;
                        expected += layer.units;
                        dense_in = layer.units;
                        break;
                    }
                }

                (void)spatial;
            }

            return expected;
        }

        LoadResult ParseMlpLayers(const char* json, const char* json_end, ArchitectureSpec& spec)
        {
            const char* layers_value = Json::FindKeyValue(json, json_end, "layers");
            if (!layers_value || *layers_value != '[')
                return Fail(LoadStatus::JsonParseFailed, "Missing layers array in JSON", NetworkKind::MLP);

            const char* cursor = layers_value;
            const char* elem_begin = nullptr;
            const char* elem_end = nullptr;

            while (spec.num_layers < kMaxLayers &&
                   Json::NextArrayElement(cursor, json_end, elem_begin, elem_end))
            {
                char layer_type[Json::kMaxStringLen] = {};
                if (!ParseLayerStringField(elem_begin, elem_end, "type", layer_type, sizeof(layer_type)))
                    return Fail(LoadStatus::LayerConfigError, "Layer missing type field", NetworkKind::MLP);

                if (std::strcmp(layer_type, "dense") != 0)
                    return Fail(LoadStatus::LayerConfigError, "MLP layers must have type dense", NetworkKind::MLP);

                DenseLayerConfig& layer = spec.dense_layers[spec.num_layers];
                if (!ParseLayerUintField(elem_begin, elem_end, "units", layer.units) || layer.units == 0)
                    return Fail(LoadStatus::LayerConfigError, "Dense layer missing units", NetworkKind::MLP);

                ParseLayerStringField(elem_begin, elem_end, "activation", layer.activation, sizeof(layer.activation));
                ParseLayerFloatField(elem_begin, elem_end, "alpha", layer.alpha, 0.01f);
                ++spec.num_layers;
            }

            if (spec.num_layers == 0)
                return Fail(LoadStatus::LayerConfigError, "MLP requires at least one layer", NetworkKind::MLP);

            spec.expected_weight_floats = ExpectedMlpWeightFloats(spec);
            return LoadResult{LoadStatus::Ok, NetworkKind::MLP, nullptr};
        }

        LoadResult ParseCnnLayers(const char* json, const char* json_end, ArchitectureSpec& spec)
        {
            const char* layers_value = Json::FindKeyValue(json, json_end, "layers");
            if (!layers_value || *layers_value != '[')
                return Fail(LoadStatus::JsonParseFailed, "Missing layers array in JSON", NetworkKind::CNN);

            const char* cursor = layers_value;
            const char* elem_begin = nullptr;
            const char* elem_end = nullptr;

            while (spec.num_layers < kMaxLayers &&
                   Json::NextArrayElement(cursor, json_end, elem_begin, elem_end))
            {
                char layer_type[Json::kMaxStringLen] = {};
                if (!ParseLayerStringField(elem_begin, elem_end, "type", layer_type, sizeof(layer_type)))
                    return Fail(LoadStatus::LayerConfigError, "Layer missing type field", NetworkKind::CNN);

                if (std::strcmp(layer_type, "conv2d") == 0)
                {
                    ConvLayerConfig& layer = spec.conv_layers[spec.num_layers];
                    spec.cnn_layer_kinds[spec.num_layers] = CnnLayerKind::Conv2D;

                    if (!ParseLayerUintField(elem_begin, elem_end, "kernel_size", layer.kernel_size) ||
                        layer.kernel_size == 0)
                        return Fail(LoadStatus::LayerConfigError, "Conv2D layer missing kernel_size", NetworkKind::CNN);

                    if (!ParseLayerUintField(elem_begin, elem_end, "filters", layer.filters) || layer.filters == 0)
                        return Fail(LoadStatus::LayerConfigError, "Conv2D layer missing filters", NetworkKind::CNN);

                    ParseLayerUintField(elem_begin, elem_end, "stride", layer.stride);
                    if (layer.stride == 0)
                        layer.stride = 1;

                    ParseLayerStringField(elem_begin, elem_end, "activation", layer.activation, sizeof(layer.activation));
                    ParseLayerFloatField(elem_begin, elem_end, "alpha", layer.alpha, 0.01f);
                }
                else if (std::strcmp(layer_type, "max_pool2d") == 0)
                {
                    PoolLayerConfig& layer = spec.pool_layers[spec.num_layers];
                    spec.cnn_layer_kinds[spec.num_layers] = CnnLayerKind::MaxPool2D;

                    ParseLayerUintField(elem_begin, elem_end, "pool_size", layer.pool_size);
                    if (layer.pool_size == 0)
                        layer.pool_size = 2;

                    ParseLayerUintField(elem_begin, elem_end, "stride", layer.stride);
                    if (layer.stride == 0)
                        layer.stride = layer.pool_size;
                }
                else if (std::strcmp(layer_type, "flatten") == 0)
                {
                    spec.cnn_layer_kinds[spec.num_layers] = CnnLayerKind::Flatten;
                }
                else if (std::strcmp(layer_type, "dense") == 0)
                {
                    DenseLayerConfig& layer = spec.cnn_dense_layers[spec.num_layers];
                    spec.cnn_layer_kinds[spec.num_layers] = CnnLayerKind::Dense;

                    if (!ParseLayerUintField(elem_begin, elem_end, "units", layer.units) || layer.units == 0)
                        return Fail(LoadStatus::LayerConfigError, "Dense layer missing units", NetworkKind::CNN);

                    ParseLayerStringField(elem_begin, elem_end, "activation", layer.activation, sizeof(layer.activation));
                    ParseLayerFloatField(elem_begin, elem_end, "alpha", layer.alpha, 0.01f);
                }
                else
                {
                    return Fail(LoadStatus::LayerConfigError, "Unsupported CNN layer type", NetworkKind::CNN);
                }

                ++spec.num_layers;
            }

            if (spec.num_layers == 0)
                return Fail(LoadStatus::LayerConfigError, "CNN requires at least one layer", NetworkKind::CNN);

            spec.expected_weight_floats = ExpectedCnnWeightFloats(spec);
            return LoadResult{LoadStatus::Ok, NetworkKind::CNN, nullptr};
        }

        const char* NetworkKindName(NetworkKind kind)
        {
            switch (kind)
            {
                case NetworkKind::MLP: return "mlp";
                case NetworkKind::CNN: return "cnn";
                default: return "unknown";
            }
        }

        const char* NetworkTypeLabel(NetworkKind kind)
        {
            switch (kind)
            {
                case NetworkKind::MLP: return "MLP";
                case NetworkKind::CNN: return "CNN";
                default: return "Unknown";
            }
        }

        bool ModelNameFromPath(const char* json_path, char* name, std::size_t name_capacity)
        {
            if (!json_path || !name || name_capacity == 0)
                return false;

            const char* base = json_path;
            for (const char* cursor = json_path; *cursor; ++cursor)
            {
                if (*cursor == '/' || *cursor == '\\')
                    base = cursor + 1;
            }

            std::strncpy(name, base, name_capacity - 1);
            name[name_capacity - 1] = '\0';

            const std::size_t name_len = std::strlen(name);
            if (name_len >= 5 && std::strcmp(name + name_len - 5, ".json") == 0)
                name[name_len - 5] = '\0';

            return name[0] != '\0';
        }

        void FormatActivationDisplay(const char* activation, char* out, std::size_t out_capacity)
        {
            if (!out || out_capacity == 0)
                return;

            out[0] = '\0';
            if (!activation || std::strcmp(activation, "none") == 0)
                return;

            if (std::strcmp(activation, "relu") == 0)
                std::strncpy(out, "ReLU", out_capacity - 1);
            else if (std::strcmp(activation, "leaky_relu") == 0)
                std::strncpy(out, "LeakyReLU", out_capacity - 1);
            else if (std::strcmp(activation, "relu6") == 0)
                std::strncpy(out, "ReLU6", out_capacity - 1);
            else if (std::strcmp(activation, "sigmoid") == 0)
                std::strncpy(out, "Sigmoid", out_capacity - 1);
            else if (std::strcmp(activation, "tanh") == 0)
                std::strncpy(out, "Tanh", out_capacity - 1);
            else if (std::strcmp(activation, "softmax") == 0)
                std::strncpy(out, "Softmax", out_capacity - 1);
            else
                std::strncpy(out, activation, out_capacity - 1);

            out[out_capacity - 1] = '\0';
        }

        void PrintInputShapeSummary(const ArchitectureSpec& spec)
        {
            for (uint32_t i = 0; i < spec.input_rank; ++i)
            {
                if (i > 0)
                    std::cout << " x ";
                std::cout << spec.input_shape[i];
            }
        }

        void PrintLayerIndex(uint32_t index)
        {
            if (index < 10)
                std::cout << ' ';
            std::cout << index << ' ';
        }

        void PrintActivationSuffix(const char* activation)
        {
            char label[Json::kMaxStringLen] = {};
            FormatActivationDisplay(activation, label, sizeof(label));
            if (label[0] != '\0')
                std::cout << "  " << label;
        }

        void PrintConvLayerSummary(const ConvLayerConfig& layer)
        {
            std::cout << " Conv2D      " << layer.filters << " filters   "
                      << layer.kernel_size << 'x' << layer.kernel_size
                      << "  stride=" << layer.stride;
            PrintActivationSuffix(layer.activation);
            std::cout << "\n";
        }

        void PrintPoolLayerSummary(const PoolLayerConfig& layer)
        {
            std::cout << " MaxPool2D   " << layer.pool_size << 'x' << layer.pool_size
                      << "           stride=" << layer.stride << "\n";
        }

        void PrintDenseLayerSummary(const DenseLayerConfig& layer)
        {
            std::cout << " Dense       " << layer.units << " units";
            PrintActivationSuffix(layer.activation);
            std::cout << "\n";
        }

        void PrintFlattenLayerSummary()
        {
            std::cout << " Flatten\n";
        }
    }

    uint32_t ComputeMlpOutputElements(const ArchitectureSpec& spec)
    {
        if (spec.num_layers == 0)
            return 0;
        return spec.input_shape[0] * spec.dense_layers[spec.num_layers - 1].units;
    }

    uint32_t ComputeCnnOutputElements(const ArchitectureSpec& spec)
    {
        uint32_t h = spec.input_shape[0];
        uint32_t w = spec.input_shape[1];
        uint32_t c = spec.input_shape[2];
        uint32_t features = h * w * c;
        bool flattened = false;

        for (uint32_t i = 0; i < spec.num_layers; ++i)
        {
            switch (spec.cnn_layer_kinds[i])
            {
                case CnnLayerKind::Conv2D:
                {
                    const ConvLayerConfig& layer = spec.conv_layers[i];
                    h = (h - layer.kernel_size) / layer.stride + 1;
                    w = (w - layer.kernel_size) / layer.stride + 1;
                    c = layer.filters;
                    break;
                }
                case CnnLayerKind::MaxPool2D:
                {
                    const PoolLayerConfig& layer = spec.pool_layers[i];
                    h = (h - layer.pool_size) / layer.stride + 1;
                    w = (w - layer.pool_size) / layer.stride + 1;
                    break;
                }
                case CnnLayerKind::Flatten:
                    features = h * w * c;
                    flattened = true;
                    break;
                case CnnLayerKind::Dense:
                    features = spec.cnn_dense_layers[i].units;
                    flattened = true;
                    break;
            }
        }

        if (flattened)
            return features;
        return h * w * c;
    }

    LoadResult ParseArchitecture(const char* json_path, ArchitectureSpec& spec)
    {
        spec = {};

        char json_buffer[kMaxJsonBytes] = {};
        std::size_t json_bytes = 0;
        if (!ReadJsonFile(json_path, json_buffer, sizeof(json_buffer) - 1, json_bytes))
            return Fail(LoadStatus::JsonOpenFailed, "Failed to open JSON architecture file");

        const char* json = json_buffer;
        const char* json_end = json_buffer + json_bytes;

        const char* version_value = Json::FindKeyValue(json, json_end, "version");
        if (!version_value)
            return Fail(LoadStatus::VersionMismatch, "Unsupported or missing JSON version (expected 1)");

        const char* version_cursor = version_value;
        if (!Json::ParseUint(version_cursor, json_end, spec.version) || spec.version != 1)
            return Fail(LoadStatus::VersionMismatch, "Unsupported or missing JSON version (expected 1)");

        if (!ParseNetworkKind(json, json_end, spec.kind))
            return Fail(LoadStatus::UnsupportedNetwork, "Missing or unsupported network field");

        if (!ParseInputShape(json, json_end, spec.input_shape, spec.input_rank))
            return Fail(LoadStatus::JsonParseFailed, "Missing or invalid input shape");

        if (spec.kind == NetworkKind::MLP)
        {
            if (spec.input_rank != 2)
                return Fail(LoadStatus::LayerConfigError, "MLP input must be a 2D shape [batch, features]", NetworkKind::MLP);
            return ParseMlpLayers(json, json_end, spec);
        }

        if (spec.kind == NetworkKind::CNN)
        {
            if (spec.input_rank != 3)
                return Fail(LoadStatus::LayerConfigError, "CNN input must be a 3D shape [H, W, C]", NetworkKind::CNN);
            return ParseCnnLayers(json, json_end, spec);
        }

        return Fail(LoadStatus::UnsupportedNetwork, "Unknown network type");
    }

    void PrintArchitecture(const ArchitectureSpec& spec)
    {
        std::cout << "  version:  " << spec.version << "\n";
        std::cout << "  network:  " << NetworkKindName(spec.kind) << "\n";
        std::cout << "  input:    [";
        for (uint32_t i = 0; i < spec.input_rank; ++i)
        {
            std::cout << spec.input_shape[i];
            if (i + 1 < spec.input_rank)
                std::cout << ", ";
        }
        std::cout << "]\n";
        std::cout << "  layers:   " << spec.num_layers << "\n";
        std::cout << "  expected weight floats: " << spec.expected_weight_floats << "\n";

        for (uint32_t i = 0; i < spec.num_layers; ++i)
        {
            std::cout << "    [" << i << "] ";

            if (spec.kind == NetworkKind::MLP)
            {
                const DenseLayerConfig& layer = spec.dense_layers[i];
                std::cout << "dense"
                          << "  units=" << layer.units
                          << "  activation=" << layer.activation;
                if (std::strcmp(layer.activation, "leaky_relu") == 0)
                    std::cout << "  alpha=" << layer.alpha;
            }
            else if (spec.kind == NetworkKind::CNN)
            {
                switch (spec.cnn_layer_kinds[i])
                {
                    case CnnLayerKind::Conv2D:
                    {
                        const ConvLayerConfig& layer = spec.conv_layers[i];
                        std::cout << "conv2d"
                                  << "  kernel_size=" << layer.kernel_size
                                  << "  stride=" << layer.stride
                                  << "  filters=" << layer.filters
                                  << "  activation=" << layer.activation;
                        if (std::strcmp(layer.activation, "leaky_relu") == 0)
                            std::cout << "  alpha=" << layer.alpha;
                        break;
                    }
                    case CnnLayerKind::MaxPool2D:
                    {
                        const PoolLayerConfig& layer = spec.pool_layers[i];
                        std::cout << "max_pool2d"
                                  << "  pool_size=" << layer.pool_size
                                  << "  stride=" << layer.stride;
                        break;
                    }
                    case CnnLayerKind::Flatten:
                        std::cout << "flatten";
                        break;
                    case CnnLayerKind::Dense:
                    {
                        const DenseLayerConfig& layer = spec.cnn_dense_layers[i];
                        std::cout << "dense"
                                  << "  units=" << layer.units
                                  << "  activation=" << layer.activation;
                        if (std::strcmp(layer.activation, "leaky_relu") == 0)
                            std::cout << "  alpha=" << layer.alpha;
                        break;
                    }
                }
            }

            std::cout << "\n";
        }
    }

    void PrintNetworkSummary(const char* json_path, const ArchitectureSpec& spec)
    {
        char name[kMaxPathLen] = {"model"};
        if (json_path)
            ModelNameFromPath(json_path, name, sizeof(name));

        std::cout << "=====================================================\n";
        std::cout << "Network Summary\n";
        std::cout << "=====================================================\n\n";
        std::cout << "Name        : " << name << "\n";
        std::cout << "Type        : " << NetworkTypeLabel(spec.kind) << "\n";
        std::cout << "Version     : " << spec.version << "\n\n";
        std::cout << "Input Shape : ";
        PrintInputShapeSummary(spec);
        std::cout << "\n\n";
        std::cout << "Layers (" << spec.num_layers << ")\n";
        std::cout << "-----------------------------------------------------\n";

        for (uint32_t i = 0; i < spec.num_layers; ++i)
        {
            PrintLayerIndex(i);

            if (spec.kind == NetworkKind::MLP)
                PrintDenseLayerSummary(spec.dense_layers[i]);
            else if (spec.kind == NetworkKind::CNN)
            {
                switch (spec.cnn_layer_kinds[i])
                {
                    case CnnLayerKind::Conv2D:
                        PrintConvLayerSummary(spec.conv_layers[i]);
                        break;
                    case CnnLayerKind::MaxPool2D:
                        PrintPoolLayerSummary(spec.pool_layers[i]);
                        break;
                    case CnnLayerKind::Flatten:
                        PrintFlattenLayerSummary();
                        break;
                    case CnnLayerKind::Dense:
                        PrintDenseLayerSummary(spec.cnn_dense_layers[i]);
                        break;
                }
            }
        }

        std::cout << "\n=====================================================\n";
    }

    void PrintWeightsSummary(const char* json_path,
                             float* weights,
                             std::size_t float_count,
                             std::size_t expected_float_count)
    {
        char bin_path[kMaxPathLen] = {};
        JsonPathToBinPath(json_path, bin_path, sizeof(bin_path));

        std::cout << "  weights file: " << bin_path << "\n";
        std::cout << "  floats loaded: " << float_count << "\n";
        std::cout << "  bytes loaded:  " << (float_count * sizeof(float)) << "\n";
        std::cout << "  expected floats: " << expected_float_count;

        if (float_count == expected_float_count)
            std::cout << " (match)\n";
        else
            std::cout << " (MISMATCH)\n";

        if (weights && float_count > 0)
        {
            std::cout << "  first values: ";
            const std::size_t preview = float_count < 8 ? float_count : 8;
            for (std::size_t i = 0; i < preview; ++i)
                std::cout << weights[i] << " ";
            if (float_count > preview)
                std::cout << "...";
            std::cout << "\n";
        }

        std::cout << "  (weights loaded into arena; not applied to network yet)\n";
    }

    bool JsonPathToBinPath(const char* json_path, char* bin_path, std::size_t bin_path_capacity)
    {
        if (!json_path || !bin_path || bin_path_capacity == 0)
            return false;

        const std::size_t path_len = std::strlen(json_path);
        if (path_len + 1 > bin_path_capacity)
            return false;

        std::strncpy(bin_path, json_path, bin_path_capacity - 1);
        bin_path[bin_path_capacity - 1] = '\0';

        if (EndsWithLiteral(bin_path, ".json"))
        {
            const std::size_t base_len = std::strlen(bin_path) - 5;
            std::strcpy(bin_path + base_len, ".bin");
            return true;
        }

        if (path_len + 4 >= bin_path_capacity)
            return false;

        std::strcat(bin_path, ".bin");
        return true;
    }

    LoadStatus LoadWeightsBin(const char* json_path,
                              Arena& arena,
                              float*& weights,
                              std::size_t& float_count,
                              const char** error_message)
    {
        char bin_path[kMaxPathLen] = {};
        if (!JsonPathToBinPath(json_path, bin_path, sizeof(bin_path)))
        {
            if (error_message)
                *error_message = "Failed to derive .bin path from JSON path";
            return LoadStatus::BinOpenFailed;
        }

        std::FILE* file = std::fopen(bin_path, "rb");
        if (!file)
        {
            if (error_message)
                *error_message = "Failed to open companion .bin weights file";
            return LoadStatus::BinOpenFailed;
        }

        if (std::fseek(file, 0, SEEK_END) != 0)
        {
            std::fclose(file);
            if (error_message)
                *error_message = "Failed to seek .bin weights file";
            return LoadStatus::BinOpenFailed;
        }

        const long file_size = std::ftell(file);
        if (file_size < 0 || (file_size % static_cast<long>(sizeof(float))) != 0)
        {
            std::fclose(file);
            if (error_message)
                *error_message = ".bin file size is not a multiple of float32";
            return LoadStatus::BinSizeMismatch;
        }

        if (std::fseek(file, 0, SEEK_SET) != 0)
        {
            std::fclose(file);
            if (error_message)
                *error_message = "Failed to rewind .bin weights file";
            return LoadStatus::BinOpenFailed;
        }

        float_count = static_cast<std::size_t>(file_size / sizeof(float));
        void* buffer = arena.alloc(file_size, alignof(float));
        if (!buffer)
        {
            std::fclose(file);
            if (error_message)
                *error_message = "Arena out of memory while loading .bin weights";
            return LoadStatus::ArenaOverflow;
        }

        const std::size_t bytes_read = std::fread(buffer, 1, static_cast<std::size_t>(file_size), file);
        std::fclose(file);

        if (bytes_read != static_cast<std::size_t>(file_size))
        {
            if (error_message)
                *error_message = "Failed to read complete .bin weights file";
            return LoadStatus::BinOpenFailed;
        }

        weights = static_cast<float*>(buffer);
        return LoadStatus::Ok;
    }

    LoadResult LoadMLP(const char* json_path,
                       Arena& arena,
                       MLPNetwork*& network,
                       std::array<uint32_t, kMaxTensorRank>& input_shape,
                       uint32_t& input_rank)
    {
        network = nullptr;

        char json_buffer[kMaxJsonBytes] = {};
        std::size_t json_bytes = 0;
        if (!ReadFile(json_path, json_buffer, sizeof(json_buffer) - 1, json_bytes))
            return Fail(LoadStatus::JsonOpenFailed, "Failed to open JSON architecture file");

        const char* json = json_buffer;
        const char* json_end = json_buffer + json_bytes;

        if (!ParseVersion(json, json_end))
            return Fail(LoadStatus::VersionMismatch, "Unsupported or missing JSON version (expected 1)");

        NetworkKind kind = NetworkKind::Unknown;
        if (!ParseNetworkKind(json, json_end, kind) || kind != NetworkKind::MLP)
            return Fail(LoadStatus::UnsupportedNetwork, "JSON network type is not mlp", kind);

        if (!ParseInputShape(json, json_end, input_shape, input_rank) || input_rank != 2)
            return Fail(LoadStatus::LayerConfigError, "MLP input must be a 2D shape [batch, features]", NetworkKind::MLP);

        const char* layers_value = Json::FindKeyValue(json, json_end, "layers");
        if (!layers_value || *layers_value != '[')
            return Fail(LoadStatus::JsonParseFailed, "Missing layers array in JSON", NetworkKind::MLP);

        const char* cursor = layers_value;
        const char* elem_begin = nullptr;
        const char* elem_end = nullptr;

        struct DenseLayerSpec
        {
            uint32_t units = 0;
            ActivationType activation = ActivationType::None;
            float leaky_alpha = 0.01f;
        };

        DenseLayerSpec layer_specs[kMaxLayers] = {};
        uint32_t num_layers = 0;

        while (num_layers < kMaxLayers &&
               Json::NextArrayElement(cursor, json_end, elem_begin, elem_end))
        {
            char layer_type[Json::kMaxStringLen] = {};
            if (!ParseLayerStringField(elem_begin, elem_end, "type", layer_type, sizeof(layer_type)))
                return Fail(LoadStatus::LayerConfigError, "Layer missing type field", NetworkKind::MLP);

            if (std::strcmp(layer_type, "dense") != 0)
                return Fail(LoadStatus::LayerConfigError, "MLP layers must have type dense", NetworkKind::MLP);

            DenseLayerSpec& spec = layer_specs[num_layers];
            if (!ParseLayerUintField(elem_begin, elem_end, "units", spec.units) || spec.units == 0)
                return Fail(LoadStatus::LayerConfigError, "Dense layer missing units", NetworkKind::MLP);

            char activation_name[Json::kMaxStringLen] = {"none"};
            ParseLayerStringField(elem_begin, elem_end, "activation", activation_name, sizeof(activation_name));
            spec.activation = ParseMlpActivation(activation_name);
            ParseLayerFloatField(elem_begin, elem_end, "alpha", spec.leaky_alpha, 0.01f);

            ++num_layers;
        }

        if (num_layers == 0)
            return Fail(LoadStatus::LayerConfigError, "MLP requires at least one layer", NetworkKind::MLP);

        float* weights = nullptr;
        std::size_t float_count = 0;
        const char* bin_error = nullptr;
        const LoadStatus bin_status = LoadWeightsBin(json_path, arena, weights, float_count, &bin_error);
        if (bin_status != LoadStatus::Ok)
            return Fail(bin_status, bin_error, NetworkKind::MLP);

        std::size_t expected_floats = 0;
        uint32_t in_features = input_shape[1];
        for (uint32_t i = 0; i < num_layers; ++i)
        {
            const uint32_t out_features = layer_specs[i].units;
            expected_floats += static_cast<std::size_t>(in_features) * out_features;
            expected_floats += out_features;
            in_features = out_features;
        }

        if (float_count != expected_floats)
            return Fail(LoadStatus::BinSizeMismatch, "MLP .bin float count does not match architecture", NetworkKind::MLP);

        void* network_mem = arena.alloc(sizeof(MLPNetwork), alignof(MLPNetwork));
        if (!network_mem)
            return Fail(LoadStatus::ArenaOverflow, "Arena out of memory while creating MLPNetwork", NetworkKind::MLP);

        network = new (network_mem) MLPNetwork(num_layers, arena);
        if (!network->IsValid())
            return Fail(LoadStatus::ArenaOverflow, "Arena out of memory while allocating MLP layers", NetworkKind::MLP);

        std::size_t weight_offset = 0;
        in_features = input_shape[1];
        for (uint32_t i = 0; i < num_layers; ++i)
        {
            const uint32_t out_features = layer_specs[i].units;
            const std::size_t weight_elems = static_cast<std::size_t>(in_features) * out_features;
            const std::size_t bias_elems = out_features;

            Tensor W = TensorFactory::View2D(weights + weight_offset, in_features, out_features);
            weight_offset += weight_elems;

            Tensor B = TensorFactory::View2D(weights + weight_offset, 1, out_features);
            weight_offset += bias_elems;

            network->InitLayer(i, W, B, layer_specs[i].activation, layer_specs[i].leaky_alpha);
            in_features = out_features;
        }

        if (!network->InitActivationBuffers(arena, input_shape[0]))
            return Fail(LoadStatus::ArenaOverflow, "Arena out of memory while allocating MLP activation buffers",
                        NetworkKind::MLP);

        if (!network->HasActivationBuffers())
            return Fail(LoadStatus::ArenaOverflow, "MLP activation buffers not ready", NetworkKind::MLP);

        return LoadResult{LoadStatus::Ok, NetworkKind::MLP, nullptr};
    }

    LoadResult LoadCNN(const char* json_path,
                       Arena& arena,
                       CNNNetwork*& network,
                       std::array<uint32_t, kMaxTensorRank>& input_shape,
                       uint32_t& input_rank)
    {
        network = nullptr;

        ArchitectureSpec spec{};
        const LoadResult parse_result = ParseArchitecture(json_path, spec);
        if (parse_result.status != LoadStatus::Ok)
            return parse_result;

        if (spec.kind != NetworkKind::CNN)
            return Fail(LoadStatus::UnsupportedNetwork, "JSON network type is not cnn", spec.kind);

        input_shape = spec.input_shape;
        input_rank = spec.input_rank;

        float* weights = nullptr;
        std::size_t float_count = 0;
        const char* bin_error = nullptr;
        const LoadStatus bin_status = LoadWeightsBin(json_path, arena, weights, float_count, &bin_error);
        if (bin_status != LoadStatus::Ok)
            return Fail(bin_status, bin_error, NetworkKind::CNN);

        if (float_count != spec.expected_weight_floats)
            return Fail(LoadStatus::BinSizeMismatch, "CNN .bin float count does not match architecture", NetworkKind::CNN);

        void* network_mem = arena.alloc(sizeof(CNNNetwork), alignof(CNNNetwork));
        if (!network_mem)
            return Fail(LoadStatus::ArenaOverflow, "Arena out of memory while creating CNNNetwork", NetworkKind::CNN);

        network = new (network_mem) CNNNetwork(spec.num_layers, arena);
        if (!network->IsValid())
            return Fail(LoadStatus::ArenaOverflow, "Arena out of memory while allocating CNN layers", NetworkKind::CNN);

        std::size_t weight_offset = 0;
        uint32_t in_channels = input_shape[2];
        uint32_t h = input_shape[0];
        uint32_t w = input_shape[1];
        uint32_t dense_in = 0;

        for (uint32_t i = 0; i < spec.num_layers; ++i)
        {
            switch (spec.cnn_layer_kinds[i])
            {
                case CnnLayerKind::Conv2D:
                {
                    const ConvLayerConfig& layer_spec = spec.conv_layers[i];
                    const std::size_t kernel_elems = static_cast<std::size_t>(layer_spec.kernel_size) *
                                                     layer_spec.kernel_size *
                                                     in_channels;
                    const std::size_t weight_elems = kernel_elems * layer_spec.filters;

                    float* layer_weights = weights + weight_offset;
                    weight_offset += weight_elems;

                    float* layer_bias = weights + weight_offset;
                    weight_offset += layer_spec.filters;

                    network->InitConvLayer(i,
                                           static_cast<int>(layer_spec.kernel_size),
                                           static_cast<int>(layer_spec.stride),
                                           static_cast<int>(in_channels),
                                           static_cast<int>(layer_spec.filters),
                                           layer_weights,
                                           layer_bias,
                                           ParseCnnActivation(layer_spec.activation),
                                           layer_spec.alpha);
                    h = (h - layer_spec.kernel_size) / layer_spec.stride + 1;
                    w = (w - layer_spec.kernel_size) / layer_spec.stride + 1;
                    in_channels = layer_spec.filters;
                    break;
                }
                case CnnLayerKind::MaxPool2D:
                {
                    const PoolLayerConfig& layer_spec = spec.pool_layers[i];
                    network->InitPoolLayer(i,
                                           static_cast<int>(layer_spec.pool_size),
                                           static_cast<int>(layer_spec.stride));
                    h = (h - layer_spec.pool_size) / layer_spec.stride + 1;
                    w = (w - layer_spec.pool_size) / layer_spec.stride + 1;
                    break;
                }
                case CnnLayerKind::Flatten:
                    dense_in = h * w * in_channels;
                    network->InitFlattenLayer(i);
                    break;
                case CnnLayerKind::Dense:
                {
                    const DenseLayerConfig& layer_spec = spec.cnn_dense_layers[i];
                    const uint32_t out_features = layer_spec.units;
                    const std::size_t weight_elems = static_cast<std::size_t>(dense_in) * out_features;
                    const std::size_t bias_elems = out_features;

                    Tensor W = TensorFactory::View2D(weights + weight_offset, dense_in, out_features);
                    weight_offset += weight_elems;

                    Tensor B = TensorFactory::View2D(weights + weight_offset, 1, out_features);
                    weight_offset += bias_elems;

                    network->InitDenseLayer(i, W, B, ParseMlpActivation(layer_spec.activation), layer_spec.alpha);
                    dense_in = out_features;
                    break;
                }
            }
        }

        if (!network->InitActivationBuffers(arena, input_shape[0], input_shape[1], input_shape[2]))
            return Fail(LoadStatus::ArenaOverflow, "Arena out of memory while allocating CNN activation buffers",
                        NetworkKind::CNN);

        if (!network->HasActivationBuffers())
            return Fail(LoadStatus::ArenaOverflow, "CNN activation buffers not ready", NetworkKind::CNN);

        return LoadResult{LoadStatus::Ok, NetworkKind::CNN, nullptr};
    }

    LoadResult Load(const char* json_path,
                    Arena& arena,
                    NetworkKind& kind,
                    MLPNetwork*& mlp,
                    CNNNetwork*& cnn,
                    std::array<uint32_t, kMaxTensorRank>& input_shape,
                    uint32_t& input_rank)
    {
        mlp = nullptr;
        cnn = nullptr;
        kind = NetworkKind::Unknown;

        char json_buffer[kMaxJsonBytes] = {};
        std::size_t json_bytes = 0;
        if (!ReadFile(json_path, json_buffer, sizeof(json_buffer) - 1, json_bytes))
            return Fail(LoadStatus::JsonOpenFailed, "Failed to open JSON architecture file");

        const char* json = json_buffer;
        const char* json_end = json_buffer + json_bytes;

        if (!ParseVersion(json, json_end))
            return Fail(LoadStatus::VersionMismatch, "Unsupported or missing JSON version (expected 1)");

        if (!ParseNetworkKind(json, json_end, kind))
            return Fail(LoadStatus::UnsupportedNetwork, "Missing or unsupported network field");

        if (kind == NetworkKind::MLP)
            return LoadMLP(json_path, arena, mlp, input_shape, input_rank);

        if (kind == NetworkKind::CNN)
            return LoadCNN(json_path, arena, cnn, input_shape, input_rank);

        return Fail(LoadStatus::UnsupportedNetwork, "Unknown network type");
    }
}
