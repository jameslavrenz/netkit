#include "vectors_loader.hpp"
#include "json_parser.hpp"
#include "model_loader.hpp"
#include "tensor_factory.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>

namespace VectorsLoader
{
    namespace
    {
        alignas(std::max_align_t) unsigned char g_arena_buffer[Arena::kDefaultCapacity];

        bool FileReadable(const char* path)
        {
            std::FILE* file = std::fopen(path, "rb");
            if (!file)
                return false;
            std::fclose(file);
            return true;
        }

        const char* ResolveModelPath(const char* rel_path, char* buffer, std::size_t buffer_size)
        {
            if (FileReadable(rel_path))
                return rel_path;

            std::snprintf(buffer, buffer_size, "../%s", rel_path);
            if (FileReadable(buffer))
                return buffer;

            return rel_path;
        }

        void ResolveSiblingPath(const char* vectors_path,
                                const char* relative,
                                char* out,
                                std::size_t out_capacity)
        {
            std::strncpy(out, vectors_path, out_capacity - 1);
            out[out_capacity - 1] = '\0';

            char* slash = std::strrchr(out, '/');
            if (!slash)
                slash = std::strrchr(out, '\\');

            if (slash)
                slash[1] = '\0';
            else
                out[0] = '\0';

            std::strncat(out, relative, out_capacity - std::strlen(out) - 1);
        }

        bool ReadTextFile(const char* path, char* buffer, std::size_t capacity, std::size_t& bytes_read)
        {
            std::FILE* file = std::fopen(path, "rb");
            if (!file)
                return false;

            bytes_read = std::fread(buffer, 1, capacity - 1, file);
            buffer[bytes_read] = '\0';
            const bool ok = !std::ferror(file);
            std::fclose(file);
            return ok;
        }

        bool ParseCaseName(const char* object_begin, const char* object_end, char* out, std::size_t out_capacity)
        {
            const char* value = Json::FindKeyValue(object_begin, object_end, "name");
            if (!value)
            {
                std::strncpy(out, "case", out_capacity);
                out[out_capacity - 1] = '\0';
                return true;
            }

            const char* p = value;
            return Json::ParseString(p, object_end, out, out_capacity);
        }

        bool ParseFloatArray(const char* object_begin,
                             const char* object_end,
                             const char* key,
                             float* values,
                             uint32_t& count,
                             uint32_t max_count)
        {
            const char* value = Json::FindKeyValue(object_begin, object_end, key);
            if (!value || *value != '[')
                return false;

            const char* cursor = value;
            const char* elem_begin = nullptr;
            const char* elem_end = nullptr;
            count = 0;

            while (count < max_count &&
                   Json::NextArrayElement(cursor, object_end, elem_begin, elem_end))
            {
                const char* p = elem_begin;
                double number = 0.0;
                if (!Json::ParseNumber(p, elem_end, number))
                    return false;
                values[count++] = static_cast<float>(number);
            }

            return count > 0;
        }

        bool FloatEq(float a, float b, float eps = 1e-5f)
        {
            return std::fabs(a - b) <= eps;
        }

        bool CheckFloats(const float* actual,
                         const float* expected,
                         uint32_t count,
                         const char* label)
        {
            for (uint32_t i = 0; i < count; ++i)
            {
                if (!FloatEq(actual[i], expected[i]))
                {
                    std::cout << "FAIL " << label << ": expected " << expected[i]
                              << " got " << actual[i] << " at index " << i << "\n";
                    return false;
                }
            }

            std::cout << "PASS " << label << "\n";
            return true;
        }

        uint32_t ExpectedInputElements(const ModelLoader::ArchitectureSpec& spec)
        {
            uint32_t count = 1;
            for (uint32_t i = 0; i < spec.input_rank; ++i)
                count *= spec.input_shape[i];
            return count;
        }

        uint32_t ExpectedMlpOutputElements(const ModelLoader::ArchitectureSpec& spec)
        {
            if (spec.num_layers == 0)
                return 0;
            return spec.input_shape[0] * spec.dense_layers[spec.num_layers - 1].units;
        }

        Tensor MakeNhwcInput(float* data, uint32_t h, uint32_t w, uint32_t c)
        {
            Tensor input{};
            input.data = data;
            input.type = DataType::Float32;
            input.rank = 3;
            input.shape[0] = h;
            input.shape[1] = w;
            input.shape[2] = c;
            input.stride[0] = w * c;
            input.stride[1] = c;
            input.stride[2] = 1;
            input.num_elements = h * w * c;
            input.bytes = input.num_elements * sizeof(float);
            return input;
        }

        bool RunMlpCase(MLPNetwork& network,
                        const ModelLoader::ArchitectureSpec& spec,
                        const std::array<uint32_t, kMaxTensorRank>& input_shape,
                        const float* input_values,
                        uint32_t input_count,
                        const float* expected,
                        uint32_t expected_count,
                        const char* case_name,
                        Arena& arena)
        {
            const uint32_t required = input_shape[0] * input_shape[1];
            if (input_count != required)
            {
                std::cout << "FAIL " << case_name << ": input length " << input_count
                          << " != expected " << required << "\n";
                return false;
            }

            const uint32_t output_elements = ExpectedMlpOutputElements(spec);
            if (expected_count != output_elements)
            {
                std::cout << "FAIL " << case_name << ": expected length " << expected_count
                          << " != model output " << output_elements << "\n";
                return false;
            }

            Tensor input = TensorFactory::Create2D(arena, input_shape[0], input_shape[1]);
            if (!input.data)
            {
                std::cout << "FAIL " << case_name << ": arena overflow while allocating input\n";
                return false;
            }

            float* input_data = static_cast<float*>(input.data);
            for (uint32_t i = 0; i < input_count; ++i)
                input_data[i] = input_values[i];

            const uint32_t output_cols = spec.dense_layers[spec.num_layers - 1].units;
            Tensor output = TensorFactory::Create2D(arena, input_shape[0], output_cols);
            if (!output.data)
            {
                std::cout << "FAIL " << case_name << ": arena overflow while allocating output\n";
                return false;
            }

            TensorFactory::PrintLabeled("Input", input);
            network.forward(input, output, arena);
            TensorFactory::PrintLabeled("Output", output);

            return CheckFloats(static_cast<const float*>(output.data), expected, expected_count, case_name);
        }

        bool RunCnnCase(CNNNetwork& network,
                        const std::array<uint32_t, kMaxTensorRank>& input_shape,
                        const float* input_values,
                        uint32_t input_count,
                        const float* expected,
                        uint32_t expected_count,
                        const char* case_name,
                        Arena& arena)
        {
            const uint32_t required = input_shape[0] * input_shape[1] * input_shape[2];
            if (input_count != required)
            {
                std::cout << "FAIL " << case_name << ": input length " << input_count
                          << " != expected " << required << "\n";
                return false;
            }

            float input_buffer[kMaxFloats] = {};
            for (uint32_t i = 0; i < input_count; ++i)
                input_buffer[i] = input_values[i];

            Tensor input = MakeNhwcInput(input_buffer, input_shape[0], input_shape[1], input_shape[2]);
            TensorFactory::PrintLabeled("Input", input);

            Tensor& output = network.forward(input, arena);
            if (!output.data)
            {
                std::cout << "FAIL " << case_name << ": arena overflow during CNN forward pass\n";
                return false;
            }

            TensorFactory::PrintLabeled("Output", output);

            if (expected_count != output.num_elements)
            {
                std::cout << "FAIL " << case_name << ": expected length " << expected_count
                          << " != output elements " << output.num_elements << "\n";
                return false;
            }

            return CheckFloats(static_cast<const float*>(output.data), expected, expected_count, case_name);
        }
    }

    RunSummary RunVectorsFile(const char* vectors_path)
    {
        RunSummary summary{};

        char path_buffer[ModelLoader::kMaxPathLen] = {};
        const char* resolved_vectors = ResolveModelPath(vectors_path, path_buffer, sizeof(path_buffer));

        char json_buffer[ModelLoader::kMaxJsonBytes] = {};
        std::size_t json_bytes = 0;
        if (!ReadTextFile(resolved_vectors, json_buffer, sizeof(json_buffer), json_bytes))
        {
            std::cout << "  Vectors load failed (" << resolved_vectors << "): cannot open file\n";
            ++summary.failed;
            return summary;
        }

        const char* json = json_buffer;
        const char* json_end = json_buffer + json_bytes;

        const char* model_value = Json::FindKeyValue(json, json_end, "model");
        if (!model_value)
        {
            std::cout << "  Vectors parse failed: missing model field\n";
            ++summary.failed;
            return summary;
        }

        char model_relative[Json::kMaxStringLen] = {};
        const char* model_parse = model_value;
        if (!Json::ParseString(model_parse, json_end, model_relative, sizeof(model_relative)))
        {
            std::cout << "  Vectors parse failed: invalid model path\n";
            ++summary.failed;
            return summary;
        }

        char model_path[ModelLoader::kMaxPathLen] = {};
        ResolveSiblingPath(resolved_vectors, model_relative, model_path, sizeof(model_path));

        char model_fallback[ModelLoader::kMaxPathLen] = {};
        const char* resolved_model = ResolveModelPath(model_path, model_fallback, sizeof(model_fallback));

        ModelLoader::ArchitectureSpec spec{};
        const ModelLoader::LoadResult arch_result = ModelLoader::ParseArchitecture(resolved_model, spec);
        if (arch_result.status != ModelLoader::LoadStatus::Ok)
        {
            std::cout << "  Model load failed (" << resolved_model << "): "
                      << (arch_result.message ? arch_result.message : "unknown error") << "\n";
            ++summary.failed;
            return summary;
        }

        std::cout << "Vectors: " << resolved_vectors << "\n";
        std::cout << "Model:   " << resolved_model << "\n";

        const char* cases_value = Json::FindKeyValue(json, json_end, "cases");
        if (!cases_value || *cases_value != '[')
        {
            std::cout << "  Vectors parse failed: missing cases array\n";
            ++summary.failed;
            return summary;
        }

        const char* cursor = cases_value;
        const char* case_begin = nullptr;
        const char* case_end = nullptr;

        while (Json::NextArrayElement(cursor, json_end, case_begin, case_end))
        {
            char case_name[Json::kMaxStringLen] = {};
            ParseCaseName(case_begin, case_end, case_name, sizeof(case_name));

            float input_values[kMaxFloats] = {};
            float expected_values[kMaxFloats] = {};
            uint32_t input_count = 0;
            uint32_t expected_count = 0;

            if (!ParseFloatArray(case_begin, case_end, "input", input_values, input_count, kMaxFloats) ||
                !ParseFloatArray(case_begin, case_end, "expected", expected_values, expected_count, kMaxFloats))
            {
                std::cout << "FAIL " << case_name << ": invalid input/expected arrays\n";
                ++summary.failed;
                continue;
            }

            if (input_count != ExpectedInputElements(spec))
            {
                std::cout << "FAIL " << case_name << ": input size " << input_count
                          << " does not match model input (" << ExpectedInputElements(spec) << " floats)\n";
                ++summary.failed;
                continue;
            }

            unsigned char* buffer = g_arena_buffer;
            Arena arena;
            arena.init(buffer, sizeof(g_arena_buffer));

            std::cout << "\nCase: " << case_name << "\n";

            if (spec.kind == ModelLoader::NetworkKind::MLP)
            {
                MLPNetwork* network = nullptr;
                std::array<uint32_t, kMaxTensorRank> input_shape{};
                uint32_t input_rank = 0;

                if (ModelLoader::LoadMLP(resolved_model, arena, network, input_shape, input_rank).status !=
                    ModelLoader::LoadStatus::Ok || !network || !network->IsValid())
                {
                    std::cout << "FAIL " << case_name << ": could not load MLP weights\n";
                    ++summary.failed;
                    continue;
                }

                if (RunMlpCase(*network, spec, input_shape, input_values, input_count,
                               expected_values, expected_count, case_name, arena))
                    ++summary.passed;
                else
                    ++summary.failed;
            }
            else if (spec.kind == ModelLoader::NetworkKind::CNN)
            {
                CNNNetwork* network = nullptr;
                std::array<uint32_t, kMaxTensorRank> input_shape{};
                uint32_t input_rank = 0;

                if (ModelLoader::LoadCNN(resolved_model, arena, network, input_shape, input_rank).status !=
                    ModelLoader::LoadStatus::Ok || !network || !network->IsValid())
                {
                    std::cout << "FAIL " << case_name << ": could not load CNN weights\n";
                    ++summary.failed;
                    continue;
                }

                if (RunCnnCase(*network, input_shape, input_values, input_count,
                               expected_values, expected_count, case_name, arena))
                    ++summary.passed;
                else
                    ++summary.failed;
            }
            else
            {
                std::cout << "FAIL " << case_name << ": unsupported network kind\n";
                ++summary.failed;
            }
        }

        return summary;
    }
}
