#include "cli.hpp"
#include "test.hpp"
#include "model_loader.hpp"
#include "tensor_factory.hpp"
#include "mlp.hpp"
#include "cnn.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace Cli
{
    namespace
    {
        constexpr uint32_t kMaxInputFloats = 4096;

        alignas(std::max_align_t) unsigned char g_arena_buffer[Arena::kDefaultCapacity];

        void PrintUsage(const char* program)
        {
            std::cout << "Usage:\n";
            std::cout << "  " << program << " test\n";
            std::cout << "  " << program << " run <model.json> --input <a,b,c,...>\n";
            std::cout << "  " << program << " inspect <model.json>\n";
        }

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

        uint32_t InputElementCount(const ModelLoader::ArchitectureSpec& spec)
        {
            uint32_t count = 1;
            for (uint32_t i = 0; i < spec.input_rank; ++i)
                count *= spec.input_shape[i];
            return count;
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

        bool ParseInputValues(const char* text, float* values, uint32_t& count, uint32_t max_count)
        {
            count = 0;
            if (!text || !*text)
                return false;

            const char* cursor = text;
            while (*cursor && count < max_count)
            {
                while (*cursor == ' ' || *cursor == ',')
                    ++cursor;

                if (!*cursor)
                    break;

                char* end = nullptr;
                const float value = std::strtof(cursor, &end);
                if (end == cursor)
                    return false;

                values[count++] = value;
                cursor = end;
            }

            while (*cursor == ' ' || *cursor == ',')
                ++cursor;

            return count > 0 && *cursor == '\0';
        }

        const char* FindOptionValue(int argc, char** argv, int start_index, const char* option)
        {
            const std::size_t option_len = std::strlen(option);

            for (int i = start_index; i < argc; ++i)
            {
                if (std::strncmp(argv[i], option, option_len) != 0)
                    continue;

                if (argv[i][option_len] == '=')
                    return argv[i] + option_len + 1;

                if (argv[i][option_len] == '\0' && i + 1 < argc)
                    return argv[i + 1];
            }

            return nullptr;
        }

        int CmdTest()
        {
            const VectorsLoader::RunSummary summary = run_all_tests();

            std::cout << "\n============================\n";
            std::cout << " C++ API SUMMARY\n";
            std::cout << "============================\n";
            std::cout << "Passed: " << summary.passed << "\n";
            std::cout << "Failed: " << summary.failed << "\n";

            return summary.failed == 0 ? 0 : 1;
        }

        int CmdInspect(const char* model_path)
        {
            char path_buffer[ModelLoader::kMaxPathLen] = {};
            const char* resolved = ResolveModelPath(model_path, path_buffer, sizeof(path_buffer));

            ModelLoader::ArchitectureSpec spec{};
            const ModelLoader::LoadResult arch_result = ModelLoader::ParseArchitecture(resolved, spec);
            if (arch_result.status != ModelLoader::LoadStatus::Ok)
            {
                std::cerr << "Failed to parse " << resolved << ": "
                          << (arch_result.message ? arch_result.message : "unknown error") << "\n";
                return 1;
            }

            std::cout << "Model: " << resolved << "\n\n";
            std::cout << "Architecture:\n";
            ModelLoader::PrintArchitecture(spec);

            Arena arena;
            arena.init(g_arena_buffer, sizeof(g_arena_buffer));

            float* weights = nullptr;
            std::size_t float_count = 0;
            const char* weight_error = nullptr;
            const ModelLoader::LoadStatus weight_status =
                ModelLoader::LoadWeightsBin(resolved, arena, weights, float_count, &weight_error);

            std::cout << "\nWeights:\n";
            if (weight_status != ModelLoader::LoadStatus::Ok)
            {
                std::cerr << "  load failed: "
                          << (weight_error ? weight_error : "unknown error") << "\n";
                return 1;
            }

            ModelLoader::PrintWeightsSummary(resolved, weights, float_count, spec.expected_weight_floats);

            arena.reset();

            const uint32_t input_elements = InputElementCount(spec);
            float input_values[kMaxInputFloats] = {};
            if (input_elements > kMaxInputFloats)
            {
                std::cerr << "Model input too large for inspect (" << input_elements << " floats)\n";
                return 1;
            }

            std::array<uint32_t, kMaxTensorRank> input_shape{};
            uint32_t input_rank = 0;

            if (spec.kind == ModelLoader::NetworkKind::MLP)
            {
                MLPNetwork* network = nullptr;
                const ModelLoader::LoadResult load_result =
                    ModelLoader::LoadMLP(resolved, arena, network, input_shape, input_rank);

                if (load_result.status != ModelLoader::LoadStatus::Ok || !network || !network->IsValid())
                {
                    std::cerr << "Failed to load MLP: "
                              << (load_result.message ? load_result.message : "unknown error") << "\n";
                    return 1;
                }

                const std::size_t bytes_after_load = arena.offset;

                Tensor input = TensorFactory::Create2D(arena, input_shape[0], input_shape[1]);
                const uint32_t output_cols = spec.dense_layers[spec.num_layers - 1].units;
                Tensor output = TensorFactory::Create2D(arena, input_shape[0], output_cols);
                if (!input.data || !output.data)
                {
                    std::cerr << "Arena overflow while allocating tensors for inspect forward pass\n";
                    return 1;
                }

                network->forward(input, output, arena);

                std::cout << "\nArena (" << arena.capacity << " bytes capacity):\n";
                std::cout << "  after load:           " << bytes_after_load << " bytes\n";
                std::cout << "  after forward (zero): " << arena.offset << " bytes\n";
                std::cout << "  remaining:            " << arena.remaining() << " bytes\n";
            }
            else if (spec.kind == ModelLoader::NetworkKind::CNN)
            {
                CNNNetwork* network = nullptr;
                const ModelLoader::LoadResult load_result =
                    ModelLoader::LoadCNN(resolved, arena, network, input_shape, input_rank);

                if (load_result.status != ModelLoader::LoadStatus::Ok || !network || !network->IsValid())
                {
                    std::cerr << "Failed to load CNN: "
                              << (load_result.message ? load_result.message : "unknown error") << "\n";
                    return 1;
                }

                const std::size_t bytes_after_load = arena.offset;

                Tensor input = MakeNhwcInput(input_values, input_shape[0], input_shape[1], input_shape[2]);
                Tensor& output = network->forward(input, arena);
                if (!output.data)
                {
                    std::cerr << "Arena overflow during CNN forward pass\n";
                    return 1;
                }

                std::cout << "\nArena (" << arena.capacity << " bytes capacity):\n";
                std::cout << "  after load:           " << bytes_after_load << " bytes\n";
                std::cout << "  after forward (zero): " << arena.offset << " bytes\n";
                std::cout << "  remaining:            " << arena.remaining() << " bytes\n";
            }
            else
            {
                std::cerr << "Unsupported network kind\n";
                return 1;
            }

            return 0;
        }

        int CmdRun(const char* model_path, const char* input_text)
        {
            if (!input_text)
            {
                std::cerr << "Missing required --input <a,b,c,...>\n";
                return 1;
            }

            char path_buffer[ModelLoader::kMaxPathLen] = {};
            const char* resolved = ResolveModelPath(model_path, path_buffer, sizeof(path_buffer));

            ModelLoader::ArchitectureSpec spec{};
            const ModelLoader::LoadResult arch_result = ModelLoader::ParseArchitecture(resolved, spec);
            if (arch_result.status != ModelLoader::LoadStatus::Ok)
            {
                std::cerr << "Failed to parse " << resolved << ": "
                          << (arch_result.message ? arch_result.message : "unknown error") << "\n";
                return 1;
            }

            float input_values[kMaxInputFloats] = {};
            uint32_t input_count = 0;
            if (!ParseInputValues(input_text, input_values, input_count, kMaxInputFloats))
            {
                std::cerr << "Invalid --input values: " << input_text << "\n";
                return 1;
            }

            const uint32_t required = InputElementCount(spec);
            if (input_count != required)
            {
                std::cerr << "Input has " << input_count << " values but model expects "
                          << required << "\n";
                return 1;
            }

            Arena arena;
            arena.init(g_arena_buffer, sizeof(g_arena_buffer));

            std::cout << "Model: " << resolved << "\n";

            std::array<uint32_t, kMaxTensorRank> input_shape{};
            uint32_t input_rank = 0;

            if (spec.kind == ModelLoader::NetworkKind::MLP)
            {
                MLPNetwork* network = nullptr;
                const ModelLoader::LoadResult load_result =
                    ModelLoader::LoadMLP(resolved, arena, network, input_shape, input_rank);

                if (load_result.status != ModelLoader::LoadStatus::Ok || !network || !network->IsValid())
                {
                    std::cerr << "Failed to load MLP: "
                              << (load_result.message ? load_result.message : "unknown error") << "\n";
                    return 1;
                }

                Tensor input = TensorFactory::Create2D(arena, input_shape[0], input_shape[1]);
                if (!input.data)
                {
                    std::cerr << "Arena overflow while allocating input tensor\n";
                    return 1;
                }

                float* input_data = static_cast<float*>(input.data);
                for (uint32_t i = 0; i < input_count; ++i)
                    input_data[i] = input_values[i];

                const uint32_t output_cols = spec.dense_layers[spec.num_layers - 1].units;
                Tensor output = TensorFactory::Create2D(arena, input_shape[0], output_cols);
                if (!output.data)
                {
                    std::cerr << "Arena overflow while allocating output tensor\n";
                    return 1;
                }

                TensorFactory::PrintLabeled("Input", input);
                network->forward(input, output, arena);
                TensorFactory::PrintLabeled("Output", output);
            }
            else if (spec.kind == ModelLoader::NetworkKind::CNN)
            {
                CNNNetwork* network = nullptr;
                const ModelLoader::LoadResult load_result =
                    ModelLoader::LoadCNN(resolved, arena, network, input_shape, input_rank);

                if (load_result.status != ModelLoader::LoadStatus::Ok || !network || !network->IsValid())
                {
                    std::cerr << "Failed to load CNN: "
                              << (load_result.message ? load_result.message : "unknown error") << "\n";
                    return 1;
                }

                Tensor input = MakeNhwcInput(input_values, input_shape[0], input_shape[1], input_shape[2]);
                TensorFactory::PrintLabeled("Input", input);

                Tensor& output = network->forward(input, arena);
                if (!output.data)
                {
                    std::cerr << "Arena overflow during CNN forward pass\n";
                    return 1;
                }

                TensorFactory::PrintLabeled("Output", output);
            }
            else
            {
                std::cerr << "Unsupported network kind\n";
                return 1;
            }

            return 0;
        }
    }

    int Run(int argc, char** argv)
    {
        if (argc < 2)
        {
            PrintUsage(argv[0]);
            return 1;
        }

        const char* command = argv[1];

        if (std::strcmp(command, "test") == 0)
            return CmdTest();

        if (std::strcmp(command, "inspect") == 0)
        {
            if (argc < 3)
            {
                std::cerr << "Missing model path\n";
                PrintUsage(argv[0]);
                return 1;
            }

            return CmdInspect(argv[2]);
        }

        if (std::strcmp(command, "run") == 0)
        {
            if (argc < 3)
            {
                std::cerr << "Missing model path\n";
                PrintUsage(argv[0]);
                return 1;
            }

            const char* input_text = FindOptionValue(argc, argv, 3, "--input");
            return CmdRun(argv[2], input_text);
        }

        std::cerr << "Unknown command: " << command << "\n";
        PrintUsage(argv[0]);
        return 1;
    }
}
