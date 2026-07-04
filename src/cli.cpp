#include "cli.hpp"
#include "netkit.h"
#include "test.hpp"
#include "nk_loader.hpp"
#include "tensor_factory.hpp"
#include "mlp.hpp"
#include "cnn.hpp"
#include "arena_util.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace Cli
{
    namespace
    {
        constexpr uint32_t kMaxInputFloats = NK_MAX_CASE_FLOATS;

#if !defined(NETKIT_ARENA_HEAP)
        alignas(std::max_align_t) unsigned char g_arena_buffer[ArenaUtil::kMnistCnnCapacity];
#endif

        std::size_t ArenaCapacityForModel(const NkLoader::ParsedModel& parsed)
        {
            const bool is_cnn = parsed.header.network_kind == NkFormat::NetworkKind::Cnn;
            return ArenaUtil::CapacityForModel(NkLoader::InputElements(parsed),
                                               is_cnn,
                                               parsed.header.weights_bytes,
                                               parsed.header.biases_bytes);
        }

        void PrintHelp(const char* program)
        {
            std::cout << "netkit — neural network inference CLI\n\n";
            std::cout << "Usage:\n";
            std::cout << "  " << program << " test\n";
            std::cout << "  " << program << " run <model.nk> --input <values>\n";
            std::cout << "  " << program << " inspect <model.nk> [--full]\n";
            std::cout << "  " << program << " help\n";
            std::cout << "  " << program << " -h | --help\n\n";

            std::cout << "Global options:\n";
            std::cout << "  -h, --help    Print this help message and exit\n";
            std::cout << "  help          Same as -h / --help\n\n";

            std::cout << "Commands:\n";
            std::cout << "  test\n";
            std::cout << "      Run the full regression suite (same as make test-cpp).\n";
            std::cout << "      Exit 0 if all cases pass, 1 otherwise.\n\n";

            std::cout << "  run <model.nk> --input <values>\n";
            std::cout << "      Load a model, print its network summary, and run one forward pass.\n";
            std::cout << "      Options:\n";
            std::cout << "        --input <values>   Required. Comma-separated float32 input values.\n";
            std::cout << "                           Forms: --input 1,2,3  or  --input=1,2,3\n";
            std::cout << "      Input count must match the model input shape:\n";
            std::cout << "        MLP: batch × features\n";
            std::cout << "        CNN: H × W × C in NHWC flatten order\n";
            std::cout << "      Maximum " << NK_MAX_CASE_FLOATS << " input floats per invocation.\n\n";

            std::cout << "  inspect <model.nk> [--full]\n";
            std::cout << "      Print a boxed network summary (architecture at a glance).\n";
            std::cout << "      Options:\n";
            std::cout << "        --full   Load weights and report arena usage after a zero-input forward pass.\n\n";

            std::cout << "Path resolution:\n";
            std::cout << "  If <model.nk> is not found in the current directory, netkit tries\n";
            std::cout << "  ../<model.nk> relative to the working directory.\n\n";

            std::cout << "Convert ONNX to .nk with: python -m netkit convert <model.onnx> -o <out.nk>\n";
            std::cout << "See docs/CLI.md for full reference.\n";
        }

        bool IsHelpRequest(int argc, char** argv, int start_index = 1)
        {
            for (int i = start_index; i < argc; ++i)
            {
                if (std::strcmp(argv[i], "-h") == 0 ||
                    std::strcmp(argv[i], "--help") == 0 ||
                    std::strcmp(argv[i], "help") == 0)
                    return true;
            }
            return false;
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

        bool HasFlag(int argc, char** argv, int start_index, const char* flag)
        {
            for (int i = start_index; i < argc; ++i)
            {
                if (std::strcmp(argv[i], flag) == 0)
                    return true;
            }
            return false;
        }

        int CmdTest()
        {
            const NkRegression::RunSummary summary = run_all_tests();

            std::cout << "\n============================\n";
            std::cout << " C++ API SUMMARY\n";
            std::cout << "============================\n";
            std::cout << "Passed: " << summary.passed << "\n";
            std::cout << "Failed: " << summary.failed << "\n";

            return summary.failed == 0 ? 0 : 1;
        }

        bool ParseNkModel(const char* nk_path, char* resolved_buffer, NkLoader::ParsedModel& parsed)
        {
            const char* resolved = ResolveModelPath(nk_path, resolved_buffer, NkLoader::kMaxPathLen);
            const NkLoader::LoadResult result = NkLoader::ParseFile(resolved, parsed);
            if (result.status != NkLoader::LoadStatus::Ok)
            {
                std::cerr << "Failed to parse " << resolved << ": "
                          << (result.message ? result.message : NkLoader::StatusMessage(result.status))
                          << "\n";
                return false;
            }
            return true;
        }

        int CmdInspect(const char* nk_path, bool full)
        {
            char path_buffer[NkLoader::kMaxPathLen] = {};
            const char* resolved = ResolveModelPath(nk_path, path_buffer, sizeof(path_buffer));

            NkLoader::ParsedModel parsed{};
            if (!ParseNkModel(nk_path, path_buffer, parsed))
                return 1;

            if (full)
            {
                NkLoader::PrintNetworkSummary(resolved, parsed);

                const std::size_t arena_capacity = ArenaCapacityForModel(parsed);
                ArenaUtil::Scoped arena_scope(arena_capacity,
#if defined(NETKIT_ARENA_HEAP)
                                              nullptr);
#else
                                              g_arena_buffer);
#endif
                if (!arena_scope)
                {
                    std::cerr << "Failed to initialize arena\n";
                    return 1;
                }
                Arena& arena = arena_scope.Get();

                const uint32_t input_elements = NkLoader::InputElements(parsed);
                float input_values[kMaxInputFloats] = {};
                if (input_elements > kMaxInputFloats)
                {
                    std::cerr << "Model input too large for inspect (" << input_elements << " floats)\n";
                    return 1;
                }

                std::array<uint32_t, kMaxTensorRank> input_shape{};
                uint32_t input_rank = 0;

                if (parsed.header.network_kind == NkFormat::NetworkKind::Mlp)
                {
                    MLPNetwork* network = nullptr;
                    const NkLoader::LoadResult load_result =
                        NkLoader::LoadMLP(resolved, arena, network, input_shape, input_rank);

                    if (load_result.status != NkLoader::LoadStatus::Ok || !network || !network->IsValid())
                    {
                        std::cerr << "Failed to load MLP\n";
                        return 1;
                    }

                    const std::size_t bytes_after_load = arena.offset;
                    Tensor input = TensorFactory::Create2D(arena, input_shape[0], input_shape[1]);
                    const uint32_t output_cols =
                        NkLoader::OutputElements(parsed) / input_shape[0];
                    Tensor output = TensorFactory::Create2D(arena, input_shape[0], output_cols);
                    network->forward(input, output, arena);

                    std::cout << "\nArena (" << arena.capacity << " bytes capacity):\n";
                    std::cout << "  after load:           " << bytes_after_load << " bytes\n";
                    std::cout << "  after forward (zero): " << arena.offset << " bytes\n";
                    std::cout << "  remaining:            " << arena.remaining() << " bytes\n";
                }
                else
                {
                    CNNNetwork* network = nullptr;
                    const NkLoader::LoadResult load_result =
                        NkLoader::LoadCNN(resolved, arena, network, input_shape, input_rank);

                    if (load_result.status != NkLoader::LoadStatus::Ok || !network || !network->IsValid())
                    {
                        std::cerr << "Failed to load CNN\n";
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

                return 0;
            }

            NkLoader::PrintNetworkSummary(resolved, parsed);
            return 0;
        }

        int CmdRun(const char* nk_path, const char* input_text)
        {
            if (!input_text)
            {
                std::cerr << "Missing required --input <a,b,c,...>\n";
                return 1;
            }

            char path_buffer[NkLoader::kMaxPathLen] = {};
            const char* resolved = ResolveModelPath(nk_path, path_buffer, sizeof(path_buffer));

            NkLoader::ParsedModel parsed{};
            if (!ParseNkModel(nk_path, path_buffer, parsed))
                return 1;

            float input_values[kMaxInputFloats] = {};
            uint32_t input_count = 0;
            if (!ParseInputValues(input_text, input_values, input_count, kMaxInputFloats))
            {
                std::cerr << "Invalid --input values: " << input_text << "\n";
                return 1;
            }

            const uint32_t required = NkLoader::InputElements(parsed);
            if (input_count != required)
            {
                std::cerr << "Input has " << input_count << " values but model expects "
                          << required << "\n";
                return 1;
            }

            const std::size_t arena_capacity = ArenaCapacityForModel(parsed);
            ArenaUtil::Scoped arena_scope(arena_capacity,
#if defined(NETKIT_ARENA_HEAP)
                                          nullptr);
#else
                                          g_arena_buffer);
#endif
            if (!arena_scope)
            {
                std::cerr << "Failed to initialize arena\n";
                return 1;
            }
            Arena& arena = arena_scope.Get();

            NkLoader::PrintNetworkSummary(resolved, parsed);
            std::cout << "\n";

            std::array<uint32_t, kMaxTensorRank> input_shape{};
            uint32_t input_rank = 0;

            if (parsed.header.network_kind == NkFormat::NetworkKind::Mlp)
            {
                MLPNetwork* network = nullptr;
                const NkLoader::LoadResult load_result =
                    NkLoader::LoadMLP(resolved, arena, network, input_shape, input_rank);

                if (load_result.status != NkLoader::LoadStatus::Ok || !network || !network->IsValid())
                {
                    std::cerr << "Failed to load MLP\n";
                    return 1;
                }

                Tensor input = TensorFactory::Create2D(arena, input_shape[0], input_shape[1]);
                float* input_data = static_cast<float*>(input.data);
                for (uint32_t i = 0; i < input_count; ++i)
                    input_data[i] = input_values[i];

                const uint32_t output_cols =
                    NkLoader::OutputElements(parsed) / input_shape[0];
                Tensor output = TensorFactory::Create2D(arena, input_shape[0], output_cols);
                TensorFactory::PrintLabeled("Input", input);
                network->forward(input, output, arena);
                TensorFactory::PrintLabeled("Output", output);
            }
            else
            {
                CNNNetwork* network = nullptr;
                const NkLoader::LoadResult load_result =
                    NkLoader::LoadCNN(resolved, arena, network, input_shape, input_rank);

                if (load_result.status != NkLoader::LoadStatus::Ok || !network || !network->IsValid())
                {
                    std::cerr << "Failed to load CNN\n";
                    return 1;
                }

                Tensor input = MakeNhwcInput(input_values, input_shape[0], input_shape[1], input_shape[2]);
                TensorFactory::PrintLabeled("Input", input);
                Tensor& output = network->forward(input, arena);
                TensorFactory::PrintLabeled("Output", output);
            }

            return 0;
        }
    }

    int Run(int argc, char** argv)
    {
        if (argc < 2)
        {
            PrintHelp(argv[0]);
            return 1;
        }

        if (IsHelpRequest(argc, argv))
        {
            PrintHelp(argv[0]);
            return 0;
        }

        const char* command = argv[1];

        if (std::strcmp(command, "test") == 0)
            return CmdTest();

        if (std::strcmp(command, "inspect") == 0)
        {
            if (argc < 3)
            {
                std::cerr << "Missing model path\n";
                PrintHelp(argv[0]);
                return 1;
            }

            return CmdInspect(argv[2], HasFlag(argc, argv, 3, "--full"));
        }

        if (std::strcmp(command, "run") == 0)
        {
            if (argc < 3)
            {
                std::cerr << "Missing model path\n";
                PrintHelp(argv[0]);
                return 1;
            }

            const char* input_text = FindOptionValue(argc, argv, 3, "--input");
            return CmdRun(argv[2], input_text);
        }

        std::cerr << "Unknown command: " << command << "\n";
        PrintHelp(argv[0]);
        return 1;
    }
}
