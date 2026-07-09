#include "nk_regression.hpp"
#include "kernel_workspace.hpp"
#include "nk_loader.hpp"
#include "quant_output.hpp"
#include "tensor_factory.hpp"
#include "arena_util.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>

namespace NkRegression
{
    namespace
    {
        constexpr std::size_t kMnistCnnArenaCapacity = ArenaUtil::kMnistCnnCapacity;

#if !defined(NETKIT_ARENA_HEAP)
        constexpr std::size_t kHandArenaCapacity = ArenaUtil::kHandCapacity;
        constexpr std::size_t kMnistMlpArenaCapacity = ArenaUtil::kMnistMlpCapacity;

        alignas(std::max_align_t) unsigned char g_hand_arena[kHandArenaCapacity];
        alignas(std::max_align_t) unsigned char g_mnist_mlp_arena[kMnistMlpArenaCapacity];
        alignas(std::max_align_t) unsigned char g_mnist_cnn_arena[kMnistCnnArenaCapacity];
#endif

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

        std::size_t ArenaCapacityForModel(const NkLoader::ParsedModel& model)
        {
            const bool is_cnn = model.header.network_kind == NkFormat::NetworkKind::Cnn;
            return ArenaUtil::CapacityForModel(NkLoader::InputElements(model),
                                               is_cnn,
                                               model.header.weights_bytes,
                                               model.header.biases_bytes);
        }

#if !defined(NETKIT_ARENA_HEAP)
        unsigned char* ArenaBufferForCapacity(std::size_t capacity)
        {
            if (capacity == kMnistCnnArenaCapacity)
                return g_mnist_cnn_arena;
            if (capacity == kMnistMlpArenaCapacity)
                return g_mnist_mlp_arena;
            return g_hand_arena;
        }
#endif

#if defined(NETKIT_ARENA_HEAP)
        Arena g_regression_heap_arena{};
        bool g_regression_heap_ready = false;
        std::size_t g_regression_heap_capacity = 0;

        Arena& RegressionHeapArena(std::size_t capacity)
        {
            if (!g_regression_heap_ready || g_regression_heap_capacity != capacity)
            {
                if (g_regression_heap_ready)
                    ArenaUtil::Release(g_regression_heap_arena);
                g_regression_heap_capacity = capacity;
                g_regression_heap_ready =
                    ArenaUtil::Init(g_regression_heap_arena, capacity, nullptr);
            }
            g_regression_heap_arena.reset();
            return g_regression_heap_arena;
        }
#endif

        bool FloatNear(float a, float b, float eps)
        {
            return std::fabs(a - b) <= eps;
        }

        bool RegressionVerboseEnabled()
        {
            const char* env = std::getenv("NETKIT_REGRESSION_VERBOSE");
            if (env != nullptr && env[0] == '1')
                return true;
            if (env != nullptr && env[0] == '0')
                return false;
            return std::getenv("GITHUB_ACTIONS") == nullptr;
        }

        bool ShouldPrintAllOutputs(uint32_t count)
        {
            if (RegressionVerboseEnabled())
                return true;
            return count <= 64;
        }

        bool ShouldPrintProgress()
        {
            return RegressionVerboseEnabled();
        }

        uint32_t ArgMax(const float* values, uint32_t count)
        {
            uint32_t best = 0;
            for (uint32_t i = 1; i < count; ++i)
            {
                if (values[i] > values[best])
                    best = i;
            }
            return best;
        }

        void CopyRegressionOutput(const Tensor& output, float* dest, uint32_t count)
        {
            // Int8 models stay int8 end-to-end — no on-device/host dequant.
            // Regression for quantized models compares via ArgMaxInt8 only.
            if (output.type == DataType::Int8)
            {
                for (uint32_t i = 0; i < count; ++i)
                    dest[i] = 0.0f;
                return;
            }

            const float* src = static_cast<const float*>(output.data);
            for (uint32_t i = 0; i < count; ++i)
                dest[i] = src[i];
        }

        uint32_t RegressionArgMax(const Tensor& output, uint32_t count)
        {
            if (output.type == DataType::Int8)
                return QuantOps::ArgMaxInt8(static_cast<const int8_t*>(output.data), count);
            return ArgMax(static_cast<const float*>(output.data), count);
        }

        void PrintElementComparison(const float* actual,
                                    const float* expected,
                                    uint32_t count,
                                    float eps)
        {
            std::cout << std::fixed << std::setprecision(4);
            for (uint32_t i = 0; i < count; ++i)
            {
                const bool ok = FloatNear(actual[i], expected[i], eps);
                std::cout << "  out[" << i << "]: actual=" << actual[i]
                          << "  expected=" << expected[i]
                          << (ok ? "  OK" : "  MISMATCH") << "\n";
            }
        }

        constexpr uint32_t kRegressionMaxInputPrint = 256;

        void PrintRegressionInput(const Tensor& input)
        {
            if (input.num_elements <= kRegressionMaxInputPrint)
                TensorFactory::PrintLabeled("Input", input);
            else
                TensorFactory::PrintLabeled("Input", input, kRegressionMaxInputPrint);
        }

        void PrintClassificationSummary(const float* actual,
                                        const float* expected,
                                        uint32_t output_count,
                                        int32_t label,
                                        float tolerance)
        {
            const uint32_t predicted = ArgMax(actual, output_count);

            std::cout << std::fixed << std::setprecision(4);
            std::cout << "  predicted class: " << predicted
                      << "  (label " << label << ")\n";
            std::cout << "  winner out[" << predicted << "]: actual=" << actual[predicted]
                      << "  expected=" << expected[predicted];
            if (FloatNear(actual[predicted], expected[predicted], tolerance))
                std::cout << "  OK\n";
            else
                std::cout << "  MISMATCH\n";

            constexpr float kRunnerUpThreshold = 0.01f;
            for (uint32_t i = 0; i < output_count; ++i)
            {
                if (i == predicted)
                    continue;
                if (actual[i] >= kRunnerUpThreshold || expected[i] >= kRunnerUpThreshold)
                {
                    std::cout << "  runner-up out[" << i << "]: actual=" << actual[i]
                              << "  expected=" << expected[i];
                    if (FloatNear(actual[i], expected[i], tolerance))
                        std::cout << "  OK\n";
                    else
                        std::cout << "  MISMATCH\n";
                }
            }
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

        Tensor MakeNhwcInputInt8(int8_t* data, uint32_t h, uint32_t w, uint32_t c)
        {
            Tensor input{};
            input.data = data;
            input.type = DataType::Int8;
            input.rank = 3;
            input.shape[0] = h;
            input.shape[1] = w;
            input.shape[2] = c;
            input.stride[0] = w * c;
            input.stride[1] = c;
            input.stride[2] = 1;
            input.num_elements = h * w * c;
            input.bytes = input.num_elements * sizeof(int8_t);
            return input;
        }

        // TCAS stores prequantized int8 as float values in [-128, 127] (Python export).
        void CopyPrequantizedInt8(const float* src, int8_t* dst, uint32_t count)
        {
            for (uint32_t i = 0; i < count; ++i)
            {
                const float v = src[i];
                int32_t q = static_cast<int32_t>(v >= 0.0f ? v + 0.5f : v - 0.5f);
                if (q < -128)
                    q = -128;
                else if (q > 127)
                    q = 127;
                dst[i] = static_cast<int8_t>(q);
            }
        }

        bool RunMlpCase(MLPNetwork& network,
                        const NkLoader::ParsedModel& model,
                        const std::array<uint32_t, kMaxTensorRank>& input_shape,
                        const NkLoader::TestCase& test_case,
                        float tolerance,
                        Arena& arena)
        {
            const uint32_t required = input_shape[0] * input_shape[1];
            if (test_case.input_count != required)
            {
                std::cout << "FAIL " << test_case.name << ": input length " << test_case.input_count
                          << " != expected " << required << "\n";
                return false;
            }

            const uint32_t output_elements = NkLoader::OutputElements(model);
            if (test_case.output_count != output_elements)
            {
                std::cout << "FAIL " << test_case.name << ": expected length " << test_case.output_count
                          << " != model output " << output_elements << "\n";
                return false;
            }

            Tensor input{};
            if (network.IsQuantized())
            {
                int8_t* in_i8 = static_cast<int8_t*>(
                    arena.alloc(test_case.input_count * sizeof(int8_t), alignof(int8_t)));
                if (!in_i8)
                {
                    std::cout << "FAIL " << test_case.name << ": arena overflow while allocating input\n";
                    return false;
                }
                CopyPrequantizedInt8(test_case.input, in_i8, test_case.input_count);
                input = TensorFactory::View2DInt8(in_i8, input_shape[0], input_shape[1]);
            }
            else
            {
                input = TensorFactory::Create2D(arena, input_shape[0], input_shape[1]);
                if (!input.data)
                {
                    std::cout << "FAIL " << test_case.name << ": arena overflow while allocating input\n";
                    return false;
                }
                float* input_data = static_cast<float*>(input.data);
                for (uint32_t i = 0; i < test_case.input_count; ++i)
                    input_data[i] = test_case.input[i];
            }

            const uint32_t output_cols = output_elements / input_shape[0];
            Tensor output{};
            if (network.IsQuantized())
            {
                int8_t* out_i8 = static_cast<int8_t*>(
                    arena.alloc(output_elements * sizeof(int8_t), alignof(int8_t)));
                output = TensorFactory::View2DInt8(out_i8, input_shape[0], output_cols);
            }
            else
            {
                output = TensorFactory::Create2D(arena, input_shape[0], output_cols);
            }
            if (!output.data)
            {
                std::cout << "FAIL " << test_case.name << ": arena overflow while allocating output\n";
                return false;
            }

            if (ShouldPrintAllOutputs(test_case.input_count))
                PrintRegressionInput(input);
            network.forward(input, output, arena);

            // Quantized models: int8 end-to-end — compare classification only (no dequant).
            if (network.IsQuantized() || output.type == DataType::Int8)
            {
                const uint32_t pred = RegressionArgMax(output, test_case.output_count);
                uint32_t expected_class = 0;
                if (test_case.label >= 0)
                    expected_class = static_cast<uint32_t>(test_case.label);
                else
                    expected_class = ArgMax(test_case.expected, test_case.output_count);

                if (pred == expected_class)
                {
                    std::cout << "PASS " << test_case.name << " (int8 argmax classification correct)\n";
                    return true;
                }
                std::cout << "FAIL " << test_case.name << ": classification mismatch (pred="
                          << pred << " expected=" << expected_class << ")\n";
                return false;
            }

            float actual_buffer[NkFormat::kMaxCaseFloats] = {};
            CopyRegressionOutput(output, actual_buffer, test_case.output_count);
            const float* actual = actual_buffer;
            if (ShouldPrintAllOutputs(test_case.output_count))
                PrintElementComparison(actual, test_case.expected, test_case.output_count, tolerance);

            bool outputs_ok = true;
            for (uint32_t i = 0; i < test_case.output_count; ++i)
            {
                if (!FloatNear(actual[i], test_case.expected[i], tolerance))
                    outputs_ok = false;
            }

            if (test_case.label >= 0)
            {
                PrintClassificationSummary(actual, test_case.expected, test_case.output_count, test_case.label,
                                           tolerance);
                const bool class_ok =
                    RegressionArgMax(output, test_case.output_count) == static_cast<uint32_t>(test_case.label);
                if (outputs_ok && class_ok)
                {
                    std::cout << "PASS " << test_case.name
                              << " (all neurons within tolerance, classification correct)\n";
                    return true;
                }

                if (!outputs_ok)
                    std::cout << "FAIL " << test_case.name << ": output neuron mismatch\n";
                if (!class_ok)
                    std::cout << "FAIL " << test_case.name << ": classification mismatch\n";
                return false;
            }

            if (outputs_ok)
            {
                std::cout << "PASS " << test_case.name << " (" << test_case.output_count
                          << " outputs match within tolerance)\n";
                return true;
            }

            std::cout << "FAIL " << test_case.name << " (mismatch in outputs)\n";
            return false;
        }

        bool RunCnnCase(CNNNetwork& network,
                        const std::array<uint32_t, kMaxTensorRank>& input_shape,
                        const NkLoader::TestCase& test_case,
                        float tolerance,
                        Arena& arena)
        {
            const uint32_t required = input_shape[0] * input_shape[1] * input_shape[2];
            if (test_case.input_count != required)
            {
                std::cout << "FAIL " << test_case.name << ": input length " << test_case.input_count
                          << " != expected " << required << "\n";
                return false;
            }

            Tensor input{};
            int8_t input_i8[NkFormat::kMaxCaseFloats] = {};
            float input_buffer[NkFormat::kMaxCaseFloats] = {};
            if (network.IsQuantized())
            {
                CopyPrequantizedInt8(test_case.input, input_i8, test_case.input_count);
                input = MakeNhwcInputInt8(input_i8, input_shape[0], input_shape[1], input_shape[2]);
            }
            else
            {
                for (uint32_t i = 0; i < test_case.input_count; ++i)
                    input_buffer[i] = test_case.input[i];
                input = MakeNhwcInput(input_buffer, input_shape[0], input_shape[1], input_shape[2]);
            }
            if (ShouldPrintAllOutputs(test_case.input_count))
                PrintRegressionInput(input);

            if (ShouldPrintProgress())
                std::cout << "  running forward...\n" << std::flush;
            Tensor& output = network.forward(input, arena);
            if (!output.data)
            {
                std::cout << "FAIL " << test_case.name << ": arena overflow during CNN forward pass\n";
                return false;
            }

            if (test_case.output_count != output.num_elements)
            {
                std::cout << "FAIL " << test_case.name << ": expected length " << test_case.output_count
                          << " != output elements " << output.num_elements << "\n";
                return false;
            }

            // Quantized models: int8 end-to-end — compare classification only (no dequant).
            if (network.IsQuantized() || output.type == DataType::Int8)
            {
                const uint32_t pred = RegressionArgMax(output, test_case.output_count);
                uint32_t expected_class = 0;
                if (test_case.label >= 0)
                    expected_class = static_cast<uint32_t>(test_case.label);
                else
                    expected_class = ArgMax(test_case.expected, test_case.output_count);

                if (pred == expected_class)
                {
                    std::cout << "PASS " << test_case.name << " (int8 argmax classification correct)\n";
                    return true;
                }
                std::cout << "FAIL " << test_case.name << ": classification mismatch (pred="
                          << pred << " expected=" << expected_class << ")\n";
                return false;
            }

            float actual_copy[NkFormat::kMaxCaseFloats] = {};
            CopyRegressionOutput(output, actual_copy, test_case.output_count);
            const float* actual = actual_copy;
            if (ShouldPrintAllOutputs(test_case.output_count))
                PrintElementComparison(actual, test_case.expected, test_case.output_count, tolerance);

            for (uint32_t i = 0; i < test_case.output_count; ++i)
            {
                if (!FloatNear(actual[i], test_case.expected[i], tolerance))
                {
                    std::cout << "FAIL " << test_case.name << " (mismatch at out[" << i << "])\n";
                    return false;
                }
            }

            if (test_case.label >= 0)
            {
                PrintClassificationSummary(actual, test_case.expected, test_case.output_count, test_case.label,
                                           tolerance);
                const bool class_ok =
                    RegressionArgMax(output, test_case.output_count) == static_cast<uint32_t>(test_case.label);
                if (!class_ok)
                {
                    std::cout << "FAIL " << test_case.name << ": classification mismatch\n";
                    return false;
                }
                std::cout << "PASS " << test_case.name
                          << " (all neurons within tolerance, classification correct)\n";
                return true;
            }

            std::cout << "PASS " << test_case.name << " (" << test_case.output_count
                      << " outputs match within tolerance)\n";
            return true;
        }

    }

    RunSummary RunModelTests(const char* nk_path)
    {
        RunSummary summary{};

        char path_buffer[NkLoader::kMaxPathLen] = {};
        const char* resolved = ResolveModelPath(nk_path, path_buffer, sizeof(path_buffer));

        NkLoader::ParsedModel parsed{};
        const NkLoader::LoadResult parse_result = NkLoader::ParseFile(resolved, parsed);
        if (parse_result.status != NkLoader::LoadStatus::Ok)
        {
            std::cout << "  Model parse failed (" << resolved << "): "
                      << (parse_result.message ? parse_result.message
                                               : NkLoader::StatusMessage(parse_result.status))
                      << "\n";
            ++summary.failed;
            return summary;
        }

        static NkLoader::TestSuite tests_storage{};
        NkLoader::TestSuite& tests = tests_storage;
        tests = NkLoader::TestSuite{};
        const NkLoader::LoadResult test_result = NkLoader::ReadTestSuite(resolved, tests);
        if (test_result.status != NkLoader::LoadStatus::Ok)
        {
            std::cout << "  Test load failed (" << resolved << "): "
                      << (test_result.message ? test_result.message
                                              : NkLoader::StatusMessage(test_result.status))
                      << "\n";
            ++summary.failed;
            return summary;
        }

        const NkLoader::NetworkKind kind = parse_result.kind;
        const std::size_t arena_capacity = ArenaCapacityForModel(parsed);

#if defined(NETKIT_ARENA_HEAP)
        // Fresh heap per model file — large backbones must not share arena with smaller models.
        ResetActiveKernelWorkspace();
        if (g_regression_heap_ready)
        {
            ArenaUtil::Release(g_regression_heap_arena);
            g_regression_heap_ready = false;
            g_regression_heap_capacity = 0;
        }
#endif

        std::cout << "Model: " << resolved << "\n";
        std::cout << "Embedded cases: " << tests.num_cases << "\n";
        std::cout << "Output tolerance: " << tests.tolerance << "\n";

        for (uint32_t i = 0; i < tests.num_cases; ++i)
        {
            const NkLoader::TestCase& test_case = tests.cases[i];
            std::cout << "\nCase: " << test_case.name << "\n";

#if defined(NETKIT_ARENA_HEAP)
            ResetActiveKernelWorkspace();
            if (g_regression_heap_ready)
            {
                ArenaUtil::Release(g_regression_heap_arena);
                g_regression_heap_ready = false;
                g_regression_heap_capacity = 0;
            }
            Arena& arena = RegressionHeapArena(arena_capacity);
            if (!g_regression_heap_ready || !arena.base)
            {
                std::cout << "FAIL " << test_case.name << ": arena init failed\n";
                ++summary.failed;
                continue;
            }
#else
            ArenaUtil::Scoped arena_scope(arena_capacity, ArenaBufferForCapacity(arena_capacity));
            if (!arena_scope)
            {
                std::cout << "FAIL " << test_case.name << ": arena init failed\n";
                ++summary.failed;
                continue;
            }
            Arena& arena = arena_scope.Get();
#endif

            if (kind == NkLoader::NetworkKind::Mlp)
            {
                MLPNetwork* network = nullptr;
                std::array<uint32_t, kMaxTensorRank> input_shape{};
                uint32_t input_rank = 0;

                if (NkLoader::LoadMLP(resolved, arena, network, input_shape, input_rank).status !=
                        NkLoader::LoadStatus::Ok ||
                    !network || !network->IsValid())
                {
                    std::cout << "FAIL " << test_case.name << ": could not load MLP weights\n";
                    ++summary.failed;
                    continue;
                }

                if (RunMlpCase(*network, parsed, input_shape, test_case, tests.tolerance, arena))
                    ++summary.passed;
                else
                    ++summary.failed;
            }
            else if (kind == NkLoader::NetworkKind::Cnn)
            {
                CNNNetwork* network = nullptr;
                std::array<uint32_t, kMaxTensorRank> input_shape{};
                uint32_t input_rank = 0;

                NkLoader::LoadResult load_result{};
                if (ShouldPrintProgress())
                    std::cout << "  loading weights...\n" << std::flush;
                load_result = NkLoader::LoadCNN(resolved, arena, network, input_shape, input_rank);
                if (load_result.status != NkLoader::LoadStatus::Ok || !network || !network->IsValid())
                {
                    std::cout << "FAIL " << test_case.name << ": could not load CNN weights";
                    if (load_result.message)
                        std::cout << " (" << load_result.message << ")";
                    std::cout << "\n";
                    ++summary.failed;
                    continue;
                }

                if (RunCnnCase(*network, input_shape, test_case, tests.tolerance, arena))
                    ++summary.passed;
                else
                    ++summary.failed;
            }
            else
            {
                std::cout << "FAIL " << test_case.name << ": unsupported network kind\n";
                ++summary.failed;
            }
        }

        return summary;
    }

    void BeginRegressionArena()
    {
#if defined(NETKIT_ARENA_HEAP)
        (void)RegressionHeapArena(kMnistCnnArenaCapacity);
#endif
    }

    void EndRegressionArena()
    {
#if defined(NETKIT_ARENA_HEAP)
        if (g_regression_heap_ready)
        {
            ArenaUtil::Release(g_regression_heap_arena);
            g_regression_heap_ready = false;
        }
#endif
    }
}
