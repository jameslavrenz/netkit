/*
 * mnist_kernel_bench.cpp — compare inference time for MNIST models.
 *
 * Backend is selected at link time (reference vs NETKIT_CMSIS_DSP=1).
 * Run via: make bench-mnist-kernels
 */
#include "arena_util.hpp"
#include "cnn.hpp"
#include "mlp.hpp"
#include "netkit_config.h"
#include "nk_loader.hpp"
#include "tensor_factory.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace
{
    constexpr const char* kMnistMlpPath = "models/mnist_mlp.nk";
    constexpr const char* kMnistCnnPath = "models/mnist_cnn.nk";
    constexpr uint32_t kWarmupIterations = 50;
    constexpr uint32_t kTimedIterations = 2000;

    const char* BackendName()
    {
#if defined(NETKIT_USE_CMSIS_DSP) && NETKIT_USE_CMSIS_DSP
        return "cmsis-dsp";
#else
        return "reference";
#endif
    }

    Tensor MakeNhwcView(float* data, uint32_t h, uint32_t w, uint32_t c)
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

    bool LoadFirstCaseInput(const char* nk_path, float* input, uint32_t input_capacity, uint32_t& input_count)
    {
        NkLoader::TestSuite suite{};
        const NkLoader::LoadResult result = NkLoader::ReadTestSuite(nk_path, suite);
        if (result.status != NkLoader::LoadStatus::Ok || suite.num_cases == 0)
            return false;

        input_count = suite.cases[0].input_count;
        if (input_count == 0 || input_count > input_capacity)
            return false;

        std::memcpy(input, suite.cases[0].input, static_cast<std::size_t>(input_count) * sizeof(float));
        return true;
    }

    double BenchMlp(Arena& arena, const float* input, uint32_t input_elements)
    {
        MLPNetwork* network = nullptr;
        std::array<uint32_t, kMaxTensorRank> input_shape{};
        uint32_t input_rank = 0;

        const NkLoader::LoadResult load_result =
            NkLoader::LoadMLP(kMnistMlpPath, arena, network, input_shape, input_rank);
        if (load_result.status != NkLoader::LoadStatus::Ok || !network || !network->IsValid())
            return -1.0;

        NkLoader::ParsedModel parsed{};
        if (NkLoader::ParseFile(kMnistMlpPath, parsed).status != NkLoader::LoadStatus::Ok)
            return -1.0;

        const uint32_t output_cols = NkLoader::OutputElements(parsed) / input_shape[0];
        Tensor input_tensor = TensorFactory::Create2D(arena, input_shape[0], input_shape[1]);
        Tensor output = TensorFactory::Create2D(arena, input_shape[0], output_cols);

        float* input_data = static_cast<float*>(input_tensor.data);
        for (uint32_t i = 0; i < input_elements; ++i)
            input_data[i] = input[i];

        for (uint32_t i = 0; i < kWarmupIterations; ++i)
            network->forward(input_tensor, output, arena);

        const auto start = std::chrono::steady_clock::now();
        for (uint32_t i = 0; i < kTimedIterations; ++i)
            network->forward(input_tensor, output, arena);
        const auto end = std::chrono::steady_clock::now();

        const double total_ms =
            std::chrono::duration<double, std::milli>(end - start).count();
        return total_ms / static_cast<double>(kTimedIterations);
    }

    double BenchCnn(Arena& arena, float* input_buffer, uint32_t input_elements)
    {
        CNNNetwork* network = nullptr;
        std::array<uint32_t, kMaxTensorRank> input_shape{};
        uint32_t input_rank = 0;

        const NkLoader::LoadResult load_result =
            NkLoader::LoadCNN(kMnistCnnPath, arena, network, input_shape, input_rank);
        if (load_result.status != NkLoader::LoadStatus::Ok || !network || !network->IsValid())
            return -1.0;

        if (input_elements != input_shape[0] * input_shape[1] * input_shape[2])
            return -1.0;

        Tensor input = MakeNhwcView(input_buffer, input_shape[0], input_shape[1], input_shape[2]);

        for (uint32_t i = 0; i < kWarmupIterations; ++i)
        {
            Tensor& output = network->forward(input, arena);
            if (!output.data)
                return -1.0;
        }

        const auto start = std::chrono::steady_clock::now();
        for (uint32_t i = 0; i < kTimedIterations; ++i)
        {
            Tensor& output = network->forward(input, arena);
            if (!output.data)
                return -1.0;
        }
        const auto end = std::chrono::steady_clock::now();

        const double total_ms =
            std::chrono::duration<double, std::milli>(end - start).count();
        return total_ms / static_cast<double>(kTimedIterations);
    }

    void PrintResult(const char* model, double avg_ms)
    {
        if (avg_ms < 0.0)
        {
            std::printf("model=%s backend=%s status=FAILED\n", model, BackendName());
            return;
        }

        const double ips = avg_ms > 0.0 ? 1000.0 / avg_ms : 0.0;
        std::printf(
            "model=%s backend=%s avg_ms=%.4f inferences_per_sec=%.1f warmup=%u iters=%u\n",
            model,
            BackendName(),
            avg_ms,
            ips,
            kWarmupIterations,
            kTimedIterations);
    }
}

int main()
{
    alignas(std::max_align_t) static unsigned char arena_memory[Arena::kDefaultCapacity];
    ArenaUtil::Scoped arena_scope(Arena::kDefaultCapacity, arena_memory);
    if (!arena_scope)
    {
        std::fprintf(stderr, "arena init failed\n");
        return 1;
    }

    float input[4096] = {};
    uint32_t input_count = 0;

    if (!LoadFirstCaseInput(kMnistMlpPath, input, static_cast<uint32_t>(std::size(input)), input_count))
    {
        std::fprintf(stderr, "failed to load MNIST MLP test input\n");
        return 1;
    }

    const double mlp_ms = BenchMlp(arena_scope.Get(), input, input_count);
    PrintResult("mnist_mlp", mlp_ms);

    if (!LoadFirstCaseInput(kMnistCnnPath, input, static_cast<uint32_t>(std::size(input)), input_count))
    {
        std::fprintf(stderr, "failed to load MNIST CNN test input\n");
        return 1;
    }

    const double cnn_ms = BenchCnn(arena_scope.Get(), input, input_count);
    PrintResult("mnist_cnn", cnn_ms);

    return (mlp_ms < 0.0 || cnn_ms < 0.0) ? 1 : 0;
}
