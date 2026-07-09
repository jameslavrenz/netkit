// netkit MobileNetV4-Conv-Small int8 ImageNet host benchmark.
//
// Loads models/mobilenetv4_imagenet_int8.nk. Feeds prequantized int8 images
// (Python export; no C++ float↔int8). ArgMax on int8 logits.

#include "arena.hpp"
#include "arena_util.hpp"
#include "cnn.hpp"
#include "cmsis_dsp_util.hpp"
#include "imagenet_mnv4_netkit_int8_test_images.h"
#include "nk_loader.hpp"
#include "tensor_factory.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <span>
#include <vector>

#ifndef NETKIT_BENCH_BACKEND
#define NETKIT_BENCH_BACKEND "reference"
#endif

#ifndef NETKIT_DW_ROW_ACCUM
#define NETKIT_DW_ROW_ACCUM 1
#endif

namespace {

constexpr const char* kDefaultModelPath = "models/mobilenetv4_imagenet_int8.nk";
constexpr int kLoops = 5;
constexpr uint32_t kInH = kImagenetMnv4Int8BenchmarkHeight;
constexpr uint32_t kInW = kImagenetMnv4Int8BenchmarkWidth;
constexpr uint32_t kInC = kImagenetMnv4Int8BenchmarkChannels;
constexpr size_t kArenaCapacity = 256 * 1024 * 1024;

Tensor MakeNhwcViewInt8(int8_t* data, uint32_t h, uint32_t w, uint32_t c)
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

int RunBenchmark(const char* model_path)
{
    NkLoader::ParsedModel parsed{};
    const NkLoader::LoadResult parse_result = NkLoader::ParseFile(model_path, parsed);
    if (parse_result.status != NkLoader::LoadStatus::Ok)
    {
        std::fprintf(stderr, "parse failed: %s\n",
                     parse_result.message ? parse_result.message
                                          : NkLoader::StatusMessage(parse_result.status));
        return 1;
    }
    if (parsed.header.network_kind != NkFormat::NetworkKind::Cnn)
    {
        std::fprintf(stderr, "expected CNN model\n");
        return 1;
    }

#if defined(NETKIT_ARENA_HEAP)
    ArenaUtil::Scoped arena_scope(kArenaCapacity, nullptr);
#else
    static unsigned char* arena_memory = new unsigned char[kArenaCapacity];
    ArenaUtil::Scoped arena_scope(kArenaCapacity, arena_memory);
#endif
    if (!arena_scope)
    {
        std::fprintf(stderr, "arena init failed (requested %zu bytes)\n", kArenaCapacity);
        return 1;
    }
    Arena& arena = arena_scope.Get();

    CNNNetwork* network = nullptr;
    std::array<uint32_t, kMaxTensorRank> input_shape{};
    uint32_t input_rank = 0;
    const NkLoader::LoadResult load_result =
        NkLoader::LoadCNN(model_path, arena, network, input_shape, input_rank);
    if (load_result.status != NkLoader::LoadStatus::Ok || !network || !network->IsValid())
    {
        std::fprintf(stderr, "load failed: %s\n",
                     load_result.message ? load_result.message
                                         : NkLoader::StatusMessage(load_result.status));
        return 1;
    }
    if (!network->IsQuantized())
    {
        std::fprintf(stderr, "expected quantized int8 model\n");
        return 1;
    }

    if (input_rank != 3 || input_shape[0] != kInH || input_shape[1] != kInW ||
        input_shape[2] != kInC)
    {
        std::fprintf(stderr, "unexpected input shape [%u,%u,%u], expected [%u,%u,%u]\n",
                     input_shape[0], input_shape[1], input_shape[2], kInH, kInW, kInC);
        return 1;
    }

    const uint32_t output_cols = NkLoader::OutputElements(parsed);
    if (output_cols != static_cast<uint32_t>(kImagenetMnv4Int8BenchmarkNumClasses))
    {
        std::fprintf(stderr, "unexpected output count %u (want %d)\n", output_cols,
                     kImagenetMnv4Int8BenchmarkNumClasses);
        return 1;
    }

    const int num_images = kImagenetMnv4Int8BenchmarkImageCount;
    const char* dw_mode = (NETKIT_DW_ROW_ACCUM ? "row-accum(4)" : "serial");
    std::printf("netkit MobileNetV4 ImageNet int8 benchmark\n");
    std::printf("  backend:     %s\n", NETKIT_BENCH_BACKEND);
    std::printf("  depthwise:   %s\n", dw_mode);
    std::printf("  dtype:       int8\n");
    std::printf("  model:       %s\n", model_path);
    std::printf("  input:       %ux%ux%u  outputs: %u\n", kInH, kInW, kInC, output_cols);
    std::printf("  arena:       %zu bytes\n", kArenaCapacity);
    std::printf("  method:      %d images x %d loops = %d invokes (all timed)\n", num_images,
                kLoops, num_images * kLoops);
#if defined(NETKIT_USE_XNNPACK) && NETKIT_USE_XNNPACK
    std::printf("  note:        XNNPACK qs8; prequantized int8 I/O (Python export)\n");
#else
    std::printf("  note:        prequantized int8 I/O (Python export); no C++ dequant\n");
#endif

    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(num_images * kLoops));
    int correct = 0;
    std::vector<int8_t> input_buf(static_cast<size_t>(kImagenetMnv4Int8BenchmarkInputSize));

    for (int loop = 0; loop < kLoops; ++loop)
    {
        for (int i = 0; i < num_images; ++i)
        {
            std::memcpy(input_buf.data(),
                        kImagenetMnv4Int8BenchmarkImages[i].pixels,
                        static_cast<size_t>(kImagenetMnv4Int8BenchmarkInputSize));
            Tensor input = MakeNhwcViewInt8(input_buf.data(), kInH, kInW, kInC);

            const auto start = std::chrono::steady_clock::now();
            Tensor& output = network->forward(input, arena);
            const auto end = std::chrono::steady_clock::now();

            if (!output.data || output.type != DataType::Int8)
            {
                std::fprintf(stderr, "forward failed on loop %d image %d\n", loop, i);
                return 1;
            }
            samples.push_back(std::chrono::duration<double, std::micro>(end - start).count());
            const int pred = static_cast<int>(
                CmsisQuantUtil::ArgMaxInt8(static_cast<const int8_t*>(output.data), output_cols));
            if (loop == 0)
            {
                if (pred == kImagenetMnv4Int8BenchmarkImages[i].label)
                    ++correct;
                std::printf("  image %d %-28s label=%4d pred=%4d %s\n", i,
                            kImagenetMnv4Int8BenchmarkImages[i].name,
                            kImagenetMnv4Int8BenchmarkImages[i].label, pred,
                            pred == kImagenetMnv4Int8BenchmarkImages[i].label ? "OK" : "MISS");
            }
        }
    }

    const double cold_us = samples.front();
    double all_sum = 0.0;
    for (double s : samples)
        all_sum += s;
    const double all_mean = all_sum / static_cast<double>(samples.size());

    double first_pass_sum = 0.0;
    for (int i = 0; i < num_images; ++i)
        first_pass_sum += samples[static_cast<size_t>(i)];
    const double first_pass_mean = first_pass_sum / static_cast<double>(num_images);

    double warm_sum = 0.0;
    double warm_min = samples[1];
    double warm_max = samples[1];
    const size_t warm_n = samples.size() - 1;
    for (size_t k = 1; k < samples.size(); ++k)
    {
        warm_sum += samples[k];
        if (samples[k] < warm_min)
            warm_min = samples[k];
        if (samples[k] > warm_max)
            warm_max = samples[k];
    }
    const double warm_mean = warm_sum / static_cast<double>(warm_n);
    double var = 0.0;
    for (size_t k = 1; k < samples.size(); ++k)
    {
        const double d = samples[k] - warm_mean;
        var += d * d;
    }
    const double warm_std = std::sqrt(var / static_cast<double>(warm_n));
    std::vector<double> warm_sorted(samples.begin() + 1, samples.end());
    std::sort(warm_sorted.begin(), warm_sorted.end());
    const double warm_median = warm_sorted[warm_sorted.size() / 2];

    const double top1 = 100.0 * static_cast<double>(correct) / static_cast<double>(num_images);
    std::printf("\n");
    std::printf("netkit MobileNetV4 ImageNet int8 summary (%s)\n", NETKIT_BENCH_BACKEND);
    std::printf("  top-1 accuracy:   %d / %d  (%.1f%%)\n", correct, num_images, top1);
    std::printf("  10-image mean:    %9.3f us (%7.3f ms)  <- primary latency\n", first_pass_mean,
                first_pass_mean / 1000.0);
    std::printf("  cold invoke:      %9.3f us (%7.3f ms)\n", cold_us, cold_us / 1000.0);
    std::printf("  warm median:      %9.3f us (%7.3f ms)\n", warm_median, warm_median / 1000.0);
    std::printf("  warm mean:        %9.3f us (%7.3f ms)  over %zu invokes\n", warm_mean,
                warm_mean / 1000.0, warm_n);
    std::printf("  warm min/max:     %9.3f / %.3f us\n", warm_min, warm_max);
    std::printf("  warm stddev:      %9.3f us\n", warm_std);
    std::printf("  all-invoke mean:  %9.3f us\n", all_mean);
    std::printf(
        "BENCHMARK_SUMMARY runtime=netkit model=mobilenetv4_imagenet dtype=int8 backend=%s "
        "top1_correct=%d top1_total=%d top1_pct=%.1f ten_image_mean_us=%.3f warm_median_us=%.3f "
        "warm_mean_us=%.3f cold_us=%.3f invokes=%zu\n",
        NETKIT_BENCH_BACKEND, correct, num_images, top1, first_pass_mean, warm_median, warm_mean,
        cold_us, samples.size());

    return 0;
}

}  // namespace

int main(int argc, char** argv)
{
    const char* model_path = (argc > 1) ? argv[1] : kDefaultModelPath;
    return RunBenchmark(model_path);
}
