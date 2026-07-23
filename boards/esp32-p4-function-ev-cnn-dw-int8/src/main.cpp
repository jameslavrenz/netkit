// Seeed ESP32-P4-Function-EV — MNIST DS-CNN int8 (netkit ESP-NN, interpreter embed).
// Methodology matches NUCLEO MCU peers: 10 runs × 10 images, discard image 0 each run.

#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "mnist_cnn_dw_int8_aot.hpp"
#include "mnist_cnn_int8_test_images.h"
#include "netkit_config.h"
#include "netkit_util.hpp"

#include <array>
#include <cstdio>
#include <cstring>

namespace aot = netkit::aot::mnist_cnn_dw_int8;

#if NETKIT_REFERENCE_QUANT_LOOPS
#define NK_CNN_DW_INT8_BACKEND_LABEL "reference-int8"
#else
#define NK_CNN_DW_INT8_BACKEND_LABEL "esp-nn-int8"
#endif

namespace {

constexpr int kRuns = 10;
constexpr int kImageCount = kMnistCnnInt8BenchmarkImageCount;
constexpr int kInputSize = kMnistCnnInt8BenchmarkInputSize;
constexpr int kOutputClasses = 10;
// MCU default is NK_ARENA_DEFAULT_CAPACITY (64 KiB). DS-CNN embed needs more
// for act ping-pong + plan scratch (act_b alone is ~21 KiB).
constexpr std::size_t kArenaCapacity = 96u * 1024u;
static_assert(NK_ARENA_DEFAULT_CAPACITY == 64u * 1024u,
              "XIAO boards assume MCU arena default of 64 KiB");

// P4: heap-allocate 96 KiB arena (sram_high). A static 96 KiB buffer in
// sram_low + 64 KiB main stack fails xTaskCreate at startup.
static unsigned char* g_arena_memory = nullptr;
alignas(std::max_align_t) static int8_t g_output_i8[aot::kOutputElements];
alignas(std::max_align_t) static int8_t g_input_staging[kInputSize];

int ArgMax10Int8(const int8_t* values)
{
    return static_cast<int>(CmsisQuantUtil::ArgMaxInt8(values, kOutputClasses));
}

void CopyInputInt8(int8_t* dst, const int8_t* src)
{
    CmsisQuantUtil::CopyInt8(src, dst, static_cast<uint32_t>(kInputSize));
}

void PrintOutI8(const int8_t* values)
{
    for (int i = 0; i < kOutputClasses; ++i)
    {
        std::printf("%s%d", i ? "," : "", static_cast<int>(values[i]));
    }
}

void PrintDigitSummary(int image,
                       int label,
                       int predicted,
                       int pred_i8,
                       const int8_t* out_i8,
                       int ok)
{
    std::printf(
        "DIGIT_SUMMARY runtime=netkit model=cnn_dw_int8 image=%d label=%d pred=%d pred_i8=%d ok=%d out_i8=",
        image,
        label,
        predicted,
        pred_i8,
        ok);
    PrintOutI8(out_i8);
    std::printf("\n");
}

}  // namespace

extern "C" void app_main(void)
{
    std::printf("\nnetkit ESP32-P4-Function-EV MNIST DS-CNN int8 benchmark\n");
    std::printf("  backend:     %s (MCU ESP32P4, interpreter embed)\n",
                NETKIT_REFERENCE_QUANT_LOOPS ? "netkit reference" : "esp-nn");
    std::printf("  weights:     flash (embedded .nk blob)\n");
    std::printf("  dtype:       int8 end-to-end\n");
    std::printf("  classify:    argmax(logits) — final Softmax omitted\n");
    std::printf("  images:      %d per run\n", kImageCount);
    std::printf("  runs:        %d (discard first invoke each run)\n", kRuns);
    std::printf("  arena bytes: %u\n", static_cast<unsigned>(kArenaCapacity));
    std::printf("  sysclk:      360000000 Hz\n");

    g_arena_memory = static_cast<unsigned char*>(
        heap_caps_malloc(kArenaCapacity, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    if (g_arena_memory == nullptr)
    {
        std::printf("ERR arena alloc\n");
        return;
    }

    Arena arena;
    arena.init(g_arena_memory, kArenaCapacity);

    aot::Model model;
    if (!model.load(arena))
    {
        std::printf("ERR model load\n");
        return;
    }
    std::printf("  model:       loaded\n");

    {
        const MnistCnnInt8BenchmarkSample& probe = kMnistCnnInt8BenchmarkImages[0];
        CopyInputInt8(g_input_staging, probe.pixels);
        if (!model.forwardInt8(arena, g_input_staging, g_output_i8))
        {
            std::printf("ERR probe forward\n");
            return;
        }
        const int predicted = ArgMax10Int8(g_output_i8);
        std::printf("  probe:       label=%d pred=%d pred_i8=%d out_i8=",
                    probe.label,
                    predicted,
                    static_cast<int>(g_output_i8[predicted]));
        PrintOutI8(g_output_i8);
        std::printf("\n");
    }

    std::array<double, kRuns> run_averages_us{};
    std::array<int, kImageCount> final_predictions{};
    std::array<int8_t, kImageCount * kOutputClasses> final_outputs{};
    int correct = 0;

    for (int run = 0; run < kRuns; ++run)
    {
        double run_total_us = 0.0;
        int counted = 0;
        for (int i = 0; i < kImageCount; ++i)
        {
            const MnistCnnInt8BenchmarkSample& sample = kMnistCnnInt8BenchmarkImages[i];
            CopyInputInt8(g_input_staging, sample.pixels);
            const int64_t start_us = esp_timer_get_time();
            if (!model.forwardInt8(arena, g_input_staging, g_output_i8))
            {
                std::printf("ERR invoke image %d\n", i);
                return;
            }
            const int64_t elapsed_us = esp_timer_get_time() - start_us;
            if (i > 0)
            {
                run_total_us += static_cast<double>(elapsed_us);
                ++counted;
            }
            if (run == kRuns - 1)
            {
                const int predicted = ArgMax10Int8(g_output_i8);
                final_predictions[static_cast<size_t>(i)] = predicted;
                std::memcpy(&final_outputs[static_cast<size_t>(i) * kOutputClasses],
                            g_output_i8,
                            static_cast<size_t>(kOutputClasses));
                if (predicted == sample.label)
                {
                    ++correct;
                }
            }
        }
        run_averages_us[static_cast<size_t>(run)] =
            run_total_us / static_cast<double>(counted);
    }

    std::printf("\n  per-digit results (final run):\n");
    std::printf("    image  label  pred  pred_i8  ok\n");
    for (int i = 0; i < kImageCount; ++i)
    {
        const MnistCnnInt8BenchmarkSample& sample = kMnistCnnInt8BenchmarkImages[i];
        const int predicted = final_predictions[static_cast<size_t>(i)];
        const int8_t* out_i8 = &final_outputs[static_cast<size_t>(i) * kOutputClasses];
        const int pred_i8 = static_cast<int>(out_i8[predicted]);
        const int ok = predicted == sample.label ? 1 : 0;
        std::printf("    %5d  %5d  %4d  %7d  %s\n",
                    i,
                    sample.label,
                    predicted,
                    pred_i8,
                    ok ? "yes" : "no");
        PrintDigitSummary(i, sample.label, predicted, pred_i8, out_i8, ok);
    }

    double mean_us = 0.0;
    for (int i = 0; i < kRuns; ++i)
    {
        mean_us += run_averages_us[static_cast<size_t>(i)];
    }
    mean_us /= static_cast<double>(kRuns);

    std::printf("  accuracy:    %d/%d on final run\n", correct, kImageCount);
    std::printf("\nnetkit MNIST cnn_dw int8 benchmark summary (%s)\n", NK_CNN_DW_INT8_BACKEND_LABEL);
    std::printf("  method:      %d runs x 10 images, discard first invoke each run\n", kRuns);
    std::printf("  per-run avg: avg of images 1-9 (us)\n\n");
    const unsigned long mean_us_i = static_cast<unsigned long>(mean_us + 0.5);
    const unsigned long mean_ms_whole = mean_us_i / 1000ul;
    const unsigned long mean_ms_frac = mean_us_i % 1000ul;
    std::printf("  mean:   %lu us (%lu.%03lu ms)\n", mean_us_i, mean_ms_whole, mean_ms_frac);
    std::printf("BENCHMARK_SUMMARY runtime=netkit model=cnn_dw_int8 backend=" NK_CNN_DW_INT8_BACKEND_LABEL
                " mean_us=%lu runs=%d\n",
                mean_us_i,
                kRuns);
    std::printf("\nDONE\n");
}
