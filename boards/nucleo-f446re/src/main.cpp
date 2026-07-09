// NUCLEO-F446RE MNIST MLP invoke benchmark — same 10 images as benchmark/netkit.
// Lowered AOT (static Kernels:: FC chain), CMSIS-DSP, flash-backed weights by default.

#include "dwt_time.h"
#include "cmsis_dsp_util.hpp"
#include "mnist_mlp_aot.hpp"
#include "mnist_test_images.h"
#include "netkit_config.h"
#include "stm32f446xx.h"
#include "uart.h"

#include <array>
#include <cstring>

// Backend label reflects the CMSIS kernels actually compiled in. For this FC MLP
// the composed dispatch runs fully-connected/activation/softmax on CMSIS-NN
// (LayerFast) and element-wise vector ops on CMSIS-DSP (VectorFast).
#if defined(NETKIT_USE_CMSIS_NN) && NETKIT_USE_CMSIS_NN &&                     \
    defined(NETKIT_USE_CMSIS_DSP) && NETKIT_USE_CMSIS_DSP
#define NK_BACKEND_LABEL "cmsis-dsp+cmsis-nn"
#elif defined(NETKIT_USE_CMSIS_NN) && NETKIT_USE_CMSIS_NN
#define NK_BACKEND_LABEL "cmsis-nn"
#elif defined(NETKIT_USE_CMSIS_DSP) && NETKIT_USE_CMSIS_DSP
#define NK_BACKEND_LABEL "cmsis-dsp"
#else
#define NK_BACKEND_LABEL "reference"
#endif

namespace aot = netkit::aot::mnist_mlp;

namespace {

constexpr int kRuns = 100;
constexpr int kImageCount = kMnistBenchmarkImageCount;
constexpr int kInputSize = kMnistBenchmarkInputSize;

alignas(std::max_align_t) static unsigned char g_arena_memory[aot::kArenaBytesRecommended];
alignas(std::max_align_t) static float g_output[aot::kOutputElements];
alignas(std::max_align_t) static float g_input_staging[kInputSize];

int ArgMax10(const float* values)
{
    return static_cast<int>(CmsisDspUtil::ArgMaxF32(values, 10));
}

void CopyInputF32(float* dst, const float* src)
{
    CmsisDspUtil::CopyF32(src, dst, static_cast<uint32_t>(kInputSize));
}

}  // namespace

extern "C" int main(void)
{
    uart_init();
    dwt_time_init();

    uart_write("\r\nnetkit NUCLEO-F446RE MNIST MLP benchmark\r\n");
    uart_printf("  backend:     " NK_BACKEND_LABEL " (MCU CM4, lowered AOT)\r\n");
    uart_printf("  weights:     flash (embedded coef arrays)\r\n");
    uart_printf("  images:      %d per run\r\n", kImageCount);
    uart_printf("  runs:        %d (discard first invoke each run)\r\n", kRuns);
    uart_printf("  arena bytes: %u\r\n", static_cast<unsigned>(aot::kArenaBytesRecommended));
    uart_printf("  sysclk:      %lu Hz\r\n", static_cast<unsigned long>(SystemCoreClock));

    Arena arena;
    if (!aot::InitArena(arena, g_arena_memory, sizeof(g_arena_memory)))
    {
        uart_write("ERR arena init\r\n");
        for (;;)
        {
        }
    }

    aot::Model model;
    if (!model.load(arena))
    {
        uart_write("ERR model load\r\n");
        for (;;)
        {
        }
    }

    std::array<double, kRuns> run_averages_us{};
    int correct = 0;

    for (int run = 0; run < kRuns; ++run)
    {
        double run_total_us = 0.0;
        int counted = 0;

        for (int i = 0; i < kImageCount; ++i)
        {
            const MnistBenchmarkSample& sample = kMnistBenchmarkImages[i];

            CopyInputF32(g_input_staging, sample.pixels);

            const uint32_t start_cycles = dwt_cycles();
            if (!model.forward(arena, g_input_staging, g_output))
            {
                uart_printf("ERR invoke image %d\r\n", i);
                for (;;)
                {
                }
            }
            const uint32_t elapsed_cycles = dwt_cycles() - start_cycles;

            const double elapsed_us = dwt_cycles_to_us(elapsed_cycles);
            if (i > 0)
            {
                run_total_us += elapsed_us;
                ++counted;
            }

            if (run == kRuns - 1)
            {
                const int predicted = ArgMax10(g_output);
                if (predicted == sample.label)
                {
                    ++correct;
                }
            }
        }

        run_averages_us[static_cast<size_t>(run)] =
            run_total_us / static_cast<double>(counted);
    }

    double mean_us = 0.0;
    for (int i = 0; i < kRuns; ++i)
    {
        mean_us += run_averages_us[static_cast<size_t>(i)];
    }
    mean_us /= static_cast<double>(kRuns);

    uart_printf("  accuracy:    %d/%d on final run\r\n", correct, kImageCount);
    uart_write("\r\nnetkit MNIST mlp benchmark summary (" NK_BACKEND_LABEL ")\r\n");
    uart_write("  method:      100 runs x 10 images, discard first invoke each run\r\n");
    uart_write("  per-run avg: avg of images 1-9 (us)\r\n\r\n");
    uart_printf("  mean:   %8.3f us (%6.3f ms)\r\n", mean_us, mean_us / 1000.0);
    uart_printf(
        "BENCHMARK_SUMMARY runtime=netkit model=mlp backend=" NK_BACKEND_LABEL " mean_us=%.3f runs=%d\r\n",
        mean_us,
        kRuns);
    uart_write("\r\nDONE\r\n");

    for (;;)
    {
    }
}
