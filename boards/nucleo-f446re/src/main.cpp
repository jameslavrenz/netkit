// NUCLEO-F446RE MNIST MLP invoke benchmark — same 10 images as benchmark/netkit.
// Lowered AOT (static Kernels:: FC chain), CMSIS-DSP, flash-backed weights by default.

#include "dwt_time.h"
#include "mnist_mlp_aot.hpp"
#include "mnist_test_images.h"
#include "netkit_config.h"
#include "stm32f446xx.h"
#include "uart.h"

#include <array>
#include <cstring>

namespace aot = netkit::aot::mnist_mlp;

namespace {

constexpr int kRuns = 100;
constexpr int kImageCount = kMnistBenchmarkImageCount;
constexpr int kInputSize = kMnistBenchmarkInputSize;

alignas(std::max_align_t) static unsigned char g_arena_memory[aot::kArenaBytesRecommended];
alignas(std::max_align_t) static float g_output[aot::kOutputElements];

int ArgMax10(const float* values)
{
    int best = 0;
    float max_val = values[0];
    for (int i = 1; i < 10; ++i)
    {
        if (values[i] > max_val)
        {
            max_val = values[i];
            best = i;
        }
    }
    return best;
}

}  // namespace

extern "C" int main(void)
{
    uart_init();
    dwt_time_init();

    uart_write("\r\nnetkit NUCLEO-F446RE MNIST MLP benchmark\r\n");
    uart_printf("  backend:     cmsis-dsp (MCU CM4, lowered AOT)\r\n");
    uart_printf("  weights:     %s\r\n",
                NETKIT_WEIGHTS_IN_RAM ? "ram (arena copy at load)"
                                      : "flash (embedded coef arrays)");
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

            const uint32_t start_cycles = dwt_cycles();
            if (!model.forward(arena, sample.pixels, g_output))
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
    uart_write("\r\nnetkit MNIST mlp benchmark summary (cmsis-dsp)\r\n");
    uart_write("  method:      100 runs x 10 images, discard first invoke each run\r\n");
    uart_write("  per-run avg: avg of images 1-9 (us)\r\n\r\n");
    uart_printf("  mean:   %8.3f us (%6.3f ms)\r\n", mean_us, mean_us / 1000.0);
    uart_printf(
        "BENCHMARK_SUMMARY runtime=netkit model=mlp backend=cmsis-dsp mean_us=%.3f runs=%d\r\n",
        mean_us,
        kRuns);
    uart_write("\r\nDONE\r\n");

    for (;;)
    {
    }
}
