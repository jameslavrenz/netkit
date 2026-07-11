// Shared timing summary for netkit / TFLM MNIST benchmarks.
#pragma once

#include <cstdio>

namespace BenchmarkStats {

constexpr int kDefaultRuns = 100;
constexpr int kCnnDefaultRuns = 10;
constexpr int kMaxRuns = 100;

// MLP microbench: many invokes in one timed window to escape ~1 µs timer noise.
constexpr int kDefaultBatchInvokes = 1000;
constexpr int kDefaultBatchPasses = 10;

struct Summary {
    int num_runs = 0;
    double mean_us = 0.0;
};

// Mean over warm runs only: discard run 0 (process / cache cold), keep runs 1..N-1.
// Each run average itself already excludes image 0 (per-run invoke warmup).
inline Summary Compute(const double* run_averages_us, int num_runs)
{
    Summary summary{};
    if (num_runs < 2)
    {
        // Need ≥2 runs so run 0 can be discarded; fall back to all runs if misconfigured.
        summary.num_runs = num_runs;
        if (num_runs == 1)
            summary.mean_us = run_averages_us[0];
        return summary;
    }

    const int warm_runs = num_runs - 1;
    summary.num_runs = warm_runs;
    double total = 0.0;
    for (int i = 1; i < num_runs; ++i)
    {
        total += run_averages_us[i];
    }
    summary.mean_us = total / static_cast<double>(warm_runs);

    return summary;
}

inline void PrintSummary(const char* runtime, const char* model, const char* backend,
                         const Summary& summary)
{
    std::printf("\n");
    std::printf("%s MNIST %s benchmark summary (%s)\n", runtime, model, backend);
    std::printf(
        "  method:      discard run 0 + first invoke each run; mean over %d warm runs x images 1-9\n",
        summary.num_runs);
    std::printf("  per-run avg: avg of images 1-9 (us)\n");
    std::printf("\n");
    std::printf("  mean:   %8.3f us (%6.3f ms)\n", summary.mean_us, summary.mean_us / 1000.0);

    std::printf("BENCHMARK_SUMMARY runtime=%s model=%s backend=%s mean_us=%.3f runs=%d\n",
                runtime, model, backend, summary.mean_us, summary.num_runs);
}

// Batch-window MLP timing: discard pass 0; each pass times batch_invokes forwards in one window.
inline void PrintBatchSummary(const char* runtime,
                              const char* model,
                              const char* backend,
                              const Summary& summary,
                              int batch_invokes)
{
    std::printf("\n");
    std::printf("%s MNIST %s benchmark summary (%s)\n", runtime, model, backend);
    std::printf(
        "  method:      discard batch pass 0; mean over %d warm passes; "
        "each pass = %d invokes in one timed window\n",
        summary.num_runs,
        batch_invokes);
    std::printf("  per-invoke:  window_us / %d\n", batch_invokes);
    std::printf("\n");
    std::printf("  mean:   %8.3f us (%6.3f ms)\n", summary.mean_us, summary.mean_us / 1000.0);

    std::printf(
        "BENCHMARK_SUMMARY runtime=%s model=%s backend=%s mean_us=%.3f runs=%d batch_invokes=%d\n",
        runtime,
        model,
        backend,
        summary.mean_us,
        summary.num_runs,
        batch_invokes);
}

}  // namespace BenchmarkStats
