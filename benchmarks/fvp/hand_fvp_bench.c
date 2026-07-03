/*
 * hand_fvp_bench.c — hand-checked MLP/CNN inference cycles on Cortex-M4F via Arm FVP.
 *
 * Uses mlp_hand.nk + cnn_hand.nk with a 64 KiB arena (MCU default).
 * Times forward passes until NETKIT_FVP_CYCLE_TARGET DWT cycles (default 100).
 */
#include "dwt_cycles.h"
#include "netkit.h"

#include <stdio.h>
#include <stdalign.h>
#include <stdint.h>
#include <unistd.h>

extern void initialise_monitor_handles(void);

extern alignas(8) const uint8_t mlp_hand_nk[];
extern const uint32_t mlp_hand_nk_size;
extern alignas(4) const float mlp_hand_nk_input[];
extern const uint32_t mlp_hand_nk_input_count;

extern alignas(8) const uint8_t cnn_hand_nk[];
extern const uint32_t cnn_hand_nk_size;
extern alignas(4) const float cnn_hand_nk_input[];
extern const uint32_t cnn_hand_nk_input_count;

#ifndef NETKIT_BENCH_BACKEND
#define NETKIT_BENCH_BACKEND "unknown"
#endif

/* Build with -DNETKIT_FVP_MODEL_MLP=1 or -DNETKIT_FVP_MODEL_CNN=1 (one model per ELF). */
#if defined(NETKIT_FVP_MODEL_MLP) + defined(NETKIT_FVP_MODEL_CNN) != 1
#error "Define exactly one of NETKIT_FVP_MODEL_MLP or NETKIT_FVP_MODEL_CNN"
#endif

#if defined(NETKIT_FVP_MODEL_MLP)
#define NETKIT_FVP_MODEL_NAME "mlp_hand"
#define NETKIT_FVP_NK_DATA mlp_hand_nk
#define NETKIT_FVP_NK_SIZE mlp_hand_nk_size
#define NETKIT_FVP_INPUT mlp_hand_nk_input
#define NETKIT_FVP_INPUT_COUNT mlp_hand_nk_input_count
#else
#define NETKIT_FVP_MODEL_NAME "cnn_hand"
#define NETKIT_FVP_NK_DATA cnn_hand_nk
#define NETKIT_FVP_NK_SIZE cnn_hand_nk_size
#define NETKIT_FVP_INPUT cnn_hand_nk_input
#define NETKIT_FVP_INPUT_COUNT cnn_hand_nk_input_count
#endif

enum
{
    kArenaBytes = NK_ARENA_DEFAULT_CAPACITY,
#ifndef NETKIT_FVP_CYCLE_TARGET
    kCycleTarget = 100u,
#else
    kCycleTarget = NETKIT_FVP_CYCLE_TARGET,
#endif
    kMaxOutputs = 16u,
    kMaxTimedIterations = 100000u
};

static alignas(max_align_t) unsigned char g_arena_memory[kArenaBytes];

static void EmitResult(const char* model_name,
                       uint32_t avg_cycles,
                       uint32_t total_cycles,
                       uint32_t iters)
{
    printf(
        "model=%s backend=%s avg_cycles=%lu total_cycles=%lu cycle_target=%u iters=%u arena_bytes=%u\n",
        model_name,
        NETKIT_BENCH_BACKEND,
        (unsigned long)avg_cycles,
        (unsigned long)total_cycles,
        (unsigned)kCycleTarget,
        (unsigned)iters,
        (unsigned)kArenaBytes);
    fflush(stdout);
}

static void EmitError(const char* model_name, const char* status, const char* err)
{
    if (err && err[0] != '\0')
    {
        printf("model=%s backend=%s status=%s err=%s\n",
               model_name,
               NETKIT_BENCH_BACKEND,
               status,
               err);
    }
    else
    {
        printf("model=%s backend=%s status=%s\n", model_name, NETKIT_BENCH_BACKEND, status);
    }
    fflush(stdout);
}

static uint32_t BenchModel(const uint8_t* nk_data,
                           uint32_t nk_size,
                           const float* input,
                           uint32_t input_count,
                           const char* model_name)
{
    nk_arena_t arena;
    nk_arena_init(&arena, g_arena_memory, sizeof(g_arena_memory));

    nk_model_t model;
    const nk_status_t load_status = nk_model_load_memory(nk_data, nk_size, &arena, &model);
    if (load_status != NK_OK)
    {
        EmitError(model_name, "LOAD_FAILED", nk_last_error());
        return 0u;
    }

    const uint32_t output_capacity = nk_model_output_count(&model);
    if (output_capacity == 0u || output_capacity > kMaxOutputs)
    {
        EmitError(model_name, "BAD_OUTPUT_CAPACITY", "");
        return 0u;
    }

    float output[kMaxOutputs] = {0.0f};
    uint32_t output_count = 0u;
    uint32_t total_cycles = 0u;
    uint32_t iters = 0u;

    while (total_cycles < kCycleTarget)
    {
        const uint32_t start = netkit_dwt_cycles();
        if (nk_model_run(&model,
                         &arena,
                         input,
                         input_count,
                         output,
                         output_capacity,
                         &output_count) != NK_OK)
        {
            EmitError(model_name, "RUN_FAILED", nk_last_error());
            return 0u;
        }
        const uint32_t end = netkit_dwt_cycles();
        total_cycles += (end - start);
        ++iters;

        if (iters >= kMaxTimedIterations)
        {
            EmitError(model_name, "ITER_LIMIT", "");
            return 0u;
        }
    }

    const uint32_t avg_cycles = total_cycles / iters;
    EmitResult(model_name, avg_cycles, total_cycles, iters);
    return avg_cycles;
}

int main(void)
{
    initialise_monitor_handles();
    netkit_dwt_enable();

    (void)BenchModel(NETKIT_FVP_NK_DATA,
                     NETKIT_FVP_NK_SIZE,
                     NETKIT_FVP_INPUT,
                     NETKIT_FVP_INPUT_COUNT,
                     NETKIT_FVP_MODEL_NAME);

    _exit(0);
}
