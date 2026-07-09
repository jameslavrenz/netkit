/*
 * embedded_smoke.c — lean runtime smoke (no NETKIT_DESKTOP-only APIs).
 *
 * Exercises nk_parse_architecture, nk_model_load, and nk_model_run against
 * small hand-checked models (test_mlp, cnn_4x4_single) using a caller-owned
 * static arena. Primary use: MCU/MPU bring-up; also runs on CPU for host validation.
 *
 * Build: make embedded-smoke  (CPU)  or  make NETKIT_TARGET=mcu embedded-smoke
 * Run:   ./tests/embedded_smoke
 * Matrix: ./tools/run_embedded_smoke.sh  (MCU/MPU + CMSIS profiles)
 */
#include "netkit.h"

#include <math.h>
#include <stdio.h>
#include <stdalign.h>
#include <stdint.h>

static int g_failures = 0;

static void ExpectTrue(int condition, const char* message)
{
    if (!condition)
    {
        fprintf(stderr, "FAIL %s\n", message);
        ++g_failures;
    }
    else
    {
        printf("PASS %s\n", message);
    }
}

static void ExpectStatus(nk_status_t actual, nk_status_t expected, const char* message)
{
    if (actual != expected)
    {
        fprintf(stderr, "FAIL %s: status=%s (%s)\n",
                message,
                nk_status_string(actual),
                nk_last_error());
        ++g_failures;
    }
    else
    {
        printf("PASS %s\n", message);
    }
}

static void ExpectFloatEq(float actual, float expected, const char* message)
{
    if (fabsf(actual - expected) > 1e-5f)
    {
        fprintf(stderr, "FAIL %s: expected %.6f got %.6f\n", message, expected, actual);
        ++g_failures;
    }
    else
    {
        printf("PASS %s\n", message);
    }
}

static void TestTargetProfile(void)
{
    printf("\n--- target profile ---\n");

#if defined(NETKIT_TARGET_MCU)
    printf("target: MCU (arena default %u bytes)\n", NK_ARENA_DEFAULT_CAPACITY);
#elif defined(NETKIT_TARGET_MPU)
    printf("target: MPU (arena default %u bytes)\n", NK_ARENA_DEFAULT_CAPACITY);
#elif defined(NETKIT_TARGET_CPU)
    printf("target: CPU host smoke (arena default %u bytes)\n", NK_ARENA_DEFAULT_CAPACITY);
#else
    printf("target: unknown\n");
#endif

#if defined(NETKIT_USE_CMSIS_NN)
    printf("backend: CMSIS-NN enabled\n");
#else
    printf("backend: reference conv/pool/FC\n");
#endif

#if defined(NETKIT_USE_CMSIS_DSP)
    printf("backend: CMSIS-DSP enabled\n");
#else
    printf("backend: reference vector ops\n");
#endif

    ExpectTrue(1, "target profile printed");
}

static void TestParseArchitecture(void)
{
    printf("\n--- parse architecture ---\n");

    nk_arch_info_t info = {0};
    ExpectStatus(nk_parse_architecture("models/test_mlp.nk", &info), NK_OK, "parse test_mlp.nk");
    ExpectTrue(info.version == 3, "mlp .nk format version");
    ExpectTrue(info.kind == NK_NETWORK_MLP, "mlp kind");
    ExpectTrue(info.input_elements == 2, "mlp input elements");
    ExpectTrue(info.output_elements == 2, "mlp output elements");

    ExpectStatus(nk_parse_architecture("models/cnn_4x4_single.nk", &info), NK_OK, "parse cnn_4x4_single.nk");
    ExpectTrue(info.kind == NK_NETWORK_CNN, "cnn kind");
    ExpectTrue(info.input_elements == 16, "cnn input elements");
    ExpectTrue(info.output_elements == 4, "cnn output elements");
}

static void TestModelLoadRun(void)
{
    printf("\n--- model load / run ---\n");

    /* Small models only — avoid NK_ARENA_DEFAULT_CAPACITY (64 MiB on CPU/MPU). */
    enum { kSmokeArenaBytes = 256 * 1024 };
    alignas(max_align_t) static unsigned char arena_memory[kSmokeArenaBytes];
    nk_arena_t arena;
    nk_arena_init(&arena, arena_memory, sizeof(arena_memory));

    nk_model_t model;
    ExpectStatus(nk_model_load("models/test_mlp.nk", &arena, &model), NK_OK, "model load mlp");

    const float mlp_input[] = {1.0f, 2.0f};
    float mlp_output[2] = {0.0f, 0.0f};
    uint32_t output_count = 0;

    ExpectStatus(nk_model_run(&model,
                            &arena,
                            mlp_input,
                            2,
                            mlp_output,
                            2,
                            &output_count),
                 NK_OK,
                 "model run mlp");
    ExpectTrue(output_count == 2, "mlp output count");
    ExpectFloatEq(mlp_output[0], 3.0f, "mlp output[0]");
    ExpectFloatEq(mlp_output[1], 3.0f, "mlp output[1]");

    nk_arena_reset(&arena);

    nk_model_t cnn_model;
    ExpectStatus(nk_model_load("models/cnn_4x4_single.nk", &arena, &cnn_model), NK_OK, "model load cnn");

    const float cnn_input[16] = {
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1
    };
    float cnn_output[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    output_count = 0;

    ExpectStatus(nk_model_run(&cnn_model,
                            &arena,
                            cnn_input,
                            16,
                            cnn_output,
                            4,
                            &output_count),
                 NK_OK,
                 "model run cnn");
    ExpectTrue(output_count == 4, "cnn output count");
    for (uint32_t i = 0; i < 4; ++i)
    {
        char label[32];
        snprintf(label, sizeof(label), "cnn output[%u]", i);
        ExpectFloatEq(cnn_output[i], 4.0f, label);
    }
}

int main(void)
{
    printf("embedded smoke: netkit lean runtime\n");

    TestTargetProfile();
    TestParseArchitecture();
    TestModelLoadRun();

    printf("\n============================\n");
    if (g_failures == 0)
    {
        printf(" embedded smoke: PASS\n");
        printf("============================\n");
        return 0;
    }

    fprintf(stderr, " embedded smoke: %d failure(s)\n", g_failures);
    fprintf(stderr, "============================\n");
    return 1;
}
