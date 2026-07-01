/*
 * test_c_api.c — C23 regression tests for netkit.h
 *
 * Validates the C API independently from the C++ test suite in src/test.cpp.
 * Build: make test-c
 */
#include "netkit.h"

#include <math.h>
#include <stdio.h>
#include <stdalign.h>
#include <string.h>

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

static void TestArena(void)
{
    printf("\n--- arena ---\n");

    alignas(max_align_t) unsigned char memory[256];
    nk_arena_t arena;
    nk_arena_init(&arena, memory, sizeof(memory));

    ExpectTrue(nk_arena_capacity(&arena) == sizeof(memory), "arena capacity");
    ExpectTrue(nk_arena_used(&arena) == 0, "arena initially empty");

    void* block = nk_arena_alloc(&arena, 64);
    ExpectTrue(block != NULL, "arena alloc");
    ExpectTrue(nk_arena_used(&arena) == 64, "arena used after alloc");
    ExpectTrue(nk_arena_remaining(&arena) == sizeof(memory) - 64, "arena remaining");

    nk_arena_reset(&arena);
    ExpectTrue(nk_arena_used(&arena) == 0, "arena reset");
}

static void TestTensorOps(void)
{
    printf("\n--- tensor / ops ---\n");

    alignas(max_align_t) unsigned char memory[1024];
    nk_arena_t arena;
    nk_arena_init(&arena, memory, sizeof(memory));

    nk_tensor_t a = {0};
    nk_tensor_t b = {0};
    nk_tensor_t c = {0};

    ExpectStatus(nk_tensor_create_2d(&arena, 2, 2, &a), NK_OK, "tensor create a");
    ExpectStatus(nk_tensor_create_2d(&arena, 2, 2, &b), NK_OK, "tensor create b");
    ExpectStatus(nk_tensor_create_2d(&arena, 2, 2, &c), NK_OK, "tensor create c");

    const float a_values[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float b_values[] = {1.0f, 0.0f, 0.0f, 1.0f};
    ExpectStatus(nk_tensor_fill(&a, a_values, 4), NK_OK, "tensor fill a");
    ExpectStatus(nk_tensor_fill(&b, b_values, 4), NK_OK, "tensor fill b");

    ExpectTrue(nk_ops_is_matmul_valid(&a, &b, &c), "matmul valid");
    nk_ops_mat_mul(&a, &b, &c);

    const float* out = nk_tensor_data_f32_const(&c);
    ExpectFloatEq(out[0], 1.0f, "matmul c[0]");
    ExpectFloatEq(out[1], 2.0f, "matmul c[1]");
    ExpectFloatEq(out[2], 3.0f, "matmul c[2]");
    ExpectFloatEq(out[3], 4.0f, "matmul c[3]");
}

static void TestParseArchitecture(void)
{
    printf("\n--- parse architecture ---\n");

    nk_arch_info_t info = {0};
    ExpectStatus(nk_parse_architecture("models/test_mlp.json", &info), NK_OK, "parse test_mlp.json");
    ExpectTrue(info.kind == NK_NETWORK_MLP, "mlp kind");
    ExpectTrue(info.input_elements == 2, "mlp input elements");
    ExpectTrue(info.output_elements == 2, "mlp output elements");
}

static void TestModelLoadRun(void)
{
    printf("\n--- model load / run ---\n");

    alignas(max_align_t) static unsigned char memory[NK_ARENA_DEFAULT_CAPACITY];
    nk_arena_t arena;
    nk_arena_init(&arena, memory, sizeof(memory));

    nk_model_t model;
    ExpectStatus(nk_model_load("models/test_mlp.json", &arena, &model), NK_OK, "model load mlp");

    const float input[] = {1.0f, 2.0f};
    float output[2] = {0.0f, 0.0f};
    uint32_t output_count = 0;

    ExpectStatus(nk_model_run(&model,
                            &arena,
                            input,
                            2,
                            output,
                            2,
                            &output_count),
                 NK_OK,
                 "model run mlp");
    ExpectTrue(output_count == 2, "mlp output count");
    ExpectFloatEq(output[0], 3.0f, "mlp output[0]");
    ExpectFloatEq(output[1], 3.0f, "mlp output[1]");

    nk_arena_reset(&arena);

    nk_model_t cnn_model;
    ExpectStatus(nk_model_load("models/cnn_4x4_single.json", &arena, &cnn_model), NK_OK, "model load cnn");

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

static void TestVectorsRegression(void)
{
    printf("\n============================\n");
    printf(" C API VECTORS TESTS\n");
    printf("============================\n");

    const nk_test_summary_t summary = nk_run_all_tests();
    ExpectTrue(summary.failed == 0, "vectors regression failed count");
    ExpectTrue(summary.passed == 8, "vectors regression passed count");
}

int main(void)
{
    printf("============================\n");
    printf(" C API TESTS\n");
    printf("============================\n");

    TestArena();
    TestTensorOps();
    TestParseArchitecture();
    TestModelLoadRun();
    TestVectorsRegression();

    printf("\n============================\n");
    printf(" C API SUMMARY\n");
    printf("============================\n");
    printf("Failures: %d\n", g_failures);

    return g_failures == 0 ? 0 : 1;
}
