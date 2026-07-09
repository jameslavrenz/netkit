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
#include <stdint.h>
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

    void* block = nk_arena_alloc(&arena, 64, 8);
    ExpectTrue(block != nullptr, "arena alloc");
    ExpectTrue(nk_arena_used(&arena) == 64, "arena used after alloc");
    ExpectTrue(nk_arena_remaining(&arena) == sizeof(memory) - 64, "arena remaining");

    nk_arena_reset(&arena);
    ExpectTrue(nk_arena_used(&arena) == 0, "arena reset");
}

#if defined(NETKIT_ARENA_HEAP)
static void TestArenaHeap(void)
{
    printf("\n--- arena heap ---\n");

    nk_arena_t arena;
    ExpectStatus(nk_arena_init_heap(&arena, 4096), NK_OK, "arena heap init");
    ExpectTrue(nk_arena_capacity(&arena) == 4096, "heap arena capacity");

    void* block = nk_arena_alloc(&arena, 128, 8);
    ExpectTrue(block != nullptr, "heap arena alloc");

#if defined(NETKIT_TARGET_CPU)
    nk_arena_destroy_heap(&arena);
    ExpectTrue(nk_arena_capacity(&arena) == 0, "heap arena destroyed");
#else
    nk_arena_destroy_heap(&arena);
    ExpectTrue(nk_arena_capacity(&arena) == 4096, "heap arena not freed on MCU/MPU");
#endif
}
#endif

static void TestArenaAlignment(void)
{
    printf("\n--- arena alignment ---\n");

    alignas(max_align_t) unsigned char memory[512];
    nk_arena_t arena;
    nk_arena_init(&arena, memory, sizeof(memory));

    /* Simulate odd float-count weight payload (28 bytes). */
    void* weights = nk_arena_alloc(&arena, 28, 4);
    ExpectTrue(weights != nullptr, "arena alloc weights");

    void* network = nk_arena_alloc(&arena, 32, 8);
    ExpectTrue(network != nullptr, "arena alloc aligned struct after odd weight blob");
    ExpectTrue(((uintptr_t)network % 8u) == 0u, "struct pointer 8-byte aligned");
    ExpectTrue(nk_arena_used(&arena) > 28, "arena used includes alignment padding");
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

static void TestConv2dSymmetricPadding(void)
{
    printf("\n--- conv2d symmetric padding (C API) ---\n");

    alignas(max_align_t) unsigned char memory[4096];
    nk_arena_t arena;
    nk_arena_init(&arena, memory, sizeof(memory));

    uint32_t shape[3] = {4, 4, 1};
    nk_tensor_t input = {0};
    nk_tensor_t output = {0};
    ExpectStatus(nk_tensor_create_nd(&arena, 3, shape, &input), NK_OK, "conv input tensor");
    ExpectStatus(nk_tensor_create_nd(&arena, 3, shape, &output), NK_OK, "conv output tensor");

    const float input_values[16] = {
        1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    };
    ExpectStatus(nk_tensor_fill(&input, input_values, 16), NK_OK, "conv input fill");

    static float weights[9] = {0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    static float bias[1] = {0.0f};

    nk_conv2d_t conv = {
        .kernel_size = 3,
        .stride = 1,
        .pad_h = 1,
        .pad_w = 1,
        .pad_h_end = NK_PAD_MIRROR,
        .pad_w_end = NK_PAD_MIRROR,
        .in_channels = 1,
        .out_channels = 1,
        .weights = weights,
        .bias = bias,
    };

    nk_conv2d_forward(&conv, &input, &output);

    const float* out = nk_tensor_data_f32_const(&output);
    ExpectFloatEq(out[0], 1.0f, "conv2d center preserved with symmetric pad");
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

    ExpectStatus(nk_parse_architecture("models/mnist_cnn.nk", &info), NK_OK, "parse mnist_cnn.nk");
    ExpectTrue(info.kind == NK_NETWORK_CNN, "mnist cnn kind");
    ExpectTrue(info.input_elements == 784, "mnist cnn input elements");
    ExpectTrue(info.output_elements == 10, "mnist cnn output elements");
}

static void TestMnistCnnLoad(void)
{
    printf("\n--- mnist cnn load ---\n");

    alignas(max_align_t) static unsigned char memory[4U * 1024U * 1024U];
    nk_arena_t arena;
    nk_arena_init(&arena, memory, sizeof(memory));

    nk_cnn_t cnn;
    nk_arch_info_t info = {0};
    ExpectStatus(nk_cnn_load("models/mnist_cnn.nk", &arena, &cnn, &info), NK_OK, "nk_cnn_load mnist_cnn");
    ExpectTrue(nk_cnn_is_valid(&cnn), "mnist cnn valid");
    ExpectTrue(info.output_elements == 10, "loaded mnist cnn output count");
}

static void TestModelLoadRun(void)
{
    printf("\n--- model load / run ---\n");

    alignas(max_align_t) static unsigned char memory[4u * 1024u * 1024u]; /* MNIST-scale; avoid 64 MiB default */
    nk_arena_t arena;
    nk_arena_init(&arena, memory, sizeof(memory));

    nk_model_t model;
    ExpectStatus(nk_model_load("models/test_mlp.nk", &arena, &model), NK_OK, "model load mlp");

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

static void TestArchPrint(void)
{
    printf("\n--- arch print ---\n");

    nk_arch_print("models/test_mlp.nk");
    printf("PASS nk_arch_print smoke (output above)\n");

    ExpectStatus(nk_arch_print("models/test_mlp.nk"), NK_OK, "nk_arch_print status");
    ExpectStatus(nk_arch_print(nullptr), NK_ERR_INVALID_ARGUMENT, "nk_arch_print null path");
}

static void TestModelMetadata(void)
{
    printf("\n--- model metadata ---\n");

    alignas(max_align_t) static unsigned char memory[4u * 1024u * 1024u]; /* MNIST-scale; avoid 64 MiB default */
    nk_arena_t arena;
    nk_arena_init(&arena, memory, sizeof(memory));

    nk_model_t model;
    ExpectStatus(nk_model_load("models/test_mlp.nk", &arena, &model), NK_OK, "model load for metadata");

    ExpectTrue(nk_model_kind(&model) == NK_NETWORK_MLP, "model kind mlp");
    ExpectTrue(nk_model_input_count(&model) == 2, "model input count");
    ExpectTrue(nk_model_output_count(&model) == 2, "model output count");

    nk_arch_info_t info = {0};
    ExpectStatus(nk_model_get_arch(&model, &info), NK_OK, "model get arch");
    ExpectTrue(info.version == 3, "arch info .nk format version");
    ExpectTrue(info.input_elements == 2, "arch info input elements");
    ExpectTrue(info.output_elements == 2, "arch info output elements");
}

static void TestBufferLoad(void)
{
    printf("\n--- buffer load (C API) ---\n");

    FILE* file = fopen("models/test_mlp.nk", "rb");
    ExpectTrue(file != nullptr, "open test_mlp.nk for buffer load");
    if (!file)
        return;

    static uint8_t blob[8192];
    const size_t nbytes = fread(blob, 1, sizeof(blob), file);
    fclose(file);
    ExpectTrue(nbytes > 0, "read test_mlp.nk bytes");

    alignas(max_align_t) static unsigned char memory[4u * 1024u * 1024u]; /* MNIST-scale; avoid 64 MiB default */
    nk_arena_t arena;
    nk_arena_init(&arena, memory, sizeof(memory));

    nk_mlp_t mlp;
    ExpectStatus(nk_mlp_load_memory(blob, nbytes, &arena, &mlp, nullptr), NK_OK, "mlp buffer load");
    ExpectTrue(nk_mlp_is_valid(&mlp), "mlp buffer load valid");
    ExpectTrue(!nk_mlp_is_quantized(&mlp), "test mlp not quantized");

    nk_model_t model;
    ExpectStatus(nk_model_load_memory(blob, nbytes, &arena, &model), NK_OK, "model buffer load");
    ExpectTrue(nk_model_kind(&model) == NK_NETWORK_MLP, "model buffer load kind");
    ExpectTrue(!nk_model_is_quantized(&model), "model buffer load not quantized");

    file = fopen("models/test_cnn.nk", "rb");
    ExpectTrue(file != nullptr, "open test_cnn.nk for buffer load");
    if (!file)
        return;

    static uint8_t cnn_blob[16384];
    const size_t cnn_bytes = fread(cnn_blob, 1, sizeof(cnn_blob), file);
    fclose(file);
    ExpectTrue(cnn_bytes > 0, "read test_cnn.nk bytes");

    nk_cnn_t cnn;
    ExpectStatus(nk_cnn_load_memory(cnn_blob, cnn_bytes, &arena, &cnn, nullptr), NK_OK, "cnn buffer load");
    ExpectTrue(nk_cnn_is_valid(&cnn), "cnn buffer load valid");
    ExpectTrue(!nk_cnn_is_quantized(&cnn), "test cnn not quantized");
}

static void TestInspectModel(void)
{
    printf("\n--- inspect model ---\n");

    alignas(max_align_t) static unsigned char memory[4u * 1024u * 1024u]; /* MNIST-scale; avoid 64 MiB default */
    nk_arena_t arena;
    nk_arena_init(&arena, memory, sizeof(memory));

    nk_inspect_info_t info = {0};
    ExpectStatus(nk_inspect_model("models/test_mlp.nk", &arena, &info), NK_OK, "inspect test_mlp");
    ExpectTrue(info.arch.input_elements == 2, "inspect input elements");
    ExpectTrue(info.weight_floats > 0, "inspect weight floats");
    ExpectTrue(info.arena_bytes_after_load > 0, "inspect arena after load");
    ExpectTrue(info.arena_bytes_after_forward >= info.arena_bytes_after_load,
               "inspect arena after forward");
    ExpectTrue(info.arch.weights_bytes > 0, "inspect weights_bytes");
    ExpectTrue(info.flash_payload_bytes > 0, "inspect flash_payload_bytes");
    ExpectTrue(info.flash_payload_bytes == info.arch.weights_bytes + info.arch.biases_bytes,
               "flash payload matches header");
}

static void TestManualMlpActivationBuffers(void)
{
    printf("\n--- manual mlp activation buffers ---\n");

    alignas(max_align_t) static unsigned char memory[4u * 1024u * 1024u]; /* MNIST-scale; avoid 64 MiB default */
    nk_arena_t arena;
    nk_arena_init(&arena, memory, sizeof(memory));

    nk_mlp_t mlp;
    ExpectStatus(nk_mlp_create(&arena, 2, &mlp), NK_OK, "mlp create");

    static float w0_data[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    static float b0_data[2] = {0.0f, 0.0f};
    static float w1_data[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    static float b1_data[2] = {0.0f, 0.0f};

    nk_tensor_t w0, b0, w1, b1, input, output;
    nk_tensor_view_2d(w0_data, 2, 2, &w0);
    nk_tensor_view_2d(b0_data, 1, 2, &b0);
    nk_tensor_view_2d(w1_data, 2, 2, &w1);
    nk_tensor_view_2d(b1_data, 1, 2, &b1);

    ExpectStatus(nk_mlp_init_layer(&mlp, 0, &w0, &b0, NK_ACTIVATION_NONE, 0.01f), NK_OK, "mlp init layer 0");
    ExpectStatus(nk_mlp_init_layer(&mlp, 1, &w1, &b1, NK_ACTIVATION_NONE, 0.01f), NK_OK, "mlp init layer 1");

    ExpectTrue(!nk_mlp_has_activation_buffers(&mlp), "mlp buffers not ready before init");

    static float in_data[2] = {1.0f, 2.0f};
    static float out_data[2] = {0.0f, 0.0f};
    nk_tensor_view_2d(in_data, 1, 2, &input);
    nk_tensor_view_2d(out_data, 1, 2, &output);

    ExpectStatus(nk_mlp_forward(&mlp, &arena, &input, &output),
                 NK_ERR_NOT_INITIALIZED,
                 "mlp forward without activation buffers");

    ExpectStatus(nk_mlp_init_activation_buffers(&mlp, &arena, 1), NK_OK, "mlp init activation buffers");
    ExpectTrue(nk_mlp_has_activation_buffers(&mlp), "mlp buffers ready after init");

    ExpectStatus(nk_mlp_forward(&mlp, &arena, &input, &output), NK_OK, "mlp forward with buffers");
    ExpectFloatEq(out_data[0], 1.0f, "manual mlp output[0]");
    ExpectFloatEq(out_data[1], 2.0f, "manual mlp output[1]");
}

static void TestCnnLoadHasBuffers(void)
{
    printf("\n--- cnn load activation buffers ---\n");

    alignas(max_align_t) static unsigned char memory[4u * 1024u * 1024u]; /* MNIST-scale; avoid 64 MiB default */
    nk_arena_t arena;
    nk_arena_init(&arena, memory, sizeof(memory));

    nk_cnn_t cnn;
    ExpectStatus(nk_cnn_load("models/cnn_4x4_single.nk", &arena, &cnn, nullptr), NK_OK, "cnn load");
    ExpectTrue(nk_cnn_has_activation_buffers(&cnn), "cnn buffers ready after load");
}

static void TestCnnExtendedOpsLoad(void)
{
    printf("\n--- cnn extended ops load ---\n");

    alignas(max_align_t) static unsigned char memory[4u * 1024u * 1024u]; /* MNIST-scale; avoid 64 MiB default */
    nk_arena_t arena;
    nk_arena_init(&arena, memory, sizeof(memory));

    nk_model_t model;
    ExpectStatus(nk_model_load("models/cnn_extended_ops.nk", &arena, &model), NK_OK, "cnn extended ops load");

    const float input[16] = {
        1.0f, 0.5f, 0.0f, 0.5f,
        0.5f, 1.0f, 0.5f, 0.0f,
        0.0f, 0.5f, 1.0f, 0.5f,
        0.5f, 0.0f, 0.5f, 1.0f,
    };
    float output[2] = {0.0f, 0.0f};
    uint32_t output_count = 0;

    ExpectStatus(nk_model_run(&model, &arena, input, 16, output, 2, &output_count),
                 NK_OK,
                 "cnn extended ops forward");
    ExpectTrue(output_count == 2, "cnn extended ops output count");
}

static void ExpectModelTestsPass(const char* nk_path, const char* label)
{
    char message[128];
    const nk_test_summary_t summary = nk_run_model_tests(nk_path);

    snprintf(message, sizeof(message), "%s failed count", label);
    ExpectTrue(summary.failed == 0, message);

    snprintf(message, sizeof(message), "%s passed count", label);
    ExpectTrue(summary.passed >= 1, message);
}

static void TestCompositeBlockLoad(void)
{
    printf("\n--- composite / ONNX import load (C API) ---\n");

    ExpectModelTestsPass("models/resnet18_basic_block.nk", "resnet18 basic block");
    ExpectModelTestsPass("models/import_resnet_basic_block.nk", "import resnet basic block");
    ExpectModelTestsPass("models/import_mobilenet_uib.nk", "import mobilenet uib");
    ExpectModelTestsPass("models/import_mobilenet_uib_skip.nk", "import mobilenet uib skip");
    ExpectModelTestsPass("models/import_convnextv2_block.nk", "import convnextv2 block");
    ExpectModelTestsPass("models/import_asym_depthwise_conv.nk", "import asym depthwise conv");
    ExpectModelTestsPass("models/yolox_mnv4_small.nk", "yolox mnv4 small");
    ExpectModelTestsPass("models/yolox_head_only.nk", "yolox head only");
}

static void TestManualYoloxDecoupledHeadLayer(void)
{
    printf("\n--- manual yolox decoupled head layer ---\n");

    ExpectTrue(NK_CNN_BLOCK_YOLOX_DECOUPLED_HEAD == 11, "yolox block enum value");

    alignas(max_align_t) static unsigned char memory[4u * 1024u * 1024u]; /* MNIST-scale; avoid 64 MiB default */
    nk_arena_t arena;
    nk_arena_init(&arena, memory, sizeof(memory));

    nk_cnn_t cnn;
    ExpectStatus(nk_cnn_create(&arena, 1, &cnn), NK_OK, "cnn create for yolox head");

    static float weights[1024];
    for (size_t i = 0; i < sizeof(weights) / sizeof(weights[0]); ++i)
        weights[i] = 0.0f;

    float* cls_conv_w[1] = {weights};
    float* cls_conv_b[1] = {weights};
    float* reg_conv_w[1] = {weights};
    float* reg_conv_b[1] = {weights};

    ExpectStatus(nk_cnn_init_yolox_decoupled_head_layer(&cnn,
                                                        &arena,
                                                        0,
                                                        2,
                                                        2,
                                                        4,
                                                        8,
                                                        2,
                                                        1,
                                                        weights,
                                                        weights,
                                                        cls_conv_w,
                                                        cls_conv_b,
                                                        reg_conv_w,
                                                        reg_conv_b,
                                                        weights,
                                                        weights,
                                                        weights,
                                                        weights,
                                                        weights,
                                                        weights),
                 NK_OK,
                 "yolox head init");

    ExpectStatus(nk_cnn_init_yolox_decoupled_head_layer(nullptr,
                                                        &arena,
                                                        0,
                                                        2,
                                                        2,
                                                        4,
                                                        8,
                                                        2,
                                                        1,
                                                        weights,
                                                        weights,
                                                        cls_conv_w,
                                                        cls_conv_b,
                                                        reg_conv_w,
                                                        reg_conv_b,
                                                        weights,
                                                        weights,
                                                        weights,
                                                        weights,
                                                        weights,
                                                        weights),
                 NK_ERR_INVALID_ARGUMENT,
                 "yolox head init null cnn");

    ExpectStatus(nk_cnn_init_activation_buffers(&cnn, &arena, 2, 2, 4),
                 NK_OK,
                 "yolox head activation buffers");
    ExpectTrue(nk_cnn_has_activation_buffers(&cnn), "yolox head buffers ready");

    uint32_t input_shape[3] = {2, 2, 4};
    nk_tensor_t input_tensor;
    nk_tensor_t output_tensor = {0};

    ExpectStatus(nk_tensor_create_nd(&arena, 3, input_shape, &input_tensor),
                 NK_OK,
                 "yolox head input tensor");
    ExpectStatus(nk_cnn_forward(&cnn, &arena, &input_tensor, &output_tensor),
                 NK_OK,
                 "yolox head forward");
    ExpectTrue(output_tensor.rank == 3, "yolox head output rank");
    ExpectTrue(output_tensor.num_elements == 28, "yolox head output elements");
}

static void TestRegression(void)
{
#if defined(NETKIT_DESKTOP)
    printf("\n============================\n");
    printf(" C API REGRESSION TESTS\n");
    printf("============================\n");

    const nk_test_summary_t summary = nk_run_all_tests();
    ExpectTrue(summary.failed == 0, "regression failed count");
    ExpectTrue(summary.passed == 89,
               "regression passed count (89 embedded cases)");
#else
    printf("\n--- regression (skipped: NETKIT_TARGET is not cpu) ---\n");
#endif
}

int main(void)
{
    printf("============================\n");
    printf(" C API TESTS\n");
    printf("============================\n");

    TestArena();
#if defined(NETKIT_ARENA_HEAP)
    TestArenaHeap();
#endif
    TestArenaAlignment();
    TestTensorOps();
    TestConv2dSymmetricPadding();
    TestParseArchitecture();
    TestArchPrint();
    TestModelMetadata();
    TestBufferLoad();
    TestInspectModel();
    TestMnistCnnLoad();
    TestModelLoadRun();
    TestManualMlpActivationBuffers();
    TestCnnLoadHasBuffers();
    TestCnnExtendedOpsLoad();
    TestManualYoloxDecoupledHeadLayer();
    TestCompositeBlockLoad();
    TestRegression();

    printf("\n============================\n");
    printf(" C API SUMMARY\n");
    printf("============================\n");
    printf("Failures: %d\n", g_failures);

    return g_failures == 0 ? 0 : 1;
}
