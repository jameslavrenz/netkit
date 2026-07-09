/*
 * netkit.h — C23 public API for netkit
 *
 * Inference: float32 models use float I/O; int8 quantized models require int8 I/O
 * (prequantized at .nk export time in Python — no C++ float→int8 or dequant).
 * Use `nk_model_run` for float32; `nk_model_run_int8` / AOT `forwardInt8` for int8.
 * Documentation:
 *   docs/PHILOSOPHY.md       — product vision, Phase 1/2, roadmap
 *   docs/GETTING_STARTED.md  — build, test, first inference
 *   docs/BUILD_TARGETS.md    — NETKIT_TARGET, arena flags, defaults
 *   docs/DATATYPES.md        — float32 and int8 today; more dtypes roadmap
 *   docs/ARENA.md            — bump allocator memory model
 *   docs/c-api.md            — full C API reference
 *   docs/API_PARITY.md       — C ↔ C++ symbol map
 *
 * Link against libnetkit.a (C++26 implementation). Compile this header with -std=c23.
 */
#ifndef NETKIT_H
#define NETKIT_H

#include "netkit_config.h"
#include <stddef.h>
#include <stdint.h>
#include <stdalign.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------- */
/* Version                                                                    */
/* -------------------------------------------------------------------------- */

#define NK_VERSION_MAJOR 0
#define NK_VERSION_MINOR 1
#define NK_VERSION_PATCH 0

#define NK_MAX_TENSOR_RANK 4
#define NK_MAX_CASE_FLOATS 16384
#define NK_MAX_LAYERS      100
#define NK_MAX_PATH_LEN    256
#define NK_MAX_MESSAGE_LEN 128

#define NK_ARENA_STORAGE_BYTES 64
#define NK_MODEL_STORAGE_BYTES 96
#define NK_MLP_STORAGE_BYTES   16
#define NK_CNN_STORAGE_BYTES   16

/* -------------------------------------------------------------------------- */
/* Status / kinds                                                             */
/* -------------------------------------------------------------------------- */

typedef enum nk_status
{
    NK_OK = 0,
    NK_ERR_MODEL_OPEN,
    NK_ERR_MODEL_READ,
    NK_ERR_MODEL_PARSE,
    NK_ERR_UNSUPPORTED_NETWORK,
    NK_ERR_VERSION_MISMATCH,
    NK_ERR_LAYER_CONFIG,
    NK_ERR_WEIGHT_MISMATCH,
    NK_ERR_ARENA_OVERFLOW,
    NK_ERR_INVALID_ARGUMENT,
    NK_ERR_BUFFER_TOO_SMALL,
    NK_ERR_MODEL_NOT_LOADED,
    NK_ERR_NOT_INITIALIZED
} nk_status_t;

typedef enum nk_network_kind
{
    NK_NETWORK_UNKNOWN = 0,
    NK_NETWORK_MLP,
    NK_NETWORK_CNN
} nk_network_kind_t;

typedef enum nk_dtype
{
    NK_DTYPE_FLOAT32 = 0,
    NK_DTYPE_INT8,
    NK_DTYPE_UINT8,
    NK_DTYPE_INT16
} nk_dtype_t;

typedef enum nk_activation
{
    NK_ACTIVATION_NONE = 0,
    NK_ACTIVATION_RELU,
    NK_ACTIVATION_SIGMOID,
    NK_ACTIVATION_TANH,
    NK_ACTIVATION_LEAKY_RELU,
    NK_ACTIVATION_RELU6,
    NK_ACTIVATION_SOFTMAX
} nk_activation_t;

typedef enum nk_conv_activation
{
    NK_CONV_ACTIVATION_NONE = 0,
    NK_CONV_ACTIVATION_RELU,
    NK_CONV_ACTIVATION_SIGMOID,
    NK_CONV_ACTIVATION_TANH,
    NK_CONV_ACTIVATION_LEAKY_RELU,
    NK_CONV_ACTIVATION_RELU6,
    NK_CONV_ACTIVATION_SOFTMAX
} nk_conv_activation_t;

typedef enum nk_cnn_block_type
{
    NK_CNN_BLOCK_CONV2D = 0,
    NK_CNN_BLOCK_DEPTHWISE_CONV2D,
    NK_CNN_BLOCK_MAX_POOL2D,
    NK_CNN_BLOCK_AVG_POOL2D,
    NK_CNN_BLOCK_BATCH_NORM2D,
    NK_CNN_BLOCK_LAYERNORM2D,
    NK_CNN_BLOCK_FLATTEN,
    NK_CNN_BLOCK_DENSE,
    NK_CNN_BLOCK_CONVNEXTV2_BLOCK,
    NK_CNN_BLOCK_MOBILENETV4_UIB,
    NK_CNN_BLOCK_RESNET_BASIC_BLOCK,
    NK_CNN_BLOCK_YOLOX_DECOUPLED_HEAD
} nk_cnn_block_type_t;

/* -------------------------------------------------------------------------- */
/* Opaque / value handles                                                     */
/* -------------------------------------------------------------------------- */

typedef struct nk_arena
{
    alignas(max_align_t) unsigned char storage[NK_ARENA_STORAGE_BYTES];
} nk_arena_t;

typedef struct nk_model
{
    alignas(max_align_t) unsigned char storage[NK_MODEL_STORAGE_BYTES];
} nk_model_t;

typedef struct nk_mlp
{
    alignas(max_align_t) unsigned char storage[NK_MLP_STORAGE_BYTES];
} nk_mlp_t;

typedef struct nk_cnn
{
    alignas(max_align_t) unsigned char storage[NK_CNN_STORAGE_BYTES];
} nk_cnn_t;

typedef struct nk_tensor
{
    void* data;
    nk_dtype_t dtype;
    uint32_t rank;
    uint32_t shape[NK_MAX_TENSOR_RANK];
    uint32_t stride[NK_MAX_TENSOR_RANK];
    uint32_t num_elements;
    uint32_t bytes;
} nk_tensor_t;

typedef struct nk_conv2d
{
    int kernel_size;
    int stride;
    int pad_h;
    int pad_w;
    int pad_h_end;
    int pad_w_end;
    int in_channels;
    int out_channels;
    float* weights;
    float* bias;
} nk_conv2d_t;

typedef struct nk_arch_info
{
    uint32_t version;
    nk_network_kind_t kind;
    uint32_t input_shape[NK_MAX_TENSOR_RANK];
    uint32_t input_rank;
    uint32_t num_layers;
    size_t expected_weight_floats;
    size_t weights_bytes;
    size_t biases_bytes;
    uint32_t input_elements;
    uint32_t output_elements;
} nk_arch_info_t;

typedef struct nk_inspect_info
{
    nk_arch_info_t arch;
    size_t weight_floats;
    size_t arena_bytes_after_load;
    size_t arena_bytes_after_forward;
    size_t arena_remaining;
    /** Weight+bias payload kept in flash/blob (not in arena peaks). */
    size_t flash_payload_bytes;
} nk_inspect_info_t;

typedef struct nk_test_summary
{
    uint32_t passed;
    uint32_t failed;
} nk_test_summary_t;

/** Pass for pad_h_end / pad_w_end to mirror pad_h / pad_w (symmetric padding). */
#define NK_PAD_MIRROR (-1)

/* -------------------------------------------------------------------------- */
/* Errors / version                                                           */
/* -------------------------------------------------------------------------- */

const char* nk_version_string(void);
const char* nk_status_string(nk_status_t status);
const char* nk_last_error(void);

/* -------------------------------------------------------------------------- */
/* Arena (arena.hpp)                                                          */
/* -------------------------------------------------------------------------- */

void nk_arena_init(nk_arena_t* arena, void* memory, size_t size);
#if defined(NETKIT_ARENA_HEAP)
nk_status_t nk_arena_init_heap(nk_arena_t* arena, size_t capacity);
void nk_arena_destroy_heap(nk_arena_t* arena);
#endif
void* nk_arena_alloc(nk_arena_t* arena, size_t size, size_t alignment);
void nk_arena_reset(nk_arena_t* arena);
size_t nk_arena_capacity(const nk_arena_t* arena);
size_t nk_arena_used(const nk_arena_t* arena);
size_t nk_arena_remaining(const nk_arena_t* arena);

/* -------------------------------------------------------------------------- */
/* Tensor factory (tensor_factory.hpp)                                        */
/* -------------------------------------------------------------------------- */

nk_status_t nk_tensor_create_2d(nk_arena_t* arena, uint32_t rows, uint32_t cols, nk_tensor_t* out);
nk_status_t nk_tensor_create_nd(nk_arena_t* arena,
                                uint32_t rank,
                                const uint32_t* shape,
                                nk_tensor_t* out);
void nk_tensor_view_2d(float* data, uint32_t rows, uint32_t cols, nk_tensor_t* out);
nk_status_t nk_tensor_fill(nk_tensor_t* tensor, const float* values, uint32_t count);
void nk_tensor_print(const nk_tensor_t* tensor);
void nk_tensor_print_labeled(const char* label, const nk_tensor_t* tensor);

/* -------------------------------------------------------------------------- */
/* Tensor access (tensor_access.hpp)                                          */
/* -------------------------------------------------------------------------- */

float* nk_tensor_data_f32(nk_tensor_t* tensor);
const float* nk_tensor_data_f32_const(const nk_tensor_t* tensor);
uint32_t nk_tensor_index_nhwc(const nk_tensor_t* tensor, uint32_t h, uint32_t w, uint32_t c);

/* -------------------------------------------------------------------------- */
/* Ops (ops.hpp)                                                              */
/* -------------------------------------------------------------------------- */

bool nk_ops_is_elementwise_valid(const nk_tensor_t* a, const nk_tensor_t* b);
bool nk_ops_check_same_shape_2d(const nk_tensor_t* a, const nk_tensor_t* b, const nk_tensor_t* c);
bool nk_ops_check_same_shape_nd(const nk_tensor_t* a, const nk_tensor_t* b, const nk_tensor_t* c);
bool nk_ops_is_matmul_valid(const nk_tensor_t* a, const nk_tensor_t* b, const nk_tensor_t* c);
bool nk_ops_is_elementwise_valid_nd(const nk_tensor_t* a, const nk_tensor_t* b, const nk_tensor_t* c);
bool nk_ops_is_unary_op_valid(const nk_tensor_t* a, const nk_tensor_t* c);

void nk_ops_mul(const nk_tensor_t* a, const nk_tensor_t* b, nk_tensor_t* c);
void nk_ops_mul_scalar(const nk_tensor_t* a, float scalar, nk_tensor_t* c);
void nk_ops_mat_add(const nk_tensor_t* a, const nk_tensor_t* b, nk_tensor_t* c);
void nk_ops_mat_add_nd(const nk_tensor_t* a, const nk_tensor_t* b, nk_tensor_t* c);
void nk_ops_mat_mul(const nk_tensor_t* a, const nk_tensor_t* b, nk_tensor_t* c);
void nk_ops_mul_nd(const nk_tensor_t* a, const nk_tensor_t* b, nk_tensor_t* c);

void nk_ops_relu(const nk_tensor_t* a, nk_tensor_t* c);
void nk_ops_sigmoid(const nk_tensor_t* a, nk_tensor_t* c);
void nk_ops_tanh(const nk_tensor_t* a, nk_tensor_t* c);
void nk_ops_leaky_relu(const nk_tensor_t* a, nk_tensor_t* c, float alpha);
void nk_ops_relu6(const nk_tensor_t* a, nk_tensor_t* c);
void nk_ops_softmax(const nk_tensor_t* a, nk_tensor_t* c);

/* -------------------------------------------------------------------------- */
/* Conv2D (conv2d.hpp)                                                        */
/* -------------------------------------------------------------------------- */

void nk_conv2d_forward(const nk_conv2d_t* conv, const nk_tensor_t* input, nk_tensor_t* output);

/* -------------------------------------------------------------------------- */
/* MLP (mlp.hpp)                                                              */
/* -------------------------------------------------------------------------- */

nk_status_t nk_mlp_create(nk_arena_t* arena, uint32_t num_layers, nk_mlp_t* mlp);
bool nk_mlp_is_valid(const nk_mlp_t* mlp);
nk_status_t nk_mlp_init_layer(nk_mlp_t* mlp,
                            uint32_t layer_idx,
                            const nk_tensor_t* weights,
                            const nk_tensor_t* bias,
                            nk_activation_t activation,
                            float leaky_alpha);
nk_status_t nk_mlp_init_activation_buffers(nk_mlp_t* mlp, nk_arena_t* arena, uint32_t batch_rows);
bool nk_mlp_has_activation_buffers(const nk_mlp_t* mlp);
nk_status_t nk_mlp_forward(nk_mlp_t* mlp,
                           nk_arena_t* arena,
                           const nk_tensor_t* input,
                           nk_tensor_t* output);

/* -------------------------------------------------------------------------- */
/* CNN (cnn.hpp)                                                              */
/* -------------------------------------------------------------------------- */

nk_status_t nk_cnn_create(nk_arena_t* arena, uint32_t num_layers, nk_cnn_t* cnn);
bool nk_cnn_is_valid(const nk_cnn_t* cnn);

/* Conv2D block */
nk_status_t nk_cnn_init_conv_layer(nk_cnn_t* cnn,
                                   uint32_t layer_idx,
                                   int kernel_size,
                                   int stride,
                                   int in_channels,
                                   int out_channels,
                                   float* weights,
                                   float* bias,
                                   nk_conv_activation_t activation,
                                   float leaky_alpha,
                                   int pad_h,
                                   int pad_w,
                                   int pad_h_end,
                                   int pad_w_end);

nk_status_t nk_cnn_init_depthwise_conv_layer(nk_cnn_t* cnn,
                                             uint32_t layer_idx,
                                             int kernel_h,
                                             int kernel_w,
                                             int stride,
                                             int channels,
                                             float* weights,
                                             float* bias,
                                             nk_conv_activation_t activation,
                                             float leaky_alpha,
                                             int pad_h,
                                             int pad_w,
                                             int pad_h_end,
                                             int pad_w_end);

nk_status_t nk_cnn_init_pool_layer(nk_cnn_t* cnn,
                                   uint32_t layer_idx,
                                   int pool_h,
                                   int pool_w,
                                   int stride,
                                   int pad_h,
                                   int pad_w,
                                   int pad_h_end,
                                   int pad_w_end);

nk_status_t nk_cnn_init_avg_pool_layer(nk_cnn_t* cnn,
                                       uint32_t layer_idx,
                                       int pool_h,
                                       int pool_w,
                                       int stride,
                                       int pad_h,
                                       int pad_w,
                                       int pad_h_end,
                                       int pad_w_end);

nk_status_t nk_cnn_init_batch_norm_layer(nk_cnn_t* cnn,
                                         uint32_t layer_idx,
                                         int channels,
                                         float* scale,
                                         float* bias);

nk_status_t nk_cnn_init_layernorm_layer(nk_cnn_t* cnn,
                                        uint32_t layer_idx,
                                        int channels,
                                        float eps,
                                        float* weight,
                                        float* bias);

nk_status_t nk_cnn_init_convnextv2_block_layer(nk_cnn_t* cnn,
                                                 nk_arena_t* arena,
                                                 uint32_t layer_idx,
                                                 uint32_t spatial_h,
                                                 uint32_t spatial_w,
                                                 int channels,
                                                 float eps,
                                                 float* dw_weights,
                                                 float* dw_bias,
                                                 float* ln_weight,
                                                 float* ln_bias,
                                                 float* pw1_weight,
                                                 float* pw1_bias,
                                                 float* grn_gamma,
                                                 float* grn_beta,
                                                 float* pw2_weight,
                                                 float* pw2_bias);

nk_status_t nk_cnn_init_mobilenetv4_uib_layer(nk_cnn_t* cnn,
                                              nk_arena_t* arena,
                                              uint32_t layer_idx,
                                              uint32_t spatial_h,
                                              uint32_t spatial_w,
                                              int in_channels,
                                              int out_channels,
                                              int start_dw_kernel,
                                              int middle_dw_kernel,
                                              int stride,
                                              int middle_dw_downsample,
                                              float expand_ratio,
                                              float* start_dw_weights,
                                              float* start_dw_bias,
                                              float* start_bn_scale,
                                              float* start_bn_bias,
                                              float* expand_weights,
                                              float* expand_bias,
                                              float* expand_bn_scale,
                                              float* expand_bn_bias,
                                              float* middle_dw_weights,
                                              float* middle_dw_bias,
                                              float* middle_bn_scale,
                                              float* middle_bn_bias,
                                              float* proj_weights,
                                              float* proj_bias,
                                              float* proj_bn_scale,
                                              float* proj_bn_bias);

nk_status_t nk_cnn_init_resnet_basic_block_layer(nk_cnn_t* cnn,
                                                 nk_arena_t* arena,
                                                 uint32_t layer_idx,
                                                 uint32_t spatial_h,
                                                 uint32_t spatial_w,
                                                 int in_channels,
                                                 int out_channels,
                                                 int stride,
                                                 float* conv1_weights,
                                                 float* conv1_bias,
                                                 float* bn1_scale,
                                                 float* bn1_bias,
                                                 float* conv2_weights,
                                                 float* conv2_bias,
                                                 float* bn2_scale,
                                                 float* bn2_bias,
                                                 float* shortcut_weights,
                                                 float* shortcut_bias,
                                                 float* shortcut_bn_scale,
                                                 float* shortcut_bn_bias);

nk_status_t nk_cnn_init_yolox_decoupled_head_layer(nk_cnn_t* cnn,
                                                 nk_arena_t* arena,
                                                 uint32_t layer_idx,
                                                 uint32_t spatial_h,
                                                 uint32_t spatial_w,
                                                 int in_channels,
                                                 int hidden_dim,
                                                 int num_classes,
                                                 int num_convs,
                                                 float* stem_weights,
                                                 float* stem_bias,
                                                 float* const* cls_conv_weights,
                                                 float* const* cls_conv_bias,
                                                 float* const* reg_conv_weights,
                                                 float* const* reg_conv_bias,
                                                 float* cls_pred_weights,
                                                 float* cls_pred_bias,
                                                 float* reg_pred_weights,
                                                 float* reg_pred_bias,
                                                 float* obj_pred_weights,
                                                 float* obj_pred_bias);

nk_status_t nk_cnn_init_flatten_layer(nk_cnn_t* cnn, uint32_t layer_idx);

nk_status_t nk_cnn_init_dense_layer(nk_cnn_t* cnn,
                                    uint32_t layer_idx,
                                    const nk_tensor_t* weights,
                                    const nk_tensor_t* bias,
                                    nk_activation_t activation,
                                    float leaky_alpha);

nk_status_t nk_cnn_init_activation_buffers(nk_cnn_t* cnn,
                                           nk_arena_t* arena,
                                           uint32_t in_h,
                                           uint32_t in_w,
                                           uint32_t in_c);
bool nk_cnn_has_activation_buffers(const nk_cnn_t* cnn);

nk_status_t nk_cnn_forward(nk_cnn_t* cnn,
                           nk_arena_t* arena,
                           const nk_tensor_t* input,
                           nk_tensor_t* output);

/* -------------------------------------------------------------------------- */
/* Model loader (.nk format)                                                  */
/* -------------------------------------------------------------------------- */

nk_status_t nk_parse_architecture(const char* nk_path, nk_arch_info_t* info);
nk_status_t nk_parse_architecture_memory(const uint8_t* data,
                                         size_t size,
                                         nk_arch_info_t* info);
nk_status_t nk_arch_print(const char* nk_path);

/** Recommended bump arena size (bytes) for load + one forward pass on CPU builds. Returns 0 on error. */
size_t nk_recommended_arena_bytes(const char* nk_path);

nk_status_t nk_mlp_load(const char* nk_path,
                        nk_arena_t* arena,
                        nk_mlp_t* mlp,
                        nk_arch_info_t* info);
/** Load embedded MLP `.nk` bytes. `data` must outlive the network (flash/blob-backed weights). */
nk_status_t nk_mlp_load_memory(const uint8_t* data,
                               size_t size,
                               nk_arena_t* arena,
                               nk_mlp_t* mlp,
                               nk_arch_info_t* info);
bool nk_mlp_is_quantized(const nk_mlp_t* mlp);

nk_status_t nk_cnn_load(const char* nk_path,
                        nk_arena_t* arena,
                        nk_cnn_t* cnn,
                        nk_arch_info_t* info);
/** Load embedded CNN `.nk` bytes. `data` must outlive the network (flash/blob-backed weights). */
nk_status_t nk_cnn_load_memory(const uint8_t* data,
                               size_t size,
                               nk_arena_t* arena,
                               nk_cnn_t* cnn,
                               nk_arch_info_t* info);
bool nk_cnn_is_quantized(const nk_cnn_t* cnn);

nk_status_t nk_model_load_auto(const char* nk_path,
                               nk_arena_t* arena,
                               nk_network_kind_t* kind,
                               nk_mlp_t* mlp,
                               nk_cnn_t* cnn,
                               nk_arch_info_t* info);

/* High-level loaded model handle (combines MLP or CNN for inference) */
nk_status_t nk_model_load(const char* nk_path, nk_arena_t* arena, nk_model_t* model);
/** Load embedded .nk bytes. `data` must outlive the model (flash .rodata / blob-backed weights). */
nk_status_t nk_model_load_memory(const uint8_t* data,
                                 size_t size,
                                 nk_arena_t* arena,
                                 nk_model_t* model);
nk_status_t nk_model_get_arch(const nk_model_t* model, nk_arch_info_t* info);
uint32_t nk_model_input_count(const nk_model_t* model);
uint32_t nk_model_output_count(const nk_model_t* model);
nk_network_kind_t nk_model_kind(const nk_model_t* model);
bool nk_model_is_quantized(const nk_model_t* model);
nk_status_t nk_model_run(const nk_model_t* model,
                         nk_arena_t* arena,
                         const float* input,
                         uint32_t input_count,
                         float* output,
                         uint32_t output_capacity,
                         uint32_t* output_count);
/** Int8 models only: prequantized int8 input → int8 output (no float quant/dequant). */
nk_status_t nk_model_run_int8(const nk_model_t* model,
                              nk_arena_t* arena,
                              const int8_t* input,
                              uint32_t input_count,
                              int8_t* output,
                              uint32_t output_capacity,
                              uint32_t* output_count);
nk_status_t nk_inspect_model(const char* nk_path, nk_arena_t* arena, nk_inspect_info_t* info);
/** Inspect embedded .nk bytes (same arena peaks as CLI inspect --full for buffer load). */
nk_status_t nk_inspect_model_memory(const uint8_t* data,
                                    size_t size,
                                    nk_arena_t* arena,
                                    nk_inspect_info_t* info);

#if defined(NETKIT_DESKTOP)
/* -------------------------------------------------------------------------- */
/* Regression tests (nk_regression.hpp, test.hpp) — CPU / desktop builds only   */
/* -------------------------------------------------------------------------- */

nk_test_summary_t nk_run_model_tests(const char* nk_path);
nk_test_summary_t nk_run_all_tests(void);

/* -------------------------------------------------------------------------- */
/* CLI (cli.hpp) — CPU / desktop builds only                                  */
/* -------------------------------------------------------------------------- */

int nk_cli_run(int argc, char** argv);
#endif /* NETKIT_DESKTOP */

#ifdef __cplusplus
}
#endif

#endif /* NETKIT_H */
