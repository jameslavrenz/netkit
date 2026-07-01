/*
 * netkit.h — C23 public API for netkit
 *
 * Inference uses float32 only. Documentation:
 *   docs/GETTING_STARTED.md  — build, test, first inference
 *   docs/c-api.md            — full C API reference
 *   docs/API_PARITY.md       — C ↔ C++ symbol map
 *
 * Link against libnetkit.a (C++26 implementation). Compile this header with -std=c23.
 */
#ifndef NETKIT_H
#define NETKIT_H

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
#define NK_MAX_LAYERS      16
#define NK_MAX_PATH_LEN    256
#define NK_MAX_MESSAGE_LEN 128
#define NK_MAX_JSON_STRING 64
#define NK_ARENA_DEFAULT_CAPACITY (64U * 1024U)

#define NK_ARENA_STORAGE_BYTES 32
#define NK_MODEL_STORAGE_BYTES 64
#define NK_MLP_STORAGE_BYTES   16
#define NK_CNN_STORAGE_BYTES   16

/* -------------------------------------------------------------------------- */
/* Status / kinds                                                             */
/* -------------------------------------------------------------------------- */

typedef enum nk_status
{
    NK_OK = 0,
    NK_ERR_JSON_OPEN,
    NK_ERR_BIN_OPEN,
    NK_ERR_JSON_PARSE,
    NK_ERR_UNSUPPORTED_NETWORK,
    NK_ERR_VERSION_MISMATCH,
    NK_ERR_LAYER_CONFIG,
    NK_ERR_BIN_SIZE_MISMATCH,
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
} nk_inspect_info_t;

typedef struct nk_test_summary
{
    uint32_t passed;
    uint32_t failed;
} nk_test_summary_t;

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
void* nk_arena_alloc(nk_arena_t* arena, size_t size);
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
nk_status_t nk_mlp_forward(nk_mlp_t* mlp,
                           nk_arena_t* arena,
                           const nk_tensor_t* input,
                           nk_tensor_t* output);

/* -------------------------------------------------------------------------- */
/* CNN (cnn.hpp)                                                              */
/* -------------------------------------------------------------------------- */

nk_status_t nk_cnn_create(nk_arena_t* arena, uint32_t num_layers, nk_cnn_t* cnn);
bool nk_cnn_is_valid(const nk_cnn_t* cnn);
nk_status_t nk_cnn_init_layer(nk_cnn_t* cnn,
                              uint32_t layer_idx,
                              int kernel_size,
                              int stride,
                              int in_channels,
                              int out_channels,
                              float* weights,
                              float* bias,
                              nk_conv_activation_t activation,
                              float leaky_alpha);
nk_status_t nk_cnn_forward(nk_cnn_t* cnn,
                           nk_arena_t* arena,
                           const nk_tensor_t* input,
                           nk_tensor_t* output);

/* -------------------------------------------------------------------------- */
/* Model loader (model_loader.hpp)                                            */
/* -------------------------------------------------------------------------- */

nk_status_t nk_parse_architecture(const char* json_path, nk_arch_info_t* info);
void nk_arch_print(const char* json_path);

bool nk_json_path_to_bin_path(const char* json_path, char* bin_path, size_t bin_path_capacity);

nk_status_t nk_load_weights_bin(const char* json_path,
                                nk_arena_t* arena,
                                float** weights,
                                size_t* float_count);

nk_status_t nk_mlp_load(const char* json_path,
                        nk_arena_t* arena,
                        nk_mlp_t* mlp,
                        nk_arch_info_t* info);

nk_status_t nk_cnn_load(const char* json_path,
                        nk_arena_t* arena,
                        nk_cnn_t* cnn,
                        nk_arch_info_t* info);

nk_status_t nk_model_load_auto(const char* json_path,
                               nk_arena_t* arena,
                               nk_network_kind_t* kind,
                               nk_mlp_t* mlp,
                               nk_cnn_t* cnn,
                               nk_arch_info_t* info);

/* High-level loaded model handle (combines MLP or CNN for inference) */
nk_status_t nk_model_load(const char* json_path, nk_arena_t* arena, nk_model_t* model);
nk_status_t nk_model_get_arch(const nk_model_t* model, nk_arch_info_t* info);
uint32_t nk_model_input_count(const nk_model_t* model);
uint32_t nk_model_output_count(const nk_model_t* model);
nk_network_kind_t nk_model_kind(const nk_model_t* model);
nk_status_t nk_model_run(const nk_model_t* model,
                         nk_arena_t* arena,
                         const float* input,
                         uint32_t input_count,
                         float* output,
                         uint32_t output_capacity,
                         uint32_t* output_count);
nk_status_t nk_inspect_model(const char* json_path, nk_arena_t* arena, nk_inspect_info_t* info);

/* -------------------------------------------------------------------------- */
/* Vectors / tests (vectors_loader.hpp, test.hpp)                             */
/* -------------------------------------------------------------------------- */

nk_test_summary_t nk_run_vectors_file(const char* vectors_path);
nk_test_summary_t nk_run_all_tests(void);

/* -------------------------------------------------------------------------- */
/* CLI (cli.hpp)                                                              */
/* -------------------------------------------------------------------------- */

int nk_cli_run(int argc, char** argv);

#ifdef __cplusplus
}
#endif

#endif /* NETKIT_H */
