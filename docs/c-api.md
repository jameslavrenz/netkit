# C API Reference (C23)

Public header: [`include/netkit.h`](../include/netkit.h)

Compile user code with `-std=c23`. Link against `libnetkit.a` using a C++ linker driver.

Every function listed here mirrors a C++26 entry point. See [`API_PARITY.md`](API_PARITY.md) for the full symbol map and contribution policy.

## Version

```c
#define NK_VERSION_MAJOR 0
#define NK_VERSION_MINOR 1
#define NK_VERSION_PATCH 0

const char* nk_version_string(void);  // "0.1.0"
```

## Constants

| Macro | Value | Description |
|-------|-------|-------------|
| `NK_MAX_TENSOR_RANK` | 4 | Max tensor rank |
| `NK_MAX_LAYERS` | 16 | Max layers in architecture metadata |
| `NK_MAX_PATH_LEN` | 256 | Path buffer size used internally |
| `NK_MAX_MESSAGE_LEN` | 128 | Max length of last error message |
| `NK_ARENA_DEFAULT_CAPACITY` | 65536 | Default arena size (64 KiB) |
| `NK_ARENA_STORAGE_BYTES` | 32 | Size of `nk_arena_t.storage` |
| `NK_MODEL_STORAGE_BYTES` | 64 | Size of `nk_model_t.storage` |
| `NK_MLP_STORAGE_BYTES` | 16 | Size of `nk_mlp_t.storage` |
| `NK_CNN_STORAGE_BYTES` | 16 | Size of `nk_cnn_t.storage` |

## Types

### `nk_status_t`

| Value | Meaning |
|-------|---------|
| `NK_OK` | Success |
| `NK_ERR_JSON_OPEN` | Could not open `.json` file |
| `NK_ERR_BIN_OPEN` | Could not open `.bin` file |
| `NK_ERR_JSON_PARSE` | JSON syntax or schema error |
| `NK_ERR_UNSUPPORTED_NETWORK` | Unknown `network` field |
| `NK_ERR_VERSION_MISMATCH` | JSON `version` is not `1` |
| `NK_ERR_LAYER_CONFIG` | Invalid layer or input shape |
| `NK_ERR_BIN_SIZE_MISMATCH` | Weight count does not match architecture |
| `NK_ERR_ARENA_OVERFLOW` | Arena exhausted |
| `NK_ERR_INVALID_ARGUMENT` | Null pointer or size mismatch |
| `NK_ERR_BUFFER_TOO_SMALL` | Output buffer too small |
| `NK_ERR_MODEL_NOT_LOADED` | Model handle not initialized |
| `NK_ERR_NOT_INITIALIZED` | Network handle not created |

```c
const char* nk_status_string(nk_status_t status);
const char* nk_last_error(void);  // detail after failed call; thread-local
```

### `nk_dtype_t`, `nk_activation_t`, `nk_conv_activation_t`

Mirror C++ `DataType`, `ActivationType`, and `ConvActivationType`. **Only `NK_DTYPE_FLOAT32` is used for inference today** — see [DATATYPES.md](DATATYPES.md).

### `nk_cnn_block_type_t`

| Value | C++ `CnnBlockType` |
|-------|---------------------|
| `NK_CNN_BLOCK_CONV2D` | `Conv2D` |
| `NK_CNN_BLOCK_MAX_POOL2D` | `MaxPool2D` |
| `NK_CNN_BLOCK_FLATTEN` | `Flatten` |
| `NK_CNN_BLOCK_DENSE` | `Dense` |

Used when building CNN pipelines manually. File-loaded models (`nk_cnn_load`) configure blocks from JSON automatically.

### `nk_tensor_t`, `nk_conv2d_t`

Mirror C++ `Tensor` and `Conv2D` layouts. Safe to pass by pointer to all `nk_tensor_*` and `nk_ops_*` functions.

### `nk_mlp_t`, `nk_cnn_t`

Opaque handles for manually constructed or file-loaded networks.

### `nk_network_kind_t`

| Value | Description |
|-------|-------------|
| `NK_NETWORK_UNKNOWN` | Not set |
| `NK_NETWORK_MLP` | Fully connected network |
| `NK_NETWORK_CNN` | Convolutional network |

### `nk_arena_t`

Opaque stack-allocatable handle. Wraps the internal bump allocator.

```c
typedef struct nk_arena {
    alignas(max_align_t) unsigned char storage[NK_ARENA_STORAGE_BYTES];
} nk_arena_t;
```

### `nk_model_t`

Opaque handle for a loaded MLP or CNN.

```c
typedef struct nk_model {
    alignas(max_align_t) unsigned char storage[NK_MODEL_STORAGE_BYTES];
} nk_model_t;
```

### `nk_arch_info_t`

Architecture metadata from JSON (no weights required).

```c
typedef struct nk_arch_info {
    uint32_t version;
    nk_network_kind_t kind;
    uint32_t input_shape[NK_MAX_TENSOR_RANK];
    uint32_t input_rank;
    uint32_t num_layers;
    size_t expected_weight_floats;
    uint32_t input_elements;   // product of input_shape
    uint32_t output_elements;  // computed from architecture
} nk_arch_info_t;
```

### `nk_inspect_info_t`

Result of `nk_inspect_model()`.

```c
typedef struct nk_inspect_info {
    nk_arch_info_t arch;
    size_t weight_floats;
    size_t arena_bytes_after_load;
    size_t arena_bytes_after_forward;
    size_t arena_remaining;
} nk_inspect_info_t;
```

## Arena functions

```c
void nk_arena_init(nk_arena_t* arena, void* memory, size_t size);
void* nk_arena_alloc(nk_arena_t* arena, size_t size, size_t alignment);
void nk_arena_reset(nk_arena_t* arena);
size_t nk_arena_capacity(const nk_arena_t* arena);
size_t nk_arena_used(const nk_arena_t* arena);
size_t nk_arena_remaining(const nk_arena_t* arena);
```

### Alignment

`nk_arena_alloc` is a **bump allocator with explicit alignment**. When the current offset is not a multiple of `alignment`, padding bytes are skipped before the returned pointer. `alignment` must be a power of two (e.g. `4`, `8`, `16`).

| Use | `alignment` |
|-----|-------------|
| Raw float buffers, tensor payload | `4` or `alignof(float)` |
| Structs, pointers, `max_align_t` types | `8` or `alignof(T)` on 64-bit targets |

Returns `NULL` when the arena is uninitialized, arguments are invalid (`size == 0`, bad alignment), or the arena is full.

**Backing memory** passed to `nk_arena_init` should be declared `alignas(max_align_t)`.

**Sizing:** pass whatever `size` your model needs — there is no arena field in model JSON. Default examples use `NK_ARENA_DEFAULT_CAPACITY` (64 KiB). MNIST needs multi-MiB buffers. Use `nk_inspect_model()` and read `arena_bytes_after_forward` in `nk_inspect_info_t`, then allocate static storage with headroom. See [ARENA.md](ARENA.md).

Model load / run APIs allocate internally with the correct alignment; you only need `nk_arena_alloc` when building custom integrations on top of the C API.

## Tensor, ops, conv, MLP, CNN

Full signatures are in [`netkit.h`](../include/netkit.h). Each group mirrors the C++ headers listed in [cpp-api.md](cpp-api.md).

### Tensor factory (`tensor_factory.hpp`)

| C function | C++ equivalent |
|------------|----------------|
| `nk_tensor_create_2d` | `TensorFactory::Create2D` |
| `nk_tensor_create_nd` | `TensorFactory::CreateND` |
| `nk_tensor_view_2d` | `TensorFactory::View2D` |
| `nk_tensor_fill` | `TensorFactory::Fill` |
| `nk_tensor_print` | `TensorFactory::Print` |
| `nk_tensor_print_labeled` | `TensorFactory::PrintLabeled` |

### Tensor access (`tensor_access.hpp`)

| C function | C++ equivalent |
|------------|----------------|
| `nk_tensor_data_f32` | `tensor_data_f32` |
| `nk_tensor_data_f32_const` | `tensor_data_f32` (const) |
| `nk_tensor_index_nhwc` | `index_nhwc` |

### Ops (`ops.hpp`)

| C function | C++ equivalent |
|------------|----------------|
| `nk_ops_is_elementwise_valid` | `IsElementwiseValid` |
| `nk_ops_check_same_shape_2d` | `CheckSameShape2D` |
| `nk_ops_check_same_shape_nd` | `CheckSameShapeND` |
| `nk_ops_is_matmul_valid` | `IsMatMulValid` |
| `nk_ops_is_elementwise_valid_nd` | `IsElementwiseValidND` |
| `nk_ops_is_unary_op_valid` | `IsUnaryOpValid` |
| `nk_ops_mul` | `Mul` |
| `nk_ops_mul_scalar` | `MulScalar` |
| `nk_ops_mat_add` | `MatAdd` |
| `nk_ops_mat_add_nd` | `MatAddND` |
| `nk_ops_mat_mul` | `MatMul` |
| `nk_ops_mul_nd` | `MulND` |
| `nk_ops_relu` | `ReLU` |
| `nk_ops_sigmoid` | `Sigmoid` |
| `nk_ops_tanh` | `Tanh` |
| `nk_ops_leaky_relu` | `LeakyReLU` |
| `nk_ops_relu6` | `ReLU6` |
| `nk_ops_softmax` | `Softmax` |

### MLP (`mlp.hpp`)

| C function | C++ equivalent |
|------------|----------------|
| `nk_mlp_create` | `MLPNetwork` constructor |
| `nk_mlp_is_valid` | `MLPNetwork::IsValid` |
| `nk_mlp_init_layer` | `MLPNetwork::InitLayer` |
| `nk_mlp_forward` | `MLPNetwork::forward` |

### CNN (`cnn.hpp`)

| C function | C++ equivalent |
|------------|----------------|
| `nk_cnn_create` | `CNNNetwork` constructor |
| `nk_cnn_is_valid` | `CNNNetwork::IsValid` |
| `nk_cnn_init_conv_layer` | `CNNNetwork::InitConvLayer` |
| `nk_cnn_init_pool_layer` | `CNNNetwork::InitPoolLayer` |
| `nk_cnn_init_flatten_layer` | `CNNNetwork::InitFlattenLayer` |
| `nk_cnn_init_dense_layer` | `CNNNetwork::InitDenseLayer` |
| `nk_cnn_init_layer` | `CNNNetwork::InitLayer` (conv alias) |
| `nk_cnn_forward` | `CNNNetwork::forward` |

### Conv2D (`conv2d.hpp`)

| C function | C++ equivalent |
|------------|----------------|
| `nk_conv2d_forward` | `Conv2D::forward` |

### CNN pipeline (manual construction)

Hybrid CNN models (conv → max pool → flatten → dense) use:

```c
nk_cnn_init_conv_layer(cnn, idx, kernel, stride, in_c, out_c, w, b, act, alpha);
nk_cnn_init_pool_layer(cnn, idx, pool_size, stride);
nk_cnn_init_flatten_layer(cnn, idx);
nk_cnn_init_dense_layer(cnn, idx, &weights, &bias, NK_ACTIVATION_RELU, 0.01f);
```

`nk_cnn_init_layer` is a backward-compatible alias for `nk_cnn_init_conv_layer`.

For file-based models (including `models/mnist_cnn.json`), use `nk_cnn_load` or `nk_model_load` — all block types are configured from JSON.

## Model loader

| C function | C++ equivalent | Notes |
|------------|----------------|-------|
| `nk_parse_architecture` | `ParseArchitecture` + `FillArchInfo` | Populates `nk_arch_info_t`; `output_elements` uses `ComputeMlpOutputElements` / `ComputeCnnOutputElements` |
| `nk_arch_print` | `PrintNetworkSummary` | Boxed network summary to stdout; returns `nk_status_t` |
| `nk_json_path_to_bin_path` | `JsonPathToBinPath` | |
| `nk_load_weights_bin` | `LoadWeightsBin` | |
| `nk_mlp_load` | `LoadMLP` | |
| `nk_cnn_load` | `LoadCNN` | Supports conv / pool / flatten / dense JSON |
| `nk_model_load_auto` | `Load` | Dispatches by JSON `network` field |

```c
nk_status_t nk_parse_architecture(const char* json_path, nk_arch_info_t* info);
nk_status_t nk_arch_print(const char* json_path);
bool nk_json_path_to_bin_path(const char* json_path, char* bin_path, size_t bin_path_capacity);
nk_status_t nk_load_weights_bin(const char* json_path, nk_arena_t* arena, float** weights, size_t* float_count);
nk_status_t nk_mlp_load(const char* json_path, nk_arena_t* arena, nk_mlp_t* mlp, nk_arch_info_t* info);
nk_status_t nk_cnn_load(const char* json_path, nk_arena_t* arena, nk_cnn_t* cnn, nk_arch_info_t* info);
nk_status_t nk_model_load_auto(const char* json_path, nk_arena_t* arena, nk_network_kind_t* kind,
                               nk_mlp_t* mlp, nk_cnn_t* cnn, nk_arch_info_t* info);
```

High-level combined handle:

```c
nk_status_t nk_model_load(const char* json_path, nk_arena_t* arena, nk_model_t* model);
nk_status_t nk_model_get_arch(const nk_model_t* model, nk_arch_info_t* info);
uint32_t nk_model_input_count(const nk_model_t* model);
uint32_t nk_model_output_count(const nk_model_t* model);
nk_network_kind_t nk_model_kind(const nk_model_t* model);
nk_status_t nk_model_run(...);
nk_status_t nk_inspect_model(...);
```

**C++-only diagnostics (no C binding):** `PrintArchitecture`, `PrintWeightsSummary` — used by `./netkit inspect --full`. See [API_PARITY.md](API_PARITY.md).

## Tests and CLI

```c
nk_test_summary_t nk_run_vectors_file(const char* vectors_path);
nk_test_summary_t nk_run_all_tests(void);
int nk_cli_run(int argc, char** argv);
```

Vectors file format: [VECTORS_TESTS.md](VECTORS_TESTS.md). CLI commands: [CLI.md](CLI.md).

## Model load and inference

```c
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
```

These mirror querying a loaded C++ network after `ModelLoader::Load` — input/output counts come from the same `ComputeMlpOutputElements` / `ComputeCnnOutputElements` logic used when filling `nk_arch_info_t`.

### `nk_model_load`

Loads architecture and weights from `json_path` (companion `.bin` resolved automatically). Allocates network state from `arena`.

### `nk_model_run`

Runs one forward pass.

- `input_count` must equal `nk_model_input_count(model)`
- `output_capacity` must be ≥ `nk_model_output_count(model)`
- On success, `*output_count` is set to the number of floats written

**Input layout**

| Network | Flat input order |
|---------|------------------|
| MLP | Row-major `[batch, features]` flattened |
| CNN | NHWC `[H, W, C]` flattened |

## Inspection

```c
nk_status_t nk_inspect_model(const char* json_path, nk_arena_t* arena, nk_inspect_info_t* info);
```

Loads the model, runs a zero-input forward pass, and reports arena high-water marks. C++ equivalent: load via `ModelLoader::Load`, run forward, read `arena.offset` — or use `./netkit inspect --full`. Use this to size embedded memory regions.

## Complete examples

| File | Build | Description |
|------|-------|-------------|
| [`examples/infer_c.c`](../examples/infer_c.c) | `make example-c` | High-level `nk_model_load` / `nk_model_run` |
| [`examples/infer_cpp.cpp`](../examples/infer_cpp.cpp) | `make example-cpp` | Native C++26 API (see [cpp-api.md](cpp-api.md)) |

Model JSON and weight layout: [MODEL_FORMAT.md](MODEL_FORMAT.md).

### Minimal C example

```c
#include <stdalign.h>
#include <stdio.h>
#include "netkit.h"

int main(void)
{
    alignas(max_align_t) static unsigned char mem[NK_ARENA_DEFAULT_CAPACITY];
    nk_arena_t arena;
    nk_model_t model;

    nk_arena_init(&arena, mem, sizeof(mem));

    if (nk_model_load("models/test_mlp.json", &arena, &model) != NK_OK)
    {
        printf("load error: %s\n", nk_last_error());
        return 1;
    }

    float in[] = {1.0f, 2.0f};
    float out[2];
    uint32_t out_n = 0;

    if (nk_model_run(&model, &arena, in, 2, out, 2, &out_n) != NK_OK)
    {
        printf("run error: %s\n", nk_last_error());
        return 1;
    }

    printf("output: %.4f, %.4f\n", out[0], out[1]);
    return 0;
}
```

## Implementation notes

- The C API is implemented in `src/netkit_api.cpp` (C++26) using `extern "C"` linkage.
- `nk_arena_t` and `nk_model_t` use fixed-size opaque storage so handles can live on the stack with no heap allocation.
- Path resolution tries the given path, then `../<path>` (for running from build subdirectories).
