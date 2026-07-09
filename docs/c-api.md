# C API Reference (C23)

Public header: [`include/netkit.h`](../include/netkit.h)  
Configuration: [`include/netkit_config.h`](../include/netkit_config.h)

The C23 API mirrors the C++26 engine for **embedded firmware** — load `.nk` via the interpreter path (`nk_model_load`, `nk_model_load_memory`) or generated **AOT** wrappers. Compile user code with `-std=c23`. Link against `libnetkit.a` using a C++ linker driver.

Every function listed here mirrors a C++26 entry point. See [`API_PARITY.md`](API_PARITY.md) for the full symbol map and contribution policy.

Overview for new users: [GETTING_STARTED.md](GETTING_STARTED.md). Philosophy and deployment modes: [PHILOSOPHY.md](PHILOSOPHY.md).

## Build configuration

Set the deployment target when building or integrating netkit:

| Makefile | `-D` macro | Role |
|----------|------------|------|
| `NETKIT_TARGET=cpu` (default) | `NETKIT_TARGET_CPU` | Desktop — CLI, regression |
| `NETKIT_TARGET=mcu` | `NETKIT_TARGET_MCU` | Lean firmware runtime |
| `NETKIT_TARGET=mpu` | `NETKIT_TARGET_MPU` | Lean firmware runtime |

| Makefile flag | Macro | Effect |
|---------------|-------|--------|
| *(CPU default)* | `NETKIT_ARENA_HEAP` | `nk_arena_init_heap()` available; default CPU examples use heap |
| `NETKIT_GLOBAL_ARENA=1` (CPU) | `NETKIT_GLOBAL_ARENA` | Static arena only on CPU; no heap helpers |
| `NETKIT_HEAP_ARENA=1` (MCU/MPU) | `NETKIT_HEAP_ARENA` → `NETKIT_ARENA_HEAP` | Optional heap API on embedded |

| `NK_ARENA_DEFAULT_CAPACITY` | MCU | CPU / MPU |
|-----------------------------|-----|-----------|
| Value | **64 KiB** | **64 MiB** |

Full guide: [BUILD_TARGETS.md](BUILD_TARGETS.md).

### Desktop-only symbols (`NETKIT_DESKTOP`)

Available only when `NETKIT_TARGET=cpu`:

```c
nk_test_summary_t nk_run_model_tests(const char* nk_path);
nk_test_summary_t nk_run_all_tests(void);
int nk_cli_run(int argc, char** argv);
```

### Heap arena symbols (`NETKIT_ARENA_HEAP`)

When compiled in (CPU default, or MCU/MPU with `NETKIT_HEAP_ARENA=1`):

```c
nk_status_t nk_arena_init_heap(nk_arena_t* arena, size_t capacity);
void nk_arena_destroy_heap(nk_arena_t* arena);
```

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
| `NK_MAX_LAYERS` | 100 | Max layers in `.nk` files (matches `NkFormat::kMaxLayers`) |
| `NK_MAX_PATH_LEN` | 256 | Path buffer size used internally |
| `NK_MAX_MESSAGE_LEN` | 128 | Max length of last error message |
| `NK_ARENA_DEFAULT_CAPACITY` | 64 KiB (MCU) / 64 MiB (CPU, MPU) | Default static arena size by build target |
| `NK_ARENA_STORAGE_BYTES` | 64 | Size of `nk_arena_t.storage` |
| `NK_MODEL_STORAGE_BYTES` | 96 | Size of `nk_model_t.storage` |
| `NK_MLP_STORAGE_BYTES` | 16 | Size of `nk_mlp_t.storage` |
| `NK_CNN_STORAGE_BYTES` | 16 | Size of `nk_cnn_t.storage` |

## Types

### `nk_status_t`

| Value | Meaning |
|-------|---------|
| `NK_OK` | Success |
| `NK_ERR_MODEL_OPEN` | Could not open model file |
| `NK_ERR_MODEL_READ` | Model read failed |
| `NK_ERR_MODEL_PARSE` | Invalid or truncated `.nk` file |
| `NK_ERR_UNSUPPORTED_NETWORK` | Unknown network kind |
| `NK_ERR_VERSION_MISMATCH` | Unsupported `.nk` version |
| `NK_ERR_LAYER_CONFIG` | Invalid layer descriptor |
| `NK_ERR_WEIGHT_MISMATCH` | Weight payload size mismatch |
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
| `NK_CNN_BLOCK_DEPTHWISE_CONV2D` | `DepthwiseConv2D` |
| `NK_CNN_BLOCK_MAX_POOL2D` | `MaxPool2D` |
| `NK_CNN_BLOCK_AVG_POOL2D` | `AvgPool2D` |
| `NK_CNN_BLOCK_BATCH_NORM2D` | `BatchNorm2d` |
| `NK_CNN_BLOCK_LAYERNORM2D` | `LayerNorm2d` |
| `NK_CNN_BLOCK_FLATTEN` | `Flatten` |
| `NK_CNN_BLOCK_DENSE` | `Dense` |
| `NK_CNN_BLOCK_CONVNEXTV2_BLOCK` | `ConvNeXtV2Block` |
| `NK_CNN_BLOCK_MOBILENETV4_UIB` | `MobilenetV4Uib` |
| `NK_CNN_BLOCK_RESNET_BASIC_BLOCK` | `ResNetBasicBlock` |
| `NK_CNN_BLOCK_YOLOX_DECOUPLED_HEAD` | `YoloxDecoupledHead` |

Numeric values match C++ `CnnBlockType` member order (starting at 0).

Used when building CNN pipelines manually. File-loaded models (`nk_cnn_load`) configure blocks from the `.nk` layer list.

### `nk_tensor_t`, `nk_conv2d_t`

Mirror C++ `Tensor` and `Conv2D` layouts. Safe to pass by pointer to all `nk_tensor_*` and `nk_ops_*` functions.

`nk_conv2d_t` mirrors C++ `Conv2D`: `pad_h`/`pad_w` are start padding; `pad_h_end`/`pad_w_end` are end padding. Pass `NK_PAD_MIRROR` (`-1`) for end padding to mirror the start (symmetric). Set all four explicitly for asymmetric padding.

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

Architecture metadata parsed from a `.nk` file (no full weight load required).

```c
typedef struct nk_arch_info {
    uint32_t version;  /* .nk format version (currently 2) */
    nk_network_kind_t kind;
    uint32_t input_shape[NK_MAX_TENSOR_RANK];
    uint32_t input_rank;
    uint32_t num_layers;
    size_t expected_weight_floats;
    size_t weights_bytes;
    size_t biases_bytes;
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
    size_t flash_payload_bytes;  /* weight+bias payload kept in flash/blob */
} nk_inspect_info_t;
```

## Arena functions

```c
void nk_arena_init(nk_arena_t* arena, void* memory, size_t size);
#if defined(NETKIT_ARENA_HEAP)   /* CPU default; MCU/MPU when NETKIT_HEAP_ARENA=1 */
nk_status_t nk_arena_init_heap(nk_arena_t* arena, size_t capacity);
void nk_arena_destroy_heap(nk_arena_t* arena);
#endif
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

**Backing memory** passed to `nk_arena_init` should be declared `alignas(max_align_t)`. This is the **default on MCU and MPU**, and on CPU when built with `NETKIT_GLOBAL_ARENA=1`. The default **CPU** build uses `nk_arena_init_heap()` instead — see [BUILD_TARGETS.md](BUILD_TARGETS.md).

`nk_arena_init_heap()` performs **one** `malloc` for the backing buffer; inference uses bump allocation inside it. `nk_arena_destroy_heap()` frees that buffer on **CPU only**; on MCU/MPU it is a no-op.

**Sizing:** CPU examples use `NK_ARENA_DEFAULT_CAPACITY` (**64 MiB**). MCU uses **64 KiB**. For custom firmware, use `nk_inspect_model()` and read `arena_bytes_after_forward`. Coefs stay in flash — use `flash_payload_bytes` plus arena peaks (see [ARENA.md](ARENA.md)).

Model load / run APIs allocate internally with the correct alignment; you only need `nk_arena_alloc` when building custom integrations on top of the C API.

## Tensor, ops, conv, MLP, CNN

Full signatures are in [`netkit.h`](../include/netkit.h). Each group mirrors the C++ headers listed in [cpp-api.md](cpp-api.md).

### Tensor factory (`tensor_factory.hpp`)

| C function | C++ equivalent |
|------------|----------------|
| `nk_tensor_create_2d` | `TensorFactory::Create2D` |
| `nk_tensor_create_nd` | `TensorFactory::CreateND` |
| `nk_tensor_view_2d` | `TensorFactory::View2D` |
| `nk_tensor_fill` | `TensorFactory::Fill` (`std::span<const float>`) |
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
| `nk_mlp_is_quantized` | `MLPNetwork::IsQuantized` |
| `nk_mlp_init_layer` | `MLPNetwork::InitLayer` |
| `nk_mlp_init_activation_buffers` | `MLPNetwork::InitActivationBuffers` |
| `nk_mlp_has_activation_buffers` | `MLPNetwork::HasActivationBuffers` |
| `nk_mlp_forward` | `MLPNetwork::forward` |

#### MLP manual construction (call order)

1. `nk_arena_init` — bind memory.
2. `nk_mlp_create(&arena, num_layers, &mlp)` — allocate layer table.
3. `nk_mlp_init_layer(&mlp, i, &weights, &bias, activation, leaky_alpha)` for each `i` in order.
4. `nk_mlp_init_activation_buffers(&mlp, &arena, batch_rows)` — **after all layers** (`batch_rows` usually `1`).
5. `nk_mlp_forward(&mlp, &arena, &input, &output)` — supply output tensor storage.

```c
alignas(max_align_t) static unsigned char memory[65536];
nk_arena_t arena;
nk_arena_init(&arena, memory, sizeof(memory));

nk_mlp_t mlp;
nk_mlp_create(&arena, 2, &mlp);
nk_mlp_init_layer(&mlp, 0, &w0, &b0, NK_ACTIVATION_RELU, 0.01f);
nk_mlp_init_layer(&mlp, 1, &w1, &b1, NK_ACTIVATION_NONE, 0.01f);
nk_mlp_init_activation_buffers(&mlp, &arena, 1);

nk_tensor_t input, output;
nk_tensor_view_2d(in_data, 1, in_features, &input);
nk_tensor_view_2d(out_data, 1, out_features, &output);
nk_mlp_forward(&mlp, &arena, &input, &output);
```

C++ equivalent: [cpp-api.md](cpp-api.md#manual-construction-call-order).

### CNN (`cnn.hpp`)

| C function | C++ equivalent |
|------------|----------------|
| `nk_cnn_create` | `CNNNetwork` constructor |
| `nk_cnn_is_valid` | `CNNNetwork::IsValid` |
| `nk_cnn_is_quantized` | `CNNNetwork::IsQuantized` |
| `nk_cnn_init_conv_layer` | `CNNNetwork::InitConvLayer` |
| `nk_cnn_init_depthwise_conv_layer` | `CNNNetwork::InitDepthwiseConvLayer` |
| `nk_cnn_init_pool_layer` | `CNNNetwork::InitPoolLayer` |
| `nk_cnn_init_avg_pool_layer` | `CNNNetwork::InitAvgPoolLayer` |
| `nk_cnn_init_batch_norm_layer` | `CNNNetwork::InitBatchNormLayer` |
| `nk_cnn_init_layernorm_layer` | `CNNNetwork::InitLayerNormLayer` |
| `nk_cnn_init_convnextv2_block_layer` | `CNNNetwork::InitConvNeXtV2BlockLayer` |
| `nk_cnn_init_mobilenetv4_uib_layer` | `CNNNetwork::InitMobilenetV4UibLayer` |
| `nk_cnn_init_resnet_basic_block_layer` | `CNNNetwork::InitResNetBasicBlockLayer` |
| `nk_cnn_init_yolox_decoupled_head_layer` | `CNNNetwork::InitYoloxDecoupledHeadLayer` |
| `nk_cnn_init_flatten_layer` | `CNNNetwork::InitFlattenLayer` |
| `nk_cnn_init_dense_layer` | `CNNNetwork::InitDenseLayer` |
| `nk_cnn_init_activation_buffers` | `CNNNetwork::InitActivationBuffers` |
| `nk_cnn_has_activation_buffers` | `CNNNetwork::HasActivationBuffers` |
| `nk_cnn_forward` | `CNNNetwork::forward` |

```c
nk_status_t nk_cnn_init_conv_layer(nk_cnn_t* cnn, uint32_t layer_idx,
    int kernel_size, int stride, int in_channels, int out_channels,
    float* weights, float* bias, nk_conv_activation_t activation, float leaky_alpha,
    int pad_h, int pad_w, int pad_h_end, int pad_w_end);
nk_status_t nk_cnn_init_pool_layer(nk_cnn_t* cnn, uint32_t layer_idx,
    int pool_h, int pool_w, int stride, int pad_h, int pad_w,
    int pad_h_end, int pad_w_end);
nk_status_t nk_cnn_init_avg_pool_layer(nk_cnn_t* cnn, uint32_t layer_idx,
    int pool_h, int pool_w, int stride, int pad_h, int pad_w,
    int pad_h_end, int pad_w_end);
nk_status_t nk_cnn_init_batch_norm_layer(nk_cnn_t* cnn, uint32_t layer_idx,
    int channels, float* scale, float* bias);
```

### Conv2D (`conv2d.hpp`)

| C function | C++ equivalent |
|------------|----------------|
| `nk_conv2d_forward` | `Conv2D::forward` |

#### CNN manual construction (call order)

1. `nk_arena_init` — bind memory.
2. `nk_cnn_create(&arena, num_layers, &cnn)` — allocate block table.
3. `nk_cnn_init_*_layer(&cnn, …)` for each **`layer_idx` from `0` to `num_layers - 1` in forward order** — one init function per block type (see `nk_cnn_block_type_t`). Composite inits (`nk_cnn_init_convnextv2_block_layer`, `nk_cnn_init_mobilenetv4_uib_layer`, `nk_cnn_init_resnet_basic_block_layer`, `nk_cnn_init_yolox_decoupled_head_layer`) also take `&arena` and **`spatial_h` / `spatial_w`** (feature-map size at that layer's input); fused scratch is allocated during the init call.
4. `nk_cnn_init_activation_buffers(&cnn, &arena, in_h, in_w, in_c)` — **after all layers**; `in_h`, `in_w`, `in_c` are the **network input** NHWC shape (same tensor you pass to forward). Allocates ping-pong buffers and CMSIS kernel workspace when applicable.
5. `nk_cnn_forward(&cnn, &arena, &input, &output)` — `input` rank-3 NHWC until flatten; `output` is filled from the network result.

**Primitive hybrid pipeline** (conv → pool → batch norm → flatten → dense):

```c
nk_cnn_create(&arena, 5, &cnn);
nk_cnn_init_conv_layer(&cnn, 0, 3, 1, 3, 16, conv_w, conv_b, NK_CONV_ACTIVATION_RELU, 0.01f,
                       1, 1, NK_PAD_MIRROR, NK_PAD_MIRROR);
nk_cnn_init_pool_layer(&cnn, 1, 2, 2, 2, 0, 0, NK_PAD_MIRROR, NK_PAD_MIRROR);
nk_cnn_init_batch_norm_layer(&cnn, 2, 16, bn_scale, bn_bias);
nk_cnn_init_flatten_layer(&cnn, 3);
nk_cnn_init_dense_layer(&cnn, 4, &W, &B, NK_ACTIVATION_NONE, 0.01f);
nk_cnn_init_activation_buffers(&cnn, &arena, in_h, in_w, in_c);
nk_cnn_forward(&cnn, &arena, &input, &output);
```

**YOLOX decoupled head only** (`NK_CNN_BLOCK_YOLOX_DECOUPLED_HEAD` = 11):

```c
nk_cnn_create(&arena, 1, &cnn);
nk_cnn_init_yolox_decoupled_head_layer(
    &cnn, &arena, 0,
    spatial_h, spatial_w,       /* feature map at head input, e.g. 2, 2 */
    in_channels, hidden_dim, num_classes, num_convs,
    stem_w, stem_b,
    cls_conv_w, cls_conv_b,     /* float* arrays[length num_convs] */
    reg_conv_w, reg_conv_b,
    cls_pred_w, cls_pred_b,
    reg_pred_w, reg_pred_b,
    obj_pred_w, obj_pred_b);
nk_cnn_init_activation_buffers(&cnn, &arena, spatial_h, spatial_w, in_channels);

uint32_t shape[3] = {spatial_h, spatial_w, in_channels};
nk_tensor_t input, output = {0};
nk_tensor_create_nd(&arena, 3, shape, &input);
nk_cnn_forward(&cnn, &arena, &input, &output);
/* output: [spatial_h, spatial_w, 4 + 1 + num_classes] NHWC */
```

Full signatures: [`netkit.h`](../include/netkit.h). C++ equivalent: [cpp-api.md](cpp-api.md#manual-construction-call-order-1). Composite block details: [YOLOX.md](YOLOX.md#manual-construction), [RESNET18.md](RESNET18.md), [MOBILENETV4.md](MOBILENETV4.md), [CONVNEXTV2.md](CONVNEXTV2.md).

Prefer **`nk_cnn_load` / `nk_model_load`** when weights come from a `.nk` file or embedded blob — the loader runs steps 2–4 for you.

### CNN pipeline (primitive layer inits)

Hybrid models can mix the init helpers below:

```c
nk_cnn_init_conv_layer(cnn, idx, kernel, stride, in_c, out_c, w, b, act, alpha,
                       pad_h, pad_w, NK_PAD_MIRROR, NK_PAD_MIRROR);
nk_cnn_init_pool_layer(cnn, idx, pool_h, pool_w, stride, pad_h, pad_w,
                       NK_PAD_MIRROR, NK_PAD_MIRROR);  /* max pool */
nk_cnn_init_avg_pool_layer(cnn, idx, pool_h, pool_w, stride, pad_h, pad_w,
                           NK_PAD_MIRROR, NK_PAD_MIRROR);
nk_cnn_init_batch_norm_layer(cnn, idx, channels, scale, bias);
nk_cnn_init_flatten_layer(cnn, idx);
nk_cnn_init_dense_layer(cnn, idx, &weights, &bias, NK_ACTIVATION_RELU, 0.01f);
```

For file-based models, use `nk_cnn_load` or `nk_model_load` — all block types (including avg pool, batch norm, padded conv) are configured from the `.nk` layer list.

For **embedded** models (AOT firmware), pass the static `.nk` byte array to `nk_model_load_memory` — same loader path, no filesystem.

## Model loader (`.nk`)

| C function | C++ equivalent | Notes |
|------------|----------------|-------|
| `nk_parse_architecture` | `NkLoader::ParseFile` + `FillArchInfo` | Populates `nk_arch_info_t` |
| `nk_recommended_arena_bytes` | `ArenaUtil::CapacityForModel` + inspect probe | Load + forward peak sizing (CPU heap builds) |
| `nk_parse_architecture_memory` | `NkLoader::ParseBuffer` + `FillArchInfo` | Parse embedded blob without a file |
| `nk_arch_print` | `NkLoader::PrintNetworkSummary` | Boxed summary to stdout |
| `nk_mlp_load` | `NkLoader::LoadMLP` | |
| `nk_mlp_load_memory` | `NkLoader::LoadMLPFromBuffer` | Embedded `.nk` blob; `data` must outlive the network |
| `nk_mlp_is_quantized` | `MLPNetwork::IsQuantized` | After load or manual init |
| `nk_cnn_load` | `NkLoader::LoadCNN` | All layer kinds including composite blocks |
| `nk_cnn_load_memory` | `NkLoader::LoadCNNFromBuffer` | Embedded `.nk` blob |
| `nk_cnn_is_quantized` | `CNNNetwork::IsQuantized` | After load or manual init |
| `nk_model_load_auto` | `NkLoader::Load` | Dispatches by network kind |

```c
nk_status_t nk_parse_architecture(const char* nk_path, nk_arch_info_t* info);
nk_status_t nk_parse_architecture_memory(const uint8_t* data, size_t size, nk_arch_info_t* info);
nk_status_t nk_arch_print(const char* nk_path);
nk_status_t nk_mlp_load(const char* nk_path, nk_arena_t* arena, nk_mlp_t* mlp, nk_arch_info_t* info);
nk_status_t nk_mlp_load_memory(const uint8_t* data, size_t size, nk_arena_t* arena, nk_mlp_t* mlp, nk_arch_info_t* info);
bool nk_mlp_is_quantized(const nk_mlp_t* mlp);
nk_status_t nk_cnn_load(const char* nk_path, nk_arena_t* arena, nk_cnn_t* cnn, nk_arch_info_t* info);
nk_status_t nk_cnn_load_memory(const uint8_t* data, size_t size, nk_arena_t* arena, nk_cnn_t* cnn, nk_arch_info_t* info);
bool nk_cnn_is_quantized(const nk_cnn_t* cnn);
nk_status_t nk_model_load_auto(const char* nk_path, nk_arena_t* arena, nk_network_kind_t* kind,
                               nk_mlp_t* mlp, nk_cnn_t* cnn, nk_arch_info_t* info);
```

High-level combined handle:

```c
nk_status_t nk_model_load(const char* nk_path, nk_arena_t* arena, nk_model_t* model);
/* Embedded blob must outlive the model (flash .rodata / blob-backed weights). */
nk_status_t nk_model_load_memory(const uint8_t* data, size_t size, nk_arena_t* arena, nk_model_t* model);
nk_status_t nk_model_get_arch(const nk_model_t* model, nk_arch_info_t* info);
uint32_t nk_model_input_count(const nk_model_t* model);
uint32_t nk_model_output_count(const nk_model_t* model);
nk_network_kind_t nk_model_kind(const nk_model_t* model);
bool nk_model_is_quantized(const nk_model_t* model);
nk_status_t nk_model_run(...);
nk_status_t nk_inspect_model(...);
nk_status_t nk_inspect_model_memory(const uint8_t* data, size_t size, nk_arena_t* arena, nk_inspect_info_t* info);
```

## Tests and CLI (CPU / desktop builds only)

Available when `NETKIT_DESKTOP` is defined (`NETKIT_TARGET=cpu`):

```c
nk_test_summary_t nk_run_model_tests(const char* nk_path);
nk_test_summary_t nk_run_all_tests(void);
int nk_cli_run(int argc, char** argv);
```

Embedded test format: [NK_FORMAT.md](NK_FORMAT.md). CLI commands: [CLI.md](CLI.md). Build profiles: [BUILD_TARGETS.md](BUILD_TARGETS.md).

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

These mirror querying a loaded C++ network after `NkLoader::Load` — input/output counts come from the `.nk` header and layer descriptors.

### `nk_model_load`

Loads architecture and weights from `nk_path`. Allocates network state from `arena`.

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
nk_status_t nk_inspect_model(const char* nk_path, nk_arena_t* arena, nk_inspect_info_t* info);
nk_status_t nk_inspect_model_memory(const uint8_t* data, size_t size,
                                      nk_arena_t* arena, nk_inspect_info_t* info);
```

Loads the model, runs a zero-input forward pass, and reports arena high-water marks. Matches `./netkit inspect --full`. Weights stay flash/blob-backed; `flash_payload_bytes` is `weights_bytes + biases_bytes`.

Use `arena_bytes_after_forward` to size embedded SRAM. Add `flash_payload_bytes` to flash budget (not arena). See [ARENA.md](ARENA.md).

## Complete examples

| File | Build | Description |
|------|-------|-------------|
| [`examples/infer_c.c`](../examples/infer_c.c) | `make example-c` | High-level `nk_model_load` / `nk_model_run` |
| [`examples/infer_cpp.cpp`](../examples/infer_cpp.cpp) | `make example-cpp` | Native C++26 API (see [cpp-api.md](cpp-api.md)) |

Model format: [NK_FORMAT.md](NK_FORMAT.md).

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

    if (nk_model_load("models/test_mlp.nk", &arena, &model) != NK_OK)
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
