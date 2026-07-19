# C / C++ API Parity

netkit maintains **two public interfaces** over one **C++26 implementation** — part of an embedded-first, multi-modal inference engine with interpreter and AOT-compiled deployment paths ([PHILOSOPHY.md](PHILOSOPHY.md)):

| Interface | Header | Standard | Role |
|-----------|--------|----------|------|
| C++ API | `include/*.hpp` | C++26 | Primary API — engine, CLI, C++ tests |
| C API | `include/netkit.h` | C23 | FFI bridge for C callers / C-only MCUs |

C source in this repository is limited to `tests/test_c_api.c` and `examples/infer_c.c`. The bridge is implemented in `src/netkit_api.cpp` (C++26).

## Policy

1. **Core runtime parity:** arena, tensor, ops, MLP/CNN construction, `.nk` loading, high-level model run, regression, and CLI entry points must have documented C equivalents in `netkit.h` before the feature is considered complete.
2. **Naming:** C functions use the `nk_` prefix and snake_case (`nk_model_load`, `nk_ops_relu`).
3. **Errors:** C functions return `nk_status_t`; details are available via `nk_last_error()`.
4. **Memory:** C handles (`nk_arena_t`, `nk_model_t`, `nk_mlp_t`, `nk_cnn_t`) use fixed-size opaque storage for stack allocation — no heap in the handle itself.
5. **Regression tests:** embedded in `.nk` files (`TCAS` section). C callers use `nk_run_model_tests()` / `nk_run_all_tests()` (**CPU / desktop builds only**).

When adding a feature, update this file and both [c-api.md](c-api.md) and [cpp-api.md](cpp-api.md).

**Build targets / backends** are shared via `include/netkit_config.h` (included by both `netkit.h` and C++ headers). Keep the Build configuration tables in [c-api.md](c-api.md#build-configuration) and [cpp-api.md](cpp-api.md#build-configuration) in sync when changing `NETKIT_TARGET_*`, arena, or CMSIS / ESP-NN / XNNPACK defaults — [BUILD_TARGETS.md](BUILD_TARGETS.md). C and C++ callers use the same `nk_*` / engine APIs; backends (`CmsisNn*`, `EspNn*`, `Xnnpack*`) are compile-time only and need no C mirror.

**Manual network construction** (layer init call order, activation buffers, forward): [cpp-api.md](cpp-api.md#manual-construction-call-order) (MLP + CNN) and [c-api.md](c-api.md#mlp-manual-construction-call-order) (C). Composite blocks (ResNet, UIB, ConvNeXt, YOLOX head): same docs plus feature guides ([YOLOX.md](YOLOX.md#manual-construction), etc.).

Related docs: [NK_FORMAT.md](NK_FORMAT.md), [CLI.md](CLI.md), [BUILD_TARGETS.md](BUILD_TARGETS.md), [PHILOSOPHY.md](PHILOSOPHY.md).

## Test suites

| Suite | Language | Command | Source |
|-------|----------|---------|--------|
| C++ API | C++26 | `make test-cpp` / `./netkit test` | `src/test.cpp` |
| C API | C23 | `make test-c` | `tests/test_c_api.c` |
| Both | — | `make test` (default) / `make test-full` (manual) | runs C++ then C; `test-full` adds full Python ONNX parity |

Both suites exercise the same **89 embedded `.nk` inference cases**; `nk_run_all_tests()` delegates to `run_all_tests()` in `src/test.cpp`. The C suite additionally smoke-tests `nk_run_model_tests()` on composite and ONNX-import fixtures before the full regression pass.

## Symbol map

### Version / errors

| C++ | C |
|-----|---|
| (constants in headers) | `NK_VERSION_*`, `nk_version_string()` |
| `NkLoader::LoadStatus` | `nk_status_t` via many-to-one map (below), `nk_status_string()` |
| `LoadResult::message` | `nk_last_error()` |

`NkLoader::LoadStatus` → `nk_status_t` (in `netkit_api.cpp`):

| `LoadStatus` | `nk_status_t` |
|--------------|---------------|
| `Ok` | `NK_OK` |
| `FileOpenFailed` | `NK_ERR_MODEL_OPEN` |
| `ReadFailed` | `NK_ERR_MODEL_READ` |
| `InvalidMagic`, `TruncatedFile` | `NK_ERR_MODEL_PARSE` |
| `UnsupportedVersion` | `NK_ERR_VERSION_MISMATCH` |
| `UnsupportedLayer` | `NK_ERR_LAYER_CONFIG` |
| `SizeMismatch` | `NK_ERR_WEIGHT_MISMATCH` |
| `ArenaOverflow` | `NK_ERR_ARENA_OVERFLOW` |

### Arena (`arena.hpp`)

| C++ | C |
|-----|---|
| `Arena::init` | `nk_arena_init` |
| `Arena::init_heap` / `destroy_heap` (`NETKIT_ARENA_HEAP`) | `nk_arena_init_heap` / `nk_arena_destroy_heap` — **CPU default**; **MPU** via `NETKIT_HEAP_ARENA=1`; **MCU forbidden** (`#error`) |
| `Arena::alloc` | `nk_arena_alloc` (size + alignment) |
| `Arena::reset` | `nk_arena_reset` |
| `Arena::capacity` / `offset` (fields) / `remaining()` | `nk_arena_capacity`, `nk_arena_used` (maps to `offset`), `nk_arena_remaining` |
| `Arena::kDefaultCapacity` | `NK_ARENA_DEFAULT_CAPACITY` |
| `Arena::attach_mapped_file` / `release_mapped_file` | Transparent inside `nk_*_load` when `NETKIT_USE_MMAP=1` |

### Tensor (`tensor.hpp`)

| C++ | C |
|-----|---|
| `Tensor` | `nk_tensor_t` |
| `DataType` (`Float32`…`Int32`) | `nk_dtype_t` (`NK_DTYPE_FLOAT32`…`NK_DTYPE_INT32`) |
| `kMaxTensorRank` | `NK_MAX_TENSOR_RANK` |

### TensorFactory (`tensor_factory.hpp`)

| C++ | C |
|-----|---|
| `Create2D` | `nk_tensor_create_2d` |
| `CreateND` | `nk_tensor_create_nd` |
| `View2D` | `nk_tensor_view_2d` |
| `View2DInt8` | `nk_tensor_view_2d_int8` |
| `View3DInt8` | `nk_tensor_view_3d_int8` |
| `View1DInt32` | `nk_tensor_view_1d_int32` |
| `Fill` | `nk_tensor_fill` |
| `Print` | `nk_tensor_print` |
| `PrintLabeled` | `nk_tensor_print_labeled` |

### Tensor access (`tensor_access.hpp`)

| C++ | C |
|-----|---|
| `tensor_data_f32` | `nk_tensor_data_f32`, `nk_tensor_data_f32_const` (nullptr if dtype ≠ float32) |
| `tensor_data_i8` | `nk_tensor_data_i8`, `nk_tensor_data_i8_const` (nullptr if dtype ≠ int8) |
| `tensor_data_i32` | `nk_tensor_data_i32`, `nk_tensor_data_i32_const` (nullptr if dtype ≠ int32) |
| `index_nhwc` | `nk_tensor_index_nhwc` |

### Ops (`ops.hpp` — namespace `Ops`)

| C++ | C |
|-----|---|
| `Ops::IsElementwiseValid` | `nk_ops_is_elementwise_valid` |
| `Ops::CheckSameShape2D` | `nk_ops_check_same_shape_2d` |
| `Ops::CheckSameShapeND` | `nk_ops_check_same_shape_nd` |
| `Ops::IsMatMulValid` | `nk_ops_is_matmul_valid` |
| `Ops::IsElementwiseValidND` | `nk_ops_is_elementwise_valid_nd` |
| `Ops::IsUnaryOpValid` | `nk_ops_is_unary_op_valid` |
| `Ops::Mul` | `nk_ops_mul` |
| `Ops::MulScalar` | `nk_ops_mul_scalar` |
| `Ops::MatAdd` | `nk_ops_mat_add` |
| `Ops::MatAddND` | `nk_ops_mat_add_nd` |
| `Ops::MatMul` | `nk_ops_mat_mul` |
| `Ops::MulND` | `nk_ops_mul_nd` |
| `Ops::ReLU` | `nk_ops_relu` |
| `Ops::Sigmoid` | `nk_ops_sigmoid` |
| `Ops::Tanh` | `nk_ops_tanh` |
| `Ops::LeakyReLU` | `nk_ops_leaky_relu` |
| `Ops::ReLU6` | `nk_ops_relu6` |
| `Ops::Softmax` | `nk_ops_softmax` |

### Conv2D (`conv2d.hpp`)

| C++ | C |
|-----|---|
| `Conv2D` | `nk_conv2d_t` (includes `pad_h`, `pad_w`, `pad_h_end`, `pad_w_end`; `NK_PAD_MIRROR` = symmetric end) |
| `Conv2D::forward` | `nk_conv2d_forward` |

### DepthwiseConv2D (`depthwise_conv2d.hpp`)

| C++ | C |
|-----|---|
| `DepthwiseConv2D` | `nk_depthwise_conv2d_t` (weights `[ch][kh][kw]`; same pad fields as conv) |
| `DepthwiseConv2D::forward` | `nk_depthwise_conv2d_forward` |

### MLP (`mlp.hpp`)

| C++ | C |
|-----|---|
| `ActivationType` | `nk_activation_t` |
| `MLPNetwork` | `nk_mlp_t` |
| `MLPNetwork::IsValid` | `nk_mlp_is_valid` |
| `MLPNetwork::IsQuantized` | `nk_mlp_is_quantized` |
| `MLPNetwork` constructor | `nk_mlp_create` |
| `InitLayer` | `nk_mlp_init_layer` |
| `InitActivationBuffers` | `nk_mlp_init_activation_buffers` |
| `HasActivationBuffers` | `nk_mlp_has_activation_buffers` |
| `SetOmitFinalSoftmax` / `OmitFinalSoftmax` | `nk_mlp_set_omit_final_softmax` / `nk_mlp_omit_final_softmax` |
| `forward` | `nk_mlp_forward` |

### CNN (`cnn.hpp`)

| C++ | C |
|-----|---|
| `CnnBlockType` | `nk_cnn_block_type_t` (same member order and numeric values) |
| `ConvActivationType` | `nk_conv_activation_t` |
| `CNNNetwork` | `nk_cnn_t` |
| `CNNNetwork::IsValid` | `nk_cnn_is_valid` |
| `CNNNetwork::IsQuantized` | `nk_cnn_is_quantized` |
| `CNNNetwork` constructor | `nk_cnn_create` |
| `InitConvLayer` | `nk_cnn_init_conv_layer` |
| `InitDepthwiseConvLayer` | `nk_cnn_init_depthwise_conv_layer` |
| `InitPoolLayer` | `nk_cnn_init_pool_layer` |
| `InitAvgPoolLayer` | `nk_cnn_init_avg_pool_layer` |
| `InitBatchNormLayer` | `nk_cnn_init_batch_norm_layer` |
| `InitLayerNormLayer` | `nk_cnn_init_layernorm_layer` |
| `InitConvNeXtV2BlockLayer` | `nk_cnn_init_convnextv2_block_layer` |
| `InitMobilenetV4UibLayer` | `nk_cnn_init_mobilenetv4_uib_layer` |
| `InitResNetBasicBlockLayer` | `nk_cnn_init_resnet_basic_block_layer` |
| `InitYoloxDecoupledHeadLayer` | `nk_cnn_init_yolox_decoupled_head_layer` |
| `InitFeatureTapLayer` | `nk_cnn_init_feature_tap_layer` |
| `InitYoloxPafpnLayer` | `nk_cnn_init_yolox_pafpn_layer` |
| `GetFeatureTapBuffer` / `GetFeatureTapElems` | `nk_cnn_get_feature_tap_buffer` / `nk_cnn_get_feature_tap_elems` |
| `InitFlattenLayer` | `nk_cnn_init_flatten_layer` |
| `InitDenseLayer` | `nk_cnn_init_dense_layer` |
| `InitActivationBuffers` | `nk_cnn_init_activation_buffers` |
| `HasActivationBuffers` | `nk_cnn_has_activation_buffers` |
| `KernelWorkspaceBytes` | `nk_cnn_kernel_workspace_bytes` |
| quant `Runtime::omit_final_softmax` | `nk_cnn_set_omit_final_softmax` / `nk_cnn_omit_final_softmax` |
| `forward` | `nk_cnn_forward` |
| `LoadCNN` (mixed conv/pool/avg pool/batch norm/flatten/dense `.nk`) | `nk_cnn_load` |

### NkLoader (`nk_loader.hpp`)

| C++ | C |
|-----|---|
| `ParseFile` + `FillArchInfo` | `nk_parse_architecture` |
| `ParseBuffer` + `FillArchInfo` | `nk_parse_architecture_memory` |
| `InputElements` / `OutputElements` | `nk_arch_info_t` via `nk_parse_architecture` / `nk_model_get_arch` |
| `ArenaUtil::CapacityForModel` (+ inspect probe on CPU) | `nk_recommended_arena_bytes` |
| `PrintNetworkSummary` | `nk_arch_print` |
| `LoadMLP` | `nk_mlp_load` |
| `LoadMLPFromBuffer` | `nk_mlp_load_memory` |
| `LoadMLPFromBuffer` (high-level) | `nk_model_load_memory` (MLP) |
| `LoadCNN` | `nk_cnn_load` |
| `LoadCNNFromBuffer` | `nk_cnn_load_memory` |
| `LoadCNNFromBuffer` (high-level) | `nk_model_load_memory` (CNN) |
| `Load` | `nk_model_load_auto` |
| `ArchInfo` (`weight_floats` only) | `nk_arch_info_t` (`expected_weight_floats` plus `weights_bytes` / `biases_bytes` from `.nk` header) |
| `inspect --full` (flash-aware buffer load) | `nk_inspect_model`, `nk_inspect_model_memory` |

High-level combined handle (C convenience):

| C++ usage pattern | C |
|-------------------|---|
| Load + run float32 inference | `nk_model_load`, `nk_model_run`, `nk_inspect_model` |
| Load + run int8 inference | `nk_model_load` / `nk_model_load_memory`, `nk_model_run_int8` |
| Load embedded `.nk` blob + run | `nk_model_load_memory`, `nk_model_run` / `nk_model_run_int8` |
| Inspect embedded blob arena peaks | `nk_inspect_model_memory` |
| Query loaded model | `nk_model_get_arch`, `nk_model_input_count`, `nk_model_output_count`, `nk_model_kind`, `nk_model_is_quantized` |
| Skip final Softmax (logits) | `nk_model_set_omit_final_softmax` / `nk_model_omit_final_softmax` (MLP + quantized CNN) |

### Weight storage (always flash/blob-backed)

| Path | Behavior |
|------|----------|
| Buffer / AOT load | Binds views into the `.nk` blob; `data` must outlive the network |
| File load + `NETKIT_USE_MMAP=1` (cpu + MPU default; forbidden on MCU) | File mmap (POSIX / Win32); arena owns mapping until `reset()` / destroy |
| File load without mmap (MCU; or MPU with `NETKIT_MMAP=0`) | Copies `.nk` into arena (prefer buffer/flash instead) |
| Arena peaks | Exclude weight/bias payload (`flash_payload_bytes` reports flash budget) |

### MCU deployment constraints

| Topic | Policy |
|-------|--------|
| Arena | Caller-owned **static/global** buffer only — `NETKIT_HEAP_ARENA` is a compile error |
| Heap | No `malloc` / `new` / `delete` / `free` on MCU firmware paths |
| Weights | Stay in the flash `.nk` image (zero-copy scales); arena holds activations + metadata |
| `NETKIT_MCU_ACCEL_ONLY` (`NETKIT_MCU_CMSIS_ONLY`) | Default on MCU class when `REFERENCE_QUANT_LOOPS=0` — QuantOps reference loops omitted (CMSIS-NN **or** ESP-NN production) |
| `NETKIT_DISABLE_IOSTREAM` | Default on MCU — `nk_arch_print` / `PrintNetworkSummary` are no-ops |
| NUCLEO-F446RE peers | **int8** CNN / DS-CNN vs TFLM + microTVM (CMSIS-NN and reference); float32 CNN/DS-CNN exceed 512 KiB flash — [STATUS.md](STATUS.md) |
| Espressif MCU (`mcu_esp`) | ESP-NN int8 production (ESP32 / S3 / C3 / C6 / P4); float32 reference (ESP-NN has no float API); same C `nk_*` load/run as Arm MCU |

### AOT deployment

| Path | C++ | C |
|------|-----|---|
| Interpreter (embed `.nk` + loader) | `netkit::aot::*::Model` or `*_aot_load` + `nk_model_load_memory` | `*_aot.h` + `nk_model_load_memory` |
| Lowered (static `Kernels::` chain) | `netkit::aot::*::Model` in `*_aot.hpp` | **C++ only** — no lowered C emitter |

Lowered AOT keeps coef arrays in flash `.rodata` (no SRAM copy at load). See `boards/nucleo-f446re/`.

### Intentional C++-only symbols

| C++ | Reason |
|-----|--------|
| `NkOpsResolver`, `NkOpList`, `CNNNetwork::SetOpsResolver` / `GetOpsResolver` | Firmware op trimming — compile-time `NkOpList<Ops...>::View()` in C++ only; file load uses default resolver internally |
| `ArenaUtil`, `BeginRegressionArena`, `EndRegressionArena` | CLI/regression sizing helpers |
| `TensorFactory::ViewND` | ND tensor views — use `nk_tensor_view_2d` or load from `.nk` |
| `MLPNetwork::GetLayer`, `CNNNetwork::GetBlock`, `CNNNetwork::GetOutput`, `CNNNetwork::layer_count` | In-memory network introspection after manual construction (feature taps: use `nk_cnn_get_feature_tap_*`) |
| `MLPNetwork::InitQuantizedLayer`, `CNNNetwork::InitQuantized*` (`InitQuantizedConvLayer`, `InitQuantizedDenseLayer`, `InitQuantizedActivationBuffers`), `SetQuantized`, `Set`/`GetQuantOutputFormat` (Int8 only; float↔int8 is Python-side), `CNNNetwork::SetQuantRuntime` / `forward_quantized` | Quantized manual construction + low-level runtime — C callers load `.nk` and use `nk_model_run_int8` (or typed `nk_mlp_*` / `nk_cnn_*` forward after load); query with `nk_*_is_quantized` |
| `CNNNetwork::forward_timed`, `MLPNetwork::forward_timed` | Benchmark-only profilers |
| `CmsisQuantPlan::Runtime` other fields | C exposes `omit_final_softmax` via `nk_cnn_*` / `nk_model_*`; remaining plan fields stay C++ |
| `CmsisNnKernel` / `CmsisNnQuant`, `EspNnKernel` / `EspNnQuant`, `XnnpackKernel` / `XnnpackQuant` | Compile-time backends selected by `NETKIT_TARGET` + `NETKIT_*` flags; C callers share the same path via `nk_model_run` / `nk_model_run_int8` |
| `NkLoader::ReadTestSuite`, `ModelPayloadBytes`, `NetworkKindName`, `FreeParsedModelExtras` | Loader utilities; C uses `nk_parse_architecture` / `nk_model_*` instead |
| `Conv2D::forward(..., fuse_activation)` / `DepthwiseConv2D::forward(..., fuse_activation)` | C forwards use default activation fusion |
| `TensorFactory::PrintLabeled(..., max_values)` | C `nk_tensor_print_labeled` omits truncation control |

### Regression tests

| C++ | C |
|-----|---|
| `NkRegression::RunSummary` | `nk_test_summary_t` |
| `NkRegression::RunModelTests` | `nk_run_model_tests` |
| `run_all_tests` | `nk_run_all_tests` |

### CLI

| C++ | C |
|-----|---|
| `Cli::Run` | `nk_cli_run` |

`libnetkit.a` contains C++ object code. Link C programs with a C++-aware linker:

```bash
clang -std=c23 -Iinclude -c app.c -o app.o
clang++ -std=c++26 -o app app.o libnetkit.a
```
