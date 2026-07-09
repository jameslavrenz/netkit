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

**Manual network construction** (layer init call order, activation buffers, forward): [cpp-api.md](cpp-api.md#manual-construction-call-order) (MLP + CNN) and [c-api.md](c-api.md#mlp-manual-construction-call-order) (C). Composite blocks (ResNet, UIB, ConvNeXt, YOLOX head): same docs plus feature guides ([YOLOX.md](YOLOX.md#manual-construction), etc.).

Related docs: [NK_FORMAT.md](NK_FORMAT.md), [CLI.md](CLI.md), [BUILD_TARGETS.md](BUILD_TARGETS.md), [PHILOSOPHY.md](PHILOSOPHY.md).

## Test suites

| Suite | Language | Command | Source |
|-------|----------|---------|--------|
| C++ API | C++26 | `make test-cpp` / `./netkit test` | `src/test.cpp` |
| C API | C23 | `make test-c` | `tests/test_c_api.c` |
| Both | — | `make test` (default) / `make test-full` (manual) | runs C++ then C; `test-full` adds full Python ONNX parity |

Both suites exercise the same **88 embedded `.nk` inference cases**; `nk_run_all_tests()` delegates to `run_all_tests()` in `src/test.cpp`. The C suite additionally smoke-tests `nk_run_model_tests()` on composite and ONNX-import fixtures before the full regression pass.

## Symbol map

### Version / errors

| C++ | C |
|-----|---|
| (constants in headers) | `NK_VERSION_*`, `nk_version_string()` |
| `NkLoader::LoadStatus` | `nk_status_t`, `nk_status_string()` |
| `LoadResult::message` | `nk_last_error()` |

### Arena (`arena.hpp`)

| C++ | C |
|-----|---|
| `Arena::init` | `nk_arena_init` |
| `Arena::init_heap` / `destroy_heap` (`NETKIT_ARENA_HEAP`) | `nk_arena_init_heap` / `nk_arena_destroy_heap` |
| `Arena::alloc` | `nk_arena_alloc` (size + alignment) |
| `Arena::reset` | `nk_arena_reset` |
| `Arena::capacity` / `offset` / `remaining` | `nk_arena_capacity`, `nk_arena_used`, `nk_arena_remaining` |
| `Arena::kDefaultCapacity` | `NK_ARENA_DEFAULT_CAPACITY` |

### Tensor (`tensor.hpp`)

| C++ | C |
|-----|---|
| `Tensor` | `nk_tensor_t` |
| `DataType` | `nk_dtype_t` |
| `kMaxTensorRank` | `NK_MAX_TENSOR_RANK` |

### TensorFactory (`tensor_factory.hpp`)

| C++ | C |
|-----|---|
| `Create2D` | `nk_tensor_create_2d` |
| `CreateND` | `nk_tensor_create_nd` |
| `View2D` | `nk_tensor_view_2d` |
| `Fill` | `nk_tensor_fill` |
| `Print` | `nk_tensor_print` |
| `PrintLabeled` | `nk_tensor_print_labeled` |

### Tensor access (`tensor_access.hpp`)

| C++ | C |
|-----|---|
| `tensor_data_f32` | `nk_tensor_data_f32`, `nk_tensor_data_f32_const` |
| `index_nhwc` | `nk_tensor_index_nhwc` |

### Ops (`ops.hpp`)

| C++ | C |
|-----|---|
| `IsElementwiseValid` | `nk_ops_is_elementwise_valid` |
| `CheckSameShape2D` | `nk_ops_check_same_shape_2d` |
| `CheckSameShapeND` | `nk_ops_check_same_shape_nd` |
| `IsMatMulValid` | `nk_ops_is_matmul_valid` |
| `IsElementwiseValidND` | `nk_ops_is_elementwise_valid_nd` |
| `IsUnaryOpValid` | `nk_ops_is_unary_op_valid` |
| `Mul` | `nk_ops_mul` |
| `MulScalar` | `nk_ops_mul_scalar` |
| `MatAdd` | `nk_ops_mat_add` |
| `MatAddND` | `nk_ops_mat_add_nd` |
| `MatMul` | `nk_ops_mat_mul` |
| `MulND` | `nk_ops_mul_nd` |
| `ReLU` | `nk_ops_relu` |
| `Sigmoid` | `nk_ops_sigmoid` |
| `Tanh` | `nk_ops_tanh` |
| `LeakyReLU` | `nk_ops_leaky_relu` |
| `ReLU6` | `nk_ops_relu6` |
| `Softmax` | `nk_ops_softmax` |

### Conv2D (`conv2d.hpp`)

| C++ | C |
|-----|---|
| `Conv2D` | `nk_conv2d_t` (includes `pad_h`, `pad_w`, `pad_h_end`, `pad_w_end`; `NK_PAD_MIRROR` = symmetric end) |
| `Conv2D::forward` | `nk_conv2d_forward` |

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
| `InitFlattenLayer` | `nk_cnn_init_flatten_layer` |
| `InitDenseLayer` | `nk_cnn_init_dense_layer` |
| `InitActivationBuffers` | `nk_cnn_init_activation_buffers` |
| `HasActivationBuffers` | `nk_cnn_has_activation_buffers` |
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
| `ArchInfo` | `nk_arch_info_t` (`weights_bytes`, `biases_bytes` from `.nk` header) |
| `inspect --full` (flash-aware buffer load) | `nk_inspect_model`, `nk_inspect_model_memory` |

High-level combined handle (C convenience):

| C++ usage pattern | C |
|-------------------|---|
| Load + run inference | `nk_model_load`, `nk_model_run`, `nk_inspect_model` |
| Load embedded `.nk` blob + run | `nk_model_load_memory`, `nk_model_run` |
| Inspect embedded blob arena peaks | `nk_inspect_model_memory` |
| Query loaded model | `nk_model_get_arch`, `nk_model_input_count`, `nk_model_output_count`, `nk_model_kind`, `nk_model_is_quantized` |

### Weight storage (`NETKIT_WEIGHTS_IN_RAM`)

| Policy | C++ buffer load | C buffer load |
|--------|-----------------|---------------|
| `NETKIT_WEIGHTS_IN_RAM=1` | Copies weight/bias payload into arena | `nk_model_load_memory` copies into arena |
| `NETKIT_WEIGHTS_IN_RAM=0` (MCU default) | Binds views into flash blob; `data` must outlive network | Same — embed `.nk` in `.rodata` |

File-based load (`nk_model_load`, `NkLoader::LoadMLP`) **always** copies payload into the arena regardless of the flag.

### AOT deployment

| Path | C++ | C |
|------|-----|---|
| Interpreter (embed `.nk` + loader) | `netkit::aot::*::Model` or `*_aot_load` + `nk_model_load_memory` | `*_aot.h` + `nk_model_load_memory` |
| Lowered (static `Kernels::` chain) | `netkit::aot::*::Model` in `*_aot.hpp` | **C++ only** — no lowered C emitter |

Lowered AOT with `--weights-in-ram` copies coef arrays from flash `.rodata` sources into the arena at `Model::load()`. MCU firmware typically uses `--no-weights-in-ram` (see `boards/nucleo-f446re/`).

### Intentional C++-only symbols

| C++ | Reason |
|-----|--------|
| `NkOpsResolver`, `NkOpList`, `CNNNetwork::SetOpsResolver` / `GetOpsResolver` | Firmware op trimming — compile-time `NkOpList<Ops...>::View()` in C++ only; file load uses default resolver internally |
| `ArenaUtil`, `BeginRegressionArena`, `EndRegressionArena` | CLI/regression sizing helpers |
| `TensorFactory::ViewND` | ND tensor views — use `nk_tensor_view_2d` or load from `.nk` |
| `MLPNetwork::GetLayer`, `CNNNetwork::GetBlock`, `CNNNetwork::GetOutput`, `CNNNetwork::layer_count` | In-memory network introspection after manual construction |
| `MLPNetwork::InitQuantizedLayer`, `CNNNetwork::InitQuantized*` (`InitQuantizedConvLayer`, `InitQuantizedDenseLayer`, `InitQuantizedActivationBuffers`), `SetQuantized`, `Set`/`GetQuantOutputFormat` (Int8 only; float↔int8 is Python-side), `CNNNetwork::SetQuantRuntime` / `forward_quantized` | Quantized manual construction + runtime — C callers use `.nk` load + `nk_model_run` (quantized path chosen internally) and query with `nk_*_is_quantized` |
| `CNNNetwork::forward_timed`, `MLPNetwork::forward_timed` | Benchmark-only profilers |
| `TensorFactory::View2DInt8`, `View3DInt8` | Manual int8 tensor views — use `.nk` load path |
| `NkLoader::ReadTestSuite`, `ModelPayloadBytes`, `NetworkKindName` | Loader utilities; C uses `nk_parse_architecture` / `nk_model_*` instead |
| `Conv2D::forward(..., fuse_activation)` | C `nk_conv2d_forward` uses default activation fusion |
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
