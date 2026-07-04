# C / C++ API Parity

netkit maintains **two public interfaces** over one **C++26 implementation**:

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
| `Conv2D` | `nk_conv2d_t` (includes `pad_h`, `pad_w`) |
| `Conv2D::forward` | `nk_conv2d_forward` |

### MLP (`mlp.hpp`)

| C++ | C |
|-----|---|
| `ActivationType` | `nk_activation_t` |
| `MLPNetwork` | `nk_mlp_t` |
| `MLPNetwork::IsValid` | `nk_mlp_is_valid` |
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
| `PrintHeader` | — (detailed binary dump only) |
| `PrintNetworkSummary` | `nk_arch_print` |
| `LoadMLP` | `nk_mlp_load` |
| `LoadMLPFromBuffer` | (via `nk_model_load_memory` for MLP) |
| `LoadCNN` | `nk_cnn_load` |
| `LoadCNNFromBuffer` | (via `nk_model_load_memory` for CNN) |
| `Load` | `nk_model_load_auto` |
| `LoadFromBuffer` | `nk_model_load_memory` |
| `ArchInfo` | `nk_arch_info_t` |

High-level combined handle (C convenience):

| C++ usage pattern | C |
|-------------------|---|
| Load + run inference | `nk_model_load`, `nk_model_run`, `nk_inspect_model` |
| Load embedded `.nk` blob + run | `nk_model_load_memory`, `nk_model_run` |
| Query loaded model | `nk_model_get_arch`, `nk_model_input_count`, `nk_model_output_count`, `nk_model_kind` |

### Intentional C++-only symbols

| C++ | Reason |
|-----|--------|
| `PrintHeader` | Detailed `.nk` header dump |
| `NkOpsResolver`, `NkOpList`, `CNNNetwork::SetOpsResolver` | Firmware op trimming — compile-time `NkOpList<Ops...>::View()` in C++ only; file load uses default resolver internally |
| `ArenaUtil`, `BeginRegressionArena`, `EndRegressionArena` | CLI/regression sizing helpers |
| `TensorFactory::ViewND` | ND tensor views — use `nk_tensor_view_2d` or load from `.nk` |
| `MLPNetwork::GetLayer`, `CNNNetwork::GetBlock`, `CNNNetwork::GetOutput` | In-memory network introspection after manual construction |
| `NkLoader::IsNkPath`, `ReadTestSuite`, `ModelPayloadBytes`, `NetworkKindName` | Loader utilities; C uses `nk_parse_architecture` / `nk_model_*` instead |
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
