# C / C++ API Parity

netkit maintains **two public interfaces** over one **C++26 implementation**:

| Interface | Header | Standard | Role |
|-----------|--------|----------|------|
| C++ API | `include/*.hpp` | C++26 | Primary API — engine, CLI, C++ tests |
| C API | `include/netkit.h` | C23 | FFI bridge for C callers / C-only MCUs |

C source in this repository is limited to `tests/test_c_api.c` and `examples/infer_c.c`. The bridge is implemented in `src/netkit_api.cpp` (C++26).

## Policy

1. **Every new stable C++ public symbol** in `include/*.hpp` must have a documented C equivalent in `netkit.h` before the feature is considered complete.
2. **Naming:** C functions use the `nk_` prefix and snake_case (`nk_model_load`, `nk_ops_relu`).
3. **Errors:** C functions return `nk_status_t`; details are available via `nk_last_error()`.
4. **Memory:** C handles (`nk_arena_t`, `nk_model_t`, `nk_mlp_t`, `nk_cnn_t`) use fixed-size opaque storage for stack allocation — no heap in the handle itself.
5. **Internal C++ only:** `Json::` parser helpers are building blocks for the model loader. C callers use `nk_parse_architecture()` and the loader functions instead of raw JSON access.

When adding a feature, update this file and both [c-api.md](c-api.md) and [cpp-api.md](cpp-api.md).

Related docs: [MODEL_FORMAT.md](MODEL_FORMAT.md), [VECTORS_TESTS.md](VECTORS_TESTS.md), [CLI.md](CLI.md).

## Test suites

| Suite | Language | Command | Source |
|-------|----------|---------|--------|
| C++ API | C++26 | `make test-cpp` / `./netkit test` | `src/test.cpp` |
| C API | C23 | `make test-c` | `tests/test_c_api.c` |
| Both | — | `make test` | runs C++ then C |

Both suites exercise the same eight vector models; the C suite adds direct API smoke tests (arena, tensor, ops, load/run).

## Symbol map

### Version / errors

| C++ | C |
|-----|---|
| (constants in headers) | `NK_VERSION_*`, `nk_version_string()` |
| `ModelLoader::LoadStatus` | `nk_status_t`, `nk_status_string()` |
| `LoadResult::message` | `nk_last_error()` |

### Arena (`arena.hpp`)

| C++ | C |
|-----|---|
| `Arena::init` | `nk_arena_init` |
| `Arena::alloc` | `nk_arena_alloc` |
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
| `Conv2D` | `nk_conv2d_t` |
| `Conv2D::forward` | `nk_conv2d_forward` |

### MLP (`mlp.hpp`)

| C++ | C |
|-----|---|
| `ActivationType` | `nk_activation_t` |
| `MLPNetwork` | `nk_mlp_t` |
| `MLPNetwork::IsValid` | `nk_mlp_is_valid` |
| `MLPNetwork` constructor | `nk_mlp_create` |
| `InitLayer` | `nk_mlp_init_layer` |
| `forward` | `nk_mlp_forward` |

### CNN (`cnn.hpp`)

| C++ | C |
|-----|---|
| `ConvActivationType` | `nk_conv_activation_t` |
| `CNNNetwork` | `nk_cnn_t` |
| `CNNNetwork::IsValid` | `nk_cnn_is_valid` |
| `CNNNetwork` constructor | `nk_cnn_create` |
| `InitLayer` | `nk_cnn_init_layer` |
| `forward` | `nk_cnn_forward` |

### ModelLoader (`model_loader.hpp`)

| C++ | C |
|-----|---|
| `ParseArchitecture` | `nk_parse_architecture` |
| `PrintArchitecture` | `nk_arch_print` |
| `JsonPathToBinPath` | `nk_json_path_to_bin_path` |
| `LoadWeightsBin` | `nk_load_weights_bin` |
| `LoadMLP` | `nk_mlp_load` |
| `LoadCNN` | `nk_cnn_load` |
| `Load` | `nk_model_load_auto` |
| `ArchitectureSpec` | `nk_arch_info_t` |

High-level combined handle (C convenience):

| C++ usage pattern | C |
|-------------------|---|
| Load + run inference | `nk_model_load`, `nk_model_run`, `nk_inspect_model` |

### Vectors / tests

| C++ | C |
|-----|---|
| `VectorsLoader::RunSummary` | `nk_test_summary_t` |
| `VectorsLoader::RunVectorsFile` | `nk_run_vectors_file` |
| `run_all_tests` | `nk_run_all_tests` |

### CLI

| C++ | C |
|-----|---|
| `Cli::Run` | `nk_cli_run` |

### JSON parser (`json_parser.hpp`)

Internal to the engine. No direct C bindings — use model loader and `nk_parse_architecture()` instead.

## Linking

`libnetkit.a` contains C++ object code. Link C programs with a C++-aware linker:

```bash
clang -std=c23 -Iinclude -c app.c -o app.o
clang++ -std=c++26 -o app app.o libnetkit.a
```
