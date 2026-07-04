# C++ API Reference (C++26)

Headers live in [`include/`](../include/). Configuration: [`include/netkit_config.h`](../include/netkit_config.h). All implementation files use `-std=c++26`.

**New users:** [GETTING_STARTED.md](GETTING_STARTED.md) · **Philosophy:** [PHILOSOPHY.md](PHILOSOPHY.md)

**Numeric type:** inference uses **float32 only** today — see [DATATYPES.md](DATATYPES.md) for the quantized-type roadmap (float16, int16, int8, int4 planned).

## Build configuration

| Makefile | Macro | Outputs |
|----------|-------|---------|
| `NETKIT_TARGET=cpu` | `NETKIT_TARGET_CPU`, `NETKIT_DESKTOP` | `netkit` CLI + full `libnetkit.a` |
| `NETKIT_TARGET=mcu` | `NETKIT_TARGET_MCU` | Lean `libnetkit.a` only |
| `NETKIT_TARGET=mpu` | `NETKIT_TARGET_MPU` | Lean `libnetkit.a` only |

| Flag | Macro | Arena |
|------|-------|-------|
| CPU default | `NETKIT_ARENA_HEAP` | Heap API; CLI uses model-sized allocation |
| `NETKIT_GLOBAL_ARENA=1` | `NETKIT_GLOBAL_ARENA` | Static buffer on CPU |
| `NETKIT_HEAP_ARENA=1` (MCU/MPU) | `NETKIT_ARENA_HEAP` | Optional heap on embedded |

`Arena::kDefaultCapacity` / `NK_ARENA_DEFAULT_CAPACITY`: **4 MiB** (CPU), **64 KiB** (MCU), **128 KiB** (MPU).

See [BUILD_TARGETS.md](BUILD_TARGETS.md).

## Core headers

| Header | Purpose |
|--------|---------|
| `netkit_config.h` | Compile-time target and arena macros (C and C++) |
| `arena.hpp` | Bump-pointer arena allocator |
| `arena_util.hpp` | `ArenaUtil::Init`, `Scoped`, model capacity helpers |
| `tensor.hpp` | `Tensor`, `DataType`, `kMaxTensorRank` |
| `tensor_factory.hpp` | Tensor creation, fill, print |
| `tensor_access.hpp` | NHWC indexing helpers |
| `ops.hpp` | Matrix ops and activations |
| `conv2d.hpp` | Low-level 2D convolution |
| `mlp.hpp` | `MLPNetwork`, `MLPLayer`, `ActivationType` |
| `cnn.hpp` | `CNNNetwork`, `Conv2DLayer`, `ConvActivationType` |
| `nk_loader.hpp` / `nk_format.hpp` | `.nk` model loading |
| `nk_regression.hpp` | Embedded `.nk` regression test runner |
| `cli.hpp` | CLI dispatch (`Cli::Run`) |
| `test.hpp` | Test suite entry (`run_all_tests`) |

For a stable C interface from C++ projects or embedded firmware, use [`netkit.h`](c-api.md). Core runtime symbols are mapped in [`API_PARITY.md`](API_PARITY.md); a few C++ helpers remain C++-only.

---

## Arena (`arena.hpp`)

See [ARENA.md](ARENA.md) for the full bump-allocator guide.

```cpp
struct Arena {
    // kDefaultCapacity: 4 MiB (CPU), 64 KiB (MCU), 128 KiB (MPU) — see netkit_config.h

    void init(void* memory, std::size_t size);
    void* alloc(std::size_t size, std::size_t alignment);  // alignment: power of two
    void reset();
    std::size_t remaining() const;
};
```

`alloc` inserts padding when the current offset is not a multiple of `alignment`. Use `alignof(float)` for tensor payloads and weight blobs; use `alignof(T)` for struct arrays and placement-new targets.

**Why alignment matters:** weight blobs can have an odd float count, leaving the arena offset at 4 mod 8 on 64-bit platforms. Without padding, a following `MLPNetwork` or `CNNNetwork` allocation would be misaligned for `placement new`. The engine passes the correct `alignof` at every internal call site.

All network and tensor allocations during load/inference draw from the arena. No `free()` — call `reset()` to reuse the buffer.

### ArenaUtil (`arena_util.hpp`)

Target-aware helpers used by the CLI and regression harness:

```cpp
namespace ArenaUtil {
    constexpr std::size_t kHandCapacity = 64 * 1024;
    constexpr std::size_t kMnistMlpCapacity = 2 * 1024 * 1024;
    constexpr std::size_t kMnistCnnCapacity = 4 * 1024 * 1024;

    std::size_t CapacityForInputElements(uint32_t input_elements, bool is_cnn);
    bool Init(Arena& arena, std::size_t capacity, void* global_buffer = nullptr);
    void Release(Arena& arena);  // CPU only: frees heap backing; no-op on MCU/MPU
    class Scoped;  // RAII — calls Release on destruction (CPU heap builds)
}
```

On CPU (heap default), `Init(capacity, nullptr)` calls `arena.init_heap()` once. Regression reuses one heap buffer for the full suite (`NkRegression::BeginRegressionArena` / `EndRegressionArena`). On MCU/MPU, pass your static buffer pointer; `destroy_heap()` never frees memory.

When `NETKIT_ARENA_HEAP` is defined, `Arena` also provides:

```cpp
bool init_heap(std::size_t size);
void destroy_heap();
```

---

## Tensor (`tensor.hpp`)

```cpp
enum class DataType : uint8_t { Float32, Int8, UInt8, Int16 };

constexpr uint32_t kMaxTensorRank = 4;

struct Tensor {
    void* data;
    DataType type;
    uint32_t rank;
    std::array<uint32_t, kMaxTensorRank> shape;
    std::array<uint32_t, kMaxTensorRank> stride;
    uint32_t num_elements;
    uint32_t bytes;
};
```

**Layouts**

- MLP tensors: 2D row-major `[rows, cols]`
- CNN tensors: NHWC `[height, width, channels]`

---

## TensorFactory (`tensor_factory.hpp`)

```cpp
namespace TensorFactory {
    Tensor Create2D(Arena& arena, uint32_t rows, uint32_t cols);
    Tensor CreateND(Arena& arena, uint32_t rank, std::span<const uint32_t> shape);
    Tensor View2D(float* data, uint32_t rows, uint32_t cols);
    Tensor ViewND(float* data, uint32_t rank, std::span<const uint32_t> shape);
    void Fill(Tensor& t, std::initializer_list<float> values);
    void Print(const Tensor& t);
    void PrintLabeled(const char* label, const Tensor& t, uint32_t max_values = 0);
    // max_values == 0 prints every element; otherwise prints the first max_values plus a total count.
}
```

Returns tensors with null `data` if the arena is full.

---

## Ops (`ops.hpp`)

Validation helpers: `IsMatMulValid`, `CheckSameShape2D`, etc.

**Arithmetic**

| Function | Description |
|----------|-------------|
| `MatMul(A, B, C)` | Matrix multiply |
| `MatAdd(A, B, C)` | 2D element-wise add |
| `MatAddND(A, B, C)` | N-D element-wise add |
| `Mul(A, B, C)` | Element-wise multiply |
| `MulND(A, B, C)` | N-D element-wise multiply |
| `MulScalar(A, scalar, C)` | Scale by scalar |

**Activations** (in-place when `A` and `C` share storage)

| Function | Description |
|----------|-------------|
| `ReLU(A, C)` | max(0, x) |
| `LeakyReLU(A, C, alpha)` | Leaky ReLU |
| `ReLU6(A, C)` | min(max(0, x), 6) |
| `Sigmoid(A, C)` | σ(x) |
| `Tanh(A, C)` | tanh(x) |
| `Softmax(A, C)` | Softmax over elements |

---

## Conv2D (`conv2d.hpp`)

Low-level convolution with per-axis padding (`pad_h`, `pad_w` — same amount on both sides of each axis; values may differ between H and W).

```cpp
struct Conv2D {
    int kernel_size, stride, pad_h, pad_w, in_channels, out_channels;
    float* weights;  // [out][kh][kw][in]
    float* bias;     // [out]
    void forward(const Tensor& input, Tensor& output);
};
```

Output spatial size: `(input + 2*pad - kernel) / stride + 1`.

---

## MLPNetwork (`mlp.hpp`)

```cpp
enum class ActivationType { None, ReLU, Sigmoid, Tanh, LeakyReLU, ReLU6, Softmax };

class MLPNetwork {
public:
    MLPNetwork(uint32_t num_layers, Arena& arena);
    bool IsValid() const;
    bool HasActivationBuffers() const;

    bool InitActivationBuffers(Arena& arena, uint32_t batch_rows);  // called by LoadMLP

    void InitLayer(uint32_t layer_idx, const Tensor& weights, const Tensor& bias,
                   ActivationType activation, float leaky_alpha = 0.01f);

    // Hidden layers use ping-pong buffers allocated at load; final layer writes to output
    void forward(const Tensor& input, Tensor& output, Arena& arena);
    MLPLayer& GetLayer(uint32_t idx);
};
```

Weight matrix shape per layer: `[out_features, in_features]` row-major (CMSIS-NN / PyTorch layout).

---

## CNNNetwork (`cnn.hpp`)

CNN pipelines support mixed blocks: conv2d, depthwise_conv2d, max_pool2d, avg_pool2d, batch_norm2d, layernorm2d, convnextv2_block, mobilenetv4_uib, resnet_basic_block, flatten, and dense (classification head). See [NK_FORMAT.md](NK_FORMAT.md), [CONVNEXTV2.md](CONVNEXTV2.md), [MOBILENETV4.md](MOBILENETV4.md), and [RESNET18.md](RESNET18.md).

```cpp
enum class CnnBlockType {
    Conv2D, DepthwiseConv2D, MaxPool2D, AvgPool2D, BatchNorm2d, LayerNorm2d,
    Flatten, Dense, ConvNeXtV2Block, MobilenetV4Uib, ResNetBasicBlock
};

class CNNNetwork {
public:
    CNNNetwork(uint32_t num_layers, Arena& arena);
    bool IsValid() const;
    bool HasActivationBuffers() const;

    bool InitActivationBuffers(Arena& arena, uint32_t in_h, uint32_t in_w, uint32_t in_c);  // LoadCNN

    void InitConvLayer(uint32_t layer_idx,
                       int kernel_size,
                       int stride,
                       int in_channels,
                       int out_channels,
                       float* weights,
                       float* bias,
                       ConvActivationType activation,
                       float leaky_alpha = 0.01f,
                       int pad_h = 0,
                       int pad_w = 0);
    void InitDepthwiseConvLayer(uint32_t layer_idx,
                                int kernel_h,
                                int kernel_w,
                                int stride,
                                int channels,
                                float* weights,
                                float* bias,
                                ConvActivationType activation,
                                float leaky_alpha = 0.01f,
                                int pad_h = 0,
                                int pad_w = 0);
    void InitPoolLayer(uint32_t layer_idx, int pool_size, int stride, int pad_h = 0, int pad_w = 0);
    void InitAvgPoolLayer(uint32_t layer_idx, int pool_size, int stride, int pad_h = 0, int pad_w = 0);
    void InitBatchNormLayer(uint32_t layer_idx, int channels, float* scale, float* bias);
    void InitLayerNormLayer(uint32_t layer_idx, int channels, float eps, float* weight, float* bias);
    void InitConvNeXtV2BlockLayer(uint32_t layer_idx, Arena& arena, uint32_t spatial_h, uint32_t spatial_w,
                                  int channels, float eps, /* weight pointers */);
    void InitMobilenetV4UibLayer(uint32_t layer_idx, Arena& arena, uint32_t spatial_h, uint32_t spatial_w,
                                 int in_channels, int out_channels, /* UIB config + weights */);
    void InitResNetBasicBlockLayer(uint32_t layer_idx, Arena& arena, uint32_t spatial_h, uint32_t spatial_w,
                                   int in_channels, int out_channels, int stride, /* conv/BN weights */);
    void InitFlattenLayer(uint32_t layer_idx);
    void InitDenseLayer(uint32_t layer_idx, const Tensor& W, const Tensor& b,
                        ActivationType activation, float leaky_alpha = 0.01f);

    void SetOpsResolver(const NkOpsResolver& resolver);  // trim via NkOpList<...>::View()
    const NkOpsResolver& GetOpsResolver() const;

    // All blocks ping-pong between two load-time buffers; result in GetOutput()
    Tensor& forward(const Tensor& input, Arena& arena);
    CnnBlock& GetBlock(uint32_t idx);
    Tensor& GetOutput();
};
```

Spatial tensors stay NHWC until flatten; dense head output is `[1, units]`. Returns null `data` on arena overflow.

`NkLoader::LoadCNN` builds full pipelines from `.nk` files (including `models/mnist_cnn.nk`).

Layer dispatch uses `NkOpList<Ops...>::View()` for compile-time op tables — see [KERNELS.md](KERNELS.md).

---

## NkLoader (`nk_loader.hpp`)

```cpp
namespace NkLoader {

enum class NetworkKind { Unknown, Mlp, Cnn };

enum class LoadStatus {
    Ok, FileOpenFailed, ReadFailed, InvalidMagic, UnsupportedVersion,
    TruncatedFile, UnsupportedLayer, SizeMismatch, ArenaOverflow
};

struct LoadResult {
    LoadStatus status;
    NetworkKind kind;
    const char* message;
};

struct ParsedModel { /* header, layers, tensor catalog */ };
struct ArchInfo { /* version, kind, input/output counts, weight_floats */ };

LoadResult ParseFile(const char* nk_path, ParsedModel& out);
LoadResult ParseBuffer(const uint8_t* data, std::size_t size, ParsedModel& out);
void FillArchInfo(const ParsedModel& model, ArchInfo& info);
uint32_t InputElements(const ParsedModel& model);
uint32_t OutputElements(const ParsedModel& model);

void PrintHeader(const char* nk_path, const ParsedModel& model);
void PrintNetworkSummary(const char* nk_path, const ParsedModel& model);

LoadResult LoadMLP(const char* nk_path, Arena& arena, MLPNetwork*& network,
                   std::array<uint32_t, kMaxTensorRank>& input_shape, uint32_t& input_rank);

LoadResult LoadMLPFromBuffer(const uint8_t* data, std::size_t size, Arena& arena, MLPNetwork*& network,
                             std::array<uint32_t, kMaxTensorRank>& input_shape, uint32_t& input_rank);

LoadResult LoadCNN(const char* nk_path, Arena& arena, CNNNetwork*& network,
                   std::array<uint32_t, kMaxTensorRank>& input_shape, uint32_t& input_rank);

LoadResult LoadCNNFromBuffer(const uint8_t* data, std::size_t size, Arena& arena, CNNNetwork*& network,
                             std::array<uint32_t, kMaxTensorRank>& input_shape, uint32_t& input_rank);

LoadResult Load(const char* nk_path, Arena& arena, NetworkKind& kind,
                MLPNetwork*& mlp, CNNNetwork*& cnn,
                std::array<uint32_t, kMaxTensorRank>& input_shape, uint32_t& input_rank);

LoadResult LoadFromBuffer(const uint8_t* data, std::size_t size, Arena& arena, NetworkKind& kind,
                          MLPNetwork*& mlp, CNNNetwork*& cnn,
                          std::array<uint32_t, kMaxTensorRank>& input_shape, uint32_t& input_rank);
}
```

**C equivalents:** `nk_parse_architecture` / `nk_parse_architecture_memory` fill `nk_arch_info_t`. `PrintNetworkSummary` → `nk_arch_print`. `PrintHeader` is a detailed binary dump (no C binding). Embedded firmware uses `LoadFromBuffer` / `LoadMLPFromBuffer` / `LoadCNNFromBuffer` (C: `nk_model_load_memory`).

**High-level C++ usage** loads with `Load` / `LoadMLP` / `LoadCNN` (file) or `LoadFromBuffer` / `LoadMLPFromBuffer` / `LoadCNNFromBuffer` (embedded `.nk` bytes) and calls `forward` directly. The C API adds `nk_model_t` + `nk_model_run` as a convenience wrapper — see [c-api.md](c-api.md).

**AOT firmware:** `python -m netkit aot` generates C++26 or C23 sources with an embedded `.nk` blob and thin wrappers — see [GETTING_STARTED.md](GETTING_STARTED.md#5-aot-compile-embed-nk-in-firmware).

**Format** — full binary layout in [NK_FORMAT.md](NK_FORMAT.md). Convert ONNX with `python -m netkit convert`.

---

## Examples

| File | Build | Description |
|------|-------|-------------|
| [`examples/infer_cpp.cpp`](../examples/infer_cpp.cpp) | `make example-cpp` | Load MLP/CNN and run forward pass via native headers |
| [`examples/infer_c.c`](../examples/infer_c.c) | `make example-c` | Same workflow via `netkit.h` |

---

## NkRegression (`nk_regression.hpp`)

```cpp
namespace NkRegression {
    struct RunSummary { uint32_t passed; uint32_t failed; };
    RunSummary RunModelTests(const char* nk_path);
}
```

Drives regression tests from embedded cases in `.nk` files. Format: [NK_FORMAT.md](NK_FORMAT.md).

---

## CLI (`cli.hpp`)

```cpp
namespace Cli {
    int Run(int argc, char** argv);
}
```

Commands: `test`, `run <model.nk> --input ...`, `inspect <model.nk> [--full]`, `help` / `-h` / `--help`. Full reference: [CLI.md](CLI.md).

---

## Test suite (`test.hpp`)

```cpp
NkRegression::RunSummary run_all_tests();
```

---

## Building as a library

```bash
make lib    # produces libnetkit.a
```

Link your C++ code:

```bash
clang++ -std=c++26 -Iinclude -o my_app my_app.cpp libnetkit.a
```

Exclude `main.cpp`, `cli.cpp`, and `test.cpp` from your link if you provide your own entry point.

---

## Design constraints

- Inference only (no training, no autodiff)
- Single-threaded
- No heap allocation in `MLPNetwork` / `CNNNetwork` layer paths
- Conv and pool layers support independent `pad_h` / `pad_w` per axis (`.nk` v3+); depthwise conv supports non-square `kernel_h` × `kernel_w`

See the [roadmap](../README.md#roadmap) for planned work (quantized types, broader ONNX import).
