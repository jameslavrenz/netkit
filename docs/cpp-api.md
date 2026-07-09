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

`Arena::kDefaultCapacity` / `NK_ARENA_DEFAULT_CAPACITY`: **64 KiB** (MCU), **64 MiB** (CPU and MPU).

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
    // kDefaultCapacity: 64 KiB (MCU), 64 MiB (CPU/MPU) — see netkit_config.h

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
    void Fill(Tensor& t, std::span<const float> values);
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

    // Benchmark-only: same as forward but invokes timing_fn(layer_idx, tag, elapsed_us, user_data) per layer
    void forward_timed(const Tensor& input, Tensor& output, LayerTimingFn timing_fn, void* user_data);

    MLPLayer& GetLayer(uint32_t idx);
};
```

Weight matrix shape per layer: `[out_features, in_features]` row-major (CMSIS-NN / PyTorch layout).

### Manual construction (call order)

Most firmware uses `NkLoader::LoadMLP` / `LoadCNN` or `nk_mlp_load` / `nk_cnn_load`. To wire layers by hand:

1. **`Arena::init`** (or `init_heap`) — bind caller-owned memory.
2. **`MLPNetwork(num_layers, arena)`** — allocates the layer array from the arena.
3. **`InitLayer(i, …)`** for each `i` from `0` to `num_layers - 1` — bind weight/bias tensors (usually views into flash or a prior arena alloc).
4. **`InitActivationBuffers(arena, batch_rows)`** — **after all layers**; `batch_rows` is the input batch (typically `1`). Allocates ping-pong hidden buffers sized to the largest intermediate.
5. **`forward(input, output, arena)`** — last layer writes to `output`; hidden layers reuse ping-pong buffers.

```cpp
alignas(std::max_align_t) static unsigned char memory[65536];
Arena arena;
arena.init(memory, sizeof(memory));

MLPNetwork net(2, arena);
net.InitLayer(0, W0, B0, ActivationType::ReLU);
net.InitLayer(1, W1, B1, ActivationType::None);
if (!net.InitActivationBuffers(arena, /*batch_rows=*/1))
    return;  // arena overflow

Tensor input = TensorFactory::View2D(in_data, 1, in_features);
Tensor output = TensorFactory::View2D(out_data, 1, out_features);
net.forward(input, output, arena);
```

C equivalent: [c-api.md](c-api.md#mlp-manual-construction-call-order) (`nk_mlp_create` → `nk_mlp_init_layer` → `nk_mlp_init_activation_buffers` → `nk_mlp_forward`).

---

## CNNNetwork (`cnn.hpp`)

CNN pipelines support mixed blocks: conv2d, depthwise_conv2d, max_pool2d, avg_pool2d, batch_norm2d, layernorm2d, convnextv2_block, mobilenetv4_uib, resnet_basic_block, yolox_decoupled_head, flatten, and dense (classification head). See [NK_FORMAT.md](NK_FORMAT.md), [CONVNEXTV2.md](CONVNEXTV2.md), [MOBILENETV4.md](MOBILENETV4.md), [RESNET18.md](RESNET18.md), and [YOLOX.md](YOLOX.md).

```cpp
enum class CnnBlockType {
    Conv2D, DepthwiseConv2D, MaxPool2D, AvgPool2D, BatchNorm2d, LayerNorm2d,
    Flatten, Dense, ConvNeXtV2Block, MobilenetV4Uib, ResNetBasicBlock,
    YoloxDecoupledHead
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
                       int pad_w = 0,
                       int pad_h_end = -1,
                       int pad_w_end = -1);
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
                                int pad_w = 0,
                                int pad_h_end = -1,
                                int pad_w_end = -1);
    void InitPoolLayer(uint32_t layer_idx,
                       int pool_h,
                       int pool_w,
                       int stride,
                       int pad_h = 0,
                       int pad_w = 0,
                       int pad_h_end = -1,
                       int pad_w_end = -1);
    void InitAvgPoolLayer(uint32_t layer_idx,
                          int pool_h,
                          int pool_w,
                          int stride,
                          int pad_h = 0,
                          int pad_w = 0,
                          int pad_h_end = -1,
                          int pad_w_end = -1);
    void InitBatchNormLayer(uint32_t layer_idx, int channels, float* scale, float* bias);
    void InitLayerNormLayer(uint32_t layer_idx, int channels, float eps, float* weight, float* bias);
    void InitConvNeXtV2BlockLayer(uint32_t layer_idx, Arena& arena, uint32_t spatial_h, uint32_t spatial_w,
                                  int channels, float eps, /* weight pointers */);
    void InitMobilenetV4UibLayer(uint32_t layer_idx, Arena& arena, uint32_t spatial_h, uint32_t spatial_w,
                                 int in_channels, int out_channels, /* UIB config + weights */);
    void InitResNetBasicBlockLayer(uint32_t layer_idx, Arena& arena, uint32_t spatial_h, uint32_t spatial_w,
                                   int in_channels, int out_channels, int stride, /* conv/BN weights */);
    void InitYoloxDecoupledHeadLayer(uint32_t layer_idx, Arena& arena, uint32_t spatial_h, uint32_t spatial_w,
                                     int in_channels, int hidden_dim, int num_classes, int num_convs,
                                     /* stem/branch/pred weight pointers */);
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

### Manual construction (call order)

1. **`Arena::init`** — bind caller-owned memory (size with `inspect --full` or `nk_inspect_model` when possible).
2. **`CNNNetwork(num_layers, arena)`** — allocates the block array from the arena.
3. **`Init*Layer(layer_idx, …)`** for each index **`0 … num_layers - 1` in forward order** — configure one block type per index. Primitive layers only need weight pointers; **composite blocks** also take `Arena&` plus **`spatial_h` / `spatial_w`** (the feature-map height/width **at that layer's input**) and may allocate fused scratch from the arena during this call.
4. **`InitActivationBuffers(arena, in_h, in_w, in_c)`** — **after every layer is configured**. Uses the **network input** NHWC shape (same as the tensor passed to `forward`), not the last layer's output shape. Allocates ping-pong activation buffers and (on CMSIS-NN builds) the shared kernel workspace — see [ARENA.md](ARENA.md#kernel-workspace-cmsis-nn).
5. **`forward(input, arena)`** — input must be rank-3 NHWC until a `Flatten` block; result is in **`GetOutput()`** (also returned by reference). Returns a tensor with null `data` on arena overflow.

**Hybrid primitive pipeline** (conv → pool → batch norm → flatten → dense):

```cpp
CNNNetwork net(5, arena);
net.InitConvLayer(0, 3, 1, 3, 16, conv_w, conv_b, ConvActivationType::ReLU, 0.01f, 1, 1);
net.InitPoolLayer(1, 2, 2, 0, 0);
net.InitBatchNormLayer(2, 16, bn_scale, bn_bias);
net.InitFlattenLayer(3);
net.InitDenseLayer(4, W, B, ActivationType::None);
if (!net.InitActivationBuffers(arena, in_h, in_w, in_c))
    return;
Tensor input = /* NHWC [in_h, in_w, in_c] */;
Tensor& output = net.forward(input, arena);
```

**Composite blocks** (same index order; pass spatial size at the block input):

| Block | Init function | Notes |
|-------|---------------|--------|
| ConvNeXt V2 | `InitConvNeXtV2BlockLayer(idx, arena, h, w, …)` | Fused scratch from arena |
| MobileNetV4 UIB | `InitMobilenetV4UibLayer(idx, arena, h, w, …)` | |
| ResNet BasicBlock | `InitResNetBasicBlockLayer(idx, arena, h, w, …)` | |
| YOLOX decoupled head | `InitYoloxDecoupledHeadLayer(idx, arena, h, w, …)` | See [YOLOX.md](YOLOX.md#manual-construction) |

**YOLOX head only** (single fused layer on backbone features):

```cpp
CNNNetwork net(1, arena);
net.InitYoloxDecoupledHeadLayer(
    0, arena,
    spatial_h, spatial_w,   // e.g. 2, 2 for 2×2 feature map
    in_channels,            // backbone output depth (e.g. 960)
    hidden_dim, num_classes, num_convs,
    stem_w, stem_b,
    cls_conv_w, cls_conv_b,  // arrays of num_convs pointers
    reg_conv_w, reg_conv_b,
    cls_pred_w, cls_pred_b,
    reg_pred_w, reg_pred_b,
    obj_pred_w, obj_pred_b);
if (!net.InitActivationBuffers(arena, spatial_h, spatial_w, in_channels))
    return;
Tensor input = /* NHWC [spatial_h, spatial_w, in_channels] */;
Tensor& output = net.forward(input, arena);  // [H, W, 4+1+num_classes]
```

C equivalent: [c-api.md](c-api.md#cnn-manual-construction-call-order) (`nk_cnn_create` → `nk_cnn_init_*` → `nk_cnn_init_activation_buffers` → `nk_cnn_forward`).

Prefer **`.nk` load** when weights live in a file or flash blob: `NkLoader::LoadCNN` / `nk_cnn_load` / `nk_model_load` perform steps 2–4 automatically.

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

void PrintNetworkSummary(const char* nk_path, const ParsedModel& model);

LoadResult LoadMLP(const char* nk_path, Arena& arena, MLPNetwork*& network,
                   std::array<uint32_t, kMaxTensorRank>& input_shape, uint32_t& input_rank);
// When NETKIT_USE_MMAP=1 (CPU default on macOS/Linux; opt-in on Linux MPU):
//   mmap MAP_PRIVATE; arena owns mapping until reset/destroy.
// Otherwise: fread into arena. Prefer Load*FromBuffer / flash on MCU and RTOS MPU.

LoadResult LoadMLPFromBuffer(const uint8_t* data, std::size_t size, Arena& arena, MLPNetwork*& network,
                             std::array<uint32_t, kMaxTensorRank>& input_shape, uint32_t& input_rank);

LoadResult LoadCNN(const char* nk_path, Arena& arena, CNNNetwork*& network,
                   std::array<uint32_t, kMaxTensorRank>& input_shape, uint32_t& input_rank);

LoadResult LoadCNNFromBuffer(const uint8_t* data, std::size_t size, Arena& arena, CNNNetwork*& network,
                             std::array<uint32_t, kMaxTensorRank>& input_shape, uint32_t& input_rank);

LoadResult Load(const char* nk_path, Arena& arena, NetworkKind& kind,
                MLPNetwork*& mlp, CNNNetwork*& cnn,
                std::array<uint32_t, kMaxTensorRank>& input_shape, uint32_t& input_rank);
}
```

**C equivalents:** `nk_parse_architecture` / `nk_parse_architecture_memory` fill `nk_arch_info_t`. `PrintNetworkSummary` → `nk_arch_print`. Embedded firmware uses `LoadMLPFromBuffer` / `LoadCNNFromBuffer` (C: `nk_mlp_load_memory` / `nk_cnn_load_memory`, or `nk_model_load_memory` for the combined handle). `IsQuantized()` → `nk_mlp_is_quantized` / `nk_cnn_is_quantized` / `nk_model_is_quantized`.

**High-level C++ usage** loads with `Load` / `LoadMLP` / `LoadCNN` (file) or `LoadMLPFromBuffer` / `LoadCNNFromBuffer` (embedded `.nk` bytes) and calls `forward` directly — the **interpreter path** via `NkOpsResolver`. The C API adds `nk_model_t` + `nk_model_run` as a convenience wrapper, plus typed `nk_mlp_load_memory` / `nk_cnn_load_memory` — see [c-api.md](c-api.md).

**Compiled firmware:** `python -m netkit aot` generates C++26 or C23 sources. Default **C++ lowered** output is a static `Kernels::` call chain (`kLowered = true`) with coefs in flash `.rodata`. **C AOT** embeds the `.nk` blob and uses `nk_model_load_memory`. See [GETTING_STARTED.md](GETTING_STARTED.md#5-aot-compile-embed-nk-in-firmware), [API_PARITY.md](API_PARITY.md), and [PHILOSOPHY.md](PHILOSOPHY.md#deployment-modes-interpreter-or-compiled).

**Format** — full binary layout in [NK_FILE_SPECIFICATION.md](NK_FILE_SPECIFICATION.md) (byte-level) and [NK_FORMAT.md](NK_FORMAT.md) (overview). Convert ONNX with `python -m netkit convert`.

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
