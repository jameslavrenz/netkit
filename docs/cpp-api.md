# C++ API Reference (C++26)

Headers live in [`include/`](../include/). All implementation files use `-std=c++26`.

**Numeric type:** inference uses **float32 only** today — see [DATATYPES.md](DATATYPES.md) for the quantized-type roadmap (float16, int16, int8, int4 planned).

## Core headers

| Header | Purpose |
|--------|---------|
| `arena.hpp` | Bump-pointer arena allocator |
| `tensor.hpp` | `Tensor`, `DataType`, `kMaxTensorRank` |
| `tensor_factory.hpp` | Tensor creation, fill, print |
| `tensor_access.hpp` | NHWC indexing helpers |
| `ops.hpp` | Matrix ops and activations |
| `conv2d.hpp` | Low-level 2D convolution |
| `mlp.hpp` | `MLPNetwork`, `MLPLayer`, `ActivationType` |
| `cnn.hpp` | `CNNNetwork`, `Conv2DLayer`, `ConvActivationType` |
| `json_parser.hpp` | Minimal JSON parser (`Json::` namespace) |
| `model_loader.hpp` | JSON + `.bin` model loading |
| `vectors_loader.hpp` | Vectors-driven test runner |
| `cli.hpp` | CLI dispatch (`Cli::Run`) |
| `test.hpp` | Test suite entry (`run_all_tests`) |

For a stable C interface from C++ projects or embedded firmware, use [`netkit.h`](c-api.md). Every C++ public symbol has a C equivalent — see [`API_PARITY.md`](API_PARITY.md).

---

## Arena (`arena.hpp`)

See [ARENA.md](ARENA.md) for the full bump-allocator guide.

```cpp
struct Arena {
    static constexpr std::size_t kDefaultCapacity = 64 * 1024;

    void init(void* memory, std::size_t size);
    void* alloc(std::size_t size, std::size_t alignment);  // alignment: power of two
    void reset();
    std::size_t remaining() const;
};
```

`alloc` inserts padding when the current offset is not a multiple of `alignment`. Use `alignof(float)` for tensor payloads and weight blobs; use `alignof(T)` for struct arrays and placement-new targets.

**Why alignment matters:** weight `.bin` files can have an odd float count, leaving the arena offset at 4 mod 8 on 64-bit platforms. Without padding, a following `MLPNetwork` or `CNNNetwork` allocation would be misaligned for `placement new`. The engine passes the correct `alignof` at every internal call site.

All network and tensor allocations during load/inference draw from the arena. No `free()` — call `reset()` to reuse the buffer.

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
    void PrintLabeled(const char* label, const Tensor& t);
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

Low-level valid convolution (no padding).

```cpp
struct Conv2D {
    int kernel_size, stride, in_channels, out_channels;
    float* weights;  // [out][kh][kw][in]
    float* bias;     // [out]
    void forward(const Tensor& input, Tensor& output);
};
```

Output spatial size: `(input - kernel) / stride + 1`.

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

Weight matrix shape per layer: `[in_features, out_features]` row-major.

---

## CNNNetwork (`cnn.hpp`)

CNN pipelines support mixed blocks: conv2d, max_pool2d, flatten, and dense (classification head). See [MODEL_FORMAT.md](MODEL_FORMAT.md).

```cpp
enum class CnnBlockType { Conv2D, MaxPool2D, Flatten, Dense };

class CNNNetwork {
public:
    CNNNetwork(uint32_t num_layers, Arena& arena);
    bool IsValid() const;
    bool HasActivationBuffers() const;

    bool InitActivationBuffers(Arena& arena, uint32_t in_h, uint32_t in_w, uint32_t in_c);  // LoadCNN

    void InitConvLayer(uint32_t layer_idx, ...);   // conv2d + activation
    void InitPoolLayer(uint32_t layer_idx, int pool_size, int stride);
    void InitFlattenLayer(uint32_t layer_idx);
    void InitDenseLayer(uint32_t layer_idx, const Tensor& W, const Tensor& b,
                        ActivationType activation, float leaky_alpha = 0.01f);

    void InitLayer(...);  // alias for InitConvLayer (pure conv stacks)

    // All blocks ping-pong between two load-time buffers; result in GetOutput()
    Tensor& forward(const Tensor& input, Arena& arena);
    CnnBlock& GetBlock(uint32_t idx);
    Tensor& GetOutput();
};
```

Spatial tensors stay NHWC until flatten; dense head output is `[1, units]`. Returns null `data` on arena overflow.

`ModelLoader::LoadCNN` builds full pipelines from JSON (including `models/mnist_cnn.json`).

---

## ModelLoader (`model_loader.hpp`)

```cpp
namespace ModelLoader {

enum class NetworkKind { Unknown, MLP, CNN };

enum class LoadStatus {
    Ok, JsonOpenFailed, BinOpenFailed, JsonParseFailed,
    UnsupportedNetwork, VersionMismatch, LayerConfigError,
    BinSizeMismatch, ArenaOverflow
};

struct LoadResult {
    LoadStatus status;
    NetworkKind kind;
    const char* message;
};

struct ArchitectureSpec { /* version, kind, input_shape, layers, expected_weight_floats */ };

LoadResult ParseArchitecture(const char* json_path, ArchitectureSpec& spec);
uint32_t ComputeMlpOutputElements(const ArchitectureSpec& spec);
uint32_t ComputeCnnOutputElements(const ArchitectureSpec& spec);
void PrintArchitecture(const ArchitectureSpec& spec);  // CLI inspect --full
void PrintNetworkSummary(const char* json_path, const ArchitectureSpec& spec);
void PrintWeightsSummary(const char* json_path, float* weights,
                         std::size_t float_count, std::size_t expected_float_count);

bool JsonPathToBinPath(const char* json_path, char* bin_path, std::size_t capacity);

LoadStatus LoadWeightsBin(const char* json_path, Arena& arena,
                          float*& weights, std::size_t& float_count,
                          const char** error_message = nullptr);

LoadResult LoadMLP(const char* json_path, Arena& arena, MLPNetwork*& network,
                   std::array<uint32_t, kMaxTensorRank>& input_shape, uint32_t& input_rank);

LoadResult LoadCNN(const char* json_path, Arena& arena, CNNNetwork*& network,
                   std::array<uint32_t, kMaxTensorRank>& input_shape, uint32_t& input_rank);

LoadResult Load(const char* json_path, Arena& arena, NetworkKind& kind,
                MLPNetwork*& mlp, CNNNetwork*& cnn,
                std::array<uint32_t, kMaxTensorRank>& input_shape, uint32_t& input_rank);
}
```

**C equivalents:** `nk_parse_architecture` fills `nk_arch_info_t` with `input_elements` and `output_elements` (same as `ComputeMlpOutputElements` / `ComputeCnnOutputElements`). `PrintNetworkSummary` → `nk_arch_print`. `PrintArchitecture` and `PrintWeightsSummary` are CLI `--full` diagnostics only (no C binding).

**High-level C++ usage** loads with `Load` / `LoadMLP` / `LoadCNN` and calls `forward` directly. The C API adds `nk_model_t` + `nk_model_run` as a convenience wrapper — see [c-api.md](c-api.md).

**JSON format** — full schema in [MODEL_FORMAT.md](MODEL_FORMAT.md).

```json
{
  "version": 1,
  "network": "mlp",
  "input": [1, 2],
  "layers": [
    { "type": "dense", "units": 2, "activation": "relu" },
    { "type": "dense", "units": 2, "activation": "none" }
  ]
}
```

**Weight layout (`.bin`)** — see [MODEL_FORMAT.md](MODEL_FORMAT.md) for byte order and CNN layout.

- Dense: `W[in×out]` row-major, then `b[out]`
- Conv2D: `W[out×k×k×in]`, then `b[out]`

---

## Examples

| File | Build | Description |
|------|-------|-------------|
| [`examples/infer_cpp.cpp`](../examples/infer_cpp.cpp) | `make example-cpp` | Load MLP/CNN and run forward pass via native headers |
| [`examples/infer_c.c`](../examples/infer_c.c) | `make example-c` | Same workflow via `netkit.h` |

---

## VectorsLoader (`vectors_loader.hpp`)

```cpp
namespace VectorsLoader {
    struct RunSummary { uint32_t passed; uint32_t failed; };
    RunSummary RunVectorsFile(const char* vectors_path);
}
```

Drives regression tests from `*.vectors.json` files. Format and tolerance: [VECTORS_TESTS.md](VECTORS_TESTS.md).

---

## CLI (`cli.hpp`)

```cpp
namespace Cli {
    int Run(int argc, char** argv);
}
```

Commands: `test`, `run <model.json> --input ...`, `inspect <model.json> [--full]`, `help` / `-h` / `--help`. Full reference: [CLI.md](CLI.md).

---

## Test suite (`test.hpp`)

```cpp
VectorsLoader::RunSummary run_all_tests();
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
- Convolutions are valid-only (no padding yet)

See the [roadmap](../README.md#roadmap) for planned ops (pooling, batch norm, quantization).
