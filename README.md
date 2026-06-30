# netkit — Neural Network Kit

netkit is a C++26 neural network kit for on-device inference on MCUs and MPUs. It is developed and validated on the desktop, then deployed to embedded targets. Companion to [memkit](https://github.com/jameslavrenz/memkit) for memory management.

Models are loaded from JSON architecture files and companion float32 `.bin` weight files during host-side development and testing.

## Features

- **MLP Layer Abstraction** - Simplified API for building stacked fully connected networks without manual sequencing
- **CNN Layer Abstraction** - High-level interface for composing multi-layer convolutional networks
- **Dense/MLP Layers** - Full support for fully connected neural network layers with multiple activation functions
- **2D Convolution** - Efficient 2D convolutional operations (1D convolution can be performed via 2D)
- **Activation Functions** - Comprehensive set of activation functions:
  - ReLU, LeakyReLU, ReLU6
  - Sigmoid, Tanh
  - Softmax
- **Memory-Efficient Arena Allocator** - Custom bump-pointer memory management
- **N-Dimensional Tensor Support** - Flexible tensor operations for various model architectures
- **NHWC Layout** - Common channel-last tensor layout for conv layers
- **Model Loader** - JSON architecture files with companion float32 `.bin` weight files
- **Vectors Test Runner** - Declarative input/expected test cases in `*.vectors.json`

## Architecture

### Core Components

- **arena.hpp/cpp** - Memory arena allocator for efficient memory management
- **tensor.hpp** - Tensor data structure and type definitions
- **tensor_factory.hpp/cpp** - Tensor creation and manipulation utilities
- **tensor_access.hpp/cpp** - Direct tensor data access and indexing functions
- **ops.hpp/cpp** - Core neural network operations (linear algebra, activations)
- **conv2d.hpp/cpp** - 2D convolution implementation
- **mlp.hpp/cpp** - MLP network abstraction for layered fully connected networks
- **cnn.hpp/cpp** - CNN network abstraction for layered convolutional networks
- **json_parser.hpp/cpp** - Minimal JSON parser for model architecture files
- **model_loader.hpp/cpp** - Load networks from JSON + `.bin` weight files
- **vectors_loader.hpp/cpp** - Generic test runner driven by `*.vectors.json` files
- **test.hpp/cpp** - Test suite orchestration and summary reporting

## Building

### Requirements
- C++26 compiler (clang++ 17+, g++ 14+, or MSVC with `/std:c++latest`)
- Make (for Unix-like systems)

### Build Commands

```bash
# Build the project
make

# Clean build artifacts
make clean

# Rebuild from scratch
make rebuild

# Build and run tests
make run
```

### Manual Compilation

```bash
clang++ -std=c++26 -g -Iinclude -o netkit src/main.cpp src/test.cpp src/arena.cpp src/tensor_factory.cpp src/tensor_access.cpp src/ops.cpp src/conv2d.cpp src/mlp.cpp src/cnn.cpp src/inference.cpp src/json_parser.cpp src/model_loader.cpp src/vectors_loader.cpp
```

## Usage Example

### MLP Network with Abstraction (Recommended)

```cpp
#include "arena.hpp"
#include "mlp.hpp"
#include "tensor_factory.hpp"

using namespace TensorFactory;

// Create memory arena
unsigned char buffer[2048];
Arena arena;
arena.init(buffer, 2048);

// Create a 2-layer MLP network (internal state allocated from arena)
MLPNetwork mlp(2, arena);

// Setup Layer 1: 2 inputs -> 3 hidden units with ReLU
Tensor W1 = Create2D(arena, 2, 3);
Fill(W1, {1, 2, 3, 4, 5, 6});

Tensor B1 = Create2D(arena, 1, 3);
Fill(B1, {0, 0, 0});

mlp.InitLayer(0, W1, B1, ActivationType::ReLU);

// Setup Layer 2: 3 hidden units -> 1 output (linear)
Tensor W2 = Create2D(arena, 3, 1);
Fill(W2, {1, 2, 3});

Tensor B2 = Create2D(arena, 1, 1);
Fill(B2, {0});

mlp.InitLayer(1, W2, B2, ActivationType::None);

// Forward pass through entire network
Tensor input = Create2D(arena, 1, 2);
Fill(input, {1.0f, 2.0f});

Tensor output = Create2D(arena, 1, 1);
mlp.forward(input, output, arena);

Print(output);
```

### MLP (Dense Layer) - Low-level Interface

```cpp
#include "arena.hpp"
#include "tensor.hpp"
#include "tensor_factory.hpp"
#include "ops.hpp"

using namespace TensorFactory;
using namespace Ops;

// Create memory arena
unsigned char buffer[1024];
Arena arena;
arena.init(buffer, 1024);

// Create input tensor
Tensor x = Create2D(arena, 1, 2);
Fill(x, {1.0f, 2.0f});

// Create weights and bias
Tensor W = Create2D(arena, 2, 3);
Fill(W, {1, 2, 3, 4, 5, 6});

Tensor b = Create2D(arena, 1, 3);
Fill(b, {0, 0, 0});

// Forward pass
Tensor h = Create2D(arena, 1, 3);
MatMul(x, W, h);
MatAdd(h, b, h);
ReLU(h, h);

Print(h);
```

### CNN Network with Abstraction (Recommended)

```cpp
#include "arena.hpp"
#include "cnn.hpp"
#include "tensor_factory.hpp"

using namespace TensorFactory;

// Create memory arena
unsigned char buffer[4096];
Arena arena;
arena.init(buffer, 4096);

// Create a 2-layer CNN (internal state allocated from arena)
CNNNetwork cnn(2, arena);

// Layer 1: 3x3 kernel, 1 input channel, 16 output channels with ReLU
float weights1[9 * 16] = {...};  // 3x3x1x16 weights
float bias1[16] = {...};

cnn.InitLayer(0,
              3,          // kernel_size
              1,          // stride
              1,          // in_channels
              16,         // out_channels
              weights1,
              bias1,
              ConvActivationType::ReLU);

// Layer 2: 1x1 kernel, 16 input channels, 32 output channels with ReLU
float weights2[1 * 1 * 16 * 32] = {...};
float bias2[32] = {...};

cnn.InitLayer(1,
              1,          // kernel_size
              1,          // stride
              16,         // in_channels
              32,         // out_channels
              weights2,
              bias2,
              ConvActivationType::ReLU);

// Create input tensor (HWC format)
Tensor input;
input.data = input_data;  // Your input data
input.rank = 3;
input.shape[0] = 32;  // height
input.shape[1] = 32;  // width
input.shape[2] = 1;   // channels

// Forward pass through entire network
Tensor& output = cnn.forward(input, arena);

Print(output);
```

### 2D Convolution - Low-level Interface

```cpp
#include "conv2d.hpp"

// Input: 4x4 with 1 channel
float input_data[16] = {1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1};
Tensor input;
input.data = input_data;
input.rank = 3;
input.shape[0] = 4; // height
input.shape[1] = 4; // width
input.shape[2] = 1; // channels

// Output: 2x2 with 1 channel
float output_data[4] = {0};
Tensor output;
output.data = output_data;
output.rank = 3;
output.shape[0] = 2;
output.shape[1] = 2;
output.shape[2] = 1;

// 3x3 kernel
float weights[9] = {1,0,1, 0,0,0, 1,0,1};
float bias[1] = {0};

Conv2D conv;
conv.kernel_size = 3;
conv.stride = 1;
conv.in_channels = 1;
conv.out_channels = 1;
conv.weights = weights;
conv.bias = bias;

conv.forward(input, output);
```

### Model File Bundles

Each model is a set of files sharing the same base name:

| File | Purpose |
|------|---------|
| `model.json` | Network architecture (layers, activations, input shape) |
| `model.bin` | Raw float32 weights in layer order (weights then bias per layer) |
| `model.vectors.json` | Test inputs and expected outputs for regression tests |

**Workflow**: edit architecture in JSON, export weights to `.bin`, add regression cases to `.vectors.json`, run `make run`.

### Loading a Model from JSON + Weights

**MLP example** (`models/test_mlp.json`):

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

**CNN example** (`models/test_cnn.json`):

```json
{
  "version": 1,
  "network": "cnn",
  "input": [2, 2, 2],
  "layers": [
    { "type": "conv2d", "kernel_size": 1, "stride": 1, "filters": 1, "activation": "relu" },
    { "type": "conv2d", "kernel_size": 1, "stride": 1, "filters": 2, "activation": "none" }
  ]
}
```

**Binary weight layout**

- **Dense layer**: `W[in_features × out_features]` row-major, then `b[out_features]`
- **Conv2D layer**: `W[out × kernel × kernel × in_channels]` (same layout as `Conv2D`), then `b[out_channels]`

**C++ usage**:

```cpp
#include "arena.hpp"
#include "model_loader.hpp"
#include "tensor_factory.hpp"

unsigned char buffer[8192];
Arena arena;
arena.init(buffer, 8192);

MLPNetwork* network = nullptr;
std::array<uint32_t, kMaxTensorRank> input_shape{};
uint32_t input_rank = 0;

ModelLoader::LoadResult result =
    ModelLoader::LoadMLP("models/test_mlp.json", arena, network, input_shape, input_rank);

if (result.status == ModelLoader::LoadStatus::Ok)
{
    Tensor input = TensorFactory::Create2D(arena, input_shape[0], input_shape[1]);
    TensorFactory::Fill(input, {1.0f, 2.0f});

    Tensor output = TensorFactory::Create2D(arena, 1, network->GetLayer(1).weights.shape[1]);
    network->forward(input, output, arena);
}
```

Use `ModelLoader::LoadCNN()` for convolutional models, or `ModelLoader::Load()` to auto-detect the network type from JSON.

Supported activations in JSON: `none`, `relu`, `sigmoid`, `tanh`, `leaky_relu`, `relu6`, `softmax`. Optional `"alpha"` field for `leaky_relu`.

### Vectors Test Files

Test cases live beside the model in `*.vectors.json`. Each file references its model and lists input/expected pairs:

```json
{
  "model": "test_mlp.json",
  "cases": [
    {
      "name": "2-layer forward",
      "input": [1, 2],
      "expected": [3, 3]
    }
  ]
}
```

`VectorsLoader::RunVectorsFile()` resolves paths, loads the model, runs each case, and prints `PASS`/`FAIL`. `run_all_tests()` in `test.cpp` drives all vector files and returns a pass/fail summary; `main()` exits with code 1 on any failure.

## Project Structure

```
netkit/
├── include/
│   ├── arena.hpp           # Memory management
│   ├── tensor.hpp          # Tensor definitions
│   ├── tensor_factory.hpp  # Tensor utilities
│   ├── tensor_access.hpp   # Tensor access functions
│   ├── ops.hpp             # Neural network operations
│   ├── conv2d.hpp          # 2D convolution (low-level)
│   ├── mlp.hpp             # MLP network abstraction
│   ├── cnn.hpp             # CNN network abstraction
│   ├── json_parser.hpp     # Minimal JSON parser
│   ├── model_loader.hpp    # JSON + .bin model loader
│   ├── vectors_loader.hpp  # Vectors-driven test runner
│   ├── inference.hpp       # Inference utilities
│   └── test.hpp            # Test suite
├── models/
│   ├── test_mlp.json                  # 2-layer MLP test model
│   ├── test_mlp.bin
│   ├── test_mlp.vectors.json          # MLP regression cases
│   ├── mlp_hand.json                  # Hand-designed 3→4→2 MLP
│   ├── mlp_hand.bin
│   ├── mlp_hand.vectors.json
│   ├── test_cnn.json                  # 2-layer CNN test model
│   ├── test_cnn.bin
│   ├── test_cnn.vectors.json          # CNN regression cases
│   ├── cnn_4x4_single.json            # Single-layer 3x3 conv on 4x4 input
│   ├── cnn_4x4_single.bin
│   ├── cnn_4x4_single.vectors.json    # Spatial conv regression cases
│   ├── cnn_hand.json                  # Hand-designed 2-channel spatial CNN
│   ├── cnn_hand.bin
│   └── cnn_hand.vectors.json
├── tools/
│   └── write_hand_models.py           # Regenerate mlp_hand.bin / cnn_hand.bin
├── src/
│   ├── main.cpp            # Entry point
│   ├── arena.cpp           # Memory management implementation
│   ├── tensor_factory.cpp  # Tensor utilities implementation
│   ├── tensor_access.cpp   # Tensor access implementation
│   ├── ops.cpp             # Operations implementation
│   ├── conv2d.cpp          # 2D convolution implementation
│   ├── mlp.cpp             # MLP network implementation
│   ├── cnn.cpp             # CNN network implementation
│   ├── json_parser.cpp     # JSON parser implementation
│   ├── model_loader.cpp    # Model loader implementation
│   ├── vectors_loader.cpp  # Vectors test runner implementation
│   ├── inference.cpp       # Inference utilities implementation
│   └── test.cpp            # Test suite implementation
├── Makefile                # Build configuration
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## Memory Management

netkit uses a bump-pointer **arena allocator** instead of `new`/`delete` or `malloc`/`free`. All tensor data and network-internal state (layer arrays, intermediate output buffers) are allocated from a single pre-sized buffer you provide.

```cpp
unsigned char buffer[4096];
Arena arena;
arena.init(buffer, 4096);

// All Create* calls and network constructors draw from this arena
MLPNetwork mlp(2, arena);
Tensor input = Create2D(arena, 1, 2);
```

Key points:
- **`Arena::kDefaultCapacity`** — 64 KiB default buffer used by the test runner
- **`Arena::init(buffer, size)`** — Bind the arena to a fixed memory region (stack buffer, static storage, etc.)
- **`Arena::alloc(size)`** — Bump-allocate bytes; returns `nullptr` if the arena is full
- **`Arena::remaining()`** — Bytes still available in the arena
- **`Arena::reset()`** — Reset the bump pointer to reuse the buffer across inference runs
- **No destructors** — `MLPNetwork` and `CNNNetwork` do not free memory; the arena owns everything

Size your buffer to hold weights, biases, intermediate activations, and the network object's internal arrays. When `alloc()` returns `nullptr`, treat it as an out-of-memory condition — there is no automatic growth.

## Layer Abstractions

### MLPNetwork
Simplifies building stacked fully connected layers:
- **`MLPNetwork(num_layers, arena)`** — Construct network; layer and intermediate tensor arrays are arena-allocated (no heap)
- **`InitLayer(idx, weights, bias, activation, alpha)`** — Configure a dense layer
- **`forward(input, output, arena)`** — Run forward pass through all layers; intermediate tensors allocated from arena
- **`GetLayer(idx)`** — Access individual layer configuration

Supported activations: `None`, `ReLU`, `Sigmoid`, `Tanh`, `LeakyReLU`, `ReLU6`, `Softmax`

### CNNNetwork
Simplifies building stacked convolutional layers:
- **`CNNNetwork(num_layers, arena)`** — Construct network; layer and intermediate tensor arrays are arena-allocated (no heap)
- **`InitLayer(idx, kernel_size, stride, in_channels, out_channels, weights, bias, activation, alpha)`** — Configure a conv layer
- **`forward(input, arena)`** — Run forward pass through all layers (returns output tensor reference); intermediate tensors allocated from arena
- **`GetLayer(idx)`** — Access individual layer configuration
- **`GetOutput()`** — Get the final output tensor

Automatically calculates output dimensions and manages intermediate tensors.

### ModelLoader
Load a trained network from disk:
- **`JsonPathToBinPath(json_path, bin_path, capacity)`** — Derive companion `.bin` path from `.json` path
- **`LoadWeightsBin(json_path, arena, weights, float_count)`** — Load float32 weights into arena
- **`LoadMLP(json_path, arena, network, input_shape, input_rank)`** — Parse JSON and build an MLP
- **`LoadCNN(json_path, arena, network, input_shape, input_rank)`** — Parse JSON and build a CNN
- **`Load(json_path, arena, kind, mlp, cnn, input_shape, input_rank)`** — Auto-detect network type

## Supported Operations

### Arithmetic
- `MatMul()` - Matrix multiplication
- `MatAdd()` - Element-wise addition
- `Mul()` - Element-wise multiplication
- `MulScalar()` - Scalar multiplication
- `MatAddND()` - N-dimensional addition
- `MulND()` - N-dimensional multiplication

### Activation Functions
- `ReLU()` - Rectified Linear Unit
- `LeakyReLU()` - Leaky ReLU with configurable alpha
- `ReLU6()` - ReLU capped at 6
- `Sigmoid()` - Sigmoid activation
- `Tanh()` - Hyperbolic tangent
- `Softmax()` - Softmax normalization

## Design Principles

- **Lightweight** - Minimal dependencies, standard C++ only
- **Memory-Conscious** - Arena allocator throughout; no dynamic heap allocation in layer abstractions
- **Single-Threaded** - Straightforward sequential execution
- **Inference-Only** - Forward pass only, not training
- **Type-Safe** - Modern C++26 with strong typing (`std::array`, `std::span`)

## Roadmap

### Near Term
- Enhanced quantization support (int8, uint8)
- Batch normalization layer
- Pooling operations (max, average)

### Future Enhancements
- On-device fine-tuning capabilities
- Transformer architecture support
- Automatic differentiation (autodiff) for training
- Model serialization/deserialization

## Testing

Run the full test suite:

```bash
make run
```

Tests are data-driven: each `models/*.vectors.json` file references a model (`*.json` + `*.bin`) and defines named cases with flat `input`/`expected` arrays. No network architecture or weights are hardcoded in source.

Current coverage:

- MLP 2-layer forward pass (`test_mlp.vectors.json`)
- MLP 3→4→2 hand-designed weights (`mlp_hand.vectors.json`)
- CNN 2-layer forward pass — channel stacking (`test_cnn.vectors.json`)
- CNN single-layer 3×3 conv on 4×4 input (`cnn_4x4_single.vectors.json`)
- CNN 2-channel 3×3 spatial conv + 1×1 mix (`cnn_hand.vectors.json`)

To add a test: create or extend a `.vectors.json` file, add a case, and register the file in `run_all_tests()` in `src/test.cpp`.

## Performance Considerations

- Tensors use row-major (C-style) layout
- NHWC format for convolutional layers
- In-place operations where possible to reduce memory allocation
- Linear indexing for fast tensor access
- Call `arena.reset()` between inference batches to reuse the same buffer without freeing individual allocations

## License

[Add license information here]

## Contributing

Contributions are welcome! Please ensure:
- Code follows the existing style
- All tests pass
- New features include test coverage
- Documentation is updated

## Contact & Support

[Add contact information here]
