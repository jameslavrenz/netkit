# TinyRT - Neural Network Inference Engine

TinyRT is a lightweight, high-performance inference engine written in modern C++ (C++17/20) designed for neural network inference on microcontroller units (MCUs) and microprocessor units (MPUs). It provides an optimized runtime for deploying trained neural network models directly on resource-constrained embedded devices.

## Features

- **MLP Layer Abstraction** - Simplified API for building stacked fully connected networks without manual sequencing
- **CNN Layer Abstraction** - High-level interface for composing multi-layer convolutional networks
- **Dense/MLP Layers** - Full support for fully connected neural network layers with multiple activation functions
- **2D Convolution** - Efficient 2D convolutional operations (1D convolution can be performed via 2D)
- **Activation Functions** - Comprehensive set of activation functions:
  - ReLU, LeakyReLU, ReLU6
  - Sigmoid, Tanh
  - Softmax
- **Memory-Efficient Arena Allocator** - Custom memory management for embedded environments
- **N-Dimensional Tensor Support** - Flexible tensor operations for various model architectures
- **NHWC Layout** - Optimized for common embedded tensor memory layout

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
- **test.hpp/cpp** - Comprehensive test suite for validation

## Building

### Requirements
- C++17 or later compiler (clang++, g++, MSVC)
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
clang++ -std=c++17 -g -Iinclude -o main src/main.cpp src/test.cpp src/arena.cpp src/tensor_factory.cpp src/tensor_access.cpp src/ops.cpp src/conv2d.cpp src/mlp.cpp src/cnn.cpp src/inference.cpp
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
Fill(W1, (float[]){1, 2, 3, 4, 5, 6});

Tensor B1 = Create2D(arena, 1, 3);
Fill(B1, (float[]){0, 0, 0});

mlp.InitLayer(0, W1, B1, ActivationType::ReLU);

// Setup Layer 2: 3 hidden units -> 1 output (linear)
Tensor W2 = Create2D(arena, 3, 1);
Fill(W2, (float[]){1, 2, 3});

Tensor B2 = Create2D(arena, 1, 1);
Fill(B2, (float[]){0});

mlp.InitLayer(1, W2, B2, ActivationType::None);

// Forward pass through entire network
Tensor input = Create2D(arena, 1, 2);
Fill(input, (float[]){1.0f, 2.0f});

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
Fill(x, (float[]){1.0f, 2.0f});

// Create weights and bias
Tensor W = Create2D(arena, 2, 3);
Fill(W, (float[]){1, 2, 3, 4, 5, 6});

Tensor b = Create2D(arena, 1, 3);
Fill(b, (float[]){0, 0, 0});

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

## Project Structure

```
tinyrt/
├── include/
│   ├── arena.hpp           # Memory management
│   ├── tensor.hpp          # Tensor definitions
│   ├── tensor_factory.hpp  # Tensor utilities
│   ├── tensor_access.hpp   # Tensor access functions
│   ├── ops.hpp             # Neural network operations
│   ├── conv2d.hpp          # 2D convolution (low-level)
│   ├── mlp.hpp             # MLP network abstraction
│   ├── cnn.hpp             # CNN network abstraction
│   ├── inference.hpp       # Inference utilities
│   └── test.hpp            # Test suite
├── src/
│   ├── main.cpp            # Entry point
│   ├── arena.cpp           # Memory management implementation
│   ├── tensor_factory.cpp  # Tensor utilities implementation
│   ├── tensor_access.cpp   # Tensor access implementation
│   ├── ops.cpp             # Operations implementation
│   ├── conv2d.cpp          # 2D convolution implementation
│   ├── mlp.cpp             # MLP network implementation
│   ├── cnn.cpp             # CNN network implementation
│   ├── inference.cpp       # Inference utilities implementation
│   └── test.cpp            # Test suite implementation
├── Makefile                # Build configuration
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## Memory Management

TinyRT uses a bump-pointer **arena allocator** instead of `new`/`delete` or `malloc`/`free`. All tensor data and network-internal state (layer arrays, intermediate output buffers) are allocated from a single pre-sized buffer you provide.

```cpp
unsigned char buffer[4096];
Arena arena;
arena.init(buffer, 4096);

// All Create* calls and network constructors draw from this arena
MLPNetwork mlp(2, arena);
Tensor input = Create2D(arena, 1, 2);
```

Key points:
- **`Arena::init(buffer, size)`** — Bind the arena to a fixed memory region (stack buffer, static RAM, etc.)
- **`Arena::alloc(size)`** — Bump-allocate bytes; used internally by tensors and network abstractions
- **`Arena::reset()`** — Reset the bump pointer to reuse the buffer across inference runs
- **No destructors** — `MLPNetwork` and `CNNNetwork` do not free memory; the arena owns everything

Size your buffer to hold weights, biases, intermediate activations, and the network object's internal arrays. If the arena runs out of space, allocations will overrun the buffer — there is no automatic growth.

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

- **Lightweight** - Minimal dependencies, suitable for embedded environments
- **Memory-Conscious** - Arena allocator throughout; no dynamic heap allocation in layer abstractions
- **Single-Threaded** - No parallelization overhead (suitable for MCUs)
- **Inference-Only** - Optimized for deployment, not training
- **Type-Safe** - Modern C++ with strong typing

## Roadmap

### Near Term
- Enhanced quantization support (int8, uint8)
- Batch normalization layer
- Pooling operations (max, average)

### Future Enhancements
- On-device fine-tuning capabilities
- Transformer architecture support
- Automatic differentiation (autodiff) for training
- Optimized kernels for specific MCU/MPU architectures
- Model serialization/deserialization

## Testing

Run the comprehensive test suite:

```bash
make run
```

Tests include:
- MLP forward pass validation
- 2D convolution with various channel configurations
- Multi-channel to single-channel convolution
- Multi-channel to multi-channel convolution

## Performance Considerations

- Tensors use row-major (C-style) layout
- NHWC format for convolutional layers (optimized for MCU memory patterns)
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
