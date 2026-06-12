#include <iostream>
#include "arena.hpp"
#include "tensor.hpp"
#include "tensor_factory.hpp"
#include "tensor_access.hpp"
#include "ops.hpp"
#include "conv2d.hpp"
#include "mlp.hpp"
#include "cnn.hpp"

using namespace TensorFactory;
using namespace Ops;

void test_mlp_abstraction(){

    std::cout << "\n============================\n";
    std::cout << " MLP ABSTRACTION TEST START\n";
    std::cout << "============================\n\n";

    unsigned char buffer[2048];
    Arena arena;
    arena.init(buffer, 2048);

    // Create a 2-layer MLP network
    // Layer 1: 2 -> 2 with ReLU
    // Layer 2: 2 -> 2 with None (linear)

    MLPNetwork mlp(2);

    // =====================================================
    // Setup Layer 1
    // =====================================================
    std::cout << "[1] Setting up Layer 1 (2 -> 2 with ReLU)\n";

    Tensor W1 = Create2D(arena, 2, 2);
    Fill(W1, (float[]){
        1, 1,
        1, 1
    });

    Tensor B1 = Create2D(arena, 1, 2);
    Fill(B1, (float[]){0, 0});

    mlp.InitLayer(0, W1, B1, ActivationType::ReLU);

    // =====================================================
    // Setup Layer 2
    // =====================================================
    std::cout << "[2] Setting up Layer 2 (2 -> 2 linear)\n";

    Tensor W2 = Create2D(arena, 2, 2);
    Fill(W2, (float[]){
        1, 0,
        0, 1
    });

    Tensor B2 = Create2D(arena, 1, 2);
    Fill(B2, (float[]){0, 0});

    mlp.InitLayer(1, W2, B2, ActivationType::None);

    // =====================================================
    // Forward pass
    // =====================================================
    std::cout << "\n[3] Forward pass\n";

    Tensor input = Create2D(arena, 1, 2);
    Fill(input, (float[]){1, 2});

    std::cout << "Input:\n";
    Print(input);

    Tensor output = Create2D(arena, 1, 2);
    mlp.forward(input, output, arena);

    std::cout << "\nFinal Output:\n";
    Print(output);

    std::cout << "\n============================\n";
    std::cout << " MLP ABSTRACTION TEST COMPLETE\n";
    std::cout << "============================\n";
}

void test_cnn_abstraction(){

    std::cout << "\n============================\n";
    std::cout << " CNN ABSTRACTION TEST START\n";
    std::cout << "============================\n\n";

    unsigned char buffer[4096];
    Arena arena;
    arena.init(buffer, 4096);

    // Create a 2-layer CNN
    // Layer 1: 1x1 kernel, 2 input channels, 1 output channel (with ReLU)
    // Layer 2: 1x1 kernel, 1 input channel, 2 output channels (with None)

    CNNNetwork cnn(2);

    // =====================================================
    // Setup Layer 1
    // =====================================================
    std::cout << "[1] Setting up Layer 1 (1x1 conv, 2->1 channels, ReLU)\n";

    // For 1x1 kernel with 2 input channels and 1 output channel:
    // weights layout: [out][kh][kw][in] = [1][1][1][2]
    float weights1[] = {1.0f, 2.0f};  // weight for output channel 0: ic0=1, ic1=2
    float bias1[] = {0.0f};

    cnn.InitLayer(0, 
                  1,           // kernel_size
                  1,           // stride
                  2,           // in_channels
                  1,           // out_channels
                  weights1, 
                  bias1,
                  ConvActivationType::ReLU);

    // =====================================================
    // Setup Layer 2
    // =====================================================
    std::cout << "[2] Setting up Layer 2 (1x1 conv, 1->2 channels, None)\n";

    // For 1x1 kernel with 1 input channel and 2 output channels:
    // weights layout: [out][kh][kw][in] = [2][1][1][1]
    float weights2[] = {1.0f, 2.0f};  // oc0=1, oc1=2
    float bias2[] = {0.0f, 0.0f};

    cnn.InitLayer(1,
                  1,           // kernel_size
                  1,           // stride
                  1,           // in_channels
                  2,           // out_channels
                  weights2,
                  bias2,
                  ConvActivationType::None);

    // =====================================================
    // Create input: 2x2 spatial, 2 channels (NHWC layout)
    // =====================================================
    std::cout << "\n[3] Creating input tensor (2x2 spatial, 2 channels)\n";

    float input_data[] = {
        1.0f, 10.0f,    // Pixel(0,0): ch0=1, ch1=10
        2.0f, 20.0f,    // Pixel(0,1): ch0=2, ch1=20
        
        3.0f, 30.0f,    // Pixel(1,0): ch0=3, ch1=30
        4.0f, 40.0f     // Pixel(1,1): ch0=4, ch1=40
    };

    Tensor input;
    input.data = input_data;
    input.rank = 3;
    input.shape[0] = 2;  // H
    input.shape[1] = 2;  // W
    input.shape[2] = 2;  // C

    std::cout << "Input tensor shape: [" << input.shape[0] << ", " << input.shape[1] << ", " << input.shape[2] << "]\n";

    // =====================================================
    // Forward pass
    // =====================================================
    std::cout << "\n[4] Forward pass\n";

    Tensor& output = cnn.forward(input, arena);

    std::cout << "Output tensor shape: [" << output.shape[0] << ", " << output.shape[1] << ", " << output.shape[2] << "]\n";
    std::cout << "Output values:\n";
    Print(output);

    std::cout << "\n============================\n";
    std::cout << " CNN ABSTRACTION TEST COMPLETE\n";
    std::cout << "============================\n";
}

void test_mlp(){

    std::cout << "\n============================\n";
    std::cout << " TinyRT MLP TEST START\n";
    std::cout << "============================\n\n";

    unsigned char buffer[1024];
    Arena arena;
    arena.init(buffer, 1024);

    // =====================================================
    // 📍 INPUT LAYER
    // =====================================================
    std::cout << "[1] Creating input tensor...\n";

    Tensor x = Create2D(arena, 1, 2);
    Fill(x, (float[]){1, 2});

    std::cout << "Input x:\n";
    Print(x);

    // =====================================================
    // 📍 LAYER 1
    // =====================================================
    std::cout << "\n[2] Layer 1: MatMul start\n";

    Tensor W1 = Create2D(arena, 2, 2);
    Fill(W1, (float[]){
        1, 1,
        1, 1
    });

    std::cout << "W1:\n";
    Print(W1);

    Tensor B1 = Create2D(arena, 1, 2);
    Fill(B1, (float[]){0, 0});

    std::cout << "B1:\n";
    Print(B1);

    std::cout << "--> MatMul(x, W1)\n";
    Tensor h1 = Create2D(arena, 1, 2);
    MatMul(x, W1, h1);

    std::cout << "After MatMul (h1 raw):\n";
    Print(h1);

    std::cout << "--> Add Bias (h1 + B1)\n";
    MatAdd(h1, B1, h1);

    std::cout << "After Bias Add:\n";
    Print(h1);

    std::cout << "--> ReLU Activation\n";
    ReLU(h1, h1);

    std::cout << "After ReLU (h1 final):\n";
    Print(h1);

    // =====================================================
    // 📍 LAYER 2
    // =====================================================
    std::cout << "\n[3] Layer 2: MatMul start\n";

    Tensor W2 = Create2D(arena, 2, 2);
    Fill(W2, (float[]){
        1, 0,
        0, 1
    });

    std::cout << "W2:\n";
    Print(W2);

    Tensor B2 = Create2D(arena, 1, 2);
    Fill(B2, (float[]){0, 0});

    std::cout << "B2:\n";
    Print(B2);

    std::cout << "--> MatMul(h1, W2)\n";

    Tensor out = Create2D(arena, 1, 2);
    MatMul(h1, W2, out);

    std::cout << "After MatMul (out raw):\n";
    Print(out);

    std::cout << "--> Add Bias (out + B2)\n";
    MatAdd(out, B2, out);

    std::cout << "Final Output:\n";
    Print(out);

    // =====================================================
    // 📍 DONE
    // =====================================================
    std::cout << "\n============================\n";
    std::cout << " MLP TEST COMPLETE\n";
    std::cout << "============================\n";

}

void cnn_test(){

    // 4x4 input (all ones)
    float input_data[16] =
    {
        1,1,1,1,
        1,1,1,1,
        1,1,1,1,
        1,1,1,1
    };

    Tensor input;
    input.data = input_data;
    input.rank = 3;
    input.shape[0] = 4;
    input.shape[1] = 4;
    input.shape[2] = 1;

    // 2x2 output
    float output_data[4] = {0};

    Tensor output;
    output.data = output_data;
    output.rank = 3;
    output.shape[0] = 2;
    output.shape[1] = 2;
    output.shape[2] = 1;

    // weights: 1 filter, 3x3x1
    float weights[9] =
    {
        1,0,1,
        0,0,0,
        1,0,1
    };

    float bias[1] = {0};

    Conv2D conv;
    conv.kernel_size = 3;
    conv.stride = 1;
    conv.in_channels = 1;
    conv.out_channels = 1;
    conv.weights = weights;
    conv.bias = bias;

    conv.forward(input, output);

    float* out = tensor_data_f32(output);

    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            std::cout << out[i * 2 + j] << " ";
        }
        std::cout << "\n";
    }

}

void multi_in_single_out_channel_cnn_test(){

std::cout << "\n===== MULTI-CHANNEL CNN TEST =====\n";

// NHWC layout:
//
// Pixel(0,0): ch0=1  ch1=10
// Pixel(0,1): ch0=2  ch1=20
// Pixel(1,0): ch0=3  ch1=30
// Pixel(1,1): ch0=4  ch1=40

float input_data[] =
{
    1, 10,
    2, 20,

    3, 30,
    4, 40
};

Tensor input;
input.data = input_data;
input.rank = 3;

input.shape[0] = 2;   // H
input.shape[1] = 2;   // W
input.shape[2] = 2;   // C

float output_data[4] = {0};

Tensor output;
output.data = output_data;
output.rank = 3;

output.shape[0] = 2;
output.shape[1] = 2;
output.shape[2] = 1;

// kernel = 1x1
//
// oc=0
// ic=0 -> weight=1
// ic=1 -> weight=2

float weights[] =
{
    1,
    2
};

float bias[] =
{
    0
};

Conv2D conv;

conv.kernel_size = 1;
conv.stride = 1;

conv.in_channels  = 2;
conv.out_channels = 1;

conv.weights = weights;
conv.bias    = bias;

conv.forward(input, output);

std::cout << "\nExpected:\n";
std::cout << "21 42\n";
std::cout << "63 84\n";

std::cout << "\nActual:\n";

for (int h = 0; h < 2; h++)
{
    for (int w = 0; w < 2; w++)
    {
        std::cout << output_data[h * 2 + w] << " ";
    }

    std::cout << "\n";
}

}

void multi_in_multi_out_channel_cnn_test(){

std::cout << "\n===== MULTI-CHANNEL / MULTI-FILTER TEST =====\n";

// NHWC layout
//
// Pixel(0,0): ch0=1  ch1=10
// Pixel(0,1): ch0=2  ch1=20
// Pixel(1,0): ch0=3  ch1=30
// Pixel(1,1): ch0=4  ch1=40

float input_data[] =
{
    1, 10,
    2, 20,

    3, 30,
    4, 40
};

Tensor input;
input.data = input_data;
input.rank = 3;

input.shape[0] = 2;   // H
input.shape[1] = 2;   // W
input.shape[2] = 2;   // C

// Output: 2x2x2

float output_data[8] = {0};

Tensor output;
output.data = output_data;
output.rank = 3;

output.shape[0] = 2;
output.shape[1] = 2;
output.shape[2] = 2;

// ----------------------------------------
// Kernel 0:
//
// output0 = ch0*1 + ch1*2
//
// Kernel 1:
//
// output1 = ch0*3 + ch1*4
// ----------------------------------------

float weights[] =
{
    // oc = 0
    1, 2,

    // oc = 1
    3, 4
};

float bias[] =
{
    0,
    0
};

Conv2D conv;

conv.kernel_size = 1;
conv.stride = 1;

conv.in_channels  = 2;
conv.out_channels = 2;

conv.weights = weights;
conv.bias    = bias;

conv.forward(input, output);

// --------------------------------------------------
// Expected results:
//
// Pixel(0,0):
//   oc0 = 1*1 + 10*2 = 21
//   oc1 = 1*3 + 10*4 = 43
//
// Pixel(0,1):
//   oc0 = 2*1 + 20*2 = 42
//   oc1 = 2*3 + 20*4 = 86
//
// Pixel(1,0):
//   oc0 = 3*1 + 30*2 = 63
//   oc1 = 3*3 + 30*4 = 129
//
// Pixel(1,1):
//   oc0 = 4*1 + 40*2 = 84
//   oc1 = 4*3 + 40*4 = 172
// --------------------------------------------------

std::cout << "\nExpected:\n";
std::cout << "(21,  43)  (42,  86)\n";
std::cout << "(63, 129)  (84, 172)\n";

std::cout << "\nActual:\n";

for (int h = 0; h < 2; h++)
{
    for (int w = 0; w < 2; w++)
    {
        int base = (h * 2 + w) * 2;

        std::cout
            << "("
            << output_data[base + 0]
            << ", "
            << output_data[base + 1]
            << ") ";
    }

    std::cout << "\n";
}

}
