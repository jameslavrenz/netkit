#include "mlp.hpp"
#include "active_kernel.hpp"
#include "activation_followup.hpp"
#include "tensor_factory.hpp"

using namespace TensorFactory;

void MLPLayer::forward(const Tensor& input, Tensor& output)
{
    const NetkitKernelActivation kernel_activation = ToKernelActivation(activation);
    const bool fused_in_kernel =
        Kernels::FullyConnectedWithBias(input, weights, bias, kernel_activation, output);
    ApplyFusedOutputActivation(kernel_activation, fused_in_kernel, output, leaky_alpha);
}

MLPNetwork::MLPNetwork(uint32_t num_layers, Arena& arena)
    : layers(nullptr), num_layers(num_layers)
{
    layers = static_cast<MLPLayer*>(arena.alloc(sizeof(MLPLayer) * num_layers, alignof(MLPLayer)));
}

bool MLPNetwork::InitActivationBuffers(Arena& arena, uint32_t batch_rows)
{
    ping_a = nullptr;
    ping_b = nullptr;
    max_activation_elements = 0;

    if (num_layers <= 1 || !layers)
        return layers != nullptr;

    for (uint32_t i = 0; i < num_layers - 1; ++i)
    {
        const uint32_t cols = layers[i].weights.shape[0];
        const uint32_t elements = batch_rows * cols;
        if (elements > max_activation_elements)
            max_activation_elements = elements;
    }

    if (max_activation_elements == 0)
        return false;

    const std::size_t bytes = static_cast<std::size_t>(max_activation_elements) * sizeof(float);
    ping_a = static_cast<float*>(arena.alloc(bytes, alignof(float)));
    ping_b = static_cast<float*>(arena.alloc(bytes, alignof(float)));
    return ping_a != nullptr && ping_b != nullptr;
}

void MLPNetwork::InitLayer(uint32_t layer_idx,
                           const Tensor& weights,
                           const Tensor& bias,
                           ActivationType activation,
                           float leaky_alpha)
{
    if (!layers || layer_idx >= num_layers)
        return;

    layers[layer_idx].weights = weights;
    layers[layer_idx].bias = bias;
    layers[layer_idx].activation = activation;
    layers[layer_idx].leaky_alpha = leaky_alpha;
}

void MLPNetwork::forward(const Tensor& input, Tensor& output, Arena& /*arena*/)
{
    if (!IsValid() || !HasActivationBuffers() || num_layers == 0)
        return;

    if (num_layers == 1)
    {
        layers[0].forward(input, output);
        return;
    }

    Tensor current_input = input;
    float* write_buffer = ping_a;

    for (uint32_t i = 0; i < num_layers; ++i)
    {
        if (i == num_layers - 1)
        {
            layers[i].forward(current_input, output);
            return;
        }

        const uint32_t rows = current_input.shape[0];
        const uint32_t cols = layers[i].weights.shape[0];
        Tensor layer_output = View2D(write_buffer, rows, cols);
        if (layer_output.num_elements > max_activation_elements)
            return;

        layers[i].forward(current_input, layer_output);
        current_input = layer_output;
        write_buffer = (write_buffer == ping_a) ? ping_b : ping_a;
    }
}
