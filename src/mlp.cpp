#include "mlp.hpp"
#include "active_kernel.hpp"
#include "activation_followup.hpp"
#include "quant_ops.hpp"
#include "tensor_factory.hpp"

#include <chrono>
#include <cstdint>

namespace
{
    void ForwardLayer(MLPLayer& layer, const Tensor& input, Tensor& output, bool omit_final_softmax)
    {
        NetkitKernelActivation kernel_activation = ToKernelActivation(layer.activation);
        if (omit_final_softmax && kernel_activation == NetkitKernelActivation::Softmax)
            kernel_activation = NetkitKernelActivation::None;
        const bool fused_in_kernel = Kernels::FullyConnectedWithBias(
            input, layer.weights, layer.bias, kernel_activation, output);
        ApplyFusedOutputActivation(kernel_activation, fused_in_kernel, output, layer.leaky_alpha);
    }

    void ForwardQuantizedLayer(MLPLayer& layer,
                               const Tensor& input,
                               Tensor& output,
                               int8_t* quant_scratch,
                               bool omit_final_softmax)
    {
        const bool apply_relu = layer.activation == ActivationType::ReLU;
        const bool apply_softmax =
            layer.activation == ActivationType::Softmax && !omit_final_softmax;
        QuantOps::ForwardQuantizedDense(input,
                                        layer.weights,
                                        layer.bias,
                                        layer.quant,
                                        apply_relu,
                                        apply_softmax,
                                        quant_scratch,
                                        output);
    }

    void ForwardLayerTimed(MLPLayer& layer,
                           const Tensor& input,
                           Tensor& output,
                           bool omit_final_softmax,
                           MLPNetwork::LayerTimingFn timing_fn,
                           void* user_data)
    {
        const auto layer_start = std::chrono::steady_clock::now();
        ForwardLayer(layer, input, output, omit_final_softmax);
        const auto layer_end = std::chrono::steady_clock::now();

        if (timing_fn)
        {
            const uint64_t duration_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(layer_end - layer_start)
                    .count());
            timing_fn("FullyConnected", duration_ns, user_data);
        }
    }

    void ForwardQuantizedLayerTimed(MLPLayer& layer,
                                    const Tensor& input,
                                    Tensor& output,
                                    int8_t* quant_scratch,
                                    bool omit_final_softmax,
                                    MLPNetwork::LayerTimingFn timing_fn,
                                    void* user_data)
    {
        const auto layer_start = std::chrono::steady_clock::now();
        ForwardQuantizedLayer(layer, input, output, quant_scratch, omit_final_softmax);
        const auto layer_end = std::chrono::steady_clock::now();

        if (timing_fn)
        {
            const uint64_t duration_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(layer_end - layer_start)
                    .count());
            timing_fn("FullyConnected", duration_ns, user_data);
        }
    }
}

void MLPLayer::forward(const Tensor& input, Tensor& output)
{
    if (quant.enabled)
        return;
    ForwardLayer(*this, input, output, /*omit_final_softmax=*/false);
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
    ping_i8_a = nullptr;
    ping_i8_b = nullptr;
    max_activation_elements = 0;
    hidden_activation_ = {};
    ping_view_a_ = {};
    ping_view_b_ = {};
    ping_i8_view_a_ = {};
    ping_i8_view_b_ = {};

    if (num_layers <= 1 || !layers)
        return layers != nullptr;

    uint32_t max_hidden_cols = 0;
    for (size_t i = 0; i < num_layers - 1; ++i)
    {
        const uint32_t cols = layers[i].weights.shape[0];
        const uint32_t elements = batch_rows * cols;
        if (elements > max_activation_elements)
            max_activation_elements = elements;
        if (cols > max_hidden_cols)
            max_hidden_cols = cols;
    }

    if (max_activation_elements == 0 || batch_rows == 0)
        return false;

    if (quantized_)
    {
        const std::size_t bytes = static_cast<std::size_t>(max_activation_elements) * sizeof(int8_t);
        ping_i8_a = static_cast<int8_t*>(arena.alloc(bytes, alignof(int8_t)));
        ping_i8_b = static_cast<int8_t*>(arena.alloc(bytes, alignof(int8_t)));
        if (!ping_i8_a || !ping_i8_b)
            return false;

        ping_i8_view_a_ = TensorFactory::View2DInt8(ping_i8_a, batch_rows, max_hidden_cols);
        ping_i8_view_b_ = TensorFactory::View2DInt8(ping_i8_b, batch_rows, max_hidden_cols);
        hidden_activation_ = ping_i8_view_a_;
        return true;
    }

    const std::size_t bytes = static_cast<std::size_t>(max_activation_elements) * sizeof(float);
    ping_a = static_cast<float*>(arena.alloc(bytes, alignof(float)));
    ping_b = static_cast<float*>(arena.alloc(bytes, alignof(float)));
    if (!ping_a || !ping_b)
        return false;

    ping_view_a_ = TensorFactory::View2D(ping_a, batch_rows, max_hidden_cols);
    ping_view_b_ = TensorFactory::View2D(ping_b, batch_rows, max_hidden_cols);
    hidden_activation_ = ping_view_a_;
    return true;
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
    layers[layer_idx].quant.enabled = false;
}

void MLPNetwork::InitQuantizedLayer(uint32_t layer_idx,
                                    const Tensor& weights,
                                    const Tensor& bias,
                                    ActivationType activation,
                                    const NkFormat::MlpLayerQuantDesc& quant,
                                    float leaky_alpha)
{
    if (!layers || layer_idx >= num_layers)
        return;

    layers[layer_idx].weights = weights;
    layers[layer_idx].bias = bias;
    layers[layer_idx].activation = activation;
    layers[layer_idx].leaky_alpha = leaky_alpha;
    layers[layer_idx].quant.params = quant;
    layers[layer_idx].quant.enabled = true;
}

void MLPNetwork::forward(const Tensor& input, Tensor& output, Arena& /*arena*/)
{
    if (!IsValid() || !HasActivationBuffers() || num_layers == 0)
        return;

    if (quantized_)
    {
        if (input.type != DataType::Int8)
            return;

        if (num_layers == 1)
        {
            ForwardQuantizedLayer(layers[0], input, output, ping_i8_a, omit_final_softmax_);
            return;
        }

        if (num_layers == 2)
        {
            ForwardQuantizedLayer(layers[0], input, hidden_activation_, ping_i8_b, false);
            ForwardQuantizedLayer(layers[1], hidden_activation_, output, ping_i8_b, omit_final_softmax_);
            return;
        }

        const Tensor* current_input = &input;
        Tensor* write_view = &ping_i8_view_a_;

        for (size_t i = 0; i < num_layers; ++i)
        {
            if (i == num_layers - 1)
            {
                ForwardQuantizedLayer(layers[i], *current_input, output, ping_i8_b, omit_final_softmax_);
                return;
            }

            const uint32_t rows = current_input->shape[0];
            const uint32_t cols = layers[i].weights.shape[0];
            if (rows * cols > max_activation_elements)
                return;

            *write_view = TensorFactory::View2DInt8(write_view == &ping_i8_view_a_ ? ping_i8_a : ping_i8_b,
                                                    rows,
                                                    cols);
            ForwardQuantizedLayer(layers[i], *current_input, *write_view, ping_i8_b, false);
            current_input = write_view;
            write_view = (write_view == &ping_i8_view_a_) ? &ping_i8_view_b_ : &ping_i8_view_a_;
        }
        return;
    }

    if (num_layers == 1)
    {
        ForwardLayer(layers[0], input, output, omit_final_softmax_);
        return;
    }

    if (num_layers == 2)
    {
        ForwardLayer(layers[0], input, hidden_activation_, false);
        ForwardLayer(layers[1], hidden_activation_, output, omit_final_softmax_);
        return;
    }

    const Tensor* current_input = &input;
    Tensor* write_view = &ping_view_a_;

    for (size_t i = 0; i < num_layers; ++i)
    {
        if (i == num_layers - 1)
        {
            ForwardLayer(layers[i], *current_input, output, omit_final_softmax_);
            return;
        }

        const uint32_t rows = current_input->shape[0];
        const uint32_t cols = layers[i].weights.shape[0];
        if (rows * cols > max_activation_elements)
            return;

        *write_view = TensorFactory::View2D(write_view == &ping_view_a_ ? ping_a : ping_b, rows, cols);
        ForwardLayer(layers[i], *current_input, *write_view, false);
        current_input = write_view;
        write_view = (write_view == &ping_view_a_) ? &ping_view_b_ : &ping_view_a_;
    }
}

void MLPNetwork::forward_timed(const Tensor& input,
                               Tensor& output,
                               LayerTimingFn timing_fn,
                               void* user_data)
{
    if (!IsValid() || !HasActivationBuffers() || num_layers == 0)
        return;

    if (quantized_)
    {
        if (input.type != DataType::Int8)
            return;

        if (num_layers == 1)
        {
            ForwardQuantizedLayerTimed(
                layers[0], input, output, ping_i8_a, omit_final_softmax_, timing_fn, user_data);
            return;
        }

        if (num_layers == 2)
        {
            ForwardQuantizedLayerTimed(
                layers[0], input, hidden_activation_, ping_i8_b, false, timing_fn, user_data);
            ForwardQuantizedLayerTimed(layers[1],
                                       hidden_activation_,
                                       output,
                                       ping_i8_b,
                                       omit_final_softmax_,
                                       timing_fn,
                                       user_data);
            return;
        }

        const Tensor* current_input = &input;
        Tensor* write_view = &ping_i8_view_a_;

        for (size_t i = 0; i < num_layers; ++i)
        {
            if (i == num_layers - 1)
            {
                ForwardQuantizedLayerTimed(layers[i],
                                           *current_input,
                                           output,
                                           ping_i8_b,
                                           omit_final_softmax_,
                                           timing_fn,
                                           user_data);
                return;
            }

            const uint32_t rows = current_input->shape[0];
            const uint32_t cols = layers[i].weights.shape[0];
            if (rows * cols > max_activation_elements)
                return;

            *write_view = TensorFactory::View2DInt8(write_view == &ping_i8_view_a_ ? ping_i8_a : ping_i8_b,
                                                    rows,
                                                    cols);
            ForwardQuantizedLayerTimed(
                layers[i], *current_input, *write_view, ping_i8_b, false, timing_fn, user_data);
            current_input = write_view;
            write_view = (write_view == &ping_i8_view_a_) ? &ping_i8_view_b_ : &ping_i8_view_a_;
        }
        return;
    }

    if (num_layers == 1)
    {
        ForwardLayerTimed(layers[0], input, output, omit_final_softmax_, timing_fn, user_data);
        return;
    }

    if (num_layers == 2)
    {
        ForwardLayerTimed(layers[0], input, hidden_activation_, false, timing_fn, user_data);
        ForwardLayerTimed(
            layers[1], hidden_activation_, output, omit_final_softmax_, timing_fn, user_data);
        return;
    }

    const Tensor* current_input = &input;
    Tensor* write_view = &ping_view_a_;

    for (size_t i = 0; i < num_layers; ++i)
    {
        if (i == num_layers - 1)
        {
            ForwardLayerTimed(
                layers[i], *current_input, output, omit_final_softmax_, timing_fn, user_data);
            return;
        }

        const uint32_t rows = current_input->shape[0];
        const uint32_t cols = layers[i].weights.shape[0];
        if (rows * cols > max_activation_elements)
            return;

        *write_view = TensorFactory::View2D(write_view == &ping_view_a_ ? ping_a : ping_b, rows, cols);
        ForwardLayerTimed(layers[i], *current_input, *write_view, false, timing_fn, user_data);
        current_input = write_view;
        write_view = (write_view == &ping_view_a_) ? &ping_view_b_ : &ping_view_a_;
    }
}
