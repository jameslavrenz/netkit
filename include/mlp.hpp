#pragma once
#include "tensor.hpp"
#include "arena.hpp"
#include "layer_quant.hpp"
#include "quant_output.hpp"

enum class ActivationType
{
    None,
    ReLU,
    Sigmoid,
    Tanh,
    LeakyReLU,
    ReLU6,
    Softmax
};

struct MLPLayer
{
    Tensor weights;      // Weight matrix [out_features, in_features] (CMSIS-NN / PyTorch layout)
    Tensor bias;         // Bias vector [1, out_features]
    ActivationType activation;
    float leaky_alpha;   // Only used if activation is LeakyReLU
    LayerQuant quant;

    // Forward pass: output = activation(FC(input, weights) + bias)
    void forward(const Tensor& input, Tensor& output);
};

class MLPNetwork
{
private:
    MLPLayer* layers;
    uint32_t num_layers;
    bool quantized_ = false;
    bool omit_final_softmax_ = false;
    QuantOutputFormat quant_output_format_ = QuantOutputFormat::Int8;
    float* ping_a{};
    float* ping_b{};
    int8_t* ping_i8_a{};
    int8_t* ping_i8_b{};
    uint32_t max_activation_elements{};
    Tensor hidden_activation_{};
    Tensor ping_view_a_{};
    Tensor ping_view_b_{};
    Tensor ping_i8_view_a_{};
    Tensor ping_i8_view_b_{};

public:
    // Constructor allocates from Arena — no heap fragmentation
    MLPNetwork(uint32_t num_layers, Arena& arena);
    
    // No destructor needed - Arena manages all memory

    bool IsValid() const { return layers != nullptr; }

    bool IsQuantized() const { return quantized_; }

    bool HasActivationBuffers() const
    {
        if (num_layers <= 1)
            return true;
        if (quantized_)
            return ping_i8_a != nullptr && ping_i8_b != nullptr;
        return ping_a != nullptr && ping_b != nullptr;
    }

    // Preallocate two ping-pong activation buffers sized to the largest hidden layer.
    bool InitActivationBuffers(Arena& arena, uint32_t batch_rows);

    // Initialize a layer at the given index
    void InitLayer(uint32_t layer_idx, const Tensor& weights, const Tensor& bias, 
                   ActivationType activation, float leaky_alpha = 0.01f);

    void InitQuantizedLayer(uint32_t layer_idx,
                            const Tensor& weights,
                            const Tensor& bias,
                            ActivationType activation,
                            const NkFormat::MlpLayerQuantDesc& quant,
                            float leaky_alpha = 0.01f);

    void SetQuantized(bool enabled) { quantized_ = enabled; }

    void SetQuantOutputFormat(QuantOutputFormat format) { quant_output_format_ = format; }

    QuantOutputFormat GetQuantOutputFormat() const { return quant_output_format_; }

    // When true, a final Dense Softmax is skipped and logits are written instead.
    // Argmax(logits) == Argmax(softmax(logits)); use for classification benches.
    void SetOmitFinalSoftmax(bool omit) { omit_final_softmax_ = omit; }

    bool OmitFinalSoftmax() const { return omit_final_softmax_; }

    // Forward pass through entire network (hidden activations reuse ping_a / ping_b)
    void forward(const Tensor& input, Tensor& output, Arena& arena);

    using LayerTimingFn = void (*)(const char* tag, uint64_t duration_ns, void* user_data);

    // Benchmark-only: per-layer timing callback (tag is "FullyConnected" per dense layer).
    void forward_timed(const Tensor& input, Tensor& output, LayerTimingFn timing_fn, void* user_data);

    // Get a specific layer
    MLPLayer& GetLayer(uint32_t idx) { return layers[idx]; }
};
