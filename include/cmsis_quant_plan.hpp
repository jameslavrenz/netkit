#pragma once

#include "quant_plan_types.hpp"
#include "quant_output.hpp"

#include <cstddef>
#include <cstdint>

class Arena;
class CNNNetwork;
class Tensor;

namespace CmsisQuantPlan
{

struct Runtime
{
    LayerPlan* layers = nullptr;
    uint32_t num_layers = 0;
    int8_t* act_a = nullptr;
    int8_t* act_b = nullptr;
    uint32_t act_a_bytes = 0;
    uint32_t act_b_bytes = 0;
    int8_t* logits = nullptr;
    uint32_t logits_elements = 0;
    // Expected int8 input element count (prequantized in Python; no C++ float→int8).
    uint32_t input_quant_elements = 0;
    uint8_t* workspace = nullptr;
    std::size_t workspace_bytes = 0;
    float input_scale = 0.0f;
    int32_t input_zero_point = 0;
    // When true, DenseSoftmax writes FC logits and skips Softmax (argmax-equivalent).
    bool omit_final_softmax = false;
    // xnn_weights_cache_t when XNNPACK enabled; owned by Runtime, deleted in DestroyRuntime.
    void* xnn_weights_cache = nullptr;
    // Shared xnn_workspace_t (TF Lite XNNPACK delegate default); owned by Runtime.
    void* xnn_workspace = nullptr;
    // Full-network XNNPACK qs8 subgraph (MobileNetV4 ImageNet int8, etc.).
    void* xnn_network_runtime = nullptr;  // xnn_runtime_t
    bool xnn_network_ready = false;
    uint32_t xnn_net_ext_in = 0;
    uint32_t xnn_net_ext_out = 1;
    // Skip xnn_setup_runtime_v2 when I/O addresses are unchanged.
    const int8_t* bound_input = nullptr;
    int8_t* bound_output = nullptr;
    // Owned per-channel bias scale arrays for network subgraph tensors.
    float** xnn_net_bias_scales = nullptr;
    uint32_t xnn_net_bias_scales_count = 0;
};

bool BuildRuntime(CNNNetwork& network, Arena& arena, uint32_t in_h, uint32_t in_w, uint32_t in_c);

void DestroyRuntime(Runtime& runtime);

bool Forward(Runtime& runtime,
             CNNNetwork& network,
             const Tensor& input,
             QuantOutputFormat output_format,
             Tensor& output_cache);

bool ForwardInt8(Runtime& runtime,
                 CNNNetwork& network,
                 const int8_t* input,
                 uint32_t input_elements,
                 QuantOutputFormat output_format,
                 Tensor& output_cache);

bool ForwardInt8ToBuffer(Runtime& runtime,
                         CNNNetwork& network,
                         const int8_t* input,
                         int8_t* output,
                         uint32_t output_elements);

#if defined(NETKIT_STAGE_TIMING) && NETKIT_STAGE_TIMING
void ResetStageTiming();
void PrintStageTimingSummary();
// Accumulate time spent inside arm_convolve_wrapper_s8 (debug builds only).
void RecordCmsisConvKernelUs(double us);
#endif

}  // namespace CmsisQuantPlan
