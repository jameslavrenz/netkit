#pragma once

#include "netkit_config.h"
#include "nk_format.hpp"
#include "quant_plan_types.hpp"

#include <cstdint>

struct Arena;  // defined in arena.hpp
struct MobileNetV4Uib;  // defined in mobilenetv4_uib.hpp
class CNNNetwork;
class MLPNetwork;

namespace CmsisQuantPlan
{
struct Runtime;
}

// XNNPACK int8 (qs8) adapters for netkit quantized conv / depthwise / pool / FC.
// Used on cpu/mpu when NETKIT_XNNPACK=1 (same flag as float32 LayerFast).
namespace XnnpackQuant
{
#if defined(NETKIT_USE_XNNPACK) && NETKIT_USE_XNNPACK && NETKIT_XNNPACK_ALLOWED
constexpr bool kEnabled = true;
#else
constexpr bool kEnabled = false;
#endif

// Create persistent XNNPACK ops (create + reshape + workspace). Prefer Arena*
// for workspace; if arena is null or alloc fails, heap-owns workspace.
#if defined(NETKIT_USE_XNNPACK) && NETKIT_USE_XNNPACK && NETKIT_XNNPACK_ALLOWED
bool CreateConv2dNhwcQuantPlan(CmsisQuantPlan::Conv2DPlan& plan,
                               const int8_t* weights,
                               const int32_t* bias,
                               Arena* arena,
                               void* weights_cache = nullptr);
bool CreateDepthwiseConv2dNhwcQuantPlan(CmsisQuantPlan::DepthwiseConv2DPlan& plan,
                                        const int8_t* weights_chw,
                                        const int32_t* bias,
                                        Arena* arena,
                                        void* weights_cache = nullptr);
bool CreateMaxPool2dNhwcQuantPlan(CmsisQuantPlan::Pool2DPlan& plan, Arena* arena);
bool CreateFullyConnectedQuantPlan(CmsisQuantPlan::FcPlan& plan,
                                   const int8_t* weights,
                                   const int32_t* bias,
                                   Arena* arena,
                                   void* weights_cache = nullptr);

void DestroyXnnpackOp(CmsisQuantPlan::XnnpackOpHoist& hoist);
// After xnn_finalize_weights_cache: reshape + workspace for ops created with a cache.
bool FinishConvAfterWeightsCache(CmsisQuantPlan::XnnpackOpHoist& hoist, Arena* arena);
bool FinishFullyConnectedAfterWeightsCache(CmsisQuantPlan::XnnpackOpHoist& hoist);

// Fused MobileNetV4 UIB subgraph: start_dw? → expand → middle_dw? → proj.
// When plan.has_residual, residual add is inside the subgraph (xnn_binary_add).
bool CreateUibSubgraph(CmsisQuantPlan::MobilenetV4UibPlan& plan,
                       const MobileNetV4Uib& uib,
                       void* weights_cache = nullptr,
                       void* workspace = nullptr);
bool FinishUibAfterWeightsCache(CmsisQuantPlan::MobilenetV4UibPlan& plan);
void DestroyUibSubgraph(CmsisQuantPlan::MobilenetV4UibPlan& plan);
bool InvokeUibSubgraph(CmsisQuantPlan::MobilenetV4UibPlan& plan,
                       const int8_t* input,
                       int8_t* output);

// Full-network qs8 subgraph (external in → all layers → external out).
bool CreateNetworkSubgraph(CmsisQuantPlan::Runtime& runtime,
                           CNNNetwork& network,
                           void* weights_cache);
void FinishNetworkAfterWeightsCache(CmsisQuantPlan::Runtime& runtime);
void DestroyNetworkSubgraph(CmsisQuantPlan::Runtime& runtime);
bool InvokeNetworkSubgraph(CmsisQuantPlan::Runtime& runtime,
                           const int8_t* input,
                           int8_t* output);

// Persistent qs8 subgraph for MLP (Dense chain). Input is [1, in_features].
struct MlpRuntime
{
    void* xnn_weights_cache = nullptr;
    void* xnn_workspace = nullptr;
    void* xnn_network_runtime = nullptr;
    bool ready = false;
    uint32_t ext_in = 0;
    uint32_t ext_out = 1;
    uint32_t in_features = 0;
    uint32_t out_features = 0;
    float** bias_scales = nullptr;
    uint32_t bias_scales_count = 0;
    // Skip xnn_setup_runtime_v2 when I/O addresses are unchanged (common in benches).
    const int8_t* bound_input = nullptr;
    int8_t* bound_output = nullptr;
};

bool BuildMlpNetworkSubgraph(MLPNetwork& network,
                             Arena& arena,
                             uint32_t in_features,
                             MlpRuntime*& out_runtime);
void DestroyMlpRuntime(MlpRuntime& runtime);
bool InvokeMlpNetworkSubgraph(MlpRuntime& runtime, const int8_t* input, int8_t* output);
#endif

bool TryConv2dNhwcQuantPlan(const CmsisQuantPlan::Conv2DPlan& plan,
                            const int8_t* input,
                            const int8_t* weights,
                            const int32_t* bias,
                            int8_t* output);

bool TryDepthwiseConv2dNhwcQuantPlan(const CmsisQuantPlan::DepthwiseConv2DPlan& plan,
                                     const int8_t* input,
                                     const int8_t* weights_chw,
                                     const int32_t* bias,
                                     int8_t* output);

bool TryMaxPool2dNhwcQuantPlan(const CmsisQuantPlan::Pool2DPlan& plan,
                               const int8_t* input,
                               int8_t* output);

bool TryFullyConnectedQuantPlan(const CmsisQuantPlan::FcPlan& plan,
                                const int8_t* input,
                                const int8_t* weights,
                                const int32_t* bias,
                                int8_t* output_int8);

bool TryConv2dNhwcQuant(const int8_t* input,
                        uint32_t in_h,
                        uint32_t in_w,
                        uint32_t in_c,
                        const int8_t* weights,
                        const int32_t* bias,
                        int kernel_size,
                        int stride,
                        int pad_h,
                        int pad_w,
                        int pad_h_end,
                        int pad_w_end,
                        int out_channels,
                        const NkFormat::MlpLayerQuantDesc& quant,
                        bool apply_relu,
                        int8_t* output);

bool TryDepthwiseConv2dNhwcQuant(const int8_t* input,
                                 uint32_t in_h,
                                 uint32_t in_w,
                                 uint32_t channels,
                                 const int8_t* weights_chw,
                                 const int32_t* bias,
                                 int kernel_h,
                                 int kernel_w,
                                 int stride,
                                 int pad_h,
                                 int pad_w,
                                 int pad_h_end,
                                 int pad_w_end,
                                 const NkFormat::MlpLayerQuantDesc& quant,
                                 bool apply_relu,
                                 int8_t* output);

bool TryMaxPool2dNhwcQuant(const int8_t* input,
                           uint32_t in_h,
                           uint32_t in_w,
                           uint32_t in_c,
                           int pool_h,
                           int pool_w,
                           int stride,
                           int pad_h,
                           int pad_w,
                           int pad_h_end,
                           int pad_w_end,
                           int8_t* output);

bool TryFullyConnectedQuant(const int8_t* input,
                            uint32_t batch,
                            uint32_t in_features,
                            const int8_t* weights,
                            const int32_t* bias,
                            uint32_t out_features,
                            const NkFormat::MlpLayerQuantDesc& quant,
                            bool apply_relu,
                            int8_t* output_int8);
}  // namespace XnnpackQuant
