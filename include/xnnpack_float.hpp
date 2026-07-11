#pragma once

#include "netkit_config.h"

#include <cstdint>

struct Arena;
class CNNNetwork;
class MLPNetwork;
struct Tensor;

// Float32 XNNPACK full-network subgraph (TF Lite XNNPACK delegate parity).
// Mirrors the int8 CreateNetworkSubgraph path: one runtime, shared weights
// cache + workspace, XNN_FLAG_DONT_SPIN_WORKERS, single invoke per forward.
namespace XnnpackFloat
{
#if defined(NETKIT_USE_XNNPACK) && NETKIT_USE_XNNPACK && NETKIT_XNNPACK_ALLOWED
constexpr bool kEnabled = true;
#else
constexpr bool kEnabled = false;
#endif

struct Runtime
{
    void* xnn_weights_cache = nullptr;  // xnn_weights_cache_t
    void* xnn_workspace = nullptr;      // xnn_workspace_t
    void* xnn_network_runtime = nullptr;  // xnn_runtime_t
    bool xnn_network_ready = false;
    uint32_t xnn_net_ext_in = 0;
    uint32_t xnn_net_ext_out = 1;
    uint32_t in_h = 0;
    uint32_t in_w = 0;
    uint32_t in_c = 0;
    uint32_t out_elements = 0;
    // Skip xnn_setup_runtime_v2 when I/O addresses are unchanged.
    const float* bound_input = nullptr;
    float* bound_output = nullptr;
    // Owned depthwise weight repacks [1,Kh,Kw,C] for subgraph (GHW → HWC).
    float** dw_hwc_owned = nullptr;
    uint32_t dw_hwc_count = 0;
};

void DestroyRuntime(Runtime& runtime);

#if defined(NETKIT_USE_XNNPACK) && NETKIT_USE_XNNPACK && NETKIT_XNNPACK_ALLOWED
// Build persistent f32 subgraph for the loaded CNN (ImageNet MobileNetV4, MNIST CNN, etc.).
// Returns false on unsupported graphs; caller keeps the layer-loop fallback.
bool BuildNetworkRuntime(CNNNetwork& network,
                         Arena& arena,
                         uint32_t in_h,
                         uint32_t in_w,
                         uint32_t in_c,
                         Runtime*& out_runtime);

// Persistent f32 subgraph for MLP (Dense chain). Input is [1, in_features].
bool BuildMlpRuntime(MLPNetwork& network,
                     Arena& arena,
                     uint32_t in_features,
                     Runtime*& out_runtime);

bool InvokeNetwork(Runtime& runtime, const float* input, float* output);
#endif
}  // namespace XnnpackFloat
