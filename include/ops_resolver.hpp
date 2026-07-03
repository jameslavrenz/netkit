#pragma once

#include "nk_format.hpp"
#include <cstdint>

struct CnnBlock;
struct Tensor;
enum class CnnBlockType;

/*
 * MCU-safe layer op registry (TFLite MicroMutableOpResolver style).
 *
 * - No std::vector, heap, virtual functions, or RTTI
 * - Function pointers + fixed static storage
 * - C++26 constinit compile-time registration tables (no dynamic static init)
 * - NkOpList<Ops...> typelist for trimmed firmware images
 */

enum class NkOpCode : uint8_t
{
    Dense = static_cast<uint8_t>(NkFormat::LayerKind::Dense),
    Conv2D = static_cast<uint8_t>(NkFormat::LayerKind::Conv2D),
    MaxPool2D = static_cast<uint8_t>(NkFormat::LayerKind::MaxPool2D),
    Flatten = static_cast<uint8_t>(NkFormat::LayerKind::Flatten),
    AvgPool2D = static_cast<uint8_t>(NkFormat::LayerKind::AvgPool2D),
    BatchNorm2d = static_cast<uint8_t>(NkFormat::LayerKind::BatchNorm2d),
    DepthwiseConv2D = static_cast<uint8_t>(NkFormat::LayerKind::DepthwiseConv2D),
    ConvNeXtV2Block = static_cast<uint8_t>(NkFormat::LayerKind::ConvNeXtV2Block),
    MobilenetV4Uib = static_cast<uint8_t>(NkFormat::LayerKind::MobilenetV4Uib),
    ResNetBasicBlock = static_cast<uint8_t>(NkFormat::LayerKind::ResNetBasicBlock),
    LayerNorm2d = static_cast<uint8_t>(NkFormat::LayerKind::LayerNorm2d),
};

constexpr uint8_t kNkOpCodeCount = 11;

struct NkCnnSpatialPlan
{
    uint32_t h = 0;
    uint32_t w = 0;
    uint32_t channels = 0;
    uint32_t* max_activation_elements = nullptr;
};

struct NkCnnOpContext
{
    CnnBlock& block;
    const Tensor& input;
    Tensor& output;
    float* write_buffer = nullptr;
    uint32_t max_activation_elements = 0;
};

using NkCnnPlanActivationFn = bool (*)(CnnBlock& block, NkCnnSpatialPlan& plan);
using NkCnnPrepareOutputFn = bool (*)(const NkCnnOpContext& ctx);
using NkCnnEvalFn = void (*)(CnnBlock& block, const Tensor& input, Tensor& output);

struct NkLayerOpRegistration
{
    uint8_t opcode = 0;
    const char* name = nullptr;
    NkCnnPlanActivationFn plan_activation = nullptr;
    NkCnnPrepareOutputFn prepare_output = nullptr;
    NkCnnEvalFn eval = nullptr;
};

struct NkOpsResolver
{
    const NkLayerOpRegistration* registrations = nullptr;
    uint8_t num_registrations = 0;

    [[nodiscard]] constexpr const NkLayerOpRegistration* Find(uint8_t opcode) const noexcept
    {
        if (!registrations)
            return nullptr;

        for (uint8_t i = 0; i < num_registrations; ++i)
        {
            if (registrations[i].opcode == opcode)
                return &registrations[i];
        }

        return nullptr;
    }
};

/*
 * Compile-time op typelist — constexpr registration array + constinit view.
 * View() returns a reference to process-lifetime static storage (MCU-safe).
 */
template<typename... OpDescriptors>
struct NkOpList
{
    static_assert(sizeof...(OpDescriptors) > 0, "NkOpList requires at least one op");
    static_assert(sizeof...(OpDescriptors) <= NkFormat::kMaxLayers, "NkOpList exceeds kMaxLayers");

    static constexpr uint8_t kMaxOps = static_cast<uint8_t>(sizeof...(OpDescriptors));

    static inline constinit NkLayerOpRegistration kRegistrations[kMaxOps] = {
        OpDescriptors::kRegistration...,
    };

    static inline constinit NkOpsResolver kView{kRegistrations, kMaxOps};

    [[nodiscard]] static const NkOpsResolver& View() noexcept { return kView; }
};

NkOpCode ToOpCode(CnnBlockType block_type);

const NkOpsResolver& GetDefaultOpsResolver();
