#pragma once

#include "ops_resolver.hpp"

bool NkPlanLayerNorm2d(CnnBlock& block, NkCnnSpatialPlan& plan);
bool NkPrepareLayerNorm2d(const NkCnnOpContext& ctx);
void NkEvalLayerNorm2d(CnnBlock& block, const Tensor& input, Tensor& output);

struct NkLayerNorm2dOpDescriptor
{
    static constexpr NkLayerOpRegistration kRegistration{
        static_cast<uint8_t>(NkOpCode::LayerNorm2d),
        "layernorm2d",
        NkPlanLayerNorm2d,
        NkPrepareLayerNorm2d,
        NkEvalLayerNorm2d,
    };
};
