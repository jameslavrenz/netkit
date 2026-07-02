#pragma once

#include "ops_resolver.hpp"

bool NkPlanFlatten(CnnBlock& block, NkCnnSpatialPlan& plan);
bool NkPrepareFlatten(const NkCnnOpContext& ctx);
void NkEvalFlatten(CnnBlock& block, const Tensor& input, Tensor& output);

struct NkFlattenOpDescriptor
{
    static constexpr NkLayerOpRegistration kRegistration = {
        static_cast<uint8_t>(NkOpCode::Flatten),
        "flatten",
        NkPlanFlatten,
        NkPrepareFlatten,
        NkEvalFlatten,
    };
};
