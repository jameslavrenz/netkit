#pragma once

#include "ops_resolver.hpp"

bool NkPlanDense(CnnBlock& block, NkCnnSpatialPlan& plan);
bool NkPrepareDense(const NkCnnOpContext& ctx);
void NkEvalDense(CnnBlock& block, const Tensor& input, Tensor& output);

struct NkDenseOpDescriptor
{
    static constexpr NkLayerOpRegistration kRegistration = {
        static_cast<uint8_t>(NkOpCode::Dense),
        "dense",
        NkPlanDense,
        NkPrepareDense,
        NkEvalDense,
    };
};
