#pragma once

#include "ops_resolver.hpp"

bool NkPlanConv2D(CnnBlock& block, NkCnnSpatialPlan& plan);
bool NkPrepareConv2D(const NkCnnOpContext& ctx);
void NkEvalConv2D(CnnBlock& block, const Tensor& input, Tensor& output);

struct NkConv2DOpDescriptor
{
    static constexpr NkLayerOpRegistration kRegistration = {
        static_cast<uint8_t>(NkOpCode::Conv2D),
        "conv2d",
        NkPlanConv2D,
        NkPrepareConv2D,
        NkEvalConv2D,
    };
};
