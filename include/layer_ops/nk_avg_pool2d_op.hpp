#pragma once

#include "ops_resolver.hpp"

bool NkPlanAvgPool2D(CnnBlock& block, NkCnnSpatialPlan& plan);
bool NkPrepareAvgPool2D(const NkCnnOpContext& ctx);
void NkEvalAvgPool2D(CnnBlock& block, const Tensor& input, Tensor& output);

struct NkAvgPool2DOpDescriptor
{
    static constexpr NkLayerOpRegistration kRegistration = {
        static_cast<uint8_t>(NkOpCode::AvgPool2D),
        "avg_pool2d",
        NkPlanAvgPool2D,
        NkPrepareAvgPool2D,
        NkEvalAvgPool2D,
    };
};
