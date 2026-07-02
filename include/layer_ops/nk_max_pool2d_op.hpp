#pragma once

#include "ops_resolver.hpp"

bool NkPlanMaxPool2D(CnnBlock& block, NkCnnSpatialPlan& plan);
bool NkPrepareMaxPool2D(const NkCnnOpContext& ctx);
void NkEvalMaxPool2D(CnnBlock& block, const Tensor& input, Tensor& output);

struct NkMaxPool2DOpDescriptor
{
    static constexpr NkLayerOpRegistration kRegistration = {
        static_cast<uint8_t>(NkOpCode::MaxPool2D),
        "max_pool2d",
        NkPlanMaxPool2D,
        NkPrepareMaxPool2D,
        NkEvalMaxPool2D,
    };
};
