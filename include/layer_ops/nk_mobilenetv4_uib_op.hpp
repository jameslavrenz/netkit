#pragma once

#include "ops_resolver.hpp"

bool NkPlanMobilenetV4Uib(CnnBlock& block, NkCnnSpatialPlan& plan);
bool NkPrepareMobilenetV4Uib(const NkCnnOpContext& ctx);
void NkEvalMobilenetV4Uib(CnnBlock& block, const Tensor& input, Tensor& output);

struct NkMobilenetV4UibOpDescriptor
{
    static constexpr NkLayerOpRegistration kRegistration = {
        static_cast<uint8_t>(NkOpCode::MobilenetV4Uib),
        "mobilenetv4_uib",
        NkPlanMobilenetV4Uib,
        NkPrepareMobilenetV4Uib,
        NkEvalMobilenetV4Uib,
    };
};
