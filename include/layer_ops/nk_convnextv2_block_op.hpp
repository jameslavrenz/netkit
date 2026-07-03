#pragma once

#include "ops_resolver.hpp"

bool NkPlanConvNeXtV2Block(CnnBlock& block, NkCnnSpatialPlan& plan);
bool NkPrepareConvNeXtV2Block(const NkCnnOpContext& ctx);
void NkEvalConvNeXtV2Block(CnnBlock& block, const Tensor& input, Tensor& output);

struct NkConvNeXtV2BlockOpDescriptor
{
    static constexpr NkLayerOpRegistration kRegistration = {
        static_cast<uint8_t>(NkOpCode::ConvNeXtV2Block),
        "convnextv2_block",
        NkPlanConvNeXtV2Block,
        NkPrepareConvNeXtV2Block,
        NkEvalConvNeXtV2Block,
    };
};
