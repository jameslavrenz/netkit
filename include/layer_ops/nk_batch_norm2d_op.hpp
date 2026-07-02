#pragma once

#include "ops_resolver.hpp"

bool NkPlanBatchNorm2d(CnnBlock& block, NkCnnSpatialPlan& plan);
bool NkPrepareBatchNorm2d(const NkCnnOpContext& ctx);
void NkEvalBatchNorm2d(CnnBlock& block, const Tensor& input, Tensor& output);

struct NkBatchNorm2dOpDescriptor
{
    static constexpr NkLayerOpRegistration kRegistration = {
        static_cast<uint8_t>(NkOpCode::BatchNorm2d),
        "batch_norm2d",
        NkPlanBatchNorm2d,
        NkPrepareBatchNorm2d,
        NkEvalBatchNorm2d,
    };
};
