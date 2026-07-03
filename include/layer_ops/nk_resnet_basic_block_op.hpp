#pragma once

#include "ops_resolver.hpp"

bool NkPlanResNetBasicBlock(CnnBlock& block, NkCnnSpatialPlan& plan);
bool NkPrepareResNetBasicBlock(const NkCnnOpContext& ctx);
void NkEvalResNetBasicBlock(CnnBlock& block, const Tensor& input, Tensor& output);

struct NkResNetBasicBlockOpDescriptor
{
    static constexpr NkLayerOpRegistration kRegistration = {
        static_cast<uint8_t>(NkOpCode::ResNetBasicBlock),
        "resnet_basic_block",
        NkPlanResNetBasicBlock,
        NkPrepareResNetBasicBlock,
        NkEvalResNetBasicBlock,
    };
};
