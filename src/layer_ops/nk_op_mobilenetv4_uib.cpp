#include "layer_ops/nk_mobilenetv4_uib_op.hpp"

#include "cnn.hpp"
#include "mobilenetv4_uib.hpp"
#include "nk_op_detail.hpp"
#include "tensor_factory.hpp"
#include <array>

using namespace TensorFactory;
using namespace nk_op_detail;

bool NkPlanMobilenetV4Uib(CnnBlock& block, NkCnnSpatialPlan& plan)
{
    const MobileNetV4Uib& uib = block.mobilenetv4_uib.block;
    const uint32_t expand_c = uib.expanded_channels();
    uint32_t out_h = plan.h;
    uint32_t out_w = plan.w;
    uib.output_spatial(plan.h, plan.w, out_h, out_w);

    BumpMaxActivation(plan, plan.h * plan.w * static_cast<uint32_t>(uib.in_channels));
    BumpMaxActivation(plan, plan.h * plan.w * expand_c);
    BumpMaxActivation(plan, out_h * out_w * static_cast<uint32_t>(uib.out_channels));

    plan.h = out_h;
    plan.w = out_w;
    plan.channels = static_cast<uint32_t>(uib.out_channels);
    return true;
}

bool NkPrepareMobilenetV4Uib(const NkCnnOpContext& ctx)
{
    const MobileNetV4Uib& uib = ctx.block.mobilenetv4_uib.block;
    uint32_t out_h = ctx.input.shape[0];
    uint32_t out_w = ctx.input.shape[1];
    uib.output_spatial(ctx.input.shape[0], ctx.input.shape[1], out_h, out_w);

    const std::array<uint32_t, 3> shape = {out_h, out_w, static_cast<uint32_t>(uib.out_channels)};
    ctx.output = ViewND(ctx.write_buffer, 3, shape);
    return ctx.output.data != nullptr && ctx.output.num_elements <= ctx.max_activation_elements;
}

void NkEvalMobilenetV4Uib(CnnBlock& block, const Tensor& input, Tensor& output)
{
    block.mobilenetv4_uib.forward(input, output);
}
