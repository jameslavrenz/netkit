#include "layer_ops/nk_yolox_decoupled_head_op.hpp"

#include "cmsis_buffer_size.hpp"
#include "cnn.hpp"
#include "nk_op_detail.hpp"
#include "tensor_factory.hpp"
#include "yolox_decoupled_head.hpp"
#include <array>

using namespace TensorFactory;
using namespace nk_op_detail;

bool NkPlanYoloxDecoupledHead(CnnBlock& block, NkCnnSpatialPlan& plan)
{
    const YoloxDecoupledHead& head = block.yolox_decoupled_head.block;
    const uint32_t hidden = static_cast<uint32_t>(head.hidden_dim);
    const uint32_t out_c = static_cast<uint32_t>(head.output_channels());

    CmsisBumpConv2dWorkspace(plan.h, plan.w, 1, 1, 0, 0, head.in_channels, head.hidden_dim);
    for (int i = 0; i < head.num_convs; ++i)
    {
        CmsisBumpConv2dWorkspace(plan.h, plan.w, 3, 1, 1, 1, head.hidden_dim, head.hidden_dim);
    }
    CmsisBumpConv2dWorkspace(plan.h, plan.w, 1, 1, 0, 0, head.hidden_dim, head.num_classes);
    for (int i = 0; i < head.num_convs; ++i)
    {
        CmsisBumpConv2dWorkspace(plan.h, plan.w, 3, 1, 1, 1, head.hidden_dim, head.hidden_dim);
    }
    CmsisBumpConv2dWorkspace(plan.h, plan.w, 1, 1, 0, 0, head.hidden_dim, 4);
    CmsisBumpConv2dWorkspace(plan.h, plan.w, 1, 1, 0, 0, head.hidden_dim, 1);

    BumpMaxActivation(plan, plan.h * plan.w * static_cast<uint32_t>(head.in_channels));
    BumpMaxActivation(plan, plan.h * plan.w * hidden);
    BumpMaxActivation(plan, plan.h * plan.w * out_c);

    plan.channels = out_c;
    return true;
}

bool NkPrepareYoloxDecoupledHead(const NkCnnOpContext& ctx)
{
    const YoloxDecoupledHead& head = ctx.block.yolox_decoupled_head.block;
    const std::array<uint32_t, 3> shape = {ctx.input.shape[0],
                                           ctx.input.shape[1],
                                           static_cast<uint32_t>(head.output_channels())};
    ctx.output = ViewND(ctx.write_buffer, 3, shape);
    return ctx.output.data != nullptr && ctx.output.num_elements <= ctx.max_activation_elements;
}

void NkEvalYoloxDecoupledHead(CnnBlock& block, const Tensor& input, Tensor& output)
{
    block.yolox_decoupled_head.forward(input, output);
}
