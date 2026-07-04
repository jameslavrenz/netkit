#include "layer_ops/nk_mobilenetv4_uib_op.hpp"

#include "cmsis_buffer_size.hpp"
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

    uint32_t cur_h = plan.h;
    uint32_t cur_w = plan.w;
    uint32_t cur_c = static_cast<uint32_t>(uib.in_channels);

    if (uib.start_dw_kernel > 0)
    {
        const int pad = (uib.start_dw_kernel - 1) / 2;
        CmsisBumpDepthwiseConv2dWorkspace(cur_h,
                                          cur_w,
                                          uib.start_dw_kernel,
                                          uib.start_dw_kernel,
                                          static_cast<int>(uib.start_dw_stride()),
                                          pad,
                                          pad,
                                          uib.in_channels);
        cur_h = CalcOutputDim(cur_h, uib.start_dw_kernel, static_cast<int>(uib.start_dw_stride()), pad);
        cur_w = CalcOutputDim(cur_w, uib.start_dw_kernel, static_cast<int>(uib.start_dw_stride()), pad);
    }

    CmsisBumpConv2dWorkspace(cur_h,
                             cur_w,
                             1,
                             1,
                             0,
                             0,
                             static_cast<int>(cur_c),
                             static_cast<int>(expand_c));

    if (uib.middle_dw_kernel > 0)
    {
        const int pad = (uib.middle_dw_kernel - 1) / 2;
        CmsisBumpDepthwiseConv2dWorkspace(cur_h,
                                          cur_w,
                                          uib.middle_dw_kernel,
                                          uib.middle_dw_kernel,
                                          static_cast<int>(uib.middle_dw_stride()),
                                          pad,
                                          pad,
                                          static_cast<int>(expand_c));
        cur_h = CalcOutputDim(cur_h, uib.middle_dw_kernel, static_cast<int>(uib.middle_dw_stride()), pad);
        cur_w = CalcOutputDim(cur_w, uib.middle_dw_kernel, static_cast<int>(uib.middle_dw_stride()), pad);
    }

    CmsisBumpConv2dWorkspace(cur_h,
                             cur_w,
                             1,
                             1,
                             0,
                             0,
                             static_cast<int>(expand_c),
                             uib.out_channels);

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
