#include "cmsis_quant_plan.hpp"

#include "arena.hpp"
#include "cnn.hpp"
#include "cmsis_buffer_size.hpp"
#include "cmsis_nn_quant.hpp"
#include "kernel_workspace.hpp"
#include "netkit_config.h"
#include "nk_op_detail.hpp"
#include "quant_ops.hpp"
#include "quant_output.hpp"
#include "quant_trace.hpp"
#include "tensor_factory.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace
{
    using namespace TensorFactory;

    void QuantizeMultiplier(double double_multiplier, int32_t* multiplier, int32_t* shift)
    {
        if (double_multiplier <= 0.0)
        {
            *multiplier = 0;
            *shift = 0;
            return;
        }

        int shift_val = 0;
        double q = std::frexp(double_multiplier, &shift_val);
        int64_t q_fixed = static_cast<int64_t>(std::llround(q * (1LL << 31)));
        if (q_fixed == (1LL << 31))
        {
            q_fixed /= 2;
            ++shift_val;
        }

        if (shift_val < -31)
        {
            shift_val = 0;
            q_fixed = 0;
        }
        if (shift_val > 30)
        {
            shift_val = 30;
            q_fixed = (1LL << 31) - 1;
        }

        *multiplier = static_cast<int32_t>(q_fixed);
        *shift = shift_val;
    }

    uint32_t PlanOutputShape(CnnBlock& block, uint32_t& h, uint32_t& w, uint32_t& c)
    {
        switch (block.type)
        {
            case CnnBlockType::Conv2D:
            {
                const Conv2D& conv = block.conv.conv;
                h = nk_op_detail::CalcOutputDimAsymmetric(
                    h, conv.kernel_size, conv.stride, conv.pad_h, conv.pad_h_end);
                w = nk_op_detail::CalcOutputDimAsymmetric(
                    w, conv.kernel_size, conv.stride, conv.pad_w, conv.pad_w_end);
                c = static_cast<uint32_t>(conv.out_channels);
                break;
            }
            case CnnBlockType::MaxPool2D:
            {
                const MaxPool2DLayer& pool = block.pool;
                h = nk_op_detail::CalcOutputDimAsymmetric(
                    h, pool.pool_h, pool.stride, pool.pad_h, pool.pad_h_end);
                w = nk_op_detail::CalcOutputDimAsymmetric(
                    w, pool.pool_w, pool.stride, pool.pad_w, pool.pad_w_end);
                break;
            }
            case CnnBlockType::Flatten:
            {
                const uint32_t features = h * w * c;
                h = 1;
                w = features;
                c = 1;
                break;
            }
            case CnnBlockType::Dense:
            {
                const uint32_t units = block.dense.weights.shape[0];
                h = 1;
                w = units;
                c = 1;
                break;
            }
            default:
                return 0;
        }

        return h * w * c;
    }

    bool BuildConvPlan(CmsisQuantPlan::Conv2DPlan& plan,
                       const Conv2D& conv,
                       const NkFormat::LayerQuantDesc& quant,
                       bool apply_relu,
                       uint32_t in_h,
                       uint32_t in_w,
                       uint32_t in_c,
                       Arena& arena)
    {
        if (conv.pad_h != conv.pad_h_end || conv.pad_w != conv.pad_w_end)
            return false;
        if (conv.out_channels <= 0 ||
            static_cast<uint32_t>(conv.out_channels) > CmsisQuantPlan::kMaxPerChannel)
            return false;
        if (quant.output_scale <= 0.0f)
            return false;

        const uint32_t out_h = nk_op_detail::CalcOutputDimAsymmetric(
            in_h, conv.kernel_size, conv.stride, conv.pad_h, conv.pad_h_end);
        const uint32_t out_w = nk_op_detail::CalcOutputDimAsymmetric(
            in_w, conv.kernel_size, conv.stride, conv.pad_w, conv.pad_w_end);

        plan.input_offset = -quant.input_zero_point;
        plan.output_offset = -quant.output_zero_point;
        plan.stride = conv.stride;
        plan.pad_h = conv.pad_h;
        plan.pad_w = conv.pad_w;
        plan.apply_relu = apply_relu;
        plan.in_h = static_cast<int32_t>(in_h);
        plan.in_w = static_cast<int32_t>(in_w);
        plan.in_c = static_cast<int32_t>(in_c);
        plan.out_h = static_cast<int32_t>(out_h);
        plan.out_w = static_cast<int32_t>(out_w);
        plan.out_c = conv.out_channels;
        plan.kernel_size = conv.kernel_size;

        const int32_t channels = conv.out_channels;
        plan.multipliers = static_cast<int32_t*>(arena.alloc(
            static_cast<std::size_t>(channels) * sizeof(int32_t) * 2, alignof(int32_t)));
        if (!plan.multipliers)
            return false;
        plan.shifts = plan.multipliers + channels;

        const double effective = static_cast<double>(quant.input_scale) *
                                 static_cast<double>(quant.weight_scale) /
                                 static_cast<double>(quant.output_scale);
        int32_t multiplier = 0;
        int32_t shift = 0;
        QuantizeMultiplier(effective, &multiplier, &shift);
        for (int32_t oc = 0; oc < channels; ++oc)
        {
            plan.multipliers[oc] = multiplier;
            plan.shifts[oc] = shift;
        }

#if defined(NETKIT_USE_CMSIS_NN) && NETKIT_USE_CMSIS_NN && NETKIT_CMSIS_NN_ALLOWED
        plan.workspace_bytes = static_cast<int32_t>(CmsisConv2dS8WorkspaceBytes(
            static_cast<uint32_t>(in_h),
            static_cast<uint32_t>(in_w),
            static_cast<uint32_t>(conv.kernel_size),
            conv.stride,
            conv.pad_h,
            conv.pad_w,
            static_cast<uint32_t>(in_c),
            static_cast<uint32_t>(conv.out_channels)));
#else
        plan.workspace_bytes = 0;
#endif
        plan.ready = true;
        CmsisNnQuant::FinalizeConv2DPlan(plan);
        return true;
    }

    bool BuildPoolPlan(CmsisQuantPlan::Pool2DPlan& plan,
                       const MaxPool2DLayer& pool,
                       uint32_t in_h,
                       uint32_t in_w,
                       uint32_t in_c)
    {
        if (pool.pad_h != pool.pad_h_end || pool.pad_w != pool.pad_w_end)
            return false;

        plan.stride = pool.stride;
        plan.pad_h = pool.pad_h;
        plan.pad_w = pool.pad_w;
        plan.pool_h = pool.pool_h;
        plan.pool_w = pool.pool_w;
        plan.in_h = static_cast<int32_t>(in_h);
        plan.in_w = static_cast<int32_t>(in_w);
        plan.in_c = static_cast<int32_t>(in_c);
        plan.out_h = static_cast<int32_t>(nk_op_detail::CalcOutputDimAsymmetric(
            in_h, pool.pool_h, pool.stride, pool.pad_h, pool.pad_h_end));
        plan.out_w = static_cast<int32_t>(nk_op_detail::CalcOutputDimAsymmetric(
            in_w, pool.pool_w, pool.stride, pool.pad_w, pool.pad_w_end));
        plan.ready = true;
        CmsisNnQuant::FinalizePool2DPlan(plan);
        return true;
    }

    bool BuildFcPlan(CmsisQuantPlan::FcPlan& plan,
                     const NkFormat::LayerQuantDesc& quant,
                     bool apply_relu,
                     uint32_t in_features,
                     uint32_t out_features)
    {
        if (in_features == 0 || out_features == 0 || quant.output_scale <= 0.0f)
            return false;

        plan.input_offset = -quant.input_zero_point;
        plan.filter_offset = -quant.weight_zero_point;
        plan.output_offset = -quant.output_zero_point;
        plan.apply_relu = apply_relu;
        plan.in_features = static_cast<int32_t>(in_features);
        plan.out_features = static_cast<int32_t>(out_features);

        const double effective = static_cast<double>(quant.input_scale) *
                               static_cast<double>(quant.weight_scale) /
                               static_cast<double>(quant.output_scale);
        QuantizeMultiplier(effective, &plan.multiplier, &plan.shift);

#if defined(NETKIT_USE_CMSIS_NN) && NETKIT_USE_CMSIS_NN && NETKIT_CMSIS_NN_ALLOWED
        plan.workspace_bytes = static_cast<int32_t>(
            CmsisFullyConnectedS8WorkspaceBytes(in_features, out_features));
#else
        plan.workspace_bytes = 0;
#endif
        plan.ready = true;
        return true;
    }

    bool BuildSoftmaxPlan(CmsisQuantPlan::SoftmaxPlan& plan, float logit_scale, int32_t row_size)
    {
        plan.params = QuantOps::ComputeSoftmaxS8Params(logit_scale);
        plan.row_size = row_size;
        plan.ready = true;
        return true;
    }
}

namespace CmsisQuantPlan
{

bool BuildRuntime(CNNNetwork& network, Arena& arena, uint32_t in_h, uint32_t in_w, uint32_t in_c)
{
    const uint32_t n = network.layer_count();
    if (n == 0)
        return false;

    Runtime* runtime = static_cast<Runtime*>(arena.alloc(sizeof(Runtime), alignof(Runtime)));
    if (!runtime)
        return false;
    std::memset(runtime, 0, sizeof(Runtime));

    runtime->layers = static_cast<LayerPlan*>(arena.alloc(sizeof(LayerPlan) * n, alignof(LayerPlan)));
    if (!runtime->layers)
        return false;
    std::memset(runtime->layers, 0, sizeof(LayerPlan) * n);
    runtime->num_layers = n;

    uint32_t h = in_h;
    uint32_t w = in_w;
    uint32_t c = in_c;
    uint32_t even_max = 0;
    uint32_t odd_max = 0;
    std::size_t workspace_bytes = 0;
    uint32_t logits_elements = 0;

    const CnnBlock& first = network.GetBlock(0);
    if (first.type == CnnBlockType::Conv2D && first.conv.quant.enabled)
    {
        runtime->input_scale = first.conv.quant.params.input_scale;
        runtime->input_zero_point = first.conv.quant.params.input_zero_point;
    }

    for (uint32_t i = 0; i < n; ++i)
    {
        LayerPlan& lp = runtime->layers[i];
        CnnBlock& block = network.GetBlock(i);
        const uint32_t in_h_layer = h;
        const uint32_t in_w_layer = w;
        const uint32_t in_c_layer = c;
        const uint32_t elements = PlanOutputShape(block, h, w, c);
        if (elements == 0)
            return false;

        lp.output_elements = elements;
        if (i % 2 == 0)
            even_max = std::max(even_max, elements);
        else
            odd_max = std::max(odd_max, elements);

        switch (block.type)
        {
            case CnnBlockType::Conv2D:
            {
                lp.kind = LayerKind::Conv2D;
                const bool apply_relu = block.conv.activation == ConvActivationType::ReLU;
                if (!BuildConvPlan(lp.conv,
                                   block.conv.conv,
                                   block.conv.quant.params,
                                   apply_relu,
                                   in_h_layer,
                                   in_w_layer,
                                   in_c_layer,
                                   arena))
                    return false;
                workspace_bytes = std::max(workspace_bytes,
                                           static_cast<std::size_t>(lp.conv.workspace_bytes));
                break;
            }
            case CnnBlockType::MaxPool2D:
            {
                lp.kind = LayerKind::MaxPool2D;
                if (!BuildPoolPlan(lp.pool, block.pool, in_h_layer, in_w_layer, in_c_layer))
                    return false;
                break;
            }
            case CnnBlockType::Flatten:
                lp.kind = LayerKind::FlattenView;
                break;
            case CnnBlockType::Dense:
            {
                const bool apply_relu = block.dense.activation == ActivationType::ReLU;
                const bool apply_softmax = block.dense.activation == ActivationType::Softmax;
                const uint32_t in_features = in_h_layer * in_w_layer * in_c_layer;
                const uint32_t out_features = block.dense.weights.shape[0];
                if (apply_softmax)
                {
                    lp.kind = LayerKind::DenseSoftmax;
                    logits_elements = out_features;
                    if (!BuildFcPlan(lp.fc,
                                     block.dense.quant.params,
                                     false,
                                     in_features,
                                     out_features))
                        return false;
                    {
                        const int8_t* weights =
                            static_cast<const int8_t*>(block.dense.weights.data);
                        const int32_t* bias = static_cast<const int32_t*>(block.dense.bias.data);
                        if (!CmsisNnQuant::FinalizeFcPlan(lp.fc, weights, bias, arena))
                            return false;
                    }
                    if (!BuildSoftmaxPlan(lp.softmax,
                                          block.dense.quant.params.output_scale > 0.0f
                                              ? block.dense.quant.params.output_scale
                                              : 1.0f,
                                          static_cast<int32_t>(out_features)))
                        return false;
                }
                else
                {
                    lp.kind = LayerKind::Dense;
                    if (!BuildFcPlan(lp.fc,
                                     block.dense.quant.params,
                                     apply_relu,
                                     in_features,
                                     out_features))
                        return false;
                    {
                        const int8_t* weights =
                            static_cast<const int8_t*>(block.dense.weights.data);
                        const int32_t* bias = static_cast<const int32_t*>(block.dense.bias.data);
                        if (!CmsisNnQuant::FinalizeFcPlan(lp.fc, weights, bias, arena))
                            return false;
                    }
                }
                workspace_bytes = std::max(workspace_bytes,
                                           static_cast<std::size_t>(lp.fc.workspace_bytes));
                break;
            }
            default:
                return false;
        }
    }

    runtime->input_quant_elements = in_h * in_w * in_c;
    runtime->staging_arena = &arena;

    runtime->act_a_bytes = even_max;
    runtime->act_b_bytes = odd_max;
    if (runtime->act_a_bytes > 0)
    {
        runtime->act_a = static_cast<int8_t*>(
            arena.alloc(static_cast<std::size_t>(runtime->act_a_bytes) * sizeof(int8_t),
                        alignof(int8_t)));
        if (!runtime->act_a)
            return false;
    }
    if (runtime->act_b_bytes > 0)
    {
        runtime->act_b = static_cast<int8_t*>(
            arena.alloc(static_cast<std::size_t>(runtime->act_b_bytes) * sizeof(int8_t),
                        alignof(int8_t)));
        if (!runtime->act_b)
            return false;
    }

    if (logits_elements > 0)
    {
        runtime->logits_elements = logits_elements;
        runtime->logits = static_cast<int8_t*>(
            arena.alloc(static_cast<std::size_t>(logits_elements) * sizeof(int8_t), alignof(int8_t)));
        if (!runtime->logits)
            return false;
    }

    if (workspace_bytes > 0)
    {
        runtime->workspace = static_cast<uint8_t*>(
            arena.alloc(workspace_bytes, alignof(std::max_align_t)));
        if (!runtime->workspace)
            return false;
        runtime->workspace_bytes = workspace_bytes;
    }

    network.SetQuantRuntime(runtime);
    QuantTrace::RecordKernelPlan(static_cast<uint32_t>(workspace_bytes),
                                 static_cast<uint32_t>(runtime->act_a_bytes + runtime->act_b_bytes));
    return true;
}

void DestroyRuntime(Runtime& runtime)
{
    runtime = {};
}

namespace
{
    bool ForwardLayers(Runtime& runtime,
                       CNNNetwork& network,
                       int8_t* current,
                       QuantOutputFormat output_format,
                       Tensor& output_cache,
                       int8_t* output_dest)
    {
        if (!runtime.layers || runtime.num_layers == 0 || !current)
            return false;

        KernelWorkspace workspace{runtime.workspace, runtime.workspace_bytes};
        KernelWorkspaceScope workspace_scope(&workspace);

        int8_t* slot_a = runtime.act_a;
        int8_t* slot_b = runtime.act_b;
        bool use_a = true;

        for (uint32_t i = 0; i < runtime.num_layers; ++i)
        {
            const LayerPlan& lp = runtime.layers[i];
            const bool is_last = i + 1 == runtime.num_layers;
            int8_t* out = nullptr;

            if (lp.kind == LayerKind::FlattenView)
            {
                continue;
            }

            if (use_a)
                out = slot_a;
            else
                out = slot_b;

            if (!out && lp.kind != LayerKind::FlattenView)
                return false;

            CnnBlock& block = network.GetBlock(i);
            switch (lp.kind)
            {
                case LayerKind::Conv2D:
                {
                    const Conv2D& conv = block.conv.conv;
                    if (!CmsisNnQuant::TryConv2dNhwcQuantPlan(
                            lp.conv, current, conv.weights_q, conv.bias_q, out))
                        return false;
                    break;
                }
                case LayerKind::MaxPool2D:
                {
                    if (!CmsisNnQuant::TryMaxPool2dNhwcQuantPlan(lp.pool, current, out))
                        return false;
                    break;
                }
                case LayerKind::Dense:
                {
                    const int8_t* weights = static_cast<const int8_t*>(block.dense.weights.data);
                    const int32_t* bias = static_cast<const int32_t*>(block.dense.bias.data);
                    int8_t* dense_out = (is_last && output_dest != nullptr) ? output_dest : out;
                    if (!CmsisNnQuant::TryFullyConnectedQuantPlan(lp.fc, current, weights, bias, dense_out))
                        return false;
                    if (dense_out == output_dest)
                        out = output_dest;
                    break;
                }
                case LayerKind::DenseSoftmax:
                {
                    const int8_t* weights = static_cast<const int8_t*>(block.dense.weights.data);
                    const int32_t* bias = static_cast<const int32_t*>(block.dense.bias.data);
                    if (!CmsisNnQuant::TryFullyConnectedQuantPlan(
                            lp.fc, current, weights, bias, runtime.logits))
                        return false;
                    int8_t* softmax_out = (output_dest != nullptr) ? output_dest : out;
                    if (!CmsisNnQuant::TrySoftmaxS8Plan(lp.softmax, runtime.logits, softmax_out))
                        return false;
                    out = softmax_out;
                    (void)output_format;
                    break;
                }
                default:
                    break;
            }

            if (lp.kind != LayerKind::FlattenView)
            {
                current = out;
                if (out != output_dest)
                    use_a = !use_a;
            }
        }

        if (output_dest != nullptr)
            return true;

        const LayerPlan& last = runtime.layers[runtime.num_layers - 1];
        output_cache = View2DInt8(current, 1, last.output_elements);
        return true;
    }
    bool EnsureInputQuantBuffer(Runtime& runtime)
    {
        if (runtime.input_quant_elements == 0)
            return true;
        if (runtime.input_quant != nullptr)
            return true;
        if (runtime.staging_arena == nullptr)
            return false;

        runtime.input_quant = static_cast<int8_t*>(
            runtime.staging_arena->alloc(runtime.input_quant_elements * sizeof(int8_t),
                                         alignof(int8_t)));
        return runtime.input_quant != nullptr;
    }
}  // namespace

bool Forward(Runtime& runtime,
             CNNNetwork& network,
             const Tensor& input,
             QuantOutputFormat output_format,
             Tensor& output_cache)
{
    if (input.type == DataType::Int8)
    {
        return ForwardInt8(runtime,
                         network,
                         static_cast<const int8_t*>(input.data),
                         input.num_elements,
                         output_format,
                         output_cache);
    }

    const float* input_f = static_cast<const float*>(input.data);
    if (!input_f || !EnsureInputQuantBuffer(runtime))
        return false;

    for (uint32_t i = 0; i < runtime.input_quant_elements; ++i)
        runtime.input_quant[i] =
            QuantOps::QuantizeFloat(input_f[i], runtime.input_scale, runtime.input_zero_point);

    return ForwardLayers(runtime, network, runtime.input_quant, output_format, output_cache, nullptr);
}

bool ForwardInt8(Runtime& runtime,
                 CNNNetwork& network,
                 const int8_t* input,
                 uint32_t input_elements,
                 QuantOutputFormat output_format,
                 Tensor& output_cache)
{
    if (!input || input_elements != runtime.input_quant_elements)
        return false;

    return ForwardLayers(runtime,
                       network,
                       const_cast<int8_t*>(input),
                       output_format,
                       output_cache,
                       nullptr);
}

bool ForwardInt8ToBuffer(Runtime& runtime,
                         CNNNetwork& network,
                         const int8_t* input,
                         int8_t* output,
                         uint32_t output_elements)
{
    if (!input || !output || runtime.num_layers == 0 || !runtime.layers)
        return false;
    if (runtime.input_quant_elements == 0 ||
        output_elements != runtime.layers[runtime.num_layers - 1].output_elements)
        return false;

    Tensor unused_cache{};
    return ForwardLayers(runtime,
                       network,
                       const_cast<int8_t*>(input),
                       QuantOutputFormat::Int8,
                       unused_cache,
                       output);
}

}  // namespace CmsisQuantPlan
