#include "quant_ops.hpp"

#include "cmsis_nn_quant.hpp"
#include "netkit_util.hpp"
#include "conv_im2col_policy.hpp"
#include "im2col_quant.hpp"
#include "kernel_workspace.hpp"
#include "netkit_config.h"
#include "nk_op_detail.hpp"
#include "quant_integer.hpp"
#include "quant_plan_types.hpp"
#include "quant_trace.hpp"
#include "xnnpack_quant.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>

namespace QuantOps
{
namespace
{
#if !NETKIT_MCU_CMSIS_ONLY
#if !defined(NETKIT_CLASS_MCU)
    // TF Lite Prepare-style: fill per-channel (or broadcast per-tensor) multipliers once.
    bool FillEffectiveMultipliers(const NkFormat::MlpLayerQuantDesc& quant,
                                  uint32_t channels,
                                  int32_t* multipliers,
                                  int32_t* shifts)
    {
        if (!multipliers || !shifts || channels == 0)
            return false;

        const bool per_channel = quant.weight_channel_scales != nullptr &&
                                 quant.num_weight_channel_scales == channels;
        if (!per_channel)
        {
            int32_t multiplier = 0;
            int32_t shift = 0;
            if (!QuantInteger::EffectiveScaleMultiplier(quant.input_scale,
                                                        quant.weight_scale,
                                                        quant.output_scale,
                                                        &multiplier,
                                                        &shift))
            {
                return false;
            }
            for (uint32_t c = 0; c < channels; ++c)
            {
                multipliers[c] = multiplier;
                shifts[c] = shift;
            }
            return true;
        }

        for (uint32_t c = 0; c < channels; ++c)
        {
            if (!QuantInteger::EffectiveScaleMultiplier(quant.input_scale,
                                                        quant.weight_channel_scales[c],
                                                        quant.output_scale,
                                                        &multipliers[c],
                                                        &shifts[c]))
            {
                return false;
            }
        }
        return true;
    }
#endif

    // Prefer caller-provided plan multipliers; otherwise heap-fill once per call
    // (CPU/MPU only). MCU never allocates — plan must supply multipliers.
    bool ResolveMultipliers(const NkFormat::MlpLayerQuantDesc& quant,
                            uint32_t channels,
                            const int32_t* in_multipliers,
                            const int32_t* in_shifts,
                            const int32_t** out_multipliers,
                            const int32_t** out_shifts,
                            std::unique_ptr<int32_t[]>& owned)
    {
        if (in_multipliers && in_shifts)
        {
            *out_multipliers = in_multipliers;
            *out_shifts = in_shifts;
            return true;
        }
#if defined(NETKIT_CLASS_MCU)
        (void)quant;
        (void)channels;
        (void)owned;
        return false;
#else
        owned.reset(new int32_t[static_cast<std::size_t>(channels) * 2u]);
        int32_t* m = owned.get();
        int32_t* s = m + channels;
        if (!FillEffectiveMultipliers(quant, channels, m, s))
            return false;
        *out_multipliers = m;
        *out_shifts = s;
        return true;
#endif
    }

    // Fully-valid output range where every kernel tap is in-bounds (no pad checks).
    void ValidOutputRange(int32_t out_size,
                          int32_t in_size,
                          int kernel,
                          int stride,
                          int pad,
                          int32_t* lo_exclusive_hi_lo,
                          int32_t* lo_exclusive_hi_hi)
    {
        if (kernel <= 0 || stride <= 0 || out_size <= 0)
        {
            *lo_exclusive_hi_lo = 0;
            *lo_exclusive_hi_hi = 0;
            return;
        }
        // oh >= ceil(pad / stride)
        int32_t lo = (pad + stride - 1) / stride;
        // oh <= floor((in_size + pad - kernel) / stride)
        int32_t hi = (in_size + pad - kernel) / stride + 1;
        if (lo < 0)
            lo = 0;
        if (hi > out_size)
            hi = out_size;
        if (lo > hi)
            lo = hi;
        *lo_exclusive_hi_lo = lo;
        *lo_exclusive_hi_hi = hi;
    }

#if !defined(NETKIT_CLASS_MCU)
    // bias' = bias + input_offset * sum(filter)  (weight_zp == 0). Bit-exact with
    // acc += filter * (input + input_offset) when every tap is applied (interior or
    // pad==0). Border pixels with skip-OOB still use the original bias + offset MAC.
    void FoldInputOffsetIntoBias(const int8_t* weights,
                                 const int32_t* bias,
                                 uint32_t out_channels,
                                 uint32_t filter_elems,
                                 int32_t input_offset,
                                 int32_t* bias_folded)
    {
        for (uint32_t oc = 0; oc < out_channels; ++oc)
        {
            const int8_t* filter = weights + static_cast<std::size_t>(oc) * filter_elems;
            int32_t sum = 0;
            for (uint32_t i = 0; i < filter_elems; ++i)
                sum += static_cast<int32_t>(filter[i]);
            bias_folded[oc] = bias[oc] + input_offset * sum;
        }
    }
#endif

    const int32_t* ResolveBiasInterior(const int32_t* bias,
                                       const int32_t* bias_folded_plan,
                                       const int8_t* weights,
                                       uint32_t out_channels,
                                       uint32_t filter_elems,
                                       int32_t input_offset,
                                       int32_t filter_offset,
                                       std::unique_ptr<int32_t[]>* scratch)
    {
        if (filter_offset != 0 || input_offset == 0)
            return bias;
        if (bias_folded_plan)
            return bias_folded_plan;
#if defined(NETKIT_CLASS_MCU)
        // No heap scratch on MCU — keep original bias; callers use offset MAC.
        (void)weights;
        (void)out_channels;
        (void)filter_elems;
        (void)scratch;
        return bias;
#else
        scratch->reset(new int32_t[out_channels]);
        FoldInputOffsetIntoBias(
            weights, bias, out_channels, filter_elems, input_offset, scratch->get());
        return scratch->get();
#endif
    }

    // Fast path: 1x1 conv, no padding — dominant UIB expand/proj shape.
    void Conv2d1x1NhwcQuantRef(const int8_t* input,
                               uint32_t in_h,
                               uint32_t in_w,
                               uint32_t in_c,
                               const int8_t* weights,
                               const int32_t* bias_use,
                               int stride,
                               int out_channels,
                               uint32_t out_h,
                               uint32_t out_w,
                               bool plain_mac,
                               int32_t input_offset,
                               const int32_t* m_ptr,
                               const int32_t* s_ptr,
                               int32_t output_zero_point,
                               int32_t baked_min,
                               int32_t baked_max,
                               int8_t* output)
    {
        (void)in_h;
        const uint32_t oc_count = static_cast<uint32_t>(out_channels);
        const uint32_t in_row_stride = in_w * in_c;
        for (size_t oh = 0; oh < out_h; ++oh)
        {
            const int8_t* in_row = input + static_cast<std::size_t>(oh) * stride * in_row_stride;
            for (size_t ow = 0; ow < out_w; ++ow)
            {
                const int8_t* in_ptr = in_row + static_cast<std::size_t>(ow) * stride * in_c;
                const uint32_t out_spatial_base = (oh * out_w + ow) * oc_count;
                for (int oc = 0; oc < out_channels; ++oc)
                {
                    int32_t acc = bias_use[oc];
                    const int8_t* wt = weights + static_cast<uint32_t>(oc) * in_c;
                    if (plain_mac)
                    {
                        for (size_t ic = 0; ic < in_c; ++ic)
                            acc += static_cast<int32_t>(wt[ic]) * static_cast<int32_t>(in_ptr[ic]);
                    }
                    else
                    {
                        for (size_t ic = 0; ic < in_c; ++ic)
                            acc += static_cast<int32_t>(wt[ic]) *
                                   (static_cast<int32_t>(in_ptr[ic]) + input_offset);
                    }
                    output[out_spatial_base + static_cast<uint32_t>(oc)] =
                        QuantInteger::RequantizeAccToInt8(
                            acc, m_ptr[oc], s_ptr[oc], output_zero_point, baked_min, baked_max);
                }
            }
        }
    }

    void Conv2dNhwcQuantRef(const int8_t* input,
                            uint32_t in_h,
                            uint32_t in_w,
                            uint32_t in_c,
                            const int8_t* weights,
                            const int32_t* bias,
                            int kernel_size,
                            int stride,
                            int pad_h,
                            int pad_w,
                            int out_channels,
                            uint32_t out_h,
                            uint32_t out_w,
                            int32_t input_offset,
                            int32_t filter_offset,
                            const int32_t* m_ptr,
                            const int32_t* s_ptr,
                            int32_t output_zero_point,
                            int32_t baked_min,
                            int32_t baked_max,
                            int8_t* output,
                            const int32_t* bias_folded_plan)
    {
        const uint32_t kernel_area =
            static_cast<uint32_t>(kernel_size) * static_cast<uint32_t>(kernel_size);
        const uint32_t filter_elems = kernel_area * in_c;
        const uint32_t oc_count = static_cast<uint32_t>(out_channels);

        std::unique_ptr<int32_t[]> bias_scratch;
        const int32_t* bias_interior = ResolveBiasInterior(bias,
                                                           bias_folded_plan,
                                                           weights,
                                                           oc_count,
                                                           filter_elems,
                                                           input_offset,
                                                           filter_offset,
                                                           &bias_scratch);
        const bool plain_interior =
            (filter_offset == 0) && (input_offset == 0 || bias_interior != bias);

        // 1x1 + no pad: skip kh/kw entirely (UIB expand/proj).
        if (kernel_size == 1 && pad_h == 0 && pad_w == 0 && filter_offset == 0)
        {
            Conv2d1x1NhwcQuantRef(input,
                                  in_h,
                                  in_w,
                                  in_c,
                                  weights,
                                  bias_interior,
                                  stride,
                                  out_channels,
                                  out_h,
                                  out_w,
                                  plain_interior,
                                  input_offset,
                                  m_ptr,
                                  s_ptr,
                                  output_zero_point,
                                  baked_min,
                                  baked_max,
                                  output);
            return;
        }

#if NETKIT_IM2COL >= 1
        // Same NETKIT_IM2COL policy as float: partial (1) / full (2) when volume warrants.
        // filter_offset != 0 needs a different lowering — keep direct loops.
        // If full-matrix scratch does not fit, degrade to partial then direct.
        if (filter_offset == 0)
        {
            const uint32_t kh = static_cast<uint32_t>(kernel_size);
            const uint32_t kw = static_cast<uint32_t>(kernel_size);
            const Conv2dExecMode mode =
                SelectConv2dExecMode(kernel_size, kernel_size, stride, in_c, out_h, out_w);
            if (mode != Conv2dExecMode::Direct)
            {
                KernelWorkspace* workspace = GetActiveKernelWorkspace();
                const int8_t pad_value =
                    plain_interior ? static_cast<int8_t>(0)
                                   : static_cast<int8_t>(-input_offset);
                const bool try_full = (mode == Conv2dExecMode::FullIm2Col);
                const bool try_partial =
                    (mode == Conv2dExecMode::PartialIm2Col) || try_full;
#if NETKIT_IM2COL >= 2
                if (try_full && workspace && workspace->data)
                {
                    const std::size_t required =
                        ConvFullIm2ColS8WorkspaceBytes(out_h, out_w, kh, kw, in_c);
                    if (workspace->size_bytes >= required &&
                        ConvFullIm2ColS8Forward(input,
                                                weights,
                                                bias_interior,
                                                output,
                                                reinterpret_cast<int8_t*>(workspace->data),
                                                in_h,
                                                in_w,
                                                in_c,
                                                out_h,
                                                out_w,
                                                out_channels,
                                                kh,
                                                kw,
                                                stride,
                                                pad_h,
                                                pad_w,
                                                pad_value,
                                                plain_interior,
                                                input_offset,
                                                m_ptr,
                                                s_ptr,
                                                output_zero_point,
                                                baked_min,
                                                baked_max))
                    {
                        return;
                    }
                }
#endif
                if (try_partial && workspace && workspace->data)
                {
                    const std::size_t required =
                        ConvPartialIm2ColS8WorkspaceBytes(kh, kw, in_c);
                    if (workspace->size_bytes >= required &&
                        ConvPartialIm2ColS8Forward(input,
                                                   weights,
                                                   bias_interior,
                                                   output,
                                                   reinterpret_cast<int8_t*>(workspace->data),
                                                   in_h,
                                                   in_w,
                                                   in_c,
                                                   out_h,
                                                   out_w,
                                                   out_channels,
                                                   kh,
                                                   kw,
                                                   stride,
                                                   pad_h,
                                                   pad_w,
                                                   pad_value,
                                                   plain_interior,
                                                   input_offset,
                                                   m_ptr,
                                                   s_ptr,
                                                   output_zero_point,
                                                   baked_min,
                                                   baked_max))
                    {
                        return;
                    }
                }
            }
        }
#endif

        int32_t oh_lo = 0;
        int32_t oh_hi = static_cast<int32_t>(out_h);
        int32_t ow_lo = 0;
        int32_t ow_hi = static_cast<int32_t>(out_w);
        if (filter_offset == 0)
        {
            ValidOutputRange(static_cast<int32_t>(out_h),
                             static_cast<int32_t>(in_h),
                             kernel_size,
                             stride,
                             pad_h,
                             &oh_lo,
                             &oh_hi);
            ValidOutputRange(static_cast<int32_t>(out_w),
                             static_cast<int32_t>(in_w),
                             kernel_size,
                             stride,
                             pad_w,
                             &ow_lo,
                             &ow_hi);
        }

        auto conv_pixel_checked = [&](size_t oh, size_t ow, const int32_t* bias_use, bool plain) {
            const int32_t in_y_origin = static_cast<int32_t>(oh) * stride - pad_h;
            const int32_t in_x_origin = static_cast<int32_t>(ow) * stride - pad_w;
            const uint32_t out_spatial_base = (oh * out_w + ow) * oc_count;
            const int32_t kh_lo = std::max(int32_t{0}, -in_y_origin);
            const int32_t kh_hi =
                std::min(static_cast<int32_t>(kernel_size), static_cast<int32_t>(in_h) - in_y_origin);
            const int32_t kw_lo = std::max(int32_t{0}, -in_x_origin);
            const int32_t kw_hi =
                std::min(static_cast<int32_t>(kernel_size), static_cast<int32_t>(in_w) - in_x_origin);
            for (int oc = 0; oc < out_channels; ++oc)
            {
                int32_t acc = bias_use[oc];
                const int8_t* filter = weights + static_cast<uint32_t>(oc) * filter_elems;
                for (int kh = kh_lo; kh < kh_hi; ++kh)
                {
                    const int32_t ih = in_y_origin + kh;
                    for (int kw = kw_lo; kw < kw_hi; ++kw)
                    {
                        const int32_t iw = in_x_origin + kw;
                        const int8_t* in_ptr =
                            input + (static_cast<uint32_t>(ih) * in_w + static_cast<uint32_t>(iw)) *
                                        in_c;
                        const int8_t* wt_ptr =
                            filter + (static_cast<uint32_t>(kh) * static_cast<uint32_t>(kernel_size) +
                                      static_cast<uint32_t>(kw)) *
                                         in_c;
                        for (size_t ic = 0; ic < in_c; ++ic)
                        {
                            const int32_t in_q = static_cast<int32_t>(in_ptr[ic]);
                            const int32_t wt_q = static_cast<int32_t>(wt_ptr[ic]);
                            if (plain)
                                acc += wt_q * in_q;
                            else if (filter_offset == 0)
                                acc += wt_q * (in_q + input_offset);
                            else
                                acc += (in_q + input_offset) * (wt_q + filter_offset);
                        }
                    }
                }
                output[out_spatial_base + static_cast<uint32_t>(oc)] =
                    QuantInteger::RequantizeAccToInt8(
                        acc, m_ptr[oc], s_ptr[oc], output_zero_point, baked_min, baked_max);
            }
        };

        // Interior: pointer-walk NHWC; 3x3/5x5 unroll kh/kw.
        // Stem is 3x3 (ImageNet L0: 224x224x3); 5x5 dense is rare — ExtraDW is depthwise.
        auto conv_mac_ic = [&](int32_t& acc, const int8_t* in_ptr, const int8_t* wt_ptr) {
            if (in_c == 3)
            {
                acc += static_cast<int32_t>(wt_ptr[0]) * static_cast<int32_t>(in_ptr[0]);
                acc += static_cast<int32_t>(wt_ptr[1]) * static_cast<int32_t>(in_ptr[1]);
                acc += static_cast<int32_t>(wt_ptr[2]) * static_cast<int32_t>(in_ptr[2]);
                return;
            }
            for (size_t ic = 0; ic < in_c; ++ic)
                acc += static_cast<int32_t>(wt_ptr[ic]) * static_cast<int32_t>(in_ptr[ic]);
        };

        // 3x3 RGB stem: gather 27 inputs once, reuse across all OCs.
        auto conv_pixel_interior_3x3_c3 = [&](size_t oh, size_t ow) {
            const int32_t in_y_origin = static_cast<int32_t>(oh) * stride - pad_h;
            const int32_t in_x_origin = static_cast<int32_t>(ow) * stride - pad_w;
            const int8_t* in00 =
                input + (static_cast<uint32_t>(in_y_origin) * in_w +
                         static_cast<uint32_t>(in_x_origin)) *
                            3u;
            const uint32_t row = in_w * 3u;
            const uint32_t out_spatial_base = (oh * out_w + ow) * oc_count;
            int8_t patch[27];
            const int8_t* src[9] = {in00,
                                    in00 + 3,
                                    in00 + 6,
                                    in00 + row,
                                    in00 + row + 3,
                                    in00 + row + 6,
                                    in00 + 2 * row,
                                    in00 + 2 * row + 3,
                                    in00 + 2 * row + 6};
            for (int t = 0; t < 9; ++t)
            {
                patch[t * 3 + 0] = src[t][0];
                patch[t * 3 + 1] = src[t][1];
                patch[t * 3 + 2] = src[t][2];
            }
            for (int oc = 0; oc < out_channels; ++oc)
            {
                int32_t acc = bias_interior[oc];
                const int8_t* f = weights + static_cast<uint32_t>(oc) * 27u;
                for (int i = 0; i < 27; ++i)
                    acc += static_cast<int32_t>(f[i]) * static_cast<int32_t>(patch[i]);
                output[out_spatial_base + static_cast<uint32_t>(oc)] =
                    QuantInteger::RequantizeAccToInt8(
                        acc, m_ptr[oc], s_ptr[oc], output_zero_point, baked_min, baked_max);
            }
        };

        auto conv_pixel_interior_3x3 = [&](size_t oh, size_t ow) {
            if (in_c == 3)
            {
                conv_pixel_interior_3x3_c3(oh, ow);
                return;
            }
            const int32_t in_y_origin = static_cast<int32_t>(oh) * stride - pad_h;
            const int32_t in_x_origin = static_cast<int32_t>(ow) * stride - pad_w;
            const int8_t* in00 =
                input + (static_cast<uint32_t>(in_y_origin) * in_w +
                         static_cast<uint32_t>(in_x_origin)) *
                            in_c;
            const uint32_t row = in_w * in_c;
            const uint32_t out_spatial_base = (oh * out_w + ow) * oc_count;
            // Gather 3x3 patch once (contiguous OHWI weights ⇒ flat MAC per OC).
            // Cap covers MNv4 stem/downsamples (in_c ≤ 96); else fall back to walks.
            constexpr uint32_t kPatchCap = 9u * 96u;
            if (filter_elems <= kPatchCap)
            {
                alignas(16) int8_t patch[kPatchCap];
                const int8_t* taps[9] = {in00,
                                         in00 + in_c,
                                         in00 + 2 * in_c,
                                         in00 + row,
                                         in00 + row + in_c,
                                         in00 + row + 2 * in_c,
                                         in00 + 2 * row,
                                         in00 + 2 * row + in_c,
                                         in00 + 2 * row + 2 * in_c};
                for (int t = 0; t < 9; ++t)
                    std::memcpy(patch + static_cast<uint32_t>(t) * in_c, taps[t], in_c);
                for (int oc = 0; oc < out_channels; ++oc)
                {
                    int32_t acc = bias_interior[oc];
                    const int8_t* f = weights + static_cast<uint32_t>(oc) * filter_elems;
                    for (uint32_t i = 0; i < filter_elems; ++i)
                        acc += static_cast<int32_t>(f[i]) * static_cast<int32_t>(patch[i]);
                    output[out_spatial_base + static_cast<uint32_t>(oc)] =
                        QuantInteger::RequantizeAccToInt8(
                            acc, m_ptr[oc], s_ptr[oc], output_zero_point, baked_min, baked_max);
                }
                return;
            }
            for (int oc = 0; oc < out_channels; ++oc)
            {
                int32_t acc = bias_interior[oc];
                const int8_t* f = weights + static_cast<uint32_t>(oc) * filter_elems;
                conv_mac_ic(acc, in00, f);
                conv_mac_ic(acc, in00 + in_c, f + in_c);
                conv_mac_ic(acc, in00 + 2 * in_c, f + 2 * in_c);
                conv_mac_ic(acc, in00 + row, f + 3 * in_c);
                conv_mac_ic(acc, in00 + row + in_c, f + 4 * in_c);
                conv_mac_ic(acc, in00 + row + 2 * in_c, f + 5 * in_c);
                conv_mac_ic(acc, in00 + 2 * row, f + 6 * in_c);
                conv_mac_ic(acc, in00 + 2 * row + in_c, f + 7 * in_c);
                conv_mac_ic(acc, in00 + 2 * row + 2 * in_c, f + 8 * in_c);
                output[out_spatial_base + static_cast<uint32_t>(oc)] =
                    QuantInteger::RequantizeAccToInt8(
                        acc, m_ptr[oc], s_ptr[oc], output_zero_point, baked_min, baked_max);
            }
        };

        auto conv_pixel_interior_5x5 = [&](size_t oh, size_t ow) {
            const int32_t in_y_origin = static_cast<int32_t>(oh) * stride - pad_h;
            const int32_t in_x_origin = static_cast<int32_t>(ow) * stride - pad_w;
            const int8_t* in00 =
                input + (static_cast<uint32_t>(in_y_origin) * in_w +
                         static_cast<uint32_t>(in_x_origin)) *
                            in_c;
            const uint32_t row = in_w * in_c;
            const uint32_t out_spatial_base = (oh * out_w + ow) * oc_count;
            for (int oc = 0; oc < out_channels; ++oc)
            {
                int32_t acc = bias_interior[oc];
                const int8_t* f = weights + static_cast<uint32_t>(oc) * filter_elems;
                // Fully unroll kh (same form as 3x3).
                for (int kh = 0; kh < 5; ++kh)
                {
                    const int8_t* in_row = in00 + static_cast<uint32_t>(kh) * row;
                    const int8_t* wt_row = f + static_cast<uint32_t>(kh) * 5u * in_c;
                    conv_mac_ic(acc, in_row, wt_row);
                    conv_mac_ic(acc, in_row + in_c, wt_row + in_c);
                    conv_mac_ic(acc, in_row + 2 * in_c, wt_row + 2 * in_c);
                    conv_mac_ic(acc, in_row + 3 * in_c, wt_row + 3 * in_c);
                    conv_mac_ic(acc, in_row + 4 * in_c, wt_row + 4 * in_c);
                }
                output[out_spatial_base + static_cast<uint32_t>(oc)] =
                    QuantInteger::RequantizeAccToInt8(
                        acc, m_ptr[oc], s_ptr[oc], output_zero_point, baked_min, baked_max);
            }
        };

        auto conv_pixel_interior = [&](size_t oh, size_t ow) {
            if (kernel_size == 3)
            {
                conv_pixel_interior_3x3(oh, ow);
                return;
            }
            if (kernel_size == 5)
            {
                conv_pixel_interior_5x5(oh, ow);
                return;
            }
            const int32_t in_y_origin = static_cast<int32_t>(oh) * stride - pad_h;
            const int32_t in_x_origin = static_cast<int32_t>(ow) * stride - pad_w;
            const int8_t* in_base =
                input + (static_cast<uint32_t>(in_y_origin) * in_w +
                         static_cast<uint32_t>(in_x_origin)) *
                            in_c;
            const uint32_t out_spatial_base = (oh * out_w + ow) * oc_count;
            for (int oc = 0; oc < out_channels; ++oc)
            {
                int32_t acc = bias_interior[oc];
                const int8_t* filter = weights + static_cast<uint32_t>(oc) * filter_elems;
                for (int kh = 0; kh < kernel_size; ++kh)
                {
                    const int8_t* in_row = in_base + static_cast<uint32_t>(kh) * in_w * in_c;
                    const int8_t* wt_row =
                        filter + static_cast<uint32_t>(kh) * static_cast<uint32_t>(kernel_size) * in_c;
                    for (int kw = 0; kw < kernel_size; ++kw)
                    {
                        const int8_t* in_ptr = in_row + static_cast<uint32_t>(kw) * in_c;
                        const int8_t* wt_ptr = wt_row + static_cast<uint32_t>(kw) * in_c;
                        conv_mac_ic(acc, in_ptr, wt_ptr);
                    }
                }
                output[out_spatial_base + static_cast<uint32_t>(oc)] =
                    QuantInteger::RequantizeAccToInt8(
                        acc, m_ptr[oc], s_ptr[oc], output_zero_point, baked_min, baked_max);
            }
        };

        const bool plain_border = (input_offset == 0 && filter_offset == 0);

        for (int32_t oh = 0; oh < oh_lo; ++oh)
            for (size_t ow = 0; ow < out_w; ++ow)
                conv_pixel_checked(static_cast<size_t>(oh), ow, bias, plain_border);

        for (int32_t oh = oh_lo; oh < oh_hi; ++oh)
        {
            for (int32_t ow = 0; ow < ow_lo; ++ow)
                conv_pixel_checked(static_cast<size_t>(oh), static_cast<size_t>(ow), bias, plain_border);
            for (int32_t ow = ow_lo; ow < ow_hi; ++ow)
                conv_pixel_interior(static_cast<size_t>(oh), static_cast<size_t>(ow));
            for (int32_t ow = ow_hi; ow < static_cast<int32_t>(out_w); ++ow)
                conv_pixel_checked(static_cast<size_t>(oh), static_cast<size_t>(ow), bias, plain_border);
        }

        for (int32_t oh = oh_hi; oh < static_cast<int32_t>(out_h); ++oh)
            for (size_t ow = 0; ow < out_w; ++ow)
                conv_pixel_checked(static_cast<size_t>(oh), ow, bias, plain_border);
    }

    void DepthwiseConv2dNhwcQuantRef(const int8_t* input,
                                     uint32_t in_h,
                                     uint32_t in_w,
                                     uint32_t channels,
                                     const int8_t* weights,
                                     const int32_t* bias,
                                     int kernel_h,
                                     int kernel_w,
                                     int stride,
                                     int pad_h,
                                     int pad_w,
                                     uint32_t out_h,
                                     uint32_t out_w,
                                     int32_t input_offset,
                                     int32_t filter_offset,
                                     const int32_t* m_ptr,
                                     const int32_t* s_ptr,
                                     int32_t output_zero_point,
                                     int32_t baked_min,
                                     int32_t baked_max,
                                     int8_t* output,
                                     const int32_t* bias_folded_plan)
    {
        const uint32_t kernel_area =
            static_cast<uint32_t>(kernel_h) * static_cast<uint32_t>(kernel_w);

        std::unique_ptr<int32_t[]> bias_scratch;
        const int32_t* bias_interior = ResolveBiasInterior(bias,
                                                           bias_folded_plan,
                                                           weights,
                                                           channels,
                                                           kernel_area,
                                                           input_offset,
                                                           filter_offset,
                                                           &bias_scratch);

        int32_t oh_lo = 0;
        int32_t oh_hi = static_cast<int32_t>(out_h);
        int32_t ow_lo = 0;
        int32_t ow_hi = static_cast<int32_t>(out_w);
        if (filter_offset == 0)
        {
            ValidOutputRange(static_cast<int32_t>(out_h),
                             static_cast<int32_t>(in_h),
                             kernel_h,
                             stride,
                             pad_h,
                             &oh_lo,
                             &oh_hi);
            ValidOutputRange(static_cast<int32_t>(out_w),
                             static_cast<int32_t>(in_w),
                             kernel_w,
                             stride,
                             pad_w,
                             &ow_lo,
                             &ow_hi);
        }

        auto dw_pixel_checked = [&](size_t oh, size_t ow, const int32_t* bias_use, bool plain) {
            const int32_t in_y_origin = static_cast<int32_t>(oh) * stride - pad_h;
            const int32_t in_x_origin = static_cast<int32_t>(ow) * stride - pad_w;
            const uint32_t out_spatial_base = (oh * out_w + ow) * channels;
            const int32_t kh_lo = std::max(int32_t{0}, -in_y_origin);
            const int32_t kh_hi =
                std::min(static_cast<int32_t>(kernel_h), static_cast<int32_t>(in_h) - in_y_origin);
            const int32_t kw_lo = std::max(int32_t{0}, -in_x_origin);
            const int32_t kw_hi =
                std::min(static_cast<int32_t>(kernel_w), static_cast<int32_t>(in_w) - in_x_origin);
            for (uint32_t c = 0; c < channels; ++c)
            {
                int32_t acc = bias_use[c];
                const int8_t* filter = weights + c * kernel_area;
                for (int kh = kh_lo; kh < kh_hi; ++kh)
                {
                    const int32_t ih = in_y_origin + kh;
                    for (int kw = kw_lo; kw < kw_hi; ++kw)
                    {
                        const int32_t iw = in_x_origin + kw;
                        const int8_t in_val_q =
                            input[(static_cast<uint32_t>(ih) * in_w + static_cast<uint32_t>(iw)) *
                                      channels +
                                  c];
                        const int8_t wt_val_q =
                            filter[static_cast<uint32_t>(kh) * static_cast<uint32_t>(kernel_w) +
                                   static_cast<uint32_t>(kw)];
                        const int32_t in_q = static_cast<int32_t>(in_val_q);
                        const int32_t wt_q = static_cast<int32_t>(wt_val_q);
                        if (plain)
                            acc += wt_q * in_q;
                        else if (filter_offset == 0)
                            acc += wt_q * (in_q + input_offset);
                        else
                            acc += (in_q + input_offset) * (wt_q + filter_offset);
                    }
                }
                output[out_spatial_base + c] = QuantInteger::RequantizeAccToInt8(
                    acc, m_ptr[c], s_ptr[c], output_zero_point, baked_min, baked_max);
            }
        };

        // Interior 3x3: unroll taps (common MobileNet DW).
        auto dw_pixel_interior_3x3 = [&](size_t oh, size_t ow) {
            const int32_t in_y_origin = static_cast<int32_t>(oh) * stride - pad_h;
            const int32_t in_x_origin = static_cast<int32_t>(ow) * stride - pad_w;
            const uint32_t out_spatial_base = (oh * out_w + ow) * channels;
            const int8_t* in00 =
                input + (static_cast<uint32_t>(in_y_origin) * in_w +
                         static_cast<uint32_t>(in_x_origin)) *
                            channels;
            const uint32_t row = in_w * channels;
            for (uint32_t c = 0; c < channels; ++c)
            {
                const int8_t* f = weights + c * 9u;
                int32_t acc = bias_interior[c];
                acc += static_cast<int32_t>(f[0]) * static_cast<int32_t>(in00[c]);
                acc += static_cast<int32_t>(f[1]) * static_cast<int32_t>(in00[c + channels]);
                acc += static_cast<int32_t>(f[2]) * static_cast<int32_t>(in00[c + 2 * channels]);
                acc += static_cast<int32_t>(f[3]) * static_cast<int32_t>(in00[c + row]);
                acc += static_cast<int32_t>(f[4]) * static_cast<int32_t>(in00[c + row + channels]);
                acc += static_cast<int32_t>(f[5]) * static_cast<int32_t>(in00[c + row + 2 * channels]);
                acc += static_cast<int32_t>(f[6]) * static_cast<int32_t>(in00[c + 2 * row]);
                acc += static_cast<int32_t>(f[7]) * static_cast<int32_t>(in00[c + 2 * row + channels]);
                acc += static_cast<int32_t>(f[8]) *
                       static_cast<int32_t>(in00[c + 2 * row + 2 * channels]);
                output[out_spatial_base + c] = QuantInteger::RequantizeAccToInt8(
                    acc, m_ptr[c], s_ptr[c], output_zero_point, baked_min, baked_max);
            }
        };

        // Interior 5x5: unroll kw per row (MobileNetV4 ExtraDW).
        auto dw_pixel_interior_5x5 = [&](size_t oh, size_t ow) {
            const int32_t in_y_origin = static_cast<int32_t>(oh) * stride - pad_h;
            const int32_t in_x_origin = static_cast<int32_t>(ow) * stride - pad_w;
            const uint32_t out_spatial_base = (oh * out_w + ow) * channels;
            const int8_t* in00 =
                input + (static_cast<uint32_t>(in_y_origin) * in_w +
                         static_cast<uint32_t>(in_x_origin)) *
                            channels;
            const uint32_t row = in_w * channels;
            for (uint32_t c = 0; c < channels; ++c)
            {
                const int8_t* f = weights + c * 25u;
                int32_t acc = bias_interior[c];
                for (int kh = 0; kh < 5; ++kh)
                {
                    const int8_t* in_row = in00 + static_cast<uint32_t>(kh) * row + c;
                    const int8_t* wt_row = f + static_cast<uint32_t>(kh) * 5;
                    acc += static_cast<int32_t>(wt_row[0]) * static_cast<int32_t>(in_row[0]);
                    acc += static_cast<int32_t>(wt_row[1]) *
                           static_cast<int32_t>(in_row[channels]);
                    acc += static_cast<int32_t>(wt_row[2]) *
                           static_cast<int32_t>(in_row[2 * channels]);
                    acc += static_cast<int32_t>(wt_row[3]) *
                           static_cast<int32_t>(in_row[3 * channels]);
                    acc += static_cast<int32_t>(wt_row[4]) *
                           static_cast<int32_t>(in_row[4 * channels]);
                }
                output[out_spatial_base + c] = QuantInteger::RequantizeAccToInt8(
                    acc, m_ptr[c], s_ptr[c], output_zero_point, baked_min, baked_max);
            }
        };

        auto dw_pixel_interior = [&](size_t oh, size_t ow) {
            if (kernel_h == 3 && kernel_w == 3)
            {
                dw_pixel_interior_3x3(oh, ow);
                return;
            }
            if (kernel_h == 5 && kernel_w == 5)
            {
                dw_pixel_interior_5x5(oh, ow);
                return;
            }
            const int32_t in_y_origin = static_cast<int32_t>(oh) * stride - pad_h;
            const int32_t in_x_origin = static_cast<int32_t>(ow) * stride - pad_w;
            const uint32_t out_spatial_base = (oh * out_w + ow) * channels;
            for (uint32_t c = 0; c < channels; ++c)
            {
                int32_t acc = bias_interior[c];
                const int8_t* filter = weights + c * kernel_area;
                for (int kh = 0; kh < kernel_h; ++kh)
                {
                    const int32_t ih = in_y_origin + kh;
                    for (int kw = 0; kw < kernel_w; ++kw)
                    {
                        const int32_t iw = in_x_origin + kw;
                        const int8_t in_val_q =
                            input[(static_cast<uint32_t>(ih) * in_w + static_cast<uint32_t>(iw)) *
                                      channels +
                                  c];
                        const int8_t wt_val_q =
                            filter[static_cast<uint32_t>(kh) * static_cast<uint32_t>(kernel_w) +
                                   static_cast<uint32_t>(kw)];
                        acc += static_cast<int32_t>(wt_val_q) * static_cast<int32_t>(in_val_q);
                    }
                }
                output[out_spatial_base + c] = QuantInteger::RequantizeAccToInt8(
                    acc, m_ptr[c], s_ptr[c], output_zero_point, baked_min, baked_max);
            }
        };

        const bool plain_border = (input_offset == 0 && filter_offset == 0);

        for (int32_t oh = 0; oh < oh_lo; ++oh)
            for (size_t ow = 0; ow < out_w; ++ow)
                dw_pixel_checked(static_cast<size_t>(oh), ow, bias, plain_border);

        for (int32_t oh = oh_lo; oh < oh_hi; ++oh)
        {
            for (int32_t ow = 0; ow < ow_lo; ++ow)
                dw_pixel_checked(static_cast<size_t>(oh), static_cast<size_t>(ow), bias, plain_border);
            for (int32_t ow = ow_lo; ow < ow_hi; ++ow)
                dw_pixel_interior(static_cast<size_t>(oh), static_cast<size_t>(ow));
            for (int32_t ow = ow_hi; ow < static_cast<int32_t>(out_w); ++ow)
                dw_pixel_checked(static_cast<size_t>(oh), static_cast<size_t>(ow), bias, plain_border);
        }

        for (int32_t oh = oh_hi; oh < static_cast<int32_t>(out_h); ++oh)
            for (size_t ow = 0; ow < out_w; ++ow)
                dw_pixel_checked(static_cast<size_t>(oh), ow, bias, plain_border);
    }

    void FullyConnectedQuantRef(const int8_t* input,
                                uint32_t batch,
                                uint32_t in_features,
                                const int8_t* weights,
                                const int32_t* bias,
                                uint32_t out_features,
                                int32_t input_offset,
                                int32_t filter_offset,
                                const int32_t* m_ptr,
                                const int32_t* s_ptr,
                                int32_t output_zero_point,
                                int32_t baked_min,
                                int32_t baked_max,
                                int8_t* output_int8,
                                const int32_t* bias_folded_plan)
    {
        std::unique_ptr<int32_t[]> bias_scratch;
        const int32_t* bias_use = ResolveBiasInterior(bias,
                                                      bias_folded_plan,
                                                      weights,
                                                      out_features,
                                                      in_features,
                                                      input_offset,
                                                      filter_offset,
                                                      &bias_scratch);
        // After fold, MAC is plain.
        if (bias_use != bias && filter_offset == 0)
            input_offset = 0;

        for (size_t b = 0; b < batch; ++b)
        {
            const int8_t* in_row = input + b * in_features;
            for (size_t oc = 0; oc < out_features; ++oc)
            {
                int32_t acc = bias_use[oc];
                const int8_t* wt_row = weights + oc * in_features;
                if (filter_offset == 0)
                {
                    if (input_offset == 0)
                    {
                        for (size_t ic = 0; ic < in_features; ++ic)
                            acc += static_cast<int32_t>(wt_row[ic]) *
                                   static_cast<int32_t>(in_row[ic]);
                    }
                    else
                    {
                        for (size_t ic = 0; ic < in_features; ++ic)
                            acc += static_cast<int32_t>(wt_row[ic]) *
                                   (static_cast<int32_t>(in_row[ic]) + input_offset);
                    }
                }
                else
                {
                    for (size_t ic = 0; ic < in_features; ++ic)
                        acc += (static_cast<int32_t>(in_row[ic]) + input_offset) *
                               (static_cast<int32_t>(wt_row[ic]) + filter_offset);
                }

                output_int8[b * out_features + oc] = QuantInteger::RequantizeAccToInt8(
                    acc, m_ptr[oc], s_ptr[oc], output_zero_point, baked_min, baked_max);
            }
        }
    }
#endif // !NETKIT_MCU_CMSIS_ONLY
}  // namespace

    void FullyConnectedQuant(const int8_t* input,
                             uint32_t batch,
                             uint32_t in_features,
                             const int8_t* weights,
                             const int32_t* bias,
                             uint32_t out_features,
                             const NkFormat::MlpLayerQuantDesc& quant,
                             QuantInteger::QuantClamp clamp,
                             int8_t* output_int8,
                             const int32_t* multipliers,
                             const int32_t* shifts,
                             const int32_t* act_min,
                             const int32_t* act_max,
                             const int32_t* bias_folded)
    {
#if defined(NETKIT_USE_XNNPACK) && NETKIT_USE_XNNPACK && NETKIT_XNNPACK_ALLOWED
        if (XnnpackQuant::TryFullyConnectedQuant(input,
                                                 batch,
                                                 in_features,
                                                 weights,
                                                 bias,
                                                 out_features,
                                                 quant,
                                                 clamp == QuantInteger::QuantClamp::ReLU ||
                                                     clamp == QuantInteger::QuantClamp::ReLU6,
                                                 output_int8))
        {
            return;
        }
#endif
#if defined(NETKIT_USE_CMSIS_NN) && NETKIT_USE_CMSIS_NN && NETKIT_CMSIS_NN_ALLOWED
        if (CmsisNnQuant::TryFullyConnectedQuant(input,
                                                 batch,
                                                 in_features,
                                                 weights,
                                                 bias,
                                                 out_features,
                                                 quant,
                                                 clamp == QuantInteger::QuantClamp::ReLU ||
                                                     clamp == QuantInteger::QuantClamp::ReLU6,
                                                 output_int8))
        {
            return;
        }
#endif
#if !NETKIT_MCU_CMSIS_ONLY
        QuantTrace::RecordFcReference();

        // Int8→int8 only: integer multiply-by-quantized-multiplier (no float requantize/dequant).
        if (!output_int8)
            return;

        std::unique_ptr<int32_t[]> owned;
        const int32_t* m_ptr = nullptr;
        const int32_t* s_ptr = nullptr;
        if (!ResolveMultipliers(quant, out_features, multipliers, shifts, &m_ptr, &s_ptr, owned))
            return;

        int32_t baked_min = -128;
        int32_t baked_max = 127;
        if (act_min && act_max)
        {
            baked_min = *act_min;
            baked_max = *act_max;
        }
        else
        {
            QuantInteger::QuantClampRange(
                clamp, quant.output_scale, quant.output_zero_point, &baked_min, &baked_max);
        }

        const int32_t input_offset = -quant.input_zero_point;
        const int32_t filter_offset = -quant.weight_zero_point;
        FullyConnectedQuantRef(input,
                               batch,
                               in_features,
                               weights,
                               bias,
                               out_features,
                               input_offset,
                               filter_offset,
                               m_ptr,
                               s_ptr,
                               quant.output_zero_point,
                               baked_min,
                               baked_max,
                               output_int8,
                               bias_folded);
#else
        (void)multipliers;
        (void)shifts;
        (void)act_min;
        (void)act_max;
        (void)bias_folded;
#endif
    }

    void ForwardQuantizedDense(const Tensor& input,
                               const Tensor& weights,
                               const Tensor& bias,
                               const LayerQuant& quant,
                               bool apply_relu,
                               bool apply_softmax,
                               int8_t* quant_scratch,
                               Tensor& output)
    {
        if (input.type != DataType::Int8 || output.type != DataType::Int8 || !input.data || !output.data)
            return;

        const uint32_t batch = input.shape[0];
        const uint32_t in_features = input.shape[1];
        const uint32_t out_features = weights.shape[0];

        const int8_t* input_i8 = static_cast<const int8_t*>(input.data);
        const int8_t* weight_q = static_cast<const int8_t*>(weights.data);
        const int32_t* bias_q = static_cast<const int32_t*>(bias.data);
        int8_t* logits_i8 = quant_scratch;

        if (apply_softmax)
        {
            FullyConnectedQuant(input_i8,
                                batch,
                                in_features,
                                weight_q,
                                bias_q,
                                out_features,
                                quant.params,
                                false,
                                logits_i8);

            const float logit_scale =
                quant.params.output_scale > 0.0f ? quant.params.output_scale : 1.0f;
            SoftmaxS8(logits_i8, out_features, logit_scale, static_cast<int8_t*>(output.data));
            return;
        }

        FullyConnectedQuant(input_i8,
                            batch,
                            in_features,
                            weight_q,
                            bias_q,
                            out_features,
                            quant.params,
                            apply_relu,
                            static_cast<int8_t*>(output.data));
    }

    void Conv2dNhwcQuant(const int8_t* input,
                         uint32_t in_h,
                         uint32_t in_w,
                         uint32_t in_c,
                         const int8_t* weights,
                         const int32_t* bias,
                         int kernel_size,
                         int stride,
                         int pad_h,
                         int pad_w,
                         int pad_h_end,
                         int pad_w_end,
                         int out_channels,
                         const NkFormat::MlpLayerQuantDesc& quant,
                         QuantInteger::QuantClamp clamp,
                         int8_t* output,
                         const ResidualAddS8* residual,
                         const int32_t* multipliers,
                         const int32_t* shifts,
                         const int32_t* act_min,
                         const int32_t* act_max,
                         const int32_t* bias_folded)
    {
        bool wrote = false;
#if defined(NETKIT_USE_XNNPACK) && NETKIT_USE_XNNPACK && NETKIT_XNNPACK_ALLOWED
        if (XnnpackQuant::TryConv2dNhwcQuant(input,
                                             in_h,
                                             in_w,
                                             in_c,
                                             weights,
                                             bias,
                                             kernel_size,
                                             stride,
                                             pad_h,
                                             pad_w,
                                             pad_h_end,
                                             pad_w_end,
                                             out_channels,
                                             quant,
                                             clamp == QuantInteger::QuantClamp::ReLU ||
                                                 clamp == QuantInteger::QuantClamp::ReLU6,
                                             output))
        {
            wrote = true;
        }
#endif
#if defined(NETKIT_USE_CMSIS_NN) && NETKIT_USE_CMSIS_NN && NETKIT_CMSIS_NN_ALLOWED
        if (!wrote && CmsisNnQuant::TryConv2dNhwcQuant(input,
                                             in_h,
                                             in_w,
                                             in_c,
                                             weights,
                                             bias,
                                             kernel_size,
                                             stride,
                                             pad_h,
                                             pad_w,
                                             pad_h_end,
                                             pad_w_end,
                                             out_channels,
                                             quant,
                                             clamp == QuantInteger::QuantClamp::ReLU ||
                                                 clamp == QuantInteger::QuantClamp::ReLU6,
                                             output))
        {
            wrote = true;
        }
#endif
#if !NETKIT_MCU_CMSIS_ONLY
        if (!wrote)
        {
            QuantTrace::RecordConv2dReference();
            const uint32_t out_h = nk_op_detail::CalcOutputDimAsymmetric(
                in_h, kernel_size, stride, pad_h, pad_h_end);
            const uint32_t out_w = nk_op_detail::CalcOutputDimAsymmetric(
                in_w, kernel_size, stride, pad_w, pad_w_end);
            const uint32_t oc_count = static_cast<uint32_t>(out_channels);

            std::unique_ptr<int32_t[]> owned;
            const int32_t* m_ptr = nullptr;
            const int32_t* s_ptr = nullptr;
            if (!ResolveMultipliers(quant, oc_count, multipliers, shifts, &m_ptr, &s_ptr, owned))
                return;

            int32_t baked_min = -128;
            int32_t baked_max = 127;
            if (act_min && act_max)
            {
                baked_min = *act_min;
                baked_max = *act_max;
            }
            else
            {
                QuantInteger::QuantClampRange(
                    clamp, quant.output_scale, quant.output_zero_point, &baked_min, &baked_max);
            }

            const int32_t input_offset = -quant.input_zero_point;
            const int32_t filter_offset = -quant.weight_zero_point;
            Conv2dNhwcQuantRef(input,
                               in_h,
                               in_w,
                               in_c,
                               weights,
                               bias,
                               kernel_size,
                               stride,
                               pad_h,
                               pad_w,
                               out_channels,
                               out_h,
                               out_w,
                               input_offset,
                               filter_offset,
                               m_ptr,
                               s_ptr,
                               quant.output_zero_point,
                               baked_min,
                               baked_max,
                               output,
                               bias_folded);
        }
#else
        (void)wrote;
        (void)multipliers;
        (void)shifts;
        (void)act_min;
        (void)act_max;
        (void)bias_folded;
#endif

        if (residual && residual->data)
        {
            const uint32_t out_h = nk_op_detail::CalcOutputDimAsymmetric(
                in_h, kernel_size, stride, pad_h, pad_h_end);
            const uint32_t out_w = nk_op_detail::CalcOutputDimAsymmetric(
                in_w, kernel_size, stride, pad_w, pad_w_end);
            const uint32_t count = out_h * out_w * static_cast<uint32_t>(out_channels);
            if (residual->add_plan && residual->add_plan->ready)
            {
                ElementwiseAddS8(output, residual->data, count, *residual->add_plan, output);
            }
            else
            {
                ElementwiseAddS8(output,
                                 residual->data,
                                 count,
                                 quant.output_scale,
                                 quant.output_zero_point,
                                 residual->scale,
                                 residual->zero_point,
                                 quant.output_scale,
                                 quant.output_zero_point,
                                 output);
            }
        }
    }

    void DepthwiseConv2dNhwcQuant(const int8_t* input,
                                  uint32_t in_h,
                                  uint32_t in_w,
                                  uint32_t channels,
                                  const int8_t* weights,
                                  const int32_t* bias,
                                  int kernel_h,
                                  int kernel_w,
                                  int stride,
                                  int pad_h,
                                  int pad_w,
                                  int pad_h_end,
                                  int pad_w_end,
                                  const NkFormat::MlpLayerQuantDesc& quant,
                                  QuantInteger::QuantClamp clamp,
                                  int8_t* output,
                                  const int32_t* multipliers,
                                  const int32_t* shifts,
                                  const int32_t* act_min,
                                  const int32_t* act_max,
                                  const int32_t* bias_folded)
    {
#if defined(NETKIT_USE_XNNPACK) && NETKIT_USE_XNNPACK && NETKIT_XNNPACK_ALLOWED
        if (XnnpackQuant::TryDepthwiseConv2dNhwcQuant(input,
                                                      in_h,
                                                      in_w,
                                                      channels,
                                                      weights,
                                                      bias,
                                                      kernel_h,
                                                      kernel_w,
                                                      stride,
                                                      pad_h,
                                                      pad_w,
                                                      pad_h_end,
                                                      pad_w_end,
                                                      quant,
                                                      clamp == QuantInteger::QuantClamp::ReLU ||
                                                          clamp == QuantInteger::QuantClamp::ReLU6,
                                                      output))
        {
            return;
        }
#endif
#if defined(NETKIT_USE_CMSIS_NN) && NETKIT_USE_CMSIS_NN && NETKIT_CMSIS_NN_ALLOWED
        if (CmsisNnQuant::TryDepthwiseConv2dNhwcQuant(input,
                                                      in_h,
                                                      in_w,
                                                      channels,
                                                      weights,
                                                      bias,
                                                      kernel_h,
                                                      kernel_w,
                                                      stride,
                                                      pad_h,
                                                      pad_w,
                                                      pad_h_end,
                                                      pad_w_end,
                                                      quant,
                                                      clamp == QuantInteger::QuantClamp::ReLU ||
                                                          clamp == QuantInteger::QuantClamp::ReLU6,
                                                      output))
        {
            return;
        }
#endif
#if !NETKIT_MCU_CMSIS_ONLY
        QuantTrace::RecordConv2dReference();
        const uint32_t out_h =
            nk_op_detail::CalcOutputDimAsymmetric(in_h, kernel_h, stride, pad_h, pad_h_end);
        const uint32_t out_w =
            nk_op_detail::CalcOutputDimAsymmetric(in_w, kernel_w, stride, pad_w, pad_w_end);

        std::unique_ptr<int32_t[]> owned;
        const int32_t* m_ptr = nullptr;
        const int32_t* s_ptr = nullptr;
        if (!ResolveMultipliers(quant, channels, multipliers, shifts, &m_ptr, &s_ptr, owned))
            return;

        int32_t baked_min = -128;
        int32_t baked_max = 127;
        if (act_min && act_max)
        {
            baked_min = *act_min;
            baked_max = *act_max;
        }
        else
        {
            QuantInteger::QuantClampRange(
                clamp, quant.output_scale, quant.output_zero_point, &baked_min, &baked_max);
        }

        const int32_t input_offset = -quant.input_zero_point;
        const int32_t filter_offset = -quant.weight_zero_point;
        DepthwiseConv2dNhwcQuantRef(input,
                                    in_h,
                                    in_w,
                                    channels,
                                    weights,
                                    bias,
                                    kernel_h,
                                    kernel_w,
                                    stride,
                                    pad_h,
                                    pad_w,
                                    out_h,
                                    out_w,
                                    input_offset,
                                    filter_offset,
                                    m_ptr,
                                    s_ptr,
                                    quant.output_zero_point,
                                    baked_min,
                                    baked_max,
                                    output,
                                    bias_folded);
#else
        (void)pad_h_end;
        (void)pad_w_end;
        (void)multipliers;
        (void)shifts;
        (void)act_min;
        (void)act_max;
        (void)bias_folded;
#endif
    }

    void MaxPool2dNhwcQuant(const int8_t* input,
                            uint32_t in_h,
                            uint32_t in_w,
                            uint32_t in_c,
                            int pool_h,
                            int pool_w,
                            int stride,
                            int pad_h,
                            int pad_w,
                            int pad_h_end,
                            int pad_w_end,
                            int8_t* output)
    {
#if defined(NETKIT_USE_XNNPACK) && NETKIT_USE_XNNPACK && NETKIT_XNNPACK_ALLOWED
        if (XnnpackQuant::TryMaxPool2dNhwcQuant(input,
                                                in_h,
                                                in_w,
                                                in_c,
                                                pool_h,
                                                pool_w,
                                                stride,
                                                pad_h,
                                                pad_w,
                                                pad_h_end,
                                                pad_w_end,
                                                output))
        {
            return;
        }
#endif
#if defined(NETKIT_USE_CMSIS_NN) && NETKIT_USE_CMSIS_NN && NETKIT_CMSIS_NN_ALLOWED
        if (CmsisNnQuant::TryMaxPool2dNhwcQuant(input,
                                                in_h,
                                                in_w,
                                                in_c,
                                                pool_h,
                                                pool_w,
                                                stride,
                                                pad_h,
                                                pad_w,
                                                pad_h_end,
                                                pad_w_end,
                                                output))
        {
            return;
        }
#endif
        QuantTrace::RecordPoolReference();
        const uint32_t out_h = nk_op_detail::CalcOutputDimAsymmetric(
            in_h, pool_h, stride, pad_h, pad_h_end);
        const uint32_t out_w = nk_op_detail::CalcOutputDimAsymmetric(
            in_w, pool_w, stride, pad_w, pad_w_end);

        for (size_t oh = 0; oh < out_h; ++oh)
        {
            for (size_t ow = 0; ow < out_w; ++ow)
            {
                for (size_t c = 0; c < in_c; ++c)
                {
                    int8_t max_val = -128;
                    bool found = false;

                    for (int kh = 0; kh < pool_h; ++kh)
                    {
                        const int32_t ih = static_cast<int32_t>(oh) * stride + kh - pad_h;
                        if (ih < 0 || ih >= static_cast<int32_t>(in_h))
                            continue;

                        for (int kw = 0; kw < pool_w; ++kw)
                        {
                            const int32_t iw = static_cast<int32_t>(ow) * stride + kw - pad_w;
                            if (iw < 0 || iw >= static_cast<int32_t>(in_w))
                                continue;

                            const int8_t value = input[(static_cast<uint32_t>(ih) * in_w +
                                                        static_cast<uint32_t>(iw)) *
                                                           in_c +
                                                       c];
                            if (!found || value > max_val)
                            {
                                max_val = value;
                                found = true;
                            }
                        }
                    }

                    output[(oh * out_w + ow) * in_c + c] = found ? max_val : static_cast<int8_t>(0);
                }
            }
        }
    }

    void ElementwiseAddS8(const int8_t* input1,
                          const int8_t* input2,
                          uint32_t count,
                          const CmsisQuantPlan::ElementwiseAddPlan& plan,
                          int8_t* output)
    {
        if (!plan.ready || !output)
            return;
#if defined(NETKIT_USE_CMSIS_NN) && NETKIT_USE_CMSIS_NN && NETKIT_CMSIS_NN_ALLOWED
        // CMSIS path still uses float-scale entry; fall through to integer plan path.
#endif
#if defined(NETKIT_STAGE_TIMING) && NETKIT_STAGE_TIMING
        // Filled by cmsis_quant_plan stage timer when residual is separate; fused
        // proj+add still hits this path — count elements for cost estimation.
        static uint64_t s_add_elems = 0;
        static uint64_t s_add_calls = 0;
        s_add_elems += count;
        ++s_add_calls;
        if ((s_add_calls % 50) == 0)
            std::fprintf(stderr,
                         "ADD_S8_PLAN calls=%llu elems_total=%llu last_count=%u\n",
                         static_cast<unsigned long long>(s_add_calls),
                         static_cast<unsigned long long>(s_add_elems),
                         count);
#endif
        const int32_t left = plan.left_shift;
        for (uint32_t i = 0; i < count; ++i)
        {
            const int32_t a = (static_cast<int32_t>(input1[i]) + plan.input1_offset) << left;
            const int32_t b = (static_cast<int32_t>(input2[i]) + plan.input2_offset) << left;
            const int32_t scaled_a =
                QuantInteger::MultiplyByQuantizedMultiplier(a, plan.input1_mult, plan.input1_shift);
            const int32_t scaled_b =
                QuantInteger::MultiplyByQuantizedMultiplier(b, plan.input2_mult, plan.input2_shift);
            const int32_t raw = QuantInteger::MultiplyByQuantizedMultiplier(
                scaled_a + scaled_b, plan.output_mult, plan.output_shift);
            const int32_t q =
                std::clamp(raw + plan.output_offset, plan.act_min, plan.act_max);
            output[i] = static_cast<int8_t>(q);
        }
    }

    void ElementwiseAddS8(const int8_t* input1,
                          const int8_t* input2,
                          uint32_t count,
                          float input1_scale,
                          int32_t input1_zero_point,
                          float input2_scale,
                          int32_t input2_zero_point,
                          float output_scale,
                          int32_t output_zero_point,
                          int8_t* output)
    {
#if defined(NETKIT_USE_CMSIS_NN) && NETKIT_USE_CMSIS_NN && NETKIT_CMSIS_NN_ALLOWED
        if (CmsisNnQuant::TryElementwiseAddS8(input1,
                                              input2,
                                              count,
                                              input1_scale,
                                              input1_zero_point,
                                              input2_scale,
                                              input2_zero_point,
                                              output_scale,
                                              output_zero_point,
                                              output))
        {
            return;
        }
#endif
        // Build a one-shot plan then run the integer hot path (no float in the loop).
        CmsisQuantPlan::ElementwiseAddPlan plan{};
        constexpr int32_t kLeftShift = 20;
        const double twice_max_input_scale =
            2.0 * std::max(static_cast<double>(input1_scale), static_cast<double>(input2_scale));
        if (twice_max_input_scale <= 0.0 || output_scale <= 0.0f)
            return;

        plan.input1_offset = -input1_zero_point;
        plan.input2_offset = -input2_zero_point;
        plan.output_offset = output_zero_point;
        plan.left_shift = kLeftShift;
        plan.block_size = static_cast<int32_t>(count);
        plan.act_min = -128;
        plan.act_max = 127;
        QuantInteger::QuantizeMultiplier(static_cast<double>(input1_scale) / twice_max_input_scale,
                                         &plan.input1_mult,
                                         &plan.input1_shift);
        QuantInteger::QuantizeMultiplier(static_cast<double>(input2_scale) / twice_max_input_scale,
                                         &plan.input2_mult,
                                         &plan.input2_shift);
        QuantInteger::QuantizeMultiplier(
            twice_max_input_scale / ((1LL << kLeftShift) * static_cast<double>(output_scale)),
            &plan.output_mult,
            &plan.output_shift);
        plan.ready = true;
        ElementwiseAddS8(input1, input2, count, plan, output);
    }

    void AvgPool2dNhwcQuant(const int8_t* input,
                            uint32_t in_h,
                            uint32_t in_w,
                            uint32_t in_c,
                            int pool_h,
                            int pool_w,
                            int stride,
                            int pad_h,
                            int pad_w,
                            int pad_h_end,
                            int pad_w_end,
                            float input_scale,
                            int32_t input_zero_point,
                            float output_scale,
                            int32_t output_zero_point,
                            int8_t* output)
    {
        const uint32_t out_h = nk_op_detail::CalcOutputDimAsymmetric(
            in_h, pool_h, stride, pad_h, pad_h_end);
        const uint32_t out_w = nk_op_detail::CalcOutputDimAsymmetric(
            in_w, pool_w, stride, pad_w, pad_w_end);

        int32_t multiplier = 0;
        int32_t shift = 0;
        const bool same_scale =
            input_scale == output_scale && input_zero_point == output_zero_point;
        if (!same_scale)
        {
            if (input_scale <= 0.0f || output_scale <= 0.0f)
                return;
            QuantInteger::QuantizeMultiplier(static_cast<double>(input_scale) /
                                                 static_cast<double>(output_scale),
                                             &multiplier,
                                             &shift);
        }

        for (uint32_t oh = 0; oh < out_h; ++oh)
        {
            for (uint32_t ow = 0; ow < out_w; ++ow)
            {
                for (uint32_t c = 0; c < in_c; ++c)
                {
                    int32_t sum = 0;
                    uint32_t count = 0;

                    for (int kh = 0; kh < pool_h; ++kh)
                    {
                        const int32_t ih = static_cast<int32_t>(oh) * stride + kh - pad_h;
                        if (ih < 0 || ih >= static_cast<int32_t>(in_h))
                            continue;

                        for (int kw = 0; kw < pool_w; ++kw)
                        {
                            const int32_t iw = static_cast<int32_t>(ow) * stride + kw - pad_w;
                            if (iw < 0 || iw >= static_cast<int32_t>(in_w))
                                continue;

                            const int8_t value = input[(static_cast<uint32_t>(ih) * in_w +
                                                        static_cast<uint32_t>(iw)) *
                                                           in_c +
                                                       c];
                            sum += static_cast<int32_t>(value) - input_zero_point;
                            ++count;
                        }
                    }

                    int32_t avg = 0;
                    if (count > 0)
                    {
                        // Rounding divide in int32 (same sign as TFLite average_pool).
                        const int32_t abs_sum = sum >= 0 ? sum : -sum;
                        int32_t q = (abs_sum + static_cast<int32_t>(count / 2)) /
                                    static_cast<int32_t>(count);
                        avg = sum >= 0 ? q : -q;
                    }

                    int32_t out_q;
                    if (same_scale)
                        out_q = avg + output_zero_point;
                    else
                        out_q = QuantInteger::MultiplyByQuantizedMultiplier(avg, multiplier, shift) +
                                output_zero_point;
                    out_q = std::clamp(out_q, int32_t{-128}, int32_t{127});
                    output[(oh * out_w + ow) * in_c + c] = static_cast<int8_t>(out_q);
                }
            }
        }
    }

    void FlattenNhwcInt8(const int8_t* input, uint32_t num_elements, int8_t* output)
    {
        CmsisQuantUtil::CopyInt8(input, output, num_elements);
    }
}
