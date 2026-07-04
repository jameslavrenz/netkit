#pragma once

#include "ops_resolver.hpp"
#include "tensor.hpp"
#include "tensor_access.hpp"
#include "nk_format.hpp"
#include <cstdint>
#include <cstring>
#include <vector>

namespace nk_op_detail
{
    inline uint32_t CalcOutputDimAsymmetric(uint32_t input_dim,
                                            int kernel_size,
                                            int stride,
                                            int pad_before,
                                            int pad_after)
    {
        return static_cast<uint32_t>(
            (static_cast<int>(input_dim) + pad_before + pad_after - kernel_size) / stride + 1);
    }

    inline uint32_t CalcOutputDim(uint32_t input_dim, int kernel_size, int stride, int pad = 0)
    {
        return CalcOutputDimAsymmetric(input_dim, kernel_size, stride, pad, pad);
    }

    inline void DecodeConvPadExtra(uint8_t pad_h,
                                   uint8_t pad_w,
                                   uint8_t pad_extra,
                                   int& pad_h_end,
                                   int& pad_w_end)
    {
        pad_h_end = static_cast<int>(pad_h) + static_cast<int>(pad_extra & 0xFU);
        pad_w_end = static_cast<int>(pad_w) + static_cast<int>((pad_extra >> 4) & 0xFU);
    }

    struct DepthwiseMeta
    {
        uint32_t kernel_h;
        uint32_t kernel_w;
        int pad_h_end;
        int pad_w_end;
    };

    inline DepthwiseMeta DecodeDepthwiseMeta(const NkFormat::ConvLayerDesc& layer,
                                             std::size_t weight_elems = 0,
                                             std::size_t channels = 0)
    {
        const uint32_t kernel_h = NkFormat::DepthwiseKernelH(layer);
        const uint8_t kw_byte = layer.kernel_w;
        const int pad_h = static_cast<int>(layer.pad_h);
        const int pad_w = static_cast<int>(layer.pad_w);

        auto pad_from_byte = [&](uint8_t byte) -> DepthwiseMeta {
            DepthwiseMeta meta{};
            meta.kernel_h = kernel_h;
            meta.kernel_w = kernel_h;
            DecodeConvPadExtra(layer.pad_h, layer.pad_w, byte, meta.pad_h_end, meta.pad_w_end);
            return meta;
        };

        auto literal = [&](uint32_t kw) -> DepthwiseMeta {
            return {kernel_h, kw, pad_h, pad_w};
        };

        if (weight_elems > 0 && channels > 0 && weight_elems % channels == 0)
        {
            const std::size_t kernel_area = weight_elems / channels;
            std::vector<DepthwiseMeta> candidates;
            if (kw_byte == kernel_h && static_cast<std::size_t>(kernel_h) * kernel_h == kernel_area)
                candidates.push_back(literal(kernel_h));
            if (static_cast<std::size_t>(kernel_h) * kernel_h == kernel_area && kw_byte != kernel_h)
                candidates.push_back(pad_from_byte(kw_byte));
            const uint32_t literal_kw = kw_byte ? kw_byte : kernel_h;
            if (static_cast<std::size_t>(kernel_h) * literal_kw == kernel_area)
                candidates.push_back(literal(literal_kw));
            if (!candidates.empty())
                return candidates.front();
        }

        if (kw_byte == kernel_h)
            return literal(kernel_h);
        const int extra_w = (static_cast<int>(kw_byte) >> 4) & 0xF;
        if (extra_w != 0)
            return pad_from_byte(kw_byte);
        const int extra_h = static_cast<int>(kw_byte) & 0xF;
        if (extra_h != 0 && kw_byte < kernel_h)
            return literal(kw_byte);
        if (extra_h != 0)
            return pad_from_byte(kw_byte);
        return literal(kw_byte ? kw_byte : kernel_h);
    }

    struct PoolMeta
    {
        int pool_h;
        int pool_w;
        int pad_h;
        int pad_w;
        int pad_h_end;
        int pad_w_end;
    };

    inline PoolMeta DecodePoolMeta(const NkFormat::PoolLayerDesc& layer)
    {
        const uint16_t reserved = layer.reserved;
        const int pool_h = static_cast<int>(layer.pool_size);
        const int pool_w = (reserved & 0xFFU) ? static_cast<int>(reserved & 0xFFU) : pool_h;
        const int extra_h = static_cast<int>((reserved >> 8) & 0xFU);
        const int extra_w = static_cast<int>((reserved >> 12) & 0xFU);
        const int pad_h = static_cast<int>(layer.pad_h);
        const int pad_w = static_cast<int>(layer.pad_w);
        return {pool_h, pool_w, pad_h, pad_w, pad_h + extra_h, pad_w + extra_w};
    }

    inline void FlattenNhwc(const Tensor& input, Tensor& output)
    {
        const float* in = tensor_data_f32(const_cast<Tensor&>(input));
        float* out = tensor_data_f32(output);
        std::memcpy(out, in, static_cast<std::size_t>(input.num_elements) * sizeof(float));
    }

    inline void BumpMaxActivation(NkCnnSpatialPlan& plan, uint32_t elements)
    {
        if (!plan.max_activation_elements)
            return;

        if (elements > *plan.max_activation_elements)
            *plan.max_activation_elements = elements;
    }
}
