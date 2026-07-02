#pragma once

#include "ops_resolver.hpp"
#include "tensor.hpp"
#include "tensor_access.hpp"
#include <cstdint>
#include <cstring>

namespace nk_op_detail
{
    inline uint32_t CalcOutputDim(uint32_t input_dim, int kernel_size, int stride, int pad = 0)
    {
        return static_cast<uint32_t>((static_cast<int>(input_dim) + 2 * pad - kernel_size) / stride + 1);
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
