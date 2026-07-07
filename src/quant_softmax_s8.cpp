#include "quant_output.hpp"

#include "cmsis_nn_quant.hpp"
#include "cmsis_dsp_util.hpp"
#include "quant_trace.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

#if defined(NETKIT_USE_CMSIS_SOFTMAX_S8) && NETKIT_USE_CMSIS_SOFTMAX_S8
extern "C" {
#include "arm_nnsupportfunctions.h"
}
#endif

#if defined(NETKIT_USE_CMSIS_NN) && NETKIT_USE_CMSIS_NN && NETKIT_CMSIS_NN_ALLOWED
#include <arm_nnfunctions.h>
#endif

namespace
{
    constexpr int kScaledDiffIntegerBits = 5;
    constexpr int kAccumBits = 12;
    constexpr int32_t kQ7Min = -128;
    constexpr int32_t kQ7Max = 127;

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

    int32_t DoublingHighMult(int32_t m1, int32_t m2)
    {
        int64_t mult = 1LL << 30;
        if ((m1 < 0) ^ (m2 < 0))
            mult = 1 - mult;
        mult = mult + static_cast<int64_t>(m1) * m2;
        return static_cast<int32_t>(mult / (1LL << 31));
    }

    int32_t DivideByPowerOfTwo(int32_t dividend, int32_t exponent)
    {
        const int32_t shift = -exponent;
        const int32_t fixup = (dividend & shift) >> 31;
        const int32_t fixed = dividend + fixup;
        return (fixed + (1 << (-shift - 1))) >> (-shift);
    }

    int32_t ExpOnNegativeValues(int32_t val)
    {
        int32_t shift = 24;
        int32_t val_mod = (val & ((1 << shift) - 1)) - (1 << shift);
        int32_t remainder = val_mod - val;
        int32_t x = (val_mod << 5) + (1 << 28);
        int32_t x2 = DoublingHighMult(x, x);
        int32_t op_1 = DivideByPowerOfTwo(DoublingHighMult(x2, x2), 2) + DoublingHighMult(x2, x);
        int32_t op_2 = x + DivideByPowerOfTwo(DoublingHighMult(op_1, 715827883) + x2, 1);
        int32_t result = 1895147668 + DoublingHighMult(1895147668, op_2);

        const int32_t selectors[] = {1672461947, 1302514674, 790015084, 290630308, 39332535, 720401, 242};
        for (int32_t sel : selectors)
        {
            if ((remainder & (1 << shift++)) != 0)
                result = DoublingHighMult(result, sel);
        }

        if (val == 0)
            result = 0x7FFFFFFF;
        return result;
    }

    int32_t OneOverOnePlusX(int32_t val)
    {
        const int64_t sum = static_cast<int64_t>(val) + (1LL << 31);
        if (sum <= (1LL << 31))
            return 0x7FFFFFFF;
        return static_cast<int32_t>(((1LL << 62) + (sum >> 1)) / sum);
    }

    uint32_t Clz32(uint32_t value)
    {
        if (value == 0)
            return 32;
#if defined(__GNUC__) || defined(__clang__)
        return static_cast<uint32_t>(__builtin_clz(value));
#else
        uint32_t count = 0;
        while ((value & 0x80000000u) == 0)
        {
            value <<= 1;
            ++count;
        }
        return count;
#endif
    }

    int32_t CalculateInputRadius(int input_integer_bits, int input_left_shift)
    {
        const double max_input_rescaled =
            255.0 * static_cast<double>(1u << (input_left_shift + input_integer_bits));
        const double inverse_scaling_factor = 255.0 / max_input_rescaled;
        const double max_input = 255.0 / inverse_scaling_factor;
        return static_cast<int32_t>(std::llround(max_input));
    }

    void ReferenceSoftmaxS8(const int8_t* input,
                            uint32_t num_rows,
                            uint32_t row_size,
                            const QuantOps::SoftmaxS8Params& params,
                            int8_t* output)
    {
        const int32_t mask = (1 << params.shift);

        for (uint32_t row = 0; row < num_rows; ++row)
        {
            int8_t max_val = input[0];
            for (uint32_t col = 1; col < row_size; ++col)
                max_val = std::max(max_val, input[col]);

            int32_t sum = 0;
            for (uint32_t col = 0; col < row_size; ++col)
            {
                const int32_t diff = static_cast<int32_t>(input[col]) - static_cast<int32_t>(max_val);
                if (diff >= params.diff_min)
                    sum += DivideByPowerOfTwo(
                        ExpOnNegativeValues(DoublingHighMult(diff * mask, params.mult)), kAccumBits);
            }

            const int32_t headroom = static_cast<int32_t>(Clz32(static_cast<uint32_t>(sum)));
            const int32_t shifted_scale =
                OneOverOnePlusX((sum > 0 ? sum << headroom : 0) - (1 << 31));
            const int32_t bits_over_unit = kAccumBits - headroom + 23;

            for (uint32_t col = 0; col < row_size; ++col)
            {
                const int32_t diff = static_cast<int32_t>(input[col]) - static_cast<int32_t>(max_val);
                if (diff >= params.diff_min)
                {
                    const int32_t res =
                        DivideByPowerOfTwo(
                            DoublingHighMult(shifted_scale,
                                             ExpOnNegativeValues(DoublingHighMult(diff * mask, params.mult))),
                            bits_over_unit) +
                        kQ7Min;
                    output[col] = static_cast<int8_t>(std::clamp(res, kQ7Min, kQ7Max));
                }
                else
                {
                    output[col] = static_cast<int8_t>(kQ7Min);
                }
            }

            input += row_size;
            output += row_size;
        }
    }
}

namespace QuantOps
{
    SoftmaxS8Params ComputeSoftmaxS8Params(float logit_scale, float beta)
    {
        SoftmaxS8Params params{};
        if (logit_scale <= 0.0f)
            return params;

        const double min_real_multiplier = std::min(
            static_cast<double>(beta) * static_cast<double>(logit_scale) *
                static_cast<double>(1LL << (31 - kScaledDiffIntegerBits)),
            static_cast<double>((1LL << 31) - 1));
        QuantizeMultiplier(min_real_multiplier, &params.mult, &params.shift);
        params.diff_min = -CalculateInputRadius(kScaledDiffIntegerBits, params.shift);
        return params;
    }

    void SoftmaxS8(const int8_t* logits, uint32_t count, float logit_scale, int8_t* output)
    {
        if (!logits || !output || count == 0)
            return;

        const float scale = logit_scale > 0.0f ? logit_scale : 1.0f;
#if defined(NETKIT_USE_CMSIS_NN) && NETKIT_USE_CMSIS_NN && NETKIT_CMSIS_NN_ALLOWED
        if (CmsisNnQuant::TrySoftmaxS8(logits, 1, count, scale, output))
        {
            QuantTrace::RecordSoftmaxCmsisOk();
            return;
        }
#endif
#if defined(NETKIT_USE_CMSIS_SOFTMAX_S8) && NETKIT_USE_CMSIS_SOFTMAX_S8
        {
            const SoftmaxS8Params params = ComputeSoftmaxS8Params(scale);
            arm_nn_softmax_common_s8(logits,
                                     1,
                                     static_cast<int32_t>(count),
                                     params.mult,
                                     params.shift,
                                     params.diff_min,
                                     false,
                                     output);
            QuantTrace::RecordSoftmaxCmsisOk();
            return;
        }
#endif
        const SoftmaxS8Params params = ComputeSoftmaxS8Params(scale);
        ReferenceSoftmaxS8(logits, 1, count, params, output);
        QuantTrace::RecordSoftmaxReference();
    }

    float DequantizeSoftmaxOutput(int8_t value)
    {
        return (static_cast<float>(value) - static_cast<float>(kSoftmaxOutputZeroPoint)) * kSoftmaxOutputScale;
    }

    void DequantizeSoftmaxOutput(const int8_t* src, float* dst, uint32_t count)
    {
        for (uint32_t i = 0; i < count; ++i)
            dst[i] = DequantizeSoftmaxOutput(src[i]);
    }

    uint32_t ArgMaxInt8(const int8_t* values, uint32_t count)
    {
        return CmsisQuantUtil::ArgMaxInt8(values, count);
    }
}

#if defined(NETKIT_USE_CMSIS_NN) && NETKIT_USE_CMSIS_NN && NETKIT_CMSIS_NN_ALLOWED

namespace CmsisNnQuant
{

bool TrySoftmaxS8(const int8_t* input,
                    uint32_t num_rows,
                    uint32_t row_size,
                    float logit_scale,
                    int8_t* output)
{
    if (!input || !output || num_rows == 0 || row_size == 0)
        return false;

    const QuantOps::SoftmaxS8Params params = QuantOps::ComputeSoftmaxS8Params(logit_scale);
    arm_softmax_s8(input,
                   static_cast<int32_t>(num_rows),
                   static_cast<int32_t>(row_size),
                   params.mult,
                   params.shift,
                   params.diff_min,
                   output);
    return true;
}

}  // namespace CmsisNnQuant

#else

namespace CmsisNnQuant
{

bool TrySoftmaxS8(const int8_t* /*input*/,
                    uint32_t /*num_rows*/,
                    uint32_t /*row_size*/,
                    float /*logit_scale*/,
                    int8_t* /*output*/)
{
    return false;
}

}  // namespace CmsisNnQuant

#endif
