#pragma once

// Integer quantization helpers shared by reference int8 kernels.
// Matches TFLite / CMSIS-NN QuantizeMultiplier + MultiplyByQuantizedMultiplier
// so int8→int8 layers never round-trip through float.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace QuantInteger
{
inline void QuantizeMultiplier(double double_multiplier, int32_t* multiplier, int32_t* shift)
{
    if (double_multiplier <= 0.0)
    {
        *multiplier = 0;
        *shift = 0;
        return;
    }

    int shift_val = 0;
    const double q = std::frexp(double_multiplier, &shift_val);
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

inline int32_t SaturatingRoundingDoublingHighMul(int32_t a, int32_t b)
{
    const bool overflow = a == b && a == std::numeric_limits<int32_t>::min();
    const int64_t ab_64 = static_cast<int64_t>(a) * static_cast<int64_t>(b);
    const int32_t nudge = ab_64 >= 0 ? (1 << 30) : (1 - (1 << 30));
    const int32_t ab_x2_high32 =
        static_cast<int32_t>((ab_64 + static_cast<int64_t>(nudge)) / (1LL << 31));
    return overflow ? std::numeric_limits<int32_t>::max() : ab_x2_high32;
}

inline int32_t RoundingDivideByPOT(int32_t x, int exponent)
{
    if (exponent == 0)
        return x;
    const int32_t mask = (1 << exponent) - 1;
    const int32_t remainder = x & mask;
    const int32_t threshold = (mask >> 1) + ((x < 0) ? 1 : 0);
    return (x >> exponent) + ((remainder > threshold) ? 1 : 0);
}

// result ≈ x * multiplier * 2^(shift - 31)
inline int32_t MultiplyByQuantizedMultiplier(int32_t x, int32_t quantized_multiplier, int shift)
{
    const int left_shift = shift > 0 ? shift : 0;
    const int right_shift = shift > 0 ? 0 : -shift;
    return RoundingDivideByPOT(
        SaturatingRoundingDoublingHighMul(x * (1 << left_shift), quantized_multiplier),
        right_shift);
}

// Fused int8 output clamp (CMSIS-NN style). ReLU6 uses quantized float 6.0.
enum class QuantClamp : uint8_t
{
    None = 0,
    ReLU,
    ReLU6,
};

inline void QuantClampRange(QuantClamp clamp,
                            float output_scale,
                            int32_t output_zero_point,
                            int32_t* act_min,
                            int32_t* act_max)
{
    *act_min = -128;
    *act_max = 127;
    if (clamp == QuantClamp::None)
        return;
    if (clamp == QuantClamp::ReLU)
    {
        *act_min = 0;
        *act_max = 127;
        return;
    }
    // ReLU6: clamp to [0, quantize(6.0)]
    *act_min = 0;
    if (output_scale <= 0.0f)
    {
        *act_max = 127;
        return;
    }
    const int32_t q6 =
        static_cast<int32_t>(std::llround(6.0 / static_cast<double>(output_scale))) +
        output_zero_point;
    *act_max = std::clamp(q6, int32_t{0}, int32_t{127});
}

// int32 accumulator → int8 via (input_scale * weight_scale / output_scale).
inline int8_t RequantizeAccToInt8(int32_t acc,
                                  int32_t multiplier,
                                  int shift,
                                  int32_t output_zero_point,
                                  QuantClamp clamp,
                                  float output_scale)
{
    int32_t q = MultiplyByQuantizedMultiplier(acc, multiplier, shift);
    q += output_zero_point;
    int32_t act_min = -128;
    int32_t act_max = 127;
    QuantClampRange(clamp, output_scale, output_zero_point, &act_min, &act_max);
    q = std::clamp(q, act_min, act_max);
    return static_cast<int8_t>(q);
}

inline int8_t RequantizeAccToInt8(int32_t acc,
                                  int32_t multiplier,
                                  int shift,
                                  int32_t output_zero_point,
                                  bool apply_relu)
{
    return RequantizeAccToInt8(acc,
                               multiplier,
                               shift,
                               output_zero_point,
                               apply_relu ? QuantClamp::ReLU : QuantClamp::None,
                               1.0f);
}

inline bool EffectiveScaleMultiplier(float input_scale,
                                     float weight_scale,
                                     float output_scale,
                                     int32_t* multiplier,
                                     int32_t* shift)
{
    if (output_scale <= 0.0f || input_scale <= 0.0f || weight_scale <= 0.0f)
        return false;
    const double effective = static_cast<double>(input_scale) * static_cast<double>(weight_scale) /
                             static_cast<double>(output_scale);
    QuantizeMultiplier(effective, multiplier, shift);
    return true;
}
}  // namespace QuantInteger
