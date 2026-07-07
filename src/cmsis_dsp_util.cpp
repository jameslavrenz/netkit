#include "cmsis_dsp_util.hpp"

#include "netkit_config.h"
#include "netkit_loop_unroll.hpp"

#include <cstring>

#if defined(NETKIT_USE_CMSIS_DSP) && NETKIT_USE_CMSIS_DSP
#include <arm_math.h>
#endif

namespace CmsisDspUtil
{
void CopyInt8(const int8_t* src, int8_t* dst, uint32_t count)
{
    if (!src || !dst || count == 0)
        return;

#if defined(NETKIT_USE_CMSIS_DSP) && NETKIT_USE_CMSIS_DSP
    arm_copy_q7(reinterpret_cast<const q7_t*>(src), reinterpret_cast<q7_t*>(dst), count);
#else
    std::memcpy(dst, src, static_cast<std::size_t>(count));
#endif
}

void CopyF32(const float* src, float* dst, uint32_t count)
{
    if (!src || !dst || count == 0)
        return;

#if defined(NETKIT_USE_CMSIS_DSP) && NETKIT_USE_CMSIS_DSP
    arm_copy_f32(src, dst, count);
#else
    std::memcpy(dst, src, static_cast<std::size_t>(count) * sizeof(float));
#endif
}

uint32_t ArgMaxInt8(const int8_t* values, uint32_t count)
{
    if (!values || count == 0)
        return 0;
    if (count == 1)
        return 0;

#if defined(NETKIT_USE_CMSIS_DSP) && NETKIT_USE_CMSIS_DSP
    q7_t max_val = 0;
    uint32_t max_index = 0;
    arm_max_q7(reinterpret_cast<const q7_t*>(values), count, &max_val, &max_index);
    return max_index;
#else
    uint32_t best = 0;
    for (uint32_t i = 1; i < count; ++i)
    {
        if (values[i] > values[best])
            best = i;
    }
    return best;
#endif
}

uint32_t ArgMaxF32(const float* values, uint32_t count)
{
    if (!values || count == 0)
        return 0;
    if (count == 1)
        return 0;

#if defined(NETKIT_USE_CMSIS_DSP) && NETKIT_USE_CMSIS_DSP
    float32_t max_val = 0.0f;
    uint32_t max_index = 0;
    arm_max_f32(values, count, &max_val, &max_index);
    return max_index;
#else
    uint32_t best = 0;
    for (uint32_t i = 1; i < count; ++i)
    {
        if (values[i] > values[best])
            best = i;
    }
    return best;
#endif
}

float DotProductF32(const float* a, const float* b, uint32_t count)
{
#if defined(NETKIT_USE_CMSIS_DSP) && NETKIT_USE_CMSIS_DSP
    float sum = 0.0f;
    arm_dot_prod_f32(const_cast<float*>(a), const_cast<float*>(b), count, &sum);
    return sum;
#else
    return NetkitLoopUnroll::dot_contiguous(a, b, count);
#endif
}

void MulF32(const float* a, const float* b, float* c, uint32_t count)
{
#if defined(NETKIT_USE_CMSIS_DSP) && NETKIT_USE_CMSIS_DSP
    arm_mult_f32(a, b, c, count);
#else
    NetkitLoopUnroll::mul_contiguous(a, b, c, count);
#endif
}

void AddF32(const float* a, const float* b, float* c, uint32_t count)
{
#if defined(NETKIT_USE_CMSIS_DSP) && NETKIT_USE_CMSIS_DSP
    arm_add_f32(a, b, c, count);
#else
    NetkitLoopUnroll::add_contiguous(a, b, c, count);
#endif
}

void MulScalarF32(const float* a, float scalar, float* c, uint32_t count)
{
#if defined(NETKIT_USE_CMSIS_DSP) && NETKIT_USE_CMSIS_DSP
    arm_scale_f32(a, scalar, c, count);
#else
    NetkitLoopUnroll::mul_scalar_contiguous(a, scalar, c, count);
#endif
}

void ScaleF32(float* c, float scalar, uint32_t count)
{
#if defined(NETKIT_USE_CMSIS_DSP) && NETKIT_USE_CMSIS_DSP
    arm_scale_f32(c, scalar, c, count);
#else
    NetkitLoopUnroll::scale_contiguous(c, scalar, count);
#endif
}
}  // namespace CmsisDspUtil
