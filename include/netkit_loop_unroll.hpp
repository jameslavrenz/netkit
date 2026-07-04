#pragma once

#include "netkit_config.h"
#include <cstdint>

// 4× manual loop unroll for netkit reference kernels only (not CMSIS).
// EXPERIMENTAL — off by default (make NETKIT_LOOP_UNROLL=1). Duplicates loop bodies and
// increases .text size; on flash-limited MCUs the firmware image may no longer fit.

namespace NetkitLoopUnroll
{
#if NETKIT_LOOP_UNROLL
inline constexpr bool kEnabled = true;
#else
inline constexpr bool kEnabled = false;
#endif

template<typename Fn>
inline void for_count(uint32_t count, Fn&& body)
{
#if NETKIT_LOOP_UNROLL
    uint32_t i = 0;
    for (; i + 4u <= count; i += 4u)
    {
        body(i);
        body(i + 1u);
        body(i + 2u);
        body(i + 3u);
    }
    for (; i < count; ++i)
        body(i);
#else
    for (uint32_t i = 0; i < count; ++i)
        body(i);
#endif
}

inline void mul_contiguous(const float* a, const float* b, float* c, uint32_t count)
{
    for_count(count, [&](uint32_t i) { c[i] = a[i] * b[i]; });
}

inline void add_contiguous(const float* a, const float* b, float* c, uint32_t count)
{
    for_count(count, [&](uint32_t i) { c[i] = a[i] + b[i]; });
}

inline void mul_scalar_contiguous(const float* a, float scalar, float* c, uint32_t count)
{
    for_count(count, [&](uint32_t i) { c[i] = a[i] * scalar; });
}

inline void scale_contiguous(float* c, float scalar, uint32_t count)
{
    for_count(count, [&](uint32_t i) { c[i] *= scalar; });
}

inline float dot_strided(const float* a,
                         uint32_t a_stride,
                         const float* b,
                         uint32_t b_stride,
                         uint32_t count)
{
    float sum = 0.0f;
#if NETKIT_LOOP_UNROLL
    uint32_t t = 0;
    for (; t + 4u <= count; t += 4u)
    {
        sum += a[t * a_stride] * b[t * b_stride];
        sum += a[(t + 1u) * a_stride] * b[(t + 1u) * b_stride];
        sum += a[(t + 2u) * a_stride] * b[(t + 2u) * b_stride];
        sum += a[(t + 3u) * a_stride] * b[(t + 3u) * b_stride];
    }
    for (; t < count; ++t)
        sum += a[t * a_stride] * b[t * b_stride];
#else
    for (uint32_t t = 0; t < count; ++t)
        sum += a[t * a_stride] * b[t * b_stride];
#endif
    return sum;
}

inline float dot_strided_b_offset(const float* a,
                                  uint32_t a_stride,
                                  const float* b,
                                  uint32_t b_stride,
                                  uint32_t b_col_offset,
                                  uint32_t count)
{
    float sum = 0.0f;
#if NETKIT_LOOP_UNROLL
    uint32_t t = 0;
    for (; t + 4u <= count; t += 4u)
    {
        sum += a[t * a_stride] * b[t * b_stride + b_col_offset];
        sum += a[(t + 1u) * a_stride] * b[(t + 1u) * b_stride + b_col_offset];
        sum += a[(t + 2u) * a_stride] * b[(t + 2u) * b_stride + b_col_offset];
        sum += a[(t + 3u) * a_stride] * b[(t + 3u) * b_stride + b_col_offset];
    }
    for (; t < count; ++t)
        sum += a[t * a_stride] * b[t * b_stride + b_col_offset];
#else
    for (uint32_t t = 0; t < count; ++t)
        sum += a[t * a_stride] * b[t * b_stride + b_col_offset];
#endif
    return sum;
}
}
