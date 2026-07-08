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

inline float dot_contiguous(const float* a, const float* b, uint32_t count)
{
    // Four independent accumulators break the serial FMA dependency chain: the FPU
    // can keep several multiply-adds in flight instead of stalling on each one's
    // latency, turning this reduction from latency-bound into throughput-bound.
    // Kept always-on (not behind NETKIT_LOOP_UNROLL) because it is the hot path for
    // the fully-connected/matmul kernels and the extra code size is a few instructions.
    float s0 = 0.0f;
    float s1 = 0.0f;
    float s2 = 0.0f;
    float s3 = 0.0f;
    uint32_t t = 0;
    for (; t + 4u <= count; t += 4u)
    {
        s0 += a[t] * b[t];
        s1 += a[t + 1u] * b[t + 1u];
        s2 += a[t + 2u] * b[t + 2u];
        s3 += a[t + 3u] * b[t + 3u];
    }
    float sum = (s0 + s1) + (s2 + s3);
    for (; t < count; ++t)
        sum += a[t] * b[t];
    return sum;
}

inline float dot_strided(const float* a,
                         uint32_t a_stride,
                         const float* b,
                         uint32_t b_stride,
                         uint32_t count)
{
    // Four independent accumulators break the serial FMA dependency chain (see dot_contiguous).
    float s0 = 0.0f;
    float s1 = 0.0f;
    float s2 = 0.0f;
    float s3 = 0.0f;
    uint32_t t = 0;
    for (; t + 4u <= count; t += 4u)
    {
        s0 += a[t * a_stride] * b[t * b_stride];
        s1 += a[(t + 1u) * a_stride] * b[(t + 1u) * b_stride];
        s2 += a[(t + 2u) * a_stride] * b[(t + 2u) * b_stride];
        s3 += a[(t + 3u) * a_stride] * b[(t + 3u) * b_stride];
    }
    float sum = (s0 + s1) + (s2 + s3);
    for (; t < count; ++t)
        sum += a[t * a_stride] * b[t * b_stride];
    return sum;
}

inline float dot_strided_b_offset(const float* a,
                                  uint32_t a_stride,
                                  const float* b,
                                  uint32_t b_stride,
                                  uint32_t b_col_offset,
                                  uint32_t count)
{
    // Four independent accumulators break the serial FMA dependency chain (see dot_contiguous).
    float s0 = 0.0f;
    float s1 = 0.0f;
    float s2 = 0.0f;
    float s3 = 0.0f;
    uint32_t t = 0;
    for (; t + 4u <= count; t += 4u)
    {
        s0 += a[t * a_stride] * b[t * b_stride + b_col_offset];
        s1 += a[(t + 1u) * a_stride] * b[(t + 1u) * b_stride + b_col_offset];
        s2 += a[(t + 2u) * a_stride] * b[(t + 2u) * b_stride + b_col_offset];
        s3 += a[(t + 3u) * a_stride] * b[(t + 3u) * b_stride + b_col_offset];
    }
    float sum = (s0 + s1) + (s2 + s3);
    for (; t < count; ++t)
        sum += a[t * a_stride] * b[t * b_stride + b_col_offset];
    return sum;
}
}
