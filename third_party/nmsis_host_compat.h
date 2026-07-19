/*
 * Host-smoke helpers for NMSIS-NN when compiling under __GNUC_PYTHON__
 * (no nmsis_core.h / RISC-V toolchain). Mirrors the portable CMSIS-NN host path.
 */
#ifndef NETKIT_NMSIS_HOST_COMPAT_H
#define NETKIT_NMSIS_HOST_COMPAT_H

#include <stdint.h>

#ifndef __RESTRICT
#define __RESTRICT restrict
#endif

#ifndef __ASM
#define __ASM __asm
#endif

/* Saturate to signed sat-bit range (CMSIS-style). */
#ifndef __SSAT
static inline int32_t __SSAT(int32_t val, uint32_t sat)
{
    if (sat >= 1u && sat <= 32u)
    {
        const int32_t max = (int32_t)((1u << (sat - 1u)) - 1u);
        const int32_t min = -1 - max;
        if (val > max)
            return max;
        if (val < min)
            return min;
    }
    return val;
}
#endif

/* Prefer memcpy-based unaligned helpers on host (see riscv_nnsupportfunctions.h). */
#ifndef __RISCV_FEATURE_UNALIGNED
#define __RISCV_FEATURE_UNALIGNED 1
#endif

#ifndef __CLZ
static inline uint32_t __CLZ(uint32_t value)
{
#if defined(__GNUC__) || defined(__clang__)
    return value ? (uint32_t)__builtin_clz(value) : 32u;
#else
    uint32_t n = 32u;
    if (value == 0u)
        return n;
    if (value >= 0x00010000u)
    {
        n -= 16u;
        value >>= 16u;
    }
    if (value >= 0x00000100u)
    {
        n -= 8u;
        value >>= 8u;
    }
    if (value >= 0x00000010u)
    {
        n -= 4u;
        value >>= 4u;
    }
    if (value >= 0x00000004u)
    {
        n -= 2u;
        value >>= 2u;
    }
    return n - (value >> 1u);
#endif
}
#endif

#endif /* NETKIT_NMSIS_HOST_COMPAT_H */
