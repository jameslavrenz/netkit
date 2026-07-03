#pragma once

#include <stdint.h>

/* Cortex-M4 DWT cycle counter (no CMSIS-Core device headers required). */
#define NETKIT_DWT_CTRL (*((volatile uint32_t*)0xE0001000u))
#define NETKIT_DWT_CYCCNT (*((volatile uint32_t*)0xE0001004u))
#define NETKIT_COREDEBUG_DEMCR (*((volatile uint32_t*)0xE000EDFCu))

static inline void netkit_dwt_enable(void)
{
    NETKIT_COREDEBUG_DEMCR |= (1u << 24); /* TRCENA */
    NETKIT_DWT_CYCCNT = 0u;
    NETKIT_DWT_CTRL |= 1u; /* CYCCNTENA */
}

static inline uint32_t netkit_dwt_cycles(void)
{
    return NETKIT_DWT_CYCCNT;
}
