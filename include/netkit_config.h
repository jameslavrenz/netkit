/*
 * netkit_config.h — compile-time build target selection (C and C++).
 *
 * Set exactly one target via Makefile NETKIT_TARGET=cpu|mcu|mpu or -D flags:
 *   NETKIT_TARGET_CPU  — desktop dev/test (CLI, regression, debug tooling)
 *   NETKIT_TARGET_MCU  — lean embedded runtime (.nk load + inference only)
 *   NETKIT_TARGET_MPU  — lean embedded runtime (.nk load + inference only)
 *
 * Arena static defaults (NK_ARENA_DEFAULT_CAPACITY / Arena::kDefaultCapacity):
 *   MCU — 64 KiB   CPU and MPU — 64 MiB
 *
 * Arena backing (override via Makefile):
 *   CPU default  — one heap malloc per session; free when session ends (CLI command or test suite).
 *                  Set NETKIT_GLOBAL_ARENA=1 for static/global only (no heap).
 *   MCU / MPU default — caller-owned static/global buffer via nk_arena_init().
 *                       NETKIT_HEAP_ARENA=1 allows one init_heap() at startup; never freed.
 *
 * Weights always stay in the .nk blob; the arena holds activations and structs.
 *   Preferred on MCU and RTOS/bare-metal MPU: flash/XIP or Load*FromBuffer (bind views).
 *   Optional POSIX mmap file load (NETKIT_USE_MMAP / Makefile NETKIT_MMAP):
 *     Default ON for CPU (macOS/Linux), OFF for MCU and MPU (RTOS-first).
 *     Opt in on embedded Linux MPU with NETKIT_MMAP=1. Arena owns the mapping.
 *
 * Optional kernel backends (Makefile / CMake — explicit NETKIT_CMSIS_*=1, profile defaults per target):
 *   NETKIT_USE_CMSIS_NN  — when NETKIT_CMSIS_NN=1 on MCU + Cortex-M NETKIT_ARCH
 *   NETKIT_USE_CMSIS_DSP — when NETKIT_CMSIS_DSP=1 (cpu/mpu/mcu)
 *   Profile defaults (Makefile): cpu=DSP only, mcu=DSP+NN, mpu=DSP only. Override with NETKIT_CMSIS_*=0.
 *   CMSIS-NN is ignored on cpu/mpu even if NETKIT_CMSIS_NN=1 (warning).
 *
 * Optional reference-kernel tuning (Makefile / CMake):
 *   NETKIT_IM2COL — float Conv2D execution strategy (single tri-state knob):
 *     0 = direct loops only (no im2col)
 *     1 = partial im2col on large layers, direct otherwise
 *     2 = full im2col + GEMM on large layers (partial/direct fallback for the rest)
 *     Default 0 (direct) on all targets (CPU/MCU/MPU). Direct convolution with the
 *     multi-accumulator dot is fastest for the small models we target; opt into
 *     im2col (1 or 2) explicitly per workload.
 *   NETKIT_REFERENCE_QUANT_LOOPS — When 1, int8 quantized forward uses netkit QuantOps
 *     reference loops (scalar conv/pool/FC + reference softmax) instead of CMSIS-NN kernels.
 *     Default 0 (CMSIS-NN). Used for MCU firmware profiling / kernel validation.
 *   NETKIT_LOOP_UNROLL — EXPERIMENTAL. When 1, 4× manual loop unroll in netkit reference
 *     kernels only (default 0). Increases .text/flash size; can exceed program memory on
 *     small MCUs. Independent of CMSIS-DSP ARM_MATH_LOOPUNROLL.
 *
 * Target architecture (Makefile NETKIT_ARCH=... / CMake -DNETKIT_ARCH=...):
 *   Maps to CMSIS ARM_MATH_* flags (CM0–M85, A32, NEON). Unset = desktop host.
 *   Also sets ARM_MATH_LOOPUNROLL (DSP), __DSP_PRESENT (M33), MVEF/MVEI (M55/M85).
 *
 * See docs/BUILD_TARGETS.md.
 */
#ifndef NETKIT_CONFIG_H
#define NETKIT_CONFIG_H

#ifndef NETKIT_KERNELS_OPTIMIZED_FOR_SPEED
#define NETKIT_KERNELS_OPTIMIZED_FOR_SPEED 0
#endif

#ifndef NETKIT_MCU_QUANT_ONLY
#define NETKIT_MCU_QUANT_ONLY 0
#endif

#if defined(NETKIT_TARGET_CPU) + defined(NETKIT_TARGET_MCU) + defined(NETKIT_TARGET_MPU) > 1
#error "Define only one of NETKIT_TARGET_CPU, NETKIT_TARGET_MCU, NETKIT_TARGET_MPU"
#endif

#if !defined(NETKIT_TARGET_CPU) && !defined(NETKIT_TARGET_MCU) && !defined(NETKIT_TARGET_MPU)
#define NETKIT_TARGET_CPU 1
#endif

#if defined(NETKIT_TARGET_CPU)
#define NETKIT_DESKTOP 1
#if !defined(NETKIT_GLOBAL_ARENA) || !NETKIT_GLOBAL_ARENA
#define NETKIT_ARENA_HEAP 1
#endif
#else
#if defined(NETKIT_HEAP_ARENA) && NETKIT_HEAP_ARENA
#define NETKIT_ARENA_HEAP 1
#endif
#endif

/* Static arena default for examples and NK_ARENA_DEFAULT_CAPACITY. */
#if defined(NETKIT_TARGET_MCU)
#define NK_ARENA_DEFAULT_CAPACITY (64U * 1024U)
#else
#define NK_ARENA_DEFAULT_CAPACITY (64U * 1024U * 1024U) /* CPU and MPU */
#endif

/*
 * float Conv2D execution strategy (single tri-state knob):
 *   0 = direct loops only, 1 = partial im2col, 2 = full im2col + GEMM.
 * Default 0 (direct) on all targets (CPU/MCU/MPU). Builds pass -DNETKIT_IM2COL=0
 * unless overridden; direct convolution with the multi-accumulator dot is fastest
 * for the small models we target.
 */
#ifndef NETKIT_IM2COL
#define NETKIT_IM2COL 0
#endif

#if NETKIT_IM2COL != 0 && NETKIT_IM2COL != 1 && NETKIT_IM2COL != 2
#error "NETKIT_IM2COL must be 0 (direct), 1 (partial), or 2 (full)"
#endif

#ifndef NETKIT_REFERENCE_QUANT_LOOPS
#define NETKIT_REFERENCE_QUANT_LOOPS 0
#endif

#if NETKIT_REFERENCE_QUANT_LOOPS != 0 && NETKIT_REFERENCE_QUANT_LOOPS != 1
#error "NETKIT_REFERENCE_QUANT_LOOPS must be 0 or 1"
#endif

#ifndef NETKIT_LOOP_UNROLL
#define NETKIT_LOOP_UNROLL 0
#endif

#if NETKIT_LOOP_UNROLL != 0 && NETKIT_LOOP_UNROLL != 1
#error "NETKIT_LOOP_UNROLL must be 0 or 1"
#endif

/* CMSIS-NN: Cortex-M MCU firmware only (not desktop CPU or Cortex-A MPU). */
#if defined(NETKIT_TARGET_MCU) &&                                                                 \
    (defined(ARM_MATH_CM0) || defined(ARM_MATH_CM0PLUS) || defined(ARM_MATH_CM3) ||               \
     defined(ARM_MATH_CM4) || defined(ARM_MATH_CM7) || defined(ARM_MATH_ARMV8MBL) ||               \
     defined(ARM_MATH_ARMV8MML) || defined(ARM_MATH_M55) || defined(ARM_MATH_M85))
#define NETKIT_CMSIS_NN_ALLOWED 1
#else
#define NETKIT_CMSIS_NN_ALLOWED 0
#endif

/* XNNPACK: host CPU and Cortex-A MPU only (not MCU). */
#if (defined(NETKIT_TARGET_CPU) || defined(NETKIT_TARGET_MPU)) && !defined(NETKIT_TARGET_MCU)
#define NETKIT_XNNPACK_ALLOWED 1
#else
#define NETKIT_XNNPACK_ALLOWED 0
#endif

/*
 * POSIX mmap for .nk file loads (macOS/Linux only at compile time).
 * Default: CPU on, MCU/MPU off (MPU often means RTOS/bare metal without mmap).
 * Override with -DNETKIT_USE_MMAP=0|1 or Makefile NETKIT_MMAP=0|1.
 * When off or unavailable, file load falls back to fread into the arena;
 * buffer/AOT load is unchanged (preferred for RTOS/firmware).
 */
#ifndef NETKIT_USE_MMAP
#if defined(NETKIT_TARGET_CPU) && (defined(__APPLE__) || defined(__linux__))
#define NETKIT_USE_MMAP 1
#else
#define NETKIT_USE_MMAP 0
#endif
#endif

#if NETKIT_USE_MMAP != 0 && NETKIT_USE_MMAP != 1
#error "NETKIT_USE_MMAP must be 0 or 1"
#endif

#endif /* NETKIT_CONFIG_H */
