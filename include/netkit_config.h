/*
 * netkit_config.h — compile-time build target selection (C and C++).
 *
 * Set exactly one target via Makefile NETKIT_TARGET=cpu|mcu|mpu or -D flags:
 *   NETKIT_TARGET_CPU  — desktop dev/test (CLI, regression, debug tooling)
 *   NETKIT_TARGET_MCU  — lean embedded runtime (.nk load + inference only)
 *   NETKIT_TARGET_MPU  — lean embedded runtime (.nk load + inference only)
 *
 * Arena static defaults (NK_ARENA_DEFAULT_CAPACITY / Arena::kDefaultCapacity):
 *   CPU — 4 MiB   MCU — 64 KiB   MPU — 128 KiB
 *
 * Arena backing (override via Makefile):
 *   CPU default  — one heap malloc per session; free when session ends (CLI command or test suite).
 *                  Set NETKIT_GLOBAL_ARENA=1 for static/global only (no heap).
 *   MCU / MPU default — caller-owned static/global buffer via nk_arena_init().
 *                       NETKIT_HEAP_ARENA=1 allows one init_heap() at startup; never freed.
 *
 * Optional weight storage (Makefile / CMake):
 *   NETKIT_WEIGHTS_IN_RAM — when 1, buffer/AOT load copies weight and bias payload into
 *     the arena (SRAM). When 0, weights stay in the .nk blob (flash/XIP); arena holds
 *     activations and network structs only.
 *     Default 0 on MCU (flash-backed), 1 on CPU and MPU. Override with NETKIT_WEIGHTS_IN_RAM=0|1.
 *     File-based load (LoadMLP path) always copies payload into the arena.
 *
 * Optional kernel backends (Makefile / CMake):
 *   NETKIT_USE_CMSIS_NN  — ARM CMSIS-NN when NETKIT_TARGET_MCU + Cortex-M NETKIT_ARCH (flag ignored elsewhere)
 *   NETKIT_USE_CMSIS_DSP — ARM CMSIS-DSP float32 vector/matrix ops + clip
 *   On MCU with both enabled, NN owns overlapping layer ops; DSP is not a fallback.
 *   On desktop and MPU, NETKIT_CMSIS_NN=1 is ignored (warning) — reference kernels and optional CMSIS-DSP apply.
 *
 * Target architecture (Makefile NETKIT_ARCH=... / CMake -DNETKIT_ARCH=...):
 *   Maps to CMSIS ARM_MATH_* flags (CM0–M85, A32, NEON). Unset = desktop host.
 *   Also sets ARM_MATH_LOOPUNROLL (DSP), __DSP_PRESENT (M33), MVEF/MVEI (M55/M85).
 *
 * See docs/BUILD_TARGETS.md.
 */
#ifndef NETKIT_CONFIG_H
#define NETKIT_CONFIG_H

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
#if defined(NETKIT_TARGET_MPU)
#define NK_ARENA_DEFAULT_CAPACITY (128U * 1024U)
#elif defined(NETKIT_TARGET_MCU)
#define NK_ARENA_DEFAULT_CAPACITY (64U * 1024U)
#else
#define NK_ARENA_DEFAULT_CAPACITY (4U * 1024U * 1024U)
#endif

/* Weight load policy: copy into arena (SRAM) vs use .nk blob in flash. */
#ifndef NETKIT_WEIGHTS_IN_RAM
#if defined(NETKIT_TARGET_MCU)
#define NETKIT_WEIGHTS_IN_RAM 0
#else
#define NETKIT_WEIGHTS_IN_RAM 1
#endif
#endif

#if NETKIT_WEIGHTS_IN_RAM != 0 && NETKIT_WEIGHTS_IN_RAM != 1
#error "NETKIT_WEIGHTS_IN_RAM must be 0 or 1"
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

#endif /* NETKIT_CONFIG_H */
