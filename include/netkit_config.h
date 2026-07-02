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
 * Optional kernel backends (Makefile / CMake):
 *   NETKIT_USE_CMSIS_NN  — ARM CMSIS-NN float32 conv/pool/FC/activations/softmax/add
 *   NETKIT_USE_CMSIS_DSP — ARM CMSIS-DSP float32 vector/matrix ops + clip
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

#endif /* NETKIT_CONFIG_H */
