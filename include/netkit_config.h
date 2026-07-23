/*
 * netkit_config.h — compile-time build target selection (C and C++).
 *
 * Set exactly one target via Makefile/CMake NETKIT_TARGET=... or -D flags:
 *   NETKIT_TARGET_CPU       — desktop dev/test (CLI, regression, debug tooling)
 *   NETKIT_TARGET_MCU_ARM   — Arm microcontroller firmware (lean runtime)
 *   NETKIT_TARGET_MPU_ARM   — Arm microprocessor / RTOS (lean runtime)
 *   NETKIT_TARGET_MCU_RISC  — non-Espressif RISC-V MCU (NMSIS-NN int8 production)
 *   NETKIT_TARGET_MPU_RISC  — RISC-V MPU (lean runtime; XNNPACK)
 *   NETKIT_TARGET_MCU_ESP   — Espressif MCU firmware (ESP-NN int8; Xtensa and RISC-V
 *                             ESP32* — C3/C6/P4 stay here, not MCU_RISC)
 *
 * Derived class / ISA macros (set automatically from the target above):
 *   NETKIT_CLASS_MCU / NETKIT_CLASS_MPU — firmware class (arena / lean API)
 *   NETKIT_ISA_ARM / NETKIT_ISA_RISC / NETKIT_ISA_ESP — instruction-set family
 *
 * Backend profile defaults (Makefile / CMake; override with NETKIT_CMSIS_NN=0|1):
 *   cpu       — XNNPACK on (any host ISA), CMSIS-NN / ESP-NN / NMSIS-NN off
 *   mcu_arm   — CMSIS-NN on (int8 production), XNNPACK forbidden; float32 uses reference
 *   mpu_arm   — XNNPACK on, CMSIS-NN / ESP-NN / NMSIS-NN off
 *   mcu_risc  — NMSIS-NN on (int8 production); float32 uses reference; XNNPACK forbidden
 *   mpu_risc  — XNNPACK on (default); CMSIS-NN / ESP-NN / NMSIS-NN forbidden
 *   mcu_esp   — ESP-NN on (int8 production; float32 uses reference — ESP-NN is int8-only)
 *
 * XNNPACK policy: default ON for cpu and all MPU targets; never allowed on MCU.
 *
 * CMSIS-DSP is not used or linked. Float32 on MCU is supported via portable/reference
 * kernels (CMSIS-NN also offers float LayerFast on Arm; ESP-NN / NMSIS-NN have no float API).
 *
 * Arena static defaults (NK_ARENA_DEFAULT_CAPACITY / Arena::kDefaultCapacity):
 *   CLASS_MCU — 64 KiB   CPU and CLASS_MPU — 64 MiB
 *   Override on the compiler command line, e.g.
 *     -DNK_ARENA_DEFAULT_CAPACITY=131072
 *   or via Make/CMake: NETKIT_ARENA_CAPACITY=<bytes> / NETKIT_ARENA_KB=<KiB>.
 *
 * Arena backing (override via Makefile):
 *   CPU default  — one heap malloc per session; free when session ends.
 *                  Set NETKIT_GLOBAL_ARENA=1 for static/global only (no heap).
 *   MCU default  — caller-owned static/global buffer via nk_arena_init() only.
 *                  Heap is forbidden forever (no malloc/new/delete/free).
 *                  Weights stay in the flash .nk image; arena holds activations.
 *   MPU default  — caller-owned static/global buffer via nk_arena_init().
 *                  NETKIT_HEAP_ARENA=1 allows one init_heap() at startup.
 *
 * Optional file mmap (NETKIT_USE_MMAP / Makefile NETKIT_MMAP):
 *   Default ON for cpu + any MPU (macOS/Linux POSIX; Windows Win32). Forbidden on MCU.
 *   Opt out on RTOS / bare-metal MPU with NETKIT_MMAP=0 (no virtual-memory OS).
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

/* Reject legacy bare MCU/MPU macros — use MCU_ARM / MPU_ARM (or *_RISC / MCU_ESP). */
#if (defined(NETKIT_TARGET_MCU) || defined(NETKIT_TARGET_MPU)) &&                                 \
    !defined(NETKIT_TARGET_MCU_ARM) && !defined(NETKIT_TARGET_MPU_ARM) &&                         \
    !defined(NETKIT_TARGET_MCU_RISC) && !defined(NETKIT_TARGET_MPU_RISC) &&                       \
    !defined(NETKIT_TARGET_MCU_ESP)
#error "NETKIT_TARGET_MCU / NETKIT_TARGET_MPU are removed — use NETKIT_TARGET_MCU_ARM, "         \
       "NETKIT_TARGET_MPU_ARM, NETKIT_TARGET_MCU_RISC, NETKIT_TARGET_MPU_RISC, or "             \
       "NETKIT_TARGET_MCU_ESP"
#endif

#if defined(NETKIT_TARGET_CPU) + defined(NETKIT_TARGET_MCU_ARM) + defined(NETKIT_TARGET_MPU_ARM) + \
        defined(NETKIT_TARGET_MCU_RISC) + defined(NETKIT_TARGET_MPU_RISC) +                       \
        defined(NETKIT_TARGET_MCU_ESP) >                                                          \
    1
#error "Define only one of NETKIT_TARGET_CPU, NETKIT_TARGET_MCU_ARM, NETKIT_TARGET_MPU_ARM, "     \
       "NETKIT_TARGET_MCU_RISC, NETKIT_TARGET_MPU_RISC, NETKIT_TARGET_MCU_ESP"
#endif

#if !defined(NETKIT_TARGET_CPU) && !defined(NETKIT_TARGET_MCU_ARM) &&                             \
    !defined(NETKIT_TARGET_MPU_ARM) && !defined(NETKIT_TARGET_MCU_RISC) &&                        \
    !defined(NETKIT_TARGET_MPU_RISC) && !defined(NETKIT_TARGET_MCU_ESP)
#define NETKIT_TARGET_CPU 1
#endif
/* Class + ISA derived from the concrete target. */
#if defined(NETKIT_TARGET_MCU_ARM) || defined(NETKIT_TARGET_MCU_RISC) ||                          \
    defined(NETKIT_TARGET_MCU_ESP)
#define NETKIT_CLASS_MCU 1
#endif
#if defined(NETKIT_TARGET_MPU_ARM) || defined(NETKIT_TARGET_MPU_RISC)
#define NETKIT_CLASS_MPU 1
#endif
#if defined(NETKIT_TARGET_MCU_ARM) || defined(NETKIT_TARGET_MPU_ARM)
#define NETKIT_ISA_ARM 1
#endif
#if defined(NETKIT_TARGET_MCU_RISC) || defined(NETKIT_TARGET_MPU_RISC)
#define NETKIT_ISA_RISC 1
#endif
#if defined(NETKIT_TARGET_MCU_ESP)
#define NETKIT_ISA_ESP 1
#endif

#if defined(NETKIT_TARGET_CPU)
#define NETKIT_DESKTOP 1
#if !defined(NETKIT_GLOBAL_ARENA) || !NETKIT_GLOBAL_ARENA
#define NETKIT_ARENA_HEAP 1
#endif
#else
#if defined(NETKIT_CLASS_MCU) && defined(NETKIT_HEAP_ARENA) && NETKIT_HEAP_ARENA
#error "NETKIT_HEAP_ARENA is forbidden on MCU — use a static/global arena; no malloc/new"
#endif
#if defined(NETKIT_CLASS_MPU) && defined(NETKIT_HEAP_ARENA) && NETKIT_HEAP_ARENA
#define NETKIT_ARENA_HEAP 1
#endif
#endif

/* MCU: no iostream (keeps libstdc++ iostream out of flash). */
#if defined(NETKIT_CLASS_MCU)
#ifndef NETKIT_DISABLE_IOSTREAM
#define NETKIT_DISABLE_IOSTREAM 1
#endif
#endif

/* Static arena default for examples and NK_ARENA_DEFAULT_CAPACITY.
 * Override: -DNK_ARENA_DEFAULT_CAPACITY=<bytes> (or Make/CMake NETKIT_ARENA_*). */
#ifndef NK_ARENA_DEFAULT_CAPACITY
#if defined(NETKIT_CLASS_MCU)
#define NK_ARENA_DEFAULT_CAPACITY (64U * 1024U) /* 64 KiB */
#else
#define NK_ARENA_DEFAULT_CAPACITY (64U * 1024U * 1024U) /* 64 MiB — CPU and MPU */
#endif
#endif

#if NK_ARENA_DEFAULT_CAPACITY == 0
#error "NK_ARENA_DEFAULT_CAPACITY must be > 0"
#endif

/*
 * Conv2D im2col strategy for float reference and int8 QuantOps (single tri-state):
 *   0 = direct loops only, 1 = partial im2col, 2 = full im2col + GEMM.
 * Default 0 (direct) on all targets. CMSIS-NN / XNNPACK / ESP-NN ignore this knob.
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

/* MCU accel-only production path omits QuantOps reference loops (flash reclaim).
 * Applies to all CLASS_MCU targets when REFERENCE_QUANT_LOOPS=0 (CMSIS-NN Arm,
 * ESP-NN Espressif, or generic MCU with loops stripped). Historical name
 * NETKIT_MCU_CMSIS_ONLY; prefer NETKIT_MCU_ACCEL_ONLY in new code. */
#if defined(NETKIT_CLASS_MCU) && !NETKIT_REFERENCE_QUANT_LOOPS
#define NETKIT_MCU_CMSIS_ONLY 1
#else
#define NETKIT_MCU_CMSIS_ONLY 0
#endif
#define NETKIT_MCU_ACCEL_ONLY NETKIT_MCU_CMSIS_ONLY

#ifndef NETKIT_LOOP_UNROLL
#define NETKIT_LOOP_UNROLL 0
#endif

#if NETKIT_LOOP_UNROLL != 0 && NETKIT_LOOP_UNROLL != 1
#error "NETKIT_LOOP_UNROLL must be 0 or 1"
#endif

/* CMSIS-NN: Arm MCU (Cortex-M) only — never cpu / mpu / RISC / ESP (override not allowed). */
#if defined(NETKIT_TARGET_MCU_ARM) &&                                                              \
    (defined(ARM_MATH_CM0) || defined(ARM_MATH_CM0PLUS) || defined(ARM_MATH_CM3) ||               \
     defined(ARM_MATH_CM4) || defined(ARM_MATH_CM7) || defined(ARM_MATH_ARMV8MBL) ||               \
     defined(ARM_MATH_ARMV8MML) || defined(ARM_MATH_M55) || defined(ARM_MATH_M85))
#define NETKIT_CMSIS_NN_ALLOWED 1
#else
#define NETKIT_CMSIS_NN_ALLOWED 0
#endif

#if defined(NETKIT_USE_CMSIS_NN) && NETKIT_USE_CMSIS_NN && !NETKIT_CMSIS_NN_ALLOWED
#error "NETKIT_USE_CMSIS_NN requires NETKIT_TARGET_MCU_ARM with a Cortex-M NETKIT_ARCH "         \
       "(CM4/M33/...); forbidden on cpu, mpu, RISC, and ESP targets"
#endif

/*
 * ESP-NN: Espressif MCU only (ESP32 / S3 / C3 / C6 / P4 via NETKIT_ARCH).
 * Int8 production path; float32 uses reference (ESP-NN has no float kernels).
 */
#if defined(NETKIT_TARGET_MCU_ESP)
#define NETKIT_ESP_NN_ALLOWED 1
#else
#define NETKIT_ESP_NN_ALLOWED 0
#endif

#if defined(NETKIT_USE_ESP_NN) && NETKIT_USE_ESP_NN && !NETKIT_ESP_NN_ALLOWED
#error "NETKIT_USE_ESP_NN requires NETKIT_TARGET_MCU_ESP; forbidden on cpu, mpu, Arm, and RISC"
#endif

/*
 * NMSIS-NN: RISC-V MCU only (Nuclei N/NX/UX + RV32* via NETKIT_ARCH).
 * Int8 production path (CMSIS-NN analogue); float32 uses reference (no f32 API).
 */
#if defined(NETKIT_TARGET_MCU_RISC)
#define NETKIT_NMSIS_NN_ALLOWED 1
#else
#define NETKIT_NMSIS_NN_ALLOWED 0
#endif

#if defined(NETKIT_USE_NMSIS_NN) && NETKIT_USE_NMSIS_NN && !NETKIT_NMSIS_NN_ALLOWED
#error "NETKIT_USE_NMSIS_NN requires NETKIT_TARGET_MCU_RISC; forbidden on cpu, mpu, Arm, and ESP"
#endif

/*
 * XNNPACK LayerFast: default for cpu + any MPU (Arm or RISC). Forbidden on MCU
 * (override not allowed) — XNNPACK can run on many ISAs, but MCU flash/RAM
 * budgets make it a poor fit; use CMSIS-NN / ESP-NN / NMSIS-NN / reference there.
 */
#if defined(NETKIT_CLASS_MCU)
#define NETKIT_XNNPACK_ALLOWED 0
#else
#define NETKIT_XNNPACK_ALLOWED 1
#endif

#if defined(NETKIT_USE_XNNPACK) && NETKIT_USE_XNNPACK && !NETKIT_XNNPACK_ALLOWED
#error "NETKIT_USE_XNNPACK is forbidden on MCU targets (cpu and MPU only)"
#endif

/*
 * mmap for .nk file loads (POSIX on macOS/Linux; Win32 on Windows).
 * Allowed on cpu + any MPU; forbidden on MCU (override not allowed).
 * Default ON when allowed on Apple/Linux/Windows; opt out with NETKIT_USE_MMAP=0
 * on RTOS / bare-metal MPU (no virtual-memory OS).
 */
#if defined(NETKIT_CLASS_MCU)
#define NETKIT_MMAP_ALLOWED 0
#else
#define NETKIT_MMAP_ALLOWED 1
#endif

#ifndef NETKIT_USE_MMAP
#if NETKIT_MMAP_ALLOWED && (defined(__APPLE__) || defined(__linux__) || defined(_WIN32))
#define NETKIT_USE_MMAP 1
#else
#define NETKIT_USE_MMAP 0
#endif
#endif

#if NETKIT_USE_MMAP != 0 && NETKIT_USE_MMAP != 1
#error "NETKIT_USE_MMAP must be 0 or 1"
#endif

#if NETKIT_USE_MMAP && !NETKIT_MMAP_ALLOWED
#error "NETKIT_USE_MMAP is forbidden on MCU targets (cpu and MPU only; use Load*FromBuffer / flash)"
#endif

#endif /* NETKIT_CONFIG_H */
