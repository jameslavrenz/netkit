# Host TFLM compile/link flags (osx + gcc + BUILD_TYPE=default).
#
# Mirrored from tensorflow/lite/micro/tools/make/Makefile for fair netkit
# vs TFLM Micro comparison (MNIST / compare.sh). Refresh if TFLM defaults change.
#
# For TF Lite / LiteRT (MPU) ImageNet benches, use tflite_host_flags.mk instead
# (BENCH_FLAG_PROFILE=tflite): -O3 -DNDEBUG and no TF_LITE_DISABLE_X86_NEON.
#
# Include from benchmark/netkit/bench.mk when BENCH_FLAG_PROFILE=tflm (default):
#   include ../common/tflm_host_flags.mk

TFLM_HOST_CC := gcc
TFLM_HOST_CXX := g++
TFLM_HOST_AR := ar

TFLM_CC_WARNINGS := \
  -Wsign-compare \
  -Wdouble-promotion \
  -Wunused-variable \
  -Wunused-function \
  -Wswitch \
  -Wvla \
  -Wall \
  -Wextra \
  -Wmissing-field-initializers \
  -Wstrict-aliasing \
  -Wno-unused-parameter

TFLM_COMMON_FLAGS := \
  -fno-unwind-tables \
  -fno-asynchronous-unwind-tables \
  -ffunction-sections \
  -fdata-sections \
  -fmessage-length=0 \
  -DTF_LITE_STATIC_MEMORY \
  -DTF_LITE_DISABLE_X86_NEON \
  $(TFLM_CC_WARNINGS)

# TFLM host (osx) adds -DTF_LITE_USE_CTIME for non-cross builds.
TFLM_HOST_COMMON_FLAGS := $(TFLM_COMMON_FLAGS) -DTF_LITE_USE_CTIME

TFLM_CXXFLAGS := \
  -fno-rtti \
  -fno-exceptions \
  -fno-threadsafe-statics \
  -Wnon-virtual-dtor \
  $(TFLM_HOST_COMMON_FLAGS) \
  -std=c++20

TFLM_CORE_OPT := -O2
TFLM_KERNEL_OPT := -O2

# osx + gcc: no --gc-sections (LLVM linker backend).
TFLM_LDFLAGS := -lm

# netkit CPU runtime defines (benchmark uses file-loaded .nk models).
# Default direct Conv2D loops (0) on cpu/mcu/mpu benchmark builds.
NETKIT_IM2COL ?= 0
NETKIT_LOOP_UNROLL ?= 0
NETKIT_BENCH_CPPFLAGS := \
  -DNETKIT_TARGET_CPU=1 \
  -DNETKIT_IM2COL=$(NETKIT_IM2COL) \
  -DNETKIT_LOOP_UNROLL=$(NETKIT_LOOP_UNROLL)

TFLM_BENCH_INCLUDES := -I$(ROOT)/include -I$(SHARED_GEN) -I../common

TFLM_BENCH_CORE_CXXFLAGS := $(TFLM_CXXFLAGS) $(TFLM_CORE_OPT) $(NETKIT_BENCH_CPPFLAGS) $(TFLM_BENCH_INCLUDES)
TFLM_BENCH_KERNEL_CXXFLAGS := $(TFLM_CXXFLAGS) $(TFLM_KERNEL_OPT) $(NETKIT_BENCH_CPPFLAGS) -I$(ROOT)/include
