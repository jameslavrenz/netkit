# TFLM microlite-aligned MCU toolchain flags (cortex-m4+fp, gcc).
#
# Mirrors tensorflow/lite/micro/tools/make/Makefile and
# targets/cortex_m_generic_makefile.inc defaults used by
# OPTIMIZED_KERNEL_DIR=cmsis_nn board firmware.
#
# Requires MCU_BOARD_COMMON (see mcu_gcc_toolchain.mk).
#
# Optimization (match TFLM kernel speed — never -Os):
#   CORE_OPTIMIZATION_LEVEL              -O2  board glue + runtime
#   KERNEL_OPTIMIZATION_LEVEL            -O2  hot inference C++
#   THIRD_PARTY_KERNEL_OPTIMIZATION_LEVEL -O2 CMSIS-NN / CMSIS-DSP kernels

include $(dir $(lastword $(MAKEFILE_LIST)))mcu_gcc_toolchain.mk

MCU_CPU := cortex-m4

MCU_FLAGS := \
	-mcpu=$(MCU_CPU) \
	-mthumb \
	-mfpu=fpv4-sp-d16 \
	-mfloat-abi=hard

TFLM_PLATFORM_FLAGS := \
	-mlittle-endian \
	-fomit-frame-pointer \
	-funsigned-char

TFLM_COMMON_FLAGS := \
	-fno-unwind-tables \
	-fno-asynchronous-unwind-tables \
	-ffunction-sections \
	-fdata-sections \
	-fmessage-length=0

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

TFLM_C_ONLY_WARNINGS := -Wimplicit-function-declaration

TFLM_CXX_WARNINGS := \
	-Wnon-virtual-dtor

# TFLM microlite board glue uses C++17; netkit boards set MCU_CXX_STD before include.
MCU_CXX_STD ?= -std=c++17

CORE_OPTIMIZATION_LEVEL := -O2
KERNEL_OPTIMIZATION_LEVEL := -O2
THIRD_PARTY_KERNEL_OPTIMIZATION_LEVEL := -O2

# Board firmware final link uses LTO (same as boards/nucleo-f446re-tflm-cnn-int8).
MCU_LINK_FLAGS := -flto

MCU_LDFLAGS := \
	$(MCU_FLAGS) \
	$(MCU_LINK_FLAGS) \
	-Wl,--fatal-warnings \
	-Wl,--gc-sections

MCU_CORE_CFLAGS := \
	$(MCU_FLAGS) \
	$(TFLM_PLATFORM_FLAGS) \
	$(TFLM_COMMON_FLAGS) \
	$(TFLM_CC_WARNINGS) \
	$(TFLM_C_ONLY_WARNINGS) \
	$(CORE_OPTIMIZATION_LEVEL) \
	-std=c17

MCU_KERNEL_CFLAGS := \
	$(MCU_FLAGS) \
	$(TFLM_PLATFORM_FLAGS) \
	$(TFLM_COMMON_FLAGS) \
	$(TFLM_CC_WARNINGS) \
	$(TFLM_C_ONLY_WARNINGS) \
	$(KERNEL_OPTIMIZATION_LEVEL) \
	-std=c17

MCU_THIRD_PARTY_KERNEL_CFLAGS := \
	$(MCU_FLAGS) \
	$(TFLM_PLATFORM_FLAGS) \
	$(TFLM_COMMON_FLAGS) \
	$(TFLM_CC_WARNINGS) \
	$(TFLM_C_ONLY_WARNINGS) \
	$(THIRD_PARTY_KERNEL_OPTIMIZATION_LEVEL) \
	-std=c17

MCU_CORE_CXXFLAGS := \
	$(MCU_FLAGS) \
	$(TFLM_PLATFORM_FLAGS) \
	$(TFLM_COMMON_FLAGS) \
	$(TFLM_CC_WARNINGS) \
	$(TFLM_CXX_WARNINGS) \
	$(CORE_OPTIMIZATION_LEVEL) \
	$(MCU_CXX_STD) \
	-fno-rtti \
	-fno-exceptions \
	-fno-threadsafe-statics

MCU_KERNEL_CXXFLAGS := \
	$(MCU_FLAGS) \
	$(TFLM_PLATFORM_FLAGS) \
	$(TFLM_COMMON_FLAGS) \
	$(TFLM_CC_WARNINGS) \
	$(TFLM_CXX_WARNINGS) \
	$(KERNEL_OPTIMIZATION_LEVEL) \
	$(MCU_CXX_STD) \
	-fno-rtti \
	-fno-exceptions \
	-fno-threadsafe-statics

MCU_ASFLAGS := $(MCU_FLAGS)

# Debug symbols for on-device bring-up (both board Makefiles pass -g).
MCU_DEBUG_FLAGS := -g
