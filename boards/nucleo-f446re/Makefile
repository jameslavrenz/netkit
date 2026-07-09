# NUCLEO-F446RE (STM32F446RET6) — MNIST MLP f32 benchmark firmware.
#
# Build profile (explicit CMSIS opt-in):
#   NETKIT_CMSIS_DSP=1, NETKIT_CMSIS_NN=1
#   NETKIT_TARGET=mcu, NETKIT_ARCH=CM4 (weights always flash/blob-backed)
#   Toolchain: mcu_tflm_toolchain.mk (CORE/KERNEL/THIRD_PARTY all -O2, -flto link)
#     so netkit and TFLM firmware are compiled with matching speed flags.
#
# Usage:
#   make cmsis-init          # once, from repo root
#   make                     # build firmware.elf
#   ./scripts/flash.sh       # program via ST-Link (USB)
#   ./scripts/monitor.sh     # read UART results (ST-Link VCP)

NETKIT_ROOT := $(abspath ../..)
BOARD_ROOT := $(abspath .)

# netkit public headers use C++20 (std::span, constinit); TFLM board glue stays C++17.
MCU_CXX_STD := -std=c++20
MCU_BOARD_COMMON := $(BOARD_ROOT)

include $(BOARD_ROOT)/mcu_tflm_toolchain.mk

BUILD_DIR := build
GEN_DIR := generated
LIB_DIR := $(BUILD_DIR)/lib
CMSIS_DSP_DIR := $(NETKIT_ROOT)/third_party/CMSIS-DSP
CMSIS_NN_DIR := $(NETKIT_ROOT)/third_party/CMSIS-NN

TARGET := mnist_mlp_nucleo_f446re
ELF := $(BUILD_DIR)/$(TARGET).elf
BIN := $(BUILD_DIR)/$(TARGET).bin

LINKER_SCRIPT := linker/STM32F446RETx_FLASH.ld

INCLUDES := \
	-I$(BOARD_ROOT)/include \
	-I$(NETKIT_ROOT)/include \
	-I$(GEN_DIR) \
	-I$(NETKIT_ROOT)/benchmark/tflm/generated \
	-I$(CMSIS_DSP_DIR)/Include \
	-I$(CMSIS_DSP_DIR)/PrivateInclude \
	-I$(CMSIS_NN_DIR)/Include \
	-I$(NETKIT_ROOT)/third_party/CMSIS-Core/CMSIS/Core/Include

NETKIT_CMSIS_DSP ?= 1
NETKIT_CMSIS_NN ?= 1

NETKIT_CPPFLAGS := \
	-DNETKIT_TARGET_MCU=1 \
	$(if $(filter 1,$(NETKIT_CMSIS_DSP)),-DNETKIT_USE_CMSIS_DSP=1,) \
	$(if $(filter 1,$(NETKIT_CMSIS_NN)),-DNETKIT_USE_CMSIS_NN=1 -DARM_NN_ENABLE_F32=1,) \
	-DNETKIT_LOOP_UNROLL=0 \
	-DNETKIT_IM2COL=0 \
	-DARM_MATH_CM4 \
	-DARM_MATH_LOOPUNROLL

# Hot inference kernels — compiled at KERNEL_OPTIMIZATION_LEVEL (-O2), the same
# tier TFLM applies to its cmsis_nn / cmsis_dsp kernels.
NETKIT_KERNEL_RUNTIME_SOURCES := \
	$(NETKIT_ROOT)/src/reference_kernel.cpp \
	$(NETKIT_ROOT)/src/ops.cpp \
	$(NETKIT_ROOT)/src/cmsis_dsp_util.cpp
ifeq ($(NETKIT_CMSIS_DSP),1)
NETKIT_KERNEL_RUNTIME_SOURCES += $(NETKIT_ROOT)/src/cmsis_dsp_backend.cpp
endif
ifeq ($(NETKIT_CMSIS_NN),1)
NETKIT_KERNEL_RUNTIME_SOURCES += $(NETKIT_ROOT)/src/cmsis_nn_backend.cpp
endif

# Runtime infrastructure — compiled at CORE_OPTIMIZATION_LEVEL (-O2).
NETKIT_CORE_RUNTIME_SOURCES := \
	$(NETKIT_ROOT)/src/arena.cpp \
	$(NETKIT_ROOT)/src/tensor_factory.cpp \
	$(NETKIT_ROOT)/src/tensor_access.cpp \
	$(NETKIT_ROOT)/src/kernel_workspace.cpp \
	$(NETKIT_ROOT)/src/cmsis_buffer_size.cpp

CMSIS_DSP_SOURCES :=
ifeq ($(NETKIT_CMSIS_DSP),1)
CMSIS_DSP_SOURCES := \
	$(CMSIS_DSP_DIR)/Source/BasicMathFunctions/arm_add_f32.c \
	$(CMSIS_DSP_DIR)/Source/BasicMathFunctions/arm_mult_f32.c \
	$(CMSIS_DSP_DIR)/Source/BasicMathFunctions/arm_dot_prod_f32.c \
	$(CMSIS_DSP_DIR)/Source/BasicMathFunctions/arm_scale_f32.c \
	$(CMSIS_DSP_DIR)/Source/BasicMathFunctions/arm_offset_f32.c \
	$(CMSIS_DSP_DIR)/Source/BasicMathFunctions/arm_clip_f32.c \
	$(CMSIS_DSP_DIR)/Source/StatisticsFunctions/arm_mean_f32.c \
	$(CMSIS_DSP_DIR)/Source/StatisticsFunctions/arm_var_f32.c \
	$(CMSIS_DSP_DIR)/Source/StatisticsFunctions/arm_max_f32.c \
	$(CMSIS_DSP_DIR)/Source/SupportFunctions/arm_copy_f32.c \
	$(CMSIS_DSP_DIR)/Source/MatrixFunctions/arm_mat_init_f32.c \
	$(CMSIS_DSP_DIR)/Source/MatrixFunctions/arm_mat_vec_mult_f32.c \
	$(CMSIS_DSP_DIR)/Source/MatrixFunctions/arm_mat_mult_f32.c
endif

# CMSIS-NN float32 kernels (opt-in via NETKIT_CMSIS_NN=1). The netkit CmsisNnKernel
# adapter lives in cmsis_nn_backend.cpp; the CMSIS-NN f32 kernels below satisfy its
# calls. Unused kernels (conv/pool/etc. for this FC-only MLP) are dropped by
# -ffunction-sections + --gc-sections, so this does not grow the image.
CMSIS_NN_SOURCES :=
ifeq ($(NETKIT_CMSIS_NN),1)
CMSIS_NN_SOURCES := \
	$(wildcard $(CMSIS_NN_DIR)/Source/ConvolutionFunctions/arm_*f32*.c) \
	$(wildcard $(CMSIS_NN_DIR)/Source/FullyConnectedFunctions/arm_*f32*.c) \
	$(wildcard $(CMSIS_NN_DIR)/Source/PoolingFunctions/arm_*f32*.c) \
	$(wildcard $(CMSIS_NN_DIR)/Source/ActivationFunctions/arm_*f32*.c) \
	$(wildcard $(CMSIS_NN_DIR)/Source/SoftmaxFunctions/arm_*f32*.c) \
	$(wildcard $(CMSIS_NN_DIR)/Source/BasicMathFunctions/arm_*f32*.c) \
	$(wildcard $(CMSIS_NN_DIR)/Source/NNSupportFunctions/arm_*f32*.c) \
	$(CMSIS_NN_DIR)/Source/NNSupportFunctions/arm_nntables_flt.c
endif

CFLAGS := $(MCU_CORE_CFLAGS) $(MCU_DEBUG_FLAGS) $(INCLUDES) $(NETKIT_CPPFLAGS)
CXXFLAGS := $(MCU_CORE_CXXFLAGS) $(MCU_DEBUG_FLAGS) $(INCLUDES) $(NETKIT_CPPFLAGS)
ASFLAGS := $(MCU_ASFLAGS) $(MCU_DEBUG_FLAGS)
LDFLAGS := $(MCU_LDFLAGS) $(NETKIT_CPPFLAGS) -T$(LINKER_SCRIPT) -Wl,-Map=$(BUILD_DIR)/$(TARGET).map

LIBNETKIT := $(LIB_DIR)/libnetkit_mnist_mlp.a

.PHONY: all clean flash monitor run export-embed export-aot check-deps size

all: check-deps export-embed $(ELF) size

check-deps:
	$(MCU_ASSERT_GCC_TOOLCHAIN)
	@test "$(NETKIT_CMSIS_DSP)" != "1" -o -f $(CMSIS_DSP_DIR)/Include/arm_math.h || \
		(echo "NETKIT_CMSIS_DSP=1 requires CMSIS-DSP — run: make -C $(NETKIT_ROOT) cmsis-init" >&2; exit 1)
	@test "$(NETKIT_CMSIS_NN)" != "1" -o -f $(CMSIS_NN_DIR)/Include/arm_nnfunctions.h || \
		(echo "NETKIT_CMSIS_NN=1 requires CMSIS-NN — run: make -C $(NETKIT_ROOT) cmsis-init" >&2; exit 1)
	@test -f $(NETKIT_ROOT)/models/mnist_mlp.nk || \
		(echo "Missing models/mnist_mlp.nk — run: make -C $(NETKIT_ROOT) export-mnist" >&2; exit 1)

export-embed: $(GEN_DIR)/.embed_stamp

$(GEN_DIR)/.embed_stamp: $(NETKIT_ROOT)/models/mnist_mlp.nk $(NETKIT_ROOT)/python/netkit/aot_lower.py
	@mkdir -p $(GEN_DIR)
	cd $(NETKIT_ROOT) && PYTHONPATH=$(NETKIT_ROOT)/python python3 -m netkit aot models/mnist_mlp.nk \
		-o $(abspath $(GEN_DIR)) --language cpp --target mcu --optimize --arena-headroom 20
	@touch $(GEN_DIR)/.embed_stamp

export-aot: export-embed

$(LIBNETKIT): $(NETKIT_KERNEL_RUNTIME_SOURCES) $(NETKIT_CORE_RUNTIME_SOURCES) $(CMSIS_DSP_SOURCES) $(CMSIS_NN_SOURCES)
	@mkdir -p $(LIB_DIR)
	@rm -f $(LIB_DIR)/*.o
	@set -e; for src in $(CMSIS_DSP_SOURCES); do \
		obj=$(LIB_DIR)/cmsis_dsp_$$(basename $$src .c).o; \
		$(CC) $(MCU_THIRD_PARTY_KERNEL_CFLAGS) $(MCU_DEBUG_FLAGS) $(INCLUDES) $(NETKIT_CPPFLAGS) -c $$src -o $$obj; \
	done
	@set -e; for src in $(CMSIS_NN_SOURCES); do \
		obj=$(LIB_DIR)/cmsis_nn_$$(basename $$src .c).o; \
		$(CC) $(MCU_THIRD_PARTY_KERNEL_CFLAGS) $(MCU_DEBUG_FLAGS) -I$(CMSIS_NN_DIR)/Include $(NETKIT_CPPFLAGS) -c $$src -o $$obj; \
	done
	@set -e; for src in $(NETKIT_CORE_RUNTIME_SOURCES); do \
		obj=$(LIB_DIR)/netkit_$$(basename $$src .cpp).o; \
		$(CXX) $(MCU_CORE_CXXFLAGS) $(MCU_DEBUG_FLAGS) $(INCLUDES) $(NETKIT_CPPFLAGS) -c $$src -o $$obj; \
	done
	@set -e; for src in $(NETKIT_KERNEL_RUNTIME_SOURCES); do \
		obj=$(LIB_DIR)/netkit_$$(basename $$src .cpp).o; \
		$(CXX) $(MCU_KERNEL_CXXFLAGS) $(MCU_DEBUG_FLAGS) $(INCLUDES) $(NETKIT_CPPFLAGS) -c $$src -o $$obj; \
	done
	$(AR) rcs $@ $(LIB_DIR)/*.o

OBJS := \
	$(BUILD_DIR)/startup.o \
	$(BUILD_DIR)/system_stm32f4xx.o \
	$(BUILD_DIR)/uart.o \
	$(BUILD_DIR)/dwt_time.o \
	$(BUILD_DIR)/syscalls.o \
	$(BUILD_DIR)/main.o \
	$(BUILD_DIR)/mnist_test_images.o \
	$(BUILD_DIR)/mnist_mlp_aot.o

$(BUILD_DIR)/startup.o: startup/startup_stm32f446xx.S
	@mkdir -p $(BUILD_DIR)
	$(CC) $(ASFLAGS) -c $< -o $@

$(BUILD_DIR)/syscalls.o: src/syscalls.c
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -fno-lto -c $< -o $@

$(BUILD_DIR)/%.o: src/%.c
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/main.o: src/main.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/mnist_test_images.o: $(NETKIT_ROOT)/benchmark/tflm/generated/mnist_test_images.cc
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Lowered AOT is the hot FC/activation call chain — KERNEL -O2, same as TFLM kernels.
$(BUILD_DIR)/mnist_mlp_aot.o: $(GEN_DIR)/mnist_mlp_aot.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(MCU_KERNEL_CXXFLAGS) $(MCU_DEBUG_FLAGS) $(INCLUDES) $(NETKIT_CPPFLAGS) -c $< -o $@

$(ELF): $(OBJS) $(LIBNETKIT)
	$(CXX) $(LDFLAGS) $(OBJS) $(LIBNETKIT) -lm -lc -lgcc -lstdc++ -lsupc++ -o $@
	$(OBJCOPY) -O binary $@ $(BIN)

size: $(ELF)
	$(SIZE) --format=berkeley $(ELF)
	@echo "Flash budget: 512 KiB — see .text + .data load size above"

clean:
	rm -rf $(BUILD_DIR)
	rm -rf $(GEN_DIR)

flash: $(ELF)
	./scripts/flash.sh $(ELF)

monitor:
	./scripts/monitor.sh

run: flash monitor
