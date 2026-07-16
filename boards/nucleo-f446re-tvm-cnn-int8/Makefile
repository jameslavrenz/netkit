# NUCLEO-F446RE — microTVM AOT (CMSIS-NN) MNIST CNN int8 benchmark.
#
# Prerequisites:
#   1) Relay-era TVM (v0.14) built with USE_MICRO=ON USE_CMSISNN=ON
#      export TVM_HOME=/path/to/apache-tvm-v0.14
#   2) make compile-model   # writes generated/mlf from mnist_cnn_int8.tflite
#   3) make && ./scripts/flash.sh
#
# Same 10 images / timing methodology as nucleo-f446re-cnn-int8 and TFLM peer.

NETKIT_ROOT := $(abspath ../..)
BOARD_NETKIT := $(abspath ../nucleo-f446re)
BOARD_ROOT := $(abspath .)

MCU_BOARD_COMMON := $(BOARD_NETKIT)
include $(BOARD_NETKIT)/mcu_tflm_toolchain.mk

TVM_HOME ?= $(abspath ../../../apache-tvm-v0.14)
BENCH_DIR := $(NETKIT_ROOT)/benchmark/tflm
MLF_DIR := $(BOARD_ROOT)/generated/mlf
CMSIS_NN_DIR := $(NETKIT_ROOT)/third_party/CMSIS-NN
CMSIS_CORE_INC := $(NETKIT_ROOT)/third_party/CMSIS-Core/CMSIS/Core/Include

MODULE_NAME := mnist_cnn_int8
# 1 = CMSIS-NN BYOC (default). 0 = pure C AOT reference kernels.
TVM_USE_CMSISNN ?= 1
BUILD_DIR := build
TARGET := mnist_cnn_int8_tvm_nucleo_f446re
ELF := $(BUILD_DIR)/$(TARGET).elf
BIN := $(BUILD_DIR)/$(TARGET).bin
LINKER_SCRIPT := $(BOARD_NETKIT)/linker/STM32F446RETx_FLASH.ld

CRT_CONFIG_DIR := $(BOARD_ROOT)/generated/crt_config
CODEGEN_SRCS := $(wildcard $(MLF_DIR)/codegen/host/src/*.c)
CRT_COMMON := $(MLF_DIR)/runtime/src/runtime/crt/common/crt_backend_api.c
CRT_STACK := $(MLF_DIR)/runtime/src/runtime/crt/memory/stack_allocator.c

CMSIS_NN_S8_SRCS := \
	$(wildcard $(CMSIS_NN_DIR)/Source/ConvolutionFunctions/arm_*s8*.c) \
	$(wildcard $(CMSIS_NN_DIR)/Source/FullyConnectedFunctions/arm_*s8*.c) \
	$(wildcard $(CMSIS_NN_DIR)/Source/PoolingFunctions/arm_*s8*.c) \
	$(wildcard $(CMSIS_NN_DIR)/Source/ActivationFunctions/arm_*s8*.c) \
	$(wildcard $(CMSIS_NN_DIR)/Source/BasicMathFunctions/arm_*s8*.c) \
	$(wildcard $(CMSIS_NN_DIR)/Source/NNSupportFunctions/arm_*s8*.c) \
	$(wildcard $(CMSIS_NN_DIR)/Source/NNSupportFunctions/arm_nn_*_s8*.c) \
	$(wildcard $(CMSIS_NN_DIR)/Source/NNSupportFunctions/arm_q7_to_q15*.c) \
	$(wildcard $(CMSIS_NN_DIR)/Source/NNSupportFunctions/arm_s8_to_s16*.c) \
	$(wildcard $(CMSIS_NN_DIR)/Source/NNSupportFunctions/arm_nn_mat_mult_kernel_s8_s16.c) \
	$(wildcard $(CMSIS_NN_DIR)/Source/NNSupportFunctions/arm_nn_mat_mult_kernel_row_offset_s8_s16.c) \
	$(wildcard $(CMSIS_NN_DIR)/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s8.c) \
	$(wildcard $(CMSIS_NN_DIR)/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s8.c) \
	$(wildcard $(CMSIS_NN_DIR)/Source/NNSupportFunctions/arm_memcpy_s8.c) \
	$(wildcard $(CMSIS_NN_DIR)/Source/NNSupportFunctions/arm_memset_s8.c) \
	$(wildcard $(CMSIS_NN_DIR)/Source/SoftmaxFunctions/arm_softmax_s8.c) \
	$(wildcard $(CMSIS_NN_DIR)/Source/SoftmaxFunctions/arm_nn_softmax_common_s8.c)

INCLUDES := \
	-I$(BOARD_NETKIT)/include \
	-I$(BOARD_ROOT)/generated \
	-I$(BENCH_DIR) \
	-I$(MLF_DIR)/codegen/host/include \
	-I$(MLF_DIR)/runtime/include \
	-I$(MLF_DIR)/runtime/src/runtime/crt/include \
	-I$(CRT_CONFIG_DIR) \
	-I$(CMSIS_NN_DIR)/Include \
	-I$(CMSIS_CORE_INC)

# Match TFLM/netkit flag tiers: board glue = CORE, AOT codegen/CRT = KERNEL,
# CMSIS-NN = THIRD_PARTY_KERNEL (all -O2 today; keeps the same switch points).
TVM_EXTRA_CFLAGS := \
	-DTVM_RUNTIME_ALLOC_ALIGNMENT_BYTES=8 \
	-Wno-unused-parameter -Wno-incompatible-pointer-types
TVM_EXTRA_CXXFLAGS := \
	-DTVM_RUNTIME_ALLOC_ALIGNMENT_BYTES=8 \
	-Wno-unused-parameter \
	-DTVM_MCU_BACKEND_LABEL=\"$(if $(filter 1,$(TVM_USE_CMSISNN)),cmsis-nn,reference)\"

CORE_CFLAGS := $(MCU_CORE_CFLAGS) $(MCU_DEBUG_FLAGS) $(INCLUDES) $(TVM_EXTRA_CFLAGS)
CORE_CXXFLAGS := $(MCU_CORE_CXXFLAGS) $(MCU_DEBUG_FLAGS) $(INCLUDES) $(TVM_EXTRA_CXXFLAGS)
KERNEL_CFLAGS := $(MCU_KERNEL_CFLAGS) $(MCU_DEBUG_FLAGS) $(INCLUDES) $(TVM_EXTRA_CFLAGS)
KERNEL_CXXFLAGS := $(MCU_KERNEL_CXXFLAGS) $(MCU_DEBUG_FLAGS) $(INCLUDES) $(TVM_EXTRA_CXXFLAGS)
CMSIS_CFLAGS := $(MCU_THIRD_PARTY_KERNEL_CFLAGS) $(MCU_DEBUG_FLAGS) $(INCLUDES) $(TVM_EXTRA_CFLAGS)

# Legacy names for any local recipes that still reference them.
CFLAGS := $(CORE_CFLAGS)
CXXFLAGS := $(CORE_CXXFLAGS)
ASFLAGS := $(MCU_ASFLAGS) $(MCU_DEBUG_FLAGS)
LDFLAGS := $(MCU_LDFLAGS) -T$(LINKER_SCRIPT) -Wl,-Map=$(BUILD_DIR)/$(TARGET).map

BOARD_OBJS := \
	$(BUILD_DIR)/startup.o \
	$(BUILD_DIR)/system_stm32f4xx.o \
	$(BUILD_DIR)/uart.o \
	$(BUILD_DIR)/dwt_time.o \
	$(BUILD_DIR)/syscalls.o \
	$(BUILD_DIR)/tvm_platform.o \
	$(BUILD_DIR)/main.o \
	$(BUILD_DIR)/mnist_cnn_int8_test_images.o \
	$(BUILD_DIR)/crt_backend_api.o \
	$(BUILD_DIR)/stack_allocator.o

CODEGEN_OBJS := $(patsubst $(MLF_DIR)/codegen/host/src/%.c,$(BUILD_DIR)/codegen/%.o,$(CODEGEN_SRCS))
# C-only AOT does not call CMSIS-NN; link it only when generated sources reference arm_*.
NEED_CMSIS := $(shell rg -l 'arm_convolve|arm_fully_connected|arm_max_pool|arm_depthwise' $(MLF_DIR)/codegen/host/src 2>/dev/null | head -1)
ifneq ($(NEED_CMSIS),)
CMSIS_OBJS := $(patsubst $(CMSIS_NN_DIR)/%.c,$(BUILD_DIR)/cmsis_nn/%.o,$(CMSIS_NN_S8_SRCS))
else
CMSIS_OBJS :=
endif

.PHONY: all clean flash monitor run compile-model check-mlf size

all: check-mlf $(ELF) size

check-mlf:
	$(MCU_ASSERT_GCC_TOOLCHAIN)
	@test -f $(BENCH_DIR)/generated/mnist_cnn_int8.tflite || \
		(echo "Run: make -C ../../benchmark/tflm export-cnn-int8" >&2; exit 1)
	@test -f $(BENCH_DIR)/generated/mnist_cnn_int8_test_images.cc || \
		(echo "Missing test images; export via TFLM tools" >&2; exit 1)
	@test -n "$(CODEGEN_SRCS)" || \
		(echo "Missing MLF codegen; run: make compile-model (needs TVM_HOME)" >&2; exit 1)
	@test -f $(CRT_CONFIG_DIR)/crt_config.h || \
		(echo "Missing crt_config.h; run: make compile-model" >&2; exit 1)

TVM_PYTHON ?= conda run -n tvm014 --no-capture-output env PYTHONPATH=$(TVM_HOME)/python

compile-model:
	@mkdir -p $(CRT_CONFIG_DIR) $(MLF_DIR)
	@test -d $(TVM_HOME)/python || (echo "Set TVM_HOME to Relay-era TVM (v0.14)" >&2; exit 1)
	@# Ensure per-channel FC is excluded from CMSIS-NN (patch is idempotent via grep).
	@if ! grep -q 'is_per_tensor' $(TVM_HOME)/python/tvm/relay/op/contrib/cmsisnn.py; then \
		echo "Applying patches/cmsisnn_skip_per_channel_fc.patch to $$TVM_HOME"; \
		patch -d $(TVM_HOME) -p1 < patches/cmsisnn_skip_per_channel_fc.patch; \
	fi
	$(TVM_PYTHON) python tools/compile_mnist_cnn_int8_tvm.py \
		--model $(BENCH_DIR)/generated/mnist_cnn_int8.tflite \
		--out-dir $(MLF_DIR) \
		--module-name $(MODULE_NAME) \
		$(if $(filter 0,$(TVM_USE_CMSISNN)),--no-cmsisnn,)
	@if [ -f $(MLF_DIR)/templates/crt_config.h.template ]; then \
		cp $(MLF_DIR)/templates/crt_config.h.template $(CRT_CONFIG_DIR)/crt_config.h; \
	elif [ -f $(MLF_DIR)/templates/crt_config-template.h ]; then \
		cp $(MLF_DIR)/templates/crt_config-template.h $(CRT_CONFIG_DIR)/crt_config.h; \
	elif [ -f $(TVM_HOME)/build/microtvm_template_projects/crt/crt_config/crt_config.h ]; then \
		cp $(TVM_HOME)/build/microtvm_template_projects/crt/crt_config/crt_config.h $(CRT_CONFIG_DIR)/crt_config.h; \
	else \
		echo "WARN: no crt_config template; using minimal stub"; \
		printf '%s\n' \
			'#pragma once' \
			'#define TVM_CRT_MAX_NDIM 6' \
			'#define TVM_CRT_MAX_ARGS 10' \
			'#define TVM_CRT_GLOBAL_FUNC_REGISTRY_SIZE_BYTES 512' \
			'#define TVM_CRT_MAX_REGISTERED_MODULES 2' \
			'#define TVM_CRT_MAX_PACKET_SIZE_BYTES 2048' \
			'#define TVM_CRT_MAX_STRLEN_DLTYPE 16' \
			'#define TVM_CRT_MAX_STRLEN_FUNCTION_NAME 120' \
			'#define TVM_CRT_MAX_STRLEN_PARAM_NAME 80' \
			> $(CRT_CONFIG_DIR)/crt_config.h; \
	fi
	@HDR=$$(ls $(MLF_DIR)/codegen/host/include/tvmgen_$(MODULE_NAME).h 2>/dev/null | head -1); \
		test -n "$$HDR" || (echo "missing generated header" >&2; exit 1); \
		$(TVM_PYTHON) python tools/gen_tvm_io_bind.py \
			--header "$$HDR" --out generated/tvm_io_bind.h --module $(MODULE_NAME)


$(BUILD_DIR)/startup.o: $(BOARD_NETKIT)/startup/startup_stm32f446xx.S
	@mkdir -p $(BUILD_DIR)
	$(CC) $(ASFLAGS) -c $< -o $@

$(BUILD_DIR)/system_stm32f4xx.o: $(BOARD_NETKIT)/src/system_stm32f4xx.c
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CORE_CFLAGS) -c $< -o $@

$(BUILD_DIR)/uart.o: $(BOARD_NETKIT)/src/uart.c
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CORE_CFLAGS) -c $< -o $@

$(BUILD_DIR)/dwt_time.o: $(BOARD_NETKIT)/src/dwt_time.c
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CORE_CFLAGS) -c $< -o $@

$(BUILD_DIR)/syscalls.o: $(BOARD_NETKIT)/src/syscalls.c
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CORE_CFLAGS) -fno-lto -c $< -o $@

$(BUILD_DIR)/tvm_platform.o: src/tvm_platform.c
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CORE_CFLAGS) -c $< -o $@

$(BUILD_DIR)/main.o: src/main.cc
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CORE_CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/mnist_cnn_int8_test_images.o: $(BENCH_DIR)/generated/mnist_cnn_int8_test_images.cc
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(KERNEL_CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/crt_backend_api.o: $(CRT_COMMON)
	@mkdir -p $(BUILD_DIR)
	$(CC) $(KERNEL_CFLAGS) -c $< -o $@

$(BUILD_DIR)/stack_allocator.o: $(CRT_STACK)
	@mkdir -p $(BUILD_DIR)
	$(CC) $(KERNEL_CFLAGS) -c $< -o $@

$(BUILD_DIR)/codegen/%.o: $(MLF_DIR)/codegen/host/src/%.c
	@mkdir -p $(dir $@)
	$(CC) $(KERNEL_CFLAGS) -c $< -o $@

$(BUILD_DIR)/cmsis_nn/%.o: $(CMSIS_NN_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CMSIS_CFLAGS) -c $< -o $@

$(ELF): $(BOARD_OBJS) $(CODEGEN_OBJS) $(CMSIS_OBJS)
	$(CXX) $(LDFLAGS) $^ -lm -lc -lgcc -lstdc++ -lsupc++ -o $@
	$(OBJCOPY) -O binary $@ $(BIN)

size: $(ELF)
	$(SIZE) --format=berkeley $(ELF)
	@echo "Flash budget: 512 KiB — microTVM AOT + CMSIS-NN"

clean:
	rm -rf $(BUILD_DIR)

flash: $(ELF)
	@../nucleo-f446re/scripts/flash.sh $(ELF)

monitor:
	@../nucleo-f446re/scripts/monitor.sh

run: flash monitor
