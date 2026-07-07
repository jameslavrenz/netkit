# NUCLEO-F446RE (STM32F446RET6) — TFLM MNIST CNN int8 benchmark firmware.
#
# Full int8 TFLite graph (int8 input/output). Same 10 MNIST images and timing
# methodology as boards/nucleo-f446re-cnn-int8 (netkit).
#
# Usage:
#   make -C ../../benchmark/tflm export-cnn-int8   # once
#   make fetch-tflm setup-deps                     # once
#   make                                           # build firmware.elf
#   ./scripts/flash.sh
#   ./scripts/monitor.sh

NETKIT_ROOT := $(abspath ../..)
BOARD_NETKIT := $(abspath ../nucleo-f446re)
BOARD_ROOT := $(abspath .)
TFLM_ROOT := $(NETKIT_ROOT)/benchmark/tflm/third_party/tflite-micro
TFLM_MAKE := $(TFLM_ROOT)/tensorflow/lite/micro/tools/make/Makefile
TFLM_DL := $(TFLM_ROOT)/tensorflow/lite/micro/tools/make/downloads
BENCH_DIR := $(NETKIT_ROOT)/benchmark/tflm
GMAKE ?= $(shell command -v gmake 2>/dev/null || command -v make 2>/dev/null)

TOOLCHAIN_BIN := $(shell $(BOARD_NETKIT)/scripts/setup-toolchain.sh 2>/dev/null || true)
ifneq ($(TOOLCHAIN_BIN),)
  CC := $(TOOLCHAIN_BIN)/arm-none-eabi-gcc
  CXX := $(TOOLCHAIN_BIN)/arm-none-eabi-g++
  OBJCOPY := $(TOOLCHAIN_BIN)/arm-none-eabi-objcopy
  SIZE := $(TOOLCHAIN_BIN)/arm-none-eabi-size
else
  CC := arm-none-eabi-gcc
  CXX := arm-none-eabi-g++
  OBJCOPY := arm-none-eabi-objcopy
  SIZE := arm-none-eabi-size
endif

TFLM_TARGET := cortex_m_generic
TFLM_ARCH := cortex-m4+fp
TFLM_OPT_KERNEL := cmsis_nn
TFLM_GENDIR := $(TFLM_ROOT)/gen/$(TFLM_TARGET)_$(TFLM_ARCH)_default_$(TFLM_OPT_KERNEL)_gcc
TFLM_LIB := $(TFLM_GENDIR)/lib/libtensorflow-microlite.a

MCU_FLAGS := -mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard
OPT_FLAGS := -Os -ffunction-sections -fdata-sections -flto
WARN_FLAGS := -Wall -Wextra

BUILD_DIR := build
TARGET := mnist_cnn_int8_tflm_nucleo_f446re
ELF := $(BUILD_DIR)/$(TARGET).elf
BIN := $(BUILD_DIR)/$(TARGET).bin

LINKER_SCRIPT := $(BOARD_NETKIT)/linker/STM32F446RETx_FLASH.ld

INCLUDES := \
	-I$(BOARD_NETKIT)/include \
	-I$(BENCH_DIR) \
	-I$(TFLM_ROOT) \
	-I$(TFLM_DL)/flatbuffers/include \
	-I$(TFLM_DL)/gemmlowp \
	-I$(TFLM_DL)/kissfft \
	-I$(TFLM_DL)/cmsis/CMSIS/Core/Include \
	-I$(TFLM_DL)/cmsis/Cortex_DFP/Device/ARMCM4/Include

TFLM_CPPFLAGS := \
	-DTF_LITE_STATIC_MEMORY \
	-DTF_LITE_MCU_DEBUG_LOG \
	-DCPU_ARMCM4=1 \
	-DARMCM4 \
	-DCMSIS_DEVICE_ARM_CORTEX_M_XX_HEADER_FILE=\"ARMCM4.h\"

CFLAGS := $(MCU_FLAGS) $(OPT_FLAGS) $(WARN_FLAGS) -std=c23 -g $(INCLUDES) $(TFLM_CPPFLAGS)
CXXFLAGS := $(MCU_FLAGS) $(OPT_FLAGS) $(WARN_FLAGS) -std=c++17 -g $(INCLUDES) $(TFLM_CPPFLAGS) \
	-fno-rtti -fno-exceptions -fno-threadsafe-statics
ASFLAGS := $(MCU_FLAGS) -g
LDFLAGS := $(MCU_FLAGS) -flto -Wl,--gc-sections -T$(LINKER_SCRIPT) -Wl,-Map=$(BUILD_DIR)/$(TARGET).map

OBJS := \
	$(BUILD_DIR)/startup.o \
	$(BUILD_DIR)/system_stm32f4xx.o \
	$(BUILD_DIR)/uart.o \
	$(BUILD_DIR)/dwt_time.o \
	$(BUILD_DIR)/syscalls.o \
	$(BUILD_DIR)/main.o \
	$(BUILD_DIR)/tflm_debug_log.o \
	$(BUILD_DIR)/mnist_cnn_int8_model_data.o \
	$(BUILD_DIR)/mnist_cnn_int8_test_images.o

.PHONY: all clean flash monitor run setup-deps fetch-tflm check-deps size microlite export-cnn-int8

all: check-deps microlite $(ELF) size

check-deps:
	@TOOLCHAIN_BIN="$$(../nucleo-f446re/scripts/setup-toolchain.sh)"; \
	if [[ ! -x "$$TOOLCHAIN_BIN/arm-none-eabi-gcc" ]]; then \
	  echo "No arm-none-eabi toolchain — run ../nucleo-f446re/scripts/setup-toolchain.sh" >&2; \
	  exit 1; \
	fi
	@test -f $(TFLM_MAKE) || (echo "Run: make fetch-tflm" >&2; exit 1)
	@test -f $(BENCH_DIR)/generated/mnist_cnn_int8.tflite || (echo "Run: make export-cnn-int8" >&2; exit 1)
	@test -f $(BENCH_DIR)/generated/mnist_cnn_int8_model_data.cc || (echo "Run: make export-cnn-int8" >&2; exit 1)
	@test -f $(BENCH_DIR)/generated/mnist_cnn_int8_test_images.cc || (echo "Run: python3 benchmark/tflm/tools/export_int8_test_images.py" >&2; exit 1)

export-cnn-int8:
	@$(MAKE) -C $(BENCH_DIR) export-cnn-int8

fetch-tflm:
	@$(MAKE) -C $(BENCH_DIR) fetch-tflm

setup-deps:
	@chmod +x scripts/setup-tflm-deps.sh
	@./scripts/setup-tflm-deps.sh

microlite: setup-deps
	@$(GMAKE) -C $(TFLM_ROOT) -f $(TFLM_MAKE) \
		TARGET=$(TFLM_TARGET) TARGET_ARCH=$(TFLM_ARCH) \
		OPTIMIZED_KERNEL_DIR=$(TFLM_OPT_KERNEL) \
		TARGET_TOOLCHAIN_ROOT=$(TOOLCHAIN_BIN)/ \
		EXTERNAL_DIR=../.. \
		microlite

$(BUILD_DIR)/startup.o: $(BOARD_NETKIT)/startup/startup_stm32f446xx.S
	@mkdir -p $(BUILD_DIR)
	$(CC) $(ASFLAGS) -c $< -o $@

$(BUILD_DIR)/system_stm32f4xx.o: $(BOARD_NETKIT)/src/system_stm32f4xx.c
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/uart.o: $(BOARD_NETKIT)/src/uart.c
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/dwt_time.o: $(BOARD_NETKIT)/src/dwt_time.c
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/syscalls.o: $(BOARD_NETKIT)/src/syscalls.c
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/main.o: src/main.cc
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/tflm_debug_log.o: src/tflm_debug_log.cc
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/mnist_cnn_int8_model_data.o: $(BENCH_DIR)/generated/mnist_cnn_int8_model_data.cc
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TFLM_LIB): microlite

$(BUILD_DIR)/mnist_cnn_int8_test_images.o: $(BENCH_DIR)/generated/mnist_cnn_int8_test_images.cc
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(ELF): $(OBJS) $(TFLM_LIB)
	$(CXX) $(LDFLAGS) $(OBJS) $(TFLM_LIB) -lm -lc -lgcc -lstdc++ -lsupc++ -o $@
	$(OBJCOPY) -O binary $@ $(BIN)

size: $(ELF)
	$(SIZE) --format=berkeley $(ELF)
	@echo "Flash budget: 512 KiB — int8 model + TFLM CMSIS-NN runtime"

clean:
	rm -rf $(BUILD_DIR)

flash: $(ELF)
	@../nucleo-f446re/scripts/flash.sh $(ELF)

monitor:
	@../nucleo-f446re/scripts/monitor.sh

run: flash monitor
