# NMSIS-NN int8 sources (Nuclei RISC-V). Mirrors third_party/cmsis_nn.mk.
# Float LayerFast is reference-only — NMSIS-NN has no f32 kernels.
NMSIS_DIR ?= third_party/NMSIS
NMSIS_NN_DIR ?= $(NMSIS_DIR)/NMSIS/NN

NMSIS_NN_CFLAGS = -std=c11 -O2 \
	-I$(NMSIS_NN_DIR)/Include \
	-I$(NMSIS_DIR)/NMSIS/Core/Include \
	$(NETKIT_ARCH_CFLAGS)

# Host smoke: portable path without nmsis_core.h intrinsics.
ifeq ($(NETKIT_HOST_SMOKE),1)
  NMSIS_NN_CFLAGS += -include third_party/nmsis_host_compat.h
endif

NMSIS_NN_S8_SOURCES := \
	$(wildcard $(NMSIS_NN_DIR)/Source/ConvolutionFunctions/riscv_*s8*.c) \
	$(wildcard $(NMSIS_NN_DIR)/Source/FullyConnectedFunctions/riscv_*s8*.c) \
	$(wildcard $(NMSIS_NN_DIR)/Source/PoolingFunctions/riscv_*s8*.c) \
	$(wildcard $(NMSIS_NN_DIR)/Source/PoolingFunctions/riscv_avgpool_s8*.c) \
	$(wildcard $(NMSIS_NN_DIR)/Source/ActivationFunctions/riscv_*s8*.c) \
	$(wildcard $(NMSIS_NN_DIR)/Source/BasicMathFunctions/riscv_*s8*.c) \
	$(wildcard $(NMSIS_NN_DIR)/Source/SoftmaxFunctions/riscv_*s8*.c) \
	$(wildcard $(NMSIS_NN_DIR)/Source/NNSupportFunctions/*.c)

# Deduplicate wildcards
NMSIS_NN_SOURCES := $(sort $(NMSIS_NN_S8_SOURCES))

NMSIS_NN_OBJECTS = $(NMSIS_NN_SOURCES:$(NMSIS_NN_DIR)/%.c=build/nmsis_nn/%.o)

build/nmsis_nn/%.o: $(NMSIS_NN_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(NMSIS_NN_CFLAGS) -c $< -o $@
