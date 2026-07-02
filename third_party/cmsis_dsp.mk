# CMSIS-DSP float32 sources (basic vector math + matrix ops used by netkit).
CMSIS_DSP_DIR ?= third_party/CMSIS-DSP

CMSIS_DSP_CFLAGS = -std=c11 -O2 \
	-I$(CMSIS_DSP_DIR)/Include \
	-I$(CMSIS_DSP_DIR)/PrivateInclude \
	$(NETKIT_ARCH_CFLAGS)

CMSIS_DSP_SOURCES = \
	$(CMSIS_DSP_DIR)/Source/BasicMathFunctions/arm_add_f32.c \
	$(CMSIS_DSP_DIR)/Source/BasicMathFunctions/arm_mult_f32.c \
	$(CMSIS_DSP_DIR)/Source/BasicMathFunctions/arm_scale_f32.c \
	$(CMSIS_DSP_DIR)/Source/BasicMathFunctions/arm_clip_f32.c \
	$(CMSIS_DSP_DIR)/Source/MatrixFunctions/arm_mat_init_f32.c \
	$(CMSIS_DSP_DIR)/Source/MatrixFunctions/arm_mat_vec_mult_f32.c \
	$(CMSIS_DSP_DIR)/Source/MatrixFunctions/arm_mat_mult_f32.c

CMSIS_DSP_OBJECTS = $(CMSIS_DSP_SOURCES:$(CMSIS_DSP_DIR)/%.c=build/cmsis_dsp/%.o)

build/cmsis_dsp/%.o: $(CMSIS_DSP_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CMSIS_DSP_CFLAGS) -c $< -o $@
