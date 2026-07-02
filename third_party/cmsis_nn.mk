# CMSIS-NN float32 sources (minimal set for conv2d + max-pool used by netkit).
CMSIS_NN_DIR ?= third_party/CMSIS-NN

CMSIS_NN_CFLAGS = -std=c11 -O2 -DARM_NN_ENABLE_F32=1 -I$(CMSIS_NN_DIR)/Include $(NETKIT_ARCH_CFLAGS)

CMSIS_NN_SOURCES = \
	$(CMSIS_NN_DIR)/Source/ConvolutionFunctions/arm_convolve_f32.c \
	$(CMSIS_NN_DIR)/Source/ConvolutionFunctions/arm_convolve_1_x_n_f32.c \
	$(CMSIS_NN_DIR)/Source/ConvolutionFunctions/arm_convolve_1x1_f32.c \
	$(CMSIS_NN_DIR)/Source/NNSupportFunctions/arm_get_buffer_size_f32.c \
	$(CMSIS_NN_DIR)/Source/NNSupportFunctions/arm_nn_pack_conv_patch_f32.c \
	$(CMSIS_NN_DIR)/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_f32.c \
	$(CMSIS_NN_DIR)/Source/NNSupportFunctions/arm_nn_mat_mult_nt_n_packed_f32.c \
	$(CMSIS_NN_DIR)/Source/NNSupportFunctions/arm_nn_conv1d_k3_f32.c \
	$(CMSIS_NN_DIR)/Source/NNSupportFunctions/arm_nn_conv1d_k3_packed_f32.c \
	$(CMSIS_NN_DIR)/Source/NNSupportFunctions/arm_nn_conv1d_k5_f32.c \
	$(CMSIS_NN_DIR)/Source/NNSupportFunctions/arm_nn_conv1d_k5_packed_f32.c \
	$(CMSIS_NN_DIR)/Source/NNSupportFunctions/arm_nntables_flt.c \
	$(CMSIS_NN_DIR)/Source/FullyConnectedFunctions/arm_fully_connected_f32.c \
	$(CMSIS_NN_DIR)/Source/PoolingFunctions/arm_max_pool_f32.c \
	$(CMSIS_NN_DIR)/Source/NNSupportFunctions/arm_nn_maxpool1d_f32.c \
	$(CMSIS_NN_DIR)/Source/ActivationFunctions/arm_nn_activation_f32.c \
	$(CMSIS_NN_DIR)/Source/SoftmaxFunctions/arm_softmax_f32.c \
	$(CMSIS_NN_DIR)/Source/BasicMathFunctions/arm_elementwise_add_f32.c

CMSIS_NN_OBJECTS = $(CMSIS_NN_SOURCES:$(CMSIS_NN_DIR)/%.c=build/cmsis_nn/%.o)

build/cmsis_nn/%.o: $(CMSIS_NN_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CMSIS_NN_CFLAGS) -c $< -o $@
