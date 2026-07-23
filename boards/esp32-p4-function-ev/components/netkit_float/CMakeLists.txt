# netkit float32 runtime for mcu_esp / ESP32-P4 (FPU). ESP-NN has no float API.
set(NETKIT_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../..")

set(NETKIT_INCLUDES ${NETKIT_ROOT}/include)

set(NETKIT_SRCS
	${NETKIT_ROOT}/src/arena.cpp
	${NETKIT_ROOT}/src/nk_mmap.cpp
	${NETKIT_ROOT}/src/tensor_factory.cpp
	${NETKIT_ROOT}/src/tensor_access.cpp
	${NETKIT_ROOT}/src/kernel_workspace.cpp
	${NETKIT_ROOT}/src/cmsis_buffer_size.cpp
	${NETKIT_ROOT}/src/im2col_quant.cpp
	${NETKIT_ROOT}/src/im2col_partial.cpp
	${NETKIT_ROOT}/src/im2col_full.cpp
	${NETKIT_ROOT}/src/quant_ops.cpp
	${NETKIT_ROOT}/src/quant_softmax_s8.cpp
	${NETKIT_ROOT}/src/netkit_util.cpp
	${NETKIT_ROOT}/src/esp_nn_backend.cpp
	${NETKIT_ROOT}/src/esp_nn_quant_backend.cpp
	${NETKIT_ROOT}/src/nk_format.cpp
	${NETKIT_ROOT}/src/cmsis_quant_plan.cpp
	${NETKIT_ROOT}/src/cnn_quant.cpp
	${NETKIT_ROOT}/src/cnn.cpp
	${NETKIT_ROOT}/src/mlp.cpp
	${NETKIT_ROOT}/src/conv2d.cpp
	${NETKIT_ROOT}/src/conv2d_layout.cpp
	${NETKIT_ROOT}/src/conv_dispatch.cpp
	${NETKIT_ROOT}/src/conv1x1_kernel.cpp
	${NETKIT_ROOT}/src/conv_depthwise_kernel.cpp
	${NETKIT_ROOT}/src/conv_direct_kernel.cpp
	${NETKIT_ROOT}/src/depthwise_conv2d.cpp
	${NETKIT_ROOT}/src/ops_resolver.cpp
	${NETKIT_ROOT}/src/ops_resolver_default.cpp
	${NETKIT_ROOT}/src/layer_ops/nk_op_conv2d.cpp
	${NETKIT_ROOT}/src/layer_ops/nk_op_depthwise_conv2d.cpp
	${NETKIT_ROOT}/src/layer_ops/nk_op_convnextv2_block.cpp
	${NETKIT_ROOT}/src/layer_ops/nk_op_mobilenetv4_uib.cpp
	${NETKIT_ROOT}/src/layer_ops/nk_op_yolox_decoupled_head.cpp
	${NETKIT_ROOT}/src/layer_ops/nk_op_feature_tap.cpp
	${NETKIT_ROOT}/src/layer_ops/nk_op_yolox_pafpn.cpp
	${NETKIT_ROOT}/src/layer_ops/nk_op_resnet_basic_block.cpp
	${NETKIT_ROOT}/src/layer_ops/nk_op_layernorm2d.cpp
	${NETKIT_ROOT}/src/layer_ops/nk_op_max_pool2d.cpp
	${NETKIT_ROOT}/src/layer_ops/nk_op_avg_pool2d.cpp
	${NETKIT_ROOT}/src/layer_ops/nk_op_batch_norm2d.cpp
	${NETKIT_ROOT}/src/layer_ops/nk_op_flatten.cpp
	${NETKIT_ROOT}/src/layer_ops/nk_op_dense.cpp
	${NETKIT_ROOT}/src/mobilenetv4_uib.cpp
	${NETKIT_ROOT}/src/resnet_basic_block.cpp
	${NETKIT_ROOT}/src/convnextv2_block.cpp
	${NETKIT_ROOT}/src/yolox_decoupled_head.cpp
	${NETKIT_ROOT}/src/yolox_pafpn.cpp
	${NETKIT_ROOT}/src/nk_loader.cpp
	${NETKIT_ROOT}/src/reference_kernel.cpp
	${NETKIT_ROOT}/src/ops.cpp
	${NETKIT_ROOT}/src/cmsis_nn_quant_backend.cpp
	${NETKIT_ROOT}/src/nmsis_nn_quant_backend.cpp
)

idf_component_register(
	SRCS ${NETKIT_SRCS}
	INCLUDE_DIRS ${NETKIT_INCLUDES}
)

target_compile_definitions(${COMPONENT_LIB} PUBLIC
	NETKIT_TARGET_MCU_ESP=1
	NETKIT_USE_ESP_NN=0
	NETKIT_MCU_QUANT_ONLY=0
	NETKIT_KERNELS_OPTIMIZED_FOR_SPEED=1
	NETKIT_USE_MMAP=0
	NETKIT_QUANT_TRACE=0
	NETKIT_LOOP_UNROLL=0
	NETKIT_IM2COL=0
	NETKIT_REFERENCE_QUANT_LOOPS=1
	CONFIG_NN_OPTIMIZED=0
)

target_compile_options(${COMPONENT_LIB} PRIVATE
	$<$<COMPILE_LANGUAGE:CXX>:-std=gnu++20>
	-Wno-unused-function
	-Wno-format
	-Wno-error=format
	-Wno-error=class-memaccess
	-Wno-error=return-type
)

include("${NETKIT_ROOT}/boards/esp32-p4-function-ev/mcu_esp_tflm_match_compile.cmake")
netkit_esp_apply_tflm_match_compile_options(${COMPONENT_LIB})

set_source_files_properties(
	${NETKIT_ROOT}/src/cmsis_quant_plan.cpp
	PROPERTIES COMPILE_OPTIONS "-Wno-class-memaccess;-Wno-error=class-memaccess"
)
