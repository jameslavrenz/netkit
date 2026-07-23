# netkit lean runtime for mcu_esp / ESP32-P4-Function-EV.
# Default: embed AOT + CmsisQuantPlan → ESP-NN P4 kernels (int8 production).
# Reference peer: -DNETKIT_XIAO_USE_ESP_NN=0 → QuantOps (ESP-NN Try* stubs).
set(NETKIT_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../..")
set(ESP_NN_DIR "${NETKIT_ROOT}/third_party/ESP-NN")

if(NOT DEFINED NETKIT_XIAO_USE_ESP_NN)
	set(NETKIT_XIAO_USE_ESP_NN 1)
endif()

set(NETKIT_INCLUDES ${NETKIT_ROOT}/include)

set(NETKIT_SRCS
	${NETKIT_ROOT}/src/arena.cpp
	${NETKIT_ROOT}/src/nk_mmap.cpp
	${NETKIT_ROOT}/src/tensor_factory.cpp
	${NETKIT_ROOT}/src/tensor_access.cpp
	${NETKIT_ROOT}/src/kernel_workspace.cpp
	${NETKIT_ROOT}/src/cmsis_buffer_size.cpp
	${NETKIT_ROOT}/src/im2col_quant.cpp
	${NETKIT_ROOT}/src/quant_ops.cpp
	${NETKIT_ROOT}/src/quant_softmax_s8.cpp
	${NETKIT_ROOT}/src/netkit_util.cpp
	${NETKIT_ROOT}/src/esp_nn_backend.cpp
	${NETKIT_ROOT}/src/esp_nn_quant_backend.cpp
	${NETKIT_ROOT}/src/nk_format.cpp
	${NETKIT_ROOT}/src/cmsis_quant_plan.cpp
	${NETKIT_ROOT}/src/cnn_quant.cpp
	${NETKIT_ROOT}/src/cnn.cpp
	${NETKIT_ROOT}/src/conv2d_layout.cpp
	${NETKIT_ROOT}/src/ops_resolver.cpp
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

if(NETKIT_XIAO_USE_ESP_NN)
	# Portable ANSI/opt (C3/C6 path). P4 PIE asm is opt-in — see esp_nn.h
	# NETKIT_ESP_NN_USE_P4_ASM (PIO gas currently rejects some immediates).
	set(ESP_NN_SRCS
		${ESP_NN_DIR}/src/activation_functions/esp_nn_relu_ansi.c
		${ESP_NN_DIR}/src/activation_functions/esp_nn_hard_swish_ansi.c
		${ESP_NN_DIR}/src/common/esp_nn_mean_ansi.c
		${ESP_NN_DIR}/src/basic_math/esp_nn_add_ansi.c
		${ESP_NN_DIR}/src/basic_math/esp_nn_mul_ansi.c
		${ESP_NN_DIR}/src/convolution/esp_nn_conv_ansi.c
		${ESP_NN_DIR}/src/convolution/esp_nn_conv_opt.c
		${ESP_NN_DIR}/src/convolution/esp_nn_depthwise_conv_ansi.c
		${ESP_NN_DIR}/src/convolution/esp_nn_depthwise_conv_opt.c
		${ESP_NN_DIR}/src/fully_connected/esp_nn_fully_connected_ansi.c
		${ESP_NN_DIR}/src/softmax/esp_nn_softmax_ansi.c
		${ESP_NN_DIR}/src/softmax/esp_nn_softmax_opt.c
		${ESP_NN_DIR}/src/logistic/esp_nn_logistic_ansi.c
		${ESP_NN_DIR}/src/pooling/esp_nn_avg_pool_ansi.c
		${ESP_NN_DIR}/src/pooling/esp_nn_max_pool_ansi.c
	)
	if(DEFINED NETKIT_ESP_NN_USE_P4_ASM AND NETKIT_ESP_NN_USE_P4_ASM)
		list(APPEND ESP_NN_SRCS
			${ESP_NN_DIR}/src/common/esp_nn_mean_s8_esp32p4.c
			${ESP_NN_DIR}/src/common/esp_nn_multiply_by_quantized_mult_esp32p4.S
			${ESP_NN_DIR}/src/activation_functions/esp_nn_hard_swish_s8_esp32p4.c
			${ESP_NN_DIR}/src/activation_functions/esp_nn_relu_s8_esp32p4.c
			${ESP_NN_DIR}/src/basic_math/esp_nn_add_s8_esp32p4.c
			${ESP_NN_DIR}/src/basic_math/esp_nn_mul_s8_esp32p4.c
			${ESP_NN_DIR}/src/convolution/esp_nn_conv_esp32p4.c
			${ESP_NN_DIR}/src/convolution/esp_nn_depthwise_conv_esp32p4.c
			${ESP_NN_DIR}/src/fully_connected/esp_nn_fully_connected_s8_esp32p4.c
			${ESP_NN_DIR}/src/pooling/esp_nn_avg_pool_s8_esp32p4.c
			${ESP_NN_DIR}/src/pooling/esp_nn_max_pool_s8_esp32p4.c
			${ESP_NN_DIR}/src/softmax/esp_nn_softmax_s8_esp32p4.c
		)
	endif()
	list(APPEND NETKIT_INCLUDES ${ESP_NN_DIR}/include ${ESP_NN_DIR}/src/common)
	set(NETKIT_ESP_NN_DEF 1)
	set(NETKIT_REF_LOOPS_DEF 0)
	set(NETKIT_CONFIG_NN_OPT 1)
else()
	set(ESP_NN_SRCS)
	set(NETKIT_ESP_NN_DEF 0)
	set(NETKIT_REF_LOOPS_DEF 1)
	set(NETKIT_CONFIG_NN_OPT 0)
endif()

idf_component_register(
	SRCS ${NETKIT_SRCS} ${ESP_NN_SRCS}
	INCLUDE_DIRS ${NETKIT_INCLUDES}
)

target_compile_definitions(${COMPONENT_LIB} PUBLIC
	NETKIT_TARGET_MCU_ESP=1
	NETKIT_USE_ESP_NN=${NETKIT_ESP_NN_DEF}
	NETKIT_MCU_QUANT_ONLY=1
	NETKIT_KERNELS_OPTIMIZED_FOR_SPEED=1
	NETKIT_USE_MMAP=0
	NETKIT_QUANT_TRACE=0
	NETKIT_LOOP_UNROLL=0
	NETKIT_IM2COL=0
	NETKIT_REFERENCE_QUANT_LOOPS=${NETKIT_REF_LOOPS_DEF}
	CONFIG_NN_OPTIMIZED=${NETKIT_CONFIG_NN_OPT}
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
