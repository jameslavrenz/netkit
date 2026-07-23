# netkit lean runtime + ESP-NN for mcu_esp / ESP32C3.
set(NETKIT_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../..")
set(ESP_NN_DIR "${NETKIT_ROOT}/third_party/ESP-NN")

# Quant-lowered AOT path (no .nk interpreter / CmsisQuantPlan runtime).
set(NETKIT_SRCS
	${NETKIT_ROOT}/src/arena.cpp
	${NETKIT_ROOT}/src/nk_mmap.cpp
	${NETKIT_ROOT}/src/tensor_factory.cpp
	${NETKIT_ROOT}/src/tensor_access.cpp
	${NETKIT_ROOT}/src/kernel_workspace.cpp
	${NETKIT_ROOT}/src/cmsis_buffer_size.cpp
	${NETKIT_ROOT}/src/reference_kernel.cpp
	${NETKIT_ROOT}/src/ops.cpp
	${NETKIT_ROOT}/src/im2col_quant.cpp
	${NETKIT_ROOT}/src/quant_ops.cpp
	${NETKIT_ROOT}/src/quant_softmax_s8.cpp
	${NETKIT_ROOT}/src/netkit_util.cpp
	${NETKIT_ROOT}/src/esp_nn_backend.cpp
	${NETKIT_ROOT}/src/esp_nn_quant_backend.cpp
	${NETKIT_ROOT}/src/nk_format.cpp
)

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

idf_component_register(
	SRCS ${NETKIT_SRCS} ${ESP_NN_SRCS}
	INCLUDE_DIRS
		${NETKIT_ROOT}/include
		${ESP_NN_DIR}/include
		${ESP_NN_DIR}/src/common
)

target_compile_definitions(${COMPONENT_LIB} PUBLIC
	NETKIT_TARGET_MCU_ESP=1
	NETKIT_USE_ESP_NN=1
	NETKIT_MCU_QUANT_ONLY=1
	NETKIT_KERNELS_OPTIMIZED_FOR_SPEED=1
	NETKIT_USE_MMAP=0
	NETKIT_QUANT_TRACE=0
	NETKIT_LOOP_UNROLL=0
	NETKIT_IM2COL=0
	NETKIT_REFERENCE_QUANT_LOOPS=0
	CONFIG_NN_OPTIMIZED=1
)

target_compile_options(${COMPONENT_LIB} PRIVATE
	$<$<COMPILE_LANGUAGE:CXX>:-std=gnu++20>
	-Wno-unused-function
	# RV32: uint32_t is long; ESP-IDF -Werror=format rejects portable %u.
	-Wno-format
	-Wno-error=format
)

# Peer A/B: match esp-tflite-micro C++ speed flags (-O3, no RTTI/exceptions…).
# ESP-NN .c stays at IDF -O2 (same as TFLM's esp-nn component).
include("${NETKIT_ROOT}/boards/xiao-esp32c3/mcu_esp_tflm_match_compile.cmake")
netkit_esp_apply_tflm_match_compile_options(${COMPONENT_LIB})
