idf_component_register(
	SRCS "main.cpp"
	     "${CMAKE_CURRENT_LIST_DIR}/../generated/mnist_cnn_dw_aot.cpp"
	     "${CMAKE_CURRENT_LIST_DIR}/../../../benchmark/tflm/generated/cnn_dw/mnist_cnn_test_images.cc"
	INCLUDE_DIRS
	     "."
	     "${CMAKE_CURRENT_LIST_DIR}/../generated"
	     "${CMAKE_CURRENT_LIST_DIR}/../../../include"
	     "${CMAKE_CURRENT_LIST_DIR}/../../../benchmark/tflm/generated/cnn_dw"
	REQUIRES netkit
	PRIV_REQUIRES esp_timer
)
include("${CMAKE_CURRENT_LIST_DIR}/../../xiao-esp32s3/mcu_esp_tflm_match_compile.cmake")
netkit_esp_apply_tflm_match_compile_options(${COMPONENT_LIB})
