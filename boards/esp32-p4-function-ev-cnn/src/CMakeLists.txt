idf_component_register(
	SRCS "main.cpp"
	     "${CMAKE_CURRENT_LIST_DIR}/../generated/mnist_cnn_aot.cpp"
	     "${CMAKE_CURRENT_LIST_DIR}/../../../benchmark/tflm/generated/mnist_cnn_test_images.cc"
	INCLUDE_DIRS
	     "."
	     "${CMAKE_CURRENT_LIST_DIR}/../generated"
	     "${CMAKE_CURRENT_LIST_DIR}/../../../include"
	     "${CMAKE_CURRENT_LIST_DIR}/../../../benchmark/tflm/generated"
	REQUIRES netkit
	PRIV_REQUIRES esp_timer
)
include("${CMAKE_CURRENT_LIST_DIR}/../../esp32-p4-function-ev/mcu_esp_tflm_match_compile.cmake")
netkit_esp_apply_tflm_match_compile_options(${COMPONENT_LIB})
