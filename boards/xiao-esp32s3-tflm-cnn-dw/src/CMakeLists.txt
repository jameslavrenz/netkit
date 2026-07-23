idf_component_register(
	SRCS "main.cpp"
	     "${CMAKE_CURRENT_LIST_DIR}/../generated/mnist_cnn_dw_model_data.cc"
	     "${CMAKE_CURRENT_LIST_DIR}/../../../benchmark/tflm/generated/cnn_dw/mnist_cnn_test_images.cc"
	INCLUDE_DIRS
	     "."
	     "${CMAKE_CURRENT_LIST_DIR}/../generated"
	     "${CMAKE_CURRENT_LIST_DIR}/../../../benchmark/tflm/generated/cnn_dw"
	PRIV_REQUIRES esp_timer esp-tflite-micro
)
