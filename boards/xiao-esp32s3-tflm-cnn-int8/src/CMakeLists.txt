idf_component_register(
	SRCS "main.cpp"
	     "${CMAKE_CURRENT_LIST_DIR}/../../../benchmark/tflm/generated/mnist_cnn_int8_model_data.cc"
	     "${CMAKE_CURRENT_LIST_DIR}/../../../benchmark/tflm/generated/mnist_cnn_int8_test_images.cc"
	INCLUDE_DIRS
	     "."
	     "${CMAKE_CURRENT_LIST_DIR}/../../../benchmark/tflm"
	     "${CMAKE_CURRENT_LIST_DIR}/../../../benchmark/tflm/generated"
	PRIV_REQUIRES esp_timer esp-tflite-micro
)
