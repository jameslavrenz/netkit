# netkit TFLM MNIST benchmarks — included via additional_tests.inc

# onnx2tf-generated model headers land under gen/generated/ (see generate_cc_arrays.py).
INCLUDES += -I$(BASE_GENDIR)
INCLUDES += -I$(EXTERNAL_DIR)/../common

MNIST_MLP_BENCHMARK_SRCS := \
$(EXTERNAL_DIR)/src/main.cc \
$(EXTERNAL_DIR)/generated/mnist_test_images.cc

MNIST_MLP_BENCHMARK_HDRS := \
$(EXTERNAL_DIR)/generated/mnist_test_images.h

MNIST_MLP_BENCHMARK_GENERATOR_INPUTS := \
$(EXTERNAL_DIR)/generated/mnist_mlp.tflite

$(eval $(call microlite_test,mnist_mlp_benchmark,\
$(MNIST_MLP_BENCHMARK_SRCS),$(MNIST_MLP_BENCHMARK_HDRS),$(MNIST_MLP_BENCHMARK_GENERATOR_INPUTS)))

MNIST_CNN_BENCHMARK_SRCS := \
$(EXTERNAL_DIR)/src/mnist_cnn_main.cc \
$(EXTERNAL_DIR)/generated/mnist_cnn_test_images.cc

MNIST_CNN_BENCHMARK_HDRS := \
$(EXTERNAL_DIR)/generated/mnist_cnn_test_images.h

MNIST_CNN_BENCHMARK_GENERATOR_INPUTS := \
$(EXTERNAL_DIR)/generated/mnist_cnn.tflite

$(eval $(call microlite_test,mnist_cnn_benchmark,\
$(MNIST_CNN_BENCHMARK_SRCS),$(MNIST_CNN_BENCHMARK_HDRS),$(MNIST_CNN_BENCHMARK_GENERATOR_INPUTS)))

MOBILENETV4_BENCHMARK_SRCS := \
$(EXTERNAL_DIR)/src/mobilenetv4_main.cc \
$(EXTERNAL_DIR)/generated/mnist_cnn_test_images.cc

MOBILENETV4_BENCHMARK_HDRS := \
$(EXTERNAL_DIR)/generated/mnist_cnn_test_images.h

MOBILENETV4_BENCHMARK_GENERATOR_INPUTS := \
$(EXTERNAL_DIR)/generated/mobilenetv4_small.tflite

$(eval $(call microlite_test,mobilenetv4_benchmark,\
$(MOBILENETV4_BENCHMARK_SRCS),$(MOBILENETV4_BENCHMARK_HDRS),$(MOBILENETV4_BENCHMARK_GENERATOR_INPUTS)))

MOBILENETV4_IMAGENET_BENCHMARK_SRCS := \
$(EXTERNAL_DIR)/src/mobilenetv4_imagenet_main.cc \
$(EXTERNAL_DIR)/generated/imagenet_mnv4_test_images.cc

MOBILENETV4_IMAGENET_BENCHMARK_HDRS := \
$(EXTERNAL_DIR)/generated/imagenet_mnv4_test_images.h

MOBILENETV4_IMAGENET_BENCHMARK_GENERATOR_INPUTS := \
$(EXTERNAL_DIR)/generated/mobilenetv4_imagenet_f32.tflite

$(eval $(call microlite_test,mobilenetv4_imagenet_benchmark,\
$(MOBILENETV4_IMAGENET_BENCHMARK_SRCS),$(MOBILENETV4_IMAGENET_BENCHMARK_HDRS),$(MOBILENETV4_IMAGENET_BENCHMARK_GENERATOR_INPUTS)))

MOBILENETV4_INT8_BENCHMARK_SRCS := \
$(EXTERNAL_DIR)/src/mobilenetv4_int8_main.cc \
$(EXTERNAL_DIR)/generated/mobilenetv4_int8_test_images.cc

MOBILENETV4_INT8_BENCHMARK_HDRS := \
$(EXTERNAL_DIR)/generated/mobilenetv4_int8_test_images.h

MOBILENETV4_INT8_BENCHMARK_GENERATOR_INPUTS := \
$(EXTERNAL_DIR)/generated/mobilenetv4_small_int8.tflite

$(eval $(call microlite_test,mobilenetv4_int8_benchmark,\
$(MOBILENETV4_INT8_BENCHMARK_SRCS),$(MOBILENETV4_INT8_BENCHMARK_HDRS),$(MOBILENETV4_INT8_BENCHMARK_GENERATOR_INPUTS)))

MOBILENETV4_IMAGENET_INT8_BENCHMARK_SRCS := \
$(EXTERNAL_DIR)/src/mobilenetv4_imagenet_int8_main.cc \
$(EXTERNAL_DIR)/generated/imagenet_mnv4_int8_test_images.cc

MOBILENETV4_IMAGENET_INT8_BENCHMARK_HDRS := \
$(EXTERNAL_DIR)/generated/imagenet_mnv4_int8_test_images.h

MOBILENETV4_IMAGENET_INT8_BENCHMARK_GENERATOR_INPUTS := \
$(EXTERNAL_DIR)/generated/mobilenetv4_imagenet_int8.tflite

$(eval $(call microlite_test,mobilenetv4_imagenet_int8_benchmark,\
$(MOBILENETV4_IMAGENET_INT8_BENCHMARK_SRCS),$(MOBILENETV4_IMAGENET_INT8_BENCHMARK_HDRS),$(MOBILENETV4_IMAGENET_INT8_BENCHMARK_GENERATOR_INPUTS)))

# Profile build only — not linked into mnist_cnn_benchmark.
MNIST_CNN_PROFILE_BENCHMARK_SRCS := \
$(EXTERNAL_DIR)/src/mnist_cnn_profile_main.cc \
$(EXTERNAL_DIR)/generated/mnist_cnn_test_images.cc

MNIST_CNN_PROFILE_BENCHMARK_HDRS := \
$(EXTERNAL_DIR)/generated/mnist_cnn_test_images.h

MNIST_CNN_PROFILE_BENCHMARK_GENERATOR_INPUTS := \
$(EXTERNAL_DIR)/generated/mnist_cnn.tflite

$(eval $(call microlite_test,mnist_cnn_profile_benchmark,\
$(MNIST_CNN_PROFILE_BENCHMARK_SRCS),$(MNIST_CNN_PROFILE_BENCHMARK_HDRS),$(MNIST_CNN_PROFILE_BENCHMARK_GENERATOR_INPUTS)))

# Profile build only — not linked into mnist_mlp_benchmark.
MNIST_MLP_PROFILE_BENCHMARK_SRCS := \
$(EXTERNAL_DIR)/src/mnist_mlp_profile_main.cc \
$(EXTERNAL_DIR)/generated/mnist_test_images.cc

MNIST_MLP_PROFILE_BENCHMARK_HDRS := \
$(EXTERNAL_DIR)/generated/mnist_test_images.h

MNIST_MLP_PROFILE_BENCHMARK_GENERATOR_INPUTS := \
$(EXTERNAL_DIR)/generated/mnist_mlp.tflite

$(eval $(call microlite_test,mnist_mlp_profile_benchmark,\
$(MNIST_MLP_PROFILE_BENCHMARK_SRCS),$(MNIST_MLP_PROFILE_BENCHMARK_HDRS),$(MNIST_MLP_PROFILE_BENCHMARK_GENERATOR_INPUTS)))
