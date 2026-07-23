# Seeed XIAO ESP32S3 — TFLM MNIST DS-CNN int8 (esp-tflite-micro + ESP-NN).
NETKIT_ROOT := $(abspath ../..)
BOARD_ROOT := $(abspath .)
BOARD_COMMON := $(abspath ../xiao-esp32s3)
include $(BOARD_COMMON)/common.mk

.PHONY: all check-deps build flash monitor run clean fullclean

all: check-deps build

check-deps:
	$(XIAO_CHECK_PIO)
	@test -f $(BOARD_ROOT)/components/esp-tflite-micro/CMakeLists.txt || \
		$(BOARD_COMMON)/scripts/fetch_esp_tflite_micro.sh \
			$(NETKIT_ROOT)/boards/xiao-esp32s3-tflm-cnn-int8/components/esp-tflite-micro
	@test -L $(BOARD_ROOT)/components/esp-tflite-micro -o -f $(BOARD_ROOT)/components/esp-tflite-micro/CMakeLists.txt || \
		ln -sfn $(NETKIT_ROOT)/boards/xiao-esp32s3-tflm-cnn-int8/components/esp-tflite-micro \
			$(BOARD_ROOT)/components/esp-tflite-micro
	@test -f $(NETKIT_ROOT)/benchmark/tflm/generated/mnist_cnn_dw_int8_model_data.cc || \
		(echo "Missing model data — see nucleo-f446re-tflm-cnn-dw-int8 Makefile export-cnn-dw-int8-model-data" >&2; exit 1)
	@test -f $(NETKIT_ROOT)/benchmark/tflm/generated/cnn_dw/mnist_cnn_int8_test_images.cc || \
		(echo "Missing cnn_dw test images" >&2; exit 1)

build:
	$(XIAO_PIO_RUN)

flash:
	$(XIAO_PIO_UPLOAD)

monitor:
	$(XIAO_PIO_MONITOR)

run: flash
	@$(MAKE) monitor

clean:
	rm -rf $(BOARD_ROOT)/.pio $(BOARD_ROOT)/managed_components $(BOARD_ROOT)/dependencies.lock

fullclean:
	$(XIAO_PIO_FULLCLEAN)
	rm -rf $(BOARD_ROOT)/.pio
