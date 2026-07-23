# ESP32-P4-Function-EV — TFLM MNIST DS-CNN float32.
NETKIT_ROOT := $(abspath ../..)
BOARD_ROOT := $(abspath .)
BOARD_COMMON := $(abspath ../esp32-p4-function-ev)
GEN_DIR := $(BOARD_ROOT)/generated
include $(BOARD_COMMON)/common.mk

.PHONY: all check-deps export-model-data build flash monitor run clean fullclean

all: check-deps export-model-data build

check-deps:
	$(XIAO_CHECK_PIO)
	@test -f $(BOARD_ROOT)/components/esp-tflite-micro/CMakeLists.txt || \
		$(BOARD_COMMON)/scripts/fetch_esp_tflite_micro.sh \
			$(NETKIT_ROOT)/boards/esp32-p4-function-ev-tflm-cnn/components/esp-tflite-micro
	@mkdir -p $(BOARD_ROOT)/components
	@test -L $(BOARD_ROOT)/components/esp-tflite-micro -o -f $(BOARD_ROOT)/components/esp-tflite-micro/CMakeLists.txt || \
		ln -sfn $(NETKIT_ROOT)/boards/esp32-p4-function-ev-tflm-cnn/components/esp-tflite-micro \
			$(BOARD_ROOT)/components/esp-tflite-micro
	@test -f $(NETKIT_ROOT)/benchmark/tflm/generated/mnist_cnn_dw.tflite || (echo "Missing tflite" >&2; exit 1)
	@test -f $(NETKIT_ROOT)/benchmark/tflm/generated/cnn_dw/mnist_cnn_test_images.cc || (echo "Missing float test images" >&2; exit 1)

export-model-data: $(GEN_DIR)/mnist_cnn_dw_model_data.cc

$(GEN_DIR)/mnist_cnn_dw_model_data.cc: $(NETKIT_ROOT)/benchmark/tflm/generated/mnist_cnn_dw.tflite
	@mkdir -p $(GEN_DIR)
	$(PYTHON) $(BOARD_COMMON)/scripts/tflite_to_model_data.py $(NETKIT_ROOT)/benchmark/tflm/generated/mnist_cnn_dw.tflite \
		--stem mnist_cnn_dw --array g_mnist_cnn_dw_model_data -o $(GEN_DIR)

build: check-deps export-model-data
	@if [ ! -d $(BOARD_ROOT)/managed_components/espressif__esp-nn ]; then \
		$(XIAO_PIO_RUN) || true; \
	fi
	@$(BOARD_COMMON)/scripts/patch_tflm_esp_nn_portable.sh $(BOARD_ROOT)
	@if [ ! -f $(BOARD_ROOT)/.esp_nn_p4_portable_stamp ]; then \
		rm -rf $(BOARD_ROOT)/.pio/build; \
		touch $(BOARD_ROOT)/.esp_nn_p4_portable_stamp; \
	fi
	$(XIAO_PIO_RUN)

flash: export-model-data
	$(XIAO_PIO_UPLOAD)

monitor:
	$(XIAO_PIO_MONITOR)

run: flash
	@$(MAKE) monitor

clean:
	rm -rf $(BOARD_ROOT)/.pio $(BOARD_ROOT)/managed_components $(BOARD_ROOT)/dependencies.lock \
		$(BOARD_ROOT)/.esp_nn_p4_portable_stamp $(GEN_DIR)

fullclean:
	$(XIAO_PIO_FULLCLEAN)
	rm -rf $(BOARD_ROOT)/.pio $(BOARD_ROOT)/.esp_nn_p4_portable_stamp
