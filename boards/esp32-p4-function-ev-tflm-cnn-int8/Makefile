# Seeed ESP32-P4-Function-EV — TFLM MNIST CNN int8 (esp-tflite-micro + ESP-NN).
NETKIT_ROOT := $(abspath ../..)
BOARD_ROOT := $(abspath .)
BOARD_COMMON := $(abspath ../esp32-p4-function-ev)
include $(BOARD_COMMON)/common.mk

.PHONY: all check-deps build flash monitor run clean fullclean

all: check-deps build

check-deps:
	$(XIAO_CHECK_PIO)
	@test -f $(BOARD_ROOT)/components/esp-tflite-micro/CMakeLists.txt || \
		$(BOARD_COMMON)/scripts/fetch_esp_tflite_micro.sh $(BOARD_ROOT)/components/esp-tflite-micro
	@test -f $(NETKIT_ROOT)/benchmark/tflm/generated/mnist_cnn_int8_model_data.cc || \
		(echo "Missing model data — run: make -C $(NETKIT_ROOT)/benchmark/tflm export-cnn-int8" >&2; exit 1)
	@test -f $(NETKIT_ROOT)/benchmark/tflm/generated/mnist_cnn_int8_test_images.cc || \
		(echo "Missing test images" >&2; exit 1)

build: check-deps
	@# Managed esp-nn is fetched on first configure; patch + drop cmake cache once.
	@if [ ! -d $(BOARD_ROOT)/managed_components/espressif__esp-nn ]; then \
		$(XIAO_PIO_RUN) || true; \
	fi
	@$(BOARD_COMMON)/scripts/patch_tflm_esp_nn_portable.sh $(BOARD_ROOT)
	@if [ ! -f $(BOARD_ROOT)/.esp_nn_p4_portable_stamp ]; then \
		rm -rf $(BOARD_ROOT)/.pio/build; \
		touch $(BOARD_ROOT)/.esp_nn_p4_portable_stamp; \
	fi
	$(XIAO_PIO_RUN)

flash:
	$(XIAO_PIO_UPLOAD)

monitor:
	$(XIAO_PIO_MONITOR)

run: flash
	@$(MAKE) monitor

clean:
	rm -rf $(BOARD_ROOT)/.pio $(BOARD_ROOT)/managed_components $(BOARD_ROOT)/dependencies.lock \
		$(BOARD_ROOT)/.esp_nn_p4_portable_stamp

fullclean:
	$(XIAO_PIO_FULLCLEAN)
	rm -rf $(BOARD_ROOT)/.pio $(BOARD_ROOT)/.esp_nn_p4_portable_stamp
