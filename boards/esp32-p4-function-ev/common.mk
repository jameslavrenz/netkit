# Shared Make helpers for boards/esp32-p4-function-ev-*
NETKIT_ROOT ?= $(abspath ../..)
BOARD_COMMON ?= $(abspath ../esp32-p4-function-ev)
PIO ?= $(HOME)/.platformio/penv/bin/pio
export PATH := $(dir $(PIO)):$(PATH)

PYTHON ?= $(shell for p in python3 /usr/bin/python3 /opt/homebrew/bin/python3 /opt/anaconda3/bin/python3; do \
	command -v $$p >/dev/null 2>&1 || continue; \
	$$p -c 'import numpy, onnx' 2>/dev/null && { command -v $$p; break; }; \
	done)

PORT ?=
# Default ESP-NN production env. Reference peer A/B: PIO_ENV=esp32_p4_ev_ref
PIO_ENV ?= esp32_p4_ev

define XIAO_CHECK_PIO
	@command -v $(notdir $(PIO)) >/dev/null 2>&1 || command -v pio >/dev/null 2>&1 || \
		(echo "PlatformIO required (pio). Install: https://platformio.org/install/cli" >&2; exit 1)
endef

define XIAO_PIO_RUN
	$(PIO) run -d $(BOARD_ROOT) -e $(PIO_ENV)
endef

define XIAO_PIO_UPLOAD
	@PORT_ARG=""; \
	if [ -n "$(PORT)" ]; then PORT_ARG="--upload-port $(PORT)"; fi; \
	$(PIO) run -d $(BOARD_ROOT) -e $(PIO_ENV) -t upload $$PORT_ARG
endef

define XIAO_PIO_MONITOR
	@PORT_ARG=""; \
	if [ -n "$(PORT)" ]; then PORT_ARG="--port $(PORT)"; fi; \
	$(PIO) device monitor -d $(BOARD_ROOT) -e $(PIO_ENV) -b 115200 $$PORT_ARG
endef

define XIAO_PIO_FULLCLEAN
	$(PIO) run -d $(BOARD_ROOT) -e $(PIO_ENV) -t fullclean
endef
