# Shared arm-none-eabi GCC toolchain for NUCLEO-F446RE board firmware.
#
# Matches TFLM microlite: TOOLCHAIN := gcc (arm-none-eabi-gcc / arm-none-eabi-g++).
# Not armclang / llvm. Both netkit and TFLM board builds pass the same
# TARGET_TOOLCHAIN_ROOT to the TFLM microlite Makefile.
#
# Before include, set:
#   MCU_BOARD_COMMON := $(abspath path/to/boards/nucleo-f446re)

ifndef MCU_BOARD_COMMON
$(error MCU_BOARD_COMMON must be set to boards/nucleo-f446re before including mcu_gcc_toolchain.mk)
endif

MCU_TOOLCHAIN := gcc

TOOLCHAIN_BIN := $(shell $(MCU_BOARD_COMMON)/scripts/setup-toolchain.sh 2>/dev/null)
ifeq ($(strip $(TOOLCHAIN_BIN)),)
$(error arm-none-eabi GCC not found — run: $(MCU_BOARD_COMMON)/scripts/setup-toolchain.sh)
endif

CC := $(TOOLCHAIN_BIN)/arm-none-eabi-gcc
CXX := $(TOOLCHAIN_BIN)/arm-none-eabi-g++
AR := $(TOOLCHAIN_BIN)/arm-none-eabi-ar
OBJCOPY := $(TOOLCHAIN_BIN)/arm-none-eabi-objcopy
SIZE := $(TOOLCHAIN_BIN)/arm-none-eabi-size

# Fail early if PATH or ARM_GNU_TOOLCHAIN pointed at clang/llvm instead of GCC.
define MCU_ASSERT_GCC_TOOLCHAIN
	@if ! $(CC) --version 2>&1 | grep -qi 'gcc'; then \
	  echo "MCU toolchain must be arm-none-eabi-gcc (TFLM TOOLCHAIN=gcc), not:" >&2; \
	  $(CC) --version 2>&1 | head -1 >&2; \
	  exit 1; \
	fi
	@if $(CC) --version 2>&1 | grep -Eiq 'clang|llvm'; then \
	  echo "Refusing clang/llvm — use arm-none-eabi-gcc from $(MCU_BOARD_COMMON)/scripts/setup-toolchain.sh" >&2; \
	  exit 1; \
	fi
endef
