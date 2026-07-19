cmake_minimum_required(VERSION 3.15)
cmake_policy(SET CMP0077 NEW)

project(netkit VERSION 0.1.0 LANGUAGES C CXX)

include(cmake/netkit_environment.cmake)

set(NETKIT_TARGET "${_netkit_target_default}" CACHE STRING
    "Build profile: cpu, mcu_arm, mpu_arm, mcu_risc, mpu_risc, or mcu_esp")
set_property(CACHE NETKIT_TARGET PROPERTY STRINGS cpu mcu_arm mpu_arm mcu_risc mpu_risc mcu_esp)

# CMSIS-NN / ESP-NN / XNNPACK defaults follow NETKIT_TARGET profile; override on the cmake command line.
# CMSIS-DSP is not used as a netkit backend.
if(NETKIT_TARGET STREQUAL "cpu")
    set(_netkit_cmsis_nn_default OFF)
    set(_netkit_esp_nn_default OFF)
    set(_netkit_xnnpack_default ON)
elseif(NETKIT_TARGET STREQUAL "mcu_arm")
    set(_netkit_cmsis_nn_default ON)
    set(_netkit_esp_nn_default OFF)
    set(_netkit_xnnpack_default OFF)
elseif(NETKIT_TARGET STREQUAL "mpu_arm")
    set(_netkit_cmsis_nn_default OFF)
    set(_netkit_esp_nn_default OFF)
    set(_netkit_xnnpack_default ON)
elseif(NETKIT_TARGET STREQUAL "mcu_risc")
    set(_netkit_cmsis_nn_default OFF)
    set(_netkit_esp_nn_default OFF)
    set(_netkit_nmsis_nn_default ON)
    set(_netkit_xnnpack_default OFF)
elseif(NETKIT_TARGET STREQUAL "mpu_risc")
    set(_netkit_cmsis_nn_default OFF)
    set(_netkit_esp_nn_default OFF)
    set(_netkit_nmsis_nn_default OFF)
    set(_netkit_xnnpack_default ON)
elseif(NETKIT_TARGET STREQUAL "mcu_esp")
    set(_netkit_cmsis_nn_default OFF)
    set(_netkit_esp_nn_default ON)
    set(_netkit_nmsis_nn_default OFF)
    set(_netkit_xnnpack_default OFF)
elseif(NETKIT_TARGET STREQUAL "mcu" OR NETKIT_TARGET STREQUAL "mpu")
    message(FATAL_ERROR "NETKIT_TARGET=${NETKIT_TARGET} is removed — use mcu_arm or mpu_arm")
else()
    set(_netkit_cmsis_nn_default OFF)
    set(_netkit_esp_nn_default OFF)
    set(_netkit_nmsis_nn_default OFF)
    set(_netkit_xnnpack_default OFF)
endif()

# Ensure NMSIS default is defined for all branches above.
if(NOT DEFINED _netkit_nmsis_nn_default)
    set(_netkit_nmsis_nn_default OFF)
endif()

option(NETKIT_CMSIS_NN "Use ARM CMSIS-NN backends (mcu_arm + Cortex-M)" ${_netkit_cmsis_nn_default})
option(NETKIT_ESP_NN "Use Espressif ESP-NN backends (mcu_esp; int8 production)" ${_netkit_esp_nn_default})
option(NETKIT_NMSIS_NN "Use Nuclei NMSIS-NN backends (mcu_risc; int8 production)" ${_netkit_nmsis_nn_default})
option(NETKIT_XNNPACK "Use Google XNNPACK LayerFast backends (cpu + any MPU; forbidden on MCU)" ${_netkit_xnnpack_default})

# Match Makefile: CI forces accel backends off so default CI stays on reference kernels.
if(DEFINED ENV{GITHUB_ACTIONS} AND "$ENV{GITHUB_ACTIONS}" STREQUAL "true")
    if(NETKIT_XNNPACK)
        message(STATUS "GITHUB_ACTIONS=true — forcing NETKIT_XNNPACK=OFF for reference-kernel CI")
        set(NETKIT_XNNPACK OFF CACHE BOOL "Use Google XNNPACK LayerFast backends (cpu + any MPU; forbidden on MCU)" FORCE)
    endif()
    if(NETKIT_NMSIS_NN)
        message(STATUS "GITHUB_ACTIONS=true — forcing NETKIT_NMSIS_NN=OFF for reference-kernel CI")
        set(NETKIT_NMSIS_NN OFF CACHE BOOL "Use Nuclei NMSIS-NN backends (mcu_risc; int8 production)" FORCE)
    endif()
endif()

option(NETKIT_GLOBAL_ARENA "Use static/global arena on CPU builds" OFF)
option(NETKIT_HEAP_ARENA "Compile heap arena helpers on MPU class builds (forbidden on MCU)" OFF)
# Optional override of NK_ARENA_DEFAULT_CAPACITY (bytes). 0 = use class default
# (MCU 64 KiB; CPU/MPU 64 MiB). Alternate: NETKIT_ARENA_KB (KiB).
set(NETKIT_ARENA_CAPACITY "0" CACHE STRING "Override NK_ARENA_DEFAULT_CAPACITY in bytes (0 = class default)")
set(NETKIT_ARENA_KB "0" CACHE STRING "Override NK_ARENA_DEFAULT_CAPACITY in KiB (0 = class default; ignored if CAPACITY set)")
# mmap: default ON for cpu + any MPU; forbidden on MCU. Opt out with -DNETKIT_MMAP=OFF
# on RTOS / bare-metal MPU (no virtual-memory OS).
if(NETKIT_TARGET STREQUAL "mcu_arm" OR NETKIT_TARGET STREQUAL "mcu_risc" OR NETKIT_TARGET STREQUAL "mcu_esp")
    set(_netkit_mmap_default OFF)
else()
    set(_netkit_mmap_default ON)
endif()
option(NETKIT_MMAP "mmap for .nk file loads (cpu + MPU default; forbidden on MCU; macOS/Linux/Windows)" ${_netkit_mmap_default})
if(NETKIT_TARGET STREQUAL "mcu_arm" OR NETKIT_TARGET STREQUAL "mcu_risc" OR NETKIT_TARGET STREQUAL "mcu_esp")
    if(NETKIT_MMAP)
        message(WARNING "NETKIT_MMAP=ON forced off — mmap is forbidden on MCU")
    endif()
    set(NETKIT_MMAP OFF CACHE BOOL "mmap for .nk file loads (cpu + MPU default; forbidden on MCU; macOS/Linux/Windows)" FORCE)
endif()
set(NETKIT_IM2COL "0" CACHE STRING "Conv2D strategy (float + int8 QuantOps): 0 direct (default), 1 partial, 2 full im2col+GEMM")
option(NETKIT_LOOP_UNROLL "EXPERIMENTAL: 4x loop unroll in netkit reference kernels (increases .text)" OFF)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(NETKIT_LAYER_OP_SOURCES
    src/layer_ops/nk_op_conv2d.cpp
    src/layer_ops/nk_op_depthwise_conv2d.cpp
    src/layer_ops/nk_op_convnextv2_block.cpp
    src/layer_ops/nk_op_mobilenetv4_uib.cpp
    src/layer_ops/nk_op_yolox_decoupled_head.cpp
    src/layer_ops/nk_op_feature_tap.cpp
    src/layer_ops/nk_op_yolox_pafpn.cpp
    src/layer_ops/nk_op_resnet_basic_block.cpp
    src/layer_ops/nk_op_layernorm2d.cpp
    src/layer_ops/nk_op_max_pool2d.cpp
    src/layer_ops/nk_op_avg_pool2d.cpp
    src/layer_ops/nk_op_batch_norm2d.cpp
    src/layer_ops/nk_op_flatten.cpp
    src/layer_ops/nk_op_dense.cpp
)

set(NETKIT_RUNTIME_SOURCES
    src/arena.cpp
    src/nk_mmap.cpp
    src/tensor_factory.cpp
    src/tensor_access.cpp
    src/reference_kernel.cpp
    src/kernel_workspace.cpp
    src/cmsis_buffer_size.cpp
    src/ops.cpp
    src/conv2d.cpp
    src/conv2d_layout.cpp
    src/conv_dispatch.cpp
    src/conv1x1_kernel.cpp
    src/conv_depthwise_kernel.cpp
    src/conv_direct_kernel.cpp
    src/im2col_partial.cpp
    src/im2col_full.cpp
    src/im2col_quant.cpp
    src/depthwise_conv2d.cpp
    src/convnextv2_block.cpp
    src/mobilenetv4_uib.cpp
    src/yolox_decoupled_head.cpp
    src/yolox_pafpn.cpp
    src/resnet_basic_block.cpp
    src/mlp.cpp
    src/quant_ops.cpp
    src/quant_softmax_s8.cpp
    src/netkit_util.cpp
    src/quant_trace.cpp
    src/cmsis_nn_quant_backend.cpp
    src/esp_nn_quant_backend.cpp
    src/nmsis_nn_quant_backend.cpp
    src/cmsis_quant_plan.cpp
    src/cnn_quant.cpp
    src/xnnpack_quant_backend.cpp
    src/cnn.cpp
    ${NETKIT_LAYER_OP_SOURCES}
    src/ops_resolver.cpp
    src/ops_resolver_default.cpp
    src/nk_format.cpp
    src/nk_loader.cpp
    src/netkit_api.cpp
)

set(NETKIT_DESKTOP_SOURCES
    src/nk_regression.cpp
    src/cli.cpp
    src/test.cpp
)

if(NETKIT_TARGET STREQUAL "cpu")
    set(NETKIT_CORE_SOURCES ${NETKIT_RUNTIME_SOURCES} ${NETKIT_DESKTOP_SOURCES})
    set(NETKIT_BUILD_CLI ON)
    set(NETKIT_BUILD_C_TESTS ON)
elseif(NETKIT_TARGET STREQUAL "mcu_arm" OR NETKIT_TARGET STREQUAL "mpu_arm" OR
       NETKIT_TARGET STREQUAL "mcu_risc" OR NETKIT_TARGET STREQUAL "mpu_risc" OR
       NETKIT_TARGET STREQUAL "mcu_esp")
    set(NETKIT_CORE_SOURCES ${NETKIT_RUNTIME_SOURCES})
    set(NETKIT_BUILD_CLI OFF)
    set(NETKIT_BUILD_C_TESTS OFF)
else()
    message(FATAL_ERROR
        "NETKIT_TARGET must be cpu, mcu_arm, mpu_arm, mcu_risc, mpu_risc, or mcu_esp (got '${NETKIT_TARGET}')")
endif()

add_library(netkit STATIC ${NETKIT_CORE_SOURCES})
add_library(netkit::netkit ALIAS netkit)

target_include_directories(netkit PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_options(netkit PRIVATE
    $<$<CXX_COMPILER_ID:Clang,AppleClang>:-fcolor-diagnostics>
    $<$<CXX_COMPILER_ID:Clang,AppleClang>:-fansi-escape-codes>
    -Wall
    -Wextra
    -g
)

if(NETKIT_TARGET STREQUAL "cpu")
    target_compile_definitions(netkit PUBLIC NETKIT_TARGET_CPU=1)
    if(NETKIT_GLOBAL_ARENA)
        target_compile_definitions(netkit PUBLIC NETKIT_GLOBAL_ARENA=1)
    endif()
elseif(NETKIT_TARGET STREQUAL "mcu_arm")
    target_compile_definitions(netkit PUBLIC NETKIT_TARGET_MCU_ARM=1)
    if(NETKIT_HEAP_ARENA)
        message(FATAL_ERROR "NETKIT_HEAP_ARENA is forbidden on MCU — use a static/global arena buffer")
    endif()
elseif(NETKIT_TARGET STREQUAL "mpu_arm")
    target_compile_definitions(netkit PUBLIC NETKIT_TARGET_MPU_ARM=1)
    if(NETKIT_HEAP_ARENA)
        target_compile_definitions(netkit PUBLIC NETKIT_HEAP_ARENA=1)
    endif()
elseif(NETKIT_TARGET STREQUAL "mcu_risc")
    target_compile_definitions(netkit PUBLIC NETKIT_TARGET_MCU_RISC=1)
    if(NETKIT_HEAP_ARENA)
        message(FATAL_ERROR "NETKIT_HEAP_ARENA is forbidden on MCU — use a static/global arena buffer")
    endif()
elseif(NETKIT_TARGET STREQUAL "mpu_risc")
    target_compile_definitions(netkit PUBLIC NETKIT_TARGET_MPU_RISC=1)
    if(NETKIT_HEAP_ARENA)
        target_compile_definitions(netkit PUBLIC NETKIT_HEAP_ARENA=1)
    endif()
elseif(NETKIT_TARGET STREQUAL "mcu_esp")
    target_compile_definitions(netkit PUBLIC NETKIT_TARGET_MCU_ESP=1)
    if(NETKIT_HEAP_ARENA)
        message(FATAL_ERROR "NETKIT_HEAP_ARENA is forbidden on MCU — use a static/global arena buffer")
    endif()
endif()

target_compile_definitions(netkit PUBLIC NETKIT_IM2COL=${NETKIT_IM2COL})

if(NETKIT_ARENA_CAPACITY AND NOT NETKIT_ARENA_CAPACITY STREQUAL "0")
    target_compile_definitions(netkit PUBLIC NK_ARENA_DEFAULT_CAPACITY=${NETKIT_ARENA_CAPACITY})
elseif(NETKIT_ARENA_KB AND NOT NETKIT_ARENA_KB STREQUAL "0")
    math(EXPR _netkit_arena_bytes "${NETKIT_ARENA_KB} * 1024")
    target_compile_definitions(netkit PUBLIC NK_ARENA_DEFAULT_CAPACITY=${_netkit_arena_bytes})
endif()


if(NETKIT_MMAP)
    target_compile_definitions(netkit PUBLIC NETKIT_USE_MMAP=1)
else()
    target_compile_definitions(netkit PUBLIC NETKIT_USE_MMAP=0)
endif()

if(NETKIT_LOOP_UNROLL)
    target_compile_definitions(netkit PUBLIC NETKIT_LOOP_UNROLL=1)
else()
    target_compile_definitions(netkit PUBLIC NETKIT_LOOP_UNROLL=0)
endif()

if(NETKIT_CMSIS_NN AND (NETKIT_TARGET STREQUAL "mcu_risc" OR NETKIT_TARGET STREQUAL "mpu_risc" OR
                        NETKIT_TARGET STREQUAL "mcu_esp"))
    message(WARNING "NETKIT_CMSIS_NN=ON forced off on ${NETKIT_TARGET} — CMSIS-NN is mcu_arm only")
    set(NETKIT_CMSIS_NN OFF)
elseif(NETKIT_CMSIS_NN AND NOT NETKIT_TARGET STREQUAL "mcu_arm")
    message(WARNING "NETKIT_CMSIS_NN=ON ignored on ${NETKIT_TARGET} — CMSIS-NN is mcu_arm + Cortex-M only")
    set(NETKIT_CMSIS_NN OFF)
endif()

if(NETKIT_ESP_NN AND NOT NETKIT_TARGET STREQUAL "mcu_esp")
    message(WARNING "NETKIT_ESP_NN=ON ignored on ${NETKIT_TARGET} — ESP-NN is mcu_esp only")
    set(NETKIT_ESP_NN OFF)
endif()

if(NETKIT_NMSIS_NN AND NOT NETKIT_TARGET STREQUAL "mcu_risc")
    message(WARNING "NETKIT_NMSIS_NN=ON ignored on ${NETKIT_TARGET} — NMSIS-NN is mcu_risc only")
    set(NETKIT_NMSIS_NN OFF)
endif()

include(cmake/netkit_cmsis.cmake)
include(cmake/netkit_esp_nn.cmake)
include(cmake/netkit_nmsis_nn.cmake)
include(cmake/netkit_xnnpack.cmake)

netkit_add_cmsis_softmax_s8(netkit)

if(NETKIT_CMSIS_NN)
    netkit_add_cmsis_nn(netkit)
endif()

if(NETKIT_ESP_NN)
    netkit_add_esp_nn(netkit)
endif()

if(NETKIT_NMSIS_NN)
    netkit_add_nmsis_nn(netkit)
endif()

if(NETKIT_XNNPACK)
    netkit_add_xnnpack(netkit)
endif()

if(NETKIT_CMSIS_NN)
    netkit_apply_cmsis_target_flags(netkit)
endif()

if(NETKIT_BUILD_CLI)
    add_executable(netkit-cli src/main.cpp)
    set_target_properties(netkit-cli PROPERTIES OUTPUT_NAME netkit)
    target_link_libraries(netkit-cli PRIVATE netkit)

    add_executable(nk_infer tools/nk_infer.c)
    target_link_libraries(nk_infer PRIVATE netkit)
    target_compile_options(nk_infer PRIVATE
        $<$<C_COMPILER_ID:Clang,AppleClang>:-fcolor-diagnostics>
        $<$<C_COMPILER_ID:Clang,AppleClang>:-fansi-escape-codes>
        -Wall
        -Wextra
        -g
    )
endif()

add_executable(infer_cpp examples/infer_cpp.cpp)
target_link_libraries(infer_cpp PRIVATE netkit)

add_executable(infer_c examples/infer_c.c)
target_link_libraries(infer_c PRIVATE netkit)
target_compile_options(infer_c PRIVATE
    $<$<C_COMPILER_ID:Clang,AppleClang>:-fcolor-diagnostics>
    $<$<C_COMPILER_ID:Clang,AppleClang>:-fansi-escape-codes>
    -Wall
    -Wextra
    -g
)

if(NETKIT_BUILD_C_TESTS)
    add_executable(test_c_api tests/test_c_api.c)
    target_link_libraries(test_c_api PRIVATE netkit)
    target_compile_options(test_c_api PRIVATE
        $<$<C_COMPILER_ID:Clang,AppleClang>:-fcolor-diagnostics>
        $<$<C_COMPILER_ID:Clang,AppleClang>:-fansi-escape-codes>
        -Wall
        -Wextra
        -g
    )
endif()
