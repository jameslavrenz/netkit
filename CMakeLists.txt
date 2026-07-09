cmake_minimum_required(VERSION 3.15)
cmake_policy(SET CMP0077 NEW)

project(netkit VERSION 0.1.0 LANGUAGES C CXX)

include(cmake/netkit_environment.cmake)

set(NETKIT_TARGET "${_netkit_target_default}" CACHE STRING "Build profile: cpu, mcu, or mpu")
set_property(CACHE NETKIT_TARGET PROPERTY STRINGS cpu mcu mpu)

# CMSIS backend defaults follow NETKIT_TARGET profile; override on the cmake command line.
if(NETKIT_TARGET STREQUAL "cpu")
    set(_netkit_cmsis_dsp_default ON)
    set(_netkit_cmsis_nn_default OFF)
    set(_netkit_xnnpack_default ON)
elseif(NETKIT_TARGET STREQUAL "mcu")
    set(_netkit_cmsis_dsp_default ON)
    set(_netkit_cmsis_nn_default ON)
    set(_netkit_xnnpack_default OFF)
elseif(NETKIT_TARGET STREQUAL "mpu")
    set(_netkit_cmsis_dsp_default ON)
    set(_netkit_cmsis_nn_default OFF)
    set(_netkit_xnnpack_default ON)
else()
    set(_netkit_cmsis_dsp_default OFF)
    set(_netkit_cmsis_nn_default OFF)
    set(_netkit_xnnpack_default OFF)
endif()

option(NETKIT_CMSIS_DSP "Use ARM CMSIS-DSP float32 backends" ${_netkit_cmsis_dsp_default})
option(NETKIT_CMSIS_NN "Use ARM CMSIS-NN float32 backends" ${_netkit_cmsis_nn_default})
option(NETKIT_XNNPACK "Use Google XNNPACK float32 LayerFast backends (cpu/mpu)" ${_netkit_xnnpack_default})

option(NETKIT_GLOBAL_ARENA "Use static/global arena on CPU builds" OFF)
option(NETKIT_HEAP_ARENA "Compile heap arena helpers on MCU/MPU builds" OFF)
option(NETKIT_WEIGHTS_IN_RAM "Copy .nk weight/bias payload into arena at load (opt-in)" OFF)
set(NETKIT_IM2COL "0" CACHE STRING "float Conv2D strategy: 0 direct (default), 1 partial, 2 full im2col+GEMM")
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
    src/depthwise_conv2d.cpp
    src/convnextv2_block.cpp
    src/mobilenetv4_uib.cpp
    src/yolox_decoupled_head.cpp
    src/resnet_basic_block.cpp
    src/mlp.cpp
    src/quant_ops.cpp
    src/quant_softmax_s8.cpp
    src/cmsis_dsp_util.cpp
    src/quant_trace.cpp
    src/cmsis_nn_quant_backend.cpp
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
elseif(NETKIT_TARGET STREQUAL "mcu")
    set(NETKIT_CORE_SOURCES ${NETKIT_RUNTIME_SOURCES})
    set(NETKIT_BUILD_CLI OFF)
    set(NETKIT_BUILD_C_TESTS OFF)
elseif(NETKIT_TARGET STREQUAL "mpu")
    set(NETKIT_CORE_SOURCES ${NETKIT_RUNTIME_SOURCES})
    set(NETKIT_BUILD_CLI OFF)
    set(NETKIT_BUILD_C_TESTS OFF)
else()
    message(FATAL_ERROR "NETKIT_TARGET must be cpu, mcu, or mpu (got '${NETKIT_TARGET}')")
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
elseif(NETKIT_TARGET STREQUAL "mcu")
    target_compile_definitions(netkit PUBLIC NETKIT_TARGET_MCU=1)
    if(NETKIT_HEAP_ARENA)
        target_compile_definitions(netkit PUBLIC NETKIT_HEAP_ARENA=1)
    endif()
elseif(NETKIT_TARGET STREQUAL "mpu")
    target_compile_definitions(netkit PUBLIC NETKIT_TARGET_MPU=1)
    if(NETKIT_HEAP_ARENA)
        target_compile_definitions(netkit PUBLIC NETKIT_HEAP_ARENA=1)
    endif()
endif()

if(NETKIT_WEIGHTS_IN_RAM)
    target_compile_definitions(netkit PUBLIC NETKIT_WEIGHTS_IN_RAM=1)
else()
    target_compile_definitions(netkit PUBLIC NETKIT_WEIGHTS_IN_RAM=0)
endif()

target_compile_definitions(netkit PUBLIC NETKIT_IM2COL=${NETKIT_IM2COL})

if(NETKIT_LOOP_UNROLL)
    target_compile_definitions(netkit PUBLIC NETKIT_LOOP_UNROLL=1)
else()
    target_compile_definitions(netkit PUBLIC NETKIT_LOOP_UNROLL=0)
endif()

include(cmake/netkit_cmsis.cmake)
include(cmake/netkit_xnnpack.cmake)

netkit_add_cmsis_softmax_s8(netkit)

if(NETKIT_CMSIS_DSP)
    netkit_add_cmsis_dsp(netkit)
endif()

if(NETKIT_CMSIS_NN)
    netkit_add_cmsis_nn(netkit)
endif()

if(NETKIT_XNNPACK)
    netkit_add_xnnpack(netkit)
endif()

if(NETKIT_CMSIS_DSP OR NETKIT_CMSIS_NN)
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
