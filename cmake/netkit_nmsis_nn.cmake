# NMSIS-NN integration for netkit (mirrors cmake/netkit_cmsis.cmake / netkit_esp_nn.cmake).

function(netkit_add_nmsis_nn target)
    if(NOT NETKIT_NMSIS_NN)
        return()
    endif()
    if(NOT NETKIT_TARGET STREQUAL "mcu_risc")
        message(WARNING
            "NETKIT_NMSIS_NN=ON ignored — requires NETKIT_TARGET=mcu_risc; using reference kernels")
        return()
    endif()

    set(NMSIS_DIR "${CMAKE_SOURCE_DIR}/third_party/NMSIS")
    set(NMSIS_NN_DIR "${NMSIS_DIR}/NMSIS/NN")
    if(NOT EXISTS "${NMSIS_NN_DIR}/Include/riscv_nnfunctions.h")
        message(FATAL_ERROR "NETKIT_NMSIS_NN=ON requires NMSIS at ${NMSIS_DIR} — run ./tools/fetch_nmsis.sh")
    endif()

    file(GLOB_RECURSE NMSIS_NN_SOURCES
        "${NMSIS_NN_DIR}/Source/ConvolutionFunctions/riscv_*s8*.c"
        "${NMSIS_NN_DIR}/Source/FullyConnectedFunctions/riscv_*s8*.c"
        "${NMSIS_NN_DIR}/Source/PoolingFunctions/riscv_*s8*.c"
        "${NMSIS_NN_DIR}/Source/PoolingFunctions/riscv_avgpool_s8*.c"
        "${NMSIS_NN_DIR}/Source/ActivationFunctions/riscv_*s8*.c"
        "${NMSIS_NN_DIR}/Source/BasicMathFunctions/riscv_*s8*.c"
        "${NMSIS_NN_DIR}/Source/SoftmaxFunctions/riscv_*s8*.c"
        "${NMSIS_NN_DIR}/Source/NNSupportFunctions/*.c"
    )
    list(REMOVE_DUPLICATES NMSIS_NN_SOURCES)

    add_library(netkit_nmsis_nn OBJECT ${NMSIS_NN_SOURCES})
    target_include_directories(netkit_nmsis_nn PUBLIC
        "${NMSIS_NN_DIR}/Include"
        "${NMSIS_DIR}/NMSIS/Core/Include")
    target_compile_options(netkit_nmsis_nn PRIVATE -std=c11 -O2 -Wno-unused-function)
    if(NETKIT_HOST_SMOKE)
        target_compile_definitions(netkit_nmsis_nn PRIVATE __GNUC_PYTHON__=1)
        target_compile_options(netkit_nmsis_nn PRIVATE
            -include "${CMAKE_SOURCE_DIR}/third_party/nmsis_host_compat.h")
    endif()

    target_sources(${target} PRIVATE
        ${CMAKE_SOURCE_DIR}/src/nmsis_nn_backend.cpp
        $<TARGET_OBJECTS:netkit_nmsis_nn>)
    target_include_directories(${target} PUBLIC
        "${NMSIS_NN_DIR}/Include"
        "${NMSIS_DIR}/NMSIS/Core/Include")
    target_compile_definitions(${target} PUBLIC NETKIT_USE_NMSIS_NN=1)
endfunction()
