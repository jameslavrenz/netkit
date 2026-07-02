# Minimal CMSIS-NN / CMSIS-DSP integration for netkit (mirrors third_party/cmsis_*.mk).

function(netkit_apply_cmsis_target_flags cmsis_target)
    if(HOST)
        target_compile_definitions(${cmsis_target} PRIVATE __GNUC_PYTHON__)
    endif()

    if(NETKIT_ENV_ARM_MATH_NEON)
        target_compile_definitions(${cmsis_target} PRIVATE ARM_MATH_NEON)
        if(EXISTS "${CMAKE_SOURCE_DIR}/third_party/CMSIS-DSP/ComputeLibrary/Include")
            target_include_directories(${cmsis_target} PRIVATE
                "${CMAKE_SOURCE_DIR}/third_party/CMSIS-DSP/ComputeLibrary/Include")
        endif()
    endif()

    if(NETKIT_ENV_ARM_MATH_LOOPUNROLL)
        target_compile_definitions(${cmsis_target} PRIVATE ARM_MATH_LOOPUNROLL)
    endif()

    foreach(def ${NETKIT_ENV_ARM_MATH_DEFINES})
        target_compile_definitions(${cmsis_target} PRIVATE ${def})
    endforeach()

    if(CMSISCORE AND EXISTS "${CMSISCORE}")
        target_include_directories(${cmsis_target} PRIVATE "${CMSISCORE}")
    endif()

    # Helium (MVE) builds may need relaxed vector conversions (matches CMSIS-DSP CMake).
    if(NETKIT_ENV_ARM_MATH_MVE)
        target_compile_options(${cmsis_target} PRIVATE
            $<$<C_COMPILER_ID:GNU>:-flax-vector-conversions>
            $<$<C_COMPILER_ID:ARMClang>:-flax-vector-conversions=integer>)
    endif()
endfunction()

function(netkit_add_cmsis_nn target)
    set(CMSIS_NN_DIR "${CMAKE_SOURCE_DIR}/third_party/CMSIS-NN")
    if(NOT EXISTS "${CMSIS_NN_DIR}/Include/arm_nnfunctions.h")
        message(FATAL_ERROR "NETKIT_CMSIS_NN=ON requires CMSIS-NN at ${CMSIS_NN_DIR} — run ./tools/fetch_cmsis_nn.sh")
    endif()

    set(CMSIS_NN_SOURCES
        Source/ConvolutionFunctions/arm_convolve_f32.c
        Source/ConvolutionFunctions/arm_convolve_1_x_n_f32.c
        Source/ConvolutionFunctions/arm_convolve_1x1_f32.c
        Source/NNSupportFunctions/arm_get_buffer_size_f32.c
        Source/NNSupportFunctions/arm_nn_pack_conv_patch_f32.c
        Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_f32.c
        Source/NNSupportFunctions/arm_nn_mat_mult_nt_n_packed_f32.c
        Source/NNSupportFunctions/arm_nn_conv1d_k3_f32.c
        Source/NNSupportFunctions/arm_nn_conv1d_k3_packed_f32.c
        Source/NNSupportFunctions/arm_nn_conv1d_k5_f32.c
        Source/NNSupportFunctions/arm_nn_conv1d_k5_packed_f32.c
        Source/NNSupportFunctions/arm_nntables_flt.c
        Source/FullyConnectedFunctions/arm_fully_connected_f32.c
        Source/PoolingFunctions/arm_max_pool_f32.c
        Source/NNSupportFunctions/arm_nn_maxpool1d_f32.c
        Source/ActivationFunctions/arm_nn_activation_f32.c
        Source/SoftmaxFunctions/arm_softmax_f32.c
        Source/BasicMathFunctions/arm_elementwise_add_f32.c
    )

    foreach(src ${CMSIS_NN_SOURCES})
        list(APPEND cmsis_nn_objects "${CMSIS_NN_DIR}/${src}")
    endforeach()

    add_library(netkit_cmsis_nn OBJECT ${cmsis_nn_objects})
    target_include_directories(netkit_cmsis_nn PUBLIC "${CMSIS_NN_DIR}/Include")
    target_compile_definitions(netkit_cmsis_nn PRIVATE ARM_NN_ENABLE_F32=1)
    target_compile_options(netkit_cmsis_nn PRIVATE -std=c11 -O2)
    netkit_apply_cmsis_target_flags(netkit_cmsis_nn)

    if(NEON AND NOT NETKIT_ENV_ARM_MATH_NEON)
        message(STATUS "NEON enabled; CMSIS-NN uses architecture NEON paths where available")
    endif()

    target_sources(${target} PRIVATE src/cmsis_nn_backend.cpp)
    target_compile_definitions(${target} PUBLIC NETKIT_USE_CMSIS_NN=1 ARM_NN_ENABLE_F32=1)
    target_include_directories(${target} PUBLIC "${CMSIS_NN_DIR}/Include")
    target_link_libraries(${target} PUBLIC netkit_cmsis_nn)
endfunction()

function(netkit_add_cmsis_dsp target)
    set(CMSIS_DSP_DIR "${CMAKE_SOURCE_DIR}/third_party/CMSIS-DSP")
    if(NOT EXISTS "${CMSIS_DSP_DIR}/Include/arm_math.h")
        message(FATAL_ERROR "NETKIT_CMSIS_DSP=ON requires CMSIS-DSP at ${CMSIS_DSP_DIR} — run ./tools/fetch_cmsis_dsp.sh")
    endif()

    set(CMSIS_DSP_SOURCES
        Source/BasicMathFunctions/arm_add_f32.c
        Source/BasicMathFunctions/arm_mult_f32.c
        Source/BasicMathFunctions/arm_scale_f32.c
        Source/BasicMathFunctions/arm_clip_f32.c
        Source/MatrixFunctions/arm_mat_init_f32.c
        Source/MatrixFunctions/arm_mat_vec_mult_f32.c
        Source/MatrixFunctions/arm_mat_mult_f32.c
    )

    foreach(src ${CMSIS_DSP_SOURCES})
        list(APPEND cmsis_dsp_objects "${CMSIS_DSP_DIR}/${src}")
    endforeach()

    add_library(netkit_cmsis_dsp OBJECT ${cmsis_dsp_objects})
    target_include_directories(netkit_cmsis_dsp PUBLIC
        "${CMSIS_DSP_DIR}/Include"
        "${CMSIS_DSP_DIR}/PrivateInclude"
    )
    target_compile_options(netkit_cmsis_dsp PRIVATE -std=c11 -O2)
    netkit_apply_cmsis_target_flags(netkit_cmsis_dsp)

    if(NEON AND NOT NETKIT_ENV_ARM_MATH_NEON)
        message(STATUS "NEON enabled; CMSIS-DSP minimal subset stays scalar (expand sources for NEON matmul)")
    endif()

    target_sources(${target} PRIVATE src/cmsis_dsp_backend.cpp)
    target_compile_definitions(${target} PUBLIC NETKIT_USE_CMSIS_DSP=1)
    target_include_directories(${target} PUBLIC
        "${CMSIS_DSP_DIR}/Include"
        "${CMSIS_DSP_DIR}/PrivateInclude"
    )
    target_link_libraries(${target} PUBLIC netkit_cmsis_dsp)
endfunction()
