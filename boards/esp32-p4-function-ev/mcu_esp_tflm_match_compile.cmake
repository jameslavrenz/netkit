# Match esp-tflite-micro hot-path compile flags for fair netkit vs TFLM A/B.
#
# Source of truth: boards/.../esp-tflite-micro/CMakeLists.txt
#   common_flags includes -O3 -ffunction-sections -fdata-sections
#   -fno-unwind-tables -fmessage-length=0
#   CXX: -fno-rtti -fno-exceptions -fno-threadsafe-statics
#
# Intentionally NOT mirrored (not needed for codegen fairness / would break netkit):
#   -std=gnu++14, -Werror, TF_LITE_* defines
#
# ESP-NN C: leave at IDF CONFIG_COMPILER_OPTIMIZATION_PERF (-O2). On the TFLM
# side, espressif/esp-nn is a separate component compiled at -O2; only the
# TFLM C++ wrappers get -O3. Applying -O3 to ESP-NN C here would bias netkit.

function(netkit_esp_apply_tflm_match_compile_options target)
	target_compile_options(${target} PRIVATE
		-ffunction-sections
		-fdata-sections
		-fmessage-length=0
		$<$<COMPILE_LANGUAGE:CXX>:-O3>
		$<$<COMPILE_LANGUAGE:CXX>:-fno-unwind-tables>
		$<$<COMPILE_LANGUAGE:CXX>:-fno-rtti>
		$<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions>
		$<$<COMPILE_LANGUAGE:CXX>:-fno-threadsafe-statics>
	)
endfunction()
