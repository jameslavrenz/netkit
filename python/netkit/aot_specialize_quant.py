"""Specialized quant AOT: direct CMSIS-NN calls with compile-time shapes (no plan/Try wrappers)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .aot_lower_quant import (
    QuantLoweredPlan,
    _cmsis_conv_s8_workspace,
    _cmsis_depthwise_s8_workspace,
    _cmsis_fc_s8_workspace,
    _format_int8_array,
    _format_int32_array,
    _mlp_input_shape,
    _plan_output_shape,
    _repack_depthwise_chw_to_hwc,
    _round_up,
    can_lower_quantized_arch,
)
from .cnn_layers import depthwise_kernel_hw
from .quantize import _quantize_multiplier
from .quant_nk_reader import read_quant_nk


_SPECIALIZE_QUANT_LAYER_TYPES = frozenset(
    {
        "conv2d",
        "depthwise_conv2d",
        "max_pool2d",
        "avg_pool2d",
        "flatten",
        "dense",
    }
)


def unsupported_specialize_quant_layers(arch: dict[str, Any]) -> list[str]:
    from .aot_lower_quant import unsupported_quant_lower_layers

    blockers = list(unsupported_quant_lower_layers(arch))
    for layer in arch.get("layers", []):
        t = layer["type"]
        if t not in _SPECIALIZE_QUANT_LAYER_TYPES and t not in blockers:
            blockers.append(t)
    return blockers


def can_specialize_quantized_arch(arch: dict[str, Any]) -> bool:
    """Primitive int8 graphs only (no UIB/composites in v1)."""
    return can_lower_quantized_arch(arch) and not unsupported_specialize_quant_layers(arch)


def _act_min_max(activation: str, output_scale: float, output_zp: int) -> tuple[int, int]:
    act_min, act_max = -128, 127
    if activation not in {"relu", "relu6"}:
        return act_min, act_max
    q0 = max(-128, min(127, int(output_zp)))
    act_min = q0
    if activation == "relu":
        return act_min, 127
    if output_scale <= 0.0:
        return act_min, 127
    q6 = int(round(6.0 / float(output_scale))) + int(output_zp)
    act_max = max(q0, min(127, q6))
    return act_min, act_max


@dataclass
class _SpecializeBuild:
    weight_blocks: list[str] = field(default_factory=list)
    layer_fns: list[str] = field(default_factory=list)
    call_lines: list[str] = field(default_factory=list)
    even_max: int = 0
    odd_max: int = 0
    workspace_bytes: int = 0
    logits_elements: int = 0
    composite_scratch: list[tuple[str, int]] = field(default_factory=list)
    input_scale: float = 1.0
    input_zero_point: int = 0
    need_uib: bool = False


def plan_specialize_quant(
    nk_path: str | Path, *, omit_final_softmax: bool = False
) -> QuantLoweredPlan:
    """Plan specialized quant emit; returns QuantLoweredPlan-compatible sizing + bodies."""
    bundle = read_quant_nk(nk_path)
    if not can_specialize_quantized_arch(bundle.arch):
        raise ValueError("architecture cannot be specialized as quantized model")

    arch = bundle.arch
    if arch.get("network") == "mlp":
        h, w, c = _mlp_input_shape(arch)
    else:
        h, w, c = bundle.input_shape

    build = _SpecializeBuild()
    if bundle.quant_layers:
        build.input_scale = bundle.quant_layers[0].input_scale
        build.input_zero_point = bundle.quant_layers[0].input_zero_point

    weight_idx = 0
    quant_idx = 0
    use_a = True

    for layer_index, layer in enumerate(arch["layers"]):
        in_h, in_w, in_c = h, w, c
        elements, h, w, c = _plan_output_shape(layer, h, w, c)
        if layer_index % 2 == 0:
            build.even_max = max(build.even_max, elements)
        else:
            build.odd_max = max(build.odd_max, elements)

        dest = "slot_a" if use_a else "slot_b"
        layer_type = layer["type"]

        if layer_type == "conv2d":
            quant = bundle.quant_layers[quant_idx]
            w_arr = bundle.weight_tensors[weight_idx]
            b_arr = bundle.bias_tensors[weight_idx]
            w_name = f"kW{weight_idx}"
            b_name = f"kB{weight_idx}"
            build.weight_blocks.append(_format_int8_array(w_name, w_arr))
            build.weight_blocks.append(_format_int32_array(b_name, b_arr))
            effective = quant.input_scale * quant.weight_scale / quant.output_scale
            mult, shift = _quantize_multiplier(effective)
            out_c = int(layer["filters"])
            k = int(layer["kernel_size"])
            stride = int(layer.get("stride", 1))
            pad_h = int(layer.get("pad_h", 0))
            pad_w = int(layer.get("pad_w", 0))
            act = layer.get("activation", "none")
            act_min, act_max = _act_min_max(act, quant.output_scale, quant.output_zero_point)
            mults = np.full(out_c, mult, dtype=np.int32)
            shifts = np.full(out_c, shift, dtype=np.int32)
            build.weight_blocks.append(_format_int32_array(f"kConv{weight_idx}Mult", mults))
            build.weight_blocks.append(_format_int32_array(f"kConv{weight_idx}Shift", shifts))
            ws = _cmsis_conv_s8_workspace(in_c, k)
            build.workspace_bytes = max(build.workspace_bytes, ws)
            fn = f"nk_conv{weight_idx}"
            # Prefer wrapper (picks 1x1/fast paths); pass nullptr upscale for s8 API variants.
            cmsis_call = (
                f"arm_convolve_wrapper_s8(&ctx, &conv, &quant_p, &in_d, in, &filt_d, {w_name}, "
                f"&bias_d, {b_name}, &out_d, out)"
            )
            build.layer_fns.append(
                f"""bool {fn}(const int8_t* in, int8_t* out)
{{
    constexpr int32_t in_h = {in_h};
    constexpr int32_t in_w = {in_w};
    constexpr int32_t in_c = {in_c};
    constexpr int32_t out_h = {h};
    constexpr int32_t out_w = {w};
    constexpr int32_t out_c = {out_c};
    constexpr int32_t kernel = {k};
    constexpr uint32_t workspace_bytes = {ws}u;
    static const cmsis_nn_conv_params conv = {{
        .input_offset = {-quant.input_zero_point},
        .output_offset = {quant.output_zero_point},
        .stride = {{.w = {stride}, .h = {stride}}},
        .padding = {{.w = {pad_w}, .h = {pad_h}}},
        .dilation = {{.w = 1, .h = 1}},
        .activation = {{.min = {act_min}, .max = {act_max}}},
    }};
    static const cmsis_nn_per_channel_quant_params quant_p = {{
        .multiplier = const_cast<int32_t*>(kConv{weight_idx}Mult),
        .shift = const_cast<int32_t*>(kConv{weight_idx}Shift),
    }};
    static const cmsis_nn_dims in_d = {{.n = 1, .h = in_h, .w = in_w, .c = in_c}};
    static const cmsis_nn_dims filt_d = {{.n = out_c, .h = kernel, .w = kernel, .c = in_c}};
    static const cmsis_nn_dims bias_d = {{.n = 1, .h = 1, .w = 1, .c = out_c}};
    static const cmsis_nn_dims out_d = {{.n = 1, .h = out_h, .w = out_w, .c = out_c}};
    cmsis_nn_context ctx = {{.buf = g_workspace, .size = workspace_bytes}};
    return {cmsis_call} == ARM_CMSIS_NN_SUCCESS;
}}
"""
            )
            build.call_lines.append(
                f"    if (!{fn}(current, {dest}))\n        return false;\n"
                f"    current = {dest};\n"
            )
            use_a = not use_a
            quant_idx += 1
            weight_idx += 1

        elif layer_type == "depthwise_conv2d":
            quant = bundle.quant_layers[quant_idx]
            kh, kw = depthwise_kernel_hw(layer)
            w_arr = bundle.weight_tensors[weight_idx]
            b_arr = bundle.bias_tensors[weight_idx]
            hwc = _repack_depthwise_chw_to_hwc(w_arr, in_c, kh, kw)
            w_hwc = f"kW{weight_idx}Hwc"
            b_name = f"kB{weight_idx}"
            build.weight_blocks.append(_format_int8_array(w_hwc, hwc))
            build.weight_blocks.append(_format_int32_array(b_name, b_arr))
            effective = quant.input_scale * quant.weight_scale / quant.output_scale
            mult, shift = _quantize_multiplier(effective)
            ch = int(layer["filters"])
            mults = np.full(ch, mult, dtype=np.int32)
            shifts = np.full(ch, shift, dtype=np.int32)
            build.weight_blocks.append(_format_int32_array(f"kDw{weight_idx}Mult", mults))
            build.weight_blocks.append(_format_int32_array(f"kDw{weight_idx}Shift", shifts))
            stride = int(layer.get("stride", 1))
            pad_h = int(layer.get("pad_h", 0))
            pad_w = int(layer.get("pad_w", 0))
            act = layer.get("activation", "none")
            act_min, act_max = _act_min_max(act, quant.output_scale, quant.output_zero_point)
            ws = _cmsis_depthwise_s8_workspace(in_h, in_w, kh, kw, in_c)
            build.workspace_bytes = max(build.workspace_bytes, ws)
            fn = f"nk_dw{weight_idx}"
            cmsis_call = (
                f"arm_depthwise_conv_wrapper_s8(&ctx, &dw, &quant_p, &in_d, in, &filt_d, {w_hwc}, "
                f"&bias_d, {b_name}, &out_d, out)"
            )
            build.layer_fns.append(
                f"""bool {fn}(const int8_t* in, int8_t* out)
{{
    constexpr int32_t in_h = {in_h};
    constexpr int32_t in_w = {in_w};
    constexpr int32_t channels = {ch};
    constexpr int32_t out_h = {h};
    constexpr int32_t out_w = {w};
    constexpr uint32_t workspace_bytes = {ws}u;
    static const cmsis_nn_dw_conv_params dw = {{
        .input_offset = {-quant.input_zero_point},
        .output_offset = {quant.output_zero_point},
        .ch_mult = 1,
        .stride = {{.w = {stride}, .h = {stride}}},
        .padding = {{.w = {pad_w}, .h = {pad_h}}},
        .dilation = {{.w = 1, .h = 1}},
        .activation = {{.min = {act_min}, .max = {act_max}}},
    }};
    static const cmsis_nn_per_channel_quant_params quant_p = {{
        .multiplier = const_cast<int32_t*>(kDw{weight_idx}Mult),
        .shift = const_cast<int32_t*>(kDw{weight_idx}Shift),
    }};
    static const cmsis_nn_dims in_d = {{.n = 1, .h = in_h, .w = in_w, .c = channels}};
    static const cmsis_nn_dims filt_d = {{.n = 1, .h = {kh}, .w = {kw}, .c = channels}};
    static const cmsis_nn_dims bias_d = {{.n = 1, .h = 1, .w = 1, .c = channels}};
    static const cmsis_nn_dims out_d = {{.n = 1, .h = out_h, .w = out_w, .c = channels}};
    cmsis_nn_context ctx = {{.buf = g_workspace, .size = workspace_bytes}};
    return {cmsis_call} == ARM_CMSIS_NN_SUCCESS;
}}
"""
            )
            build.call_lines.append(
                f"    if (!{fn}(current, {dest}))\n        return false;\n"
                f"    current = {dest};\n"
            )
            use_a = not use_a
            quant_idx += 1
            weight_idx += 1

        elif layer_type == "max_pool2d":
            pool_h = int(layer["pool_size"])
            pool_w = int(layer.get("pool_w", pool_h))
            stride = int(layer.get("stride", pool_h))
            pad_h = int(layer.get("pad_h", 0))
            pad_w = int(layer.get("pad_w", 0))
            fn = f"nk_pool{layer_index}"
            build.layer_fns.append(
                f"""bool {fn}(const int8_t* in, int8_t* out)
{{
    constexpr int32_t in_h = {in_h};
    constexpr int32_t in_w = {in_w};
    constexpr int32_t in_c = {in_c};
    constexpr int32_t out_h = {h};
    constexpr int32_t out_w = {w};
    static const cmsis_nn_pool_params pool = {{
        .stride = {{.w = {stride}, .h = {stride}}},
        .padding = {{.w = {pad_w}, .h = {pad_h}}},
        .activation = {{.min = -128, .max = 127}},
    }};
    static const cmsis_nn_dims in_d = {{.n = 1, .h = in_h, .w = in_w, .c = in_c}};
    static const cmsis_nn_dims filt_d = {{.n = 1, .h = {pool_h}, .w = {pool_w}, .c = 1}};
    static const cmsis_nn_dims out_d = {{.n = 1, .h = out_h, .w = out_w, .c = in_c}};
    cmsis_nn_context ctx = {{.buf = nullptr, .size = 0}};
    return arm_max_pool_s8(&ctx, &pool, &in_d, in, &filt_d, &out_d, out) == ARM_CMSIS_NN_SUCCESS;
}}
"""
            )
            build.call_lines.append(
                f"    if (!{fn}(current, {dest}))\n        return false;\n"
                f"    current = {dest};\n"
            )
            use_a = not use_a

        elif layer_type == "avg_pool2d":
            # Identity spatial shrink via CMSIS avg pool when present; else memcpy path.
            pool_h = int(layer["pool_size"])
            pool_w = int(layer.get("pool_w", pool_h))
            stride = int(layer.get("stride", pool_h))
            pad_h = int(layer.get("pad_h", 0))
            pad_w = int(layer.get("pad_w", 0))
            fn = f"nk_avgpool{layer_index}"
            build.layer_fns.append(
                f"""bool {fn}(const int8_t* in, int8_t* out)
{{
    constexpr int32_t in_h = {in_h};
    constexpr int32_t in_w = {in_w};
    constexpr int32_t in_c = {in_c};
    constexpr int32_t out_h = {h};
    constexpr int32_t out_w = {w};
    static const cmsis_nn_pool_params pool = {{
        .stride = {{.w = {stride}, .h = {stride}}},
        .padding = {{.w = {pad_w}, .h = {pad_h}}},
        .activation = {{.min = -128, .max = 127}},
    }};
    static const cmsis_nn_dims in_d = {{.n = 1, .h = in_h, .w = in_w, .c = in_c}};
    static const cmsis_nn_dims filt_d = {{.n = 1, .h = {pool_h}, .w = {pool_w}, .c = 1}};
    static const cmsis_nn_dims out_d = {{.n = 1, .h = out_h, .w = out_w, .c = in_c}};
    cmsis_nn_context ctx = {{.buf = nullptr, .size = 0}};
    return arm_avgpool_s8(&ctx, &pool, &in_d, in, &filt_d, &out_d, out) == ARM_CMSIS_NN_SUCCESS;
}}
"""
            )
            build.call_lines.append(
                f"    if (!{fn}(current, {dest}))\n        return false;\n"
                f"    current = {dest};\n"
            )
            use_a = not use_a

        elif layer_type == "flatten":
            build.call_lines.append("    // flatten: zero-copy view\n")

        elif layer_type == "dense":
            quant = bundle.quant_layers[quant_idx]
            w_arr = bundle.weight_tensors[weight_idx]
            b_arr = bundle.bias_tensors[weight_idx]
            w_name = f"kW{weight_idx}"
            b_name = f"kB{weight_idx}"
            build.weight_blocks.append(_format_int8_array(w_name, w_arr))
            build.weight_blocks.append(_format_int32_array(b_name, b_arr))
            in_features = in_h * in_w * in_c
            out_features = int(layer["units"])
            effective = quant.input_scale * quant.weight_scale / quant.output_scale
            mult, shift = _quantize_multiplier(effective)
            build.weight_blocks.append(
                f"static int32_t kFc{weight_idx}Mult[1] = {{{mult}}};\n"
                f"static int32_t kFc{weight_idx}Shift[1] = {{{shift}}};"
            )
            act = layer.get("activation", "none")
            if omit_final_softmax and act == "softmax" and layer is arch["layers"][-1]:
                act = "none"
            act_min, act_max = _act_min_max(act, quant.output_scale, quant.output_zero_point)
            ws = _cmsis_fc_s8_workspace(in_features, out_features)
            build.workspace_bytes = max(build.workspace_bytes, ws)
            is_last = layer is arch["layers"][-1]
            fn = f"nk_fc{weight_idx}"
            build.layer_fns.append(
                f"""bool {fn}(const int8_t* in, int8_t* out)
{{
    constexpr int32_t in_features = {in_features};
    constexpr int32_t out_features = {out_features};
    constexpr uint32_t workspace_bytes = {ws}u;
    static const cmsis_nn_fc_params fc = {{
        .input_offset = {-quant.input_zero_point},
        .filter_offset = {-quant.weight_zero_point},
        .output_offset = {quant.output_zero_point},
        .activation = {{.min = {act_min}, .max = {act_max}}},
    }};
    static const cmsis_nn_quant_params quant_p = {{
        .multiplier = kFc{weight_idx}Mult,
        .shift = kFc{weight_idx}Shift,
        .is_per_channel = 0,
    }};
    static const cmsis_nn_dims in_d = {{.n = 1, .h = 1, .w = 1, .c = in_features}};
    static const cmsis_nn_dims filt_d = {{.n = in_features, .h = 1, .w = 1, .c = out_features}};
    static const cmsis_nn_dims bias_d = {{.n = 1, .h = 1, .w = 1, .c = out_features}};
    static const cmsis_nn_dims out_d = {{.n = 1, .h = 1, .w = 1, .c = out_features}};
    cmsis_nn_context ctx = {{.buf = g_workspace, .size = workspace_bytes}};
    return arm_fully_connected_wrapper_s8(
               &ctx, &fc, &quant_p, &in_d, in, &filt_d, {w_name}, &bias_d, {b_name}, &out_d, out)
           == ARM_CMSIS_NN_SUCCESS;
}}
"""
            )
            if is_last:
                build.call_lines.append(
                    f"    if (!{fn}(current, output))\n        return false;\n"
                )
                build.logits_elements = out_features
            else:
                build.call_lines.append(
                    f"    if (!{fn}(current, {dest}))\n        return false;\n"
                    f"    current = {dest};\n"
                )
                use_a = not use_a
            quant_idx += 1
            weight_idx += 1

        elif layer_type == "mobilenetv4_uib":
            raise ValueError(
                "specialize does not yet emit MobileNetV4 UIB as raw CMSIS; "
                "use default lower (CmsisQuantPlan) for UIB graphs"
            )

        else:
            raise ValueError(f"unsupported specialize layer: {layer_type}")

    workspace = _round_up(max(build.workspace_bytes, 1), 64)
    forward_body = (
        "    const int8_t* current = input;\n"
        "    int8_t* slot_a = g_act_a;\n"
        "    int8_t* slot_b = g_act_b;\n"
        "\n" + "".join(build.call_lines) + "    return true;\n"
    )
    load_body = "    loaded_ = true;\n    return true;\n"

    # Stash generated fns/weights on the plan via forward_body prefix convention:
    # render reads specialized extras from a module-level side channel.
    global _LAST_SPECIALIZE_EXTRAS
    _LAST_SPECIALIZE_EXTRAS = {
        "weight_blocks": build.weight_blocks,
        "layer_fns": build.layer_fns,
        "need_uib": build.need_uib,
    }

    return QuantLoweredPlan(
        input_scale=build.input_scale,
        input_zero_point=build.input_zero_point,
        act_a_elements=build.even_max,
        act_b_elements=build.odd_max,
        workspace_bytes=workspace,
        logits_elements=build.logits_elements,
        weight_arrays=[],
        bias_arrays=[],
        forward_body=forward_body,
        load_body=load_body,
        arena_after_load=0,
        arena_after_forward=0,
        composite_scratch=build.composite_scratch,
        quant_desc_arrays=[],
    )


_LAST_SPECIALIZE_EXTRAS: dict[str, Any] = {
    "weight_blocks": [],
    "layer_fns": [],
    "need_uib": False,
}


def render_specialize_quant_cpp_header(
    symbol: str,
    network: str,
    input_elements: int,
    output_elements: int,
    input_shape: list[int],
    plan: QuantLoweredPlan,
    arena_recommended: int,
) -> str:
    shape_literals = ", ".join(str(v) for v in input_shape)
    return f"""#pragma once
/* Generated by netkit AOT (quant specialized) — direct CMSIS-NN, constexpr shapes */

#include "arena.hpp"

#include <cstddef>
#include <cstdint>

namespace netkit::aot::{symbol} {{

inline constexpr const char* kName = "{symbol}";
inline constexpr const char* kNetwork = "{network}";
inline constexpr bool kLowered = true;
inline constexpr bool kQuantLowered = true;
inline constexpr bool kSpecialized = true;
inline constexpr std::uint32_t kInputElements = {input_elements}u;
inline constexpr std::uint32_t kOutputElements = {output_elements}u;
inline constexpr std::uint32_t kInputShape[] = {{{shape_literals}}};
inline constexpr std::uint32_t kInputRank = {len(input_shape)}u;
inline constexpr float kInputScale = {plan.input_scale:.8f}f;
inline constexpr int32_t kInputZeroPoint = {plan.input_zero_point};
inline constexpr std::size_t kActAElements = {plan.act_a_elements}u;
inline constexpr std::size_t kActBElements = {plan.act_b_elements}u;
inline constexpr std::size_t kWorkspaceBytes = {plan.workspace_bytes}u;
inline constexpr std::size_t kArenaBytesAfterLoad = {plan.arena_after_load}u;
inline constexpr std::size_t kArenaBytesAfterForward = {plan.arena_after_forward}u;
inline constexpr std::size_t kArenaBytesRecommended = {arena_recommended}u;
inline constexpr std::size_t kNkBytes = 0u;

inline bool InitArena(Arena& arena, void* memory, std::size_t capacity)
{{
    if (!memory || capacity < kArenaBytesRecommended)
        return false;
    arena.init(memory, capacity);
    return true;
}}

class Model {{
public:
    Model() = default;

    bool load(Arena& arena);
    bool forwardInt8(Arena& arena, const int8_t* input, int8_t* output) const;
    [[nodiscard]] bool isLoaded() const {{ return loaded_; }}

private:
    bool loaded_ = false;
}};

}}  // namespace netkit::aot::{symbol}
"""


def render_specialize_quant_cpp_source(
    symbol: str,
    plan: QuantLoweredPlan,
    *,
    flash_section: bool,
) -> str:
    extras = _LAST_SPECIALIZE_EXTRAS
    weight_blocks = "\n\n".join(extras.get("weight_blocks") or [])
    layer_fns = "\n".join(extras.get("layer_fns") or [])

    if flash_section:
        flash_attr = (
            "\n#if defined(__GNUC__)\n"
            "#if defined(__ELF__)\n"
            "#define NETKIT_AOT_FLASH_CONST __attribute__((section(\".rodata\"), aligned(8)))\n"
            "#else\n"
            "#define NETKIT_AOT_FLASH_CONST __attribute__((aligned(8)))\n"
            "#endif\n"
            "#else\n"
            "#define NETKIT_AOT_FLASH_CONST\n"
            "#endif\n"
        )
    else:
        flash_attr = "\n#define NETKIT_AOT_FLASH_CONST\n"

    scratch = ""
    if plan.act_a_elements > 0:
        scratch += f"alignas(16) static int8_t g_act_a[{plan.act_a_elements}];\n"
    if plan.act_b_elements > 0:
        scratch += f"alignas(16) static int8_t g_act_b[{plan.act_b_elements}];\n"
    scratch += f"alignas(16) static std::uint8_t g_workspace[{max(plan.workspace_bytes, 1)}];\n"

    return f"""/* Generated by netkit AOT compiler (quant specialized) — direct CMSIS-NN */
{flash_attr}
#include "{symbol}_aot.hpp"

#include <arm_nnfunctions.h>

#include <cstdint>

namespace netkit::aot::{symbol} {{

namespace {{

{weight_blocks}

{scratch}

{layer_fns}

}}  // namespace

bool Model::load(Arena& /*arena*/)
{{
{plan.load_body}}}

bool Model::forwardInt8(Arena& /*arena*/, const int8_t* input, int8_t* output) const
{{
    if (!loaded_ || !input || !output)
        return false;
{plan.forward_body}}}

}}  // namespace netkit::aot::{symbol}
"""
