"""Lower .nk graphs to static C++ kernel call chains (no loader / ops resolver)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .nk_optimize import _GraphLayer, _ShapeState, _decompose

LOWERABLE_LAYER_TYPES = frozenset(
    {
        "dense",
        "conv2d",
        "depthwise_conv2d",
        "max_pool2d",
        "avg_pool2d",
        "flatten",
        "batch_norm2d",
    }
)

_ACTIVATION_CPP = {
    "none": "NetkitKernelActivation::None",
    "relu": "NetkitKernelActivation::ReLU",
    "sigmoid": "NetkitKernelActivation::Sigmoid",
    "tanh": "NetkitKernelActivation::Tanh",
    "leaky_relu": "NetkitKernelActivation::LeakyReLU",
    "relu6": "NetkitKernelActivation::ReLU6",
    "softmax": "NetkitKernelActivation::Softmax",
}


@dataclass(frozen=True)
class LoweredPlan:
    scratch_floats: int
    scratch_buffers: int
    arena_after_load: int
    arena_after_forward: int
    weight_arrays: list[tuple[str, np.ndarray]]
    load_body: str
    forward_body: str


def can_lower_arch(arch: dict[str, Any]) -> bool:
    return all(layer["type"] in LOWERABLE_LAYER_TYPES for layer in arch["layers"])


def _out_dim(in_dim: int, kernel: int, stride: int, pad_before: int, pad_after: int) -> int:
    return (in_dim + pad_before + pad_after - kernel) // stride + 1


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _weight_runtime_ref(name: str, weights_in_ram: bool) -> str:
    return f"g{name}" if weights_in_ram else name


def _weight_data_expr(name: str, weights_in_ram: bool) -> str:
    if weights_in_ram:
        return f"g{name}"
    return f"const_cast<float*>({name})"


def _build_load_body(weight_arrays: list[tuple[str, np.ndarray]], *, weights_in_ram: bool) -> str:
    if not weights_in_ram:
        return "    loaded_ = true;\n    return true;\n"
    lines: list[str] = []
    for name, array in weight_arrays:
        nbytes = int(np.asarray(array, dtype=np.float32).nbytes)
        lines.append(f"    g{name} = static_cast<float*>(arena.alloc({nbytes}u, 16u));\n")
        lines.append(f"    if (!g{name})\n        return false;\n")
        lines.append(f"    std::memcpy(g{name}, {name}_src, {nbytes}u);\n")
    lines.append("    loaded_ = true;\n    return true;\n")
    return "".join(lines)


def _total_weight_bytes(weight_arrays: list[tuple[str, np.ndarray]]) -> int:
    return sum(int(np.asarray(array, dtype=np.float32).nbytes) for _, array in weight_arrays)


def _format_float_array(name: str, values: np.ndarray, *, const: bool = True) -> str:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    row: list[str] = []
    lines = [f"alignas(16) static {'const ' if const else ''}float {name}[{flat.size}] = {{"]
    for index, value in enumerate(flat):
        row.append(f"{float(value):.8f}f")
        if len(row) == 8:
            lines.append(f"    {', '.join(row)},")
            row = []
    if row:
        lines.append(f"    {', '.join(row)},")
    lines.append("};")
    return "\n".join(lines)


def _emit_activation_followup(var: str, activation: str, *, alpha: float, only_if_not_fused: bool) -> str:
    if activation == "none":
        return ""
    prefix = "    if (!act_fused)\n    " if only_if_not_fused else "    "
    if activation == "softmax":
        return f"{prefix}Kernels::Softmax({var}, {var});\n"
    if activation == "sigmoid":
        return f"{prefix}Kernels::Sigmoid({var}, {var});\n"
    if activation == "tanh":
        return f"{prefix}Kernels::Tanh({var}, {var});\n"
    if activation == "leaky_relu":
        return f"{prefix}Kernels::LeakyReLU({var}, {var}, {alpha:.8f}f);\n"
    if activation == "relu":
        return f"{prefix}Kernels::ReLU({var}, {var});\n"
    if activation == "relu6":
        return f"{prefix}Kernels::ReLU6({var}, {var});\n"
    raise ValueError(f"unsupported activation: {activation}")


def _view2d(data: str, rows: int, cols: int, var: str) -> str:
    return (
        f"    {var} = TensorFactory::View2D({data}, {rows}u, {cols}u);\n"
        f"    if (!{var}.data)\n        return false;\n"
    )


def _view3d(data: str, h: int, w: int, c: int, var: str) -> str:
    return (
        f"    {{\n"
        f"        const std::array<std::uint32_t, 3> shape = {{{h}u, {w}u, {c}u}};\n"
        f"        {var} = TensorFactory::ViewND({data}, 3, shape);\n"
        f"    }}\n"
        f"    if (!{var}.data)\n        return false;\n"
    )


def _plan_mlp(arch: dict[str, Any], layers: list[_GraphLayer], *, weights_in_ram: bool) -> LoweredPlan:
    batch, in_features = arch["input"]
    weight_arrays: list[tuple[str, np.ndarray]] = []
    forward_lines: list[str] = [
        "    float* read_buffer = const_cast<float*>(input);\n",
        _view2d("read_buffer", batch, in_features, "current"),
    ]

    scratch_floats = 0
    for index, layer in enumerate(layers):
        assert layer.tensors.weight is not None and layer.tensors.bias is not None
        out_features = layer.spec["units"]
        in_features = layer.tensors.weight.shape[1]
        w_name = f"kWeight{index}"
        b_name = f"kBias{index}"
        weight_arrays.append((w_name, layer.tensors.weight))
        weight_arrays.append((b_name, layer.tensors.bias.reshape(1, out_features)))

        activation = layer.spec.get("activation", "none")
        alpha = float(layer.spec.get("alpha", 0.01))
        act_cpp = _ACTIVATION_CPP[activation]
        is_last = index == len(layers) - 1

        if is_last:
            write_ptr = "output"
            out_rows, out_cols = batch, out_features
        else:
            scratch_floats = max(scratch_floats, batch * out_features)
            write_ptr = "scratch_a"
            out_rows, out_cols = batch, out_features

        forward_lines.append(_view2d(_weight_data_expr(w_name, weights_in_ram), out_features, in_features, "weights"))
        forward_lines.append(_view2d(_weight_data_expr(b_name, weights_in_ram), 1, out_features, "bias"))
        forward_lines.append(_view2d(write_ptr, out_rows, out_cols, "layer_out"))
        forward_lines.append(
            f"    act_fused = Kernels::FullyConnectedWithBias("
            f"current, weights, bias, {act_cpp}, layer_out);\n"
        )
        followup = _emit_activation_followup("layer_out", activation, alpha=alpha, only_if_not_fused=True)
        if followup:
            forward_lines.append(followup)

        if not is_last:
            forward_lines.append("    current = layer_out;\n")
            in_features = out_features

    arena_forward = _round_up(scratch_floats * 4, 64)
    weight_bytes = _round_up(_total_weight_bytes(weight_arrays), 64) if weights_in_ram else 0
    return LoweredPlan(
        scratch_floats=scratch_floats,
        scratch_buffers=1,
        arena_after_load=weight_bytes,
        arena_after_forward=max(arena_forward, weight_bytes),
        weight_arrays=weight_arrays,
        load_body=_build_load_body(weight_arrays, weights_in_ram=weights_in_ram),
        forward_body="".join(forward_lines),
    )


def _append_ping_pong(
    forward_lines: list[str],
    *,
    rank2: bool,
    height: int,
    width: int,
    channels: int,
    features: int,
) -> None:
    forward_lines.append(
        "    read_buffer = write_buffer;\n"
        "    write_buffer = (write_buffer == scratch_a) ? scratch_b : scratch_a;\n"
    )
    if rank2:
        forward_lines.append(_view2d("read_buffer", 1, features, "current"))
    else:
        forward_lines.append(_view3d("read_buffer", height, width, channels, "current"))


def _plan_cnn(arch: dict[str, Any], layers: list[_GraphLayer], *, weights_in_ram: bool) -> LoweredPlan:
    height, width, channels = arch["input"]
    dense_in = 0
    weight_arrays: list[tuple[str, np.ndarray]] = []
    forward_lines: list[str] = [
        "    float* read_buffer = const_cast<float*>(input);\n",
        "    float* write_buffer = scratch_a;\n",
        _view3d("read_buffer", height, width, channels, "current"),
    ]

    max_activation = height * width * channels
    layer_index = 0

    for layer in layers:
        layer_type = layer.spec["type"]
        is_last = layer is layers[-1]
        dest = "output" if is_last else "write_buffer"
        if layer_type == "conv2d":
            k = layer.spec["kernel_size"]
            stride = layer.spec.get("stride", 1)
            pad_h = layer.spec.get("pad_h", 0)
            pad_w = layer.spec.get("pad_w", 0)
            pad_h_end = layer.spec.get("pad_h_end", pad_h)
            pad_w_end = layer.spec.get("pad_w_end", pad_w)
            out_c = layer.spec["filters"]
            in_c = layer.tensors.weight.shape[3]
            out_h = _out_dim(height, k, stride, pad_h, pad_h_end)
            out_w = _out_dim(width, k, stride, pad_w, pad_w_end)
            activation = layer.spec.get("activation", "none")
            alpha = float(layer.spec.get("alpha", 0.01))
            act_cpp = _ACTIVATION_CPP[activation]

            w_name = f"kWeight{layer_index}"
            b_name = f"kBias{layer_index}"
            weight_arrays.append((w_name, layer.tensors.weight))
            weight_arrays.append((b_name, layer.tensors.bias))
            forward_lines.append(_view3d(dest, out_h, out_w, out_c, "layer_out"))
            forward_lines.append(
                f"    act_fused = Kernels::Conv2dForward(current, {_weight_data_expr(w_name, weights_in_ram)}, "
                f"{_weight_data_expr(b_name, weights_in_ram)}, {k}, {stride}, {pad_h}, {pad_w}, "
                f"{in_c}, {out_c}, {act_cpp}, layer_out);\n"
            )
            followup = _emit_activation_followup("layer_out", activation, alpha=alpha, only_if_not_fused=True)
            if followup:
                forward_lines.append(followup)
            if not is_last:
                _append_ping_pong(
                    forward_lines,
                    rank2=False,
                    height=out_h,
                    width=out_w,
                    channels=out_c,
                    features=0,
                )
            height, width, channels = out_h, out_w, out_c
            layer_index += 1
        elif layer_type == "depthwise_conv2d":
            kh = layer.spec.get("kernel_h", layer.spec.get("kernel_size", 3))
            kw = layer.spec.get("kernel_w", kh)
            stride = layer.spec.get("stride", 1)
            pad_h = layer.spec.get("pad_h", 0)
            pad_w = layer.spec.get("pad_w", 0)
            pad_h_end = layer.spec.get("pad_h_end", pad_h)
            pad_w_end = layer.spec.get("pad_w_end", pad_w)
            ch = layer.spec["filters"]
            out_h = _out_dim(height, kh, stride, pad_h, pad_h_end)
            out_w = _out_dim(width, kw, stride, pad_w, pad_w_end)
            activation = layer.spec.get("activation", "none")
            alpha = float(layer.spec.get("alpha", 0.01))
            act_cpp = _ACTIVATION_CPP[activation]

            w_name = f"kWeight{layer_index}"
            b_name = f"kBias{layer_index}"
            weight_arrays.append((w_name, layer.tensors.weight))
            weight_arrays.append((b_name, layer.tensors.bias))
            forward_lines.append(_view3d(dest, out_h, out_w, ch, "layer_out"))
            forward_lines.append(
                f"    act_fused = Kernels::DepthwiseConv2dForward(current, {_weight_data_expr(w_name, weights_in_ram)}, "
                f"{_weight_data_expr(b_name, weights_in_ram)}, {kh}, {kw}, {stride}, {pad_h}, {pad_w}, "
                f"{pad_h_end}, {pad_w_end}, {ch}, {act_cpp}, layer_out);\n"
            )
            followup = _emit_activation_followup("layer_out", activation, alpha=alpha, only_if_not_fused=True)
            if followup:
                forward_lines.append(followup)
            if not is_last:
                _append_ping_pong(
                    forward_lines,
                    rank2=False,
                    height=out_h,
                    width=out_w,
                    channels=ch,
                    features=0,
                )
            height, width, channels = out_h, out_w, ch
            layer_index += 1
        elif layer_type == "max_pool2d":
            pool_h = layer.spec["pool_size"]
            pool_w = layer.spec.get("pool_w", pool_h)
            stride = layer.spec.get("stride", pool_h)
            pad_h = layer.spec.get("pad_h", 0)
            pad_w = layer.spec.get("pad_w", 0)
            pad_h_end = layer.spec.get("pad_h_end", pad_h)
            pad_w_end = layer.spec.get("pad_w_end", pad_w)
            out_h = _out_dim(height, pool_h, stride, pad_h, pad_h_end)
            out_w = _out_dim(width, pool_w, stride, pad_w, pad_w_end)
            forward_lines.append(_view3d(dest, out_h, out_w, channels, "layer_out"))
            forward_lines.append(
                f"    Kernels::MaxPool2dForward(current, {pool_h}, {stride}, "
                f"{pad_h}, {pad_w}, layer_out);\n"
            )
            if not is_last:
                _append_ping_pong(
                    forward_lines,
                    rank2=False,
                    height=out_h,
                    width=out_w,
                    channels=channels,
                    features=0,
                )
            height, width = out_h, out_w
        elif layer_type == "avg_pool2d":
            pool_h = layer.spec["pool_size"]
            pool_w = layer.spec.get("pool_w", pool_h)
            stride = layer.spec.get("stride", pool_h)
            pad_h = layer.spec.get("pad_h", 0)
            pad_w = layer.spec.get("pad_w", 0)
            pad_h_end = layer.spec.get("pad_h_end", pad_h)
            pad_w_end = layer.spec.get("pad_w_end", pad_w)
            out_h = _out_dim(height, pool_h, stride, pad_h, pad_h_end)
            out_w = _out_dim(width, pool_w, stride, pad_w, pad_w_end)
            forward_lines.append(_view3d(dest, out_h, out_w, channels, "layer_out"))
            forward_lines.append(
                f"    Kernels::AvgPool2dForward(current, {pool_h}, {stride}, "
                f"{pad_h}, {pad_w}, layer_out);\n"
            )
            if not is_last:
                _append_ping_pong(
                    forward_lines,
                    rank2=False,
                    height=out_h,
                    width=out_w,
                    channels=channels,
                    features=0,
                )
            height, width = out_h, out_w
        elif layer_type == "batch_norm2d":
            ch = layer.spec["channels"]
            scale_name = f"kScale{layer_index}"
            bias_name = f"kBias{layer_index}"
            weight_arrays.append((scale_name, layer.tensors.weight))
            weight_arrays.append((bias_name, layer.tensors.bias))
            forward_lines.append(_view3d(dest, height, width, ch, "layer_out"))
            forward_lines.append(
                f"    Kernels::BatchNorm2dForward(current, "
                f"{_weight_runtime_ref(scale_name, weights_in_ram)}, "
                f"{_weight_runtime_ref(bias_name, weights_in_ram)}, "
                f"{ch}, layer_out);\n"
            )
            if not is_last:
                _append_ping_pong(
                    forward_lines,
                    rank2=False,
                    height=height,
                    width=width,
                    channels=ch,
                    features=0,
                )
            channels = ch
            layer_index += 1
        elif layer_type == "flatten":
            dense_in = height * width * channels
            forward_lines.append(
                f"    if (current.num_elements != {dense_in}u)\n        return false;\n"
                f"    std::memcpy(write_buffer, current.data, "
                f"static_cast<std::size_t>({dense_in}) * sizeof(float));\n"
            )
            forward_lines.append(_view2d("write_buffer", 1, dense_in, "current"))
            height, width, channels = 1, dense_in, 1
            if not is_last:
                _append_ping_pong(
                    forward_lines,
                    rank2=True,
                    height=1,
                    width=dense_in,
                    channels=1,
                    features=dense_in,
                )
        elif layer_type == "dense":
            out_features = layer.spec["units"]
            activation = layer.spec.get("activation", "none")
            alpha = float(layer.spec.get("alpha", 0.01))
            act_cpp = _ACTIVATION_CPP[activation]
            w_name = f"kWeight{layer_index}"
            b_name = f"kBias{layer_index}"
            weight_arrays.append((w_name, layer.tensors.weight))
            weight_arrays.append((b_name, layer.tensors.bias.reshape(1, out_features)))
            is_last = layer is layers[-1]
            if is_last:
                forward_lines.append(_view2d("output", 1, out_features, "layer_out"))
            else:
                forward_lines.append(_view2d("write_buffer", 1, out_features, "layer_out"))
            forward_lines.append(
                _view2d(_weight_data_expr(w_name, weights_in_ram), out_features, dense_in, "weights")
            )
            forward_lines.append(
                _view2d(_weight_data_expr(b_name, weights_in_ram), 1, out_features, "bias")
            )
            forward_lines.append(
                f"    act_fused = Kernels::FullyConnectedWithBias("
                f"current, weights, bias, {act_cpp}, layer_out);\n"
            )
            followup = _emit_activation_followup("layer_out", activation, alpha=alpha, only_if_not_fused=True)
            if followup:
                forward_lines.append(followup)
            if not is_last:
                _append_ping_pong(
                    forward_lines,
                    rank2=True,
                    height=1,
                    width=out_features,
                    channels=1,
                    features=out_features,
                )
            dense_in = out_features
            layer_index += 1
        else:
            raise ValueError(f"unsupported layer type: {layer_type}")

        max_activation = max(max_activation, height * width * channels)

    body = "".join(forward_lines)

    arena_forward = _round_up(max_activation * 4 * 2, 64)
    weight_bytes = _round_up(_total_weight_bytes(weight_arrays), 64) if weights_in_ram else 0
    return LoweredPlan(
        scratch_floats=max_activation * 2,
        scratch_buffers=2,
        arena_after_load=weight_bytes,
        arena_after_forward=max(arena_forward, weight_bytes),
        weight_arrays=weight_arrays,
        load_body=_build_load_body(weight_arrays, weights_in_ram=weights_in_ram),
        forward_body=body,
    )


def plan_lowered(arch: dict[str, Any], weights: np.ndarray, *, weights_in_ram: bool = False) -> LoweredPlan:
    if not can_lower_arch(arch):
        raise ValueError("architecture contains layers that cannot be lowered")
    layers, _state = _decompose(arch, weights)
    if arch["network"] == "mlp":
        return _plan_mlp(arch, layers, weights_in_ram=weights_in_ram)
    if arch["network"] == "cnn":
        return _plan_cnn(arch, layers, weights_in_ram=weights_in_ram)
    raise ValueError(f"unsupported network: {arch['network']}")


def render_lowered_cpp_header(
    symbol: str,
    network: str,
    input_elements: int,
    output_elements: int,
    input_shape: list[int],
    plan: LoweredPlan,
    arena_recommended: int,
) -> str:
    shape_literals = ", ".join(str(v) for v in input_shape)
    return f"""#pragma once
/* Generated by netkit AOT compiler (lowered) — static kernel call chain */

#include "arena.hpp"

#include <cstddef>
#include <cstdint>

namespace netkit::aot::{symbol} {{

inline constexpr const char* kName = "{symbol}";
inline constexpr const char* kNetwork = "{network}";
inline constexpr bool kLowered = true;
inline constexpr std::uint32_t kInputElements = {input_elements}u;
inline constexpr std::uint32_t kOutputElements = {output_elements}u;
inline constexpr std::uint32_t kInputShape[] = {{{shape_literals}}};
inline constexpr std::uint32_t kInputRank = {len(input_shape)}u;
inline constexpr std::size_t kScratchFloats = {plan.scratch_floats}u;
inline constexpr std::size_t kArenaBytesAfterLoad = {plan.arena_after_load}u;
inline constexpr std::size_t kArenaBytesAfterForward = {plan.arena_after_forward}u;
inline constexpr std::size_t kArenaBytesRecommended = {arena_recommended}u;

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
    bool forward(Arena& arena, const float* input, float* output) const;
    [[nodiscard]] bool isLoaded() const {{ return loaded_; }}

private:
    bool loaded_ = false;
}};

}}  // namespace netkit::aot::{symbol}
"""


def render_lowered_cpp_source(
    symbol: str,
    plan: LoweredPlan,
    include_main: bool,
    flash_section: bool,
    *,
    weights_in_ram: bool = False,
) -> str:
    if weights_in_ram:
        weight_blocks = "\n\n".join(
            _format_float_array(f"{name}_src", array) for name, array in plan.weight_arrays
        )
        ptr_decls = "\n".join(f"static float* g{name} = nullptr;" for name, _ in plan.weight_arrays)
        weight_section = f"{weight_blocks}\n\n{ptr_decls}\n"
    else:
        weight_blocks = "\n\n".join(_format_float_array(name, array) for name, array in plan.weight_arrays)
        weight_section = f"{weight_blocks}\n"
    scratch_decl = ""
    if plan.scratch_floats > 0:
        scratch_decl = f"alignas(16) static float scratch_a[{plan.scratch_floats}];\n"
        if plan.scratch_buffers > 1:
            scratch_decl += f"alignas(16) static float scratch_b[{plan.scratch_floats}];\n"

    main_block = ""
    if include_main:
        main_block = f"""

#ifdef NETKIT_AOT_MAIN
#include "arena_util.hpp"
#include <cstdio>
#include <stdalign.h>

int main(void)
{{
    alignas(std::max_align_t) static unsigned char arena_mem[netkit::aot::{symbol}::kArenaBytesRecommended];
    Arena arena;
    if (!netkit::aot::{symbol}::InitArena(arena, arena_mem, sizeof(arena_mem)))
        return 1;

    netkit::aot::{symbol}::Model model;
    if (!model.load(arena))
        return 1;

    float input[netkit::aot::{symbol}::kInputElements] = {{0.0f}};
    float output[netkit::aot::{symbol}::kOutputElements] = {{0.0f}};
    if (!model.forward(arena, input, output))
        return 1;

    for (std::uint32_t i = 0; i < netkit::aot::{symbol}::kOutputElements; ++i)
        std::printf(i ? ",%.6f" : "%.6f", output[i]);
    std::printf("\\n");
    return 0;
}}
#endif
"""

    flash_attr = ""
    place_weights_in_flash = flash_section or weights_in_ram
    if place_weights_in_flash:
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

    load_arena_param = "arena" if weights_in_ram else "/*arena*/"

    return f"""/* Generated by netkit AOT compiler (lowered) — compile with -std=c++26 for firmware */
{flash_attr}
#include "{symbol}_aot.hpp"

#include "active_kernel.hpp"
#include "tensor_factory.hpp"

#include <array>
#include <cstring>
#include <span>

namespace netkit::aot::{symbol} {{

namespace {{

{weight_section}
{scratch_decl}
}}  // namespace

bool Model::load(Arena& {load_arena_param})
{{
    if (loaded_)
        return true;
{plan.load_body}
}}

bool Model::forward(Arena& /*arena*/, const float* input, float* output) const
{{
    if (!loaded_ || !input || !output)
        return false;
    Tensor current{{}};
    Tensor layer_out{{}};
    Tensor weights{{}};
    Tensor bias{{}};
    bool act_fused = false;
{plan.forward_body}
    return true;
}}

}}  // namespace netkit::aot::{symbol}
{main_block}
"""
