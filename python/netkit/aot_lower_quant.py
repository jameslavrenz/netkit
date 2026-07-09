"""Lower quantized CNN graphs to static CmsisQuantPlan call chains (no .nk loader)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .arch_writer import _make_divisible, _uib_output_spatial
from .quantize import _quantize_multiplier, _softmax_s8_params
from .quant_nk_reader import QuantNkBundle, read_quant_nk
from .reference_forward import _out_dim

LOWERABLE_QUANT_LAYER_TYPES = frozenset(
    {
        "conv2d",
        "max_pool2d",
        "avg_pool2d",
        "flatten",
        "dense",
        "mobilenetv4_uib",
    }
)


@dataclass(frozen=True)
class QuantLoweredPlan:
    input_scale: float
    input_zero_point: int
    act_a_elements: int
    act_b_elements: int
    workspace_bytes: int
    logits_elements: int
    weight_arrays: list[tuple[str, np.ndarray]]
    bias_arrays: list[tuple[str, np.ndarray]]
    forward_body: str
    load_body: str
    arena_after_load: int
    arena_after_forward: int
    composite_scratch: list[tuple[str, int]] = field(default_factory=list)
    quant_desc_arrays: list[tuple[str, Any]] = field(default_factory=list)


def can_lower_quantized_arch(arch: dict[str, Any]) -> bool:
    network = arch.get("network")
    if network not in {"cnn", "mlp"}:
        return False
    return all(layer["type"] in LOWERABLE_QUANT_LAYER_TYPES for layer in arch["layers"])


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _plan_output_shape(layer: dict[str, Any], h: int, w: int, c: int) -> tuple[int, int, int, int]:
    layer_type = layer["type"]
    if layer_type == "conv2d":
        k = layer["kernel_size"]
        stride = layer.get("stride", 1)
        pad_h = layer.get("pad_h", 0)
        pad_w = layer.get("pad_w", 0)
        pad_h_end = layer.get("pad_h_end", pad_h)
        pad_w_end = layer.get("pad_w_end", pad_w)
        oh = _out_dim(h, k, stride, pad_h, pad_h_end)
        ow = _out_dim(w, k, stride, pad_w, pad_w_end)
        oc = layer["filters"]
        return oh * ow * oc, oh, ow, oc
    if layer_type == "max_pool2d":
        pool_h = layer["pool_size"]
        pool_w = layer.get("pool_w", pool_h)
        stride = layer.get("stride", pool_h)
        pad_h = layer.get("pad_h", 0)
        pad_w = layer.get("pad_w", 0)
        pad_h_end = layer.get("pad_h_end", pad_h)
        pad_w_end = layer.get("pad_w_end", pad_w)
        oh = _out_dim(h, pool_h, stride, pad_h, pad_h_end)
        ow = _out_dim(w, pool_w, stride, pad_w, pad_w_end)
        return oh * ow * c, oh, ow, c
    if layer_type == "avg_pool2d":
        pool_h = layer["pool_size"]
        pool_w = layer.get("pool_w", pool_h)
        stride = layer.get("stride", pool_h)
        pad_h = layer.get("pad_h", 0)
        pad_w = layer.get("pad_w", 0)
        pad_h_end = layer.get("pad_h_end", pad_h)
        pad_w_end = layer.get("pad_w_end", pad_w)
        oh = _out_dim(h, pool_h, stride, pad_h, pad_h_end)
        ow = _out_dim(w, pool_w, stride, pad_w, pad_w_end)
        return oh * ow * c, oh, ow, c
    if layer_type == "mobilenetv4_uib":
        oh, ow = _uib_output_spatial(h, w, layer)
        oc = int(layer["out_channels"])
        return oh * ow * oc, oh, ow, oc
    if layer_type == "flatten":
        features = h * w * c
        return features, 1, features, 1
    if layer_type == "dense":
        units = layer["units"]
        return units, 1, units, 1
    raise ValueError(f"unsupported layer for quant lowering: {layer_type}")


def _uib_subop_count(layer: dict[str, Any]) -> int:
    count = 2  # expand + project
    if int(layer.get("start_dw_kernel", 0)):
        count += 1
    if int(layer.get("middle_dw_kernel", 0)):
        count += 1
    return count


def _format_quant_desc(name: str, quant: Any) -> str:
    return (
        f"static const NkFormat::MlpLayerQuantDesc {name} = {{\n"
        f"    .input_scale = {quant.input_scale:.8f}f,\n"
        f"    .input_zero_point = {quant.input_zero_point},\n"
        f"    .weight_scale = {quant.weight_scale:.8f}f,\n"
        f"    .weight_zero_point = {quant.weight_zero_point},\n"
        f"    .bias_scale = {quant.bias_scale:.8f}f,\n"
        f"    .bias_zero_point = {quant.bias_zero_point},\n"
        f"    .output_scale = {quant.output_scale:.8f}f,\n"
        f"    .output_zero_point = {quant.output_zero_point},\n"
        f"}};"
    )


def _mlp_input_shape(arch: dict[str, Any]) -> tuple[int, int, int]:
    """Treat MLP [batch, features] as 1x1xfeatures for dense-only planning."""
    shape = arch["input"]
    if len(shape) == 2:
        return 1, 1, int(shape[1])
    if len(shape) == 3:
        return int(shape[0]), int(shape[1]), int(shape[2])
    raise ValueError(f"unsupported input shape for quant lowering: {shape}")

def _cmsis_conv_s8_workspace(in_c: int, kernel: int) -> int:
    rhs_cols = kernel * kernel * in_c
    remainder = rhs_cols % 4
    aligned_rhs_cols = rhs_cols if remainder == 0 else rhs_cols + 4 - remainder
    return 2 * aligned_rhs_cols * 2


def _cmsis_fc_s8_workspace(in_features: int, out_features: int) -> int:
    # CM4 DSP path returns 0 from arm_fully_connected_s8_get_buffer_size.
    (in_features, out_features)
    return 0


def _format_int8_array(name: str, values: np.ndarray) -> str:
    flat = np.asarray(values, dtype=np.int8).reshape(-1)
    row: list[str] = []
    lines = [f"alignas(16) NETKIT_AOT_FLASH_CONST int8_t {name}[{flat.size}] = {{"]
    for index, value in enumerate(flat):
        row.append(str(int(value)))
        if len(row) == 16:
            lines.append(f"    {', '.join(row)},")
            row = []
    if row:
        lines.append(f"    {', '.join(row)},")
    lines.append("};")
    return "\n".join(lines)


def _format_int32_array(name: str, values: np.ndarray) -> str:
    flat = np.asarray(values, dtype=np.int32).reshape(-1)
    row: list[str] = []
    lines = [f"alignas(16) NETKIT_AOT_FLASH_CONST int32_t {name}[{flat.size}] = {{"]
    for index, value in enumerate(flat):
        row.append(str(int(value)))
        if len(row) == 8:
            lines.append(f"    {', '.join(row)},")
            row = []
    if row:
        lines.append(f"    {', '.join(row)},")
    lines.append("};")
    return "\n".join(lines)


def plan_lowered_quant(
    nk_path: str | Path, *, omit_final_softmax: bool = False
) -> QuantLoweredPlan:
    bundle = read_quant_nk(nk_path)
    if not can_lower_quantized_arch(bundle.arch):
        raise ValueError("architecture cannot be lowered as quantized model")

    arch = bundle.arch
    if arch.get("network") == "mlp":
        h, w, c = _mlp_input_shape(arch)
    else:
        h, w, c = bundle.input_shape
    even_max = 0
    odd_max = 0
    workspace_bytes = 0
    logits_elements = 0
    layer_workspace: list[tuple[str, int]] = []

    weight_arrays: list[tuple[str, np.ndarray]] = []
    bias_arrays: list[tuple[str, np.ndarray]] = []
    composite_scratch: list[tuple[str, int]] = []
    quant_desc_arrays: list[tuple[str, Any]] = []
    forward_lines: list[str] = [
        "    KernelWorkspace workspace{g_workspace, kWorkspaceBytes};",
        "    KernelWorkspaceScope workspace_scope(&workspace);",
        "    const int8_t* current = input;",
        "    int8_t* slot_a = g_act_a;",
        "    int8_t* slot_b = g_act_b;",
        "    bool use_a = true;",
        "",
    ]

    weight_idx = 0
    quant_idx = 0
    # Track activation scale/zp for avg_pool (mirrors CmsisQuantPlan chaining).
    act_scale = bundle.quant_layers[0].input_scale if bundle.quant_layers else 1.0
    act_zp = bundle.quant_layers[0].input_zero_point if bundle.quant_layers else 0

    for layer_index, layer in enumerate(arch["layers"]):
        in_h, in_w, in_c = h, w, c
        elements, h, w, c = _plan_output_shape(layer, h, w, c)
        if layer_index % 2 == 0:
            even_max = max(even_max, elements)
        else:
            odd_max = max(odd_max, elements)

        layer_type = layer["type"]
        if layer_type == "conv2d":
            w_arr = bundle.weight_tensors[weight_idx]
            b_arr = bundle.bias_tensors[weight_idx]
            w_name = f"kW{weight_idx}"
            b_name = f"kB{weight_idx}"
            plan_name = f"kConv{weight_idx}Plan"
            weight_arrays.append((w_name, w_arr))
            bias_arrays.append((b_name, b_arr))
            workspace_bytes = max(
                workspace_bytes,
                _cmsis_conv_s8_workspace(in_c, layer["kernel_size"]),
            )
            layer_workspace.append((f"kConv{weight_idx}Plan", _cmsis_conv_s8_workspace(in_c, layer["kernel_size"])))
            forward_lines.extend(
                [
                    "    if (use_a) {",
                    f"        if (!CmsisNnQuant::TryConv2dNhwcQuantPlan(",
                    f"                {plan_name}, current, {w_name}, {b_name}, slot_a))",
                    "            return false;",
                    "        current = slot_a;",
                    "    } else {",
                    f"        if (!CmsisNnQuant::TryConv2dNhwcQuantPlan(",
                    f"                {plan_name}, current, {w_name}, {b_name}, slot_b))",
                    "            return false;",
                    "        current = slot_b;",
                    "    }",
                    "    use_a = !use_a;",
                    "",
                ]
            )
            quant_idx += 1
            weight_idx += 1
            act_scale = bundle.quant_layers[quant_idx - 1].output_scale
            act_zp = bundle.quant_layers[quant_idx - 1].output_zero_point

        elif layer_type == "max_pool2d":
            plan_name = f"kPool{layer_index}Plan"
            forward_lines.extend(
                [
                    "    if (use_a) {",
                    f"        if (!CmsisNnQuant::TryMaxPool2dNhwcQuantPlan({plan_name}, current, slot_a))",
                    "            return false;",
                    "        current = slot_a;",
                    "    } else {",
                    f"        if (!CmsisNnQuant::TryMaxPool2dNhwcQuantPlan({plan_name}, current, slot_b))",
                    "            return false;",
                    "        current = slot_b;",
                    "    }",
                    "    use_a = !use_a;",
                    "",
                ]
            )

        elif layer_type == "avg_pool2d":
            plan_name = f"kAvgPool{layer_index}Plan"
            pool_h = layer["pool_size"]
            pool_w = layer.get("pool_w", pool_h)
            stride = layer.get("stride", pool_h)
            pad_h = layer.get("pad_h", 0)
            pad_w = layer.get("pad_w", 0)
            pad_h_end = layer.get("pad_h_end", pad_h)
            pad_w_end = layer.get("pad_w_end", pad_w)
            forward_lines.extend(
                [
                    "    {",
                    "        int8_t* dest = use_a ? slot_a : slot_b;",
                    f"        QuantOps::AvgPool2dNhwcQuant(",
                    f"            current, {in_h}u, {in_w}u, {in_c}u,",
                    f"            {pool_h}, {pool_w}, {stride},",
                    f"            {pad_h}, {pad_w}, {pad_h_end}, {pad_w_end},",
                    f"            {act_scale:.8f}f, {act_zp},",
                    f"            {act_scale:.8f}f, {act_zp}, dest);",
                    "        current = dest;",
                    "        use_a = !use_a;",
                    "    }",
                    "",
                ]
            )

        elif layer_type == "mobilenetv4_uib":
            n_sub = _uib_subop_count(layer)
            in_c_u = int(layer["in_channels"])
            out_c_u = int(layer["out_channels"])
            start_k = int(layer.get("start_dw_kernel", 0))
            middle_k = int(layer.get("middle_dw_kernel", 0))
            stride = int(layer.get("stride", 1))
            middle_down = bool(layer.get("middle_dw_downsample", 1))
            expand_ratio = float(layer["expand_ratio"])
            expand_c = _make_divisible(in_c_u * expand_ratio, 8)
            residual = in_c_u if (stride == 1 and in_c_u == out_c_u) else 0
            scratch_bytes = (2 * in_h * in_w * expand_c + in_h * in_w * residual)
            scratch_name = f"g_uib_i8_scratch_{layer_index}"
            composite_scratch.append((scratch_name, scratch_bytes))

            block_input_scale = act_scale
            block_input_zp = act_zp
            sub_names: list[tuple[str, str, str]] = []
            for sub_i in range(n_sub):
                w_arr = bundle.weight_tensors[weight_idx]
                b_arr = bundle.bias_tensors[weight_idx]
                q = bundle.quant_layers[quant_idx]
                w_name = f"kW{weight_idx}"
                b_name = f"kB{weight_idx}"
                q_name = f"kUibQ{layer_index}_{sub_i}"
                weight_arrays.append((w_name, w_arr))
                bias_arrays.append((b_name, b_arr))
                quant_desc_arrays.append((q_name, q))
                sub_names.append((w_name, b_name, q_name))
                weight_idx += 1
                quant_idx += 1

            # Sub-op order: optional start_dw, expand, optional middle_dw, proj
            si = 0
            start_w = start_b = "nullptr"
            start_q = "NkFormat::MlpLayerQuantDesc{}"
            if start_k:
                start_w, start_b, start_q = sub_names[si]
                si += 1
            expand_w, expand_b, expand_q = sub_names[si]
            si += 1
            middle_w = middle_b = "nullptr"
            middle_q = "NkFormat::MlpLayerQuantDesc{}"
            if middle_k:
                middle_w, middle_b, middle_q = sub_names[si]
                si += 1
            proj_w, proj_b, proj_q = sub_names[si]

            act_scale = bundle.quant_layers[quant_idx - 1].output_scale
            act_zp = bundle.quant_layers[quant_idx - 1].output_zero_point

            forward_lines.extend(
                [
                    "    {",
                    "        int8_t* dest = use_a ? slot_a : slot_b;",
                    "        MobileNetV4Uib uib{};",
                    f"        uib.in_channels = {in_c_u};",
                    f"        uib.out_channels = {out_c_u};",
                    f"        uib.start_dw_kernel = {start_k};",
                    f"        uib.middle_dw_kernel = {middle_k};",
                    f"        uib.stride = {stride};",
                    f"        uib.middle_dw_downsample = {'true' if middle_down else 'false'};",
                    f"        uib.expand_ratio = {expand_ratio:.8f}f;",
                    "        uib.quant_enabled = true;",
                    f"        uib.block_input_scale = {block_input_scale:.8f}f;",
                    f"        uib.block_input_zero_point = {block_input_zp};",
                    f"        uib.start_dw_weights_q = {start_w if start_k else 'nullptr'};",
                    f"        uib.start_dw_bias_q = {start_b if start_k else 'nullptr'};",
                    f"        uib.start_dw_quant = {start_q if start_k else 'NkFormat::MlpLayerQuantDesc{}'};",
                    f"        uib.expand_weights_q = {expand_w};",
                    f"        uib.expand_bias_q = {expand_b};",
                    f"        uib.expand_quant = {expand_q};",
                    f"        uib.middle_dw_weights_q = {middle_w if middle_k else 'nullptr'};",
                    f"        uib.middle_dw_bias_q = {middle_b if middle_k else 'nullptr'};",
                    f"        uib.middle_dw_quant = {middle_q if middle_k else 'NkFormat::MlpLayerQuantDesc{}'};",
                    f"        uib.proj_weights_q = {proj_w};",
                    f"        uib.proj_bias_q = {proj_b};",
                    f"        uib.proj_quant = {proj_q};",
                    f"        uib.scratch_i8 = {scratch_name};",
                    f"        uib.scratch_i8_bytes = {scratch_bytes}u;",
                    f"        uib.forward_quant(current, dest, {in_h}u, {in_w}u);",
                    "        current = dest;",
                    "        use_a = !use_a;",
                    "    }",
                    "",
                ]
            )

        elif layer_type == "flatten":
            forward_lines.append("    // flatten: zero-copy view (current unchanged)\n")

        elif layer_type == "dense":
            w_arr = bundle.weight_tensors[weight_idx]
            b_arr = bundle.bias_tensors[weight_idx]
            w_name = f"kW{weight_idx}"
            b_name = f"kB{weight_idx}"
            plan_name = f"kFc{weight_idx}Plan"
            weight_arrays.append((w_name, w_arr))
            bias_arrays.append((b_name, b_arr))

            in_features = in_h * in_w * in_c
            out_features = layer["units"]
            fc_ws = _cmsis_fc_s8_workspace(in_features, out_features)
            workspace_bytes = max(workspace_bytes, fc_ws)
            layer_workspace.append((f"kFc{weight_idx}Plan", fc_ws))

            if layer.get("activation") == "softmax" and not omit_final_softmax:
                logits_elements = out_features
                sm_name = f"kSoftmax{weight_idx}Plan"
                forward_lines.extend(
                    [
                        f"    if (!CmsisNnQuant::TryFullyConnectedQuantPlan(",
                        f"            {plan_name}, current, {w_name}, {b_name}, g_logits))",
                        "        return false;",
                        f"    if (!CmsisNnQuant::TrySoftmaxS8Plan({sm_name}, g_logits, output))",
                        "        return false;",
                        "    return true;",
                        "",
                    ]
                )
            elif layer.get("activation") == "softmax" and omit_final_softmax:
                # Classification: write logits directly (argmax-equivalent).
                forward_lines.extend(
                    [
                        f"    if (!CmsisNnQuant::TryFullyConnectedQuantPlan(",
                        f"            {plan_name}, current, {w_name}, {b_name}, output))",
                        "        return false;",
                        "    return true;",
                        "",
                    ]
                )
            else:
                forward_lines.extend(
                    [
                        "    if (use_a) {",
                        f"        if (!CmsisNnQuant::TryFullyConnectedQuantPlan(",
                        f"                {plan_name}, current, {w_name}, {b_name}, slot_a))",
                        "            return false;",
                        "        current = slot_a;",
                        "    } else {",
                        f"        if (!CmsisNnQuant::TryFullyConnectedQuantPlan(",
                        f"                {plan_name}, current, {w_name}, {b_name}, slot_b))",
                        "            return false;",
                        "        current = slot_b;",
                        "    }",
                        "    use_a = !use_a;",
                        "",
                    ]
                )
            quant_idx += 1
            weight_idx += 1
            act_scale = bundle.quant_layers[quant_idx - 1].output_scale
            act_zp = bundle.quant_layers[quant_idx - 1].output_zero_point

        else:
            raise ValueError(f"unsupported layer for quant lowering: {layer_type}")

    has_softmax_tail = any(
        layer.get("type") == "dense" and layer.get("activation") == "softmax"
        for layer in arch["layers"]
    )
    if not has_softmax_tail or omit_final_softmax:
        # Softmax-omitted path already wrote to `output`; only copy when the last
        # op left results in `current`.
        if not (has_softmax_tail and omit_final_softmax):
            forward_lines.append(
                "    for (std::uint32_t i = 0; i < kOutputElements; ++i)\n"
                "        output[i] = current[i];\n"
            )
    forward_lines.append("    return true;\n")

    first_quant = bundle.quant_layers[0]
    load_lines: list[str] = []
    w_idx = 0
    for layer_index, layer in enumerate(arch["layers"]):
        if layer["type"] == "conv2d":
            load_lines.append(f"CmsisNnQuant::FinalizeConv2DPlan(kConv{w_idx}Plan);")
            w_idx += 1
        elif layer["type"] == "max_pool2d":
            load_lines.append(f"CmsisNnQuant::FinalizePool2DPlan(kPool{layer_index}Plan);")
        elif layer["type"] == "avg_pool2d":
            pass
        elif layer["type"] == "mobilenetv4_uib":
            w_idx += _uib_subop_count(layer)
        elif layer["type"] == "dense":
            load_lines.append(
                f"CmsisNnQuant::FinalizeFcPlan(kFc{w_idx}Plan, kW{w_idx}, kB{w_idx}, arena);"
            )
            if layer.get("activation") == "softmax" and not omit_final_softmax:
                q = bundle.quant_layers[w_idx]
                load_lines.append(
                    f"kSoftmax{w_idx}Plan.params = "
                    f"QuantOps::ComputeSoftmaxS8Params({q.output_scale:.8f}f);"
                )
                load_lines.append(f"kSoftmax{w_idx}Plan.ready = true;")
            w_idx += 1
    load_lines.append("loaded_ = true;")
    load_lines.append("return true;")
    load_body = "".join(f"    {line}\n" for line in load_lines)

    return QuantLoweredPlan(
        input_scale=first_quant.input_scale,
        input_zero_point=first_quant.input_zero_point,
        act_a_elements=even_max,
        act_b_elements=odd_max,
        workspace_bytes=_round_up(max(workspace_bytes, 1), 64),
        logits_elements=logits_elements,
        weight_arrays=weight_arrays,
        bias_arrays=bias_arrays,
        forward_body="".join(forward_lines),
        load_body=load_body,
        arena_after_load=0,
        arena_after_forward=0,
        composite_scratch=composite_scratch,
        quant_desc_arrays=quant_desc_arrays,
    )


def _emit_plan_structs(bundle: QuantNkBundle, *, omit_final_softmax: bool = False) -> str:
    lines: list[str] = []
    if bundle.arch.get("network") == "mlp":
        h, w, c = _mlp_input_shape(bundle.arch)
    else:
        h, w, c = bundle.input_shape
    weight_idx = 0
    quant_idx = 0

    for layer_index, layer in enumerate(bundle.arch["layers"]):
        in_h, in_w, in_c = h, w, c
        _elements, h, w, c = _plan_output_shape(layer, h, w, c)
        layer_type = layer["type"]

        if layer_type == "conv2d":
            quant = bundle.quant_layers[quant_idx]
            ws = _cmsis_conv_s8_workspace(in_c, layer["kernel_size"])
            act = layer.get("activation", "none")
            if act == "relu":
                clamp = "QuantInteger::QuantClamp::ReLU"
            elif act == "relu6":
                clamp = "QuantInteger::QuantClamp::ReLU6"
            else:
                clamp = "QuantInteger::QuantClamp::None"
            lines.append(
                f"""static CmsisQuantPlan::Conv2DPlan kConv{weight_idx}Plan = {{
    .input_offset = {-quant.input_zero_point},
    .output_offset = {-quant.output_zero_point},
    .stride = {layer.get("stride", 1)},
    .pad_h = {layer.get("pad_h", 0)},
    .pad_w = {layer.get("pad_w", 0)},
    .clamp = {clamp},
    .output_scale = {quant.output_scale:.8f}f,
    .in_h = {in_h},
    .in_w = {in_w},
    .in_c = {in_c},
    .out_h = {h},
    .out_w = {w},
    .out_c = {c},
    .kernel_size = {layer["kernel_size"]},
    .workspace_bytes = {ws},
    .multipliers = const_cast<int32_t*>(kConv{weight_idx}Mult),
    .shifts = const_cast<int32_t*>(kConv{weight_idx}Shift),
    .ready = true,
}};"""
            )
            quant_idx += 1
            weight_idx += 1

        elif layer_type == "max_pool2d":
            pool_h = layer["pool_size"]
            pool_w = layer.get("pool_w", pool_h)
            lines.append(
                f"""static CmsisQuantPlan::Pool2DPlan kPool{layer_index}Plan = {{
    .stride = {layer.get("stride", pool_h)},
    .pad_h = {layer.get("pad_h", 0)},
    .pad_w = {layer.get("pad_w", 0)},
    .pool_h = {pool_h},
    .pool_w = {pool_w},
    .in_h = {in_h},
    .in_w = {in_w},
    .in_c = {in_c},
    .out_h = {h},
    .out_w = {w},
    .ready = true,
}};"""
            )

        elif layer_type in {"avg_pool2d", "flatten"}:
            pass

        elif layer_type == "mobilenetv4_uib":
            n_sub = _uib_subop_count(layer)
            quant_idx += n_sub
            weight_idx += n_sub

        elif layer_type == "dense":
            quant = bundle.quant_layers[quant_idx]
            in_features = in_h * in_w * in_c
            out_features = layer["units"]
            effective = quant.input_scale * quant.weight_scale / quant.output_scale
            mult, shift = _quantize_multiplier(effective)
            act = layer.get("activation", "none")
            if act == "relu":
                clamp = "QuantInteger::QuantClamp::ReLU"
            elif act == "relu6":
                clamp = "QuantInteger::QuantClamp::ReLU6"
            else:
                clamp = "QuantInteger::QuantClamp::None"
            fc_ws = _cmsis_fc_s8_workspace(in_features, out_features)
            lines.append(
                f"""static CmsisQuantPlan::FcPlan kFc{weight_idx}Plan = {{
    .input_offset = {-quant.input_zero_point},
    .filter_offset = {-quant.weight_zero_point},
    .output_offset = {-quant.output_zero_point},
    .clamp = {clamp},
    .output_scale = {quant.output_scale:.8f}f,
    .in_features = {in_features},
    .out_features = {out_features},
    .multiplier = {mult},
    .shift = {shift},
    .workspace_bytes = {fc_ws},
    .ready = true,
}};"""
            )
            if layer.get("activation") == "softmax" and not omit_final_softmax:
                lines.append(
                    f"""static CmsisQuantPlan::SoftmaxPlan kSoftmax{weight_idx}Plan = {{
    .params = {{}},
    .row_size = {out_features},
    .ready = false,
}};"""
                )
            quant_idx += 1
            weight_idx += 1

    return "\n\n".join(lines)


def render_lowered_quant_cpp_header(
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
/* Generated by netkit AOT compiler (quant lowered) — static CmsisQuantPlan call chain */

#include "arena.hpp"

#include <cstddef>
#include <cstdint>

namespace netkit::aot::{symbol} {{

inline constexpr const char* kName = "{symbol}";
inline constexpr const char* kNetwork = "{network}";
inline constexpr bool kLowered = true;
inline constexpr bool kQuantLowered = true;
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


def render_lowered_quant_cpp_source(
    symbol: str,
    nk_path: Path,
    plan: QuantLoweredPlan,
    include_main: bool,
    flash_section: bool,
    *,
    omit_final_softmax: bool = False,
) -> str:
    bundle = read_quant_nk(nk_path)
    weight_blocks: list[str] = []
    for name, array in plan.weight_arrays:
        weight_blocks.append(_format_int8_array(name, array))
    for name, array in plan.bias_arrays:
        weight_blocks.append(_format_int32_array(name, array))

    weight_idx = 0
    quant_idx = 0
    for layer in bundle.arch["layers"]:
        if layer["type"] == "conv2d":
            quant = bundle.quant_layers[quant_idx]
            effective = quant.input_scale * quant.weight_scale / quant.output_scale
            mult, shift = _quantize_multiplier(effective)
            channels = layer["filters"]
            mults = np.full(channels, mult, dtype=np.int32)
            shifts = np.full(channels, shift, dtype=np.int32)
            weight_blocks.append(_format_int32_array(f"kConv{weight_idx}Mult", mults))
            weight_blocks.append(_format_int32_array(f"kConv{weight_idx}Shift", shifts))
            quant_idx += 1
            weight_idx += 1
        elif layer["type"] == "mobilenetv4_uib":
            n_sub = _uib_subop_count(layer)
            quant_idx += n_sub
            weight_idx += n_sub
        elif layer["type"] == "dense":
            quant_idx += 1
            weight_idx += 1

    for name, quant in plan.quant_desc_arrays:
        weight_blocks.append(_format_quant_desc(name, quant))

    plan_structs = _emit_plan_structs(bundle, omit_final_softmax=omit_final_softmax)

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
    if plan.logits_elements > 0:
        scratch += f"alignas(16) static int8_t g_logits[{plan.logits_elements}];\n"
    if plan.workspace_bytes > 0:
        scratch += f"alignas(16) static std::uint8_t g_workspace[{plan.workspace_bytes}];\n"
    else:
        scratch += "alignas(16) static std::uint8_t g_workspace[1];\n"
    for name, elems in plan.composite_scratch:
        scratch += f"alignas(16) static int8_t {name}[{elems}];\n"

    need_uib = any(layer["type"] == "mobilenetv4_uib" for layer in bundle.arch["layers"])
    uib_include = '#include "mobilenetv4_uib.hpp"\n' if need_uib else ""

    return f"""/* Generated by netkit AOT compiler (quant lowered) */
{flash_attr}
#include "{symbol}_aot.hpp"

#include "cmsis_nn_quant.hpp"
#include "kernel_workspace.hpp"
#include "nk_format.hpp"
#include "quant_ops.hpp"
#include "quant_plan_types.hpp"
{uib_include}
#include <cstdint>

namespace netkit::aot::{symbol} {{

namespace {{

{chr(10).join(weight_blocks)}

{plan_structs}

{scratch}
}}  // namespace

bool Model::load(Arena& arena)
{{
{plan.load_body}}}

bool Model::forwardInt8(Arena& /*arena*/, const int8_t* input, int8_t* output) const
{{
    if (!loaded_ || !input || !output)
        return false;
{plan.forward_body}}}

}}  // namespace netkit::aot::{symbol}
"""
