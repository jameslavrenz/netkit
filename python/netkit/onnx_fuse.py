"""Fuse ONNX subgraphs into netkit composite CNN layers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .batch_norm_fold import fold_batch_norm_params
from .onnx_graph import OnnxGraph, trace_batch_norm, trace_conv, trace_through_activations
from .pad_encoding import onnx_spatial_pads
from .format import Activation
from .writer import LayerSpec


def _onnx_conv_to_netkit(weight: np.ndarray) -> np.ndarray:
    out_c, in_c, kh, kw = weight.shape
    w = weight.reshape(out_c, in_c, kh, kw)
    return np.transpose(w, (0, 2, 3, 1)).copy()


def _onnx_depthwise_conv_to_netkit(weight: np.ndarray) -> np.ndarray:
    out_c, _, kh, kw = weight.shape
    return weight.reshape(out_c, kh, kw).astype(np.float32).copy()


def _attr_int(node, name: str, default: int = 0) -> int:
    for attr in node.attribute:
        if attr.name == name:
            return int(attr.i)
    return default


def _attr_float(node, name: str, default: float = 0.0) -> float:
    for attr in node.attribute:
        if attr.name == name:
            return float(attr.f)
    return default


def _attr_ints(node, name: str, default: list[int] | None = None) -> list[int]:
    for attr in node.attribute:
        if attr.name == name:
            return list(attr.ints)
    return default or []


def _symmetric_conv_pads(node) -> tuple[int, int]:
    top, left, bottom, right = onnx_spatial_pads(node)
    if top != bottom or left != right:
        raise ValueError("ResNet fusion requires symmetric conv padding")
    return top, left


def _onnx_conv_to_netkit(weight: np.ndarray) -> np.ndarray:
    out_c, in_c, kh, kw = weight.shape
    w = weight.reshape(out_c, in_c, kh, kw)
    return np.transpose(w, (0, 2, 3, 1)).copy()


@dataclass
class FuseResult:
    layer: LayerSpec
    weight_tensors: list[np.ndarray]
    bias_tensors: list[np.ndarray]
    consumed_indices: set[int]
    out_channels: int
    spatial_h: int
    spatial_w: int


@dataclass
class _ConvBnChain:
    conv_index: int
    bn_index: int | None
    conv_w: np.ndarray
    conv_b: np.ndarray | None
    conv_input: str


def _fold_bn_node(graph: OnnxGraph, bn_index: int) -> tuple[np.ndarray, np.ndarray]:
    from .onnx_graph import get_initializer

    node = graph.node(bn_index)
    scale = get_initializer(graph, node.input[1])
    beta = get_initializer(graph, node.input[2])
    mean = get_initializer(graph, node.input[3])
    var = get_initializer(graph, node.input[4])
    eps = _attr_float(node, "epsilon", 1e-5)
    return fold_batch_norm_params(scale, beta, mean, var, eps=eps)


def _bn_tensors(graph: OnnxGraph, bn_index: int | None, channels: int) -> tuple[np.ndarray, np.ndarray]:
    if bn_index is None:
        return np.ones(channels, dtype=np.float32), np.zeros(channels, dtype=np.float32)
    return _fold_bn_node(graph, bn_index)


def _conv_output_dim(size: int, kernel: int, stride: int, pad: int) -> int:
    return (size + 2 * pad - kernel) // stride + 1


def _activation_indices_after(graph: OnnxGraph, tensor: str) -> set[int]:
    consumed: set[int] = set()
    for consumer_idx, _ in graph.consumers.get(tensor, []):
        node = graph.node(consumer_idx)
        if node.op_type in {"Relu", "Sigmoid", "Tanh", "LeakyRelu", "Clip", "Softmax"}:
            consumed.add(consumer_idx)
    return consumed


def _resolve_conv_bn_chain(graph: OnnxGraph, tensor: str) -> _ConvBnChain | None:
    tensor = trace_through_activations(graph, tensor)
    node = graph.producer_node(tensor)
    if node is None:
        return None
    if node.op_type == "BatchNormalization":
        bn_index = graph.producers[node.output[0]]
        conv = trace_conv(graph, node.input[0])
        if conv is None:
            return None
        conv_index, conv_input, conv_w, conv_b = conv
        return _ConvBnChain(conv_index, bn_index, conv_w, conv_b, conv_input)
    if node.op_type == "Conv":
        conv = trace_conv(graph, tensor)
        if conv is None:
            return None
        conv_index, conv_input, conv_w, conv_b = conv
        return _ConvBnChain(conv_index, None, conv_w, conv_b, conv_input)
    return None


def _resolve_skip(
    graph: OnnxGraph,
    tensor: str,
    block_input: str,
) -> tuple[bool, int | None, int | None, np.ndarray | None, np.ndarray | None]:
    if tensor == block_input:
        return True, None, None, None, None

    tensor = trace_through_activations(graph, tensor)
    if tensor == block_input:
        return True, None, None, None, None

    chain = _resolve_conv_bn_chain(graph, tensor)
    if chain is None or chain.conv_input != block_input:
        return False, None, None, None, None
    return False, chain.conv_index, chain.bn_index, chain.conv_w, chain.conv_b


def _parse_main_branch(graph: OnnxGraph, main_tensor: str) -> tuple | None:
    conv2 = _resolve_conv_bn_chain(graph, main_tensor)
    if conv2 is None:
        return None

    conv2_input = trace_through_activations(graph, conv2.conv_input)
    conv1 = _resolve_conv_bn_chain(graph, conv2_input)
    if conv1 is None:
        return None

    return (
        conv1.conv_input,
        conv1.conv_index,
        conv1.bn_index,
        conv2.conv_index,
        conv2.bn_index,
        conv1.conv_w,
        conv1.conv_b,
        conv2.conv_w,
        conv2.conv_b,
    )


def try_fuse_resnet_basic_block(
    graph: OnnxGraph,
    add_index: int,
    *,
    spatial_h: int,
    spatial_w: int,
    in_channels: int,
) -> FuseResult | None:
    add_node = graph.node(add_index)
    if add_node.op_type != "Add" or len(add_node.input) < 2:
        return None

    left, right = add_node.input[0], add_node.input[1]
    parsed = _parse_main_branch(graph, left)
    skip_tensor = right
    if parsed is None:
        parsed = _parse_main_branch(graph, right)
        skip_tensor = left
    if parsed is None:
        return None

    (
        block_input,
        conv1_index,
        bn1_index,
        conv2_index,
        bn2_index,
        conv1_w,
        conv1_b,
        conv2_w,
        conv2_b,
    ) = parsed

    identity, shortcut_conv_idx, shortcut_bn_idx, shortcut_w, shortcut_b = _resolve_skip(
        graph, skip_tensor, block_input
    )
    if not identity and shortcut_conv_idx is None:
        return None

    conv1_node = graph.node(conv1_index)
    strides = _attr_ints(conv1_node, "strides")
    stride = int(strides[0]) if strides else 1
    pad_h, pad_w = _symmetric_conv_pads(conv1_node)
    out_c = int(conv1_w.shape[0])
    in_c = int(in_channels)
    if identity and not (stride == 1 and in_c == out_c):
        return None
    if not identity and (stride == 1 and in_c == out_c):
        return None

    consumed = {conv1_index, conv2_index, add_index}
    if bn1_index is not None:
        consumed.add(bn1_index)
    if bn2_index is not None:
        consumed.add(bn2_index)
    consumed.update(_activation_indices_after(graph, graph.node(conv1_index).output[0]))
    consumed.update(_activation_indices_after(graph, graph.node(conv2_index).output[0]))
    if shortcut_conv_idx is not None:
        consumed.add(shortcut_conv_idx)
    if shortcut_bn_idx is not None:
        consumed.add(shortcut_bn_idx)
    consumed.update(_activation_indices_after(graph, add_node.output[0]))

    bn1_scale, bn1_bias = _bn_tensors(graph, bn1_index, out_c)
    bn2_scale, bn2_bias = _bn_tensors(graph, bn2_index, out_c)

    weight_tensors: list[np.ndarray] = [
        _onnx_conv_to_netkit(conv1_w.astype(np.float32)),
        bn1_scale,
        _onnx_conv_to_netkit(conv2_w.astype(np.float32)),
        bn2_scale,
    ]
    bias_tensors: list[np.ndarray] = [
        conv1_b.astype(np.float32) if conv1_b is not None else np.zeros(out_c, dtype=np.float32),
        bn1_bias,
        conv2_b.astype(np.float32) if conv2_b is not None else np.zeros(out_c, dtype=np.float32),
        bn2_bias,
    ]

    if not identity:
        assert shortcut_w is not None
        shortcut_scale, shortcut_bias = _bn_tensors(graph, shortcut_bn_idx, out_c)
        weight_tensors.extend(
            [
                _onnx_conv_to_netkit(shortcut_w.astype(np.float32)),
                shortcut_scale,
            ]
        )
        bias_tensors.extend(
            [
                shortcut_b.astype(np.float32)
                if shortcut_b is not None
                else np.zeros(out_c, dtype=np.float32),
                shortcut_bias,
            ]
        )

    out_h = _conv_output_dim(spatial_h, 3, stride, pad_h)
    out_w = _conv_output_dim(spatial_w, 3, stride, pad_w)

    return FuseResult(
        layer=LayerSpec(
            kind="resnet_basic_block",
            in_channels=in_c,
            out_channels=out_c,
            stride=stride,
        ),
        weight_tensors=weight_tensors,
        bias_tensors=bias_tensors,
        consumed_indices=consumed,
        out_channels=out_c,
        spatial_h=out_h,
        spatial_w=out_w,
    )


def resnet_fuse_result_to_primitives(
    fused: FuseResult,
) -> tuple[list[LayerSpec], list[np.ndarray], list[np.ndarray]]:
    """Expand a fused ResNet BasicBlock into primitive conv / batch-norm layers."""
    layer = fused.layer
    in_c = int(layer.in_channels)
    out_c = int(layer.out_channels)
    stride = int(layer.stride)
    identity = stride == 1 and in_c == out_c

    layers: list[LayerSpec] = []
    weight_tensors: list[np.ndarray] = []
    bias_tensors: list[np.ndarray] = []
    idx = 0

    def append_pair(spec: LayerSpec) -> None:
        nonlocal idx
        layers.append(spec)
        weight_tensors.append(fused.weight_tensors[idx])
        bias_tensors.append(fused.bias_tensors[idx])
        idx += 1

    append_pair(
        LayerSpec(
            kind="conv2d",
            kernel_size=3,
            stride=stride,
            filters=out_c,
            pad_h=1,
            pad_w=1,
            activation=Activation.NONE,
        )
    )
    append_pair(LayerSpec(kind="batch_norm2d", channels=out_c))
    append_pair(
        LayerSpec(
            kind="conv2d",
            kernel_size=3,
            stride=1,
            filters=out_c,
            pad_h=1,
            pad_w=1,
            activation=Activation.NONE,
        )
    )
    append_pair(LayerSpec(kind="batch_norm2d", channels=out_c))

    if not identity:
        append_pair(
            LayerSpec(
                kind="conv2d",
                kernel_size=1,
                stride=stride,
                filters=out_c,
                activation=Activation.NONE,
            )
        )
        append_pair(LayerSpec(kind="batch_norm2d", channels=out_c))

    return layers, weight_tensors, bias_tensors


def _identity_bn(channels: int) -> tuple[np.ndarray, np.ndarray]:
    return np.ones(channels, dtype=np.float32), np.zeros(channels, dtype=np.float32)


def _conv_stride(node) -> int:
    strides = _attr_ints(node, "strides")
    return int(strides[0]) if strides else 1


def _conv_kernel_hw(weight: np.ndarray, node) -> tuple[int, int]:
    kernel_shape = _attr_ints(node, "kernel_shape")
    kh = int(kernel_shape[0]) if kernel_shape else int(weight.shape[2])
    kw = int(kernel_shape[1]) if len(kernel_shape) > 1 else int(weight.shape[3])
    return kh, kw


def _relu_indices_after(graph: OnnxGraph, tensor: str) -> set[int]:
    consumed: set[int] = set()
    for consumer_idx, _ in graph.consumers.get(tensor, []):
        node = graph.node(consumer_idx)
        if node.op_type == "Relu":
            consumed.add(consumer_idx)
    return consumed


def _append_uib_conv_stage(
    parts_w: list[np.ndarray],
    parts_b: list[np.ndarray],
    graph: OnnxGraph,
    conv_idx: int,
    bn_idx: int | None,
    *,
    depthwise: bool,
) -> None:
    from .onnx_graph import get_initializer, resolve_initializer_name

    node = graph.node(conv_idx)
    weight = get_initializer(graph, node.input[1])
    bias = None
    if len(node.input) >= 3:
        bias_name = resolve_initializer_name(graph, node.input[2])
        if bias_name in graph.initializers:
            bias = graph.initializers[bias_name]
    out_c = int(weight.shape[0])
    if depthwise:
        parts_w.append(_onnx_depthwise_conv_to_netkit(weight.astype(np.float32)))
    else:
        parts_w.append(_onnx_conv_to_netkit(weight.astype(np.float32)))
    parts_b.append(bias.astype(np.float32) if bias is not None else np.zeros(out_c, dtype=np.float32))
    if bn_idx is not None:
        scale, beta = _fold_bn_node(graph, bn_idx)
    else:
        scale, beta = _identity_bn(out_c)
    parts_w.append(scale)
    parts_b.append(beta)


def _infer_uib_stride_from_stages(
    *,
    start_k: int,
    start_stride: int,
    middle_k: int,
    middle_stride: int,
) -> tuple[int, int]:
    if middle_k > 0:
        middle_down = 1 if middle_stride > 1 else 0
        if middle_down:
            return middle_stride, 1
        return start_stride if start_k > 0 else 1, 0
    if start_k > 0:
        return start_stride, 1
    return 1, 1


def _parse_uib_from_project(
    graph: OnnxGraph,
    project_tensor: str,
    *,
    block_input: str,
    in_channels: int,
    spatial_h: int,
    spatial_w: int,
    require_identity_skip: bool,
) -> FuseResult | None:
    from .onnx_graph import trace_batch_norm, trace_conv

    proj = trace_conv(graph, project_tensor)
    if proj is None:
        return None
    proj_idx, proj_in, proj_w, _proj_b = proj
    out_c = int(proj_w.shape[0])
    if require_identity_skip and out_c != in_channels:
        return None

    consumed: set[int] = {proj_idx}
    consumed.update(_relu_indices_after(graph, graph.node(proj_idx).output[0]))

    stage_specs: list[tuple[int, int | None, bool]] = []
    start_k = 0
    start_stride = 1
    middle_k = 0
    middle_stride = 1
    expand_c = in_channels

    tensor = trace_through_activations(graph, proj_in)
    mid = trace_conv(graph, tensor)
    if mid is not None:
        mid_idx, mid_in, mid_w, _mid_b = mid
        mid_node = graph.node(mid_idx)
        group = _attr_int(mid_node, "group", 1)
        mid_channels = int(mid_w.shape[0])
        if group > 1 and group == mid_channels:
            middle_k, _ = _conv_kernel_hw(mid_w, mid_node)
            middle_stride = _conv_stride(mid_node)
            bn_idx = None
            traced = trace_batch_norm(graph, mid_in)
            if traced is not None:
                bn_idx = traced[0]
                consumed.add(bn_idx)
            consumed.add(mid_idx)
            consumed.update(_relu_indices_after(graph, mid_node.output[0]))
            stage_specs.append((mid_idx, bn_idx, True))
            expand_c = mid_channels
            tensor = traced[1] if traced is not None else mid_in
            tensor = trace_through_activations(graph, tensor)

    expand = trace_conv(graph, tensor)
    if expand is None:
        return None
    exp_idx, exp_in, exp_w, _exp_b = expand
    expand_c = int(exp_w.shape[0])
    exp_node = graph.node(exp_idx)
    if _attr_int(exp_node, "group", 1) != 1:
        return None
    bn_idx = None
    traced = trace_batch_norm(graph, exp_in)
    if traced is not None:
        bn_idx = traced[0]
        consumed.add(bn_idx)
    consumed.add(exp_idx)
    consumed.update(_relu_indices_after(graph, exp_node.output[0]))
    stage_specs.insert(0, (exp_idx, bn_idx, False))

    tensor = traced[1] if traced is not None else exp_in
    tensor = trace_through_activations(graph, tensor)
    start = trace_conv(graph, tensor)
    if start is not None:
        st_idx, st_in, st_w, _st_b = start
        st_node = graph.node(st_idx)
        group = _attr_int(st_node, "group", 1)
        if group == in_channels:
            start_k, _ = _conv_kernel_hw(st_w, st_node)
            start_stride = _conv_stride(st_node)
            bn_idx = None
            traced = trace_batch_norm(graph, st_in)
            if traced is not None:
                bn_idx = traced[0]
                consumed.add(bn_idx)
            consumed.add(st_idx)
            stage_specs.insert(0, (st_idx, bn_idx, True))
            tensor = st_in
    if trace_through_activations(graph, tensor) != block_input and tensor != block_input:
        return None

    stage_specs.append((proj_idx, None, False))

    weight_tensors: list[np.ndarray] = []
    bias_tensors: list[np.ndarray] = []
    for conv_idx, bn_idx, depthwise in stage_specs:
        _append_uib_conv_stage(
            weight_tensors, bias_tensors, graph, conv_idx, bn_idx, depthwise=depthwise
        )

    block_stride, middle_down = _infer_uib_stride_from_stages(
        start_k=start_k,
        start_stride=start_stride,
        middle_k=middle_k,
        middle_stride=middle_stride,
    )
    if middle_k > 0 and middle_down and start_k > 0 and start_stride != 1:
        return None
    if middle_k > 0 and not middle_down and middle_stride != 1:
        return None

    expand_ratio = float(expand_c) / float(in_channels)
    pad = 1
    if start_k:
        pad = (start_k - 1) // 2
        dw_stride = 1 if middle_down and middle_k else start_stride
        out_h = _conv_output_dim(spatial_h, start_k, dw_stride, pad)
        out_w = _conv_output_dim(spatial_w, start_k, dw_stride, pad)
    elif middle_k > 0 and middle_down:
        pad = (middle_k - 1) // 2
        out_h = _conv_output_dim(spatial_h, middle_k, middle_stride, pad)
        out_w = _conv_output_dim(spatial_w, middle_k, middle_stride, pad)
    else:
        out_h, out_w = spatial_h, spatial_w

    return FuseResult(
        layer=LayerSpec(
            kind="mobilenetv4_uib",
            in_channels=in_channels,
            out_channels=out_c,
            start_dw_kernel=start_k,
            middle_dw_kernel=middle_k,
            stride=block_stride,
            expand_ratio=expand_ratio,
            middle_dw_downsample=middle_down,
        ),
        weight_tensors=weight_tensors,
        bias_tensors=bias_tensors,
        consumed_indices=consumed,
        out_channels=out_c,
        spatial_h=out_h,
        spatial_w=out_w,
    )


def try_fuse_mobilenetv4_uib(
    graph: OnnxGraph,
    add_index: int,
    *,
    spatial_h: int,
    spatial_w: int,
    in_channels: int,
) -> FuseResult | None:
    add_node = graph.node(add_index)
    if add_node.op_type != "Add" or len(add_node.input) < 2:
        return None

    left, right = add_node.input[0], add_node.input[1]
    parsed = _parse_uib_from_project(
        graph,
        left,
        block_input=right,
        in_channels=in_channels,
        spatial_h=spatial_h,
        spatial_w=spatial_w,
        require_identity_skip=True,
    )
    skip_tensor = right
    if parsed is None:
        parsed = _parse_uib_from_project(
            graph,
            right,
            block_input=left,
            in_channels=in_channels,
            spatial_h=spatial_h,
            spatial_w=spatial_w,
            require_identity_skip=True,
        )
        skip_tensor = left
    if parsed is None:
        return None

    parsed.consumed_indices.add(add_index)
    return parsed


def try_fuse_mobilenetv4_uib_chain(
    graph: OnnxGraph,
    conv_index: int,
    *,
    spatial_h: int,
    spatial_w: int,
    in_channels: int,
    block_input: str,
) -> FuseResult | None:
    node = graph.node(conv_index)
    if node.op_type != "Conv" or not node.output:
        return None
    parsed = _parse_uib_from_project(
        graph,
        node.output[0],
        block_input=block_input,
        in_channels=in_channels,
        spatial_h=spatial_h,
        spatial_w=spatial_w,
        require_identity_skip=False,
    )
    if parsed is None:
        return None
    if parsed.out_channels == in_channels:
        return None
    return parsed


def mobilenet_uib_fuse_result_to_primitives(
    fused: FuseResult,
) -> tuple[list[LayerSpec], list[np.ndarray], list[np.ndarray]]:
    """Expand a fused UIB into primitive conv / depthwise / batch-norm layers."""
    from .arch_writer import _arch_to_spec
    from .nk_fuse import expand_mobilenetv4_uib_to_linear

    layer = fused.layer
    arch = {
        "network": "cnn",
        "input": [fused.spatial_h, fused.spatial_w, layer.in_channels],
        "layers": [
            {
                "type": "mobilenetv4_uib",
                "in_channels": layer.in_channels,
                "out_channels": layer.out_channels,
                "start_dw_kernel": layer.start_dw_kernel,
                "middle_dw_kernel": layer.middle_dw_kernel,
                "stride": layer.stride,
                "expand_ratio": layer.expand_ratio,
                "middle_dw_downsample": layer.middle_dw_downsample,
            }
        ],
    }
    parts: list[np.ndarray] = []
    for w, b in zip(fused.weight_tensors, fused.bias_tensors):
        parts.append(np.asarray(w, dtype=np.float32).reshape(-1))
        parts.append(np.asarray(b, dtype=np.float32).reshape(-1))
    weights = np.concatenate(parts).astype(np.float32)
    expanded_arch, expanded_weights = expand_mobilenetv4_uib_to_linear(arch, weights)
    spec = _arch_to_spec(expanded_arch, expanded_weights)
    return spec.layers, spec.weight_tensors, spec.bias_tensors
