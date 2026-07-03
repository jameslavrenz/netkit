"""Fuse ONNX subgraphs into netkit composite CNN layers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .batch_norm_fold import fold_batch_norm_params
from .onnx_graph import OnnxGraph, trace_batch_norm, trace_conv, trace_through_activations
from .writer import LayerSpec


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
    pads = _attr_ints(node, "pads", [0, 0, 0, 0])
    if len(pads) < 2:
        return 0, 0
    pad_h = int(pads[0])
    pad_w = int(pads[1])
    if len(pads) >= 4 and (pads[0] != pads[2] or pads[1] != pads[3]):
        raise ValueError("asymmetric conv padding is not supported")
    return pad_h, pad_w


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
    node = graph.node(bn_index)
    scale = graph.initializers[node.input[1]]
    beta = graph.initializers[node.input[2]]
    mean = graph.initializers[node.input[3]]
    var = graph.initializers[node.input[4]]
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
