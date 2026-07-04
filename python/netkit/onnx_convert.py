"""Convert supported ONNX graphs into netkit .nk models."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import onnx
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Requires onnx: pip install onnx numpy") from exc

from .format import Activation, activation_from_name
from .pad_encoding import onnx_spatial_pads, spatial_output_dim
from .writer import LayerSpec, ModelSpec, write_nk, write_nk_bytes

_STANDALONE_ACTIVATIONS = {
    "Relu",
    "Softmax",
    "Sigmoid",
    "Tanh",
    "LeakyRelu",
    "Clip",
}


def _initializer_map(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    from onnx import numpy_helper

    return {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}


def _resolve_input_shape(model: onnx.ModelProto) -> tuple[str, list[int]]:
    if not model.graph.input:
        raise ValueError("ONNX graph has no inputs")

    dims = []
    for dim in model.graph.input[0].type.tensor_type.shape.dim:
        dims.append(int(dim.dim_value))

    if len(dims) == 2:
        batch = dims[0] if dims[0] > 0 else 1
        return "mlp", [batch, dims[1]]
    if len(dims) == 4:
        _, channels, height, width = dims
        return "cnn", [height, width, channels]
    if len(dims) == 3:
        return "cnn", dims
    raise ValueError("unsupported ONNX input rank")


def _onnx_conv_to_netkit(weight: np.ndarray) -> np.ndarray:
    # ONNX [O, I, Kh, Kw] -> netkit [O, Kh, Kw, I]
    out_c, in_c, kh, kw = weight.shape
    w = weight.reshape(out_c, in_c, kh, kw)
    return np.transpose(w, (0, 2, 3, 1)).copy()


def _onnx_depthwise_conv_to_netkit(weight: np.ndarray) -> np.ndarray:
    # ONNX depthwise [C, 1, Kh, Kw] -> netkit [C, Kh, Kw]
    out_c, _, kh, kw = weight.shape
    return weight.reshape(out_c, kh, kw).astype(np.float32).copy()


def _expand_grouped_conv_to_netkit(weight: np.ndarray, *, group: int, in_channels: int) -> np.ndarray:
    """Expand ONNX grouped Conv weights into dense netkit conv2d layout [O, Kh, Kw, I]."""
    out_c, in_per_group, kh, kw = weight.shape
    if in_per_group * group != in_channels:
        raise ValueError("grouped conv weight shape does not match input channels")
    if out_c % group != 0:
        raise ValueError("grouped conv output channels must divide group count")
    full = np.zeros((out_c, kh, kw, in_channels), dtype=np.float32)
    out_per_group = out_c // group
    w = np.transpose(weight.astype(np.float32), (0, 2, 3, 1))  # [O, Kh, Kw, I/G]
    for g in range(group):
        o_base = g * out_per_group
        i_base = g * in_per_group
        for oc in range(out_per_group):
            for ic in range(in_per_group):
                full[o_base + oc, :, :, i_base + ic] = w[o_base + oc, :, :, ic]
    return full


def _depthwise_layer_spec(
    *,
    kh: int,
    kw: int,
    stride: int,
    channels: int,
    activation: Activation,
    alpha: float,
    pad_top: int,
    pad_left: int,
    pad_bottom: int,
    pad_right: int,
) -> LayerSpec:
    spec = LayerSpec(
        kind="depthwise_conv2d",
        kernel_h=kh,
        kernel_w=kw,
        stride=stride,
        filters=channels,
        activation=activation,
        alpha=alpha,
        pad_h=pad_top,
        pad_w=pad_left,
    )
    if pad_bottom != pad_top or pad_right != pad_left:
        if kh != kw:
            raise ValueError("asymmetric padding on non-square depthwise conv is not supported")
        spec.pad_h_end = pad_bottom
        spec.pad_w_end = pad_right
    return spec


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


def _onnx_gemm_weight_to_netkit(weight: np.ndarray, *, trans_b: int) -> tuple[np.ndarray, int]:
    """Normalize ONNX Gemm B matrix to netkit dense layout [out, in]."""
    if trans_b:
        return weight.astype(np.float32), int(weight.shape[0])
    return weight.T.astype(np.float32), int(weight.shape[1])


def _attr_ints(node, name: str, default: list[int] | None = None) -> list[int]:
    for attr in node.attribute:
        if attr.name == name:
            return list(attr.ints)
    return default or []


def _conv_output_dim(size: int, kernel: int, stride: int, pad_top: int, pad_bottom: int) -> int:
    return spatial_output_dim(size, kernel, stride, pad_top, pad_bottom)


def _pool_output_dim(size: int, kernel: int, stride: int, pad_top: int, pad_bottom: int) -> int:
    return spatial_output_dim(size, kernel, stride, pad_top, pad_bottom)


def _pool_kernel_shape(node) -> tuple[int, int]:
    kernel_shape = _attr_ints(node, "kernel_shape")
    if not kernel_shape:
        return 2, 2
    kh = int(kernel_shape[0])
    kw = int(kernel_shape[1]) if len(kernel_shape) > 1 else kh
    return kh, kw


def _pool_stride(node, *, default: int) -> int:
    strides = _attr_ints(node, "strides")
    if not strides:
        return default
    return int(strides[0])


def _layer_pad_fields(node) -> tuple[int, int, int, int]:
    top, left, bottom, right = onnx_spatial_pads(node)
    return top, left, bottom, right


def _symmetric_pool_pads(node) -> tuple[int, int]:
    top, left, bottom, right = onnx_spatial_pads(node)
    if top != bottom or left != right:
        raise ValueError("legacy symmetric pool padding helper requires equal top/bottom and left/right")
    return top, left


def _symmetric_conv_pads(node) -> tuple[int, int]:
    top, left, bottom, right = onnx_spatial_pads(node)
    if top != bottom or left != right:
        raise ValueError("legacy symmetric conv padding helper requires equal top/bottom and left/right")
    return top, left


def _clip_is_relu6(node, initializers: dict[str, np.ndarray]) -> bool:
    if len(node.input) < 3:
        return False
    min_name, max_name = node.input[1], node.input[2]
    if min_name not in initializers or max_name not in initializers:
        return False
    min_val = float(initializers[min_name].reshape(-1)[0])
    max_val = float(initializers[max_name].reshape(-1)[0])
    return min_val == 0.0 and max_val == 6.0


def _peek_fused_activation(
    nodes, index: int, initializers: dict[str, np.ndarray]
) -> tuple[Activation, float, int]:
    if index + 1 >= len(nodes):
        return Activation.NONE, 0.01, 0

    nxt = nodes[index + 1]
    return _activation_from_node(nxt, initializers)


def _activation_at(
    nodes, index: int, initializers: dict[str, np.ndarray]
) -> tuple[Activation, float, int]:
    if index >= len(nodes):
        return Activation.NONE, 0.01, 0
    return _activation_from_node(nodes[index], initializers)


def _activation_from_node(
    node, initializers: dict[str, np.ndarray]
) -> tuple[Activation, float, int]:
    if node.op_type == "Relu":
        return Activation.RELU, 0.01, 1
    if node.op_type == "Softmax":
        return Activation.SOFTMAX, 0.01, 1
    if node.op_type == "Sigmoid":
        return Activation.SIGMOID, 0.01, 1
    if node.op_type == "Tanh":
        return Activation.TANH, 0.01, 1
    if node.op_type == "LeakyRelu":
        return Activation.LEAKY_RELU, _attr_float(node, "alpha", 0.01), 1
    if node.op_type == "Clip" and _clip_is_relu6(node, initializers):
        return Activation.RELU6, 0.01, 1
    return Activation.NONE, 0.01, 0


def _optional_initializer(initializers: dict[str, np.ndarray], name: str) -> np.ndarray | None:
    if not name or name not in initializers:
        return None
    return initializers[name]


def _optional_bias(initializers: dict[str, np.ndarray], node) -> np.ndarray | None:
    if len(node.input) < 3:
        return None
    return _optional_initializer(initializers, node.input[2])


def _optional_bias_graph(graph, node) -> np.ndarray | None:
    from .onnx_graph import OnnxGraph, resolve_initializer_name

    if not isinstance(graph, OnnxGraph) or len(node.input) < 3:
        return None
    name = resolve_initializer_name(graph, node.input[2])
    return graph.initializers.get(name)


def _has_graph_branches(nodes, initializers: dict[str, np.ndarray]) -> bool:
    """True when the graph has residual/skip Add nodes, not fused MatMul/Gemm bias adds."""
    for node in nodes:
        if node.op_type != "Add":
            continue
        if len(node.input) >= 2 and node.input[1] in initializers:
            continue
        return True
    return False


def _peek_fused_activation_graph(graph, node_index: int) -> tuple[Activation, float, set[int]]:
    from .onnx_graph import OnnxGraph

    assert isinstance(graph, OnnxGraph)
    node = graph.node(node_index)
    if not node.output:
        return Activation.NONE, 0.01, set()
    out = node.output[0]
    for consumer_idx, _ in graph.consumers.get(out, []):
        nxt = graph.node(consumer_idx)
        if nxt.op_type == "Relu":
            return Activation.RELU, 0.01, {consumer_idx}
        if nxt.op_type == "Softmax":
            return Activation.SOFTMAX, 0.01, {consumer_idx}
        if nxt.op_type == "Sigmoid":
            return Activation.SIGMOID, 0.01, {consumer_idx}
        if nxt.op_type == "Tanh":
            return Activation.TANH, 0.01, {consumer_idx}
        if nxt.op_type == "LeakyRelu":
            return Activation.LEAKY_RELU, _attr_float(nxt, "alpha", 0.01), {consumer_idx}
        if nxt.op_type == "Clip" and _clip_is_relu6(nxt, graph.initializers):
            return Activation.RELU6, 0.01, {consumer_idx}
    return Activation.NONE, 0.01, set()


def _emit_cnn_primitive(
    graph,
    node_index: int,
    *,
    initializers: dict[str, np.ndarray],
    channels: int,
    spatial_h: int,
    spatial_w: int,
    consumed: set[int],
) -> tuple[list[LayerSpec], list[np.ndarray], list[np.ndarray], int, int, int, set[int]] | None:
    """Emit one primitive layer from an ONNX node index."""
    from .onnx_graph import OnnxGraph, get_initializer

    assert isinstance(graph, OnnxGraph)
    node = graph.node(node_index)
    if node.op_type in _STANDALONE_ACTIVATIONS:
        consumed.add(node_index)
        return [], [], [], spatial_h, spatial_w, channels, consumed

    activation, alpha, act_nodes = _peek_fused_activation_graph(graph, node_index)
    consumed.update(act_nodes)
    layers: list[LayerSpec] = []
    weight_tensors: list[np.ndarray] = []
    bias_tensors: list[np.ndarray] = []

    if node.op_type in {"Identity", "Dropout", "Reshape", "Squeeze", "Unsqueeze", "Transpose"}:
        consumed.add(node_index)
        return layers, weight_tensors, bias_tensors, spatial_h, spatial_w, channels, consumed

    if node.op_type == "Conv":
        weight = get_initializer(graph, node.input[1])
        bias = _optional_bias_graph(graph, node)
        group = _attr_int(node, "group", 1)
        kernel_shape = _attr_ints(node, "kernel_shape")
        strides = _attr_ints(node, "strides")
        kh = int(kernel_shape[0]) if kernel_shape else int(weight.shape[2])
        kw = int(kernel_shape[1]) if len(kernel_shape) > 1 else int(weight.shape[3])
        stride = int(strides[0]) if strides else 1
        pad_top, pad_left, pad_bottom, pad_right = _layer_pad_fields(node)
        out_c = int(weight.shape[0])
        in_c = int(channels)
        if group > 1 and group == in_c and out_c == in_c:
            layers.append(
                _depthwise_layer_spec(
                    kh=kh,
                    kw=kw,
                    stride=stride,
                    channels=out_c,
                    activation=activation,
                    alpha=alpha,
                    pad_top=pad_top,
                    pad_left=pad_left,
                    pad_bottom=pad_bottom,
                    pad_right=pad_right,
                )
            )
            weight_tensors.append(_onnx_depthwise_conv_to_netkit(weight.astype(np.float32)))
        elif group == 1:
            layers.append(
                LayerSpec(
                    kind="conv2d",
                    kernel_size=kh,
                    stride=stride,
                    filters=out_c,
                    activation=activation,
                    alpha=alpha,
                    pad_h=pad_top,
                    pad_w=pad_left,
                    pad_h_end=pad_bottom,
                    pad_w_end=pad_right,
                )
            )
            weight_tensors.append(_onnx_conv_to_netkit(weight.astype(np.float32)))
        elif group > 1 and in_c % group == 0 and out_c % group == 0:
            layers.append(
                LayerSpec(
                    kind="conv2d",
                    kernel_size=kh,
                    stride=stride,
                    filters=out_c,
                    activation=activation,
                    alpha=alpha,
                    pad_h=pad_top,
                    pad_w=pad_left,
                    pad_h_end=pad_bottom,
                    pad_w_end=pad_right,
                )
            )
            weight_tensors.append(
                _expand_grouped_conv_to_netkit(weight.astype(np.float32), group=group, in_channels=in_c)
            )
        else:
            raise ValueError("grouped conv (non-depthwise) is not supported")
        bias_tensors.append(
            bias.astype(np.float32) if bias is not None else np.zeros(out_c, dtype=np.float32)
        )
        spatial_h = _conv_output_dim(spatial_h, kh, stride, pad_top, pad_bottom)
        spatial_w = _conv_output_dim(spatial_w, kw, stride, pad_left, pad_right)
        if group == 1 or (
            group > 1 and in_c % group == 0 and out_c % group == 0 and not (group == in_c and out_c == in_c)
        ):
            channels = out_c
        consumed.add(node_index)
        return layers, weight_tensors, bias_tensors, spatial_h, spatial_w, channels, consumed

    if node.op_type == "MaxPool":
        pool_h, pool_w = _pool_kernel_shape(node)
        stride = _pool_stride(node, default=pool_h)
        pad_top, pad_left, pad_bottom, pad_right = _layer_pad_fields(node)
        layers.append(
            LayerSpec(
                kind="max_pool2d",
                pool_size=pool_h,
                pool_w=pool_w,
                stride=stride,
                pad_h=pad_top,
                pad_w=pad_left,
                pad_h_end=pad_bottom,
                pad_w_end=pad_right,
            )
        )
        spatial_h = _pool_output_dim(spatial_h, pool_h, stride, pad_top, pad_bottom)
        spatial_w = _pool_output_dim(spatial_w, pool_w, stride, pad_left, pad_right)
        consumed.add(node_index)
        return layers, weight_tensors, bias_tensors, spatial_h, spatial_w, channels, consumed

    if node.op_type in {"AveragePool", "AvgPool"}:
        pool_h, pool_w = _pool_kernel_shape(node)
        stride = _pool_stride(node, default=pool_h)
        pad_top, pad_left, pad_bottom, pad_right = _layer_pad_fields(node)
        layers.append(
            LayerSpec(
                kind="avg_pool2d",
                pool_size=pool_h,
                pool_w=pool_w,
                stride=stride,
                pad_h=pad_top,
                pad_w=pad_left,
                pad_h_end=pad_bottom,
                pad_w_end=pad_right,
            )
        )
        spatial_h = _pool_output_dim(spatial_h, pool_h, stride, pad_top, pad_bottom)
        spatial_w = _pool_output_dim(spatial_w, pool_w, stride, pad_left, pad_right)
        consumed.add(node_index)
        return layers, weight_tensors, bias_tensors, spatial_h, spatial_w, channels, consumed

    if node.op_type == "GlobalAveragePool":
        if spatial_h != spatial_w:
            raise ValueError("GlobalAveragePool requires square spatial dims in this converter")
        layers.append(
            LayerSpec(kind="avg_pool2d", pool_size=spatial_h, stride=1, pad_h=0, pad_w=0)
        )
        spatial_h = 1
        spatial_w = 1
        consumed.add(node_index)
        return layers, weight_tensors, bias_tensors, spatial_h, spatial_w, channels, consumed

    if node.op_type == "BatchNormalization":
        scale = initializers[node.input[1]]
        beta = initializers[node.input[2]]
        mean = initializers[node.input[3]]
        var = initializers[node.input[4]]
        epsilon = _attr_float(node, "epsilon", 1e-5)
        out_c = int(scale.shape[0])
        inv_std = 1.0 / np.sqrt(var.astype(np.float64) + epsilon)
        folded_scale = (scale.astype(np.float64) * inv_std).astype(np.float32)
        folded_bias = (beta.astype(np.float64) - mean.astype(np.float64) * folded_scale).astype(np.float32)
        layers.append(LayerSpec(kind="batch_norm2d", channels=out_c))
        weight_tensors.append(folded_scale)
        bias_tensors.append(folded_bias)
        consumed.add(node_index)
        return layers, weight_tensors, bias_tensors, spatial_h, spatial_w, channels, consumed

    if node.op_type == "Flatten":
        layers.append(LayerSpec(kind="flatten"))
        consumed.add(node_index)
        return layers, weight_tensors, bias_tensors, spatial_h, spatial_w, channels, consumed

    if node.op_type == "Gemm":
        weight = initializers[node.input[1]]
        bias = _optional_bias(initializers, node)
        trans_b = _attr_int(node, "transB", 0)
        packed_w, out_features = _onnx_gemm_weight_to_netkit(weight, trans_b=trans_b)
        layers.append(
            LayerSpec(kind="dense", units=out_features, activation=activation, alpha=alpha)
        )
        weight_tensors.append(packed_w)
        bias_tensors.append(
            bias.astype(np.float32) if bias is not None else np.zeros(out_features, dtype=np.float32)
        )
        consumed.add(node_index)
        return layers, weight_tensors, bias_tensors, spatial_h, spatial_w, channels, consumed

    if node.op_type == "MatMul":
        if node.input[1] not in initializers:
            raise ValueError("MatMul weight must be an initializer for CNN import")
        weight = initializers[node.input[1]]
        packed_w, out_features = _matmul_weight_to_netkit(weight)
        bias, bias_nodes = _peek_fused_bias_add_graph(graph, node_index, initializers)
        consumed.update(bias_nodes)
        layers.append(
            LayerSpec(kind="dense", units=out_features, activation=activation, alpha=alpha)
        )
        weight_tensors.append(packed_w)
        bias_tensors.append(
            bias.astype(np.float32) if bias is not None else np.zeros(out_features, dtype=np.float32)
        )
        consumed.add(node_index)
        return layers, weight_tensors, bias_tensors, spatial_h, spatial_w, channels, consumed

    if node.op_type == "Add":
        raise ValueError(
            "unsupported Add node — enable composite fusion or simplify the ONNX graph"
        )

    raise ValueError(f"unsupported ONNX op for CNN: {node.op_type}")


def _primitive_shape_delta(
    graph,
    node_index: int,
    *,
    initializers: dict[str, np.ndarray],
    channels: int,
    spatial_h: int,
    spatial_w: int,
) -> tuple[int, int, int] | None:
    """Return updated (h, w, channels) after a primitive ONNX op, or None if no shape change."""
    node = graph.node(node_index)
    if node.op_type in _STANDALONE_ACTIVATIONS or node.op_type in {"Identity", "Dropout", "Add"}:
        return spatial_h, spatial_w, channels
    if node.op_type == "Conv":
        from .onnx_graph import get_initializer

        weight = get_initializer(graph, node.input[1])
        group = _attr_int(node, "group", 1)
        kernel_shape = _attr_ints(node, "kernel_shape")
        strides = _attr_ints(node, "strides")
        kh = int(kernel_shape[0]) if kernel_shape else int(weight.shape[2])
        kw = int(kernel_shape[1]) if len(kernel_shape) > 1 else int(weight.shape[3])
        stride = int(strides[0]) if strides else 1
        pad_top, pad_left, pad_bottom, pad_right = _layer_pad_fields(node)
        out_c = int(weight.shape[0])
        in_c = int(channels)
        spatial_h = _conv_output_dim(spatial_h, kh, stride, pad_top, pad_bottom)
        spatial_w = _conv_output_dim(spatial_w, kw, stride, pad_left, pad_right)
        if group == 1 or (
            group > 1 and in_c % group == 0 and out_c % group == 0 and not (group == in_c and out_c == in_c)
        ):
            channels = out_c
        return spatial_h, spatial_w, channels
    if node.op_type == "MaxPool":
        pool_h, pool_w = _pool_kernel_shape(node)
        stride = _pool_stride(node, default=pool_h)
        pad_top, pad_left, pad_bottom, pad_right = _layer_pad_fields(node)
        spatial_h = _pool_output_dim(spatial_h, pool_h, stride, pad_top, pad_bottom)
        spatial_w = _pool_output_dim(spatial_w, pool_w, stride, pad_left, pad_right)
        return spatial_h, spatial_w, channels
    if node.op_type in {"AveragePool", "AvgPool"}:
        pool_h, pool_w = _pool_kernel_shape(node)
        stride = _pool_stride(node, default=pool_h)
        pad_top, pad_left, pad_bottom, pad_right = _layer_pad_fields(node)
        spatial_h = _pool_output_dim(spatial_h, pool_h, stride, pad_top, pad_bottom)
        spatial_w = _pool_output_dim(spatial_w, pool_w, stride, pad_left, pad_right)
        return spatial_h, spatial_w, channels
    if node.op_type == "GlobalAveragePool":
        return 1, 1, channels
    if node.op_type in {
        "BatchNormalization",
        "Flatten",
        "Gemm",
        "MatMul",
        "Reshape",
        "Squeeze",
        "Unsqueeze",
        "Transpose",
    }:
        return spatial_h, spatial_w, channels
    return None


def _onnx_to_spec_cnn_branched(model, *, composite: bool) -> ModelSpec:
    from .onnx_fuse import (
        mobilenet_uib_fuse_result_to_primitives,
        resnet_fuse_result_to_primitives,
        try_fuse_mobilenetv4_uib,
        try_fuse_mobilenetv4_uib_chain,
        try_fuse_resnet_basic_block,
    )
    from .onnx_graph import build_onnx_graph, topo_order

    graph = build_onnx_graph(model)
    initializers = graph.initializers
    order = topo_order(graph)

    fusion_by_index: dict[int, object] = {}
    fusion_consumed: set[int] = set()
    spatial_h, spatial_w, channels = graph.input_shape
    block_input = graph.input_name

    for idx in order:
        if idx in fusion_consumed:
            continue
        node = graph.node(idx)
        if node.op_type == "Add":
            fused = try_fuse_resnet_basic_block(
                graph,
                idx,
                spatial_h=spatial_h,
                spatial_w=spatial_w,
                in_channels=channels,
            )
            if fused is None:
                fused = try_fuse_mobilenetv4_uib(
                    graph,
                    idx,
                    spatial_h=spatial_h,
                    spatial_w=spatial_w,
                    in_channels=channels,
                )
            if fused is not None:
                fusion_by_index[idx] = fused
                fusion_consumed.update(fused.consumed_indices)
                spatial_h, spatial_w, channels = fused.spatial_h, fused.spatial_w, fused.out_channels
                continue
        elif node.op_type == "Conv" and node.output:
            fused = try_fuse_mobilenetv4_uib_chain(
                graph,
                idx,
                spatial_h=spatial_h,
                spatial_w=spatial_w,
                in_channels=channels,
                block_input=block_input,
            )
            if fused is not None:
                fusion_by_index[idx] = fused
                fusion_consumed.update(fused.consumed_indices)
                spatial_h, spatial_w, channels = fused.spatial_h, fused.spatial_w, fused.out_channels
                continue
        shape = _primitive_shape_delta(
            graph,
            idx,
            initializers=initializers,
            channels=channels,
            spatial_h=spatial_h,
            spatial_w=spatial_w,
        )
        if shape is not None:
            spatial_h, spatial_w, channels = shape
            if node.output:
                block_input = node.output[0]

    layers: list[LayerSpec] = []
    weight_tensors: list[np.ndarray] = []
    bias_tensors: list[np.ndarray] = []
    spatial_h, spatial_w, channels = graph.input_shape
    block_input = graph.input_name
    consumed: set[int] = set()

    for idx in order:
        if idx in fusion_consumed and idx not in fusion_by_index:
            continue
        if idx in fusion_by_index:
            fused = fusion_by_index[idx]
            if composite:
                layers.append(fused.layer)
                weight_tensors.extend(fused.weight_tensors)
                bias_tensors.extend(fused.bias_tensors)
            elif fused.layer.kind == "resnet_basic_block":
                prim_layers, prim_w, prim_b = resnet_fuse_result_to_primitives(fused)
                layers.extend(prim_layers)
                weight_tensors.extend(prim_w)
                bias_tensors.extend(prim_b)
            elif fused.layer.kind == "mobilenetv4_uib":
                prim_layers, prim_w, prim_b = mobilenet_uib_fuse_result_to_primitives(fused)
                layers.extend(prim_layers)
                weight_tensors.extend(prim_w)
                bias_tensors.extend(prim_b)
            spatial_h, spatial_w, channels = fused.spatial_h, fused.spatial_w, fused.out_channels
            node = graph.node(idx)
            if node.output:
                block_input = node.output[0]
            continue
        emitted = _emit_cnn_primitive(
            graph,
            idx,
            initializers=initializers,
            channels=channels,
            spatial_h=spatial_h,
            spatial_w=spatial_w,
            consumed=consumed,
        )
        if emitted is None:
            continue
        new_layers, new_w, new_b, spatial_h, spatial_w, channels, consumed = emitted
        layers.extend(new_layers)
        weight_tensors.extend(new_w)
        bias_tensors.extend(new_b)
        node = graph.node(idx)
        if node.output:
            block_input = node.output[0]

    return ModelSpec(
        network="cnn",
        input_shape=graph.input_shape,
        layers=layers,
        weight_tensors=weight_tensors,
        bias_tensors=bias_tensors,
    )


def _matmul_weight_to_netkit(weight: np.ndarray) -> tuple[np.ndarray, int]:
    if weight.ndim != 2:
        raise ValueError("MatMul weight initializer must be rank-2")
    return weight.T.astype(np.float32), int(weight.shape[1])


def _peek_fused_bias_add(
    nodes, index: int, initializers: dict[str, np.ndarray]
) -> tuple[np.ndarray | None, int]:
    if index + 1 >= len(nodes):
        return None, 0
    nxt = nodes[index + 1]
    if nxt.op_type != "Add" or len(nxt.input) < 2:
        return None, 0
    bias_name = nxt.input[1]
    if bias_name not in initializers:
        return None, 0
    return initializers[bias_name].astype(np.float32).reshape(-1), 1


def _peek_fused_bias_add_graph(
    graph, node_index: int, initializers: dict[str, np.ndarray]
) -> tuple[np.ndarray | None, set[int]]:
    if node_index + 1 >= len(graph.nodes):
        return None, set()
    nxt = graph.node(node_index + 1)
    if nxt.op_type != "Add" or len(nxt.input) < 2:
        return None, set()
    bias_name = nxt.input[1]
    if bias_name not in initializers:
        return None, set()
    return initializers[bias_name].astype(np.float32).reshape(-1), {node_index + 1}


def onnx_to_spec(onnx_path: str | Path, *, fuse_composite: bool = True) -> ModelSpec:
    model = onnx.load(onnx_path)
    network, input_shape = _resolve_input_shape(model)
    initializers = _initializer_map(model)
    nodes = list(model.graph.node)

    layers: list[LayerSpec] = []
    weight_tensors: list[np.ndarray] = []
    bias_tensors: list[np.ndarray] = []

    if network == "mlp":
        in_features = input_shape[1]
        i = 0
        while i < len(nodes):
            node = nodes[i]
            if node.op_type in _STANDALONE_ACTIVATIONS:
                i += 1
                continue
            if node.op_type in {"Reshape", "Squeeze", "Unsqueeze", "Identity", "Dropout"}:
                i += 1
                continue

            activation, alpha, act_skip = _peek_fused_activation(nodes, i, initializers)

            if node.op_type == "Gemm":
                weight = initializers[node.input[1]]
                bias = _optional_bias(initializers, node)
                trans_b = _attr_int(node, "transB", 0)
                packed_w, out_features = _onnx_gemm_weight_to_netkit(weight, trans_b=trans_b)
                i += 1 + act_skip
            elif node.op_type == "MatMul":
                if node.input[1] not in initializers:
                    raise ValueError("MatMul weight must be an initializer for MLP import")
                weight = initializers[node.input[1]]
                packed_w, out_features = _matmul_weight_to_netkit(weight)
                bias, bias_skip = _peek_fused_bias_add(nodes, i, initializers)
                post_bias = i + 1 + bias_skip
                activation, alpha, act_skip = _activation_at(nodes, post_bias, initializers)
                i = post_bias + act_skip
            else:
                raise ValueError(f"unsupported ONNX op for MLP: {node.op_type}")

            layers.append(
                LayerSpec(kind="dense", units=out_features, activation=activation, alpha=alpha)
            )
            weight_tensors.append(packed_w)
            bias_tensors.append(
                bias.astype(np.float32) if bias is not None else np.zeros(out_features, dtype=np.float32)
            )
            in_features = out_features

        return ModelSpec(
            network="mlp",
            input_shape=input_shape,
            layers=layers,
            weight_tensors=weight_tensors,
            bias_tensors=bias_tensors,
        )

    if network == "cnn" and _has_graph_branches(nodes, initializers):
        return _onnx_to_spec_cnn_branched(model, composite=fuse_composite)

    # CNN (linear scan)
    channels = input_shape[2]
    spatial_h, spatial_w = input_shape[0], input_shape[1]
    dense_in = 0
    i = 0
    while i < len(nodes):
        node = nodes[i]
        if node.op_type in _STANDALONE_ACTIVATIONS:
            i += 1
            continue

        if node.op_type in {"Reshape", "Squeeze", "Unsqueeze", "Identity", "Dropout", "Transpose"}:
            i += 1
            continue

        activation, alpha, skip = _peek_fused_activation(nodes, i, initializers)

        if node.op_type == "Conv":
            weight = initializers[node.input[1]]
            bias = _optional_bias(initializers, node)
            group = _attr_int(node, "group", 1)
            kernel_shape = _attr_ints(node, "kernel_shape")
            strides = _attr_ints(node, "strides")
            kh = int(kernel_shape[0]) if kernel_shape else int(weight.shape[2])
            kw = int(kernel_shape[1]) if len(kernel_shape) > 1 else int(weight.shape[3])
            stride = int(strides[0]) if strides else 1
            pad_top, pad_left, pad_bottom, pad_right = _layer_pad_fields(node)
            out_c = int(weight.shape[0])
            in_c = int(channels)
            if group > 1 and group == in_c and out_c == in_c:
                layers.append(
                    _depthwise_layer_spec(
                        kh=kh,
                        kw=kw,
                        stride=stride,
                        channels=out_c,
                        activation=activation,
                        alpha=alpha,
                        pad_top=pad_top,
                        pad_left=pad_left,
                        pad_bottom=pad_bottom,
                        pad_right=pad_right,
                    )
                )
                weight_tensors.append(_onnx_depthwise_conv_to_netkit(weight.astype(np.float32)))
            elif group == 1:
                layers.append(
                    LayerSpec(
                        kind="conv2d",
                        kernel_size=kh,
                        stride=stride,
                        filters=out_c,
                        activation=activation,
                        alpha=alpha,
                        pad_h=pad_top,
                        pad_w=pad_left,
                        pad_h_end=pad_bottom,
                        pad_w_end=pad_right,
                    )
                )
                weight_tensors.append(_onnx_conv_to_netkit(weight.astype(np.float32)))
            elif group > 1 and in_c % group == 0 and out_c % group == 0:
                layers.append(
                    LayerSpec(
                        kind="conv2d",
                        kernel_size=kh,
                        stride=stride,
                        filters=out_c,
                        activation=activation,
                        alpha=alpha,
                        pad_h=pad_top,
                        pad_w=pad_left,
                        pad_h_end=pad_bottom,
                        pad_w_end=pad_right,
                    )
                )
                weight_tensors.append(
                    _expand_grouped_conv_to_netkit(weight.astype(np.float32), group=group, in_channels=in_c)
                )
            else:
                raise ValueError("grouped conv (non-depthwise) is not supported")
            bias_tensors.append(
                (bias.astype(np.float32) if bias is not None else np.zeros(out_c, dtype=np.float32))
            )
            spatial_h = _conv_output_dim(spatial_h, kh, stride, pad_top, pad_bottom)
            spatial_w = _conv_output_dim(spatial_w, kw, stride, pad_left, pad_right)
            if group == 1 or (group > 1 and in_c % group == 0 and out_c % group == 0 and not (group == in_c and out_c == in_c)):
                channels = out_c
            i += 1 + skip
            continue

        if node.op_type == "MaxPool":
            pool_h, pool_w = _pool_kernel_shape(node)
            stride = _pool_stride(node, default=pool_h)
            pad_top, pad_left, pad_bottom, pad_right = _layer_pad_fields(node)
            layers.append(
                LayerSpec(
                    kind="max_pool2d",
                    pool_size=pool_h,
                    pool_w=pool_w,
                    stride=stride,
                    pad_h=pad_top,
                    pad_w=pad_left,
                    pad_h_end=pad_bottom,
                    pad_w_end=pad_right,
                )
            )
            spatial_h = _pool_output_dim(spatial_h, pool_h, stride, pad_top, pad_bottom)
            spatial_w = _pool_output_dim(spatial_w, pool_w, stride, pad_left, pad_right)
            i += 1
            continue

        if node.op_type == "AveragePool" or node.op_type == "AvgPool":
            pool_h, pool_w = _pool_kernel_shape(node)
            stride = _pool_stride(node, default=pool_h)
            pad_top, pad_left, pad_bottom, pad_right = _layer_pad_fields(node)
            layers.append(
                LayerSpec(
                    kind="avg_pool2d",
                    pool_size=pool_h,
                    pool_w=pool_w,
                    stride=stride,
                    pad_h=pad_top,
                    pad_w=pad_left,
                    pad_h_end=pad_bottom,
                    pad_w_end=pad_right,
                )
            )
            spatial_h = _pool_output_dim(spatial_h, pool_h, stride, pad_top, pad_bottom)
            spatial_w = _pool_output_dim(spatial_w, pool_w, stride, pad_left, pad_right)
            i += 1
            continue

        if node.op_type == "GlobalAveragePool":
            if spatial_h != spatial_w:
                raise ValueError("GlobalAveragePool requires square spatial dims in this converter")
            layers.append(
                LayerSpec(
                    kind="avg_pool2d",
                    pool_size=spatial_h,
                    stride=1,
                    pad_h=0,
                    pad_w=0,
                )
            )
            spatial_h = 1
            spatial_w = 1
            i += 1
            continue

        if node.op_type == "BatchNormalization":
            scale = initializers[node.input[1]]
            beta = initializers[node.input[2]]
            mean = initializers[node.input[3]]
            var = initializers[node.input[4]]
            epsilon = _attr_float(node, "epsilon", 1e-5)
            out_c = int(scale.shape[0])
            inv_std = 1.0 / np.sqrt(var.astype(np.float64) + epsilon)
            folded_scale = (scale.astype(np.float64) * inv_std).astype(np.float32)
            folded_bias = (beta.astype(np.float64) - mean.astype(np.float64) * folded_scale).astype(
                np.float32
            )
            layers.append(LayerSpec(kind="batch_norm2d", channels=out_c))
            weight_tensors.append(folded_scale)
            bias_tensors.append(folded_bias)
            i += 1
            continue

        if node.op_type == "Flatten":
            layers.append(LayerSpec(kind="flatten"))
            dense_in = spatial_h * spatial_w * channels
            i += 1
            continue

        if node.op_type == "Gemm":
            weight = initializers[node.input[1]]
            bias = _optional_bias(initializers, node)
            trans_b = _attr_int(node, "transB", 0)
            packed_w, out_features = _onnx_gemm_weight_to_netkit(weight, trans_b=trans_b)
            layers.append(
                LayerSpec(kind="dense", units=out_features, activation=activation, alpha=alpha)
            )
            weight_tensors.append(packed_w)
            bias_tensors.append(
                (bias.astype(np.float32) if bias is not None else np.zeros(out_features, dtype=np.float32))
            )
            dense_in = out_features
            i += 1 + skip
            continue

        if node.op_type == "MatMul":
            if node.input[1] not in initializers:
                raise ValueError("MatMul weight must be an initializer for CNN import")
            weight = initializers[node.input[1]]
            packed_w, out_features = _matmul_weight_to_netkit(weight)
            bias, bias_skip = _peek_fused_bias_add(nodes, i + skip, initializers)
            layers.append(
                LayerSpec(kind="dense", units=out_features, activation=activation, alpha=alpha)
            )
            weight_tensors.append(packed_w)
            bias_tensors.append(
                bias.astype(np.float32) if bias is not None else np.zeros(out_features, dtype=np.float32)
            )
            dense_in = out_features
            i += 1 + skip + bias_skip
            continue

        raise ValueError(f"unsupported ONNX op for CNN: {node.op_type}")

    return ModelSpec(
        network="cnn",
        input_shape=input_shape,
        layers=layers,
        weight_tensors=weight_tensors,
        bias_tensors=bias_tensors,
    )


def convert_onnx_to_nk(
    onnx_path: str | Path,
    output_path: str | Path | None = None,
    *,
    fuse_composite: bool = True,
    packager_fuse: bool | None = None,
    optimize: bool = True,
) -> Path:
    from .arch_writer import _arch_to_spec
    from .nk_optimize import optimize_nk
    from .reader import read_nk_bytes

    if packager_fuse is None:
        packager_fuse = True

    onnx_path = Path(onnx_path)
    output_path = Path(output_path) if output_path else onnx_path.with_suffix(".nk")
    spec = onnx_to_spec(onnx_path, fuse_composite=fuse_composite)
    if optimize:
        from .nk_optimize import OptimizeOptions

        arch, weights = read_nk_bytes(write_nk_bytes(spec))
        opt = optimize_nk(
            arch,
            weights,
            options=OptimizeOptions(fuse_composite=packager_fuse),
        )
        spec = _arch_to_spec(opt.arch, opt.weights)
    elif packager_fuse:
        from .nk_fuse import fuse_composite_blocks

        arch, weights = read_nk_bytes(write_nk_bytes(spec))
        fused = fuse_composite_blocks(arch, weights, verify_output=False)
        spec = _arch_to_spec(fused.arch, fused.weights)
    write_nk(output_path, spec)
    return output_path
