"""Convert supported ONNX graphs into netkit .nk models."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import onnx
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Requires onnx: pip install onnx numpy") from exc

from .format import Activation, activation_from_name
from .writer import LayerSpec, ModelSpec, write_nk

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


def _conv_output_dim(size: int, kernel: int, stride: int, pad: int) -> int:
    return (size + 2 * pad - kernel) // stride + 1


def _symmetric_pool_pads(node) -> tuple[int, int]:
    pads = _attr_ints(node, "pads", [0, 0, 0, 0])
    if len(pads) < 2:
        return 0, 0
    pad_h = int(pads[0])
    pad_w = int(pads[1])
    if len(pads) >= 4 and (pads[0] != pads[2] or pads[1] != pads[3]):
        raise ValueError("asymmetric pool padding is not supported")
    return pad_h, pad_w


def _pool_output_dim(size: int, kernel: int, stride: int, pad: int) -> int:
    return (size + 2 * pad - kernel) // stride + 1


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
    if nxt.op_type == "Relu":
        return Activation.RELU, 0.01, 1
    if nxt.op_type == "Softmax":
        return Activation.SOFTMAX, 0.01, 1
    if nxt.op_type == "Sigmoid":
        return Activation.SIGMOID, 0.01, 1
    if nxt.op_type == "Tanh":
        return Activation.TANH, 0.01, 1
    if nxt.op_type == "LeakyRelu":
        return Activation.LEAKY_RELU, _attr_float(nxt, "alpha", 0.01), 1
    if nxt.op_type == "Clip" and _clip_is_relu6(nxt, initializers):
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


def _symmetric_conv_pads(node) -> tuple[int, int]:
    pads = _attr_ints(node, "pads", [0, 0, 0, 0])
    if len(pads) < 2:
        return 0, 0
    pad_h = int(pads[0])
    pad_w = int(pads[1])
    if len(pads) >= 4 and (pads[0] != pads[2] or pads[1] != pads[3]):
        raise ValueError("asymmetric conv padding is not supported")
    return pad_h, pad_w


def _has_graph_branches(nodes) -> bool:
    return any(node.op_type == "Add" for node in nodes)


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
    from .onnx_graph import OnnxGraph

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

    if node.op_type in {"Identity", "Dropout"}:
        consumed.add(node_index)
        return layers, weight_tensors, bias_tensors, spatial_h, spatial_w, channels, consumed

    if node.op_type == "Conv":
        weight = initializers[node.input[1]]
        bias = _optional_bias(initializers, node)
        group = _attr_int(node, "group", 1)
        kernel_shape = _attr_ints(node, "kernel_shape")
        strides = _attr_ints(node, "strides")
        kh = int(kernel_shape[0]) if kernel_shape else int(weight.shape[2])
        kw = int(kernel_shape[1]) if len(kernel_shape) > 1 else int(weight.shape[3])
        stride = int(strides[0]) if strides else 1
        pad_h, pad_w = _symmetric_conv_pads(node)
        out_c = int(weight.shape[0])
        in_c = int(channels)
        if group == in_c and out_c == in_c:
            layers.append(
                LayerSpec(
                    kind="depthwise_conv2d",
                    kernel_h=kh,
                    kernel_w=kw,
                    stride=stride,
                    filters=out_c,
                    activation=activation,
                    alpha=alpha,
                    pad_h=pad_h,
                    pad_w=pad_w,
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
                    pad_h=pad_h,
                    pad_w=pad_w,
                )
            )
            weight_tensors.append(_onnx_conv_to_netkit(weight.astype(np.float32)))
        else:
            raise ValueError("grouped conv (non-depthwise) is not supported")
        bias_tensors.append(
            bias.astype(np.float32) if bias is not None else np.zeros(out_c, dtype=np.float32)
        )
        spatial_h = _conv_output_dim(spatial_h, kh, stride, pad_h)
        spatial_w = _conv_output_dim(spatial_w, kw, stride, pad_w)
        if group == 1:
            channels = out_c
        consumed.add(node_index)
        return layers, weight_tensors, bias_tensors, spatial_h, spatial_w, channels, consumed

    if node.op_type == "MaxPool":
        kernel_shape = _attr_ints(node, "kernel_shape")
        strides = _attr_ints(node, "strides")
        kernel = int(kernel_shape[0]) if kernel_shape else 2
        stride = int(strides[0]) if strides else kernel
        pad_h, pad_w = _symmetric_pool_pads(node)
        layers.append(
            LayerSpec(kind="max_pool2d", pool_size=kernel, stride=stride, pad_h=pad_h, pad_w=pad_w)
        )
        spatial_h = _pool_output_dim(spatial_h, kernel, stride, pad_h)
        spatial_w = _pool_output_dim(spatial_w, kernel, stride, pad_w)
        consumed.add(node_index)
        return layers, weight_tensors, bias_tensors, spatial_h, spatial_w, channels, consumed

    if node.op_type in {"AveragePool", "AvgPool"}:
        kernel_shape = _attr_ints(node, "kernel_shape")
        strides = _attr_ints(node, "strides")
        kernel = int(kernel_shape[0]) if kernel_shape else 2
        stride = int(strides[0]) if strides else kernel
        pad_h, pad_w = _symmetric_pool_pads(node)
        layers.append(
            LayerSpec(kind="avg_pool2d", pool_size=kernel, stride=stride, pad_h=pad_h, pad_w=pad_w)
        )
        spatial_h = _pool_output_dim(spatial_h, kernel, stride, pad_h)
        spatial_w = _pool_output_dim(spatial_w, kernel, stride, pad_w)
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
        weight = initializers[node.input[1]]
        group = _attr_int(node, "group", 1)
        kernel_shape = _attr_ints(node, "kernel_shape")
        strides = _attr_ints(node, "strides")
        kh = int(kernel_shape[0]) if kernel_shape else int(weight.shape[2])
        kw = int(kernel_shape[1]) if len(kernel_shape) > 1 else int(weight.shape[3])
        stride = int(strides[0]) if strides else 1
        pad_h, pad_w = _symmetric_conv_pads(node)
        out_c = int(weight.shape[0])
        spatial_h = _conv_output_dim(spatial_h, kh, stride, pad_h)
        spatial_w = _conv_output_dim(spatial_w, kw, stride, pad_w)
        if group == 1:
            channels = out_c
        return spatial_h, spatial_w, channels
    if node.op_type == "MaxPool":
        kernel_shape = _attr_ints(node, "kernel_shape")
        strides = _attr_ints(node, "strides")
        kernel = int(kernel_shape[0]) if kernel_shape else 2
        stride = int(strides[0]) if strides else kernel
        pad_h, pad_w = _symmetric_pool_pads(node)
        spatial_h = _pool_output_dim(spatial_h, kernel, stride, pad_h)
        spatial_w = _pool_output_dim(spatial_w, kernel, stride, pad_w)
        return spatial_h, spatial_w, channels
    if node.op_type in {"AveragePool", "AvgPool"}:
        kernel_shape = _attr_ints(node, "kernel_shape")
        strides = _attr_ints(node, "strides")
        kernel = int(kernel_shape[0]) if kernel_shape else 2
        stride = int(strides[0]) if strides else kernel
        pad_h, pad_w = _symmetric_pool_pads(node)
        spatial_h = _pool_output_dim(spatial_h, kernel, stride, pad_h)
        spatial_w = _pool_output_dim(spatial_w, kernel, stride, pad_w)
        return spatial_h, spatial_w, channels
    if node.op_type == "GlobalAveragePool":
        return 1, 1, channels
    if node.op_type in {"BatchNormalization", "Flatten", "Gemm"}:
        return spatial_h, spatial_w, channels
    return None


def _onnx_to_spec_cnn_fused(model) -> ModelSpec:
    from .onnx_fuse import try_fuse_resnet_basic_block
    from .onnx_graph import build_onnx_graph, topo_order

    graph = build_onnx_graph(model)
    initializers = graph.initializers
    order = topo_order(graph)

    fusion_by_add: dict[int, object] = {}
    fusion_consumed: set[int] = set()
    spatial_h, spatial_w, channels = graph.input_shape

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
            if fused is not None:
                fusion_by_add[idx] = fused
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

    layers: list[LayerSpec] = []
    weight_tensors: list[np.ndarray] = []
    bias_tensors: list[np.ndarray] = []
    spatial_h, spatial_w, channels = graph.input_shape
    consumed: set[int] = set()

    for idx in order:
        if idx in fusion_consumed and idx not in fusion_by_add:
            continue
        if idx in fusion_by_add:
            fused = fusion_by_add[idx]
            layers.append(fused.layer)
            weight_tensors.extend(fused.weight_tensors)
            bias_tensors.extend(fused.bias_tensors)
            spatial_h, spatial_w, channels = fused.spatial_h, fused.spatial_w, fused.out_channels
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

    return ModelSpec(
        network="cnn",
        input_shape=graph.input_shape,
        layers=layers,
        weight_tensors=weight_tensors,
        bias_tensors=bias_tensors,
    )


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

            if node.op_type != "Gemm":
                raise ValueError(f"unsupported ONNX op for MLP: {node.op_type}")

            activation, alpha, skip = _peek_fused_activation(nodes, i, initializers)

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
            in_features = out_features
            i += 1 + skip

        return ModelSpec(
            network="mlp",
            input_shape=input_shape,
            layers=layers,
            weight_tensors=weight_tensors,
            bias_tensors=bias_tensors,
        )

    if network == "cnn" and fuse_composite and _has_graph_branches(nodes):
        return _onnx_to_spec_cnn_fused(model)

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
            pad_h, pad_w = _symmetric_conv_pads(node)
            out_c = int(weight.shape[0])
            in_c = int(channels)
            if group == in_c and out_c == in_c:
                layers.append(
                    LayerSpec(
                        kind="depthwise_conv2d",
                        kernel_h=kh,
                        kernel_w=kw,
                        stride=stride,
                        filters=out_c,
                        activation=activation,
                        alpha=alpha,
                        pad_h=pad_h,
                        pad_w=pad_w,
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
                        pad_h=pad_h,
                        pad_w=pad_w,
                    )
                )
                weight_tensors.append(_onnx_conv_to_netkit(weight.astype(np.float32)))
            else:
                raise ValueError("grouped conv (non-depthwise) is not supported")
            bias_tensors.append(
                (bias.astype(np.float32) if bias is not None else np.zeros(out_c, dtype=np.float32))
            )
            spatial_h = _conv_output_dim(spatial_h, kh, stride, pad_h)
            spatial_w = _conv_output_dim(spatial_w, kw, stride, pad_w)
            if group == 1:
                channels = out_c
            i += 1 + skip
            continue

        if node.op_type == "MaxPool":
            kernel_shape = _attr_ints(node, "kernel_shape")
            strides = _attr_ints(node, "strides")
            kernel = int(kernel_shape[0]) if kernel_shape else 2
            stride = int(strides[0]) if strides else kernel
            pad_h, pad_w = _symmetric_pool_pads(node)
            layers.append(
                LayerSpec(kind="max_pool2d", pool_size=kernel, stride=stride, pad_h=pad_h, pad_w=pad_w)
            )
            spatial_h = _pool_output_dim(spatial_h, kernel, stride, pad_h)
            spatial_w = _pool_output_dim(spatial_w, kernel, stride, pad_w)
            i += 1
            continue

        if node.op_type == "AveragePool" or node.op_type == "AvgPool":
            kernel_shape = _attr_ints(node, "kernel_shape")
            strides = _attr_ints(node, "strides")
            kernel = int(kernel_shape[0]) if kernel_shape else 2
            stride = int(strides[0]) if strides else kernel
            pad_h, pad_w = _symmetric_pool_pads(node)
            layers.append(
                LayerSpec(kind="avg_pool2d", pool_size=kernel, stride=stride, pad_h=pad_h, pad_w=pad_w)
            )
            spatial_h = _pool_output_dim(spatial_h, kernel, stride, pad_h)
            spatial_w = _pool_output_dim(spatial_w, kernel, stride, pad_w)
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
) -> Path:
    onnx_path = Path(onnx_path)
    output_path = Path(output_path) if output_path else onnx_path.with_suffix(".nk")
    spec = onnx_to_spec(onnx_path, fuse_composite=fuse_composite)
    write_nk(output_path, spec)
    return output_path
