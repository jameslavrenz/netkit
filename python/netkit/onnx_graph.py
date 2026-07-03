"""ONNX value graph helpers for pattern fusion."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    import onnx
except ImportError:  # pragma: no cover
    onnx = None  # type: ignore[assignment]


@dataclass
class OnnxGraph:
    nodes: list
    initializers: dict[str, np.ndarray]
    input_name: str
    input_shape: list[int]  # NHWC for CNN
    producers: dict[str, int] = field(default_factory=dict)
    consumers: dict[str, list[tuple[int, int]]] = field(default_factory=dict)

    def node(self, index: int):
        return self.nodes[index]

    def producer_index(self, tensor: str) -> int | None:
        if tensor == self.input_name or tensor in self.initializers:
            return None
        idx = self.producers.get(tensor)
        return idx if idx is not None else None

    def producer_node(self, tensor: str):
        idx = self.producer_index(tensor)
        return self.nodes[idx] if idx is not None else None


def build_onnx_graph(model) -> OnnxGraph:
    from onnx import numpy_helper

    nodes = list(model.graph.node)
    initializers = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
    if not model.graph.input:
        raise ValueError("ONNX graph has no inputs")

    input_value = model.graph.input[0]
    input_name = input_value.name
    dims = [int(d.dim_value) for d in input_value.type.tensor_type.shape.dim]
    if len(dims) == 4:
        _, channels, height, width = dims
        input_shape = [height, width, channels]
    elif len(dims) == 3:
        input_shape = dims
    else:
        raise ValueError("graph fusion requires CNN input rank 3 or 4")

    producers: dict[str, int] = {}
    consumers: dict[str, list[tuple[int, int]]] = {}
    for idx, node in enumerate(nodes):
        for out in node.output:
            producers[out] = idx
        for input_idx, inp in enumerate(node.input):
            consumers.setdefault(inp, []).append((idx, input_idx))

    return OnnxGraph(
        nodes=nodes,
        initializers=initializers,
        input_name=input_name,
        input_shape=input_shape,
        producers=producers,
        consumers=consumers,
    )


def topo_order(graph: OnnxGraph) -> list[int]:
    """Kahn topological sort over ONNX nodes."""
    indegree = [0] * len(graph.nodes)
    for idx, node in enumerate(graph.nodes):
        for inp in node.input:
            if graph.producer_index(inp) is not None:
                indegree[idx] += 1

    ready = [i for i, d in enumerate(indegree) if d == 0]
    order: list[int] = []
    while ready:
        idx = ready.pop()
        order.append(idx)
        for out in graph.nodes[idx].output:
            for consumer_idx, _ in graph.consumers.get(out, []):
                indegree[consumer_idx] -= 1
                if indegree[consumer_idx] == 0:
                    ready.append(consumer_idx)

    if len(order) != len(graph.nodes):
        raise ValueError("ONNX graph has cycles or unresolved inputs")
    return order


def skip_activation(node) -> bool:
    return node.op_type in {"Relu", "Sigmoid", "Tanh", "LeakyRelu", "Clip", "Softmax"}


def trace_through_activations(graph: OnnxGraph, tensor: str) -> str:
    """Follow single-consumer activation chains backward to the pre-activation tensor."""
    while True:
        node = graph.producer_node(tensor)
        if node is None or not skip_activation(node):
            return tensor
        tensor = node.input[0]


def trace_batch_norm(graph: OnnxGraph, tensor: str) -> tuple[int, str] | None:
    tensor = trace_through_activations(graph, tensor)
    node = graph.producer_node(tensor)
    if node is None or node.op_type != "BatchNormalization":
        return None
    idx = graph.producers[node.output[0]]
    return idx, node.input[0]


def trace_conv(graph: OnnxGraph, tensor: str) -> tuple[int, str, np.ndarray, np.ndarray | None] | None:
    tensor = trace_through_activations(graph, tensor)
    node = graph.producer_node(tensor)
    if node is None or node.op_type != "Conv":
        return None
    weight = graph.initializers[node.input[1]]
    bias = None
    if len(node.input) >= 3 and node.input[2] in graph.initializers:
        bias = graph.initializers[node.input[2]]
    idx = graph.producers[node.output[0]]
    return idx, node.input[0], weight, bias
