"""Graph optimizations for .nk models before AOT embedding."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .cnn_layers import conv2d_input_channels, depthwise_kernel_hw
from .reference_forward import forward_cnn, forward_mlp, pack_cnn_weights, pack_mlp_weights


@dataclass(frozen=True)
class OptimizeOptions:
    """Stable packager-side optimizations (default-on when AOT optimize is enabled)."""

    fold_conv_batch_norm: bool = True
    fold_batch_norm_into_dense: bool = True
    merge_linear_dense: bool = True
    remove_identity_batch_norm: bool = True
    fuse_composite: bool = False
    verbose_fuse: bool = False


@dataclass
class OptimizeResult:
    arch: dict[str, Any]
    weights: np.ndarray
    applied: list[str] = field(default_factory=list)


def _out_dim(in_dim: int, kernel: int, stride: int, pad: int = 0) -> int:
    return (in_dim + 2 * pad - kernel) // stride + 1


@dataclass
class _TensorPair:
    weight: np.ndarray | None = None
    bias: np.ndarray | None = None


@dataclass
class _GraphLayer:
    spec: dict[str, Any]
    tensors: _TensorPair
    # Multi-tensor composites (UIB / ResNet / ConvNeXt); empty for primitives.
    tensor_pairs: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)


@dataclass
class _ShapeState:
    height: int
    width: int
    channels: int
    dense_in: int = 0


def _decompose(arch: dict[str, Any], weights: np.ndarray) -> tuple[list[_GraphLayer], _ShapeState]:
    network = arch["network"]
    if network == "mlp":
        return _decompose_mlp(arch, weights)
    if network == "cnn":
        return _decompose_cnn(arch, weights)
    raise ValueError(f"unsupported network: {network}")


def _decompose_mlp(arch: dict[str, Any], weights: np.ndarray) -> tuple[list[_GraphLayer], _ShapeState]:
    layers: list[_GraphLayer] = []
    offset = 0
    in_features = arch["input"][1]
    for layer in arch["layers"]:
        out_features = layer["units"]
        w_size = in_features * out_features
        w = weights[offset : offset + w_size].reshape(out_features, in_features).copy()
        offset += w_size
        b = weights[offset : offset + out_features].copy()
        offset += out_features
        layers.append(_GraphLayer(spec=dict(layer), tensors=_TensorPair(weight=w, bias=b)))
        in_features = out_features
    if offset != len(weights):
        raise ValueError(f"weight count mismatch: used {offset}, file has {len(weights)}")
    return layers, _ShapeState(height=1, width=arch["input"][1], channels=1, dense_in=in_features)


def _decompose_cnn(arch: dict[str, Any], weights: np.ndarray) -> tuple[list[_GraphLayer], _ShapeState]:
    from .arch_writer import _make_divisible, _resnet_output_spatial, _uib_output_spatial

    layers: list[_GraphLayer] = []
    height, width, channels = arch["input"]
    dense_in = 0
    offset = 0
    for layer in arch["layers"]:
        layer_type = layer["type"]
        if layer_type == "conv2d":
            k = layer["kernel_size"]
            stride = layer.get("stride", 1)
            pad_h = layer.get("pad_h", 0)
            pad_w = layer.get("pad_w", 0)
            out_c = layer["filters"]
            in_c = conv2d_input_channels(layer, channels)
            kernel_elems = k * k * in_c
            w = weights[offset : offset + kernel_elems * out_c].reshape(out_c, k, k, in_c).copy()
            offset += kernel_elems * out_c
            b = weights[offset : offset + out_c].copy()
            offset += out_c
            layers.append(_GraphLayer(spec=dict(layer), tensors=_TensorPair(weight=w, bias=b)))
            height = _out_dim(height, k, stride, pad_h)
            width = _out_dim(width, k, stride, pad_w)
            channels = out_c
        elif layer_type == "depthwise_conv2d":
            kh, kw = depthwise_kernel_hw(layer)
            stride = layer.get("stride", 1)
            pad_h = layer.get("pad_h", 0)
            pad_w = layer.get("pad_w", 0)
            ch = layer["filters"]
            kernel_elems = kh * kw * ch
            w = weights[offset : offset + kernel_elems].reshape(ch, kh, kw).copy()
            offset += kernel_elems
            b = weights[offset : offset + ch].copy()
            offset += ch
            layers.append(_GraphLayer(spec=dict(layer), tensors=_TensorPair(weight=w, bias=b)))
            height = _out_dim(height, kh, stride, pad_h)
            width = _out_dim(width, kw, stride, pad_w)
        elif layer_type in {"max_pool2d", "avg_pool2d"}:
            pool = layer["pool_size"]
            stride = layer.get("stride", pool)
            pad_h = layer.get("pad_h", 0)
            pad_w = layer.get("pad_w", 0)
            layers.append(_GraphLayer(spec=dict(layer), tensors=_TensorPair()))
            height = _out_dim(height, pool, stride, pad_h)
            width = _out_dim(width, pool, stride, pad_w)
        elif layer_type == "batch_norm2d":
            ch = layer["channels"]
            scale = weights[offset : offset + ch].copy()
            offset += ch
            bias = weights[offset : offset + ch].copy()
            offset += ch
            layers.append(_GraphLayer(spec=dict(layer), tensors=_TensorPair(weight=scale, bias=bias)))
        elif layer_type == "layernorm2d":
            ch = layer["channels"]
            ln_w = weights[offset : offset + ch].copy()
            offset += ch
            ln_b = weights[offset : offset + ch].copy()
            offset += ch
            layers.append(_GraphLayer(spec=dict(layer), tensors=_TensorPair(weight=ln_w, bias=ln_b)))
        elif layer_type == "mobilenetv4_uib":
            in_c = layer["in_channels"]
            out_c = layer["out_channels"]
            start_k = int(layer.get("start_dw_kernel", 0))
            middle_k = int(layer.get("middle_dw_kernel", 0))
            expand_c = _make_divisible(in_c * float(layer["expand_ratio"]), 8)
            pairs: list[tuple[np.ndarray, np.ndarray]] = []
            if start_k:
                dw_elems = start_k * start_k * in_c
                w = weights[offset : offset + dw_elems].reshape(in_c, start_k, start_k).copy()
                offset += dw_elems
                b = weights[offset : offset + in_c].copy()
                offset += in_c
                pairs.append((w, b))
                scale = weights[offset : offset + in_c].copy()
                offset += in_c
                beta = weights[offset : offset + in_c].copy()
                offset += in_c
                pairs.append((scale, beta))
            expand_elems = expand_c * in_c
            w = weights[offset : offset + expand_elems].reshape(expand_c, 1, 1, in_c).copy()
            offset += expand_elems
            b = weights[offset : offset + expand_c].copy()
            offset += expand_c
            pairs.append((w, b))
            scale = weights[offset : offset + expand_c].copy()
            offset += expand_c
            beta = weights[offset : offset + expand_c].copy()
            offset += expand_c
            pairs.append((scale, beta))
            if middle_k:
                dw_elems = middle_k * middle_k * expand_c
                w = weights[offset : offset + dw_elems].reshape(expand_c, middle_k, middle_k).copy()
                offset += dw_elems
                b = weights[offset : offset + expand_c].copy()
                offset += expand_c
                pairs.append((w, b))
                scale = weights[offset : offset + expand_c].copy()
                offset += expand_c
                beta = weights[offset : offset + expand_c].copy()
                offset += expand_c
                pairs.append((scale, beta))
            proj_elems = out_c * expand_c
            w = weights[offset : offset + proj_elems].reshape(out_c, 1, 1, expand_c).copy()
            offset += proj_elems
            b = weights[offset : offset + out_c].copy()
            offset += out_c
            pairs.append((w, b))
            scale = weights[offset : offset + out_c].copy()
            offset += out_c
            beta = weights[offset : offset + out_c].copy()
            offset += out_c
            pairs.append((scale, beta))
            layers.append(_GraphLayer(spec=dict(layer), tensors=_TensorPair(), tensor_pairs=pairs))
            height, width = _uib_output_spatial(height, width, layer)
            channels = out_c
        elif layer_type == "resnet_basic_block":
            in_c = layer["in_channels"]
            out_c = layer["out_channels"]
            stride = int(layer.get("stride", 1))
            identity = stride == 1 and in_c == out_c
            pairs = []
            conv1_elems = out_c * 3 * 3 * in_c
            w = weights[offset : offset + conv1_elems].reshape(out_c, 3, 3, in_c).copy()
            offset += conv1_elems
            b = weights[offset : offset + out_c].copy()
            offset += out_c
            pairs.append((w, b))
            scale = weights[offset : offset + out_c].copy()
            offset += out_c
            beta = weights[offset : offset + out_c].copy()
            offset += out_c
            pairs.append((scale, beta))
            conv2_elems = out_c * 3 * 3 * out_c
            w = weights[offset : offset + conv2_elems].reshape(out_c, 3, 3, out_c).copy()
            offset += conv2_elems
            b = weights[offset : offset + out_c].copy()
            offset += out_c
            pairs.append((w, b))
            scale = weights[offset : offset + out_c].copy()
            offset += out_c
            beta = weights[offset : offset + out_c].copy()
            offset += out_c
            pairs.append((scale, beta))
            if not identity:
                shortcut_elems = out_c * in_c
                w = weights[offset : offset + shortcut_elems].reshape(out_c, 1, 1, in_c).copy()
                offset += shortcut_elems
                b = weights[offset : offset + out_c].copy()
                offset += out_c
                pairs.append((w, b))
                scale = weights[offset : offset + out_c].copy()
                offset += out_c
                beta = weights[offset : offset + out_c].copy()
                offset += out_c
                pairs.append((scale, beta))
            layers.append(_GraphLayer(spec=dict(layer), tensors=_TensorPair(), tensor_pairs=pairs))
            height, width = _resnet_output_spatial(height, width, layer)
            channels = out_c
        elif layer_type == "convnextv2_block":
            ch = layer["channels"]
            expanded = ch * 4
            pairs = []
            dw_elems = 7 * 7 * ch
            w = weights[offset : offset + dw_elems].reshape(ch, 7, 7).copy()
            offset += dw_elems
            b = weights[offset : offset + ch].copy()
            offset += ch
            pairs.append((w, b))
            ln_w = weights[offset : offset + ch].copy()
            offset += ch
            ln_b = weights[offset : offset + ch].copy()
            offset += ch
            pairs.append((ln_w, ln_b))
            pw1_elems = expanded * ch
            w = weights[offset : offset + pw1_elems].reshape(expanded, ch).copy()
            offset += pw1_elems
            b = weights[offset : offset + expanded].copy()
            offset += expanded
            pairs.append((w, b))
            grn_g = weights[offset : offset + expanded].copy()
            offset += expanded
            grn_b = weights[offset : offset + expanded].copy()
            offset += expanded
            pairs.append((grn_g, grn_b))
            pw2_elems = ch * expanded
            w = weights[offset : offset + pw2_elems].reshape(ch, expanded).copy()
            offset += pw2_elems
            b = weights[offset : offset + ch].copy()
            offset += ch
            pairs.append((w, b))
            layers.append(_GraphLayer(spec=dict(layer), tensors=_TensorPair(), tensor_pairs=pairs))
        elif layer_type == "flatten":
            layers.append(_GraphLayer(spec=dict(layer), tensors=_TensorPair()))
            dense_in = height * width * channels
        elif layer_type == "dense":
            out_f = layer["units"]
            w_size = dense_in * out_f
            w = weights[offset : offset + w_size].reshape(out_f, dense_in).copy()
            offset += w_size
            b = weights[offset : offset + out_f].copy()
            offset += out_f
            layers.append(_GraphLayer(spec=dict(layer), tensors=_TensorPair(weight=w, bias=b)))
            dense_in = out_f
        else:
            raise ValueError(f"unsupported layer type: {layer_type}")
    if offset != len(weights):
        raise ValueError(f"weight count mismatch: used {offset}, file has {len(weights)}")
    state = _ShapeState(height=height, width=width, channels=channels, dense_in=dense_in)
    return layers, state


def _compose_mlp(arch: dict[str, Any], layers: list[_GraphLayer]) -> tuple[dict[str, Any], np.ndarray]:
    packed: list[tuple[np.ndarray, np.ndarray]] = []
    for layer in layers:
        if layer.spec["type"] != "dense":
            raise ValueError("MLP compose expects dense layers only")
        assert layer.tensors.weight is not None and layer.tensors.bias is not None
        packed.append((layer.tensors.weight, layer.tensors.bias))
    new_arch = {"network": "mlp", "input": list(arch["input"]), "layers": [layer.spec for layer in layers]}
    return new_arch, pack_mlp_weights(packed)


def _compose_cnn(arch: dict[str, Any], layers: list[_GraphLayer]) -> tuple[dict[str, Any], np.ndarray]:
    tensors: list[tuple[np.ndarray, np.ndarray] | None] = []
    for layer in layers:
        layer_type = layer.spec["type"]
        if layer_type in {"max_pool2d", "avg_pool2d", "flatten"}:
            tensors.append(None)
        elif layer_type in {"conv2d", "depthwise_conv2d", "batch_norm2d", "dense"}:
            assert layer.tensors.weight is not None and layer.tensors.bias is not None
            tensors.append((layer.tensors.weight, layer.tensors.bias))
        else:
            raise ValueError(f"unsupported layer type: {layer_type}")
    new_arch = {"network": "cnn", "input": list(arch["input"]), "layers": [layer.spec for layer in layers]}
    return new_arch, pack_cnn_weights(tensors)


def _compose(arch: dict[str, Any], layers: list[_GraphLayer]) -> tuple[dict[str, Any], np.ndarray]:
    if arch["network"] == "mlp":
        return _compose_mlp(arch, layers)
    return _compose_cnn(arch, layers)


def _simulate_shape(arch: dict[str, Any]) -> dict[int, _ShapeState]:
    """Map layer index -> spatial shape entering that layer."""
    shapes: dict[int, _ShapeState] = {}
    if arch["network"] == "mlp":
        return shapes
    height, width, channels = arch["input"]
    dense_in = 0
    for index, layer in enumerate(arch["layers"]):
        shapes[index] = _ShapeState(height=height, width=width, channels=channels, dense_in=dense_in)
        layer_type = layer["type"]
        if layer_type == "conv2d":
            k = layer["kernel_size"]
            stride = layer.get("stride", 1)
            pad_h = layer.get("pad_h", 0)
            pad_w = layer.get("pad_w", 0)
            height = _out_dim(height, k, stride, pad_h)
            width = _out_dim(width, k, stride, pad_w)
            channels = layer["filters"]
        elif layer_type == "depthwise_conv2d":
            kh, kw = depthwise_kernel_hw(layer)
            stride = layer.get("stride", 1)
            pad_h = layer.get("pad_h", 0)
            pad_w = layer.get("pad_w", 0)
            height = _out_dim(height, kh, stride, pad_h)
            width = _out_dim(width, kw, stride, pad_w)
        elif layer_type in {"max_pool2d", "avg_pool2d"}:
            pool = layer["pool_size"]
            stride = layer.get("stride", pool)
            pad_h = layer.get("pad_h", 0)
            pad_w = layer.get("pad_w", 0)
            height = _out_dim(height, pool, stride, pad_h)
            width = _out_dim(width, pool, stride, pad_w)
        elif layer_type == "flatten":
            dense_in = height * width * channels
        elif layer_type == "dense":
            dense_in = layer["units"]
    return shapes


def _channel_for_flat_index(index: int, height: int, width: int, channels: int) -> int:
    rem = index % (width * channels)
    return rem % channels


def _fold_conv_batch_norm(layers: list[_GraphLayer]) -> bool:
    changed = False
    index = 0
    while index < len(layers) - 1:
        conv = layers[index]
        bn = layers[index + 1]
        if conv.spec.get("type") != "conv2d" or bn.spec.get("type") != "batch_norm2d":
            index += 1
            continue
        if conv.spec["filters"] != bn.spec["channels"]:
            index += 1
            continue
        assert conv.tensors.weight is not None and conv.tensors.bias is not None
        assert bn.tensors.weight is not None and bn.tensors.bias is not None
        scale = bn.tensors.weight
        bn_bias = bn.tensors.bias
        conv.tensors.weight = conv.tensors.weight * scale.reshape(-1, 1, 1, 1)
        conv.tensors.bias = conv.tensors.bias * scale + bn_bias
        del layers[index + 1]
        changed = True
    return changed


def _remove_identity_batch_norm(layers: list[_GraphLayer]) -> bool:
    changed = False
    kept: list[_GraphLayer] = []
    for layer in layers:
        if layer.spec.get("type") != "batch_norm2d":
            kept.append(layer)
            continue
        assert layer.tensors.weight is not None and layer.tensors.bias is not None
        if np.allclose(layer.tensors.weight, 1.0) and np.allclose(layer.tensors.bias, 0.0):
            changed = True
            continue
        kept.append(layer)
    if changed:
        layers[:] = kept
    return changed


def _fold_batch_norm_into_dense(
    arch: dict[str, Any], layers: list[_GraphLayer]
) -> bool:
    if arch["network"] != "cnn":
        return False
    shapes = _simulate_shape(arch)
    changed = False
    index = 0
    while index < len(layers) - 2:
        bn = layers[index]
        flatten = layers[index + 1]
        dense = layers[index + 2]
        if (
            bn.spec.get("type") != "batch_norm2d"
            or flatten.spec.get("type") != "flatten"
            or dense.spec.get("type") != "dense"
        ):
            index += 1
            continue
        shape = shapes.get(index)
        if shape is None:
            index += 1
            continue
        assert bn.tensors.weight is not None and bn.tensors.bias is not None
        assert dense.tensors.weight is not None and dense.tensors.bias is not None
        scale = bn.tensors.weight
        bn_bias = bn.tensors.bias
        height, width, channels = shape.height, shape.width, shape.channels
        in_features = dense.tensors.weight.shape[1]
        if in_features != height * width * channels:
            index += 1
            continue
        dense_w = dense.tensors.weight.copy()
        dense_b = dense.tensors.bias.copy()
        bias_offset = np.zeros(in_features, dtype=np.float32)
        for col in range(in_features):
            channel = _channel_for_flat_index(col, height, width, channels)
            bias_offset[col] = bn_bias[channel]
        dense.tensors.bias = dense_b + dense_w @ bias_offset
        for col in range(in_features):
            channel = _channel_for_flat_index(col, height, width, channels)
            dense_w[:, col] *= scale[channel]
        dense.tensors.weight = dense_w
        del layers[index]
        changed = True
    return changed


def _merge_linear_dense(layers: list[_GraphLayer]) -> bool:
    changed = False
    index = 0
    while index < len(layers) - 1:
        first = layers[index]
        second = layers[index + 1]
        if first.spec.get("type") != "dense" or second.spec.get("type") != "dense":
            index += 1
            continue
        if first.spec.get("activation", "none") != "none":
            index += 1
            continue
        assert first.tensors.weight is not None and first.tensors.bias is not None
        assert second.tensors.weight is not None and second.tensors.bias is not None
        w1, b1 = first.tensors.weight, first.tensors.bias
        w2, b2 = second.tensors.weight, second.tensors.bias
        merged_spec = dict(second.spec)
        first.spec = merged_spec
        first.tensors.weight = w2 @ w1
        first.tensors.bias = w2 @ b1 + b2
        del layers[index + 1]
        changed = True
    return changed


def _forward(arch: dict[str, Any], weights: np.ndarray, flat_input: np.ndarray) -> np.ndarray:
    if arch["network"] == "mlp":
        return np.asarray(forward_mlp(flat_input, arch, weights), dtype=np.float32)
    return np.asarray(forward_cnn(flat_input, arch, weights), dtype=np.float32)


def _optimization_passes_enabled(opts: OptimizeOptions) -> bool:
    return any(
        (
            opts.fold_conv_batch_norm,
            opts.fold_batch_norm_into_dense,
            opts.merge_linear_dense,
            opts.remove_identity_batch_norm,
        )
    )


_COMPOSITE_LAYER_TYPES = frozenset(
    {"mobilenetv4_uib", "resnet_basic_block", "convnextv2_block"}
)


def _has_composite_layers(arch: dict[str, Any]) -> bool:
    return any(layer.get("type") in _COMPOSITE_LAYER_TYPES for layer in arch.get("layers", []))


def optimize_nk(
    arch: dict[str, Any],
    weights: np.ndarray,
    *,
    options: OptimizeOptions | None = None,
    atol: float = 1e-5,
) -> OptimizeResult:
    """Apply stable graph optimizations and verify numeric parity against the input model."""
    opts = options or OptimizeOptions()
    weights = np.asarray(weights, dtype=np.float32)

    probe_count = 3
    if arch["network"] == "mlp":
        batch, features = arch["input"]
        probes = [np.random.default_rng(7 + i).standard_normal(batch * features).astype(np.float32) for i in range(probe_count)]
    else:
        h, w, c = arch["input"]
        probes = [np.random.default_rng(7 + i).standard_normal(h * w * c).astype(np.float32) for i in range(probe_count)]

    current_arch = arch
    current_weights = weights
    applied: list[str] = []

    if opts.fuse_composite and current_arch.get("network") == "cnn":
        from .nk_fuse import FuseOptions, fuse_composite_blocks

        fused = fuse_composite_blocks(
            current_arch,
            current_weights,
            atol=atol,
            verify_output=False,
            options=FuseOptions(verbose_fuse=opts.verbose_fuse),
        )
        current_arch = fused.arch
        current_weights = fused.weights
        applied.extend(fused.applied)

    baseline = [_forward(current_arch, current_weights, probe) for probe in probes]

    while _optimization_passes_enabled(opts) and not _has_composite_layers(current_arch):
        layers, _ = _decompose(current_arch, current_weights)
        changed = False
        if opts.remove_identity_batch_norm and _remove_identity_batch_norm(layers):
            applied.append("remove_identity_batch_norm")
            changed = True
        if opts.fold_conv_batch_norm and _fold_conv_batch_norm(layers):
            applied.append("fold_conv_batch_norm")
            changed = True
        if opts.fold_batch_norm_into_dense and _fold_batch_norm_into_dense(current_arch, layers):
            applied.append("fold_batch_norm_into_dense")
            changed = True
        if opts.merge_linear_dense and _merge_linear_dense(layers):
            applied.append("merge_linear_dense")
            changed = True
        if not changed:
            break
        current_arch, current_weights = _compose(
            {"network": arch["network"], "input": list(arch["input"])},
            layers,
        )
        for probe, expected in zip(probes, baseline):
            actual = _forward(current_arch, current_weights, probe)
            if not np.allclose(actual, expected, rtol=0.0, atol=atol):
                raise ValueError("optimization changed model output beyond tolerance")

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_applied: list[str] = []
    for name in applied:
        if name not in seen:
            seen.add(name)
            unique_applied.append(name)

    return OptimizeResult(arch=current_arch, weights=current_weights, applied=unique_applied)
