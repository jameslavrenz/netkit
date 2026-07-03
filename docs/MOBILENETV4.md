# MobileNetV4 UIB (`mobilenetv4_uib`)

Fused **Universal Inverted Bottleneck (UIB)** block from [MobileNetV4](https://arxiv.org/abs/2404.10518) — the core building block of **MNv4-Conv-Small** and other MobileNetV4 variants.

Layer kind **`9`**. Full **MNv4-Conv-Small** (stem + 18 UIB blocks + classifier) fits in one `.nk` file with `kMaxLayers=100`; see fixture below.

## Block math (NHWC)

Matches the reference PyTorch implementation (`d-li14/mobilenetv4.pytorch`):

1. **Optional start depthwise** — kernel 3 or 5 (0 = skip); BN; no ReLU
2. **Expand** — 1×1 conv to `make_divisible(in × expand_ratio, 8)` channels; BN; ReLU
3. **Optional middle depthwise** — kernel 3 or 5 (0 = skip); BN; ReLU
4. **Project** — 1×1 conv to `out_channels`; BN; no ReLU
5. **Residual** — add input when `stride == 1` and `in_channels == out_channels`

### Stride / downsampling

When `stride > 1`:

- If `middle_dw_kernel > 0` and `middle_dw_downsample` (default): middle DW uses `stride`
- Else if `start_dw_kernel > 0`: start DW uses `stride`

## MNv4-Conv-Small UIB configs (reference)

| Block | start_dw | middle_dw | stride | out | expand |
|-------|----------|-----------|--------|-----|--------|
| ExtraDW | 5 | 5 | 2 | 96 | 3.0 |
| IB | 0 | 3 | 1 | 96 | 2.0 |
| ConvNeXt-style | 3 | 0 | 1 | 96 | 4.0 |

Full Small spec: `mobilenetv4_conv_small()` in [mobilenetv4.pytorch](https://github.com/d-li14/mobilenetv4.pytorch).

## `.nk` layer descriptor (kind `9`)

| Field | Type |
|-------|------|
| `in_channels` | `uint32` |
| `out_channels` | `uint32` |
| `start_dw_kernel` | `uint8` (0, 3, or 5) |
| `middle_dw_kernel` | `uint8` (0, 3, or 5) |
| `stride` | `uint8` |
| `middle_dw_downsample` | `uint8` (0 or 1) |
| `expand_ratio` | `float32` |
| `reserved` | `uint32` |

## Weights (variable tensor pairs)

Each stage: conv weight + conv bias, then BN scale + BN beta (folded inference BN).

Order:

1. *(if start_dw)* start DW `[Cin,Kh,Kw]` + bias `[Cin]`; BN `[Cin]` + `[Cin]`
2. expand `[Cexp,Cin]` + bias `[Cexp]`; BN `[Cexp]` + `[Cexp]`
3. *(if middle_dw)* middle DW `[Cexp,Kh,Kw]` + bias `[Cexp]`; BN `[Cexp]` + `[Cexp]`
4. project `[Cout,Cexp]` + bias `[Cout]`; BN `[Cout]` + `[Cout]`

`Cexp = make_divisible(in_channels × expand_ratio, 8)`.

## Fixtures

### UIB block only

`models/mobilenetv4_small_uib.nk` — 4×4×4 input, IB-style UIB (`0,3,1,4,2.0`), one TCAS case.

```bash
python tools/write_mobilenetv4_small_uib_fixture.py
```

### Full MNv4-Conv-Small

`models/mobilenetv4_small.nk` — 56×56×3 input, 22 layers (stem + 18 backbone UIB blocks + global avg pool + 1×1 conv 1280 + flatten + dense 10), ~2.5M weights, one TCAS case. Uses the **large CNN arena** tier on CPU (32 MiB heap) because weights alone exceed 4 MiB.

```bash
python tools/write_mobilenetv4_small_fixture.py
```

Builder: `python/netkit/mobilenetv4_small.py` (`build_mobilenetv4_small_arch`). Block specs match [mobilenetv4.pytorch](https://github.com/d-li14/mobilenetv4.pytorch).

## Python

```python
arch = {
    "network": "cnn",
    "input": [H, W, C_in],
    "layers": [{
        "type": "mobilenetv4_uib",
        "in_channels": C_in,
        "out_channels": C_out,
        "start_dw_kernel": 0,
        "middle_dw_kernel": 3,
        "stride": 1,
        "expand_ratio": 2.0,
        "middle_dw_downsample": 1,
    }],
}
```

## C++

- Kernel: `MobileNetV4Uib::forward()` in `src/mobilenetv4_uib.cpp`
- Loader: `NkFormat::LayerKind::MobilenetV4Uib`
- Op registry: `NkMobilenetV4UibOpDescriptor`

Scratch: `2 × H × W × Cexp + (residual ? H × W × Cin : 0)` floats at load time.
