# ResNet-18 BasicBlock (`resnet_basic_block`)

Fused **ResNet BasicBlock** used in ResNet-18 and ResNet-34 (not the bottleneck block in ResNet-50+).

Layer kind **`10`**. A full ResNet-18 backbone (stem + 8 blocks + classifier) can be expressed in one `.nk` file with `kMaxLayers=100`.

## Block math (NHWC)

Matches the standard torchvision ResNet block (inference BN folded into scale/bias):

1. **Conv 3×3** — `in_channels → out_channels`, stride `s`, pad 1
2. **BatchNorm** + **ReLU**
3. **Conv 3×3** — `out_channels → out_channels`, stride 1, pad 1
4. **BatchNorm**
5. **Residual** — identity when `stride == 1` and `in_channels == out_channels`; else 1×1 conv + BN projection with stride `s`
6. **ReLU**

## `.nk` layer descriptor (kind `10`)

| Field | Type |
|-------|------|
| `in_channels` | `uint32` |
| `out_channels` | `uint32` |
| `stride` | `uint8` (1 or 2) |
| `reserved` | 3 bytes |

## Weights (tensor pairs: W then B)

Identity shortcut (`stride=1`, `in == out`):

1. conv1 `[out, 3, 3, in]` + bias `[out]`; BN scale + beta `[out]` each
2. conv2 `[out, 3, 3, out]` + bias; BN scale + beta

Projection shortcut (otherwise), append:

3. shortcut `[out, 1, 1, in]` + bias; BN scale + beta

## Fixture

`models/resnet18_basic_block.nk` — 4×4×4 input, identity BasicBlock, one TCAS case.

```bash
python tools/write_resnet18_basic_block_fixture.py
```

## Full backbone

`models/resnet18.nk` — stem (7×7 conv, ReLU, max pool) + 8 BasicBlocks + avg pool + dense head. Input **56×56×3**, 13 layers, ~11.2M weights. Regenerate:

```bash
python tools/write_resnet18_fixture.py
```

Builder: `python/netkit/resnet18.py` (`build_resnet18_arch`). Embedded TCAS tolerance **1e-4** (deep-network float accumulation).

### Pack PyTorch checkpoint

```bash
python -m netkit pack --arch resnet18 -o models/my_resnet18.nk --height 56 --width 56 --num-classes 10
# or
python tools/pack_resnet18_checkpoint.py -o models/my_resnet18.nk
```

Uses `python/netkit/torch_backbone_pack.py` to fold BatchNorm and map torchvision weights into composite BasicBlock tensors.

## Python

```python
arch = {
    "network": "cnn",
    "input": [H, W, C_in],
    "layers": [{
        "type": "resnet_basic_block",
        "in_channels": C_in,
        "out_channels": C_out,
        "stride": 1,
    }],
}
```

## C++

- Kernel: `ResNetBasicBlock::forward()` in `src/resnet_basic_block.cpp`
- Loader: `NkFormat::LayerKind::ResNetBasicBlock`
- Op registry: `NkResNetBasicBlockOpDescriptor`
