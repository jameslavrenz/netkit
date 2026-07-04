# ConvNeXt V2 block (`convnextv2_block`)

Fused **ConvNeXt V2** residual block for `.nk` CNN models. Matches the block used in Meta [ConvNeXt V2](https://github.com/facebookresearch/ConvNeXt-V2) (including Atto-scale configs): depthwise 7×7 → LayerNorm → 4× MLP → GELU → GRN → project → residual.

Layer kind **`8`**. Standalone **LayerNorm2d** (kind **`11`**) is used for stem, stage downsample, and head norms in the full Atto backbone.

## Block math (NHWC)

For input `X` with shape `[H, W, C]`:

1. **Depthwise conv** — 7×7, stride 1, pad 3, per channel
2. **LayerNorm** — normalize across `C` at each spatial position; scale + bias per channel
3. **Pointwise expand** — dense `C → 4C` at each pixel
4. **GELU**
5. **GRN** — `γ * (x * Nx) + β + x` where `Nx[c] = ||X_c||_2 / (mean_c(||X_c||_2) + ε)`
6. **Pointwise project** — dense `4C → C` at each pixel
7. **Residual** — add input `X`

Spatial size is unchanged (`H`, `W` constant).

## `.nk` layer descriptor

| Field | Type | Notes |
|-------|------|-------|
| `kind` | `uint8` | `8` |
| `channels` | `uint32` | Must match input channel count |
| `reserved` | `uint32` | `0` |
| `eps` | `float32` | LayerNorm + GRN epsilon (default `1e-6`) |

## Weights (5 tensor pairs)

Catalog order: weight then bias for each pair.

| Pair | Weight shape | Bias shape |
|------|--------------|------------|
| Depthwise | `[C, 7, 7]` | `[C]` |
| LayerNorm | `[C]` | `[C]` |
| Expand | `[4C, C]` | `[4C]` |
| GRN | `[4C]` | `[4C]` |
| Project | `[C, 4C]` | `[C]` |

Total floats for channel count `C`: `98C + 4C²` (e.g. `C=2` → 162 floats).

## Fixture

`models/convnextv2_atto_block.nk` — 4×4×2 input, one block, one embedded TCAS case. Regenerate:

```bash
python tools/write_convnextv2_atto_block_fixture.py
```

## Full backbone (ConvNeXt V2-Atto)

`models/convnextv2_atto.nk` — official Atto depths `[2,2,6,2]`, dims `[40,80,160,320]`, stem + four stages + head. Input **32×32×3** (divisible by 32), 24 layers. Regenerate:

```bash
python tools/write_convnextv2_atto_fixture.py
```

Builder: `python/netkit/convnextv2_atto.py` (`build_convnextv2_atto_arch`). Uses `layernorm2d` (kind `11`) between stem/stage convs and before the classifier pool.

### Pack PyTorch checkpoint

Requires `pip install -e "python[train]"` (torch + timm):

```bash
python -m netkit pack --arch convnextv2_atto -o models/my_convnextv2_atto.nk --height 32 --width 32 --num-classes 10
# or
python tools/pack_convnextv2_atto_checkpoint.py -o models/my_convnextv2_atto.nk
```

Uses `python/netkit/torch_backbone_pack.py` to map timm `convnextv2_atto` weights into composite block tensors (LayerNorm2d, ConvNeXt V2 blocks, classifier head).

Parity: `python/tests/test_torch_backbone_pack.py` and `test_torch_backbone_runtime_parity.py` — see [TESTING.md](TESTING.md).

## Python

```python
from netkit.arch_writer import write_nk_from_arch

arch = {
    "network": "cnn",
    "input": [H, W, C],
    "layers": [{"type": "convnextv2_block", "channels": C, "eps": 1e-6}],
}
```

Reference forward: `netkit.reference_forward.forward_cnn`.

## C++

- Kernel: `ConvNeXtV2Block::forward()` in `src/convnextv2_block.cpp`
- Loader: `NkFormat::LayerKind::ConvNeXtV2Block` in `src/nk_loader.cpp`
- Op registry: `NkConvNeXtV2BlockOpDescriptor`

Scratch for the block (`H*W*4*C + 4*C` floats) is allocated from the load arena at init time.
