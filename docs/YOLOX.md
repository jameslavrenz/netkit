# YOLOX single-scale detector (MobileNetV4-Small)

Netkit ships a **single-scale, anchor-free YOLOX-style object detector** built from the existing **MobileNetV4-Conv-Small** backbone plus a fused **decoupled detection head** layer kind (`yolox_decoupled_head`).

## Architecture

```
Input (H×W×3)
  └─ MobileNetV4-Conv-Small backbone (18 UIB/conv blocks, no classifier head)
  └─ YOLOX decoupled head (fused composite layer)
       ├─ stem: 1×1 conv + SiLU
       ├─ cls branch: stacked 3×3 convs + SiLU → 1×1 cls prediction
       ├─ reg branch: stacked 3×3 convs + SiLU → 1×1 box regression (ltrb)
       └─ obj branch: 1×1 objectness (from reg features)
Output (H'×W'×(4+1+num_classes))  — NHWC, single tensor
```

Channel layout per spatial location:

| Channels | Meaning |
|----------|---------|
| 0–3 | Box offsets (left, top, right, bottom) |
| 4 | Objectness logit |
| 5… | Class logits (`num_classes`) |

The head runs cls/reg/obj branches **in parallel inside the fused layer** (netkit graphs are otherwise sequential). This matches YOLOX decoupled-head semantics at a single stride.

## Fixture model

| File | Input | Head | Output grid |
|------|-------|------|-------------|
| `models/yolox_mnv4_small.nk` | 56×56×3 | hidden=64, 2 stacked convs, 10 classes | 2×2×15 (60 floats) |
| `models/yolox_head_only.nk` | 2×2×960 (synthetic backbone features) | hidden=32, 2 stacked convs, 5 classes | 2×2×10 (40 floats) |

Regenerate:

```bash
python tools/write_yolox_mnv4_detector_fixture.py
python tools/write_yolox_head_only_fixture.py
```

## Python API

```python
from netkit.yolox_detector import build_yolox_mnv4_small_detector
from netkit.yolox_decode import decode_yolox_output
from netkit.reference_forward import forward_cnn

arch = build_yolox_mnv4_small_detector(height=416, width=416, num_classes=80, hidden_dim=256)
# ... pack weights, write .nk with write_nk_from_arch ...
output = forward_cnn(flat_input, arch, weights)
detections = decode_yolox_output(output, num_classes=80, input_height=416, input_width=416)
```

Builder parameters:

- `height`, `width` — input resolution (backbone stride ≈ 32 → output grid is H/32 × W/32)
- `num_classes` — detection classes (default 80)
- `hidden_dim` — head width (default 256, YOLOX-style)
- `num_convs` — stacked 3×3 convs per branch (default 2, max 4)

## C++ runtime

The fused layer is registered in the op resolver as `yolox_decoupled_head` (`LayerKind` value **12**). Load and run like any CNN:

```bash
./netkit run models/yolox_mnv4_small.nk --input <9408 comma-separated floats>
./netkit inspect models/yolox_mnv4_small.nk --full
```

Post-processing (sigmoid, grid decode, NMS) stays on the host — see `python/netkit/yolox_decode.py`.

## Manual construction

Most deployments load `yolox_mnv4_small.nk` or `yolox_head_only.nk` via the loader. To attach a head to your own backbone features in code, follow the **same call order as any CNN**: create network → init layers in index order → `InitActivationBuffers` with **network input** shape → `forward`.

| Step | C++ | C |
|------|-----|---|
| 1. Arena | `arena.init(memory, size)` | `nk_arena_init(&arena, memory, size)` |
| 2. Network | `CNNNetwork(n, arena)` | `nk_cnn_create(&arena, n, &cnn)` |
| 3. Layers | `InitConvLayer` / … / `InitYoloxDecoupledHeadLayer(i, arena, h, w, …)` | `nk_cnn_init_*_layer` / `nk_cnn_init_yolox_decoupled_head_layer` |
| 4. Buffers | `InitActivationBuffers(arena, in_h, in_w, in_c)` | `nk_cnn_init_activation_buffers(&cnn, &arena, in_h, in_w, in_c)` |
| 5. Inference | `forward(input, arena)` → `GetOutput()` | `nk_cnn_forward(&cnn, &arena, &input, &output)` |

**Head-only** (backbone already produced `[H, W, C]` features): use `n = 1`, set `spatial_h/w` to the feature map size, `in_channels = C`, then call `InitActivationBuffers(arena, spatial_h, spatial_w, in_channels)`.

**Full detector** (backbone + head): init UIB / conv layers for the backbone first (`layer_idx` 0…N−2), then `InitYoloxDecoupledHeadLayer(N−1, arena, h', w', …)` where `h', w'` are the spatial size **entering the head** (e.g. 2×2 on the bundled model). `InitActivationBuffers` always uses the **original network input** (e.g. 56×56×3), not the head's feature size.

Stacked branch conv pointers: `cls_conv_weights[i]` and `reg_conv_weights[i]` for `i = 0 … num_convs−1` (max 4). Output tensor layout: NHWC `[H, W, 4 + 1 + num_classes]` — ltrb box (4), objectness (1), class logits (`num_classes`).

Worked examples: [cpp-api.md](cpp-api.md#manual-construction-call-order-1), [c-api.md](c-api.md#cnn-manual-construction-call-order). Tests: `tests/test_c_api.c` (`TestManualYoloxDecoupledHeadLayer`).

## Arena sizing

Use `./netkit inspect models/yolox_mnv4_small.nk --full` for activation high-water. The fused head allocates scratch proportional to `3 × H' × W' × hidden_dim` inside the layer (not counted in ping-pong buffers).

For MCU firmware, weights stay flash/blob-backed; size SRAM from inspect arena peaks and use `flash_payload_bytes` for flash budget (see [ARENA.md](ARENA.md)).

## Tests

Synthetic regression covers the **full MobileNetV4-Small → YOLOX chain** and an **isolated head** fixture:

| Layer | What is verified |
|-------|------------------|
| **C++ embedded TCAS** | `yolox_mnv4_small.nk` (19 layers, random 56×56×3 input) and `yolox_head_only.nk` (head on 2×2×960 features) vs Python reference |
| **Backbone chain** | YOLOX backbone layers match `build_mobilenetv4_small_arch(include_head=False)`; shared weight prefix; backbone-only forward equals MNv4 without head |
| **Head composition** | `forward(backbone) → head` equals full-detector forward (fixture + fresh random weights) |
| **Decode golden** | Hand-built 1×1 output tensor → expected box geometry and score threshold behavior |
| **Runtime parity** | `tools/nk_infer` matches NumPy reference on both fixtures and on a temp-written model |

Python: `python/tests/test_yolox_detector.py` (run via `make test-python`).

C++: `src/test.cpp` includes both `.nk` fixtures (**88** total embedded cases across the suite).

C API smoke tests: `tests/test_c_api.c` (`TestManualYoloxDecoupledHeadLayer`, `yolox_head_only.nk` in composite load pass). Manual construction patterns: [c-api.md](c-api.md#cnn-manual-construction-call-order), [cpp-api.md](cpp-api.md#manual-construction-call-order-1).

## Limitations (Phase 1)

- **Single scale only** — no FPN/PAN multi-stride neck
- **No NMS in runtime** — host decode helper only
- **Random-weight fixture** — timm/YOLOX-trained checkpoint packing is future work

See also: [MOBILENETV4.md](MOBILENETV4.md), [NK_FORMAT.md](NK_FORMAT.md).
