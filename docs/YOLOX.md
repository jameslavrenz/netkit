# YOLOX multi-scale detector (MobileNetV4-Small + Nano PAFPN)

Netkit ships an **anchor-free YOLOX-style object detector** with a **MobileNetV4-Conv-Small** backbone, **feature taps**, a fused **Nano-style PAFPN neck**, and **three decoupled heads** (`yolox_pafpn_multiscale`).

This is a **reimplementation for netkit** (sequential CNN + `.nk`), not a vendored copy of Megvii YOLOX. Architecture ideas (decoupled heads, PAFPN, Nano depthwise refine, nearest-2× top-down, exp-LTRB decode) follow the public YOLOX design; the backbone comes from **timm** `mobilenetv4_conv_small`.

## Architecture

```
Input (H×W×3)
  └─ MobileNetV4-Conv-Small backbone
       ├─ … → C3 (stride 8, 64 ch)  → feature_tap[0]
       ├─ … → C4 (stride 16, 96 ch) → feature_tap[1]
       └─ … → C5 (stride 32, 960 ch) ──┐
  └─ yolox_pafpn_multiscale (fused)     │
       ├─ lateral 1×1: C3/C4/C5 → H     │
       ├─ top-down nearest-2× upsample + add + DW+PW
       ├─ bottom-up stride-2 DW+PW      │
       └─ 3× YoloxDecoupledHead on N3/N4/N5
Output: flat concat [P3 | P4 | P5], each Hi×Wi×(4+1+num_classes)
```

The CNN runtime is **sequential (one tensor in/out)**. `feature_tap` is an identity pass-through that also copies the activation into a side buffer. The PAFPN layer reads `tap[0]=C3`, `tap[1]=C4`, and its layer input as `C5`.

### Why two new layer kinds (not a free FPN graph)

| Kind | Id | Role |
|------|----|------|
| `feature_tap` | **13** | Keep the backbone sequential while snapshotting C3/C4 for the neck |
| `yolox_pafpn_multiscale` | **14** | Fused Nano PAFPN + three heads (one composite op) |

**Nearest-neighbor 2× upsample** is required for the top-down path (C5→C4, C4→C3). It is implemented **inside** the PAFPN op (`UpsampleNearest2x` / `_upsample_nearest_2x`), not as a standalone `.nk` layer — same packaging choice as folding laterals, add, DW/PW, and heads into kind 14 so the runtime stays one-tensor-in/out.

Channel layout per spatial location:

| Channels | Meaning |
|----------|---------|
| 0–3 | Box **log**-distances (left, top, right, bottom); decode with `exp` → positive LTRB |
| 4 | Objectness logit |
| 5… | Class logits (`num_classes`) |

## Tap points (MNv4-Conv-Small)

0-based backbone block indices **before** inserting taps (`MOBILENETV4_CONV_SMALL_BLOCKS`):

| Feature | After block index | Spec | Stride / channels |
|---------|-------------------|------|-------------------|
| C3 | 4 | `conv_bn 1,1,64` | 8 / 64 |
| C4 | 10 | `uib 3,0,1,96,4.0` | 16 / 96 |
| C5 | 17 | `conv_bn 1,1,960` | 32 / 960 |

After inserting two taps, layer indices shift; the builder inserts taps immediately after producing C3 and C4.

## Weight catalog order (`yolox_pafpn_multiscale`)

1. **Laterals** `lat3`, `lat4`, `lat5` — 1×1 W+B each  
2. **Top-down** `td_p4`, `td_p3` — each DW 3×3 s1 + PW 1×1 (W+B each)  
3. **Bottom-up** `bu_n4`, `bu_n5` — each DW 3×3 s2 + PW 1×1  
4. **Heads** `head_p3`, `head_p4`, `head_p5` — same order as `pack_yolox_head_weights_flat` (stem, cls×N, reg×N, cls/reg/obj preds)

## Descriptor layouts

- **FeatureTap (kind 13):** `uint32 channels`, `uint8 tap_id`, `uint8 reserved[3]`
- **YoloxPafpnMultiscale (kind 14):** `uint32 c3_ch, c4_ch, c5_ch, hidden_dim, num_classes`; `uint8 num_convs`; `uint8 reserved[3]`

## Fixture models

| File | Input | Notes |
|------|-------|-------|
| `models/yolox_mnv4_small.nk` | 64×64×3 | Full backbone + taps + PAFPN; grids 8×8 / 4×4 / 2×2; 10 classes, hidden=64 (CI / TCAS fixture) |
| `models/yolox_pafpn_taps.nk` | 8×8×64 | Synthetic C3→tap→downsample→C4→tap→C5→PAFPN |
| `models/yolox_mnv4_pafpn_trained.nk` | packed from mini-train | Trained weights; **does not** replace the CI fixture |

Regenerate fixtures:

```bash
python tools/write_yolox_mnv4_detector_fixture.py
python tools/write_yolox_mnv4_pafpn_fixture.py   # same as detector fixture
python tools/write_yolox_head_only_fixture.py    # writes yolox_pafpn_taps.nk
```

## Python API

```python
from netkit.yolox_detector import build_yolox_mnv4_small_detector
from netkit.yolox_decode import decode_yolox_output
from netkit.reference_forward import forward_cnn

arch = build_yolox_mnv4_small_detector(height=416, width=416, num_classes=80, hidden_dim=64)
# ... pack weights, write .nk ...
output = forward_cnn(flat_input, arch, weights)
detections = decode_yolox_output(
    output, num_classes=80, input_height=416, input_width=416
)  # strides [8,16,32]; box channels via exp(LTRB)
```

`build_yolox_mnv4_small_detector` **always** builds backbone + taps + PAFPN (neck required).

## Smoke train (no COCO)

```bash
python tools/train_yolox_mnv4_pafpn_smoke.py --steps 30
```

Uses timm `mobilenetv4_conv_small` (ImageNet pretrained, features_only), freezes or lightly trains the backbone, trains neck+heads on synthetic random boxes for ~30 Adam steps, and saves a checkpoint under `models/checkpoints/`.

## Mini real-data train (coco128 / COCO val / train subset)

```bash
# tiny set
python tools/train_yolox_mnv4_pafpn_mini.py --source coco128 --steps 1000 --batch 4 --size 320

# bigger-but-cheap: COCO val2017 (~5k images), freeze then unfreeze
python tools/train_yolox_mnv4_pafpn_mini.py --source coco_val --max-images 5000 --holdout 200 \
  --steps 10000 --unfreeze-after 5000 --batch 4 --size 320 \
  --out models/checkpoints/yolox_mnv4_pafpn_coco_val.pt

# EMA fine-tune (no mosaic) — first pack bar clear (~0.185 COCO AP@0.5 w/ NMS@0.65)
python tools/train_yolox_mnv4_pafpn_mini.py --source coco_train --data data --max-images 50000 --holdout 200 \
  --steps 15000 --unfreeze-after 0 --batch 4 --size 320 --assign simota \
  --mosaic-prob 0.0 --ema-decay 0.999 --lr 5e-4 --backbone-lr 5e-5 \
  --init-from models/checkpoints/yolox_mnv4_pafpn_coco_train_50k_mosaic.pt \
  --out models/checkpoints/yolox_mnv4_pafpn_coco_train_50k_ema.pt

# EMA v2 — mosaic + multiscale continuation (~0.224 COCO AP@0.5 w/ NMS@0.65)
python tools/train_yolox_mnv4_pafpn_mini.py --source coco_train --data data --max-images 50000 --holdout 200 \
  --steps 25000 --unfreeze-after 0 --batch 4 --size 320 --assign simota \
  --mosaic-prob 0.5 --mosaic-close-frac 0.2 --multiscale \
  --ema-decay 0.999 --lr 2e-4 --backbone-lr 2e-5 \
  --init-from models/checkpoints/yolox_mnv4_pafpn_coco_train_50k_ema.pt \
  --out models/checkpoints/yolox_mnv4_pafpn_coco_train_50k_ema_v2.pt

# EMA v3 — longer lower-LR continuation (target ~0.30 COCO AP@0.5)
NETKIT_TRAIN_DEVICE=mps python tools/train_yolox_mnv4_pafpn_mini.py --source coco_train --data data \
  --max-images 50000 --holdout 200 \
  --steps 40000 --unfreeze-after 0 --batch 4 --size 320 --assign simota \
  --mosaic-prob 0.5 --mosaic-close-frac 0.2 --multiscale \
  --ema-decay 0.999 --lr 1e-4 --backbone-lr 1e-5 \
  --init-from models/checkpoints/yolox_mnv4_pafpn_coco_train_50k_ema_v2.pt \
  --out models/checkpoints/yolox_mnv4_pafpn_coco_train_50k_ema_v3.pt

# re-score with NMS (torch and/or exported TF Lite)
python tools/eval_yolox_holdout.py --backend both \
  --ckpt models/checkpoints/yolox_mnv4_pafpn_coco_train_50k_ema_v2.pt \
  --tflite benchmark/tflm/generated/yolox_mnv4_pafpn_320.tflite \
  --data data --holdout 200 --nms-iou 0.65

# pack only if holdout COCO AP@0.5 (with NMS) ≥ 0.15
python tools/pack_yolox_mnv4_pafpn_checkpoint.py \
  --ckpt models/checkpoints/yolox_mnv4_pafpn_coco_train_50k_ema_v2.pt \
  --hidden 64 --height 320 --width 320 \
  --out models/yolox_mnv4_pafpn_trained.nk
```

Downloads Ultralytics **coco128**, official **COCO val2017**, or a boxed **train2017 subset** (no 18GB zip). For `coco_train`, hold-out comes from **val2017** (no overlap). Training uses freeze→unfreeze, flip+color jitter, optional **4-tile Mosaic** (`--mosaic-prob`) with late close, optional **multi-scale** around `--size` (`--multiscale`), optional **EMA** (`--ema-decay`, eval/save EMA weights), **SimOTA** (default) or center-radius multi-positive (`--assign center`), exp-LTRB + GIoU box loss, and hold-out scoring (confidence, boxes, rough greedy mAP@0.5, and COCO-style AP@0.5 with class-aware NMS). `--init-from` / pack load with `pretrained=False` so warm-starts do not hit Hugging Face. Pack writes `yolox_mnv4_pafpn_trained.nk` only when COCO AP@0.5 on hold-out is worth keeping (CI fixture untouched). Current packed trained weights report hold-out **coco_map50 ≈ 0.224** (NMS@0.65).

## C++ runtime

Layer kinds: `feature_tap` (**13**), `yolox_pafpn_multiscale` (**14**). Load and run like any CNN. Post-processing stays on the host (`python/netkit/yolox_decode.py`).

**AOT:** use **float lower** on host/MPU (`python -m netkit aot models/yolox_pafpn_taps.nk -o out --strict-lower` → `g_feature_tap_*` + `YoloxPafpnMultiscale::forward`). Quant AOT is out of scope for YOLOX — detectors run float on Linux-class devices in practice.

Host sanity (packed trained weights vs torch EMA, decode+NMS):

```bash
make tools/nk_infer
python tools/sanity_yolox_trained_host.py --images 3
```

`nk_infer` accepts `@in.bin` / `--out-bin out.bin` for large 320² tensors (argv float lists hit OS limits). `nk_model_run` views the caller buffer directly (no 16k stack cap).

Manual construction: init backbone layers and `InitFeatureTapLayer` / `nk_cnn_init_feature_tap_layer` for tap ids 0/1, then `InitYoloxPafpnLayer` / `nk_cnn_init_yolox_pafpn_layer` (wires heads after init via `GetBlock(...).yolox_pafpn.block.heads[i]` in C++, or load from `.nk` in C), then `InitActivationBuffers` with the **network input** shape. After forward, read taps with `GetFeatureTapBuffer` / `nk_cnn_get_feature_tap_buffer`.

Arena: PAFPN scratch scales with `~8 × H3×W3×hidden` plus three head workspaces reused sequentially; tap buffers are side allocations sized to each tapped map. Nearest-2× upsample buffers are part of that neck scratch (not a separate op allocation).

## Tests

| Layer | What is verified |
|-------|------------------|
| **C++ TCAS** | `yolox_mnv4_small.nk`, `yolox_pafpn_taps.nk` vs Python reference |
| **Arch / taps** | Block indices, tap insertion, output element count |
| **Decode** | Multi-scale flat concat with strides 8/16/32; exp-LTRB geometry |
| **Runtime** | `tools/nk_infer` parity |

Python: `python/tests/test_yolox_pafpn.py`, `python/tests/test_yolox_detector.py`.

## Limitations

- **No NMS in runtime** — host decode helper supports optional class-aware NMS (`nms_iou_threshold`); train hold-out eval uses it  
- **No standalone upsample layer** — nearest-2× is private to the PAFPN composite  
- **Training is mini-scale** — SimOTA (default) + optional Mosaic/multi-scale; hold-out reports both greedy rough mAP and COCO-style AP@0.5 with NMS (not a full pycocotools suite)  
- **Depthwise Nano PAFPN** — add-based laterals/top-down/bottom-up (no extra bottom-up refine convs)

See also: [MOBILENETV4.md](MOBILENETV4.md), [NK_FORMAT.md](NK_FORMAT.md).
