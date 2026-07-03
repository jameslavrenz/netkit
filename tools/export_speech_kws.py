#!/usr/bin/env python3
"""Train Speech Commands KWS CNN and export models/speech_kws.nk + ONNX parity sidecar.

Run from repo root:
    python3 tools/export_speech_kws.py

Requires: pip install -e "python[train]"

On first run, downloads Speech Commands v0.02, extracts mel features to
data/speech_commands/kws_16x10.npz, trains, and writes bundled regression cases.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from netkit import RegressionSuite, write_nk_from_arch
from netkit.datasets import SPEECH_KWS_KEYWORDS, load_speech_commands_kws
from netkit.torch_models import SpeechKwsCnn
from netkit.torch_pack import assert_packed_matches_reference, forward_speech_kws_netkit, pack_speech_kws
from netkit.torch_train import select_keyword_cases, train_cnn_classifier

MODELS = ROOT / "models"

EPOCHS = 40
BATCH_SIZE = 128
LEARNING_RATE = 0.0008
SEED = 42
NUM_CASES = 12
PER_CLASS_LIMIT = 800
IMG_H = 16
IMG_W = 10

ARCH = {
    "network": "cnn",
    "input": [IMG_H, IMG_W, 1],
    "layers": [
        {"type": "conv2d", "kernel_size": 3, "stride": 1, "filters": 8, "pad_h": 1, "pad_w": 1, "activation": "relu"},
        {"type": "depthwise_conv2d", "kernel_h": 3, "kernel_w": 3, "stride": 1, "filters": 8, "pad_h": 1, "pad_w": 1, "activation": "relu"},
        {"type": "conv2d", "kernel_size": 1, "stride": 1, "filters": 16, "activation": "relu"},
        {"type": "max_pool2d", "pool_size": 2, "stride": 2},
        {"type": "depthwise_conv2d", "kernel_h": 3, "kernel_w": 3, "stride": 1, "filters": 16, "pad_h": 1, "pad_w": 1, "activation": "relu"},
        {"type": "conv2d", "kernel_size": 1, "stride": 1, "filters": 24, "activation": "relu"},
        {"type": "max_pool2d", "pool_size": 2, "stride": 2},
        {"type": "flatten"},
        {"type": "dense", "units": 12, "activation": "none"},
    ],
}


def main() -> None:
    x_train, y_train, x_test, y_test = load_speech_commands_kws(
        per_class_limit=PER_CLASS_LIMIT,
        rebuild_cache=True,
    )

    best_acc = -1.0
    best_model: SpeechKwsCnn | None = None
    best_cases = None
    best_seed = SEED

    for attempt, seed in enumerate((SEED, 7, 13)):
        print(
            f"\nTraining Speech KWS CNN attempt {attempt + 1} on {x_train.shape[0]} feature maps "
            f"(seed={seed}, Adam lr={LEARNING_RATE}, batch={BATCH_SIZE}, epochs={EPOCHS}) ..."
        )
        model = SpeechKwsCnn(num_classes=len(SPEECH_KWS_KEYWORDS))
        train_cnn_classifier(
            model,
            x_train,
            y_train,
            forward_logits=model.forward_logits,
            img_h=IMG_H,
            img_w=IMG_W,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            seed=seed,
        )

        model.eval()
        import torch

        with torch.no_grad():
            nchw = x_test.reshape(-1, IMG_H, IMG_W, 1).transpose(0, 3, 1, 2).astype("float32")
            logits = model.forward_logits(torch.from_numpy(nchw.copy()))
            pred = logits.argmax(dim=1).cpu().numpy()
        test_acc = float((pred == y_test).mean())
        print(f"Test accuracy: {test_acc * 100:.2f}%")

        try:
            cases = select_keyword_cases(
                lambda x, m=model: forward_speech_kws_netkit(m, x, img_h=IMG_H, img_w=IMG_W),
                x_test,
                y_test,
                keyword_names=list(SPEECH_KWS_KEYWORDS),
                num_cases=NUM_CASES,
                name_fmt="KWS {keyword} (test idx {i})",
                x_train=x_train,
                y_train=y_train,
            )
        except RuntimeError as exc:
            print(exc)
            if test_acc > best_acc:
                best_acc = test_acc
                best_model = model
                best_seed = seed
            continue

        best_acc = test_acc
        best_model = model
        best_cases = cases
        best_seed = seed
        break

    if best_model is None or best_cases is None:
        raise RuntimeError("failed to train a model with 12 correctly-classified keyword regression cases")

    print(f"Selected training seed {best_seed} (test accuracy {best_acc * 100:.2f}%)")
    print("Published Speech Commands micro-model tutorials typically report ~85–95% on 12-class subsets.")

    weights = pack_speech_kws(best_model)
    assert_packed_matches_reference(
        ARCH,
        weights,
        lambda inp: forward_speech_kws_netkit(best_model, inp, img_h=IMG_H, img_w=IMG_W),
        seed=best_seed,
        atol=1e-4,
    )

    MODELS.mkdir(parents=True, exist_ok=True)
    nk_path = write_nk_from_arch(
        ARCH,
        weights,
        MODELS / "speech_kws.nk",
        RegressionSuite(tolerance=0.0001, cases=best_cases),
    )
    print(f"Wrote {nk_path} ({weights.nbytes} bytes, {len(best_cases)} embedded test cases)")


if __name__ == "__main__":
    main()
