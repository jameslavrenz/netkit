#!/usr/bin/env python3
"""Embed a .tflite as C arrays for esp-tflite-micro firmwares."""
from __future__ import annotations
import argparse
from pathlib import Path

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tflite", type=Path)
    ap.add_argument("--stem", required=True, help="e.g. mnist_cnn → mnist_cnn_model_data.*")
    ap.add_argument("--array", required=True, help="C symbol, e.g. g_mnist_cnn_model_data")
    ap.add_argument("-o", "--outdir", type=Path, required=True)
    args = ap.parse_args()
    data = args.tflite.read_bytes()
    args.outdir.mkdir(parents=True, exist_ok=True)
    h = args.outdir / f"{args.stem}_model_data.h"
    cc = args.outdir / f"{args.stem}_model_data.cc"
    h.write_text(
        "#pragma once\n\n#include <cstdint>\n\n"
        f"constexpr unsigned int {args.array}_size = {len(data)};\n"
        f"extern const unsigned char {args.array}[];\n"
    )
    out = [f'#include "{args.stem}_model_data.h"\n', f"alignas(16) const unsigned char {args.array}[] = {{\n"]
    for i in range(0, len(data), 12):
        chunk = data[i : i + 12]
        out.append("  " + ", ".join(f"0x{b:02x}" for b in chunk) + ",\n")
    out.append("};\n")
    cc.write_text("".join(out))
    print(f"wrote {h.name} {cc.name} ({len(data)} bytes)")

if __name__ == "__main__":
    main()
