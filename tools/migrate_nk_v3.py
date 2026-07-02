#!/usr/bin/env python3
"""Upgrade .nk v2 pool layers to v3 (add pad_h, pad_w, reserved). Idempotent on v3."""

from __future__ import annotations

import io
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from netkit.format import HEADER_BYTES, MAGIC, LayerKind  # noqa: E402

MODELS = ROOT / "models"


def _layer_body_v2(kind: int) -> int:
    if kind == LayerKind.DENSE:
        return 12
    if kind == LayerKind.CONV2D:
        return 20
    if kind in (LayerKind.MAX_POOL2D, LayerKind.AVG_POOL2D):
        return 8
    if kind == LayerKind.BATCH_NORM2D:
        return 8
    if kind == LayerKind.FLATTEN:
        return 0
    raise ValueError(f"unsupported layer kind: {kind}")


def migrate_bytes(data: bytes) -> bytes | None:
    if len(data) < HEADER_BYTES:
        raise ValueError("truncated .nk header")

    fields = struct.unpack("<4sIBBH4IIIIII", data[:HEADER_BYTES])
    magic, version = fields[0], fields[1]
    if magic != MAGIC:
        raise ValueError("invalid .nk magic")
    if version == 3:
        return None
    if version != 2:
        raise ValueError(f"unsupported .nk version: {version}")

    num_layers = fields[9]
    stream = io.BytesIO(data)
    stream.seek(HEADER_BYTES)

    layers = bytearray()
    for _ in range(num_layers):
        kind = struct.unpack("<B", stream.read(1))[0]
        stream.read(3)
        body = stream.read(_layer_body_v2(kind))
        layers += struct.pack("<BBBB", kind, 0, 0, 0) + body
        if kind in (LayerKind.MAX_POOL2D, LayerKind.AVG_POOL2D):
            layers += struct.pack("<BBH", 0, 0, 0)

    tail = stream.read()
    new_header = struct.pack(
        "<4sIBBH4IIIIII",
        MAGIC,
        3,
        fields[2],
        fields[3],
        fields[4],
        *fields[5:9],
        fields[9],
        fields[10],
        fields[11],
        fields[12],
        fields[13],
    )
    return new_header + layers + tail


def migrate_file(path: Path) -> bool:
    original = path.read_bytes()
    updated = migrate_bytes(original)
    if updated is None:
        print(f"skip {path} (already v3)")
        return False
    path.write_bytes(updated)
    print(f"migrated {path} v2 -> v3")
    return True


def main() -> None:
    paths = sorted(MODELS.glob("*.nk"))
    if not paths:
        raise SystemExit("no .nk files found")

    migrated = 0
    for path in paths:
        if migrate_file(path):
            migrated += 1
    print(f"done: {migrated} file(s) migrated")


if __name__ == "__main__":
    main()
