"""MCU flash-backed weights: buffer load keeps payload bytes out of the arena."""

from __future__ import annotations

import io
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from netkit.arch_writer import arch_to_nk_bytes
from netkit.format import HEADER_BYTES, skip_payload_alignment_padding, unpack_header, weight_payload_bytes
from netkit.inspect import _read_layer_body, _read_tensor_desc
from netkit.reader import read_nk, read_test_suite

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
LIB = ROOT / "libnetkit.a"


def _require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise unittest.SkipTest(f"{name} not found")


def _require_lib() -> None:
    if not LIB.is_file():
        raise unittest.SkipTest("libnetkit.a missing — run `make lib` first")


def _weight_payload_offset(nk_bytes: bytes) -> int:
    stream = io.BytesIO(nk_bytes)
    header = unpack_header(stream.read(HEADER_BYTES))
    for _ in range(header["num_layers"]):
        kind = struct.unpack("<B", stream.read(1))[0]
        stream.read(3)
        _read_layer_body(stream, kind)
    for _ in range(header["num_weight_tensors"]):
        _read_tensor_desc(stream)
    for _ in range(header["num_bias_tensors"]):
        _read_tensor_desc(stream)
    meta_end = stream.tell()
    return skip_payload_alignment_padding(stream, meta_end)


def _format_byte_array(data: bytes, *, width: int = 12) -> str:
    lines: list[str] = []
    row: list[str] = []
    for byte in data:
        row.append(f"0x{byte:02x}")
        if len(row) == width:
            lines.append("    " + ", ".join(row) + ",")
            row = []
    if row:
        lines.append("    " + ", ".join(row) + ",")
    return "\n".join(lines)


def _rebuild_lib(*make_args: str) -> None:
    subprocess.run(
        ["make", "clean"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["make", *make_args, "lib"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _write_probe_harness(tmp: Path, nk_bytes: bytes) -> Path:
    harness = tmp / "mcu_flash_harness.c"
    harness.write_text(
        f"""#include "netkit.h"
#include <stdio.h>
#include <stddef.h>
#include <stdalign.h>

alignas(8) static const unsigned char nk_blob[] = {{
{_format_byte_array(nk_bytes)}
}};
static const size_t nk_size = sizeof(nk_blob);

int main(void)
{{
    alignas(max_align_t) static unsigned char arena_mem[65536];
    nk_arena_t arena;
    nk_arena_init(&arena, arena_mem, sizeof(arena_mem));

    nk_model_t model;
    if (nk_model_load_memory(nk_blob, nk_size, &arena, &model) != NK_OK)
        return 1;

    printf("%zu\\n", nk_arena_used(&arena));
    return 0;
}}
""",
        encoding="utf-8",
    )
    return harness


def _probe_arena_used(tmp: Path, harness: Path) -> int:
    binary = tmp / "mcu_flash_probe"
    subprocess.run(
        [
            "clang++",
            "-std=c++26",
            "-Wall",
            "-Wextra",
            "-DNETKIT_TARGET_MCU=1",
            f"-I{ROOT / 'include'}",
            str(harness),
            str(LIB),
            "-o",
            str(binary),
        ],
        cwd=tmp,
        check=True,
        capture_output=True,
        text=True,
    )
    proc = subprocess.run([str(binary)], cwd=tmp, check=True, capture_output=True, text=True)
    return int(proc.stdout.strip())


class TestMcuFlashWeights(unittest.TestCase):
    def setUp(self) -> None:
        _require_tool("clang++")
        _require_lib()

    def test_new_exports_align_weight_payload(self) -> None:
        arch, weights = read_nk(MODELS / "mlp_hand.nk")
        nk_bytes = arch_to_nk_bytes(arch, weights)
        payload_offset = _weight_payload_offset(nk_bytes)
        self.assertEqual(payload_offset % 4, 0)

    def test_buffer_load_arena_excludes_payload(self) -> None:
        nk_path = MODELS / "mlp_hand.nk"
        arch, weights = read_nk(nk_path)
        tests = read_test_suite(nk_path)
        nk_bytes = arch_to_nk_bytes(arch, weights, tests=tests)
        header = unpack_header(nk_bytes[:HEADER_BYTES])
        payload_bytes = weight_payload_bytes(header)
        self.assertGreater(payload_bytes, 0)
        self.assertEqual(_weight_payload_offset(nk_bytes) % 4, 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            harness = _write_probe_harness(tmp, nk_bytes)

            _rebuild_lib("NETKIT_TARGET=mcu", "NETKIT_WEIGHTS_IN_RAM=0")
            try:
                used_flash = _probe_arena_used(tmp, harness)
                _rebuild_lib("NETKIT_TARGET=mcu", "NETKIT_WEIGHTS_IN_RAM=1")
                used_ram = _probe_arena_used(tmp, harness)
            finally:
                _rebuild_lib()

        self.assertGreater(used_ram, used_flash)
        self.assertGreaterEqual(used_ram - used_flash, payload_bytes)


if __name__ == "__main__":
    unittest.main()
