"""CLI: python -m netkit convert|inspect ..."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .onnx_convert import convert_onnx_to_nk
from .aot_compile import compile_aot
from .inspect import inspect_nk


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="netkit model tools (.nk format)")
    sub = parser.add_subparsers(dest="command", required=True)

    convert = sub.add_parser("convert", help="Convert ONNX to .nk")
    convert.add_argument("input", help="Path to .onnx model")
    convert.add_argument("-o", "--output", help="Output .nk path")

    inspect = sub.add_parser("inspect", help="Print .nk header and tensor catalog")
    inspect.add_argument("input", help="Path to .nk model")

    aot = sub.add_parser("aot", help="AOT compile .nk to embeddable C/C++ source")
    aot.add_argument("input", help="Path to .nk model")
    aot.add_argument("-o", "--output", required=True, help="Output directory")
    aot.add_argument(
        "--language",
        choices=("cpp", "c"),
        default="cpp",
        help="Output language (default: cpp / C++26)",
    )
    aot.add_argument("--name", help="Model symbol prefix (default: input stem)")
    aot.add_argument(
        "--main",
        action="store_true",
        help="Include optional NETKIT_AOT_MAIN smoke main in the generated source",
    )

    aot.add_argument(
        "--optimize",
        action="store_true",
        help="Apply stable graph optimizations before embedding (BN fold, linear dense merge)",
    )
    aot.add_argument(
        "--arena-headroom",
        type=int,
        default=12,
        metavar="PCT",
        help="Extra arena headroom %% above measured forward peak (default: 12)",
    )
    aot.add_argument(
        "--no-flash-section",
        action="store_true",
        help="Do not place the .nk blob in a GCC .rodata section",
    )

    args = parser.parse_args(argv)
    input_path = Path(args.input)

    if args.command == "convert":
        if input_path.suffix.lower() != ".onnx":
            print(f"Unsupported input type: {input_path.suffix} (expected .onnx)", file=sys.stderr)
            return 1
        out = convert_onnx_to_nk(input_path, args.output)
        print(f"Wrote {out}")
        return 0

    if args.command == "inspect":
        inspect_nk(input_path)
        return 0

    if args.command == "aot":
        if input_path.suffix.lower() != ".nk":
            print(f"Unsupported input type: {input_path.suffix} (expected .nk)", file=sys.stderr)
            return 1
        result = compile_aot(
            input_path,
            args.output,
            language=args.language,
            model_name=args.name,
            include_main=args.main,
            optimize=args.optimize,
            arena_headroom_percent=args.arena_headroom,
            flash_section=not args.no_flash_section,
        )
        print(f"Wrote {result.header_path}")
        print(f"Wrote {result.source_path}")
        print(
            f"network={result.network} input={result.input_elements} "
            f"output={result.output_elements} bytes={result.nk_bytes} language={result.language.value}"
        )
        print(
            f"arena_after_load={result.arena_bytes_after_load} "
            f"arena_after_forward={result.arena_bytes_after_forward} "
            f"arena_recommended={result.arena_bytes_recommended}"
        )
        if result.optimizations_applied:
            print(f"optimizations={','.join(result.optimizations_applied)}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
