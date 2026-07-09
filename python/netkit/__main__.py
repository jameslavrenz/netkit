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
    convert.add_argument(
        "--no-fuse",
        action="store_true",
        help="Disable composite block fusion (ResNet BasicBlock from Add nodes)",
    )
    convert.add_argument(
        "--no-optimize",
        action="store_true",
        help="Disable packager graph optimizations (BN fold, conv+BN merge, linear dense merge)",
    )
    convert.add_argument(
        "--verbose-fuse",
        action="store_true",
        help="Log ONNX/packager composite fusion attempts that do not match",
    )

    pack = sub.add_parser("pack", help="Pack PyTorch backbone checkpoint to .nk")
    pack.add_argument(
        "--arch",
        choices=("resnet18", "convnextv2_atto", "mobilenetv4_small"),
        required=True,
    )
    pack.add_argument("-o", "--output", required=True, help="Output .nk path")
    pack.add_argument("--height", type=int, default=None, help="Input height (arch default if omitted)")
    pack.add_argument("--width", type=int, default=None, help="Input width (arch default if omitted)")
    pack.add_argument("--num-classes", type=int, default=10)
    pack.add_argument(
        "--pretrained",
        action="store_true",
        help="Load ImageNet-pretrained timm weights (default: random init)",
    )

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
        "--no-lower",
        action="store_true",
        help="Embed .nk blob and use runtime loader (C++ only; default is static kernel lowering)",
    )
    aot.add_argument(
        "--no-flash-section",
        action="store_true",
        help="Do not place embedded coefs in a GCC .rodata section",
    )
    aot.add_argument(
        "--target",
        choices=("cpu", "mpu", "mcu"),
        default="cpu",
        help="Firmware target for arena sizing (mcu subtracts flash payload from probe peaks)",
    )
    aot.add_argument(
        "--omit-final-softmax",
        action="store_true",
        help=(
            "Skip final Dense Softmax and emit logits instead "
            "(argmax-equivalent; for classification benches)"
        ),
    )

    args = parser.parse_args(argv)

    if args.command == "convert":
        input_path = Path(args.input)
        if input_path.suffix.lower() != ".onnx":
            print(f"Unsupported input type: {input_path.suffix} (expected .onnx)", file=sys.stderr)
            return 1
        out = convert_onnx_to_nk(
            input_path,
            args.output,
            fuse_composite=not args.no_fuse,
            optimize=not args.no_optimize,
            verbose_fuse=args.verbose_fuse,
        )
        print(f"Wrote {out}")
        return 0

    if args.command == "pack":
        try:
            import torch  # noqa: F401
        except ImportError:
            print('pack requires torch: pip install -e "python[train]"', file=sys.stderr)
            return 1
        import numpy as np
        from .arch_writer import write_nk_from_arch
        from .reference_forward import forward_cnn
        from .torch_backbone_pack import (
            PACK_ARCH_DEFAULTS,
            backbone_torch_forward,
            load_backbone_model,
            pack_backbone_from_torch,
        )
        from .torch_pack import assert_packed_matches_reference
        from .writer import RegressionCase, RegressionSuite

        default_h, default_w = PACK_ARCH_DEFAULTS[args.arch]
        height = args.height if args.height is not None else default_h
        width = args.width if args.width is not None else default_w

        model = load_backbone_model(
            args.arch, num_classes=args.num_classes, pretrained=args.pretrained
        )
        arch, weights = pack_backbone_from_torch(
            args.arch,
            model,
            height=height,
            width=width,
            num_classes=args.num_classes,
        )

        def torch_forward(inp: np.ndarray) -> np.ndarray:
            return backbone_torch_forward(model, inp, height=height, width=width)

        parity_samples = 4 if args.arch == "mobilenetv4_small" else 8
        assert_packed_matches_reference(
            arch, weights, torch_forward, seed=42, atol=1e-4, samples=parity_samples
        )
        rng = np.random.default_rng(0)
        inp = rng.standard_normal(height * width * 3, dtype=np.float32) * 0.1
        expected = forward_cnn(inp, arch, weights)
        label = args.arch.replace("_", " ").title()
        out = Path(args.output)
        write_nk_from_arch(
            arch,
            weights,
            out,
            RegressionSuite(
                tolerance=1e-4,
                cases=[RegressionCase(name=f"{label} packed checkpoint", input=inp, expected=expected)],
            ),
        )
        print(
            f"Wrote {out} (timm, {len(arch['layers'])} layers, {weights.nbytes} bytes, "
            f"{height}x{width}x3)"
        )
        return 0

    if args.command == "inspect":
        inspect_nk(Path(args.input))
        return 0

    if args.command == "aot":
        input_path = Path(args.input)
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
            lower=not args.no_lower,
            omit_final_softmax=args.omit_final_softmax,
        )
        print(f"Wrote {result.header_path}")
        print(f"Wrote {result.source_path}")
        print(
            f"network={result.network} input={result.input_elements} "
            f"output={result.output_elements} bytes={result.nk_bytes} language={result.language.value}"
        )
        if result.quant_fast:
            print("quant_fast=true (quant lowered static CmsisQuantPlan call chain)")
        elif result.lowered:
            print("lowered=true (static Kernels:: call chain)")
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
