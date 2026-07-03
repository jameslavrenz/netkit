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

    pack = sub.add_parser("pack", help="Pack PyTorch backbone checkpoint to .nk")
    pack.add_argument("--arch", choices=("resnet18",), required=True)
    pack.add_argument("-o", "--output", required=True, help="Output .nk path")
    pack.add_argument("--height", type=int, default=56)
    pack.add_argument("--width", type=int, default=56)
    pack.add_argument("--num-classes", type=int, default=10)

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
        out = convert_onnx_to_nk(input_path, args.output, fuse_composite=not args.no_fuse)
        print(f"Wrote {out}")
        return 0

    if args.command == "pack":
        try:
            import torch
            from torchvision.models import resnet18
        except ImportError:
            print('pack requires torch/torchvision: pip install -e "python[train]" torchvision', file=sys.stderr)
            return 1
        import numpy as np
        from .arch_writer import write_nk_from_arch
        from .reference_forward import forward_cnn
        from .torch_backbone_pack import pack_resnet18_from_torch
        from .torch_pack import assert_packed_matches_reference
        from .writer import RegressionCase, RegressionSuite

        if args.arch != "resnet18":
            print(f"unsupported arch: {args.arch}", file=sys.stderr)
            return 1
        model = resnet18(weights=None)
        model.eval()
        if args.num_classes != model.fc.out_features:
            model.fc = torch.nn.Linear(model.fc.in_features, args.num_classes)
        arch, weights = pack_resnet18_from_torch(
            model,
            height=args.height,
            width=args.width,
            num_classes=args.num_classes,
        )

        def torch_forward(inp: np.ndarray) -> np.ndarray:
            x = torch.from_numpy(
                inp.reshape(1, args.height, args.width, 3).transpose(0, 3, 1, 2).copy()
            )
            with torch.no_grad():
                logits = model(x)
            return logits.cpu().numpy().reshape(-1)

        assert_packed_matches_reference(arch, weights, torch_forward, seed=42, atol=1e-4)
        rng = np.random.default_rng(0)
        inp = rng.standard_normal(args.height * args.width * 3, dtype=np.float32) * 0.1
        expected = forward_cnn(inp, arch, weights)
        out = Path(args.output)
        write_nk_from_arch(
            arch,
            weights,
            out,
            RegressionSuite(
                tolerance=1e-4,
                cases=[RegressionCase(name="ResNet-18 packed checkpoint", input=inp, expected=expected)],
            ),
        )
        print(f"Wrote {out} ({len(arch['layers'])} layers, {weights.nbytes} bytes)")
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
