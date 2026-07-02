"""netkit Python tools — convert ONNX models to .nk and inspect them."""

from .aot_compile import AotCompileResult, AotLanguage, compile_aot
from .arch_writer import write_nk_from_arch
from .onnx_convert import convert_onnx_to_nk
from .inspect import inspect_nk
from .reader import read_nk, read_test_suite
from .writer import RegressionCase, RegressionSuite, write_nk

__all__ = [
    "AotCompileResult",
    "AotLanguage",
    "compile_aot",
    "convert_onnx_to_nk",
    "inspect_nk",
    "read_nk",
    "read_test_suite",
    "write_nk",
    "write_nk_from_arch",
    "RegressionCase",
    "RegressionSuite",
]
