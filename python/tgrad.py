"""Tgrad Python authoring layer — L6.b real FFI via ctypes ↔ @[export].

Loads `libtgrad.dylib` (built by `c/Makefile`'s `dylib` target),
calls into Lean's @[export] entries through C trampolines. No
subprocess; sub-millisecond per-call overhead so L7's perf parity is
meaningful.

Usage:
    python -m tgrad bench --shape 64x64x64

    >>> import tgrad
    >>> a = tgrad.Tensor.from_numpy(np_array_a)
    >>> b = tgrad.Tensor.from_numpy(np_array_b)
    >>> c = a @ b
    >>> c.numpy()  # → np.ndarray
"""
from __future__ import annotations
import argparse
import ctypes
import json
import os
import statistics
import sys
import time
import weakref
from pathlib import Path

import numpy as np

REPO_ROOT = Path(os.environ.get("TGRAD_ROOT", Path(__file__).resolve().parents[1])).expanduser().resolve(strict=False)
DEFAULT_LIB = REPO_ROOT / ".lake" / "build" / "lib" / "libtgrad.dylib"
LIB_PATH = Path(os.environ.get("TGRAD_LIB", str(DEFAULT_LIB))).expanduser().resolve(strict=False)
PERF_PROFILE = os.environ.get("TGRAD_PERF_PROFILE", "apple_m4_mini_release")

if not LIB_PATH.exists():
    raise RuntimeError(
        f"libtgrad.dylib not found at {LIB_PATH}. Build it via "
        f"`lake build Tgrad:shared tgrad-cli tgrad-tests && make -C c dylib` "
        f"from the Tgrad repo root, or set TGRAD_LIB."
    )

# RTLD_GLOBAL: libtgrad.dylib's force_loaded Lean code has flat-namespace
# references to the C bridge symbols (lean_theograd_metal_alloc, etc.).
# Without RTLD_GLOBAL, dyld's per-handle private namespace prevents the
# Lean code from seeing libtgrad.dylib's own bridge symbols at load.
_lib = ctypes.CDLL(str(LIB_PATH), mode=ctypes.RTLD_GLOBAL)

_lib.tgrad_init.argtypes = []
_lib.tgrad_init.restype  = ctypes.c_int
_lib.tgrad_handle_inc.argtypes = [ctypes.c_void_p]
_lib.tgrad_handle_inc.restype  = None
_lib.tgrad_handle_dec.argtypes = [ctypes.c_void_p]
_lib.tgrad_handle_dec.restype  = None

_lib.tgrad_tensor_alloc.argtypes = [ctypes.c_size_t]
_lib.tgrad_tensor_alloc.restype  = ctypes.c_uint64
_lib.tgrad_tensor_free.argtypes  = [ctypes.c_uint64, ctypes.c_size_t]
_lib.tgrad_tensor_free.restype   = None
_lib.tgrad_tensor_write_bytes.argtypes = [
    ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
_lib.tgrad_tensor_write_bytes.restype  = ctypes.c_int
_lib.tgrad_tensor_read_bytes.argtypes  = [
    ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
_lib.tgrad_tensor_read_bytes.restype   = ctypes.c_int
_lib.tgrad_dtype_query.argtypes = [ctypes.c_uint8, ctypes.c_uint8]
_lib.tgrad_dtype_query.restype = ctypes.c_uint64
_lib.tgrad_dtype_binary_query.argtypes = [
    ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8]
_lib.tgrad_dtype_binary_query.restype = ctypes.c_uint8
_lib.tgrad_dtype_unary_query.argtypes = [ctypes.c_uint8, ctypes.c_uint8]
_lib.tgrad_dtype_unary_query.restype = ctypes.c_uint8
_lib.tgrad_dtype_lub_many.argtypes = [
    ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
_lib.tgrad_dtype_lub_many.restype = ctypes.c_uint8
_lib.tgrad_dtype_infer_python.argtypes = [
    ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
_lib.tgrad_dtype_infer_python.restype = ctypes.c_uint8
_lib.tgrad_dtype_default.argtypes = [ctypes.c_uint8]
_lib.tgrad_dtype_default.restype = ctypes.c_uint8
_lib.tgrad_dtype_set_default.argtypes = [ctypes.c_uint8, ctypes.c_uint8]
_lib.tgrad_dtype_set_default.restype = ctypes.c_uint8
_lib.tgrad_dtype_creation_default.argtypes = []
_lib.tgrad_dtype_creation_default.restype = ctypes.c_uint8
_lib.tgrad_dtype_backend_name.argtypes = [
    ctypes.c_uint8, ctypes.POINTER(ctypes.c_char), ctypes.c_size_t]
_lib.tgrad_dtype_backend_name.restype = ctypes.c_size_t
_lib.tgrad_dtype_public_name.argtypes = [
    ctypes.c_uint8, ctypes.POINTER(ctypes.c_char), ctypes.c_size_t]
_lib.tgrad_dtype_public_name.restype = ctypes.c_size_t
_lib.tgrad_dtype_display_name.argtypes = [
    ctypes.c_uint8, ctypes.POINTER(ctypes.c_char), ctypes.c_size_t]
_lib.tgrad_dtype_display_name.restype = ctypes.c_size_t
_lib.tgrad_dtype_table_query.argtypes = [
    ctypes.c_uint8, ctypes.c_uint8, ctypes.c_size_t, ctypes.c_size_t]
_lib.tgrad_dtype_table_query.restype = ctypes.c_uint64
_lib.tgrad_dtype_table_name.argtypes = [
    ctypes.c_uint8, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_char), ctypes.c_size_t]
_lib.tgrad_dtype_table_name.restype = ctypes.c_size_t
_lib.tgrad_bf16_pack_bytes.argtypes = [
    ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
_lib.tgrad_bf16_pack_bytes.restype = ctypes.c_int
_lib.tgrad_bf16_expand_bytes.argtypes = [
    ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
_lib.tgrad_bf16_expand_bytes.restype = ctypes.c_int
_lib.tgrad_bf16_round_bits.argtypes = [ctypes.c_uint32]
_lib.tgrad_bf16_round_bits.restype = ctypes.c_uint32
_lib.tgrad_matmul_64x64.argtypes = [
    ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
_lib.tgrad_matmul_64x64.restype  = ctypes.c_int32

# L11: general sentinel matmul. ShapeSentinel.ofTriple bounds the accepted
# set; Lean then generates the parametric TC declaration and launch geometry.
_lib.tgrad_matmul.argtypes = [
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
]
_lib.tgrad_matmul.restype  = ctypes.c_int32

# L12: alternate generated-emitter cache. Same generated declaration as
# tgrad_matmul; the distinct symbol preserves alternate-route observability.
_lib.tgrad_matmul_alg.argtypes = [
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
]
_lib.tgrad_matmul_alg.restype  = ctypes.c_int32

# L13.B: scalar-matmul entry for below-TC-tile shapes (any dim < 8).
# Kernel rendered from `scalarMatmulKernelDecl`. Distinct symbol so
# the L13.B gate can observably route through this path.
_lib.tgrad_matmul_small.argtypes = [
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
]
_lib.tgrad_matmul_small.restype  = ctypes.c_int32

# L13.F: TC-general matmul for non-sentinel TC-eligible shapes.
# Kernel rendered from `tcMatmulKernelDecl` (pure on (M, K, N)).
# Uses simdgroup_load/multiply_accumulate/store — generated source
# always contains `simdgroup_multiply_accumulate` (L13.F gate check).
_lib.tgrad_matmul_tc.argtypes = [
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
]
_lib.tgrad_matmul_tc.restype  = ctypes.c_int32

# L13.F.STRICT.B: manually-loaded TC-general matmul. Bench-only until
# L13.F.STRICT.C flips the production TC route.
_lib.tgrad_matmul_tc_manual_load.argtypes = [
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
]
_lib.tgrad_matmul_tc_manual_load.restype  = ctypes.c_int32

# L13.F: TC eligibility query. Pure call into Lean — returns 1 if
# `tcMatmulKernelDecl(M, K, N)` would produce a valid plan, else 0.
# Python uses this BEFORE dispatch to route non-sentinel TC. The
# routing decision is Lean's, not Python's (L13.F gate D3).
_lib.tgrad_matmul_tc_eligible.argtypes = [
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
]
_lib.tgrad_matmul_tc_eligible.restype  = ctypes.c_int32

# L14.A: opaque-handle tensor registry. `tgrad_tensor_from_buffer`
# constructs a Lean-side Tensor (UOp graph) and returns an opaque
# uint64 handle. Query entries (rank/shape_dim/raw_buffer) round-trip
# for L14.B's view methods.
_lib.tgrad_tensor_from_buffer.argtypes = [
    ctypes.c_uint64, ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t, ctypes.c_uint8,
]
_lib.tgrad_tensor_from_buffer.restype = ctypes.c_uint64
_lib.tgrad_tensor_rank.argtypes       = [ctypes.c_uint64]
_lib.tgrad_tensor_rank.restype        = ctypes.c_size_t
_lib.tgrad_tensor_shape_dim.argtypes  = [ctypes.c_uint64, ctypes.c_size_t]
_lib.tgrad_tensor_shape_dim.restype   = ctypes.c_size_t
_lib.tgrad_tensor_raw_buffer.argtypes = [ctypes.c_uint64]
_lib.tgrad_tensor_raw_buffer.restype  = ctypes.c_uint64

# L14.B.1: view methods. Each composes a movement node on the
# underlying Tensor's uop and returns a new opaque handle. Pure
# graph transforms; no buffer allocation; no kernel dispatch.
_lib.tgrad_tensor_transpose.argtypes = [ctypes.c_uint64]
_lib.tgrad_tensor_transpose.restype  = ctypes.c_uint64
_lib.tgrad_tensor_permute.argtypes   = [
    ctypes.c_uint64, ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t]
_lib.tgrad_tensor_permute.restype    = ctypes.c_uint64
_lib.tgrad_tensor_reshape.argtypes   = [
    ctypes.c_uint64, ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t]
_lib.tgrad_tensor_reshape.restype    = ctypes.c_uint64
_lib.tgrad_tensor_expand.argtypes    = [
    ctypes.c_uint64, ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t]
_lib.tgrad_tensor_expand.restype     = ctypes.c_uint64
_lib.tgrad_tensor_slice.argtypes     = [
    ctypes.c_uint64, ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t]
_lib.tgrad_tensor_slice.restype      = ctypes.c_uint64

# L14.B.1: query the UOp kind of a tensor handle's root. Returns
# 0=BUFFER, 1=PERMUTE, 2=RESHAPE, 3=EXPAND, 4=SLICE, 255=other.
# L14.B.2.c uses this to route view inputs to tgrad_matmul_view
# (replacing L14.B.1's MatmulOnNonBufferUop typed-error guard).
_lib.tgrad_tensor_uop_kind.argtypes = [ctypes.c_uint64]
_lib.tgrad_tensor_uop_kind.restype  = ctypes.c_uint8

# L14.B.2.c: view-aware matmul. When either input has a non-BUFFER
# uop, Python routes here; the Lean side runs the parametric scalar
# matmul (`Pipeline.realizeView`) with A/B load-index UOps derived
# from the input uop chains.
# Graph-indexed realize. Tensor methods compose a UOp graph and one
# entry lowers it, so a new op costs a node constructor and a table row
# rather than another export/trampoline/binding triple.
_lib.tgrad_tensor_binop.argtypes = [ctypes.c_uint8, ctypes.c_uint64, ctypes.c_uint64]
_lib.tgrad_tensor_binop.restype  = ctypes.c_uint64
_lib.tgrad_tensor_reduce.argtypes = [ctypes.c_uint8, ctypes.c_uint64, ctypes.c_size_t]
_lib.tgrad_tensor_reduce.restype  = ctypes.c_uint64
_lib.tgrad_realize.argtypes = [ctypes.c_uint64]
_lib.tgrad_realize.restype  = ctypes.c_uint64
_lib.tgrad_tensor_dtype.argtypes = [ctypes.c_uint64]
_lib.tgrad_tensor_dtype.restype  = ctypes.c_uint8

# Constant-fill creation. Lean resolves the current runtime floating default,
# applies dtype admission, and owns the GPU fill; Python only marshals shape /
# fill / dtype code.
_lib.tgrad_tensor_full.argtypes = [
    ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t,
    ctypes.c_double, ctypes.c_uint8]
_lib.tgrad_tensor_full.restype  = ctypes.c_uint64

_lib.tgrad_matmul_view.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
_lib.tgrad_matmul_view.restype  = ctypes.c_uint64

# Stable FFI dtype encoding (matches PythonFFI.lean's `dtypeOfCode`):
_DTYPE_BF16 = 0
_DTYPE_F32  = 1
_DTYPE_F16  = 2
_DTYPE_I32  = 3
_DTYPE_BOOL = 4
_DTYPE_WEAKINT = 5
_DTYPE_I8 = 6
_DTYPE_U8 = 7
_DTYPE_I16 = 8
_DTYPE_U16 = 9
_DTYPE_U32 = 10
_DTYPE_I64 = 11
_DTYPE_U64 = 12
_DTYPE_WEAKFLOAT = 13
_DTYPE_FP8E4M3 = 14
_DTYPE_FP8E5M2 = 15
_DTYPE_FP8E4M3FNUZ = 16
_DTYPE_FP8E5M2FNUZ = 17
_DTYPE_F64 = 18
_DTYPE_VOID = 254
# Creation-surface sentinels (matches PythonFFI.creationDtype?):
_CREATION_DTYPE_DEFAULT = 255  # dtype=None → Lean resolves runtime default
_CREATION_DTYPE_UNKNOWN = 253  # unrecognised name → Lean rejects
_DTYPE_CODES = {
    "bf16": _DTYPE_BF16, "bfloat16": _DTYPE_BF16,
    "f32": _DTYPE_F32, "float32": _DTYPE_F32,
    "f16": _DTYPE_F16, "float16": _DTYPE_F16,
    "i32": _DTYPE_I32, "int32": _DTYPE_I32,
    "bool": _DTYPE_BOOL, "weakint": _DTYPE_WEAKINT,
    "int8": _DTYPE_I8, "uint8": _DTYPE_U8,
    "int16": _DTYPE_I16, "uint16": _DTYPE_U16,
    "uint32": _DTYPE_U32, "int64": _DTYPE_I64, "uint64": _DTYPE_U64,
    "weakfloat": _DTYPE_WEAKFLOAT,
    "fp8e4m3": _DTYPE_FP8E4M3, "fp8e5m2": _DTYPE_FP8E5M2,
    "fp8e4m3fnuz": _DTYPE_FP8E4M3FNUZ,
    "fp8e5m2fnuz": _DTYPE_FP8E5M2FNUZ,
    "float64": _DTYPE_F64, "void": _DTYPE_VOID,
}
def _dtype_code(name: str) -> int:
    return _DTYPE_CODES.get(name, 255)

def _marshaled_dtype_code(dtype) -> int | None:
    """Extract a public dtype object's stable code without interpreting it.

    The strict tinygrad shim passes Lean-backed singleton objects here. This is
    boundary plumbing: admission and meaning remain in Lean.
    """
    code = getattr(dtype, "code", None)
    return int(code) if isinstance(code, int) else None


def _native_dtype_name(dtype) -> str:
    if isinstance(dtype, str):
        return dtype
    code = _marshaled_dtype_code(dtype)
    if code is None or code not in _DTYPE_OF_CODE:
        raise TgradTypeError(f"unsupported dtype object {dtype!r}")
    return _DTYPE_OF_CODE[code]


def _creation_dtype_code(dtype) -> int:
    """Marshal a creation dtype kwarg for Lean. Does not validate —
    Lean owns runtime-default resolution and compute admission."""
    if dtype is None:
        return _CREATION_DTYPE_DEFAULT
    code = _marshaled_dtype_code(dtype)
    if code is not None:
        return code
    return _DTYPE_CODES.get(dtype, _CREATION_DTYPE_UNKNOWN)

# Legacy L12 cache toggle. Both entries execute the generated declaration;
# the alternate symbol remains as an observable cache-isolation probe.
_USE_ALGEBRAIC: bool = False
_USE_MANUAL_LOAD_TC: bool = False

def set_use_algebraic(flag: bool) -> None:
    """Switch between the primary and alternate generated-sentinel caches."""
    global _USE_ALGEBRAIC
    _USE_ALGEBRAIC = bool(flag)

def set_use_manual_load_tc(flag: bool) -> None:
    """Switch TC-general routing between the legacy simdgroup-load
    kernel and the L13.F.STRICT.B manual-load kernel."""
    global _USE_MANUAL_LOAD_TC
    _USE_MANUAL_LOAD_TC = bool(flag)

_rc = _lib.tgrad_init()
if _rc != 0:
    raise RuntimeError(f"tgrad_init failed (rc={_rc})")


class TgradError(RuntimeError):
    """Base for all Tgrad FFI errors."""


class NotInLeanScope(TgradError):
    """Raised when the requested shape isn't supported by the L6.b scope."""


class TgradTypeError(TgradError):
    """Raised for dtype / shape contract violations."""


class MatmulOnNonBufferUop(TgradError):
    """Raised by `Tensor.__matmul__` when either input's Lean-side
    `uop` is not a BUFFER leaf — i.e. the user composed a view chain
    (`a.transpose() @ b`) and tried to matmul before L14.B.2 wires
    `Schedule.Rangeify` into the realize path. The view methods are
    plumbed; the kernel-side view-aware codegen lands at L14.B.2."""


# Op codes shared with PythonFFI.binOpOfCode.
_BINOP_ADD = 0
_BINOP_MUL = 1
_BINOP_SUB = 2

_REDUCE_DTYPES = {"bf16", "f32"}
# Bytes per element, used to check that a materialized buffer is
# exactly as large as its declared shape requires.
_DTYPE_BYTES = {"bf16": 2, "f32": 4, "i32": 4}


_DTYPE_OF_CODE = {0: "bf16", 1: "f32", 2: "f16", 3: "i32"}


def _dtype_query(code: int, query: int) -> int:
    return int(_lib.tgrad_dtype_query(code, query))


def _dtype_binary_query(query: int, left: int, right: int) -> int:
    value = int(_lib.tgrad_dtype_binary_query(query, left, right))
    if value == 255:
        raise TgradTypeError(
            f"invalid Lean binary dtype query={query} left={left} right={right}")
    return value


def _dtype_unary_query(query: int, code: int) -> int:
    value = int(_lib.tgrad_dtype_unary_query(query, code))
    if value == 255:
        raise TgradTypeError(f"invalid Lean unary dtype query={query} code={code}")
    return value


def _dtype_lub_many(codes) -> int:
    values = tuple(int(code) for code in codes)
    if not values:
        raise TgradTypeError("least-upper dtype requires at least one dtype")
    arr = (ctypes.c_uint8 * len(values))(*values)
    result = int(_lib.tgrad_dtype_lub_many(arr, len(values)))
    if result == 255:
        raise TgradTypeError(f"Lean rejected dtype-code sequence {values!r}")
    return result


def _dtype_infer_python(tags) -> int:
    values = tuple(int(tag) for tag in tags)
    arr = (ctypes.c_uint8 * len(values))(*values)
    result = int(_lib.tgrad_dtype_infer_python(arr, len(values)))
    if result == 255:
        raise TgradTypeError(f"Lean rejected Python dtype tags {values!r}")
    return result


def _dtype_default(which: int) -> int:
    value = int(_lib.tgrad_dtype_default(which))
    if value == 255:
        raise TgradTypeError(f"invalid Lean default dtype selector={which}")
    return value


def _dtype_set_default(which: int, code: int) -> bool:
    return bool(_lib.tgrad_dtype_set_default(which, code))


def _dtype_creation_default() -> int:
    """Current default after Lean creation admission; 255 means rejected."""
    return int(_lib.tgrad_dtype_creation_default())


def _dtype_backend_name(code: int) -> str:
    needed = int(_lib.tgrad_dtype_backend_name(code, None, 0))
    if needed == 0:
        raise TgradTypeError(f"invalid Lean dtype name code={code}")
    out = ctypes.create_string_buffer(needed + 1)
    actual = int(_lib.tgrad_dtype_backend_name(code, out, len(out)))
    if actual != needed:
        raise TgradError(f"dtype name length changed: {needed} -> {actual}")
    return out.value.decode("utf-8")


def _dtype_public_name(code: int) -> str:
    needed = int(_lib.tgrad_dtype_public_name(code, None, 0))
    if needed == 0:
        raise TgradTypeError(f"invalid Lean public dtype name code={code}")
    out = ctypes.create_string_buffer(needed + 1)
    actual = int(_lib.tgrad_dtype_public_name(code, out, len(out)))
    if actual != needed:
        raise TgradError(f"public dtype name length changed: {needed} -> {actual}")
    return out.value.decode("utf-8")


def _dtype_display_name(code: int) -> str:
    needed = int(_lib.tgrad_dtype_display_name(code, None, 0))
    if needed == 0:
        raise TgradTypeError(f"invalid Lean display dtype name code={code}")
    out = ctypes.create_string_buffer(needed + 1)
    actual = int(_lib.tgrad_dtype_display_name(code, out, len(out)))
    if actual != needed:
        raise TgradError(f"display dtype name length changed: {needed} -> {actual}")
    return out.value.decode("utf-8")


def _dtype_table_query(table: int, query: int, row: int = 0, column: int = 0) -> int:
    return int(_lib.tgrad_dtype_table_query(table, query, row, column))


def _dtype_table_name(table: int, row: int) -> str:
    needed = int(_lib.tgrad_dtype_table_name(table, row, None, 0))
    if needed == 0:
        raise TgradTypeError(f"invalid Lean dtype table name table={table} row={row}")
    out = ctypes.create_string_buffer(needed + 1)
    actual = int(_lib.tgrad_dtype_table_name(table, row, out, len(out)))
    if actual != needed:
        raise TgradError(f"dtype table name length changed: {needed} -> {actual}")
    return out.value.decode("utf-8")


# Compute admission is a Lean relation. This is only its authoring-name
# projection; a newly admitted Lean code without a boundary spelling fails
# closed instead of silently disappearing from Python.
_SUPPORTED_DTYPE_CODES = frozenset(
    code for code in range(255)
    if _dtype_query(code, 0) == 1 and _dtype_query(code, 15) == 1)
_UNNAMED_SUPPORTED_DTYPE_CODES = _SUPPORTED_DTYPE_CODES.difference(_DTYPE_OF_CODE)
if _UNNAMED_SUPPORTED_DTYPE_CODES:
    raise RuntimeError(
        "Lean compute-supported dtype codes lack Python authoring spellings: "
        f"{sorted(_UNNAMED_SUPPORTED_DTYPE_CODES)}")
_SUPPORTED_DTYPES = frozenset(
    _DTYPE_OF_CODE[code] for code in _SUPPORTED_DTYPE_CODES)


def _bf16_round_bits(bits: int) -> int:
    """Lean-owned scalar float_to_bf16 result, represented as fp32 bits."""
    return int(_lib.tgrad_bf16_round_bits(int(bits)))


def _dtype_of_handle(h: int) -> str:
    """Result dtype as Lean computed it. Python never recomputes the
    promotion lattice; `Dtype.lub` stays single-sourced."""
    code = _lib.tgrad_tensor_dtype(h)
    name = _DTYPE_OF_CODE.get(code)
    if name is None:
        raise TgradError(f"tgrad_tensor_dtype returned unknown code {code}")
    return name


def _bytes_from_numpy(arr: np.ndarray, dtype: str) -> bytes:
    if dtype == "bf16":
        return _bf16_from_fp32(arr)
    if dtype == "f32":
        return arr.astype(np.float32).tobytes()
    if dtype == "i32":
        return arr.astype(np.int32).tobytes()
    raise TgradTypeError(f"unsupported dtype {dtype!r}")


def _numpy_from_bytes(b: bytes, shape: tuple[int, ...], dtype: str) -> np.ndarray:
    if dtype == "bf16":
        return _fp32_from_bf16(b, shape)
    if dtype == "f32":
        return np.frombuffer(b, dtype=np.float32).reshape(shape).copy()
    if dtype == "i32":
        return np.frombuffer(b, dtype=np.int32).reshape(shape).copy()
    raise TgradTypeError(f"unsupported dtype {dtype!r}")


def _numel(shape: tuple[int, ...]) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n


def _argfix(*x):
    """Match tinygrad.helpers.argfix: a lone tuple/list is the shape;
    otherwise the varargs are the shape. ``ones((1,3))`` and ``ones(1,3)``
    are both valid; ``ones((1,3), 2)`` raises ValueError."""
    if x and x[0].__class__ in (tuple, list):
        if len(x) != 1:
            raise ValueError(f"bad arg {x}")
        return tuple(x[0])
    return x


# L13.C+ scope: arbitrary 2D operand shapes are accepted. The
# concrete shape support per matmul `(M, K) @ (K, N)` is decided at
# __matmul__ time:
#   - (M, K, N) in _TRIPLE_SET → generated TC sentinel path (tgrad_matmul)
#   - otherwise, Lean wide eligibility → generated TC-general path
#   - otherwise → scalar correctness fallback (tgrad_matmul_small)
# `_SUPPORTED_SHAPES` is kept as a documentation set listing the
# operand shapes the manifest fixtures exercise; from_numpy /
# from_bf16_bytes now accept any 2D ndarray with dims ≥ 1.
_SUPPORTED_SHAPES = {
    # L11 sentinel operand shapes
    (64, 64),  (1024, 1024),  (2048, 2048),  (4096, 4096),  (8192, 8192),
    (8192, 1024),  (4096, 1024),  (2048, 1024),
    (1024, 8192), (1024, 4096),  (1024, 2048),
    # L13.B below-TC-tile operand shapes (any dim < 8)
    (4, 4), (8, 8), (8, 4), (4, 32), (32, 4), (6, 32), (32, 6),
}
_L11_TRIPLES = [
    (64,   64,   64),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
    (8192, 1024, 1024),
    (4096, 1024, 1024),
    (2048, 1024, 1024),
    (1024, 1024, 8192),
    (1024, 1024, 4096),
    (1024, 1024, 2048),
]
# L13.B below-TC-tile triples (any dim < 8) — matches the 5
# "below_tc_tile" entries in `fixtures/bench/general_shape_manifest.json`.
_L13_B_SMALL_TRIPLES = [
    (4, 4, 4),
    (8, 8, 4),
    (4, 32, 4),
    (6, 32, 6),
    (4, 4, 32),
]
_TRIPLE_SET = set(_L11_TRIPLES)
_SMALL_TRIPLE_SET = set(_L13_B_SMALL_TRIPLES)


def _bf16_from_fp32(arr_fp32: np.ndarray) -> bytes:
    """Marshal fp32 bytes through Lean's bf16 round-to-nearest-even rule."""
    src = np.asarray(arr_fp32, dtype=np.float32).ravel().tobytes()
    src_buf = (ctypes.c_uint8 * len(src)).from_buffer_copy(src)
    dst_len = len(src) // 2
    dst_buf = (ctypes.c_uint8 * dst_len)()
    rc = _lib.tgrad_bf16_pack_bytes(src_buf, len(src), dst_buf, dst_len)
    if rc != 0:
        raise TgradError(f"Lean bf16 pack returned rc={rc}")
    return bytes(dst_buf)


def _fp32_from_bf16(b: bytes, shape: tuple[int, ...]) -> np.ndarray:
    """Marshal bf16 storage through Lean's exact fp32 expansion."""
    src_buf = (ctypes.c_uint8 * len(b)).from_buffer_copy(b)
    dst_len = len(b) * 2
    dst_buf = (ctypes.c_uint8 * dst_len)()
    rc = _lib.tgrad_bf16_expand_bytes(src_buf, len(b), dst_buf, dst_len)
    if rc != 0:
        raise TgradError(f"Lean bf16 expansion returned rc={rc}")
    return np.frombuffer(bytes(dst_buf), dtype=np.float32).reshape(shape).copy()


class Tensor:
    """Tgrad tensor — owns an MTLBuffer via the Lean allocator.

    `__slots__` includes `__weakref__` so `weakref.finalize` can
    register a cleanup; per learnings/05_opaque_handle/python.py.
    """
    __slots__ = ("_buf", "_size", "_shape", "_dtype", "_handle", "_fin",
                 "_base", "__weakref__")

    def __init__(self, data, dtype: str = "f32"):
        """Construct a materialized Tensor from public Python data.

        Internal runtime code uses ``_from_buffer`` so an integer MTLBuffer
        address cannot be confused with scalar tensor data.
        """
        try:
            arr = np.asarray(data)
        except (TypeError, ValueError) as exc:
            raise TgradTypeError(
                f"Tensor data is not rectangular numeric input: {exc}") from exc
        self._init_from_numpy(arr, dtype)

    def _init_buffer(self, buf: int, size: int, shape: tuple[int, ...],
                     dtype: str, handle: int | None = None,
                     owns_buf: bool = True,
                     base: "Tensor | None" = None) -> None:
        """Initialize Tgrad's internal buffer/view representation.

        `buf` and `size` describe the underlying MTLBuffer.
        `shape` and `dtype` are the *effective* (post-view) shape +
        dtype. `handle` is the Lean-side opaque tensor handle; if
        None (L14.A path), it's auto-registered via
        `tgrad_tensor_from_buffer`. `owns_buf` controls whether
        garbage collection should free the buffer — view-derived
        Tensors share the underlying buffer with the source, so
        only the root sets `owns_buf=True`.

        `base` is the Tensor this one is a view of. A view holds a
        strong reference to it so the parent cannot be collected
        while the view is alive. Without it the parent's finalizer
        returns the MTLBuffer to the LRU pool (or releases it once
        the pool is full) while the view still points at it, so the
        view reads recycled or freed GPU memory."""
        self._buf   = buf
        self._size  = size
        self._shape = shape
        self._dtype = dtype
        self._base  = base
        # A materialized buffer must be exactly the size its shape
        # implies. Views are exempt: `expand` legitimately presents a
        # shape whose element count exceeds the allocation (stride 0),
        # and the bound there is enforced by the Lean `Schedule.View`
        # algebra, not by byte count.
        if owns_buf:
            want = _numel(shape) * _DTYPE_BYTES[dtype]
            if want != size:
                raise TgradTypeError(
                    f"shape {shape} of dtype {dtype} needs {want} bytes "
                    f"but the buffer is {size}; refusing to build a Tensor "
                    f"whose shape disagrees with its allocation")
        if handle is None:
            shape_arr = (ctypes.c_size_t * len(shape))(*shape)
            handle = _lib.tgrad_tensor_from_buffer(
                buf, shape_arr, len(shape), _dtype_code(dtype))
            if handle == 0:
                raise TgradError(
                    f"tgrad_tensor_from_buffer(buf={buf}, shape={shape}) returned 0")
        self._handle = handle
        # On collection, return the buffer to the LRU (only if we own it).
        if owns_buf:
            self._fin = weakref.finalize(self, _lib.tgrad_tensor_free, buf, size)
        else:
            self._fin = None

    @classmethod
    def _from_buffer(cls, buf: int, size: int, shape: tuple[int, ...],
                     dtype: str, handle: int | None = None,
                     owns_buf: bool = True,
                     base: "Tensor | None" = None) -> "Tensor":
        tensor = cls.__new__(cls)
        tensor._init_buffer(buf, size, shape, dtype, handle, owns_buf, base)
        return tensor

    def _init_from_numpy(self, arr: np.ndarray, dtype: str) -> None:
        dtype = _native_dtype_name(dtype)
        if dtype not in _SUPPORTED_DTYPES:
            raise TgradTypeError(
                f"unsupported dtype {dtype!r}; supported: {sorted(_SUPPORTED_DTYPES)}")
        if arr.ndim > 3:
            raise NotInLeanScope(
                f"public Tensor construction supports rank 0 through 3 "
                f"(got ndim={arr.ndim})")
        if arr.dtype.kind not in "biuf":
            raise TgradTypeError(
                f"Tensor data must be a rectangular bool/int/float array "
                f"(got dtype={arr.dtype})")
        shape = tuple(int(s) for s in arr.shape)
        if any(s < 1 for s in shape):
            raise NotInLeanScope(
                f"materialized Tensor dimensions must be positive (got {shape})")
        bytes_ = _bytes_from_numpy(arr, dtype)
        size = len(bytes_)
        buf = _lib.tgrad_tensor_alloc(size)
        if buf == 0:
            raise TgradError(f"tgrad_tensor_alloc({size}) returned 0")
        arr_buf = (ctypes.c_uint8 * size).from_buffer_copy(bytes_)
        rc = _lib.tgrad_tensor_write_bytes(buf, arr_buf, size)
        if rc != 0:
            _lib.tgrad_tensor_free(buf, size)
            raise TgradError(f"tgrad_tensor_write_bytes returned rc={rc}")
        try:
            self._init_buffer(buf, size, shape, dtype)
        except BaseException:
            _lib.tgrad_tensor_free(buf, size)
            raise

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> str:
        return self._dtype

    @classmethod
    def from_numpy(cls, arr: np.ndarray, dtype: str = "bf16") -> "Tensor":
        tensor = cls.__new__(cls)
        tensor._init_from_numpy(np.asarray(arr), dtype)
        return tensor

    @classmethod
    def full(cls, shape, fill_value, dtype: str | None = None, **kwargs) -> "Tensor":
        """Create a tensor filled with ``fill_value``.

        Python owns the calling convention (``_argfix``, kwargs).
        Lean owns dtype default/admission, GPU allocation, and the
        constant-fill kernel. Unknown kwargs (e.g. ``device=``) raise
        rather than being silently ignored.
        """
        if kwargs:
            raise TypeError(
                f"Tensor.full: unsupported keyword argument(s) "
                f"{sorted(kwargs)}; Tgrad accepts only dtype= "
                f"(supported dtypes: {sorted(_SUPPORTED_DTYPES)})")
        shape = tuple(int(s) for s in _argfix(shape))
        code = _creation_dtype_code(dtype)
        shape_arr = (ctypes.c_size_t * len(shape))(*shape)
        h = _lib.tgrad_tensor_full(
            shape_arr, len(shape), float(fill_value), code)
        if h == 0:
            # Lean refused. Map known refusals to the public exceptions
            # without re-implementing admission before the call.
            if code not in (_DTYPE_BF16, _DTYPE_F32, _DTYPE_I32,
                            _CREATION_DTYPE_DEFAULT):
                raise TgradTypeError(
                    f"unsupported dtype {dtype!r}; "
                    f"supported: {sorted(_SUPPORTED_DTYPES)}")
            if any(s < 1 for s in shape):
                raise NotInLeanScope(
                    f"materialized Tensor dimensions must be positive "
                    f"(got {shape})")
            if len(shape) > 3:
                raise NotInLeanScope(
                    f"public Tensor construction supports rank 0 through 3 "
                    f"(got ndim={len(shape)})")
            raise TgradError(
                f"tgrad_tensor_full(shape={shape}, fill={fill_value!r}) "
                f"returned 0")
        out_buf = _lib.tgrad_tensor_raw_buffer(h)
        out_rank = int(_lib.tgrad_tensor_rank(h))
        out_shape = tuple(
            int(_lib.tgrad_tensor_shape_dim(h, i))
            for i in range(out_rank))
        out_dtype = _dtype_of_handle(h)
        return cls._from_buffer(
            out_buf, _numel(out_shape) * _DTYPE_BYTES[out_dtype],
            out_shape, out_dtype, handle=h, owns_buf=True)

    @classmethod
    def zeros(cls, *shape, **kwargs) -> "Tensor":
        """Create a tensor filled with zeros. Shape via ``_argfix``."""
        return cls.full(_argfix(*shape), 0.0, **kwargs)

    @classmethod
    def ones(cls, *shape, **kwargs) -> "Tensor":
        """Create a tensor filled with ones. Shape via ``_argfix``."""
        return cls.full(_argfix(*shape), 1.0, **kwargs)

    @classmethod
    def from_bf16_bytes(cls, raw: bytes, shape: tuple[int, ...]) -> "Tensor":
        """Construct from already-bf16 bytes (e.g. captured tinygrad fixture).
        L13.C+: accepts any 2D shape with dims ≥ 1."""
        shape = tuple(int(s) for s in shape)
        if len(shape) != 2 or any(s < 1 for s in shape):
            raise NotInLeanScope(
                f"shape {shape} must be 2D with positive dims.")
        want = _numel(shape) * _DTYPE_BYTES["bf16"]
        if len(raw) != want:
            raise TgradTypeError(
                f"from_bf16_bytes: shape {shape} needs {want} bytes, got "
                f"{len(raw)}. A shape larger than its buffer makes the "
                f"kernel index past the allocation.")
        size = len(raw)
        buf = _lib.tgrad_tensor_alloc(size)
        if buf == 0:
            raise TgradError(f"tgrad_tensor_alloc({size}) returned 0")
        arr_buf = (ctypes.c_uint8 * size).from_buffer_copy(raw)
        rc = _lib.tgrad_tensor_write_bytes(buf, arr_buf, size)
        if rc != 0:
            _lib.tgrad_tensor_free(buf, size)
            raise TgradError(f"tgrad_tensor_write_bytes returned rc={rc}")
        return cls._from_buffer(buf, size, shape, "bf16")

    _VIEW_KIND_NAMES = {1: "permute", 2: "reshape", 3: "expand", 4: "slice"}

    def _materialize_for_readback(self, method: str) -> "Tensor":
        """Return a contiguous Tensor suitable for host readback.

        BUFFER tensors already are contiguous. Movement views are realized by
        Lean's indexed-copy kernel. The existing two-handle view trampoline
        reserves a zero second handle for this unary operation, avoiding a new
        C ABI entry while still keeping rendering, allocation, and dispatch in
        Lean.

        The temporary output owns its fresh buffer and is freed after the
        caller copies the bytes to Python."""
        kind = self._uop_kind_code()
        if kind == 0:
            return self
        name = self._VIEW_KIND_NAMES.get(kind, f"uop-kind-{kind}")
        out_handle = _lib.tgrad_matmul_view(self._handle, 0)
        if out_handle == 0:
            raise NotInLeanScope(
                f"Tensor.{method}() could not materialize {name} view with "
                f"shape {self._shape}; the movement chain may be invalid, "
                f"unsupported, or empty")
        out_buf = _lib.tgrad_tensor_raw_buffer(out_handle)
        if out_buf == 0:
            raise TgradError(
                f"Tensor.{method}(): materialized handle {out_handle} has no buffer")
        out_rank = _lib.tgrad_tensor_rank(out_handle)
        out_shape = tuple(
            int(_lib.tgrad_tensor_shape_dim(out_handle, i))
            for i in range(out_rank))
        out_size = _numel(out_shape) * _DTYPE_BYTES[self._dtype]
        if out_shape != self._shape:
            _lib.tgrad_tensor_free(out_buf, out_size)
            raise TgradError(
                f"Tensor.{method}(): materialized shape {out_shape} disagrees "
                f"with Python view shape {self._shape}")
        return Tensor._from_buffer(out_buf, out_size, out_shape, self._dtype,
                                   handle=out_handle, owns_buf=True)

    def numpy(self) -> np.ndarray:
        tensor = self._materialize_for_readback("numpy")
        out = (ctypes.c_uint8 * tensor._size)()
        rc = _lib.tgrad_tensor_read_bytes(tensor._buf, out, tensor._size)
        if rc != 0:
            raise TgradError(f"tgrad_tensor_read_bytes returned rc={rc}")
        return _numpy_from_bytes(bytes(out), tensor._shape, tensor._dtype)

    def to_bytes(self) -> bytes:
        tensor = self._materialize_for_readback("to_bytes")
        out = (ctypes.c_uint8 * tensor._size)()
        rc = _lib.tgrad_tensor_read_bytes(tensor._buf, out, tensor._size)
        if rc != 0:
            raise TgradError(f"tgrad_tensor_read_bytes returned rc={rc}")
        return bytes(out)

    def tolist(self):
        """Return exact logical readback using Python's nested-list shape."""
        return self.numpy().tolist()

    def realize(self) -> "Tensor":
        """A materialized Tgrad Tensor realizes to the identical object."""
        return self

    # L14.B.1: view methods — compose movement nodes on the Lean-side
    # uop; return a new Tensor sharing the underlying buffer. Pure
    # graph transforms; no buffer allocation; no kernel dispatch.

    def transpose(self) -> "Tensor":
        new_h = _lib.tgrad_tensor_transpose(self._handle)
        if new_h == 0:
            raise TgradError(f"tgrad_tensor_transpose(handle={self._handle}) returned 0")
        # Effective shape after transpose (2-D PERMUTE [1,0]):
        if len(self._shape) != 2:
            raise TgradTypeError(
                f"Tensor.transpose: 2-D only (got shape={self._shape})")
        new_shape = (self._shape[1], self._shape[0])
        return Tensor._from_buffer(self._buf, self._size, new_shape, self._dtype,
                                   handle=new_h, owns_buf=False, base=self)

    T = property(transpose)

    def permute(self, *axes: int) -> "Tensor":
        if len(axes) != len(self._shape):
            raise TgradTypeError(
                f"Tensor.permute: axes={axes} doesn't match rank={len(self._shape)}")
        n = len(axes)
        arr = (ctypes.c_size_t * n)(*axes)
        new_h = _lib.tgrad_tensor_permute(self._handle, arr, n)
        if new_h == 0:
            raise TgradError(f"tgrad_tensor_permute(axes={axes}) returned 0")
        new_shape = tuple(self._shape[i] for i in axes)
        return Tensor._from_buffer(self._buf, self._size, new_shape, self._dtype,
                                   handle=new_h, owns_buf=False, base=self)

    def reshape(self, *new_shape: int) -> "Tensor":
        if len(new_shape) == 1 and isinstance(new_shape[0], (tuple, list)):
            new_shape = tuple(new_shape[0])
        try:
            new_shape = tuple(int(dim) for dim in new_shape)
        except (TypeError, ValueError) as exc:
            raise TgradTypeError(f"Tensor.reshape: invalid shape {new_shape}") from exc
        if any(dim < 1 for dim in new_shape):
            raise TgradTypeError(
                f"Tensor.reshape: dimensions must be positive (got {new_shape})")
        if _numel(new_shape) != _numel(self._shape):
            raise TgradTypeError(
                f"Tensor.reshape: cannot reshape {self._shape} to {new_shape}")
        n = len(new_shape)
        arr = (ctypes.c_size_t * n)(*new_shape)
        new_h = _lib.tgrad_tensor_reshape(self._handle, arr, n)
        if new_h == 0:
            raise TgradError(f"tgrad_tensor_reshape(shape={new_shape}) returned 0")
        return Tensor._from_buffer(self._buf, self._size, tuple(new_shape), self._dtype,
                                   handle=new_h, owns_buf=False, base=self)

    def expand(self, *new_shape: int) -> "Tensor":
        n = len(new_shape)
        arr = (ctypes.c_size_t * n)(*new_shape)
        new_h = _lib.tgrad_tensor_expand(self._handle, arr, n)
        if new_h == 0:
            raise TgradError(f"tgrad_tensor_expand(shape={new_shape}) returned 0")
        return Tensor._from_buffer(self._buf, self._size, tuple(new_shape), self._dtype,
                                   handle=new_h, owns_buf=False, base=self)

    def __getitem__(self, key) -> "Tensor":
        # Accept a single slice or a tuple of slices.
        if isinstance(key, slice):
            key = (key,)
        if not isinstance(key, tuple):
            raise TgradTypeError(
                f"Tensor.__getitem__: only slice or tuple-of-slice supported (got {type(key).__name__})")
        n = len(key)
        flat = (ctypes.c_size_t * (3 * n))()
        new_shape = list(self._shape)
        for i, sl in enumerate(key):
            if not isinstance(sl, slice):
                raise TgradTypeError(f"Tensor.__getitem__[{i}]: expected slice, got {type(sl).__name__}")
            dim = self._shape[i] if i < len(self._shape) else 0
            start = 0 if sl.start is None else sl.start
            stop  = dim if sl.stop is None else sl.stop
            step  = 1 if sl.step is None else sl.step
            if step <= 0:
                raise TgradTypeError(
                    f"Tensor.__getitem__: only positive steps supported (got step={step})")
            # These land in a `c_size_t` array; a negative value wraps
            # to ~1.8e19, which Lean then clamps to an empty axis while
            # Python computes a *larger* dim than the source. Reject.
            if start < 0 or stop < 0:
                raise TgradTypeError(
                    f"Tensor.__getitem__[{i}]: negative indices are not "
                    f"supported (got start={sl.start}, stop={sl.stop})")
            flat[3 * i]     = start
            flat[3 * i + 1] = stop
            flat[3 * i + 2] = step
            new_shape[i] = max(0, (min(stop, dim) - min(start, dim) + step - 1) // step)
        new_h = _lib.tgrad_tensor_slice(self._handle, flat, 3 * n)
        if new_h == 0:
            raise TgradError(f"tgrad_tensor_slice returned 0")
        return Tensor._from_buffer(self._buf, self._size, tuple(new_shape), self._dtype,
                                   handle=new_h, owns_buf=False, base=self)

    def _uop_kind_code(self) -> int:
        """Return the underlying UOp kind: 0=BUFFER, 1=PERMUTE,
        2=RESHAPE, 3=EXPAND, 4=SLICE, 255=other/unregistered."""
        return _lib.tgrad_tensor_uop_kind(self._handle)

    # Pointwise ops. Each is a table row: build a binop node, hand the
    # graph to realize. No new FFI symbol, no new kernel generator, and
    # views work for free because the index expressions come from the
    # View algebra. Adding another operator is one line here plus one
    # row in Renderer.Elementwise.elementwiseOpStr.
    def _pointwise(self, other: "Tensor", op_code: int, name: str) -> "Tensor":
        if not isinstance(other, Tensor):
            return NotImplemented
        for d in (self._dtype, other._dtype):
            if d not in _SUPPORTED_DTYPES:
                raise TgradTypeError(f"{name}: unsupported dtype {d!r}")
        # Validate numpy-style right-aligned broadcasting at the public
        # boundary so invalid shapes receive a stable, informative error.
        # Lean independently pads the shorter View with leading stride-zero
        # size-one axes before lowering the graph.
        rank = max(len(self._shape), len(other._shape))
        if rank > 3:
            raise NotInLeanScope(f"{name}: ranks above 3 are not supported")
        a_shape = (1,) * (rank - len(self._shape)) + self._shape
        b_shape = (1,) * (rank - len(other._shape)) + other._shape
        for x, y in zip(a_shape, b_shape):
            if x != y and x != 1 and y != 1:
                raise TgradTypeError(
                    f"{name}: shapes {self._shape} and {other._shape} are not "
                    f"broadcastable")
        h = _lib.tgrad_tensor_binop(op_code, self._handle, other._handle)
        if h == 0:
            raise TgradError(f"tgrad_tensor_binop({name}) returned 0")
        out_handle = _lib.tgrad_realize(h)
        if out_handle == 0:
            raise TgradError(f"tgrad_realize({name}) failed for {self._shape}")
        out_buf = _lib.tgrad_tensor_raw_buffer(out_handle)
        out_rank = int(_lib.tgrad_tensor_rank(out_handle))
        out_shape = tuple(
            int(_lib.tgrad_tensor_shape_dim(out_handle, i))
            for i in range(out_rank))
        out_dtype = _dtype_of_handle(out_handle)
        return Tensor._from_buffer(
            out_buf, _numel(out_shape) * _DTYPE_BYTES[out_dtype], out_shape, out_dtype,
            handle=out_handle, owns_buf=True, base=None)

    def __add__(self, other: "Tensor") -> "Tensor":
        return self._pointwise(other, _BINOP_ADD, "add")

    def __sub__(self, other: "Tensor") -> "Tensor":
        return self._pointwise(other, _BINOP_SUB, "sub")

    def __mul__(self, other: "Tensor") -> "Tensor":
        return self._pointwise(other, _BINOP_MUL, "mul")

    # Reductions. Same table-row shape as the pointwise ops: the kernel
    # is shared, only the operator and identity element differ. Keepdim
    # is the only mode, so `sum(axis=1)` on (rows, cols) gives (rows, 1).
    def _reduce(self, op_code: int, axis: int, name: str) -> "Tensor":
        if axis not in (0, 1):
            raise TgradTypeError(f"{name}: axis must be 0 or 1 (got {axis})")
        # The current reduction renderer accumulates in float. Int32 is a
        # pointwise/storage capability only until a native integer reduction
        # packet lands; admitting it here would silently launder precision.
        if self._dtype not in _REDUCE_DTYPES:
            raise TgradTypeError(f"{name}: unsupported dtype {self._dtype!r}")
        h = _lib.tgrad_tensor_reduce(op_code, self._handle, axis)
        if h == 0:
            raise TgradError(f"tgrad_tensor_reduce({name}) returned 0")
        out_handle = _lib.tgrad_realize(h)
        if out_handle == 0:
            raise TgradError(f"tgrad_realize({name}) failed for {self._shape}")
        out_buf = _lib.tgrad_tensor_raw_buffer(out_handle)
        m = _lib.tgrad_tensor_shape_dim(out_handle, 0)
        n = _lib.tgrad_tensor_shape_dim(out_handle, 1)
        dt = _dtype_of_handle(out_handle)
        return Tensor._from_buffer(
            out_buf, m * n * _DTYPE_BYTES[dt], (m, n), dt,
            handle=out_handle, owns_buf=True, base=None)

    def sum(self, axis: int = 1) -> "Tensor":
        return self._reduce(_BINOP_ADD, axis, "sum")

    def prod(self, axis: int = 1) -> "Tensor":
        return self._reduce(_BINOP_MUL, axis, "prod")

    def __matmul__(self, other: "Tensor") -> "Tensor":
        if not isinstance(other, Tensor):
            return NotImplemented
        if self._dtype != "bf16" or other._dtype != "bf16":
            raise TgradTypeError(
                f"matmul: bf16 only (got {self._dtype} @ {other._dtype})")
        # L14.B.2.c: route view inputs through Pipeline.realizeView
        # (which runs the parametric scalar matmul with view-derived
        # index UOps). Replaces L14.B.1's MatmulOnNonBufferUop guard.
        M, K_a = self._shape
        K_b, N = other._shape
        if K_a != K_b:
            raise TgradTypeError(
                f"matmul: contraction dim mismatch ({self._shape} @ {other._shape})")

        if not _USE_ALGEBRAIC:
            # Graph-indexed path. Build `reduce add (mul a b)` and hand it
            # to one entry; Lean picks sentinel / TC / scalar / view. The
            # shape inspection that used to happen here now happens where
            # the type system can see it.
            prod = _lib.tgrad_tensor_binop(_BINOP_MUL, self._handle, other._handle)
            if prod == 0:
                raise TgradError("tgrad_tensor_binop(mul) returned 0")
            graph = _lib.tgrad_tensor_reduce(_BINOP_ADD, prod, 1)
            if graph == 0:
                raise TgradError("tgrad_tensor_reduce(add) returned 0")
            out_handle = _lib.tgrad_realize(graph)
            if out_handle == 0:
                raise TgradError(
                    f"tgrad_realize failed for {self._shape} @ {other._shape}")
            out_buf = _lib.tgrad_tensor_raw_buffer(out_handle)
            out_M = _lib.tgrad_tensor_shape_dim(out_handle, 0)
            out_N = _lib.tgrad_tensor_shape_dim(out_handle, 1)
            return Tensor._from_buffer(
                out_buf, out_M * out_N * 2, (out_M, out_N), "bf16",
                handle=out_handle, owns_buf=True, base=None)

        # Algebraic-emit route, retained so L12 can observe a distinct
        # cache path via --use-algebraic-emit.
        a_kind = self._uop_kind_code()
        b_kind = other._uop_kind_code()
        if a_kind != 0 or b_kind != 0:
            out_handle = _lib.tgrad_matmul_view(self._handle, other._handle)
            if out_handle == 0:
                raise TgradError(
                    f"tgrad_matmul_view(a.uop={a_kind}, b.uop={b_kind}) "
                    f"returned 0 — Pipeline.realizeView failed")
            out_buf = _lib.tgrad_tensor_raw_buffer(out_handle)
            out_rank = _lib.tgrad_tensor_rank(out_handle)
            if out_rank != 2:
                raise TgradError(
                    f"tgrad_matmul_view: expected 2-D output, got rank {out_rank}")
            out_M = _lib.tgrad_tensor_shape_dim(out_handle, 0)
            out_N = _lib.tgrad_tensor_shape_dim(out_handle, 1)
            out_size = out_M * out_N * 2  # bf16 = 2 bytes/elem
            return Tensor._from_buffer(
                out_buf, out_size, (out_M, out_N), "bf16",
                handle=out_handle, owns_buf=True)
        M, K_a = self._shape
        K_b, N = other._shape
        if K_a != K_b:
            raise TgradTypeError(
                f"matmul: contraction dim mismatch ({self._shape} @ {other._shape})")
        K = K_a
        if (M, K, N) in _TRIPLE_SET:
            # L11/L12 sentinel path. Both entry points render the parametric
            # TC declaration; the alternate symbol keeps cache-route
            # observability for L12's generated sweep.
            entry = _lib.tgrad_matmul_alg if _USE_ALGEBRAIC else _lib.tgrad_matmul
            entry_name = "tgrad_matmul_alg" if _USE_ALGEBRAIC else "tgrad_matmul"
        elif _lib.tgrad_matmul_tc_eligible(M, K, N) == 1:
            # L13.F TC-eligible non-sentinel: route through the Lean-
            # owned WMMA kernel emit. Routing decision comes from
            # Lean's `tcMatmulKernelDecl` legality check, NOT a
            # Python-side shape table (per L13.F gate D3).
            if _USE_MANUAL_LOAD_TC:
                entry = _lib.tgrad_matmul_tc_manual_load
                entry_name = "tgrad_matmul_tc_manual_load"
            else:
                entry = _lib.tgrad_matmul_tc
                entry_name = "tgrad_matmul_tc"
        else:
            # L13.B + L13.C + L13.D: catch-all scalar path. Used for
            # below-TC-tile + TC-misaligned shapes that Lean's plan
            # query rejects.
            entry = _lib.tgrad_matmul_small
            entry_name = "tgrad_matmul_small"
        out_size = M * N * 2  # bf16: 2 bytes/element
        out_buf = _lib.tgrad_tensor_alloc(out_size)
        if out_buf == 0:
            raise TgradError(f"tgrad_tensor_alloc({out_size}) for matmul output returned 0")
        rc = entry(M, K, N, self._buf, other._buf, out_buf)
        if rc != 0:
            _lib.tgrad_tensor_free(out_buf, out_size)
            raise TgradError(
                f"{entry_name}(M={M}, K={K}, N={N}) returned rc={rc} "
                f"(see PythonFFI.lean for rc → reason mapping)")
        return Tensor._from_buffer(out_buf, out_size, (M, N), "bf16")


def bench(shape: str = "64x64x64", dtype: str = "bf16") -> dict:
    """Run a single-shape FFI bench: matmul on captured tinygrad inputs,
    compare result vs captured expected output. Returns parsed status."""
    if shape != "64x64x64":
        raise NotInLeanScope(f"L6.b bench scope: 64x64x64 only (got {shape})")
    if dtype != "bf16":
        raise TgradTypeError(f"L6.b bench dtype: bf16 only (got {dtype})")
    fix_dir = REPO_ROOT / "fixtures" / "pipeline"
    a_bytes = (fix_dir / "matmul_64x64_bf16_seed42_a.bin").read_bytes()
    b_bytes = (fix_dir / "matmul_64x64_bf16_seed42_b.bin").read_bytes()
    e_bytes = (fix_dir / "matmul_64x64_bf16_seed42_expected.bin").read_bytes()
    a = Tensor.from_bf16_bytes(a_bytes, (64, 64))
    b = Tensor.from_bf16_bytes(b_bytes, (64, 64))
    c = a @ b
    actual = c.to_bytes()
    matched = actual == e_bytes
    return {
        "shape":     shape,
        "dtype":     dtype,
        "actual_len": len(actual),
        "expected_len": len(e_bytes),
        "byte_match": matched,
    }


def bench_timing(shape: str = "64x64x64", dtype: str = "bf16",
                 n_warmup: int = 50, n_measured: int = 200) -> dict:
    """Time Tgrad's matmul via the ctypes FFI. Returns a dict with
    median/percentile ms statistics. Mirrors the tinygrad capture's
    methodology so the lean_ms ↔ tinygrad_ms ratio is honest.

    Before timing, byte-matches the matmul output against the captured
    tinygrad expected fixture — this closes the "skip a@b in the loop"
    sabotage gap (a no-op timing would pass the perf predicate but the
    byte-match would fail first)."""
    if shape != "64x64x64":
        raise NotInLeanScope(f"L7 timing scope: 64x64x64 only (got {shape})")
    if dtype != "bf16":
        raise TgradTypeError(f"L7 timing dtype: bf16 only (got {dtype})")
    fix_dir = REPO_ROOT / "fixtures" / "pipeline"
    a_bytes = (fix_dir / "matmul_64x64_bf16_seed42_a.bin").read_bytes()
    b_bytes = (fix_dir / "matmul_64x64_bf16_seed42_b.bin").read_bytes()
    e_bytes = (fix_dir / "matmul_64x64_bf16_seed42_expected.bin").read_bytes()
    a = Tensor.from_bf16_bytes(a_bytes, (64, 64))
    b = Tensor.from_bf16_bytes(b_bytes, (64, 64))
    # Pre-timing correctness anchor: same path bench_timing measures must
    # produce bit-identical output to L5.b's expected. Catches the "fake
    # the timing by skipping a @ b" sabotage at L7's own gate level.
    c0 = a @ b
    if c0.to_bytes() != e_bytes:
        raise TgradError(
            "bench-timing: matmul output does NOT byte-match captured fixture; "
            "the timing measurement would not reflect correct work")
    # Warmup (cache warm; no further correctness checks — we've established it).
    for _ in range(n_warmup):
        _ = a @ b
    times_ms: list[float] = []
    for _ in range(n_measured):
        t0 = time.perf_counter()
        _ = a @ b
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)
    times_ms.sort()
    return {
        "shape":      shape,
        "dtype":      dtype,
        "n_warmup":   n_warmup,
        "n_measured": n_measured,
        "min":        times_ms[0],
        "p25":        times_ms[int(0.25 * len(times_ms))],
        "median":     statistics.median(times_ms),
        "p75":        times_ms[int(0.75 * len(times_ms))],
        "max":        times_ms[-1],
    }


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Tgrad Python authoring layer (L6.b)")
    sub = p.add_subparsers(dest="cmd", required=True)
    bp = sub.add_parser("bench", help="run a single-shape FFI bench")
    bp.add_argument("--shape", required=True, help="e.g. 64x64x64")
    bp.add_argument("--dtype", default="bf16")
    tp = sub.add_parser("bench-timing", help="time Tgrad's matmul (L7 perf parity)")
    tp.add_argument("--shape", required=True, help="e.g. 64x64x64")
    tp.add_argument("--dtype", default="bf16")
    tp.add_argument("--warmup",  type=int, default=50)
    tp.add_argument("--measured", type=int, default=200)
    bs = sub.add_parser("bench-small",
                        help="L13.B: correctness-only sweep over below-TC-tile shapes")
    bs.add_argument("--manifest",
                    default=str(REPO_ROOT / "fixtures" / "bench" / "general_shape_manifest.json"))
    bs.add_argument("--output", default="/tmp/tgrad_L13_B_bench.jsonl")
    bg = sub.add_parser("bench-general",
                        help="L13.C: correctness-only sweep over non-below-TC-tile manifest entries")
    bg.add_argument("--manifest",
                    default=str(REPO_ROOT / "fixtures" / "bench" / "general_shape_manifest.json"))
    bg.add_argument("--output", default="/tmp/tgrad_L13_C_bench.jsonl")
    br = sub.add_parser("bench-random-shapes",
                        help="L13.D: random shape sweep (HEAD-derived seed; correctness only)")
    br.add_argument("--seed", required=True,
                    help="hex seed (typically 'git rev-parse HEAD' prefix)")
    br.add_argument("--count", type=int, default=30)
    br.add_argument("--output", default="/tmp/tgrad_L13_D_bench.jsonl")
    btc = sub.add_parser("bench-tc-general",
                         help="L13.F: 8 pinned TC-eligible non-sentinel shapes (correctness + ratio)")
    btc.add_argument("--manifest",
                     default=str(REPO_ROOT / "fixtures" / "bench" / "tc_general_manifest.json"))
    btc.add_argument("--baseline", default=None,
                     help="default: fixtures/perf/tinygrad_baseline_tc_general_<profile>.json")
    btc.add_argument("--output", default="/tmp/tgrad_L13_F_tc_general.jsonl")
    btc.add_argument("--warmup",   type=int, default=10)
    btc.add_argument("--measured", type=int, default=30)
    btc.add_argument("--use-manual-load", action="store_true",
                     help="route TC-general matmul through the "
                          "L13.F.STRICT.B manual-load kernel")
    brtc = sub.add_parser("bench-random-tc-general",
                          help="L13.F: random TC-eligible non-sentinel shapes (correctness only)")
    brtc.add_argument("--seed", required=True)
    brtc.add_argument("--count", type=int, default=10)
    brtc.add_argument("--output", default="/tmp/tgrad_L13_F_random_tc.jsonl")
    brtc.add_argument("--use-manual-load", action="store_true",
                      help="route TC-general matmul through the "
                           "L13.F.STRICT.B manual-load kernel")
    bv = sub.add_parser("bench-views",
                        help="L14.B.3: 16 pinned view-matmul cases (correctness)")
    bv.add_argument("--manifest",
                    default=str(REPO_ROOT / "fixtures" / "bench" / "view_manifest.json"))
    bv.add_argument("--output", default="/tmp/tgrad_L14_B_3_views.jsonl")
    brv = sub.add_parser("bench-random-views",
                         help="L14.C: anti-hardcoding random view-matmul sweep (seed=HEAD)")
    brv.add_argument("--seed", required=True,
                     help="hex string — first 16 chars of git rev-parse HEAD")
    brv.add_argument("--count", type=int, default=20)
    brv.add_argument("--output", default="/tmp/tgrad_L14_C_random.jsonl")
    bf = sub.add_parser("bench-full", help="run the L11 50-pair sweep (correctness + ratio)")
    bf.add_argument("--manifest",
                    default=str(REPO_ROOT / "fixtures" / "bench" / "pair_manifest.json"))
    bf.add_argument("--baseline", default=None,
                    help="default: fixtures/perf/tinygrad_baseline_<profile>_full.json")
    bf.add_argument("--output",   default="/tmp/tgrad_L11_bench.jsonl")
    bf.add_argument("--warmup",   type=int, default=10)
    bf.add_argument("--measured", type=int, default=30)
    bf.add_argument("--use-algebraic-emit", dest="use_algebraic",
                    action="store_true",
                    help="route generated matmul through L12's independent "
                         "alternate cache entry (tgrad_matmul_alg)")
    args = p.parse_args(argv)
    if args.cmd == "bench":
        try:
            r = bench(args.shape, args.dtype)
        except NotInLeanScope as exc:
            print(f"NotInLeanScope: {exc}", file=sys.stderr)
            return 1
        except TgradError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"py_shape: {args.shape}")
        print(f"py_dtype: {args.dtype}")
        print(f"py_actual_len: {r['actual_len']}")
        print(f"py_expected_len: {r['expected_len']}")
        print(f"py_byte_match: {'true' if r['byte_match'] else 'false'}")
        print(f"py_pipeline_ok: {'true' if r['byte_match'] else 'false'}")
        return 0 if r["byte_match"] else 1
    if args.cmd == "bench-timing":
        try:
            r = bench_timing(args.shape, args.dtype, args.warmup, args.measured)
        except NotInLeanScope as exc:
            print(f"NotInLeanScope: {exc}", file=sys.stderr); return 1
        except TgradError as exc:
            print(f"error: {exc}", file=sys.stderr); return 1
        print(f"py_shape: {args.shape}")
        print(f"py_dtype: {args.dtype}")
        print(f"py_n_warmup: {r['n_warmup']}")
        print(f"py_n_measured: {r['n_measured']}")
        print(f"py_lean_ms_min: {r['min']:.4f}")
        print(f"py_lean_ms_p25: {r['p25']:.4f}")
        print(f"py_lean_ms_median: {r['median']:.4f}")
        print(f"py_lean_ms_p75: {r['p75']:.4f}")
        print(f"py_lean_ms_max: {r['max']:.4f}")
        return 0
    if args.cmd == "bench-small":
        import tgrad_bench
        try:
            summary = tgrad_bench.run_bench_small(args.manifest, args.output)
        except NotInLeanScope as exc:
            print(f"NotInLeanScope: {exc}", file=sys.stderr); return 1
        except TgradError as exc:
            print(f"error: {exc}", file=sys.stderr); return 1
        print(f"py_bench_small_output: {args.output}")
        print(f"py_bench_small_below_tc_tile_count: {summary['below_tc_tile_count']}")
        print(f"py_bench_small_n_correct: {summary['n_correct']}")
        if summary["failed"]:
            print(f"py_bench_small_errors: {len(summary['failed'])}")
            for f in summary["failed"]:
                print(f"  ERROR: {f['shape']:20s} {f['dist']:20s} "
                      f"max_abs_diff={f['max_abs_diff']:.6f}",
                      file=sys.stderr)
            return 1
        print(f"py_bench_small_ok: true")
        return 0
    if args.cmd == "bench-general":
        import tgrad_bench
        try:
            summary = tgrad_bench.run_bench_general(args.manifest, args.output)
        except NotInLeanScope as exc:
            print(f"NotInLeanScope: {exc}", file=sys.stderr); return 1
        except TgradError as exc:
            print(f"error: {exc}", file=sys.stderr); return 1
        print(f"py_bench_general_output: {args.output}")
        print(f"py_bench_general_count: {summary['general_count']}")
        print(f"py_bench_general_n_correct: {summary['n_correct']}")
        if summary["failed"]:
            print(f"py_bench_general_errors: {len(summary['failed'])}")
            for f in summary["failed"]:
                print(f"  ERROR: {f['shape']:20s} bucket={f['bucket']:25s} "
                      f"max_abs_diff={f['max_abs_diff']:.6f}",
                      file=sys.stderr)
            return 1
        print(f"py_bench_general_ok: true")
        return 0
    if args.cmd == "bench-tc-general":
        import tgrad_bench
        baseline_path = (args.baseline if args.baseline is not None else
                         str(REPO_ROOT / "fixtures" / "perf"
                             / f"tinygrad_baseline_tc_general_{PERF_PROFILE}.json"))
        try:
            summary = tgrad_bench.run_bench_tc_general(
                args.manifest, baseline_path, args.output,
                warmup=args.warmup, measured=args.measured,
                use_manual_load=args.use_manual_load)
        except NotInLeanScope as exc:
            print(f"NotInLeanScope: {exc}", file=sys.stderr); return 1
        except TgradError as exc:
            print(f"error: {exc}", file=sys.stderr); return 1
        print(f"py_bench_tc_general_output: {args.output}")
        print(f"py_bench_tc_general_count: {summary['manifest_count']}")
        print(f"py_bench_tc_general_n_correct: {summary['n_correct']}")
        print(f"py_bench_tc_general_n_tc_route: {summary['n_tc_route']}")
        print(f"py_bench_tc_general_n_scalar_route: {summary['n_scalar_route']}")
        if "ratio_max" in summary:
            print(f"py_bench_tc_general_ratio_max: {summary['ratio_max']:.4f}")
            print(f"py_bench_tc_general_ratio_median: {summary['ratio_median']:.4f}")
        if summary["failed"]:
            print(f"py_bench_tc_general_issues: {len(summary['failed'])}")
            for f in summary["failed"]:
                print(f"  ISSUE: {f['shape']:25s} route={f['route']:10s} "
                      f"correct={f['correct']} ratio={f['ratio']:.4f}",
                      file=sys.stderr)
            return 1
        print("py_bench_tc_general_ok: true")
        return 0
    if args.cmd == "bench-random-tc-general":
        import tgrad_bench
        try:
            summary = tgrad_bench.run_bench_random_tc_general(
                args.seed, args.count, args.output,
                use_manual_load=args.use_manual_load)
        except NotInLeanScope as exc:
            print(f"NotInLeanScope: {exc}", file=sys.stderr); return 1
        except TgradError as exc:
            print(f"error: {exc}", file=sys.stderr); return 1
        print(f"py_bench_random_tc_output: {args.output}")
        print(f"py_bench_random_tc_seed: {summary['seed_hex']}")
        print(f"py_bench_random_tc_count: {summary['count']}")
        print(f"py_bench_random_tc_actual: {summary['actual_count']}")
        print(f"py_bench_random_tc_n_correct: {summary['n_correct']}")
        print(f"py_bench_random_tc_n_tc_route: {summary['n_tc_route']}")
        if summary["failed"]:
            print(f"py_bench_random_tc_issues: {len(summary['failed'])}")
            return 1
        print("py_bench_random_tc_ok: true")
        return 0
    if args.cmd == "bench-random-shapes":
        import tgrad_bench
        try:
            summary = tgrad_bench.run_bench_random_shapes(args.seed, args.count, args.output)
        except NotInLeanScope as exc:
            print(f"NotInLeanScope: {exc}", file=sys.stderr); return 1
        except TgradError as exc:
            print(f"error: {exc}", file=sys.stderr); return 1
        print(f"py_bench_random_output: {args.output}")
        print(f"py_bench_random_seed: {summary['seed_hex']}")
        print(f"py_bench_random_count: {summary['count']}")
        print(f"py_bench_random_n_correct: {summary['n_correct']}")
        if summary["failed"]:
            print(f"py_bench_random_errors: {len(summary['failed'])}")
            for f in summary["failed"]:
                print(f"  ERROR: idx={f['idx']:3d} {f['shape']:20s} "
                      f"max_abs_diff={f['max_abs_diff']:.6f}",
                      file=sys.stderr)
            return 1
        print(f"py_bench_random_ok: true")
        return 0
    if args.cmd == "bench-views":
        import tgrad_bench
        try:
            summary = tgrad_bench.run_bench_views(args.manifest, args.output)
        except NotInLeanScope as exc:
            print(f"NotInLeanScope: {exc}", file=sys.stderr); return 1
        except TgradError as exc:
            print(f"error: {exc}", file=sys.stderr); return 1
        print(f"py_bench_views_output: {args.output}")
        print(f"py_bench_views_count: {summary['manifest_count']}")
        print(f"py_bench_views_n_correct: {summary['n_correct']}")
        print(f"py_bench_views_n_route_view: {summary['n_route_view']}")
        print(f"py_bench_views_n_route_buffer: {summary['n_route_buffer']}")
        if summary["failed"]:
            print(f"py_bench_views_errors: {len(summary['failed'])}")
            for f in summary["failed"]:
                err = f.get("error", f"max_abs_diff={f.get('max_abs_diff', '?')}")
                shape_str = f"{f.get('M','?')}x{f.get('K','?')}x{f.get('N','?')}"
                print(f"  ERROR: {f.get('op_chain', '?'):16s} {shape_str:14s}  {err}",
                      file=sys.stderr)
            return 1
        print("py_bench_views_ok: true")
        return 0
    if args.cmd == "bench-random-views":
        import tgrad_bench
        try:
            summary = tgrad_bench.run_bench_random_views(args.seed, args.count, args.output)
        except NotInLeanScope as exc:
            print(f"NotInLeanScope: {exc}", file=sys.stderr); return 1
        except TgradError as exc:
            print(f"error: {exc}", file=sys.stderr); return 1
        print(f"py_random_views_output: {args.output}")
        print(f"py_random_views_count: {summary['count']}")
        print(f"py_random_views_n_correct: {summary['n_correct']}")
        print(f"py_random_views_n_route_view: {summary['n_route_view']}")
        print(f"py_random_views_ops_used: {','.join(summary['ops_used'])}")
        print(f"py_random_views_seed: {summary['seed_hex']}")
        if summary["failed"]:
            print(f"py_random_views_errors: {len(summary['failed'])}")
            for f in summary["failed"]:
                err = f.get("error") or f"max_abs_diff={f.get('max_abs_diff','?')}"
                shape_str = f"{f.get('M','?')}x{f.get('K','?')}x{f.get('N','?')}"
                print(f"  ERROR: {f.get('op_chain','?'):16s} {shape_str:14s}  {err}",
                      file=sys.stderr)
            return 1
        print("py_random_views_ok: true")
        return 0
    if args.cmd == "bench-full":
        import tgrad_bench
        baseline_path = (args.baseline if args.baseline is not None else
                         str(REPO_ROOT / "fixtures" / "perf"
                             / f"tinygrad_baseline_{PERF_PROFILE}_full.json"))
        # L12: switch generated matmul routing to the alternate FFI cache.
        # Set BEFORE invoking run_bench_full so warm-up + timing both
        # exercise the algebraic path.
        if args.use_algebraic:
            set_use_algebraic(True)
        try:
            summary = tgrad_bench.run_bench_full(
                args.manifest, baseline_path, args.output,
                warmup=args.warmup, measured=args.measured,
            )
        except NotInLeanScope as exc:
            print(f"NotInLeanScope: {exc}", file=sys.stderr); return 1
        except TgradError as exc:
            print(f"error: {exc}", file=sys.stderr); return 1
        print(f"py_bench_full_output: {args.output}")
        print(f"py_bench_full_use_algebraic: {'true' if args.use_algebraic else 'false'}")
        print(f"py_bench_full_n_pairs: {summary['manifest_count']}")
        print(f"py_bench_full_n_correct: {summary['n_correct']}")
        print(f"py_bench_full_n_ratio_ok: {summary['n_ratio_ok']}")
        print(f"py_bench_full_ratio_min: {summary['ratio_min']:.4f}")
        print(f"py_bench_full_ratio_median: {summary['ratio_median']:.4f}")
        print(f"py_bench_full_ratio_max: {summary['ratio_max']:.4f}")
        if summary["failed"]:
            correctness_errors = [f for f in summary["failed"] if not f["correct"]]
            perf_misses = [f for f in summary["failed"] if f["correct"]]
            if correctness_errors:
                print(f"py_bench_full_correctness_errors: {len(correctness_errors)}")
                for f in correctness_errors:
                    print(
                        f"  ERROR: {f['shape']:20s} {f['dist']:20s} "
                        f"correct={f['correct']} ratio={f['ratio']:.4f}",
                        file=sys.stderr)
            if perf_misses:
                print(f"py_bench_full_perf_miss: {len(perf_misses)}")
                for f in perf_misses:
                    print(
                        f"  PERF_MISS: {f['shape']:20s} {f['dist']:20s} "
                        f"correct={f['correct']} ratio={f['ratio']:.4f}",
                        file=sys.stderr)
            # Distinct exit codes, because these are distinct claims.
            #   1 = a NUMERICAL result was wrong
            #   2 = every result was correct; some pair missed the ratio
            # They used to share exit 1, which meant a caller that cares
            # only about correctness was failed by performance. L12 is
            # exactly such a caller: its own evidence records
            # "performance_predicate": "not evaluated by semantic gate",
            # yet it died on this return before reaching its 50/50
            # correctness assertion. Callers that DO claim performance
            # (L11) assert on n_ratio_ok themselves and get a better
            # diagnostic from that than from an opaque non-zero exit.
            return 1 if correctness_errors else 2
        print(f"py_bench_full_ok: true")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
