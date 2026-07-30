"""Strict tinygrad dtype spelling backed by Tgrad's Lean semantic core.

This module exposes metadata and promotion that Tgrad now implements. It does
not add tensor kernels: compute admission remains `tgrad._SUPPORTED_DTYPES`.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import struct

import tgrad as _tgrad

from ._unsupported import missing_attribute, unsupported


_VALID_CODES = tuple(code for code in range(255) if _tgrad._dtype_query(code, 0) == 1)
_PUBLIC_BY_CODE = {code: _tgrad._dtype_public_name(code) for code in _VALID_CODES}


@dataclass(frozen=True, eq=False, slots=True, init=False)
class DType:
    """Identity singleton whose semantic answers come from Lean.

    Upstream DTypes are objects, not strings: equality and hashing are object
    identity.  The strict shim intentionally inherits neither string equality
    nor ordering operations that upstream does not define.
    """
    code: int
    public_name: str
    priority: int
    bitsize: int
    name: str
    fmt: str | None
    itemsize: int

    def __init__(self, code: int):
        public = _PUBLIC_BY_CODE[code]
        if _tgrad._dtype_query(code, 0) != 1:
            raise _tgrad.TgradTypeError(f"Lean rejected dtype code {code}")
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "public_name", public)
        priority = _tgrad._dtype_query(code, 1)
        object.__setattr__(
            self, "priority", -1 if priority == (1 << 64) - 1 else priority)
        object.__setattr__(self, "bitsize", _tgrad._dtype_query(code, 2))
        object.__setattr__(self, "itemsize", _tgrad._dtype_query(code, 3))
        object.__setattr__(self, "name", _tgrad._dtype_backend_name(code))
        fmt_code = _tgrad._dtype_query(code, 14)
        object.__setattr__(self, "fmt", chr(fmt_code) if fmt_code else None)

    def __repr__(self):
        return f"dtypes.{_tgrad._dtype_display_name(self.code)}"

    __str__ = __repr__

    def __lt__(self, other):
        if not isinstance(other, DType): return NotImplemented
        return bool(_tgrad._dtype_binary_query(2, self.code, other.code))

    def scalar(self):
        return self

    @property
    def min(self):
        kind = _tgrad._dtype_query(self.code, 8)
        if kind == 0: return False
        if kind == 1:
            encoded = _tgrad._dtype_query(self.code, 9)
            return encoded - (1 << 64) if encoded & (1 << 63) else encoded
        if kind == 2: return 0
        if kind == 3: return -math.inf
        raise _tgrad.TgradTypeError(f"{self!r} has no concrete representable minimum")

    @property
    def max(self):
        kind = _tgrad._dtype_query(self.code, 8)
        if kind == 0: return True
        if kind in (1, 2): return _tgrad._dtype_query(self.code, 10)
        if kind == 3: return math.inf
        raise _tgrad.TgradTypeError(f"{self!r} has no concrete representable maximum")


_BY_CODE = {code: DType(code) for code in _PUBLIC_BY_CODE}


class InvalidType:
    _instance = None
    def __new__(cls):
        if cls._instance is None: cls._instance = object.__new__(cls)
        return cls._instance
    def __repr__(self): return "Invalid"


Invalid = InvalidType()


class _DTypes:
    @property
    def default_int(self): return _BY_CODE[_tgrad._dtype_default(0)]
    @default_int.setter
    def default_int(self, dtype):
        if not isinstance(dtype, DType) or not _tgrad._dtype_set_default(0, dtype.code):
            raise _tgrad.TgradTypeError(f"Lean rejected integer default {dtype!r}")
    @property
    def default_float(self): return _BY_CODE[_tgrad._dtype_default(1)]
    @default_float.setter
    def default_float(self, dtype):
        if not isinstance(dtype, DType) or not _tgrad._dtype_set_default(1, dtype.code):
            raise _tgrad.TgradTypeError(f"Lean rejected floating default {dtype!r}")

    @staticmethod
    def is_float(dtype) -> bool: return bool(_tgrad._dtype_query(dtype.code, 4))
    @staticmethod
    def is_int(dtype) -> bool: return bool(_tgrad._dtype_query(dtype.code, 5))
    @staticmethod
    def is_unsigned(dtype) -> bool: return bool(_tgrad._dtype_query(dtype.code, 6))
    @staticmethod
    def is_bool(dtype) -> bool: return bool(_tgrad._dtype_query(dtype.code, 7))
    @staticmethod
    def finfo(dtype):
        exponent = _tgrad._dtype_query(dtype.code, 11)
        mantissa = _tgrad._dtype_query(dtype.code, 12)
        if exponent == 255 or mantissa == 255:
            raise ValueError(f"{dtype!r} is not a floating point type")
        return exponent, mantissa
    @staticmethod
    def from_py(value):
        return _BY_CODE[_tgrad._dtype_infer_python(_python_dtype_tags(value))]


def _python_dtype_tags(value):
    """Normalize Python syntax to primitive tags; Lean owns their meaning."""
    if isinstance(value, (bool, InvalidType)): return [0]
    if isinstance(value, float): return [2]
    if isinstance(value, int): return [1]
    if isinstance(value, (list, tuple)):
        if not value: return [3]
        tags = []
        for item in value: tags.extend(_python_dtype_tags(item))
        return tags
    raise RuntimeError(f"Could not infer dtype of {value} with type {type(value)}")


dtypes = _DTypes()
for code, dtype in _BY_CODE.items():
    setattr(dtypes, _PUBLIC_BY_CODE[code], dtype)

_ALIASES = {}
for row in range(_tgrad._dtype_table_query(0, 0)):
    name = _tgrad._dtype_table_name(0, row)
    target = _BY_CODE[_tgrad._dtype_table_query(0, 1, row)]
    _ALIASES[name] = target
    if name not in ("default_int", "default_float"):
        setattr(dtypes, name, target)

for row in range(_tgrad._dtype_table_query(1, 0)):
    name = _tgrad._dtype_table_name(1, row)
    count = _tgrad._dtype_table_query(1, 1, row)
    setattr(dtypes, name, tuple(
        _BY_CODE[_tgrad._dtype_table_query(1, 2, row, column)]
        for column in range(count)))

DTYPES_DICT = {dtype.public_name: dtype for dtype in dtypes.all}
DTYPES_DICT.update({
    name: target for name, target in _ALIASES.items()
    if name not in ("default_int", "default_float")})


def to_dtype(dtype):
    if isinstance(dtype, DType): return dtype
    try: return getattr(dtypes, str(dtype).lower())
    except AttributeError as exc: raise RuntimeError(f"unknown dtype {dtype!r}") from exc


def strong_dtype(dtype):
    return _BY_CODE[_tgrad._dtype_unary_query(0, to_dtype(dtype).code)]


def least_upper_dtype(*values):
    if not values: raise TypeError("least_upper_dtype requires at least one dtype")
    codes = [to_dtype(value).code for value in values]
    return _BY_CODE[_tgrad._dtype_lub_many(codes)]


def least_upper_float(dtype):
    return _BY_CODE[_tgrad._dtype_unary_query(1, to_dtype(dtype).code)]


def can_lossless_cast(source, target):
    return bool(_tgrad._dtype_binary_query(1, to_dtype(source).code, to_dtype(target).code))


def float_to_bf16(value):
    # Normalize through fp32 exactly as upstream does, then marshal the bit
    # pattern through Lean's scalar-result rule. Tensor storage conversion is
    # intentionally a different boundary: it cannot preserve low NaN payload.
    normalized = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    rounded = _tgrad._bf16_round_bits(normalized)
    return struct.unpack("<f", struct.pack("<I", rounded))[0]


def float_to_fp16(value):
    input_bits = struct.unpack("<Q", struct.pack("<d", float(value)))[0]
    result_bits = _tgrad._low_precision_scalar(2, dtypes.float16.code, input_bits)
    return struct.unpack("<d", struct.pack("<Q", result_bits))[0]


def _low_precision_dtype_code(dtype):
    """Marshal object identity only; Lean decides descriptor admission."""
    return dtype.code if isinstance(dtype, DType) else 255


def float_to_fp8(value, dtype):
    try:
        return _tgrad._low_precision_value_public(
            0, value, _low_precision_dtype_code(dtype))
    except _tgrad._LowPrecisionBoundaryError as exc:
        if exc.reason == 1:
            raise AssertionError("Only for fp8s") from exc
        if exc.reason == 5:
            raise OverflowError("int too large to convert to float") from exc
        raise TypeError(
            f"must be real number, not {type(value).__name__}") from exc


def fp8_to_float(payload, dtype):
    try:
        result_bits = _tgrad._low_precision_decode_public(
            payload, _low_precision_dtype_code(dtype))
    except _tgrad._LowPrecisionBoundaryError as exc:
        if exc.reason == 1:
            raise AssertionError("Only for fp8s") from exc
        raise TypeError(
            f"unsupported operand type(s) for &: "
            f"'{type(payload).__name__}' and 'int'") from exc
    return struct.unpack("<d", struct.pack("<Q", result_bits))[0]


def _truncate_fp8(value, dtype):
    try:
        result_bits = _tgrad._low_precision_value_public(
            3, value, dtype.code)
    except _tgrad._LowPrecisionBoundaryError as exc:
        if exc.reason == 1:
            raise AssertionError("Only for fp8s") from exc
        if exc.reason == 5:
            raise OverflowError("int too large to convert to float") from exc
        raise TypeError(
            f"must be real number, not {type(value).__name__}") from exc
    return struct.unpack("<d", struct.pack("<Q", result_bits))[0]


def _to_np_dtype(dtype):
    import numpy as np
    dtype = to_dtype(dtype)
    if dtype in (dtypes.bfloat16, *dtypes.fp8s): return np.float32
    return np.dtype(dtype.fmt).type if dtype.fmt is not None else None


# Every callable delegates to Lean. This table exposes a dtype-indexed
# relation; it does not carry conversion arithmetic or tensor admission.
truncate = {
    dtypes.float16: float_to_fp16,
    dtypes.bfloat16: float_to_bf16,
    dtypes.fp8e4m3: lambda value: _truncate_fp8(value, dtypes.fp8e4m3),
    dtypes.fp8e5m2: lambda value: _truncate_fp8(value, dtypes.fp8e5m2),
    dtypes.fp8e4m3fnuz: lambda value: _truncate_fp8(value, dtypes.fp8e4m3fnuz),
    dtypes.fp8e5m2fnuz: lambda value: _truncate_fp8(value, dtypes.fp8e5m2fnuz),
}
_to_torch_dtype = unsupported("tinygrad.dtype._to_torch_dtype")
_from_torch_dtype = unsupported("tinygrad.dtype._from_torch_dtype")

__all__ = (
    "dtypes", "DTYPES_DICT", "DType", "InvalidType", "Invalid", "to_dtype",
    "can_lossless_cast", "truncate", "float_to_fp16", "float_to_bf16",
    "_to_np_dtype", "least_upper_dtype", "least_upper_float", "fp8_to_float",
    "float_to_fp8", "_to_torch_dtype", "_from_torch_dtype",
)


def __getattr__(name: str):
    missing_attribute("tinygrad.dtype", name)
