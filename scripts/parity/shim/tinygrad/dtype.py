"""Tgrad's implemented dtype identifiers under tinygrad import spellings."""
from __future__ import annotations

import numpy as np
import tgrad as _tgrad

from ._unsupported import missing_attribute, unsupported, unsupported_type


_BF16 = "bf16"
_F32 = "f32"


class _DTypes:
    bfloat16 = _BF16
    float32 = _F32
    float = _F32
    default_float = _F32
    floats = (_BF16, _F32)
    fp8s = ()

    # These identifiers are importable for test parametrization, but any
    # attempted dtype behavior raises through the marker.
    bool = unsupported("tinygrad.dtype.dtypes.bool")
    int8 = unsupported("tinygrad.dtype.dtypes.int8")
    int16 = unsupported("tinygrad.dtype.dtypes.int16")
    int32 = unsupported("tinygrad.dtype.dtypes.int32")
    int64 = long = unsupported("tinygrad.dtype.dtypes.int64")
    uint8 = unsupported("tinygrad.dtype.dtypes.uint8")
    uint16 = unsupported("tinygrad.dtype.dtypes.uint16")
    uint32 = unsupported("tinygrad.dtype.dtypes.uint32")
    uint64 = ulong = unsupported("tinygrad.dtype.dtypes.uint64")
    float16 = half = unsupported("tinygrad.dtype.dtypes.float16")
    float64 = double = unsupported("tinygrad.dtype.dtypes.float64")
    fp8e4m3 = unsupported("tinygrad.dtype.dtypes.fp8e4m3")
    fp8e5m2 = unsupported("tinygrad.dtype.dtypes.fp8e5m2")
    fp8e4m3fnuz = unsupported("tinygrad.dtype.dtypes.fp8e4m3fnuz")
    fp8e5m2fnuz = unsupported("tinygrad.dtype.dtypes.fp8e5m2fnuz")

    @staticmethod
    def is_float(dtype) -> bool:
        if dtype in _tgrad._SUPPORTED_DTYPES:
            return True
        raise _tgrad.TgradTypeError(f"unsupported dtype {dtype!r}")

    @staticmethod
    def is_int(dtype) -> bool:
        if dtype in _tgrad._SUPPORTED_DTYPES:
            return False
        raise _tgrad.TgradTypeError(f"unsupported dtype {dtype!r}")

    @staticmethod
    def is_unsigned(dtype) -> bool:
        if dtype in _tgrad._SUPPORTED_DTYPES:
            return False
        raise _tgrad.TgradTypeError(f"unsupported dtype {dtype!r}")


dtypes = _DTypes()
DTYPES_DICT = {"bfloat16": _BF16, "float32": _F32}


def _to_np_dtype(dtype):
    if dtype in (_BF16, _F32):
        # Tgrad's bf16 host readback is deliberately lifted to float32.
        return np.dtype(np.float32)
    raise _tgrad.TgradTypeError(f"unsupported dtype {dtype!r}")


DType = unsupported_type("tinygrad.dtype.DType")
InvalidType = _tgrad.TgradTypeError
Invalid = unsupported("tinygrad.dtype.Invalid")
to_dtype = unsupported("tinygrad.dtype.to_dtype")
can_lossless_cast = unsupported("tinygrad.dtype.can_lossless_cast")
truncate = unsupported("tinygrad.dtype.truncate")
float_to_fp16 = unsupported("tinygrad.dtype.float_to_fp16")
float_to_bf16 = unsupported("tinygrad.dtype.float_to_bf16")
least_upper_dtype = unsupported("tinygrad.dtype.least_upper_dtype")
least_upper_float = unsupported("tinygrad.dtype.least_upper_float")
fp8_to_float = unsupported("tinygrad.dtype.fp8_to_float")
float_to_fp8 = unsupported("tinygrad.dtype.float_to_fp8")
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
