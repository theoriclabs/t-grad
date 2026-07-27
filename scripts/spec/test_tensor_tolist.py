#!/usr/bin/env python3
"""Contract tests for WORK-PY-F32-VIEW-READBACK-V1."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import tgrad


class TensorToListTests(unittest.TestCase):
    def test_buffer_scalar_and_matrix_preserve_shape(self):
        self.assertEqual(tgrad.Tensor(1.5, dtype="f32").tolist(), 1.5)
        self.assertEqual(
            tgrad.Tensor([[1, 2], [3, 4]], dtype="f32").tolist(),
            [[1.0, 2.0], [3.0, 4.0]])

    def test_float32_reshape_view_readback(self):
        source = tgrad.Tensor([1, 2, 3, 4, 5, 6], dtype="f32")
        view = source.reshape((2, 3))
        self.assertEqual(view.tolist(), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    def test_float32_view_copy_is_bit_preserving(self):
        bits = np.asarray([0x80000000, 0x3f800001, 0x7fc01234, 0xff800000],
                          dtype=np.uint32)
        values = bits.view(np.float32)
        source = tgrad.Tensor(values, dtype="f32")
        view = source.reshape((2, 2))
        self.assertEqual(view.to_bytes(), bits.tobytes())


if __name__ == "__main__":
    unittest.main()
