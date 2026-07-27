#!/usr/bin/env python3
"""Contract tests for WORK-PY-REALIZE-IDENTITY-V1."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import tgrad


class TensorRealizeTests(unittest.TestCase):
    def test_materialized_tensor_realizes_to_itself_repeatedly(self):
        tensor = tgrad.Tensor([[1, 2], [3, 4]], dtype="f32")
        buf = tensor._buf
        first = tensor.realize()
        second = tensor.realize()
        self.assertIs(first, tensor)
        self.assertIs(second, tensor)
        self.assertIs(first, second)
        self.assertEqual(tensor._buf, buf)

    def test_pointwise_result_realizes_to_itself(self):
        left = tgrad.Tensor([[1, 2], [3, 4]], dtype="f32")
        right = tgrad.Tensor([[10, 20], [30, 40]], dtype="f32")
        result = left + right
        self.assertIs(result.realize(), result)


if __name__ == "__main__":
    unittest.main()
