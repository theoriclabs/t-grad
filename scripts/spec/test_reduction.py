#!/usr/bin/env python3
"""Positive differential tests for standalone sum/prod lowering."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import tgrad


class ReductionTests(unittest.TestCase):
    def assertReduction(self, values, axis, method, expected, *, dtype="f32"):
        tensor = tgrad.Tensor(np.asarray(values, dtype=np.float32), dtype=dtype)
        before = tensor.numpy().copy()
        result = getattr(tensor, method)(axis=axis)
        np.testing.assert_array_equal(
            result.numpy(), np.asarray(expected, dtype=np.float32))
        np.testing.assert_array_equal(tensor.numpy(), before)
        self.assertEqual(result.dtype, dtype)
        self.assertIs(result.realize(), result)
        return result

    def test_sum_both_axes_keeps_rank(self):
        values = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        by_row = self.assertReduction(
            values, 1, "sum", values.sum(axis=1, keepdims=True))
        by_col = self.assertReduction(
            values, 0, "sum", values.sum(axis=0, keepdims=True))
        self.assertEqual(by_row.shape, (2, 1))
        self.assertEqual(by_col.shape, (1, 3))

    def test_prod_uses_distinct_operator_and_identity(self):
        values = np.asarray([[2, 3, 4], [5, 6, 7]], dtype=np.float32)
        result = self.assertReduction(
            values, 1, "prod", values.prod(axis=1, keepdims=True))
        self.assertEqual(result.shape, (2, 1))

    def test_view_indexing_is_used_by_reduction(self):
        values = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        tensor = tgrad.Tensor(values, dtype="f32").transpose()
        result = tensor.sum(axis=1)
        np.testing.assert_array_equal(
            result.numpy(), values.T.sum(axis=1, keepdims=True))
        self.assertEqual(result.shape, (3, 1))

    def test_bfloat16_store_rounds_to_nearest_even(self):
        # Each row lands exactly halfway between adjacent bf16 values.  The
        # first lower endpoint is even and rounds down; the second is odd and
        # rounds up.  Every input is exactly representable as bf16, so host
        # conversion cannot manufacture the expected answer.
        values = np.asarray(
            [[1.0, 0.00390625], [1.0078125, 0.00390625]],
            dtype=np.float32,
        )
        result = self.assertReduction(
            values, 1, "sum", [[1.0], [1.015625]], dtype="bf16")
        self.assertEqual(result.shape, (2, 1))

    def test_invalid_axis_is_rejected_before_ffi(self):
        tensor = tgrad.Tensor([[1, 2], [3, 4]], dtype="f32")
        with self.assertRaisesRegex(tgrad.TgradTypeError, "axis must be 0 or 1"):
            tensor.sum(axis=2)


if __name__ == "__main__":
    unittest.main()
