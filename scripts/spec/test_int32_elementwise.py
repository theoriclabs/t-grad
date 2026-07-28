#!/usr/bin/env python3
"""Focused verifier for WORK-DTYPE-I32-ELEMENTWISE-V1."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import tgrad


class Int32ElementwiseTests(unittest.TestCase):
    def test_public_constructor_and_negative_view_readback_are_four_byte_i32(self):
        values = np.asarray([16777217, -16777217, 0, 2147483000], dtype=np.int32)
        tensor = tgrad.Tensor(values, dtype="i32")
        self.assertEqual(tensor.dtype, "i32")
        self.assertEqual(tensor._size, values.size * 4)
        self.assertEqual(tensor.numpy().dtype, np.dtype(np.int32))
        np.testing.assert_array_equal(tensor.numpy(), values)

        view = tensor.reshape(2, 2).transpose()
        np.testing.assert_array_equal(view.numpy(), values.reshape(2, 2).T)
        self.assertEqual(view.tolist(), values.reshape(2, 2).T.tolist())

    def test_frozen_right_aligned_rank_extension(self):
        left_np = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        right_np = np.asarray([10, 20, 30], dtype=np.int32)
        left = tgrad.Tensor(left_np, dtype="i32")
        right = tgrad.Tensor(right_np, dtype="i32")
        left_before, right_before = left.numpy().copy(), right.numpy().copy()

        result = left + right
        self.assertEqual(result.shape, (2, 3))
        self.assertEqual(result.dtype, "i32")
        np.testing.assert_array_equal(result.numpy(), left_np + right_np)
        self.assertIs(result.realize(), result)
        np.testing.assert_array_equal(result.numpy(), result.numpy())
        np.testing.assert_array_equal(left.numpy(), left_before)
        np.testing.assert_array_equal(right.numpy(), right_before)

    def test_native_integer_add_sub_mul_do_not_launder_through_float(self):
        left_np = np.asarray([16777217, -16777217, 4097], dtype=np.int32)
        right_np = np.asarray([1, -1, 4097], dtype=np.int32)
        left = tgrad.Tensor(left_np, dtype="i32")
        right = tgrad.Tensor(right_np, dtype="i32")

        for result, expected in (
            (left + right, left_np + right_np),
            (left - right, left_np - right_np),
            (left * right, left_np * right_np),
        ):
            self.assertEqual(result.dtype, "i32")
            np.testing.assert_array_equal(result.numpy(), expected)

    def test_mixed_int32_float32_scalar_uses_lub_and_float_compute(self):
        left_np = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        left = tgrad.Tensor(left_np, dtype="i32")
        right = tgrad.Tensor(0.5, dtype="f32")
        result = left + right

        self.assertEqual(result.shape, (2, 3))
        self.assertEqual(result.dtype, "f32")
        np.testing.assert_array_equal(result.numpy(), left_np.astype(np.float32) + 0.5)

    def test_int32_reduction_remains_explicitly_out_of_scope(self):
        tensor = tgrad.Tensor([[1, 2], [3, 4]], dtype="i32")
        with self.assertRaisesRegex(tgrad.TgradTypeError, "unsupported dtype"):
            tensor.sum(axis=1)


if __name__ == "__main__":
    unittest.main()
