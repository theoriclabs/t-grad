#!/usr/bin/env python3
"""Differential tests for WORK-EW-RANKED-BROADCAST-V1."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import tgrad


class RankedBroadcastTests(unittest.TestCase):
    def assertOp(self, left, right, op, expected, *, dtype="f32"):
        a = tgrad.Tensor(left, dtype=dtype)
        b = tgrad.Tensor(right, dtype=dtype)
        a_before = a.numpy().copy()
        b_before = b.numpy().copy()
        result = op(a, b)
        np.testing.assert_array_equal(result.numpy(), np.asarray(expected, dtype=np.float32))
        np.testing.assert_array_equal(a.numpy(), a_before)
        np.testing.assert_array_equal(b.numpy(), b_before)
        self.assertIs(result.realize(), result)
        return result

    def test_rank_zero_output(self):
        result = self.assertOp(2.0, 3.0, lambda a, b: a + b, 5.0)
        self.assertEqual(result.shape, ())

    def test_rank_one_output(self):
        left = np.asarray([1, 2, 3], dtype=np.float32)
        right = np.asarray([10], dtype=np.float32)
        result = self.assertOp(left, right, lambda a, b: a + b, left + right)
        self.assertEqual(result.shape, (3,))

    def test_frozen_two_sided_rank_three_add(self):
        left = np.asarray([1, 2, 3, 4, 5, 6], dtype=np.float32).reshape(2, 1, 3)
        right = np.asarray([10, 20], dtype=np.float32).reshape(1, 2, 1)
        result = self.assertOp(left, right, lambda a, b: a + b, left + right)
        self.assertEqual(result.shape, (2, 2, 3))
        self.assertEqual(result.dtype, "f32")

    def test_right_aligned_rank_extension(self):
        left = np.arange(6, dtype=np.float32).reshape(2, 3)
        right = np.asarray([10, 20, 30], dtype=np.float32)
        self.assertOp(left, right, lambda a, b: a + b, left + right)

    def test_scalar_broadcast(self):
        left = np.arange(6, dtype=np.float32).reshape(2, 3)
        self.assertOp(left, 0.5, lambda a, b: a + b, left + 0.5)

    def test_rank_three_sub_and_mul_keep_distinct_kernels(self):
        left = np.asarray([1, 2, 3], dtype=np.float32).reshape(1, 1, 3)
        right = np.asarray([2, 4], dtype=np.float32).reshape(1, 2, 1)
        self.assertOp(left, right, lambda a, b: a - b, left - right)
        self.assertOp(left, right, lambda a, b: a * b, left * right)

    def test_rank_three_bfloat16(self):
        left = np.asarray([1, 2, 3, 4, 5, 6], dtype=np.float32).reshape(2, 1, 3)
        right = np.asarray([8, 16], dtype=np.float32).reshape(1, 2, 1)
        result = self.assertOp(
            left, right, lambda a, b: a + b, left + right, dtype="bf16")
        self.assertEqual(result.shape, (2, 2, 3))
        self.assertEqual(result.dtype, "bf16")

    def test_right_incompatible_shape_is_rejected(self):
        left = tgrad.Tensor(np.zeros((2, 3), dtype=np.float32), dtype="f32")
        right = tgrad.Tensor(np.zeros((2,), dtype=np.float32), dtype="f32")
        with self.assertRaisesRegex(tgrad.TgradTypeError, "not broadcastable"):
            _ = left + right

    def test_rank_two_singleton_regression(self):
        left = np.arange(6, dtype=np.float32).reshape(2, 3)
        right = np.asarray([[10], [20]], dtype=np.float32)
        self.assertOp(left, right, lambda a, b: a + b, left + right)


if __name__ == "__main__":
    unittest.main()
