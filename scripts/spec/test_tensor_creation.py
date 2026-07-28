#!/usr/bin/env python3
"""Contract tests for Tensor.full / ones / zeros (M0 creation surface)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import tgrad


class TensorCreationTests(unittest.TestCase):
    def test_ones_tuple_and_variadic_shape(self):
        by_tuple = tgrad.Tensor.ones((2, 3))
        by_args = tgrad.Tensor.ones(2, 3)
        self.assertEqual(by_tuple.shape, (2, 3))
        self.assertEqual(by_args.shape, (2, 3))
        self.assertEqual(by_tuple.dtype, "f32")
        np.testing.assert_array_equal(
            by_tuple.numpy(), np.ones((2, 3), dtype=np.float32))
        np.testing.assert_array_equal(
            by_args.numpy(), np.ones((2, 3), dtype=np.float32))

    def test_zeros_and_full(self):
        z = tgrad.Tensor.zeros(2, 2)
        f = tgrad.Tensor.full((2, 2), 7.0)
        self.assertEqual(z.dtype, "f32")
        self.assertEqual(f.dtype, "f32")
        np.testing.assert_array_equal(
            z.numpy(), np.zeros((2, 2), dtype=np.float32))
        np.testing.assert_array_equal(
            f.numpy(), np.full((2, 2), 7.0, dtype=np.float32))

    def test_ones_rank3_milestone_shape(self):
        t = tgrad.Tensor.ones((1, 3, 4096))
        self.assertEqual(t.shape, (1, 3, 4096))
        self.assertEqual(t.dtype, "f32")
        out = t.realize().numpy()
        self.assertTrue((out == 1).all())

    def test_dtype_kwarg_supported_and_unsupported(self):
        for dtype, np_dtype, fill in (
            ("bf16", np.float32, 1.0),
            ("f32", np.float32, 1.0),
            ("i32", np.int32, 1),
        ):
            t = tgrad.Tensor.ones(2, 2, dtype=dtype)
            self.assertEqual(t.dtype, dtype)
            np.testing.assert_array_equal(
                t.numpy(), np.ones((2, 2), dtype=np_dtype))
        with self.assertRaises(tgrad.TgradTypeError):
            tgrad.Tensor.ones(2, 2, dtype="float64")
        with self.assertRaises(tgrad.TgradTypeError):
            tgrad.Tensor.full((2, 2), 0.0, dtype="f16")

    def test_argfix_rejects_tuple_plus_extra(self):
        with self.assertRaises(ValueError) as ctx:
            tgrad.Tensor.ones((1, 2), 3)
        self.assertIn("bad arg", str(ctx.exception))

    def test_unsupported_kwarg_raises(self):
        with self.assertRaises(TypeError) as ctx:
            tgrad.Tensor.ones(2, 2, device="CPU")
        self.assertIn("device", str(ctx.exception))
        with self.assertRaises(TypeError):
            tgrad.Tensor.full((2, 2), 1.0, device="CPU")
        with self.assertRaises(TypeError):
            tgrad.Tensor.zeros(2, buffer=False)

    def test_scalar_shape_supported_zero_dim_rejected(self):
        scalar = tgrad.Tensor.ones()
        self.assertEqual(scalar.shape, ())
        self.assertEqual(float(scalar.numpy()), 1.0)
        with self.assertRaises(tgrad.NotInLeanScope):
            tgrad.Tensor.zeros(0, 3)


if __name__ == "__main__":
    unittest.main()
