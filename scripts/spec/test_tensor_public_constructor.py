#!/usr/bin/env python3
"""Contract tests for WORK-PY-TENSOR-PUBLIC-CONSTRUCTOR-V2."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import tgrad


class PublicTensorConstructorTests(unittest.TestCase):
    def assertTensor(self, data, shape, expected):
        tensor = tgrad.Tensor(data, dtype="f32")
        self.assertEqual(tensor.shape, shape)
        self.assertEqual(tensor.dtype, "f32")
        np.testing.assert_array_equal(
            tensor.numpy(), np.asarray(expected, dtype=np.float32))

    def test_scalar(self):
        self.assertTensor(1.5, (), np.asarray(1.5, dtype=np.float32))

    def test_flat_list(self):
        self.assertTensor([1, 2, 3], (3,), [1, 2, 3])

    def test_nested_list(self):
        self.assertTensor([[1, 2], [3, 4]], (2, 2), [[1, 2], [3, 4]])

    def test_rank_three_numpy(self):
        data = np.arange(6, dtype=np.float32).reshape(2, 1, 3)
        self.assertTensor(data, (2, 1, 3), data)

    def test_tuple_and_variadic_reshape_construct_views(self):
        source = tgrad.Tensor([1, 2, 3, 4, 5, 6], dtype="f32")
        by_tuple = source.reshape((2, 3))
        by_args = source.reshape(2, 3)
        self.assertEqual(by_tuple.shape, (2, 3))
        self.assertEqual(by_args.shape, (2, 3))
        self.assertIs(by_tuple._base, source)
        self.assertIs(by_args._base, source)

    def test_int32_constructor_preserves_dtype_and_values(self):
        tensor = tgrad.Tensor([1, -2, 16777217], dtype="i32")
        self.assertEqual(tensor.dtype, "i32")
        self.assertEqual(tensor.shape, (3,))
        np.testing.assert_array_equal(
            tensor.numpy(), np.asarray([1, -2, 16777217], dtype=np.int32))

    def test_still_unsupported_dtype_fails_before_allocation(self):
        with self.assertRaises(tgrad.TgradTypeError):
            tgrad.Tensor([1, 2], dtype="f16")

    def test_ragged_input_is_rejected(self):
        with self.assertRaises(tgrad.TgradTypeError):
            tgrad.Tensor([[1], [2, 3]], dtype="f32")

    def test_empty_materialization_is_rejected(self):
        with self.assertRaises(tgrad.NotInLeanScope):
            tgrad.Tensor([], dtype="f32")

    def test_internal_buffer_protocol_is_not_public_positional_api(self):
        with self.assertRaises(TypeError):
            tgrad.Tensor(1234, 16, (2, 2), "f32")


if __name__ == "__main__":
    unittest.main()
