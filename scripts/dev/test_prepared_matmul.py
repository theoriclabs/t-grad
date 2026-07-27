#!/usr/bin/env python3
"""Metal lifecycle/correctness checks for Tgrad's prepared matmul boundary.

Run serially after rebuilding the dylib. This is not a benchmark.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import tgrad  # noqa: E402


class PreparedMatmulTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fixtures = ROOT / "fixtures" / "pipeline"
        cls.a_bytes = (fixtures / "matmul_64x64_bf16_seed42_a.bin").read_bytes()
        cls.b_bytes = (fixtures / "matmul_64x64_bf16_seed42_b.bin").read_bytes()
        cls.expected = (fixtures / "matmul_64x64_bf16_seed42_expected.bin").read_bytes()

    def test_foreign_fixture_matches_and_output_buffer_is_reused(self) -> None:
        a = tgrad.Tensor.from_bf16_bytes(self.a_bytes, (64, 64))
        b = tgrad.Tensor.from_bf16_bytes(self.b_bytes, (64, 64))
        prepared = a.prepare_matmul(b)
        self.assertEqual(prepared.route, "sentinel")
        output_buffer = prepared.output._buf
        prepared.poison_output()
        first = prepared.run()
        self.assertIs(first, prepared.output)
        self.assertEqual(first._buf, output_buffer)
        self.assertEqual(first.to_bytes(), self.expected)
        prepared.poison_output(0x5A)
        second = prepared.run()
        self.assertEqual(second._buf, output_buffer)
        self.assertEqual(second.to_bytes(), self.expected)
        prepared.close()

    def test_same_shape_inputs_can_be_replaced_without_repreparing(self) -> None:
        a = tgrad.Tensor.from_bf16_bytes(self.a_bytes, (64, 64))
        b = tgrad.Tensor.from_bf16_bytes(self.b_bytes, (64, 64))
        a2 = tgrad.Tensor.from_bf16_bytes(self.b_bytes, (64, 64))
        b2 = tgrad.Tensor.from_bf16_bytes(self.a_bytes, (64, 64))
        expected = (a2 @ b2).to_bytes()
        with a.prepare_matmul(b) as prepared:
            output_buffer = prepared.output._buf
            actual = prepared.run(a2, b2)
            self.assertEqual(actual._buf, output_buffer)
            self.assertEqual(actual.to_bytes(), expected)

    def test_release_and_view_misuse_fail_loudly(self) -> None:
        a = tgrad.Tensor.from_bf16_bytes(self.a_bytes, (64, 64))
        b = tgrad.Tensor.from_bf16_bytes(self.b_bytes, (64, 64))
        with self.assertRaises(tgrad.TgradTypeError):
            a.T.prepare_matmul(b)
        prepared = a.prepare_matmul(b)
        prepared.close()
        with self.assertRaisesRegex(tgrad.TgradError, "released"):
            prepared.run()


if __name__ == "__main__":
    unittest.main()
