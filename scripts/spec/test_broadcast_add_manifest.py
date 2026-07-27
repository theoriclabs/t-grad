#!/usr/bin/env python3
"""Adversarial checks for the prospective broadcast-add freeze boundary."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.spec import generate_broadcast_add_manifest as generator


class ManifestTests(unittest.TestCase):
    def load(self) -> dict:
        return json.loads(generator.DEFAULT_INPUT.read_text(encoding="utf-8"))

    def write(self, root: Path, document: dict) -> Path:
        path = root / "manifest.json"
        path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
        return path

    def test_committed_manifest_is_valid_and_complete(self) -> None:
        document, content_hash = generator.parse(generator.DEFAULT_INPUT)
        self.assertEqual(6, len(document["_scenario_ids"]))
        self.assertEqual(8, len(document["_mutation_ids"]))
        self.assertEqual(64, len(content_hash))

    def test_missing_scenario_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            document = self.load()
            document["scenarios"].pop()
            with self.assertRaisesRegex(RuntimeError, "exactly six scenarios"):
                generator.parse(self.write(Path(raw), document))

    def test_duplicate_mutation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            document = self.load()
            document["mutations"][1]["id"] = document["mutations"][0]["id"]
            with self.assertRaisesRegex(RuntimeError, "contains duplicates"):
                generator.parse(self.write(Path(raw), document))

    def test_unknown_mutation_dimension_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            document = self.load()
            document["mutations"][0]["must_fail"] = ["performance"]
            with self.assertRaisesRegex(RuntimeError, "unknown dimensions"):
                generator.parse(self.write(Path(raw), document))

    def test_nonprospective_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            document = self.load()
            document["frozen_before_product_change"] = False
            with self.assertRaisesRegex(RuntimeError, "not declared prospectively"):
                generator.parse(self.write(Path(raw), document))

    def test_operand_value_count_must_match_shape(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            document = self.load()
            document["scenarios"][0]["left"]["values"].pop()
            with self.assertRaisesRegex(RuntimeError, "value count does not match"):
                generator.parse(self.write(Path(raw), document))

    def test_unknown_observation_token_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            document = self.load()
            document["scenarios"][0]["observe"].append("looks_good")
            with self.assertRaisesRegex(RuntimeError, "unknown observation tokens"):
                generator.parse(self.write(Path(raw), document))


if __name__ == "__main__":
    unittest.main()
