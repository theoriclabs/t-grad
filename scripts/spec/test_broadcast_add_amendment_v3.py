from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.spec import generate_broadcast_add_amendment_v3 as generator


class BroadcastAddAmendmentV3Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = json.loads(generator.DEFAULT_INPUT.read_text(encoding="utf-8"))

    def parse_mutation(self, mutate) -> None:
        document = copy.deepcopy(self.document)
        mutate(document)
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "amendment.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            generator.parse(path)

    def test_committed_amendment_is_narrow_and_bound(self) -> None:
        document, amendment_hash, effective_hash = generator.parse(generator.DEFAULT_INPUT)
        self.assertEqual(2, len(document["trace_footprint_amendments"]))
        self.assertEqual(64, len(amendment_hash))
        self.assertEqual(64, len(effective_hash))

    def test_third_trace_amendment_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "exactly the two"):
            self.parse_mutation(lambda doc: doc["trace_footprint_amendments"].append(
                copy.deepcopy(doc["trace_footprint_amendments"][0])))

    def test_semantic_dimension_amendment_is_not_available(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "exactly the two"):
            self.parse_mutation(lambda doc: doc["trace_footprint_amendments"][0].__setitem__(
                "to", ["value"]))

    def test_refuting_diagnostic_drift_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "refuting diagnostic"):
            self.parse_mutation(lambda doc: doc.__setitem__(
                "refuting_diagnostic_sha256", "0" * 64))

    def test_product_chronology_cannot_be_weakened(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "chronology or inheritance"):
            self.parse_mutation(lambda doc: doc.__setitem__(
                "frozen_before_product_candidate", False))


if __name__ == "__main__":
    unittest.main()
