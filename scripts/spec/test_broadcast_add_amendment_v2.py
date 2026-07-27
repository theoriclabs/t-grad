from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.spec import generate_broadcast_add_amendment_v2 as generator


class BroadcastAddAmendmentV2Tests(unittest.TestCase):
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
        self.assertEqual(2, len(document["amendments"]))
        self.assertEqual(64, len(amendment_hash))
        self.assertEqual(64, len(effective_hash))

    def test_third_amendment_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "exactly two"):
            self.parse_mutation(lambda doc: doc["amendments"].append(
                copy.deepcopy(doc["amendments"][0])))

    def test_dimension_change_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "more than dtype observability"):
            self.parse_mutation(lambda doc: doc["amendments"][0].__setitem__(
                "dimension", "shape"))

    def test_base_manifest_drift_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "exact V1 manifest"):
            self.parse_mutation(lambda doc: doc.__setitem__(
                "base_manifest_sha256", "0" * 64))

    def test_v1_disposition_cannot_be_erased(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "disposition"):
            self.parse_mutation(lambda doc: doc.__setitem__("v1_disposition", "passed"))


if __name__ == "__main__":
    unittest.main()
