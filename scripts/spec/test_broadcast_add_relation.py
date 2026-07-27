from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from scripts.spec import broadcast_add_relation as relation

REPO = Path(__file__).resolve().parents[2]
BASELINE = REPO / "fixtures/requirements/observations/84a58222575eab06ecc72889e1dbbe2a2084849356673a8d299c21ad2e41a844/observation.json"


class BroadcastAddRelationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference = json.loads(BASELINE.read_text(encoding="utf-8"))
        self.candidate = copy.deepcopy(self.reference)
        self.candidate["identity"]["verifier"]["observer_sha256"] = "f" * 64
        self.candidate["identity"]["verifier"]["git"]["revision"] = "e" * 40
        self.candidate["identity"]["verifier"]["git"]["tree"] = "d" * 40

    def test_revision_only_provenance_difference_is_accepted(self) -> None:
        self.assertTrue(relation.equivalent(self.reference, self.candidate))

    def assert_mutant_rejected(self, mutate) -> None:
        candidate = copy.deepcopy(self.candidate)
        mutate(candidate)
        self.assertFalse(relation.equivalent(self.reference, candidate))

    def test_probe_mismatch_is_rejected(self) -> None:
        self.assert_mutant_rejected(lambda doc: doc["identity"]["verifier"].__setitem__(
            "probe_sha256", "0" * 64))

    def test_schema_mismatch_is_rejected(self) -> None:
        self.assert_mutant_rejected(lambda doc: doc["identity"]["verifier"].__setitem__(
            "schema_version", 99))

    def test_manifest_mismatch_is_rejected(self) -> None:
        self.assert_mutant_rejected(lambda doc: doc["identity"]["manifest"].__setitem__(
            "effective_sha256", "0" * 64))

    def test_semantic_lock_mismatch_is_rejected(self) -> None:
        self.assert_mutant_rejected(lambda doc: doc["identity"]["lock"].__setitem__(
            "semantic_lock_sha256", "0" * 64))

    def test_missing_relation_identity_is_rejected(self) -> None:
        del self.candidate["identity"]["verifier"]["probe_sha256"]
        with self.assertRaisesRegex(RuntimeError, "probe_sha256"):
            relation.equivalent(self.reference, self.candidate)


if __name__ == "__main__":
    unittest.main()
