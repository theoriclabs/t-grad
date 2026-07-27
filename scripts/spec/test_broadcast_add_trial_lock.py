from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.spec import check_broadcast_add_trial_lock as lock


class BroadcastAddTrialLockTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = json.loads(lock.DEFAULT_LOCK.read_text(encoding="utf-8"))

    def verify_mutation(self, mutate) -> None:
        document = copy.deepcopy(self.document)
        mutate(document)
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "lock.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            lock.verify(path, require_current=False)

    def test_committed_lock_verifies_current_tree(self) -> None:
        lock.verify(lock.DEFAULT_LOCK, require_current=True)

    def test_tampered_frozen_hash_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "frozen file hash mismatch"):
            self.verify_mutation(lambda doc: doc["definition_files"].__setitem__(
                "Tgrad/Requirements/Relation.lean", "0" * 64))

    def test_weakened_policy_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "weakens a required promotion policy"):
            self.verify_mutation(lambda doc: doc["promotion_policy"].__setitem__(
                "product_candidate_forbidden_before_baseline_observation", False))

    def test_incomplete_change_set_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "change set does not match"):
            self.verify_mutation(lambda doc: doc["definition_change_set"].pop())

    def test_definition_revision_cannot_contain_observer(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "observer already existed"):
            self.verify_mutation(lambda doc: doc["pending_artifacts"].__setitem__(
                "observer", "scripts/spec/generate_broadcast_add_manifest.py"))

    def test_backend_cannot_drift(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "execution backend changed"):
            self.verify_mutation(lambda doc: doc["execution_boundary"].__setitem__(
                "backend", "CPU"))


if __name__ == "__main__":
    unittest.main()
