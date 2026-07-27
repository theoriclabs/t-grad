from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.spec import check_broadcast_add_v4_tooling_amendment as checker


class BroadcastAddV4ToolingAmendmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = checker.verify()

    def verify_mutation(self, mutate) -> None:
        document = copy.deepcopy(self.document)
        mutate(document)
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "v4.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            checker.verify(path)

    def test_committed_amendment_is_exact(self) -> None:
        self.assertEqual("current_worktree_file", self.document["tooling_amendment"]["from"])

    def test_broadening_the_change_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "changes more"):
            self.verify_mutation(lambda doc: doc["tooling_amendment"].__setitem__(
                "binding", "all_verifier_files"))

    def test_product_chronology_is_required(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "chronology"):
            self.verify_mutation(lambda doc: doc.__setitem__(
                "frozen_before_product_candidate", False))


if __name__ == "__main__":
    unittest.main()
