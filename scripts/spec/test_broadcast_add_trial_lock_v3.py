from __future__ import annotations

import copy
import json
import unittest

from scripts.spec import generate_broadcast_add_trial_lock_v3 as lock


class BroadcastAddTrialLockV3Tests(unittest.TestCase):
    def test_committed_lock_matches_frozen_git_facts(self) -> None:
        document = json.loads(lock.DEFAULT_OUTPUT.read_text(encoding="utf-8"))
        lock.verify(document, require_current=True)

    def test_tampered_observer_identity_is_rejected(self) -> None:
        document = lock.build()
        document["v2_observer_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "differs from frozen Git facts"):
            lock.verify(document, require_current=False)

    def test_product_prohibition_is_not_optional(self) -> None:
        document = copy.deepcopy(lock.build())
        document["promotion_policy"][
            "product_candidate_forbidden_before_baseline_observation"
        ] = False
        with self.assertRaisesRegex(RuntimeError, "differs from frozen Git facts"):
            lock.verify(document, require_current=False)


if __name__ == "__main__":
    unittest.main()
