#!/usr/bin/env python3
"""Narrow tests for scripts/contract/check_cycle_chronology.py.

Focus: synthetic Git mutation-then-reversion under tempfile, proving
endpoint-only comparison misses intermediate drift while full-history
checking catches it; and branch/merge cases where --ancestry-path excludes
unrelated reachable commits. Read-only against the Tgrad repository.
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

_CHECKER = Path(__file__).resolve().parent / "check_cycle_chronology.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location(
        "check_cycle_chronology", _CHECKER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {_CHECKER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


chrono = _load_checker()


class CycleChronologyGitTests(unittest.TestCase):
    def test_mutation_reversion_caught_by_full_history_only(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tgrad-cycle-chronology-test-") as raw:
            root = Path(raw)
            history = chrono.build_mutation_reversion_repo(root)

            self.assertNotEqual(history.freeze, history.intermediate)
            self.assertNotEqual(history.intermediate, history.candidate)
            self.assertEqual(history.freeze_closure, history.candidate_closure)
            self.assertNotEqual(
                history.freeze_closure, history.intermediate_closure
            )

            path = chrono.ancestry_commits(
                history.repo, history.freeze, history.candidate
            )
            self.assertGreaterEqual(len(path), 3)
            self.assertEqual(path[0], history.freeze)
            self.assertEqual(path[-1], history.candidate)
            self.assertIn(history.intermediate, path)

            self.assertTrue(
                chrono.endpoint_only_freeze_ok(
                    history.repo, history.freeze, history.candidate
                ),
                "endpoint-only must pass when candidate restores freeze identity",
            )
            self.assertFalse(
                chrono.full_history_freeze_ok(
                    history.repo, history.freeze, history.candidate
                ),
                "full history must reject intermediate mutation before reversion",
            )
            self.assertEqual(
                chrono.drifted_commits(
                    history.repo, history.freeze, history.candidate
                ),
                [history.intermediate],
            )

    def test_demonstrate_mutation_reversion_api(self) -> None:
        result = chrono.demonstrate_mutation_reversion()
        self.assertIs(result["authentication_claim"], False)
        self.assertIs(result["content_digest_cryptographic"], False)
        self.assertIs(result["imported_capture_authenticated"], False)
        self.assertTrue(result["endpoint_only_ok"])
        self.assertFalse(result["full_history_ok"])
        self.assertEqual(len(result["ancestry_path"]), 3)
        self.assertEqual(
            result["drifted_commits"],
            [result["commits"]["intermediate"]],
        )

    def test_stable_history_passes_full_history(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tgrad-cycle-chronology-stable-") as raw:
            root = Path(raw)
            repo = root / "stable"
            chrono._init_repo(repo)
            chrono._write_closure(repo, "stable-closure")
            a = chrono._commit(repo, "a")
            chrono._git(repo, "commit", "--allow-empty", "-m", "b")
            b = chrono._git(repo, "rev-parse", "HEAD").stdout.strip()
            chrono._git(repo, "commit", "--allow-empty", "-m", "c")
            c = chrono._git(repo, "rev-parse", "HEAD").stdout.strip()
            self.assertTrue(chrono.endpoint_only_freeze_ok(repo, a, c))
            self.assertTrue(chrono.full_history_freeze_ok(repo, a, c))
            self.assertEqual(chrono.drifted_commits(repo, a, c), [])
            self.assertEqual(len(chrono.ancestry_commits(repo, a, c)), 3)
            self.assertEqual(b, chrono.ancestry_commits(repo, a, c)[1])

    def test_ancestry_path_excludes_unrelated_merge_side(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tgrad-cycle-chronology-merge-") as raw:
            history = chrono.build_merge_with_unrelated_side_repo(Path(raw))
            chrono.ensure_freeze_is_ancestor(
                history.repo, history.freeze, history.candidate
            )

            naive = chrono.reachable_without_ancestry_path(
                history.repo, history.freeze, history.candidate
            )
            path = chrono.ancestry_commits(
                history.repo, history.freeze, history.candidate
            )

            self.assertIn(history.side, naive)
            self.assertNotIn(
                history.side,
                path,
                "unrelated merge-reachable side must be excluded by --ancestry-path",
            )
            self.assertIn(history.freeze, path)
            self.assertIn(history.mid, path)
            self.assertIn(history.candidate, path)

    def test_non_ancestor_freeze_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tgrad-cycle-chronology-na-") as raw:
            history = chrono.build_mutation_reversion_repo(Path(raw))
            with self.assertRaisesRegex(RuntimeError, "not an ancestor"):
                chrono.ancestry_commits(
                    history.repo, history.candidate, history.freeze
                )

    def test_documents_role_separation(self) -> None:
        source = _CHECKER.read_text(encoding="utf-8")
        self.assertIn("is NOT\nauthentication of either artifact", source)
        self.assertIn("read-only against the Tgrad", source)
        self.assertIn("tempfile", source)
        self.assertIn("--ancestry-path", source)
        self.assertIn("merge-base", source)


if __name__ == "__main__":
    unittest.main()
