#!/usr/bin/env python3
"""Independent Git full-history freeze-integrity checker.

Packet: mechanics.cycle-chronology-v1

Role separation (explicit):
  * Lean (`Tgrad.Contract.Chronology`) validates typed judging-input graphs,
    ancestry manifests, and cycle registries as a schema substrate.
  * This Python module independently exercises Git ancestry mechanics on a
    synthetic temporary repository. It is read-only against the Tgrad repo.

Agreement between Lean native_decide checks and this Python exercise is NOT
authentication of either artifact. Both are agent-authorable. Cryptographic
binding of ContentDigest tokens is not claimed. The imported capture remains
unauthenticated until a grounded extractor exists.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


CLOSURE_FILE = "judging_closure.txt"


@dataclass(frozen=True)
class SyntheticHistory:
    """Linear freeze → mutate → revert history with three commits."""

    repo: Path
    freeze: str
    intermediate: str
    candidate: str
    freeze_closure: str
    intermediate_closure: str
    candidate_closure: str


@dataclass(frozen=True)
class MergeSideHistory:
    """Freeze→candidate path with an unrelated side branch merged in."""

    repo: Path
    root: str
    freeze: str
    mid: str
    side: str
    candidate: str


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed: {result.stderr.strip() or result.stdout.strip()}"
        )
    return result


def _write_closure(repo: Path, value: str) -> None:
    (repo / CLOSURE_FILE).write_text(value + "\n", encoding="utf-8")


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", CLOSURE_FILE)
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


def _init_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=False)
    _git(repo, "init")
    _git(repo, "config", "user.email", "chronology-check@example.com")
    _git(repo, "config", "user.name", "Chronology Checker")


def build_mutation_reversion_repo(root: Path) -> SyntheticHistory:
    """Create a temp Git repo: freeze C1, mutate C2, revert to C1.

    Uses tempfile-provided roots only — never fixed /tmp/tgrad_* paths.
    """
    repo = root / "synthetic-chronology"
    _init_repo(repo)

    freeze_closure = "closure-identity-v1"
    mutated_closure = "closure-identity-MUTATED"
    candidate_closure = freeze_closure

    _write_closure(repo, freeze_closure)
    freeze = _commit(repo, "freeze judging inputs")

    _write_closure(repo, mutated_closure)
    intermediate = _commit(repo, "mutate judging inputs")

    _write_closure(repo, candidate_closure)
    candidate = _commit(repo, "revert judging inputs")

    return SyntheticHistory(
        repo=repo,
        freeze=freeze,
        intermediate=intermediate,
        candidate=candidate,
        freeze_closure=freeze_closure,
        intermediate_closure=mutated_closure,
        candidate_closure=candidate_closure,
    )


def build_merge_with_unrelated_side_repo(root: Path) -> MergeSideHistory:
    """History where an unrelated side commit is reachable but off the ancestry path.

    Topology::

        root -- freeze -- mid ------\\
          \\                          candidate (merge)
           side ---------------------/

    ``freeze..candidate`` without ``--ancestry-path`` includes ``side``.
    With ``--ancestry-path``, ``side`` is excluded.
    """
    repo = root / "synthetic-merge-side"
    _init_repo(repo)

    _write_closure(repo, "root-closure")
    root_commit = _commit(repo, "root")

    _write_closure(repo, "freeze-closure")
    freeze = _commit(repo, "freeze")

    _write_closure(repo, "mid-closure")
    mid = _commit(repo, "mid")

    _git(repo, "checkout", "-b", "side-branch", root_commit)
    _write_closure(repo, "side-closure")
    side = _commit(repo, "unrelated side")

    _git(repo, "checkout", "-b", "main-line", mid)
    # Merge unrelated side into the candidate tip. Closure content may conflict;
    # resolve without changing the commit topology under test.
    merge = _git(repo, "merge", side, "-m", "merge unrelated side", check=False)
    if merge.returncode != 0:
        _write_closure(repo, "candidate-after-merge")
        _git(repo, "add", CLOSURE_FILE)
        _git(repo, "commit", "-m", "merge unrelated side")
    candidate = _git(repo, "rev-parse", "HEAD").stdout.strip()

    return MergeSideHistory(
        repo=repo,
        root=root_commit,
        freeze=freeze,
        mid=mid,
        side=side,
        candidate=candidate,
    )


def closure_at(repo: Path, commit: str) -> str:
    raw = _git(repo, "show", f"{commit}:{CLOSURE_FILE}").stdout
    return raw.strip()


def ensure_freeze_is_ancestor(repo: Path, freeze: str, candidate: str) -> None:
    result = _git(
        repo, "merge-base", "--is-ancestor", freeze, candidate, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"freeze {freeze} is not an ancestor of candidate {candidate}"
        )


def reachable_without_ancestry_path(
    repo: Path, freeze: str, candidate: str
) -> list[str]:
    """Naive range listing (includes merge-reachable unrelated commits)."""
    ensure_freeze_is_ancestor(repo, freeze, candidate)
    out = _git(
        repo,
        "rev-list",
        "--reverse",
        f"{freeze}..{candidate}",
    ).stdout.splitlines()
    return [freeze, *[c for c in out if c]]


def ancestry_commits(repo: Path, freeze: str, candidate: str) -> list[str]:
    """Commits on the ancestry path from candidate back to freeze, oldest-first.

    Requires ``merge-base --is-ancestor`` and uses ``rev-list --ancestry-path``.
    """
    ensure_freeze_is_ancestor(repo, freeze, candidate)
    out = _git(
        repo,
        "rev-list",
        "--reverse",
        "--ancestry-path",
        f"{freeze}..{candidate}",
    ).stdout.splitlines()
    return [freeze, *[c for c in out if c]]


def endpoint_only_freeze_ok(repo: Path, freeze: str, candidate: str) -> bool:
    """Naive check: only compare freeze and candidate closure identities."""
    return closure_at(repo, freeze) == closure_at(repo, candidate)


def full_history_freeze_ok(repo: Path, freeze: str, candidate: str) -> bool:
    """Full-history check: every commit on the ancestry path must match freeze."""
    expected = closure_at(repo, freeze)
    for commit in ancestry_commits(repo, freeze, candidate):
        if closure_at(repo, commit) != expected:
            return False
    return True


def drifted_commits(repo: Path, freeze: str, candidate: str) -> list[str]:
    expected = closure_at(repo, freeze)
    return [
        commit
        for commit in ancestry_commits(repo, freeze, candidate)
        if closure_at(repo, commit) != expected
    ]


def demonstrate_mutation_reversion(root: Path | None = None) -> dict:
    """Prove endpoint-only misses mutation/reversion; full history catches it.

    Read-only against the Tgrad checkout: all Git mutation happens under
    ``tempfile.TemporaryDirectory`` (or a caller-provided temp root).
    """
    if root is None:
        with tempfile.TemporaryDirectory(prefix="tgrad-cycle-chronology-") as tmp:
            return demonstrate_mutation_reversion(Path(tmp))

    history = build_mutation_reversion_repo(root)
    endpoint_ok = endpoint_only_freeze_ok(
        history.repo, history.freeze, history.candidate
    )
    full_ok = full_history_freeze_ok(
        history.repo, history.freeze, history.candidate
    )
    drifts = drifted_commits(history.repo, history.freeze, history.candidate)
    path = ancestry_commits(history.repo, history.freeze, history.candidate)

    if len(path) < 3:
        raise RuntimeError("synthetic history must contain at least three commits")
    if history.intermediate not in path:
        raise RuntimeError("intermediate commit missing from ancestry path")
    if not endpoint_ok:
        raise RuntimeError(
            "endpoint-only check unexpectedly failed; mutation/reversion fixture broken"
        )
    if full_ok:
        raise RuntimeError(
            "full-history check unexpectedly passed; mutation/reversion not detected"
        )
    if drifts != [history.intermediate]:
        raise RuntimeError(
            f"expected sole drift at intermediate commit, got {drifts!r}"
        )

    return {
        "role": "python-git-ancestry-mechanics",
        "authentication_claim": False,
        "content_digest_cryptographic": False,
        "imported_capture_authenticated": False,
        "repo_root_policy": "tempfile-only; read-only against Tgrad checkout",
        "commits": {
            "freeze": history.freeze,
            "intermediate": history.intermediate,
            "candidate": history.candidate,
        },
        "closures": {
            "freeze": history.freeze_closure,
            "intermediate": history.intermediate_closure,
            "candidate": history.candidate_closure,
        },
        "ancestry_path": path,
        "endpoint_only_ok": endpoint_ok,
        "full_history_ok": full_ok,
        "drifted_commits": drifts,
        "lesson": (
            "endpoint-only comparison misses mutation-then-reversion; "
            "full-history checking rejects the intermediate drift"
        ),
        "closure_file_sha256_at_freeze": hashlib.sha256(
            (history.freeze_closure + "\n").encode()
        ).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Exercise full-history vs endpoint-only freeze checking on a "
            "synthetic temporary Git repository. Does not modify the Tgrad repo."
        )
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the demonstration result as JSON on stdout",
    )
    args = parser.parse_args()
    result = demonstrate_mutation_reversion()
    if args.json:
        import json

        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(result["lesson"])
        print(
            f"endpoint_only_ok={result['endpoint_only_ok']} "
            f"full_history_ok={result['full_history_ok']} "
            f"drifted={result['drifted_commits']}"
        )
        print(
            "Note: this Python check does not authenticate Lean chronology "
            "results; roles remain independent."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
