#!/usr/bin/env python3
"""L15.C — verdict + EXPERIMENT_RESULT.md authoring audit. Run by L15_C.sh.

Emits a single JSON document on stdout with:
- `criteria[]`: 2 entries (lean_better_evidence / honesty), each with
  `verdict: pass|fail|inconclusive` + `evidence` text.
- `lean_invariants[]`: at least 5 Lean-explicit invariants named with
  file + negative_case.
- `experiment_result_word_count`, `experiment_result_sha256`.
- `result`: yes | no | inconclusive (final computed verdict over
  L15.A's 3 + L15.B's 3 + L15.C's 2 + `len(lean_invariants) >= 5`).
"""
from __future__ import annotations
import hashlib
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MEMO = REPO / "EXPERIMENT_RESULT.md"
_evidence_raw = os.environ.get("TGRAD_EVIDENCE_DIR")
if not _evidence_raw or not Path(_evidence_raw).is_absolute():
    raise RuntimeError("TGRAD_EVIDENCE_DIR must be an explicit absolute run-owned path")
EVIDENCE = Path(_evidence_raw)


def _load(name: str) -> dict:
    p = EVIDENCE / name
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def _memo_text() -> str:
    return MEMO.read_text() if MEMO.exists() else ""


def _section_blocks(text: str) -> dict[str, str]:
    """Split memo into `## Heading -> body` blocks. Returns the body string
    for each `## ...` header. Stops at the next `## ` line."""
    blocks: dict[str, str] = {}
    parts = re.split(r"^## ", text, flags=re.M)
    for chunk in parts[1:]:
        head, *rest = chunk.split("\n", 1)
        body = rest[0] if rest else ""
        blocks[head.strip()] = body
    return blocks


# -------- Criterion 7: Lean-better evidence ---------------------------

def gather_lean_invariants() -> list[dict]:
    """The invariants are authored in EXPERIMENT_RESULT.md's
    "Where Lean Helped" section; this function parses the section into
    structured entries. Each numbered item there has a bold name + a
    `File:` line + a `Negative case:` line.
    """
    blocks = _section_blocks(_memo_text())
    helped = blocks.get("Where Lean Helped", "")
    if not helped:
        return []
    items: list[dict] = []
    # Match: `N. **Name.**` then a `- File: path` and `- Negative case: ...`
    pattern = re.compile(
        r"^\d+\.\s+\*\*([^*]+)\*\*[^\n]*\n"
        r"((?:[ \t]+- [^\n]+\n?)+)",
        re.M,
    )
    for m in pattern.finditer(helped):
        name = m.group(1).strip().rstrip(".")
        body = m.group(2)
        file_match = re.search(r"-\s+File:\s+`([^`]+)`", body)
        neg_match = re.search(r"-\s+Negative case:\s+(.+?)(?=\n[ \t]*- |\n\n|\Z)",
                              body, re.S)
        items.append({
            "name": name,
            "file": file_match.group(1) if file_match else "",
            "negative_case": neg_match.group(1).strip().replace("\n", " ") if neg_match else "",
        })
    return items


def check_lean_better_evidence(lean_invariants: list[dict]) -> dict:
    n = len(lean_invariants)
    have_files = sum(1 for inv in lean_invariants if inv["file"])
    have_neg = sum(1 for inv in lean_invariants if inv["negative_case"])
    verdict = "pass" if (n >= 5 and have_files == n and have_neg == n) else "fail"
    return {
        "criterion": "lean_better_evidence",
        "verdict": verdict,
        "artifact_paths": ["Tgrad/EXPERIMENT_RESULT.md"],
        "evidence": (
            f"lean_invariants_named={n}/>=5, "
            f"with_file={have_files}/{n}, with_negative_case={have_neg}/{n}"
        ),
    }


# -------- Criterion 8: Honesty ----------------------------------------

REQUIRED_NOT_CLAIMED = [
    "full tinygrad replacement",
    "arbitrary dtypes",  # OR "bf16 only" — checked below
    "autograd",
    "full beam",
]
FORBIDDEN_PHRASES = [
    r"\bfully proves equivalence\b",
    r"\bcomplete proof\b",
    r"\ball dtypes\b",
    r"\bfull tinygrad replacement\b",
]


def check_honesty() -> dict:
    text = _memo_text().lower()
    blocks = _section_blocks(_memo_text())
    not_claimed_body = blocks.get("Not Claimed", "").lower()
    scope_body = blocks.get("Scope", "").lower()

    # The phrases must appear in the "Not Claimed" section (or scope).
    not_claimed_hits = sum(1 for phrase in REQUIRED_NOT_CLAIMED
                           if phrase in not_claimed_body
                              or (phrase == "arbitrary dtypes" and ("bf16" in scope_body or "bf16 only" in not_claimed_body))
                              or (phrase == "full beam" and ("beam" in not_claimed_body or "beam" in scope_body)))

    # No over-claim phrases ANYWHERE.
    overclaim_hits = 0
    overclaim_details: list[str] = []
    for pat in FORBIDDEN_PHRASES:
        for m in re.finditer(pat, text):
            # Allow the literal "full tinygrad replacement" if it appears
            # in the Not Claimed block.
            phrase = m.group(0)
            start = m.start()
            # Find which block this occurrence is in.
            in_not_claimed = (
                "## Not Claimed" in _memo_text()[:start + 200]
                and "## " not in _memo_text()[start:start + 200].split("Not Claimed", 1)[-1]
            )
            if phrase == "full tinygrad replacement" and in_not_claimed:
                continue
            overclaim_hits += 1
            overclaim_details.append(phrase)

    # Slice is named precisely.
    slice_named = "bf16" in text and ("metal" in text or "apple" in text)

    verdict = "pass" if (
        not_claimed_hits >= 4 and overclaim_hits == 0 and slice_named
    ) else "fail"
    return {
        "criterion": "honesty",
        "verdict": verdict,
        "artifact_paths": ["Tgrad/EXPERIMENT_RESULT.md"],
        "evidence": (
            f"not_claimed_required={not_claimed_hits}/4, "
            f"overclaim_hits={overclaim_hits}/0 {overclaim_details}, "
            f"slice_named_precisely={slice_named}"
        ),
    }


# -------- Verdict computation -----------------------------------------

def compute_verdict(criteria_c: list[dict], lean_invariants: list[dict]) -> str:
    # L15.A + L15.B criteria must already be pass (the gate runs after both).
    l15a = _load("L15_A.json")
    l15b = _load("L15_B.json")
    a_pass = sum(1 for c in l15a.get("criteria", []) if c.get("verdict") == "pass")
    b_pass = sum(1 for c in l15b.get("criteria", []) if c.get("verdict") == "pass")
    c_pass = sum(1 for c in criteria_c if c.get("verdict") == "pass")
    total_pass = a_pass + b_pass + c_pass
    have_invariants = len(lean_invariants) >= 5
    if total_pass == 8 and have_invariants:
        return "yes"
    if any(c.get("verdict") == "fail" for c in (criteria_c + l15a.get("criteria", []) + l15b.get("criteria", []))):
        return "no"
    return "inconclusive"


def main() -> int:
    lean_invariants = gather_lean_invariants()
    criteria = [
        check_lean_better_evidence(lean_invariants),
        check_honesty(),
    ]
    memo_text = _memo_text()
    word_count = len(memo_text.split()) if memo_text else 0
    sha = hashlib.sha256(memo_text.encode()).hexdigest() if memo_text else ""
    out = {
        "criteria": criteria,
        "lean_invariants": lean_invariants,
        "experiment_result_word_count": word_count,
        "experiment_result_sha256": sha,
        "result": compute_verdict(criteria, lean_invariants),
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
