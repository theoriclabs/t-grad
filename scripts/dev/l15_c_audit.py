#!/usr/bin/env python3
"""L15.C — verdict + EXPERIMENT_RESULT.md authoring audit. Run by L15_C.sh.

Emits a single JSON document on stdout with:
- `criteria[]`: 2 entries (lean_better_evidence / honesty), each with
  `verdict: pass|fail|inconclusive` + `evidence` text.
- `lean_invariants[]`: at least 5 Lean-explicit invariants named with
  file + negative_case.
- `experiment_result_word_count`, `experiment_result_sha256`.
- `computed_answer`: yes | no | inconclusive (the eight-criterion answer).
- `promoted_result`: yes | no | inconclusive (parsed only from the memo's
  Verdict section).
- `memo_contract`: section, declaration, and promotion-coherence evidence.
"""
from __future__ import annotations
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MEMO = REPO / "EXPERIMENT_RESULT.md"


def _load(name: str) -> dict:
    p = REPO / "fixtures" / "gate_evidence" / name
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def _memo_text() -> str:
    return MEMO.read_text() if MEMO.exists() else ""


REQUIRED_NON_EVIDENCE_SECTIONS = (
    "Verdict",
    "Scope",
    "Where Lean Helped",
    "Where Lean Did Not Yet Help",
    "Performance Interpretation",
    "Not Claimed",
    "Next Move",
)
EVIDENCE_SECTIONS = (
    "Evidence",
    "Historical Evidence",
    "Mixed Historical Evidence",
)
PROMOTED_RESULTS = ("yes", "no", "inconclusive")


def _section_entries(text: str) -> list[dict]:
    """Return every level-two section with source offsets and body."""
    matches = list(re.finditer(r"^## ([^\n]+)[ \t]*$", text, re.M))
    entries: list[dict] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        entries.append({
            "name": match.group(1).strip(),
            "start": match.start(),
            "body_start": match.end(),
            "end": end,
            "body": text[match.end():end].lstrip("\n"),
        })
    return entries


def _section_blocks(text: str) -> dict[str, str]:
    """Split memo into `## Heading -> body` blocks. Returns the body string
    for each `## ...` header. Stops at the next `## ` line."""
    return {entry["name"]: entry["body"] for entry in _section_entries(text)}


def analyze_memo_contract(text: str) -> dict:
    """Parse the promoted result and evidence provenance from the memo.

    The declaration is intentionally not a whole-file substring search: one
    exact declaration must occur in the one Verdict section, and declaration-
    like lines elsewhere make the artifact ambiguous and therefore invalid.
    """
    entries = _section_entries(text)
    counts = Counter(entry["name"] for entry in entries)
    errors: list[str] = []

    for name in REQUIRED_NON_EVIDENCE_SECTIONS:
        if counts[name] != 1:
            errors.append(f"section {name!r} count={counts[name]} expected=1")

    evidence_names = [name for name in EVIDENCE_SECTIONS if counts[name] > 0]
    evidence_count = sum(counts[name] for name in EVIDENCE_SECTIONS)
    if evidence_count != 1:
        errors.append(
            "evidence section count="
            f"{evidence_count} expected=1 among {list(EVIDENCE_SECTIONS)!r}"
        )
    evidence_section = evidence_names[0] if evidence_count == 1 else None
    evidence_kind = (
        "current" if evidence_section == "Evidence"
        else "historical" if evidence_section in {
            "Historical Evidence", "Mixed Historical Evidence"
        }
        else None
    )

    declaration_like = list(re.finditer(
        r"^[ \t]*current promoted result[^\n]*$", text, re.M | re.I
    ))
    promoted_result = None
    declaration_section = None
    if len(declaration_like) != 1:
        errors.append(
            f"current promoted result declaration count={len(declaration_like)} expected=1"
        )
    else:
        declaration = declaration_like[0]
        line = declaration.group(0).strip()
        exact = re.fullmatch(
            r"current promoted result: (yes|no|inconclusive)", line
        )
        if exact is None:
            errors.append(f"malformed current promoted result declaration: {line!r}")
        else:
            promoted_result = exact.group(1)
        containing = [
            entry for entry in entries
            if entry["body_start"] <= declaration.start() < entry["end"]
        ]
        declaration_section = containing[0]["name"] if len(containing) == 1 else None
        if declaration_section != "Verdict":
            errors.append(
                "current promoted result declaration is in "
                f"{declaration_section!r}, expected 'Verdict'"
            )

    coherence_ok = not (
        promoted_result == "yes" and evidence_kind == "historical"
    )
    if not coherence_ok:
        errors.append(
            "promoted result 'yes' is incoherent with historical-only evidence"
        )

    return {
        "valid": not errors,
        "errors": errors,
        "section_counts": {
            name: counts[name]
            for name in (*REQUIRED_NON_EVIDENCE_SECTIONS, *EVIDENCE_SECTIONS)
        },
        "evidence_section": evidence_section,
        "evidence_kind": evidence_kind,
        "promoted_result": promoted_result,
        "declaration_section": declaration_section,
        "coherence_ok": coherence_ok,
    }


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

def compute_answer(criteria_c: list[dict], lean_invariants: list[dict]) -> str:
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


def self_test() -> int:
    """Falsify parser scope and promotion coherence without gate evidence."""
    common = """# Memo
## Verdict
{declaration}
## Scope
scope
## {evidence}
evidence
## Where Lean Helped
helped
## Where Lean Did Not Yet Help
not yet
## Performance Interpretation
perf
## Not Claimed
not claimed
## Next Move
next
"""
    honest = analyze_memo_contract(common.format(
        declaration="current promoted result: inconclusive",
        evidence="Mixed Historical Evidence",
    ))
    false_promotion = analyze_memo_contract(common.format(
        declaration="current promoted result: yes",
        evidence="Mixed Historical Evidence",
    ))
    current_yes = analyze_memo_contract(common.format(
        declaration="current promoted result: yes",
        evidence="Evidence",
    ))
    duplicate = analyze_memo_contract(
        common.format(
            declaration="current promoted result: inconclusive",
            evidence="Mixed Historical Evidence",
        ) + "\ncurrent promoted result: yes\n"
    )
    if not (
        honest["valid"]
        and honest["promoted_result"] == "inconclusive"
        and honest["evidence_kind"] == "historical"
        and not false_promotion["valid"]
        and not false_promotion["coherence_ok"]
        and current_yes["valid"]
        and not duplicate["valid"]
    ):
        print("l15_c_audit_self_test: FAIL", file=sys.stderr)
        print(json.dumps({
            "honest": honest,
            "false_promotion": false_promotion,
            "current_yes": current_yes,
            "duplicate": duplicate,
        }, indent=2), file=sys.stderr)
        return 1
    print("l15_c_audit_self_test: pass")
    print("l15_c_self_test_promoted_result: inconclusive")
    print("l15_c_self_test_evidence_kind: historical")
    return 0


def main() -> int:
    if sys.argv[1:] == ["--self-test"]:
        return self_test()
    if sys.argv[1:]:
        print("usage: l15_c_audit.py [--self-test]", file=sys.stderr)
        return 2
    lean_invariants = gather_lean_invariants()
    criteria = [
        check_lean_better_evidence(lean_invariants),
        check_honesty(),
    ]
    memo_text = _memo_text()
    memo_contract = analyze_memo_contract(memo_text)
    word_count = len(memo_text.split()) if memo_text else 0
    sha = hashlib.sha256(memo_text.encode()).hexdigest() if memo_text else ""
    out = {
        "criteria": criteria,
        "lean_invariants": lean_invariants,
        "memo_contract": memo_contract,
        "experiment_result_word_count": word_count,
        "experiment_result_sha256": sha,
        "computed_answer": compute_answer(criteria, lean_invariants),
        "promoted_result": memo_contract["promoted_result"],
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
