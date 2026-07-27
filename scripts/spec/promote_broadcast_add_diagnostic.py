#!/usr/bin/env python3
"""Promote a replay-valid broadcast-add diagnostic blocker, never a baseline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.spec import observe_broadcast_add as observer

DEFAULT_DESTINATION = REPO / "fixtures/requirements/observations"


def promote(source: Path, destination_root: Path) -> Path:
    source = source.resolve()
    document = json.loads(source.read_text(encoding="utf-8"))
    if document.get("result_kind") != "diagnostic_blocker":
        raise RuntimeError("only a diagnostic blocker may use this promoter")
    if document.get("inference_policy", {}).get("baseline_eligible") is not False:
        raise RuntimeError("diagnostic does not explicitly forbid baseline use")
    if document.get("evidence_id") != observer.computed_evidence_id(document):
        raise RuntimeError("diagnostic evidence id is invalid")
    _, _, manifest = observer.validate_v2_definition()
    observer.replay_observation(source, document, manifest)
    destination = destination_root.resolve() / document["evidence_id"]
    for ref in document["artifacts"]:
        raw = observer.read_bound_artifact(source, ref)
        observer.write_once(destination / ref["path"], raw)
    observer.write_once(destination / "diagnostic.json", source.read_bytes())
    copied = json.loads((destination / "diagnostic.json").read_text())
    observer.replay_observation(destination / "diagnostic.json", copied, manifest)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args()
    destination = promote(args.source, args.destination)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
