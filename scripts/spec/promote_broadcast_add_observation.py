#!/usr/bin/env python3
"""Promote a replay-valid V4 observation into content-addressed evidence."""
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
    if document.get("result_kind") != "observation":
        raise RuntimeError("only a valid observation may use this promoter")
    if document.get("evidence_id") != observer.computed_evidence_id(document):
        raise RuntimeError("observation evidence id is invalid")
    _, _, manifest = observer.validate_v4_definition()
    observer.replay_observation(source, document, manifest)
    if document.get("against") == "upstream":
        calibrations = document.get("calibrations")
        if not isinstance(calibrations, list) or len(calibrations) != 8 or any(
            item.get("outcome") != "validator_rejected_mutant"
            for item in calibrations
        ):
            raise RuntimeError("upstream observation is not fully calibrated")
    elif document.get("against") == "tgrad":
        if document.get("calibrations") is not None or not isinstance(
            document.get("upstream_comparison"), list
        ):
            raise RuntimeError("Tgrad observation lacks its upstream comparison")
    else:
        raise RuntimeError("unknown observation subject")
    destination = destination_root.resolve() / document["evidence_id"]
    for ref in document["artifacts"]:
        raw = observer.read_bound_artifact(source, ref)
        observer.write_once(destination / ref["path"], raw)
    observer.write_once(destination / "observation.json", source.read_bytes())
    copied = json.loads((destination / "observation.json").read_text())
    observer.replay_observation(destination / "observation.json", copied, manifest)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args()
    print(promote(args.source, args.destination))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
