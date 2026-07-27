"""Machine-readable pytest event reporter for parity observations.

The suite observer sets ``TGRAD_PYTEST_REPORT`` to a fresh path for each test
file.  This plugin records phase/outcome facts only; human diagnostics remain
in pytest's normalized stdout/stderr artifact.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from _pytest._io.saferepr import saferepr


REPORT_ENV = "TGRAD_PYTEST_REPORT"


def _emit(event: dict) -> None:
    destination = os.environ.get(REPORT_ENV)
    if not destination:
        return
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def pytest_collection_finish(session) -> None:
    descriptors = []
    for item in session.items:
        callspec = getattr(item, "callspec", None)
        descriptors.append({
            "nodeid": item.nodeid,
            "parameters": {
                key: saferepr(value)
                for key, value in sorted(
                    getattr(callspec, "params", {}).items()
                )
            },
            "fixtures": sorted(set(getattr(item, "fixturenames", []))),
            "marks": sorted([
                {
                    "name": mark.name,
                    "args": [saferepr(value) for value in mark.args],
                    "kwargs": {
                        key: saferepr(value)
                        for key, value in sorted(mark.kwargs.items())
                    },
                }
                for mark in item.iter_markers()
            ], key=lambda mark: json.dumps(mark, sort_keys=True)),
        })
    _emit({
        "event": "collection_finish",
        "count": len(session.items),
        "nodeids": sorted(item.nodeid for item in session.items),
        "cases": sorted(descriptors, key=lambda item: item["nodeid"]),
    })


def pytest_collectreport(report) -> None:
    if report.failed:
        _emit({
            "event": "collection_error",
            "nodeid": report.nodeid,
            "outcome": report.outcome,
        })


def pytest_runtest_logreport(report) -> None:
    event = {
        "event": "test_report",
        "nodeid": report.nodeid,
        "phase": report.when,
        "outcome": report.outcome,
    }
    if hasattr(report, "wasxfail"):
        event["wasxfail"] = str(report.wasxfail)
    context = getattr(report, "context", None)
    if context is not None:
        event["subtest"] = {
            "msg": None if context.msg is None else str(context.msg),
            "kwargs": dict(context.kwargs),
        }
    if report.failed:
        # Keep classification machine-readable without making the reporter an
        # oracle for product semantics.  The full diagnostic is retained by
        # the suite observer as a content-addressed artifact.
        text = getattr(report, "longreprtext", "") or str(report.longrepr)
        matches = re.findall(r"(?:^|\n)E\s+([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception))(?::|$)", text)
        if matches:
            event["failure_type"] = matches[-1].rsplit(".", 1)[-1]
    _emit(event)


def pytest_sessionfinish(session, exitstatus) -> None:
    _emit({"event": "session_finish", "exitstatus": int(exitstatus)})
