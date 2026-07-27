"""Machine-readable pytest event reporter for parity observations.

The suite observer sets ``TGRAD_PYTEST_REPORT`` to a fresh path for each test
file.  This plugin records phase/outcome facts only; human diagnostics remain
in pytest's normalized stdout/stderr artifact.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


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
    _emit({
        "event": "collection_finish",
        "count": len(session.items),
        "nodeids": sorted(item.nodeid for item in session.items),
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
    _emit(event)


def pytest_sessionfinish(session, exitstatus) -> None:
    _emit({"event": "session_finish", "exitstatus": int(exitstatus)})
