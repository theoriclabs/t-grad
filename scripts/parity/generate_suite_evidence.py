#!/usr/bin/env python3
"""Generate Lean execution facts from a replay-validated observation pair."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.parity import promote_suite_observations as promotion
from scripts.parity import run_upstream_suite as observer

DEFAULT_OUTPUT = REPO / "Tgrad" / "Evidence" / "SuiteGenerated.lean"

PHASES = {
    "collection": ".collection",
    "execution": ".execution",
    "setup_teardown": ".setupTeardown",
    "environment": ".environment",
    "oracle": ".oracle",
    "mixed": ".mixed",
    "harness": ".harness",
    "no_tests": ".noTests",
}
OUTCOMES = {
    "passed": ".passed",
    "blocked_product_surface": ".blockedProductSurface",
    "blocked_environment": ".blockedEnvironment",
    "nonconforming": ".nonconforming",
    "unobserved_environment": ".unobservedEnvironment",
    "unobserved_upstream": ".unobservedUpstream",
    "mixed": ".mixed",
    "collection_mismatch": ".collectionMismatch",
    "collection_error": ".collectionError",
    "timeout": ".timeout",
    "empty": ".empty",
    "verifier_error": ".verifierError",
}


def q(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def opt(value: str | None) -> str:
    return "none" if value is None else f"some {q(value)}"


def string_list(values: list[str]) -> str:
    return "[" + ", ".join(q(value) for value in values) + "]"


def render_counts(counts: dict) -> str:
    return (
        "{ collected := %d, passed := %d, failed := %d, skipped := %d, "
        "xfailed := %d, xpassed := %d, errors := %d }"
        % tuple(counts.get(key, 0) for key in (
            "collected", "passed", "failed", "skipped", "xfailed",
            "xpassed", "errors",
        ))
    )


def render_aggregate(aggregate: dict) -> str:
    return (
        "{ files := %d, passed := %d, failed := %d, skipped := %d, "
        "xfailed := %d, xpassed := %d, errors := %d }"
        % tuple(aggregate.get(key, 0) for key in (
            "files", "tests_passed", "tests_failed", "tests_skipped",
            "tests_xfailed", "tests_xpassed", "tests_errors",
        ))
    )


def render_file(cell: dict) -> str:
    try:
        phase = PHASES[cell["phase"]]
        outcome = OUTCOMES[cell["outcome"]]
    except KeyError as error:
        raise RuntimeError(f"unmodelled suite fact: {error}") from error
    return "".join([
        "{ path := " + q(cell["file"]),
        f", phase := {phase}",
        f", outcome := {outcome}",
        ", reasonCodes := " + string_list(cell.get("reason_codes", [])),
        ", counts := " + render_counts(cell["counts"]),
        f", nodeIdCount := {cell['nodeid_count']}",
        f", caseCount := {cell['case_count']}",
        ", nodeIdManifestHash := " + q(cell["nodeid_manifest_sha256"]),
        ", collectionCaseManifestHash := " +
        q(cell["collection_case_manifest_sha256"]),
        ", terminalCaseManifestHash := " + q(cell["case_manifest_sha256"]),
        ", caseOutcomeManifestHash := " +
        q(cell["case_outcome_manifest_sha256"]),
        ", sourceHash := " + q(cell["source_sha256"]),
        ", reportHash := " + q(cell["report_sha256"]) + " }",
    ])


def render_oracle(value: dict | None) -> str:
    if value is None:
        return "none"
    return "\n".join([
        "some",
        "      { upstreamEligible := %d" % value["upstream_eligible_case_count"],
        "        upstreamUnobserved := %d" % value["upstream_unobserved_case_count"],
        "        subjectMatched := %d" % value["subject_matched_case_count"],
        "        subjectPassed := %d" % value["subject_passed_case_count"],
        "        subjectNonpassing := %d" % value["subject_nonpassing_case_count"],
        "        subjectMissing := %d" % value["subject_missing_case_count"],
        "        subjectDescriptorMismatches := %d }" %
        value["subject_descriptor_mismatch_count"],
    ])


def render_run(name: str, path: Path, document: dict) -> str:
    identity = document["identity"]
    subject = identity["subject"]
    product = subject.get("product_sources", {})
    runtime = subject.get("runtime", {})
    baseline = identity.get("upstream_baseline")
    baseline_value = "none" if baseline is None else "\n".join([
        "some",
        "      { artifactHash := " + q(baseline["artifact_sha256"]),
        "        resultId := " + q(baseline["result_id"]),
        "        runArtifactId := " + q(baseline["run_artifact_id"]) + " }",
    ])
    files = ",\n        ".join(
        render_file(cell) for cell in document["observation"]["cells"]
    )
    return "\n".join([
        f"def {name} : SuiteRunFact :=",
        f"  {{ schemaVersion := {document['schema_version']}",
        f"    subjectKind := .{document['against']}",
        f"    evidencePath := {q(path.relative_to(REPO).as_posix())}",
        f"    evidenceHash := {q(observer.digest(path.read_bytes()))}",
        f"    resultId := {q(document['result_id'])}",
        f"    runArtifactId := {q(document['run_artifact_id'])}",
        f"    scenarioId := {q(document['scenario_id'])}",
        f"    upstreamRevision := {q(identity['upstream']['revision'])}",
        f"    upstreamTree := {q(identity['upstream']['tree'])}",
        f"    subjectRevision := {q(subject['revision'])}",
        f"    subjectTree := {q(subject['tree'])}",
        f"    subjectDirty := {str(subject['dirty']).lower()}",
        f"    verifierHash := {q(identity['verifier']['runner_sha256'])}",
        f"    reporterHash := {q(identity['verifier']['reporter_sha256'])}",
        f"    contractHash := {q(identity['contract']['sha256'])}",
        f"    environmentHash := {q(identity['environment']['sha256'])}",
        "    environmentFactsHash := " +
        q(observer.digest(observer.canonical(identity["environment"]["facts"]))),
        f"    backend := {q(identity['environment']['backend']['default_device'])}",
        "    hardwareIdentityHash := " +
        q(identity["environment"]["backend"]["hardware"]["identity_sha256"]),
        f"    adapterHash := {opt(subject.get('adapter', {}).get('content_sha256'))}",
        f"    runtimeArtifactHash := {opt(runtime.get('artifact_sha256'))}",
        f"    productSourcesHash := {opt(product.get('combined_sha256'))}",
        f"    upstreamBaseline := {baseline_value}",
        f"    aggregate := {render_aggregate(document['observation']['aggregate'])}",
        "    files :=",
        "      [ " + files + "]",
        "    oracleCases := " + render_oracle(document["observation"].get("oracle_cases")) + " }",
    ])


def generated(upstream_path: Path, upstream: dict,
              tgrad_path: Path, tgrad: dict) -> str:
    return "\n\n".join([
        "import Tgrad.Evidence.Suite",
        """/- GENERATED by scripts/parity/generate_suite_evidence.py.

The source pair was replayed from its raw pytest event streams before this
file was emitted.  These are execution facts only, not requirement evidence.
-/""",
        "namespace Tgrad.Evidence.SuiteGenerated\n\nopen Tgrad.Evidence.Suite",
        render_run("upstreamRun", upstream_path, upstream),
        render_run("tgradRun", tgrad_path, tgrad),
        """theorem upstream_run_well_formed : upstreamRun.wellFormed := by native_decide
theorem tgrad_run_well_formed : tgradRun.wellFormed := by native_decide
theorem paired_sources_equal :
    upstreamRun.files.map (fun fact => (fact.path, fact.sourceHash)) =
      tgradRun.files.map (fun fact => (fact.path, fact.sourceHash)) := by
  native_decide
theorem canonical_contract_has_34_files :
    upstreamRun.files.length = 34 ∧ tgradRun.files.length = 34 := by native_decide
theorem paired_execution_boundary :
    upstreamRun.upstreamRevision = tgradRun.upstreamRevision ∧
    upstreamRun.upstreamTree = tgradRun.upstreamTree ∧
    upstreamRun.scenarioId = tgradRun.scenarioId ∧
    upstreamRun.verifierHash = tgradRun.verifierHash ∧
    upstreamRun.reporterHash = tgradRun.reporterHash ∧
    upstreamRun.contractHash = tgradRun.contractHash ∧
    upstreamRun.environmentFactsHash = tgradRun.environmentFactsHash ∧
    upstreamRun.backend = tgradRun.backend ∧
    upstreamRun.hardwareIdentityHash = tgradRun.hardwareIdentityHash := by native_decide
theorem tgrad_baseline_binds_upstream :
    tgradRun.upstreamBaseline = some
      { artifactHash := upstreamRun.evidenceHash
        resultId := upstreamRun.resultId
        runArtifactId := upstreamRun.runArtifactId } := by native_decide
theorem tgrad_product_identity_is_present :
    tgradRun.subjectDirty = false ∧
    tgradRun.adapterHash.isSome ∧
    tgradRun.runtimeArtifactHash.isSome ∧
    tgradRun.productSourcesHash.isSome := by native_decide
theorem recorded_oracle_case_accounting :
    tgradRun.oracleCases = some
      { upstreamEligible := 1145
        upstreamUnobserved := 156
        subjectMatched := 1004
        subjectPassed := 17
        subjectNonpassing := 987
        subjectMissing := 141
        subjectDescriptorMismatches := 98 } := by native_decide
theorem oracle_case_arithmetic :
    1004 + 141 = 1145 ∧ 17 + 987 = 1004 := by native_decide

end Tgrad.Evidence.SuiteGenerated""",
        "",
    ])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--tgrad", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    upstream_path = args.upstream.resolve()
    tgrad_path = args.tgrad.resolve()
    upstream, upstream_raw = promotion.validate_document(upstream_path)
    tgrad, _ = promotion.validate_document(tgrad_path)
    observer.validate_tgrad_observation(
        tgrad_path, tgrad, upstream_path, observer.DEFAULT_CHECKOUT
    )
    promotion.validate_pair(upstream, upstream_raw, tgrad)
    output = generated(upstream_path, upstream, tgrad_path, tgrad)
    target = args.output.resolve()
    if args.check:
        if not target.is_file() or target.read_text(encoding="utf-8") != output:
            raise RuntimeError(f"generated suite evidence is stale: {target}")
        print(f"checked {target}")
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_temp = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temp = Path(raw_temp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(output)
        os.replace(temp, target)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
