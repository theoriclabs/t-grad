#!/usr/bin/env python3
"""Derive a release-performance decision from retained paired-runtime runs.

The certificate names completed artifact sets; it does not author run/session
counts or a verdict.  Both candidate collection and the promoted-snapshot
auditor call this module to reconstruct those facts from the raw observations.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any


SHA256_LENGTH = 64
COMPARISON_ID = "tgrad_prepared_vs_tinygrad_tinyjit_replay"
BOUNDARY_ID = "prepared_runtime"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == SHA256_LENGTH and all(
        char in "0123456789abcdef" for char in value
    )


def load_object(path: Path, label: str, problems: list[str]) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        problems.append(f"{label}: cannot parse {path}: {error}")
        return None
    if not isinstance(value, dict):
        problems.append(f"{label}: root is not an object")
        return None
    return value


def safe_relative(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path == Path("."):
        return None
    return path


def resolve_regular(root: Path, relative: Path) -> Path | None:
    unresolved = root / relative
    if unresolved.is_symlink():
        return None
    current = unresolved.parent
    while current != root:
        if current.is_symlink():
            return None
        if current == current.parent:
            return None
        current = current.parent
    path = unresolved.resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return None
    if not path.is_file() or path.is_symlink():
        return None
    return path


def stable_seed(seed: int, *parts: Any) -> int:
    material = canonical_json([seed, *parts]).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def quantile(ordered: list[float], probability: float) -> float:
    if not ordered:
        raise ValueError("empty quantile")
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_upper(
    sessions: dict[int, list[float]], resamples: int, confidence: float, seed: int,
) -> float:
    session_ids = sorted(sessions)
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(resamples):
        means: list[float] = []
        for _ in session_ids:
            chosen = rng.choice(session_ids)
            values = sessions[chosen]
            sampled = [rng.choice(values) for _ in values]
            means.append(sum(sampled) / len(sampled))
        estimates.append(sum(means) / len(means))
    estimates.sort()
    return math.exp(quantile(estimates, 1.0 - (1.0 - confidence) / 2.0))


def parse_raw_run(
    raw_path: Path, run_id: str, configuration: dict[str, Any], problems: list[str],
    label: str,
) -> dict[str, Any] | None:
    sessions = configuration.get("sessions")
    samples = configuration.get("samples_per_session")
    resamples = configuration.get("bootstrap_resamples")
    confidence = configuration.get("confidence_level")
    analysis_seed = configuration.get("analysis_seed")
    if not all(isinstance(value, int) for value in (sessions, samples, resamples, analysis_seed)) or \
       isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or \
       sessions <= 0 or samples <= 0 or resamples <= 0 or not 0.0 < confidence < 1.0:
        problems.append(f"{label}: measurement configuration is malformed")
        return None
    pairs: dict[tuple[int, int], dict[str, int]] = {}
    try:
        lines = raw_path.read_text().splitlines()
    except OSError as error:
        problems.append(f"{label}: cannot read raw observations: {error}")
        return None
    for line_number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            problems.append(f"{label}: raw line {line_number} is invalid JSON: {error}")
            return None
        if not isinstance(row, dict) or row.get("run_id") != run_id:
            problems.append(f"{label}: raw line {line_number} has the wrong run identity")
            return None
        if not (
            row.get("record_type") == "observation" and
            row.get("phase") == "measured" and
            row.get("included_in_analysis") is True and
            row.get("comparison_id") == COMPARISON_ID
        ):
            continue
        session, sample, side, duration = (
            row.get("session_index"), row.get("sample_index"),
            row.get("side"), row.get("duration_ns"),
        )
        if not isinstance(session, int) or not isinstance(sample, int) or \
           side not in {"numerator", "denominator"} or \
           not isinstance(duration, int) or duration <= 0:
            problems.append(f"{label}: malformed measured observation at line {line_number}")
            return None
        group = pairs.setdefault((session, sample), {})
        if side in group:
            problems.append(f"{label}: duplicate {side} observation for session/sample")
            return None
        group[side] = duration
    expected = {(session, sample) for session in range(sessions) for sample in range(samples)}
    if set(pairs) != expected or any(set(pair) != {"numerator", "denominator"}
                                     for pair in pairs.values()):
        problems.append(f"{label}: raw observations do not exactly cover configured pairs")
        return None
    by_session: dict[int, list[float]] = {session: [] for session in range(sessions)}
    for (session, _sample), pair in sorted(pairs.items()):
        by_session[session].append(math.log(pair["numerator"] / pair["denominator"]))
    session_means = [sum(by_session[index]) / len(by_session[index]) for index in range(sessions)]
    mean_log = sum(session_means) / len(session_means)
    upper = bootstrap_upper(
        by_session, resamples, float(confidence),
        stable_seed(analysis_seed, COMPARISON_ID, "bootstrap"),
    )
    return {
        "session_count": sessions,
        "pair_count": len(pairs),
        "geometric_mean_ratio": math.exp(mean_log),
        "hierarchical_bootstrap_upper_ratio": upper,
    }


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def validate_release_certificate(
    certificate_path: Path, expected_source: dict[str, Any], policy_id: str,
    variance_model_path: Path, decision_rule_path: Path,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], dict[str, Any] | None, list[str]]:
    """Return certificate, retained files, derived attestation, and problems."""
    problems: list[str] = []
    root = certificate_path.resolve().parent
    document = load_object(certificate_path, "performance certificate", problems)
    if document is None:
        return None, [], None, problems
    required = {
        "schema_version", "certificate_id", "state", "source", "boundary_id",
        "evaluation_runs", "variance_model_sha256", "decision_rule_sha256",
        "content_sha256",
    }
    if set(document) != required:
        problems.append("performance certificate schema keys are not exact")
    actual_content = canonical_sha256({
        key: value for key, value in document.items() if key != "content_sha256"
    })
    if document.get("content_sha256") != actual_content:
        problems.append("performance certificate content hash mismatch")
    if document.get("schema_version") != 2 or document.get("certificate_id") != policy_id or \
       document.get("source") != expected_source or document.get("boundary_id") != BOUNDARY_ID:
        problems.append("performance certificate identity/source/boundary mismatch")
    model_digest = file_sha256(variance_model_path) if variance_model_path.is_file() else None
    rule_digest = file_sha256(decision_rule_path) if decision_rule_path.is_file() else None
    if document.get("variance_model_sha256") != model_digest:
        problems.append("performance certificate variance model is not reviewed source")
    if document.get("decision_rule_sha256") != rule_digest:
        problems.append("performance certificate decision rule is not reviewed source")
    rule = load_object(decision_rule_path, "performance decision rule", problems)
    minimum_runs = 3
    minimum_sessions = 3
    minimum_samples = 30
    minimum_resamples = 2000
    required_confidence = 0.95
    acceptance: dict[str, Any] | None = None
    if rule is not None:
        required_rule = {
            "schema_version", "policy_id", "state", "comparison_id", "boundary_id",
            "minimum_independent_calibration_runs", "minimum_independent_evaluation_runs",
            "minimum_logical_sessions_per_run", "minimum_samples_per_session",
            "minimum_bootstrap_resamples", "confidence_level", "acceptance_rule",
            "calibration_artifacts", "missing",
        }
        if set(rule) != required_rule:
            problems.append("performance decision rule schema keys are not exact")
        if rule.get("schema_version") != 1 or rule.get("policy_id") != policy_id or \
           rule.get("comparison_id") != COMPARISON_ID or \
           rule.get("boundary_id") != BOUNDARY_ID:
            problems.append("performance decision rule identity/boundary mismatch")
        declared_runs = rule.get("minimum_independent_evaluation_runs")
        declared_calibration_runs = rule.get("minimum_independent_calibration_runs")
        declared_sessions = rule.get("minimum_logical_sessions_per_run")
        declared_samples = rule.get("minimum_samples_per_session")
        declared_resamples = rule.get("minimum_bootstrap_resamples")
        declared_confidence = finite_number(rule.get("confidence_level"))
        if not isinstance(declared_calibration_runs, int) or declared_calibration_runs < 3 or \
           not isinstance(declared_runs, int) or declared_runs < 3 or \
           not isinstance(declared_sessions, int) or declared_sessions < 3:
            problems.append("performance decision rule minimums are below policy")
        else:
            minimum_runs = declared_runs
            minimum_sessions = declared_sessions
        if not isinstance(declared_samples, int) or declared_samples < 30 or \
           not isinstance(declared_resamples, int) or declared_resamples < 2000 or \
           declared_confidence != 0.95:
            problems.append("performance decision rule sampling requirements are below policy")
        else:
            minimum_samples = declared_samples
            minimum_resamples = declared_resamples
            required_confidence = declared_confidence
        if rule.get("state") != "reviewed":
            problems.append("performance decision rule has not been calibrated and reviewed")
        acceptance = rule.get("acceptance_rule")
        if not isinstance(acceptance, dict) or set(acceptance) != {
            "statistic", "operator", "threshold"
        } or acceptance.get("statistic") != \
           "maximum_run_hierarchical_bootstrap_upper_geometric_mean_ratio" or \
           acceptance.get("operator") != "<=" or \
           isinstance(acceptance.get("threshold"), bool) or \
           not isinstance(acceptance.get("threshold"), (int, float)) or \
           not math.isfinite(float(acceptance["threshold"])) or acceptance["threshold"] <= 0:
            problems.append("performance decision rule acceptance rule is unsupported")
            acceptance = None
        calibration = rule.get("calibration_artifacts")
        if not isinstance(calibration, list) or \
           not isinstance(declared_calibration_runs, int) or \
           len(calibration) < declared_calibration_runs:
            problems.append("reviewed performance decision rule has no calibration artifacts")
        elif rule.get("state") == "reviewed":
            repository = decision_rule_path.resolve().parents[2]
            seen_calibration: set[Path] = set()
            for artifact in calibration:
                if not isinstance(artifact, dict) or set(artifact) != {"path", "sha256"}:
                    problems.append("performance calibration artifact entry is malformed")
                    continue
                relative = safe_relative(artifact.get("path"))
                path = resolve_regular(repository, relative) if relative else None
                if relative in seen_calibration or path is None or \
                   file_sha256(path) != artifact.get("sha256"):
                    problems.append("performance calibration artifact does not resolve in source")
                    continue
                seen_calibration.add(relative)

    entries = document.get("evaluation_runs")
    if not isinstance(entries, list) or len(entries) < minimum_runs:
        problems.append("performance certificate has too few evaluation runs")
        entries = []
    retained: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    seen_relative: set[Path] = set()
    seen_instances: set[str] = set()
    seen_run_ids: set[str] = set()
    for index, entry in enumerate(entries):
        label = f"evaluation run {index}"
        if not isinstance(entry, dict) or set(entry) != {"completion_path", "completion_sha256"}:
            problems.append(f"{label}: entry is malformed")
            continue
        completion_relative = safe_relative(entry.get("completion_path"))
        completion = resolve_regular(root, completion_relative) if completion_relative else None
        if completion is None or file_sha256(completion) != entry.get("completion_sha256"):
            problems.append(f"{label}: completion artifact does not resolve")
            continue
        completion_doc = load_object(completion, f"{label} completion", problems)
        if completion_doc is None:
            continue
        required_completion = {"schema_version", "run_id", "raw", "summary", "artifact_set_id", "state"}
        if set(completion_doc) != required_completion or completion_doc.get("schema_version") != 3 or \
           completion_doc.get("state") != "complete" or not is_sha256(completion_doc.get("artifact_set_id")):
            problems.append(f"{label}: completion schema/state is malformed")
            continue
        identity = {key: completion_doc[key] for key in ("schema_version", "run_id", "raw", "summary")}
        if completion_doc["artifact_set_id"] != canonical_sha256(identity):
            problems.append(f"{label}: completion artifact-set identity mismatch")
        run_id = completion_doc.get("run_id")
        if not is_sha256(run_id) or run_id in seen_run_ids:
            problems.append(f"{label}: run id is malformed or duplicated")
            continue
        seen_run_ids.add(run_id)
        files: dict[str, Path] = {"completion": completion}
        for kind in ("raw", "summary"):
            descriptor = completion_doc.get(kind)
            if not isinstance(descriptor, dict) or \
               set(descriptor) != ({"file", "sha256", "observation_count"} if kind == "raw" else {"file", "sha256"}):
                problems.append(f"{label}: {kind} descriptor is malformed")
                continue
            file_relative = safe_relative(descriptor.get("file"))
            if file_relative is None or file_relative.parent != Path("."):
                problems.append(f"{label}: {kind} filename is unsafe")
                continue
            relative = completion_relative.parent / file_relative
            path = resolve_regular(root, relative)
            if path is None or file_sha256(path) != descriptor.get("sha256"):
                problems.append(f"{label}: {kind} artifact does not resolve")
                continue
            files[kind] = path
        if set(files) != {"completion", "raw", "summary"}:
            continue
        summary = load_object(files["summary"], f"{label} summary", problems)
        if summary is None:
            continue
        if summary.get("schema_version") != 3 or summary.get("run_id") != run_id or \
           summary.get("completion") != "complete" or summary.get("errors") != []:
            problems.append(f"{label}: summary identity/completion is invalid")
            continue
        instance = summary.get("run_instance")
        instance_id = instance.get("id") if isinstance(instance, dict) else None
        if not isinstance(instance_id, str) or not instance_id or instance_id in seen_instances:
            problems.append(f"{label}: run instance is absent or duplicated")
            continue
        seen_instances.add(instance_id)
        provenance = summary.get("provenance")
        tgrad = provenance.get("tgrad") if isinstance(provenance, dict) else None
        tinygrad = provenance.get("tinygrad") if isinstance(provenance, dict) else None
        tgrad_source = tgrad.get("source") if isinstance(tgrad, dict) else None
        runtime = tgrad.get("environment", {}).get("runtime_binary") if isinstance(tgrad, dict) else None
        if not isinstance(tgrad_source, dict) or \
           tgrad_source.get("commit") != expected_source.get("commit") or \
           tgrad_source.get("tree") != expected_source.get("tree") or \
           tgrad_source.get("dirty") is not False or \
           tgrad.get("revision_validation") != "clean_attributed" or \
           not isinstance(runtime, dict) or runtime.get("build_validation") != "rebuilt_from_clean_subject" or \
           runtime.get("source_commit") != expected_source.get("commit") or \
           runtime.get("source_tree") != expected_source.get("tree"):
            problems.append(f"{label}: Tgrad subject/runtime provenance mismatch")
        if not isinstance(tinygrad, dict) or tinygrad.get("revision_validation") != "pinned_clean":
            problems.append(f"{label}: tinygrad reference was not the pinned clean source")
        comparisons = summary.get("comparisons")
        comparison = next(
            (value for value in comparisons if isinstance(value, dict) and value.get("id") == COMPARISON_ID),
            None,
        ) if isinstance(comparisons, list) else None
        if not isinstance(comparison, dict) or \
           comparison.get("numerator_adapter") != "tgrad" or \
           comparison.get("numerator_boundary") != BOUNDARY_ID or \
           comparison.get("denominator_adapter") != "tinygrad" or \
           comparison.get("denominator_boundary") != "tinyjit_replay":
            problems.append(f"{label}: prepared-runtime comparison boundary mismatch")
        configuration = summary.get("configuration")
        if not isinstance(configuration, dict) or \
           not isinstance(configuration.get("sessions"), int) or \
           configuration["sessions"] < minimum_sessions or \
           not isinstance(configuration.get("samples_per_session"), int) or \
           configuration["samples_per_session"] < minimum_samples or \
           not isinstance(configuration.get("bootstrap_resamples"), int) or \
           configuration["bootstrap_resamples"] < minimum_resamples or \
           finite_number(configuration.get("confidence_level")) != required_confidence:
            problems.append(f"{label}: sampling design is below the reviewed minimum")
            continue
        derived = parse_raw_run(files["raw"], run_id, configuration, problems, label)
        if derived is None:
            continue
        analysis = summary.get("analysis", {}).get(COMPARISON_ID) \
            if isinstance(summary.get("analysis"), dict) else None
        summary_ratio = finite_number(analysis.get("geometric_mean_ratio")) \
            if isinstance(analysis, dict) else None
        interval = analysis.get("bootstrap", {}).get("geometric_mean_ratio_interval") \
            if isinstance(analysis, dict) and isinstance(analysis.get("bootstrap"), dict) \
            else None
        summary_upper = finite_number(interval[1]) \
            if isinstance(interval, list) and len(interval) == 2 else None
        if summary_ratio is None or summary_upper is None or not math.isclose(
            summary_ratio, derived["geometric_mean_ratio"], rel_tol=1e-12, abs_tol=1e-12,
        ) or not math.isclose(
            summary_upper, derived["hierarchical_bootstrap_upper_ratio"],
            rel_tol=1e-12, abs_tol=1e-12,
        ):
            problems.append(f"{label}: summary statistics disagree with raw observations")
        for kind, path in files.items():
            relative = path.relative_to(root)
            if relative in seen_relative:
                problems.append(f"{label}: artifact file is reused by another evaluation run")
                continue
            seen_relative.add(relative)
            retained.append({
                "path": str(relative), "sha256": file_sha256(path),
                "bytes": path.stat().st_size, "kind": kind,
            })
        run_rows.append({"run_id": run_id, "run_instance_id": instance_id, **derived})

    observed = max(
        (row["hierarchical_bootstrap_upper_ratio"] for row in run_rows),
        default=None,
    )
    accepted = (
        acceptance is not None and observed is not None and
        observed <= float(acceptance["threshold"]) and len(run_rows) >= minimum_runs
    )
    if document.get("state") != "promoted" or not accepted:
        problems.append("performance certificate is not promoted by the derived decision")
    attestation = {
        "run_count": len(run_rows),
        "session_count": sum(row["session_count"] for row in run_rows),
        "statistic": acceptance.get("statistic") if acceptance else None,
        "observed": observed,
        "threshold": acceptance.get("threshold") if acceptance else None,
        "accepted": accepted,
        "runs": run_rows,
    }
    return document, retained, attestation, problems
