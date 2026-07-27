#!/usr/bin/env python3
"""Live, paired Tgrad/tinygrad runtime measurements.

This module deliberately contains no performance policy.  It records raw,
same-process observations and descriptive paired statistics; it never emits a
threshold or a parity verdict.  Real adapters are imported lazily so the
orchestration and statistics can be calibrated with CPU-only fakes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import random
import statistics
import struct
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence


SCHEMA_VERSION = 2
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TINYGRAD_SOURCE = Path("/tmp/tgrad-upstream-19c4d736")
EXPECTED_TINYGRAD_COMMIT = "19c4d736f2bc8e26d21f08b28ffd6298408da00f"
EXPECTED_TINYGRAD_TREE = "855cca3b00c38841a6d3a043284f3a2ca696d4b0"
KNOWN_UPSTREAM_PYTHON = Path("/tmp/tgrad-upstream-py312/bin/python")


class HarnessError(RuntimeError):
    """Base error for configuration, provenance, or execution failures."""


class RevisionError(HarnessError):
    """The supplied tinygrad checkout is not the declared reference."""


class CorrectnessError(HarnessError):
    """Adapters disagreed before timing began."""


class MeasurementRunError(HarnessError):
    """A timed observation failed after the evidence stream was opened."""

    def __init__(self, message: str, summary: Mapping[str, Any]):
        super().__init__(message)
        self.summary = dict(summary)


@dataclass(frozen=True)
class GitState:
    path: str
    commit: str | None
    tree: str | None
    dirty: bool | None
    status_digest: str | None
    error: str | None = None


@dataclass(frozen=True)
class AdapterProvenance:
    name: str
    implementation: str
    source: GitState
    device: str
    environment: Mapping[str, Any]
    revision_validation: str
    revision_diagnostics: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["revision_diagnostics"] = list(self.revision_diagnostics)
        return value


@dataclass(frozen=True)
class RuntimeBinaryProvenance:
    path: str
    sha256: str
    size_bytes: int
    source_commit: str | None
    source_tree: str | None
    build_validation: str
    build_commands: tuple[tuple[str, ...], ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["build_commands"] = [list(command) for command in self.build_commands]
        return value


@dataclass(frozen=True)
class BoundarySpec:
    id: str
    category: str
    description: str
    includes: tuple[str, ...]
    excludes: tuple[str, ...]
    available: bool = True
    diagnostic: bool = False
    unavailable_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["includes"] = list(self.includes)
        value["excludes"] = list(self.excludes)
        return value


@dataclass(frozen=True)
class Measurement:
    duration_ns: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.duration_ns <= 0:
            raise ValueError("duration_ns must be positive")


@dataclass(frozen=True)
class PhaseObservation:
    boundary_id: str
    label: str
    measurement: Measurement


@dataclass(frozen=True)
class PreparedCorrectness:
    output: bytes
    metadata: Mapping[str, Any] = field(default_factory=dict)


class AdapterSession(Protocol):
    """One logical session.  Implementations may share process-global caches."""

    def boundary_specs(self) -> Mapping[str, BoundarySpec]: ...

    def correctness_output(self) -> bytes: ...

    def prepare(self) -> Iterable[PhaseObservation]: ...

    def prepared_correctness(self) -> PreparedCorrectness: ...

    def measure(self, boundary_id: str) -> Measurement: ...

    def close(self) -> None: ...


class Adapter(Protocol):
    name: str

    def provenance(self) -> AdapterProvenance: ...

    def create_session(
        self, workload: "Workload", a_payload: bytes, b_payload: bytes
    ) -> AdapterSession: ...


@dataclass(frozen=True)
class Workload:
    m: int = 64
    k: int = 64
    n: int = 64
    dtype: str = "bf16"
    input_seed: int = 42
    operation: str = "matmul"

    def validate(self) -> None:
        if self.operation != "matmul":
            raise HarnessError(f"unsupported operation: {self.operation}")
        if self.dtype != "bf16":
            raise HarnessError(f"unsupported dtype: {self.dtype}")
        if min(self.m, self.k, self.n) <= 0:
            raise HarnessError("matmul dimensions must be positive")

    @property
    def shape(self) -> str:
        return f"{self.m}x{self.k}x{self.n}"


@dataclass(frozen=True)
class Comparison:
    id: str
    numerator_adapter: str
    numerator_boundary: str
    denominator_adapter: str
    denominator_boundary: str
    interpretation: str
    diagnostic: bool = False
    kernel_speed_claim_eligible: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HarnessConfig:
    raw_output: Path
    summary_output: Path
    sessions: int = 3
    samples_per_session: int = 30
    warmup_pairs: int = 5
    order_seed: int = 20260727
    analysis_seed: int = 20260728
    bootstrap_resamples: int = 2000
    confidence_level: float = 0.95
    # Tests may pin these to make fake artifacts reproducible.  The real CLI
    # leaves them unset, producing a fresh identity and timestamp per run.
    run_instance_id: str | None = None
    captured_at_utc: str | None = None

    @property
    def completion_output(self) -> Path:
        return self.summary_output.with_name(self.summary_output.name + ".complete.json")

    def validate(self) -> None:
        if self.sessions <= 0:
            raise HarnessError("sessions must be positive")
        if self.samples_per_session < 2:
            raise HarnessError("samples_per_session must be at least 2 (one AB and one BA)")
        if self.warmup_pairs < 0:
            raise HarnessError("warmup_pairs cannot be negative")
        if self.bootstrap_resamples <= 0:
            raise HarnessError("bootstrap_resamples must be positive")
        if not 0.0 < self.confidence_level < 1.0:
            raise HarnessError("confidence_level must be between zero and one")
        if self.run_instance_id is not None and not self.run_instance_id.strip():
            raise HarnessError("run_instance_id must not be blank")
        if self.captured_at_utc is not None and not self.captured_at_utc.strip():
            raise HarnessError("captured_at_utc must not be blank")
        if self.raw_output.resolve() == self.summary_output.resolve():
            raise HarnessError("raw and summary output paths must differ")
        if self.raw_output.resolve().parent != self.summary_output.resolve().parent:
            raise HarnessError("raw and summary outputs must share one run directory")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _stable_seed(seed: int, *parts: Any) -> int:
    material = _canonical_json([seed, *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def deterministic_bf16_payload(numel: int, seed: int) -> bytes:
    """Generate finite bf16 payload bytes without depending on NumPy."""
    if numel <= 0:
        raise ValueError("numel must be positive")
    rng = random.Random(seed)
    out = bytearray(2 * numel)
    for index in range(numel):
        value = rng.uniform(-1.0, 1.0)
        f32_bits = struct.unpack("<I", struct.pack("<f", value))[0]
        struct.pack_into("<H", out, 2 * index, f32_bits >> 16)
    return bytes(out)


def workload_material(workload: Workload) -> tuple[bytes, bytes, dict[str, Any], str]:
    workload.validate()
    a_payload = deterministic_bf16_payload(workload.m * workload.k, workload.input_seed)
    b_payload = deterministic_bf16_payload(
        workload.k * workload.n, _stable_seed(workload.input_seed, "b")
    )
    manifest = {
        "operation": workload.operation,
        "shape": workload.shape,
        "dimensions": {"m": workload.m, "k": workload.k, "n": workload.n},
        "dtype": workload.dtype,
        "input_generation": {
            "algorithm": "python_random_uniform_f32_then_truncate_high_16_bits",
            "seed": workload.input_seed,
            "b_seed_derivation": "sha256(canonical_json([seed, 'b']))[:8]",
        },
        "payloads": {
            "a_bytes": len(a_payload),
            "a_sha256": _sha256_bytes(a_payload),
            "b_bytes": len(b_payload),
            "b_sha256": _sha256_bytes(b_payload),
        },
    }
    return a_payload, b_payload, manifest, _sha256_json(manifest)


def _git_command(path: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def inspect_git_checkout(path: Path) -> GitState:
    resolved = path.resolve()
    try:
        commit = _git_command(resolved, "rev-parse", "HEAD")
        tree = _git_command(resolved, "rev-parse", "HEAD^{tree}")
        status = _git_command(resolved, "status", "--porcelain", "--untracked-files=normal")
        diff = _git_command(resolved, "diff", "--binary", "HEAD", "--")
        untracked_text = _git_command(
            resolved, "ls-files", "--others", "--exclude-standard"
        )
        digest = hashlib.sha256()
        digest.update(status.encode("utf-8"))
        digest.update(diff.encode("utf-8"))
        for relative in sorted(line for line in untracked_text.splitlines() if line):
            candidate = resolved / relative
            digest.update(relative.encode("utf-8"))
            if candidate.is_symlink():
                digest.update(os.readlink(candidate).encode("utf-8"))
            elif candidate.is_file():
                digest.update(candidate.read_bytes())
        return GitState(
            path=str(resolved),
            commit=commit,
            tree=tree,
            dirty=bool(status),
            status_digest=digest.hexdigest(),
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return GitState(
            path=str(resolved), commit=None, tree=None, dirty=None,
            status_digest=None, error=f"{type(exc).__name__}: {exc}",
        )


def validate_tinygrad_checkout(
    path: Path, allow_unknown_revision: bool = False
) -> tuple[GitState, str, tuple[str, ...]]:
    state = inspect_git_checkout(path)
    diagnostics: list[str] = []
    if state.error is not None:
        diagnostics.append(f"git inspection failed: {state.error}")
    if state.commit != EXPECTED_TINYGRAD_COMMIT:
        diagnostics.append(
            f"commit is {state.commit!r}, expected {EXPECTED_TINYGRAD_COMMIT}"
        )
    if state.tree != EXPECTED_TINYGRAD_TREE:
        diagnostics.append(f"tree is {state.tree!r}, expected {EXPECTED_TINYGRAD_TREE}")
    if state.dirty is not False:
        diagnostics.append(f"dirty state is {state.dirty!r}, expected False")
    if diagnostics and not allow_unknown_revision:
        raise RevisionError(
            "tinygrad checkout is not the pinned clean source; pass "
            "--allow-unknown-tinygrad-revision only for conspicuous diagnostics: "
            + "; ".join(diagnostics)
        )
    validation = "pinned_clean" if not diagnostics else "diagnostic_override"
    return state, validation, tuple(diagnostics)


def validate_tgrad_checkout(
    path: Path, allow_dirty: bool = False
) -> tuple[GitState, str, tuple[str, ...]]:
    """Require an attributable local subject unless explicitly diagnostic."""
    state = inspect_git_checkout(path)
    diagnostics: list[str] = []
    if state.error is not None:
        diagnostics.append(f"git inspection failed: {state.error}")
    if state.commit is None or state.tree is None:
        diagnostics.append("commit/tree identity is unavailable")
    if state.dirty is not False:
        diagnostics.append(f"dirty state is {state.dirty!r}, expected False")
    if diagnostics and not allow_dirty:
        raise RevisionError(
            "Tgrad checkout is not a clean attributable subject; pass "
            "--allow-dirty-tgrad only for conspicuous diagnostics: "
            + "; ".join(diagnostics)
        )
    validation = "clean_attributed" if not diagnostics else "diagnostic_override"
    return state, validation, tuple(diagnostics)


def build_and_attest_tgrad_runtime(
    repository: Path,
    source: GitState,
    *,
    diagnostic_override: bool = False,
) -> RuntimeBinaryProvenance:
    """Build the measured dylib from the inspected subject and bind its bytes to it.

    A path and source revision alone do not establish which native code Python
    loaded.  Promotable runs therefore rebuild through the repository build
    graph, reject an alternate ``TGRAD_LIB``, re-check the source afterwards,
    and hash the exact dylib that the adapter will import.
    """
    resolved_repository = repository.resolve()
    expected = (resolved_repository / ".lake" / "build" / "lib" / "libtgrad.dylib").resolve()
    configured = Path(os.environ.get("TGRAD_LIB", str(expected))).expanduser().resolve()
    if configured != expected and not diagnostic_override:
        raise RevisionError(
            f"TGRAD_LIB resolves to {configured}, but promotable measurements require {expected}"
        )

    commands: tuple[tuple[str, ...], ...] = (
        ("lake", "build", "Tgrad:shared", "tgrad-cli", "tgrad-tests"),
        ("make", "-C", "c", "dylib"),
    )
    if not diagnostic_override:
        for command in commands:
            try:
                subprocess.run(
                    command,
                    cwd=resolved_repository,
                    check=True,
                    stdout=sys.stderr,
                    stderr=sys.stderr,
                )
            except (OSError, subprocess.CalledProcessError) as exc:
                raise HarnessError(
                    f"Tgrad runtime build failed for {' '.join(command)}: {exc}"
                ) from exc
        validation = "rebuilt_from_clean_subject"
    else:
        validation = "diagnostic_existing_binary"

    after = inspect_git_checkout(resolved_repository)
    if (
        not diagnostic_override
        and (after.commit != source.commit or after.tree != source.tree or after.dirty is not False)
    ):
        raise RevisionError(
            "Tgrad source identity changed while building the measured runtime: "
            f"before=({source.commit}, {source.tree}, {source.dirty}) "
            f"after=({after.commit}, {after.tree}, {after.dirty})"
        )
    if not configured.is_file():
        raise HarnessError(f"measured Tgrad dylib does not exist: {configured}")
    return RuntimeBinaryProvenance(
        path=str(configured),
        sha256=_sha256_file(configured),
        size_bytes=configured.stat().st_size,
        source_commit=source.commit,
        source_tree=source.tree,
        build_validation=validation,
        build_commands=commands if not diagnostic_override else (),
    )


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _import_from_source(module_name: str, source_root: Path, expected_file_root: Path) -> Any:
    existing = sys.modules.get(module_name)
    if existing is not None:
        existing_file = getattr(existing, "__file__", None)
        if existing_file is None or not _is_within(Path(existing_file), expected_file_root):
            raise HarnessError(
                f"{module_name} was already imported from {existing_file!r}, not {expected_file_root}"
            )
        return existing
    source = str(source_root.resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    module = importlib.import_module(module_name)
    module_file = getattr(module, "__file__", None)
    if module_file is None or not _is_within(Path(module_file), expected_file_root):
        raise HarnessError(
            f"imported {module_name} from {module_file!r}, expected beneath {expected_file_root}"
        )
    return module


def _toolchain_metadata() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_compiler": platform.python_compiler(),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": platform.node(),
        "processor": platform.processor(),
        "mac_version": platform.mac_ver()[0],
    }


def _maybe_reexec_python(requested: Path | None, argv: Sequence[str]) -> None:
    """Replace this process with an explicitly requested interpreter once.

    This is a launcher boundary, not a benchmark boundary: after ``execve``
    both adapters are imported and sampled in the same replacement process.
    No child process is launched per observation.
    """
    if requested is None:
        return
    resolved = requested.resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise HarnessError(f"requested Python executable is not executable: {resolved}")
    os.environ["TGRAD_PAIRED_REQUESTED_PYTHON"] = str(resolved)
    os.environ.setdefault("TGRAD_PAIRED_ORIGINAL_DEV", os.environ.get("DEV", "<unset>"))
    os.environ.setdefault("TGRAD_PAIRED_ORIGINAL_METAL", os.environ.get("METAL", "<unset>"))
    if resolved == Path(sys.executable).resolve():
        return
    environment = os.environ.copy()
    environment["TGRAD_PAIRED_REQUESTED_PYTHON"] = str(resolved)
    environment["TGRAD_PAIRED_ORIGINAL_DEV"] = environment.get("DEV", "<unset>")
    environment["TGRAD_PAIRED_ORIGINAL_METAL"] = environment.get("METAL", "<unset>")
    environment["DEV"] = "METAL"
    environment.pop("METAL", None)
    try:
        os.execve(
            str(resolved),
            [str(resolved), str(Path(__file__).resolve()), *argv],
            environment,
        )
    except OSError as exc:
        raise HarnessError(f"could not execute requested Python {resolved}: {exc}") from exc


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("quantile of empty values")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _sample_variance(values: Sequence[float]) -> float | None:
    return statistics.variance(values) if len(values) >= 2 else None


def _hierarchical_bootstrap(
    by_session: Mapping[int, Sequence[float]], resamples: int,
    confidence_level: float, seed: int,
) -> tuple[float, float]:
    session_ids = sorted(by_session)
    if not session_ids:
        raise ValueError("bootstrap requires observations")
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(resamples):
        sampled_session_means: list[float] = []
        for _session_slot in session_ids:
            chosen_id = rng.choice(session_ids)
            source = by_session[chosen_id]
            sampled_pairs = [rng.choice(source) for _ in range(len(source))]
            sampled_session_means.append(statistics.fmean(sampled_pairs))
        estimates.append(statistics.fmean(sampled_session_means))
    estimates.sort()
    tail = (1.0 - confidence_level) / 2.0
    return _quantile(estimates, tail), _quantile(estimates, 1.0 - tail)


def analyze_pairs(
    pairs: Sequence[Mapping[str, Any]], comparison: Comparison,
    config: HarnessConfig, workload: Workload,
) -> dict[str, Any] | None:
    selected = [p for p in pairs if p["comparison_id"] == comparison.id]
    if not selected:
        return None
    by_session: dict[int, list[float]] = {}
    log_ratios: list[float] = []
    for pair in selected:
        log_ratio = math.log(pair["numerator_duration_ns"] / pair["denominator_duration_ns"])
        log_ratios.append(log_ratio)
        by_session.setdefault(int(pair["session_index"]), []).append(log_ratio)
    ordered = sorted(log_ratios)
    probabilities = (("p05", 0.05), ("p25", 0.25), ("p50", 0.50),
                     ("p75", 0.75), ("p95", 0.95))
    log_quantiles = {name: _quantile(ordered, p) for name, p in probabilities}
    session_rows = []
    within_numerator = 0.0
    within_denominator = 0
    for session_index in sorted(by_session):
        values = by_session[session_index]
        variance = _sample_variance(values)
        if variance is not None:
            within_numerator += (len(values) - 1) * variance
            within_denominator += len(values) - 1
        session_rows.append({
            "session_index": session_index,
            "count": len(values),
            "mean_log_ratio": statistics.fmean(values),
            "sample_variance_log_ratio": variance,
        })
    session_means = [row["mean_log_ratio"] for row in session_rows]
    bootstrap_seed = _stable_seed(config.analysis_seed, comparison.id, "bootstrap")
    ci_low, ci_high = _hierarchical_bootstrap(
        by_session, config.bootstrap_resamples, config.confidence_level, bootstrap_seed
    )
    # Match the hierarchical bootstrap estimand: each logical session has
    # equal weight, even if an interrupted run leaves unequal pair counts.
    mean_log = statistics.fmean(session_means)
    operation_flops = 2 * workload.m * workload.k * workload.n

    def absolute_side(side: str) -> dict[str, Any]:
        durations = sorted(float(pair[f"{side}_duration_ns"]) for pair in selected)
        throughputs = sorted(operation_flops / (duration * 1_000.0) for duration in durations)
        return {
            "duration_ns_quantiles": {
                name: _quantile(durations, probability)
                for name, probability in probabilities
            },
            "effective_operational_rate_tflops_quantiles": {
                name: _quantile(throughputs, probability)
                for name, probability in probabilities
            },
            "arithmetic_convention": "2*M*K*N floating-point operations",
            "rate_scope": (
                "effective operation count divided by the observed boundary duration; "
                "not isolated kernel throughput"
            ),
        }

    return {
        "orientation": (
            f"log({comparison.numerator_adapter}/{comparison.denominator_adapter})"
        ),
        "count": len(log_ratios),
        "mean_log_ratio": mean_log,
        "geometric_mean_ratio": math.exp(mean_log),
        "log_ratio_quantiles": log_quantiles,
        "ratio_quantiles": {key: math.exp(value) for key, value in log_quantiles.items()},
        "absolute": {
            "numerator": absolute_side("numerator"),
            "denominator": absolute_side("denominator"),
        },
        "bootstrap": {
            "method": "hierarchical sessions-then-pairs, percentile interval of mean log-ratio",
            "analysis_seed": config.analysis_seed,
            "derived_seed": bootstrap_seed,
            "resamples": config.bootstrap_resamples,
            "confidence_level": config.confidence_level,
            "mean_log_ratio_interval": [ci_low, ci_high],
            "geometric_mean_ratio_interval": [math.exp(ci_low), math.exp(ci_high)],
        },
        "variance": {
            "pooled_within_session_log_ratio": (
                within_numerator / within_denominator if within_denominator else None
            ),
            "observed_between_session_mean_log_ratio": _sample_variance(session_means),
            "between_definition": (
                "sample variance of session means; observed, not deconvolved from within-session noise"
            ),
        },
        "sessions": session_rows,
    }


def _balanced_orders(count: int, seed: int) -> list[str]:
    if count < 0:
        raise ValueError("order count cannot be negative")
    rng = random.Random(seed)
    orders = [order for _ in range(count // 2) for order in ("AB", "BA")]
    if count % 2:
        orders.append(rng.choice(("AB", "BA")))
    rng.shuffle(orders)
    return orders


def _configuration_dict(config: HarnessConfig) -> dict[str, Any]:
    return {
        "sessions": config.sessions,
        "samples_per_session": config.samples_per_session,
        "warmup_pairs": config.warmup_pairs,
        "order_seed": config.order_seed,
        "analysis_seed": config.analysis_seed,
        "bootstrap_resamples": config.bootstrap_resamples,
        "confidence_level": config.confidence_level,
    }


def _open_atomic_text(path: Path) -> tuple[Any, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    )
    return handle, Path(handle.name)


def _write_json_temporary(path: Path, value: Mapping[str, Any]) -> Path:
    handle, temporary = _open_atomic_text(path)
    try:
        with handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return temporary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = _write_json_temporary(path, value)
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _error_dict(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)}


def _close_sessions(sessions: Iterable[AdapterSession]) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    for session in reversed(list(sessions)):
        try:
            session.close()
        except BaseException as exc:
            errors.append(_error_dict(exc))
    return errors


def _boundary_catalog(
    adapters: Mapping[str, Adapter], workload: Workload,
    a_payload: bytes, b_payload: bytes,
) -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    sessions: list[AdapterSession] = []
    try:
        for name in sorted(adapters):
            adapter = adapters[name]
            session = adapter.create_session(workload, a_payload, b_payload)
            sessions.append(session)
            specs = session.boundary_specs()
            for boundary_id, spec in specs.items():
                if boundary_id != spec.id:
                    raise HarnessError(
                        f"adapter {name} boundary key {boundary_id!r} disagrees with spec id {spec.id!r}"
                    )
            catalog[name] = {
                boundary_id: spec.to_dict()
                for boundary_id, spec in sorted(specs.items())
            }
    finally:
        close_errors = _close_sessions(sessions)
        if close_errors and sys.exc_info()[0] is None:
            raise HarnessError(f"boundary-catalog session close failed: {close_errors}")
    return catalog


def _correctness_preflight(
    adapters: Mapping[str, Adapter], comparisons: Sequence[Comparison],
    workload: Workload, a_payload: bytes, b_payload: bytes,
) -> dict[str, Any]:
    sessions: dict[str, AdapterSession] = {}
    outputs: dict[str, bytes] = {}
    try:
        for name in sorted(adapters):
            adapter = adapters[name]
            sessions[name] = adapter.create_session(workload, a_payload, b_payload)
        for name in sorted(sessions):
            outputs[name] = sessions[name].correctness_output()
        expected_bytes = workload.m * workload.n * 2
        for name, output in outputs.items():
            if len(output) != expected_bytes:
                raise CorrectnessError(
                    f"correctness preflight output from {name} has {len(output)} bytes; "
                    f"expected {expected_bytes} for {workload.m}x{workload.n} bf16"
                )
        reference_name = sorted(outputs)[0]
        for name, output in outputs.items():
            if output != outputs[reference_name]:
                raise CorrectnessError(
                    f"correctness preflight mismatch: {reference_name}="
                    f"{_sha256_bytes(outputs[reference_name])} {name}={_sha256_bytes(output)}"
                )
        for comparison in comparisons:
            numerator = outputs[comparison.numerator_adapter]
            denominator = outputs[comparison.denominator_adapter]
            if numerator != denominator:
                raise CorrectnessError(
                    f"correctness preflight mismatch for {comparison.id}: "
                    f"{comparison.numerator_adapter}={_sha256_bytes(numerator)} "
                    f"{comparison.denominator_adapter}={_sha256_bytes(denominator)}"
                )
        return {
            name: {"bytes": len(value), "sha256": _sha256_bytes(value)}
            for name, value in sorted(outputs.items())
        }
    finally:
        close_errors = _close_sessions(sessions.values())
        if close_errors and sys.exc_info()[0] is None:
            raise HarnessError(f"correctness-preflight session close failed: {close_errors}")


def _validate_comparisons(
    comparisons: Sequence[Comparison], adapters: Mapping[str, Adapter],
    boundaries: Mapping[str, Mapping[str, Any]],
) -> None:
    if not comparisons:
        raise HarnessError("at least one comparison is required")
    ids: set[str] = set()
    for comparison in comparisons:
        if comparison.id in ids:
            raise HarnessError(f"duplicate comparison id: {comparison.id}")
        ids.add(comparison.id)
        if comparison.numerator_adapter == comparison.denominator_adapter:
            raise HarnessError(f"comparison {comparison.id} cannot compare an adapter to itself")
        for adapter_name, boundary_id in (
            (comparison.numerator_adapter, comparison.numerator_boundary),
            (comparison.denominator_adapter, comparison.denominator_boundary),
        ):
            if adapter_name not in adapters:
                raise HarnessError(f"unknown adapter in {comparison.id}: {adapter_name}")
            spec = boundaries.get(adapter_name, {}).get(boundary_id)
            if spec is None:
                raise HarnessError(
                    f"unknown boundary in {comparison.id}: {adapter_name}.{boundary_id}"
                )
            if not spec["available"]:
                raise HarnessError(
                    f"unavailable boundary in {comparison.id}: {adapter_name}.{boundary_id}: "
                    f"{spec['unavailable_reason']}"
                )


def run_harness(
    config: HarnessConfig,
    workload: Workload,
    adapters: Mapping[str, Adapter],
    comparisons: Sequence[Comparison],
) -> dict[str, Any]:
    """Run correctness preflight, paired sessions, and deterministic analysis.

    Correctness failures occur before either output path is created.  Once
    timing starts, every attempted observation (including errors and warmups)
    is appended to the raw stream.
    """
    config.validate()
    if len(adapters) < 2:
        raise HarnessError("paired measurement requires at least two adapters")
    if (
        config.raw_output.exists()
        or config.summary_output.exists()
        or config.completion_output.exists()
    ):
        raise HarnessError("refusing to overwrite an existing run artifact")
    a_payload, b_payload, manifest, workload_hash = workload_material(workload)
    provenance: dict[str, dict[str, Any]] = {}
    for name, adapter in sorted(adapters.items()):
        adapter_provenance = adapter.provenance()
        if adapter.name != name or adapter_provenance.name != name:
            raise HarnessError(
                f"adapter identity mismatch for mapping key {name!r}: "
                f"adapter.name={adapter.name!r}, provenance.name={adapter_provenance.name!r}"
            )
        provenance[name] = adapter_provenance.to_dict()
    boundaries = _boundary_catalog(adapters, workload, a_payload, b_payload)
    _validate_comparisons(comparisons, adapters, boundaries)

    # No output file exists until all implementations agree on exact bf16 bytes.
    correctness = _correctness_preflight(
        adapters, comparisons, workload, a_payload, b_payload
    )

    config_dict = _configuration_dict(config)
    run_instance = {
        "id": config.run_instance_id or uuid.uuid4().hex,
        "captured_at_utc": config.captured_at_utc or datetime.now(timezone.utc).isoformat(),
    }
    toolchain = _toolchain_metadata()
    run_identity = {
        "schema_version": SCHEMA_VERSION,
        "run_instance": run_instance,
        "workload_hash": workload_hash,
        "configuration": config_dict,
        "toolchain": toolchain,
        "provenance": provenance,
        "timing_boundaries": boundaries,
        "comparisons": [comparison.to_dict() for comparison in comparisons],
        "correctness_preflight": correctness,
    }
    run_id = _sha256_json(run_identity)
    raw_handle, raw_temporary = _open_atomic_text(config.raw_output)
    observation_index = 0
    raw_count = 0
    pairs: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    completion = "complete"
    unexpected_failure: BaseException | None = None

    def emit(record: Mapping[str, Any]) -> None:
        nonlocal observation_index, raw_count
        enriched = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "run_instance_id": run_instance["id"],
            "workload_hash": workload_hash,
            "observation_index": observation_index,
            **record,
        }
        raw_handle.write(_canonical_json(enriched) + "\n")
        raw_handle.flush()
        observation_index += 1
        raw_count += 1

    try:
        for session_index in range(config.sessions):
            sessions: dict[str, AdapterSession] = {}
            try:
                for name in sorted(adapters):
                    adapter = adapters[name]
                    sessions[name] = adapter.create_session(workload, a_payload, b_payload)
                for adapter_name in sorted(sessions):
                    try:
                        for phase_index, phase in enumerate(sessions[adapter_name].prepare()):
                            emit({
                                "record_type": "observation",
                                "phase": "preparation",
                                "included_in_analysis": False,
                                "session_index": session_index,
                                "phase_index": phase_index,
                                "adapter": adapter_name,
                                "boundary_id": phase.boundary_id,
                                "label": phase.label,
                                "duration_ns": phase.measurement.duration_ns,
                                "metadata": dict(phase.measurement.metadata),
                            })
                        prepared = sessions[adapter_name].prepared_correctness()
                        prepared_sha256 = _sha256_bytes(prepared.output)
                        expected = correctness[adapter_name]
                        if (
                            len(prepared.output) != expected["bytes"]
                            or prepared_sha256 != expected["sha256"]
                        ):
                            raise CorrectnessError(
                                f"prepared timed route for {adapter_name} disagrees with preflight: "
                                f"bytes={len(prepared.output)} sha256={prepared_sha256}; "
                                f"expected bytes={expected['bytes']} sha256={expected['sha256']}"
                            )
                        emit({
                            "record_type": "prepared_correctness",
                            "phase": "preparation",
                            "included_in_analysis": False,
                            "session_index": session_index,
                            "adapter": adapter_name,
                            "output_bytes": len(prepared.output),
                            "output_sha256": prepared_sha256,
                            "metadata": dict(prepared.metadata),
                        })
                    except BaseException as exc:
                        error = {
                            "session_index": session_index,
                            "adapter": adapter_name,
                            "phase": "preparation",
                            "error": _error_dict(exc),
                        }
                        errors.append(error)
                        emit({
                            "record_type": "observation_error",
                            "included_in_analysis": False,
                            **error,
                        })
                        raise MeasurementRunError(
                            f"preparation failed for {adapter_name} in session {session_index}", {}
                        ) from exc
                for comparison in comparisons:
                    phases = (("warmup", config.warmup_pairs),
                              ("measured", config.samples_per_session))
                    for phase_name, count in phases:
                        orders = _balanced_orders(
                            count,
                            _stable_seed(
                                config.order_seed, session_index, comparison.id, phase_name
                            ),
                        )
                        for sample_index, order in enumerate(orders):
                            sides = (
                                (("numerator", comparison.numerator_adapter,
                                  comparison.numerator_boundary),
                                 ("denominator", comparison.denominator_adapter,
                                  comparison.denominator_boundary))
                                if order == "AB" else
                                (("denominator", comparison.denominator_adapter,
                                  comparison.denominator_boundary),
                                 ("numerator", comparison.numerator_adapter,
                                  comparison.numerator_boundary))
                            )
                            durations: dict[str, int] = {}
                            pending_records: list[dict[str, Any]] = []
                            pair_failed = False
                            for order_position, (side, adapter_name, boundary_id) in enumerate(sides):
                                try:
                                    measured = sessions[adapter_name].measure(boundary_id)
                                    durations[side] = measured.duration_ns
                                    pending_records.append({
                                        "record_type": "observation",
                                        "phase": phase_name,
                                        "session_index": session_index,
                                        "comparison_id": comparison.id,
                                        "sample_index": sample_index,
                                        "order": order,
                                        "order_position": order_position,
                                        "side": side,
                                        "adapter": adapter_name,
                                        "boundary_id": boundary_id,
                                        "duration_ns": measured.duration_ns,
                                        "metadata": dict(measured.metadata),
                                    })
                                except BaseException as exc:
                                    pair_failed = True
                                    error = {
                                        "session_index": session_index,
                                        "comparison_id": comparison.id,
                                        "phase": phase_name,
                                        "sample_index": sample_index,
                                        "order": order,
                                        "order_position": order_position,
                                        "adapter": adapter_name,
                                        "boundary_id": boundary_id,
                                        "error": _error_dict(exc),
                                    }
                                    errors.append(error)
                                    for pending in pending_records:
                                        emit({"included_in_analysis": False, **pending})
                                    emit({
                                        "record_type": "observation_error",
                                        "included_in_analysis": False,
                                        **error,
                                    })
                                    break
                            if pair_failed:
                                completion = "measurement_error"
                                raise MeasurementRunError(
                                    f"timed observation failed in {comparison.id}", {}
                                )
                            for pending in pending_records:
                                emit({
                                    "included_in_analysis": phase_name == "measured",
                                    **pending,
                                })
                            if phase_name == "measured":
                                pairs.append({
                                    "session_index": session_index,
                                    "comparison_id": comparison.id,
                                    "sample_index": sample_index,
                                    "order": order,
                                    "numerator_duration_ns": durations["numerator"],
                                    "denominator_duration_ns": durations["denominator"],
                                })
            finally:
                close_errors = _close_sessions(sessions.values())
                if close_errors and sys.exc_info()[0] is None:
                    raise HarnessError(f"timed-session close failed: {close_errors}")
    except MeasurementRunError as exc:
        completion = "measurement_error"
        unexpected_failure = exc
    except BaseException as exc:
        completion = "measurement_error"
        unexpected_failure = exc
        error = {
            "phase": "session_setup_or_preparation",
            "error": _error_dict(exc),
        }
        errors.append(error)
        emit({
            "record_type": "observation_error",
            "included_in_analysis": False,
            **error,
        })
    finally:
        raw_handle.flush()
        os.fsync(raw_handle.fileno())
        raw_handle.close()

    raw_sha256 = _sha256_file(raw_temporary)

    analyses = {
        comparison.id: analyze_pairs(pairs, comparison, config, workload)
        for comparison in comparisons
    }
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "run_instance": run_instance,
        "completion": completion,
        "workload": manifest,
        "workload_hash": workload_hash,
        "configuration": config_dict,
        "toolchain": toolchain,
        "provenance": provenance,
        "timing_boundaries": boundaries,
        "comparisons": [comparison.to_dict() for comparison in comparisons],
        "correctness_preflight": correctness,
        "raw_artifact": {
            "file": config.raw_output.name,
            "sha256": raw_sha256,
            "observation_count": raw_count,
        },
        "completion_marker": config.completion_output.name,
        "raw_observation_count": raw_count,
        "complete_pair_count": len(pairs),
        "analysis": analyses,
        "errors": errors,
        "methodology": {
            "ordering": "seeded balanced AB/BA, shuffled independently per session/comparison/phase",
            "session_scope": (
                "logical sessions in one process; implementation process-global caches may persist"
            ),
            "policy": "descriptive observations only; no threshold or performance conclusion",
        },
    }
    summary_temporary: Path | None = None
    try:
        summary_temporary = _write_json_temporary(config.summary_output, summary)
        summary_sha256 = _sha256_file(summary_temporary)
        artifact_identity = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "raw": {
                "file": config.raw_output.name,
                "sha256": raw_sha256,
                "observation_count": raw_count,
            },
            "summary": {
                "file": config.summary_output.name,
                "sha256": summary_sha256,
            },
        }
        completion_manifest = {
            **artifact_identity,
            "artifact_set_id": _sha256_json(artifact_identity),
            "state": "complete",
        }
        os.replace(raw_temporary, config.raw_output)
        os.replace(summary_temporary, config.summary_output)
        summary_temporary = None
        # This marker is the commit record for the two-file artifact set. A
        # raw or summary file without this marker is incomplete evidence.
        _write_json_atomic(config.completion_output, completion_manifest)
    except BaseException:
        raw_temporary.unlink(missing_ok=True)
        if summary_temporary is not None:
            summary_temporary.unlink(missing_ok=True)
        if not config.completion_output.exists():
            config.raw_output.unlink(missing_ok=True)
            config.summary_output.unlink(missing_ok=True)
        raise
    if completion != "complete":
        raise MeasurementRunError("paired measurement ended with an error", summary) from unexpected_failure
    return summary


class _TgradSession:
    def __init__(self, module: Any, workload: Workload, a_payload: bytes, b_payload: bytes):
        self._module = module
        self._workload = workload
        self._a = module.Tensor.from_bf16_bytes(a_payload, (workload.m, workload.k))
        self._b = module.Tensor.from_bf16_bytes(b_payload, (workload.k, workload.n))
        self._last = None
        unavailable = lambda boundary_id, category, reason: BoundarySpec(
            id=boundary_id, category=category, description=reason, includes=(), excludes=(),
            available=False, unavailable_reason=reason,
        )
        self._boundaries = {
            "repeated_synchronized_matmul": BoundarySpec(
                id="repeated_synchronized_matmul",
                category="end_to_end",
                description=(
                    "Wall time around local Tensor.__matmul__; includes Python route choice, "
                    "output allocation, FFI, Metal command submission, and waitUntilCompleted."
                ),
                includes=("Python call", "route selection", "output allocation", "FFI",
                          "Metal dispatch", "device synchronization"),
                excludes=("input construction", "output readback", "output release"),
            ),
            "dispatch_runtime": unavailable(
                "dispatch_runtime", "dispatch_runtime",
                "Tgrad exposes no isolated public dispatch-only boundary in this harness",
            ),
            "compile_capture": unavailable(
                "compile_capture", "compile_capture",
                "first-call compilation is inseparable from correctness setup and runtime",
            ),
            "tinyjit_replay": unavailable(
                "tinyjit_replay", "tinyjit_replay", "Tgrad has no TinyJit API",
            ),
        }

    def boundary_specs(self) -> Mapping[str, BoundarySpec]:
        return self._boundaries

    def correctness_output(self) -> bytes:
        return (self._a @ self._b).to_bytes()

    def prepare(self) -> Sequence[PhaseObservation]:
        return ()

    def prepared_correctness(self) -> PreparedCorrectness:
        output = self.correctness_output()
        return PreparedCorrectness(
            output,
            {"route": "repeated_synchronized_matmul", "prepared_state": "not_applicable"},
        )

    def measure(self, boundary_id: str) -> Measurement:
        if boundary_id != "repeated_synchronized_matmul":
            raise HarnessError(f"unsupported Tgrad boundary: {boundary_id}")
        started = time.perf_counter_ns()
        result = self._a @ self._b
        duration = time.perf_counter_ns() - started
        self._last = result
        return Measurement(duration, {"synchronization": "inside Tgrad Metal dispatch"})

    def close(self) -> None:
        self._last = None
        self._a = None
        self._b = None


class TgradAdapter:
    name = "tgrad"

    def __init__(
        self, repository: Path = REPO_ROOT, allow_dirty: bool = False,
        validated: tuple[GitState, str, tuple[str, ...]] | None = None,
        runtime_binary: RuntimeBinaryProvenance | None = None,
    ):
        self._repository = repository.resolve()
        state, validation, diagnostics = (
            validated if validated is not None
            else validate_tgrad_checkout(self._repository, allow_dirty)
        )
        binary = runtime_binary or build_and_attest_tgrad_runtime(
            self._repository,
            state,
            diagnostic_override=validation == "diagnostic_override",
        )
        python_root = self._repository / "python"
        self._module = _import_from_source("tgrad", python_root, python_root)
        loaded_path = Path(self._module.LIB_PATH).resolve()
        if loaded_path != Path(binary.path).resolve():
            raise RevisionError(
                f"Tgrad imported {loaded_path}, but the attested runtime is {binary.path}"
            )
        loaded_sha256 = _sha256_file(loaded_path)
        if loaded_sha256 != binary.sha256:
            raise RevisionError(
                "Tgrad dylib changed between attestation and import: "
                f"attested={binary.sha256} loaded={loaded_sha256}"
            )
        self._provenance = AdapterProvenance(
            name=self.name,
            implementation="local Tgrad Python/Lean/Metal route",
            source=state,
            device="METAL",
            environment={
                "DEV": os.environ.get("DEV"),
                "METAL": os.environ.get("METAL"),
                "runtime_binary": binary.to_dict(),
            },
            revision_validation=validation,
            revision_diagnostics=diagnostics,
        )

    def provenance(self) -> AdapterProvenance:
        return self._provenance

    def create_session(
        self, workload: Workload, a_payload: bytes, b_payload: bytes
    ) -> AdapterSession:
        return _TgradSession(self._module, workload, a_payload, b_payload)


def _bf16_payload_to_numpy(payload: bytes, shape: tuple[int, int]) -> Any:
    numpy = importlib.import_module("numpy")
    words = numpy.frombuffer(payload, dtype="<u2").astype(numpy.uint32)
    return (words << 16).view(numpy.float32).reshape(shape)


def _numpy_to_bf16_payload(value: Any) -> bytes:
    numpy = importlib.import_module("numpy")
    f32 = numpy.asarray(value, dtype=numpy.float32)
    words = (f32.view(numpy.uint32) >> 16).astype("<u2", copy=False)
    return words.tobytes(order="C")


class _TinygradSession:
    def __init__(self, module: Any, workload: Workload, a_payload: bytes, b_payload: bytes):
        self._module = module
        self._workload = workload
        self._Tensor = module.Tensor
        self._TinyJit = module.TinyJit
        self._Device = module.Device
        self._dtypes = module.dtypes
        self._a = self._Tensor(
            _bf16_payload_to_numpy(a_payload, (workload.m, workload.k)),
            device="METAL", dtype=self._dtypes.bfloat16,
        ).realize()
        self._b = self._Tensor(
            _bf16_payload_to_numpy(b_payload, (workload.k, workload.n)),
            device="METAL", dtype=self._dtypes.bfloat16,
        ).realize()
        self._last = None
        self._capture_verified = False

        def jit_body(a: Any, b: Any) -> Any:
            return (a @ b).cast(self._dtypes.bfloat16).realize()

        self._jit = self._TinyJit(jit_body)
        unavailable = lambda boundary_id, category, reason: BoundarySpec(
            id=boundary_id, category=category, description=reason, includes=(), excludes=(),
            available=False, unavailable_reason=reason,
        )
        self._boundaries = {
            "tinyjit_first_call": BoundarySpec(
                id="tinyjit_first_call", category="compile_capture",
                description="TinyJit count-zero call plus Device synchronization; not isolated compile time.",
                includes=("graph construction", "scheduling", "compilation/cache lookup", "runtime", "synchronization"),
                excludes=("input construction", "output readback"), diagnostic=True,
            ),
            "tinyjit_capture_call": BoundarySpec(
                id="tinyjit_capture_call", category="compile_capture",
                description="TinyJit capture call plus Device synchronization; capture and runtime are combined.",
                includes=("capture", "lowering", "compilation/cache lookup", "runtime", "synchronization"),
                excludes=("input construction", "output readback"), diagnostic=True,
            ),
            "tinyjit_replay": BoundarySpec(
                id="tinyjit_replay", category="tinyjit_replay",
                description="Prepared TinyJit replay call plus explicit Device synchronization.",
                includes=("Python TinyJit call", "captured program replay", "device synchronization"),
                excludes=("initial capture", "initial compile", "input construction", "output readback"),
            ),
            "unjit_end_to_end": BoundarySpec(
                id="unjit_end_to_end", category="end_to_end",
                description="Fresh un-JIT matmul graph, realize, and Device synchronization.",
                includes=("Python graph construction", "schedule/lower/cache lookup", "runtime", "synchronization"),
                excludes=("input construction", "output readback"), diagnostic=True,
            ),
            "dispatch_runtime": unavailable(
                "dispatch_runtime", "dispatch_runtime",
                "this adapter does not use a private dispatch-only tinygrad API",
            ),
        }

    def _sync(self) -> None:
        self._Device[self._Device.DEFAULT].synchronize()

    def _unjit(self) -> Any:
        result = (self._a @ self._b).cast(self._dtypes.bfloat16).realize()
        self._sync()
        return result

    def boundary_specs(self) -> Mapping[str, BoundarySpec]:
        return self._boundaries

    def correctness_output(self) -> bytes:
        result = self._unjit()
        return _numpy_to_bf16_payload(result.numpy())

    def _measure_call(self, call: Any, metadata: Mapping[str, Any]) -> Measurement:
        started = time.perf_counter_ns()
        result = call()
        self._sync()
        duration = time.perf_counter_ns() - started
        self._last = result
        return Measurement(duration, metadata)

    def prepare(self) -> Iterable[PhaseObservation]:
        first = self._measure_call(
            lambda: self._jit(self._a, self._b),
            {"tinyjit_count_before": 0, "boundary_is_composite": True},
        )
        yield PhaseObservation("tinyjit_first_call", "TinyJit first call", first)
        capture = self._measure_call(
            lambda: self._jit(self._a, self._b),
            {"tinyjit_count_before": 1, "boundary_is_composite": True},
        )
        yield PhaseObservation("tinyjit_capture_call", "TinyJit capture call", capture)
        if self._jit.cnt != 2 or self._jit.captured is None:
            raise CorrectnessError(
                "TinyJit did not enter captured replay state after preparation: "
                f"cnt={self._jit.cnt}, captured={self._jit.captured is not None}"
            )
        self._capture_verified = True

    def prepared_correctness(self) -> PreparedCorrectness:
        if not self._capture_verified or self._jit.captured is None or self._jit.cnt < 2:
            raise CorrectnessError("TinyJit replay correctness requested before verified capture")
        count_before = self._jit.cnt
        result = self._jit(self._a, self._b)
        self._sync()
        if self._jit.captured is None or self._jit.cnt != count_before + 1:
            raise CorrectnessError("TinyJit replay state changed unexpectedly during correctness check")
        return PreparedCorrectness(
            _numpy_to_bf16_payload(result.numpy()),
            {
                "tinyjit_phase": "verified_replay",
                "tinyjit_count_before": count_before,
                "tinyjit_count_after": self._jit.cnt,
                "captured": True,
                "explicit_device_synchronize": True,
            },
        )

    def measure(self, boundary_id: str) -> Measurement:
        if boundary_id == "tinyjit_replay":
            return self._measure_call(
                lambda: self._jit(self._a, self._b),
                {"tinyjit_phase": "replay", "explicit_device_synchronize": True},
            )
        if boundary_id == "unjit_end_to_end":
            return self._measure_call(
                lambda: (self._a @ self._b).cast(self._dtypes.bfloat16).realize(),
                {"tinyjit_phase": "not_used", "explicit_device_synchronize": True},
            )
        raise HarnessError(f"unsupported tinygrad boundary: {boundary_id}")

    def close(self) -> None:
        self._last = None
        self._jit = None
        self._a = None
        self._b = None


class TinygradAdapter:
    name = "tinygrad"

    def __init__(
        self, source: Path = DEFAULT_TINYGRAD_SOURCE,
        allow_unknown_revision: bool = False,
        validated: tuple[GitState, str, tuple[str, ...]] | None = None,
    ):
        self._source = source.resolve()
        state, validation, diagnostics = (
            validated if validated is not None
            else validate_tinygrad_checkout(self._source, allow_unknown_revision)
        )
        inherited_dev = os.environ.pop(
            "TGRAD_PAIRED_ORIGINAL_DEV", os.environ.get("DEV", "<unset>")
        )
        inherited_legacy_metal_marker = os.environ.pop(
            "TGRAD_PAIRED_ORIGINAL_METAL", None
        )
        inherited_legacy_metal = os.environ.pop("METAL", None)
        original_legacy_metal = (
            inherited_legacy_metal_marker
            if inherited_legacy_metal_marker is not None
            else (inherited_legacy_metal if inherited_legacy_metal is not None else "<unset>")
        )
        requested_python = os.environ.pop("TGRAD_PAIRED_REQUESTED_PYTHON", None)
        inherited_jit = os.environ.get("JIT", "<unset>")
        os.environ["DEV"] = "METAL"
        os.environ["JIT"] = "1"
        self._module = _import_from_source("tinygrad", self._source, self._source / "tinygrad")
        self._provenance = AdapterProvenance(
            name=self.name,
            implementation="pinned official tinygrad",
            source=state,
            device="METAL",
            environment={
                "DEV": "METAL",
                "DEV_inherited": inherited_dev,
                "legacy_METAL_removed": original_legacy_metal != "<unset>",
                "legacy_METAL_inherited_value": original_legacy_metal,
                "METAL_effective": os.environ.get("METAL"),
                "JIT_inherited": inherited_jit,
                "JIT_effective": os.environ.get("JIT"),
                "JIT_requirement": "capture object and replay output verified before timing",
                "requested_python_executable": requested_python,
                "effective_python_executable": str(Path(sys.executable).resolve()),
            },
            revision_validation=validation,
            revision_diagnostics=diagnostics,
        )

    def provenance(self) -> AdapterProvenance:
        return self._provenance

    def create_session(
        self, workload: Workload, a_payload: bytes, b_payload: bytes
    ) -> AdapterSession:
        return _TinygradSession(self._module, workload, a_payload, b_payload)


def default_comparisons(include_unjit_diagnostic: bool = False) -> list[Comparison]:
    comparisons = [Comparison(
        id="tgrad_repeated_vs_tinygrad_tinyjit_replay",
        numerator_adapter="tgrad",
        numerator_boundary="repeated_synchronized_matmul",
        denominator_adapter="tinygrad",
        denominator_boundary="tinyjit_replay",
        interpretation=(
            "Operational repeated-call comparison. Tgrad includes route/allocation/FFI/dispatch/sync; "
            "tinygrad includes TinyJit replay/Python/sync. It is not an isolated kernel comparison."
        ),
        kernel_speed_claim_eligible=False,
    )]
    if include_unjit_diagnostic:
        comparisons.append(Comparison(
            id="tgrad_repeated_vs_tinygrad_unjit_diagnostic",
            numerator_adapter="tgrad",
            numerator_boundary="repeated_synchronized_matmul",
            denominator_adapter="tinygrad",
            denominator_boundary="unjit_end_to_end",
            interpretation=(
                "Diagnostic only: tinygrad rebuilds and realizes an un-JIT graph while Tgrad uses its "
                "repeated static route. This deliberately asymmetric boundary is reported separately."
            ),
            diagnostic=True,
            kernel_speed_claim_eligible=False,
        ))
    return comparisons


def _parse_shape(value: str) -> tuple[int, int, int]:
    try:
        dimensions = tuple(int(part) for part in value.lower().split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape must be MxKxN") from exc
    if len(dimensions) != 3 or min(dimensions) <= 0:
        raise argparse.ArgumentTypeError("shape must contain three positive dimensions")
    return dimensions  # type: ignore[return-value]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="live paired Tgrad/tinygrad runtime observations (no performance verdict)"
    )
    parser.add_argument("--tinygrad-source", type=Path, default=DEFAULT_TINYGRAD_SOURCE)
    parser.add_argument(
        "--python-executable", type=Path,
        help=(
            "replace this process with the requested Python before importing either runtime; "
            f"known working upstream environment: {KNOWN_UPSTREAM_PYTHON}"
        ),
    )
    parser.add_argument("--allow-unknown-tinygrad-revision", action="store_true")
    parser.add_argument(
        "--allow-dirty-tgrad", action="store_true",
        help="allow a dirty/unattributed Tgrad subject for diagnostic output only",
    )
    parser.add_argument("--shape", type=_parse_shape, default=(64, 64, 64))
    parser.add_argument("--input-seed", type=int, default=42)
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--order-seed", type=int, default=20260727)
    parser.add_argument("--analysis-seed", type=int, default=20260728)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--include-unjit-diagnostic", action="store_true")
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(effective_argv)
    m, k, n = args.shape
    try:
        _maybe_reexec_python(args.python_executable, effective_argv)
        # Establish both subjects before either runtime import. Environment
        # selection then happens in TinygradAdapter before both imports.
        tinygrad_validation = validate_tinygrad_checkout(
            args.tinygrad_source, args.allow_unknown_tinygrad_revision
        )
        tgrad_validation = validate_tgrad_checkout(
            REPO_ROOT, args.allow_dirty_tgrad
        )
        tgrad_binary = build_and_attest_tgrad_runtime(
            REPO_ROOT,
            tgrad_validation[0],
            diagnostic_override=tgrad_validation[1] == "diagnostic_override",
        )
        tinygrad_adapter = TinygradAdapter(
            args.tinygrad_source, args.allow_unknown_tinygrad_revision,
            validated=tinygrad_validation,
        )
        tgrad_adapter = TgradAdapter(
            REPO_ROOT, args.allow_dirty_tgrad, validated=tgrad_validation,
            runtime_binary=tgrad_binary,
        )
        summary = run_harness(
            HarnessConfig(
                raw_output=args.raw_output,
                summary_output=args.summary_output,
                sessions=args.sessions,
                samples_per_session=args.samples,
                warmup_pairs=args.warmup,
                order_seed=args.order_seed,
                analysis_seed=args.analysis_seed,
                bootstrap_resamples=args.bootstrap_resamples,
                confidence_level=args.confidence_level,
            ),
            Workload(m=m, k=k, n=n, input_seed=args.input_seed),
            {"tgrad": tgrad_adapter, "tinygrad": tinygrad_adapter},
            default_comparisons(args.include_unjit_diagnostic),
        )
    except MeasurementRunError as exc:
        print(f"paired runtime measurement error: {exc}", file=sys.stderr)
        return 1
    except HarnessError as exc:
        print(f"paired runtime setup error: {exc}", file=sys.stderr)
        return 2
    print(_canonical_json({
        "completion": summary["completion"],
        "raw_output": str(args.raw_output),
        "summary_output": str(args.summary_output),
        "completion_output": str(
            args.summary_output.with_name(args.summary_output.name + ".complete.json")
        ),
        "raw_observation_count": summary["raw_observation_count"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
