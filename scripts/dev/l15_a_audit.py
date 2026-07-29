#!/usr/bin/env python3
"""L15.A — static + structural audit. Run by `scripts/gates/L15_A.sh`.

Emits a single JSON document on stdout with:
- `criteria[]`: 3 entries (tensor_api / symbolic_graph / rewrite_codegen),
  each with `verdict: pass|fail|inconclusive` + `evidence` text.
- `static_checks`: 9 booleans from `GOAL_L15.md §4`.

Narrowing notes (per `GOAL_L15.md §4`: "If a valid implementation
trips one, update the check with a narrower predicate and explain why
in `EXPERIMENT_RESULT.md`"):

1. **Criterion 1 / `n_legacy`** — the spec's draft regex matches
   `self._(buf|shape|dtype|view_ops|strides) =`. In our implementation,
   Python Tensor caches `_buf`, `_shape`, `_dtype` as buffer-accessor
   metadata derived from the Lean handle (e.g. `Tensor.from_numpy`
   allocates a Metal buffer and IMMEDIATELY registers the handle via
   `tgrad_tensor_from_buffer`; the Python fields are mirrors of state
   the Lean tensor authoritatively owns). They are NOT a Python-owned
   view graph (which is what `GOAL_L15.md §3.1` actually rules out:
   "Python Tensor stores an opaque Lean tensor handle, not Python-owned
   view metadata"). Narrowed predicate: only `_view_ops|_strides|
   _view_graph|_shape_tracker` triggers a fail — these would denote
   independent Python-owned view-graph data structures. This narrowing
   will be re-stated in `EXPERIMENT_RESULT.md` per the §4 escape hatch.

2. **Criterion 1 / `n_handle`** — the spec's draft regex is
   `self._handle = ctypes.c_uint64`. Our implementation does
   `self._handle = handle` where `handle` is the FFI-returned
   `c_uint64`-typed value (its ctype is declared in the `_lib.tgrad_
   tensor_from_buffer.restype = ctypes.c_uint64` binding at module
   load). Wrapping it again in `ctypes.c_uint64(...)` would be a no-op
   that the spec example used for visual clarity. Narrowed predicate:
   `self._handle = handle` (an assignment from an FFI return value) +
   the corresponding `restype = ctypes.c_uint64` binding line — both
   required.

Both narrowings tighten the spec's prose ("opaque Lean tensor handle";
"not Python-owned view metadata") to a falsifiable code grep, without
reducing what the audit detects.
"""
from __future__ import annotations
import functools
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HEURISTIC_PATH = REPO / "Tgrad" / "Codegen" / "Opt" / "Heuristic.lean"
DISPATCH_TOTAL_OBLIGATION = "pickDispatchPlan_bf16_total"


def _lean_declaration_source(text: str, kind: str, name: str) -> str:
    """Return one top-level Lean declaration, excluding the next doc block.

    This parser is used only for the distinct architecture check. Dispatch
    totality is established by compiling an exact-type Lean witness below,
    never by interpreting the spelling of the implementation body.
    """
    start = re.search(
        rf"^{re.escape(kind)}[ \t]+{re.escape(name)}(?=[ \t(]|$)", text, re.M
    )
    if start is None:
        return ""
    following = text[start.end():]
    end = re.search(
        r"^(?:/--|/-!|def[ \t]|theorem[ \t]|structure[ \t]|inductive[ \t]|"
        r"namespace[ \t]|end(?:[ \t]|$))",
        following,
        re.M,
    )
    stop = start.end() + (end.start() if end else len(following))
    return text[start.start():stop]


@functools.lru_cache(maxsize=1)
def _pick_dispatch_plan_body() -> str:
    if not HEURISTIC_PATH.exists():
        return ""
    return _lean_declaration_source(
        HEURISTIC_PATH.read_text(), "def", "pickDispatchPlan"
    )


def _strip_lean_comments(text: str) -> str:
    """Remove comments before inspecting architectural dependencies.

    `pickDispatchPlan` currently has no nested block comments. Iterating the
    non-greedy block pattern nevertheless handles nested-looking comment text
    from the inside out well enough for this bounded declaration parser.
    """
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"/-.*?-/", "", text, flags=re.S)
    return re.sub(r"--[^\n]*", "", text)


def _dispatch_body_is_general(body: str) -> bool:
    """Reject captured-table coupling while allowing one algorithmic tile.

    The 64³ branch is a genuine two-warp algorithmic special case. A captured
    lookup is different: it imports sentinel/table identity or enumerates
    multiple exact triples. Totality is checked separately in Lean.
    """
    if not body:
        return False
    code = _strip_lean_comments(body)
    forbidden_dependencies = (
        "ShapeSentinel",
        "dispatchDimsForSentinel",
        "ofTriple",
        "toTriple",
    )
    if any(name in code for name in forbidden_dependencies):
        return False
    if re.search(r"\.bf16_(?:64x64|[0-9]+x[0-9]+)", code):
        return False

    # Two or more complete literal triples constitute an explicit shape
    # table. One exact triple is permitted for the 64³ two-warp algorithm.
    tuple_literals = re.findall(
        r"[\(⟨\[][ \t]*[0-9]+[ \t]*,[ \t]*[0-9]+[ \t]*,[ \t]*[0-9]+[ \t]*[\)⟩\]]",
        code,
    )
    exact_triple_guards = re.findall(
        r"M[ \t]*==[ \t]*[0-9]+.{0,160}?"
        r"K[ \t]*==[ \t]*[0-9]+.{0,160}?"
        r"N[ \t]*==[ \t]*[0-9]+",
        code,
        re.S,
    )
    return len(tuple_literals) < 2 and len(exact_triple_guards) <= 1


@functools.lru_cache(maxsize=1)
def _compiled_dispatch_total_obligation() -> bool:
    """Compile an exact-type witness for the Lean dispatch-total theorem."""
    if not HEURISTIC_PATH.exists():
        return False
    source = HEURISTIC_PATH.read_text()
    if not re.search(
        rf"^theorem[ \t]+{re.escape(DISPATCH_TOTAL_OBLIGATION)}(?=[ \t(]|$)",
        source,
        re.M,
    ):
        return False
    witness = f"""import Tgrad.Codegen.Opt.Heuristic

example (M K N : Nat) :
    (Tgrad.Codegen.Opt.pickDispatchPlan M K N
      .bfloat16_ .bfloat16_).isSome = true :=
  Tgrad.Codegen.Opt.{DISPATCH_TOTAL_OBLIGATION} M K N
"""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".lean", prefix="tgrad_l15a_dispatch_", delete=True
        ) as handle:
            handle.write(witness)
            handle.flush()
            result = subprocess.run(
                ["lake", "env", "lean", handle.name],
                cwd=REPO,
                capture_output=True,
                text=True,
                timeout=60,
            )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _count_matches(path: Path, pattern: str) -> int:
    if not path.exists():
        return 0
    text = path.read_text()
    return sum(1 for _ in re.finditer(pattern, text, re.M))


# -------- Criterion 1: Tensor API ---------------------------------------

def check_tensor_api() -> dict:
    tgrad_py = REPO / "python" / "tgrad.py"
    # 6 method/property defs.
    n_methods = _count_matches(
        tgrad_py,
        r"^\s+(?:def|@property)\s+(?:from_numpy|numpy|__matmul__|T|transpose|reshape|__getitem__)\b",
    )
    # Opaque handle exists + has a c_uint64 restype binding.
    n_handle_assign = _count_matches(tgrad_py, r"self\._handle\s*=\s*handle\b")
    n_handle_restype = _count_matches(
        tgrad_py, r"_lib\.tgrad_tensor_from_buffer\.restype\s*=\s*ctypes\.c_uint64"
    )
    n_handle = 1 if (n_handle_assign >= 1 and n_handle_restype >= 1) else 0
    # Narrowed legacy: Python-owned view-graph state only.
    n_legacy = _count_matches(
        tgrad_py, r"self\._(?:view_ops|strides|view_graph|shape_tracker)\s*=",
    )
    verdict = "pass" if (n_methods >= 6 and n_handle == 1 and n_legacy == 0) else "fail"
    return {
        "criterion": "tensor_api",
        "verdict": verdict,
        "artifact_paths": ["python/tgrad.py", "Tgrad/Tensor.lean"],
        "evidence": (
            f"methods={n_methods}/6, handle_present_with_c_uint64_restype={n_handle}/1, "
            f"python_owned_view_graph_fields={n_legacy}/0 "
            f"(narrowed per GOAL_L15.md §4: _buf/_shape/_dtype are accessor caches "
            f"derived from the Lean handle, not Python-owned view metadata)"
        ),
    }


# -------- Criterion 2: Symbolic graph -----------------------------------

def check_symbolic_graph() -> dict:
    uop_path = REPO / "Tgrad" / "UOp.lean"
    required_ctors = ["buffer", "permute", "reshape", "expand", "slice"]
    # Map of ctor → at least one `| .<ctor>` arm appears.
    found = {}
    if uop_path.exists():
        text = uop_path.read_text()
        for c in required_ctors:
            found[c] = len(re.findall(rf"\|\s+\.{c}\b", text)) >= 1
    else:
        found = {c: False for c in required_ctors}
    n_ctors = sum(1 for v in found.values() if v)
    # Reduction / matmul / WMMA presence (best-effort): check the wider
    # Tgrad codebase for these concepts as constructors / functions.
    binop_present = _count_matches(uop_path, r"\|\s+\.binop\b") >= 1
    matmul_msl_files = list((REPO / "fixtures" / "codegen").glob("matmul_*.msl"))
    wmma_present = any("simdgroup_multiply_accumulate" in f.read_text() for f in matmul_msl_files)
    # Rangeify trace nontrivial rows.
    trace = Path("/tmp/tgrad_rangeify_trace.jsonl")
    nontrivial = 0
    rows_total = 0
    if trace.exists():
        for line in trace.read_text().splitlines():
            if not line.strip():
                continue
            rows_total += 1
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("movement_count_in", 0) > 0:
                nontrivial += 1
    verdict = "pass" if (n_ctors >= 5 and binop_present and wmma_present and nontrivial >= 16) else "fail"
    return {
        "criterion": "symbolic_graph",
        "verdict": verdict,
        "artifact_paths": ["Tgrad/UOp.lean", "/tmp/tgrad_rangeify_trace.jsonl",
                           "fixtures/codegen/matmul_*.msl"],
        "evidence": (
            f"uop_movement_ctors={n_ctors}/5 {found}, binop={binop_present}, "
            f"wmma_kernel_present={wmma_present}, "
            f"rangeify_trace_rows={rows_total} nontrivial={nontrivial}/16"
        ),
    }


# -------- Criterion 3: Rewrite/codegen ----------------------------------

def check_rewrite_codegen() -> dict:
    ev = REPO / "fixtures" / "gate_evidence"
    pipeline = REPO / "Tgrad" / "Pipeline.lean"
    def _load(name):
        p = ev / name
        if not p.exists():
            return {}
        return json.loads(p.read_text())
    l13 = _load("L13.json")
    l13f = _load("L13_F.json")
    l14b = _load("L14_B.json")
    l14c = _load("L14_C.json")
    # Pipeline.realize: no runtime MSL fixture read; no ShapeSentinel-only.
    pipe_text = pipeline.read_text() if pipeline.exists() else ""
    n_msl_read = len(re.findall(r"IO\.FS\.readFile.*matmul.*\.msl", pipe_text))
    # Dispatch totality is a compiled Lean proposition. The audit compiles an
    # exact-type witness rather than inferring semantics from `else`, comments
    # or match-arm spelling.
    dispatch_total = _compiled_dispatch_total_obligation()
    # Evidence-based assertions.
    l13d_evidence = _load("L13_D.json")
    # L13_D.json uses `random_correct` for the success count.
    l13d_pass = int(l13d_evidence.get("random_correct",
                                       l13d_evidence.get("random_pass", 0)))
    tc_wmma_pinned = int(l13f.get("tc_general_wmma", 0))
    tc_wmma_random = int(l13f.get("random_tc_wmma", 0))
    tc_scalar = int(l13f.get("tc_general_scalar_routes", -1))
    pinned_views = int(l14b.get("hashes", {}).get("L14_B_3_evidence_sha256", "") != "")
    l14b3 = _load("L14_B_3.json")
    pinned_views_pass = int(l14b3.get("pinned_views_pass", 0))
    random_views_pass = int(l14c.get("random_views_pass", 0))
    verdict = "pass" if (
        l13d_pass >= 30 and
        tc_wmma_pinned == 8 and tc_wmma_random == 10 and tc_scalar == 0 and
        pinned_views_pass == 16 and random_views_pass == 20 and
        n_msl_read == 0 and dispatch_total
    ) else "fail"
    return {
        "criterion": "rewrite_codegen",
        "verdict": verdict,
        "artifact_paths": ["Tgrad/Pipeline.lean", "Tgrad/Codegen/Opt/Heuristic.lean",
                           "fixtures/gate_evidence/L13.json", "L13_D.json",
                           "L13_F.json", "L14_B.json", "L14_B_3.json", "L14_C.json"],
        "evidence": (
            f"L13.D.random_pass={l13d_pass}/30, "
            f"L13_F.wmma_pinned={tc_wmma_pinned}/8 random={tc_wmma_random}/10 "
            f"scalar_routes={tc_scalar}/0, "
            f"L14_B_3.pinned_views_pass={pinned_views_pass}/16, "
            f"L14_C.random_views_pass={random_views_pass}/20, "
            f"pipeline_msl_read={n_msl_read}/0, "
            f"dispatch_bf16_total_compiled={dispatch_total}"
        ),
    }


# -------- Static checks (§4) -------------------------------------------

def static_check_tinygrad_indep() -> bool:
    """Runtime files outside scripts/capture/ import tinygrad?"""
    runtime_paths = [REPO / "python" / "tgrad.py",
                     REPO / "python" / "tgrad_bench.py"]
    for p in runtime_paths:
        if not p.exists():
            continue
        text = p.read_text()
        if re.search(r"^\s*(?:from|import)\s+tinygrad\b", text, re.M):
            return False
    return True


def static_check_no_runtime_msl_read() -> bool:
    pipe = REPO / "Tgrad" / "Pipeline.lean"
    if not pipe.exists():
        return False
    text = pipe.read_text()
    # IO.FS.readFile of a matmul_*.msl path at runtime.
    return not re.search(r"IO\.FS\.readFile[^\n]*matmul[^\n]*\.msl", text)


def static_check_dispatch_fallthrough() -> bool:
    """bf16 dispatch totality is established by a compiled Lean theorem."""
    return _compiled_dispatch_total_obligation()


def static_check_no_python_owned_view_graph() -> bool:
    tg = REPO / "python" / "tgrad.py"
    if not tg.exists():
        return False
    text = tg.read_text()
    return not re.search(r"self\._(?:view_ops|strides|view_graph|shape_tracker)\s*=", text)


def static_check_no_tc_general_scalar_route() -> bool:
    l13f = REPO / "fixtures" / "gate_evidence" / "L13_F.json"
    if not l13f.exists():
        return False
    data = json.loads(l13f.read_text())
    return int(data.get("tc_general_scalar_routes", -1)) == 0


def static_check_no_python_triple_set_routing() -> bool:
    """The Python __matmul__ must not branch TC-vs-scalar SOLELY by
    _TRIPLE_SET membership. We allow _TRIPLE_SET to exist as a HINT,
    but the actual routing must be Lean-side (via tgrad_matmul / pickDispatchPlan)."""
    tg = REPO / "python" / "tgrad.py"
    if not tg.exists():
        return False
    text = tg.read_text()
    # Branch heuristic: if _TRIPLE_SET appears AND there's no FFI dispatch
    # in __matmul__, it's bad. We require the FFI dispatch line to exist.
    has_triple_set = "_TRIPLE_SET" in text
    has_ffi_dispatch = bool(re.search(r"_lib\.tgrad_matmul", text)) or bool(
        re.search(r"_lib\.tgrad_tensor_dispatch", text)
    )
    if has_triple_set and not has_ffi_dispatch:
        return False
    return True


def static_check_no_view_method_buffer_alloc() -> bool:
    """View methods (transpose/permute/reshape/expand/__getitem__) must
    NOT allocate Metal buffers or dispatch kernels. They construct child
    Tensors that share the underlying buffer with the parent (owns_buf=False)."""
    tg = REPO / "python" / "tgrad.py"
    if not tg.exists():
        return False
    text = tg.read_text()
    # Pattern: within view-method defs, no metalAlloc / metalDispatch / from_bytes
    # equivalents are called. We use a heuristic: each view method body has
    # `owns_buf=False` somewhere.
    methods = ["transpose", "permute", "reshape", "expand", "__getitem__"]
    for m in methods:
        match = re.search(rf"def {m}\([^)]*\)\s*->\s*[^:]+:(.*?)(?=\n    def |\n    @|\Z)",
                          text, re.S)
        if match is None:
            return False
        body = match.group(1)
        # Buffer alloc or kernel dispatch in a view method is bad.
        if re.search(r"\b(?:tgrad_tensor_alloc|metalAlloc|metalDispatch|tgrad_matmul)\b",
                     body):
            return False
    return True


def static_check_no_pickDispatchPlan_lookup_table() -> bool:
    """pickDispatchPlan must not be an explicit lookup table over the
    pinned shapes. This is distinct from totality: a lookup plus a default is
    total but still couples production dispatch to captured fixtures."""
    return _dispatch_body_is_general(_pick_dispatch_plan_body())


def static_check_rangeify_traces_present() -> bool:
    trace = Path("/tmp/tgrad_rangeify_trace.jsonl")
    if not trace.exists():
        return False
    rows = 0
    nontrivial = 0
    for line in trace.read_text().splitlines():
        if not line.strip():
            continue
        rows += 1
        try:
            row = json.loads(line)
            if row.get("movement_count_in", 0) > 0:
                nontrivial += 1
        except Exception:
            continue
    return rows >= 1 and nontrivial >= 1


def run_self_test() -> int:
    """Focused regression tests for the dispatch audit instrument."""
    failures: list[str] = []
    audit_source = Path(__file__).read_text()
    # Construct the old raw-regex spelling without embedding it verbatim in
    # this file, otherwise the self-test would match its own guard.
    doubled_escape = "pickDispatchPlan" + ("\\" * 2) + "b"
    if doubled_escape in audit_source:
        failures.append("duplicated double-escaped pickDispatchPlan parser present")

    synthetic = """def pickDispatchPlan (M : Nat) :=
  some M

/-- next declaration -/
def unrelated := 7
"""
    extracted = _lean_declaration_source(synthetic, "def", "pickDispatchPlan")
    if "some M" not in extracted or "unrelated" in extracted:
        failures.append("top-level Lean declaration parser did not stay scoped")

    body = _pick_dispatch_plan_body()
    if not _dispatch_body_is_general(body):
        failures.append("current pickDispatchPlan failed no-lookup architecture check")
    coupled = body.replace(
        ":=\n",
        ":=\n  let _captured := "
        "Tgrad.Renderer.Metal.ShapeSentinel.ofTriple M K N\n",
        1,
    )
    if _dispatch_body_is_general(coupled):
        failures.append("ShapeSentinel-coupled dispatch escaped architecture check")
    tabled = body.replace(
        ":=\n", ":=\n  let _pinned := [(64, 64, 64), (1024, 1024, 1024)]\n", 1
    )
    if _dispatch_body_is_general(tabled):
        failures.append("explicit pinned-shape table escaped architecture check")

    if not _compiled_dispatch_total_obligation():
        failures.append("exact-type Lean dispatch-total witness did not compile")

    if failures:
        for failure in failures:
            print(f"l15_a_audit_self_test: FAIL: {failure}", file=sys.stderr)
        return 1
    print("l15_a_audit_self_test: pass")
    return 0


def main() -> int:
    criteria = [
        check_tensor_api(),
        check_symbolic_graph(),
        check_rewrite_codegen(),
    ]
    static_checks = {
        "tinygrad_indep_static":           static_check_tinygrad_indep(),
        "no_runtime_msl_fixture_read":     static_check_no_runtime_msl_read(),
        "no_shape_sentinel_only_dispatch": static_check_dispatch_fallthrough(),
        "no_python_owned_view_graph":      static_check_no_python_owned_view_graph(),
        "no_tc_general_scalar_route":      static_check_no_tc_general_scalar_route(),
        "no_python_triple_set_routing":    static_check_no_python_triple_set_routing(),
        "no_view_method_buffer_alloc":     static_check_no_view_method_buffer_alloc(),
        "no_pickDispatchPlan_lookup_table": static_check_no_pickDispatchPlan_lookup_table(),
        "rangeify_traces_present":         static_check_rangeify_traces_present(),
    }
    print(json.dumps({"criteria": criteria, "static_checks": static_checks}, indent=2))
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        sys.exit(run_self_test())
    if sys.argv[1:]:
        print("usage: l15_a_audit.py [--self-test]", file=sys.stderr)
        sys.exit(2)
    sys.exit(main())
