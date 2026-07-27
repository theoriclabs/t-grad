#!/usr/bin/env bash
# Gate L1 — type system & graph-rewrite engine.
#
# Verifies Dtype + Shape + UOp + UPat + GraphRewrite + Rules.Symbolic
# (16-rule subset) lift cleanly with their theorems preserved. The
# behavioural checks (lub table, cast table, shape ops, symbolic DAG)
# are cross-validated against captured tinygrad output — the agent
# cannot fake the matches because they don't control the inputs.
#
# Anti-shortcut design: each predicate is verified by the gate runner,
# not by inspecting tgrad-tests stdout. Tgrad code must structurally
# contain the theorems; the runner cross-checks them against fixtures
# captured from tinygrad.
set -euo pipefail
if [[ -z "${REPO_ROOT:-}" ]]; then
  export REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
if [[ -z "${TGRAD_DIR:-}" ]]; then
  export TGRAD_DIR="$REPO_ROOT"
fi
source "$TGRAD_DIR/scripts/lib/checks.sh"
L1_LUB="$(tgrad_run_path L1_lub.json)"
L1_CAST="$(tgrad_run_path L1_cast.json)"
L1_SHAPE="$(tgrad_run_path L1_shape.json)"
L1_MV="$(tgrad_run_path L1_mv.json)"
L1_SYM="$(tgrad_run_path L1_sym.json)"
L1_NEG_LEAN="$(tgrad_run_path L1_negative_test.lean)"
L1_NEG_LOG="$(tgrad_run_path L1_negative_test.log)"

echo "[L1] types & graph-rewrite engine"
run_preflight

# -----------------------------------------------------------------------
# Predicate 1: required Tgrad modules exist.
# -----------------------------------------------------------------------
required_modules=(
  Tgrad/Dtype.lean
  Tgrad/Shape.lean
  Tgrad/UOp.lean
  Tgrad/UPat.lean
  Tgrad/GraphRewrite.lean
  Tgrad/Rules/Symbolic.lean
)
for m in "${required_modules[@]}"; do
  [[ -f "$REPO_ROOT/$m" ]] || { echo "  ✗ missing required module: $m"; exit 1; }
done
echo "  ✓ all 6 required modules present"

# -----------------------------------------------------------------------
# Predicate 2: required theorems are declared.
# A `theorem` declaration that survived `lake build` (which preflight
# enforced) proved. `axiom`/`sorry` already rejected by preflight.
# -----------------------------------------------------------------------
required_theorems=(
  "Tgrad/Dtype.lean:lub_comm_holds"
  "Tgrad/Dtype.lean:lub_assoc_holds"
  "Tgrad/Dtype.lean:canLosslessCast_self"
  "Tgrad/Dtype.lean:canLosslessCast_from_bool"
  "Tgrad/Shape.lean:numel_nil"
  "Tgrad/Shape.lean:numel_append"
  "Tgrad/Shape.lean:sintNumel_lift"
  "Tgrad/Shape.lean:reshape_preserves_numel_concrete"
)
for entry in "${required_theorems[@]}"; do
  file="${entry%:*}"; thm="${entry##*:}"
  if ! grep -qE "^theorem[[:space:]]+$thm\b" "$REPO_ROOT/$file"; then
    echo "  ✗ missing theorem: $thm in $file"
    exit 1
  fi
done
echo "  ✓ all 8 required theorems declared (and proved — preflight rejected sorry/axiom)"

# -----------------------------------------------------------------------
# Predicate 3: fixtures lifted to fixtures/.
# -----------------------------------------------------------------------
required_fixtures=(
  fixtures/dtype/lub_table.json
  fixtures/dtype/can_lossless_cast_table.json
  fixtures/shape/shape_table.json
  fixtures/shape/movement_table.json
  fixtures/symbolic/dag_in.json
  fixtures/symbolic/dag_out_expected.json
)
for f in "${required_fixtures[@]}"; do
  [[ -f "$REPO_ROOT/$f" ]] || { echo "  ✗ missing required fixture: $f"; exit 1; }
done
echo "  ✓ all 6 required fixtures present in fixtures/"

# -----------------------------------------------------------------------
# Predicate 4: cross-validate behaviour against captures + tinygrad.
#
# Each subcommand below asks Tgrad to compute something, then the gate
# diffs the output byte-for-byte against the captured ground truth.
#
# Anti-shortcut property: the agent cannot trivially make these pass
# unless `Tgrad.Dtype.lub` actually agrees with tinygrad's
# `least_upper_dtype` on every dtype pair, etc.
# -----------------------------------------------------------------------

# Sub-predicate 4a: lub table — Tgrad recomputes the 14×14 lub table,
# emits as JSON; diff against the captured table.
./.lake/build/bin/tgrad-cli emit-lub-table >"$L1_LUB" 2>&1 || {
  echo "  ✗ tgrad-cli emit-lub-table failed"; cat "$L1_LUB"; exit 1
}
if ! diff -q "$L1_LUB" "$TGRAD_DIR/fixtures/dtype/lub_table.json" >/dev/null; then
  echo "  ✗ Tgrad.Dtype.lub disagrees with captured lub table"
  diff "$L1_LUB" "$TGRAD_DIR/fixtures/dtype/lub_table.json" | head -20
  exit 1
fi
echo "  ✓ Tgrad.Dtype.lub matches captured 14×14 lub table"

# Sub-predicate 4b: cast table.
./.lake/build/bin/tgrad-cli emit-cast-table >"$L1_CAST" 2>&1 || {
  echo "  ✗ tgrad-cli emit-cast-table failed"; cat "$L1_CAST"; exit 1
}
if ! diff -q "$L1_CAST" "$TGRAD_DIR/fixtures/dtype/can_lossless_cast_table.json" >/dev/null; then
  echo "  ✗ Tgrad.Dtype.canLosslessCast disagrees with captured table"
  exit 1
fi
echo "  ✓ Tgrad.Dtype.canLosslessCast matches captured 14×14 cast table"

# Sub-predicate 4c: shape ops.
./.lake/build/bin/tgrad-cli emit-shape-table >"$L1_SHAPE" 2>&1 || {
  echo "  ✗ tgrad-cli emit-shape-table failed"; cat "$L1_SHAPE"; exit 1
}
if ! diff -q "$L1_SHAPE" "$TGRAD_DIR/fixtures/shape/shape_table.json" >/dev/null; then
  echo "  ✗ Tgrad.Shape disagrees with captured shape table"
  exit 1
fi
echo "  ✓ Tgrad.Shape (numel/align/broadcast) matches captured table"

./.lake/build/bin/tgrad-cli emit-movement-table >"$L1_MV" 2>&1 || {
  echo "  ✗ tgrad-cli emit-movement-table failed"; cat "$L1_MV"; exit 1
}
if ! diff -q "$L1_MV" "$TGRAD_DIR/fixtures/shape/movement_table.json" >/dev/null; then
  echo "  ✗ Tgrad.Shape movement ops disagree with captured table"
  exit 1
fi
echo "  ✓ Tgrad.Shape.{reshape,permute,expand} matches captured 15 cases"

# Sub-predicate 4d: symbolic reduction on phase 03's 47-node DAG.
./.lake/build/bin/tgrad-cli reduce-symbolic-dag \
    "$TGRAD_DIR/fixtures/symbolic/dag_in.json" \
    >"$L1_SYM" 2>&1 || {
  echo "  ✗ tgrad-cli reduce-symbolic-dag failed"; cat "$L1_SYM"; exit 1
}
if ! diff -q "$L1_SYM" "$TGRAD_DIR/fixtures/symbolic/dag_out_expected.json" >/dev/null; then
  echo "  ✗ Tgrad.GraphRewrite.run + Tgrad.Rules.Symbolic 16-rule subset disagrees with captured output"
  exit 1
fi
echo "  ✓ Tgrad.GraphRewrite reduces phase 03's 47-node DAG to captured output"

# -----------------------------------------------------------------------
# Predicate 5: negative tests — broken inputs should fail to typecheck.
#
# Anti-shortcut: prove the type system actually rejects what it should.
# -----------------------------------------------------------------------
# Try compiling a deliberately broken UOp construction; expect failure.
cat >"$L1_NEG_LEAN" <<'EOF'
-- Deliberately broken: indexing should require a UOp buffer + UOp offset.
-- A stringly-typed call should not typecheck.
import Tgrad
open Tgrad
def badUOp : UOp := .index "this is not a UOp" "neither is this"
EOF
if (cd "$TGRAD_DIR" && lake env lean "$L1_NEG_LEAN") >"$L1_NEG_LOG" 2>&1; then
  echo "  ✗ negative test compiled — type system isn't rejecting bad UOp"
  exit 1
fi
echo "  ✓ negative test correctly rejected by typechecker"

# -----------------------------------------------------------------------
# Evidence: write hashes of outputs so re-runs are deterministic.
# -----------------------------------------------------------------------
ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
host="$(hostname)"; plat="$(uname -srm)"
lub_hash="$(shasum -a 256 "$L1_LUB" | awk '{print $1}')"
cast_hash="$(shasum -a 256 "$L1_CAST" | awk '{print $1}')"
shape_hash="$(shasum -a 256 "$L1_SHAPE" | awk '{print $1}')"
mv_hash="$(shasum -a 256 "$L1_MV" | awk '{print $1}')"
sym_hash="$(shasum -a 256 "$L1_SYM" | awk '{print $1}')"
cat >"$TGRAD_DIR/fixtures/gate_evidence/L1.json" <<EOF
{
  "gate": "L1",
  "ts_utc": "$ts",
  "host": "$host",
  "platform": "$plat",
  "commit": "$commit",
  "hashes": {
    "lub_table_sha256":   "$lub_hash",
    "cast_table_sha256":  "$cast_hash",
    "shape_table_sha256": "$shape_hash",
    "movement_table_sha256": "$mv_hash",
    "symbolic_dag_sha256":   "$sym_hash"
  }
}
EOF
check_evidence_for L1 || exit 1
check_falsifiability_verified L1 || exit 1
echo "  ✓ L1 type-system gate green (evidence recorded)"
