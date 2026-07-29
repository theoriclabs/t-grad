#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

shared_module="Tgrad/Backend/FillPlan.lean"

# Falsifiability recipe: change DeviceProfile.beq to compare compiler mode and
# tool but omit compiler version. The Lean executable must still build, then
# fail the typed semantic and kernel identity separation assertions even though
# the derived serialization still contains the version.
# A second boundary recipe changes CopyInRequest.hostBytes to return empty.
# Both witness leaves must then reject their otherwise valid nonempty H2D call.
# The compiler-negative fixtures below are also load-bearing: making any sealed
# constructor public, weakening an ABI proof field, exposing a witness probe
# credential, or permitting a renderer without an exact semantic projection
# must make the corresponding fixture compile and therefore turn this gate red.

if grep -Eqi '(cuda|rocm|nvrtc|nvcc|hiprtc|hipcc|sm_|gfx)|(^|[^[:alnum:]_])hip([^[:alnum:]_]|$)' "$shared_module"; then
  echo "FAIL shared spine contains vendor-specific identity or dialect"
  exit 1
fi
echo "PASS shared spine excludes vendor-specific identity and dialect"

if grep -Eq '^import Tgrad\.(Renderer|Runtime)' "$shared_module"; then
  echo "FAIL shared spine imports a vendor renderer or runtime"
  exit 1
fi
echo "PASS shared spine has no renderer/runtime dependency"

lake build backend-shared-tests
.lake/build/bin/backend-shared-tests

negative_dir="$(mktemp -d "$repo_root/.lake/backend-shared-negative.XXXXXX")"
cleanup_negative_dir() {
  case "$negative_dir" in
    "$repo_root"/.lake/backend-shared-negative.*) rm -rf -- "$negative_dir" ;;
    *) echo "FAIL refusing to clean unexpected negative-fixture path" >&2 ;;
  esac
}
trap cleanup_negative_dir EXIT

expect_lean_failure() {
  local name="$1"
  local diagnostic="$2"
  local fixture="$negative_dir/$name.lean"
  local output="$negative_dir/$name.out"
  if lake env lean "$fixture" >"$output" 2>&1; then
    echo "FAIL compiler-negative fixture unexpectedly compiled: $name"
    exit 1
  fi
  if ! grep -Eq "$diagnostic" "$output"; then
    echo "FAIL compiler-negative fixture missed intended diagnostic: $name"
    sed -n '1,120p' "$output"
    exit 1
  fi
  echo "PASS compiler-negative fixture rejected: $name"
}

cat >"$negative_dir/false_abi_witness.lean" <<'EOF'
import Tgrad.Backend.FillPlan

open Tgrad.Backend

def forgedAbiU64 : AbiU64 := {
  value := maxAbiU64 + 1
  fits := by decide
}
EOF
expect_lean_failure false_abi_witness 'Tactic `decide` proved that the proposition|of_decide_eq_true|failed to synthesize'

cat >"$negative_dir/false_abi_u32_witness.lean" <<'EOF'
import Tgrad.Backend.FillPlan

open Tgrad.Backend

def forgedAbiU32 : AbiU32 := {
  value := maxAbiU32 + 1
  fits := by decide
}
EOF
expect_lean_failure false_abi_u32_witness 'Tactic `decide` proved that the proposition|of_decide_eq_true|failed to synthesize'

for authority in FillPlan SourceArtifact CompileRequest AvailableCapability \
    AuthorizedPlan BoundBuffer BoundCompiledKernel CopyInRequest CopyOutRequest; do
  cat >"$negative_dir/private_${authority}.lean" <<EOF
import Tgrad.Backend.FillPlan

open Tgrad.Backend

#check ${authority}.mk
EOF
  expect_lean_failure "private_${authority}" "Unknown constant.*${authority}\\.mk"
done

for credential in WitnessCredentialA WitnessCredentialB; do
  cat >"$negative_dir/private_${credential}.lean" <<EOF
import BackendSharedTests

#check ${credential}
EOF
  expect_lean_failure "private_${credential}" "Unknown identifier.*${credential}"
done

cat >"$negative_dir/constant_renderer.lean" <<'EOF'
import Tgrad.Backend.FillPlan

open Tgrad.Backend

def constantRenderer (identity : RendererContractIdentity) :
    RendererContract RenderedSemantics := {
  identity
  render := fun _ => {
    scalar := .int32 0
    elementCount := 1
    outputIndex := .blockOnlyX
    bounds := .unguarded
  }
  projectSemantics := some
  renderPreserves := by
    intro plan
    rfl
}
EOF
expect_lean_failure constant_renderer 'tactic.*rfl.*failed|rfl.*failed|type mismatch'
