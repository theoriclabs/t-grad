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

expect_lean_success() {
  local name="$1"
  local fixture="$negative_dir/$name.lean"
  local output="$negative_dir/$name.out"
  if ! lake env lean "$fixture" >"$output" 2>&1; then
    echo "FAIL compiler-positive fixture did not compile: $name"
    sed -n '1,120p' "$output"
    exit 1
  fi
  echo "PASS compiler-positive fixture accepted: $name"
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

for authority in FillPlan SourceArtifact CompileRequest CompleteProfileObservation \
    AvailableCapability AuthorizedPlan BoundBuffer BoundCompiledKernel CopyInRequest \
    CopyOutRequest; do
  cat >"$negative_dir/private_${authority}.lean" <<EOF
import Tgrad.Backend.FillPlan

open Tgrad.Backend

#check ${authority}.mk
EOF
  expect_lean_failure "private_${authority}" "Unknown constant.*${authority}\\.mk"
done

cat >"$negative_dir/capability_has_no_credential.lean" <<'EOF'
import Tgrad.Backend.FillPlan

open Tgrad.Backend

def extractCredential {Credential : Type} (authority : ProbeAuthority Credential)
    (capability : AvailableCapability authority) : Credential :=
  capability.credential
EOF
expect_lean_failure capability_has_no_credential 'Invalid field.*credential|Unknown identifier.*credential'

cat >"$negative_dir/capability_cannot_remint.lean" <<'EOF'
import Tgrad.Backend.FillPlan

open Tgrad.Backend

noncomputable def remintAtNextOrdinal {Credential : Type}
    (authority : ProbeAuthority Credential)
    (capability : AvailableCapability authority) :
    ProbeObservation authority String :=
  profileObservationFromProbe authority
    capability.credential
    (capability.count + 1) (capability.deviceOf.ordinal + 1)
    (.complete capability.deviceOf.profile) "forged remint"
EOF
expect_lean_failure capability_cannot_remint 'Invalid field.*credential|Unknown identifier.*credential'

cat >"$negative_dir/capability_recursor_has_no_credential.lean" <<'EOF'
import Tgrad.Backend.FillPlan

open Tgrad.Backend

noncomputable def extractCapabilityCredentialViaRecursor {Credential : Type}
    {authority : ProbeAuthority Credential}
    (capability : AvailableCapability authority) : Credential :=
  AvailableCapability.rec (motive := fun _ => Credential)
    (fun credential _ _ _ _ _ _ => credential) capability
EOF
expect_lean_failure capability_recursor_has_no_credential 'Application type mismatch|function expected at|Type mismatch'

cat >"$negative_dir/capability_cannot_rebind_count.lean" <<'EOF'
import Tgrad.Backend.FillPlan

open Tgrad.Backend

def rebindCapabilityCount {Credential : Type}
    {authority : ProbeAuthority Credential}
    (capability : AvailableCapability authority) :
    AvailableCapability authority :=
  { capability with deviceCount := capability.count + 1 }
EOF
expect_lean_failure capability_cannot_rebind_count 'invalid \{\.\.\.\} notation.*private|Unknown constant.*AvailableCapability\.mk'

cat >"$negative_dir/capability_cannot_rebind_device.lean" <<'EOF'
import Tgrad.Backend.FillPlan

open Tgrad.Backend

def rebindCapabilityDevice {Credential : Type}
    {authority : ProbeAuthority Credential}
    (capability : AvailableCapability authority) :
    AvailableCapability authority :=
  { capability with
    device := { capability.deviceOf with ordinal := capability.count } }
EOF
expect_lean_failure capability_cannot_rebind_device 'invalid \{\.\.\.\} notation.*private|Unknown constant.*AvailableCapability\.mk'

cat >"$negative_dir/observation_has_no_credential.lean" <<'EOF'
import Tgrad.Backend.FillPlan

open Tgrad.Backend

noncomputable def extractObservationCredential {Credential : Type}
    {authority : ProbeAuthority Credential}
    (observation : CompleteProfileObservation authority) : Credential :=
  Classical.choose observation.profileAuthorized
EOF
expect_lean_failure observation_has_no_credential 'Field `profileAuthorized`.*is private|Invalid field.*profileAuthorized|Unknown identifier.*profileAuthorized'

cat >"$negative_dir/probe_pattern_has_no_credential.lean" <<'EOF'
import Tgrad.Backend.FillPlan

open Tgrad.Backend

def extractProfiledCredential {Credential Detail : Type}
    {authority : ProbeAuthority Credential} :
    ProbeObservation authority Detail → Option Credential
  | .profiled credential _ _ _ _ => some credential
  | _ => none
EOF
expect_lean_failure probe_pattern_has_no_credential 'Function expected at|Invalid pattern|Application type mismatch|Type mismatch|invalid alternative'

cat >"$negative_dir/observation_cannot_rebind_count.lean" <<'EOF'
import Tgrad.Backend.FillPlan

open Tgrad.Backend

def rebindObservationCount {Credential : Type}
    {authority : ProbeAuthority Credential}
    (observation : CompleteProfileObservation authority) :
    CompleteProfileObservation authority :=
  { observation with deviceCount := observation.deviceCount + 1 }
EOF
expect_lean_failure observation_cannot_rebind_count 'invalid \{\.\.\.\} notation.*private|Unknown constant.*CompleteProfileObservation\.mk|Invalid field.*deviceCountPositive'

cat >"$negative_dir/observation_cannot_rebind_device.lean" <<'EOF'
import Tgrad.Backend.FillPlan

open Tgrad.Backend

def rebindObservationDevice {Credential : Type}
    {authority : ProbeAuthority Credential}
    (observation : CompleteProfileObservation authority) :
    CompleteProfileObservation authority :=
  { observation with
    device := { observation.device with ordinal := observation.device.ordinal + 1 } }
EOF
expect_lean_failure observation_cannot_rebind_device 'invalid \{\.\.\.\} notation.*private|Unknown constant.*CompleteProfileObservation\.mk|Invalid field.*ordinalValid'

cat >"$negative_dir/exact_observation_only.lean" <<'EOF'
import Tgrad.Backend.FillPlan

open Tgrad.Backend

def mintExactObservation {Credential Detail : Type}
    (authority : ProbeAuthority Credential)
    (observation : CompleteProfileObservation authority) (detail : Detail) :
    Availability (AvailableCapability authority) Detail :=
  availabilityFromProbe authority (.profiled observation detail)
EOF
expect_lean_success exact_observation_only

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
