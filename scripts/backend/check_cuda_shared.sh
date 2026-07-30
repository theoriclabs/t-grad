#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"

accepted_shared=ad2e3510f414e32a37f3c02095f49eed63801fd6
vendor_files=(
  Tgrad/Backend/Cuda.lean
  Tgrad/Renderer/Cuda.lean
  Tgrad/Runtime/Cuda.lean
)

if ! git diff --quiet "$accepted_shared" -- \
    Tgrad/Backend/FillPlan.lean BackendSharedTests.lean \
    scripts/backend/check_shared_spine.sh; then
  echo "FAIL accepted shared spine was edited" >&2
  exit 1
fi

if rg -n '^(structure|inductive) (FillPlan|FillValue|SemanticIdentity|KernelIdentity|SourceArtifact|AvailableCapability|AuthorizedPlan|CopyInRequest|CopyOutRequest)\b' \
    "${vendor_files[@]}"; then
  echo "FAIL CUDA leaf duplicates shared semantic authority" >&2
  exit 1
fi

if rg -n 'Tgrad\.(Runtime|Renderer)\.(Metal|Hip|Rocm)|Tgrad\.Python|tinygrad|numpy' \
    "${vendor_files[@]}"; then
  echo "FAIL CUDA leaf imports or depends on a fallback/authoring implementation" >&2
  exit 1
fi

if rg -n '\.available\b' Tgrad/Runtime/Cuda.lean; then
  echo "FAIL no-hardware CUDA leaf contains a positive availability path" >&2
  exit 1
fi

missing_operations=(
  'allocate := fun _ => missingResult \.allocation'
  'compile := fun _ _ => missingResult \.compilation'
  'launch := fun _ _ _ => missingResult \.launch'
  'synchronize := fun _ _ => missingResult \.synchronization'
  'copyIn := fun _ _ _ => missingResult \.copy'
  'copyOut := fun _ _ _ => missingResult \.copy'
  'releaseBuffer := fun _ _ => missingResult \.release'
  'releaseKernel := fun _ _ => missingResult \.release'
)
for operation in "${missing_operations[@]}"; do
  if ! rg -q "$operation" Tgrad/Runtime/Cuda.lean; then
    echo "FAIL no-hardware CUDA runtime operation is not explicitly fail-closed: $operation" >&2
    exit 1
  fi
done

lake build backend-shared-tests cuda-shared-tests
.lake/build/bin/backend-shared-tests
.lake/build/bin/cuda-shared-tests

fixture_dir="$root/scripts/backend/cuda_shared_fixtures"

expect_lean_failure() {
  local name="$1"
  local expected="$2"
  local source="$fixture_dir/$name.lean"
  local output="$root/.lake/build/cuda-shared-$name.out"
  if lake env lean "$source" >"$output" 2>&1; then
    echo "FAIL $name unexpectedly compiled" >&2
    exit 1
  fi
  if ! rg -q "$expected" "$output"; then
    echo "FAIL $name did not fail at the intended boundary" >&2
    sed -n '1,160p' "$output" >&2
    exit 1
  fi
  echo "PASS private fixture $name"
}

expect_lean_success() {
  local name="$1"
  local source="$fixture_dir/$name.lean"
  local output="$root/.lake/build/cuda-shared-$name.out"
  if ! lake env lean "$source" >"$output" 2>&1; then
    echo "FAIL $name did not compile" >&2
    sed -n '1,160p' "$output" >&2
    exit 1
  fi
  echo "PASS public fixture $name"
}

expect_lean_failure private_probe_credential 'Unknown identifier|Unknown constant'
expect_lean_failure private_buffer_handle 'Unknown constant.*BufferHandle\.mk'
expect_lean_failure private_kernel_handle 'Unknown constant.*KernelHandle\.mk'
expect_lean_failure no_positive_probe_constructor 'Unknown constant.*ProbeResult\.profiled|Invalid dotted identifier notation|Invalid constructor|Unknown identifier'
expect_lean_success public_runtime_shape

if command -v nvcc >/dev/null 2>&1; then
  echo "VENDOR_COMPILE=UNOBSERVED_ENVIRONMENT reason=nvcc_present_but_not_invoked_in_cpu_static_gate"
else
  echo "VENDOR_COMPILE=UNOBSERVED_ENVIRONMENT reason=nvcc_not_found"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "HARDWARE=UNOBSERVED_ENVIRONMENT reason=nvidia_smi_present_but_hardware_use_not_authorized"
else
  echo "HARDWARE=UNOBSERVED_ENVIRONMENT reason=nvidia_smi_not_found_and_probe_absent"
fi

echo "cuda-shared static gate: green"
