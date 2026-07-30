#!/bin/sh
set -eu

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$repo_dir"

make -C c -f Makefile.hip
lake build rocm-shared-tests
"$repo_dir/.lake/build/bin/rocm-shared-tests"

negative_out="$repo_dir/.lake/build/hip-probe-credential-negative.out"
if lake env lean scripts/backend/negative/HipProbeCredentialForge.lean \
    >"$negative_out" 2>&1; then
  echo "FAIL private HIP probe credential became constructible" >&2
  exit 1
fi
if ! grep -Eq "Unknown constant.*ProbeCredential\.mk|unknown.*ProbeCredential\.mk" \
    "$negative_out"; then
  echo "FAIL private HIP probe credential rejected for an unexpected reason" >&2
  sed -n '1,80p' "$negative_out" >&2
  exit 1
fi
echo "PASS private HIP probe credential is sealed"

if rg -n 'gfx[0-9]|hipcc|hiprtc|hip/hip_runtime|libamdhip' \
    Tgrad/Backend/FillPlan.lean >/dev/null; then
  echo "FAIL HIP vendor policy leaked into shared FillPlan" >&2
  exit 1
fi
echo "PASS HIP gfx/tool/dialect/runtime policy remains vendor-local"

if rg -n '^import Tgrad\.(Runtime|Renderer)\.(Metal|CPU)|Tgrad\.(Runtime|Renderer)\.(Metal|CPU)' \
    Tgrad/Runtime/Hip.lean >/dev/null; then
  echo "FAIL alternate backend imported by HIP runtime leaf" >&2
  exit 1
fi
echo "PASS HIP runtime imports no alternate-backend fallback"

if command -v hipcc >/dev/null 2>&1; then
  echo "VENDOR_COMPILE=UNOBSERVED_ENVIRONMENT reason=hipcc_present_but_not_invoked"
else
  echo "VENDOR_COMPILE=UNOBSERVED_ENVIRONMENT reason=hipcc_not_found"
fi

if command -v rocm-smi >/dev/null 2>&1; then
  echo "HARDWARE=UNOBSERVED_ENVIRONMENT reason=rocm_smi_present_but_hardware_use_not_authorized"
else
  echo "HARDWARE=UNOBSERVED_ENVIRONMENT reason=rocm_smi_not_found_and_probe_incomplete_or_negative"
fi
