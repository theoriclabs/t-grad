#!/bin/sh
set -eu

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$repo_dir"

make -C c -f Makefile.hip
lake build rocm-tests
"$repo_dir/.lake/build/bin/rocm-tests"

if command -v hipcc >/dev/null 2>&1; then
  echo "VENDOR_COMPILE=UNOBSERVED_ENVIRONMENT reason=hipcc_present_but_no_compiler_claim_recorded"
  exit 1
else
  echo "VENDOR_COMPILE=UNOBSERVED_ENVIRONMENT reason=hipcc_not_found"
fi

if command -v rocm-smi >/dev/null 2>&1; then
  echo "HARDWARE=UNOBSERVED_ENVIRONMENT reason=rocm_smi_present_but_hardware_use_not_authorized"
  exit 1
else
  echo "HARDWARE=UNOBSERVED_ENVIRONMENT reason=rocm_smi_not_found_and_runtime_probe_negative"
fi
