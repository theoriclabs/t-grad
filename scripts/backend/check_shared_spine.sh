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
exec .lake/build/bin/backend-shared-tests
