#!/usr/bin/env bash
# Emit the derived pilot status report for the CURRENT product tree.
#
# Lean cannot observe git during elaboration, so this script computes the
# live subject-tree identity (revision + tree hash + dirty flag for tracked
# product sources) and passes it to PilotStatusMain via
# TGRAD_PRODUCT_SUBJECT_TREE.  If that variable is already set, it is left
# alone so callers can feed a deliberate tree identity for checks.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

# Product sources bound into the pilot observation identity.  Dirty means the
# working tree differs from HEAD in any of these tracked paths.
product_paths=(python scripts/parity/shim)

current_product_subject_tree() {
  local revision content_hash dirty
  revision="$(git rev-parse HEAD)"
  content_hash="$(git rev-parse "HEAD^{tree}")"
  if git diff --quiet HEAD -- "${product_paths[@]}"; then
    dirty=false
  else
    dirty=true
  fi
  printf '%s:%s:%s' "$revision" "$content_hash" "$dirty"
}

if [[ -z "${TGRAD_PRODUCT_SUBJECT_TREE:-}" ]]; then
  TGRAD_PRODUCT_SUBJECT_TREE="$(current_product_subject_tree)"
  export TGRAD_PRODUCT_SUBJECT_TREE
fi

exec lake env lean Tgrad/Growth/PilotStatusMain.lean
