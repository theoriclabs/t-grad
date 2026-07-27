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

# Product-determining paths: ONE set used for BOTH content_hash and dirty.
# In (observed product behaviour):
#   python/              — Python product surface exercised by the pilot
#   scripts/parity/shim/ — import shim bound into adapter/runtime identity
#   Tgrad/               — Lean sources that build libtgrad (observed runtime)
#   c/                   — C trampolines linked into libtgrad.dylib
# Out (cannot change observed product behaviour):
#   docs/                — documentation only
#   fixtures/            — recorded evidence / test data, not product code
#   scripts/gates/       — gate runners, not the product under observation
product_paths=(python scripts/parity/shim Tgrad c)

# Deterministic content hash over exactly product_paths.
# Stage those paths into a throwaway index (so a clean tree matches
# `git rev-parse HEAD:<path>` per path), take each path's tree OID via
# `write-tree --prefix`, then hash the path→oid listing in declared order.
product_content_hash() {
  local tmp_index path oid listing=""
  tmp_index="$(mktemp "${TMPDIR:-/tmp}/tgrad-product-index.XXXXXX")"
  rm -f "$tmp_index"
  GIT_INDEX_FILE="$tmp_index" git read-tree HEAD
  GIT_INDEX_FILE="$tmp_index" git add -u -- "${product_paths[@]}"
  for path in "${product_paths[@]}"; do
    oid="$(GIT_INDEX_FILE="$tmp_index" git write-tree --prefix="${path}/")"
    listing+="${path}"$'\t'"${oid}"$'\n'
  done
  rm -f "$tmp_index"
  printf '%s' "$listing" | git hash-object --stdin
}

current_product_subject_tree() {
  local revision content_hash dirty
  revision="$(git rev-parse HEAD)"
  content_hash="$(product_content_hash)"
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
