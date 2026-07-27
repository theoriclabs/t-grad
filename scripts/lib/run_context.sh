#!/usr/bin/env bash
# Run-scoped temporary artifact management for gates and development checks.
#
# Source this file, then call tgrad_run_context_init once near the start of a
# top-level script.  A caller that inherits TGRAD_RUN_DIR joins the existing
# run; a caller without one creates a private mktemp root.  Only the process
# that created the root owns its cleanup trap.

_tgrad_run_error() {
  echo "tgrad run context: $*" >&2
  return 1
}

_tgrad_run_canonical_root() {
  local root="${1:-}"
  [[ -n "$root" ]] || { _tgrad_run_error "run root is empty"; return 1; }
  case "$root" in
    /*) ;;
    *) _tgrad_run_error "run root must be absolute: $root"; return 1 ;;
  esac
  case "$root" in
    *$'\n'*|*$'\r'*) _tgrad_run_error "run root contains a line break"; return 1 ;;
  esac
  [[ -d "$root" ]] || { _tgrad_run_error "run root is not a directory: $root"; return 1; }
  [[ ! -L "$root" ]] || { _tgrad_run_error "run root must not be a symlink: $root"; return 1; }

  local canonical
  canonical="$(cd -P -- "$root" 2>/dev/null && pwd -P)" || {
    _tgrad_run_error "cannot resolve run root: $root"
    return 1
  }
  case "$canonical" in
    /|/tmp|/private/tmp) _tgrad_run_error "refusing broad run root: $canonical"; return 1 ;;
  esac
  printf '%s\n' "$canonical"
}

tgrad_run_context_is_owner() {
  [[ -n "${TGRAD_RUN_DIR:-}" ]] || return 1
  [[ -n "${TGRAD_RUN_OWNER_PID:-}" ]] || return 1
  [[ -n "${TGRAD_RUN_OWNER_TOKEN:-}" ]] || return 1
  [[ "${BASHPID:-$$}" == "$TGRAD_RUN_OWNER_PID" ]] || return 1
  [[ -f "$TGRAD_RUN_DIR/.tgrad-run-owner" ]] || return 1
  [[ "$(<"$TGRAD_RUN_DIR/.tgrad-run-owner")" == "$TGRAD_RUN_OWNER_TOKEN" ]] || return 1
}

_tgrad_run_validate_shared_root() {
  local root="$1"
  local marker="$root/.tgrad-run-shared"
  if [[ -e "$marker" && ! -d "$marker" ]]; then
    _tgrad_run_error "shared-root marker is not a directory: $marker"
    return 1
  fi
  if [[ -d "$marker" ]]; then
    return 0
  fi

  # A caller-supplied root must be dedicated to this protocol.  Refuse to
  # adopt a pre-existing non-empty directory, since artifact names are direct
  # children and could otherwise overwrite unrelated files.
  local first_entry
  first_entry="$(find "$root" -mindepth 1 -maxdepth 1 -print -quit)"
  [[ -z "$first_entry" ]] || {
    _tgrad_run_error "unmarked shared run root is not empty: $root"
    return 1
  }

  # mkdir is the marker: it is atomic when two coordinators join an empty root.
  mkdir "$marker" 2>/dev/null || [[ -d "$marker" ]] || {
    _tgrad_run_error "cannot mark shared run root: $root"
    return 1
  }
}

tgrad_run_context_cleanup() {
  tgrad_run_context_is_owner || return 0
  if [[ "${TGRAD_KEEP_RUN_DIR:-0}" == "1" ]]; then
    echo "tgrad run context: preserving $TGRAD_RUN_DIR" >&2
    return 0
  fi

  local root
  root="$(_tgrad_run_canonical_root "$TGRAD_RUN_DIR")" || return 1
  [[ "$root" == "$TGRAD_RUN_DIR" ]] || {
    _tgrad_run_error "run root changed before cleanup"
    return 1
  }
  [[ "$(<"$root/.tgrad-run-owner")" == "$TGRAD_RUN_OWNER_TOKEN" ]] || {
    _tgrad_run_error "ownership marker changed before cleanup"
    return 1
  }
  rm -rf -- "$root"
}

_tgrad_run_context_on_exit() {
  local status=$?
  trap - EXIT
  tgrad_run_context_cleanup || true
  exit "$status"
}

tgrad_run_context_init() {
  local root created=0
  if [[ -z "${TGRAD_RUN_DIR:-}" ]]; then
    root="$(mktemp -d -t tgrad_run)" || {
      _tgrad_run_error "mktemp failed"
      return 1
    }
    created=1
  else
    root="$TGRAD_RUN_DIR"
  fi

  root="$(_tgrad_run_canonical_root "$root")" || {
    [[ "$created" == "1" ]] && rm -rf -- "$root"
    return 1
  }
  export TGRAD_RUN_DIR="$root"

  if [[ "$created" == "1" ]]; then
    export TGRAD_RUN_OWNER_PID="${BASHPID:-$$}"
    export TGRAD_RUN_OWNER_TOKEN="${TGRAD_RUN_OWNER_PID}.${RANDOM}.${RANDOM}"
    printf '%s\n' "$TGRAD_RUN_OWNER_TOKEN" >"$TGRAD_RUN_DIR/.tgrad-run-owner"
  elif [[ -n "${TGRAD_RUN_OWNER_PID:-}" || -n "${TGRAD_RUN_OWNER_TOKEN:-}" ]]; then
    [[ -n "${TGRAD_RUN_OWNER_PID:-}" && -n "${TGRAD_RUN_OWNER_TOKEN:-}" ]] || {
      _tgrad_run_error "incomplete inherited ownership metadata"
      return 1
    }
    [[ -f "$TGRAD_RUN_DIR/.tgrad-run-owner" ]] || {
      _tgrad_run_error "inherited run root has no ownership marker"
      return 1
    }
    [[ "$(<"$TGRAD_RUN_DIR/.tgrad-run-owner")" == "$TGRAD_RUN_OWNER_TOKEN" ]] || {
      _tgrad_run_error "inherited run root ownership marker does not match"
      return 1
    }
  else
    _tgrad_run_validate_shared_root "$TGRAD_RUN_DIR" || return 1
  fi

  if tgrad_run_context_is_owner && [[ "${_TGRAD_RUN_TRAP_PID:-}" != "${BASHPID:-$$}" ]]; then
    _TGRAD_RUN_TRAP_PID="${BASHPID:-$$}"
    trap _tgrad_run_context_on_exit EXIT
  fi
}

# Return one path immediately beneath TGRAD_RUN_DIR.  Deliberately reject
# separators instead of normalizing them: callers must provide a resolved,
# bounded artifact name, never a user-controlled relative path.
tgrad_run_path() {
  [[ "$#" == "1" ]] || { _tgrad_run_error "tgrad_run_path expects one child name"; return 1; }
  local name="$1"
  [[ -n "${TGRAD_RUN_DIR:-}" ]] || { _tgrad_run_error "run context is not initialized"; return 1; }
  [[ "$name" != "." && "$name" != ".." ]] || {
    _tgrad_run_error "refusing broad child name: $name"
    return 1
  }
  [[ "$name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
    _tgrad_run_error "invalid child name: $name"
    return 1
  }
  printf '%s/%s\n' "$TGRAD_RUN_DIR" "$name"
}

# Create a unique directory beneath the run root for a gate that may be
# invoked more than once by nested regression checks.
tgrad_run_subdir() {
  [[ "$#" == "1" ]] || { _tgrad_run_error "tgrad_run_subdir expects one prefix"; return 1; }
  local prefix="$1"
  [[ "$prefix" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
    _tgrad_run_error "invalid subdirectory prefix: $prefix"
    return 1
  }
  mktemp -d "$TGRAD_RUN_DIR/${prefix}.XXXXXX"
}

tgrad_run_prepare_rangeify_trace() {
  local target
  target="$(tgrad_run_path rangeify_trace.jsonl)" || return 1
  : >"$target"
  printf '%s\n' "$target"
}
