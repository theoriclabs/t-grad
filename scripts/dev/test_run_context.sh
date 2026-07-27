#!/usr/bin/env bash
# Focused, non-GPU self-test for scripts/lib/run_context.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_CONTEXT="$REPO_ROOT/scripts/lib/run_context.sh"
COORD_DIR="$(mktemp -d -t tgrad_run_context_test)"
trap 'rm -rf -- "$COORD_DIR"' EXIT

probe() {
  local label="$1"
  env -u TGRAD_RUN_DIR -u TGRAD_RUN_OWNER_PID -u TGRAD_RUN_OWNER_TOKEN \
    TGRAD_KEEP_RUN_DIR=1 bash -c '
      set -euo pipefail
      source "$1"
      tgrad_run_context_init
      artifact="$(tgrad_run_path representative.log)"
      printf "%s\n" "$2" >"$artifact"
      trace="$(tgrad_run_prepare_rangeify_trace)"
      printf "trace-%s\n" "$2" >"$trace"
      nested="$(bash -c '\''
        set -euo pipefail
        source "$1"
        tgrad_run_context_init
        [[ "${BASHPID:-$$}" != "$TGRAD_RUN_OWNER_PID" ]]
        printf "%s\n" "$TGRAD_RUN_DIR"
        printf "nested-%s\n" "$2" >"$(tgrad_run_path nested.log)"
      '\'' _ "$1" "$2")"
      [[ "$nested" == "$TGRAD_RUN_DIR" ]]
      printf "%s\n" "$TGRAD_RUN_DIR"
    ' _ "$RUN_CONTEXT" "$label"
}

probe alpha >"$COORD_DIR/alpha.root" &
pid_alpha=$!
probe beta >"$COORD_DIR/beta.root" &
pid_beta=$!
wait "$pid_alpha"
wait "$pid_beta"

root_alpha="$(<"$COORD_DIR/alpha.root")"
root_beta="$(<"$COORD_DIR/beta.root")"
[[ "$root_alpha" != "$root_beta" ]]
[[ "$(<"$root_alpha/representative.log")" == "alpha" ]]
[[ "$(<"$root_beta/representative.log")" == "beta" ]]
[[ "$(<"$root_alpha/nested.log")" == "nested-alpha" ]]
[[ "$(<"$root_beta/nested.log")" == "nested-beta" ]]
[[ "$(<"$root_alpha/rangeify_trace.jsonl")" == "trace-alpha" ]]
[[ "$(<"$root_beta/rangeify_trace.jsonl")" == "trace-beta" ]]

# An explicitly supplied root is usable but has no creator marker, so the
# helper must never install an owning cleanup trap for it.
external_root="$COORD_DIR/external root"
mkdir "$external_root"
env -u TGRAD_RUN_OWNER_PID -u TGRAD_RUN_OWNER_TOKEN \
  TGRAD_RUN_DIR="$external_root" bash -c '
  set -euo pipefail
  source "$1"
  tgrad_run_context_init
  ! tgrad_run_context_is_owner
  printf "external\n" >"$(tgrad_run_path artifact.log)"
  first="$(tgrad_run_subdir repeated_work)"
  second="$(tgrad_run_subdir repeated_work)"
  [[ "$first" != "$second" && -d "$first" && -d "$second" ]]
' _ "$RUN_CONTEXT"
[[ -f "$external_root/artifact.log" ]]
[[ -d "$external_root/.tgrad-run-shared" ]]

# An unrelated non-empty directory cannot be adopted as a run root.
unmarked_root="$COORD_DIR/unmarked"
mkdir "$unmarked_root"
printf 'do not overwrite\n' >"$unmarked_root/existing.txt"
if env -u TGRAD_RUN_OWNER_PID -u TGRAD_RUN_OWNER_TOKEN \
    TGRAD_RUN_DIR="$unmarked_root" bash -c \
    'source "$1"; tgrad_run_context_init' _ "$RUN_CONTEXT" >/dev/null 2>&1; then
  echo "accepted non-empty unmarked run root" >&2
  exit 1
fi
[[ "$(<"$unmarked_root/existing.txt")" == "do not overwrite" ]]

env -u TGRAD_RUN_OWNER_PID -u TGRAD_RUN_OWNER_TOKEN \
  TGRAD_RUN_DIR="$external_root" bash -c '
  set -euo pipefail
  source "$1"
  tgrad_run_context_init
  for bad in "" . .. ../escape /absolute child/path '"'"'line
break'"'"'; do
    if tgrad_run_path "$bad" >/dev/null 2>&1; then
      echo "accepted invalid child: $bad" >&2
      exit 1
    fi
  done
' _ "$RUN_CONTEXT"

if env -u TGRAD_RUN_OWNER_PID -u TGRAD_RUN_OWNER_TOKEN \
    TGRAD_RUN_DIR=/tmp bash -c 'source "$1"; tgrad_run_context_init' _ "$RUN_CONTEXT" \
    >/dev/null 2>&1; then
  echo "accepted broad run root" >&2
  exit 1
fi

# Without the keep flag, the creating process removes its own root on exit.
auto_root="$(env -u TGRAD_RUN_DIR -u TGRAD_RUN_OWNER_PID -u TGRAD_RUN_OWNER_TOKEN \
  bash -c 'source "$1"; tgrad_run_context_init; printf "%s\n" "$TGRAD_RUN_DIR"' \
  _ "$RUN_CONTEXT")"
[[ ! -e "$auto_root" ]]

# The probes explicitly requested preservation.  Validate their exact roots
# before removing them as part of the self-test's own cleanup.
for root in "$root_alpha" "$root_beta"; do
  [[ "$root" == /tmp/* || "$root" == /private/tmp/* || "$root" == /private/var/folders/* ]]
  [[ -f "$root/.tgrad-run-owner" ]]
  rm -rf -- "$root"
done

echo "run_context self-test: ok"
