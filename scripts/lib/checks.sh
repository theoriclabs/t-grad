# Tgrad gate runner — shared universal checks.
#
# Every gate (L1+) runs these before its specific predicates. They are
# the "no shortcuts" enforcement layer: any of these failing means the
# gate fails, regardless of what the gate's own predicates say.
#
# Sourced by scripts/gates/L<n>.sh. Do not run directly.

_TGRAD_CHECKS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_TGRAD_CHECKS_DIR/run_context.sh"
tgrad_run_context_init
unset _TGRAD_CHECKS_DIR

# -----------------------------------------------------------------------
# check_no_sorry — fail if any Tgrad module contains `sorry`.
#
# `sorry` skips proofs; an agent could fake a theorem with it.
# -----------------------------------------------------------------------
check_no_sorry() {
  local hits
  hits="$(grep -RnE '(^|[^a-zA-Z_])sorry([^a-zA-Z_]|$)' \
            "$TGRAD_DIR/Tgrad" "$TGRAD_DIR/Main.lean" "$TGRAD_DIR/Tests.lean" \
            2>/dev/null | grep -v '^[[:space:]]*--' || true)"
  if [[ -n "$hits" ]]; then
    echo "  ✗ check_no_sorry: 'sorry' found in:"
    echo "$hits" | sed 's/^/      /'
    return 1
  fi
}

# -----------------------------------------------------------------------
# check_no_axiom — fail if any Tgrad module declares an axiom that's not
# allowlisted as an FFI-extern.
#
# `axiom foo : T` is how an agent could declare a fake theorem.
# The only legal axiom-ish form in Tgrad is `@[extern "name"] opaque`
# (FFI boundary), which is structurally different.
# -----------------------------------------------------------------------
check_no_axiom() {
  local hits
  hits="$(grep -RnE '^[[:space:]]*axiom\b' \
            "$TGRAD_DIR/Tgrad" "$TGRAD_DIR/Main.lean" "$TGRAD_DIR/Tests.lean" \
            2>/dev/null || true)"
  if [[ -n "$hits" ]]; then
    echo "  ✗ check_no_axiom: 'axiom' declarations found:"
    echo "$hits" | sed 's/^/      /'
    echo "      use @[extern \"...\"] opaque for FFI; everything else must prove"
    return 1
  fi
}

# -----------------------------------------------------------------------
# check_no_unsafe — fail if any Tgrad module uses `unsafe` declarations.
#
# `unsafe` bypasses Lean's totality / termination checks.
# -----------------------------------------------------------------------
check_no_unsafe() {
  local hits
  hits="$(grep -RnE '^[[:space:]]*unsafe\b' \
            "$TGRAD_DIR/Tgrad" "$TGRAD_DIR/Main.lean" "$TGRAD_DIR/Tests.lean" \
            2>/dev/null || true)"
  if [[ -n "$hits" ]]; then
    echo "  ✗ check_no_unsafe: 'unsafe' declarations found:"
    echo "$hits" | sed 's/^/      /'
    return 1
  fi
}

# -----------------------------------------------------------------------
# check_warnings — fail if `lake build`'s stderr contains warning lines
# that are not in scripts/lib/warning_allowlist.txt.
#
# Reads the run-scoped clean rebuild log produced by check_clean_rebuild.
# Call check_clean_rebuild first.
# -----------------------------------------------------------------------
check_warnings() {
  local log="$(tgrad_run_path clean_rebuild.log)"
  local allowlist="$TGRAD_DIR/scripts/lib/warning_allowlist.txt"
  [[ -f "$log" ]] || return 0   # if no log, check_clean_rebuild handles it
  [[ -f "$allowlist" ]] || { echo "  ✗ check_warnings: warning_allowlist.txt missing"; return 1; }
  # Strip comments + blanks from allowlist to produce the regex list.
  local tmp_re="$(tgrad_run_path warning_allowlist.regex)"
  grep -vE '^[[:space:]]*(#|$)' "$allowlist" > "$tmp_re"
  # Find warning lines that don't match any allowlisted pattern.
  local stray
  stray="$(grep -E 'warning:' "$log" | (grep -vE -f "$tmp_re" || true))"
  rm -f "$tmp_re"
  if [[ -n "$stray" ]]; then
    echo "  ✗ check_warnings: build emitted warnings not in allowlist:"
    echo "$stray" | sed 's/^/      /'
    echo "      add to scripts/lib/warning_allowlist.txt or fix the source"
    return 1
  fi
}

# -----------------------------------------------------------------------
# check_clean_rebuild — fail if a from-scratch build doesn't work.
#
# Stale .olean files can mask real failures (e.g. removed-but-still-cached
# constants). A stale .lake/build/bin/ binary can mask real source
# regressions (gate predicates that shell to `tgrad-cli` would otherwise
# run against a pre-sabotage binary). A green build on a clean tree —
# library AND executables — is the real signal.
#
# Lake's `lake clean` is blocked by mathlib-cache rules elsewhere in this
# repo; we approximate by removing all generated artifacts under
# .lake/build/{lib,ir,bin} and rebuilding everything the gates use.
# -----------------------------------------------------------------------
check_clean_rebuild() {
  cd "$TGRAD_DIR"
  rm -rf .lake/build/lib .lake/build/ir .lake/build/bin 2>/dev/null
  local rebuild_log
  rebuild_log="$(tgrad_run_path clean_rebuild.log)"
  : >"$rebuild_log"
  # If a C bridge is present (L4+), build the .o files before lake.
  # The Tgrad lakefile.lean links them into tgrad-cli + tgrad-tests
  # via moreLinkArgs; missing .o files would surface as link errors.
  if [[ -f c/Makefile ]]; then
    make -C c >>"$rebuild_log" 2>&1 || {
      echo "  ✗ check_clean_rebuild: C bridge build (make -C c) failed"
      sed 's/^/      /' "$rebuild_log"
      return 1
    }
  fi
  # Build library, shared library, CLI, and tests — every artifact a gate
  # may invoke. The Python-facing dylib target relinks against
  # libtgrad_Tgrad.dylib, so the clean rebuild must produce Tgrad:shared
  # before any gate calls ensure_dylib.
  lake build Tgrad:shared tgrad-cli tgrad-tests >>"$rebuild_log" 2>&1 || {
    echo "  ✗ check_clean_rebuild: build failed after rm -rf .lake/build/{lib,ir,bin}"
    sed 's/^/      /' "$rebuild_log"
    return 1
  }
}

# -----------------------------------------------------------------------
# ensure_dylib [log]
#
# Build the Python-facing dynamic library incrementally by default.
# Set TGRAD_FORCE_REBUILD=1 to force `make -B dylib` for final clean
# verification or when debugging stale native artifacts.
# -----------------------------------------------------------------------
ensure_dylib() {
  local log="${1:-$(tgrad_run_path dylib.log)}"
  local make_args=(dylib)
  if [[ "${TGRAD_FORCE_REBUILD:-0}" == "1" ]]; then
    make_args=(-B dylib)
  fi
  ( cd "$TGRAD_DIR/c" && make "${make_args[@]}" ) >"$log" 2>&1 || {
    echo "  ✗ ensure_dylib: make ${make_args[*]} failed"
    sed 's/^/      /' "$log"
    return 1
  }
  local dylib="$TGRAD_DIR/.lake/build/lib/libtgrad.dylib"
  if [[ ! -f "$dylib" ]]; then
    echo "  ✗ ensure_dylib: $dylib missing after make ${make_args[*]}"
    sed 's/^/      /' "$log"
    return 1
  fi
}

# Execute a nested regression gate without allowing it to overwrite the
# top-level candidate evidence already bound to an earlier process outcome.
# Run artifacts still share the serial run root and are preserved by content
# hash after each top-level gate; evidence documents get a private namespace.
run_gate_isolated() {
  local script="$1"
  local nested_root
  nested_root="$(tgrad_run_subdir nested_gate)" || return 1
  mkdir -p "$nested_root/evidence"
  TGRAD_EVIDENCE_DIR="$nested_root/evidence" bash "$script"
}

# -----------------------------------------------------------------------
# check_no_gate_regression — verify GREEN_GATES never shrinks vs git HEAD.
#
# Catches the case where an agent removes a gate from the green list to
# unblock themselves on an unrelated issue.
# -----------------------------------------------------------------------
check_no_gate_regression() {
  git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1 || return 0
  local committed_green current_green committed_tokens current_tokens
  committed_green="$(git -C "$REPO_ROOT" show HEAD:scripts/gate.sh 2>/dev/null | \
                     grep -o 'GREEN_GATES=([^)]*)' | head -1 || true)"
  current_green="$(grep -o 'GREEN_GATES=([^)]*)' "$TGRAD_DIR/scripts/gate.sh" | head -1 || true)"
  # Each green gate in HEAD must still appear in the current version.
  if [[ -z "$committed_green" ]]; then return 0; fi
  committed_tokens="$(echo "$committed_green" | tr -d '()' | sed 's/GREEN_GATES=//' | tr -d "\"")"
  current_tokens="$(echo "$current_green" | tr -d '()' | sed 's/GREEN_GATES=//' | tr -d "\"")"
  for g in $committed_tokens; do
    if [[ " $current_tokens " != *" $g "* ]]; then
      echo "  ✗ check_no_gate_regression: gate $g was green in HEAD but is missing now"
      echo "      committed: $committed_green"
      echo "      current:   $current_green"
      return 1
    fi
  done
}

# -----------------------------------------------------------------------
# check_evidence_for <gate>
#
# Each gate writes a run-owned evidence file recording timestamp,
# machine/profile, commit, and computation hashes. This local check only
# confirms the legacy common fields; release publication separately resolves
# every hash against the reviewed locator contract and retained artifact set.
# -----------------------------------------------------------------------
check_evidence_for() {
  local gate="$1"
  local file="$TGRAD_EVIDENCE_DIR/$gate.json"
  if [[ ! -f "$file" ]]; then
    echo "  ✗ check_evidence_for $gate: $file missing — gate hasn't produced evidence"
    return 1
  fi
  # Minimal schema check: must have ts_utc, hashes, commit, and either
  # host or host_profile. Public release evidence uses host_profile.
  for key in ts_utc hashes commit; do
    if ! grep -q "\"$key\":" "$file"; then
      echo "  ✗ check_evidence_for $gate: missing '$key' field in $file"
      return 1
    fi
  done
  if ! grep -Eq '"host"|"host_profile"' "$file"; then
    echo "  ✗ check_evidence_for $gate: missing 'host' or 'host_profile' field in $file"
    return 1
  fi
}

# -----------------------------------------------------------------------
# check_falsifiability_verified <gate>
#
# Per Rule 10, each gate has a `<gate>_falsifiability.md` with a table
# enumerating sabotages. Each row's last column is the "Verified?"
# flag — must be `✓` (or `yes` / a date / a commit sha). Anything
# matching `—`, `pending`, `TODO`, or blank counts as unverified.
#
# The gate cannot flip green until every row is verified.
# -----------------------------------------------------------------------
check_falsifiability_verified() {
  local gate="$1"
  local file="$TGRAD_DIR/scripts/gates/${gate}_falsifiability.md"
  if [[ ! -f "$file" ]]; then
    echo "  ✗ check_falsifiability_verified $gate: $file missing"
    return 1
  fi
  # Find the table header line (has "Verified" in last column).
  # Verified column entries are extracted via awk on '|'-separated rows
  # that begin with a row-number-or-asterisk-or-text marker.
  # We look at every data row (line starts with `|` and has ≥ 4 pipes)
  # below the table header.
  local unverified
  unverified="$(awk -F'|' '
    /Verified\?/ { in_table=1; next }
    in_table && /^\|[[:space:]]*[-:]+[[:space:]]*\|/ { next }   # separator
    in_table && NF>=5 {
      last=$(NF-1)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", last)
      if (last == "" || last == "—" || last == "-" || \
          tolower(last) == "pending" || tolower(last) == "todo" || \
          last == "?") {
        print NR": "$0
      }
    }
  ' "$file")"
  if [[ -n "$unverified" ]]; then
    echo "  ✗ $file has unverified sabotage row(s):"
    echo "$unverified" | sed 's/^/      /'
    echo "      Per Rule 10, every sabotage must be run + verified to fail"
    echo "      for the right reason BEFORE flipping the gate. Mark each"
    echo "      row's Verified? column with ✓ (or a commit sha) after"
    echo "      running the sabotage and confirming the gate rejects it."
    return 1
  fi
}

# -----------------------------------------------------------------------
# run_global_preflight — full-sweep or single-gate clean preflight.
#
# A full gate sweep can run dozens of gates. The strong stale-artifact
# check must happen, but it only needs to happen once at the start of
# the sweep. Child gate scripts inherit TGRAD_GLOBAL_PREFLIGHT_DONE=1
# and keep running the cheap anti-shortcut checks without repeatedly
# wiping .lake/build.
# -----------------------------------------------------------------------
run_global_preflight() {
  echo "  [global preflight]"
  check_no_sorry           || return 1
  check_no_axiom           || return 1
  check_no_unsafe          || return 1
  check_no_gate_regression || return 1
  check_clean_rebuild      || return 1
  check_warnings           || return 1
  export TGRAD_GLOBAL_PREFLIGHT_DONE=1
  echo "  [global preflight ✓]"
}

# -----------------------------------------------------------------------
# run_preflight — every gate calls this before its specific predicates.
# Aborts the gate on any failure.
# -----------------------------------------------------------------------
run_preflight() {
  echo "  [preflight]"
  check_no_sorry           || return 1
  check_no_axiom           || return 1
  check_no_unsafe          || return 1
  check_no_gate_regression || return 1
  if [[ "${TGRAD_GLOBAL_PREFLIGHT_DONE:-0}" == "1" ]]; then
    echo "  [preflight] clean rebuild already covered by global preflight"
  else
    check_clean_rebuild    || return 1
    check_warnings         || return 1
  fi
  echo "  [preflight ✓]"
}
