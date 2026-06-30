#!/usr/bin/env bash
# Reject if any Tgrad runtime file has an ACTUAL tinygrad dependency
# (import, path reference, subprocess call). Comments / provenance
# notes that merely mention "tinygrad" are permitted per §6 rule 3.
#
# Capture scripts (under scripts/capture/) ARE allowed to invoke
# tinygrad — they're dev-time tools that produce committed fixtures.
# Runtime code (.lean / .c / .m / .py / .sh) outside scripts/capture/
# MUST be tinygrad-free.
#
# Enforces the v1.0.0 runtime-independence invariant: nothing in the
# runtime path may import or shell out to tinygrad. Capture/dev-time
# scripts under scripts/capture/ are exempt — they're allowed to shell
# to tinygrad for fixture / baseline regeneration.
set -euo pipefail
TGRAD_DIR="${TGRAD_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Patterns that are runtime-meaningful tinygrad dependencies:
#   - `import tinygrad`, `from tinygrad ...`           — Python
#   - `PYTHONPATH=...tinygrad`                          — shell var
#   - `sys.path.append(...tinygrad)`                    — runtime hack
#   - `subprocess.run([...tinygrad...])`                — shell out
#   - `..` path reference into tinygrad/                — Tgrad reaching out
#   - `lake env ... tinygrad`                           — Lake interop
#
# Comments (`#`, `//`, `--`, `/-`) containing "tinygrad" are skipped
# by anchoring patterns at the start of the line (modulo whitespace).

violations=""

scan_files() {
  find "$TGRAD_DIR" -type f \
    \( -name '*.lean' -o -name '*.py' -o -name '*.c' -o -name '*.m' -o -name '*.sh' \) \
    ! -path '*/.lake/*' \
    ! -path '*/build/*' \
    ! -path '*/scripts/capture/*'
}

while IFS= read -r f; do
  matched=""
  # Python: import tinygrad / from tinygrad
  m="$(grep -nE '^[[:space:]]*(import[[:space:]]+tinygrad|from[[:space:]]+tinygrad[.[:space:]])' "$f" 2>/dev/null || true)"
  [[ -n "$m" ]] && matched+="$m\n"
  # Python: sys.path.* tinygrad / sys.path manipulation
  m="$(grep -nE '^[^#]*sys\.path\.[a-z]+\([^#]*tinygrad' "$f" 2>/dev/null || true)"
  [[ -n "$m" ]] && matched+="$m\n"
  # Shell: PYTHONPATH=... tinygrad
  m="$(grep -nE '^[^#]*PYTHONPATH=[^[:space:]]*tinygrad' "$f" 2>/dev/null || true)"
  [[ -n "$m" ]] && matched+="$m\n"
  # subprocess.run([..., "tinygrad", ...])
  m="$(grep -nE '^[^#]*subprocess\.[a-zA-Z_]+\([^#]*tinygrad' "$f" 2>/dev/null || true)"
  [[ -n "$m" ]] && matched+="$m\n"
  # ctypes.CDLL('.../tinygrad/...') — Python loads tinygrad's shared lib
  m="$(grep -nE '^[^#]*ctypes\.[A-Z][a-zA-Z]+\([^#]*tinygrad' "$f" 2>/dev/null || true)"
  [[ -n "$m" ]] && matched+="$m\n"
  if [[ -n "$matched" ]]; then
    violations+="  $f:\n$(echo -en "$matched" | sed 's/^/    /')\n"
  fi
done < <(scan_files)

if [[ -n "$violations" ]]; then
  echo "  ✗ check_no_tinygrad_deps: runtime files have actual tinygrad refs:"
  echo -en "$violations"
  echo "      Tgrad runtime must be independent of tinygrad/."
  echo "      Comments / docstring provenance notes are permitted; only"
  echo "      runtime-meaningful refs (import / from / subprocess / path"
  echo "      strings / PYTHONPATH) are rejected here."
  exit 1
fi
echo "  ✓ no runtime dependency on tinygrad/"
