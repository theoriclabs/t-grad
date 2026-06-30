#!/usr/bin/env python3
"""DEV-TIME transpiler: captured matmul MSL → Lean KernelDecl source.

Reads each fixtures/codegen/matmul_*.msl, parses it into a typed
AST mirroring Tgrad/Renderer/Metal.lean's `Stmt`/`KernelDecl`,
and emits a Lean source file (Tgrad/Renderer/MatmulDecls.lean)
that defines `matmulKernelDeclFor : ShapeSentinel → KernelDecl`.

The generated Lean module is committed; this script is never invoked
at runtime. The L12 gate verifies byte-equality between the captured
MSL and the rendered output of the generated `KernelDecl`s — so a
broken transpiler turns into a gate failure, never a silent miss.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Shape ↔ ShapeSentinel mapping (mirrors Renderer.Metal.ShapeSentinel)
# ---------------------------------------------------------------------------
SHAPE_TRIPLES: list[tuple[str, str, tuple[int, int, int]]] = [
    # (msl filename stem, ShapeSentinel constructor, (M, K, N))
    ("matmul_64x64",             "bf16_64x64",              (64,   64,   64)),
    ("matmul_1024x1024x1024",    "bf16_1024x1024",          (1024, 1024, 1024)),
    ("matmul_2048x2048x2048",    "bf16_2048x2048",          (2048, 2048, 2048)),
    ("matmul_4096x4096x4096",    "bf16_4096x4096",          (4096, 4096, 4096)),
    ("matmul_8192x8192x8192",    "bf16_8192x8192",          (8192, 8192, 8192)),
    ("matmul_8192x1024x1024",    "bf16_8192x1024x1024",     (8192, 1024, 1024)),
    ("matmul_4096x1024x1024",    "bf16_4096x1024x1024",     (4096, 1024, 1024)),
    ("matmul_2048x1024x1024",    "bf16_2048x1024x1024",     (2048, 1024, 1024)),
    ("matmul_1024x1024x8192",    "bf16_1024x1024x8192",     (1024, 1024, 8192)),
    ("matmul_1024x1024x4096",    "bf16_1024x1024x4096",     (1024, 1024, 4096)),
    ("matmul_1024x1024x2048",    "bf16_1024x1024x2048",     (1024, 1024, 2048)),
]


# ---------------------------------------------------------------------------
# AST mirroring Tgrad/Renderer/Metal.lean
# ---------------------------------------------------------------------------
class Stmt:  # base
    pass


class DeclAccArray(Stmt):
    def __init__(self, name: str, size: int): self.name, self.size = name, size

class DeclInt(Stmt):
    def __init__(self, name: str, expr: str, comment: str | None):
        self.name, self.expr, self.comment = name, expr, comment

class DeclBfloat(Stmt):
    def __init__(self, name: str, expr: str): self.name, self.expr = name, expr

class DeclBfloat2(Stmt):
    def __init__(self, name: str, expr: str): self.name, self.expr = name, expr

class DeclFloat2(Stmt):
    def __init__(self, name: str, expr: str): self.name, self.expr = name, expr

class AccStore(Stmt):
    def __init__(self, name: str, offset: int, rhs: str):
        self.name, self.offset, self.rhs = name, offset, rhs

class AccZeroInit(Stmt):
    def __init__(self, name: str, size: int): self.name, self.size = name, size

class DataStore(Stmt):
    def __init__(self, buf: str, offset: str, rhs: str):
        self.buf, self.offset, self.rhs = buf, offset, rhs

class WmmaCall(Stmt):
    def __init__(self, out: str, prelude: str, a: str, b: str, c: str):
        self.out, self.prelude, self.a, self.b, self.c = out, prelude, a, b, c

class ForLoop(Stmt):
    def __init__(self, ivar: str, hi: int, body: list[Stmt]):
        self.ivar, self.hi, self.body = ivar, hi, body


# ---------------------------------------------------------------------------
# Line classifier
# ---------------------------------------------------------------------------
RE_DECL_ACC   = re.compile(r"^(\s*)float (\w+)\[(\d+)\];$")
RE_INT_COMM   = re.compile(r"^(\s*)int (\w+) = (.+); /\* (.+) \*/$")
RE_INT        = re.compile(r"^(\s*)int (\w+) = (.+);$")
RE_BFLOAT     = re.compile(r"^(\s*)bfloat (\w+) = (.+);$")
RE_BFLOAT2    = re.compile(r"^(\s*)bfloat2 (\w+) = (.+);$")
RE_FLOAT2     = re.compile(r"^(\s*)float2 (\w+) = (.+);$")
RE_ACC_ZERO   = re.compile(r"^(\s*)\*\((acc\d+)\+(\d+)\) = 0\.0f;$")
RE_ACC_STORE  = re.compile(r"^(\s*)\*\((acc\d+)\+(\d+)\) = (.+);$")
RE_FOR        = re.compile(r"^(\s*)for \(int (\w+) = 0; \w+ < (\d+); \w+\+\+\) \{$")
RE_END_BRACE  = re.compile(r"^(\s*)\}$")
# Data store comes in two flavours:
#   *(data0_4096+(alu74+8)) = ((bfloat)((*(acc0+2))));
#   *(data0_4096+alu74)      = ((bfloat)((*(acc0+0))));
# Capture (buf, offset, rhs). The offset is everything between the
# first `+` after buf and the matching `)` that closes the outer `*(...)`.
RE_DATA_STORE = re.compile(
    r"^(\s*)\*\((data\d+_\d+)\+(.+)\) = \(\(bfloat\)\(\((.+)\)\)\);$"
)


def _split_top_level_commas(s: str) -> list[str]:
    """Split a comma-separated argument list at the top level only —
    commas inside `(...)` / `[...]` / `{...}` stay inside their group."""
    out: list[str] = []
    depth = 0
    cur: list[str] = []
    i = 0
    while i < len(s):
        c = s[i]
        if c in "([{":
            depth += 1
            cur.append(c)
        elif c in ")]}":
            depth -= 1
            cur.append(c)
        elif c == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
            # Skip following space
            if i + 1 < len(s) and s[i + 1] == " ":
                i += 1
        else:
            cur.append(c)
        i += 1
    if cur:
        out.append("".join(cur).strip())
    return out


def parse_stmt(line: str, indent_inner: str | None = None) -> Stmt | str | None:
    """Map a single line to a Stmt, or return None/'__end__'/'__forStart__'
    for control lines. `indent_inner` is unused here but kept for future
    extension."""
    if not line.strip():
        return None
    m = RE_FOR.match(line)
    if m:
        return ("__for__", m.group(2), int(m.group(3)))
    if RE_END_BRACE.match(line):
        return "__end_brace__"
    m = RE_DECL_ACC.match(line)
    if m:
        return DeclAccArray(m.group(2), int(m.group(3)))
    m = RE_INT_COMM.match(line)
    if m:
        return DeclInt(m.group(2), m.group(3), m.group(4))
    # Special-case zero-init lines — caller collapses runs of them
    m = RE_ACC_ZERO.match(line)
    if m:
        return ("__acc_zero__", m.group(2), int(m.group(3)))
    m = RE_ACC_STORE.match(line)
    if m:
        return AccStore(m.group(2), int(m.group(3)), m.group(4))
    m = RE_INT.match(line)
    if m:
        return DeclInt(m.group(2), m.group(3), None)
    m = RE_BFLOAT.match(line)
    if m:
        return DeclBfloat(m.group(2), m.group(3))
    m = RE_BFLOAT2.match(line)
    if m:
        return DeclBfloat2(m.group(2), m.group(3))
    m = RE_FLOAT2.match(line)
    if m:
        # Distinguish WMMA call from generic float2 decl
        rhs = m.group(3)
        if rhs.startswith("__WMMA_"):
            # Parse __WMMA_<...>(a, b, c)
            paren = rhs.index("(")
            prelude = rhs[:paren]
            # rhs[paren+1:-1] strips outer parens
            inner = rhs[paren + 1:-1]
            args = _split_top_level_commas(inner)
            assert len(args) == 3, f"WMMA expected 3 args, got {args} in {line!r}"
            return WmmaCall(m.group(2), prelude, args[0], args[1], args[2])
        return DeclFloat2(m.group(2), rhs)
    m = RE_DATA_STORE.match(line)
    if m:
        return DataStore(m.group(2), m.group(3), m.group(4))
    return ("__raw__", line)  # unrecognised


def parse_msl(path: Path) -> tuple[str, list[tuple[str, str, str]], list[Stmt]]:
    """Parse a captured MSL into (kernelName, argList, body).

    argList entries are tuples:
      buffer arg:  ("buffer", "<qualifier>", "<baseType>", "<name>")  [4-tuple]
      attribute :  ("attr",   "<baseType>", "<name>", "<attrStr>")     [4-tuple]
    For uniformity both are 4-tuples; caller distinguishes by entry[0].
    """
    text = path.read_text()
    lines = text.splitlines()
    # Skip prelude — we know there's exactly one WMMA prelude block per
    # capture; everything after the prelude's closing `}` is the kernel.
    i = 0
    # Skip #include and using namespace
    assert lines[0] == "#include <metal_stdlib>"
    assert lines[1] == "using namespace metal;"
    i = 2
    # WMMA prelude: from `float2 __WMMA_...(...){` to first `}` at col 0
    assert lines[i].startswith("float2 __WMMA_"), lines[i]
    i += 1
    while lines[i] != "}":
        i += 1
    i += 1
    # kernel signature
    sig = lines[i]
    m = re.match(r"^kernel void (\w+)\((.+)\) \{$", sig)
    assert m, f"bad kernel sig: {sig!r}"
    kernel_name = m.group(1)
    args_raw = _split_top_level_commas(m.group(2))
    args: list[tuple] = []
    for a in args_raw:
        # buffer: "device bfloat* <name>"
        m_buf = re.match(r"^(device(?: const)?|threadgroup) (\w+)\* (\w+)$", a)
        if m_buf:
            args.append(("buffer", m_buf.group(1), m_buf.group(2), m_buf.group(3)))
            continue
        # attr: "uint3 <name> [[...]]"
        m_at = re.match(r"^(\w+) (\w+) (\[\[.+\]\])$", a)
        if m_at:
            args.append(("attr", m_at.group(1), m_at.group(2), m_at.group(3)))
            continue
        raise AssertionError(f"unknown kernel arg form: {a!r}")
    i += 1
    # Parse body until the matching `}` at column 0
    body: list[Stmt] = []
    stack: list[list[Stmt]] = [body]
    for_ctx: list[tuple[str, int]] = []  # (ivar, hi) per open forLoop
    pending_zero_init: list[tuple[str, int]] = []  # collected (name, offset)
    def flush_zero_init():
        nonlocal pending_zero_init
        if not pending_zero_init:
            return
        # Verify contiguous 0..N-1 indices
        name = pending_zero_init[0][0]
        offsets = [p[1] for p in pending_zero_init]
        assert offsets == list(range(len(offsets))), f"non-contiguous zero init: {offsets}"
        stack[-1].append(AccZeroInit(name, len(offsets)))
        pending_zero_init = []

    while i < len(lines):
        line = lines[i]
        if line == "}":
            # End of kernel
            assert not for_ctx, f"unclosed for loops: {for_ctx}"
            flush_zero_init()
            i += 1
            break
        parsed = parse_stmt(line)
        if parsed is None:
            i += 1
            continue
        if isinstance(parsed, tuple) and parsed[0] == "__for__":
            flush_zero_init()
            _, ivar, hi = parsed
            new_body: list[Stmt] = []
            stack[-1].append(ForLoop(ivar, hi, new_body))
            stack.append(new_body)
            for_ctx.append((ivar, hi))
            i += 1
            continue
        if isinstance(parsed, str) and parsed == "__end_brace__":
            flush_zero_init()
            # End of innermost forLoop
            assert for_ctx, "unmatched `}` at line: " + repr(line)
            for_ctx.pop()
            stack.pop()
            i += 1
            continue
        if isinstance(parsed, tuple) and parsed[0] == "__acc_zero__":
            _, name, off = parsed
            pending_zero_init.append((name, off))
            i += 1
            continue
        if isinstance(parsed, tuple) and parsed[0] == "__raw__":
            raise AssertionError(f"unrecognised line at {path}:{i+1}: {line!r}")
        # Any regular Stmt — flush pending zero-init first
        flush_zero_init()
        stack[-1].append(parsed)
        i += 1
    return kernel_name, args, body


# ---------------------------------------------------------------------------
# Lean source emitter
# ---------------------------------------------------------------------------
def lean_str(s: str) -> str:
    """Lean 4 string literal — only " and \\ need escaping in our captures."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


_OFFSET_VARCONST_RE = re.compile(r'\(([A-Za-z_][A-Za-z0-9_]*)\+(\d+)\)')
_OFFSET_VAR_RE      = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)')

def parse_offset_to_uop_lean(off: str) -> str:
    """L14.B.2.b: parse a tinygrad-captured offset expression into a
    Lean `UOp` constructor call usable as a `storeIndexed` idx arg.

    Forms encountered in `fixtures/codegen/matmul_*.msl`:
      - bare var: `alu74`
      - var + literal: `(alu74+8)`
    Both render byte-equal via `UOp.renderIndexExpr` (the BinOp.add
    arm emits `(<a>+<b>)`).
    """
    s = off.strip()
    m = _OFFSET_VARCONST_RE.fullmatch(s)
    if m:
        v, c = m.group(1), m.group(2)
        return (
            f"(.binop .add (.var {lean_str(v)} .int32_) "
            f"(.const .int32_ (.i {c})) .int32_)"
        )
    m = _OFFSET_VAR_RE.fullmatch(s)
    if m:
        v = m.group(1)
        return f"(.var {lean_str(v)} .int32_)"
    raise AssertionError(f"L14.B.2.b: unparseable offset string {off!r}")


def emit_stmt_lean(s: Stmt, indent: int) -> str:
    pad = "  " * indent
    if isinstance(s, DeclAccArray):
        return f"{pad}.declAccArray {lean_str(s.name)} {s.size}"
    if isinstance(s, DeclInt):
        if s.comment is None:
            return f"{pad}.declInt {lean_str(s.name)} {lean_str(s.expr)}"
        return f"{pad}.declInt {lean_str(s.name)} {lean_str(s.expr)} (some {lean_str(s.comment)})"
    if isinstance(s, DeclBfloat):
        return f"{pad}.declBfloat {lean_str(s.name)} {lean_str(s.expr)}"
    if isinstance(s, DeclBfloat2):
        return f"{pad}.declBfloat2 {lean_str(s.name)} {lean_str(s.expr)}"
    if isinstance(s, DeclFloat2):
        return f"{pad}.declFloat2 {lean_str(s.name)} {lean_str(s.expr)}"
    if isinstance(s, AccStore):
        return f"{pad}.accStore {lean_str(s.name)} {s.offset} {lean_str(s.rhs)}"
    if isinstance(s, AccZeroInit):
        return f"{pad}.accZeroInit {lean_str(s.name)} {s.size}"
    if isinstance(s, DataStore):
        # L14.B.2.b: emit `.storeIndexed` driven by a UOp idx tree
        # (parsed from the captured offset string). The renderer's
        # output matches the prior `.dataStore` format byte-for-byte
        # because `UOp.renderIndexExpr` on the parsed UOp reproduces
        # the same `(varname+const)` parenthesisation tinygrad emits.
        idx_uop = parse_offset_to_uop_lean(s.offset)
        return f"{pad}.storeIndexed {lean_str(s.buf)} {idx_uop} {lean_str(s.rhs)}"
    if isinstance(s, WmmaCall):
        return (f"{pad}.wmmaCall {lean_str(s.out)} {lean_str(s.prelude)} "
                f"{lean_str(s.a)} {lean_str(s.b)} {lean_str(s.c)}")
    if isinstance(s, ForLoop):
        inner = ",\n".join(emit_stmt_lean(b, indent + 1) for b in s.body)
        return (f"{pad}.forLoop {lean_str(s.ivar)} {s.hi} [\n"
                f"{inner}\n{pad}]")
    raise AssertionError(f"unknown stmt: {s}")


def emit_args_lean(args: list[tuple], indent: int) -> str:
    pad = "  " * indent
    lines = []
    for a in args:
        if a[0] == "buffer":
            _, qual, base, name = a
            lines.append(f"{pad}.buffer {{ qualifier := {lean_str(qual)}, "
                         f"baseType := {lean_str(base)}, name := {lean_str(name)} }}")
        else:
            _, base, name, attr = a
            lines.append(f"{pad}.attr   {{ baseType := {lean_str(base)}, "
                         f"name := {lean_str(name)}, attrStr := {lean_str(attr)} }}")
    return ",\n".join(lines)


def emit_kernel_decl(sentinel: str, kernel_name: str,
                     args: list[tuple], body: list[Stmt],
                     trailing_newline: bool) -> str:
    args_lean = emit_args_lean(args, indent=2)
    body_lean = ",\n".join(emit_stmt_lean(b, indent=2) for b in body)
    trailer = "true" if trailing_newline else "false"
    return (f"-- {sentinel} — kernel {kernel_name} ({len(body)} body stmts)\n"
            f"private def {sentinel}_decl : KernelDecl :=\n"
            f"  {{ name     := {lean_str(kernel_name)},\n"
            f"    wmmaArgs := [WmmaArg.bf16Float],\n"
            f"    args     := [\n{args_lean}\n    ],\n"
            f"    body     := [\n{body_lean}\n    ],\n"
            f"    trailingNewline := {trailer} }}\n")


def main():
    repo = Path(__file__).resolve().parents[2]
    msl_dir = repo / "fixtures" / "codegen"
    out_path = repo / "Tgrad" / "Renderer" / "MatmulDecls.lean"

    parts: list[str] = []
    parts.append('import Tgrad.Renderer.Metal')
    parts.append('')
    parts.append('/-! # Tgrad.Renderer.MatmulDecls — L12 algebraic matmul KernelDecls')
    parts.append('')
    parts.append('  Generated by `scripts/dev/lower_matmul.py` from the 10')
    parts.append('  captured matmul kernels in `fixtures/codegen/`. Each')
    parts.append('  per-shape `KernelDecl` is structurally encoded — when its')
    parts.append('  `renderKernel` output is byte-equal to the captured `.msl`,')
    parts.append('  L12 is green for that shape.')
    parts.append('')
    parts.append('  This file is regenerated by re-running the transpiler:')
    parts.append('    python3 scripts/dev/lower_matmul.py')
    parts.append('  The generator is dev-time only; at runtime / gate-time the')
    parts.append('  rendered output of these decls is what L12.sh diffs against')
    parts.append('  the captured fixtures.')
    parts.append('-/')
    parts.append('namespace Tgrad')
    parts.append('namespace Renderer')
    parts.append('namespace Metal')
    parts.append('')

    for stem, sentinel, _ in SHAPE_TRIPLES:
        msl_path = msl_dir / f"{stem}.msl"
        if not msl_path.exists():
            print(f"WARN: missing {msl_path}", file=sys.stderr)
            continue
        kernel_name, args, body = parse_msl(msl_path)
        # Pin trailing-newline from the captured fixture so the renderKernel
        # output is byte-identical (the 10 production captures end with `}`
        # but matmul_64x64 ends with `}\n`).
        raw = msl_path.read_bytes()
        trailing_newline = raw.endswith(b"\n")
        parts.append(emit_kernel_decl(sentinel, kernel_name, args, body,
                                      trailing_newline))

    # Dispatcher
    parts.append('/-- Dispatcher: ShapeSentinel → KernelDecl. -/')
    parts.append('def matmulKernelDeclFor : ShapeSentinel → KernelDecl')
    for _, sentinel, _ in SHAPE_TRIPLES:
        parts.append(f"  | .{sentinel} => {sentinel}_decl")

    parts.append('')
    parts.append('end Metal')
    parts.append('end Renderer')
    parts.append('end Tgrad')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts) + "\n")
    n_shapes = sum(1 for stem, _, _ in SHAPE_TRIPLES if (msl_dir / f"{stem}.msl").exists())
    print(f"wrote {out_path} ({n_shapes} shapes)")


if __name__ == "__main__":
    main()
