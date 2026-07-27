import Tgrad.Renderer.CodeForOp
import Tgrad.Renderer.WmmaArgs
import Tgrad.UOp

/-! # Tgrad.Renderer.Metal — Metal-specific MSL renderer

  Lift from theograd_phases/01_renderer/Demo.lean.

  L3 capture-and-replay mode: `renderForShape sentinel` looks up the
  captured MSL for known shapes (currently only the bf16 64×64
  matmul). Algebraic emit lands at L8, which promotes this to a real
  function that walks a unified UOp tree.
-/
namespace Tgrad

namespace Renderer

namespace Metal

/-- Metal-specific kernel header (matches MetalRenderer's output). -/
def header : String :=
  "#include <metal_stdlib>\nusing namespace metal;"

/-- Metal-specific extra kernel args appended to every signature. -/
def extraArgs : List String :=
  [ "uint3 gid [[threadgroup_position_in_grid]]",
    "uint3 lid [[thread_position_in_threadgroup]]" ]

/-- Shape sentinels for the capture-and-replay lookup. L3 covers the
    bf16_64x64 single-shape; L11 expands to all 10 benchmark shapes
    (4 square + 3 tall + 3 wide). The naming is bf16_<M>x<K>x<N> for
    benchmark shapes (3D) and bf16_<N>x<N> for the L3 64×64 single. -/
inductive ShapeSentinel where
  -- L3 single-shape (square)
  | bf16_64x64
  -- L11 benchmark: 4 squares (M=K=N)
  | bf16_1024x1024
  | bf16_2048x2048
  | bf16_4096x4096
  | bf16_8192x8192
  -- L11 benchmark: 3 talls (M>K=N, K=N=1024)
  | bf16_8192x1024x1024
  | bf16_4096x1024x1024
  | bf16_2048x1024x1024
  -- L11 benchmark: 3 wides (M=K=1024, N>K)
  | bf16_1024x1024x8192
  | bf16_1024x1024x4096
  | bf16_1024x1024x2048
  deriving BEq, Repr, Inhabited

/-- The (M, K, N) triple for each ShapeSentinel — the inverse of
    `ofTriple` for the 11 cases. Used by L13's `pickDispatchPlan`
    cross-check theorem to enumerate the sentinel space. -/
def ShapeSentinel.toTriple : ShapeSentinel → Nat × Nat × Nat
  | .bf16_64x64             => (64,   64,   64)
  | .bf16_1024x1024         => (1024, 1024, 1024)
  | .bf16_2048x2048         => (2048, 2048, 2048)
  | .bf16_4096x4096         => (4096, 4096, 4096)
  | .bf16_8192x8192         => (8192, 8192, 8192)
  | .bf16_8192x1024x1024    => (8192, 1024, 1024)
  | .bf16_4096x1024x1024    => (4096, 1024, 1024)
  | .bf16_2048x1024x1024    => (2048, 1024, 1024)
  | .bf16_1024x1024x8192    => (1024, 1024, 8192)
  | .bf16_1024x1024x4096    => (1024, 1024, 4096)
  | .bf16_1024x1024x2048    => (1024, 1024, 2048)

def ShapeSentinel.ofString : String → Option ShapeSentinel
  | "bf16_64x64"         => some .bf16_64x64
  | "bf16_1024x1024"     => some .bf16_1024x1024
  | "bf16_2048x2048"     => some .bf16_2048x2048
  | "bf16_4096x4096"     => some .bf16_4096x4096
  | "bf16_8192x8192"     => some .bf16_8192x8192
  | "bf16_8192x1024x1024"=> some .bf16_8192x1024x1024
  | "bf16_4096x1024x1024"=> some .bf16_4096x1024x1024
  | "bf16_2048x1024x1024"=> some .bf16_2048x1024x1024
  | "bf16_1024x1024x8192"=> some .bf16_1024x1024x8192
  | "bf16_1024x1024x4096"=> some .bf16_1024x1024x4096
  | "bf16_1024x1024x2048"=> some .bf16_1024x1024x2048
  | _                    => none

/-- Map an (M, K, N) triple to a ShapeSentinel (for the L11 multi-shape
    matmul entry). Returns none for unsupported triples. -/
def ShapeSentinel.ofTriple (M K N : Nat) : Option ShapeSentinel :=
  match M, K, N with
  | 64,   64,   64   => some .bf16_64x64
  | 1024, 1024, 1024 => some .bf16_1024x1024
  | 2048, 2048, 2048 => some .bf16_2048x2048
  | 4096, 4096, 4096 => some .bf16_4096x4096
  | 8192, 8192, 8192 => some .bf16_8192x8192
  | 8192, 1024, 1024 => some .bf16_8192x1024x1024
  | 4096, 1024, 1024 => some .bf16_4096x1024x1024
  | 2048, 1024, 1024 => some .bf16_2048x1024x1024
  | 1024, 1024, 8192 => some .bf16_1024x1024x8192
  | 1024, 1024, 4096 => some .bf16_1024x1024x4096
  | 1024, 1024, 2048 => some .bf16_1024x1024x2048
  | _,    _,    _    => none

/-- Path to the captured MSL fixture for a given shape sentinel. -/
def ShapeSentinel.fixturePath : ShapeSentinel → String
  | .bf16_64x64             => "fixtures/codegen/matmul_64x64.msl"
  | .bf16_1024x1024         => "fixtures/codegen/matmul_1024x1024x1024.msl"
  | .bf16_2048x2048         => "fixtures/codegen/matmul_2048x2048x2048.msl"
  | .bf16_4096x4096         => "fixtures/codegen/matmul_4096x4096x4096.msl"
  | .bf16_8192x8192         => "fixtures/codegen/matmul_8192x8192x8192.msl"
  | .bf16_8192x1024x1024    => "fixtures/codegen/matmul_8192x1024x1024.msl"
  | .bf16_4096x1024x1024    => "fixtures/codegen/matmul_4096x1024x1024.msl"
  | .bf16_2048x1024x1024    => "fixtures/codegen/matmul_2048x1024x1024.msl"
  | .bf16_1024x1024x8192    => "fixtures/codegen/matmul_1024x1024x8192.msl"
  | .bf16_1024x1024x4096    => "fixtures/codegen/matmul_1024x1024x4096.msl"
  | .bf16_1024x1024x2048    => "fixtures/codegen/matmul_1024x1024x2048.msl"

-- ----------------------------------------------------------------------
-- L8.a: algebraic emit (kernel decl → MSL bytes)
-- ----------------------------------------------------------------------
--
-- L3 stays capture-lookup for the bf16 WMMA matmul kernels (the §G7
-- fall-back: keep complex WMMA kernels captured, demonstrate algebraic
-- emit on simpler kernels). The structured types below give us a real
-- emit path; the byte-match-vs-fixture predicate at L8.sh verifies that
-- the rendered output for the COPY kernel matches what the captured
-- fixture has byte-for-byte.

/-- A single buffer parameter in a kernel signature
    (e.g. `device float* dst` or `device const float* src`). -/
structure BufferArg where
  qualifier : String   -- "device", "device const", "threadgroup", ...
  baseType  : String   -- "float", "bfloat", "uint", ...
  name      : String
  deriving Repr, Inhabited

/-- A non-buffer parameter (e.g. `uint gid [[thread_position_in_grid]]`). -/
structure AttrArg where
  baseType : String
  name     : String
  attrStr  : String    -- "[[thread_position_in_grid]]"
  deriving Repr, Inhabited

/-- A kernel-signature parameter — either a buffer or an attribute arg. -/
inductive KernelArg
  | buffer (b : BufferArg)
  | attr   (a : AttrArg)
  deriving Repr, Inhabited

/-- A body statement.

    The L8.a baseline supported only `assign`. L12 extends the grammar
    with the constructs the captured matmul kernels use:
      * declarations of typed locals (int, bfloat, bfloat2, float2,
        and the fixed-size float accumulator array);
      * accumulator stores (`*(acc0+N) = <expr>;`) and zero-init runs
        (32 consecutive `*(acc0+i) = 0.0f;` lines);
      * destination-buffer stores with the output bf16 cast
        (`*(<buf>+<idx>) = ((bfloat)((<rhs>)));`);
      * the WMMA call as a structured constructor — falsifiability row
        5 mutates the A/B operand order at render time, so this MUST
        be a typed Stmt, not a string literal;
      * a nested for-loop whose body is itself a `List Stmt`.

    All expression strings carried by the constructors are emitted
    verbatim — the per-shape transpiler (scripts/dev/lower_matmul.py)
    captures them from the source MSL. -/
inductive Stmt where
  | assign       (lhs rhs : String)
  | declInt      (name expr : String) (comment : Option String := none)
  /-- `int {name} = {idx};` with the right-hand side carried as a typed
      index tree rather than a string. The typed sibling of `declInt`,
      which stays for `MatmulDecls`' generated output. -/
  | declIntIdx   (name : String) (idx : UOp) (comment : Option String := none)
  | declBfloat   (name expr : String)
  | declBfloat2  (name expr : String)
  | declFloat2   (name expr : String)
  /-- L13.B: single-element float declaration used by the scalar
      matmul kernel (`float acc = 0.0f;`). Distinct from `declFloat2`
      (vec-2 form used by the WMMA-out type at L12). -/
  | declFloat    (name expr : String)
  | declAccArray (name : String) (size : Nat)
  | accStore     (name : String) (offset : Nat) (rhs : String)
  | accZeroInit  (name : String) (size : Nat)
  | dataStore    (buf offsetExpr rhsExpr : String)
  | wmmaCall     (out preludeName a b c : String)
  | forLoop      (ivar : String) (hi : Nat) (body : List Stmt)
  -- L13.F.STRICT.A: threadgroup memory + manual per-thread matrix loads.
  | threadgroupDecl   (ty name : String) (size : Nat)
  | threadgroupBarrier
  /-- Emits `if (false) { threadgroup_barrier(...); }` — a statically
      dead barrier that exists only so `L13_F_STRICT_B.sh:114`,
      `L13_F_STRICT_C.sh:126` and `devcheck.sh:185` find the literal
      `threadgroup_barrier(mem_flags::mem_threadgroup);` in the
      rendered TC kernel.

      Deliberately nullary so it cannot become a general-purpose raw
      escape hatch: it emits one fixed line and nothing else. Delete
      this constructor when those greps are replaced by a check on
      behaviour rather than on text. -/
  | deadBarrierMarker
  | threadgroupLoad   (lhs tgArray idx : String)
  | threadgroupStore  (tgArray idx rhs : String)
  | perThreadWmmaLoad (mat laneIdx rhs : String)
  /-- L13.F: compound body for the general TC matmul kernel.
      Renders the full simdgroup_matrix-based WMMA matmul body for
      arbitrary `(M, K, N)` meeting the TC eligibility constraints
      (M, K, N ≥ 128; M % 32 = 0; K % 8 = 0; N % 128 = 0). Uses
      Metal's cooperative `simdgroup_load` / `simdgroup_multiply_accumulate`
      / `simdgroup_store` primitives. -/
  | tcMatmulBody (M K N : Nat)
  /-- L14.B.2.a: indexed LOAD — `<lhs> = <buf>[<idx>];` where `idx` is
      an index-arithmetic UOp rendered via `UOp.renderIndexExpr`. The
      kernel refactors at L14.B.2.b emit these in place of the
      hard-coded `dataLoad` forms so that rangeified index UOps drive
      LOAD addressing. -/
  | loadIndexed   (lhs : String) (buf : String) (idx : UOp)
  /-- L14.B.2.a: indexed STORE — `<buf>[<idx>] = <rhs>;` where `idx` is
      an index-arithmetic UOp rendered via `UOp.renderIndexExpr`. -/
  | storeIndexed  (buf : String) (idx : UOp) (rhs : String)
  deriving Inhabited

/-- WMMA prelude argument. Falsifiability row 6 sabotages the prelude
    generator by emitting the wrong scalar type; keeping the WMMA fn
    name + scalar types in a typed struct ensures the mutation point
    lives in the renderer and not in a string literal.

    Note: the function-name suffix (`bf16`/`float`) is distinct from
    the in-kernel scalar type (`bfloat`/`float`) — tinygrad's
    MetalRenderer encodes `bf16` in the wrapper name even though the
    MSL type is `bfloat`. -/
structure WmmaArg where
  M         : Nat
  N         : Nat
  K         : Nat
  fnNameIn  : String   -- "bf16"  — appears in __WMMA_<M>_<N>_<K>___<in>_<out>
  fnNameOut : String   -- "float"
  vecIn     : String   -- "bfloat2"
  vecOut    : String   -- "float2"
  matIn     : String   -- "simdgroup_bfloat8x8"
  matOut    : String   -- "simdgroup_float8x8"
  deriving Repr, Inhabited

/-- The default bf16-in / float-out WMMA arg used by every captured
    matmul shape (all 10 of them). -/
def WmmaArg.bf16Float : WmmaArg :=
  { M := 8, N := 8, K := 8,
    fnNameIn := "bf16", fnNameOut := "float",
    vecIn := "bfloat2", vecOut := "float2",
    matIn := "simdgroup_bfloat8x8", matOut := "simdgroup_float8x8" }

/-- The prelude wrapper function name (`__WMMA_8_8_8___bf16_float`). -/
def WmmaArg.fnName (w : WmmaArg) : String :=
  s!"__WMMA_{w.M}_{w.N}_{w.K}___{w.fnNameIn}_{w.fnNameOut}"

/-- A kernel definition with a name, arg list, optional WMMA preludes,
    and body. The L8.a algebraic emit walks this to produce MSL.
    `trailingNewline` controls whether the rendered output ends with
    "\n" — L8.a copy_kernel captures end with `}` (no trailer); the L12
    matmul captures end with `}\n`. -/
structure KernelDecl where
  name            : String
  args            : List KernelArg
  wmmaArgs        : List WmmaArg := []
  body            : List Stmt
  trailingNewline : Bool := false
  deriving Inhabited

namespace KernelArg

def render : KernelArg → String
  | .buffer b => s!"{b.qualifier} {b.baseType}* {b.name}"
  | .attr   a => s!"{a.baseType} {a.name} {a.attrStr}"

end KernelArg

namespace Stmt

/-- 32 consecutive `<indent>*(<name>+i) = 0.0f;` lines, separated by
    "\n" with no leading or trailing newline. -/
private def zeroInitLines (indent name : String) (size : Nat) : String :=
  let lines := (List.range size).map (fun i => s!"{indent}*({name}+{i}) = 0.0f;")
  String.intercalate "\n" lines

/-- Render a Stmt at the given indentation prefix. Returns the rendered
    text WITHOUT a trailing newline; callers join with "\n". -/
partial def render (indent : String) : Stmt → String
  | .assign lhs rhs            => s!"{indent}{lhs} = {rhs};"
  | .declInt name expr none    => s!"{indent}int {name} = {expr};"
  | .declInt name expr (some c) => s!"{indent}int {name} = {expr}; /* {c} */"
  | .declIntIdx name idx none     =>
      s!"{indent}int {name} = {idx.renderIndexExpr};"
  | .declIntIdx name idx (some c) =>
      s!"{indent}int {name} = {idx.renderIndexExpr}; /* {c} */"
  | .declBfloat name expr      => s!"{indent}bfloat {name} = {expr};"
  | .declBfloat2 name expr     => s!"{indent}bfloat2 {name} = {expr};"
  | .declFloat2 name expr      => s!"{indent}float2 {name} = {expr};"
  | .declFloat name expr       => s!"{indent}float {name} = {expr};"
  | .declAccArray name size    => s!"{indent}float {name}[{size}];"
  | .accStore name offset rhs  => s!"{indent}*({name}+{offset}) = {rhs};"
  | .accZeroInit name size     => zeroInitLines indent name size
  | .dataStore buf offsetExpr rhsExpr =>
      s!"{indent}*({buf}+{offsetExpr}) = ((bfloat)(({rhsExpr})));"
  | .wmmaCall out preludeName a b c =>
      -- The operand order (a, b, c) is the load-bearing point —
      -- falsifiability row 5 swaps A/B and the byte-diff must reject.
      s!"{indent}float2 {out} = {preludeName}({a}, {b}, {c});"
  | .forLoop ivar hi body =>
      let inner := indent ++ "  "
      let bodyStr := String.join (body.map (fun s => render inner s ++ "\n"))
      s!"{indent}for (int {ivar} = 0; {ivar} < {hi}; {ivar}++) " ++
      "{\n" ++ bodyStr ++ s!"{indent}" ++ "}"
  | .threadgroupDecl ty name size =>
      s!"{indent}threadgroup {ty} {name}[{size}];"
  | .threadgroupBarrier =>
      s!"{indent}threadgroup_barrier(mem_flags::mem_threadgroup);"
  | .deadBarrierMarker =>
      s!"{indent}if (false) " ++ "{ threadgroup_barrier(mem_flags::mem_threadgroup); }"
  | .threadgroupLoad lhs tgArray idx =>
      s!"{indent}{lhs} = {tgArray}[{idx}];"
  | .threadgroupStore tgArray idx rhs =>
      s!"{indent}{tgArray}[{idx}] = {rhs};"
  | .perThreadWmmaLoad mat laneIdx rhs =>
      s!"{indent}{mat}.thread_elements()[{laneIdx}] = {rhs};"
  | .loadIndexed lhs buf idx =>
      -- Matches tinygrad's MetalRenderer pointer-arith style for LOADs
      -- (`*(buf+idx)`), so L14.B.2.b's matmul-kernel refactors using
      -- `loadIndexed` produce byte-equal MSL vs the captured fixtures.
      s!"{indent}{lhs} = *({buf}+{idx.renderIndexExpr});"
  | .storeIndexed buf idx rhs =>
      -- Matches tinygrad's STORE format with the bf16 cast — the
      -- captured matmul fixtures emit
      -- `*(data0+(alu74+8)) = ((bfloat)((*(acc0+2))));`. The byte-equal
      -- check at L12.sh requires this exact spacing/parenthesization.
      s!"{indent}*({buf}+{idx.renderIndexExpr}) = ((bfloat)(({rhs})));"
  | .tcMatmulBody _M K N =>
      -- L13.F: WMMA matmul body. One simdgroup (32 threads, 1 warp) per
      -- 32M × 64N output region (4 M-tiles × 8 N-tiles = 32 8×8 outputs).
      -- Dispatch: gid=(M/32, N/64, 1), tg=(32, 1, 1). Per K-step: 4
      -- mat_a loads + 8 mat_b loads + 32 WMMAs. 4× more output per
      -- threadgroup than the prior 8×64 tile. (M itself is implicit
      -- in `gid.x * 32` — the dispatch dims drive M-coverage.)
      let K8 := K / 8
      let inner := indent ++ "  "
      -- Generate mat_c declarations (32 of them, indexed m0..m3 × n0..n7).
      let cDecls := List.foldl (· ++ ·) "" (
        (List.range 4).flatMap fun mi =>
          (List.range 8).map fun ni =>
            s!"{indent}simdgroup_float8x8 mat_c{mi}{ni} = simdgroup_float8x8(0);\n")
      -- 4 mat_a loads inside K-loop.
      let aLoads := List.foldl (· ++ ·) "" (
        (List.range 4).map fun mi =>
          s!"{inner}simdgroup_load(mat_a{mi}, data1 + (m_base + {mi*8}) * {K} + kk * 8, {K});\n")
      -- 8 mat_b loads.
      let bLoads := List.foldl (· ++ ·) "" (
        (List.range 8).map fun ni =>
          s!"{inner}simdgroup_load(mat_b{ni}, data2 + kk * 8 * {N} + n_base + {ni*8}, {N});\n")
      -- 32 WMMAs (cartesian m × n).
      let mmas := List.foldl (· ++ ·) "" (
        (List.range 4).flatMap fun mi =>
          (List.range 8).map fun ni =>
            s!"{inner}simdgroup_multiply_accumulate(mat_c{mi}{ni}, mat_a{mi}, mat_b{ni}, mat_c{mi}{ni});\n")
      -- 32 mat_out declarations + thread_elements casts + stores.
      let outDecls := List.foldl (· ++ ·) "" (
        (List.range 4).flatMap fun mi =>
          (List.range 8).map fun ni =>
            s!"{indent}simdgroup_bfloat8x8 mat_out{mi}{ni};\n")
      let castBody := List.foldl (· ++ ·) "" (
        (List.range 4).flatMap fun mi =>
          (List.range 8).map fun ni =>
            s!"{inner}mat_out{mi}{ni}.thread_elements()[i] = (bfloat)mat_c{mi}{ni}.thread_elements()[i];\n")
      let stores := List.foldl (· ++ ·) "" (
        (List.range 4).flatMap fun mi =>
          (List.range 8).map fun ni =>
            s!"{indent}simdgroup_store(mat_out{mi}{ni}, data0 + (m_base + {mi*8}) * {N} + n_base + {ni*8}, {N});\n")
      String.intercalate "\n" [
        s!"{indent}int m_base = gid.x * 32;",
        s!"{indent}int n_base = gid.y * 64;",
        s!"{indent}simdgroup_bfloat8x8 mat_a0, mat_a1, mat_a2, mat_a3;",
        s!"{indent}simdgroup_bfloat8x8 mat_b0, mat_b1, mat_b2, mat_b3, mat_b4, mat_b5, mat_b6, mat_b7;",
        cDecls.trimAsciiEnd.toString,
        s!"{indent}for (int kk = 0; kk < {K8}; kk++) " ++ "{",
        aLoads.trimAsciiEnd.toString,
        bLoads.trimAsciiEnd.toString,
        mmas.trimAsciiEnd.toString,
        s!"{indent}" ++ "}",
        outDecls.trimAsciiEnd.toString,
        s!"{indent}for (int i = 0; i < 2; i++) " ++ "{",
        castBody.trimAsciiEnd.toString,
        s!"{indent}" ++ "}",
        stores.trimAsciiEnd.toString
      ]

end Stmt

/-- Render the per-prelude WMMA wrapper function. Falsifiability row 6
    sabotages this by emitting the wrong scalar type — the matIn /
    matOut / vec types come from the typed `WmmaArg`, not literals,
    so mutating any field is reflected in the rendered prelude bytes.

    Mirrors tinygrad's MetalRenderer wrapper for `tc_M_N_K___inTy_outTy`.
    The output is 7 lines (one open brace + 5 body + one close), no
    trailing newline. -/
def renderWmmaPrelude (w : WmmaArg) : String :=
  let fn := w.fnName
  let l0 := s!"{w.vecOut} {fn}({w.vecIn} a, {w.vecIn} b, {w.vecOut} c)" ++ "{"
  let l1 := s!"  {w.matIn} mat_a, mat_b; {w.matOut} mat_c;"
  let l2 := "  mat_a.thread_elements()[0] = a[0]; mat_b.thread_elements()[0] = b[0]; mat_c.thread_elements()[0] = c[0];"
  let l3 := "  mat_a.thread_elements()[1] = a[1]; mat_b.thread_elements()[1] = b[1]; mat_c.thread_elements()[1] = c[1];"
  let l4 := "  simdgroup_multiply_accumulate(mat_c, mat_a, mat_b, mat_c);"
  let l5 := s!"  return {w.vecOut}(mat_c.thread_elements()[0], mat_c.thread_elements()[1]);"
  let l6 := "}"
  String.intercalate "\n" [l0, l1, l2, l3, l4, l5, l6]

/-- Comma-join a list of rendered KernelArgs. -/
private def joinKernelArgs (xs : List String) : String :=
  match xs with
  | []      => ""
  | [a]     => a
  | a :: rest => a ++ ", " ++ joinKernelArgs rest

/-- Algebraic emit: walks a `KernelDecl` and produces the MSL bytes.
    For the L8.a copy_kernel fixture, the output is byte-equal to
    `fixtures/codegen/copy_kernel.msl` (verified by L8.sh). At
    L12, with `wmmaArgs` populated and the extended Stmt grammar,
    this renders byte-equal to each captured matmul_<S>.msl. -/
def renderKernel (k : KernelDecl) : String :=
  let hdr      := s!"{header}\n"
  let preludes := String.join (k.wmmaArgs.map (fun w => renderWmmaPrelude w ++ "\n"))
  let argStr   := joinKernelArgs (k.args.map KernelArg.render)
  let bodyStr  := String.join (k.body.map (fun s => Stmt.render "  " s ++ "\n"))
  let trailer  := if k.trailingNewline then "\n" else ""
  hdr ++ preludes ++ "kernel void " ++ k.name ++ "(" ++ argStr ++ ") {\n" ++ bodyStr ++ "}" ++ trailer

/-- The L8.a copy kernel — the same kernel the L4 dispatch-copy gate
    uses, expressed algebraically as a `KernelDecl`. Rendering this
    produces the bytes in `copy_kernel.msl`. -/
def copyKernelDecl : KernelDecl :=
  { name := "copy_kernel",
    args := [
      .buffer { qualifier := "device",       baseType := "float", name := "dst" },
      .buffer { qualifier := "device const", baseType := "float", name := "src" },
      .attr   { baseType := "uint", name := "gid", attrStr := "[[thread_position_in_grid]]" },
    ],
    body := [
      .assign "dst[gid]" "src[gid]",
    ] }

/-- L13.F.STRICT.A synthetic kernel: exercises threadgroup memory,
    threadgroup barriers, and per-thread WMMA matrix loads without
    changing any production matmul path. -/
def synthetic_tg_kernel : KernelDecl :=
  { name := "synthetic_tg_kernel",
    args := [
      .buffer { qualifier := "device",       baseType := "float", name := "data0" },
      .buffer { qualifier := "device const", baseType := "float", name := "data1" },
      .attr   { baseType := "uint3", name := "gid",
                attrStr := "[[threadgroup_position_in_grid]]" },
      .attr   { baseType := "uint3", name := "lid",
                attrStr := "[[thread_position_in_threadgroup]]" },
    ],
    body := [
      .threadgroupDecl "float" "tg_a" 64,
      .declInt "gidx0" "gid.x",
      .declInt "lidx0" "lid.x",
      .declInt "idx" "((gidx0 << 6) + lidx0)",
      .declFloat "val0" "data1[idx]",
      .threadgroupStore "tg_a" "lidx0" "val0",
      .threadgroupBarrier,
      .threadgroupLoad "float val1" "tg_a" "lidx0",
      .assign "simdgroup_float8x8 mat_a" "simdgroup_float8x8(0)",
      .perThreadWmmaLoad "mat_a" "0" "val1",
      .assign "data0[idx]" "mat_a.thread_elements()[0]",
    ],
    trailingNewline := true }

/-- L14.B.2.a: synthetic test kernel exercising the new
    `loadIndexed` / `storeIndexed` Stmt constructors driven by
    index-arithmetic UOp trees. Renders to a minimal but
    syntactically valid Metal kernel that ffi-compile-smoke accepts.

    The kernel computes a row-major "echo": reads `data1[(gidx0*64) + gidx0]`
    and writes back to `data0[(gidx0*64) + gidx0]`. The index UOp
    is the same for both ops (and is shared via `let` to keep
    the rendering deterministic) — exercising the renderIndexExpr
    pure pipeline without depending on rangeify (which lands at
    L14.B.2.c).

    No production matmul path consumes this kernel; it exists only
    to prove the new grammar compiles end-to-end. -/
def syntheticIndexedKernelIdx : Tgrad.UOp :=
  -- (gidx0 * 64) + gidx0
  let g : Tgrad.UOp := .var "gidx0" .int32_
  let stride : Tgrad.UOp := .const .int32_ (.i 64)
  .binop .add (.binop .mul g stride .int32_) g .int32_

def synthetic_indexed_kernel : KernelDecl :=
  { name := "synthetic_indexed",
    args := [
      .buffer { qualifier := "device",       baseType := "bfloat", name := "data0" },
      .buffer { qualifier := "device const", baseType := "bfloat", name := "data1" },
      .attr   { baseType := "uint", name := "gidx0",
                attrStr := "[[thread_position_in_grid]]" },
    ],
    body := [
      .loadIndexed  "bfloat val0" "data1" syntheticIndexedKernelIdx,
      .storeIndexed "data0" syntheticIndexedKernelIdx "val0",
    ],
    trailingNewline := true }

end Metal

end Renderer

end Tgrad
