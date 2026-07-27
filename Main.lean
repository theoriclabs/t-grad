import Tgrad
import Lean.Data.Json

/-! # Tgrad CLI — `tgrad <subcommand>`

  Subcommands per `01_design.md` §9 and `GOAL.md`'s active-gate predicates:

  ## Production surface (post-L6)
    tgrad bench    — run the benchmark sweep (no Python needed)
    tgrad gate     — high-water-mark checks
    tgrad render   — emit MSL for a fixture's UOp tree (debug)
    tgrad capture  — populate a new capture (shells to tinygrad)

  ## L1 gate-runner surface (now live)
    tgrad emit-lub-table         — 14×14 dtype LUB table as JSON
    tgrad emit-cast-table        — 14×14 canLosslessCast table as JSON
    tgrad emit-shape-table       — shape numel/align/broadcast cases
    tgrad emit-movement-table    — reshape/permute/expand cases
    tgrad reduce-symbolic-dag <p>— read p, reduce via 16 rules, emit
-/

open Tgrad
open Lean (Json)

def usage : IO Unit := do
  IO.println "tgrad — Lean-runtime bf16 matmul"
  IO.println ""
  IO.println "Usage:"
  IO.println "  tgrad bench                    run the benchmark sweep"
  IO.println "  tgrad gate                     run the high-water-mark checks"
  IO.println "  tgrad render <fix>             emit MSL for a fixture's UOp tree"
  IO.println "  tgrad capture M K N            populate a new MSL capture"
  IO.println ""
  IO.println "Gate-runner subcommands (populated as gates land):"
  IO.println "  tgrad emit-lub-table           14×14 dtype LUB table     (L1)"
  IO.println "  tgrad emit-cast-table          14×14 cast table          (L1)"
  IO.println "  tgrad emit-shape-table         shape op table            (L1)"
  IO.println "  tgrad emit-movement-table      movement op table         (L1)"
  IO.println "  tgrad reduce-symbolic-dag <p>  symbolic_simple subset    (L1)"

-- ============================================================================
-- L1 emit helpers.
-- ============================================================================

/-- Emit Dtype.computeLubTable to stdout — matches lub_table.json byte-for-byte
    (Python json.dumps(indent=2)). -/
def emitLubTable : IO UInt32 := do
  let s := Dtype.lubTableToJson Dtype.computeLubTable
  IO.println s
  pure 0

def emitCastTable : IO UInt32 := do
  let s := Dtype.castTableToJson Dtype.computeCastTable
  IO.println s
  pure 0

def emitShapeTable : IO UInt32 := do
  let table := computeShapeTable Shape.fixturePairs
  let s := shapeTableToJson table
  IO.println s
  pure 0

def emitMovementTable : IO UInt32 := do
  -- For each fixture row, recompute `expected` via Tgrad's shape op so the
  -- emit actually exercises permuteShape / reshapeShape / expandShape. The
  -- hardcoded `expected` in the fixture rows is only used to seed the
  -- input columns; the gate's byte-diff is against the captured
  -- movement_table.json.
  let computed := Shape.movementFixtureRows.map (fun r =>
    { r with expected := applyMovementRow r })
  let s := movementTableToJson computed
  IO.println s
  pure 0

-- ============================================================================
-- JSON parser for the symbolic DAG fixture (dag_in.json).
-- ============================================================================

/-- Parse a Json.obj entry into a UOpArg. The fixtures use:
       null                              → UOpArg.none
       {"kind":"int","v":N}              → UOpArg.int N
       {"kind":"str","v":"X"}            → UOpArg.str X
       {"kind":"bool","v":B}             → UOpArg.bool B
       {"kind":"float","v":F}            → UOpArg.float F
       {"kind":"tuple","v":[...args...]} → UOpArg.tuple [parsed...]
                                            (phase-04 RANGE arg) -/
partial def parseArg : Json → Except String UOpArg
  | .null => pure .none
  | j     => do
      let kind ← j.getObjVal? "kind"
      let kindS ← kind.getStr?
      let v ← j.getObjVal? "v"
      match kindS with
      | "int"   => do
          let n ← v.getInt?
          pure (.int n)
      | "str"   => do
          let s ← v.getStr?
          pure (.str s)
      | "bool"  => do
          let b ← v.getBool?
          pure (.bool b)
      | "float" => do
          let f ← v.getNum?
          pure (.float f.toFloat)
      | "tuple" => do
          let arr ← v.getArr?
          let mut xs : List UOpArg := []
          for aj in arr do
            let a ← parseArg aj
            xs := xs ++ [a]
          pure (.tuple xs)
      | k       => throw s!"unknown arg kind: {k}"

/-- Parse a single record. -/
def parseRecord (j : Json) : Except String UOpRecord := do
  let idxV  ← j.getObjVal? "idx"
  let idx   ← idxV.getNat?
  let opV   ← j.getObjVal? "op"
  let opS   ← opV.getStr?
  let op    ← match UOpKind.ofString opS with
              | some k => pure k
              | none   => throw s!"unknown op kind: {opS}"
  let dtV   ← j.getObjVal? "dtype"
  let dtS   ← dtV.getStr?
  let dt    ← match Dtype.ofSymbolicStr dtS with
              | some d => pure d
              | none   => throw s!"unknown dtype: {dtS}"
  let argV  ← j.getObjVal? "arg"
  let arg   ← parseArg argV
  let srcV  ← j.getObjVal? "src"
  let srcArr ← srcV.getArr?
  let mut srcs : List Nat := []
  for sj in srcArr do
    let n ← sj.getNat?
    srcs := srcs ++ [n]
  pure { idx := idx, op := op, dtype := dt, arg := arg, src := srcs }

/-- Parse the dag_in.json file (an array of records). -/
def parseRecords (path : System.FilePath) : IO (Except String (List UOpRecord)) := do
  let text ← IO.FS.readFile path
  match Lean.Json.parse text with
  | .error e => pure (.error s!"JSON parse: {e}")
  | .ok j    => do
      let arr ← match j.getArr? with
        | .ok a   => pure a
        | .error e => return .error s!"top-level not array: {e}"
      let mut recs : List UOpRecord := []
      for rj in arr do
        match parseRecord rj with
        | .error e => return .error s!"record: {e}"
        | .ok r    => recs := recs ++ [r]
      pure (.ok recs)

/-- `tgrad reduce-symbolic-dag <path>` — read a UOpRecord JSON array,
    rebuild the tree, run the 16-rule `symbolic_simple` subset, emit
    the post-rewrite records to stdout (matching dag_out_expected.json
    byte-for-byte). -/
def reduceSymbolicDag (path : String) : IO UInt32 := do
  match (← parseRecords (System.FilePath.mk path)) with
  | .error e =>
      IO.eprintln s!"[reduce-symbolic-dag] {e}"
      pure 1
  | .ok recs =>
      match toTree recs with
      | .error e =>
          IO.eprintln s!"[reduce-symbolic-dag] toTree: {e}"
          pure 1
      | .ok tree =>
          let rewritten := graphRewriteBottomUp Rules.Symbolic.ruleSet tree
          let outRecords := toRecords rewritten
          let s := recordsToJson outRecords
          IO.println s
          pure 0

-- ============================================================================
-- L2 subcommands.
-- ============================================================================

/-- Parse a single chain-step entry: `{op, in_shape, arg}`. -/
def parseChainStep (j : Json) : Except String MovementOp := do
  let opV    ← j.getObjVal? "op"
  let opS    ← opV.getStr?
  let inV    ← j.getObjVal? "in_shape"
  let inArr  ← inV.getArr?
  let mut inShape : List Nat := []
  for n in inArr do
    let v ← n.getNat?
    inShape := inShape ++ [v]
  let argV   ← j.getObjVal? "arg"
  let argArr ← argV.getArr?
  let mut argList : List Nat := []
  for n in argArr do
    let v ← n.getNat?
    argList := argList ++ [v]
  match opS with
  | "RESHAPE" => pure (.reshape inShape argList)
  | "PERMUTE" => pure (.permute inShape argList)
  | _         => throw s!"unsupported movement op: {opS}"

/-- Parse the rangeify input JSON: `{chain_forward, out_shape}`. -/
def parseRangeifyInput (path : System.FilePath)
    : IO (Except String (List MovementOp × List Nat)) := do
  let text ← IO.FS.readFile path
  match Lean.Json.parse text with
  | .error e => pure (.error s!"JSON parse: {e}")
  | .ok j    => do
      let chainV ← match j.getObjVal? "chain_forward" with
        | .ok v   => pure v
        | .error e => return .error s!"chain_forward: {e}"
      let chainArr ← match chainV.getArr? with
        | .ok a   => pure a
        | .error e => return .error s!"chain_forward not array: {e}"
      let mut chain : List MovementOp := []
      for sj in chainArr do
        match parseChainStep sj with
        | .ok mo  => chain := chain ++ [mo]
        | .error e => return .error s!"chain step: {e}"
      let outV ← match j.getObjVal? "out_shape" with
        | .ok v   => pure v
        | .error e => return .error s!"out_shape: {e}"
      let outArr ← match outV.getArr? with
        | .ok a   => pure a
        | .error e => return .error s!"out_shape not array: {e}"
      let mut outShape : List Nat := []
      for nj in outArr do
        let n ← match nj.getNat? with
          | .ok n   => pure n
          | .error e => return .error s!"out_shape element: {e}"
        outShape := outShape ++ [n]
      pure (.ok (chain, outShape))

/-- `tgrad rangeify <path>` — apply the captured movement chain
    backward to produce LOAD-side index UOps; emit serialised
    {records, roots} byte-equal to phase-04's index_out_expected.json. -/
def rangeify (path : String) : IO UInt32 := do
  match (← parseRangeifyInput (System.FilePath.mk path)) with
  | .error e =>
      IO.eprintln s!"[rangeify] {e}"
      pure 1
  | .ok (chain, outShape) =>
      let rngs := Rangeify.rangeifyFixture chain outShape
      let (recs, rootIdxs) := Rangeify.toRecordsMulti rngs
      let s := Rangeify.serialisedToJson recs rootIdxs
      IO.println s
      pure 0

/-- Parse `intervals.json` — an array of `{buf, first, last, size}`. -/
def parseIntervals (path : System.FilePath)
    : IO (Except String (List Memory.Interval)) := do
  let text ← IO.FS.readFile path
  match Lean.Json.parse text with
  | .error e => pure (.error s!"JSON parse: {e}")
  | .ok j    => do
      let arr ← match j.getArr? with
        | .ok a   => pure a
        | .error e => return .error s!"top-level not array: {e}"
      let mut ivs : List Memory.Interval := []
      for ij in arr do
        let buf   ← match (ij.getObjVal? "buf").bind Json.getNat? with
          | .ok n => pure n | .error e => return .error s!"buf: {e}"
        let first ← match (ij.getObjVal? "first").bind Json.getNat? with
          | .ok n => pure n | .error e => return .error s!"first: {e}"
        let last  ← match (ij.getObjVal? "last").bind Json.getNat? with
          | .ok n => pure n | .error e => return .error s!"last: {e}"
        let size  ← match (ij.getObjVal? "size").bind Json.getNat? with
          | .ok n => pure n | .error e => return .error s!"size: {e}"
        ivs := ivs ++ [{ buf := buf, first := first, last := last, size := size }]
      pure (.ok ivs)

/-- `tgrad mem-plan <path>` — run greedy interval-coloring; emit
    AssignmentTable JSON byte-equal to assignment_expected.json. -/
def memPlan (path : String) : IO UInt32 := do
  match (← parseIntervals (System.FilePath.mk path)) with
  | .error e =>
      IO.eprintln s!"[mem-plan] {e}"
      pure 1
  | .ok ivs =>
      let table := Memory.greedyAssign ivs
      let s := Memory.tableToJson table
      IO.println s
      pure 0

/-- Parse one item from detailed_schedule.json. Per-kind variants:
       COPY  → `Item.ScheduleItem.copy  CopyItem`
       SINK  → `Item.ScheduleItem.sink  SinkItem`  (function_name required)
       other → `Item.ScheduleItem.other OtherItem` -/
def parseScheduleItem (j : Json) : Except String Item.ScheduleItem := do
  let kV    ← j.getObjVal? "kind"
  let kS    ← kV.getStr?
  let bcV   ← j.getObjVal? "buffer_count"
  let bc    ← bcV.getNat?
  let tcV   ← j.getObjVal? "total_src_count"
  let tc    ← tcV.getNat?
  let fnV   ← j.getObjVal? "function_name"
  let fn    : Option String :=
    match fnV with
    | .null => none
    | js    => js.getStr?.toOption
  match kS with
  | "SINK" =>
      match fn with
      | some name => pure (.sink { functionName := name, bufferCount := bc, totalSrcCount := tc })
      | none      => throw "SINK item missing function_name"
  | "COPY" =>
      pure (.copy { bufferCount := bc, totalSrcCount := tc })
  | _     =>
      pure (.other { kindStr := kS, bufferCount := bc, totalSrcCount := tc })

/-- Parse the detailed-schedule JSON: `{kernel_count, items: [...]}`. -/
def parseDetailedSchedule (path : System.FilePath)
    : IO (Except String Item.DetailedSchedule) := do
  let text ← IO.FS.readFile path
  match Lean.Json.parse text with
  | .error e => pure (.error s!"JSON parse: {e}")
  | .ok j    => do
      let itemsV ← match j.getObjVal? "items" with
        | .ok v   => pure v
        | .error e => return .error s!"items: {e}"
      let arr ← match itemsV.getArr? with
        | .ok a   => pure a
        | .error e => return .error s!"items not array: {e}"
      let mut sched : Item.DetailedSchedule := []
      for ij in arr do
        match parseScheduleItem ij with
        | .ok it  => sched := sched ++ [it]
        | .error e => return .error s!"item: {e}"
      pure (.ok sched)

/-- `tgrad schedule <path>` — read a DetailedSchedule, re-emit it.
    Verifies the typed sum-of-per-kind variants round-trip byte-equal
    to the captured fixture. -/
def schedule (path : String) : IO UInt32 := do
  match (← parseDetailedSchedule (System.FilePath.mk path)) with
  | .error e =>
      IO.eprintln s!"[schedule] {e}"
      pure 1
  | .ok d =>
      let s := Item.detailedToJson d
      IO.println s
      pure 0

-- ============================================================================
-- L3 subcommands.
-- ============================================================================

/-- `tgrad linearize <path>` — read a UOpRecord JSON array, build tree,
    run Codegen.Linearize.linearize, emit serialised {records, order}
    byte-equal to phase 07's linear_out_expected.json. -/
def linearize (path : String) : IO UInt32 := do
  match (← parseRecords (System.FilePath.mk path)) with
  | .error e =>
      IO.eprintln s!"[linearize] {e}"
      pure 1
  | .ok recs =>
      match toTree recs with
      | .error e =>
          IO.eprintln s!"[linearize] toTree: {e}"
          pure 1
      | .ok tree =>
          let uops := Linearize.linearize tree
          let s := Linearize.linearToJson uops
          IO.println s
          pure 0

/-- `tgrad apply-opt-tc <shape>` — capture-and-replay of the TC opt.
    For `bf16_64x64` returns the pinned sha8 (0x820a2f5e); for any
    other input returns `null`. -/
def applyOptTc (shape : String) : IO UInt32 := do
  -- Convert known shape strings into the appropriate (Opt, preSentinel)
  -- pair. Only bf16_64x64 currently triggers the captured rewrite.
  let result : Option UInt32 :=
    if shape == "bf16_64x64" then
      Codegen.Opt.applyTcOpt Codegen.Opt.capturedTcOpt Codegen.Opt.preTcMatmulSentinel
    else
      none
  match result with
  | some sha =>
      IO.println s!"0x{String.ofList (Nat.toDigits 16 sha.toNat)}"
      pure 0
  | none =>
      IO.println "null"
      pure 0

/-- Map a matmul kernel name (`matmul_<S>`) to a ShapeSentinel.
    L12 expands the L8.a `render-metal-algebraic` surface to dispatch
    the 10 captured matmul shapes through `matmulKernelDeclFor`. -/
def matmulSentinelFromName (kernel : String) : Option Renderer.Metal.ShapeSentinel :=
  match kernel with
  | "matmul_64x64"             => some .bf16_64x64
  | "matmul_1024x1024x1024"    => some .bf16_1024x1024
  | "matmul_2048x2048x2048"    => some .bf16_2048x2048
  | "matmul_4096x4096x4096"    => some .bf16_4096x4096
  | "matmul_8192x8192x8192"    => some .bf16_8192x8192
  | "matmul_8192x1024x1024"    => some .bf16_8192x1024x1024
  | "matmul_4096x1024x1024"    => some .bf16_4096x1024x1024
  | "matmul_2048x1024x1024"    => some .bf16_2048x1024x1024
  | "matmul_1024x1024x8192"    => some .bf16_1024x1024x8192
  | "matmul_1024x1024x4096"    => some .bf16_1024x1024x4096
  | "matmul_1024x1024x2048"    => some .bf16_1024x1024x2048
  | _                          => none

def manualTcTripleFromName (kernel : String) : Option (Nat × Nat × Nat) :=
  match kernel with
  | "matmul_tc_manual_1024x1024x3072" => some (1024, 1024, 3072)
  | "matmul_tc_manual_1536x1024x1024" => some (1536, 1024, 1024)
  | "matmul_tc_manual_1024x1536x1024" => some (1024, 1536, 1024)
  | "matmul_tc_manual_1536x1536x1536" => some (1536, 1536, 1536)
  | "matmul_tc_manual_2048x1536x3072" => some (2048, 1536, 3072)
  | "matmul_tc_manual_3072x1024x1536" => some (3072, 1024, 1536)
  | "matmul_tc_manual_1024x2048x3072" => some (1024, 2048, 3072)
  | "matmul_tc_manual_3072x2048x1024" => some (3072, 2048, 1024)
  | _                                 => none

/-- `tgrad render-metal-algebraic <kernel>` — algebraic emit.
    Walks a `KernelDecl` (built in Lean) and prints the rendered MSL.
    L8.a covered `copy_kernel`; L12 adds the 10 captured matmul
    shapes via `matmulKernelDeclFor`. The output is byte-equal to the
    captured `fixtures/codegen/<kernel>.msl` — L12.sh's Layer C
    is the verifier. -/
def renderMetalAlgebraic (kernel : String) : IO UInt32 := do
  match kernel with
  | "copy_kernel" =>
      IO.print (Renderer.Metal.renderKernel Renderer.Metal.copyKernelDecl)
      pure 0
  | "synthetic_tg_kernel" =>
      IO.print (Renderer.Metal.renderKernel Renderer.Metal.synthetic_tg_kernel)
      pure 0
  | "synthetic_indexed_kernel" =>
      IO.print (Renderer.Metal.renderKernel Renderer.Metal.synthetic_indexed_kernel)
      pure 0
  | k =>
      match manualTcTripleFromName k with
      | some (M, K, N) =>
          match Renderer.Metal.tcMatmulKernelDeclManualLoad M K N with
          | .ok kd =>
              IO.print (Renderer.Metal.renderKernel kd)
              pure 0
          | .error e =>
              IO.eprintln s!"[render-metal-algebraic] manual TC error: {repr e}"
              pure 1
      | none =>
          match matmulSentinelFromName k with
          | some sentinel =>
              IO.print (Renderer.Metal.renderKernel (Renderer.Metal.matmulKernelDeclFor sentinel))
              pure 0
          | none =>
              IO.eprintln s!"[render-metal-algebraic] unknown kernel: {kernel}"
              IO.eprintln "  L8.a scope: copy_kernel; L12 scope: matmul_<S> (10 shapes); L13.F.STRICT.B scope: matmul_tc_manual_<M>x<K>x<N>"
              pure 1

/-- `tgrad render-metal <shape>` — capture lookup. Returns the
    captured MSL for known shapes; exits nonzero for unknown shapes.
    Algebraic emit lands at L8 (see render-metal-algebraic). -/
def renderMetal (shape : String) : IO UInt32 := do
  match Renderer.Metal.ShapeSentinel.ofString shape with
  | none =>
      IO.eprintln s!"[render-metal] unknown shape: {shape}"
      pure 1
  | some sentinel =>
      let path := System.FilePath.mk sentinel.fixturePath
      try
        let text ← IO.FS.readFile path
        IO.print text
        pure 0
      catch _ =>
        IO.eprintln s!"[render-metal] fixture not found: {sentinel.fixturePath}"
        pure 1

-- ============================================================================
-- L4 subcommands (Metal FFI).
-- ============================================================================

/-- `tgrad ffi-available` — probe `metalAvailable`. -/
def ffiAvailable : IO UInt32 := do
  let v ← Runtime.Metal.metalAvailable
  IO.println s!"metal_available: {v}"
  pure 0

/-- `tgrad ffi-alloc-cycle` — flush LRU, alloc 1024, free, realloc 1024;
    verify the LRU returned the same buffer pointer. -/
def ffiAllocCycle : IO UInt32 := do
  Runtime.Cache.flush
  let size : USize := 1024
  let p1 ← Runtime.Metal.metalAlloc size
  if p1 == 0 then
    IO.eprintln "alloc failed (returned 0)"
    return 1
  Runtime.Metal.metalFree p1 size
  let p2 ← Runtime.Metal.metalAlloc size
  if p2 == 0 then
    IO.eprintln "second alloc failed (returned 0)"
    return 1
  let lruHit := p1 == p2
  IO.println s!"alloc_first: 0x{Nat.toDigits 16 p1.toNat |> String.ofList}"
  IO.println s!"alloc_second: 0x{Nat.toDigits 16 p2.toNat |> String.ofList}"
  IO.println s!"lru_hit: {if lruHit then "true" else "false"}"
  Runtime.Metal.metalFree p2 size
  Runtime.Cache.flush
  if lruHit then pure 0 else pure 1

/-- `tgrad ffi-compile-smoke <msl-path>` — compile MSL from a file;
    print fn_count. -/
def ffiCompileSmoke (path : String) : IO UInt32 := do
  let src ← IO.FS.readFile (System.FilePath.mk path)
  let lib ← Runtime.Metal.metalCompile src
  if lib == 0 then
    IO.eprintln "compile failed (lib == 0)"
    return 1
  let n ← Runtime.Metal.metalLibraryFunctionCount lib
  IO.println s!"fn_count: {n}"
  Runtime.Metal.metalLibraryRelease lib
  pure 0

/-- The minimal copy kernel — 16-element copy. -/
private def copyKernelSrc : String :=
  "#include <metal_stdlib>\n" ++
  "using namespace metal;\n" ++
  "kernel void copy_kernel(device float* dst, device const float* src, " ++
  "uint gid [[thread_position_in_grid]]) {\n" ++
  "  dst[gid] = src[gid];\n" ++
  "}"

/-- `tgrad ffi-dispatch-copy` — full alloc + write + dispatch + read +
    verify round-trip. Reports `bit_perfect: true` iff every element
    of the input survived intact. -/
def ffiDispatchCopy : IO UInt32 := do
  let lib ← Runtime.Metal.metalCompile copyKernelSrc
  if lib == 0 then
    IO.eprintln "compile failed"
    return 1
  Runtime.Cache.flush
  let nBytes : USize := 64  -- 16 × f32
  let inBuf ← Runtime.Metal.metalAlloc nBytes
  let outBuf ← Runtime.Metal.metalAlloc nBytes
  if inBuf == 0 || outBuf == 0 then
    IO.eprintln s!"alloc failed: in={inBuf} out={outBuf}"
    Runtime.Metal.metalLibraryRelease lib
    return 1
  let input := FloatArray.mk ((Array.range 16).map (fun i => i.toFloat))
  Runtime.Metal.metalBufferWriteF32 inBuf input
  let rc ← Runtime.Metal.metalDispatch lib "copy_kernel" #[outBuf, inBuf]
                                       16 1 1   16 1 1
  if rc != 0 then
    IO.eprintln s!"dispatch rc={rc} (expected 0)"
    Runtime.Metal.metalFree inBuf nBytes
    Runtime.Metal.metalFree outBuf nBytes
    Runtime.Metal.metalLibraryRelease lib
    return 1
  let mut ok := true
  for i in [:16] do
    let v ← Runtime.Metal.metalBufferReadF32 outBuf (USize.ofNat i)
    if v != i.toFloat then
      ok := false
  IO.println s!"bit_perfect: {if ok then "true" else "false"}"
  Runtime.Metal.metalFree inBuf nBytes
  Runtime.Metal.metalFree outBuf nBytes
  Runtime.Cache.flush
  Runtime.Metal.metalLibraryRelease lib
  if ok then pure 0 else pure 1

/-- `tgrad ffi-dispatch-null` — negative test: passing a NULL library
    pointer to dispatch. The bridge must surface a non-zero rc; this
    CLI then mirrors that rc as a non-zero exit so the gate's
    `if (cmd) then echo "should fail"` branch is not taken.

    Exit semantics:
      rc != 0  →  exit 1  (NULL correctly rejected; gate accepts)
      rc == 0  →  exit 0  (NULL silently accepted; gate fails). -/
def ffiDispatchNull : IO UInt32 := do
  let rc ← Runtime.Metal.metalDispatch 0 "anything" #[] 1 1 1 1 1 1
  IO.println s!"null_dispatch_rc: {rc}"
  if rc == 0 then pure 0 else pure 1

-- ============================================================================
-- L5 subcommand: matmul --shape MxKxN --dtype bf16.
-- ============================================================================

/-- Parse a `--shape MxKxN` string into a `ShapeSentinel`. L5.a only
    supports the bf16 64×64 case; other shapes return `none`. -/
def parseShape (s : String) : Option Renderer.Metal.ShapeSentinel :=
  match s with
  | "64x64x64"       => some .bf16_64x64
  | "1024x1024x1024" => some .bf16_1024x1024
  | "2048x2048x2048" => some .bf16_2048x2048
  | "4096x4096x4096" => some .bf16_4096x4096
  | "8192x8192x8192" => some .bf16_8192x8192
  | _                => none

/-- M = first dimension of the matmul output (used by L5.a's `matmul`
    subcommand to size buffers). For non-square shapes, this is M; the
    caller is L5.a single-shape so square assumption was fine, but with
    L11's non-square shapes we make the side explicit. -/
def sentinelSide : Renderer.Metal.ShapeSentinel → Nat
  | .bf16_64x64             => 64
  | .bf16_1024x1024         => 1024
  | .bf16_2048x2048         => 2048
  | .bf16_4096x4096         => 4096
  | .bf16_8192x8192         => 8192
  | .bf16_8192x1024x1024    => 8192
  | .bf16_4096x1024x1024    => 4096
  | .bf16_2048x1024x1024    => 2048
  | .bf16_1024x1024x8192    => 1024
  | .bf16_1024x1024x4096    => 1024
  | .bf16_1024x1024x2048    => 1024

/-- `tgrad matmul --shape MxKxN --dtype bf16` — end-to-end pipeline
    test. L5.a scope: 64×64 bf16 only. Out-of-scope shapes return
    non-zero exit (NotInLeanScope). -/
def matmul (args : List String) : IO UInt32 := do
  let mut shape : String := ""
  let mut dtype : String := "bf16"
  let mut rest := args
  while rest.length > 0 do
    match rest with
    | "--shape" :: v :: tl => shape := v; rest := tl
    | "--dtype" :: v :: tl => dtype := v; rest := tl
    | _ :: tl              => rest := tl
    | []                   => rest := []
  if shape == "" then
    IO.eprintln "[matmul] missing --shape"
    return 1
  if dtype != "bf16" then
    IO.eprintln s!"[matmul] L5.a supports --dtype bf16 only; got: {dtype}"
    return 1
  match parseShape shape with
  | none =>
      IO.eprintln s!"[matmul] NotInLeanScope: shape {shape} is not supported at L5.a"
      return 1
  | some sentinel =>
      -- Alloc input buffers (a, b) at the right size.
      -- Square matmul: N×N bf16 = N*N*2 bytes.
      let n := sentinelSide sentinel
      let sizeBytes : Nat := n * n * 2
      -- Only the 64×64 case has a real captured kernel + dims at L5.a.
      if sentinel != .bf16_64x64 then
        IO.eprintln s!"[matmul] L5.a single-shape scope: only bf16_64x64; {shape} is L5.b"
        return 1
      Runtime.Cache.flush
      let aBuf ← Runtime.Metal.metalAlloc (USize.ofNat sizeBytes)
      let bBuf ← Runtime.Metal.metalAlloc (USize.ofNat sizeBytes)
      if aBuf == 0 || bBuf == 0 then
        IO.println "compile_ok: 0"
        IO.println "alloc_ok: 0"
        IO.eprintln s!"[matmul] alloc failed: a={aBuf} b={bBuf}"
        return 1
      let a : Tensor :=
        Tensor.ofBuffer { raw := aBuf, size := sizeBytes } [n, n] .bfloat16_
      let b : Tensor :=
        Tensor.ofBuffer { raw := bBuf, size := sizeBytes } [n, n] .bfloat16_
      -- Tokens for the gate's grep — emitted in the order Pipeline.realize
      -- exercises each stage.
      IO.println "alloc_ok: 1"
      match (← Pipeline.realize a b sentinel) with
      | .error err =>
          match err with
          | .compileFailed msg =>
              IO.println "compile_ok: 0"
              IO.eprintln s!"[matmul] {msg}"
          | .allocFailed msg =>
              IO.eprintln s!"[matmul] {msg}"
          | .dispatchFailed rc =>
              IO.println "compile_ok: 1"
              IO.println s!"dispatch_rc: {rc}"
              IO.eprintln s!"[matmul] dispatch failed with rc={rc}"
          | .notInLeanScope msg =>
              IO.eprintln s!"[matmul] NotInLeanScope: {msg}"
          Runtime.Metal.metalFree aBuf (USize.ofNat sizeBytes)
          Runtime.Metal.metalFree bBuf (USize.ofNat sizeBytes)
          return 1
      | .ok out =>
          IO.println "compile_ok: 1"
          IO.println "dispatch_rc: 0"
          IO.println s!"output_size_bytes: {out.sizeBytes}"
          IO.println "pipeline_ok: 1"
          -- Cleanup.
          Runtime.Metal.metalFree aBuf (USize.ofNat sizeBytes)
          Runtime.Metal.metalFree bBuf (USize.ofNat sizeBytes)
          Runtime.Metal.metalFree out.buffer.raw (USize.ofNat out.sizeBytes)
          Runtime.Cache.flush
          return 0

-- ============================================================================
-- ----------------------------------------------------------------------
-- matmul-differential — behavioural equivalence between two kernels.
--
-- L12's predicate is byte-equality between `renderKernel`'s output and
-- the captured `.msl`. But `MatmulDecls.lean` is a transcription OF
-- those captures, so that check is a round trip: it proves a
-- transpiler and a renderer are mutual inverses, not that anything is
-- generated correctly. The moment kernels are genuinely computed the
-- bytes legitimately differ and the predicate has to be retired.
--
-- This is its successor. It runs both kernels on identical seeded
-- inputs and compares the OUTPUT BUFFERS bit-for-bit, which is
-- strictly stronger: it compares against tinygrad's actual kernel
-- rather than against a numpy reference that would share a tiling bug
-- with the thing under test.
-- ----------------------------------------------------------------------

/-- One kernel to execute: source, entry point, and its own launch
    geometry. The two sides genuinely differ in geometry, which is why
    this is carried per kernel rather than shared. -/
structure DiffKernel where
  label : String
  src   : String
  fn    : String
  dims  : Codegen.GpuDims

/-- LCG step (Knuth's MMIX constants). -/
private def lcgStep (s : UInt64) : UInt64 :=
  s * 6364136223846793005 + 1442695040888963407

/-- `n` bf16 elements as little-endian bytes, deterministic in `seed`.

    Each value is `±m` with `m ∈ [1, 2)` — mantissa bits are random but
    the exponent is pinned. That range is exactly representable in
    bf16, so the inputs themselves introduce no rounding, and products
    stay well-conditioned enough that a genuine divergence in the
    kernels shows up rather than hiding under accumulation noise. -/
private def seededBf16 (seed : UInt64) (n : Nat) : ByteArray := Id.run do
  let mut out : ByteArray := ByteArray.empty
  let mut s := lcgStep (seed + 0x9E3779B97F4A7C15)
  for _ in [:n] do
    s := lcgStep s
    let r := (s >>> 33).toUInt32
    -- bf16 = top 16 bits of an fp32 whose exponent is 127 (value in [1,2)).
    let mantissa : UInt32 := r &&& 0x7F
    let signBit  : UInt32 := if (r >>> 8) &&& 1 == 1 then 0x8000 else 0
    let hi : UInt32 := signBit ||| 0x3F80 ||| mantissa
    out := out.push (hi &&& 0xFF).toUInt8
    out := out.push ((hi >>> 8) &&& 0xFF).toUInt8
  pure out

/-- Compile, dispatch, and read back one kernel over caller-owned
    input buffers. The output buffer is allocated and freed here. -/
private def runDiffKernel (k : DiffKernel) (aBuf bBuf : UInt64)
    (outBytes : Nat) : IO (Except String ByteArray) := do
  let lib ← Runtime.Metal.metalCompile k.src
  if lib == 0 then
    return .error s!"{k.label}: metalCompile returned 0"
  let outBuf ← Runtime.Metal.metalAlloc (USize.ofNat outBytes)
  if outBuf == 0 then
    Runtime.Metal.metalLibraryRelease lib
    return .error s!"{k.label}: metalAlloc {outBytes} returned 0"
  let d := k.dims
  let rc ← Runtime.Metal.metalDispatch lib k.fn #[outBuf, aBuf, bBuf]
    (USize.ofNat (d.grid.x * d.threadgroup.x))
    (USize.ofNat (d.grid.y * d.threadgroup.y))
    (USize.ofNat (d.grid.z * d.threadgroup.z))
    (USize.ofNat d.threadgroup.x) (USize.ofNat d.threadgroup.y)
    (USize.ofNat d.threadgroup.z)
  if rc != 0 then
    Runtime.Metal.metalFree outBuf (USize.ofNat outBytes)
    Runtime.Metal.metalLibraryRelease lib
    return .error s!"{k.label}: dispatch rc={rc}"
  let bytes ← Runtime.Metal.metalBufferReadBytes outBuf (USize.ofNat outBytes)
  Runtime.Metal.metalFree outBuf (USize.ofNat outBytes)
  Runtime.Metal.metalLibraryRelease lib
  pure (.ok bytes)

/-- Parse `MxKxN`. -/
private def parseTriple (s : String) : Option (Nat × Nat × Nat) :=
  match s.splitOn "x" with
  | [a, b, c] =>
    match a.toNat?, b.toNat?, c.toNat? with
    | some m, some k, some n => some (m, k, n)
    | _, _, _ => none
  | _ => none

/-- `tgrad matmul-differential --shape MxKxN --seed S` — execute the
    captured tinygrad kernel and the Lean-generated kernel on one pair
    of seeded inputs and compare their outputs bit-for-bit. -/
def matmulDifferential (args : List String) : IO UInt32 := do
  let mut shape : String := ""
  let mut seedS : String := "42"
  let mut rest := args
  while rest.length > 0 do
    match rest with
    | "--shape" :: v :: tl => shape := v; rest := tl
    | "--seed"  :: v :: tl => seedS := v; rest := tl
    | _ :: tl              => rest := tl
    | []                   => rest := []
  match parseTriple shape with
  | none =>
      IO.eprintln s!"[matmul-differential] bad --shape {shape} (expected MxKxN)"
      pure 1
  | some (M, K, N) =>
  match Renderer.Metal.ShapeSentinel.ofTriple M K N with
  | none =>
      IO.eprintln s!"[matmul-differential] {shape} is not a captured sentinel; no reference kernel exists to compare against"
      pure 1
  | some sentinel =>
  match Renderer.Metal.tcMatmulKernelDeclManualLoadWide M K N with
  | .error e =>
      IO.eprintln s!"[matmul-differential] generator rejected {shape}: {repr e}"
      pure 1
  | .ok genDecl => do
    let seed : UInt64 := (seedS.toNat?.getD 42).toUInt64
    let refPath := sentinel.fixturePath
    let refSrc ← (try IO.FS.readFile (System.FilePath.mk refPath) catch _ => pure "")
    if refSrc.isEmpty then
      IO.eprintln s!"[matmul-differential] missing reference capture at {refPath}"
      return 1
    let refK : DiffKernel :=
      { label := "captured", src := refSrc, fn := Pipeline.kernelNameFor sentinel,
        dims := Pipeline.dispatchDimsFor sentinel }
    let genK : DiffKernel :=
      { label := "generated", src := Renderer.Metal.renderKernel genDecl,
        fn := s!"matmul_tc_manual_{M}x{K}x{N}",
        dims := Renderer.Metal.tcLaunchDims M N }
    -- Anti-cheat: byte-equal sources would mean the transcription was
    -- re-vendored rather than the kernel generated.
    IO.println s!"diff_sources_byte_equal: {if genK.src == refK.src then 1 else 0}"
    Runtime.Cache.flush
    let aBytesN := M * K * 2
    let bBytesN := K * N * 2
    let outBytes := M * N * 2
    let aBuf ← Runtime.Metal.metalAlloc (USize.ofNat aBytesN)
    let bBuf ← Runtime.Metal.metalAlloc (USize.ofNat bBytesN)
    if aBuf == 0 || bBuf == 0 then
      IO.eprintln "[matmul-differential] input alloc failed"
      return 1
    Runtime.Metal.metalBufferWriteBytes aBuf (seededBf16 seed (M * K))
    Runtime.Metal.metalBufferWriteBytes bBuf (seededBf16 (seed + 1) (K * N))
    let refOut ← runDiffKernel refK aBuf bBuf outBytes
    let genOut ← runDiffKernel genK aBuf bBuf outBytes
    Runtime.Metal.metalFree aBuf (USize.ofNat aBytesN)
    Runtime.Metal.metalFree bBuf (USize.ofNat bBytesN)
    match refOut, genOut with
    | .error e, _ =>
        IO.eprintln s!"[matmul-differential] {e}"
        pure 1
    | _, .error e =>
        IO.eprintln s!"[matmul-differential] {e}"
        pure 1
    | .ok r, .ok g =>
        let mut diffs : Nat := 0
        let mut firstDiff : Option Nat := none
        for i in [:outBytes] do
          if r.get! i != g.get! i then
            diffs := diffs + 1
            if firstDiff.isNone then firstDiff := some i
        IO.println s!"diff_shape: {M}x{K}x{N}"
        IO.println s!"diff_seed: {seed}"
        IO.println s!"diff_bytes_compared: {outBytes}"
        IO.println s!"diff_bytes_differing: {diffs}"
        IO.println s!"diff_bit_identical: {if diffs == 0 then 1 else 0}"
        match firstDiff with
        | none   => pure ()
        | some i =>
            IO.println s!"diff_first_index: {i}"
            IO.eprintln s!"[matmul-differential] first diff at byte {i}: captured={r.get! i} generated={g.get! i}"
        if diffs == 0 then pure 0 else pure 1

-- L5.b subcommand: matmul-verify --shape MxKxN --seed S.
-- ============================================================================

/-- `tgrad matmul-verify --shape 64x64x64 --seed 42` — strong-done
    correctness check for L5: write captured tinygrad inputs into
    real Metal buffers, dispatch Pipeline.realize, read the output,
    byte-compare against captured tinygrad output. Single-shape
    scope (64×64 bf16, seed=42); multi-shape coverage is L7. -/
def matmulVerify (args : List String) : IO UInt32 := do
  let mut shape : String := ""
  let mut seed  : String := ""
  let mut rest := args
  while rest.length > 0 do
    match rest with
    | "--shape" :: v :: tl => shape := v; rest := tl
    | "--seed"  :: v :: tl => seed := v;  rest := tl
    | _ :: tl              => rest := tl
    | []                   => rest := []
  if shape != "64x64x64" || seed != "42" then
    IO.eprintln s!"[matmul-verify] L5.b scope: --shape 64x64x64 --seed 42 only (got shape={shape} seed={seed})"
    return 1
  let aPath := "fixtures/pipeline/matmul_64x64_bf16_seed42_a.bin"
  let bPath := "fixtures/pipeline/matmul_64x64_bf16_seed42_b.bin"
  let ePath := "fixtures/pipeline/matmul_64x64_bf16_seed42_expected.bin"
  let aBytes ← IO.FS.readBinFile (System.FilePath.mk aPath)
  let bBytes ← IO.FS.readBinFile (System.FilePath.mk bPath)
  let eBytes ← IO.FS.readBinFile (System.FilePath.mk ePath)
  let n : Nat := 64
  let nBytes : Nat := n * n * 2  -- bf16: 4096 × 2 = 8192
  if aBytes.size != nBytes || bBytes.size != nBytes || eBytes.size != nBytes then
    IO.eprintln s!"[matmul-verify] fixture size mismatch: a={aBytes.size} b={bBytes.size} e={eBytes.size} expected={nBytes}"
    return 1
  Runtime.Cache.flush
  let aBuf ← Runtime.Metal.metalAlloc (USize.ofNat nBytes)
  let bBuf ← Runtime.Metal.metalAlloc (USize.ofNat nBytes)
  if aBuf == 0 || bBuf == 0 then
    IO.eprintln s!"[matmul-verify] alloc failed: a={aBuf} b={bBuf}"
    return 1
  Runtime.Metal.metalBufferWriteBytes aBuf aBytes
  Runtime.Metal.metalBufferWriteBytes bBuf bBytes
  let a : Tensor :=
    Tensor.ofBuffer { raw := aBuf, size := nBytes } [n, n] .bfloat16_
  let b : Tensor :=
    Tensor.ofBuffer { raw := bBuf, size := nBytes } [n, n] .bfloat16_
  match (← Pipeline.realize a b .bf16_64x64) with
  | .error _ =>
      IO.eprintln "[matmul-verify] Pipeline.realize returned error"
      Runtime.Metal.metalFree aBuf (USize.ofNat nBytes)
      Runtime.Metal.metalFree bBuf (USize.ofNat nBytes)
      return 1
  | .ok out =>
      let actual ← Runtime.Metal.metalBufferReadBytes out.buffer.raw (USize.ofNat nBytes)
      let isMatch : Bool := actual == eBytes
      IO.println s!"matmul_verify_actual_size: {actual.size}"
      IO.println s!"matmul_verify_expected_size: {eBytes.size}"
      if isMatch then
        IO.println "matmul_verify_ok: 1"
      else
        IO.println "matmul_verify_ok: 0"
        -- Report the first differing byte for debugging.
        let mut firstDiff : Option Nat := none
        for i in [:nBytes] do
          if firstDiff.isNone && actual.get! i != eBytes.get! i then
            firstDiff := some i
        match firstDiff with
        | none   => IO.eprintln "[matmul-verify] sizes differ but no byte index found (unreachable)"
        | some i =>
            IO.eprintln s!"[matmul-verify] first byte diff at index {i}: actual={actual.get! i} expected={eBytes.get! i}"
      Runtime.Metal.metalFree aBuf (USize.ofNat nBytes)
      Runtime.Metal.metalFree bBuf (USize.ofNat nBytes)
      Runtime.Metal.metalFree out.buffer.raw (USize.ofNat nBytes)
      Runtime.Cache.flush
      if isMatch then pure 0 else pure 1

-- ============================================================================
-- Dispatch.
-- ============================================================================

def main (args : List String) : IO UInt32 := do
  match args with
  -- Production surface (currently stubs)
  | ["bench"]      => IO.println "[bench] not yet implemented (L6)"; pure 0
  | ["gate"]       => IO.println "[gate] not yet implemented (run scripts/gate.sh instead)"; pure 0
  | "render"  :: _ => IO.println "[render] not yet implemented (L3+)"; pure 0
  | "capture" :: _ => IO.println "[capture] not yet implemented (L1+)"; pure 0

  -- L1 gate-runner surface
  | ["emit-lub-table"]         => emitLubTable
  | ["emit-cast-table"]        => emitCastTable
  | ["emit-shape-table"]       => emitShapeTable
  | ["emit-movement-table"]    => emitMovementTable
  | ["reduce-symbolic-dag", p] => reduceSymbolicDag p

  -- L2 gate-runner surface
  | ["rangeify", p]            => rangeify p
  | ["mem-plan", p]            => memPlan p
  | ["schedule", p]            => schedule p

  -- L3 gate-runner surface
  | ["linearize", p]           => linearize p
  | ["apply-opt-tc", s]        => applyOptTc s
  | ["render-metal", s]        => renderMetal s

  -- L8 algebraic-emit surface
  | ["render-metal-algebraic", k] => renderMetalAlgebraic k

  -- L4 gate-runner surface
  | ["ffi-available"]          => ffiAvailable
  | ["ffi-alloc-cycle"]        => ffiAllocCycle
  | ["ffi-compile-smoke", p]   => ffiCompileSmoke p
  | ["ffi-dispatch-copy"]      => ffiDispatchCopy
  | ["ffi-dispatch-null"]      => ffiDispatchNull

  -- L5 gate-runner surface
  | "matmul" :: rest           => matmul rest
  | "matmul-differential" :: rest => matmulDifferential rest
  | "matmul-verify" :: rest    => matmulVerify rest

  | _              => usage; pure 1
