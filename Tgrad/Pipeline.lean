import Tgrad.Tensor
import Tgrad.Renderer.Metal
import Tgrad.Renderer.MatmulScalar
import Tgrad.Runtime.MetalProgram
import Tgrad.Codegen.GpuDims
import Tgrad.Codegen.Opt.Heuristic
import Tgrad.Schedule.Rangeify

/-! # Tgrad.Pipeline — end-to-end realize composition

  L5 is the discovery point of phase composition. Per §G5 and §7's
  L5 fall-back, we scope to single-shape (bf16 64×64) at L5.a; L5.b
  promotes to multi-shape + byte-match-vs-tinygrad when a bf16
  host-write extern + captured (input, output) fixture exist.

  `Pipeline.realize` composes the lifted stages into a single Lean
  IO function:

      Tensor a, Tensor b  →  Tensor c
              ↓ Renderer.Metal.ShapeSentinel lookup
              ↓ load captured MSL
              ↓ Runtime.MetalProgram.metalCompile
              ↓ Runtime.MetalAllocator.metalAlloc (output buffer)
              ↓ Runtime.MetalProgram.metalDispatch
              ↓ output Tensor wrapping the result buffer

  At L5.a the input buffers are taken as-pre-allocated (the caller
  is responsible for filling them). L6 adds the Python entry that
  marshals numpy arrays into BufferHandles via @[export] entries.
-/
namespace Tgrad

/-- Errors `Pipeline.realize` surfaces. -/
inductive PipelineError where
  | notInLeanScope    (msg : String)
  | compileFailed     (msg : String)
  | allocFailed       (msg : String)
  | dispatchFailed    (rc  : UInt32)
  deriving Repr

namespace Pipeline

/-! ## L14.B.2.c: rangeify trace + view-aware matmul

  `Pipeline.realize` (the matmul-verify entry point used by Main.lean's
  CLI subcommands) consumes a SINK UOp tree built from the input
  Tensors' uop graphs. For BUFFER-only inputs the SINK is a pass-through
  through `Schedule_Rangeify_rangeify`; for view-chain inputs we route
  through a parametric scalar matmul whose A/B load-index UOps are
  derived from the input uop's movement chain.

  The trace records movement-node counts (in vs out of rangeify) so
  the L14.B.2.c smoke can prove rangeify saw a non-empty chain.
-/

/-- L14.B.2.c: one row per `Pipeline.realize` call when
    `TGRAD_RANGEIFY_TRACE=1` is set; emitted as JSONL to
    `/tmp/tgrad_rangeify_trace.jsonl`. -/
structure RangeifyTraceRow where
  input_uop_kind          : String
  output_uop_kind         : String
  movement_count_in       : Nat
  movement_count_out      : Nat
  root_indexed_ops_count  : Nat
  deriving Repr, Inhabited

/-- L14.B.2.c: write a trace row to `/tmp/tgrad_rangeify_trace.jsonl`
    when the `TGRAD_RANGEIFY_TRACE` env var is set. No-op otherwise —
    L11/L13/L13_F sweeps don't set the env var and stay hot-path-free. -/
def RangeifyTrace.maybeEmit (row : RangeifyTraceRow) : IO Unit := do
  let some _ ← IO.getEnv "TGRAD_RANGEIFY_TRACE" | pure ()
  let path := "/tmp/tgrad_rangeify_trace.jsonl"
  let line :=
    s!"\{\"input_uop_kind\":\"{row.input_uop_kind}\","
      ++ s!"\"output_uop_kind\":\"{row.output_uop_kind}\","
      ++ s!"\"movement_count_in\":{row.movement_count_in},"
      ++ s!"\"movement_count_out\":{row.movement_count_out},"
      ++ s!"\"root_indexed_ops_count\":{row.root_indexed_ops_count}" ++ "}"
  let h ← IO.FS.Handle.mk path IO.FS.Mode.append
  h.putStrLn line

/-- L14.B.2.c: synthesize a tiny SINK UOp tree from `a.uop` and
    `b.uop` for the trace. Not used by codegen — the matmul kernels
    consume `Tensor.buffer.raw` directly. The point of this tree is
    that `UOp.countMovementNodes` walked over it reports the right
    count for the smoke. -/
def buildMatmulSink (aUop bUop : UOp) : UOp :=
  .sink (.binop .add aUop bUop .void_)

/-- L14.B.3: derive the LOAD-A index UOp from a Tensor's uop chain.
    Handles the matmul kernel's per-thread access pattern (gidx0 =
    output row, Ridx0 = K loop iterator). Returns the index UOp for
    `data1[idx]` in scalar matmul of effective shape (M, K, N).

    Supported uop chains:
    - `.buffer _ shape _` (row-major): walks shape; for 2-D input
      `[M_orig, K_orig]` returns `gidx0 * K_orig + Ridx0`.
    - `.permute (.buffer _ _ _) [1, 0]` (2-D transpose):
      `Ridx0 * M_orig + gidx0` where `M_orig` is the buffer's first
      dim (which equals the matmul's M_effective when transposed).
    - `.slice (.buffer _ shape _) [{start, stop, step}, _]`
      (M-axis slice): `gidx0 * step * K_orig + start * K_orig + Ridx0`.
    - `.reshape (.buffer _ shape _) _` (numel-preserving reshape):
      row-major `gidx0 * K + Ridx0` using the matmul's effective K
      (the data is contiguous; reshape is a relabel).

    Other movement ops (EXPAND over A; nested chains; non-2-D) panic
    — L14.B.3's manifest doesn't exercise them on the A side. -/
partial def viewIndexUOpForA (uop : UOp) (M K : Nat) : UOp :=
  let rowMajorIdx (k : Nat) : UOp :=
    .binop .add
      (.binop .mul (.var "gidx0" .int32_)
                   (.const .int32_ (.i (Int.ofNat k))) .int32_)
      (.var "Ridx0" .int32_) .int32_
  match uop with
  | .buffer _ _ _ => rowMajorIdx K
  | .reshape (.buffer _ _ _) _ => rowMajorIdx K
  | .permute (.buffer _ _ _) [1, 0] =>
      -- 2-D transpose: Ridx0 * M + gidx0 (M_orig of buffer == M_view of matmul)
      .binop .add
        (.binop .mul (.var "Ridx0" .int32_)
                     (.const .int32_ (.i (Int.ofNat M))) .int32_)
        (.var "gidx0" .int32_) .int32_
  | .slice (.buffer _ origShape _) (slc :: _) =>
      -- M-axis slice with `slc = {start, stop, step}`. Original K
      -- dim is `origShape[1]`. The view reads
      --   a_view[gidx0][Ridx0] = original[gidx0*step + start][Ridx0]
      -- so idx = (gidx0*step + start) * K_orig + Ridx0
      let kOrig := (origShape[1]?).getD K
      let stepConst : UOp := .const .int32_ (.i (Int.ofNat slc.step))
      let startConst : UOp := .const .int32_ (.i (Int.ofNat slc.start))
      let kOrigConst : UOp := .const .int32_ (.i (Int.ofNat kOrig))
      .binop .add
        (.binop .mul
          (.binop .add
            (.binop .mul (.var "gidx0" .int32_) stepConst .int32_)
            startConst .int32_)
          kOrigConst .int32_)
        (.var "Ridx0" .int32_) .int32_
  | _ => panic! "L14.B.3: viewIndexUOpForA: unsupported uop chain"

/-- L14.B.3: derive the LOAD-B index UOp. Supported uop chains:
    - `.buffer _ _ _` (row-major): `Ridx0 * N + gidx1`.
    - `.permute (.buffer _ _ _) [1, 0]` (b transposed):
      `gidx1 * K + Ridx0`.
    - `.expand (.buffer _ origShape _) _` (broadcast singleton on N):
      `Ridx0 * 1 + 0 = Ridx0` (when original is shape [K, 1]).
      For other expand shapes panic; the manifest only exercises
      the `b.expand(K, N)` from `(K, 1)` case. -/
partial def viewIndexUOpForB (uop : UOp) (K N : Nat) : UOp :=
  let rowMajorIdx (n : Nat) : UOp :=
    .binop .add
      (.binop .mul (.var "Ridx0" .int32_)
                   (.const .int32_ (.i (Int.ofNat n))) .int32_)
      (.var "gidx1" .int32_) .int32_
  match uop with
  | .buffer _ _ _ => rowMajorIdx N
  | .reshape (.buffer _ _ _) _ => rowMajorIdx N
  | .permute (.buffer _ _ _) [1, 0] =>
      -- 2-D transpose: gidx1 * K + Ridx0
      .binop .add
        (.binop .mul (.var "gidx1" .int32_)
                     (.const .int32_ (.i (Int.ofNat K))) .int32_)
        (.var "Ridx0" .int32_) .int32_
  | .expand (.buffer _ origShape _) _ =>
      -- The original buffer has shape [K, 1] (singleton on N-axis).
      -- Each "expanded" element is just data2[Ridx0 * 1 + 0] = data2[Ridx0].
      let origN := (origShape[1]?).getD 1
      .binop .add
        (.binop .mul (.var "Ridx0" .int32_)
                     (.const .int32_ (.i (Int.ofNat origN))) .int32_)
        (.const .int32_ (.i 0)) .int32_
  | _ => panic! "L14.B.3: viewIndexUOpForB: unsupported uop chain"

/-- Dispatch dims for a shape sentinel. L13.A refactor: delegates to
    `Codegen.Opt.pickDispatchPlan`, the pure heuristic function that
    handles the 11 L11 sentinels exhaustively (closed-form formula for
    M,K,N ≥ 1024 + a 64×64 below-TC-tile branch).

    The `panic!` arm is unreachable: the theorem
    `Codegen.Opt.pickDispatchPlan_matches_capture` proves that every
    `ShapeSentinel` produces `some _` under the formula. Lean's
    type-checker rejects the theorem if any sentinel doesn't match,
    so a wrong formula can't reach runtime. -/
def dispatchDimsFor (s : Renderer.Metal.ShapeSentinel) : Codegen.GpuDims :=
  let (M, K, N) := s.toTriple
  match Codegen.Opt.pickDispatchPlan M K N .bfloat16_ .bfloat16_ with
  | some plan => plan.dims
  | none      =>
      -- Unreachable per `pickDispatchPlan_matches_capture`. The
      -- panic is a safety belt for the Inhabited return type only.
      panic! s!"Pipeline.dispatchDimsFor: pickDispatchPlan returned none for sentinel"

/-- The captured kernel function name per shape. -/
def kernelNameFor : Renderer.Metal.ShapeSentinel → String
  | .bf16_64x64             => "r_2_32_2_2_4_4_8"
  | .bf16_1024x1024         => "r_32_8_32_4_2_4_4_128"
  | .bf16_2048x2048         => "r_64_16_32_4_2_4_4_256"
  | .bf16_4096x4096         => "r_128_32_32_4_2_4_4_512"
  | .bf16_8192x8192         => "r_256_64_32_4_2_4_4_1024"
  | .bf16_8192x1024x1024    => "r_256_8_32_4_2_4_4_128"
  | .bf16_4096x1024x1024    => "r_128_8_32_4_2_4_4_128"
  | .bf16_2048x1024x1024    => "r_64_8_32_4_2_4_4_128"
  | .bf16_1024x1024x8192    => "r_32_64_32_4_2_4_4_128"
  | .bf16_1024x1024x4096    => "r_32_32_32_4_2_4_4_128"
  | .bf16_1024x1024x2048    => "r_32_16_32_4_2_4_4_128"

/-- L14.B.2.c: shared rangeify+trace hook. Called by both
    `Pipeline.realize` (matmul-verify path) and `Pipeline.realizeView`
    (Python view matmul via `tgrad_matmul_view_lean`). Builds a SINK
    from the input uops, runs `Schedule_Rangeify_rangeify` (no-op
    pass-through at L14.B.2.c scope — L14.B.3 lifts the real rule
    set), and emits a JSONL trace row if `TGRAD_RANGEIFY_TRACE=1`. -/
def runRangeifyAndTrace (a b : Tensor) : IO Unit := do
  let matmulSink := buildMatmulSink a.uop b.uop
  let rangeified := Schedule.Rangeify.rangeify matmulSink
  RangeifyTrace.maybeEmit
    { input_uop_kind         := matmulSink.kind.toStr
    , output_uop_kind        := rangeified.kind.toStr
    , movement_count_in      := UOp.countMovementNodes matmulSink
    , movement_count_out     := UOp.countMovementNodes rangeified
    , root_indexed_ops_count := UOp.countIndexedLoadStore rangeified
    }

/-- `Pipeline.realize` for the bf16 N×N matmul, single-shape (L5.a).

    Takes pre-allocated input Tensors `a`, `b` (caller responsible
    for filling). Allocates the output buffer, compiles the captured
    MSL for the shape, dispatches the kernel, returns the output
    Tensor.

    L14.B.2.c: this function now invokes `runRangeifyAndTrace` at the
    top, which calls `Schedule_Rangeify_rangeify` on a SINK built from
    `a.uop` and `b.uop` and emits the JSONL trace row. For BUFFER-only
    inputs this is a structural no-op (L11/L13/L13_F bit-identical).

    Errors surface as `Except PipelineError`. -/
def realize (a b : Tensor) (shape : Renderer.Metal.ShapeSentinel) :
    IO (Except PipelineError Tensor) := do
  -- L14.B.2.c: rangeify + trace before the existing dispatch path.
  -- For BUFFER-only inputs this is a structural no-op (rangeify is
  -- identity); the trace records movement-node counts so view-input
  -- callers (`Pipeline.realizeView` via `tgrad_matmul_view`) can
  -- verify rangeify saw the chain.
  runRangeifyAndTrace a b
  -- 1. Load captured MSL for the shape.
  let mslPath := shape.fixturePath
  let msl ←
    try
      IO.FS.readFile (System.FilePath.mk mslPath)
    catch _ =>
      return .error (.notInLeanScope s!"no captured MSL at {mslPath}")
  -- 2. Compile.
  let libPtr ← Runtime.Metal.metalCompile msl
  if libPtr == 0 then
    return .error (.compileFailed s!"metalCompile returned 0 for {mslPath}")
  -- 3. Alloc output buffer (same size as inputs at L5.a — square matmul).
  let outBuf ← Runtime.Metal.metalAlloc (USize.ofNat a.sizeBytes)
  if outBuf == 0 then
    Runtime.Metal.metalLibraryRelease libPtr
    return .error (.allocFailed s!"metalAlloc {a.sizeBytes} bytes returned 0")
  -- 4. Dispatch. The Metal bridge uses `dispatchThreads:threadsPerThreadgroup:`,
  -- which takes TOTAL threads (not threadgroup count). The captured
  -- `dims.grid` is the threadgroup count (matches tinygrad's capture
  -- convention), so multiply through to get total threads.
  let dims := dispatchDimsFor shape
  let fnName := kernelNameFor shape
  let totalX : USize := USize.ofNat (dims.grid.x * dims.threadgroup.x)
  let totalY : USize := USize.ofNat (dims.grid.y * dims.threadgroup.y)
  let totalZ : USize := USize.ofNat (dims.grid.z * dims.threadgroup.z)
  let rc ← Runtime.Metal.metalDispatch libPtr fnName
    #[outBuf, a.buffer.raw, b.buffer.raw]
    totalX totalY totalZ
    (USize.ofNat dims.threadgroup.x) (USize.ofNat dims.threadgroup.y) (USize.ofNat dims.threadgroup.z)
  if rc != 0 then
    Runtime.Metal.metalFree outBuf (USize.ofNat a.sizeBytes)
    Runtime.Metal.metalLibraryRelease libPtr
    return .error (.dispatchFailed rc)
  -- 5. Construct output Tensor wrapping the result buffer.
  -- L14.A: the Tensor now carries a `uop : UOp` graph; for the
  -- contiguous-buffer path here, `Tensor.ofBuffer` wraps the raw
  -- buffer in a `.buffer` UOp + records shape + dtype.
  Runtime.Metal.metalLibraryRelease libPtr
  pure (.ok (Tensor.ofBuffer { raw := outBuf, size := a.sizeBytes } a.shape a.dtype))

/-- L14.B.2.c: view-aware matmul. When either input has a non-BUFFER
    uop (e.g. `.permute` from `a.transpose()`), the captured matmul
    kernels can't be used directly — they assume row-major contiguous
    buffer access. This routes through `scalarMatmulKernelDeclWithIdx`
    with A/B load-index UOps derived from the input uop chains via
    `viewIndexUOpForA` / `viewIndexUOpForB`.

    Caller pre-condition: at least one of `a.uop` / `b.uop` is NOT a
    BUFFER (else `Pipeline.realize` takes the existing captured path).

    L14.B.2.c smoke scope: handles `a.transpose() @ b` for square
    shapes (M = K = N). The `viewIndexUOp*` helpers panic on
    unsupported uop chains; L14.B.3 generalises. -/
def realizeView (a b : Tensor) : IO (Except PipelineError Tensor) := do
  -- L14.B.2.c: emit the rangeify trace (the same one Pipeline.realize
  -- emits) so the smoke at the Python `__matmul__`-via-`tgrad_matmul_view`
  -- entry can verify rangeify saw the view chain.
  runRangeifyAndTrace a b
  let aShape := a.shape
  let bShape := b.shape
  match aShape, bShape with
  | [M, K], [Kb, N] =>
      if K != Kb then
        return .error (.notInLeanScope s!"realizeView: contraction dim mismatch ({M}×{K} @ {Kb}×{N})")
      -- Derive index UOps from the input uop chains.
      let aIdx := viewIndexUOpForA a.uop M K
      let bIdx := viewIndexUOpForB b.uop K N
      -- L14.B.3 fix: the kernel function name must uniquely identify
      -- the *access pattern*, not just (M,K,N). Otherwise two view
      -- variants on the same shape produce the same `fnName` and the
      -- C-side pipeline cache (keyed on `library_ptr:fn_name`) can
      -- return a stale pipeline if a freed lib's address is reused.
      -- We derive the tag from the rendered index expressions, which
      -- are stable strings encoding the full access pattern.
      let viewTag : String :=
        let h := String.hash (aIdx.renderIndexExpr ++ "|" ++ bIdx.renderIndexExpr)
        s!"view{h.toNat}"
      let kd := Renderer.Metal.scalarMatmulKernelDeclWithIdx M K N aIdx bIdx viewTag
      let msl := Renderer.Metal.renderKernel kd
      if msl.isEmpty then
        return .error (.compileFailed "realizeView: renderKernel produced empty MSL")
      let libPtr ← Runtime.Metal.metalCompile msl
      if libPtr == 0 then
        return .error (.compileFailed s!"realizeView: metalCompile failed for matmul_scalar_{viewTag}_{M}x{K}x{N}")
      let outBytes := M * N * a.dtype.sizeBytes
      let outBuf ← Runtime.Metal.metalAlloc (USize.ofNat outBytes)
      if outBuf == 0 then
        Runtime.Metal.metalLibraryRelease libPtr
        return .error (.allocFailed s!"realizeView: metalAlloc {outBytes} returned 0")
      let fnName := s!"matmul_scalar_{viewTag}_{M}x{K}x{N}"
      let rc ← Runtime.Metal.metalDispatch libPtr fnName
        #[outBuf, a.buffer.raw, b.buffer.raw]
        (USize.ofNat M) (USize.ofNat N) 1
        1 1 1
      Runtime.Metal.metalLibraryRelease libPtr
      if rc != 0 then
        Runtime.Metal.metalFree outBuf (USize.ofNat outBytes)
        return .error (.dispatchFailed rc)
      pure (.ok (Tensor.ofBuffer { raw := outBuf, size := outBytes } [M, N] a.dtype))
  | _, _ =>
      return .error (.notInLeanScope s!"realizeView: only 2-D shapes supported (got {aShape} @ {bShape})")

end Pipeline

end Tgrad
