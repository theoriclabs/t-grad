import Tgrad.Runtime.Buffer
import Tgrad.Runtime.MetalProgram
import Tgrad.Runtime.Cache
import Tgrad.Pipeline
import Tgrad.Tensor
import Tgrad.Renderer.Elementwise
import Tgrad.Renderer.Creation
import Tgrad.Renderer.MatmulScalar
import Tgrad.Renderer.MatmulTc
import Tgrad.Codegen.Opt.Heuristic

/-! # Tgrad.PythonFFI — `@[export]` entries for Python ctypes

  The Lean-owns-the-runtime side of the L6 bridge. Python calls in
  via tiny C trampolines in `c/tgrad_python.c`, which translate
  ctypes-friendly args (uint64, const uint8_t*, size_t) into Lean's
  calling convention and dispatch the `@[export]` entries below.

  Direction is **inverted** vs L4's `@[extern]` (which is Lean → C):
  here Python → C → Lean. Both directions coexist in the same
  `libtgrad.dylib` produced by `c/Makefile`'s `dylib` target.

  Long-form spec lived in the pre-v1.0.0 `Tgrad/GOAL_NEXT.md`
  §G_L6_b ladder; the v1 contract is the @[export] surface below
  plus `python/tgrad.py`.
-/
namespace Tgrad.PythonFFI

/-- Allocate `nBytes` of MTLBuffer memory. Returns the raw MTLBuffer
    pointer as `UInt64`; 0 on failure. -/
@[export tgrad_tensor_alloc_lean]
def tensorAlloc (nBytes : USize) : IO UInt64 :=
  Tgrad.Runtime.Metal.metalAlloc nBytes

/-- Free a buffer (returns to LRU pool, or releases if pool is full). -/
@[export tgrad_tensor_free_lean]
def tensorFree (bufPtr : UInt64) (nBytes : USize) : IO Unit :=
  Tgrad.Runtime.Metal.metalFree bufPtr nBytes

/-- Write a `ByteArray` of host bytes into a buffer at offset 0. -/
@[export tgrad_tensor_write_bytes_lean]
def tensorWriteBytes (bufPtr : UInt64) (bytes : @& ByteArray) : IO Unit :=
  Tgrad.Runtime.Metal.metalBufferWriteBytes bufPtr bytes

/-- Read `nBytes` from a buffer's contents into a fresh `ByteArray`. -/
@[export tgrad_tensor_read_bytes_lean]
def tensorReadBytes (bufPtr : UInt64) (nBytes : USize) : IO ByteArray :=
  Tgrad.Runtime.Metal.metalBufferReadBytes bufPtr nBytes

/-- Cache for the compiled bf16 64×64 matmul library. Lazy: first
    matmul64x64 call renders the KernelDecl and compiles; subsequent
    calls reuse the cached library pointer. Skipping the per-call
    `renderKernel` + `metalCompile` + `metalLibraryRelease` cycle
    is the load-bearing perf optimization for L7's ratio predicate. -/
initialize cachedMatmul64Lib : IO.Ref UInt64 ← IO.mkRef 0

/-- Run the rendered bf16 64×64 matmul kernel with caller-supplied
    input and output MTLBuffer pointers. Returns 0 on success;
    negative codes:
       -1: renderKernel produced empty MSL
       -2: metalCompile returned 0
       other: dispatch's signed rc reinterpreted via `UInt32.toInt32`. -/
@[export tgrad_matmul_64x64_lean]
def matmul64x64 (aPtr bPtr outPtr : UInt64) : IO Int32 := do
  -- Load the compiled library once; reuse across calls.
  let mut libPtr ← cachedMatmul64Lib.get
  if libPtr == 0 then
    -- Rendered, not read. See `compileOrCacheGet`.
    let kernelDecl ← match Tgrad.Pipeline.generatedKernelDeclFor .bf16_64x64 with
      | .error _ => return -1
      | .ok decl => pure decl
    let msl := Tgrad.Renderer.Metal.renderKernel kernelDecl
    if msl.isEmpty then return -1
    libPtr ← Tgrad.Pipeline.compileNativeBf16OrZero msl
    if libPtr == 0 then return -2
    cachedMatmul64Lib.set libPtr
  let dims := Tgrad.Pipeline.generatedDispatchDimsFor .bf16_64x64
  let totalX : USize := USize.ofNat (dims.grid.x * dims.threadgroup.x)
  let totalY : USize := USize.ofNat (dims.grid.y * dims.threadgroup.y)
  let totalZ : USize := USize.ofNat (dims.grid.z * dims.threadgroup.z)
  let rc ← Tgrad.Runtime.Metal.metalDispatch libPtr
    (Tgrad.Pipeline.generatedKernelNameFor .bf16_64x64)
    #[outPtr, aPtr, bPtr]
    totalX totalY totalZ
    (USize.ofNat dims.threadgroup.x) (USize.ofNat dims.threadgroup.y)
    (USize.ofNat dims.threadgroup.z)
  pure rc.toInt32

-- ----------------------------------------------------------------------
-- L11: general matmul entry — accepts (M, K, N), routes via ShapeSentinel.
-- ----------------------------------------------------------------------

/-- Per-shape compiled-library cache. Key: ShapeSentinel toString.
    Value: MTLLibrary pointer (UInt64; 0 = not yet compiled). First
    call for a shape compiles + caches; subsequent calls reuse. -/
initialize libCache : IO.Ref (List (String × UInt64)) ← IO.mkRef []

private def cacheKey : Tgrad.Renderer.Metal.ShapeSentinel → String
  | .bf16_64x64             => "bf16_64x64"
  | .bf16_1024x1024         => "bf16_1024x1024"
  | .bf16_2048x2048         => "bf16_2048x2048"
  | .bf16_4096x4096         => "bf16_4096x4096"
  | .bf16_8192x8192         => "bf16_8192x8192"
  | .bf16_8192x1024x1024    => "bf16_8192x1024x1024"
  | .bf16_4096x1024x1024    => "bf16_4096x1024x1024"
  | .bf16_2048x1024x1024    => "bf16_2048x1024x1024"
  | .bf16_1024x1024x8192    => "bf16_1024x1024x8192"
  | .bf16_1024x1024x4096    => "bf16_1024x1024x4096"
  | .bf16_1024x1024x2048    => "bf16_1024x1024x2048"

private def cacheLookup (key : String) : List (String × UInt64) → UInt64
  | []              => 0
  | (k, v) :: rest  => if k == key then v else cacheLookup key rest

private def compileOrCacheGet (sentinel : Tgrad.Renderer.Metal.ShapeSentinel) : IO UInt64 := do
  let key := cacheKey sentinel
  let cache ← libCache.get
  let cached := cacheLookup key cache
  if cached != 0 then
    return cached
  let kernelDecl ← match Tgrad.Pipeline.generatedKernelDeclFor sentinel with
    | .error _ => return 0
    | .ok decl => pure decl
  let msl := Tgrad.Renderer.Metal.renderKernel kernelDecl
  if msl.isEmpty then return 0
  let lib ← Tgrad.Pipeline.compileNativeBf16OrZero msl
  if lib == 0 then return 0
  libCache.modify (fun c => (key, lib) :: c)
  pure lib

/-- General bf16 matmul. (M, K, N) selects the right `KernelDecl` via
    `ShapeSentinel.ofTriple`; the MSL is emitted by `renderKernel`.
    Returns 0 on success; negative codes:
       -1: unsupported (M, K, N) — not in L11 manifest
       -2: renderKernel produced empty MSL OR metalCompile failed
       other: dispatch rc reinterpreted via UInt32.toInt32.

    No filesystem access: `fixtures/codegen/*.msl` are a differential
    reference for the gates, not a runtime input. -/
@[export tgrad_matmul_lean]
def matmulGeneral (M K N : USize) (aPtr bPtr outPtr : UInt64) : IO Int32 := do
  let sentinel ← match Tgrad.Renderer.Metal.ShapeSentinel.ofTriple M.toNat K.toNat N.toNat with
    | none   => return -1
    | some s => pure s
  let libPtr ← compileOrCacheGet sentinel
  if libPtr == 0 then return -2
  let dims := Tgrad.Pipeline.generatedDispatchDimsFor sentinel
  let totalX : USize := USize.ofNat (dims.grid.x * dims.threadgroup.x)
  let totalY : USize := USize.ofNat (dims.grid.y * dims.threadgroup.y)
  let totalZ : USize := USize.ofNat (dims.grid.z * dims.threadgroup.z)
  let fnName := Tgrad.Pipeline.generatedKernelNameFor sentinel
  let rc ← Tgrad.Runtime.Metal.metalDispatch libPtr fnName
    #[outPtr, aPtr, bPtr]
    totalX totalY totalZ
    (USize.ofNat dims.threadgroup.x) (USize.ofNat dims.threadgroup.y) (USize.ofNat dims.threadgroup.z)
  pure rc.toInt32

-- ----------------------------------------------------------------------
-- L12: alternate generated-emitter entry. Both sentinel entries now render
-- the parametric declaration; this symbol remains distinct while the legacy
-- L12 routing/caching checks are retired with the transcription.
-- ----------------------------------------------------------------------

/-- Algebraic-path compiled-library cache. Separate from `libCache` so
    the legacy benchmark toggle remains observable while both entries
    exercise the same generated declaration and launch geometry. -/
initialize libCacheAlg : IO.Ref (List (String × UInt64)) ← IO.mkRef []

private def compileOrCacheGetAlg (sentinel : Tgrad.Renderer.Metal.ShapeSentinel) : IO UInt64 := do
  let key := cacheKey sentinel
  let cache ← libCacheAlg.get
  let cached := cacheLookup key cache
  if cached != 0 then
    return cached
  let kernelDecl ← match Tgrad.Pipeline.generatedKernelDeclFor sentinel with
    | .error _ => return 0
    | .ok decl => pure decl
  let msl := Tgrad.Renderer.Metal.renderKernel kernelDecl
  if msl.isEmpty then return 0
  let lib ← Tgrad.Pipeline.compileNativeBf16OrZero msl
  if lib == 0 then return 0
  libCacheAlg.modify (fun c => (key, lib) :: c)
  pure lib

/-- General bf16 matmul via the alternate generated-emitter cache. (M, K, N)
    routes through `ShapeSentinel.ofTriple`, then through the same parametric
    declaration and generated launch geometry as `tgrad_matmul_lean`.

    L12 evidence: the dylib must export `_tgrad_matmul_alg` (separate
    symbol from `_tgrad_matmul`). The L12 gate's anti-cheat predicate
    requires the bench's `--use-algebraic-emit` flag to resolve to
    THIS symbol (and only this one), not silently fall through to the
    primary cache. -/
@[export tgrad_matmul_alg_lean]
def matmulGeneralAlg (M K N : USize) (aPtr bPtr outPtr : UInt64) : IO Int32 := do
  let sentinel ← match Tgrad.Renderer.Metal.ShapeSentinel.ofTriple M.toNat K.toNat N.toNat with
    | none   => return -1
    | some s => pure s
  let libPtr ← compileOrCacheGetAlg sentinel
  if libPtr == 0 then return -2
  let dims := Tgrad.Pipeline.generatedDispatchDimsFor sentinel
  let totalX : USize := USize.ofNat (dims.grid.x * dims.threadgroup.x)
  let totalY : USize := USize.ofNat (dims.grid.y * dims.threadgroup.y)
  let totalZ : USize := USize.ofNat (dims.grid.z * dims.threadgroup.z)
  let fnName := Tgrad.Pipeline.generatedKernelNameFor sentinel
  let rc ← Tgrad.Runtime.Metal.metalDispatch libPtr fnName
    #[outPtr, aPtr, bPtr]
    totalX totalY totalZ
    (USize.ofNat dims.threadgroup.x) (USize.ofNat dims.threadgroup.y) (USize.ofNat dims.threadgroup.z)
  pure rc.toInt32

-- ----------------------------------------------------------------------
-- L13.B: scalar-matmul entry for below-TC-tile shapes (M, K, or N < 8).
-- The kernel is emitted algebraically from `scalarMatmulKernelDecl`;
-- one thread per output element, no WMMA. Distinct symbol from
-- `tgrad_matmul_lean` so the L13.B gate's anti-cheat can confirm the
-- below-TC-tile path is observably wired (not silent fall-through to
-- the sentinel path).
-- ----------------------------------------------------------------------

/-- Scalar-path compiled-library cache, keyed by `"<M>x<K>x<N>"`. -/
initialize libCacheSmall : IO.Ref (List (String × UInt64)) ← IO.mkRef []

private def compileOrCacheGetSmall (M K N : Nat) : IO UInt64 := do
  let key := s!"{M}x{K}x{N}"
  let cache ← libCacheSmall.get
  let cached := cacheLookup key cache
  if cached != 0 then
    return cached
  let msl := Tgrad.Renderer.Metal.renderKernel
    (Tgrad.Renderer.Metal.scalarMatmulKernelDecl M K N)
  if msl.isEmpty then return 0
  let lib ← Tgrad.Runtime.Metal.metalCompile msl
  if lib == 0 then return 0
  libCacheSmall.modify (fun c => (key, lib) :: c)
  pure lib

/-- L13.B + L13.C scalar matmul for shapes rejected by the parametric TC
    generator. Python queries Lean eligibility before selecting this entry.
    Covers:
      * L13.B: below-TC-tile (M, K, or N < 8) — 5 manifest entries
      * L13.C: catch-all general path — any other non-sentinel bf16
                shape (TC-aligned-non-pow2, pow2-non-benchmark,
                asym-tall, asym-wide, large-mixed)
    Returns 0 on success;
       -1 : (M, K, N) is TC-eligible (caller should use a generated
            tensor-core entry instead)
       -2 : compile failed
       other : dispatch rc -/
@[export tgrad_matmul_small_lean]
def matmulSmall (M K N : USize) (aPtr bPtr outPtr : UInt64) : IO Int32 := do
  let Mn := M.toNat
  let Kn := K.toNat
  let Nn := N.toNat
  -- The scalar dispatch dimensions are always (M, N, 1). This entry remains
  -- a correctness-only fallback; the caller is responsible for selecting it
  -- only after `matmulTcEligible` rejects the shape.
  if Mn < 1 ∨ Kn < 1 ∨ Nn < 1 then return -1
  if (Tgrad.Renderer.Metal.tcMatmulKernelDeclManualLoadWide Mn Kn Nn).isOk then
    return -1
  let libPtr ← compileOrCacheGetSmall Mn Kn Nn
  if libPtr == 0 then return -2
  -- Scalar dispatch: grid=(M, N, 1), tg=(1, 1, 1).
  let totalX : USize := USize.ofNat Mn
  let totalY : USize := USize.ofNat Nn
  let totalZ : USize := 1
  let fnName := s!"matmul_scalar_{Mn}x{Kn}x{Nn}"
  let rc ← Tgrad.Runtime.Metal.metalDispatch libPtr fnName
    #[outPtr, aPtr, bPtr]
    totalX totalY totalZ
    1 1 1
  pure rc.toInt32

-- ----------------------------------------------------------------------
-- L13.F: TC-eligible non-sentinel general matmul. The legacy explicit TC
-- exports below remain WMMA-only and return -2 when this backend rejects
-- native bf16. The default graph realizer is different: `runBufferMatmul`
-- treats structural eligibility and backend capability separately and falls
-- back to portable scalar execution when this compile probe returns zero.
-- ----------------------------------------------------------------------

initialize libCacheTcManual : IO.Ref (List (String × UInt64)) ← IO.mkRef []

private def compileOrCacheGetTcManual (M K N : Nat) : IO UInt64 := do
  let key := s!"manual:{M}x{K}x{N}"
  let cache ← libCacheTcManual.get
  let cached := cacheLookup key cache
  if cached != 0 then
    return cached
  match Tgrad.Renderer.Metal.tcMatmulKernelDeclManualLoadWide M K N with
  | .error _ => return 0
  | .ok kd =>
      let msl := Tgrad.Renderer.Metal.renderKernel kd
      if msl.isEmpty then return 0
      let lib ← Tgrad.Pipeline.compileNativeBf16OrZero msl
      if lib == 0 then return 0
      libCacheTcManual.modify (fun c => (key, lib) :: c)
      pure lib

/-- L13.F TC matmul. For shapes meeting the generated kernel's eligibility
    constraints (M ≥ 32; K ≥ 8; N ≥ 32; M % 32 = 0; K % 8 = 0;
    N is divisible by its `32 * min 4 (N / 32)` output tile).
    Generates a Lean-owned manual-load WMMA kernel.
    Returns -1 if shape is NOT TC-eligible (caller should route via
    `tgrad_matmul_small` instead). -/
@[export tgrad_matmul_tc_lean]
def matmulTc (M K N : USize) (aPtr bPtr outPtr : UInt64) : IO Int32 := do
  let Mn := M.toNat
  let Kn := K.toNat
  let Nn := N.toNat
  if !(Tgrad.Renderer.Metal.tcMatmulKernelDeclManualLoadWide Mn Kn Nn).isOk then
    return -1
  let libPtr ← compileOrCacheGetTcManual Mn Kn Nn
  if libPtr == 0 then return -2
  let dims := Tgrad.Renderer.Metal.tcLaunchDims Mn Nn
  let totalX : USize := USize.ofNat (dims.grid.x * dims.threadgroup.x)
  let totalY : USize := USize.ofNat (dims.grid.y * dims.threadgroup.y)
  let totalZ : USize := USize.ofNat (dims.grid.z * dims.threadgroup.z)
  let fnName := s!"matmul_tc_manual_{Mn}x{Kn}x{Nn}"
  let rc ← Tgrad.Runtime.Metal.metalDispatch libPtr fnName
    #[outPtr, aPtr, bPtr]
    totalX totalY totalZ
    (USize.ofNat dims.threadgroup.x) (USize.ofNat dims.threadgroup.y)
    (USize.ofNat dims.threadgroup.z)
  pure rc.toInt32

/-- L13.F.STRICT.B manual-load TC matmul. Installed as a distinct
    entry so the bench harness can prove correctness/perf before
    L13.F.STRICT.C makes it the production TC path. -/
@[export tgrad_matmul_tc_manual_load_lean]
def matmulTcManualLoad (M K N : USize) (aPtr bPtr outPtr : UInt64) : IO Int32 := do
  let Mn := M.toNat
  let Kn := K.toNat
  let Nn := N.toNat
  if !(Tgrad.Renderer.Metal.tcMatmulKernelDeclManualLoadWide Mn Kn Nn).isOk then
    return -1
  let libPtr ← compileOrCacheGetTcManual Mn Kn Nn
  if libPtr == 0 then return -2
  let dims := Tgrad.Renderer.Metal.tcLaunchDims Mn Nn
  let totalX : USize := USize.ofNat (dims.grid.x * dims.threadgroup.x)
  let totalY : USize := USize.ofNat (dims.grid.y * dims.threadgroup.y)
  let totalZ : USize := USize.ofNat (dims.grid.z * dims.threadgroup.z)
  let fnName := s!"matmul_tc_manual_{Mn}x{Kn}x{Nn}"
  let rc ← Tgrad.Runtime.Metal.metalDispatch libPtr fnName
    #[outPtr, aPtr, bPtr]
    totalX totalY totalZ
    (USize.ofNat dims.threadgroup.x) (USize.ofNat dims.threadgroup.y)
    (USize.ofNat dims.threadgroup.z)
  pure rc.toInt32

/-- L13.F query: does Lean's manual-load TC decl consider this shape
    TC-eligible? Pure check that Python can call BEFORE dispatch to decide
    TC vs scalar routing. Returns 1 if eligible, 0 if not. -/
@[export tgrad_matmul_tc_eligible_lean]
def matmulTcEligible (M K N : USize) : IO UInt32 := do
  let Mn := M.toNat
  let Kn := K.toNat
  let Nn := N.toNat
  match Tgrad.Renderer.Metal.tcMatmulKernelDeclManualLoadWide Mn Kn Nn with
  | .error _ => pure 0
  | .ok _    => pure 1

-- ----------------------------------------------------------------------
-- L14.A: TensorRegistry + tgrad_tensor_from_buffer.
--
-- Python holds an opaque `UInt64` handle; Lean owns the underlying
-- `Tensor` (UOp graph + dtype). L14.B's view methods (transpose,
-- reshape, permute, expand, slice) consume a handle, look up the
-- Tensor, compose a movement-node onto its uop, and register the
-- result, returning a new handle. L14.A only ships the BUFFER path:
-- Python allocates a buffer with `tgrad_tensor_alloc`, writes bytes
-- in, then registers a Tensor whose uop is `.buffer (h, shape, dtype)`.
-- ----------------------------------------------------------------------

/-- Lean-side Tensor registry: opaque `UInt64` handle → `Tensor`. The
    map is append-only at L14.A (release is L14.B+ work — for now
    Python's existing `tgrad_tensor_free` reclaims the underlying
    MTLBuffer, and the registry entry is never re-used). -/
initialize tensorRegistry : IO.Ref (List (UInt64 × Tgrad.Tensor)) ← IO.mkRef []

/-- Monotonic counter for the next handle to hand out. -/
initialize nextTensorHandle : IO.Ref UInt64 ← IO.mkRef 1

namespace TensorRegistry

/-- Register a Tensor; return its handle. -/
def register (t : Tgrad.Tensor) : IO UInt64 := do
  let h ← nextTensorHandle.get
  nextTensorHandle.set (h + 1)
  tensorRegistry.modify (fun reg => (h, t) :: reg)
  pure h

/-- Look up the Tensor for a handle. Returns `none` if the handle was
    never registered (or was released — but L14.A is append-only). -/
def get? (h : UInt64) : IO (Option Tgrad.Tensor) := do
  let reg ← tensorRegistry.get
  pure (reg.find? (fun p => p.1 == h) |>.map Prod.snd)

end TensorRegistry

/-- Stable FFI dtype decoding. Invalid codes remain invalid; no metadata or
tensor path may silently substitute bf16. -/
def dtypeOfCode? (code : UInt8) : Option Tgrad.Dtype := Tgrad.Dtype.ofCode? code

/-- L14.A: construct a `Tensor` with `uop := .buffer h shape dtype`,
    register it, return the opaque handle.

    `shape` is an Array USize (arbitrary rank); `dtypeCode` is the
    stable FFI encoding (`dtypeOfCode`). The C trampoline marshals a
    `const size_t* shape + size_t ndim` into the Lean Array. -/
@[export tgrad_tensor_from_buffer_lean]
def tensorFromBuffer
    (bufHandle : UInt64) (shape : @& Array USize) (dtypeCode : UInt8) :
    IO UInt64 := do
  let some dt := dtypeOfCode? dtypeCode | return 0
  if !dt.computeSupported then return 0
  let dims : List Nat := shape.toList.map USize.toNat
  let t    : Tgrad.Tensor := { uop := .buffer bufHandle dims dt, dtype := dt }
  TensorRegistry.register t

/-! ## Foreign-grounded dtype query boundary

One stable code identifies a semantic dtype. Query selectors are intentionally
small integers so C/Python only marshal answers computed in Lean.

`dtypeQuery`: 0 valid, 1 priority (two's complement), 2 bits, 3 itemsize,
4 float, 5 int, 6 unsigned, 7 bool, 8 range kind, 9 min bits, 10 max bits,
11 finfo exponent, 12 finfo mantissa, 13 immediate-parent bitmask,
14 format ASCII, 15 compute supported, 16 non-associative triple count.

`dtypeBinaryQuery`: 0 LUB code, 1 lossless-cast bool, 2 less-than bool.
`dtypeUnaryQuery`: 0 strong dtype, 1 least-upper float.
-/

/-- Process-local dtype defaults.  The state is in Lean so every public
observer sees one authority; Python only transports codes to these refs. -/
initialize dtypeDefaultIntState : IO.Ref Tgrad.Dtype ←
  IO.mkRef Tgrad.Dtype.defaultInt
initialize dtypeDefaultFloatState : IO.Ref Tgrad.Dtype ←
  IO.mkRef Tgrad.Dtype.defaultFloat

private def readDtypeDefaults : IO (Tgrad.Dtype × Tgrad.Dtype) := do
  pure (← dtypeDefaultIntState.get, ← dtypeDefaultFloatState.get)

private def boolCode (b : Bool) : UInt64 := if b then 1 else 0

private def parentMask (d : Tgrad.Dtype) : UInt64 :=
  d.immediateParents.foldl (fun mask parent =>
    mask ||| ((1 : UInt64) <<< parent.code.toUInt64)) 0

@[export tgrad_dtype_query_lean]
def dtypeQuery (code query : UInt8) : IO UInt64 := do
  let some d := Tgrad.Dtype.ofCode? code | return (0 : UInt64) - 1
  pure (match query with
    | 0 => 1
    | 1 => if d == .void_ then (0 : UInt64) - 1 else UInt64.ofNat d.priority.natAbs
    | 2 => UInt64.ofNat d.bits
    | 3 => UInt64.ofNat d.sizeBytes
    | 4 => boolCode d.isFloat
    | 5 => boolCode d.isInt
    | 6 => boolCode d.isUnsigned
    | 7 => boolCode d.isBool
    | 8 => d.rangeKind.toUInt64
    | 9 => d.rangeMinBits
    | 10 => d.rangeMaxBits
    | 11 => UInt64.ofNat ((d.finfo.map Prod.fst).getD 255)
    | 12 => UInt64.ofNat ((d.finfo.map Prod.snd).getD 255)
    | 13 => parentMask d
    | 14 => d.formatCode.toUInt64
    | 15 => boolCode d.computeSupported
    | 16 => UInt64.ofNat Tgrad.Dtype.nonAssociativeTriples.length
    | _ => (0 : UInt64) - 1)

@[export tgrad_dtype_binary_query_lean]
def dtypeBinaryQuery (query leftCode rightCode : UInt8) : IO UInt8 := do
  let some left := Tgrad.Dtype.ofCode? leftCode | return 255
  let some right := Tgrad.Dtype.ofCode? rightCode | return 255
  pure (match query with
    | 0 => (Tgrad.Dtype.lub left right).code
    | 1 => if Tgrad.Dtype.canLosslessCast left right then 1 else 0
    | 2 => if Tgrad.Dtype.lt left right then 1 else 0
    | _ => 255)

@[export tgrad_dtype_unary_query_lean]
def dtypeUnaryQuery (query code : UInt8) : IO UInt8 := do
  let some d := Tgrad.Dtype.ofCode? code | return 255
  let (defaultInt, defaultFloat) ← readDtypeDefaults
  pure (match query with
    | 0 => (Tgrad.Dtype.strongWithDefaults defaultInt defaultFloat d).code
    | 1 => (Tgrad.Dtype.leastUpperFloatWithDefault defaultFloat d).code
    | _ => 255)

@[export tgrad_dtype_lub_many_lean]
def dtypeLubMany (codes : @& Array UInt8) : IO UInt8 := do
  let decoded := codes.toList.mapM Tgrad.Dtype.ofCode?
  let some dtypes := decoded | return 255
  pure ((Tgrad.Dtype.leastUpperMany? dtypes).map Tgrad.Dtype.code |>.getD 255)

@[export tgrad_dtype_infer_python_lean]
def dtypeInferPython (tags : @& Array UInt8) : IO UInt8 := do
  let (defaultInt, defaultFloat) ← readDtypeDefaults
  pure ((Tgrad.Dtype.inferPythonTagsWithDefaults?
    defaultInt defaultFloat tags.toList).map Tgrad.Dtype.code |>.getD 255)

@[export tgrad_dtype_default_lean]
def dtypeDefault (which : UInt8) : IO UInt8 := do
  if which == 0 then return (← dtypeDefaultIntState.get).code
  if which == 1 then return (← dtypeDefaultFloatState.get).code
  pure 255

/-- Mutate one runtime default iff the decoded dtype belongs to the exact
foreign-admitted set for that role.  Returns one on mutation and zero on every
rejection, including invalid codes. -/
@[export tgrad_dtype_set_default_lean]
def dtypeSetDefault (which code : UInt8) : IO UInt8 := do
  let some d := Tgrad.Dtype.ofCode? code | return 0
  if which == 0 && d.integerDefaultAllowed then
    dtypeDefaultIntState.set d
    return 1
  if which == 1 && d.floatingDefaultAllowed then
    dtypeDefaultFloatState.set d
    return 1
  pure 0

/-- Resolve the current floating default through creation admission without
allocating or dispatching.  `255` means the semantic default is valid but the
current compute backend does not admit it; no substitution occurs. -/
@[export tgrad_dtype_creation_default_lean]
def dtypeCreationDefault : IO UInt8 := do
  let d ← dtypeDefaultFloatState.get
  pure (if d.computeSupported then d.code else 255)

@[export tgrad_dtype_backend_name_lean]
def dtypeBackendName (code : UInt8) : IO String :=
  pure ((Tgrad.Dtype.ofCode? code).map Tgrad.Dtype.backendName |>.getD "")

@[export tgrad_dtype_public_name_lean]
def dtypePublicName (code : UInt8) : IO String :=
  pure ((Tgrad.Dtype.ofCode? code).map Tgrad.Dtype.toStr |>.getD "")

@[export tgrad_dtype_display_name_lean]
def dtypeDisplayName (code : UInt8) : IO String :=
  pure ((Tgrad.Dtype.ofCode? code).map Tgrad.Dtype.displayName |>.getD "")

/-- Indexed access to compiler-checked Lean alias/collection declarations.
Table 0 is aliases: query 0 row-count, query 1 target code.
Table 1 is collections: query 0 row-count, query 1 member-count,
query 2 member code at `column`. Invalid requests return UInt64 max. -/
@[export tgrad_dtype_table_query_lean]
def dtypeTableQuery (table query : UInt8) (row column : USize) : IO UInt64 := do
  if table == 0 then
    if query == 0 then return UInt64.ofNat Tgrad.Dtype.aliases.length
    let some entry := Tgrad.Dtype.aliases[row.toNat]? | return (0 : UInt64) - 1
    if query == 1 then return entry.2.code.toUInt64
    return (0 : UInt64) - 1
  if table == 1 then
    if query == 0 then return UInt64.ofNat Tgrad.Dtype.collections.length
    let some entry := Tgrad.Dtype.collections[row.toNat]? | return (0 : UInt64) - 1
    if query == 1 then return UInt64.ofNat entry.2.length
    if query == 2 then
      let some member := entry.2[column.toNat]? | return (0 : UInt64) - 1
      return member.code.toUInt64
    return (0 : UInt64) - 1
  pure ((0 : UInt64) - 1)

@[export tgrad_dtype_table_name_lean]
def dtypeTableName (table : UInt8) (row : USize) : IO String :=
  pure (if table == 0 then (Tgrad.Dtype.aliases[row.toNat]?).map Prod.fst |>.getD ""
        else if table == 1 then (Tgrad.Dtype.collections[row.toNat]?).map Prod.fst |>.getD ""
        else "")

@[export tgrad_bf16_pack_bytes_lean]
def bf16PackBytes (bytes : @& ByteArray) : IO ByteArray :=
  pure ((Tgrad.Dtype.packFp32BytesToBf16? bytes).getD ByteArray.empty)

@[export tgrad_bf16_expand_bytes_lean]
def bf16ExpandBytes (bytes : @& ByteArray) : IO ByteArray :=
  pure ((Tgrad.Dtype.expandBf16BytesToFp32? bytes).getD ByteArray.empty)

@[export tgrad_bf16_round_bits_lean]
def bf16RoundBits (bits : UInt32) : IO UInt32 :=
  pure (Tgrad.Dtype.bf16RoundedF32Bits bits)

/-- L14.A: query the registered shape rank for a tensor handle. Lets
    L14.B's view-method tests verify lookups round-trip via the FFI.
    Returns 0 if the handle isn't registered. -/
@[export tgrad_tensor_rank_lean]
def tensorRank (h : UInt64) : IO USize := do
  let some t ← TensorRegistry.get? h | pure 0
  pure (USize.ofNat t.shape.length)

/-- L14.A: query a single shape dim for a tensor handle. `i` is the
    axis index (0-based). Returns 0 if handle/index invalid. -/
@[export tgrad_tensor_shape_dim_lean]
def tensorShapeDim (h : UInt64) (i : USize) : IO USize := do
  let some t ← TensorRegistry.get? h | pure 0
  pure (USize.ofNat ((t.shape[i.toNat]?).getD 0))

/-- L14.A: query the underlying MTLBuffer pointer for a tensor handle. -/
@[export tgrad_tensor_raw_buffer_lean]
def tensorRawBuffer (h : UInt64) : IO UInt64 := do
  let some t ← TensorRegistry.get? h | pure 0
  pure t.buffer.raw

-- ----------------------------------------------------------------------
-- L14.B.1: view-method @[export] entries.
--
-- Each takes an opaque tensor handle + op-specific args; composes a
-- movement node onto the underlying Tensor's uop; registers the result;
-- returns a new opaque handle. Pure: no buffer allocation, no kernel
-- dispatch, no IO beyond IO.Ref bookkeeping.
-- ----------------------------------------------------------------------

/-- L14.B.1: `Tensor.transpose` (2-D PERMUTE [1, 0]). -/
@[export tgrad_tensor_transpose_lean]
def tensorTranspose (h : UInt64) : IO UInt64 := do
  let some t ← TensorRegistry.get? h
    | throw <| IO.userError s!"tgrad_tensor_transpose: handle {h} not registered"
  TensorRegistry.register t.transpose

/-- L14.B.1: `Tensor.permute axes`. `axes` is the new axis order as
    `Array USize`. -/
@[export tgrad_tensor_permute_lean]
def tensorPermute (h : UInt64) (axes : @& Array USize) : IO UInt64 := do
  let some t ← TensorRegistry.get? h
    | throw <| IO.userError s!"tgrad_tensor_permute: handle {h} not registered"
  let axesNat : List Nat := axes.toList.map USize.toNat
  TensorRegistry.register (t.permute axesNat)

/-- L14.B.1: `Tensor.reshape newShape`. `newShape` is `Array USize`. -/
@[export tgrad_tensor_reshape_lean]
def tensorReshape (h : UInt64) (newShape : @& Array USize) : IO UInt64 := do
  let some t ← TensorRegistry.get? h
    | throw <| IO.userError s!"tgrad_tensor_reshape: handle {h} not registered"
  let shape : List Nat := newShape.toList.map USize.toNat
  TensorRegistry.register (t.reshape shape)

/-- L14.B.1: `Tensor.expand newShape`. `newShape` is `Array USize`. -/
@[export tgrad_tensor_expand_lean]
def tensorExpand (h : UInt64) (newShape : @& Array USize) : IO UInt64 := do
  let some t ← TensorRegistry.get? h
    | throw <| IO.userError s!"tgrad_tensor_expand: handle {h} not registered"
  let shape : List Nat := newShape.toList.map USize.toNat
  TensorRegistry.register (t.expand shape)

/-- L14.B.1: `Tensor.slice slices`. `slices` is a flat `Array USize`
    holding `[start_0, stop_0, step_0, start_1, stop_1, step_1, ...]`
    per axis (3 USize per axis). -/
@[export tgrad_tensor_slice_lean]
def tensorSlice (h : UInt64) (slicesFlat : @& Array USize) : IO UInt64 := do
  let some t ← TensorRegistry.get? h
    | throw <| IO.userError s!"tgrad_tensor_slice: handle {h} not registered"
  let n := slicesFlat.size / 3
  let slices : List Tgrad.Slice := List.range n |>.map (fun i =>
    let start := (slicesFlat[3 * i]?).getD 0
    let stop  := (slicesFlat[3 * i + 1]?).getD 0
    let step  := (slicesFlat[3 * i + 2]?).getD 1
    { start := start.toNat, stop := stop.toNat, step := step.toNat })
  TensorRegistry.register (t.slice slices)

/-- L14.B.1: query the UOp kind of a tensor handle's root. Returns a
    code per `UOpKind.toStr` (`tgrad/python/tgrad.py` decodes):
       0 = BUFFER, 1 = PERMUTE, 2 = RESHAPE, 3 = EXPAND, 4 = SLICE,
       255 = other / handle unregistered.
    Used by Python's matmul to route to `tgrad_matmul_view` when the
    input has a non-BUFFER uop (L14.B.2.c replaced L14.B.1's typed
    error). -/
@[export tgrad_tensor_uop_kind_lean]
def tensorUopKind (h : UInt64) : IO UInt8 := do
  let some t ← TensorRegistry.get? h | pure 255
  pure (match t.uop with
    | .buffer _ _ _  => 0
    | .permute _ _   => 1
    | .reshape _ _   => 2
    | .expand _ _    => 3
    | .slice _ _     => 4
    | _              => 255)

-- ----------------------------------------------------------------------
-- L14.B.2.c: view-aware matmul and unary view materialization.
-- Replaces the L14.B.1 typed-error
-- guard (`MatmulOnNonBufferUop`) — when either input has a non-BUFFER
-- uop, Python routes through this entry which calls
-- `Pipeline.realizeView` (parametric scalar matmul with index UOps
-- derived from the input uop chains). Returns the new opaque tensor
-- handle for the output (with BUFFER root); 0 on error.
--
-- The existing C trampoline has two UInt64 handle arguments. A second handle
-- of 0 is otherwise invalid and is reserved for unary view materialization;
-- this keeps the materializer within the existing stable C ABI.
-- ----------------------------------------------------------------------

@[export tgrad_matmul_view_lean]
def matmulView (aHandle bHandle : UInt64) : IO UInt64 := do
  let some a ← TensorRegistry.get? aHandle | pure 0
  let res ← if bHandle == 0 then
    Tgrad.Pipeline.materializeView a
  else do
    let some b ← TensorRegistry.get? bHandle | pure (.error
      (.notInLeanScope s!"matmulView: handle {bHandle} not registered"))
    Tgrad.Pipeline.realizeView a b
  match res with
  | .error _ => pure 0
  | .ok t    => TensorRegistry.register t

-- ----------------------------------------------------------------------
-- Graph-indexed realize.
--
-- Eight of this module's exports are matmul routes, and Python picks
-- between them by inspecting shapes. That does not scale: every new op
-- would cost another export, another C trampoline, another ctypes
-- binding and another branch in `__matmul__`. It also puts the routing
-- decision in Python, where it is invisible to Lean's type system.
--
-- These three entries replace that shape. Tensor methods build a UOp
-- graph; `tgrad_realize` lowers whatever graph it is handed and decides
-- the route itself. An op then costs a node constructor and a table
-- row, with no ABI surface at all.
-- ----------------------------------------------------------------------

private def binOpOfCode : UInt8 → Option BinOp
  | 0 => some .add
  | 1 => some .mul
  | 2 => some .sub
  | _ => none

/-- Compose a binary node over two graphs. -/
@[export tgrad_tensor_binop_lean]
def tensorBinop (opCode : UInt8) (h1 h2 : UInt64) : IO UInt64 := do
  let some t1 ← TensorRegistry.get? h1 | return 0
  let some t2 ← TensorRegistry.get? h2 | return 0
  let some op := binOpOfCode opCode | return 0
  TensorRegistry.register { uop := .binop op t1.uop t2.uop t1.dtype, dtype := t1.dtype }

/-- Compose a reduction node over one graph. -/
@[export tgrad_tensor_reduce_lean]
def tensorReduce (opCode : UInt8) (h : UInt64) (axis : USize) : IO UInt64 := do
  let some t ← TensorRegistry.get? h | return 0
  let some op := binOpOfCode opCode | return 0
  let ax : UOp := .const .int32_ (.i (Int.ofNat axis.toNat))
  TensorRegistry.register { uop := .reduce op t.uop [ax], dtype := t.dtype }

/-- Buffer-operand matmul for shapes with no captured sentinel: the
    parametric WMMA kernel where the generator accepts the shape, the
    scalar fallback otherwise. Allocates its own output, so the caller
    hands over a graph and receives a materialised tensor. -/
private def runBufferMatmul (a b : Tgrad.Tensor) (M K N : Nat) :
    IO (Option Tgrad.Tensor) := do
  let outBytes := M * N * 2
  let tcOk := (Tgrad.Renderer.Metal.tcMatmulKernelDeclManualLoadWide M K N).isOk
  let tcLibPtr ← if tcOk then compileOrCacheGetTcManual M K N else pure 0
  -- Structural tile eligibility is not hardware support. If this Metal
  -- compiler rejects the native-bf16 WMMA source, use the same portable
  -- scalar generator as an ineligible shape. On capable devices `tcLibPtr`
  -- remains nonzero and the optimized route is unchanged.
  let useTc := tcOk && tcLibPtr != 0
  let libPtr ← if useTc then pure tcLibPtr else compileOrCacheGetSmall M K N
  if libPtr == 0 then return none
  let outBuf ← Tgrad.Runtime.Metal.metalAlloc (USize.ofNat outBytes)
  if outBuf == 0 then return none
  let d := Tgrad.Renderer.Metal.tcLaunchDims M N
  let fnName := if useTc then s!"matmul_tc_manual_{M}x{K}x{N}"
                else s!"matmul_scalar_{M}x{K}x{N}"
  let tx := if useTc then d.grid.x * d.threadgroup.x else M
  let ty := if useTc then d.grid.y * d.threadgroup.y else N
  let tz := if useTc then d.grid.z * d.threadgroup.z else 1
  let lx := if useTc then d.threadgroup.x else 1
  let ly := if useTc then d.threadgroup.y else 1
  let lz := if useTc then d.threadgroup.z else 1
  let rc ← Tgrad.Runtime.Metal.metalDispatch libPtr fnName
    #[outBuf, a.buffer.raw, b.buffer.raw]
    (USize.ofNat tx) (USize.ofNat ty) (USize.ofNat tz)
    (USize.ofNat lx) (USize.ofNat ly) (USize.ofNat lz)
  if rc != 0 then
    Tgrad.Runtime.Metal.metalFree outBuf (USize.ofNat outBytes)
    return none
  pure (some (Tgrad.Tensor.ofBuffer { raw := outBuf, size := outBytes } [M, N] .bfloat16_))


/-- Dtype of a tensor handle, as the code `dtypeOfCode` decodes.

    A query, not an operation: it does not grow with the op set. It
    exists so Python can learn a promoted result type instead of
    reimplementing `Dtype.lub`, which would put a second copy of the
    lattice on the other side of the FFI where nothing checks it. -/
@[export tgrad_tensor_dtype_lean]
def tensorDtype (h : UInt64) : IO UInt8 := do
  let some t ← TensorRegistry.get? h | return 255
  pure t.dtype.code

/-- Compiled-kernel cache for pointwise ops. Keyed on the operator, the
    output extent and a hash of BOTH rendered index expressions, so two
    different view chains never collide on one kernel. -/
initialize libCacheEw : IO.Ref (List (String × UInt64)) ← IO.mkRef []

private def compileOrCacheGetEw (op : BinOp) (outShape : List Nat)
    (aIdx bIdx : UOp) (aTy bTy outTy : Tgrad.Dtype) :
    IO (Option (UInt64 × String)) := do
  let tag := toString (String.hash (aIdx.renderIndexExpr ++ "|" ++ bIdx.renderIndexExpr))
  match Tgrad.Renderer.Metal.elementwiseKernelDeclRanked op outShape aIdx bIdx
          aTy bTy outTy tag with
  | none => return none
  | some decl =>
    let key := decl.name
    let cache ← libCacheEw.get
    let cached := cacheLookup key cache
    if cached != 0 then return some (cached, decl.name)
    let msl := Tgrad.Renderer.Metal.renderKernel decl
    if msl.isEmpty then return none
    let lib ← Tgrad.Runtime.Metal.metalCompile msl
    if lib == 0 then return none
    libCacheEw.modify (fun c => (key, lib) :: c)
    pure (some (lib, decl.name))

/-- Lower a pointwise binary graph. One thread per output element; the
    per-operand index expressions come from the `View` algebra, so an
    operand that is a view costs nothing extra. -/
private def runElementwise (op : BinOp) (a b : Tgrad.Tensor) :
    IO (Option Tgrad.Tensor) := do
  let aShape := a.shape
  let bShape := b.shape
  let rank := Nat.max aShape.length bShape.length
  if rank > 3 then return none
  -- Broadcast to a common extent. `View.expand` gives the smaller
  -- operand stride 0 on the stretched axis, so the kernel is unchanged
  -- and a broadcast operand costs no extra code — the same reason views
  -- are free here.
  let outShape := Tgrad.broadcast aShape bShape
  match ((Tgrad.Schedule.viewOfUOp a.uop).bind (fun v => v.padLeftToRank rank)).bind
          (fun v => v.expand outShape),
        ((Tgrad.Schedule.viewOfUOp b.uop).bind (fun v => v.padLeftToRank rank)).bind
          (fun v => v.expand outShape) with
  | some va, some vb => do
    let vars : List UOp := (List.range rank).map (fun i => .var s!"gidx{i}" .int32_)
    let aIdx := Tgrad.Schedule.View.indexOf va vars
    let bIdx := Tgrad.Schedule.View.indexOf vb vars
    -- The promoted result type comes from the dtype lattice. `Dtype.lub`
    -- has existed and been correct since L1 with no caller outside the
    -- JSON table emitters; this is its first load-bearing use.
    let outTy := Tgrad.Dtype.lub a.dtype b.dtype
    match (← compileOrCacheGetEw op outShape aIdx bIdx a.dtype b.dtype outTy) with
    | none => return none
    | some (lib, fnName) => do
      let outBytes := Tgrad.numel outShape * outTy.sizeBytes
      let outBuf ← Tgrad.Runtime.Metal.metalAlloc (USize.ofNat outBytes)
      if outBuf == 0 then return none
      let dx := (outShape[0]?).getD 1
      let dy := (outShape[1]?).getD 1
      let dz := (outShape[2]?).getD 1
      let rc ← Tgrad.Runtime.Metal.metalDispatch lib fnName
        #[outBuf, a.buffer.raw, b.buffer.raw]
        (USize.ofNat dx) (USize.ofNat dy) (USize.ofNat dz) 1 1 1
      if rc != 0 then
        Tgrad.Runtime.Metal.metalFree outBuf (USize.ofNat outBytes)
        return none
      pure (some (Tgrad.Tensor.ofBuffer { raw := outBuf, size := outBytes }
                  outShape outTy))
  | _, _ => return none


initialize libCacheRed : IO.Ref (List (String × UInt64)) ← IO.mkRef []

/-- Lower a reduction over one axis of a rank-2 operand, keepdim.

    Same structure as the pointwise path: the operand index comes from
    the `View` algebra, so reducing a transposed or sliced tensor needs
    no extra code. The loop variable is the contracted coordinate. -/
private def runReduce (op : BinOp) (a : Tgrad.Tensor) (axis : Nat) :
    IO (Option Tgrad.Tensor) := do
  let aShape := a.shape
  if aShape.length != 2 then return none
  match aShape[0]?, aShape[1]?, Tgrad.Schedule.viewOfUOp a.uop with
  | some rows, some cols, some va => do
    let gid : UOp := .var "gidx0" .int32_
    let rid : UOp := .var "ridx0" .int32_
    -- axis 1 contracts columns: operand coordinate is (out, loop).
    -- axis 0 contracts rows:    operand coordinate is (loop, out).
    let vars := if axis == 1 then [gid, rid] else [rid, gid]
    let operandIdx := Tgrad.Schedule.View.indexOf va vars
    let outTy := a.dtype
    let tag := toString (String.hash operandIdx.renderIndexExpr)
    match Tgrad.Renderer.Metal.reduceKernelDecl op rows cols axis
            operandIdx a.dtype outTy tag with
    | none => return none
    | some decl => do
      let cache ← libCacheRed.get
      let cached := cacheLookup decl.name cache
      let lib ← if cached != 0 then pure cached else do
        let msl := Tgrad.Renderer.Metal.renderKernel decl
        let l ← Tgrad.Runtime.Metal.metalCompile msl
        if l != 0 then libCacheRed.modify (fun c => (decl.name, l) :: c)
        pure l
      if lib == 0 then return none
      let outLen := if axis == 1 then rows else cols
      let outShape := if axis == 1 then [rows, 1] else [1, cols]
      let outBytes := outLen * outTy.sizeBytes
      let outBuf ← Tgrad.Runtime.Metal.metalAlloc (USize.ofNat outBytes)
      if outBuf == 0 then return none
      let rc ← Tgrad.Runtime.Metal.metalDispatch lib decl.name
        #[outBuf, a.buffer.raw] (USize.ofNat outLen) 1 1 1 1 1
      if rc != 0 then
        Tgrad.Runtime.Metal.metalFree outBuf (USize.ofNat outBytes)
        return none
      pure (some (Tgrad.Tensor.ofBuffer { raw := outBuf, size := outBytes }
                  outShape outTy))
  | _, _, _ => return none

initialize libCacheFused : IO.Ref (List (String × UInt64)) ← IO.mkRef []

/-- Lower `reduce(redOp) over the last axis of (a ewOp b)` as ONE kernel.

    This is the general contraction path, and the reason matmul does not
    need a matmul-shaped generator. `a @ b` is
    `reduce add (mul (expand a) (expand b))` over the contracted axis;
    lowering that literally would allocate the `M*N*K` product first —
    two gigabytes at 1024³ — so the elementwise op is instead consumed
    inside the accumulator loop and never materialised. That is what
    tinygrad's scheduler does, and the operand indices come from the
    `View` algebra, so the expands and the permute cost nothing.

    Slow but correct: one thread per output element, no tiling, no
    threadgroup memory, no WMMA. The specialised kernels remain the fast
    route for shapes that qualify; this is the path they are chosen
    *over*, and `scripts/differential_codegen.sh` is what keeps the two
    answering identically. -/
private def runFusedReduce (redOp ewOp : BinOp) (a b : Tgrad.Tensor) :
    IO (Option Tgrad.Tensor) := do
  let aShape := a.shape
  if aShape.length != 3 || b.shape != aShape then return none
  match aShape[0]?, aShape[1]?, aShape[2]?,
        Tgrad.Schedule.viewOfUOp a.uop, Tgrad.Schedule.viewOfUOp b.uop with
  | some d0, some d1, some d2, some va, some vb => do
    let vars : List UOp :=
      [.var "gidx0" .int32_, .var "gidx1" .int32_, .var "ridx0" .int32_]
    let aIdx := Tgrad.Schedule.View.indexOf va vars
    let bIdx := Tgrad.Schedule.View.indexOf vb vars
    let outTy := Tgrad.Dtype.lub a.dtype b.dtype
    let tag := toString (String.hash (aIdx.renderIndexExpr ++ "|" ++ bIdx.renderIndexExpr))
    match Tgrad.Renderer.Metal.fusedReduceKernelDecl redOp ewOp d0 d1 d2
            aIdx bIdx a.dtype b.dtype outTy tag with
    | none => return none
    | some decl => do
      let cache ← libCacheFused.get
      let cached := cacheLookup decl.name cache
      let lib ← if cached != 0 then pure cached else do
        let msl := Tgrad.Renderer.Metal.renderKernel decl
        let l ← Tgrad.Runtime.Metal.metalCompile msl
        if l != 0 then libCacheFused.modify (fun c => (decl.name, l) :: c)
        pure l
      if lib == 0 then return none
      let outBytes := d0 * d1 * outTy.sizeBytes
      let outBuf ← Tgrad.Runtime.Metal.metalAlloc (USize.ofNat outBytes)
      if outBuf == 0 then return none
      let rc ← Tgrad.Runtime.Metal.metalDispatch lib decl.name
        #[outBuf, a.buffer.raw, b.buffer.raw]
        (USize.ofNat d0) (USize.ofNat d1) 1 1 1 1
      if rc != 0 then
        Tgrad.Runtime.Metal.metalFree outBuf (USize.ofNat outBytes)
        return none
      pure (some (Tgrad.Tensor.ofBuffer { raw := outBuf, size := outBytes }
                  [d0, d1, 1] outTy))
  | _, _, _, _, _ => return none

/-- Materialise a graph and return a handle to the result.

    The accepted shape today is the matmul marker
    `reduce add (mul a b)`. That marker is built from existing UOp
    constructors rather than a bespoke `.matmul` node, so the step that
    replaces this special-case lowering with generic elementwise +
    reduce lowering changes only the *lowering* and not the graph.

    Honest limitation: the operands are not broadcast to a common shape,
    so the `mul` node is not yet a well-typed elementwise product. Making
    it well-typed is the generic-elementwise step's job. -/
@[export tgrad_realize_lean]
def realizeGraph (h : UInt64) : IO UInt64 := do
  let some t ← TensorRegistry.get? h | return 0
  match t.uop with
  | .reduce .add (.binop .mul aU bU _) _ =>
      -- Operand dtypes come from their own leaves only on the general
      -- path; the rank-2 route below is pinned to bf16 by its sentinel
      -- fixtures and must keep reading them as bf16.
      let a : Tgrad.Tensor := { uop := aU, dtype := .bfloat16_ }
      let b : Tgrad.Tensor := { uop := bU, dtype := .bfloat16_ }
      let aShape := a.shape
      -- Rank 3 is the general contraction written out as a graph. It
      -- goes to the fused kernel; the rank-2 marker below keeps the
      -- specialised WMMA route for the shapes that earn it.
      if aShape.length == 3 then
        match (← runFusedReduce .add .mul
                  { uop := aU, dtype := aU.dtypeOf }
                  { uop := bU, dtype := bU.dtypeOf }) with
        | none     => return 0
        | some out => TensorRegistry.register out
      else
      let bShape := b.shape
      let some M := aShape[0]? | return 0
      let some K := aShape[1]? | return 0
      let some N := bShape[1]? | return 0
      if !a.isBufferUop || !b.isBufferUop then
        match (← Tgrad.Pipeline.realizeView a b) with
        | .error _ => return 0
        | .ok out  => TensorRegistry.register out
      else
        match Tgrad.Renderer.Metal.ShapeSentinel.ofTriple M K N with
        | some s =>
            match (← Tgrad.Pipeline.realize a b s) with
            | .error _ => return 0
            | .ok out  => TensorRegistry.register out
        | none =>
            match (← runBufferMatmul a b M K N) with
            | none     => return 0
            | some out => TensorRegistry.register out
  | .reduce op bodyU axes =>
      -- A reduction whose body is not the matmul marker. The matmul arm
      -- above is matched first because it is the more specific pattern.
      let a : Tgrad.Tensor := { uop := bodyU, dtype := bodyU.dtypeOf }
      let axis := match axes with
                  | (.const _ (.i n)) :: _ => n.toNat
                  | _                      => 1
      match (← runReduce op a axis) with
      | none     => return 0
      | some out => TensorRegistry.register out
  | .binop op aU bU _ =>
      -- Pointwise. Reachable for every operator `elementwiseOpStr`
      -- accepts; the rest fail to build a kernel rather than emitting
      -- something plausible.
      -- Operand dtypes come from their own buffer leaves, so a mixed
      -- pair promotes rather than being silently read as bf16.
      let a : Tgrad.Tensor := { uop := aU, dtype := aU.dtypeOf }
      let b : Tgrad.Tensor := { uop := bU, dtype := bU.dtypeOf }
      match (← runElementwise op a b) with
      | none     => return 0
      | some out => TensorRegistry.register out
  | _ => return 0


-- ----------------------------------------------------------------------
-- Constant-fill creation (`Tensor.full` / `ones` / `zeros`).
--
-- Python normalises the calling convention (`_argfix`, kwargs) and
-- marshals signed shape + fill + dtype code. Lean classifies the signed shape
-- before any conversion to `Nat`; for `255`, Lean resolves the current runtime
-- floating default and then applies dtype admission (bf16/f32/i32 only), GPU
-- allocation, fill-kernel compile/dispatch, and registry insert.
-- A host-side `numpy.full` upload is deliberately not a path.
-- ----------------------------------------------------------------------

/-- Lean-owned admission result for a materialized creation shape. Negative
    dimensions take precedence over the current zero and rank limitations so
    every upstream negative-dimension case has one stable public reason. -/
inductive CreationShapeAdmission where
  | accepted
  | negativeDimension
  | zeroNonmaterializable
  | rankUnsupported
  deriving Repr, DecidableEq, BEq

namespace CreationShapeAdmission

/-- Stable boundary code. Python maps these reasons to public exception
    classes but does not reconstruct the shape predicate. -/
def code : CreationShapeAdmission → UInt8
  | .accepted => 0
  | .negativeDimension => 1
  | .zeroNonmaterializable => 2
  | .rankUnsupported => 3

end CreationShapeAdmission

/-- Classify signed dimensions before they can wrap through `size_t`/`USize`.
    Rank zero through three remains admitted; zero-sized materialization and
    rank above three remain explicitly outside the current implementation. -/
def creationShapeAdmission (shape : List Int) : CreationShapeAdmission :=
  if shape.any (fun d => d < 0) then .negativeDimension
  else if shape.any (fun d => d == 0) then .zeroNonmaterializable
  else if shape.length > 3 then .rankUnsupported
  else .accepted

example : creationShapeAdmission [(-3 : Int)] = .negativeDimension := by native_decide
example : creationShapeAdmission [(2 : Int), -3] = .negativeDimension := by native_decide
example : creationShapeAdmission [(2 : Int), -3, 0] = .negativeDimension := by native_decide
example : creationShapeAdmission [(2 : Int), 3] = .accepted := by native_decide
example : creationShapeAdmission [(2 : Int), 0] = .zeroNonmaterializable := by native_decide
example : creationShapeAdmission [(1 : Int), 2, 3, 4] = .rankUnsupported := by native_decide

/-- Allocation-free signed-shape query used by the public exception boundary. -/
@[export tgrad_creation_shape_admission_lean]
def creationShapeAdmissionQuery (shape : @& Array Int) : IO UInt8 :=
  pure (creationShapeAdmission shape.toList).code

/-- Primitive scalar category preserved by the Python authoring boundary.
    These codes intentionally match `Dtype.pythonTagDtypeWithDefaults?`; that
    existing function remains the single authority for primitive meaning. -/
inductive FillScalarTag where
  | bool
  | int
  | float
  deriving Repr, DecidableEq, BEq

namespace FillScalarTag

def ofCode? : UInt8 → Option FillScalarTag
  | 0 => some .bool
  | 1 => some .int
  | 2 => some .float
  | _ => none

def pythonTagCode : FillScalarTag → UInt8
  | .bool => 0
  | .int => 1
  | .float => 2

end FillScalarTag

/-- Pure creation resolver under an explicit default environment. Explicit
    dtype wins before fill-tag decoding. Inference delegates primitive meaning
    to the existing dtype authority, then current compute admission is applied
    exactly once and never substitutes a supported dtype. -/
def resolveCreationDtypeWithDefaults? (defaultInt defaultFloat : Tgrad.Dtype)
    (fillTag dtypeCode : UInt8) : Option Tgrad.Dtype :=
  let candidate :=
    if dtypeCode == 255 then
      match FillScalarTag.ofCode? fillTag with
      | some tag => Tgrad.Dtype.pythonTagDtypeWithDefaults?
          defaultInt defaultFloat tag.pythonTagCode
      | none => none
    else
      Tgrad.Dtype.ofCode? dtypeCode
  match candidate with
  | some d => if d.computeSupported then some d else none
  | none => none

example : resolveCreationDtypeWithDefaults? .int32_ .float32_ 1 255 =
    some .int32_ := by native_decide
example : resolveCreationDtypeWithDefaults? .int32_ .float32_ 2 255 =
    some .float32_ := by native_decide
example : resolveCreationDtypeWithDefaults? .int32_ .float32_ 0 255 =
    none := by native_decide
example : resolveCreationDtypeWithDefaults? .int32_ .float32_ 1 0 =
    some .bfloat16_ := by native_decide
example : resolveCreationDtypeWithDefaults? .int32_ .float32_ 254 255 =
    none := by native_decide
example : resolveCreationDtypeWithDefaults? .int64_ .float32_ 1 255 =
    none := by native_decide
example : resolveCreationDtypeWithDefaults? .int32_ .bfloat16_ 2 255 =
    some .bfloat16_ := by native_decide
example : resolveCreationDtypeWithDefaults? .int32_ .float16_ 2 255 =
    none := by native_decide
example : resolveCreationDtypeWithDefaults? .int32_ .float32_ 254 1 =
    some .float32_ := by native_decide

/-- Shared stateful resolver used by both the allocation-free query and the
    public allocation path. -/
def resolveCreationDtype (fillTag dtypeCode : UInt8) :
    IO (Option Tgrad.Dtype) := do
  let (defaultInt, defaultFloat) ← readDtypeDefaults
  pure (resolveCreationDtypeWithDefaults?
    defaultInt defaultFloat fillTag dtypeCode)

/-- Allocation-free observation of the exact dtype decision `tensorFull`
    will use. `255` means rejected after inference/override and compute
    admission; no allocation, compilation, or dispatch occurs. -/
@[export tgrad_creation_dtype_resolve_lean]
def creationDtypeResolveQuery (fillTag dtypeCode : UInt8) : IO UInt8 := do
  let some d ← resolveCreationDtype fillTag dtypeCode | return 255
  pure d.code

/-- The only source facts that full-like creation is allowed to inherit.
    Keeping this projection typed makes the pure resolver independent of the
    registry while the stateful resolver below remains the sole authority for
    obtaining it from a live tensor handle. -/
structure FullLikeSource where
  shape : List Nat
  dtype : Tgrad.Dtype
  deriving Repr, DecidableEq

/-- Shape and dtype selected for a full-like allocation. -/
structure FullLikeResolution where
  shape : List Nat
  dtype : Tgrad.Dtype
  deriving Repr, DecidableEq

/-- Pure full-like semantics under an explicit default environment.

    An omitted dtype inherits the source dtype exactly. An explicit dtype uses
    the same resolver and compute-admission relation as `tensorFull`. The fill
    tag is validated in both cases so an invalid scalar cannot be hidden by
    inheritance. -/
def resolveFullLikeWithDefaults? (source : Option FullLikeSource)
    (defaultInt defaultFloat : Tgrad.Dtype) (fillTag dtypeCode : UInt8) :
    Option FullLikeResolution := do
  let source ← source
  let _ ← FillScalarTag.ofCode? fillTag
  if creationShapeAdmission (source.shape.map Int.ofNat) != .accepted then
    none
  else
    let dtype? :=
      if dtypeCode == 255 then
        if source.dtype.computeSupported then some source.dtype else none
      else
        resolveCreationDtypeWithDefaults?
          defaultInt defaultFloat fillTag dtypeCode
    let dtype ← dtype?
    some { shape := source.shape, dtype := dtype }

private def fullLikeF32Source : FullLikeSource :=
  { shape := [2, 3], dtype := .float32_ }

private def fullLikeI32Source : FullLikeSource :=
  { shape := [2, 3], dtype := .int32_ }

example : resolveFullLikeWithDefaults? (some fullLikeF32Source)
    .int32_ .float32_ 1 255 =
    some { shape := [2, 3], dtype := .float32_ } := by native_decide
example : resolveFullLikeWithDefaults? (some fullLikeI32Source)
    .int32_ .float32_ 1 255 =
    some { shape := [2, 3], dtype := .int32_ } := by native_decide
example : resolveFullLikeWithDefaults? (some fullLikeF32Source)
    .int32_ .float32_ 1 3 =
    some { shape := [2, 3], dtype := .int32_ } := by native_decide
example : resolveFullLikeWithDefaults? (some fullLikeI32Source)
    .int32_ .float32_ 1 0 =
    some { shape := [2, 3], dtype := .bfloat16_ } := by native_decide
example : resolveFullLikeWithDefaults?
    (some { shape := [2, 3], dtype := Tgrad.Dtype.float16_ })
    .int32_ .float32_ 1 255 = none := by native_decide
example : resolveFullLikeWithDefaults? (some fullLikeF32Source)
    .int32_ .float32_ 1 253 = none := by native_decide
example : resolveFullLikeWithDefaults? none
    .int32_ .float32_ 1 255 = none := by native_decide

/-- Resolve full-like inheritance from a registered source tensor. Both the
    allocation-free query and the allocating entry point call this function,
    so the observer cannot drift from runtime behavior. -/
def resolveFullLike (sourceHandle : UInt64) (fillTag dtypeCode : UInt8) :
    IO (Option FullLikeResolution) := do
  let some source ← TensorRegistry.get? sourceHandle | return none
  let (defaultInt, defaultFloat) ← readDtypeDefaults
  pure (resolveFullLikeWithDefaults?
    (some { shape := source.shape, dtype := source.dtype })
    defaultInt defaultFloat fillTag dtypeCode)

/-- Allocation-free observation of full-like resolution. Query 0 returns the
    dtype code, query 1 the rank, and query 2 the dimension at `index`.
    Invalid handles, dtypes, tags, or queries return UInt64 max. -/
@[export tgrad_full_like_query_lean]
def fullLikeQuery (sourceHandle : UInt64) (fillTag dtypeCode query : UInt8)
    (index : USize) : IO UInt64 := do
  let some spec ← resolveFullLike sourceHandle fillTag dtypeCode |
    return (0 : UInt64) - 1
  pure (match query with
    | 0 => spec.dtype.code.toUInt64
    | 1 => UInt64.ofNat spec.shape.length
    | 2 => match spec.shape[index.toNat]? with
      | some dim => UInt64.ofNat dim
      | none => (0 : UInt64) - 1
    | _ => (0 : UInt64) - 1)


-- ----------------------------------------------------------------------
-- Arithmetic-progression creation (`Tensor.arange`).
-- ----------------------------------------------------------------------

/-- A scalar transported by the Python/C authoring boundary. Both payloads
    are retained so extending the Lean resolver to floating progressions does
    not require moving scalar interpretation into Python or changing the ABI.
    The current packet admits integral/bool tags and rejects floating tags in
    Lean. -/
structure RangeBoundaryScalar where
  intValue : Int
  floatValue : Float
  tag : UInt8
  deriving Repr

structure RangeResolution where
  start : Int
  stop : Int
  step : Int
  length : Nat
  last : Int
  dtype : Tgrad.Dtype
  computeAdmitted : Bool
  deriving Repr, DecidableEq

inductive RangeDecision where
  | accepted (spec : RangeResolution)
  | zeroStep
  | scalarUnsupported
  | invalidDtype
  | unrepresentable
  | lengthOverflow
  deriving Repr, DecidableEq

namespace RangeDecision

def code : RangeDecision → UInt8
  | .accepted _ => 0
  | .zeroStep => 1
  | .scalarUnsupported => 2
  | .invalidDtype => 3
  | .unrepresentable => 4
  | .lengthOverflow => 5

end RangeDecision

private def rangeIntegralValue? (scalar : RangeBoundaryScalar) : Option Int :=
  match FillScalarTag.ofCode? scalar.tag with
  | some .bool | some .int => some scalar.intValue
  | some .float | none => none

/-- Exact half-open length for an integral progression. All divisions occur
    on positive values, avoiding host-language signed-division conventions. -/
def integerRangeLength? (start stop step : Int) : Option Nat :=
  if step == 0 then none
  else if step > 0 then
    if stop ≤ start then some 0
    else some (((stop - start - 1) / step) + 1).toNat
  else
    if stop ≥ start then some 0
    else some (((start - stop - 1) / (-step)) + 1).toNat

private def belowDtypeMinimum (dtype : Tgrad.Dtype) (value : Int) : Bool :=
  let pow2 (bits : Nat) : Int := Int.ofNat (2 ^ bits)
  match dtype.rangeKind with
  | 0 | 2 => value < 0
  | 1 =>
    let bound := pow2 (dtype.bits - 1)
    value < -bound
  | 3 => false
  | _ => true

private def aboveDtypeMaximum (dtype : Tgrad.Dtype) (value : Int) : Bool :=
  let pow2 (bits : Nat) : Int := Int.ofNat (2 ^ bits)
  match dtype.rangeKind with
  | 0 => 1 < value
  | 1 => pow2 (dtype.bits - 1) - 1 < value
  | 2 => pow2 dtype.bits - 1 < value
  | 3 => false
  | _ => true

/-- Pinned tinygrad's exact one-sided representability predicate:
    `lo < dtype.min || dtype.max < hi`. In particular it does not require
    `lo ≤ max` or `min ≤ hi` for an empty wrong-direction range. -/
private def outsidePinnedRangeBounds (dtype : Tgrad.Dtype) (lo hi : Int) : Bool :=
  belowDtypeMinimum dtype lo || aboveDtypeMaximum dtype hi

private def rangeLengthFitsBoundary (length : Nat) : Bool :=
  length ≤ 18446744073709551615

private def rangeByteCountFits (length elementBytes : Nat) : Bool :=
  elementBytes != 0 &&
    length ≤ 18446744073709551615 / elementBytes

example : rangeByteCountFits 4611686018427387903 4 = true := by native_decide
example : rangeByteCountFits 4611686018427387904 4 = false := by native_decide

private def rangeByteCountFitsBoundary (spec : RangeResolution) : Bool :=
  rangeByteCountFits spec.length spec.dtype.sizeBytes

/-- Pure range semantics under explicit runtime defaults. Dtype selection is
    semantic first; `computeAdmitted` records the independent current backend
    relation and never substitutes another dtype. -/
def resolveRangeWithDefaults (startArg stopArg stepArg : RangeBoundaryScalar)
    (hasStop dtypeCode : UInt8) (defaultInt _defaultFloat : Tgrad.Dtype) :
    RangeDecision :=
  let startValue? := rangeIntegralValue? startArg
  let stopValue? := if hasStop == 0 then some 0 else rangeIntegralValue? stopArg
  let stepValue? := rangeIntegralValue? stepArg
  match startValue?, stopValue?, stepValue? with
  | some rawStart, some rawStop, some step =>
    let start := if hasStop == 0 then 0 else rawStart
    let stop := if hasStop == 0 then rawStart else rawStop
    let lo := if step > 0 then start else stop - step
    let hi := if step > 0 then stop - step else start
    let semantic? :=
      if dtypeCode == 255 then some defaultInt
      else Tgrad.Dtype.ofCode? dtypeCode
    match semantic? with
    | none => .invalidDtype
    | some initialDtype =>
      let dtype :=
        if dtypeCode == 255 && initialDtype == defaultInt &&
            outsidePinnedRangeBounds initialDtype lo hi then
          Tgrad.Dtype.int64_
        else initialDtype
      if outsidePinnedRangeBounds dtype lo hi then
        .unrepresentable
      else if step == 0 then
        .zeroStep
      else
        match integerRangeLength? start stop step with
        | none => .zeroStep
        | some length =>
          if !rangeLengthFitsBoundary length then .lengthOverflow else
          let last := if length == 0 then start
            else start + Int.ofNat (length - 1) * step
          .accepted {
            start := start, stop := stop, step := step, length := length,
            last := last, dtype := dtype,
            computeAdmitted := dtype.computeSupported }
  | _, _, _ => .scalarUnsupported

private def rangeInt (value : Int) (tag : UInt8 := 1) : RangeBoundaryScalar :=
  { intValue := value, floatValue := 0.0, tag := tag }

private def rangeMatches (decision : RangeDecision) (start stop step : Int)
    (length : Nat) (last : Int) (dtype : Tgrad.Dtype)
    (computeAdmitted : Bool) : Bool :=
  match decision with
  | .accepted spec =>
    spec.start == start && spec.stop == stop && spec.step == step &&
      spec.length == length && spec.last == last && spec.dtype == dtype &&
      spec.computeAdmitted == computeAdmitted
  | _ => false

example : rangeMatches
    (resolveRangeWithDefaults (rangeInt 10) (rangeInt 0) (rangeInt 1)
      0 255 .int32_ .float32_)
    0 10 1 10 9 .int32_ true = true := by native_decide
example : rangeMatches
    (resolveRangeWithDefaults (rangeInt 10) (rangeInt 5) (rangeInt (-3))
      1 255 .int32_ .float32_)
    10 5 (-3) 2 7 .int32_ true = true := by native_decide
example : rangeMatches
    (resolveRangeWithDefaults (rangeInt 5) (rangeInt 10) (rangeInt (-1))
      1 255 .int32_ .float32_)
    5 10 (-1) 0 5 .int32_ true = true := by native_decide
example : resolveRangeWithDefaults (rangeInt 0) (rangeInt 10) (rangeInt 0)
    1 255 .int32_ .float32_ = .zeroStep := by native_decide
example : resolveRangeWithDefaults
    (rangeInt 1099511627776) (rangeInt 0) (rangeInt 0)
    1 3 .int32_ .float32_ = .unrepresentable := by native_decide
example : resolveRangeWithDefaults
    (rangeInt 1099511627776) (rangeInt 0) (rangeInt 0)
    1 6 .int32_ .float32_ = .unrepresentable := by native_decide
example : rangeMatches
    (resolveRangeWithDefaults (rangeInt 2) (rangeInt 9) (rangeInt 2)
      1 1 .int32_ .float32_)
    2 9 2 4 8 .float32_ true = true := by native_decide
example : rangeMatches
    (resolveRangeWithDefaults (rangeInt 0) (rangeInt 4) (rangeInt 1)
      1 2 .int32_ .float32_)
    0 4 1 4 3 .float16_ false = true := by native_decide
example : rangeMatches
    (resolveRangeWithDefaults (rangeInt 0) (rangeInt 4) (rangeInt 1)
      1 255 .int64_ .float32_)
    0 4 1 4 3 .int64_ false = true := by native_decide
example : rangeMatches
    (resolveRangeWithDefaults
      (rangeInt (-2147483648)) (rangeInt 2147483648)
      (rangeInt 2147483647) 1 255 .int32_ .float32_)
    (-2147483648) 2147483648 2147483647 3 2147483646
      .int32_ true = true := by native_decide
example : rangeMatches
    (resolveRangeWithDefaults (rangeInt 125) (rangeInt 130) (rangeInt 3)
      1 6 .int32_ .float32_)
    125 130 3 2 128 .int8_ false = true := by native_decide
example : rangeMatches
    (resolveRangeWithDefaults (rangeInt 128) (rangeInt 0) (rangeInt 1)
      1 6 .int32_ .float32_)
    128 0 1 0 128 .int8_ false = true := by native_decide

/-- Shared stateful authority used by both the allocation-free query and the
    allocating path. -/
def resolveRange (start stop step : RangeBoundaryScalar)
    (hasStop dtypeCode : UInt8) : IO RangeDecision := do
  let (defaultInt, defaultFloat) ← readDtypeDefaults
  pure (resolveRangeWithDefaults start stop step hasStop dtypeCode
    defaultInt defaultFloat)

private def boundaryRangeScalar (intValue : Int) (floatValue : Float)
    (tag : UInt8) : RangeBoundaryScalar :=
  { intValue := intValue, floatValue := floatValue, tag := tag }

/-- Allocation-free observation of the exact stateful range decision.
    Query 0 is semantic dtype, 1 length, 2 compute admission, 3 normalized
    start bits, 4 step bits, 5 last bits, and 6 the decision code. -/
@[export tgrad_range_query_lean]
def rangeQuery (startInt stopInt stepInt : @& Int)
    (startFloat stopFloat stepFloat : Float)
    (startTag stopTag stepTag hasStop dtypeCode query : UInt8) : IO UInt64 := do
  let start := boundaryRangeScalar startInt startFloat startTag
  let stop := boundaryRangeScalar stopInt stopFloat stopTag
  let step := boundaryRangeScalar stepInt stepFloat stepTag
  let decision ← resolveRange start stop step hasStop dtypeCode
  if query == 6 then return decision.code.toUInt64
  match decision with
  | .accepted spec =>
    pure (match query with
      | 0 => spec.dtype.code.toUInt64
      | 1 => UInt64.ofNat spec.length
      | 2 => boolCode spec.computeAdmitted
      | 3 => if spec.start < 0 then
          (0 : UInt64) - UInt64.ofNat (-spec.start).toNat
        else UInt64.ofNat spec.start.toNat
      | 4 => if spec.step < 0 then
          (0 : UInt64) - UInt64.ofNat (-spec.step).toNat
        else UInt64.ofNat spec.step.toNat
      | 5 => if spec.last < 0 then
          (0 : UInt64) - UInt64.ofNat (-spec.last).toNat
        else UInt64.ofNat spec.last.toNat
      | _ => (0 : UInt64) - 1)
  | _ => pure ((0 : UInt64) - 1)

initialize libCacheFill : IO.Ref (List (String × UInt64)) ← IO.mkRef []

private def compileOrCacheGetFill (shape : List Nat) (ty : Tgrad.Dtype)
    (fill : Float) : IO (Option (UInt64 × String)) := do
  match Tgrad.Renderer.Metal.fillKernelDecl shape ty fill with
  | none => return none
  | some decl =>
    let key := decl.name
    let cache ← libCacheFill.get
    let cached := cacheLookup key cache
    if cached != 0 then return some (cached, decl.name)
    let msl := Tgrad.Renderer.Metal.renderKernel decl
    if msl.isEmpty then return none
    let lib ← Tgrad.Runtime.Metal.metalCompile msl
    if lib == 0 then return none
    libCacheFill.modify (fun c => (key, lib) :: c)
    pure (some (lib, decl.name))

private def allocateFilledTensor (dims : List Nat) (ty : Tgrad.Dtype)
    (fill : Float) : IO UInt64 := do
  match (← compileOrCacheGetFill dims ty fill) with
  | none => return 0
  | some (lib, fnName) => do
    let outBytes := Tgrad.numel dims * ty.sizeBytes
    let outBuf ← Tgrad.Runtime.Metal.metalAlloc (USize.ofNat outBytes)
    if outBuf == 0 then return 0
    let dx := (dims[0]?).getD 1
    let dy := (dims[1]?).getD 1
    let dz := (dims[2]?).getD 1
    let rc ← Tgrad.Runtime.Metal.metalDispatch lib fnName
      #[outBuf]
      (USize.ofNat dx) (USize.ofNat dy) (USize.ofNat dz) 1 1 1
    if rc != 0 then
      Tgrad.Runtime.Metal.metalFree outBuf (USize.ofNat outBytes)
      return 0
    TensorRegistry.register
      (Tgrad.Tensor.ofBuffer { raw := outBuf, size := outBytes } dims ty)

/-- Allocate a GPU buffer, fill it with a constant via the generated
    kernel, register the tensor, return its handle. `0` on any refusal
    (unsupported dtype, empty/oversized rank, compile/dispatch failure). -/
@[export tgrad_tensor_full_lean]
def tensorFull (shape : @& Array Int) (fill : Float)
    (fillTag dtypeCode : UInt8) :
    IO UInt64 := do
  let signedDims := shape.toList
  if creationShapeAdmission signedDims != .accepted then return 0
  let some ty ← resolveCreationDtype fillTag dtypeCode | return 0
  let dims : List Nat := signedDims.map Int.toNat
  allocateFilledTensor dims ty fill

/-- Lean-owned full-like creation. The source handle is resolved through the
    registry, shape/dtype inheritance is decided in Lean, and the existing fill
    implementation performs allocation, dispatch, and registration. -/
@[export tgrad_tensor_full_like_lean]
def tensorFullLike (sourceHandle : UInt64) (fill : Float)
    (fillTag dtypeCode : UInt8) : IO UInt64 := do
  let some spec ← resolveFullLike sourceHandle fillTag dtypeCode | return 0
  allocateFilledTensor spec.shape spec.dtype fill

initialize libCacheRange : IO.Ref (List (String × UInt64)) ← IO.mkRef []

private def compileOrCacheGetRange (spec : RangeResolution) :
    IO (Option (UInt64 × String)) := do
  match Tgrad.Renderer.Metal.rangeKernelDecl
      spec.length spec.dtype spec.start spec.step with
  | none => return none
  | some decl =>
    let key := decl.name
    let cache ← libCacheRange.get
    let cached := cacheLookup key cache
    if cached != 0 then return some (cached, decl.name)
    let msl := Tgrad.Renderer.Metal.renderKernel decl
    if msl.isEmpty then return none
    let lib ← Tgrad.Runtime.Metal.metalCompile msl
    if lib == 0 then return none
    libCacheRange.modify (fun c => (key, lib) :: c)
    pure (some (lib, decl.name))

private def allocateRangeTensor (spec : RangeResolution) : IO UInt64 := do
  if !spec.computeAdmitted || spec.length == 0 ||
      !rangeByteCountFitsBoundary spec then return 0
  match (← compileOrCacheGetRange spec) with
  | none => return 0
  | some (lib, fnName) => do
    let outBytes := spec.length * spec.dtype.sizeBytes
    let outBuf ← Tgrad.Runtime.Metal.metalAlloc (USize.ofNat outBytes)
    if outBuf == 0 then return 0
    let rc ← Tgrad.Runtime.Metal.metalDispatch lib fnName #[outBuf]
      (USize.ofNat spec.length) 1 1 1 1 1
    if rc != 0 then
      Tgrad.Runtime.Metal.metalFree outBuf (USize.ofNat outBytes)
      return 0
    TensorRegistry.register
      (Tgrad.Tensor.ofBuffer { raw := outBuf, size := outBytes }
        [spec.length] spec.dtype)

/-- Lean-owned arithmetic-progression allocation and dispatch. -/
@[export tgrad_tensor_arange_lean]
def tensorArange (startInt stopInt stepInt : @& Int)
    (startFloat stopFloat stepFloat : Float)
    (startTag stopTag stepTag hasStop dtypeCode : UInt8) : IO UInt64 := do
  let start := boundaryRangeScalar startInt startFloat startTag
  let stop := boundaryRangeScalar stopInt stopFloat stopTag
  let step := boundaryRangeScalar stepInt stepFloat stepTag
  let decision ← resolveRange start stop step hasStop dtypeCode
  match decision with
  | .accepted spec => allocateRangeTensor spec
  | _ => return 0

end Tgrad.PythonFFI
