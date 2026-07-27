import Tgrad.Runtime.Buffer
import Tgrad.Runtime.MetalProgram
import Tgrad.Runtime.Cache
import Tgrad.Pipeline
import Tgrad.Tensor
import Tgrad.Renderer.Elementwise
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
    libPtr ← Tgrad.Runtime.Metal.metalCompile msl
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
  let lib ← Tgrad.Runtime.Metal.metalCompile msl
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
  let lib ← Tgrad.Runtime.Metal.metalCompile msl
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
-- L13.F: TC-eligible non-sentinel general matmul. Production now routes
-- through the manual-load WMMA kernel from L13.F.STRICT.B/C. There is no
-- scalar fallback for shapes meeting the TC-eligibility predicate.
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
      let lib ← Tgrad.Runtime.Metal.metalCompile msl
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

/-- Stable FFI dtype encoding. Matches what `python/tgrad.py`
    passes for L14.A constructors. -/
def dtypeOfCode (code : UInt8) : Tgrad.Dtype :=
  match code with
  | 0 => .bfloat16_
  | 1 => .float32_
  | 2 => .float16_
  | 3 => .int32_
  | _ => .bfloat16_  -- default; FFI callers should not exceed the table

/-- L14.A: construct a `Tensor` with `uop := .buffer h shape dtype`,
    register it, return the opaque handle.

    `shape` is an Array USize (arbitrary rank); `dtypeCode` is the
    stable FFI encoding (`dtypeOfCode`). The C trampoline marshals a
    `const size_t* shape + size_t ndim` into the Lean Array. -/
@[export tgrad_tensor_from_buffer_lean]
def tensorFromBuffer
    (bufHandle : UInt64) (shape : @& Array USize) (dtypeCode : UInt8) :
    IO UInt64 := do
  let dt   := dtypeOfCode dtypeCode
  let dims : List Nat := shape.toList.map USize.toNat
  let t    : Tgrad.Tensor := { uop := .buffer bufHandle dims dt, dtype := dt }
  TensorRegistry.register t

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
  let libPtr ← if tcOk then compileOrCacheGetTcManual M K N
               else compileOrCacheGetSmall M K N
  if libPtr == 0 then return none
  let outBuf ← Tgrad.Runtime.Metal.metalAlloc (USize.ofNat outBytes)
  if outBuf == 0 then return none
  let d := Tgrad.Renderer.Metal.tcLaunchDims M N
  let fnName := if tcOk then s!"matmul_tc_manual_{M}x{K}x{N}"
                else s!"matmul_scalar_{M}x{K}x{N}"
  let tx := if tcOk then d.grid.x * d.threadgroup.x else M
  let ty := if tcOk then d.grid.y * d.threadgroup.y else N
  let tz := if tcOk then d.grid.z * d.threadgroup.z else 1
  let lx := if tcOk then d.threadgroup.x else 1
  let ly := if tcOk then d.threadgroup.y else 1
  let lz := if tcOk then d.threadgroup.z else 1
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
  pure (match t.dtype with
        | .bfloat16_ => 0
        | .float32_  => 1
        | .float16_  => 2
        | .int32_    => 3
        | _          => 255)

/-- Compiled-kernel cache for pointwise ops. Keyed on the operator, the
    output extent and a hash of BOTH rendered index expressions, so two
    different view chains never collide on one kernel. -/
initialize libCacheEw : IO.Ref (List (String × UInt64)) ← IO.mkRef []

private def compileOrCacheGetEw (op : BinOp) (rows cols : Nat)
    (aIdx bIdx : UOp) (aTy bTy outTy : Tgrad.Dtype) :
    IO (Option (UInt64 × String)) := do
  let tag := toString (String.hash (aIdx.renderIndexExpr ++ "|" ++ bIdx.renderIndexExpr))
  match Tgrad.Renderer.Metal.elementwiseKernelDecl op rows cols aIdx bIdx
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
  if aShape != bShape then return none
  if aShape.length != 2 then return none
  match aShape[0]?, aShape[1]?,
        Tgrad.Schedule.viewOfUOp a.uop, Tgrad.Schedule.viewOfUOp b.uop with
  | some rows, some cols, some va, some vb => do
    let vars : List UOp := [.var "gidx0" .int32_, .var "gidx1" .int32_]
    let aIdx := Tgrad.Schedule.View.indexOf va vars
    let bIdx := Tgrad.Schedule.View.indexOf vb vars
    -- The promoted result type comes from the dtype lattice. `Dtype.lub`
    -- has existed and been correct since L1 with no caller outside the
    -- JSON table emitters; this is its first load-bearing use.
    let outTy := Tgrad.Dtype.lub a.dtype b.dtype
    match (← compileOrCacheGetEw op rows cols aIdx bIdx a.dtype b.dtype outTy) with
    | none => return none
    | some (lib, fnName) => do
      let outBytes := rows * cols * outTy.sizeBytes
      let outBuf ← Tgrad.Runtime.Metal.metalAlloc (USize.ofNat outBytes)
      if outBuf == 0 then return none
      let rc ← Tgrad.Runtime.Metal.metalDispatch lib fnName
        #[outBuf, a.buffer.raw, b.buffer.raw]
        (USize.ofNat rows) (USize.ofNat cols) 1 1 1 1
      if rc != 0 then
        Tgrad.Runtime.Metal.metalFree outBuf (USize.ofNat outBytes)
        return none
      pure (some (Tgrad.Tensor.ofBuffer { raw := outBuf, size := outBytes }
                  [rows, cols] outTy))
  | _, _, _, _ => return none

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
      let a : Tgrad.Tensor := { uop := aU, dtype := .bfloat16_ }
      let b : Tgrad.Tensor := { uop := bU, dtype := .bfloat16_ }
      let aShape := a.shape
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


end Tgrad.PythonFFI
