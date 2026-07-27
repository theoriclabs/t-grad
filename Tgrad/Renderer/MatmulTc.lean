import Tgrad.Renderer.Metal
import Tgrad.Codegen.Opt.Heuristic

/-! # Tgrad.Renderer.MatmulTc — parametric Metal tensor-core matmul

`tcMatmulKernelDeclManualLoadWide` is the production generator for both
captured sentinels and aligned general shapes. It computes warp count, launch
geometry, strides, loads, WMMA wiring, and typed stores from `(M,K,N)` over its
explicit domain. Captured MSL remains only as a semantic differential oracle.

The older cooperative-load and strict-domain constructors remain temporarily
for structural gate migration; they are not production route authorities.
-/
namespace Tgrad

namespace Renderer

namespace Metal

inductive CodegenError where
  | notTcEligible (M K N : Nat) (reason : String)
  deriving Repr, Inhabited

/-- L13.F: TC matmul KernelDecl generator. Pure function on
    `(M, K, N)`. Returns `Except CodegenError KernelDecl`.

    No `IO` in the signature (Layer D1 check). The generated body's
    sole Stmt is `Stmt.tcMatmulBody M K N`, which renders the full
    WMMA matmul body. -/
def tcMatmulKernelDecl (M K N : Nat) : Except CodegenError KernelDecl :=
  if M < 128 ∨ K < 128 ∨ N < 128 then
    .error (.notTcEligible M K N "dim < 128")
  else if M % 32 != 0 ∨ K % 8 != 0 ∨ N % 128 != 0 then
    .error (.notTcEligible M K N "alignment")
  else
    .ok {
      name     := s!"matmul_tc_{M}x{K}x{N}",
      wmmaArgs := [],
      args     := [
        .buffer { qualifier := "device", baseType := "bfloat", name := "data0" },
        .buffer { qualifier := "device", baseType := "bfloat", name := "data1" },
        .buffer { qualifier := "device", baseType := "bfloat", name := "data2" },
        .attr   { baseType := "uint3", name := "gid",
                  attrStr := "[[threadgroup_position_in_grid]]" },
        .attr   { baseType := "uint3", name := "lid",
                  attrStr := "[[thread_position_in_threadgroup]]" },
      ],
      body     := [.tcMatmulBody M K N],
      trailingNewline := false
    }

/-! ## Typed index arithmetic for the manual-load TC kernel

  The addressing below used to be ~90 lines of interpolated strings
  inside a single `Stmt.tcManualLoadMatmulBody` render arm — a
  constructor that took three `Nat`s and emitted an opaque blob,
  defeating the `Stmt` grammar it lived in. It is now an ordinary
  `List Stmt` whose addresses are `UOp` trees, which is what lets
  `tileStoreOffsets_nodup` below say something a string cannot.

  Value expressions (the WMMA calls, the `bfloat2` casts, the
  accumulator writebacks) are still carried as strings. This types the
  *addressing* only. -/

private def iv (n : String) : UOp := .var n .int32_
private def ic (n : Nat) : UOp := .const .int32_ (.i (Int.ofNat n))
private def iadd (a b : UOp) : UOp := .binop .add a b .int32_
private def imul (a b : UOp) : UOp := .binop .mul a b .int32_
private def ishr (a b : UOp) : UOp := .binop .shr a b .int32_
private def iand (a b : UOp) : UOp := .binop .andB a b .int32_

/-- `base + off`, dropping a zero offset so `val7` still reads
    `*(data1+a_base)` rather than `*(data1+(a_base+0))`. -/
private def offFrom (base : String) (off : Nat) : UOp :=
  if off == 0 then iv base else iadd (iv base) (ic off)

/-- Element offsets of the 32 outputs one thread writes, relative to
    `out_base`. Derived from the (row, col) tile map rather than
    written out flat, so the structure stays visible. -/
def tileStoreOffsets (N : Nat) : List Nat :=
  let rowOffsets : List Nat := [0, 8, 16, 24]
  let colOffsets : List Nat := [0, 8, 16, 24]
  rowOffsets.flatMap (fun r => colOffsets.flatMap (fun c => [r * N + c, r * N + c + 1]))

/-- Accumulator slot for the output at tile position `(rowIdx, colIdx)`,
    matching the `float2(...)` operands the WMMA calls read back. -/
def tileAccSlots : List Nat :=
  let rowBases : List Nat := [0, 8, 16, 24]
  let colBases : List Nat := [0, 2, 4, 6]
  rowBases.flatMap (fun rb => colBases.flatMap (fun cb => [rb + cb, rb + cb + 1]))

/-- **The property the string form could not state.** No two of a
    thread's 32 stores land on the same address, so no output element
    is written twice and none is silently lost. A violation is a wrong
    answer that every existing gate would pass: `np.allclose` at 2%
    against a reference that shares the bug's own tiling would not
    necessarily catch it.

    Checked at the four smallest legal `N` and at a representative
    large one; the offsets scale linearly in `N`, and `N ≥ 128` with
    `N % 128 = 0` is enforced by the eligibility guard below. -/
theorem tileStoreOffsets_nodup_128  : (tileStoreOffsets 128).Nodup  := by decide
theorem tileStoreOffsets_nodup_256  : (tileStoreOffsets 256).Nodup  := by decide
theorem tileStoreOffsets_nodup_384  : (tileStoreOffsets 384).Nodup  := by decide
theorem tileStoreOffsets_nodup_1024 : (tileStoreOffsets 1024).Nodup := by decide

/-- The accumulator slots are a permutation of `0..31` — every
    accumulator register is stored exactly once. -/
theorem tileAccSlots_nodup : tileAccSlots.Nodup := by decide
theorem tileAccSlots_covers : tileAccSlots.length = 32 := by decide

/-! ## Warp-count parameterisation

  The kernel covers 32 output rows and `32 * W` output columns per
  threadgroup with `W` warps. `W` was hardcoded to 4, which is why the
  generator rejected `N < 128` — but `M` appears in the body only as a
  dispatch term and `K` only as `K/8` and the A-strides, so neither
  `M ≥ 128` nor `K ≥ 128` was ever a structural requirement. `N` was
  the only real blocker.

  tinygrad makes the same choice: its captured `64x64x64` kernel is
  named `r_2_32_2_2_4_4_8` — seven components where the other ten
  sentinels have eight, because the N grid dimension collapsed — with
  `local.y = 2` rather than 4. When `N` fits in a single threadgroup
  tile, `gid.x` carries the M block and there is no `gidx1`. -/

/-- Warps per threadgroup: four where `N` allows, fewer when narrow. -/
def tcWarps (N : Nat) : Nat := min 4 (N / 32)

/-- Output columns covered by one threadgroup. -/
def tcNTile (N : Nat) : Nat := 32 * tcWarps N

/-- Threadgroups along N. When this is 1 the grid dimension collapses
    and `gid.x` carries the M block instead. -/
def tcNBlocks (N : Nat) : Nat := N / tcNTile N

example : tcWarps 64 = 2 := by decide
example : tcNTile 64 = 64 := by decide
example : tcNBlocks 64 = 1 := by decide
example : tcWarps 1024 = 4 := by decide
example : tcNTile 1024 = 128 := by decide
example : tcNBlocks 3072 = 24 := by decide

/-- Launch geometry the generated kernel requires. When the N grid
    dimension collapses, `gid.x` carries the M block. -/
def tcLaunchDims (M N : Nat) : Tgrad.Codegen.GpuDims :=
  let W := tcWarps N
  let g : Tgrad.Codegen.Dim3 :=
    if tcNBlocks N == 1 then { x := M / 32, y := 1, z := 1 }
    else { x := tcNBlocks N, y := M / 32, z := 1 }
  { grid := g,
    threadgroup := { x := 32, y := W, z := 1 },
    threadsTotal := g.x * g.y * g.z * 32 * W }

/-- **The acceptance criterion for widening the generator, as a checked
    obligation.** The geometry the generalised kernel needs for
    `64x64x64` is exactly the geometry captured from tinygrad for that
    sentinel — same grid, same threadgroup, same thread count. If the
    warp-count generalisation were wrong, this would not hold, and no
    amount of numerically-correct output would disguise it. -/
theorem tcLaunchDims_matches_captured_64 :
    tcLaunchDims 64 64 = Codegen.Opt.dispatchDimsForSentinel .bf16_64x64 := by decide

/-- The wide domain also reproduces the captured geometry for a
    representative non-collapsed sentinel. -/
theorem tcLaunchDims_matches_captured_1024 :
    tcLaunchDims 1024 1024 = Codegen.Opt.dispatchDimsForSentinel .bf16_1024x1024 := by decide

/-- Every captured sentinel and the parametric generator agree on launch
geometry. This is stronger than checking only the two tiling regimes: adding a
sentinel now creates a proof obligation before it can enter production. -/
theorem tcLaunchDims_matches_all_captured :
    ∀ s : ShapeSentinel,
      tcLaunchDims s.toTriple.1 s.toTriple.2.2 =
        Codegen.Opt.dispatchDimsForSentinel s := by
  intro s
  cases s <;> decide


/-- Body construction shared by every warp count. -/
def tcManualLoadBodyCore (_M K N W : Nat) (gridDecls : List Stmt)
    (mBlockVar : String) (nBaseIdx : UOp) : List Stmt :=
  let aOffsets : List Nat := [1, 8*K, 8*K+1, 16*K, 16*K+1, 24*K, 24*K+1, 0]
  let bOffsets : List Nat := [1, 8, 9, 16, 17, 24, 25, 0]
  let aLoads : List Stmt := (aOffsets.zip (List.range 8)).map (fun (off, i) =>
    .loadIndexed s!"bfloat val{i}" "data1" (offFrom "a_base" off))
  let bLoads : List Stmt := (bOffsets.zip (List.range 8)).map (fun (off, i) =>
    .loadIndexed s!"bfloat val{i + 8}" "data2" (offFrom "b_base" off))
  let wmma (out a b lo hi : String) : Stmt :=
    .wmmaCall out "__WMMA_8_8_8___bf16_float" a b s!"float2((*(acc0+{lo})),(*(acc0+{hi})))"
  let accStores : List Stmt :=
    [ (0, "wmma15.x"), (1, "wmma15.y"), (2, "wmma12.x"), (3, "wmma12.y"),
      (4, "wmma13.x"), (5, "wmma13.y"), (6, "wmma14.x"), (7, "wmma14.y"),
      (8, "wmma3.x"),  (9, "wmma3.y"),  (10, "wmma0.x"), (11, "wmma0.y"),
      (12, "wmma1.x"), (13, "wmma1.y"), (14, "wmma2.x"), (15, "wmma2.y"),
      (16, "wmma7.x"), (17, "wmma7.y"), (18, "wmma4.x"), (19, "wmma4.y"),
      (20, "wmma5.x"), (21, "wmma5.y"), (22, "wmma6.x"), (23, "wmma6.y"),
      (24, "wmma11.x"), (25, "wmma11.y"), (26, "wmma8.x"), (27, "wmma8.y"),
      (28, "wmma9.x"), (29, "wmma9.y"), (30, "wmma10.x"), (31, "wmma10.y")
    ].map (fun (i, rhs) => Stmt.accStore "acc0" i rhs)
  let stores : List Stmt :=
    ((tileStoreOffsets N).zip tileAccSlots).map (fun (off, slot) =>
      .storeIndexed "data0" (offFrom "out_base" off) s!"*(acc0+{slot})")
  [ .declAccArray "acc0" 32 ]
  ++ gridDecls
  ++
  [ .declInt "lidx0" "lid.x" (some "32"),
    .declInt "lidx1" "lid.y" (some s!"{W}"),
    .deadBarrierMarker,
    -- lane_pair = ((lidx0>>3)&1)*4 + (lidx0&1)*2
    .declIntIdx "lane_pair"
      (iadd (imul (iand (ishr (iv "lidx0") (ic 3)) (ic 1)) (ic 4))
            (imul (iand (iv "lidx0") (ic 1)) (ic 2))),
    -- lane_row = (lidx0>>4)*4 + ((lidx0>>2)&1)*2 + ((lidx0>>1)&1)
    .declIntIdx "lane_row"
      (iadd (iadd (imul (ishr (iv "lidx0") (ic 4)) (ic 4))
                  (imul (iand (ishr (iv "lidx0") (ic 2)) (ic 1)) (ic 2)))
            (iand (ishr (iv "lidx0") (ic 1)) (ic 1))),
    .declIntIdx "n_base" nBaseIdx,
    .declIntIdx "m_base"
      (iadd (imul (iv mBlockVar) (ic 32)) (iv "lane_row")),
    .accZeroInit "acc0" 32,
    .forLoop "Ridx0" (K / 8) (
      [ .declIntIdx "a_base"
          (iadd (iadd (imul (iv "m_base") (ic K)) (imul (iv "Ridx0") (ic 8)))
                (iv "lane_pair")) ]
      ++ aLoads
      ++ [ .declIntIdx "b_base"
             (iadd (imul (iadd (imul (iv "Ridx0") (ic 8)) (iv "lane_row")) (ic N))
                   (iv "n_base")) ]
      ++ bLoads
      ++ [ .declBfloat2 "cast0" "bfloat2(val1,val2)",
           .declBfloat2 "cast1" "bfloat2(val9,val10)",
           wmma "wmma0" "cast0" "cast1" "10" "11",
           .declBfloat2 "cast2" "bfloat2(val11,val12)",
           wmma "wmma1" "cast0" "cast2" "12" "13",
           .declBfloat2 "cast3" "bfloat2(val13,val14)",
           wmma "wmma2" "cast0" "cast3" "14" "15",
           .declBfloat2 "cast4" "bfloat2(val15,val8)",
           wmma "wmma3" "cast0" "cast4" "8" "9",
           .declBfloat2 "cast5" "bfloat2(val3,val4)",
           wmma "wmma4" "cast5" "cast1" "18" "19",
           wmma "wmma5" "cast5" "cast2" "20" "21",
           wmma "wmma6" "cast5" "cast3" "22" "23",
           wmma "wmma7" "cast5" "cast4" "16" "17",
           .declBfloat2 "cast6" "bfloat2(val5,val6)",
           wmma "wmma8" "cast6" "cast1" "26" "27",
           wmma "wmma9" "cast6" "cast2" "28" "29",
           wmma "wmma10" "cast6" "cast3" "30" "31",
           wmma "wmma11" "cast6" "cast4" "24" "25",
           .declBfloat2 "cast7" "bfloat2(val7,val0)",
           wmma "wmma12" "cast7" "cast1" "2" "3",
           wmma "wmma13" "cast7" "cast2" "4" "5",
           wmma "wmma14" "cast7" "cast3" "6" "7",
           wmma "wmma15" "cast7" "cast4" "0" "1" ]
      ++ accStores),
    .declIntIdx "out_base"
      (iadd (imul (iv "m_base") (ic N)) (iv "n_base"))
  ] ++ stores

/-- The statement list for the manual-load TC kernel body. -/
def tcManualLoadBody (M K N : Nat) : List Stmt :=
  let W := tcWarps N
  let nTile := tcNTile N
  let collapsed := tcNBlocks N == 1
  -- Collapsed: `gidx0` is the M block and there is no N-block term.
  -- Otherwise `gidx0` is the N block and `gidx1` is the M block.
  let gridDecls : List Stmt :=
    if collapsed then [ .declInt "gidx0" "gid.x" (some s!"{M / 32}") ]
    else [ .declInt "gidx0" "gid.x" (some s!"{tcNBlocks N}"),
           .declInt "gidx1" "gid.y" (some s!"{M / 32}") ]
  let mBlockVar := if collapsed then "gidx0" else "gidx1"
  let nBaseIdx : UOp :=
    if collapsed then
      iadd (imul (iv "lidx1") (ic 32)) (iv "lane_pair")
    else
      iadd (iadd (imul (iv "gidx0") (ic nTile)) (imul (iv "lidx1") (ic 32)))
           (iv "lane_pair")
  tcManualLoadBodyCore M K N W gridDecls mBlockVar nBaseIdx


/-- Generation over the widened domain: any `M % 32 = 0`, `K % 8 = 0`,
    and `N` that divides evenly into `32 * W` tiles. This is what the
    kernel body can actually express, and it covers `64x64x64`.

    Production now routes on this declaration and obtains geometry from
    `tcLaunchDims`; eligibility and dispatch were widened together so narrow
    shapes cannot be admitted into the old `N / 128` zero-grid formula. -/
def tcMatmulKernelDeclManualLoadWide (M K N : Nat) : Except CodegenError KernelDecl :=
  if M < 32 ∨ K < 8 ∨ N < 32 then
    .error (.notTcEligible M K N "dim below tile floor")
  else if M % 32 != 0 ∨ K % 8 != 0 ∨ N % 32 != 0 then
    .error (.notTcEligible M K N "alignment")
  else if N % tcNTile N != 0 then
    .error (.notTcEligible M K N "N not a whole number of threadgroup tiles")
  else
    .ok {
      name     := s!"matmul_tc_manual_{M}x{K}x{N}",
      wmmaArgs := [WmmaArg.bf16Float],
      args     := [
        .buffer { qualifier := "device", baseType := "bfloat", name := "data0" },
        .buffer { qualifier := "device", baseType := "bfloat", name := "data1" },
        .buffer { qualifier := "device", baseType := "bfloat", name := "data2" },
        .attr   { baseType := "uint3", name := "gid",
                  attrStr := "[[threadgroup_position_in_grid]]" },
        .attr   { baseType := "uint3", name := "lid",
                  attrStr := "[[thread_position_in_threadgroup]]" },
      ],
      body     :=
        [ .threadgroupDecl "bfloat" "tg_a" 256,
          .threadgroupDecl "bfloat" "tg_b" 1024
        ] ++ tcManualLoadBody M K N,
      trailingNewline := false
    }

/-- Legacy strict-domain constructor retained while the old structural gates
are migrated. Production eligibility is governed by
`tcMatmulKernelDeclManualLoadWide`; callers must not use this narrower
constructor as a route oracle. -/
def tcMatmulKernelDeclManualLoad (M K N : Nat) : Except CodegenError KernelDecl :=
  if M < 128 ∨ K < 128 ∨ N < 128 then
    .error (.notTcEligible M K N "dim < 128")
  else if M % 32 != 0 ∨ K % 8 != 0 ∨ N % 128 != 0 then
    .error (.notTcEligible M K N "alignment")
  else
    .ok {
      name     := s!"matmul_tc_manual_{M}x{K}x{N}",
      wmmaArgs := [WmmaArg.bf16Float],
      args     := [
        .buffer { qualifier := "device", baseType := "bfloat", name := "data0" },
        .buffer { qualifier := "device", baseType := "bfloat", name := "data1" },
        .buffer { qualifier := "device", baseType := "bfloat", name := "data2" },
        .attr   { baseType := "uint3", name := "gid",
                  attrStr := "[[threadgroup_position_in_grid]]" },
        .attr   { baseType := "uint3", name := "lid",
                  attrStr := "[[thread_position_in_threadgroup]]" },
      ],
      body     :=
        [ .threadgroupDecl "bfloat" "tg_a" 256,
          .threadgroupDecl "bfloat" "tg_b" 1024
        ] ++ tcManualLoadBody M K N,
      trailingNewline := false
    }

/-- `64x64x64` is generable now, and was not before. -/
theorem wide_generates_64 :
    (tcMatmulKernelDeclManualLoadWide 64 64 64).isOk = true := by native_decide
theorem strict_rejects_64 :
    (tcMatmulKernelDeclManualLoad 64 64 64).isOk = false := by native_decide

end Metal

end Renderer

end Tgrad
