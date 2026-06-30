import Tgrad.Dtype

/-! # Tgrad.UOp — the unified UOp inductive

  Load-bearing type per `01_design.md` §4 + README §3. Replaces the
  per-phase UOp variants (phases 01, 02, 03, 04, 07) with one
  constructor-per-op-kind inductive consumed by every downstream
  module.

  L1 extensions (added during the L1 lift):
    - BinOp gains: floordiv, floormod, cmplt, cmpne.
    - New constructor: defineVar (symbolic-rewrite leaf used by the
      symbolic_simple rule set lifted from phase 03).
    - New constructor: vconst (vectorized const) — placeholder hook
      for L3+; declared but not consumed at L1.
    - Projections: kind, dtypeOf, children, beq, constArgInt — these
      give the rewrite engine a uniform view of a unified UOp without
      requiring a separate flat-tagged type.

  `src` (in tinygrad's sense) is implicit in each constructor's
  arguments; the `arg : Any` of tinygrad's UOp is replaced by typed
  per-constructor fields. Indexing UOp by its result Dtype was
  considered (phase 01 negative-example block A) and rejected for v1:
  it multiplies the constructor explosion. Dtype is a field;
  per-constructor argument invariants are enforced as runtime
  predicates in `UOp/Validate.lean` (deferred).
-/

namespace Tgrad

/-- Compile-time constants embedded in UOp.const / vconst. -/
inductive ConstVal where
  | b (v : Bool)
  | i (v : Int)
  | f (v : Float)
  deriving BEq, Repr, Inhabited

/-- ALU binops covering both the bf16 matmul kernel and the symbolic
    rule set lifted from phase 03 (FLOORDIV / FLOORMOD / CMPLT / CMPNE
    are required by symbolic_simple even though the matmul kernel
    doesn't emit them). -/
inductive BinOp where
  | add | sub | mul
  | shl | shr | xor | andB | orB
  | floordiv | floormod | cmplt | cmpne
  deriving BEq, Repr, Inhabited, DecidableEq

/-- Workitem-ID kind: thread vs threadgroup position. -/
inductive SpecialKind where
  | local_     -- thread_position_in_threadgroup
  | global_    -- threadgroup_position_in_grid
  deriving BEq, Repr, Inhabited

/-- Range kind (LOOP / REDUCE / WARP / LOCAL / UPCAST / UNROLL).
    Subset shown at L1; expand as the pipeline grows. -/
inductive RangeKind where
  | loop
  | reduce
  | warp
  deriving BEq, Repr, Inhabited

def RangeKind.toUpperStr : RangeKind → String
  | .loop   => "LOOP"
  | .reduce => "REDUCE"
  | .warp   => "WARP"

def RangeKind.ofString : String → Option RangeKind
  | "LOOP"   => some .loop
  | "REDUCE" => some .reduce
  | "WARP"   => some .warp
  | _        => none

/-- Tensor-core tile dims. Concrete for Metal arm64 (8,8,8); a
    backend typeclass would generalize this. -/
structure TileDims where
  M : Nat
  N : Nat
  K : Nat
  deriving BEq, Repr, Inhabited

/-- L14.B.1: per-axis slice spec. `(start, stop, step)` mirrors numpy's
    slice arg. `step` defaults to 1 if absent; negative steps are
    rejected at rangeify time (out of scope at L14.B.1; tested by L14.B
    falsifiability D6). -/
structure Slice where
  start : Nat
  stop  : Nat
  step  : Nat
  deriving BEq, Repr, Inhabited

/-- The unified UOp.

    Constructors are per-op-kind. tinygrad's `src` is implicit in the
    fields (`binop`'s `a` and `b` are its src). tinygrad's `arg : Any`
    is replaced by typed per-constructor fields.

    Negative example (L1 gate): `UOp.index "str" "str"` must fail to
    typecheck (both fields are typed `UOp`). -/
inductive UOp : Type where
  /- IO / parameters / symbolic leaves -/
  | param     (idx : Nat) (dtype : Dtype) (size : Nat)
  | const     (dtype : Dtype) (value : ConstVal)
  | vconst    (dtype : Dtype) (values : List ConstVal)
  | defineVar (name : String) (dtype : Dtype) (lo : UOp) (hi : UOp)
  | special   (kind : SpecialKind) (axis : Char) (range : Nat)
  /- L14.A: tensor-leaf buffer (raw MTLBuffer + shape + dtype). The
     root of a Tensor's uop graph for a contiguous-buffer tensor. -/
  | buffer    (handle : UInt64) (shape : List Nat) (dtype : Dtype)
  /- L14.B.1: movement-op nodes — pure graph transforms over a source
     UOp. `Schedule.Rangeify` (wired in at L14.B.2) pushes these into
     LOAD-index expressions. -/
  | permute   (src : UOp) (axes : List Nat)
  | reshape   (src : UOp) (newShape : List Nat)
  | expand    (src : UOp) (newShape : List Nat)
  | slice     (src : UOp) (slices : List Slice)
  /- L14.B.2.a: variable reference by string name. Used by
     `renderIndexExpr` to emit kernel-local variable refs in
     index-arithmetic UOps (e.g. `gidx0`, `m`, `k_iter`). Distinct
     from `defineVar` which carries (lo, hi) bounds and represents
     a symbolic loop variable. -/
  | var       (name : String) (dtype : Dtype)
  /- pointer arithmetic -/
  | index     (buf : UOp) (offset : UOp)
  | load      (addr : UOp) (dtype : Dtype)
  | store     (addr : UOp) (value : UOp)
  /- ALU -/
  | binop     (op : BinOp) (a : UOp) (b : UOp) (dtype : Dtype)
  | cast      (target : Dtype) (e : UOp)
  | gep       (e : UOp) (lane : Nat)
  /- control flow / structure -/
  | range     (idx : Nat) (kind : RangeKind) (bound : UOp)
  | endR      (r : UOp)
  | reduce    (op : BinOp) (body : UOp) (axes : List UOp)
  | sink      (body : UOp)
  /- tensor-core -/
  | wmma      (name : String) (a : UOp) (b : UOp) (c : UOp)
              (dtypeIn : Dtype) (dtypeOut : Dtype) (tile : TileDims)
  deriving Inhabited

-- ============================================================================
-- Op-kind tag (used by UPat for matching against the unified UOp).
-- ============================================================================

/-- Op-kind tag. Mirrors the strings tinygrad uses in its `Ops` enum.
    Maps to a single Tgrad.UOp constructor. -/
inductive UOpKind where
  | DEFINE_VAR | CONST | VCONST
  | ADD | SUB | MUL | FLOORDIV | FLOORMOD
  | AND | OR | XOR | SHL | SHR | CMPLT | CMPNE
  | PARAM | SPECIAL | BUFFER | INDEX | LOAD | STORE | CAST | GEP
  | RANGE | END | REDUCE | SINK | WMMA
  | PERMUTE | RESHAPE | EXPAND | SLICE
  | VAR
  deriving BEq, Repr, Inhabited, DecidableEq

def UOpKind.toStr : UOpKind → String
  | .DEFINE_VAR => "DEFINE_VAR"
  | .CONST      => "CONST"
  | .VCONST     => "VCONST"
  | .ADD        => "ADD"
  | .SUB        => "SUB"
  | .MUL        => "MUL"
  | .FLOORDIV   => "FLOORDIV"
  | .FLOORMOD   => "FLOORMOD"
  | .AND        => "AND"
  | .OR         => "OR"
  | .XOR        => "XOR"
  | .SHL        => "SHL"
  | .SHR        => "SHR"
  | .CMPLT      => "CMPLT"
  | .CMPNE      => "CMPNE"
  | .PARAM      => "PARAM"
  | .SPECIAL    => "SPECIAL"
  | .BUFFER     => "BUFFER"
  | .INDEX      => "INDEX"
  | .LOAD       => "LOAD"
  | .STORE      => "STORE"
  | .CAST       => "CAST"
  | .GEP        => "GEP"
  | .RANGE      => "RANGE"
  | .END        => "END"
  | .REDUCE     => "REDUCE"
  | .SINK       => "SINK"
  | .WMMA       => "WMMA"
  | .PERMUTE    => "PERMUTE"
  | .RESHAPE    => "RESHAPE"
  | .EXPAND     => "EXPAND"
  | .SLICE      => "SLICE"
  | .VAR        => "VAR"

def UOpKind.ofString (s : String) : Option UOpKind :=
  match s with
  | "DEFINE_VAR" => some .DEFINE_VAR
  | "CONST"      => some .CONST
  | "VCONST"     => some .VCONST
  | "ADD"        => some .ADD
  | "SUB"        => some .SUB
  | "MUL"        => some .MUL
  | "FLOORDIV"   => some .FLOORDIV
  | "FLOORMOD"   => some .FLOORMOD
  | "AND"        => some .AND
  | "OR"         => some .OR
  | "XOR"        => some .XOR
  | "SHL"        => some .SHL
  | "SHR"        => some .SHR
  | "CMPLT"      => some .CMPLT
  | "CMPNE"      => some .CMPNE
  | "PARAM"      => some .PARAM
  | "SPECIAL"    => some .SPECIAL
  | "BUFFER"     => some .BUFFER
  | "INDEX"      => some .INDEX
  | "LOAD"       => some .LOAD
  | "STORE"      => some .STORE
  | "CAST"       => some .CAST
  | "GEP"        => some .GEP
  | "RANGE"      => some .RANGE
  | "END"        => some .END
  | "REDUCE"     => some .REDUCE
  | "SINK"       => some .SINK
  | "WMMA"       => some .WMMA
  | "PERMUTE"    => some .PERMUTE
  | "RESHAPE"    => some .RESHAPE
  | "EXPAND"     => some .EXPAND
  | "SLICE"      => some .SLICE
  | "VAR"        => some .VAR
  | _            => none

/-- Map a BinOp to the matching UOpKind tag. -/
def BinOp.toKind : BinOp → UOpKind
  | .add      => .ADD
  | .sub      => .SUB
  | .mul      => .MUL
  | .shl      => .SHL
  | .shr      => .SHR
  | .xor      => .XOR
  | .andB     => .AND
  | .orB      => .OR
  | .floordiv => .FLOORDIV
  | .floormod => .FLOORMOD
  | .cmplt    => .CMPLT
  | .cmpne    => .CMPNE

def BinOp.ofKind : UOpKind → Option BinOp
  | .ADD      => some .add
  | .SUB      => some .sub
  | .MUL      => some .mul
  | .SHL      => some .shl
  | .SHR      => some .shr
  | .XOR      => some .xor
  | .AND      => some .andB
  | .OR       => some .orB
  | .FLOORDIV => some .floordiv
  | .FLOORMOD => some .floormod
  | .CMPLT    => some .cmplt
  | .CMPNE    => some .cmpne
  | _         => none

-- ============================================================================
-- Projections — UPat / GraphRewrite consume these to view a unified UOp
-- as the flat (kind, dtype, arg, src) shape phase 02/03's engine expects.
-- ============================================================================

namespace UOp

/-- Op-kind of a UOp. -/
def kind : UOp → UOpKind
  | .param _ _ _         => .PARAM
  | .const _ _           => .CONST
  | .vconst _ _          => .VCONST
  | .defineVar _ _ _ _   => .DEFINE_VAR
  | .special _ _ _       => .SPECIAL
  | .buffer _ _ _        => .BUFFER
  | .index _ _           => .INDEX
  | .load _ _            => .LOAD
  | .store _ _           => .STORE
  | .binop op _ _ _      => op.toKind
  | .cast _ _            => .CAST
  | .gep _ _             => .GEP
  | .range _ _ _         => .RANGE
  | .endR _              => .END
  | .reduce _ _ _        => .REDUCE
  | .sink _              => .SINK
  | .wmma _ _ _ _ _ _ _  => .WMMA
  | .permute _ _         => .PERMUTE
  | .reshape _ _         => .RESHAPE
  | .expand _ _          => .EXPAND
  | .slice _ _           => .SLICE
  | .var _ _             => .VAR

/-- Result dtype of a UOp. For nodes whose dtype is implicit (e.g.
    range / endR / sink / store), we return a sentinel `.void`-like
    value; symbolic_simple's UPat patterns never gate on these. We
    map them to `.weakint_` (the matchless "any-int" tag tinygrad
    treats similarly). -/
def dtypeOf : UOp → Dtype
  | .param _ d _         => d
  | .const d _           => d
  | .vconst d _          => d
  | .defineVar _ d _ _   => d
  | .special _ _ _       => .int32_
  | .buffer _ _ d        => d
  | .index _ _           => .int32_  -- pointer-arith; treat as int32
  | .load _ d            => d
  | .store _ _           => .void_   -- sentinel (no result)
  | .binop _ _ _ d       => d
  | .cast d _            => d
  | .gep e _             => e.dtypeOf
  | .range _ _ _         => .weakint_
  | .endR _              => .void_   -- sentinel
  | .reduce _ b _        => b.dtypeOf
  | .sink _              => .void_   -- sentinel
  | .wmma _ _ _ _ _ d _  => d
  /- L14.B.1: movement ops inherit their source's dtype. -/
  | .permute src _       => src.dtypeOf
  | .reshape src _       => src.dtypeOf
  | .expand src _        => src.dtypeOf
  | .slice src _         => src.dtypeOf
  /- L14.B.2.a: var ref carries its own dtype. -/
  | .var _ d             => d

/-- Children of a UOp (sub-tree edges). Matches the post-order
    `src : List UOp` view phase 02/03 used. Order is significant:
    symbolic-simple matches by positional arity. -/
def children : UOp → List UOp
  | .param _ _ _              => []
  | .const _ _                => []
  | .vconst _ _               => []
  | .defineVar _ _ lo hi      => [lo, hi]
  | .special _ _ _            => []
  | .buffer _ _ _             => []
  | .index buf off            => [buf, off]
  | .load addr _              => [addr]
  | .store addr v             => [addr, v]
  | .binop _ a b _            => [a, b]
  | .cast _ e                 => [e]
  | .gep e _                  => [e]
  | .range _ _ b              => [b]
  | .endR r                   => [r]
  | .reduce _ body axes       => body :: axes
  | .sink body                => [body]
  | .wmma _ a b c _ _ _       => [a, b, c]
  /- L14.B.1: movement ops have a single source child. -/
  | .permute src _            => [src]
  | .reshape src _            => [src]
  | .expand src _             => [src]
  | .slice src _              => [src]
  /- L14.B.2.a: var ref is a leaf — no children. -/
  | .var _ _                  => []

/-- Const-int extraction. Used by UPat to match `constInt n`. Returns
    `none` for non-CONST nodes or non-int values. -/
def constArgInt : UOp → Option Int
  | .const _ (.i n) => some n
  | _               => none

/-- DEFINE_VAR's name (for round-tripping the symbolic fixture). -/
def defineVarName : UOp → Option String
  | .defineVar n _ _ _ => some n
  | _                  => none

/-- Structural equality. `partial` because the unified UOp is mutually
    recursive in lists (`reduce`, `wmma`, etc.). -/
partial def beq : UOp → UOp → Bool
  | u1, u2 =>
    u1.kind == u2.kind ∧ u1.dtypeOf == u2.dtypeOf ∧
    childrenEq u1.children u2.children ∧
    constArgsEqual u1 u2
where
  /-- Lists of UOps compare by length-then-element. -/
  childrenEq : List UOp → List UOp → Bool
    | [], []            => true
    | [], _             => false
    | _, []             => false
    | a :: as, b :: bs  => a.beq b ∧ childrenEq as bs
  /-- Embedded args (non-UOp scalar fields) compare per-constructor. -/
  constArgsEqual : UOp → UOp → Bool
    | .const _ v1,           .const _ v2           => v1 == v2
    | .vconst _ vs1,         .vconst _ vs2         => vs1 == vs2
    | .defineVar n1 _ _ _,   .defineVar n2 _ _ _   => n1 == n2
    | .param i1 _ s1,        .param i2 _ s2        => i1 == i2 ∧ s1 == s2
    | .special k1 a1 r1,     .special k2 a2 r2     => k1 == k2 ∧ a1 == a2 ∧ r1 == r2
    | .buffer h1 s1 d1,      .buffer h2 s2 d2      => h1 == h2 ∧ s1 == s2 ∧ d1 == d2
    | .range i1 k1 _,        .range i2 k2 _        => i1 == i2 ∧ k1 == k2
    | .gep _ l1,             .gep _ l2             => l1 == l2
    | .wmma n1 _ _ _ _ _ t1, .wmma n2 _ _ _ _ _ t2 => n1 == n2 ∧ t1 == t2
    | .permute _ a1,         .permute _ a2         => a1 == a2
    | .reshape _ s1,         .reshape _ s2         => s1 == s2
    | .expand _ s1,          .expand _ s2          => s1 == s2
    | .slice _ s1,           .slice _ s2           => s1 == s2
    | .var n1 d1,            .var n2 d2            => n1 == n2 ∧ d1 == d2
    | _,                     _                     => true  -- structural-only

instance : BEq UOp := ⟨UOp.beq⟩

/-- Build a CONST node with a given dtype + Int arg. Matches phase 03's
    `UOp.constInt`. -/
def constInt (d : Dtype) (n : Int) : UOp :=
  .const d (.i n)

/-- L14.B.2.c: count movement-op nodes in a UOp tree.
    Used by `RangeifyTrace.maybeEmit` to record whether rangeify saw
    nontrivial movement chains. -/
partial def countMovementNodes : UOp → Nat
  | .permute src _ => 1 + countMovementNodes src
  | .reshape src _ => 1 + countMovementNodes src
  | .expand src _  => 1 + countMovementNodes src
  | .slice src _   => 1 + countMovementNodes src
  | u => u.children.foldl (fun acc c => acc + countMovementNodes c) 0

/-- L14.B.2.c: count `.load` nodes (indexed LOADs) in a UOp tree.
    Used by `RangeifyTrace.maybeEmit` to record whether rangeify
    produced indexed loads. -/
partial def countIndexedLoadStore : UOp → Nat
  | .load _ _ => 1
  | u => u.children.foldl (fun acc c => acc + countIndexedLoadStore c) 0

/-- L14.B.2.a: render an index-arithmetic UOp tree as a parenthesised
    C-style expression string. Used by `Stmt.loadIndexed` /
    `Stmt.storeIndexed` to emit `buf[idx_expr]` LOAD/STORE statements
    from rangeify's output.

    Pure (no IO). Handles:
      - `.var name _`          → `name`
      - `.const _ (.i n)`      → `toString n`
      - `.binop op a b _`      → `({a} OP {b})` for ADD/SUB/MUL/FLOORDIV/FLOORMOD

    Panics on non-arithmetic UOp kinds (LOAD / CAST / matmul-sink / etc.)
    so that a misused codegen surface is caught at runtime rather than
    silently rendered as garbage. L14.B.2.b's kernel refactors are the
    only legitimate callers; L14.B.2.c wires the rangeify-produced
    index UOps through this. -/
partial def renderIndexExpr : UOp → String
  | .var name _              => name
  | .const _ (.i n)          => toString n
  | .binop .add a b _        => s!"({renderIndexExpr a}+{renderIndexExpr b})"
  | .binop .sub a b _        => s!"({renderIndexExpr a}-{renderIndexExpr b})"
  | .binop .mul a b _        => s!"({renderIndexExpr a}*{renderIndexExpr b})"
  | .binop .floordiv a b _   => s!"({renderIndexExpr a}/{renderIndexExpr b})"
  | .binop .floormod a b _   => s!"({renderIndexExpr a}%{renderIndexExpr b})"
  | u =>
      panic! s!"UOp.renderIndexExpr: non-index UOp kind {u.kind.toStr}"

end UOp

end Tgrad
