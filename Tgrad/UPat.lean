import Tgrad.UOp

/-! # Tgrad.UPat — pattern AST + matcher

  Ports tinygrad's `UPat` (`tinygrad/uop/ops.py:1187-1205`) and phases
  02/03's typed pattern matcher into the unified `Tgrad` namespace.

  Phase 02/03 used a flat-tagged UOp (op kind + arg : Any + src list);
  Tgrad's UOp is per-constructor structured (see Tgrad.UOp). UPat
  bridges them by matching on `UOp.kind`, `UOp.dtypeOf`,
  `UOp.constArgInt`, and `UOp.children` projections rather than direct
  constructor destructuring. This keeps the unified UOp single while
  reusing phase 02/03's matcher contract verbatim.

  Negative example: `UPat.collectNames` is shape-preserving and
  doesn't introduce `partial def`s on the result type — so the rule's
  names-consistency check is decidable at gate-runner time.
-/

namespace Tgrad

mutual
inductive SrcPat where
  | none_
  | exact (xs : List UPat)
  deriving Inhabited
inductive UPat where
  | mk
      (op        : Option (List UOpKind))
      (dtype     : Option (List Dtype))
      (constInt  : Option Int)
      (name      : Option String)
      (src       : SrcPat)
      (strictLen : Bool)
  deriving Inhabited
end

namespace UPat

/-- Match-anything wildcard. -/
def wild : UPat := UPat.mk none none none none .none_ false

/-- Variable: matches anything (optionally constrained by dtype), binds
    to `name`. -/
def var (name : String) (dtypes : Option (List Dtype) := none) : UPat :=
  UPat.mk none dtypes none (some name) .none_ false

/-- CONST-only variable (any int / float / bool literal). -/
def cvar (name : String) (dtypes : Option (List Dtype) := none) : UPat :=
  UPat.mk (some [.CONST]) dtypes none (some name) .none_ false

/-- Specific integer CONST. -/
def constInt (n : Int) : UPat :=
  UPat.mk (some [.CONST]) none (some n) none .none_ false

/-- ALU binop with fixed positional arity. -/
def alu2 (kind : UOpKind) (a b : UPat) : UPat :=
  UPat.mk (some [kind]) none none none (.exact [a, b]) true

def add  (a b : UPat) : UPat := alu2 .ADD a b
def mul  (a b : UPat) : UPat := alu2 .MUL a b
def sub  (a b : UPat) : UPat := alu2 .SUB a b
def div  (a b : UPat) : UPat := alu2 .FLOORDIV a b
def modP (a b : UPat) : UPat := alu2 .FLOORMOD a b
def xorP (a b : UPat) : UPat := alu2 .XOR a b
def andP (a b : UPat) : UPat := alu2 .AND a b
def orP  (a b : UPat) : UPat := alu2 .OR  a b

end UPat

-- ============================================================================
-- Bindings + match
-- ============================================================================

abbrev Bindings := List (String × UOp)

def Bindings.lookup (b : Bindings) (n : String) : Option UOp :=
  match b with
  | []              => none
  | (k, v) :: rest  => if k == n then some v else Bindings.lookup rest n

def Bindings.insert (b : Bindings) (n : String) (u : UOp) : Bindings :=
  (n, u) :: b

mutual
/-- Try to match `p` against `u`, extending `store`. Returns `none` on
    failure or `some store'` on success. Ports tinygrad's `UPat.match`,
    restricted to v1 src-pattern support (none_ / exact). -/
partial def matchPat (p : UPat) (u : UOp) (store : Bindings) : Option Bindings := do
  let .mk pOp pDt pCi pName pSrc pStrict := p
  if let some kinds := pOp then
    if !kinds.contains u.kind then failure
  if let some dts := pDt then
    if !dts.contains u.dtypeOf then failure
  if let some n := pCi then
    match u.constArgInt with
    | some n' => if n != n' then failure
    | none    => failure
  let store ←
    (match pName with
     | none      => some store
     | some name =>
         match store.lookup name with
         | some prev => if prev.beq u then some store else none
         | none      => some (store.insert name u))
  let srcs := u.children
  match pSrc with
  | .none_      =>
      if pStrict ∧ srcs.length ≠ 0 then failure
      pure store
  | .exact xs   =>
      if pStrict ∧ srcs.length ≠ xs.length then failure
      if srcs.length < xs.length then failure
      matchSeq xs srcs store

partial def matchSeq (ps : List UPat) (us : List UOp) (s : Bindings) : Option Bindings :=
  match ps, us with
  | [], _              => some s
  | _, []              => none
  | p :: rest, u :: us =>
      match matchPat p u s with
      | some s' => matchSeq rest us s'
      | none    => none
end

-- ============================================================================
-- Names-consistency walk.
-- ============================================================================

/-- Collect every `name` mentioned in the pattern's `name`-fields,
    recursively. Used by `Rule.namesConsistent` (in GraphRewrite). -/
partial def UPat.collectNames (p : UPat) : List String :=
  let .mk _ _ _ pname psrc _ := p
  let here := match pname with | some n => [n] | none => []
  let subs := match psrc with
    | .none_      => []
    | .exact xs   => xs.foldl (fun acc q => acc ++ q.collectNames) []
  here ++ subs

end Tgrad
