import Tgrad.GraphRewrite

/-! # Tgrad.Rules.Symbolic — extended `symbolic_simple` subset.

  L1.a: 16-rule slice ported from theograd_phases/03_pattern_inventory.
  L9.a (P7): +6 rules — commutative duals + idempotents + subtraction
  identities. Each rule annotated with its source line in
  `tinygrad/uop/symbolic.py` so drift in tinygrad's rule body is
  immediately visible.

  Rules in tinygrad's `symbolic_simple` (~30 + propagate_invalid) that
  are saved for L9.b:
    - `x // -1 → -x` (needs Int negation rule)
    - bool `&` / `|` rules (needs DType.bool handling in matchPat)
    - `where` / cast / bitcast rules (needs ternary + cast UPat helpers)
    - `(x % c) + (x // c) * c = x` (fold_add_divmod_recombine)
    - threefry rules
    - propagate_invalid (whole sub-PatternMatcher)
-/

namespace Tgrad

namespace Rules

namespace Symbolic

-- ============================================================================
-- Rewrite helpers (each maps `Bindings → Option UOp`).
-- ============================================================================

/-- Identity replacement: yields the value bound to "x". -/
private def ruleId : Bindings → Option UOp := fun b => b.lookup "x"

/-- Rule 6: `(x ^ y) ^ y → x`. Same shape as `ruleId` but separate name
    documents which rule it implements (the name-reuse semantics in
    `matchPat`'s `pName` branch verifies the two "y"s bind to the same
    UOp). -/
private def ruleXorYY : Bindings → Option UOp := fun b => b.lookup "x"

/-- Yield `x.const_like(v)` — a CONST node with x's dtype and value v. -/
private def ruleConst (v : Int) : Bindings → Option UOp :=
  fun b => b.lookup "x" |>.map (fun x => UOp.constInt x.dtypeOf v)

/-- Generic binary const-fold helper. `combine` produces the folded value
    when both operands are int CONSTs in `c1`/`c2` bindings; otherwise
    falls through. -/
private def ruleCFInt (combine : Int → Int → Option Int) : Bindings → Option UOp :=
  fun b => do
    let c1 ← b.lookup "c1"; let c2 ← b.lookup "c2"
    match c1.constArgInt, c2.constArgInt with
    | some n1, some n2 =>
        match combine n1 n2 with
        | some n => some (UOp.constInt c1.dtypeOf n)
        | none   => none
    | _, _             => none

/-- Bitwise on Int via Nat (our fixture stays in [0, 256]). -/
private def intBitop (op : Nat → Nat → Nat) (n1 n2 : Int) : Option Int :=
  if n1 ≥ 0 ∧ n2 ≥ 0 then some (Int.ofNat (op n1.toNat n2.toNat)) else none

private def ruleCFAdd : Bindings → Option UOp :=
  ruleCFInt (fun a b => some (a + b))
private def ruleCFMul : Bindings → Option UOp :=
  ruleCFInt (fun a b => some (a * b))
private def ruleCFXor : Bindings → Option UOp :=
  ruleCFInt (fun a b => intBitop Nat.xor a b)
private def ruleCFAnd : Bindings → Option UOp :=
  ruleCFInt (fun a b => intBitop Nat.land a b)
private def ruleCFDiv : Bindings → Option UOp :=
  ruleCFInt (fun a b => if b == 0 then none else some (a / b))
private def ruleCFMod : Bindings → Option UOp :=
  ruleCFInt (fun a b => if b == 0 then none else some (a % b))

-- ============================================================================
-- Rule set — aligned byte-for-byte with `tinygrad/uop/symbolic.py`'s
-- `symbolic_simple` slice that fires on phase 03's 47-node DAG.
-- ============================================================================

/-- Extended `symbolic_simple` subset. L1.a (16 rules) + L9.a (+6 rules
    = 22 total). Each rule annotated with the source line of its
    `symbolic_simple` counterpart. -/
def ruleSet : PatternMatcher := [
  -- self-folding (symbolic.py:77-83)
  { pat := UPat.add  (UPat.var "x") (UPat.constInt 0),                  names := ["x"],      rewrite := ruleId },        -- L77
  { pat := UPat.mul  (UPat.var "x") (UPat.constInt 1),                  names := ["x"],      rewrite := ruleId },        -- L78
  { pat := UPat.xorP (UPat.var "x") (UPat.constInt 0),                  names := ["x"],      rewrite := ruleId },        -- L79
  { pat := UPat.div  (UPat.var "x") (UPat.var "x"),                     names := ["x"],      rewrite := ruleConst 1 },   -- L80
  { pat := UPat.div  (UPat.var "x") (UPat.constInt 1),                  names := ["x"],      rewrite := ruleId },        -- L81
  { pat := UPat.xorP (UPat.xorP (UPat.var "x") (UPat.var "y")) (UPat.var "y"),
    names := ["x", "y"], rewrite := ruleXorYY },                                                                          -- L83
  -- zero-folding (symbolic.py:99-101, 127)
  { pat := UPat.andP (UPat.var "x") (UPat.constInt 0),                  names := ["x"],      rewrite := ruleConst 0 },   -- L101
  { pat := UPat.modP (UPat.var "x") (UPat.var "x"),                     names := ["x"],      rewrite := ruleConst 0 },   -- L99
  { pat := UPat.xorP (UPat.var "x") (UPat.var "x"),                     names := ["x"],      rewrite := ruleConst 0 },   -- L100
  { pat := UPat.mul  (UPat.var "x") (UPat.constInt 0),                  names := ["x"],      rewrite := ruleConst 0 },   -- L127
  -- constant-folding (symbolic.py:110)
  { pat := UPat.add  (UPat.cvar "c1") (UPat.cvar "c2"),                 names := ["c1","c2"], rewrite := ruleCFAdd },     -- L110 ADD
  { pat := UPat.mul  (UPat.cvar "c1") (UPat.cvar "c2"),                 names := ["c1","c2"], rewrite := ruleCFMul },     -- L110 MUL
  { pat := UPat.xorP (UPat.cvar "c1") (UPat.cvar "c2"),                 names := ["c1","c2"], rewrite := ruleCFXor },     -- L110 XOR
  { pat := UPat.div  (UPat.cvar "c1") (UPat.cvar "c2"),                 names := ["c1","c2"], rewrite := ruleCFDiv },     -- L110 FLOORDIV
  { pat := UPat.modP (UPat.cvar "c1") (UPat.cvar "c2"),                 names := ["c1","c2"], rewrite := ruleCFMod },     -- L110 FLOORMOD
  { pat := UPat.andP (UPat.cvar "c1") (UPat.cvar "c2"),                 names := ["c1","c2"], rewrite := ruleCFAnd },     -- L110 AND
  -- L9.a additions — commutative duals + idempotents + subtraction identities.
  -- tinygrad's symbolic_simple doesn't list commutative duals explicitly
  -- (its UPat doesn't auto-permute non-list src) — these are equivalent
  -- to the LHS-binding versions above, useful for fixtures with
  -- captured 0+x / 1*x / 0|x patterns that the L1 set would miss.
  { pat := UPat.add  (UPat.constInt 0) (UPat.var "x"),                  names := ["x"],      rewrite := ruleId },        -- L9.a-1: 0 + x → x
  { pat := UPat.mul  (UPat.constInt 1) (UPat.var "x"),                  names := ["x"],      rewrite := ruleId },        -- L9.a-2: 1 * x → x
  { pat := UPat.sub  (UPat.var "x") (UPat.var "x"),                     names := ["x"],      rewrite := ruleConst 0 },   -- L9.a-3: x - x → 0
  { pat := UPat.sub  (UPat.var "x") (UPat.constInt 0),                  names := ["x"],      rewrite := ruleId },        -- L9.a-4: x - 0 → x
  { pat := UPat.orP  (UPat.var "x") (UPat.constInt 0),                  names := ["x"],      rewrite := ruleId },        -- L9.a-5: x | 0 → x (sympy.py:88-ish idempotent)
  { pat := UPat.orP  (UPat.var "x") (UPat.var "x"),                     names := ["x"],      rewrite := ruleId }         -- L9.a-6: x | x → x (sympy.py:93 idempotent)
]

end Symbolic

end Rules

end Tgrad
