import Tgrad.UPat

/-! # Tgrad.GraphRewrite — rule + matcher + bottom-up driver

  Ports phase 02/03's `Rule`, `PatternMatcher`, `graphRewriteBottomUp`,
  and the JSON round-trip helpers (toTree / toRecords / recordsToJson)
  into Tgrad's unified-UOp shape.

  The flat `UOpRecord` shape stays — it's the fixture's serialized
  form. `toTree` parses a record list into a Tgrad.UOp tree by
  dispatching on UOpKind + arg + dtype; `toRecords` walks a tree and
  emits the equivalent records (interning structurally-equal subtrees
  to match tinygrad's `UOpMetaClass` behaviour).
-/

namespace Tgrad

-- ============================================================================
-- Argument shape carried alongside a UOpRecord (phase 02/03 schema).
-- ============================================================================

/-- The arg-shape that appears in fixture JSON. Tgrad's unified UOp
    encodes most of these as typed constructor fields; UOpArg is the
    boundary representation used when parsing/emitting the fixture's
    `arg` object. The `tuple` variant covers phase-04 RANGE's
    `(axis_idx, "LOOP")` arg (and any future nested tuples). -/
inductive UOpArg where
  | none
  | bool  (b : Bool)
  | int   (n : Int)
  | float (f : Float)
  | str   (s : String)
  | tuple (xs : List UOpArg)
  deriving Inhabited, Repr

/-- Structural equality. `partial` because the inductive is
    self-recursive via List. -/
partial def UOpArg.beq : UOpArg → UOpArg → Bool
  | .none,     .none      => true
  | .bool a,   .bool b    => a == b
  | .int a,    .int b     => a == b
  | .float a,  .float b   => a == b
  | .str a,    .str b     => a == b
  | .tuple xs, .tuple ys  =>
      xs.length == ys.length ∧ (List.zip xs ys).all (fun p => UOpArg.beq p.1 p.2)
  | _, _ => false

instance : BEq UOpArg := ⟨UOpArg.beq⟩

/-- Pretty-print a UOpArg as Python-json-dumps(indent=2)-byte-equivalent
    JSON, with the leading line's `{` at column 0 of the line where the
    arg appears (callers handle outer indentation). `indent` is the
    column at which `}` should close — i.e. matching the position of
    the `"arg":` key in the enclosing UOpRecord (= column 4 in our
    output). -/
partial def UOpArg.toJsonAt (indent : Nat) : UOpArg → String
  | .none     => "null"
  | .bool b   =>
      let inner := "".pushn ' ' (indent + 2)
      let close := "".pushn ' ' indent
      "{\n" ++ inner ++ "\"kind\": \"bool\",\n" ++ inner ++
      "\"v\": " ++ (if b then "true" else "false") ++ "\n" ++ close ++ "}"
  | .int n    =>
      let inner := "".pushn ' ' (indent + 2)
      let close := "".pushn ' ' indent
      "{\n" ++ inner ++ "\"kind\": \"int\",\n" ++ inner ++
      "\"v\": " ++ toString n ++ "\n" ++ close ++ "}"
  | .float f  =>
      let inner := "".pushn ' ' (indent + 2)
      let close := "".pushn ' ' indent
      "{\n" ++ inner ++ "\"kind\": \"float\",\n" ++ inner ++
      "\"v\": " ++ toString f ++ "\n" ++ close ++ "}"
  | .str s    =>
      let inner := "".pushn ' ' (indent + 2)
      let close := "".pushn ' ' indent
      let esc := s.foldl (fun acc c =>
        if c == '\\' then acc ++ "\\\\"
        else if c == '"' then acc ++ "\\\""
        else acc.push c) ""
      "{\n" ++ inner ++ "\"kind\": \"str\",\n" ++ inner ++
      "\"v\": \"" ++ esc ++ "\"\n" ++ close ++ "}"
  | .tuple [] =>
      let inner := "".pushn ' ' (indent + 2)
      let close := "".pushn ' ' indent
      "{\n" ++ inner ++ "\"kind\": \"tuple\",\n" ++ inner ++ "\"v\": []\n" ++ close ++ "}"
  | .tuple xs =>
      let inner := "".pushn ' ' (indent + 2)
      let close := "".pushn ' ' indent
      let item := "".pushn ' ' (indent + 4)
      let renderedItems := xs.map (fun x => item ++ x.toJsonAt (indent + 4))
      "{\n" ++ inner ++ "\"kind\": \"tuple\",\n" ++
      inner ++ "\"v\": [\n" ++
      String.intercalate ",\n" renderedItems ++ "\n" ++ inner ++ "]\n" ++ close ++ "}"

def UOpArg.toJson (a : UOpArg) : String := a.toJsonAt 4

-- ============================================================================
-- UOpRecord — flat record shape used for the fixture round-trip.
-- ============================================================================

structure UOpRecord where
  idx   : Nat
  op    : UOpKind
  dtype : Dtype
  arg   : UOpArg
  src   : List Nat
  deriving Repr, Inhabited

/-- Build a Tgrad.UOp from a parsed (op, dtype, arg, srcUOps) tuple.
    Dispatches on op kind. Returns `Except` on shape mismatches
    (e.g. CONST with no int arg, ADD with ≠ 2 children). -/
def UOp.ofParsed (op : UOpKind) (dt : Dtype) (arg : UOpArg) (srcs : List UOp)
    : Except String UOp := do
  match op, srcs with
  | .CONST, [] =>
      match arg with
      | .int n   => pure (UOp.const dt (.i n))
      | .bool b  => pure (UOp.const dt (.b b))
      | .float f => pure (UOp.const dt (.f f))
      | _        => throw "CONST requires an int/bool/float arg"
  | .DEFINE_VAR, [lo, hi] =>
      match arg with
      | .str s => pure (UOp.defineVar s dt lo hi)
      | _      => throw "DEFINE_VAR requires a string-name arg"
  | .RANGE, [bound] =>
      match arg with
      | .tuple [.int idx, .str kindStr] =>
          match RangeKind.ofString kindStr with
          | some k => pure (UOp.range idx.toNat k bound)
          | none   => throw s!"unknown range kind: {kindStr}"
      | _ => throw "RANGE requires (int, str) tuple arg"
  | .SINK, [body] =>
      pure (UOp.sink body)
  | k, [a, b] =>
      match BinOp.ofKind k with
      | some bop => pure (UOp.binop bop a b dt)
      | none     => throw s!"unsupported binary op kind {k.toStr}"
  | _, _ =>
      throw s!"unsupported (op={op.toStr}, arity={srcs.length})"

/-- Rebuild a tree from a post-order-indexed record list. Returns
    `Except String UOp`; structurally-invalid input (out-of-bounds
    src, post-order violation, empty list) is reported, not panicked.

    Adapted from phase 02/03's toTree. -/
def toTree (records : List UOpRecord) : Except String UOp := do
  let arr := records.toArray
  let n := arr.size
  if n = 0 then throw "toTree: empty record list"
  let mut memo : Array UOp := Array.replicate n default
  for i in [:n] do
    let r := arr[i]!
    let mut srcUOps : List UOp := []
    for s in r.src do
      if s < i then
        srcUOps := srcUOps ++ [memo[s]!]
      else
        throw s!"toTree: rec {i} (op={r.op.toStr}) references src idx {s} (post-order violated)"
    let u ← UOp.ofParsed r.op r.dtype r.arg srcUOps
    memo := memo.set! i u
  pure memo[n - 1]!

-- ============================================================================
-- Rule + PatternMatcher
-- ============================================================================

/-- A rewrite rule. `names` declares the bound names the pattern
    produces; `rewrite` is the replacement function. -/
structure Rule where
  pat     : UPat
  names   : List String := []
  rewrite : Bindings → Option UOp

abbrev PatternMatcher := List Rule

/-- Names-consistency check. -/
def Rule.namesConsistent (r : Rule) : Option String :=
  let patNames := r.pat.collectNames
  let missing := patNames.filter (fun n => ¬ r.names.contains n)
  let extra   := r.names.filter (fun n => ¬ patNames.contains n)
  if missing.isEmpty ∧ extra.isEmpty then none
  else
    let parts :=
      (if missing.isEmpty then [] else [s!"pattern binds {missing} but rule.names omits them"]) ++
      (if extra.isEmpty   then [] else [s!"rule.names lists {extra} but pattern doesn't bind them"])
    some (String.intercalate "; " parts)

def PatternMatcher.validate (pm : PatternMatcher) : Option (Nat × String) :=
  let rec loop (i : Nat) : List Rule → Option (Nat × String)
    | []      => none
    | r :: rs =>
        match r.namesConsistent with
        | some msg => some (i, msg)
        | none     => loop (i+1) rs
  loop 0 pm

/-- Try every rule in order; return the first non-`none` result. -/
def PatternMatcher.tryRules (pm : PatternMatcher) (u : UOp) : Option UOp :=
  match pm with
  | []          => none
  | r :: rest   =>
      match matchPat r.pat u [] with
      | some bindings =>
          match r.rewrite bindings with
          | some u' => some u'
          | none    => PatternMatcher.tryRules rest u
      | none          => PatternMatcher.tryRules rest u

-- ============================================================================
-- Bottom-up rewrite (with fuel for fixed-point termination)
-- ============================================================================

inductive FpResult where
  | done          (u : UOp)
  | fuelExhausted (u : UOp)
  deriving Inhabited

partial def rewriteFixedPoint (pm : PatternMatcher) (u : UOp) (fuel : Nat := 1000) : FpResult :=
  match fuel with
  | 0      => FpResult.fuelExhausted u
  | n + 1  =>
      match pm.tryRules u with
      | none    => FpResult.done u
      | some u' => rewriteFixedPoint pm u' n

/-- Bottom-up: rewrite each src, rebuild parent, run fixed-point at root.

    Returns `none` if fuel was exhausted at any node (an oscillation
    indicator). To rebuild a parent after rewriting its children, we
    dispatch on the UOp constructor — preserving Tgrad's unified
    structure rather than going through the flat `(op, dtype, arg, src)`
    rebuild that phase 02/03 used. -/
partial def graphRewriteBottomUp? (pm : PatternMatcher) (u : UOp) : Option UOp :=
  let rebuilt := rebuildWithRewrittenChildren pm u
  rebuilt.bind (fun u' =>
    match rewriteFixedPoint pm u' with
    | FpResult.done u''           => some u''
    | FpResult.fuelExhausted _    => none)
where
  rebuildWithRewrittenChildren (pm : PatternMatcher) (u : UOp) : Option UOp :=
    match u with
    | .param _ _ _      => some u
    | .const _ _        => some u
    | .vconst _ _       => some u
    | .special _ _ _    => some u
    | .buffer _ _ _     => some u
    | .range idx kind bound => do
        let bound' ← graphRewriteBottomUp? pm bound
        some (.range idx kind bound')
    | .defineVar n d lo hi => do
        let lo' ← graphRewriteBottomUp? pm lo
        let hi' ← graphRewriteBottomUp? pm hi
        some (.defineVar n d lo' hi')
    | .index buf off    => do
        let buf' ← graphRewriteBottomUp? pm buf
        let off' ← graphRewriteBottomUp? pm off
        some (.index buf' off')
    | .load addr d      => do
        let addr' ← graphRewriteBottomUp? pm addr
        some (.load addr' d)
    | .store addr v     => do
        let addr' ← graphRewriteBottomUp? pm addr
        let v'    ← graphRewriteBottomUp? pm v
        some (.store addr' v')
    | .binop op a b d   => do
        let a' ← graphRewriteBottomUp? pm a
        let b' ← graphRewriteBottomUp? pm b
        some (.binop op a' b' d)
    | .cast d e         => do
        let e' ← graphRewriteBottomUp? pm e
        some (.cast d e')
    | .bitcast d e      => do
        let e' ← graphRewriteBottomUp? pm e
        some (.bitcast d e')
    | .gep e l          => do
        let e' ← graphRewriteBottomUp? pm e
        some (.gep e' l)
    | .endR r           => do
        let r' ← graphRewriteBottomUp? pm r
        some (.endR r')
    | .reduce op body axes => do
        let body' ← graphRewriteBottomUp? pm body
        let axes' ← mapAxes pm axes
        some (.reduce op body' axes')
    | .sink body        => do
        let body' ← graphRewriteBottomUp? pm body
        some (.sink body')
    | .wmma n a b c di do_ t => do
        let a' ← graphRewriteBottomUp? pm a
        let b' ← graphRewriteBottomUp? pm b
        let c' ← graphRewriteBottomUp? pm c
        some (.wmma n a' b' c' di do_ t)
    /- L14.B.1: movement ops recurse into src. -/
    | .permute src axes => do
        let src' ← graphRewriteBottomUp? pm src
        some (.permute src' axes)
    | .reshape src ns   => do
        let src' ← graphRewriteBottomUp? pm src
        some (.reshape src' ns)
    | .expand src ns    => do
        let src' ← graphRewriteBottomUp? pm src
        some (.expand src' ns)
    | .slice src sls    => do
        let src' ← graphRewriteBottomUp? pm src
        some (.slice src' sls)
    /- L14.B.2.a: var is a leaf — no children to rewrite. -/
    | .var _ _ => some u
  mapAxes (pm : PatternMatcher) : List UOp → Option (List UOp)
    | []      => some []
    | a :: as =>
        match graphRewriteBottomUp? pm a with
        | none   => none
        | some a' => match mapAxes pm as with
                     | none    => none
                     | some as' => some (a' :: as')

partial def graphRewriteBottomUp (pm : PatternMatcher) (u : UOp) : UOp :=
  match graphRewriteBottomUp? pm u with
  | some u' => u'
  | none    => panic! "graphRewriteBottomUp: fuel exhausted"

-- ============================================================================
-- toRecords — emit the post-order record list for a UOp tree.
-- ============================================================================

/-- Extract the UOpArg for a given UOp constructor (the boundary
    representation of the per-constructor structured field).

    For RANGE: returns `.tuple [.int idx, .str kind.toUpperStr]` to
    match phase-04's fixture format. -/
def UOp.toArg : UOp → UOpArg
  | .const _ (.i n)         => .int n
  | .const _ (.b b)         => .bool b
  | .const _ (.f f)         => .float f
  | .vconst _ _             => .none -- L1 doesn't emit VCONST round-trips
  | .defineVar n _ _ _      => .str n
  | .range idx kind _       => .tuple [.int (Int.ofNat idx), .str kind.toUpperStr]
  | _                       => .none

private def findIdx (seen : Array (UOp × Nat)) (u : UOp) : Option Nat :=
  seen.foldl
    (fun acc (entry : UOp × Nat) =>
      match acc with
      | some _ => acc
      | none   => if entry.1.beq u then some entry.2 else none)
    none

private abbrev WalkSt := Array UOpRecord × Array (UOp × Nat)

mutual
partial def walkUOp (u : UOp) (st : WalkSt) : Nat × WalkSt :=
  match findIdx st.2 u with
  | some i => (i, st)
  | none   =>
      let (srcIdxs, st1) := walkUOps u.children st
      let idx := st1.1.size
      let row : UOpRecord :=
        { idx := idx, op := u.kind, dtype := u.dtypeOf, arg := u.toArg, src := srcIdxs }
      (idx, (st1.1.push row, st1.2.push (u, idx)))

partial def walkUOps (us : List UOp) (st : WalkSt) : List Nat × WalkSt :=
  match us with
  | []      => ([], st)
  | u :: ts =>
      let (i, st1) := walkUOp u st
      let (is, st2) := walkUOps ts st1
      (i :: is, st2)
end

/-- Flatten a tree back to a canonical post-order record list. -/
def toRecords (root : UOp) : List UOpRecord :=
  let (_, (recs, _)) := walkUOp root (#[], #[])
  recs.toList

-- ============================================================================
-- Record JSON emit — matches phase 03's dag_out_expected.json byte format
-- (Python json.dumps(indent=2)).
-- ============================================================================

private def srcListJson (xs : List Nat) : String :=
  if xs.isEmpty then "[]"
  else "[\n      " ++ String.intercalate ",\n      " (xs.map toString) ++ "\n    ]"

def UOpRecord.toJson (r : UOpRecord) : String :=
  let argLine := match r.arg with
    | .none => "null"
    | _     => r.arg.toJson
  "  {\n" ++
  "    \"idx\": "   ++ toString r.idx               ++ ",\n" ++
  "    \"op\": \""  ++ r.op.toStr                   ++ "\",\n" ++
  "    \"dtype\": \"" ++ r.dtype.toSymbolicStr      ++ "\",\n" ++
  "    \"arg\": "   ++ argLine                       ++ ",\n" ++
  "    \"src\": "   ++ srcListJson r.src             ++ "\n  }"

def recordsToJson (rs : List UOpRecord) : String :=
  "[\n" ++ String.intercalate ",\n" (rs.map UOpRecord.toJson) ++ "\n]"

end Tgrad
