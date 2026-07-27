import Tgrad.Growth.PilotReport

/- This file is intentionally not imported by `TgradSpec`: invoking it emits
the deterministic pilot status without adding build-time output noise.

The promotion subject tree is supplied at run time via
`TGRAD_PRODUCT_SUBJECT_TREE=revision:contentHash:dirty` because Lean cannot
shell out during elaboration.  `scripts/spec/pilot_status.sh` computes the
actual current product tree from git and exports that variable. -/

open Tgrad.Evidence

/-- Parse `true` / `false` (case-sensitive, matching the shell exporter). -/
def parseDirty (value : String) : Option Bool :=
  if value == "true" then some true
  else if value == "false" then some false
  else none

/-- Decode `revision:contentHash:dirty` into a `TreeRef`. -/
def parseSubjectTree (raw : String) : Option TreeRef :=
  match raw.splitOn ":" with
  | [revision, contentHash, dirtyTok] =>
      if revision.isEmpty || contentHash.isEmpty then none
      else
        match parseDirty dirtyTok with
        | some dirty => some { revision, contentHash, dirty }
        | none => none
  | _ => none

def readCurrentSubjectTree : IO TreeRef := do
  let some raw ← IO.getEnv "TGRAD_PRODUCT_SUBJECT_TREE"
    | throw <| IO.userError
        "TGRAD_PRODUCT_SUBJECT_TREE is unset; run via scripts/spec/pilot_status.sh"
  match parseSubjectTree raw with
  | some tree =>
      if !tree.wellFormed then
        throw <| IO.userError s!"TGRAD_PRODUCT_SUBJECT_TREE is malformed: {raw}"
      pure tree
  | none =>
      throw <| IO.userError
        s!"TGRAD_PRODUCT_SUBJECT_TREE must be revision:contentHash:true|false, got: {raw}"

#eval show IO Unit from do
  let subject ← readCurrentSubjectTree
  IO.println (Tgrad.Growth.PilotReport.reportJsonFor subject)
