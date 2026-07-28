import Tgrad.Growth.PilotState
import Tgrad.Growth.Work

namespace Tgrad.Growth.PilotReport

open Tgrad.Evidence
open Tgrad.Growth.PilotState

/-- Status JSON for a concrete product subject tree.  The live report path
supplies the actual current tree; the evidence-bound snapshot uses
`productBaseline`. -/
def reportJsonFor (subjectTree : TreeRef) : String :=
  let ctx := contextFor subjectTree
  let state := statusJsonFor ctx
  let stateBody := (state.dropEnd 2).toString ++ ",\n"
  stateBody ++
  "  \"derived_work\": " ++
    Tgrad.Growth.Work.workJsonFor (helpersStateFor ctx) ctx ++ "\n" ++
  "}"

/-- Evidence-bound snapshot.  `scripts/spec/pilot_status.sh` prints
`reportJsonFor` with the runtime current product tree instead. -/
def reportJson : String := reportJsonFor productBaseline

end Tgrad.Growth.PilotReport
