import Tgrad.Growth.PilotState
import Tgrad.Growth.Work

namespace Tgrad.Growth.PilotReport

def reportJson : String :=
  let state := Tgrad.Growth.PilotState.statusJson
  let stateBody := (state.dropEnd 2).toString ++ ",\n"
  stateBody ++
  "  \"derived_work\": " ++ Tgrad.Growth.Work.workJson ++ "\n" ++
  "}"

end Tgrad.Growth.PilotReport
