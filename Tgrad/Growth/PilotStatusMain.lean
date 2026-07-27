import Tgrad.Growth.PilotReport

/- This file is intentionally not imported by `TgradSpec`: invoking it emits
the deterministic pilot status without adding build-time output noise. -/

#eval IO.println Tgrad.Growth.PilotReport.reportJson
