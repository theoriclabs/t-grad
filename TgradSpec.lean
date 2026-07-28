import Tgrad.Ontology
import Tgrad.Spec.Epistemic
import Tgrad.Spec.Architecture
import Tgrad.Spec.Findings
import Tgrad.Spec.LiveConditions
import Tgrad.Spec.RuntimeWork
import Tgrad.Spec.Growth
import Tgrad.Spec.Evolution
import Tgrad.Spec.Work
import Tgrad.Spec.Parity
import Tgrad.Requirements.World
import Tgrad.Requirements.Relation
import Tgrad.Requirements.Requirements
import Tgrad.Requirements.Pilot
import Tgrad.Requirements.BroadcastAddPilot
import Tgrad.Specification.Boundary
import Tgrad.Specification.Pilot
import Tgrad.Specification.BroadcastAddPilot
import Tgrad.Conformance.Claims
import Tgrad.Evidence.Observations
import Tgrad.Evidence.PilotGenerated
import Tgrad.Evidence.SuiteGenerated
import Tgrad.Growth.Derived
import Tgrad.Growth.PilotState
import Tgrad.Growth.Work
import Tgrad.Growth.BroadcastAddManifest
import Tgrad.Growth.BroadcastAddManifestGenerated
import Tgrad.Growth.BroadcastAddManifestV2Generated
import Tgrad.Growth.BroadcastAddManifestV2
import Tgrad.Growth.BroadcastAddPacketV2
import Tgrad.Growth.BroadcastAddManifestV3Generated
import Tgrad.Growth.BroadcastAddManifestV3
import Tgrad.Growth.BroadcastAddPacketV3
import Tgrad.Growth.BroadcastAddPacketV4
import Tgrad.Growth.BroadcastAddPacketV5
import Tgrad.Growth.BroadcastAddPacketV6
import Tgrad.Growth.BroadcastAddObservationV1
import Tgrad.Growth.BroadcastAddConstructorCandidateV1
import Tgrad.Growth.BroadcastAddConstructorCandidateV2
import Tgrad.Growth.BroadcastAddObservationV2
import Tgrad.Growth.BroadcastAddReadbackCandidateV1
import Tgrad.Growth.BroadcastAddObservationV3
import Tgrad.Growth.BroadcastAddRealizeCandidateV1
import Tgrad.Growth.BroadcastAddObservationV4
import Tgrad.Growth.BroadcastAddRankedCandidateV1
import Tgrad.Growth.BroadcastAddObservationV5
import Tgrad.Growth.BroadcastAddRankedVerificationAmendmentV1
import Tgrad.Growth.BroadcastAddInt32CandidateV1
import Tgrad.Growth.BroadcastAddObservationV6
import Tgrad.Growth.BroadcastAddPacket
import Tgrad.Growth.PilotReport

/-! # TgradSpec — checked specification, separate from the product runtime

`Tgrad` is the product library. `TgradSpec` is the epistemic and operational
model used to inspect and plan changes to it. Keeping separate roots ensures
the spec is compiled and tested without linking roadmap/finding data into the
Python-facing shared library.
-/
