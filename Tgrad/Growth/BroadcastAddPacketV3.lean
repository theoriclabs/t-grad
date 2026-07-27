import Tgrad.Growth.BroadcastAddPacketV2
import Tgrad.Growth.BroadcastAddManifestV3

/-! # Prospective V3 observer-repair packet

This packet changes work *by* the verifier, not product behavior. It binds the
refuting diagnostic and predicts exactly two trace-contract changes. Product
authoring remains forbidden until a calibrated upstream baseline exists.
-/

namespace Tgrad.Growth.BroadcastAddPacketV3

open Tgrad.Growth.BroadcastAddManifestV3Generated

structure CalibrationContractPacket where
  id : String
  baselineRevision : String
  behavioralPacket : Tgrad.Growth.BroadcastAddPacket.ProspectivePacket
  contractHash : String
  refutingEvidenceId : String
  amendments : List TraceFootprintAmendment
  observerWriteSet : List String
  productWriteSet : List String
  deriving Repr, Inhabited

def packet : CalibrationContractPacket :=
  { id := packetId
    baselineRevision := baselineRevision
    behavioralPacket := Tgrad.Growth.BroadcastAddPacketV2.packet
    contractHash := effectiveContractHash
    refutingEvidenceId := refutingEvidenceId
    amendments := traceAmendments
    observerWriteSet :=
      ["scripts/spec/observe_broadcast_add.py",
       "scripts/spec/test_broadcast_add_observer.py"]
    productWriteSet := [] }

def CalibrationContractPacket.wellFormed (candidate : CalibrationContractPacket) : Bool :=
  !candidate.id.trimAscii.isEmpty &&
  !candidate.baselineRevision.trimAscii.isEmpty &&
  candidate.behavioralPacket.wellFormed &&
  !candidate.contractHash.trimAscii.isEmpty &&
  !candidate.refutingEvidenceId.trimAscii.isEmpty &&
  candidate.amendments.length == 2 &&
  (candidate.amendments.map (·.mutationId)).Nodup &&
  !candidate.observerWriteSet.isEmpty && candidate.observerWriteSet.Nodup &&
  candidate.productWriteSet.isEmpty

theorem packet_is_well_formed : packet.wellFormed := by native_decide

theorem repair_forbids_product_authoring : packet.productWriteSet = [] := by rfl

theorem behavioral_work_shape_is_unchanged :
    packet.behavioralPacket = Tgrad.Growth.BroadcastAddPacketV2.packet := by rfl

end Tgrad.Growth.BroadcastAddPacketV3
