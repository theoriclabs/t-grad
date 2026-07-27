import Tgrad.Growth.BroadcastAddPacketV3

/-! # V4 chronology-tooling correction

V3's semantic correction is retained. V4 changes only where the tooling reads
the old observer hash: from the mutable worktree to V3's frozen Git object.
-/

namespace Tgrad.Growth.BroadcastAddPacketV4

structure ToolingAmendment where
  id : String
  baselineRevision : String
  binding : String
  before : String
  after : String
  inheritedPacket : Tgrad.Growth.BroadcastAddPacketV3.CalibrationContractPacket
  productWriteSet : List String
  deriving Repr, Inhabited

def packet : ToolingAmendment :=
  { id := "WORK-SPEC-BROADCAST-ADD-PROSPECTIVE-V4"
    baselineRevision := "15db155fadbaf68ab26ee2f08747064b0f08480e"
    binding := "v2_observer_sha256"
    before := "current_worktree_file"
    after := "frozen_git_object_at_v3_definition_revision"
    inheritedPacket := Tgrad.Growth.BroadcastAddPacketV3.packet
    productWriteSet := [] }

def ToolingAmendment.wellFormed (candidate : ToolingAmendment) : Bool :=
  !candidate.id.trimAscii.isEmpty &&
  !candidate.baselineRevision.trimAscii.isEmpty &&
  !candidate.binding.trimAscii.isEmpty &&
  candidate.before != candidate.after &&
  candidate.inheritedPacket.wellFormed &&
  candidate.productWriteSet.isEmpty

theorem packet_is_well_formed : packet.wellFormed := by native_decide

theorem semantic_v3_packet_is_inherited :
    packet.inheritedPacket = Tgrad.Growth.BroadcastAddPacketV3.packet := by rfl

theorem product_authoring_remains_forbidden : packet.productWriteSet = [] := by rfl

end Tgrad.Growth.BroadcastAddPacketV4
