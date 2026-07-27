import Tgrad.Growth.BroadcastAddPacket
import Tgrad.Growth.BroadcastAddManifestV2

/-! # Repaired prospective broadcast-add packet

The work shape is inherited from V1. Only the manifest identity, baseline and
the two corrected mutation obligations change.
-/

namespace Tgrad.Growth.BroadcastAddPacketV2

open Tgrad.Growth.BroadcastAddPacket
open Tgrad.Growth.BroadcastAddManifestV2

def packet : ProspectivePacket :=
  { BroadcastAddPacket.packet with
    id := manifest.packetId
    baselineRevision := manifest.baselineRevision
    manifestPath := manifest.path
    manifestHash := manifest.contentHash
    mutations := manifest.mutations }

theorem packet_is_well_formed : packet.wellFormed := by native_decide

theorem work_shape_is_inherited :
    packet.requirements = BroadcastAddPacket.packet.requirements ∧
    packet.scenarios = BroadcastAddPacket.packet.scenarios ∧
    packet.candidateComponents = BroadcastAddPacket.packet.candidateComponents ∧
    packet.definitionResources = BroadcastAddPacket.packet.definitionResources ∧
    packet.observerResources = BroadcastAddPacket.packet.observerResources ∧
    packet.observationOutcomeEnvelope =
      BroadcastAddPacket.packet.observationOutcomeEnvelope ∧
    packet.mustRemainUnchanged = BroadcastAddPacket.packet.mustRemainUnchanged ∧
    packet.forbiddenInferences = BroadcastAddPacket.packet.forbiddenInferences := by
  native_decide

end Tgrad.Growth.BroadcastAddPacketV2
