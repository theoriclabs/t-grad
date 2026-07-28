#!/usr/bin/env python3
"""Canonical generated-claim regenerator / guard.

Packet: mechanics.generated-claim-guard-v1

Role separation (explicit):
  * Lean (`Tgrad.Contract.GeneratedClaim`) validates typed requests with a
    private smart constructor and renders human-facing claim language.
  * This Python module regenerates a canonical machine-readable claim artifact
    from raw synthetic inputs and rejects handwritten / stale / colluding
    claim artifacts.

Trust boundary (honest, non-release):
  * Activation is synthetic-non-release only.
  * ContentDigest values are shape tokens, not cryptographic digests.
  * The synthetic one-requirement fixture is unauthenticated.
  * No real Tgrad completeness / tinygrad-compatibility claim exists.
  * Full cross-language generation from Lean is NOT available; this checker
    mirrors the canonical encoding over the same synthetic raw inputs.
  * Agreement between Lean native_decide checks and this regenerator is NOT
    authentication of either artifact.
  * This checker does not self-grep repository sources and does not trust a
    supplied output digest as its own ground truth. Identity is always
    recomputed from raw inputs.
  * Registry freeze/ancestry payloads are required premises and participate in
    claim identity.
  * Before emitting synthetic_complete, this checker validates the inherited
    catalog-closure and judging-input-closure contracts mirrored from Lean
    Completion/Chronology (totality, uniqueness, required-denominator mapping,
    accepted-exclusion references, required categories, roots, dependencies,
    reachability, nodup, cycles). The Git chronology exercise in
    check_cycle_chronology.py remains the independent ancestry oracle; this
    module mirrors the typed judging-graph schema checks, not the Git walker.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
ACTIVATION = "synthetic-non-release"

# Mirror Lean Tgrad.Contract.Chronology.allRequiredJudgingCategories.
REQUIRED_JUDGING_CATEGORIES = [
    "targetProfileDecision",
    "requirementDenotation",
    "boundaryDenotation",
    "adapter",
    "relation",
    "scenarioGenerator",
    "validator",
    "calibrationPolicy",
    "importedHelper",
    "environmentPolicy",
    "toolchain",
    "claimRenderer",
]

ALL_OBLIGATION_CLASSES = [
    "adequacy",
    "catalogClosure",
    "requirementDischarge",
    "scenarioObservation",
    "performanceQualification",
]

ALL_ACCEPTANCE_RULES = [
    "requireProof",
    "acceptBoundedSearch",
    "acceptJudgment",
    "releaseEligible",
]

REQUIRED_FIELD_NAMES = [
    "target",
    "profile",
    "sourceClosure",
    "subjectTree",
    "runtime",
    "assurancePolicy",
    "judgingClosureIdentity",
    "claimRenderer",
    "catalogDenominator",
    "discharges",
    "cycleRegistry",
    "activation",
    "selectedPromotedEvent",
]

# Exact raw schema. Unknown keys are rejected (no ignored colluding fields).
REQUIRED_RAW_KEYS = frozenset(
    {
        "activation",
        "target",
        "profile",
        "source_closure",
        "subject_tree",
        "runtime_artifact",
        "runtime_source_tree",
        "assurance_policy",
        "inventory",
        "dispositions",
        "requirement_entries",
        "accepted_judgments",
        "discharges",
        "judging_nodes",
        "judging_roots",
        "judging_discovered_inventory",
        "claim_renderer",
        "cycle_descriptors",
        "cycle_entries",
        "selected_promoted_event",
        "field_provenance",
    }
)

FORBIDDEN_RAW_CONCLUSION_KEYS = frozenset(
    {
        "complete",
        "pass",
        "certificate_digest",
        "claim_identity",
        "synthetic_complete",
        "judging_closure_identity",  # must be recomputed from judging graph
        "exclusions",  # must be derived from dispositions
        "tolerances",  # not independently authorable in this packet
    }
)

EPISTEMIC_LIMITATIONS = [
    "ContentDigest is a shape token, not a cryptographic digest",
    "synthetic one-requirement fixture is unauthenticated",
    "no real Tgrad completeness or tinygrad-compatibility claim exists",
    "Lean/Python agreement is not authentication",
    "activation is synthetic-non-release; not a product release guard",
    (
        "field provenance classification is authored premise, "
        "not cryptographic field grounding"
    ),
    (
        "Python encoding mirrors Lean shape tokens for this synthetic fixture; "
        "it is not authenticated cross-language generation from Lean"
    ),
]

# Discharge fields that must exactly match the catalog requirement entry.
DISCHARGE_CATALOG_BINDINGS = [
    ("requirement_id", "requirement_id"),
    ("requirement_semantics", "requirement_semantics"),
    ("specification_id", "specification_id"),
    ("specification_semantics", "specification_semantics"),
    ("boundary_id", "boundary_id"),
    ("boundary_semantics", "boundary_semantics"),
    ("scenario_id", "scenario_id"),
    ("scenario_digest", "scenario_digest"),
    ("relation_id", "relation_id"),
    ("relation_digest", "relation_digest"),
    ("adapter_id", "adapter_id"),
    ("adapter_digest", "adapter_digest"),
    ("validator_id", "validator_id"),
    ("validator_version", "validator_version"),
    ("calibration_id", "calibration_id"),
    ("calibration_campaign", "calibration_campaign"),
    ("calibration_fault_model", "calibration_fault_model"),
]


class ClaimGuardError(Exception):
    """Diagnosed rejection with a stable machine reason code."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        msg = reason if not detail else f"{reason}: {detail}"
        super().__init__(msg)


def enc_str(s: str) -> str:
    return f"{len(s)}:{s}"


def enc_nat(n: int) -> str:
    return enc_str(str(n))


def enc_digest(value: str) -> str:
    return enc_str(value)


def enc_seq(parts: list[str]) -> str:
    return enc_nat(len(parts)) + "".join(parts)


def digest_token(payload: str) -> str:
    """Shape token only — deterministic, not claimed as cryptographic auth."""
    return payload


def canonical_json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, separators=(",", ": ")) + "\n"


def encode_provenance(prov: dict[str, Any]) -> str:
    kind = prov["kind"]
    if kind == "imported":
        return (
            enc_str("imported")
            + enc_str(prov["id"])
            + enc_digest(prov["source_closure"])
        )
    if kind == "derived":
        return (
            enc_str("derived")
            + enc_str(prov["id"])
            + enc_str(prov["verifier"])
            + enc_digest(prov["input_closure"])
        )
    if kind == "calibrated":
        return (
            enc_str("calibrated")
            + enc_str(prov["id"])
            + enc_digest(prov["campaign"])
            + enc_str(prov["fault_model"])
        )
    if kind == "judgment":
        return (
            enc_str("judgment")
            + enc_str(prov["id"])
            + enc_str(prov["authority"])
            + enc_digest(prov["scope"])
            + enc_digest(prov["invalidation"])
        )
    raise ClaimGuardError("malformed_provenance", kind)


def provenance_label(prov: dict[str, Any]) -> str:
    kind = prov["kind"]
    if kind == "imported":
        return f"imported({prov['id']})"
    if kind == "derived":
        return f"derived({prov['id']}/{prov['verifier']})"
    if kind == "calibrated":
        return f"calibrated({prov['id']})"
    if kind == "judgment":
        return f"judgment({prov['id']}/{prov['authority']})"
    raise ClaimGuardError("malformed_provenance", kind)


def encode_assurance_state(state: dict[str, Any]) -> str:
    kind = state["kind"]
    if kind == "open":
        return enc_str("open")
    if kind == "blocked":
        return enc_str("blocked") + enc_str(state["id"]) + enc_str(state["blocker_kind"])
    if kind == "refuted":
        return enc_str("refuted") + enc_str(state["id"]) + enc_digest(state["artifact"])
    if kind == "survivedSearch":
        return (
            enc_str("survivedSearch")
            + enc_str(state["id"])
            + enc_str(state["generator_id"])
            + enc_digest(state["source_closure"])
            + enc_digest(state["seeds"])
            + enc_str(state["budget_desc"])
            + enc_digest(state["partitions"])
            + enc_nat(int(state["divergence_count"]))
            + enc_str(state["calibration"])
        )
    if kind == "acceptedBy":
        return (
            enc_str("acceptedBy")
            + enc_str(state["id"])
            + enc_str(state["authority"])
            + enc_digest(state["scope"])
            + enc_digest(state["invalidation"])
        )
    if kind == "proved":
        return enc_str("proved") + enc_str(state["id"]) + enc_str(state["theorem_name"])
    raise ClaimGuardError("malformed_assurance_state", kind)


def encode_target(t: dict[str, Any]) -> str:
    disposition = "promoted" if t.get("disposition") == "promoted" else "extractedCandidate"
    return (
        enc_str(t["id"])
        + enc_str(t["repository"])
        + enc_digest(t["revision"])
        + enc_digest(t["source_closure"])
        + enc_str(t["profile"])
        + enc_str(disposition)
    )


def encode_tree(tree: dict[str, Any]) -> str:
    dirty = "dirty" if tree.get("dirty") else "clean"
    return enc_str(tree["revision"]) + enc_digest(tree["content"]) + enc_str(dirty)


def encode_env(e: dict[str, Any]) -> str:
    return enc_str(e["id"]) + enc_digest(e["digest"])


def encode_evidence(e: dict[str, Any]) -> str:
    return enc_str(e["id"]) + enc_digest(e["digest"])


def encode_disposition(row: dict[str, Any]) -> str:
    d = row["disposition"]
    kind = d["kind"]
    if kind == "required":
        body = enc_str("required") + enc_str(d["requirement"])
    elif kind == "excluded":
        body = enc_str("excluded") + enc_str(d["judgment"])
    elif kind == "ambiguous":
        body = enc_str("ambiguous") + enc_str(d["resolve_by"])
    elif kind == "superseded":
        body = enc_str("superseded") + enc_str(d["replacement"])
    else:
        raise ClaimGuardError("malformed_disposition", kind)
    return enc_str(row["item"]) + body


def encode_requirement_entry(e: dict[str, Any]) -> str:
    envs = sorted(e["environments"], key=lambda x: x["id"])
    return (
        enc_str(e["requirement_id"])
        + enc_digest(e["requirement_semantics"])
        + enc_str(e["specification_id"])
        + enc_digest(e["specification_semantics"])
        + enc_str(e["boundary_id"])
        + enc_digest(e["boundary_semantics"])
        + enc_str(e["scenario_id"])
        + enc_digest(e["scenario_digest"])
        + enc_str(e["relation_id"])
        + enc_digest(e["relation_digest"])
        + enc_str(e["adapter_id"])
        + enc_digest(e["adapter_digest"])
        + enc_str(e["validator_id"])
        + enc_digest(e["validator_version"])
        + enc_str(e["calibration_id"])
        + enc_digest(e["calibration_campaign"])
        + enc_str(e["calibration_fault_model"])
        + enc_seq([encode_env(x) for x in envs])
    )


def encode_discharge(d: dict[str, Any]) -> str:
    envs = sorted(d["environments"], key=lambda x: x["id"])
    evs = sorted(d["evidence"], key=lambda x: x["id"])
    purpose = d.get("purpose", "semanticCompatibility")
    return (
        enc_str(d["requirement_id"])
        + enc_digest(d["requirement_semantics"])
        + enc_str(d["specification_id"])
        + enc_digest(d["specification_semantics"])
        + enc_str(d["boundary_id"])
        + enc_digest(d["boundary_semantics"])
        + enc_str(d["scenario_id"])
        + enc_digest(d["scenario_digest"])
        + enc_str(d["relation_id"])
        + enc_digest(d["relation_digest"])
        + enc_str(d["adapter_id"])
        + enc_digest(d["adapter_digest"])
        + enc_str(d["validator_id"])
        + enc_digest(d["validator_version"])
        + enc_str(d["calibration_id"])
        + enc_digest(d["calibration_campaign"])
        + enc_str(d["calibration_fault_model"])
        + encode_target(d["target"])
        + enc_str(d["profile"])
        + encode_tree(d["subject_tree"])
        + enc_digest(d["runtime_artifact"])
        + encode_tree(d["runtime_source_tree"])
        + enc_seq([encode_env(x) for x in envs])
        + enc_seq([encode_evidence(x) for x in evs])
        + enc_str(purpose)
        + encode_assurance_state(d["assurance_state"])
    )


def encode_cycle_descriptor(d: dict[str, Any]) -> str:
    return (
        enc_str(d["id"])
        + encode_target(d["target"])
        + enc_str(d["profile"])
        + enc_digest(d["expected_frozen_judging_closure_identity"])
        + enc_str(d["outcome"])
        + enc_str(d["freeze"])
        + enc_str(d["candidate"])
    )


def encode_ancestry_record(r: dict[str, Any]) -> str:
    parents = sorted(r.get("parents_within_capture", []))
    return (
        enc_str(r["commit"])
        + enc_seq([enc_str(p) for p in parents])
        + enc_digest(r["judging_closure_identity"])
    )


def encode_ancestry_manifest(m: dict[str, Any]) -> str:
    captured = sorted(m.get("captured_commit_ids", []))
    records = sorted(m.get("records", []), key=lambda r: r["commit"])
    return (
        enc_str(m["capture_identity"])
        + enc_digest(m["extractor_identity"])
        + encode_target(m["target"])
        + enc_str(m["profile"])
        + enc_digest(m["frozen_judging_closure_identity"])
        + enc_str(m["freeze"])
        + enc_str(m["candidate"])
        + enc_seq([enc_str(c) for c in captured])
        + enc_seq([encode_ancestry_record(r) for r in records])
    )


def encode_freeze_integrity(fi: dict[str, Any]) -> str:
    snaps = sorted(fi.get("supplied_snapshots", []), key=lambda r: r["commit"])
    protocol = fi.get("prospective_protocol")
    if protocol is None:
        protocol_enc = enc_str("none")
    else:
        protocol_enc = (
            enc_str("some")
            + enc_str(protocol["id"])
            + enc_digest(protocol["digest"])
        )
    return (
        enc_str(fi["claimed_kind"])
        + protocol_enc
        + encode_ancestry_manifest(fi["manifest"])
        + enc_seq([encode_ancestry_record(r) for r in snaps])
    )


def encode_cycle_entry(e: dict[str, Any]) -> str:
    body = encode_cycle_descriptor(e["descriptor"])
    fi = e.get("freeze_integrity")
    if fi is None:
        return body + enc_str("none")
    return body + enc_str("some") + encode_freeze_integrity(fi)


def encode_claim_field(f: dict[str, Any]) -> str:
    return enc_str(f["name"]) + encode_provenance(f["provenance"])


def encode_judging_node(n: dict[str, Any]) -> str:
    deps = sorted(n.get("dependencies", []))
    return (
        enc_str(n["id"])
        + enc_str(n["category"])
        + enc_digest(n["content"])
        + enc_seq([enc_str(d) for d in deps])
        + encode_provenance(n["provenance"])
    )


def policy_content_id(policy: dict[str, Any]) -> str:
    """Mirror Lean ProfileAssurancePolicy.contentId shape token.

    Lean: digest \"{profileId}|v{version}|{sorted class.tag=rule.tag}\"
    """
    rules = sorted(policy["rules"], key=lambda r: r["obligation_class"])
    body = ";".join(f"{r['obligation_class']}={r['rule']}" for r in rules)
    return digest_token(f"{policy['profile_id']}|v{policy['version']}|{body}")


def encode_frozen_catalog_binding(
    target: dict[str, Any],
    profile: str,
    source_closure: str,
    inventory: list[str],
    dispositions: list[dict[str, Any]],
    requirement_entries: list[dict[str, Any]],
    accepted_judgments: list[dict[str, Any]],
    policy: dict[str, Any],
) -> str:
    """Mirror Lean encodeFrozenCatalogBinding.

    Binds full catalog denotations into the judging targetProfileDecision node.
    Discharge evidence/assurance are intentionally excluded (post-freeze).
    """
    inv = [enc_str(i) for i in sorted(inventory)]
    disp = [
        encode_disposition(r)
        for r in sorted(dispositions, key=lambda x: x["item"])
    ]
    entries = [
        encode_requirement_entry(e)
        for e in sorted(requirement_entries, key=lambda x: x["requirement_id"])
    ]
    judgments = [
        enc_str(j["id"])
        + enc_str(j["authority"])
        + enc_digest(j["scope"])
        + enc_digest(j["invalidation"])
        for j in sorted(accepted_judgments, key=lambda x: x["id"])
    ]
    return (
        encode_target(target)
        + enc_str(profile)
        + enc_digest(source_closure)
        + enc_seq(inv)
        + enc_seq(disp)
        + enc_seq(entries)
        + enc_seq(judgments)
        + enc_digest(policy_content_id(policy))
    )


def encode_judging_scope_binding(
    target: dict[str, Any],
    profile: str,
    source_closure: str,
    policy: dict[str, Any],
    *,
    inventory: list[str] | None = None,
    dispositions: list[dict[str, Any]] | None = None,
    requirement_entries: list[dict[str, Any]] | None = None,
    accepted_judgments: list[dict[str, Any]] | None = None,
) -> str:
    """Compatibility wrapper; prefer encode_frozen_catalog_binding with full catalog."""
    if inventory is None or dispositions is None or requirement_entries is None:
        raise ClaimGuardError(
            "incomplete_frozen_catalog_binding",
            "judging scope binding requires full catalog denotations",
        )
    return encode_frozen_catalog_binding(
        target,
        profile,
        source_closure,
        inventory,
        dispositions,
        requirement_entries,
        accepted_judgments or [],
        policy,
    )


def claim_renderer_binding_payload() -> str:
    return enc_str("generated-claim-renderer") + enc_nat(SCHEMA_VERSION)


def compute_judging_closure_identity(
    nodes: list[dict[str, Any]], roots: list[str]
) -> str:
    sorted_nodes = sorted(nodes, key=lambda n: n["id"])
    node_part = enc_seq([encode_judging_node(n) for n in sorted_nodes])
    edge_part = enc_seq(
        [
            enc_str(n["id"]) + enc_str(dep)
            for n in sorted_nodes
            for dep in sorted(n.get("dependencies", []))
        ]
    )
    root_part = enc_seq([enc_str(r) for r in sorted(roots)])
    return digest_token(enc_nat(1) + node_part + edge_part + root_part)


def residual_from_assurance(req_id: str, state: dict[str, Any]) -> dict[str, Any] | None:
    kind = state["kind"]
    if kind == "proved":
        return None
    if kind == "open":
        return {"requirement": req_id, "kind": "open", "detail": "unresolved"}
    if kind == "blocked":
        return {
            "requirement": req_id,
            "kind": "blocked",
            "detail": f"{state['id']}:{state['blocker_kind']}",
        }
    if kind == "refuted":
        return {"requirement": req_id, "kind": "refuted", "detail": state["id"]}
    if kind == "survivedSearch":
        return {
            "requirement": req_id,
            "kind": "survivedSearch",
            "detail": (
                f"generator={state['generator_id']};"
                f"budget={state['budget_desc']};"
                f"divergences={state['divergence_count']}"
            ),
        }
    if kind == "acceptedBy":
        return {
            "requirement": req_id,
            "kind": "acceptedByJudgment",
            "detail": state["id"],
        }
    raise ClaimGuardError("malformed_assurance_state", kind)


def assurance_label(state: dict[str, Any]) -> str:
    kind = state["kind"]
    if kind == "open":
        return "open"
    if kind == "blocked":
        return f"blocked({state['id']}/{state['blocker_kind']})"
    if kind == "refuted":
        return f"refuted({state['id']})"
    if kind == "survivedSearch":
        return (
            f"survivedSearch(generator={state['generator_id']};"
            f"budget={state['budget_desc']};divergences={state['divergence_count']})"
        )
    if kind == "acceptedBy":
        return f"acceptedByJudgment({state['id']}/authority={state['authority']})"
    if kind == "proved":
        return f"proved({state['id']}/{state['theorem_name']})"
    raise ClaimGuardError("malformed_assurance_state", kind)


def derived_exclusions(dispositions: list[dict[str, Any]]) -> list[str]:
    rows = []
    for row in dispositions:
        d = row["disposition"]
        if d["kind"] == "excluded":
            rows.append(f"{row['item']}->{d['judgment']}")
    return sorted(rows)


@dataclass
class RawClaimInputs:
    """Premises only — no authored complete/pass Boolean or certificate digest."""

    activation: str
    target: dict[str, Any]
    profile: str
    source_closure: str
    subject_tree: dict[str, Any]
    runtime_artifact: str
    runtime_source_tree: dict[str, Any]
    assurance_policy: dict[str, Any]
    inventory: list[str]
    dispositions: list[dict[str, Any]]
    requirement_entries: list[dict[str, Any]]
    accepted_judgments: list[dict[str, Any]]
    discharges: list[dict[str, Any]]
    judging_nodes: list[dict[str, Any]]
    judging_roots: list[str]
    judging_discovered_inventory: list[str]
    claim_renderer: dict[str, Any]
    cycle_descriptors: list[dict[str, Any]]
    cycle_entries: list[dict[str, Any]]
    selected_promoted_event: str
    field_provenance: list[dict[str, Any]]

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "RawClaimInputs":
        keys = frozenset(data.keys())
        forbidden = sorted(keys & FORBIDDEN_RAW_CONCLUSION_KEYS)
        if forbidden:
            raise ClaimGuardError(
                "authored_conclusion_field",
                ",".join(forbidden),
            )
        unknown = sorted(keys - REQUIRED_RAW_KEYS)
        if unknown:
            raise ClaimGuardError("unknown_raw_field", ",".join(unknown))
        missing = sorted(REQUIRED_RAW_KEYS - keys)
        if missing:
            raise ClaimGuardError("missing_raw_field", ",".join(missing))
        return RawClaimInputs(
            activation=data["activation"],
            target=data["target"],
            profile=data["profile"],
            source_closure=data["source_closure"],
            subject_tree=data["subject_tree"],
            runtime_artifact=data["runtime_artifact"],
            runtime_source_tree=data["runtime_source_tree"],
            assurance_policy=data["assurance_policy"],
            inventory=list(data["inventory"]),
            dispositions=list(data["dispositions"]),
            requirement_entries=list(data["requirement_entries"]),
            accepted_judgments=list(data["accepted_judgments"]),
            discharges=list(data["discharges"]),
            judging_nodes=list(data["judging_nodes"]),
            judging_roots=list(data["judging_roots"]),
            judging_discovered_inventory=list(data["judging_discovered_inventory"]),
            claim_renderer=data["claim_renderer"],
            cycle_descriptors=list(data["cycle_descriptors"]),
            cycle_entries=list(data["cycle_entries"]),
            selected_promoted_event=data["selected_promoted_event"],
            field_provenance=list(data["field_provenance"]),
        )


def _duplicates(items: list[Any]) -> list[Any]:
    seen: set[Any] = set()
    dupes: list[Any] = []
    for item in items:
        if item in seen and item not in dupes:
            dupes.append(item)
        seen.add(item)
    return dupes


def _superseded_edges(
    dispositions: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    edges: list[tuple[str, str]] = []
    for row in dispositions:
        d = row["disposition"]
        if d["kind"] == "superseded":
            edges.append((row["item"], d["replacement"]))
    return edges


def _superseded_cycle_from(edges: list[tuple[str, str]], start: str) -> bool:
    successors: dict[str, list[str]] = {}
    for src, dst in edges:
        successors.setdefault(src, []).append(dst)

    def go(current: str, path: list[str], fuel: int) -> bool:
        if fuel <= 0:
            return False
        if current in path:
            return True
        return any(go(nxt, path + [current], fuel - 1) for nxt in successors.get(current, []))

    return go(start, [], len(edges) + 1)


def diagnose_assurance_policy(policy: dict[str, Any]) -> list[str]:
    """Mirror Lean ProfileAssurancePolicy.wellFormed + content discipline."""
    faults: list[str] = []
    if int(policy.get("version", 0)) <= 0:
        faults.append("malformed_policy:version")
    rules = list(policy.get("rules", []))
    classes = [r.get("obligation_class") for r in rules]
    for dup in _duplicates(classes):
        faults.append(f"duplicate_policy_class:{dup}")
    for cls in ALL_OBLIGATION_CLASSES:
        if cls not in classes:
            faults.append(f"missing_policy_class:{cls}")
    for cls in classes:
        if cls not in ALL_OBLIGATION_CLASSES:
            faults.append(f"unknown_policy_class:{cls}")
    for rule in rules:
        tag = rule.get("rule")
        if tag not in ALL_ACCEPTANCE_RULES:
            faults.append(f"unknown_acceptance_rule:{tag}")
    if set(classes) != set(ALL_OBLIGATION_CLASSES) or len(classes) != len(
        set(classes)
    ):
        # Keep a stable umbrella diagnosis for incomplete/non-total policies.
        faults.append("malformed_policy")
    return faults


def _walk_parents(
    records: list[dict[str, Any]], start: str
) -> set[str]:
    parent_map = {
        r["commit"]: list(r.get("parents_within_capture", [])) for r in records
    }
    edge_count = sum(len(ps) for ps in parent_map.values())
    fuel = len(records) + edge_count + 1
    frontier = [start]
    seen: set[str] = set()
    while frontier and fuel > 0:
        fuel -= 1
        current = frontier.pop(0)
        if current in seen:
            continue
        seen.add(current)
        frontier.extend(parent_map.get(current, []))
    return seen


def _records_set_eq(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> bool:
    def key(r: dict[str, Any]) -> tuple:
        return (
            r.get("commit"),
            tuple(sorted(r.get("parents_within_capture", []))),
            r.get("judging_closure_identity"),
        )

    return sorted(left, key=key) == sorted(right, key=key)


def diagnose_freeze_integrity(fi: dict[str, Any], event_id: str) -> list[str]:
    """Mirror Lean freezeIntegrityDiagnose for one promoted cycle entry."""
    prefix = f"freeze_integrity:{event_id}:"
    faults: list[str] = []
    manifest = fi.get("manifest") or {}
    records = list(manifest.get("records", []))
    snapshots = list(fi.get("supplied_snapshots", []))
    captured = list(manifest.get("captured_commit_ids", []))
    freeze = manifest.get("freeze")
    candidate = manifest.get("candidate")
    frozen_id = manifest.get("frozen_judging_closure_identity")

    if not manifest.get("capture_identity"):
        faults.append(prefix + "malformed_capture_identity")
    if not manifest.get("extractor_identity"):
        faults.append(prefix + "malformed_extractor_identity")
    if not freeze:
        faults.append(prefix + "malformed_freeze")
    if not candidate:
        faults.append(prefix + "malformed_candidate")
    if not frozen_id:
        faults.append(prefix + "malformed_frozen_closure_identity")
    if not captured:
        faults.append(prefix + "empty_captured_inventory")
    if not records:
        faults.append(prefix + "empty_ancestry")

    for dup in _duplicates(captured):
        faults.append(prefix + f"duplicate_captured_commit:{dup}")
    record_ids = [r.get("commit") for r in records]
    for dup in _duplicates(record_ids):
        faults.append(prefix + f"duplicate_commit:{dup}")

    if set(captured) != set(record_ids):
        faults.append(prefix + "capture_inventory_mismatch")
    for cid in captured:
        if cid not in record_ids:
            faults.append(prefix + f"omitted_ancestry_commit:{cid}")

    if freeze not in captured:
        faults.append(prefix + "freeze_not_in_ancestry")
    if candidate not in captured:
        faults.append(prefix + "candidate_not_in_ancestry")

    for rec in records:
        for parent in rec.get("parents_within_capture", []):
            if parent not in captured:
                faults.append(
                    prefix + f"unresolved_parent:{rec.get('commit')}:{parent}"
                )
        if len(rec.get("parents_within_capture", [])) != len(
            set(rec.get("parents_within_capture", []))
        ):
            faults.append(prefix + f"malformed_commit_record:{rec.get('commit')}")

    freeze_rec = next((r for r in records if r.get("commit") == freeze), None)
    if freeze_rec is not None and freeze_rec.get("parents_within_capture"):
        faults.append(prefix + "freeze_not_interval_root")

    if candidate and records:
        reachable = _walk_parents(records, candidate)
        if freeze not in reachable:
            faults.append(prefix + "freeze_not_ancestor")
        if set(reachable) != set(captured):
            for cid in set(captured) - set(reachable):
                faults.append(prefix + f"omitted_ancestry_commit:{cid}")
            for cid in set(reachable) - set(captured):
                faults.append(prefix + f"omitted_ancestry_commit:{cid}")

    if not _records_set_eq(records, snapshots):
        faults.append(prefix + "snapshot_inventory_mismatch")

    if freeze_rec is not None and freeze_rec.get("judging_closure_identity") != frozen_id:
        faults.append(prefix + "freeze_closure_mismatch")

    for rec in records:
        if rec.get("judging_closure_identity") != frozen_id:
            faults.append(
                prefix + f"judging_closure_drift:{rec.get('commit')}"
            )

    claimed = fi.get("claimed_kind")
    protocol = fi.get("prospective_protocol")
    if claimed == "retrospectiveFreezeIntegrity":
        if protocol is not None:
            faults.append(prefix + "retrospective_carries_protocol")
    elif claimed == "prospective":
        if protocol is None:
            faults.append(prefix + "prospective_without_protocol")
        elif not protocol.get("id") or not protocol.get("digest"):
            faults.append(prefix + "malformed_prospective_protocol")
    else:
        faults.append(prefix + f"unknown_claimed_kind:{claimed}")

    return faults


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _diagnose_environment(env: dict[str, Any], prefix: str) -> list[str]:
    faults: list[str] = []
    if not _nonempty(env.get("id")) or not _nonempty(env.get("digest")):
        faults.append(f"{prefix}:malformed_environment")
    return faults


def _diagnose_requirement_entry(entry: dict[str, Any]) -> list[str]:
    """Mirror Lean RequirementClosureEntry.wellFormed."""
    rid = entry.get("requirement_id", "")
    prefix = f"requirement_entry:{rid}"
    faults: list[str] = []
    required_fields = [
        ("requirement_id", "requirement_id"),
        ("requirement_semantics", "requirement_semantics"),
        ("specification_id", "specification_id"),
        ("specification_semantics", "specification_semantics"),
        ("boundary_id", "boundary_id"),
        ("boundary_semantics", "boundary_semantics"),
        ("scenario_id", "scenario_id"),
        ("scenario_digest", "scenario_digest"),
        ("relation_id", "relation_id"),
        ("relation_digest", "relation_digest"),
        ("adapter_id", "adapter_id"),
        ("adapter_digest", "adapter_digest"),
        ("validator_id", "validator_id"),
        ("validator_version", "validator_version"),
        ("calibration_id", "calibration_id"),
        ("calibration_campaign", "calibration_campaign"),
        ("calibration_fault_model", "calibration_fault_model"),
    ]
    for key, label in required_fields:
        if not _nonempty(entry.get(key)):
            faults.append(f"{prefix}:malformed_{label}")
    envs = list(entry.get("environments") or [])
    if not envs:
        faults.append(f"{prefix}:empty_environments")
        faults.append("malformed_requirement_entry")
    env_ids = [e.get("id") for e in envs]
    for dup in _duplicates(env_ids):
        faults.append(f"{prefix}:duplicate_environment:{dup}")
    for env in envs:
        faults.extend(_diagnose_environment(env, prefix))
    if faults and "malformed_requirement_entry" not in faults:
        # Umbrella diagnosis matching Lean CatalogFault.malformedRequirementEntry.
        if any(
            f.startswith(prefix + ":malformed_")
            or f.endswith(":empty_environments")
            or ":duplicate_environment:" in f
            for f in faults
        ):
            faults.append("malformed_requirement_entry")
    return faults


def _diagnose_target_identity(target: dict[str, Any], prefix: str = "target") -> list[str]:
    faults: list[str] = []
    if not _nonempty(target.get("id")):
        faults.append(f"malformed_{prefix}_id")
    if not _nonempty(target.get("repository")):
        faults.append(f"malformed_{prefix}_repository")
    if not _nonempty(target.get("revision")):
        faults.append(f"malformed_{prefix}_revision")
    if not _nonempty(target.get("source_closure")):
        faults.append(f"malformed_{prefix}_source_closure")
    if not _nonempty(target.get("profile")):
        faults.append(f"malformed_{prefix}_profile")
    return faults


def _diagnose_judgment_identity(j: dict[str, Any]) -> list[str]:
    faults: list[str] = []
    jid = j.get("id", "")
    if not _nonempty(jid):
        faults.append("malformed_judgment:id")
    if not _nonempty(j.get("authority")):
        faults.append(f"malformed_judgment:{jid}:authority")
    if not _nonempty(j.get("scope")):
        faults.append(f"malformed_judgment:{jid}:scope")
    if not _nonempty(j.get("invalidation")):
        faults.append(f"malformed_judgment:{jid}:invalidation")
    return faults


def diagnose_catalog_closure(raw: RawClaimInputs) -> list[str]:
    """Mirror Lean CatalogClosureCandidate / catalogDiagnose contracts."""
    faults: list[str] = []
    if not raw.inventory or not raw.requirement_entries:
        faults.append("empty_denominator")
    faults.extend(_diagnose_target_identity(raw.target))
    if not _nonempty(raw.profile):
        faults.append("malformed_profile")
    if not _nonempty(raw.source_closure):
        faults.append("malformed_source_closure")
    if raw.target.get("disposition") != "promoted":
        faults.append("unpromoted_target")
    if raw.target.get("profile") != raw.profile:
        faults.append("mismatched_target_profile")
    if raw.target.get("source_closure") != raw.source_closure:
        faults.append("mismatched_source_closure")
    if raw.assurance_policy.get("profile_id") != raw.profile:
        faults.append("mismatched_policy_profile")
    faults.extend(diagnose_assurance_policy(raw.assurance_policy))

    for entry in raw.requirement_entries:
        faults.extend(_diagnose_requirement_entry(entry))

    req_ids = [e["requirement_id"] for e in raw.requirement_entries]
    for dup in _duplicates(req_ids):
        faults.append(f"duplicate_requirement:{dup}")
    for dup in _duplicates(list(raw.inventory)):
        faults.append(f"duplicate_inventory_item:{dup}")
    for item in raw.inventory:
        if not _nonempty(item):
            faults.append("malformed_inventory_item")

    disposition_items = [row["item"] for row in raw.dispositions]
    for dup in _duplicates(disposition_items):
        faults.append(f"duplicate_disposition:{dup}")
    for row in raw.dispositions:
        if not _nonempty(row.get("item")):
            faults.append("malformed_disposition")

    inv_set = set(raw.inventory)
    disp_set = set(disposition_items)
    for item in raw.inventory:
        if item not in disp_set:
            faults.append(f"missing_disposition:{item}")
    for item in disposition_items:
        if item not in inv_set:
            faults.append(f"extra_disposition:{item}")

    for row in raw.dispositions:
        kind = row["disposition"]["kind"]
        if kind == "ambiguous":
            faults.append(f"unresolved_applicable:{row['item']}")
        if kind == "superseded":
            replacement = row["disposition"]["replacement"]
            if replacement not in inv_set:
                faults.append(
                    f"superseded_not_in_inventory:{row['item']}->{replacement}"
                )
            if replacement == row["item"]:
                faults.append(f"superseded_self:{row['item']}")
        if kind == "required" and not _nonempty(row["disposition"].get("requirement")):
            faults.append(f"malformed_disposition:{row['item']}")
        if kind == "excluded" and not _nonempty(row["disposition"].get("judgment")):
            faults.append(f"malformed_disposition:{row['item']}")

    edges = _superseded_edges(raw.dispositions)
    for start, _ in edges:
        if _superseded_cycle_from(edges, start):
            fault = f"superseded_cycle:{start}"
            if fault not in faults:
                faults.append(fault)

    required_from_disp = [
        row["disposition"]["requirement"]
        for row in raw.dispositions
        if row["disposition"]["kind"] == "required"
    ]
    for rid in req_ids:
        if rid not in required_from_disp:
            faults.append(f"required_not_in_denominator:{rid}")
    for rid in required_from_disp:
        if rid not in req_ids:
            faults.append(f"required_disposition_omitted_from_denominator:{rid}")

    accepted_ids = [j["id"] for j in raw.accepted_judgments]
    for dup in _duplicates(accepted_ids):
        faults.append(f"duplicate_judgment:{dup}")
    for j in raw.accepted_judgments:
        faults.extend(_diagnose_judgment_identity(j))
    exclusion_ids = [
        row["disposition"]["judgment"]
        for row in raw.dispositions
        if row["disposition"]["kind"] == "excluded"
    ]
    for jid in exclusion_ids:
        if jid not in accepted_ids:
            faults.append(f"unaccepted_exclusion:{jid}")
    for jid in accepted_ids:
        if jid not in exclusion_ids:
            faults.append(f"unreferenced_judgment:{jid}")

    return faults


def _lookup_node(
    nodes: list[dict[str, Any]], node_id: str
) -> dict[str, Any] | None:
    return next((n for n in nodes if n["id"] == node_id), None)


def _reachable_from_roots(
    nodes: list[dict[str, Any]], roots: list[str]
) -> set[str]:
    by_id = {n["id"]: n for n in nodes}
    edge_count = sum(len(n.get("dependencies", [])) for n in nodes)
    fuel = len(nodes) + edge_count + len(roots) + 1
    frontier = list(roots)
    seen: set[str] = set()
    while frontier and fuel > 0:
        fuel -= 1
        current = frontier.pop(0)
        if current in seen:
            continue
        seen.add(current)
        node = by_id.get(current)
        if node is not None:
            frontier.extend(node.get("dependencies", []))
    return seen


def _dependency_cycle_from(nodes: list[dict[str, Any]], start: str) -> bool:
    by_id = {n["id"]: n for n in nodes}
    edge_count = sum(len(n.get("dependencies", [])) for n in nodes)
    fuel0 = len(nodes) + edge_count + 1

    def go(current: str, path: list[str], fuel: int) -> bool:
        if fuel <= 0:
            return False
        if current in path:
            return True
        node = by_id.get(current)
        if node is None:
            return False
        return any(
            go(dep, path + [current], fuel - 1) for dep in node.get("dependencies", [])
        )

    return go(start, [], fuel0)


def diagnose_judging_input_closure(
    nodes: list[dict[str, Any]],
    roots: list[str],
    discovered_inventory: list[str],
    *,
    schema_version: int = 1,
) -> list[str]:
    """Mirror Lean JudgingInputClosureCandidate / judgingClosureDiagnose."""
    faults: list[str] = []
    if schema_version != SCHEMA_VERSION:
        faults.append("wrong_schema_version")
    if not nodes:
        faults.append("empty_nodes")
    if not roots:
        faults.append("empty_roots")

    ids = [n["id"] for n in nodes]
    id_set = set(ids)
    for dup in _duplicates(ids):
        faults.append(f"duplicate_node:{dup}")
    for dup in _duplicates(list(roots)):
        faults.append(f"duplicate_root:{dup}")

    for node in nodes:
        deps = list(node.get("dependencies", []))
        for dup in _duplicates(deps):
            faults.append(f"duplicate_dependency:{node['id']}:{dup}")
        for dep in deps:
            if dep not in id_set:
                faults.append(f"unresolved_dependency:{node['id']}:{dep}")
        prov = node.get("provenance")
        if not isinstance(prov, dict) or prov.get("kind") not in {
            "imported",
            "derived",
            "calibrated",
            "judgment",
        }:
            faults.append(f"inadmissible_provenance:{node['id']}")

    for root in roots:
        if not root:
            faults.append(f"malformed_root:{root}")
        if root not in id_set:
            faults.append(f"unresolved_root:{root}")

    reachable = _reachable_from_roots(nodes, roots)
    for node_id in ids:
        if node_id not in reachable:
            faults.append(f"unreachable_node:{node_id}")

    for node_id in ids:
        if _dependency_cycle_from(nodes, node_id):
            fault = f"dependency_cycle:{node_id}"
            if fault not in faults:
                faults.append(fault)

    present = {n.get("category") for n in nodes}
    for cat in REQUIRED_JUDGING_CATEGORIES:
        if cat not in present:
            faults.append(f"missing_required_category:{cat}")

    for dup in _duplicates(list(discovered_inventory)):
        faults.append(f"duplicate_discovered_input:{dup}")
    discovered_set = set(discovered_inventory)
    for node_id in discovered_inventory:
        if node_id not in id_set:
            faults.append(f"undeclared_discovered_input:{node_id}")
    for node_id in ids:
        if node_id not in discovered_set:
            faults.append(f"undiscovered_declared_node:{node_id}")
    if set(ids) != discovered_set:
        faults.append("discovered_inventory_mismatch")

    return faults


def validate_raw(raw: RawClaimInputs) -> list[str]:
    """Structural + epistemic diagnosis over raw premises.

    Emits completion language only after catalog-closure and judging-input-
    closure contracts succeed (mirrors Lean validateProfileFromCatalog +
    validateJudgingInputClosure prerequisites of validateGeneratedClaim).
    """
    faults: list[str] = []
    if raw.activation != ACTIVATION:
        faults.append("wrong_activation")

    faults.extend(diagnose_catalog_closure(raw))
    faults.extend(
        diagnose_judging_input_closure(
            raw.judging_nodes,
            raw.judging_roots,
            raw.judging_discovered_inventory,
        )
    )

    if raw.subject_tree.get("dirty") or raw.runtime_source_tree.get("dirty"):
        faults.append("dirty_tree")
    if raw.runtime_source_tree != raw.subject_tree:
        faults.append("runtime_tree_mismatch")

    names = [f["name"] for f in raw.field_provenance]
    if len(names) != len(set(names)):
        faults.append("duplicate_field_provenance")
    if set(names) != set(REQUIRED_FIELD_NAMES):
        for required in REQUIRED_FIELD_NAMES:
            if required not in names:
                faults.append(f"missing_field_provenance:{required}")
        for name in names:
            if name not in REQUIRED_FIELD_NAMES:
                faults.append(f"extra_field_provenance:{name}")

    if not raw.claim_renderer.get("id") or not raw.claim_renderer.get("content"):
        faults.append("missing_claim_renderer")

    expected_renderer = digest_token(claim_renderer_binding_payload())
    if raw.claim_renderer.get("content") != expected_renderer:
        faults.append("mismatched_claim_renderer_binding")

    expected_scope = digest_token(
        encode_frozen_catalog_binding(
            raw.target,
            raw.profile,
            raw.source_closure,
            raw.inventory,
            raw.dispositions,
            raw.requirement_entries,
            raw.accepted_judgments,
            raw.assurance_policy,
        )
    )
    scope_nodes = [
        n for n in raw.judging_nodes if n.get("category") == "targetProfileDecision"
    ]
    if not scope_nodes or any(n.get("content") != expected_scope for n in scope_nodes):
        faults.append("mismatched_judging_scope_binding")

    node_ids = sorted(n["id"] for n in raw.judging_nodes)
    if node_ids != sorted(raw.judging_discovered_inventory):
        faults.append("judging_discovered_inventory_mismatch")

    judging_identity = compute_judging_closure_identity(
        raw.judging_nodes, raw.judging_roots
    )

    req_ids = sorted(e["requirement_id"] for e in raw.requirement_entries)
    discharged = sorted(d["requirement_id"] for d in raw.discharges)
    if req_ids != discharged:
        faults.append("missing_or_extra_discharge")

    entries_by_id = {e["requirement_id"]: e for e in raw.requirement_entries}
    for d in raw.discharges:
        entry = entries_by_id.get(d["requirement_id"])
        if entry is None:
            faults.append("discharge_without_catalog_entry")
            continue
        for d_key, e_key in DISCHARGE_CATALOG_BINDINGS:
            if d.get(d_key) != entry.get(e_key):
                faults.append(f"discharge_catalog_mismatch:{d_key}")
        entry_env_ids = sorted(e["id"] for e in entry.get("environments", []))
        discharge_env_ids = sorted(e["id"] for e in d.get("environments", []))
        if entry_env_ids != discharge_env_ids:
            faults.append("discharge_catalog_mismatch:environments")
        for env in d.get("environments", []):
            match = next(
                (e for e in entry.get("environments", []) if e["id"] == env["id"]),
                None,
            )
            if match is None or match != env:
                faults.append("discharge_catalog_mismatch:environment_identity")
            faults.extend(
                _diagnose_environment(env, f"discharge:{d.get('requirement_id')}")
            )
        if not d.get("environments"):
            faults.append(f"discharge:{d.get('requirement_id')}:empty_environments")
        if d.get("purpose") == "performance":
            faults.append("performance_purpose_rejected")
        if d["profile"] != raw.profile:
            faults.append("mismatched_discharge_profile")
        if d["target"] != raw.target:
            faults.append("mismatched_discharge_target")
        faults.extend(_diagnose_target_identity(d.get("target") or {}, "discharge_target"))
        if not _nonempty(d.get("profile")):
            faults.append(f"discharge:{d.get('requirement_id')}:malformed_profile")
        if d["subject_tree"] != raw.subject_tree:
            faults.append("mismatched_discharge_subject_tree")
        if d["runtime_artifact"] != raw.runtime_artifact:
            faults.append("mismatched_discharge_runtime_artifact")
        if d["runtime_source_tree"] != raw.runtime_source_tree:
            faults.append("mismatched_discharge_runtime_source_tree")
        if not _nonempty(d.get("runtime_artifact")):
            faults.append(f"discharge:{d.get('requirement_id')}:malformed_runtime")
        subject = d.get("subject_tree") or {}
        if not _nonempty(subject.get("revision")) or not _nonempty(subject.get("content")):
            faults.append(f"discharge:{d.get('requirement_id')}:malformed_subject_tree")
        if not d.get("evidence"):
            faults.append("empty_discharge_evidence")
        ev_ids = [e.get("id") for e in d.get("evidence", [])]
        for dup in _duplicates(ev_ids):
            faults.append(f"discharge:{d.get('requirement_id')}:duplicate_evidence:{dup}")
        for ev in d.get("evidence", []):
            if not _nonempty(ev.get("id")) or not _nonempty(ev.get("digest")):
                faults.append(
                    f"discharge:{d.get('requirement_id')}:malformed_evidence"
                )
        state = d["assurance_state"]
        policy_rules = {
            r["obligation_class"]: r["rule"] for r in raw.assurance_policy["rules"]
        }
        rule = policy_rules.get("requirementDischarge")
        if rule == "requireProof" and state["kind"] != "proved":
            faults.append("policy_does_not_admit")
        if rule == "acceptBoundedSearch" and state["kind"] != "survivedSearch":
            faults.append("policy_does_not_admit")
        if state["kind"] == "survivedSearch" and rule == "requireProof":
            faults.append("bounded_search_mislabeled_as_proof_path")

    desc_by_id = {d["id"]: d for d in raw.cycle_descriptors}
    entry_descs = [e["descriptor"] for e in raw.cycle_entries]
    if sorted(d["id"] for d in raw.cycle_descriptors) != sorted(
        d["id"] for d in entry_descs
    ):
        faults.append("cycle_descriptor_entry_mismatch")
    for entry in raw.cycle_entries:
        desc = entry["descriptor"]
        if desc_by_id.get(desc["id"]) != desc:
            faults.append("cycle_descriptor_entry_mismatch")
        if desc["outcome"] == "promoted":
            if entry.get("freeze_integrity") is None:
                faults.append("promoted_missing_freeze_integrity")
            else:
                fi = entry["freeze_integrity"]
                manifest = fi["manifest"]
                if manifest.get("target") != desc["target"]:
                    faults.append("promoted_target_mismatch")
                if manifest.get("profile") != desc["profile"]:
                    faults.append("promoted_profile_mismatch")
                if (
                    manifest.get("frozen_judging_closure_identity")
                    != desc["expected_frozen_judging_closure_identity"]
                ):
                    faults.append("promoted_closure_mismatch")
                if (
                    manifest.get("freeze") != desc["freeze"]
                    or manifest.get("candidate") != desc["candidate"]
                ):
                    faults.append("promoted_commit_mismatch")
                faults.extend(diagnose_freeze_integrity(fi, desc["id"]))
        elif entry.get("freeze_integrity") is not None:
            faults.append("non_promoted_carries_chronology")

    selected = desc_by_id.get(raw.selected_promoted_event)
    if selected is None:
        faults.append("selected_event_missing")
    elif selected["outcome"] != "promoted":
        # Non-promoted selection is the primary diagnosis; do not also demand
        # freeze integrity that non-promoted events are forbidden to carry.
        faults.append("selected_event_not_promoted")
    else:
        selected_entry = next(
            (
                e
                for e in raw.cycle_entries
                if e["descriptor"]["id"] == raw.selected_promoted_event
            ),
            None,
        )
        if selected_entry is None or selected_entry.get("freeze_integrity") is None:
            faults.append("selected_event_lacks_freeze_integrity")
        # Cross-bind only the selected promoted event to this claim's scope.
        if selected["target"] != raw.target:
            faults.append("mismatched_chronology_target")
        if selected["profile"] != raw.profile:
            faults.append("mismatched_chronology_profile")
        if selected["expected_frozen_judging_closure_identity"] != judging_identity:
            faults.append("stale_or_mismatched_judging_closure_identity")
        if raw.subject_tree.get("revision") != selected["candidate"]:
            faults.append("subject_candidate_mismatch")
        if selected_entry is not None and selected_entry.get("freeze_integrity"):
            fi = selected_entry["freeze_integrity"]
            manifest = fi["manifest"]
            if manifest.get("candidate") != raw.subject_tree.get("revision"):
                faults.append("subject_candidate_mismatch")
            if manifest.get("frozen_judging_closure_identity") != judging_identity:
                faults.append("stale_or_mismatched_judging_closure_identity")

    return sorted(set(faults))


def compute_claim_identity(raw: RawClaimInputs) -> str:
    judging_identity = compute_judging_closure_identity(
        raw.judging_nodes, raw.judging_roots
    )
    inventory = [enc_str(i) for i in sorted(raw.inventory)]
    dispositions = [
        encode_disposition(r)
        for r in sorted(raw.dispositions, key=lambda x: x["item"])
    ]
    entries = [
        encode_requirement_entry(e)
        for e in sorted(raw.requirement_entries, key=lambda x: x["requirement_id"])
    ]
    judgments = [
        enc_str(j["id"])
        + enc_str(j["authority"])
        + enc_digest(j["scope"])
        + enc_digest(j["invalidation"])
        for j in sorted(raw.accepted_judgments, key=lambda x: x["id"])
    ]
    discharges = [
        encode_discharge(d)
        for d in sorted(raw.discharges, key=lambda x: x["requirement_id"])
    ]
    registry_entries = [
        encode_cycle_entry(e)
        for e in sorted(raw.cycle_entries, key=lambda x: x["descriptor"]["id"])
    ]
    fields = [
        encode_claim_field(f)
        for f in sorted(raw.field_provenance, key=lambda x: x["name"])
    ]
    renderer_nodes = sorted(
        [n for n in raw.judging_nodes if n.get("category") == "claimRenderer"],
        key=lambda n: n["id"],
    )
    renderer_part = enc_seq(
        [
            enc_str(n["id"])
            + enc_digest(n["content"])
            + encode_provenance(n["provenance"])
            for n in renderer_nodes
        ]
    )
    payload = (
        enc_nat(SCHEMA_VERSION)
        + enc_str(raw.activation)
        + enc_str(raw.selected_promoted_event)
        + encode_target(raw.target)
        + enc_str(raw.profile)
        + enc_digest(raw.source_closure)
        + encode_tree(raw.subject_tree)
        + enc_digest(raw.runtime_artifact)
        + encode_tree(raw.runtime_source_tree)
        + enc_digest(policy_content_id(raw.assurance_policy))
        + enc_digest(judging_identity)
        + enc_seq(inventory)
        + enc_seq(dispositions)
        + enc_seq(entries)
        + enc_seq(judgments)
        + enc_seq(discharges)
        + enc_seq(registry_entries)
        + enc_seq(fields)
        + renderer_part
    )
    return digest_token(payload)


def derive_residuals(raw: RawClaimInputs) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for d in raw.discharges:
        residual = residual_from_assurance(d["requirement_id"], d["assurance_state"])
        if residual is not None:
            rows.append(residual)
    return sorted(rows, key=lambda r: (r["requirement"], r["kind"]))


def render_report(
    raw: RawClaimInputs, claim_identity: str, residuals: list[dict[str, Any]]
) -> str:
    denom_empty = not raw.inventory or not raw.requirement_entries
    unresolved = any(r["kind"] in {"open", "blocked", "refuted"} for r in residuals)
    if denom_empty:
        status = "STATUS: incomplete (empty denominator; complete language forbidden)"
        synthetic_complete = False
    elif unresolved:
        status = "STATUS: incomplete (open/blocked/refuted obligations remain)"
        synthetic_complete = False
    elif raw.activation == ACTIVATION:
        status = (
            "STATUS: synthetic-complete under declared scope "
            "(NON-RELEASE; not a real Tgrad claim)"
        )
        synthetic_complete = True
    else:
        status = "STATUS: incomplete"
        synthetic_complete = False

    env_ids = sorted(
        {e["id"] for d in raw.discharges for e in d["environments"]}
    )
    grades = [
        f"  - {d['requirement_id']}: {assurance_label(d['assurance_state'])}"
        for d in raw.discharges
    ]
    residual_lines = (
        ["  (none)"]
        if not residuals
        else [f"  - {r['requirement']}: {r['kind']} ({r['detail']})" for r in residuals]
    )
    search_bounds = []
    for d in raw.discharges:
        st = d["assurance_state"]
        if st["kind"] == "survivedSearch":
            search_bounds.append(
                f"  - {d['requirement_id']}: budget={st['budget_desc']}; seeds={st['seeds']}"
            )
    if not search_bounds:
        search_bounds = ["  (none)"]

    for d in raw.discharges:
        label = assurance_label(d["assurance_state"])
        if d["assurance_state"]["kind"] != "proved" and label.startswith("proved("):
            raise ClaimGuardError("bounded_search_mislabeled_as_proof")

    exclusions = derived_exclusions(raw.dispositions)
    provenance_lines = [
        f"  - {f['name']}: {provenance_label(f['provenance'])}"
        for f in sorted(raw.field_provenance, key=lambda x: x["name"])
    ]
    selected = next(
        d for d in raw.cycle_descriptors if d["id"] == raw.selected_promoted_event
    )
    clean = "dirty" if raw.subject_tree.get("dirty") else "clean"
    judging_identity = compute_judging_closure_identity(
        raw.judging_nodes, raw.judging_roots
    )
    lines = [
        "GENERATED CLAIM REPORT",
        f"schema_version={SCHEMA_VERSION}",
        f"activation={raw.activation}",
        f"claim_identity={claim_identity}",
        "EPISTEMIC LIMITATIONS:",
        *[f"  - {item}" for item in EPISTEMIC_LIMITATIONS],
        (
            f"target={raw.target['id']}@{raw.target['revision']} "
            f"repo={raw.target['repository']}"
        ),
        f"profile={raw.profile}",
        f"source_closure={raw.source_closure}",
        (
            f"subject_tree={raw.subject_tree['revision']}/"
            f"{raw.subject_tree['content']}/{clean}"
        ),
        f"runtime_artifact={raw.runtime_artifact}",
        (
            f"runtime_source_tree={raw.runtime_source_tree['revision']}/"
            f"{raw.runtime_source_tree['content']}"
        ),
        f"selected_promoted_event={raw.selected_promoted_event}",
        f"selected_candidate_commit={selected['candidate']}",
        (
            f"subject_candidate_binding={raw.subject_tree['revision']}"
            f"=={selected['candidate']}"
        ),
        "environment_scope={" + ",".join(env_ids) + "}",
        f"denominator_inventory={len(raw.inventory)}",
        f"denominator_requirements={len(raw.requirement_entries)}",
        f"exclusions=[{','.join(exclusions)}]",
        "tolerances: (none declared in synthetic fixture)",
        "search_bounds:",
        *search_bounds,
        f"assurance_policy={policy_content_id(raw.assurance_policy)}",
        f"judging_closure_identity={judging_identity}",
        "field_provenance:",
        *provenance_lines,
        "assurance_grades:",
        *grades,
        "residual_obligations:",
        *residual_lines,
        status,
        "NOTE: bounded search and accepted judgment are never printed as proof.",
        "NOTE: no scalar parity percentage is defined or printed.",
        f"synthetic_complete={str(synthetic_complete).lower()}",
    ]
    text = "\n".join(lines)
    if "parity%" in text or "% compatible" in text:
        raise ClaimGuardError("scalar_parity_percentage_forbidden")
    return text


def render_failure_report(raw: RawClaimInputs, faults: list[str]) -> dict[str, Any]:
    residuals = derive_residuals(raw)
    residual_lines = (
        ["  (none)"]
        if not residuals
        else [f"  - {r['requirement']}: {r['kind']} ({r['detail']})" for r in residuals]
    )
    fault_lines = [f"  - {f}" for f in faults]
    text = "\n".join(
        [
            "GENERATED CLAIM ATTEMPT REPORT",
            f"schema_version={SCHEMA_VERSION}",
            f"activation={raw.activation}",
            "EPISTEMIC LIMITATIONS:",
            *[f"  - {item}" for item in EPISTEMIC_LIMITATIONS],
            "validation_ok=false",
            "synthetic_complete=false",
            "STATUS: incomplete (validation failed; completion language forbidden)",
            "diagnosed_faults:",
            *fault_lines,
            "residual_obligations_from_raw_discharges:",
            *residual_lines,
            "NOTE: this failure report may be incomplete but never claims completion.",
            "NOTE: bounded search and accepted judgment are never printed as proof.",
            "NOTE: no scalar parity percentage is defined or printed.",
        ]
    )
    if "synthetic-complete" in text or "STATUS: COMPLETE" in text:
        raise ClaimGuardError("failure_report_emitted_completion_language")
    return {
        "schema_version": SCHEMA_VERSION,
        "activation": raw.activation,
        "validation_ok": False,
        "synthetic_complete": False,
        "authentication_claim": False,
        "content_digest_cryptographic": False,
        "lean_python_agreement_is_authentication": False,
        "cross_language_generation_from_lean": False,
        "epistemic_limitations": list(EPISTEMIC_LIMITATIONS),
        "diagnosed_faults": list(faults),
        "residual_obligations": residuals,
        "rendered_report": text,
        "trust_boundary": (
            "synthetic-non-release mirror of canonical encoding; "
            "not a release guard; not cryptographic authentication"
        ),
    }


def regenerate_claim(raw: RawClaimInputs) -> dict[str, Any]:
    faults = validate_raw(raw)
    if faults:
        raise ClaimGuardError(faults[0], ",".join(faults))
    claim_identity = compute_claim_identity(raw)
    residuals = derive_residuals(raw)
    report = render_report(raw, claim_identity, residuals)
    unresolved = any(r["kind"] in {"open", "blocked", "refuted"} for r in residuals)
    denom_empty = not raw.inventory or not raw.requirement_entries
    judging_identity = compute_judging_closure_identity(
        raw.judging_nodes, raw.judging_roots
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "activation": raw.activation,
        "epistemic_limitations": list(EPISTEMIC_LIMITATIONS),
        "authentication_claim": False,
        "content_digest_cryptographic": False,
        "lean_python_agreement_is_authentication": False,
        "cross_language_generation_from_lean": False,
        "trust_boundary": (
            "synthetic-non-release mirror of canonical encoding; "
            "not a release guard; not cryptographic authentication"
        ),
        "claim_identity": claim_identity,
        "judging_closure_identity": judging_identity,
        "selected_promoted_event": raw.selected_promoted_event,
        "scope": {
            "target": raw.target,
            "profile": raw.profile,
            "source_closure": raw.source_closure,
            "subject_tree": raw.subject_tree,
            "runtime_artifact": raw.runtime_artifact,
            "runtime_source_tree": raw.runtime_source_tree,
            "environments": sorted(
                {e["id"] for d in raw.discharges for e in d["environments"]}
            ),
            "assurance_policy_content_id": policy_content_id(raw.assurance_policy),
        },
        "denominator": {
            "inventory": sorted(raw.inventory),
            "requirements": sorted(
                e["requirement_id"] for e in raw.requirement_entries
            ),
            "exclusions": derived_exclusions(raw.dispositions),
            "tolerances": [],
        },
        "field_provenance": [
            {"name": f["name"], "label": provenance_label(f["provenance"])}
            for f in sorted(raw.field_provenance, key=lambda x: x["name"])
        ],
        "assurance_grades": [
            {
                "requirement": d["requirement_id"],
                "grade": d["assurance_state"]["kind"],
                "label": assurance_label(d["assurance_state"]),
            }
            for d in sorted(raw.discharges, key=lambda x: x["requirement_id"])
        ],
        "residual_obligations": residuals,
        "synthetic_complete": (
            raw.activation == ACTIVATION and not denom_empty and not unresolved
        ),
        "rendered_report": report,
    }


def attempt_claim(raw: RawClaimInputs) -> dict[str, Any]:
    """Always produce a report. Failures never emit completion language."""
    faults = validate_raw(raw)
    if faults:
        return render_failure_report(raw, faults)
    return regenerate_claim(raw)


def verify_claim_artifact(raw: RawClaimInputs, artifact: dict[str, Any]) -> dict[str, Any]:
    """Regenerate from raw inputs; require exact canonical artifact equality.

    Ground truth is the regenerated claim. A supplied claim_identity is never
    trusted as its own verifier input. Partial field compares are insufficient.
    """
    expected = regenerate_claim(raw)
    if canonical_json(artifact) != canonical_json(expected):
        # Emit a useful primary diagnosis when identity/complete diverge.
        if artifact.get("claim_identity") != expected["claim_identity"]:
            raise ClaimGuardError(
                "stale_or_forged_claim_identity",
                "supplied identity does not match identity derived from raw inputs",
            )
        if artifact.get("synthetic_complete") != expected["synthetic_complete"]:
            raise ClaimGuardError("handwritten_complete_mismatch")
        if artifact.get("rendered_report") != expected["rendered_report"]:
            raise ClaimGuardError("handwritten_or_stale_rendered_report")
        if artifact.get("residual_obligations") != expected["residual_obligations"]:
            raise ClaimGuardError("hidden_or_altered_residual_obligations")
        if artifact.get("assurance_grades") != expected["assurance_grades"]:
            raise ClaimGuardError("assurance_grade_mismatch")
        if artifact.get("denominator") != expected["denominator"]:
            raise ClaimGuardError("denominator_mismatch")
        if artifact.get("scope") != expected["scope"]:
            raise ClaimGuardError("tampered_structured_scope")
        if artifact.get("epistemic_limitations") != expected["epistemic_limitations"]:
            raise ClaimGuardError("tampered_epistemic_limitations")
        if artifact.get("trust_boundary") != expected["trust_boundary"]:
            raise ClaimGuardError("tampered_trust_boundary")
        if artifact.get("cross_language_generation_from_lean") is not False:
            raise ClaimGuardError("false_cross_language_generation_claim")
        if artifact.get("authentication_claim") is True:
            raise ClaimGuardError("false_authentication_claim")
        if artifact.get("content_digest_cryptographic") is True:
            raise ClaimGuardError("false_cryptographic_claim")
        raise ClaimGuardError(
            "artifact_not_canonical",
            "regenerated artifact differs from supplied artifact",
        )
    if artifact.get("activation") != ACTIVATION:
        raise ClaimGuardError("wrong_activation")
    return {
        "ok": True,
        "reason": "regenerated_from_raw_inputs_exact_equality",
        "claim_identity": expected["claim_identity"],
        "authentication_claim": False,
    }


def _toy_imported() -> dict[str, Any]:
    return {
        "kind": "imported",
        "id": "import.upstream.closure",
        "source_closure": "src-hash",
    }


def _toy_derived() -> dict[str, Any]:
    return {
        "kind": "derived",
        "id": "derived.status",
        "verifier": "verifier.pilot-status",
        "input_closure": "in-hash",
    }


def _toy_calibrated() -> dict[str, Any]:
    return {
        "kind": "calibrated",
        "id": "cal.helpers",
        "campaign": "cal-hash",
        "fault_model": "missing-public-name",
    }


def _toy_judgment() -> dict[str, Any]:
    return {
        "kind": "judgment",
        "id": "judgment.scope.demo",
        "authority": "owner",
        "scope": "scope-hash",
        "invalidation": "inv-hash",
    }


def _toy_node(
    node_id: str,
    category: str,
    content: str,
    deps: list[str],
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "category": category,
        "content": content,
        "dependencies": deps,
        "provenance": provenance or _toy_imported(),
    }


def toy_raw_inputs() -> RawClaimInputs:
    """Unauthenticated synthetic one-requirement fixture (non-release)."""
    proved = {
        "kind": "proved",
        "id": "proof.adeq.demo",
        "theorem_name": "D_and_S_entails_R",
    }
    imported = _toy_imported()
    derived = _toy_derived()
    calibrated = _toy_calibrated()
    judgment = _toy_judgment()
    target = {
        "id": "target.toy",
        "repository": "github.com/tinygrad/tinygrad",
        "revision": "19c4d736",
        "source_closure": "toy-source-closure",
        "profile": "profile.toy.v1",
        "disposition": "promoted",
    }
    # Subject revision must equal the selected promoted candidate commit.
    subject = {
        "revision": "commit-c-candidate",
        "content": "subj-tree",
        "dirty": False,
    }
    env = {"id": "env.macos-metal", "digest": "env-hash"}
    policy = {
        "profile_id": "profile.toy.v1",
        "version": 1,
        "rules": [
            {"obligation_class": cls, "rule": "requireProof"}
            for cls in [
                "adequacy",
                "catalogClosure",
                "performanceQualification",
                "requirementDischarge",
                "scenarioObservation",
            ]
        ],
    }
    entry = {
        "requirement_id": "REQ-TOY-IMPORT",
        "requirement_semantics": "req-sem",
        "specification_id": "SPEC-TOY",
        "specification_semantics": "spec-sem",
        "boundary_id": "BND-TOY",
        "boundary_semantics": "bnd-sem",
        "scenario_id": "SCN-TOY",
        "scenario_digest": "scn-hash",
        "relation_id": "REL-TOY",
        "relation_digest": "rel-hash",
        "adapter_id": "ADP-TOY",
        "adapter_digest": "adp-hash",
        "validator_id": "VAL-TOY",
        "validator_version": "val-v1",
        "calibration_id": "CAL-TOY",
        "calibration_campaign": "cal-hash",
        "calibration_fault_model": "missing-public-name",
        "environments": [env],
    }
    discharge = {
        **{k: entry[k] for k in entry if k != "environments"},
        "target": target,
        "profile": "profile.toy.v1",
        "subject_tree": subject,
        "runtime_artifact": "runtime-dylib",
        "runtime_source_tree": subject,
        "environments": [env],
        "evidence": [{"id": "ev.toy.1", "digest": "ev-hash"}],
        "purpose": "semanticCompatibility",
        "assurance_state": proved,
    }
    scope_content = digest_token(
        encode_frozen_catalog_binding(
            target,
            "profile.toy.v1",
            "toy-source-closure",
            ["src.helpers"],
            [
                {
                    "item": "src.helpers",
                    "disposition": {"kind": "required", "requirement": "REQ-TOY-IMPORT"},
                }
            ],
            [entry],
            [],
            policy,
        )
    )
    renderer_content = digest_token(claim_renderer_binding_payload())
    judging_nodes = [
        _toy_node("n.claimRenderer", "claimRenderer", renderer_content, ["n.toolchain"]),
        _toy_node("n.toolchain", "toolchain", "tc", ["n.envPolicy"]),
        _toy_node("n.envPolicy", "environmentPolicy", "ep", ["n.helper"]),
        _toy_node("n.helper", "importedHelper", "ih", ["n.cal"]),
        _toy_node("n.cal", "calibrationPolicy", "cp", ["n.val"]),
        _toy_node("n.val", "validator", "vl", ["n.scn"]),
        _toy_node("n.scn", "scenarioGenerator", "sg", ["n.rel"]),
        _toy_node("n.rel", "relation", "rl", ["n.adp"]),
        _toy_node("n.adp", "adapter", "ad", ["n.bnd"]),
        _toy_node("n.bnd", "boundaryDenotation", "bd", ["n.req"]),
        _toy_node("n.req", "requirementDenotation", "rd", ["n.tgt"]),
        _toy_node("n.tgt", "targetProfileDecision", scope_content, []),
    ]
    judging_roots = ["n.claimRenderer"]
    judging_identity = compute_judging_closure_identity(judging_nodes, judging_roots)
    ancestry = [
        {
            "commit": "commit-a-freeze",
            "parents_within_capture": [],
            "judging_closure_identity": judging_identity,
        },
        {
            "commit": "commit-b-middle",
            "parents_within_capture": ["commit-a-freeze"],
            "judging_closure_identity": judging_identity,
        },
        {
            "commit": "commit-c-candidate",
            "parents_within_capture": ["commit-b-middle"],
            "judging_closure_identity": judging_identity,
        },
    ]
    freeze_integrity = {
        "claimed_kind": "retrospectiveFreezeIntegrity",
        "prospective_protocol": None,
        "manifest": {
            "capture_identity": "capture.claim.v1",
            "extractor_identity": "extractor-claim",
            "target": target,
            "profile": "profile.toy.v1",
            "frozen_judging_closure_identity": judging_identity,
            "freeze": "commit-a-freeze",
            "candidate": "commit-c-candidate",
            "captured_commit_ids": [
                "commit-a-freeze",
                "commit-b-middle",
                "commit-c-candidate",
            ],
            "records": copy.deepcopy(ancestry),
        },
        "supplied_snapshots": copy.deepcopy(ancestry),
    }
    promoted = {
        "id": "cycle.claim.promoted.1",
        "target": target,
        "profile": "profile.toy.v1",
        "expected_frozen_judging_closure_identity": judging_identity,
        "outcome": "promoted",
        "freeze": "commit-a-freeze",
        "candidate": "commit-c-candidate",
    }
    rejected = {
        "id": "cycle.claim.rejected.1",
        "target": target,
        "profile": "profile.toy.v1",
        "expected_frozen_judging_closure_identity": judging_identity,
        "outcome": "rejected",
        "freeze": "commit-a-freeze",
        "candidate": "commit-c-candidate",
    }
    foreign = {
        "id": "cycle.foreign.rejected.1",
        "target": {
            **target,
            "id": "target.foreign",
            "profile": "profile.foreign.v1",
        },
        "profile": "profile.foreign.v1",
        "expected_frozen_judging_closure_identity": "foreign-judging-shape",
        "outcome": "rejected",
        "freeze": "commit-a-freeze",
        "candidate": "commit-c-candidate",
    }
    fields = [
        {"name": "target", "provenance": imported},
        {"name": "profile", "provenance": judgment},
        {"name": "sourceClosure", "provenance": imported},
        {"name": "subjectTree", "provenance": derived},
        {"name": "runtime", "provenance": derived},
        {"name": "assurancePolicy", "provenance": judgment},
        {"name": "judgingClosureIdentity", "provenance": derived},
        {"name": "claimRenderer", "provenance": derived},
        {"name": "catalogDenominator", "provenance": imported},
        {"name": "discharges", "provenance": calibrated},
        {"name": "cycleRegistry", "provenance": derived},
        {"name": "activation", "provenance": judgment},
        {"name": "selectedPromotedEvent", "provenance": derived},
    ]
    return RawClaimInputs(
        activation=ACTIVATION,
        target=target,
        profile="profile.toy.v1",
        source_closure="toy-source-closure",
        subject_tree=subject,
        runtime_artifact="runtime-dylib",
        runtime_source_tree=subject,
        assurance_policy=policy,
        inventory=["src.helpers"],
        dispositions=[
            {
                "item": "src.helpers",
                "disposition": {"kind": "required", "requirement": "REQ-TOY-IMPORT"},
            }
        ],
        requirement_entries=[entry],
        accepted_judgments=[],
        discharges=[discharge],
        judging_nodes=judging_nodes,
        judging_roots=judging_roots,
        judging_discovered_inventory=[n["id"] for n in judging_nodes],
        claim_renderer={
            "id": "n.claimRenderer",
            "content": renderer_content,
            "provenance": derived,
        },
        cycle_descriptors=[promoted, rejected, foreign],
        cycle_entries=[
            {"descriptor": promoted, "freeze_integrity": freeze_integrity},
            {"descriptor": rejected, "freeze_integrity": None},
            {"descriptor": foreign, "freeze_integrity": None},
        ],
        selected_promoted_event="cycle.claim.promoted.1",
        field_provenance=fields,
    )


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(canonical_json(data), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def raw_to_dict(raw: RawClaimInputs) -> dict[str, Any]:
    return {
        "activation": raw.activation,
        "target": raw.target,
        "profile": raw.profile,
        "source_closure": raw.source_closure,
        "subject_tree": raw.subject_tree,
        "runtime_artifact": raw.runtime_artifact,
        "runtime_source_tree": raw.runtime_source_tree,
        "assurance_policy": raw.assurance_policy,
        "inventory": raw.inventory,
        "dispositions": raw.dispositions,
        "requirement_entries": raw.requirement_entries,
        "accepted_judgments": raw.accepted_judgments,
        "discharges": raw.discharges,
        "judging_nodes": raw.judging_nodes,
        "judging_roots": raw.judging_roots,
        "judging_discovered_inventory": raw.judging_discovered_inventory,
        "claim_renderer": raw.claim_renderer,
        "cycle_descriptors": raw.cycle_descriptors,
        "cycle_entries": raw.cycle_entries,
        "selected_promoted_event": raw.selected_promoted_event,
        "field_provenance": raw.field_provenance,
    }


def demonstrate_regeneration(root: Path | None = None) -> dict[str, Any]:
    if root is None:
        with tempfile.TemporaryDirectory(prefix="tgrad-generated-claim-") as tmp:
            return demonstrate_regeneration(Path(tmp))

    raw = toy_raw_inputs()
    raw_path = root / "raw_inputs.json"
    claim_path = root / "claim.json"
    write_json(raw_path, raw_to_dict(raw))
    claim = regenerate_claim(raw)
    write_json(claim_path, claim)
    verify = verify_claim_artifact(raw, claim)
    return {
        "role": "python-generated-claim-regenerator",
        "authentication_claim": False,
        "content_digest_cryptographic": False,
        "cross_language_generation_from_lean": False,
        "activation": ACTIVATION,
        "trust_boundary": (
            "synthetic-non-release mirror of canonical encoding; "
            "not a release guard; not cryptographic authentication"
        ),
        "raw_path": str(raw_path),
        "claim_path": str(claim_path),
        "claim_identity_sha256": hashlib.sha256(
            claim["claim_identity"].encode()
        ).hexdigest(),
        "verify": verify,
        "synthetic_complete": claim["synthetic_complete"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate/verify synthetic generated-claim artifacts from raw "
            "inputs. Does not trust supplied digests as ground truth."
        )
    )
    parser.add_argument(
        "--emit-toy",
        metavar="DIR",
        help="write toy raw inputs + regenerated claim under DIR (or tempfile)",
    )
    parser.add_argument("--raw", type=Path, help="raw synthetic inputs JSON")
    parser.add_argument(
        "--check",
        type=Path,
        help="claim artifact to verify against regeneration from --raw",
    )
    parser.add_argument(
        "--write-claim",
        type=Path,
        help="write regenerated claim artifact to this path",
    )
    parser.add_argument("--json", action="store_true", help="print JSON result")
    args = parser.parse_args()

    if args.emit_toy is not None:
        root = Path(args.emit_toy)
        root.mkdir(parents=True, exist_ok=True)
        result = demonstrate_regeneration(root)
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(result["trust_boundary"])
            print(f"verify={result['verify']}")
        return 0

    if args.raw is None:
        result = demonstrate_regeneration()
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(
                "synthetic regeneration ok; "
                f"activation={result['activation']} "
                f"auth={result['authentication_claim']}"
            )
        return 0

    raw = RawClaimInputs.from_dict(load_json(args.raw))
    claim = regenerate_claim(raw)
    if args.write_claim is not None:
        write_json(args.write_claim, claim)
    if args.check is not None:
        artifact = load_json(args.check)
        verify_claim_artifact(raw, artifact)
        if args.json:
            print(
                json.dumps(
                    {"ok": True, "claim_identity": claim["claim_identity"]}, indent=2
                )
            )
        else:
            print("ok: regenerated claim matches artifact")
        return 0
    if args.json:
        print(json.dumps(claim, indent=2, sort_keys=True))
    else:
        print(claim["rendered_report"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ClaimGuardError as exc:
        print(f"REJECT {exc.reason}" + (f" ({exc.detail})" if exc.detail else ""))
        raise SystemExit(1) from exc
