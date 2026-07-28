#!/usr/bin/env python3
"""Deterministic mutation tests for scripts/contract/check_generated_claims.py.

Each failure asserts the intended diagnosis reason. Uses tempfile only.
CPU-only; read-only against the Tgrad repository checkout.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

_CHECKER = Path(__file__).resolve().parent / "check_generated_claims.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_generated_claims", _CHECKER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {_CHECKER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


claim = _load_checker()


def _raw_dict() -> dict:
    return claim.raw_to_dict(claim.toy_raw_inputs())


def _rebind_frozen_catalog(data: dict) -> dict:
    """Recompute frozen catalog binding, judging identity, and cycle closures."""
    scope = claim.digest_token(
        claim.encode_frozen_catalog_binding(
            data["target"],
            data["profile"],
            data["source_closure"],
            data["inventory"],
            data["dispositions"],
            data["requirement_entries"],
            data["accepted_judgments"],
            data["assurance_policy"],
        )
    )
    for node in data["judging_nodes"]:
        if node["category"] == "targetProfileDecision":
            node["content"] = scope
    jid = claim.compute_judging_closure_identity(
        data["judging_nodes"], data["judging_roots"]
    )
    for desc in data["cycle_descriptors"]:
        if desc["id"].startswith("cycle.claim."):
            desc["expected_frozen_judging_closure_identity"] = jid
    for entry in data["cycle_entries"]:
        desc = entry["descriptor"]
        if desc["id"].startswith("cycle.claim."):
            desc["expected_frozen_judging_closure_identity"] = jid
        fi = entry.get("freeze_integrity")
        if fi is not None:
            fi["manifest"]["frozen_judging_closure_identity"] = jid
            for row in fi["manifest"]["records"]:
                row["judging_closure_identity"] = jid
            for row in fi["supplied_snapshots"]:
                row["judging_closure_identity"] = jid
    return data


def _with_search_policy(data: dict) -> dict:
    for rule in data["assurance_policy"]["rules"]:
        if rule["obligation_class"] == "requirementDischarge":
            rule["rule"] = "acceptBoundedSearch"
    return _rebind_frozen_catalog(data)


class GeneratedClaimGuardTests(unittest.TestCase):
    def test_positive_regeneration_and_verify(self) -> None:
        raw = claim.toy_raw_inputs()
        artifact = claim.regenerate_claim(raw)
        self.assertEqual(artifact["activation"], "synthetic-non-release")
        self.assertIs(artifact["authentication_claim"], False)
        self.assertIs(artifact["content_digest_cryptographic"], False)
        self.assertIs(artifact["lean_python_agreement_is_authentication"], False)
        self.assertIs(artifact["cross_language_generation_from_lean"], False)
        self.assertTrue(artifact["synthetic_complete"])
        self.assertIn("shape token", artifact["rendered_report"])
        self.assertIn("NON-RELEASE", artifact["rendered_report"])
        self.assertIn("field_provenance:", artifact["rendered_report"])
        self.assertIn("subject_candidate_binding=", artifact["rendered_report"])
        self.assertNotIn("parity%", artifact["rendered_report"])
        self.assertEqual(artifact["residual_obligations"], [])
        self.assertEqual(
            raw.subject_tree["revision"],
            next(
                d["candidate"]
                for d in raw.cycle_descriptors
                if d["id"] == raw.selected_promoted_event
            ),
        )
        verify = claim.verify_claim_artifact(raw, artifact)
        self.assertTrue(verify["ok"])
        self.assertEqual(verify["reason"], "regenerated_from_raw_inputs_exact_equality")

    def test_order_independence_of_semantic_sets(self) -> None:
        raw_a = claim.toy_raw_inputs()
        data = _raw_dict()
        data["field_provenance"] = list(reversed(data["field_provenance"]))
        data["cycle_descriptors"] = list(reversed(data["cycle_descriptors"]))
        data["cycle_entries"] = list(reversed(data["cycle_entries"]))
        data["inventory"] = list(reversed(data["inventory"]))
        raw_b = claim.RawClaimInputs.from_dict(data)
        id_a = claim.compute_claim_identity(raw_a)
        id_b = claim.compute_claim_identity(raw_b)
        self.assertEqual(id_a, id_b)
        self.assertEqual(
            claim.regenerate_claim(raw_a)["claim_identity"],
            claim.regenerate_claim(raw_b)["claim_identity"],
        )

    def test_alternate_valid_global_registry_entry_preserved(self) -> None:
        raw = claim.toy_raw_inputs()
        self.assertTrue(
            any(d["id"] == "cycle.foreign.rejected.1" for d in raw.cycle_descriptors)
        )
        art = claim.regenerate_claim(raw)
        self.assertTrue(art["synthetic_complete"])

    def test_handwritten_complete_rejected(self) -> None:
        raw = claim.toy_raw_inputs()
        artifact = claim.regenerate_claim(raw)
        forged = copy.deepcopy(artifact)
        forged["synthetic_complete"] = True
        forged["residual_obligations"] = []
        forged["rendered_report"] = artifact["rendered_report"].replace(
            "STATUS: synthetic-complete",
            "STATUS: COMPLETE",
        )
        with self.assertRaises(claim.ClaimGuardError) as ctx:
            claim.verify_claim_artifact(raw, forged)
        self.assertEqual(ctx.exception.reason, "handwritten_or_stale_rendered_report")

        forged2 = copy.deepcopy(artifact)
        forged2["synthetic_complete"] = not artifact["synthetic_complete"]
        with self.assertRaises(claim.ClaimGuardError) as ctx2:
            claim.verify_claim_artifact(raw, forged2)
        self.assertEqual(ctx2.exception.reason, "handwritten_complete_mismatch")

    def test_handwritten_release_claim_rejected(self) -> None:
        """Mechanical guard rejects release activation; does not trust output+digest."""
        raw = claim.RawClaimInputs.from_dict({**_raw_dict(), "activation": "release"})
        with self.assertRaises(claim.ClaimGuardError) as ctx:
            claim.regenerate_claim(raw)
        self.assertEqual(ctx.exception.reason, "wrong_activation")

        # Colluding output+digest for a release story still fails exact equality
        # against regeneration from honest raw.
        honest_raw = claim.toy_raw_inputs()
        honest = claim.regenerate_claim(honest_raw)
        release_story = copy.deepcopy(honest)
        release_story["activation"] = "release"
        release_story["synthetic_complete"] = True
        release_story["claim_identity"] = "authored-release-digest"
        with self.assertRaises(claim.ClaimGuardError) as ctx2:
            claim.verify_claim_artifact(honest_raw, release_story)
        self.assertEqual(ctx2.exception.reason, "stale_or_forged_claim_identity")

    def test_stale_derived_identity_rejected(self) -> None:
        raw = claim.toy_raw_inputs()
        artifact = claim.regenerate_claim(raw)
        artifact["claim_identity"] = "stale-handwritten-identity"
        with self.assertRaises(claim.ClaimGuardError) as ctx:
            claim.verify_claim_artifact(raw, artifact)
        self.assertEqual(ctx.exception.reason, "stale_or_forged_claim_identity")

    def test_missing_requirement_discharge_rejected(self) -> None:
        data = _raw_dict()
        data["discharges"] = []
        raw = claim.RawClaimInputs.from_dict(data)
        with self.assertRaises(claim.ClaimGuardError) as ctx:
            claim.regenerate_claim(raw)
        self.assertEqual(ctx.exception.reason, "missing_or_extra_discharge")

    def test_altered_discharge_semantics_rejected(self) -> None:
        data = _raw_dict()
        data["discharges"][0]["specification_semantics"] = "SPEC-SEM-ATTACK"
        raw = claim.RawClaimInputs.from_dict(data)
        with self.assertRaises(claim.ClaimGuardError) as ctx:
            claim.regenerate_claim(raw)
        self.assertEqual(
            ctx.exception.reason, "discharge_catalog_mismatch:specification_semantics"
        )

    def test_extra_provenance_field_rejected(self) -> None:
        data = _raw_dict()
        data["field_provenance"].append(
            {
                "name": "sneaky",
                "provenance": data["field_provenance"][0]["provenance"],
            }
        )
        raw = claim.RawClaimInputs.from_dict(data)
        with self.assertRaises(claim.ClaimGuardError) as ctx:
            claim.regenerate_claim(raw)
        self.assertEqual(ctx.exception.reason, "extra_field_provenance:sneaky")

    def test_promotion_tree_mismatch_rejected(self) -> None:
        data = _raw_dict()
        other = {"revision": "unrelated-rev", "content": "subj-tree", "dirty": False}
        data["subject_tree"] = other
        data["runtime_source_tree"] = other
        data["discharges"][0]["subject_tree"] = other
        data["discharges"][0]["runtime_source_tree"] = other
        raw = claim.RawClaimInputs.from_dict(data)
        with self.assertRaises(claim.ClaimGuardError) as ctx:
            claim.regenerate_claim(raw)
        self.assertEqual(ctx.exception.reason, "subject_candidate_mismatch")

    def test_hidden_exclusion_derived_not_authored(self) -> None:
        data = _raw_dict()
        data["inventory"].append("src.hidden")
        data["dispositions"].append(
            {
                "item": "src.hidden",
                "disposition": {"kind": "excluded", "judgment": "j.hide"},
            }
        )
        data["accepted_judgments"].append(
            {
                "id": "j.hide",
                "authority": "owner",
                "scope": "s",
                "invalidation": "i",
            }
        )
        # Authored exclusions key is forbidden in raw schema.
        with self.assertRaises(claim.ClaimGuardError) as ctx_authored:
            claim.RawClaimInputs.from_dict({**data, "exclusions": []})
        self.assertEqual(ctx_authored.exception.reason, "authored_conclusion_field")

        # Catalog denotation changed: rebound frozen judging binding + chronology.
        _rebind_frozen_catalog(data)

        raw = claim.RawClaimInputs.from_dict(data)
        art = claim.regenerate_claim(raw)
        self.assertIn("src.hidden->j.hide", art["denominator"]["exclusions"])
        excl_line = [
            line
            for line in art["rendered_report"].splitlines()
            if line.startswith("exclusions=")
        ][0]
        self.assertIn("src.hidden->j.hide", excl_line)

    def test_tampered_structured_scope_rejected(self) -> None:
        raw = claim.toy_raw_inputs()
        honest = claim.regenerate_claim(raw)
        mut = copy.deepcopy(honest)
        mut["scope"] = copy.deepcopy(mut["scope"])
        mut["scope"]["target"] = copy.deepcopy(mut["scope"]["target"])
        mut["scope"]["target"]["id"] = "target.ATTACK"
        with self.assertRaises(claim.ClaimGuardError) as ctx:
            claim.verify_claim_artifact(raw, mut)
        self.assertEqual(ctx.exception.reason, "tampered_structured_scope")

    def test_unknown_raw_field_rejected(self) -> None:
        data = _raw_dict()
        data["colluding_hidden_complete"] = True
        with self.assertRaises(claim.ClaimGuardError) as ctx:
            claim.RawClaimInputs.from_dict(data)
        self.assertEqual(ctx.exception.reason, "unknown_raw_field")

    def test_authored_judging_identity_and_exclusions_rejected(self) -> None:
        data = _raw_dict()
        data["judging_closure_identity"] = "authored-token"
        with self.assertRaises(claim.ClaimGuardError) as ctx:
            claim.RawClaimInputs.from_dict(data)
        self.assertEqual(ctx.exception.reason, "authored_conclusion_field")

    def test_changed_target_rejected(self) -> None:
        data = _raw_dict()
        data["target"] = {**data["target"], "id": "target.other"}
        # Keep selected chronology on old target → chronology mismatch.
        raw = claim.RawClaimInputs.from_dict(data)
        with self.assertRaises(claim.ClaimGuardError) as ctx:
            claim.regenerate_claim(raw)
        self.assertIn(
            ctx.exception.reason,
            {
                "mismatched_discharge_target",
                "mismatched_chronology_target",
                "mismatched_judging_scope_binding",
            },
        )

    def test_changed_profile_rejected(self) -> None:
        data = _raw_dict()
        data["profile"] = "profile.other.v1"
        raw = claim.RawClaimInputs.from_dict(data)
        with self.assertRaises(claim.ClaimGuardError) as ctx:
            claim.regenerate_claim(raw)
        self.assertIn(
            ctx.exception.reason,
            {
                "mismatched_target_profile",
                "mismatched_policy_profile",
                "mismatched_discharge_profile",
                "mismatched_chronology_profile",
                "mismatched_judging_scope_binding",
            },
        )

    def test_changed_source_closure_rejected(self) -> None:
        data = _raw_dict()
        data["source_closure"] = "other-source-closure"
        # Rebound judging scope so the primary diagnosis is source-closure drift
        # against target.source_closure, not the scope-binding secondary.
        scope = claim.digest_token(
            claim.encode_frozen_catalog_binding(
                data["target"],
                data["profile"],
                data["source_closure"],
                data["inventory"],
                data["dispositions"],
                data["requirement_entries"],
                data["accepted_judgments"],
                data["assurance_policy"],
            )
        )
        for node in data["judging_nodes"]:
            if node["category"] == "targetProfileDecision":
                node["content"] = scope
        raw = claim.RawClaimInputs.from_dict(data)
        with self.assertRaises(claim.ClaimGuardError) as ctx:
            claim.regenerate_claim(raw)
        self.assertEqual(ctx.exception.reason, "mismatched_source_closure")

    def test_changed_subject_tree_changes_identity(self) -> None:
        raw_a = claim.toy_raw_inputs()
        data = _raw_dict()
        other = {
            "revision": "commit-c-candidate",
            "content": "other-tree",
            "dirty": False,
        }
        data["subject_tree"] = other
        data["runtime_source_tree"] = other
        data["discharges"][0]["subject_tree"] = other
        data["discharges"][0]["runtime_source_tree"] = other
        raw_b = claim.RawClaimInputs.from_dict(data)
        self.assertNotEqual(
            claim.compute_claim_identity(raw_a),
            claim.compute_claim_identity(raw_b),
        )
        self.assertTrue(claim.regenerate_claim(raw_b)["synthetic_complete"])

    def test_changed_runtime_changes_identity(self) -> None:
        raw_a = claim.toy_raw_inputs()
        data = _raw_dict()
        data["runtime_artifact"] = "runtime-OTHER"
        data["discharges"][0]["runtime_artifact"] = "runtime-OTHER"
        raw_b = claim.RawClaimInputs.from_dict(data)
        self.assertNotEqual(
            claim.compute_claim_identity(raw_a),
            claim.compute_claim_identity(raw_b),
        )

    def test_unchanged_closure_changed_policy_rejected(self) -> None:
        data = _raw_dict()
        data["assurance_policy"] = {**data["assurance_policy"], "version": 2}
        # Intentionally do NOT rebound judging scope node.
        raw = claim.RawClaimInputs.from_dict(data)
        with self.assertRaises(claim.ClaimGuardError) as ctx:
            claim.regenerate_claim(raw)
        self.assertEqual(ctx.exception.reason, "mismatched_judging_scope_binding")

    def test_post_freeze_catalog_denotation_drift_rejected(self) -> None:
        """Mutate catalog+discharge semantics together; leave judging/cycle frozen."""
        data = _raw_dict()
        data["requirement_entries"][0]["requirement_semantics"] = "mutated-after-freeze"
        data["discharges"][0]["requirement_semantics"] = "mutated-after-freeze"
        raw = claim.RawClaimInputs.from_dict(data)
        with self.assertRaises(claim.ClaimGuardError) as ctx:
            claim.regenerate_claim(raw)
        self.assertEqual(ctx.exception.reason, "mismatched_judging_scope_binding")

    def test_post_freeze_denotation_rebound_changes_identity(self) -> None:
        raw_a = claim.toy_raw_inputs()
        data = _raw_dict()
        data["requirement_entries"][0]["requirement_semantics"] = "mutated-after-freeze"
        data["discharges"][0]["requirement_semantics"] = "mutated-after-freeze"
        scope = claim.digest_token(
            claim.encode_frozen_catalog_binding(
                data["target"],
                data["profile"],
                data["source_closure"],
                data["inventory"],
                data["dispositions"],
                data["requirement_entries"],
                data["accepted_judgments"],
                data["assurance_policy"],
            )
        )
        for node in data["judging_nodes"]:
            if node["category"] == "targetProfileDecision":
                node["content"] = scope
        jid = claim.compute_judging_closure_identity(
            data["judging_nodes"], data["judging_roots"]
        )
        for desc in data["cycle_descriptors"]:
            if desc["id"].startswith("cycle.claim."):
                desc["expected_frozen_judging_closure_identity"] = jid
        for entry in data["cycle_entries"]:
            desc = entry["descriptor"]
            if desc["id"].startswith("cycle.claim."):
                desc["expected_frozen_judging_closure_identity"] = jid
            fi = entry.get("freeze_integrity")
            if fi is not None:
                fi["manifest"]["frozen_judging_closure_identity"] = jid
                for row in fi["manifest"]["records"]:
                    row["judging_closure_identity"] = jid
                for row in fi["supplied_snapshots"]:
                    row["judging_closure_identity"] = jid
        raw_b = claim.RawClaimInputs.from_dict(data)
        art_b = claim.regenerate_claim(raw_b)
        self.assertTrue(art_b["synthetic_complete"])
        self.assertNotEqual(
            claim.compute_claim_identity(raw_a),
            claim.compute_claim_identity(raw_b),
        )

    def test_post_freeze_evidence_only_still_validates(self) -> None:
        data = _raw_dict()
        data["discharges"][0]["evidence"] = [
            {"id": "ev.toy.POST", "digest": "ev-hash-POST"}
        ]
        raw = claim.RawClaimInputs.from_dict(data)
        art = claim.regenerate_claim(raw)
        self.assertTrue(art["synthetic_complete"])

    def test_missing_disposition_rejected_after_rebind(self) -> None:
        data = _raw_dict()
        data["inventory"] = list(data["inventory"]) + ["src.undisposed"]
        _rebind_frozen_catalog(data)
        raw = claim.RawClaimInputs.from_dict(data)
        with self.assertRaises(claim.ClaimGuardError) as ctx:
            claim.regenerate_claim(raw)
        self.assertEqual(ctx.exception.reason, "missing_disposition:src.undisposed")

    def test_extra_disposition_rejected(self) -> None:
        data = _raw_dict()
        data["dispositions"] = list(data["dispositions"]) + [
            {
                "item": "src.ghost",
                "disposition": {"kind": "required", "requirement": "REQ-TOY-IMPORT"},
            }
        ]
        _rebind_frozen_catalog(data)
        raw = claim.RawClaimInputs.from_dict(data)
        with self.assertRaises(claim.ClaimGuardError) as ctx:
            claim.regenerate_claim(raw)
        self.assertEqual(ctx.exception.reason, "extra_disposition:src.ghost")

    def test_unaccepted_exclusion_rejected(self) -> None:
        data = _raw_dict()
        data["inventory"] = list(data["inventory"]) + ["src.hidden"]
        data["dispositions"] = list(data["dispositions"]) + [
            {
                "item": "src.hidden",
                "disposition": {"kind": "excluded", "judgment": "j.unaccepted"},
            }
        ]
        # No matching accepted_judgments entry.
        _rebind_frozen_catalog(data)
        raw = claim.RawClaimInputs.from_dict(data)
        with self.assertRaises(claim.ClaimGuardError) as ctx:
            claim.regenerate_claim(raw)
        self.assertEqual(ctx.exception.reason, "unaccepted_exclusion:j.unaccepted")

    def test_missing_required_judging_category_rejected_after_rebind(self) -> None:
        data = _raw_dict()
        data["judging_nodes"] = [
            n for n in data["judging_nodes"] if n["category"] != "validator"
        ]
        for node in data["judging_nodes"]:
            node["dependencies"] = [d for d in node["dependencies"] if d != "n.val"]
            if node["id"] == "n.cal":
                node["dependencies"] = ["n.scn"]
        data["judging_discovered_inventory"] = [n["id"] for n in data["judging_nodes"]]
        _rebind_frozen_catalog(data)
        raw = claim.RawClaimInputs.from_dict(data)
        with self.assertRaises(claim.ClaimGuardError) as ctx:
            claim.regenerate_claim(raw)
        self.assertEqual(ctx.exception.reason, "missing_required_category:validator")

    def test_unreachable_judging_node_rejected(self) -> None:
        data = _raw_dict()
        data["judging_nodes"] = list(data["judging_nodes"]) + [
            {
                "id": "n.orphan",
                "category": "importedHelper",
                "content": "orphan",
                "dependencies": [],
                "provenance": data["judging_nodes"][0]["provenance"],
            }
        ]
        data["judging_discovered_inventory"] = [n["id"] for n in data["judging_nodes"]]
        _rebind_frozen_catalog(data)
        raw = claim.RawClaimInputs.from_dict(data)
        with self.assertRaises(claim.ClaimGuardError) as ctx:
            claim.regenerate_claim(raw)
        # Orphan is unreachable; may also trip duplicate category (ok) — primary
        # diagnosis after sort may be dependency/category related; require exact.
        self.assertEqual(ctx.exception.reason, "unreachable_node:n.orphan")

    def test_judging_dependency_cycle_rejected(self) -> None:
        data = _raw_dict()
        # Minimal two-node cycle on the leaf edge of the DAG.
        for node in data["judging_nodes"]:
            if node["id"] == "n.tgt":
                node["dependencies"] = ["n.req"]
            if node["id"] == "n.req":
                node["dependencies"] = ["n.tgt"]
        _rebind_frozen_catalog(data)
        raw = claim.RawClaimInputs.from_dict(data)
        with self.assertRaises(claim.ClaimGuardError) as ctx:
            claim.regenerate_claim(raw)
        self.assertTrue(
            ctx.exception.reason.startswith("dependency_cycle:"),
            ctx.exception.reason,
        )

    def test_freeze_interval_judging_closure_drift_rejected(self) -> None:
        """Mutation/reversion inside freeze..candidate must not complete."""
        data = _raw_dict()
        for entry in data["cycle_entries"]:
            fi = entry.get("freeze_integrity")
            if fi is None:
                continue
            for row in fi["manifest"]["records"]:
                if row["commit"] == "commit-b-middle":
                    row["judging_closure_identity"] = "MUTATED-IN-CYCLE"
            for row in fi["supplied_snapshots"]:
                if row["commit"] == "commit-b-middle":
                    row["judging_closure_identity"] = "MUTATED-IN-CYCLE"
        raw = claim.RawClaimInputs.from_dict(data)
        with self.assertRaises(claim.ClaimGuardError) as ctx:
            claim.regenerate_claim(raw)
        self.assertEqual(
            ctx.exception.reason,
            "freeze_integrity:cycle.claim.promoted.1:judging_closure_drift:commit-b-middle",
        )

    def test_incomplete_assurance_policy_rejected_after_rebind(self) -> None:
        data = _raw_dict()
        data["assurance_policy"] = {
            "profile_id": "profile.toy.v1",
            "version": 1,
            "rules": [
                {
                    "obligation_class": "requirementDischarge",
                    "rule": "requireProof",
                }
            ],
        }
        _rebind_frozen_catalog(data)
        raw = claim.RawClaimInputs.from_dict(data)
        with self.assertRaises(claim.ClaimGuardError) as ctx:
            claim.regenerate_claim(raw)
        # Primary sorted diagnosis is the first missing class / umbrella.
        self.assertIn(
            ctx.exception.reason,
            {
                "malformed_policy",
                "missing_policy_class:adequacy",
                "missing_policy_class:catalogClosure",
                "missing_policy_class:performanceQualification",
                "missing_policy_class:scenarioObservation",
            },
        )
        # Full fault set must include totality diagnosis.
        faults = claim.validate_raw(raw)
        self.assertIn("malformed_policy", faults)
        self.assertIn("missing_policy_class:adequacy", faults)
        self.assertEqual(len([f for f in faults if f.startswith("missing_policy_class:")]), 4)

    def test_duplicate_policy_class_rejected(self) -> None:
        data = _raw_dict()
        data["assurance_policy"] = {
            **data["assurance_policy"],
            "rules": data["assurance_policy"]["rules"]
            + [
                {
                    "obligation_class": "requirementDischarge",
                    "rule": "acceptBoundedSearch",
                }
            ],
        }
        _rebind_frozen_catalog(data)
        raw = claim.RawClaimInputs.from_dict(data)
        with self.assertRaises(claim.ClaimGuardError) as ctx:
            claim.regenerate_claim(raw)
        faults = claim.validate_raw(raw)
        self.assertIn("duplicate_policy_class:requirementDischarge", faults)
        self.assertIn("malformed_policy", faults)

    def test_unknown_acceptance_rule_rejected(self) -> None:
        data = _raw_dict()
        for rule in data["assurance_policy"]["rules"]:
            if rule["obligation_class"] == "requirementDischarge":
                rule["rule"] = "acceptAnything"
        _rebind_frozen_catalog(data)
        raw = claim.RawClaimInputs.from_dict(data)
        faults = claim.validate_raw(raw)
        self.assertIn("unknown_acceptance_rule:acceptAnything", faults)

    def test_freeze_not_interval_root_rejected(self) -> None:
        data = _raw_dict()
        for entry in data["cycle_entries"]:
            fi = entry.get("freeze_integrity")
            if fi is None:
                continue
            for row in fi["manifest"]["records"]:
                if row["commit"] == "commit-a-freeze":
                    row["parents_within_capture"] = ["commit-b-middle"]
            for row in fi["supplied_snapshots"]:
                if row["commit"] == "commit-a-freeze":
                    row["parents_within_capture"] = ["commit-b-middle"]
        raw = claim.RawClaimInputs.from_dict(data)
        faults = claim.validate_raw(raw)
        self.assertIn(
            "freeze_integrity:cycle.claim.promoted.1:freeze_not_interval_root",
            faults,
        )

    def test_snapshot_record_mismatch_rejected(self) -> None:
        data = _raw_dict()
        for entry in data["cycle_entries"]:
            fi = entry.get("freeze_integrity")
            if fi is None:
                continue
            # Mutate only snapshots, leave records honest.
            for row in fi["supplied_snapshots"]:
                if row["commit"] == "commit-b-middle":
                    row["judging_closure_identity"] = "SNAPSHOT-ONLY-MUTATION"
        raw = claim.RawClaimInputs.from_dict(data)
        faults = claim.validate_raw(raw)
        self.assertIn(
            "freeze_integrity:cycle.claim.promoted.1:snapshot_inventory_mismatch",
            faults,
        )

    def test_empty_requirement_environments_rejected_after_rebind(self) -> None:
        data = _raw_dict()
        data["requirement_entries"][0]["environments"] = []
        data["discharges"][0]["environments"] = []
        _rebind_frozen_catalog(data)
        raw = claim.RawClaimInputs.from_dict(data)
        with self.assertRaises(claim.ClaimGuardError) as ctx:
            claim.regenerate_claim(raw)
        faults = claim.validate_raw(raw)
        self.assertIn(
            "requirement_entry:REQ-TOY-IMPORT:empty_environments", faults
        )
        self.assertIn("malformed_requirement_entry", faults)
        self.assertIn(
            "discharge:REQ-TOY-IMPORT:empty_environments", faults
        )
        self.assertIn(
            ctx.exception.reason,
            {
                "malformed_requirement_entry",
                "requirement_entry:REQ-TOY-IMPORT:empty_environments",
                "discharge:REQ-TOY-IMPORT:empty_environments",
            },
        )

    def test_rebound_policy_changes_identity(self) -> None:
        raw_a = claim.toy_raw_inputs()
        data = _with_search_policy(_raw_dict())
        # Keep proved discharge under requireProof sibling classes but switch
        # requirementDischarge to search and update discharge state.
        data["discharges"][0]["assurance_state"] = {
            "kind": "survivedSearch",
            "id": "search.scenario.demo",
            "generator_id": "gen.scenario.v1",
            "source_closure": "sc-hash",
            "seeds": "seeds-hash",
            "budget_desc": "seeds=8,cases<=256",
            "partitions": "part-hash",
            "divergence_count": 0,
            "calibration": "cal.omit-ops-mixin",
        }
        raw_b = claim.RawClaimInputs.from_dict(data)
        art_b = claim.regenerate_claim(raw_b)
        self.assertTrue(art_b["synthetic_complete"])
        self.assertNotEqual(
            claim.compute_claim_identity(raw_a),
            claim.compute_claim_identity(raw_b),
        )

    def test_ancestry_capture_mutation_changes_identity(self) -> None:
        raw_a = claim.toy_raw_inputs()
        data = _raw_dict()
        for entry in data["cycle_entries"]:
            fi = entry.get("freeze_integrity")
            if fi is not None:
                fi["manifest"]["capture_identity"] = "capture.MUTATED"
                fi["manifest"]["extractor_identity"] = "extractor-MUTATED"
        raw_b = claim.RawClaimInputs.from_dict(data)
        self.assertNotEqual(
            claim.compute_claim_identity(raw_a),
            claim.compute_claim_identity(raw_b),
        )
        self.assertTrue(claim.regenerate_claim(raw_b)["synthetic_complete"])

    def test_selected_rejected_event_rejected(self) -> None:
        data = _raw_dict()
        data["selected_promoted_event"] = "cycle.claim.rejected.1"
        raw = claim.RawClaimInputs.from_dict(data)
        with self.assertRaises(claim.ClaimGuardError) as ctx:
            claim.regenerate_claim(raw)
        self.assertEqual(ctx.exception.reason, "selected_event_not_promoted")

    def test_selected_missing_event_rejected(self) -> None:
        data = _raw_dict()
        data["selected_promoted_event"] = "cycle.absent.1"
        raw = claim.RawClaimInputs.from_dict(data)
        with self.assertRaises(claim.ClaimGuardError) as ctx:
            claim.regenerate_claim(raw)
        self.assertEqual(ctx.exception.reason, "selected_event_missing")

    def test_failure_report_lists_blocker_without_completion(self) -> None:
        data = _raw_dict()
        data["discharges"][0]["assurance_state"] = {
            "kind": "blocked",
            "id": "blocker.env.metal",
            "blocker_kind": "environment",
        }
        raw = claim.RawClaimInputs.from_dict(data)
        report = claim.attempt_claim(raw)
        self.assertFalse(report["validation_ok"])
        self.assertFalse(report["synthetic_complete"])
        self.assertIn("policy_does_not_admit", report["diagnosed_faults"])
        self.assertTrue(
            any(r["kind"] == "blocked" for r in report["residual_obligations"])
        )
        self.assertIn("completion language forbidden", report["rendered_report"])
        self.assertNotIn("synthetic-complete", report["rendered_report"])

    def test_hidden_open_blocker_rejected(self) -> None:
        data = _raw_dict()
        data["discharges"][0]["assurance_state"] = {
            "kind": "blocked",
            "id": "blocker.env.metal",
            "blocker_kind": "environment",
        }
        raw = claim.RawClaimInputs.from_dict(data)
        with self.assertRaises(claim.ClaimGuardError) as ctx:
            claim.regenerate_claim(raw)
        self.assertIn(
            ctx.exception.reason,
            {"policy_does_not_admit", "bounded_search_mislabeled_as_proof_path"},
        )

        data2 = _with_search_policy(_raw_dict())
        data2["discharges"][0]["assurance_state"] = {
            "kind": "survivedSearch",
            "id": "search.scenario.demo",
            "generator_id": "gen.scenario.v1",
            "source_closure": "sc-hash",
            "seeds": "seeds-hash",
            "budget_desc": "seeds=8,cases<=256",
            "partitions": "part-hash",
            "divergence_count": 0,
            "calibration": "cal.omit-ops-mixin",
        }
        raw2 = claim.RawClaimInputs.from_dict(data2)
        artifact = claim.regenerate_claim(raw2)
        self.assertTrue(
            any(r["kind"] == "survivedSearch" for r in artifact["residual_obligations"])
        )
        forged = copy.deepcopy(artifact)
        forged["residual_obligations"] = []
        with self.assertRaises(claim.ClaimGuardError) as ctx2:
            claim.verify_claim_artifact(raw2, forged)
        self.assertEqual(ctx2.exception.reason, "hidden_or_altered_residual_obligations")

    def test_bounded_search_mislabeled_proof_rejected(self) -> None:
        data = _raw_dict()
        data["discharges"][0]["assurance_state"] = {
            "kind": "survivedSearch",
            "id": "search.scenario.demo",
            "generator_id": "gen.scenario.v1",
            "source_closure": "sc-hash",
            "seeds": "seeds-hash",
            "budget_desc": "seeds=8,cases<=256",
            "partitions": "part-hash",
            "divergence_count": 0,
            "calibration": "cal.omit-ops-mixin",
        }
        raw = claim.RawClaimInputs.from_dict(data)
        with self.assertRaises(claim.ClaimGuardError) as ctx:
            claim.regenerate_claim(raw)
        self.assertIn(
            ctx.exception.reason,
            {"policy_does_not_admit", "bounded_search_mislabeled_as_proof_path"},
        )

        data_ok = _with_search_policy(_raw_dict())
        data_ok["discharges"][0]["assurance_state"] = {
            "kind": "survivedSearch",
            "id": "search.scenario.demo",
            "generator_id": "gen.scenario.v1",
            "source_closure": "sc-hash",
            "seeds": "seeds-hash",
            "budget_desc": "seeds=8,cases<=256",
            "partitions": "part-hash",
            "divergence_count": 0,
            "calibration": "cal.omit-ops-mixin",
        }
        raw_ok = claim.RawClaimInputs.from_dict(data_ok)
        artifact = claim.regenerate_claim(raw_ok)
        self.assertIn("survivedSearch", artifact["assurance_grades"][0]["label"])
        self.assertNotIn("proved(", artifact["assurance_grades"][0]["label"])
        forged = copy.deepcopy(artifact)
        forged["assurance_grades"][0] = {
            "requirement": "REQ-TOY-IMPORT",
            "grade": "proved",
            "label": "proved(proof.fake/Fake)",
        }
        with self.assertRaises(claim.ClaimGuardError) as ctx2:
            claim.verify_claim_artifact(raw_ok, forged)
        self.assertEqual(ctx2.exception.reason, "assurance_grade_mismatch")

    def test_performance_evidence_rejected_for_semantic(self) -> None:
        data = _raw_dict()
        data["discharges"][0]["purpose"] = "performance"
        raw = claim.RawClaimInputs.from_dict(data)
        with self.assertRaises(claim.ClaimGuardError) as ctx:
            claim.regenerate_claim(raw)
        self.assertEqual(ctx.exception.reason, "performance_purpose_rejected")

    def test_changed_provenance_changes_identity(self) -> None:
        raw_a = claim.toy_raw_inputs()
        data = _raw_dict()
        for field in data["field_provenance"]:
            if field["name"] == "target":
                field["provenance"] = {
                    **field["provenance"],
                    "source_closure": "src-hash-MUTATED",
                }
        raw_b = claim.RawClaimInputs.from_dict(data)
        self.assertNotEqual(
            claim.compute_claim_identity(raw_a),
            claim.compute_claim_identity(raw_b),
        )

    def test_output_digest_collusion_without_raw_identity(self) -> None:
        raw = claim.toy_raw_inputs()
        honest = claim.regenerate_claim(raw)
        colluded = copy.deepcopy(honest)
        colluded["rendered_report"] = honest["rendered_report"] + "\nCOLLUDED COMPLETE"
        colluded["synthetic_complete"] = True
        colluded["claim_identity"] = "colluded-identity-over-edited-body"
        with self.assertRaises(claim.ClaimGuardError) as ctx:
            claim.verify_claim_artifact(raw, colluded)
        self.assertEqual(ctx.exception.reason, "stale_or_forged_claim_identity")

        colluded2 = copy.deepcopy(honest)
        colluded2["rendered_report"] = honest["rendered_report"] + "\nEXTRA"
        colluded2["claim_identity"] = claim.digest_token(colluded2["rendered_report"])
        with self.assertRaises(claim.ClaimGuardError) as ctx2:
            claim.verify_claim_artifact(raw, colluded2)
        self.assertEqual(ctx2.exception.reason, "stale_or_forged_claim_identity")

    def test_generated_output_digest_collusion_exact_equality(self) -> None:
        raw = claim.toy_raw_inputs()
        honest = claim.regenerate_claim(raw)
        colluded = copy.deepcopy(honest)
        colluded["epistemic_limitations"] = ["nothing to see"]
        colluded["trust_boundary"] = "RELEASE GUARD"
        colluded["cross_language_generation_from_lean"] = True
        colluded["extra_unknown_field"] = "COLLUDE"
        with self.assertRaises(claim.ClaimGuardError) as ctx:
            claim.verify_claim_artifact(raw, colluded)
        self.assertIn(
            ctx.exception.reason,
            {
                "tampered_epistemic_limitations",
                "tampered_trust_boundary",
                "false_cross_language_generation_claim",
                "artifact_not_canonical",
            },
        )

    def test_authored_conclusion_fields_rejected_in_raw(self) -> None:
        data = _raw_dict()
        data["complete"] = True
        with self.assertRaises(claim.ClaimGuardError) as ctx:
            claim.RawClaimInputs.from_dict(data)
        self.assertEqual(ctx.exception.reason, "authored_conclusion_field")

    def test_stale_judging_closure_in_cycle_registry(self) -> None:
        data = _raw_dict()
        data["cycle_descriptors"][0][
            "expected_frozen_judging_closure_identity"
        ] = "stale-judging-closure-identity"
        # Keep entry descriptor in sync.
        data["cycle_entries"][0]["descriptor"] = data["cycle_descriptors"][0]
        raw = claim.RawClaimInputs.from_dict(data)
        with self.assertRaises(claim.ClaimGuardError) as ctx:
            claim.regenerate_claim(raw)
        self.assertIn(
            ctx.exception.reason,
            {
                "stale_or_mismatched_judging_closure_identity",
                "promoted_closure_mismatch",
            },
        )

    def test_empty_denominator_rejected(self) -> None:
        data = _raw_dict()
        data["inventory"] = []
        data["requirement_entries"] = []
        data["discharges"] = []
        data["dispositions"] = []
        raw = claim.RawClaimInputs.from_dict(data)
        with self.assertRaises(claim.ClaimGuardError) as ctx:
            claim.regenerate_claim(raw)
        self.assertEqual(ctx.exception.reason, "empty_denominator")

    def test_tempfile_emit_and_check_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tgrad-generated-claim-test-") as tmp:
            root = Path(tmp)
            result = claim.demonstrate_regeneration(root)
            self.assertIs(result["authentication_claim"], False)
            raw = claim.RawClaimInputs.from_dict(
                json.loads((root / "raw_inputs.json").read_text(encoding="utf-8"))
            )
            artifact = json.loads((root / "claim.json").read_text(encoding="utf-8"))
            verify = claim.verify_claim_artifact(raw, artifact)
            self.assertTrue(verify["ok"])

    def test_trust_boundary_is_artifact_behavior(self) -> None:
        """Behavioral evidence: regenerated artifact carries non-auth claims."""
        art = claim.regenerate_claim(claim.toy_raw_inputs())
        self.assertIs(art["authentication_claim"], False)
        self.assertIs(art["content_digest_cryptographic"], False)
        self.assertIs(art["lean_python_agreement_is_authentication"], False)
        self.assertIs(art["cross_language_generation_from_lean"], False)
        self.assertEqual(art["activation"], "synthetic-non-release")
        self.assertIn("not a release guard", art["trust_boundary"])
        self.assertIn(
            "not cryptographic field grounding",
            "\n".join(art["epistemic_limitations"]),
        )
        # Exact schema: unknown artifact key fails verify.
        mut = copy.deepcopy(art)
        mut["sneaky"] = True
        with self.assertRaises(claim.ClaimGuardError) as ctx:
            claim.verify_claim_artifact(claim.toy_raw_inputs(), mut)
        self.assertEqual(ctx.exception.reason, "artifact_not_canonical")


if __name__ == "__main__":
    unittest.main()
