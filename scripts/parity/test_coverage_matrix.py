#!/usr/bin/env python3
"""Offline falsifiers for the atomic parity coverage-matrix contract.

These are handed to the serial verifier; this module performs no GPU work.
"""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.parity import coverage_matrix as matrix  # noqa: E402
from scripts.parity.coverage_model import (  # noqa: E402
    EXPECTED_REQUIREMENT_COUNT,
    attach_content_sha256,
    build_requirement_inventory,
)


MANIFEST = ROOT / "fixtures" / "parity" / "upstream_19c4d736f2bc.json"


def rehash(document: dict) -> dict:
    document.pop("content_sha256", None)
    document.update(attach_content_sha256(document))
    return document


def confirmed_file(identifier: str, digit: str) -> dict:
    return {
        "state": "confirmed",
        "id": identifier,
        "path": f"/evidence/{identifier}.json",
        "sha256": digit * 64,
    }


class CoverageMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.requirements = build_requirement_inventory(MANIFEST)

    def make_matrix(self) -> dict:
        with tempfile.TemporaryDirectory(prefix="tgrad-coverage-test-") as directory:
            profile = Path(directory) / "profile.json"
            profile.write_text(
                json.dumps(
                    attach_content_sha256(
                        {
                            "schema_version": 1,
                            "kind": "tgrad-parity-profile",
                            "profile": "publicApi",
                            "environment_ids": ["host"],
                        }
                    )
                )
            )
            identities = [
                {
                    "state": "confirmed",
                    "repository": "/subject",
                    "commit": "1" * 40,
                    "tree": "2" * 40,
                    "dirty": False,
                },
                {
                    "state": "confirmed",
                    "repository": "/verifier",
                    "commit": "3" * 40,
                    "tree": "4" * 40,
                    "dirty": False,
                },
            ]
            with mock.patch.object(matrix, "git_identity", side_effect=identities):
                return matrix.initialize_matrix(
                    self.requirements,
                    Path("/subject"),
                    Path("/verifier"),
                    "publicApi",
                    profile,
                )

    def confirm_optional_identities(self, document: dict) -> None:
        document["classification"] = confirmed_file("public-contract-v1", "b")
        document["adapter"] = confirmed_file("thin-adapter-v1", "c")
        document["relation_registry"] = confirmed_file("relations-v1", "d")
        document["environment"] = confirmed_file("host", "e")
        document["oracle_contract"] = confirmed_file("upstream-suite-v1", "f")

    def classify_test_rows(self, document: dict) -> None:
        for cell in document["cells"]:
            if cell["disposition"] == "unclassified":
                cell["disposition"] = "excluded"
                cell["classification"] = "public-contract-v1:internal-only"
                cell["rationale"] = "reviewed internal-representation test fixture"
        reconciled = matrix.reconcile_obligations(document, self.requirements)
        document.clear()
        document.update(reconciled)

    def attach_requirement_observations(
        self, document: dict, requirement_id: str, *, origin: str = "upstream_suite"
    ) -> None:
        obligation_rows = [
            value
            for value in document["obligations"]
            if value["requirement_id"] == requirement_id
        ]
        self.assertTrue(obligation_rows)
        for index, obligation in enumerate(obligation_rows):
            calibration_id = f"calibration-{index}"
            validator_id = f"validator-{index}"
            evidence_id = f"evidence-{index}"
            document["calibrations"].append(
                {
                    "id": calibration_id,
                    "verifier_tree": document["verifier"]["tree"],
                    "validator_definition_sha256": str(index + 1) * 64,
                    "mutant_tree": str(index + 2) * 40,
                    "fault_model": f"wrong {obligation['dimension']} result",
                    "dimension": obligation["dimension"],
                    "environment_id": obligation["environment_id"],
                    "equivalence_relation": obligation["equivalence_relation"],
                    "adapter_sha256": document["adapter"]["sha256"],
                    "relation_registry_sha256": document["relation_registry"]["sha256"],
                    "environment_sha256": document["environment"]["sha256"],
                    "oracle_contract_sha256": document["oracle_contract"]["sha256"],
                    "scenario_manifest_sha256": "6" * 64,
                    "artifact_sha256": "7" * 64,
                    "outcome": "validator_rejected_mutant",
                }
            )
            document["validators"].append(
                {
                    "id": validator_id,
                    "verifier_tree": document["verifier"]["tree"],
                    "definition_sha256": str(index + 1) * 64,
                    "dimensions": [obligation["dimension"]],
                    "calibration_ids": [calibration_id],
                }
            )
            document["evidence"].append(
                {
                    "id": evidence_id,
                    "origin": origin,
                    "kind": "upstream_test",
                    "obligation_id": obligation["id"],
                    "requirement_id": requirement_id,
                    "dimension": obligation["dimension"],
                    "environment_id": obligation["environment_id"],
                    "equivalence_relation": obligation["equivalence_relation"],
                    "outcome": "pass",
                    "upstream_revision": document["target"]["upstream_ref"],
                    "subject_tree": document["subject"]["tree"],
                    "verifier_tree": document["verifier"]["tree"],
                    "adapter_sha256": document["adapter"]["sha256"],
                    "classification_sha256": document["classification"]["sha256"],
                    "relation_registry_sha256": document["relation_registry"]["sha256"],
                    "environment_sha256": document["environment"]["sha256"],
                    "oracle_contract_sha256": document["oracle_contract"]["sha256"],
                    "scenario_manifest_sha256": "8" * 64,
                    "artifact_sha256": "9" * 64,
                    "artifact_size_bytes": 1,
                    "oracle_tree": "a" * 40,
                    "independence_basis_sha256": "b" * 64,
                    "validator_id": validator_id,
                    "calibration_ids": [calibration_id],
                }
            )
            document["observations"].append(
                {
                    "id": f"observation-{index}",
                    "obligation_id": obligation["id"],
                    "outcome": "pass",
                    "evidence_id": evidence_id,
                }
            )

    def prepare_reportable(self, document: dict) -> None:
        self.confirm_optional_identities(document)
        self.classify_test_rows(document)

    def test_generated_denominator_is_exactly_590_unique_rows(self) -> None:
        rows = self.requirements["requirements"]
        self.assertEqual(EXPECTED_REQUIREMENT_COUNT, 590)
        self.assertEqual(len(rows), 590)
        self.assertEqual(len({row["id"] for row in rows}), 590)
        self.assertEqual([row["ordinal"] for row in rows], list(range(590)))
        self.assertTrue(all(len(row["identity_sha256"]) == 64 for row in rows))
        self.assertEqual(self.requirements["reviewed_requirement_count"], 590)

    def test_initial_matrix_is_total_but_not_reportable(self) -> None:
        document = self.make_matrix()
        summary = matrix.validate_matrix(document, self.requirements)
        self.assertTrue(summary["structurally_total"])
        self.assertEqual(summary["cell_count"], 590)
        self.assertGreater(summary["obligation_count"], 0)
        self.assertFalse(summary["reportable"])
        self.assertFalse(summary["fully_observed"])
        self.assertFalse(summary["conformant"])
        self.assertNotIn("percentage", json.dumps(summary).lower())
        self.assertNotIn("score", json.dumps(summary).lower())

    def test_missing_duplicate_and_reordered_cells_are_rejected(self) -> None:
        for mutation in ("missing", "duplicate", "reordered"):
            with self.subTest(mutation=mutation):
                document = self.make_matrix()
                if mutation == "missing":
                    document["cells"].pop()
                elif mutation == "duplicate":
                    document["cells"][-1] = copy.deepcopy(document["cells"][0])
                else:
                    document["cells"][0], document["cells"][1] = (
                        document["cells"][1], document["cells"][0]
                    )
                rehash(document)
                with self.assertRaisesRegex(matrix.CoverageMatrixError, "exact ordered total"):
                    matrix.validate_matrix(document, self.requirements)

    def test_stale_matrix_hash_is_rejected(self) -> None:
        document = self.make_matrix()
        document["cells"][0]["rationale"] = "tampered after hashing"
        with self.assertRaisesRegex(matrix.CoverageMatrixError, "content_sha256 mismatch"):
            matrix.validate_matrix(document, self.requirements)

    def test_authored_pass_state_is_rejected(self) -> None:
        document = self.make_matrix()
        document["cells"][0]["result"] = "pass"
        rehash(document)
        with self.assertRaisesRegex(matrix.CoverageMatrixError, "derived, not authored"):
            matrix.validate_matrix(document, self.requirements)

    def test_reporting_identity_does_not_create_conformance(self) -> None:
        document = self.make_matrix()
        self.prepare_reportable(document)
        rehash(document)
        summary = matrix.validate_matrix(document, self.requirements)
        self.assertTrue(summary["reportable"])
        self.assertFalse(summary["fully_observed"])
        self.assertFalse(summary["conformant"])
        self.assertGreater(summary["required_counts"]["unobserved"], 0)

    def test_self_referential_evidence_cannot_promote_pass(self) -> None:
        document = self.make_matrix()
        self.prepare_reportable(document)
        requirement_id = document["obligations"][0]["requirement_id"]
        self.attach_requirement_observations(
            document, requirement_id, origin="self_referential"
        )
        rehash(document)
        with self.assertRaisesRegex(matrix.CoverageMatrixError, "self-referential"):
            matrix.validate_matrix(document, self.requirements)

    def test_atomic_observations_can_pass_one_row_without_claiming_parity(self) -> None:
        document = self.make_matrix()
        self.prepare_reportable(document)
        requirement_id = document["obligations"][0]["requirement_id"]
        self.attach_requirement_observations(document, requirement_id)
        rehash(document)
        summary = matrix.validate_matrix(document, self.requirements)
        self.assertEqual(summary["required_counts"]["pass"], 1)
        self.assertFalse(summary["fully_observed"])
        self.assertFalse(summary["conformant"])

    def test_one_dimension_cannot_launder_a_multi_dimension_row(self) -> None:
        document = self.make_matrix()
        self.prepare_reportable(document)
        requirement_id = document["obligations"][0]["requirement_id"]
        self.attach_requirement_observations(document, requirement_id)
        document["observations"].pop()
        used = document["evidence"].pop()["id"]
        self.assertNotIn(used, {row["evidence_id"] for row in document["observations"]})
        document["validators"].pop()
        document["calibrations"].pop()
        rehash(document)
        summary = matrix.validate_matrix(document, self.requirements)
        self.assertEqual(summary["cell_results"][requirement_id], "unobserved")

    def test_wrong_subject_and_orphan_evidence_are_rejected(self) -> None:
        for mutation in ("wrong-subject", "orphan"):
            with self.subTest(mutation=mutation):
                document = self.make_matrix()
                self.prepare_reportable(document)
                requirement_id = document["obligations"][0]["requirement_id"]
                self.attach_requirement_observations(document, requirement_id)
                if mutation == "wrong-subject":
                    document["evidence"][0]["subject_tree"] = "0" * 40
                    expected = "subject_tree"
                else:
                    orphan = copy.deepcopy(document["evidence"][0])
                    orphan["id"] = "orphan-evidence"
                    document["evidence"].append(orphan)
                    expected = "orphan evidence"
                rehash(document)
                with self.assertRaisesRegex(matrix.CoverageMatrixError, expected):
                    matrix.validate_matrix(document, self.requirements)


if __name__ == "__main__":
    unittest.main()
