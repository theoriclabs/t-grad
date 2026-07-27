#!/usr/bin/env python3
"""Offline falsifiers for the diagnostic 590-requirement suite projection."""
from __future__ import annotations

import copy
import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.parity import project_suite_coverage as projection  # noqa: E402


def load_inputs() -> dict:
    return copy.deepcopy(projection.load_projection_inputs(ROOT))


def project(inputs: dict) -> dict:
    return projection.project_documents(
        inputs["requirements"],
        inputs["oracle"],
        inputs["upstream"],
        inputs["tgrad"],
    )


def repair_aggregate(suite: dict) -> None:
    suite["aggregate"] = projection._computed_aggregate(suite["results"])


class SuiteCoverageProjectionTests(unittest.TestCase):
    def test_projection_has_one_ordered_cell_per_reviewed_requirement(self) -> None:
        document = project(load_inputs())
        self.assertEqual(len(document["cells"]), 590)
        self.assertEqual(
            [cell["ordinal"] for cell in document["cells"]],
            list(range(590)),
        )
        self.assertEqual(
            len({cell["requirement_id"] for cell in document["cells"]}),
            590,
        )
        self.assertEqual(
            Counter(cell["disposition"] for cell in document["cells"]),
            Counter({"required": 471, "excluded": 104, "not_applicable": 15}),
        )

    def test_public_contract_is_calibrated_upstream_before_tgrad_is_read(self) -> None:
        document = project(load_inputs())
        calibration = document["suite_diagnostics"]["public_contract_upstream"]
        self.assertEqual(
            calibration["file_status_counts"],
            {"pass": 34, "fail": 0, "collect_error": 0, "timeout": 0, "empty": 0},
        )
        self.assertEqual(
            calibration["test_outcome_counts"],
            {"passed": 1003, "failed": 0, "errors": 0, "skipped": 146},
        )

    def test_missing_api_result_is_fatal_even_with_repaired_aggregate(self) -> None:
        inputs = load_inputs()
        suite = inputs["tgrad"]["null"]
        removed = suite["results"].pop()
        self.assertEqual(removed["file"], "test/null/test_rearrange_einops.py")
        repair_aggregate(suite)
        with self.assertRaisesRegex(projection.ProjectionError, "Tgrad API coverage mismatch.*missing"):
            project(inputs)

    def test_extra_non_api_result_is_fatal_even_with_repaired_aggregate(self) -> None:
        inputs = load_inputs()
        suite = inputs["tgrad"]["null"]
        extra = copy.deepcopy(suite["results"][0])
        extra["file"] = "test/null/test_attention.py"
        suite["results"].append(extra)
        repair_aggregate(suite)
        with self.assertRaisesRegex(projection.ProjectionError, "Tgrad API coverage mismatch.*extra"):
            project(inputs)

    def test_collection_status_cannot_be_relabelled_as_execution_failure(self) -> None:
        document = project(load_inputs())
        cell = next(
            cell
            for cell in document["cells"]
            if cell["tgrad_diagnostic"].get("status") == "collect_error"
        )
        self.assertEqual(cell["tgrad_diagnostic"]["stage"], "collection")
        cell["tgrad_diagnostic"]["stage"] = "post_collection_execution"
        with self.assertRaisesRegex(projection.ProjectionError, "status/stage conflation"):
            projection._validate_projection_structure(
                document, require_source_bindings=False
            )

    def test_fixture_status_mutation_cannot_conflate_failure_stages(self) -> None:
        inputs = load_inputs()
        suite = inputs["tgrad"]["null"]
        result = next(row for row in suite["results"] if row["status"] == "collect_error")
        result["status"] = "fail"
        result["errors"] = 0
        result["failed"] = 1
        repair_aggregate(suite)
        with self.assertRaisesRegex(projection.ProjectionError, "Tgrad status distribution changed"):
            project(inputs)

    def test_unknown_attribution_forbids_promotable_or_reportable_claims(self) -> None:
        for field in ("promotable", "reportable_as_conformance"):
            with self.subTest(field=field):
                document = project(load_inputs())
                self.assertTrue(document["claim_boundary"]["diagnostic_only"])
                self.assertEqual(
                    [
                        blocker["field"]
                        for blocker in document["claim_boundary"]["unknown_attribution_blockers"]
                    ],
                    ["tgrad_subject_tree", "verifier_tree", "execution_environment",
                     "raw_diagnostics", "equivalence_relation_registry",
                     "validator_calibration"],
                )
                document["claim_boundary"][field] = True
                with self.assertRaisesRegex(
                    projection.ProjectionError,
                    "cannot be promotable|cannot report conformance",
                ):
                    projection._validate_projection_structure(
                        document, require_source_bindings=False
                    )

    def test_unknown_attribution_blocker_cannot_be_silently_dropped(self) -> None:
        document = project(load_inputs())
        document["claim_boundary"]["unknown_attribution_blockers"].pop()
        with self.assertRaisesRegex(projection.ProjectionError, "attribution blockers"):
            projection._validate_projection_structure(
                document, require_source_bindings=False
            )

    def test_scalar_score_or_percentage_is_forbidden(self) -> None:
        for forbidden_key in ("score", "percentage"):
            with self.subTest(forbidden_key=forbidden_key):
                document = project(load_inputs())
                document[forbidden_key] = 0
                with self.assertRaisesRegex(projection.ProjectionError, "scalar parity key"):
                    projection._validate_projection_structure(
                        document, require_source_bindings=False
                    )

    def test_projection_is_deterministic_for_identical_documents(self) -> None:
        inputs = load_inputs()
        first = project(copy.deepcopy(inputs))
        second = project(copy.deepcopy(inputs))
        self.assertEqual(first, second)

    def test_bound_source_digest_mutation_is_fatal(self) -> None:
        document = projection.build_document(ROOT)
        document["source_bindings"]["strict_adapter_files"][0]["sha256"] = "0" * 64
        document["content_sha256"] = projection._canonical_document_sha256(document)
        with self.assertRaisesRegex(projection.ProjectionError, "adapter_bundle_sha256"):
            projection.validate_projection(document)

    def test_reference_provenance_mutation_is_fatal(self) -> None:
        document = projection.build_document(ROOT)
        binding = document["source_bindings"]["source_fixtures"][0]
        binding["reference_artifact"]["state"] = "byte_identical"
        document["content_sha256"] = projection._canonical_document_sha256(document)
        with self.assertRaisesRegex(projection.ProjectionError, "absent-reference binding"):
            projection.validate_projection(document)

    def test_content_digest_mutation_is_fatal(self) -> None:
        document = projection.build_document(ROOT)
        document["suite_diagnostics"]["tgrad"]["test_outcome_counts"]["errors"] += 1
        with self.assertRaisesRegex(projection.ProjectionError, "content_sha256 mismatch"):
            projection.validate_projection(document)

    def test_final_artifact_requires_source_bindings(self) -> None:
        document = projection.build_document(ROOT)
        del document["source_bindings"]
        document["content_sha256"] = projection._canonical_document_sha256(document)
        with self.assertRaisesRegex(projection.ProjectionError, "requires source bindings"):
            projection.validate_projection(document)

    def test_freshly_rehashed_semantic_mutation_is_fatal(self) -> None:
        document = projection.build_document(ROOT)
        observed = next(
            cell for cell in document["cells"]
            if cell["tgrad_diagnostic"].get("state") == "observed"
        )
        observed["tgrad_diagnostic"]["counters"]["errors"] += 1
        document["content_sha256"] = projection._canonical_document_sha256(document)
        with self.assertRaisesRegex(projection.ProjectionError, "rederived from bound inputs"):
            projection.validate_projection(document)


if __name__ == "__main__":
    unittest.main()
