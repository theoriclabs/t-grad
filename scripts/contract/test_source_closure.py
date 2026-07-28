from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from scripts.contract import generate_source_closure
from scripts.parity import ensure_oracle, extract_upstream, upstream_target


def run_git(repo: Path | None, *args: str) -> str:
    command = ["git"] if repo is None else ["git", "-C", str(repo)]
    result = subprocess.run(
        [*command, *args], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise AssertionError(f"{' '.join(command + list(args))}: {result.stderr}")
    return result.stdout.strip()


def synthetic_oracle(root: Path, *, detached: bool = True) -> tuple[Path, ensure_oracle.OracleExpectation]:
    repo = root / "oracle"
    repo.mkdir()
    run_git(None, "init", "-q", str(repo))
    run_git(repo, "config", "user.email", "source-closure@example.invalid")
    run_git(repo, "config", "user.name", "Source Closure Test")
    (repo / "sample.py").write_text("VALUE = 1\n", encoding="utf-8")
    run_git(repo, "add", "sample.py")
    run_git(repo, "commit", "-q", "-m", "pin")
    revision = run_git(repo, "rev-parse", "HEAD")
    tree = run_git(repo, "rev-parse", "HEAD^{tree}")
    object_format = run_git(repo, "rev-parse", "--show-object-format")
    origin = "https://github.com/example/source-closure-fixture.git"
    run_git(repo, "remote", "add", "origin", origin)
    if detached:
        run_git(repo, "checkout", "-q", "--detach", revision)
    expectation = ensure_oracle.OracleExpectation(
        repository="github.com/example/source-closure-fixture",
        revision=revision,
        tree=tree,
        object_format=object_format,
        clone_url=origin,
    )
    return repo, expectation


class SourceClosureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.checkout = Path(
            os.environ.get(
                "TGRAD_SOURCE_CLOSURE_TEST_CHECKOUT",
                str(upstream_target.DEFAULT_ORACLE),
            )
        )
        cls.live_ok, cls.live_detail = ensure_oracle.verify(cls.checkout)
        try:
            cls.canonical = extract_upstream.SOURCE_CLOSURE_OUTPUT.read_bytes()
        except OSError as exc:
            raise AssertionError(
                f"committed canonical source-closure fixture is required: {exc}"
            ) from exc
        cls.document = extract_upstream.parse_source_closure_bytes(
            cls.canonical, authenticate_extractor_sources=True
        )

    def require_live_checkout(self) -> None:
        if not self.live_ok:
            self.skipTest(f"live pinned checkout unavailable: {self.live_detail}")

    def coherent(self, mutate) -> dict:
        document = copy.deepcopy(self.document)
        mutate(document)
        return extract_upstream.refresh_source_closure_digests(document)

    def coherent_signature_change(self) -> dict:
        def mutate(doc: dict) -> None:
            row = doc["tensor_api"]["declarations"][0]
            shape = json.loads(row["structural_signature"])
            shape["mutant"] = True
            encoded = extract_upstream.canonical_payload(shape).decode("utf-8")
            row["structural_signature"] = encoded
            row["signature_sha256"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()

        return self.coherent(mutate)

    def coherent_non_special_omission(self) -> dict:
        """Remove one ordinary tracked source and every predicate-derived reference."""
        omitted = "tinygrad/helpers.py"

        def mutate(doc: dict) -> None:
            doc["files"] = [row for row in doc["files"] if row["path"] != omitted]
            for category in doc["categories"]:
                category["paths"] = [path for path in category["paths"] if path != omitted]

        mutant = self.coherent(mutate)
        self.assertNotIn(omitted, [row["path"] for row in mutant["files"]])
        return mutant

    def assertPureRejects(self, document: dict) -> None:
        with self.assertRaises(extract_upstream.ExtractionError):
            extract_upstream.validate_source_closure_document(document)

    def test_extract_twice_is_byte_identical(self) -> None:
        self.require_live_checkout()
        first = extract_upstream.build_source_closure(self.checkout)
        again = extract_upstream.build_source_closure(self.checkout)
        self.assertEqual(
            extract_upstream.canonical_bytes(first),
            extract_upstream.canonical_bytes(again),
        )
        self.assertEqual(self.canonical, extract_upstream.canonical_bytes(again))

    def test_corrected_tensor_denominator_and_test_categories(self) -> None:
        tensor = self.document["tensor_api"]
        categories = {row["id"]: row for row in self.document["categories"]}
        self.assertEqual(tensor["method_count"], 295)
        self.assertEqual(tensor["property_count"], 5)
        self.assertEqual(tensor["direct_method_count"], 47)
        self.assertNotIn("get", tensor["method_names"])
        self.assertNotIn("set", tensor["method_names"])
        self.assertEqual(categories["upstream_tests"]["file_count"], 331)
        self.assertEqual(categories["api_surface_tests"]["file_count"], 138)
        self.assertNotEqual(
            categories["upstream_tests"]["inventory_sha256"],
            categories["api_surface_tests"]["inventory_sha256"],
        )

    def test_machine_independent_parser_policy_identity(self) -> None:
        extractor = self.document["extractor"]
        self.assertEqual(extractor["parser_policy_id"], extract_upstream.PARSER_POLICY_ID)
        self.assertEqual(extractor["parser_grammar_feature"], "3.12")
        self.assertEqual(extractor["python_implementation"], "cpython")
        self.assertEqual(extractor["python_major_minor_min"], "3.12")
        self.assertEqual(extractor["python_major_minor_max"], "3.14")
        forbidden = {
            "generated_at", "timestamp", "platform", "python_patch",
            "python_version", "runtime_version", "absolute_path", "temp_path",
        }

        def keys(value: object) -> set[str]:
            if isinstance(value, dict):
                return set(value) | set().union(*(keys(item) for item in value.values()))
            if isinstance(value, list):
                return set().union(*(keys(item) for item in value))
            return set()

        self.assertFalse(keys(self.document) & forbidden)
        self.assertNotIn(str(self.checkout.resolve()), self.canonical.decode("utf-8"))

        mutant = self.coherent(
            lambda doc: doc["extractor"].__setitem__(
                "parser_policy_id", "cpython-ast.structural-signature.stale"
            )
        )
        self.assertPureRejects(mutant)

    def test_missing_checkout(self) -> None:
        with TemporaryDirectory(prefix="tgrad-source-closure-") as temporary:
            ok, detail = ensure_oracle.verify(Path(temporary) / "missing")
        self.assertFalse(ok)
        self.assertIn("no checkout", detail)

    def test_oracle_rejects_wrong_origin_attached_revision_and_tree(self) -> None:
        with TemporaryDirectory(prefix="tgrad-source-closure-") as temporary:
            repo, expectation = synthetic_oracle(Path(temporary), detached=False)
            ok, detail = ensure_oracle.verify(repo, expectation)
            self.assertFalse(ok)
            self.assertIn("attached", detail)
        with TemporaryDirectory(prefix="tgrad-source-closure-") as temporary:
            repo, expectation = synthetic_oracle(Path(temporary))
            run_git(repo, "remote", "set-url", "origin", "https://github.com/wrong/repo.git")
            ok, detail = ensure_oracle.verify(repo, expectation)
            self.assertFalse(ok)
            self.assertIn("origin is", detail)
        with TemporaryDirectory(prefix="tgrad-source-closure-") as temporary:
            repo, expectation = synthetic_oracle(Path(temporary))
            (repo / "sample.py").write_text("VALUE = 2\n", encoding="utf-8")
            run_git(repo, "add", "sample.py")
            run_git(repo, "commit", "-q", "-m", "alternate head")
            alternate_revision = run_git(repo, "rev-parse", "HEAD")
            ok, detail = ensure_oracle.verify(repo, expectation)
            self.assertFalse(ok)
            self.assertIn("HEAD is", detail)
            ok, detail = ensure_oracle.verify(
                repo,
                replace(
                    expectation,
                    revision=alternate_revision,
                    tree="0" * len(expectation.tree),
                ),
            )
            self.assertFalse(ok)
            self.assertIn("HEAD tree", detail)

    def test_oracle_rejects_tracked_and_untracked_dirt(self) -> None:
        with TemporaryDirectory(prefix="tgrad-source-closure-") as temporary:
            repo, expectation = synthetic_oracle(Path(temporary))
            (repo / "sample.py").write_text("VALUE = 2\n", encoding="utf-8")
            ok, detail = ensure_oracle.verify(repo, expectation)
            self.assertFalse(ok)
            self.assertIn("dirty", detail)
        with TemporaryDirectory(prefix="tgrad-source-closure-") as temporary:
            repo, expectation = synthetic_oracle(Path(temporary))
            (repo / "untracked.py").write_text("VALUE = 3\n", encoding="utf-8")
            ok, detail = ensure_oracle.verify(repo, expectation)
            self.assertFalse(ok)
            self.assertIn("dirty", detail)

    def test_oracle_rejects_missing_required_object(self) -> None:
        with TemporaryDirectory(prefix="tgrad-source-closure-") as temporary:
            repo, expectation = synthetic_oracle(Path(temporary))
            oid = run_git(repo, "rev-parse", "HEAD:sample.py")
            payload = (repo / "sample.py").read_bytes()
            self.assertEqual(
                ensure_oracle.git_blob_oid(expectation.object_format, payload), oid
            )
            self.assertNotEqual(
                ensure_oracle.git_blob_oid(
                    expectation.object_format, payload + b"# substituted\n"
                ),
                oid,
            )
            object_path = repo / ".git" / "objects" / oid[:2] / oid[2:]
            self.assertTrue(object_path.is_file())
            object_path.unlink()
            ok, detail = ensure_oracle.verify(repo, expectation)
            self.assertFalse(ok)
            self.assertIn("strict full fsck failed", detail)

    def test_oracle_fsck_rejects_missing_reachable_tree_object(self) -> None:
        with TemporaryDirectory(prefix="tgrad-source-closure-") as temporary:
            repo, expectation = synthetic_oracle(Path(temporary))
            tree_path = (
                repo / ".git" / "objects" /
                expectation.tree[:2] / expectation.tree[2:]
            )
            self.assertTrue(tree_path.is_file())
            tree_path.unlink()
            ok, detail = ensure_oracle.verify(repo, expectation)
            self.assertFalse(ok)
            self.assertIn("strict full fsck failed", detail)

    def test_oracle_verify_rehashes_the_cat_file_payload(self) -> None:
        with TemporaryDirectory(prefix="tgrad-source-closure-") as temporary:
            repo, expectation = synthetic_oracle(Path(temporary))
            real_git = ensure_oracle.git

            def substituted_git(
                git_repo: Path | None,
                *args: str,
                input_bytes: bytes | None = None,
            ) -> subprocess.CompletedProcess[bytes]:
                result = real_git(git_repo, *args, input_bytes=input_bytes)
                if args != ("cat-file", "--batch") or result.returncode != 0:
                    return result
                altered = bytearray(result.stdout)
                header_end = altered.index(b"\n")
                size = int(bytes(altered[:header_end]).split(b" ")[2])
                payload_start = header_end + 1
                self.assertGreater(size, 0)
                altered[payload_start] ^= 1
                self.assertEqual(len(altered), len(result.stdout))
                return subprocess.CompletedProcess(
                    result.args,
                    result.returncode,
                    bytes(altered),
                    result.stderr,
                )

            with mock.patch.object(
                ensure_oracle, "git", side_effect=substituted_git
            ):
                ok, detail = ensure_oracle.verify(repo, expectation)
            self.assertFalse(ok)
            self.assertIn("payload object id mismatch", detail)

    def test_oracle_rejects_replacement_refs_and_sanitizes_inherited_git_overrides(self) -> None:
        with TemporaryDirectory(prefix="tgrad-source-closure-") as temporary:
            repo, expectation = synthetic_oracle(Path(temporary))
            original_oid = run_git(repo, "rev-parse", "HEAD:sample.py")
            replacement_file = repo / "replacement.py"
            replacement_file.write_text("VALUE = 99\n", encoding="utf-8")
            replacement_oid = run_git(repo, "hash-object", "-w", "replacement.py")
            run_git(repo, "replace", original_oid, replacement_oid)
            ok, detail = ensure_oracle.verify(repo, expectation)
            self.assertFalse(ok)
            self.assertIn("replacement refs are forbidden", detail)

        with TemporaryDirectory(prefix="tgrad-source-closure-") as temporary:
            repo, expectation = synthetic_oracle(Path(temporary))
            poisoned = Path(temporary) / "poisoned-object-directory"
            poisoned.mkdir()
            with mock.patch.dict(
                os.environ,
                {
                    "GIT_DIR": str(Path(temporary) / "wrong-git-dir"),
                    "GIT_OBJECT_DIRECTORY": str(poisoned),
                    "GIT_REPLACE_REF_BASE": "refs/poisoned",
                    "GIT_CONFIG_COUNT": "1",
                    "GIT_CONFIG_KEY_0": "core.repositoryformatversion",
                    "GIT_CONFIG_VALUE_0": "99",
                },
            ):
                ok, detail = ensure_oracle.verify(repo, expectation)
            self.assertTrue(ok, detail)

    def test_category_and_selected_file_removals_are_rejected(self) -> None:
        removed_category = self.coherent(lambda doc: doc["categories"].pop())
        self.assertPureRejects(removed_category)

        def remove_tensor_file(doc: dict) -> None:
            doc["files"] = [row for row in doc["files"] if row["path"] != "tinygrad/tensor.py"]
            for category in doc["categories"]:
                category["paths"] = [
                    path for path in category["paths"] if path != "tinygrad/tensor.py"
                ]

        self.assertPureRejects(self.coherent(remove_tensor_file))

    def test_duplicate_category_file_and_declaration_are_rejected(self) -> None:
        def duplicate_category(doc: dict) -> None:
            doc["categories"].append(copy.deepcopy(doc["categories"][-1]))

        def duplicate_file(doc: dict) -> None:
            doc["files"].append(copy.deepcopy(doc["files"][0]))
            doc["files"].sort(key=lambda row: row["path"])

        def duplicate_declaration(doc: dict) -> None:
            rows = doc["tensor_api"]["declarations"]
            rows.append(copy.deepcopy(rows[0]))
            rows.sort(key=lambda row: (row["kind"], row["name"], row["source"], row["declaring_class"]))

        for mutate in (duplicate_category, duplicate_file, duplicate_declaration):
            with self.subTest(mutate=mutate.__name__):
                self.assertPureRejects(self.coherent(mutate))

    def test_tensor_only_47_undercount_is_rejected(self) -> None:
        def mutate(doc: dict) -> None:
            tensor = doc["tensor_api"]
            tensor["declarations"] = [
                row for row in tensor["declarations"] if row["source"] == "tinygrad/tensor.py"
            ]
            tensor["method_names"] = sorted({
                row["name"] for row in tensor["declarations"] if row["kind"] == "method"
            })
            tensor["property_names"] = sorted({
                row["name"] for row in tensor["declarations"] if row["kind"] == "property"
            })
            tensor["direct_methods"] = list(tensor["method_names"])

        mutant = self.coherent(mutate)
        self.assertEqual(mutant["tensor_api"]["method_count"], 47)
        self.assertPureRejects(mutant)

    def test_duplicate_name_declaration_undercount_306_is_rejected(self) -> None:
        """Remove one duplicate-name declaration; unique-name pins alone must not pass."""

        def mutate(doc: dict) -> None:
            tensor = doc["tensor_api"]
            removed = False
            kept: list[dict] = []
            for row in tensor["declarations"]:
                if (
                    not removed
                    and row["kind"] == "method"
                    and row["name"] == "alu"
                    and row["source"] == "tinygrad/mixin/elementwise.py"
                    and row["declaring_class"] == "ElementwiseMixin"
                ):
                    removed = True
                    continue
                kept.append(row)
            if not removed:
                raise AssertionError("expected ElementwiseMixin.alu duplicate-name declaration")
            tensor["declarations"] = kept

        mutant = self.coherent(mutate)
        tensor = mutant["tensor_api"]
        self.assertEqual(tensor["declaration_count"], 306)
        self.assertEqual(tensor["direct_method_count"], 47)
        self.assertEqual(tensor["method_count"], 295)
        self.assertEqual(tensor["property_count"], 5)
        self.assertEqual(len(tensor["direct_methods"]), 47)
        self.assertEqual(len(tensor["method_names"]), 295)
        self.assertEqual(len(tensor["property_names"]), 5)
        self.assertIn("alu", tensor["method_names"])
        self.assertPureRejects(mutant)

    def test_legacy_helper_contamination_297_is_named_and_rejected(self) -> None:
        self.assertEqual(extract_upstream.LEGACY_HELPER_CONTAMINATION_METHOD_COUNT, 297)

        def mutate(doc: dict) -> None:
            tensor = doc["tensor_api"]
            signature = tensor["declarations"][0]["structural_signature"]
            signature_sha = hashlib.sha256(signature.encode("utf-8")).hexdigest()
            for name in ("get", "set"):
                tensor["declarations"].append({
                    "declaring_class": "_ContextVar",
                    "kind": "method",
                    "name": name,
                    "signature_sha256": signature_sha,
                    "source": "tinygrad/tensor.py",
                    "structural_signature": signature,
                })
            tensor["declarations"].sort(
                key=lambda row: (row["kind"], row["name"], row["source"], row["declaring_class"])
            )
            tensor["method_names"] = sorted(set(tensor["method_names"]) | {"get", "set"})

        mutant = self.coherent(mutate)
        self.assertEqual(mutant["tensor_api"]["method_count"], 297)
        self.assertPureRejects(mutant)

    def test_ops_wrong_source_and_test_category_conflation_are_rejected(self) -> None:
        def redirect_ops(doc: dict) -> None:
            doc["ops"]["source"] = "tinygrad/uop/ops.py"
            for row in doc["ops"]["declarations"]:
                row["source"] = "tinygrad/uop/ops.py"
            next(row for row in doc["categories"] if row["id"] == "ops")["paths"] = [
                "tinygrad/uop/ops.py"
            ]

        self.assertPureRejects(self.coherent(redirect_ops))

        def conflate(doc: dict) -> None:
            all_tests = next(row for row in doc["categories"] if row["id"] == "upstream_tests")
            api_tests = next(row for row in doc["categories"] if row["id"] == "api_surface_tests")
            api_tests["paths"] = list(all_tests["paths"])
            doc["tests"]["sources"] = list(all_tests["paths"])

        mutant = self.coherent(conflate)
        self.assertEqual(mutant["tests"]["count"], 331)
        self.assertPureRejects(mutant)

    def test_nondeterminism_timestamp_unknown_ids_and_path_traversal_are_rejected(self) -> None:
        def reorder(doc: dict) -> None:
            doc["files"][0], doc["files"][1] = doc["files"][1], doc["files"][0]

        def timestamp(doc: dict) -> None:
            doc["timestamp"] = "2026-07-28T00:00:00Z"

        def unknown_category(doc: dict) -> None:
            doc["categories"][0]["id"] = "not_a_category"

        def unknown_limit(doc: dict) -> None:
            doc["limits"][0]["id"] = "not_a_limit"

        def traversal(doc: dict) -> None:
            doc["files"][0]["path"] = "../escape.py"

        for mutate in (reorder, timestamp, unknown_category, unknown_limit, traversal):
            with self.subTest(mutate=mutate.__name__):
                self.assertPureRejects(self.coherent(mutate))

    def test_malformed_hash_non_utf8_ast_failure_and_partial_json_are_rejected(self) -> None:
        malformed = self.coherent(lambda doc: doc["files"][0].__setitem__("sha256", "xyz"))
        self.assertPureRejects(malformed)
        with self.assertRaises(extract_upstream.ExtractionError):
            extract_upstream.parse_python_bytes("bad.py", b"\xff")
        with self.assertRaises(extract_upstream.ExtractionError):
            extract_upstream.parse_python_bytes("bad.py", b"def broken(:\n")
        for raw in (b"", b"{}\n", b'{"schema":1,"schema":2}\n'):
            with self.subTest(raw=raw):
                with self.assertRaises(extract_upstream.ExtractionError):
                    extract_upstream.parse_source_closure_bytes(raw)

    def test_signature_removal_is_purely_rejected(self) -> None:
        def mutate(doc: dict) -> None:
            row = doc["tensor_api"]["declarations"][0]
            row["structural_signature"] = ""
            row["signature_sha256"] = hashlib.sha256(b"").hexdigest()

        self.assertPureRejects(self.coherent(mutate))

    def test_coherent_signature_change_passes_pure_representation_validation(self) -> None:
        mutant = self.coherent_signature_change()
        extract_upstream.validate_source_closure_document(mutant)

    def test_coherent_signature_change_requires_git_reextraction(self) -> None:
        self.require_live_checkout()
        mutant = self.coherent_signature_change()
        with TemporaryDirectory(prefix="tgrad-source-closure-") as temporary:
            path = Path(temporary) / "mutant.json"
            path.write_bytes(extract_upstream.canonical_bytes(mutant))
            with self.assertRaisesRegex(
                extract_upstream.ExtractionError, "differs from foreign Git re-extraction"
            ):
                extract_upstream.check_source_closure_against_git(path, self.checkout)

    def test_load_bearing_source_identity_staleness_requires_local_authentication(self) -> None:
        for source_path in (
            "scripts/parity/ensure_oracle.py",
            "scripts/contract/generate_source_closure.py",
        ):
            with self.subTest(source_path=source_path):
                def mutate(doc: dict) -> None:
                    row = next(
                        row for row in doc["extractor"]["source_files"]
                        if row["path"] == source_path
                    )
                    row["sha256"] = (
                        "0" * 64 if row["sha256"] != "0" * 64 else "1" * 64
                    )

                mutant = self.coherent(mutate)
                extract_upstream.validate_source_closure_document(mutant)
                with self.assertRaisesRegex(
                    extract_upstream.ExtractionError, "identities are stale"
                ):
                    extract_upstream.validate_source_closure_document(
                        mutant, authenticate_extractor_sources=True
                    )

        real_source_identity = extract_upstream._source_identity

        def edited_generator(path: str) -> dict:
            identity = real_source_identity(path)
            if path == "scripts/contract/generate_source_closure.py":
                identity["byte_size"] += 1
                identity["sha256"] = "0" * 64
            return identity

        with mock.patch.object(
            extract_upstream, "_source_identity", side_effect=edited_generator
        ):
            with self.assertRaisesRegex(
                extract_upstream.ExtractionError, "identities are stale"
            ):
                extract_upstream.validate_source_closure_document(
                    self.document, authenticate_extractor_sources=True
                )

    def leaf_identity_mutants(self) -> dict[str, dict]:
        mutations = {
            "content-size": lambda row: row.__setitem__("byte_size", row["byte_size"] + 1),
            "oid": lambda row: row.__setitem__(
                "blob_oid", "0" * len(row["blob_oid"]) if set(row["blob_oid"]) != {"0"} else "1" * len(row["blob_oid"])
            ),
            "sha256": lambda row: row.__setitem__(
                "sha256", "0" * 64 if row["sha256"] != "0" * 64 else "1" * 64
            ),
        }
        return {
            label: self.coherent(lambda doc, change=mutate_row: change(doc["files"][0]))
            for label, mutate_row in mutations.items()
        }

    def test_coherent_leaf_identity_changes_pass_pure_representation_validation(self) -> None:
        for label, mutant in self.leaf_identity_mutants().items():
            with self.subTest(label=label):
                extract_upstream.validate_source_closure_document(mutant)

    def test_coherent_leaf_identity_changes_require_git_reextraction(self) -> None:
        self.require_live_checkout()
        for label, mutant in self.leaf_identity_mutants().items():
            with self.subTest(label=label):
                with TemporaryDirectory(prefix="tgrad-source-closure-") as temporary:
                    path = Path(temporary) / "mutant.json"
                    path.write_bytes(extract_upstream.canonical_bytes(mutant))
                    with self.assertRaisesRegex(
                        extract_upstream.ExtractionError,
                        "differs from foreign Git re-extraction",
                    ):
                        extract_upstream.check_source_closure_against_git(path, self.checkout)

    def test_coherent_non_special_omission_passes_pure_validation(self) -> None:
        extract_upstream.validate_source_closure_document(
            self.coherent_non_special_omission()
        )

    def test_coherent_non_special_omission_requires_git_reextraction(self) -> None:
        self.require_live_checkout()
        mutant = self.coherent_non_special_omission()
        with TemporaryDirectory(prefix="tgrad-source-closure-") as temporary:
            path = Path(temporary) / "omission.json"
            path.write_bytes(extract_upstream.canonical_bytes(mutant))
            with self.assertRaisesRegex(
                extract_upstream.ExtractionError, "differs from foreign Git re-extraction"
            ):
                extract_upstream.check_source_closure_against_git(path, self.checkout)

    def test_stale_generated_lean_is_rejected_offline(self) -> None:
        fixture = extract_upstream.SOURCE_CLOSURE_OUTPUT
        with TemporaryDirectory(prefix="tgrad-source-closure-") as temporary:
            stale = Path(temporary) / "SourceClosureGenerated.lean"
            stale.write_text("stale\n", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(generate_source_closure.__file__),
                    "--check-projection",
                    "--input", str(fixture),
                    "--output", str(stale),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(result.returncode, 1)
        self.assertIn("generated Lean projection is stale", result.stderr)

    def test_full_check_reextracts_git_before_lean_projection_comparison(self) -> None:
        self.require_live_checkout()
        mutant = self.coherent_non_special_omission()
        with TemporaryDirectory(prefix="tgrad-source-closure-") as temporary:
            fixture = Path(temporary) / "omission.json"
            fixture.write_bytes(extract_upstream.canonical_bytes(mutant))
            stale = Path(temporary) / "SourceClosureGenerated.lean"
            stale.write_text("stale\n", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(generate_source_closure.__file__),
                    "--check",
                    "--input", str(fixture),
                    "--output", str(stale),
                    "--checkout", str(self.checkout),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(result.returncode, 1)
        self.assertIn("differs from foreign Git re-extraction", result.stderr)
        self.assertNotIn("generated Lean projection is stale", result.stderr)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SourceClosureTests)
    count = suite.countTestCases()
    print(f"source-closure tests discovered: {count}", file=sys.stderr)
    if count == 0:
        raise SystemExit("refusing a green source-closure run with zero tests")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
