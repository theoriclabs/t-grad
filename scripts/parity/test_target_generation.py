#!/usr/bin/env python3
"""Offline falsification tests for the upstream parity denominator."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "fixtures/parity/upstream_19c4d736f2bc.json"
TARGET = ROOT / "Tgrad/Spec/ParityTarget.lean"
EXTRACTOR = ROOT / "scripts/parity/extract_upstream.py"
RENDERER = ROOT / "scripts/parity/render_lean_target.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


extract = load_module("tgrad_extract_upstream", EXTRACTOR)


def canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def rehash(data: dict) -> None:
    data["section_sha256"] = {
        section: canonical_sha(data[section])
        for section in ("tensor_api", "dtypes", "ops", "backends", "tests")
    }
    body = {
        key: value for key, value in data.items()
        if key not in ("extracted_at_utc", "content_sha256")
    }
    data["content_sha256"] = canonical_sha(body)


class ExtractionFalsificationTests(unittest.TestCase):
    def test_wrong_ops_location_cannot_emit_empty_vocabulary(self) -> None:
        wrong_module = "from tinygrad.uop import Ops\n"
        with mock.patch.object(extract, "source_of", return_value=wrong_module):
            with self.assertRaisesRegex(extract.ExtractionError, "class Ops not found"):
                extract.ops_inventory("candidate")

    def test_missing_tensor_mixins_is_fatal(self) -> None:
        tensor_source = "class Tensor:\n    def realize(self): pass\n"
        with mock.patch.object(extract, "gh", return_value=[]), \
             mock.patch.object(extract, "source_of", return_value=tensor_source):
            with self.assertRaisesRegex(extract.ExtractionError, "no mixin modules"):
                extract.tensor_api("candidate")


class TargetGenerationTests(unittest.TestCase):
    def run_renderer(self, manifest: Path, output: Path, check: bool = False,
                     manifest_label: str | None = None) -> subprocess.CompletedProcess[str]:
        command = [sys.executable, str(RENDERER), "--manifest", str(manifest), "--output", str(output)]
        if manifest_label is not None:
            command += ["--manifest-label", manifest_label]
        if check:
            command.append("--check")
        return subprocess.run(command, cwd=ROOT, text=True, capture_output=True)

    def test_committed_target_is_current(self) -> None:
        result = self.run_renderer(MANIFEST, TARGET, check=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_wrong_but_internally_rehashed_ops_inventory_is_stale(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tgrad-parity-mutation-") as directory:
            mutated_path = Path(directory) / "mutated.json"
            data = json.loads(MANIFEST.read_text())
            data["ops"]["members"] = data["ops"]["members"][1:]
            data["counts"]["ops_members"] -= 1
            rehash(data)
            mutated_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
            result = self.run_renderer(
                mutated_path, TARGET, check=True,
                manifest_label="fixtures/parity/upstream_19c4d736f2bc.json",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("STALE", result.stdout)

    def test_hash_mutation_is_rejected_before_render(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tgrad-parity-hash-") as directory:
            mutated_path = Path(directory) / "mutated.json"
            data = copy.deepcopy(json.loads(MANIFEST.read_text()))
            data["content_sha256"] = "0" * 64
            mutated_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
            result = self.run_renderer(mutated_path, TARGET)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("content_sha256", result.stdout)

    def test_implicit_exclusions_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tgrad-parity-exclusions-") as directory:
            mutated_path = Path(directory) / "mutated.json"
            data = json.loads(MANIFEST.read_text())
            del data["exclusions"]
            rehash(data)
            mutated_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
            result = self.run_renderer(mutated_path, TARGET)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("exclusions", result.stdout)


if __name__ == "__main__":
    unittest.main()
