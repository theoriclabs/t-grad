#!/usr/bin/env python3
"""Project the canonical source-closure JSON into a raw Lean candidate.

Lean validates the imported representation.  In check mode this script first
re-extracts the foreign Git object database and byte-compares the canonical
JSON, which is the content-authentication boundary.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.parity import extract_upstream, upstream_target

DEFAULT_INPUT = extract_upstream.SOURCE_CLOSURE_OUTPUT
DEFAULT_OUTPUT = REPO_ROOT / "Tgrad" / "Contract" / "SourceClosureGenerated.lean"


def lean_string(value: str) -> str:
    pieces = ['"']
    for character in value:
        code = ord(character)
        if character == '"':
            pieces.append(r'\"')
        elif character == "\\":
            pieces.append(r"\\")
        elif character == "\n":
            pieces.append(r"\n")
        elif character == "\r":
            pieces.append(r"\r")
        elif character == "\t":
            pieces.append(r"\t")
        elif 0x20 <= code <= 0x7E:
            pieces.append(character)
        else:
            pieces.append(f"\\u{{{code:x}}}")
    pieces.append('"')
    return "".join(pieces)


def lean_list(values: list, render, *, indent: int = 2) -> str:
    if not values:
        return "[]"
    padding = " " * indent
    return "[\n" + ",\n".join(padding + "  " + render(value) for value in values) + f"\n{padding}]"


def render_target(value: dict) -> str:
    return (
        "{ repository := " + lean_string(value["repository"])
        + ", revision := " + lean_string(value["revision"])
        + ", tree := " + lean_string(value["tree"])
        + ", objectFormat := " + lean_string(value["object_format"])
        + ", disposition := .extractedCandidate }"
    )


def render_source_identity(value: dict) -> str:
    return (
        "{ path := " + lean_string(value["path"])
        + f", byteSize := {value['byte_size']}"
        + ", sha256 := " + lean_string(value["sha256"]) + " }"
    )


def render_file_identity(value: dict) -> str:
    return (
        "{ path := " + lean_string(value["path"])
        + ", mode := " + lean_string(value["mode"])
        + ", blobOid := " + lean_string(value["blob_oid"])
        + f", byteSize := {value['byte_size']}"
        + ", sha256 := " + lean_string(value["sha256"]) + " }"
    )


CATEGORY_CONSTRUCTORS = {
    "api_surface_tests": ".apiSurfaceTests",
    "backends": ".backends",
    "dtypes": ".dtypes",
    "extractor": ".extractor",
    "ops": ".ops",
    "tensor_api": ".tensorApi",
    "tinygrad_python": ".tinygradPython",
    "upstream_tests": ".upstreamTests",
}


def render_category(value: dict) -> str:
    paths = lean_list(value["paths"], lean_string, indent=6)
    source_kind = ".localExtractor" if value["source_kind"] == "local_extractor" else ".foreignGit"
    return (
        "{ id := " + CATEGORY_CONSTRUCTORS[value["id"]]
        + ", status := .complete, sourceKind := " + source_kind
        + ", paths := " + paths
        + f", fileCount := {value['file_count']}"
        + ", inventorySha256 := " + lean_string(value["inventory_sha256"]) + " }"
    )


def render_tensor_scope(value: dict) -> str:
    declaring_class = (
        "none" if value["declaring_class"] is None
        else "some " + lean_string(value["declaring_class"])
    )
    status = ".selectedClass" if value["status"] == "selected_class" else ".explicitNoClass"
    return (
        "{ source := " + lean_string(value["source"])
        + ", declaringClass := " + declaring_class
        + ", status := " + status + " }"
    )


def render_tensor_declaration(value: dict) -> str:
    return (
        "{ source := " + lean_string(value["source"])
        + ", declaringClass := " + lean_string(value["declaring_class"])
        + ", kind := ." + value["kind"]
        + ", name := " + lean_string(value["name"])
        + ", structuralSignature := " + lean_string(value["structural_signature"])
        + ", signatureSha256 := " + lean_string(value["signature_sha256"]) + " }"
    )


def render_named(value: dict) -> str:
    return (
        "{ name := " + lean_string(value["name"])
        + ", source := " + lean_string(value["source"]) + " }"
    )


LIMIT_CONSTRUCTORS = {
    "backend_execution": ".backendExecution",
    "catalog_closure": ".catalogClosure",
    "docs_anchors": ".docsAnchors",
    "official_workloads": ".officialWorkloads",
    "public_export_semantics": ".publicExportSemantics",
    "pytest_node_ids": ".pytestNodeIds",
    "requirement_interpretation": ".requirementInterpretation",
    "requirement_rows_590": ".requirementRows590",
    "runtime_build_attestation": ".runtimeBuildAttestation",
    "runtime_parity": ".runtimeParity",
    "runtime_resolved_tensor_behavior": ".runtimeResolvedTensorBehavior",
    "scenario_adequacy": ".scenarioAdequacy",
    "target_promotion": ".targetPromotion",
}


def render_candidate(document: dict) -> str:
    extractor = document["extractor"]
    tensor = document["tensor_api"]
    ops = document["ops"]
    dtypes = document["dtypes"]
    backends = document["backends"]
    tests = document["tests"]
    lines = [
        "import Tgrad.Contract.SourceClosure",
        "",
        "/-! This file is generated from the canonical source-closure JSON.",
        "It constructs only the public raw candidate.  Python re-extraction",
        "authenticates Git content; Lean validates this imported representation. -/",
        "",
        "namespace Tgrad.Contract",
        "",
        "set_option maxRecDepth 100000 in",
        "def sourceClosureCandidate : SourceClosureCandidate :=",
        "  { schema := " + lean_string(document["schema"]),
        "    target := " + render_target(document["target"]),
        "    repository := { entryCount := " + str(document["repository"]["entry_count"])
        + ", inventorySha256 := " + lean_string(document["repository"]["inventory_sha256"]) + " }",
        "    extractor :=",
        "      { policyId := " + lean_string(extractor["policy_id"]),
        "        canonicalizerId := " + lean_string(extractor["canonicalizer_id"]),
        "        parserPolicyId := " + lean_string(extractor["parser_policy_id"]),
        "        parserGrammarFeature := " + lean_string(extractor["parser_grammar_feature"]),
        "        pythonImplementation := " + lean_string(extractor["python_implementation"]),
        "        pythonMajorMinorMin := " + lean_string(extractor["python_major_minor_min"]),
        "        pythonMajorMinorMax := " + lean_string(extractor["python_major_minor_max"]),
        "        sourceFiles := " + lean_list(extractor["source_files"], render_source_identity, indent=8),
        "        sourceBundleSha256 := " + lean_string(extractor["source_bundle_sha256"]) + " }",
        "    files := " + lean_list(document["files"], render_file_identity, indent=4),
        "    categories := " + lean_list(document["categories"], render_category, indent=4),
        "    tensorApi :=",
        "      { sources := " + lean_list(tensor["sources"], lean_string, indent=8),
        "        sourceScopes := " + lean_list(tensor["source_scopes"], render_tensor_scope, indent=8),
        "        declarations := " + lean_list(tensor["declarations"], render_tensor_declaration, indent=8),
        "        directMethods := " + lean_list(tensor["direct_methods"], lean_string, indent=8),
        "        methodNames := " + lean_list(tensor["method_names"], lean_string, indent=8),
        "        propertyNames := " + lean_list(tensor["property_names"], lean_string, indent=8),
        f"        declarationCount := {tensor['declaration_count']}",
        f"        directMethodCount := {tensor['direct_method_count']}",
        f"        methodCount := {tensor['method_count']}",
        f"        propertyCount := {tensor['property_count']} }}",
        "    ops :=",
        "      { source := " + lean_string(ops["source"]),
        "        declarations := " + lean_list(ops["declarations"], render_named, indent=8),
        f"        count := {ops['count']} }}",
        "    dtypes :=",
        "      { source := " + lean_string(dtypes["source"]),
        "        declarations := " + lean_list(dtypes["declarations"], render_named, indent=8),
        f"        count := {dtypes['count']} }}",
        "    backends :=",
        "      { declarations := " + lean_list(backends["declarations"], render_named, indent=8),
        f"        count := {backends['count']} }}",
        "    tests :=",
        "      { policyId := " + lean_string(tests["policy_id"]),
        "        groups := " + lean_list(tests["groups"], lean_string, indent=8),
        "        sources := " + lean_list(tests["sources"], lean_string, indent=8),
        f"        count := {tests['count']} }}",
        "    limits := [" + ", ".join(LIMIT_CONSTRUCTORS[row["id"]] for row in document["limits"]) + "]",
        "    closureSha256 := " + lean_string(document["closure_sha256"]) + " }",
        "",
        "set_option maxRecDepth 100000 in",
        "theorem sourceClosureCandidateValid :",
        "    (validateSourceClosure sourceClosureCandidate).isOk := by",
        "  native_decide",
        "",
        "end Tgrad.Contract",
        "",
    ]
    return "\n".join(lines)


def load_document(path: Path) -> dict:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise extract_upstream.ExtractionError(f"cannot read source closure {path}: {exc}") from exc
    return extract_upstream.parse_source_closure_bytes(
        raw, authenticate_extractor_sources=True
    )


def check_projection(input_path: Path, output_path: Path) -> dict:
    """Offline JSON-to-Lean staleness check; this does not authenticate Git."""
    document = load_document(input_path)
    expected = render_candidate(document).encode("utf-8")
    try:
        current = output_path.read_bytes()
    except FileNotFoundError as exc:
        raise extract_upstream.ExtractionError(
            f"generated Lean projection is missing: {output_path}"
        ) from exc
    if current != expected:
        raise extract_upstream.ExtractionError(
            f"generated Lean projection is stale: {output_path}"
        )
    return document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkout", type=Path, default=upstream_target.DEFAULT_ORACLE)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--check-projection",
        action="store_true",
        help="offline JSON-to-Lean staleness check (does not authenticate foreign Git)",
    )
    args = parser.parse_args()
    if args.check and args.check_projection:
        parser.error("--check and --check-projection are mutually exclusive")
    try:
        if args.check_projection:
            check_projection(args.input, args.output)
            print(f"source closure Lean projection is current: {args.output} (offline)")
            return 0
        if args.check:
            document = extract_upstream.check_source_closure_against_git(
                args.input, args.checkout
            )
        else:
            document = load_document(args.input)
        expected = render_candidate(document).encode("utf-8")
        if args.check:
            # Foreign Git authentication above deliberately precedes this
            # projection comparison.
            check_projection(args.input, args.output)
            print(f"source closure JSON and Lean projection are current: {args.output}")
            return 0
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(expected)
        print(f"wrote {args.output}")
        return 0
    except extract_upstream.ExtractionError as exc:
        print(f"source closure generation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
