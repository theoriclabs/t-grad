#!/usr/bin/env python3
"""Generate the upstream parity manifest from a pinned tinygrad revision.

`Tgrad/Spec/Parity.lean` holds the upstream target as `.unknown` because
no denominator exists. Without one, "N% compatible" has nothing to
divide by, and a hand-authored checklist would be exactly the
self-referential denominator PARITY.md forbids. This produces the
denominator by reading tinygrad itself.

Everything is read at one pinned commit, so a re-run on the same SHA
produces the same manifest. The inventories are parsed with `ast`, not
regex, so a formatting change upstream does not silently alter counts.

If any inventory cannot be extracted, this **fails loudly and writes
nothing**. A partial manifest would become a wrong denominator, which
is worse than no denominator: it would make parity look closer than it
is, permanently, and nothing downstream could detect it.

Usage:
    python3 scripts/parity/extract_upstream.py            # pinned SHA
    python3 scripts/parity/extract_upstream.py --ref SHA  # another rev
    python3 scripts/parity/extract_upstream.py --print    # no write

Requires an authenticated `gh`.
"""
from __future__ import annotations

import argparse
import ast
import base64
import copy
import hashlib
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.parity import ensure_oracle, upstream_target

# The research candidate named in PARITY.md. Pinning it here is what
# makes the manifest reproducible; bump it deliberately, never silently.
DEFAULT_REF = upstream_target.REVISION
REPO = upstream_target.REPOSITORY_SLUG
OUT_DIR = REPO_ROOT / "fixtures" / "parity"

SOURCE_CLOSURE_SCHEMA = "tgrad.source-closure.v1"
SOURCE_CLOSURE_OUTPUT = REPO_ROOT / "fixtures" / "contract" / "source_closure_19c4d736f2bc.json"
EXTRACTOR_POLICY_ID = "tgrad.source-closure.extractor-policy.v1"
CANONICALIZER_ID = "tgrad.canonical-json.sorted-compact-utf8.v1"
PARSER_POLICY_ID = "cpython-ast.structural-signature.v1"
SUPPORTED_PYTHON_IMPLEMENTATION = "cpython"
SUPPORTED_PYTHON_MIN = (3, 12)
SUPPORTED_PYTHON_MAX = (3, 14)
PARSER_GRAMMAR_FEATURE_VERSION = (3, 12)
API_SURFACE_TEST_POLICY_ID = "tinygrad.api-surface-tests.null-unit-backend.non-init.v1"
EXTRACTOR_SOURCE_PATHS = (
    "scripts/contract/generate_source_closure.py",
    "scripts/parity/ensure_oracle.py",
    "scripts/parity/extract_upstream.py",
    "scripts/parity/upstream_target.py",
)

TENSOR_SOURCE_SCOPES = {
    "tinygrad/mixin/creation.py": "CreationMixin",
    "tinygrad/mixin/dtype.py": "DTypeMixin",
    "tinygrad/mixin/elementwise.py": "ElementwiseMixin",
    "tinygrad/mixin/gradient.py": None,
    "tinygrad/mixin/movement.py": "MovementMixin",
    "tinygrad/mixin/op.py": "OpMixin",
    "tinygrad/mixin/rand.py": "RandMixin",
    "tinygrad/mixin/reduce.py": "ReduceMixin",
    "tinygrad/tensor.py": "Tensor",
}

CATEGORY_IDS = (
    "api_surface_tests",
    "backends",
    "dtypes",
    "extractor",
    "ops",
    "tensor_api",
    "tinygrad_python",
    "upstream_tests",
)

LIMIT_STATEMENTS = {
    "backend_execution": "This packet does not extract or prove backend execution.",
    "catalog_closure": "This packet does not extract or prove catalog closure and does not construct it.",
    "docs_anchors": "This packet does not extract or prove documentation anchors.",
    "official_workloads": "This packet does not extract or prove official workloads.",
    "public_export_semantics": "This packet does not extract or prove public-export semantics.",
    "pytest_node_ids": "This packet does not extract or prove pytest node IDs.",
    "requirement_interpretation": "This packet does not extract or prove requirement interpretation.",
    "requirement_rows_590": "This packet does not extract or prove the 590 requirement rows and does not import, validate, or discharge them.",
    "runtime_build_attestation": "This packet does not extract or prove runtime/build attestation and does not construct it.",
    "runtime_parity": "This packet does not extract or prove runtime parity.",
    "runtime_resolved_tensor_behavior": "This packet does not extract or prove runtime-resolved Tensor behavior.",
    "scenario_adequacy": "This packet does not extract or prove scenario adequacy.",
    "target_promotion": "This packet does not extract or prove target promotion and does not construct it.",
}

EXPECTED_PIN_COUNTS = {
    "tensor_direct_methods": 47,
    "tensor_methods": 295,
    "tensor_properties": 5,
    "ops": 82,
    "dtypes": 52,
    "backends": 16,
    "api_surface_tests": 138,
    "upstream_tests": 331,
}

# Regression-only number produced by the legacy every-class walk, which
# incorrectly admitted _ContextVar.get and _ContextVar.set as Tensor methods.
LEGACY_HELPER_CONTAMINATION_METHOD_COUNT = 297

TEST_GROUPS = ("null", "unit", "backend")


class ExtractionError(RuntimeError):
    pass


def gh(path: str) -> object:
    p = subprocess.run(
        ["gh", "api", f"repos/{REPO}/{path}"], capture_output=True, text=True
    )
    if p.returncode != 0:
        raise ExtractionError(f"gh api {path} failed: {p.stderr.strip()[:200]}")
    return json.loads(p.stdout)


def source_of(path: str, ref: str) -> str:
    blob = gh(f"contents/{path}?ref={ref}")
    if not isinstance(blob, dict) or "content" not in blob:
        raise ExtractionError(f"{path}: no file content at {ref[:12]}")
    return base64.b64decode(blob["content"]).decode("utf-8")


def parse(path: str, src: str) -> ast.Module:
    try:
        return ast.parse(src)
    except SyntaxError as e:
        raise ExtractionError(f"{path}: unparseable ({e})") from e


def class_named(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise ExtractionError(f"class {name} not found")


def _public_members(cls: ast.ClassDef) -> tuple[set, set]:
    methods, properties = set(), set()
    for node in cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_") and not node.name.startswith("__"):
                continue
            decorators = {
                d.id for d in node.decorator_list if isinstance(d, ast.Name)
            }
            (properties if "property" in decorators else methods).add(node.name)
    return methods, properties


def tensor_api(ref: str) -> dict:
    """Public callables and properties reachable on tinygrad's Tensor.

    `class Tensor(RandMixin)` carries only a fraction of the surface;
    the rest lives in `tinygrad/mixin/` (creation, dtype, elementwise,
    gradient, movement, op, rand, reduce). Reading tensor.py alone
    reports 47 methods against a true surface several times larger.

    That error only ever points one way — it shrinks the denominator
    and makes parity look closer than it is — so the mixin chain is
    walked and provenance is recorded per source.
    """
    sources = ["tinygrad/tensor.py"]
    listing = gh(f"contents/tinygrad/mixin?ref={ref}")
    sources += sorted(
        f"tinygrad/mixin/{e['name']}"
        for e in listing
        if e["name"].endswith(".py") and e["name"] != "__init__.py"
    )
    if len(sources) == 1:
        raise ExtractionError("no mixin modules found; Tensor surface would be understated")

    methods, properties, by_source = set(), set(), {}
    for path in sources:
        tree = parse(path, source_of(path, ref))
        m_here, p_here = set(), set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                m, p = _public_members(node)
                m_here |= m
                p_here |= p
        methods |= m_here
        properties |= p_here
        by_source[path] = sorted(m_here | p_here)

    if not methods:
        raise ExtractionError("Tensor exposes no public methods; parse is wrong")
    # `mixin/gradient.py` legitimately contributes 0: it holds module-level
    # autograd functions (compute_gradient, call_gradient), not Tensor
    # methods. Do not "fix" that zero by widening this to module scope.
    return {
        "sources": sources,
        "methods": sorted(methods),
        "properties": sorted(properties),
        "by_source": by_source,
    }


def dtype_inventory(ref: str) -> dict:
    """Names declared on tinygrad's `dtypes` class."""
    path = "tinygrad/dtype.py"
    cls = class_named(parse(path, source_of(path, ref)), "dtypes")
    names = set()
    for node in cls.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and not t.id.startswith("_"):
                    names.add(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if not node.target.id.startswith("_"):
                names.add(node.target.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                names.add(node.name)
    if not names:
        raise ExtractionError("dtypes class is empty; parse is wrong")
    return {"source": path, "names": sorted(names)}


def ops_inventory(ref: str) -> dict:
    """Members of the Ops vocabulary.

    Ops lives in `tinygrad/uop/__init__.py`, not `ops.py` — `ops.py`
    does `from tinygrad.uop import Ops`. Worth pinning in a comment
    because the obvious guess is wrong and a silent miss here would
    understate the denominator.
    """
    path = "tinygrad/uop/__init__.py"
    cls = class_named(parse(path, source_of(path, ref)), "Ops")
    members = set()
    for node in cls.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and not t.id.startswith("_"):
                    members.add(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if not node.target.id.startswith("_"):
                members.add(node.target.id)
    if not members:
        raise ExtractionError("Ops enum is empty; parse is wrong")
    return {"source": path, "members": sorted(members)}


def backend_inventory(ref: str) -> dict:
    listing = gh(f"contents/tinygrad/runtime?ref={ref}")
    names = sorted(
        e["name"][len("ops_"):-len(".py")]
        for e in listing
        if e["name"].startswith("ops_") and e["name"].endswith(".py")
    )
    if not names:
        raise ExtractionError("no runtime ops_*.py backends found")
    return {"source": "tinygrad/runtime/", "names": names}


def test_inventory(ref: str) -> dict:
    groups = {}
    for group in TEST_GROUPS:
        listing = gh(f"contents/test/{group}?ref={ref}")
        files = sorted(
            {"name": e["name"], "bytes": e["size"]}.items() and
            (e["name"], e["size"])
            for e in listing
            if e["name"].endswith(".py") and e["name"] != "__init__.py"
        )
        if not files:
            raise ExtractionError(f"test/{group} has no test files")
        groups[group] = {
            "files": [{"name": n, "bytes": b} for n, b in files],
            "count": len(files),
            "bytes": sum(b for _, b in files),
        }
    return {"source": "test/", "groups": groups}


def legacy_main(args: argparse.Namespace) -> int:
    try:
        head = gh(f"commits/{args.ref}")
        ref = head["sha"]
        manifest = {
            "upstream_repo": REPO,
            "upstream_ref": ref,
            "upstream_committed_at": head["commit"]["committer"]["date"],
            "upstream_subject": head["commit"]["message"].split("\n")[0],
            "extracted_at_utc": datetime.now(timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "extractor": "scripts/parity/extract_upstream.py",
            "tensor_api": tensor_api(ref),
            "dtypes": dtype_inventory(ref),
            "ops": ops_inventory(ref),
            "backends": backend_inventory(ref),
            "tests": test_inventory(ref),
        }
    except ExtractionError as e:
        print(f"extract_upstream: FAILED — {e}", file=sys.stderr)
        print(
            "  Nothing written. A partial manifest becomes a wrong denominator,\n"
            "  which is worse than none: it makes parity look closer than it is\n"
            "  and nothing downstream can detect it.",
            file=sys.stderr,
        )
        return 1

    t = manifest["tests"]["groups"]
    counts = {
        "tensor_methods": len(manifest["tensor_api"]["methods"]),
        "tensor_properties": len(manifest["tensor_api"]["properties"]),
        "dtype_names": len(manifest["dtypes"]["names"]),
        "ops_members": len(manifest["ops"]["members"]),
        "backends": len(manifest["backends"]["names"]),
        "test_files": sum(g["count"] for g in t.values()),
        "test_files_no_backend": t["null"]["count"],
    }
    manifest["counts"] = counts
    # Content digest over everything except the wall-clock stamp, so two
    # runs at the same ref are comparable byte-for-byte. The evidence
    # audit exists because unverifiable provenance is how this repo ended
    # up with 37 files certifying a commit that was not in the tree.
    body = {k: v for k, v in manifest.items() if k != "extracted_at_utc"}
    manifest["content_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True).encode()
    ).hexdigest()

    print(f"upstream {REPO} @ {ref[:12]}  ({manifest['upstream_committed_at']})")
    print(f"  {manifest['upstream_subject']}")
    for k, v in counts.items():
        print(f"  {k:24s} {v}")

    if args.print_only:
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"upstream_{ref[:12]}.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {out.relative_to(OUT_DIR.parents[1])}")
    return 0


# ---------------------------------------------------------------------------
# Source-closure v1.  This is an extension of the existing extractor rather
# than a second authority.  The legacy manifest path above intentionally keeps
# its historical GitHub-API behavior and output unchanged.


def canonical_payload(value: object) -> bytes:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ExtractionError(f"cannot canonicalize JSON: {exc}") from exc
    return rendered.encode("utf-8")


def canonical_bytes(value: object) -> bytes:
    return canonical_payload(value) + b"\n"


def digest_value(value: object) -> str:
    return hashlib.sha256(canonical_payload(value)).hexdigest()


def _git_bytes(checkout: Path, *args: str, input_bytes: bytes | None = None) -> bytes:
    result = ensure_oracle.git(checkout, *args, input_bytes=input_bytes)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()[:400]
        raise ExtractionError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout


def _read_foreign_tree(checkout: Path) -> tuple[list[dict], dict[str, bytes]]:
    """Enumerate the pinned tree and read every blob from Git's object DB."""
    raw_tree = _git_bytes(
        checkout, "ls-tree", "-r", "-z", upstream_target.REVISION
    )
    entries: list[tuple[str, str, str, str]] = []
    try:
        for raw in raw_tree.split(b"\0"):
            if not raw:
                continue
            meta, path_bytes = raw.split(b"\t", 1)
            mode, object_type, oid = meta.decode("ascii").split(" ")
            repo_path = path_bytes.decode("utf-8")
            if object_type != "blob":
                raise ExtractionError(
                    f"tracked entry is not a blob: {repo_path} ({object_type})"
                )
            entries.append((mode, object_type, oid, repo_path))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ExtractionError(f"malformed or non-UTF-8 ls-tree output: {exc}") from exc
    entries.sort(key=lambda row: row[3])
    if not entries:
        raise ExtractionError("pinned tree has no tracked blobs")

    query = b"".join(row[2].encode("ascii") + b"\n" for row in entries)
    batch = _git_bytes(checkout, "cat-file", "--batch", input_bytes=query)
    offset = 0
    contents: dict[str, bytes] = {}
    files: list[dict] = []
    try:
        for mode, _, expected_oid, repo_path in entries:
            newline = batch.index(b"\n", offset)
            header = batch[offset:newline].decode("ascii")
            offset = newline + 1
            parts = header.split(" ")
            if len(parts) != 3 or parts[1] != "blob":
                raise ExtractionError(f"missing blob for {repo_path}: {header}")
            actual_oid, _, size_text = parts
            size = int(size_text)
            blob = batch[offset:offset + size]
            offset += size
            if len(blob) != size or batch[offset:offset + 1] != b"\n":
                raise ExtractionError(f"truncated blob stream for {repo_path}")
            offset += 1
            if actual_oid != expected_oid:
                raise ExtractionError(
                    f"blob identity mismatch for {repo_path}: {actual_oid} != {expected_oid}"
                )
            recomputed_oid = ensure_oracle.git_blob_oid(
                upstream_target.OBJECT_FORMAT, blob
            )
            if recomputed_oid != expected_oid:
                raise ExtractionError(
                    f"blob payload identity mismatch for {repo_path}: "
                    f"{recomputed_oid} != {expected_oid}"
                )
            contents[repo_path] = blob
            files.append(
                {
                    "blob_oid": actual_oid,
                    "byte_size": len(blob),
                    "mode": mode,
                    "path": repo_path,
                    "sha256": hashlib.sha256(blob).hexdigest(),
                }
            )
    except (UnicodeDecodeError, ValueError) as exc:
        raise ExtractionError(f"malformed cat-file batch stream: {exc}") from exc
    if offset != len(batch):
        raise ExtractionError("cat-file batch stream has trailing bytes")
    return files, contents


def parse_python_bytes(path: str, raw: bytes) -> ast.Module:
    try:
        source = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ExtractionError(f"{path}: Python source is not UTF-8 ({exc})") from exc
    try:
        return ast.parse(
            source, filename=path, feature_version=PARSER_GRAMMAR_FEATURE_VERSION
        )
    except SyntaxError as exc:
        raise ExtractionError(f"{path}: Python AST parse failed ({exc})") from exc


def validate_parser_runtime() -> None:
    implementation = sys.implementation.name
    major_minor = (sys.version_info.major, sys.version_info.minor)
    if (
        implementation != SUPPORTED_PYTHON_IMPLEMENTATION
        or not (SUPPORTED_PYTHON_MIN <= major_minor <= SUPPORTED_PYTHON_MAX)
    ):
        raise ExtractionError(
            "unsupported Python AST tool identity: "
            f"{implementation} {major_minor[0]}.{major_minor[1]}; expected "
            f"{SUPPORTED_PYTHON_IMPLEMENTATION} "
            f"{SUPPORTED_PYTHON_MIN[0]}.{SUPPORTED_PYTHON_MIN[1]} through "
            f"{SUPPORTED_PYTHON_MAX[0]}.{SUPPORTED_PYTHON_MAX[1]}"
        )


def _is_tinygrad_python(path: str) -> bool:
    return path.startswith("tinygrad/") and path.endswith(".py")


def _is_upstream_test(path: str) -> bool:
    return path.startswith("test/") and path.endswith(".py")


def _is_api_surface_test(path: str) -> bool:
    return (
        path.endswith(".py")
        and path.rsplit("/", 1)[-1] != "__init__.py"
        and any(path.startswith(f"test/{group}/") for group in TEST_GROUPS)
    )


def _is_tensor_source(path: str) -> bool:
    return path == "tinygrad/tensor.py" or (
        path.startswith("tinygrad/mixin/")
        and path.endswith(".py")
        and path.rsplit("/", 1)[-1] != "__init__.py"
    )


def _is_backend_source(path: str) -> bool:
    prefix = "tinygrad/runtime/ops_"
    return path.startswith(prefix) and path.endswith(".py") and "/" not in path[len(prefix):]


def _source_identity(path: str) -> dict:
    local = REPO_ROOT / path
    try:
        raw = local.read_bytes()
    except OSError as exc:
        raise ExtractionError(f"cannot read extractor source {path}: {exc}") from exc
    return {
        "byte_size": len(raw),
        "path": path,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _category(
    category_id: str,
    source_kind: str,
    paths: list[str],
    identities: dict[str, dict],
) -> dict:
    ordered = sorted(paths)
    members = []
    for path in ordered:
        try:
            members.append(identities[path])
        except KeyError as exc:
            raise ExtractionError(
                f"category {category_id} references missing identity {path}"
            ) from exc
    return {
        "file_count": len(ordered),
        "id": category_id,
        "inventory_sha256": digest_value(members),
        "paths": ordered,
        "source_kind": source_kind,
        "status": "complete",
    }


def _top_level_class(tree: ast.Module, name: str, path: str) -> ast.ClassDef:
    matches = [
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name
    ]
    if len(matches) != 1:
        raise ExtractionError(
            f"{path}: expected exactly one top-level class {name}, found {len(matches)}"
        )
    return matches[0]


def _stable_constant(value: object) -> dict:
    if value is None:
        return {"kind": "none"}
    if value is Ellipsis:
        return {"kind": "ellipsis"}
    if isinstance(value, bool):
        return {"kind": "bool", "value": value}
    if isinstance(value, int):
        return {"kind": "int", "value": str(value)}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ExtractionError("non-finite float in structural signature")
        return {"kind": "float", "value": value.hex()}
    if isinstance(value, complex):
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ExtractionError("non-finite complex in structural signature")
        return {
            "imag": value.imag.hex(),
            "kind": "complex",
            "real": value.real.hex(),
        }
    if isinstance(value, str):
        return {"kind": "str", "value": value}
    if isinstance(value, bytes):
        return {"kind": "bytes", "value": value.hex()}
    raise ExtractionError(
        f"unsupported constant {type(value).__name__} in structural signature"
    )


def _stable_expr(node: ast.expr | None) -> object:
    """Select only versioned, source-semantic AST fields used by policy v1."""
    if node is None:
        return None
    if isinstance(node, ast.Constant):
        return {"constant": _stable_constant(node.value)}
    if isinstance(node, ast.Name):
        return {"id": node.id, "node": "name"}
    if isinstance(node, ast.Attribute):
        return {"attr": node.attr, "node": "attribute", "value": _stable_expr(node.value)}
    if isinstance(node, ast.Subscript):
        return {"node": "subscript", "slice": _stable_expr(node.slice), "value": _stable_expr(node.value)}
    if isinstance(node, ast.Starred):
        return {"node": "starred", "value": _stable_expr(node.value)}
    if isinstance(node, ast.Slice):
        return {
            "lower": _stable_expr(node.lower),
            "node": "slice",
            "step": _stable_expr(node.step),
            "upper": _stable_expr(node.upper),
        }
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return {
            "elements": [_stable_expr(element) for element in node.elts],
            "node": type(node).__name__.lower(),
        }
    if isinstance(node, ast.Dict):
        return {
            "items": [
                {"key": _stable_expr(key), "value": _stable_expr(value)}
                for key, value in zip(node.keys, node.values, strict=True)
            ],
            "node": "dict",
        }
    if isinstance(node, ast.Call):
        return {
            "arguments": [_stable_expr(argument) for argument in node.args],
            "function": _stable_expr(node.func),
            "keywords": [
                {"name": keyword.arg, "value": _stable_expr(keyword.value)}
                for keyword in node.keywords
            ],
            "node": "call",
        }
    if isinstance(node, ast.UnaryOp):
        return {
            "node": "unary",
            "operand": _stable_expr(node.operand),
            "operator": type(node.op).__name__,
        }
    if isinstance(node, ast.BinOp):
        return {
            "left": _stable_expr(node.left),
            "node": "binary",
            "operator": type(node.op).__name__,
            "right": _stable_expr(node.right),
        }
    if isinstance(node, ast.BoolOp):
        return {
            "node": "boolean",
            "operator": type(node.op).__name__,
            "values": [_stable_expr(value) for value in node.values],
        }
    if isinstance(node, ast.Compare):
        return {
            "comparators": [_stable_expr(value) for value in node.comparators],
            "left": _stable_expr(node.left),
            "node": "compare",
            "operators": [type(operator).__name__ for operator in node.ops],
        }
    if isinstance(node, ast.IfExp):
        return {
            "body": _stable_expr(node.body),
            "condition": _stable_expr(node.test),
            "else": _stable_expr(node.orelse),
            "node": "conditional",
        }
    if isinstance(node, ast.Lambda):
        return {
            "arguments": _stable_arguments(node.args),
            "body": _stable_expr(node.body),
            "node": "lambda",
        }
    if isinstance(node, ast.NamedExpr):
        return {
            "node": "named_expression",
            "target": _stable_expr(node.target),
            "value": _stable_expr(node.value),
        }
    if isinstance(node, ast.JoinedStr):
        return {"node": "joined_string", "values": [_stable_expr(value) for value in node.values]}
    if isinstance(node, ast.FormattedValue):
        return {
            "conversion": node.conversion,
            "format": _stable_expr(node.format_spec),
            "node": "formatted_value",
            "value": _stable_expr(node.value),
        }
    raise ExtractionError(
        f"unsupported AST expression {type(node).__name__} in structural signature; "
        "the parser policy must be versioned before accepting it"
    )


def _stable_argument(argument: ast.arg, kind: str, default: ast.expr | None) -> dict:
    return {
        "annotation": _stable_expr(argument.annotation),
        "default": _stable_expr(default),
        "kind": kind,
        "name": argument.arg,
        "type_comment": argument.type_comment,
    }


def _stable_arguments(arguments: ast.arguments) -> list[dict]:
    positional = list(arguments.posonlyargs) + list(arguments.args)
    defaults: list[ast.expr | None] = [None] * (len(positional) - len(arguments.defaults)) + list(arguments.defaults)
    rows = [
        _stable_argument(
            argument,
            "positional_only" if index < len(arguments.posonlyargs) else "positional_or_keyword",
            defaults[index],
        )
        for index, argument in enumerate(positional)
    ]
    if arguments.vararg is not None:
        rows.append(_stable_argument(arguments.vararg, "variadic_positional", None))
    rows.extend(
        _stable_argument(argument, "keyword_only", default)
        for argument, default in zip(arguments.kwonlyargs, arguments.kw_defaults, strict=True)
    )
    if arguments.kwarg is not None:
        rows.append(_stable_argument(arguments.kwarg, "variadic_keyword", None))
    return rows


def _structural_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[str, str]:
    type_params = getattr(node, "type_params", [])
    if type_params:
        raise ExtractionError(
            "function type parameters are outside structural-signature policy v1"
        )
    shape = {
        "arguments": _stable_arguments(node.args),
        "async": isinstance(node, ast.AsyncFunctionDef),
        "decorators": [_stable_expr(decorator) for decorator in node.decorator_list],
        "returns": _stable_expr(node.returns),
        "type_comment": node.type_comment,
    }
    encoded = canonical_payload(shape).decode("utf-8")
    return encoded, hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _tensor_declarations(
    source: str, declaring_class: str, cls: ast.ClassDef
) -> list[dict]:
    declarations: list[dict] = []
    for node in cls.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("_") and not node.name.startswith("__"):
            continue
        decorators = {
            decorator.id
            for decorator in node.decorator_list
            if isinstance(decorator, ast.Name)
        }
        kind = "property" if "property" in decorators else "method"
        structural_signature, signature_sha256 = _structural_signature(node)
        declarations.append(
            {
                "declaring_class": declaring_class,
                "kind": kind,
                "name": node.name,
                "signature_sha256": signature_sha256,
                "source": source,
                "structural_signature": structural_signature,
            }
        )
    return sorted(
        declarations,
        key=lambda row: (row["kind"], row["name"], row["source"], row["declaring_class"]),
    )


def build_source_closure(checkout: Path = upstream_target.DEFAULT_ORACLE) -> dict:
    checkout = checkout.resolve()
    validate_parser_runtime()
    ok, detail = ensure_oracle.verify(checkout)
    if not ok:
        raise ExtractionError(f"oracle verification failed: {detail}")

    files, contents = _read_foreign_tree(checkout)
    file_by_path = {row["path"]: row for row in files}
    paths = sorted(file_by_path)
    extractor_sources = [_source_identity(path) for path in EXTRACTOR_SOURCE_PATHS]
    extractor_by_path = {row["path"]: row for row in extractor_sources}

    category_paths = {
        "api_surface_tests": [path for path in paths if _is_api_surface_test(path)],
        "backends": [path for path in paths if _is_backend_source(path)],
        "dtypes": ["tinygrad/dtype.py"] if "tinygrad/dtype.py" in file_by_path else [],
        "extractor": list(EXTRACTOR_SOURCE_PATHS),
        "ops": ["tinygrad/uop/__init__.py"] if "tinygrad/uop/__init__.py" in file_by_path else [],
        "tensor_api": [path for path in paths if _is_tensor_source(path)],
        "tinygrad_python": [path for path in paths if _is_tinygrad_python(path)],
        "upstream_tests": [path for path in paths if _is_upstream_test(path)],
    }
    categories = [
        _category(
            category_id,
            "local_extractor" if category_id == "extractor" else "foreign_git",
            category_paths[category_id],
            extractor_by_path if category_id == "extractor" else file_by_path,
        )
        for category_id in CATEGORY_IDS
    ]

    # Decode and parse every Python file in both complete source categories.
    parsed: dict[str, ast.Module] = {}
    for path in sorted(
        set(category_paths["tinygrad_python"]) | set(category_paths["upstream_tests"])
    ):
        parsed[path] = parse_python_bytes(path, contents[path])

    if set(category_paths["tensor_api"]) != set(TENSOR_SOURCE_SCOPES):
        raise ExtractionError(
            "Tensor source set changed; parser policy requires an explicit scope decision"
        )
    tensor_source_scopes: list[dict] = []
    tensor_declarations: list[dict] = []
    for path in category_paths["tensor_api"]:
        declaring_class = TENSOR_SOURCE_SCOPES[path]
        if declaring_class is None:
            if any(isinstance(node, ast.ClassDef) for node in parsed[path].body):
                raise ExtractionError(
                    f"{path}: explicit no-class source unexpectedly declares a top-level class"
                )
            tensor_source_scopes.append(
                {"declaring_class": None, "source": path, "status": "explicit_no_class"}
            )
            continue
        selected = _top_level_class(parsed[path], declaring_class, path)
        tensor_source_scopes.append(
            {
                "declaring_class": declaring_class,
                "source": path,
                "status": "selected_class",
            }
        )
        tensor_declarations.extend(_tensor_declarations(path, declaring_class, selected))
    tensor_source_scopes.sort(key=lambda row: row["source"])
    tensor_declarations.sort(
        key=lambda row: (row["kind"], row["name"], row["source"], row["declaring_class"])
    )
    tensor_method_names = sorted(
        {row["name"] for row in tensor_declarations if row["kind"] == "method"}
    )
    tensor_property_names = sorted(
        {row["name"] for row in tensor_declarations if row["kind"] == "property"}
    )
    direct_methods = sorted({
        row["name"]
        for row in tensor_declarations
        if row["source"] == "tinygrad/tensor.py"
        and row["declaring_class"] == "Tensor"
        and row["kind"] == "method"
    })

    ops_source = "tinygrad/uop/__init__.py"
    ops_class = class_named(parsed[ops_source], "Ops")
    ops_members: set[str] = set()
    for node in ops_class.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    ops_members.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if not node.target.id.startswith("_"):
                ops_members.add(node.target.id)

    dtype_source = "tinygrad/dtype.py"
    dtype_class = class_named(parsed[dtype_source], "dtypes")
    dtype_names: set[str] = set()
    for node in dtype_class.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    dtype_names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if not node.target.id.startswith("_"):
                dtype_names.add(node.target.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                dtype_names.add(node.name)

    backends = [
        {
            "name": path[len("tinygrad/runtime/ops_"):-len(".py")],
            "source": path,
        }
        for path in category_paths["backends"]
    ]
    document = {
        "backends": {"count": len(backends), "declarations": backends},
        "categories": categories,
        "dtypes": {
            "count": len(dtype_names),
            "declarations": [
                {"name": name, "source": dtype_source} for name in sorted(dtype_names)
            ],
            "source": dtype_source,
        },
        "extractor": {
            "canonicalizer_id": CANONICALIZER_ID,
            "parser_policy_id": PARSER_POLICY_ID,
            "policy_id": EXTRACTOR_POLICY_ID,
            "python_implementation": SUPPORTED_PYTHON_IMPLEMENTATION,
            "python_major_minor_max": f"{SUPPORTED_PYTHON_MAX[0]}.{SUPPORTED_PYTHON_MAX[1]}",
            "python_major_minor_min": f"{SUPPORTED_PYTHON_MIN[0]}.{SUPPORTED_PYTHON_MIN[1]}",
            "parser_grammar_feature": f"{PARSER_GRAMMAR_FEATURE_VERSION[0]}.{PARSER_GRAMMAR_FEATURE_VERSION[1]}",
            "source_bundle_sha256": digest_value(extractor_sources),
            "source_files": extractor_sources,
        },
        "files": files,
        "limits": [
            {"id": limit_id, "statement": LIMIT_STATEMENTS[limit_id]}
            for limit_id in sorted(LIMIT_STATEMENTS)
        ],
        "ops": {
            "count": len(ops_members),
            "declarations": [
                {"name": name, "source": ops_source} for name in sorted(ops_members)
            ],
            "source": ops_source,
        },
        "repository": {
            "entry_count": len(files),
            "inventory_sha256": digest_value(files),
        },
        "schema": SOURCE_CLOSURE_SCHEMA,
        "target": {
            "disposition": "extracted_candidate",
            "object_format": upstream_target.OBJECT_FORMAT,
            "repository": upstream_target.REPOSITORY,
            "revision": upstream_target.REVISION,
            "tree": upstream_target.TREE,
        },
        "tensor_api": {
            "declaration_count": len(tensor_declarations),
            "declarations": tensor_declarations,
            "direct_method_count": len(direct_methods),
            "direct_methods": sorted(direct_methods),
            "method_count": len(tensor_method_names),
            "method_names": tensor_method_names,
            "property_count": len(tensor_property_names),
            "property_names": tensor_property_names,
            "source_scopes": tensor_source_scopes,
            "sources": sorted(category_paths["tensor_api"]),
        },
        "tests": {
            "count": len(category_paths["api_surface_tests"]),
            "groups": sorted(TEST_GROUPS),
            "policy_id": API_SURFACE_TEST_POLICY_ID,
            "sources": sorted(category_paths["api_surface_tests"]),
        },
    }
    document["closure_sha256"] = digest_value(document)
    validate_source_closure_document(document, authenticate_extractor_sources=True)
    return document


def _strict_object(value: object, expected: set[str], where: str) -> dict:
    if not isinstance(value, dict):
        raise ExtractionError(f"{where}: expected object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ExtractionError(f"{where}: fields mismatch; missing={missing}, unknown={unknown}")
    return value


def _strict_list(value: object, where: str) -> list:
    if not isinstance(value, list):
        raise ExtractionError(f"{where}: expected array")
    return value


def _valid_repo_path(path: object) -> bool:
    if not isinstance(path, str) or not path or "\\" in path or "\x00" in path:
        return False
    parsed = PurePosixPath(path)
    return (
        not parsed.is_absolute()
        and str(parsed) == path
        and all(part not in ("", ".", "..") for part in parsed.parts)
    )


def _valid_lower_hex(value: object, length: int) -> bool:
    return isinstance(value, str) and re.fullmatch(f"[0-9a-f]{{{length}}}", value) is not None


def _validate_file_identity(row: object, where: str, oid_len: int) -> dict:
    item = _strict_object(
        row, {"blob_oid", "byte_size", "mode", "path", "sha256"}, where
    )
    if not _valid_repo_path(item["path"]):
        raise ExtractionError(f"{where}: invalid repository path {item['path']!r}")
    if not isinstance(item["mode"], str) or re.fullmatch(r"[0-7]{6}", item["mode"]) is None:
        raise ExtractionError(f"{where}: malformed Git mode")
    if not _valid_lower_hex(item["blob_oid"], oid_len):
        raise ExtractionError(f"{where}: malformed blob OID")
    if not isinstance(item["byte_size"], int) or isinstance(item["byte_size"], bool) or item["byte_size"] < 0:
        raise ExtractionError(f"{where}: invalid byte size")
    if not _valid_lower_hex(item["sha256"], 64):
        raise ExtractionError(f"{where}: malformed SHA-256")
    return item


def _validate_source_identity(row: object, where: str) -> dict:
    item = _strict_object(row, {"byte_size", "path", "sha256"}, where)
    if not _valid_repo_path(item["path"]):
        raise ExtractionError(f"{where}: invalid extractor path")
    if not isinstance(item["byte_size"], int) or isinstance(item["byte_size"], bool) or item["byte_size"] < 0:
        raise ExtractionError(f"{where}: invalid byte size")
    if not _valid_lower_hex(item["sha256"], 64):
        raise ExtractionError(f"{where}: malformed SHA-256")
    return item


def refresh_source_closure_digests(document: dict) -> dict:
    """Recompute all derived counts/digests for coherent mutation tests."""
    doc = copy.deepcopy(document)
    files = doc.get("files", [])
    doc["repository"]["entry_count"] = len(files)
    doc["repository"]["inventory_sha256"] = digest_value(files)
    sources = doc["extractor"]["source_files"]
    doc["extractor"]["source_bundle_sha256"] = digest_value(sources)
    foreign = {row["path"]: row for row in files}
    local = {row["path"]: row for row in sources}
    for category in doc.get("categories", []):
        identities = local if category.get("source_kind") == "local_extractor" else foreign
        members = [identities[path] for path in category.get("paths", []) if path in identities]
        category["file_count"] = len(category.get("paths", []))
        category["inventory_sha256"] = digest_value(members)
    tensor = doc.get("tensor_api", {})
    tensor["declaration_count"] = len(tensor.get("declarations", []))
    tensor["method_count"] = len(tensor.get("method_names", []))
    tensor["property_count"] = len(tensor.get("property_names", []))
    tensor["direct_method_count"] = len(tensor.get("direct_methods", []))
    for key in ("ops", "dtypes"):
        if key in doc:
            doc[key]["count"] = len(doc[key].get("declarations", []))
    if "backends" in doc:
        doc["backends"]["count"] = len(doc["backends"].get("declarations", []))
    if "tests" in doc:
        doc["tests"]["count"] = len(doc["tests"].get("sources", []))
    body = {key: value for key, value in doc.items() if key != "closure_sha256"}
    doc["closure_sha256"] = digest_value(body)
    return doc


def validate_source_closure_document(
    document: object, *, authenticate_extractor_sources: bool = False
) -> dict:
    doc = _strict_object(
        document,
        {
            "backends", "categories", "closure_sha256", "dtypes", "extractor",
            "files", "limits", "ops", "repository", "schema", "target",
            "tensor_api", "tests",
        },
        "source closure",
    )
    if doc["schema"] != SOURCE_CLOSURE_SCHEMA:
        raise ExtractionError("source closure: wrong schema")
    target = _strict_object(
        doc["target"],
        {"disposition", "object_format", "repository", "revision", "tree"},
        "target",
    )
    expected_target = {
        "disposition": "extracted_candidate",
        "object_format": upstream_target.OBJECT_FORMAT,
        "repository": upstream_target.REPOSITORY,
        "revision": upstream_target.REVISION,
        "tree": upstream_target.TREE,
    }
    if target != expected_target:
        raise ExtractionError(f"target identity mismatch: {target!r}")
    oid_len = 40 if target["object_format"] == "sha1" else 64

    files_raw = _strict_list(doc["files"], "files")
    files = [
        _validate_file_identity(row, f"files[{index}]", oid_len)
        for index, row in enumerate(files_raw)
    ]
    file_paths = [row["path"] for row in files]
    if file_paths != sorted(file_paths) or len(file_paths) != len(set(file_paths)):
        raise ExtractionError("files: paths are unsorted or duplicated")
    repository = _strict_object(
        doc["repository"], {"entry_count", "inventory_sha256"}, "repository"
    )
    if repository["entry_count"] != len(files):
        raise ExtractionError("repository: entry_count is not derived from files")
    if repository["inventory_sha256"] != digest_value(files):
        raise ExtractionError("repository: inventory digest mismatch")

    extractor = _strict_object(
        doc["extractor"],
        {
            "canonicalizer_id", "parser_policy_id", "policy_id",
            "parser_grammar_feature", "python_implementation",
            "python_major_minor_max", "python_major_minor_min",
            "source_bundle_sha256", "source_files",
        },
        "extractor",
    )
    if extractor["policy_id"] != EXTRACTOR_POLICY_ID:
        raise ExtractionError("extractor: stale or unknown policy identity")
    if extractor["canonicalizer_id"] != CANONICALIZER_ID:
        raise ExtractionError("extractor: stale or unknown canonicalizer identity")
    if extractor["parser_policy_id"] != PARSER_POLICY_ID:
        raise ExtractionError("extractor: stale or unknown parser policy identity")
    if (
        extractor["python_implementation"] != SUPPORTED_PYTHON_IMPLEMENTATION
        or extractor["python_major_minor_min"]
            != f"{SUPPORTED_PYTHON_MIN[0]}.{SUPPORTED_PYTHON_MIN[1]}"
        or extractor["python_major_minor_max"]
            != f"{SUPPORTED_PYTHON_MAX[0]}.{SUPPORTED_PYTHON_MAX[1]}"
        or extractor["parser_grammar_feature"]
            != f"{PARSER_GRAMMAR_FEATURE_VERSION[0]}.{PARSER_GRAMMAR_FEATURE_VERSION[1]}"
    ):
        raise ExtractionError("extractor: unsupported Python AST tool identity")
    source_files_raw = _strict_list(extractor["source_files"], "extractor.source_files")
    source_files = [
        _validate_source_identity(row, f"extractor.source_files[{index}]")
        for index, row in enumerate(source_files_raw)
    ]
    source_paths = [row["path"] for row in source_files]
    if source_paths != sorted(EXTRACTOR_SOURCE_PATHS) or len(source_paths) != len(set(source_paths)):
        raise ExtractionError("extractor: exact source file set mismatch")
    if extractor["source_bundle_sha256"] != digest_value(source_files):
        raise ExtractionError("extractor: source bundle digest mismatch")
    if authenticate_extractor_sources:
        expected_sources = [_source_identity(path) for path in EXTRACTOR_SOURCE_PATHS]
        if source_files != expected_sources:
            raise ExtractionError("extractor: recorded source identities are stale")

    categories_raw = _strict_list(doc["categories"], "categories")
    categories: list[dict] = []
    for index, raw in enumerate(categories_raw):
        category = _strict_object(
            raw,
            {"file_count", "id", "inventory_sha256", "paths", "source_kind", "status"},
            f"categories[{index}]",
        )
        if category["id"] not in CATEGORY_IDS:
            raise ExtractionError(f"categories[{index}]: unknown category {category['id']!r}")
        if category["status"] != "complete":
            raise ExtractionError(f"categories[{index}]: category is not complete")
        expected_kind = "local_extractor" if category["id"] == "extractor" else "foreign_git"
        if category["source_kind"] != expected_kind:
            raise ExtractionError(f"categories[{index}]: wrong source kind")
        paths = _strict_list(category["paths"], f"categories[{index}].paths")
        if (
            not all(_valid_repo_path(path) for path in paths)
            or paths != sorted(paths)
            or len(paths) != len(set(paths))
        ):
            raise ExtractionError(f"categories[{index}]: invalid or duplicate paths")
        identities = {row["path"]: row for row in (source_files if expected_kind == "local_extractor" else files)}
        if not set(paths) <= set(identities):
            raise ExtractionError(f"categories[{index}]: path does not cross-reference an identity")
        members = [identities[path] for path in paths]
        if category["file_count"] != len(paths):
            raise ExtractionError(f"categories[{index}]: file_count is not derived")
        if category["inventory_sha256"] != digest_value(members):
            raise ExtractionError(f"categories[{index}]: inventory digest mismatch")
        categories.append(category)
    category_ids = [category["id"] for category in categories]
    if category_ids != list(CATEGORY_IDS) or len(category_ids) != len(set(category_ids)):
        raise ExtractionError("categories: exact required category set/order mismatch")
    by_category = {category["id"]: category for category in categories}

    expected_category_paths = {
        "api_surface_tests": sorted(path for path in file_paths if _is_api_surface_test(path)),
        "backends": sorted(path for path in file_paths if _is_backend_source(path)),
        "dtypes": ["tinygrad/dtype.py"],
        "extractor": sorted(EXTRACTOR_SOURCE_PATHS),
        "ops": ["tinygrad/uop/__init__.py"],
        "tensor_api": sorted(path for path in file_paths if _is_tensor_source(path)),
        "tinygrad_python": sorted(path for path in file_paths if _is_tinygrad_python(path)),
        "upstream_tests": sorted(path for path in file_paths if _is_upstream_test(path)),
    }
    for category_id, expected_paths in expected_category_paths.items():
        if by_category[category_id]["paths"] != expected_paths:
            raise ExtractionError(f"category {category_id}: extraction predicate mismatch")

    upstream_paths = by_category["upstream_tests"]["paths"]
    api_paths = by_category["api_surface_tests"]["paths"]
    if (
        len(upstream_paths) != EXPECTED_PIN_COUNTS["upstream_tests"]
        or len(api_paths) != EXPECTED_PIN_COUNTS["api_surface_tests"]
        or not set(api_paths) < set(upstream_paths)
        or by_category["upstream_tests"]["inventory_sha256"]
            == by_category["api_surface_tests"]["inventory_sha256"]
    ):
        raise ExtractionError(
            "upstream_tests and api_surface_tests are conflated or have wrong pinned counts"
        )

    tensor = _strict_object(
        doc["tensor_api"],
        {
            "declaration_count", "declarations", "direct_method_count",
            "direct_methods", "method_count", "method_names", "property_count",
            "property_names", "source_scopes", "sources",
        },
        "tensor_api",
    )
    tensor_sources = _strict_list(tensor["sources"], "tensor_api.sources")
    if tensor_sources != by_category["tensor_api"]["paths"]:
        raise ExtractionError("tensor_api: source/category cross-reference mismatch")

    scopes_raw = _strict_list(tensor["source_scopes"], "tensor_api.source_scopes")
    scopes: list[dict] = []
    for index, raw in enumerate(scopes_raw):
        scope = _strict_object(
            raw,
            {"declaring_class", "source", "status"},
            f"tensor_api.source_scopes[{index}]",
        )
        source = scope["source"]
        expected_class = TENSOR_SOURCE_SCOPES.get(source, object())
        expected_status = "explicit_no_class" if expected_class is None else "selected_class"
        if (
            source not in TENSOR_SOURCE_SCOPES
            or scope["declaring_class"] != expected_class
            or scope["status"] != expected_status
        ):
            raise ExtractionError(f"tensor_api.source_scopes[{index}]: scope mismatch")
        scopes.append(scope)
    expected_scopes = [
        {
            "declaring_class": TENSOR_SOURCE_SCOPES[source],
            "source": source,
            "status": (
                "explicit_no_class"
                if TENSOR_SOURCE_SCOPES[source] is None
                else "selected_class"
            ),
        }
        for source in sorted(TENSOR_SOURCE_SCOPES)
    ]
    if scopes != expected_scopes or [row["source"] for row in scopes] != tensor_sources:
        raise ExtractionError("tensor_api: exact source scope policy mismatch")

    declarations_raw = _strict_list(tensor["declarations"], "tensor_api.declarations")
    declarations: list[dict] = []
    for index, raw in enumerate(declarations_raw):
        row = _strict_object(
            raw,
            {
                "declaring_class", "kind", "name", "signature_sha256",
                "source", "structural_signature",
            },
            f"tensor_api.declarations[{index}]",
        )
        source = row["source"]
        if (
            source not in TENSOR_SOURCE_SCOPES
            or TENSOR_SOURCE_SCOPES[source] is None
            or row["declaring_class"] != TENSOR_SOURCE_SCOPES[source]
            or row["kind"] not in {"method", "property"}
            or not isinstance(row["name"], str)
            or not row["name"]
            or not isinstance(row["structural_signature"], str)
            or not row["structural_signature"]
            or not _valid_lower_hex(row["signature_sha256"], 64)
        ):
            raise ExtractionError(f"tensor_api.declarations[{index}]: invalid declaration")
        try:
            signature_value = json.loads(
                row["structural_signature"], object_pairs_hook=_reject_duplicate_pairs
            )
        except (json.JSONDecodeError, ExtractionError) as exc:
            raise ExtractionError(
                f"tensor_api.declarations[{index}]: invalid structural signature"
            ) from exc
        signature_bytes = row["structural_signature"].encode("utf-8")
        if (
            canonical_payload(signature_value) != signature_bytes
            or hashlib.sha256(signature_bytes).hexdigest() != row["signature_sha256"]
        ):
            raise ExtractionError(
                f"tensor_api.declarations[{index}]: structural signature digest mismatch"
            )
        declarations.append(row)
    declaration_keys = [
        (row["kind"], row["name"], row["source"], row["declaring_class"])
        for row in declarations
    ]
    if declaration_keys != sorted(declaration_keys) or len(declaration_keys) != len(set(declaration_keys)):
        raise ExtractionError("tensor_api: declarations are unsorted or duplicated")

    method_names = _strict_list(tensor["method_names"], "tensor_api.method_names")
    property_names = _strict_list(tensor["property_names"], "tensor_api.property_names")
    derived_method_names = sorted(
        {row["name"] for row in declarations if row["kind"] == "method"}
    )
    derived_property_names = sorted(
        {row["name"] for row in declarations if row["kind"] == "property"}
    )
    if method_names != derived_method_names or property_names != derived_property_names:
        raise ExtractionError("tensor_api: unique names are not derived from declarations")
    direct_methods = _strict_list(tensor["direct_methods"], "tensor_api.direct_methods")
    derived_direct_methods = sorted({
        row["name"]
        for row in declarations
        if row["source"] == "tinygrad/tensor.py"
        and row["declaring_class"] == "Tensor"
        and row["kind"] == "method"
    })
    if (
        direct_methods != derived_direct_methods
        or direct_methods != sorted(direct_methods)
        or len(direct_methods) != len(set(direct_methods))
    ):
        raise ExtractionError("tensor_api: invalid direct Tensor method inventory")
    if set(method_names) & set(property_names):
        raise ExtractionError("tensor_api: method/property declaration overlap")
    direct_count = len(direct_methods)
    if (
        tensor["declaration_count"] != len(declarations)
        or tensor["method_count"] != len(method_names)
        or tensor["property_count"] != len(property_names)
        or tensor["direct_method_count"] != direct_count
    ):
        raise ExtractionError("tensor_api: counts are not derived")
    if (
        direct_count != EXPECTED_PIN_COUNTS["tensor_direct_methods"]
        or len(method_names) != EXPECTED_PIN_COUNTS["tensor_methods"]
        or len(property_names) != EXPECTED_PIN_COUNTS["tensor_properties"]
        or len(tensor_sources) <= 1
    ):
        raise ExtractionError("tensor_api: pinned source-composition trap detected")

    ops = _strict_object(doc["ops"], {"count", "declarations", "source"}, "ops")
    if ops["source"] != "tinygrad/uop/__init__.py" or by_category["ops"]["paths"] != [ops["source"]]:
        raise ExtractionError("ops: must be extracted exactly from tinygrad/uop/__init__.py")
    ops_rows = _strict_list(ops["declarations"], "ops.declarations")
    ops_names: list[str] = []
    for index, raw in enumerate(ops_rows):
        row = _strict_object(raw, {"name", "source"}, f"ops.declarations[{index}]")
        if not isinstance(row["name"], str) or not row["name"] or row["source"] != ops["source"]:
            raise ExtractionError(f"ops.declarations[{index}]: invalid declaration")
        ops_names.append(row["name"])
    if ops_names != sorted(ops_names) or len(ops_names) != len(set(ops_names)):
        raise ExtractionError("ops: declarations are unsorted or duplicated")
    if ops["count"] != len(ops_rows) or len(ops_rows) != EXPECTED_PIN_COUNTS["ops"]:
        raise ExtractionError("ops: count mismatch or wrong-source undercount")

    dtypes = _strict_object(doc["dtypes"], {"count", "declarations", "source"}, "dtypes")
    if dtypes["source"] != "tinygrad/dtype.py" or by_category["dtypes"]["paths"] != [dtypes["source"]]:
        raise ExtractionError("dtypes: exact source mismatch")
    dtype_rows = _strict_list(dtypes["declarations"], "dtypes.declarations")
    dtype_names: list[str] = []
    for index, raw in enumerate(dtype_rows):
        row = _strict_object(raw, {"name", "source"}, f"dtypes.declarations[{index}]")
        if not isinstance(row["name"], str) or not row["name"] or row["source"] != dtypes["source"]:
            raise ExtractionError(f"dtypes.declarations[{index}]: invalid declaration")
        dtype_names.append(row["name"])
    if dtype_names != sorted(dtype_names) or len(dtype_names) != len(set(dtype_names)):
        raise ExtractionError("dtypes: declarations are unsorted or duplicated")
    if dtypes["count"] != len(dtype_rows) or len(dtype_rows) != EXPECTED_PIN_COUNTS["dtypes"]:
        raise ExtractionError("dtypes: count mismatch")

    backends = _strict_object(doc["backends"], {"count", "declarations"}, "backends")
    backend_rows = _strict_list(backends["declarations"], "backends.declarations")
    backend_names: list[str] = []
    backend_sources: list[str] = []
    for index, raw in enumerate(backend_rows):
        row = _strict_object(raw, {"name", "source"}, f"backends.declarations[{index}]")
        source = row["source"]
        expected_name = (
            source[len("tinygrad/runtime/ops_"):-len(".py")]
            if isinstance(source, str) and _is_backend_source(source)
            else None
        )
        if row["name"] != expected_name:
            raise ExtractionError(f"backends.declarations[{index}]: name/source mismatch")
        backend_names.append(row["name"])
        backend_sources.append(source)
    if (
        backend_names != sorted(backend_names)
        or len(backend_names) != len(set(backend_names))
        or backend_sources != by_category["backends"]["paths"]
        or backends["count"] != len(backend_rows)
        or len(backend_rows) != EXPECTED_PIN_COUNTS["backends"]
    ):
        raise ExtractionError("backends: derived inventory mismatch")

    tests = _strict_object(doc["tests"], {"count", "groups", "policy_id", "sources"}, "tests")
    test_sources = _strict_list(tests["sources"], "tests.sources")
    if (
        tests["policy_id"] != API_SURFACE_TEST_POLICY_ID
        or tests["groups"] != sorted(TEST_GROUPS)
        or test_sources != by_category["api_surface_tests"]["paths"]
        or tests["count"] != len(test_sources)
        or len(test_sources) != EXPECTED_PIN_COUNTS["api_surface_tests"]
    ):
        raise ExtractionError("tests: api-surface subset policy mismatch")

    limits_raw = _strict_list(doc["limits"], "limits")
    limits: list[dict] = []
    for index, raw in enumerate(limits_raw):
        row = _strict_object(raw, {"id", "statement"}, f"limits[{index}]")
        if row["id"] not in LIMIT_STATEMENTS:
            raise ExtractionError(f"limits[{index}]: unknown limit {row['id']!r}")
        if row["statement"] != LIMIT_STATEMENTS[row["id"]]:
            raise ExtractionError(f"limits[{index}]: statement mismatch")
        limits.append(row)
    limit_ids = [row["id"] for row in limits]
    if limit_ids != sorted(LIMIT_STATEMENTS) or len(limit_ids) != len(set(limit_ids)):
        raise ExtractionError("limits: exact mandatory set/order mismatch")

    if not _valid_lower_hex(doc["closure_sha256"], 64):
        raise ExtractionError("source closure: malformed closure_sha256")
    body = {key: value for key, value in doc.items() if key != "closure_sha256"}
    if doc["closure_sha256"] != digest_value(body):
        raise ExtractionError("source closure: closure_sha256 mismatch")
    return doc


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ExtractionError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def parse_source_closure_bytes(
    raw: bytes, *, authenticate_extractor_sources: bool = False
) -> dict:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ExtractionError(f"source closure JSON is not UTF-8: {exc}") from exc
    try:
        document = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ExtractionError(f"non-finite JSON number is forbidden: {value}")
            ),
        )
    except ExtractionError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ExtractionError(f"invalid source closure JSON: {exc}") from exc
    validate_source_closure_document(
        document, authenticate_extractor_sources=authenticate_extractor_sources
    )
    if raw != canonical_bytes(document):
        raise ExtractionError(
            "source closure JSON is not canonical compact sorted-key UTF-8 with one final newline"
        )
    return document


def check_source_closure_against_git(
    output: Path = SOURCE_CLOSURE_OUTPUT,
    checkout: Path = upstream_target.DEFAULT_ORACLE,
) -> dict:
    regenerated = build_source_closure(checkout)
    regenerated_bytes = canonical_bytes(regenerated)
    try:
        current = output.read_bytes()
    except FileNotFoundError as exc:
        raise ExtractionError(f"source closure fixture is missing: {output}") from exc
    # Authenticity boundary: byte comparison against a fresh object-database
    # extraction precedes any generated-Lean projection check.
    if current != regenerated_bytes:
        raise ExtractionError(
            f"source closure fixture differs from foreign Git re-extraction: {output}"
        )
    return parse_source_closure_bytes(current, authenticate_extractor_sources=True)


def source_closure_main(args: argparse.Namespace) -> int:
    try:
        if args.check:
            document = check_source_closure_against_git(args.output, args.checkout)
            print(
                f"source closure up to date: {args.output} "
                f"({document['closure_sha256']}, {document['repository']['entry_count']} entries)"
            )
            return 0
        document = build_source_closure(args.checkout)
        rendered = canonical_bytes(document)
        if args.print_only:
            sys.stdout.buffer.write(rendered)
            return 0
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(rendered)
        print(
            f"wrote {args.output}: closure {document['closure_sha256']}, "
            f"{document['repository']['entry_count']} entries, "
            f"{EXPECTED_PIN_COUNTS['upstream_tests']} upstream tests, "
            f"{EXPECTED_PIN_COUNTS['api_surface_tests']} api-surface tests"
        )
        return 0
    except ExtractionError as exc:
        print(f"extract_upstream source-closure: FAILED — {exc}", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", default=DEFAULT_REF)
    parser.add_argument("--print", dest="print_only", action="store_true")
    parser.add_argument(
        "--source-closure", action="store_true", help="extract source-closure-v1 from local Git objects"
    )
    parser.add_argument("--checkout", type=Path, default=upstream_target.DEFAULT_ORACLE)
    parser.add_argument("--output", type=Path, default=SOURCE_CLOSURE_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.source_closure:
        return source_closure_main(args)
    if args.check or args.checkout != upstream_target.DEFAULT_ORACLE or args.output != SOURCE_CLOSURE_OUTPUT:
        parser.error("--check/--checkout/--output require --source-closure")
    return legacy_main(args)


if __name__ == "__main__":
    sys.exit(main())
