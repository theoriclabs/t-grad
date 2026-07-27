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
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# The research candidate named in PARITY.md. Pinning it here is what
# makes the manifest reproducible; bump it deliberately, never silently.
DEFAULT_REF = "19c4d736f2bc8e26d21f08b28ffd6298408da00f"
REPO = "tinygrad/tinygrad"
OUT_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "parity"

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=DEFAULT_REF)
    ap.add_argument("--print", dest="print_only", action="store_true")
    args = ap.parse_args()

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


if __name__ == "__main__":
    sys.exit(main())
