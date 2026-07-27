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
SOURCE_TREE: Path | None = None

TEST_GROUPS = ("null", "unit", "backend")


class ExtractionError(RuntimeError):
    pass


def gh(path: str) -> object:
    if SOURCE_TREE is not None:
        return local_api(path)
    p = subprocess.run(
        ["gh", "api", f"repos/{REPO}/{path}"], capture_output=True, text=True
    )
    if p.returncode != 0:
        raise ExtractionError(f"gh api {path} failed: {p.stderr.strip()[:200]}")
    return json.loads(p.stdout)


def local_api(path: str) -> object:
    """The small GitHub-contents subset needed by this extractor, from a checkout."""
    assert SOURCE_TREE is not None
    if path.startswith("commits/"):
        requested = path.removeprefix("commits/")
        process = subprocess.run(
            ["git", "-C", str(SOURCE_TREE), "show", "-s",
             "--format=%H%x00%cI%x00%s", "HEAD"],
            capture_output=True, text=True,
        )
        if process.returncode != 0:
            raise ExtractionError(f"cannot inspect source checkout: {process.stderr.strip()}")
        sha, committed_at, subject = process.stdout.strip().split("\0", 2)
        committed_at = datetime.fromisoformat(committed_at).astimezone(
            timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        if not sha.startswith(requested) and requested != sha:
            raise ExtractionError(
                f"source checkout is {sha}, not requested revision {requested}"
            )
        status = subprocess.run(
            ["git", "-C", str(SOURCE_TREE), "status", "--porcelain"],
            capture_output=True, text=True,
        )
        if status.returncode != 0 or status.stdout:
            raise ExtractionError("source checkout must be a clean immutable tree")
        return {
            "sha": sha,
            "commit": {"committer": {"date": committed_at}, "message": subject},
        }
    if not path.startswith("contents/"):
        raise ExtractionError(f"unsupported local API request: {path}")
    relative, _, requested = path.removeprefix("contents/").partition("?ref=")
    head = subprocess.run(
        ["git", "-C", str(SOURCE_TREE), "rev-parse", "HEAD"],
        capture_output=True, text=True,
    )
    if head.returncode != 0 or not head.stdout.strip().startswith(requested):
        raise ExtractionError(f"local content request does not match checkout: {relative}")
    target = SOURCE_TREE / relative
    if target.is_file():
        return {"content": base64.b64encode(target.read_bytes()).decode("ascii")}
    if target.is_dir():
        return [
            {"name": child.name, "size": child.stat().st_size}
            for child in sorted(target.iterdir())
        ]
    raise ExtractionError(f"missing path in source checkout: {relative}")


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
    global SOURCE_TREE
    if sys.version_info < (3, 11):
        print(
            "extract_upstream: FAILED — Python 3.11+ is required to parse the pinned upstream syntax",
            file=sys.stderr,
        )
        return 1
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=DEFAULT_REF)
    ap.add_argument("--print", dest="print_only", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--source-tree", type=Path)
    args = ap.parse_args()
    if args.print_only and args.check:
        ap.error("--print and --check are mutually exclusive")
    SOURCE_TREE = args.source_tree.resolve() if args.source_tree is not None else None
    if SOURCE_TREE is not None and not SOURCE_TREE.is_dir():
        print(f"extract_upstream: FAILED — no source tree at {SOURCE_TREE}", file=sys.stderr)
        return 1

    try:
        head = gh(f"commits/{args.ref}")
        ref = head["sha"]
        manifest = {
            "schema_version": 1,
            "upstream_repo": REPO,
            "upstream_ref": ref,
            "upstream_committed_at": head["commit"]["committer"]["date"],
            "upstream_subject": head["commit"]["message"].split("\n")[0],
            "extracted_at_utc": datetime.now(timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "extractor": "scripts/parity/extract_upstream.py",
            "extractor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "tensor_api": tensor_api(ref),
            "dtypes": dtype_inventory(ref),
            "ops": ops_inventory(ref),
            "backends": backend_inventory(ref),
            "tests": test_inventory(ref),
            # Exclusions are policy, not extraction accidents.  The ledger is
            # explicit and empty by default so a missing inventory can never be
            # laundered into an implicit exclusion.
            "exclusions": [],
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
    manifest["section_sha256"] = {
        name: hashlib.sha256(
            json.dumps(manifest[name], sort_keys=True).encode()
        ).hexdigest()
        for name in ("tensor_api", "dtypes", "ops", "backends", "tests")
    }
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

    out = OUT_DIR / f"upstream_{ref[:12]}.json"
    if args.check:
        try:
            committed = json.loads(out.read_text())
        except (OSError, json.JSONDecodeError) as error:
            print(f"extract_upstream: STALE — cannot read {out}: {error}", file=sys.stderr)
            return 1
        if (committed.get("upstream_ref") != ref or
                committed.get("content_sha256") != manifest["content_sha256"]):
            print(f"extract_upstream: STALE — {out}", file=sys.stderr)
            return 1
        print(f"\nextract_upstream: OK — {out.relative_to(OUT_DIR.parents[1])}")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {out.relative_to(OUT_DIR.parents[1])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
