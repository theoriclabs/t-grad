#!/usr/bin/env python3
"""Materialize and verify the pinned tinygrad foreign Git checkout.

Source-closure verification needs more than a directory at the right-looking
revision.  The checkout must have the normalized foreign origin, an exact
detached HEAD and tree, no tracked or untracked dirt, and every blob required
by the pinned tree available locally.  Object completeness is checked with
lazy fetching disabled.

Usage:
    python3 scripts/parity/ensure_oracle.py
    python3 scripts/parity/ensure_oracle.py --verify-only
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.parity import upstream_target


@dataclass(frozen=True)
class OracleExpectation:
    repository: str
    revision: str
    tree: str
    object_format: str
    clone_url: str


EXPECTED = OracleExpectation(
    repository=upstream_target.REPOSITORY,
    revision=upstream_target.REVISION,
    tree=upstream_target.TREE,
    object_format=upstream_target.OBJECT_FORMAT,
    clone_url=upstream_target.UPSTREAM_URL,
)
DEFAULT_ORACLE = upstream_target.DEFAULT_ORACLE

# Backwards-compatible names used by existing callers.
PINNED_REF = upstream_target.REVISION
UPSTREAM_URL = upstream_target.UPSTREAM_URL


def git(
    repo: Path | None,
    *args: str,
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    base = ["git"] if repo is None else ["git", "-C", str(repo)]
    # The caller's Git plumbing must not redirect the repository, object
    # database, config, namespace, index, or replacement-ref base.  Keep the
    # ordinary process environment, discard every inherited Git override, and
    # then add only explicit non-fetch/non-replacement policy.
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env["GIT_NO_LAZY_FETCH"] = "1"
    env["GIT_NO_REPLACE_OBJECTS"] = "1"
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_SYSTEM"] = os.devnull
    env["GIT_ATTR_NOSYSTEM"] = "1"
    env["LC_ALL"] = "C"
    env["LANG"] = "C"
    return subprocess.run(
        [*base, *args],
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=env,
    )


def git_blob_oid(object_format: str, payload: bytes) -> str:
    framed = f"blob {len(payload)}\0".encode("ascii") + payload
    if object_format == "sha1":
        return hashlib.sha1(framed).hexdigest()
    if object_format == "sha256":
        return hashlib.sha256(framed).hexdigest()
    raise ValueError(f"unsupported Git object format: {object_format}")


def _text(result: subprocess.CompletedProcess[bytes]) -> str:
    return result.stdout.decode("utf-8", "replace").strip()


def normalize_origin(value: str) -> str:
    """Normalize common GitHub HTTPS/SSH spellings to ``host/owner/repo``."""
    raw = value.strip()
    if re.match(r"^[^/@:]+@[^:]+:.+", raw):
        user_host, path = raw.split(":", 1)
        host = user_host.rsplit("@", 1)[-1]
    else:
        parsed = urlparse(raw if "://" in raw else f"ssh://{raw}")
        host = parsed.hostname or ""
        path = parsed.path
    path = path.strip("/")
    if path.endswith(".git"):
        path = path[:-4]
    normalized_path = str(PurePosixPath(path)) if path else ""
    return f"{host.lower()}/{normalized_path}".rstrip("/")


def _tree_entries(path: Path, revision: str) -> tuple[list[tuple[str, str, str, str]] | None, str]:
    result = git(path, "ls-tree", "-r", "-z", revision)
    if result.returncode != 0:
        return None, f"cannot enumerate pinned tree: {result.stderr.decode('utf-8', 'replace').strip()[:300]}"
    entries: list[tuple[str, str, str, str]] = []
    try:
        for raw in result.stdout.split(b"\0"):
            if not raw:
                continue
            meta, path_bytes = raw.split(b"\t", 1)
            mode, object_type, oid = meta.decode("ascii").split(" ")
            repo_path = path_bytes.decode("utf-8")
            entries.append((mode, object_type, oid, repo_path))
    except (UnicodeDecodeError, ValueError) as exc:
        return None, f"malformed or non-UTF-8 ls-tree entry: {exc}"
    return entries, ""


def _verify_required_objects(
    path: Path, entries: list[tuple[str, str, str, str]], expectation: OracleExpectation
) -> tuple[bool, str]:
    expected_oid_len = 40 if expectation.object_format == "sha1" else 64
    for mode, object_type, oid, repo_path in entries:
        if object_type != "blob":
            return False, f"required entry is not a blob: {repo_path} ({mode} {object_type})"
        if len(oid) != expected_oid_len:
            return False, f"malformed object id for {repo_path}: {oid}"

    # --batch reads every payload, so a missing or corrupt blob cannot pass a
    # metadata-only batch-check.  GIT_NO_LAZY_FETCH prevents a partial clone
    # from consulting the network during verification.
    payload = b"".join(oid.encode("ascii") + b"\n" for _, _, oid, _ in entries)
    result = git(path, "cat-file", "--batch", input_bytes=payload)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()[:300]
        return False, f"cannot read complete required object set: {detail}"
    offset = 0
    try:
        for (_, _, expected_oid, repo_path) in entries:
            newline = result.stdout.index(b"\n", offset)
            header = result.stdout[offset:newline].decode("ascii")
            offset = newline + 1
            parts = header.split(" ")
            if len(parts) != 3 or parts[1] != "blob":
                return False, f"missing required object for {repo_path}: {header}"
            actual_oid, _, size_text = parts
            size = int(size_text)
            if actual_oid != expected_oid:
                return False, f"object mismatch for {repo_path}: {actual_oid} != {expected_oid}"
            blob = result.stdout[offset:offset + size]
            if len(blob) != size:
                return False, f"truncated required object for {repo_path}"
            recomputed_oid = git_blob_oid(expectation.object_format, blob)
            if recomputed_oid != expected_oid:
                return False, (
                    f"payload object id mismatch for {repo_path}: "
                    f"{recomputed_oid} != {expected_oid}"
                )
            offset += size
            if result.stdout[offset:offset + 1] != b"\n":
                return False, f"truncated required object for {repo_path}"
            offset += 1
    except (UnicodeDecodeError, ValueError) as exc:
        return False, f"malformed required object stream: {exc}"
    if offset != len(result.stdout):
        return False, "unexpected trailing bytes in required object stream"
    return True, f"{len(entries)} required blobs complete"


def verify(
    path: Path, expectation: OracleExpectation = EXPECTED
) -> tuple[bool, str]:
    git_dir = git(path, "rev-parse", "--git-dir") if path.exists() else None
    if git_dir is None or git_dir.returncode != 0:
        return False, f"no checkout at {path}"

    origin = git(path, "remote", "get-url", "origin")
    if origin.returncode != 0:
        return False, "checkout has no origin remote"
    actual_origin = normalize_origin(_text(origin))
    if actual_origin != expectation.repository:
        return False, f"origin is {actual_origin}, expected {expectation.repository}"

    object_format = git(path, "rev-parse", "--show-object-format")
    if object_format.returncode != 0 or _text(object_format) != expectation.object_format:
        return False, (
            f"object format is {_text(object_format) or '<unknown>'}, "
            f"expected {expectation.object_format}"
        )

    replacements = git(path, "for-each-ref", "--format=%(refname)", "refs/replace")
    if replacements.returncode != 0:
        return False, "cannot inspect replacement refs"
    if replacements.stdout.strip():
        first = replacements.stdout.decode("utf-8", "replace").splitlines()[0]
        return False, f"replacement refs are forbidden: {first}"

    for alternate_name in ("objects/info/alternates", "objects/info/http-alternates"):
        alternate_result = git(path, "rev-parse", "--git-path", alternate_name)
        if alternate_result.returncode != 0:
            return False, f"cannot resolve Git object alternate path: {alternate_name}"
        alternate_path = Path(_text(alternate_result))
        if not alternate_path.is_absolute():
            alternate_path = path / alternate_path
        try:
            has_alternates = alternate_path.is_file() and bool(alternate_path.read_bytes().strip())
        except OSError as exc:
            return False, f"cannot inspect Git object alternates: {exc}"
        if has_alternates:
            return False, f"Git object alternates are forbidden: {alternate_path}"

    # Authenticate commit/tree structure before using rev-parse or ls-tree as
    # an identity oracle.  This is distinct from the later per-blob payload
    # re-hash: fsck validates the reachable object graph and object encodings;
    # the payload loop validates exactly the bytes returned to the extractor.
    fsck = git(
        path,
        "fsck",
        "--strict",
        "--full",
        "--no-dangling",
        expectation.revision,
    )
    if fsck.returncode != 0:
        detail = (fsck.stderr or fsck.stdout).decode("utf-8", "replace").strip()[:300]
        return False, f"strict full fsck failed for pinned revision: {detail}"

    head = git(path, "rev-parse", "--verify", "HEAD^{commit}")
    if head.returncode != 0:
        return False, "checkout has no readable HEAD commit"
    actual_head = _text(head)
    if actual_head != expectation.revision:
        return False, f"HEAD is {actual_head or '<missing>'}, expected {expectation.revision}"

    symbolic = git(path, "symbolic-ref", "-q", "HEAD")
    if symbolic.returncode == 0:
        return False, f"HEAD is attached to {_text(symbolic)}; expected detached HEAD"
    if symbolic.returncode not in (1,):
        return False, "cannot determine whether HEAD is detached"

    tree = git(path, "rev-parse", "HEAD^{tree}")
    actual_tree = _text(tree) if tree.returncode == 0 else ""
    if actual_tree != expectation.tree:
        return False, f"HEAD tree is {actual_tree or '<missing>'}, expected {expectation.tree}"

    entries, error = _tree_entries(path, expectation.revision)
    if entries is None:
        return False, error
    objects_ok, object_detail = _verify_required_objects(path, entries, expectation)
    if not objects_ok:
        return False, object_detail

    status = git(path, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    if status.returncode != 0:
        return False, f"cannot inspect checkout cleanliness: {status.stderr.decode('utf-8', 'replace').strip()[:300]}"
    if status.stdout:
        first = status.stdout.split(b"\0", 1)[0].decode("utf-8", "replace")
        return False, f"checkout is dirty (tracked or untracked): {first}"

    return True, (
        f"origin {expectation.repository}, detached {expectation.revision}, "
        f"tree {expectation.tree}, {object_detail}, clean"
    )


def _fetch_pin(path: Path, expectation: OracleExpectation) -> tuple[bool, str]:
    result = git(path, "fetch", "--quiet", "--no-tags", "origin", expectation.revision)
    if result.returncode != 0:
        return False, result.stderr.decode("utf-8", "replace").strip()[:300]
    return True, ""


def ensure(path: Path, expectation: OracleExpectation = EXPECTED) -> int:
    ok, detail = verify(path, expectation)
    if ok:
        print(f"oracle ok: {path} ({detail})")
        return 0

    path.parent.mkdir(parents=True, exist_ok=True)
    if git(path, "rev-parse", "--git-dir").returncode != 0:
        if path.exists() and any(path.iterdir()):
            print(f"refusing to replace non-checkout path: {path}", file=sys.stderr)
            return 1
        print(f"cloning upstream oracle into {path} ...")
        result = git(
            None, "clone", "--quiet", "--no-checkout", expectation.clone_url, str(path)
        )
        if result.returncode != 0:
            print(
                f"clone failed: {result.stderr.decode('utf-8', 'replace').strip()[:300]}",
                file=sys.stderr,
            )
            return 1
    else:
        origin = git(path, "remote", "get-url", "origin")
        actual = normalize_origin(_text(origin)) if origin.returncode == 0 else "<missing>"
        if actual != expectation.repository:
            print(
                f"refusing checkout with origin {actual}; expected {expectation.repository}",
                file=sys.stderr,
            )
            return 1
        status = git(path, "status", "--porcelain=v1", "-z", "--untracked-files=all")
        if status.returncode != 0 or status.stdout:
            print("refusing to overwrite dirty oracle checkout", file=sys.stderr)
            return 1

    present = git(path, "cat-file", "-e", f"{expectation.revision}^{{commit}}")
    if present.returncode != 0:
        print("pinned revision absent; fetching exact revision ...")
        fetched, fetch_detail = _fetch_pin(path, expectation)
        if not fetched:
            print(f"fetch failed: {fetch_detail}", file=sys.stderr)
            return 1

    result = git(path, "checkout", "--quiet", "--detach", expectation.revision)
    if result.returncode != 0:
        print(
            f"checkout of pin failed: {result.stderr.decode('utf-8', 'replace').strip()[:300]}",
            file=sys.stderr,
        )
        return 1

    ok, detail = verify(path, expectation)
    if not ok:
        print(f"oracle verification FAILED after provisioning: {detail}", file=sys.stderr)
        return 1
    print(f"oracle ready: {path} ({detail})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    path = args.path.resolve()

    if args.verify_only:
        ok, detail = verify(path)
        print(("oracle ok: " if ok else "oracle NOT usable: ") + detail)
        return 0 if ok else 1
    return ensure(path)


if __name__ == "__main__":
    raise SystemExit(main())
