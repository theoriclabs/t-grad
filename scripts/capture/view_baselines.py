#!/usr/bin/env python3
"""DEV-TIME capture: 16 pinned view-matmul reference shas via numpy
bf16-roundtripped reference (route b of L13.E)."""
from __future__ import annotations
import hashlib, json, sys, zlib
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "fixtures" / "bench" / "view_manifest.json"


def _seed(M, K, N, dist, op):
    return (0xBF16 ^ (M * 73856093) ^ (K * 19349663) ^ (N * 83492791)
            ^ zlib.crc32(dist.encode()) ^ zlib.crc32(op.encode())) & 0xFFFFFFFF


def _to_bf16_f32(arr):
    flat = arr.astype(np.float32).flatten()
    view = flat.view(np.uint32)
    hi = (view >> 16).astype(np.uint16)
    lifted = hi.astype(np.uint32) << 16
    return lifted.view(np.float32).reshape(arr.shape).copy()


def _to_bf16_bytes(arr):
    flat = arr.astype(np.float32).flatten()
    view = flat.view(np.uint32)
    hi = (view >> 16).astype(np.uint16)
    return hi.tobytes()


def _alloc_shapes_for(op, M, K, N):
    if op == "transpose_left":   return (K, M), (K, N)
    if op == "transpose_right":  return (M, K), (N, K)
    if op == "transpose_both":   return (K, M), (N, K)
    if op == "slice_2":          return (2 * M, K), (K, N)
    if op == "reshape_split":    return (2 * M, K // 2), (K, N)
    if op == "expand_right":     return (M, K), (K, 1)
    raise ValueError(op)


def _apply_op_numpy(op, a, b, M, K, N):
    if op == "transpose_left":   return a.T @ b
    if op == "transpose_right":  return a @ b.T
    if op == "transpose_both":   return a.T @ b.T
    if op == "slice_2":          return a[::2, :] @ b
    if op == "reshape_split":    return a.reshape(M, K) @ b
    if op == "expand_right":     return a @ np.broadcast_to(b, (K, N))
    raise ValueError(op)


MANIFEST_SOURCE = [
    ("transpose_left", 64,  64,  64,  "gauss"),
    ("transpose_left", 32,  64,  32,  "gauss"),
    ("transpose_left", 64,  32,  64,  "gauss"),
    ("transpose_left", 128, 64,  64,  "gauss"),
    ("transpose_right", 64,  64,  64, "gauss"),
    ("transpose_right", 32,  32,  64, "gauss"),
    ("transpose_right", 64,  32,  32, "gauss"),
    ("transpose_right", 32,  64,  32, "gauss"),
    ("transpose_both", 64,  64,  64, "gauss"),
    ("transpose_both", 32,  64,  32, "gauss"),
    ("slice_2", 32, 32, 32, "gauss"),
    ("slice_2", 64, 32, 32, "gauss"),
    ("reshape_split", 32, 32, 32, "gauss"),
    ("reshape_split", 64, 64, 32, "gauss"),
    ("expand_right", 32, 32, 32, "gauss"),
    ("expand_right", 64, 32, 64, "gauss"),
]


def main():
    entries = []
    for op, M, K, N, dist in MANIFEST_SOURCE:
        seed = _seed(M, K, N, dist, op)
        rng = np.random.default_rng(seed)
        a_shape, b_shape = _alloc_shapes_for(op, M, K, N)
        a_np = rng.standard_normal(a_shape, dtype=np.float32)
        b_np = rng.standard_normal(b_shape, dtype=np.float32)
        a_bf16 = _to_bf16_f32(a_np)
        b_bf16 = _to_bf16_f32(b_np)
        ref = _apply_op_numpy(op, a_bf16, b_bf16, M, K, N).astype(np.float32)
        ref_bytes = _to_bf16_bytes(ref)
        entries.append({
            "op_chain": op,
            "M": M, "K": K, "N": N,
            "dist": dist,
            "seed": seed,
            "expected_bytes_sha256": hashlib.sha256(ref_bytes).hexdigest(),
        })
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(entries, indent=2) + "\n")
    print(f"wrote {MANIFEST} ({len(entries)} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
