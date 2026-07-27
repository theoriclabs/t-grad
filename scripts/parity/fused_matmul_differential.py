#!/usr/bin/env python3
"""Matmul via the general graph, differentiated against the specialised one.

`a @ b` is `reduce add (mul (expand a) (expand b))` contracted over the
last axis. This builds exactly that --- reshape, expand, permute, mul,
sum, and nothing matmul-shaped --- and checks two things:

  1. it agrees with numpy, so the general path is correct on its own;
  2. it agrees BIT-FOR-BIT with `a @ b`, so the specialised WMMA route
     is an optimisation of this expression rather than a different
     computation that happens to be called matmul.

(2) is the load-bearing one. It is what makes deleting or bypassing the
specialised route a decision about speed instead of a gamble about
answers.

Note the lowering is fused: the `M*N*K` product is never materialised
(it would be 2 GB at 1024**3). See `fusedReduceKernelDecl`.

The numpy check is scale-relative, and the tail of this file falsifies
that metric before trusting it --- an all-green comparison whose
comparator cannot go red is not evidence.

Usage:  DYLD_LIBRARY_PATH=.lake/build/lib .venv/bin/python \
          scripts/parity/fused_matmul_differential.py
"""
import sys, ctypes, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "python"))
import tgrad
from tgrad import Tensor, _lib, _BINOP_MUL, _BINOP_ADD, _dtype_of_handle, _DTYPE_BYTES

def fused_matmul(a, b):
    """a @ b via the GENERAL graph: reduce add (mul (expand a) (expand b)).

    No matmul-shaped call anywhere: only reshape/expand/permute/mul/sum."""
    M, K = a.shape; K2, N = b.shape; assert K == K2
    a3 = a.reshape(M, 1, K).expand(M, N, K)
    b3 = b.reshape(1, K, N).permute(0, 2, 1).expand(M, N, K)
    prod = _lib.tgrad_tensor_binop(_BINOP_MUL, a3._handle, b3._handle)
    assert prod, "binop mul returned 0"
    graph = _lib.tgrad_tensor_reduce(_BINOP_ADD, prod, 2)
    assert graph, "reduce add returned 0"
    out = _lib.tgrad_realize(graph)
    assert out, "realize returned 0"
    buf = _lib.tgrad_tensor_raw_buffer(out)
    dims = [_lib.tgrad_tensor_shape_dim(out, i) for i in range(3)]
    dt = _dtype_of_handle(out)
    return Tensor._from_buffer(
        buf, dims[0]*dims[1]*_DTYPE_BYTES[dt], (dims[0], dims[1]),
        dt, handle=out, owns_buf=True, base=None), dims

fails = 0
for (M, K, N) in [(4,4,4), (8,4,6), (16,16,16), (3,5,7), (64,64,64), (2,32,9)]:
    rng = np.random.default_rng(M*1000+K*10+N)
    an = (rng.standard_normal((M,K)).astype(np.float32)*0.5)
    bn = (rng.standard_normal((K,N)).astype(np.float32)*0.5)
    a = Tensor.from_numpy(an.astype(np.float32), dtype="bf16")
    b = Tensor.from_numpy(bn.astype(np.float32), dtype="bf16")
    try:
        got, dims = fused_matmul(a, b)
        g = got.numpy()
    except Exception as e:
        print(f"  {M}x{K}x{N}  FUSED RAISED {type(e).__name__}: {e}"); fails += 1; continue
    # oracle: numpy, in bf16-rounded inputs
    def bf16(x):
        u = x.astype(np.float32).view(np.uint32)
        return ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).view(np.float32)
    want = bf16(an) @ bf16(bn)
    # Scale-relative, not element-relative: the output is stored as bf16
    # (8 mantissa bits), so an entry where the dot product nearly cancels
    # has enormous *relative* error while being a perfectly good answer.
    # Comparing against the magnitude of the result matrix is the metric
    # that actually distinguishes a wrong kernel from bf16 rounding.
    scale = float(np.abs(want).max()) + 1e-6
    err = float(np.max(np.abs(g.astype(np.float32) - want)) / scale)
    # and the SPECIALISED path, same inputs
    spec = (a @ b).numpy()
    agree = np.array_equal(g, spec)
    ok = err < 0.02 and agree
    print(f"  {M}x{K}x{N:<9} dims={dims} scale_rel_err={err:.5f} "
          f"fused==specialised: {agree}  {'ok' if ok else 'FAIL'}")
    if not ok: fails += 1

# Falsify the comparator. Both checks above are equalities, and an
# equality that has never been shown capable of failing is decoration.
# Perturb one output element by one part in fifty of the matrix scale
# and require BOTH arms to notice.
rng = np.random.default_rng(0)
_w = rng.standard_normal((8, 8)).astype(np.float32)
_scale = float(np.abs(_w).max())
_bad = _w.copy(); _bad[3, 5] += 0.05 * _scale
if float(np.max(np.abs(_bad - _w)) / (_scale + 1e-6)) < 0.02:
    print("  FALSIFICATION FAILED: numpy metric ignores a 5% perturbation")
    fails += 1
if np.array_equal(_bad, _w):
    print("  FALSIFICATION FAILED: bit-equality ignores a perturbation")
    fails += 1
print("  falsification: metric and bit-equality both reject a perturbed result")

print("FAILS:", fails)
sys.exit(1 if fails else 0)
