"""
Benchmark for the Kronecker-factorized linear layer.

Reports three things:
  1. Correctness  -- the vec-trick matches the explicit Kronecker product, and
                     a true rank-2 matrix is reconstructed exactly.
  2. Params/speed -- dense vs Kronecker for representative layer sizes.
  3. Quality      -- reconstruction error vs Kronecker rank R on a dense matrix.

Run:  python benchmark.py
"""

import time
import numpy as np
from kron_linear import kron_matvec, KroneckerLinear, nearest_kron_product

rng = np.random.default_rng(0)


def timeit(fn, n_iter):
    for _ in range(3):          # warmup
        fn()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    return (time.perf_counter() - t0) / n_iter


# ---------------------------------------------------------------------------
# 1) Correctness
# ---------------------------------------------------------------------------
A = rng.standard_normal((8, 7))
B = rng.standard_normal((6, 5))
x = rng.standard_normal((7 * 5,))
assert np.allclose(kron_matvec(A, B, x), np.kron(A, B) @ x), "vec-trick mismatch"

m1, n1, m2, n2 = 8, 7, 6, 5
A1, B1 = rng.standard_normal((m1, n1)), rng.standard_normal((m2, n2))
A2, B2 = rng.standard_normal((m1, n1)), rng.standard_normal((m2, n2))
W2 = np.kron(A1, B1) + np.kron(A2, B2)
approx2 = nearest_kron_product(W2, m1, n1, m2, n2, R=2)
rel2 = np.linalg.norm(W2 - approx2.to_dense()) / np.linalg.norm(W2)
assert rel2 < 1e-10, f"rank-2 reconstruction not exact: {rel2}"
print("1) correctness")
print(f"   vec-trick == dense Kronecker product: OK")
print(f"   exact rank-2 reconstruction, rel. error = {rel2:.2e}")


# ---------------------------------------------------------------------------
# 2) Parameters and speed (rank-1) for two layer sizes
# ---------------------------------------------------------------------------
print("\n2) parameters and forward-pass speed (rank-1 Kronecker)")
print(f"   {'layer':>12} {'dense params':>14} {'kron params':>12} {'param ratio':>12} "
      f"{'dense fwd':>11} {'kron fwd':>11} {'speedup':>8}")
for f, n_iter in [(32, 300), (64, 80)]:
    M = N = f * f
    W = rng.standard_normal((M, N))
    xv = rng.standard_normal((N,))
    Af = rng.standard_normal((f, f))
    Bf = rng.standard_normal((f, f))
    kl = KroneckerLinear([(Af, Bf)])

    dense_p = M * N
    kron_p = kl.num_params()
    t_dense = timeit(lambda: W @ xv, n_iter)
    t_kron = timeit(lambda: kl.forward(xv), n_iter)
    print(f"   {M:>5}x{N:<6} {dense_p:>14,} {kron_p:>12,} "
          f"{dense_p / kron_p:>11.0f}x {t_dense*1e6:>9.1f}us {t_kron*1e6:>9.1f}us "
          f"{t_dense / t_kron:>7.1f}x")


# ---------------------------------------------------------------------------
# 3) Quality vs rank: how well can a Kronecker sum approximate a dense matrix?
#    (A random matrix is a WORST case -- it has no Kronecker structure. Trained
#     weight matrices typically compress much better than this.)
# ---------------------------------------------------------------------------
print("\n3) reconstruction error vs Kronecker rank (1024x1024 random matrix)")
m1 = n1 = m2 = n2 = 32
M = N = 1024
W = rng.standard_normal((M, N))
dense_p = M * N
print(f"   {'rank R':>7} {'kron params':>12} {'compression':>12} {'rel. Frobenius error':>22}")
for R in [1, 2, 4, 8, 16, 32, 64]:
    approx = nearest_kron_product(W, m1, n1, m2, n2, R=R)
    err = np.linalg.norm(W - approx.to_dense()) / np.linalg.norm(W)
    comp = dense_p / approx.num_params()
    print(f"   {R:>7} {approx.num_params():>12,} {comp:>11.1f}x {err:>21.3f}")
