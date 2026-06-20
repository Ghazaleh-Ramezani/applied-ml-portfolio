# Kronecker-Factorized Linear Layers — a from-scratch study

A small, self-contained implementation I wrote to understand how Kronecker
structure reduces the parameter count and inference cost of a linear layer, and
what it costs in expressivity. **This is a learning project**, not a research
contribution: the goal was to build the mechanism from first principles in plain
NumPy, measure the tradeoffs myself, and be able to explain every line.

## The idea

A dense linear layer applies `y = W x` with `W` of shape `(M, N)` — that is
`M·N` parameters and `O(M·N)` work per forward pass. If `W` is structured as a
Kronecker product `W = A ⊗ B` with `A: (m1,n1)`, `B: (m2,n2)`,
`M = m1·m2`, `N = n1·n2`, then:

- parameters drop from `M·N` to `m1·n1 + m2·n2`, and
- the forward pass never forms `W`, using the **vec trick**:
  `(A ⊗ B) x = vec(A · X · Bᵀ)` where `X = reshape(x, (n1, n2))`.

One term is very restrictive, so the layer supports a **sum of `R` terms**
(Kronecker rank `R`): `W ≈ Σ A_i ⊗ B_i`. Larger `R` buys back expressivity at
the cost of parameters — that tradeoff is the whole point of the study.

To approximate an existing dense matrix, the code uses the
**Van Loan–Pitsianis** construction: rearrange `W` into a matrix whose SVD
yields the best rank-`R` Kronecker approximation.

## What I measured

Verified correctness (the vec trick matches the explicit Kronecker product, and
a genuinely rank-2 matrix reconstructs to machine precision, rel. error
`≈ 3e-16`), then measured parameters, speed, and approximation quality.

**Parameters and forward-pass speed (rank-1, NumPy, CPU):**

| layer | dense params | kron params | param ratio | dense fwd | kron fwd | speedup |
|---|---|---|---|---|---|---|
| 1024×1024 | 1,048,576 | 2,048 | 512× | 312 µs | 7.5 µs | 41× |
| 4096×4096 | 16,777,216 | 8,192 | 2048× | 5491 µs | 28 µs | 195× |

The speedup grows with size because dense cost scales as `O(M·N)` while the
Kronecker forward scales far more slowly — the asymptotic gap is the real story.

**Approximation quality vs rank (1024×1024 random matrix):**

| rank R | kron params | compression | rel. Frobenius error |
|---|---|---|---|
| 1 | 2,048 | 512× | 0.998 |
| 4 | 8,192 | 128× | 0.992 |
| 16 | 32,768 | 32× | 0.970 |
| 64 | 131,072 | 8× | 0.889 |

**Honest caveat (and the most interesting result):** a *random* matrix is a
worst case — it has no Kronecker structure, so even at rank 64 the error is
still high. This is the right lesson to take away: Kronecker factorization only
pays off when the underlying operator actually has this structure, which trained
weight matrices often do far more than random ones do. Quantifying that on real
trained weights is the natural next experiment.

## Files

- `kron_linear.py` — the layer: `kron_matvec` (vec-trick forward),
  `KroneckerLinear` (rank-`R`), and `nearest_kron_product` (Van Loan–Pitsianis).
- `benchmark.py` — correctness checks, the speed/param table, the rank sweep.

## Run it

```bash
pip install numpy
python benchmark.py
```


