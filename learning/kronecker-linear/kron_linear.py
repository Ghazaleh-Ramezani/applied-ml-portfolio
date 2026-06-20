"""
Kronecker-factorized linear layer — plain NumPy, from scratch.

Three public names (imported by benchmark.py):
  kron_matvec(A, B, x)          -- single-term vec-trick forward pass
  KroneckerLinear(terms)        -- rank-R layer (list of (A, B) pairs)
  nearest_kron_product(W, ...)  -- Van Loan–Pitsianis approximation

Math recap
----------
For W = A ⊗ B with A:(m1,n1), B:(m2,n2), M=m1*m2, N=n1*n2:
  (A ⊗ B) x  =  vec( B · reshape(x, n1, n2).T · A.T )
             =  vec( A · reshape(x, n2, n1).T · B.T ).T   [equiv form]

The reshape trick avoids forming the M×N Kronecker product explicitly.

For rank-R approximation:  W ≈ Σ_{i=1}^{R}  A_i ⊗ B_i

Van Loan–Pitsianis construction
--------------------------------
Rearrange W into W_hat of shape (m1*m2, n1*n2) → this equals the sum of
Kronecker products in "rearranged" form.  The SVD of W_hat gives the best
rank-R approximation: sigma_i, u_i, v_i → A_i = sqrt(sigma_i)*reshape(u_i),
B_i = sqrt(sigma_i)*reshape(v_i).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Vec-trick forward pass for a single Kronecker term
# ---------------------------------------------------------------------------

def kron_matvec(A: np.ndarray, B: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Compute (A ⊗ B) @ x without forming the Kronecker product.

    Parameters
    ----------
    A : (m1, n1)
    B : (m2, n2)
    x : (n1*n2,)

    Returns
    -------
    y : (m1*m2,)
    """
    m1, n1 = A.shape
    m2, n2 = B.shape
    assert x.shape == (n1 * n2,), f"x must have shape ({n1*n2},), got {x.shape}"

    # reshape x → (n2, n1), apply B and A, reshape back
    # (A ⊗ B) x = vec(B · X · Aᵀ)  where X = reshape(x, n2, n1)
    # np.kron(A,B) lays out blocks as: kron[i*m2+k, j*n2+l] = A[i,j]*B[k,l]
    # So (kron(A,B) @ x)[i*m2+k] = Σ_{j,l} A[i,j]*B[k,l]*x[j*n2+l]
    #                             = Σ_j A[i,j] * (B @ X[j,:])  where X=x.reshape(n1,n2)
    # = (A @ X @ B.T).ravel()
    X = x.reshape(n1, n2)          # (n1, n2)
    return (A @ X @ B.T).ravel()   # (m1, m2) → (m1*m2,)


# ---------------------------------------------------------------------------
# Rank-R Kronecker linear layer
# ---------------------------------------------------------------------------

class KroneckerLinear:
    """
    A linear map  y = (Σ_i  A_i ⊗ B_i) x,  stored as a list of (A_i, B_i).

    Parameters
    ----------
    terms : list of (A_i, B_i) pairs, each A_i:(m1,n1), B_i:(m2,n2)
    """

    def __init__(self, terms: list):
        assert len(terms) >= 1
        self.terms = terms
        m1, n1 = terms[0][0].shape
        m2, n2 = terms[0][1].shape
        self.M = m1 * m2
        self.N = n1 * n2

    def forward(self, x: np.ndarray) -> np.ndarray:
        """y = (Σ A_i ⊗ B_i) x   using the vec trick for each term."""
        assert x.shape == (self.N,)
        y = np.zeros(self.M)
        for A, B in self.terms:
            y += kron_matvec(A, B, x)
        return y

    def to_dense(self) -> np.ndarray:
        """Materialize the full M×N weight matrix (for testing / error checks)."""
        W = np.zeros((self.M, self.N))
        for A, B in self.terms:
            W += np.kron(A, B)
        return W

    def num_params(self) -> int:
        """Total number of parameters across all Kronecker factors."""
        return sum(A.size + B.size for A, B in self.terms)


# ---------------------------------------------------------------------------
# Van Loan–Pitsianis: best rank-R Kronecker approximation
# ---------------------------------------------------------------------------

def _rearrange(W: np.ndarray, m1: int, n1: int, m2: int, n2: int) -> np.ndarray:
    """
    Rearrange W (m1*m2, n1*n2) into W_hat (m1*n1, m2*n2) such that
    the SVD of W_hat gives the Kronecker factors via Van Loan–Pitsianis.

    W_hat[i*n1+j, k*n2+l] = W[i*m2+k, j*n2+l]
    Equivalently: reshape W into (m1,m2,n1,n2), then transpose to (m1,n1,m2,n2),
    then reshape to (m1*n1, m2*n2).
    """
    return W.reshape(m1, m2, n1, n2).transpose(0, 2, 1, 3).reshape(m1 * n1, m2 * n2)


def nearest_kron_product(
    W: np.ndarray,
    m1: int, n1: int,
    m2: int, n2: int,
    R: int = 1,
) -> KroneckerLinear:
    """
    Best rank-R Kronecker approximation of W using Van Loan–Pitsianis / SVD.

    Parameters
    ----------
    W   : (m1*m2, n1*n2) dense weight matrix to approximate
    m1,n1 : shape of factor A
    m2,n2 : shape of factor B
    R   : Kronecker rank (number of terms in the sum)

    Returns
    -------
    KroneckerLinear with R terms, minimizing ||W - Σ A_i ⊗ B_i||_F
    """
    assert W.shape == (m1 * m2, n1 * n2), (
        f"W must have shape ({m1*m2}, {n1*n2}), got {W.shape}"
    )

    W_hat = _rearrange(W, m1, n1, m2, n2)          # (m1*n1, m2*n2)
    U, s, Vt = np.linalg.svd(W_hat, full_matrices=False)

    terms = []
    for i in range(min(R, len(s))):
        scale = np.sqrt(s[i])
        A_i = (scale * U[:, i]).reshape(m1, n1)
        B_i = (scale * Vt[i, :]).reshape(m2, n2)
        terms.append((A_i, B_i))

    return KroneckerLinear(terms)
