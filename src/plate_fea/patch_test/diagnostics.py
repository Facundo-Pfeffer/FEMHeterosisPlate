"""Stability and consistency diagnostics for constrained stiffness K."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import eigsh, LinearOperator


@dataclass(frozen=True)
class ConstrainedSystemDiagnostics:
    n_free: int
    rank_dense: int | None
    min_eigenvalue: float | None
    max_eigenvalue: float | None
    condition_estimate: float | None
    smallest_singular_value_dense: float | None
    note: str


def _to_dense(K_ff: csr_matrix | np.ndarray) -> np.ndarray:
    if issparse(K_ff):
        return K_ff.toarray()
    return np.asarray(K_ff, dtype=float)


def diagnose_constrained_stiffness(
    K_ff: csr_matrix | np.ndarray,
    *,
    dense_threshold: int = 400,
    eigsh_nev: int = 8,
) -> ConstrainedSystemDiagnostics:
    """
    Rank / eigenvalue / condition information for the free-free partition ``K_ff``.

    For ``n_free`` larger than ``dense_threshold``, uses sparse ``eigsh`` on a few modes only
    (smallest-magnitude eigenvalues); full rank is not computed in that regime.
    """
    if issparse(K_ff):
        n = int(K_ff.shape[0])
    else:
        n = int(np.asarray(K_ff).shape[0])
    if n == 0:
        return ConstrainedSystemDiagnostics(
            n_free=0,
            rank_dense=None,
            min_eigenvalue=None,
            max_eigenvalue=None,
            condition_estimate=None,
            smallest_singular_value_dense=None,
            note="empty K_ff",
        )

    if n <= dense_threshold:
        Kd = _to_dense(K_ff)
        rank = int(np.linalg.matrix_rank(Kd, tol=max(np.linalg.norm(Kd, ord=2), 1.0) * 1.0e-10))
        s = np.linalg.svd(Kd, compute_uv=False)
        smin = float(s[-1]) if s.size else float("nan")
        sym = np.max(np.abs(Kd - Kd.T))
        if sym < 1e-10 * (1.0 + np.linalg.norm(Kd, ord="fro")):
            w = np.linalg.eigvalsh(0.5 * (Kd + Kd.T))
            lam_min = float(np.min(w))
            lam_max = float(np.max(w))
            cond_est = float(lam_max / max(abs(lam_min), 1e-30))
            return ConstrainedSystemDiagnostics(
                n_free=n,
                rank_dense=rank,
                min_eigenvalue=lam_min,
                max_eigenvalue=lam_max,
                condition_estimate=cond_est,
                smallest_singular_value_dense=smin,
                note="dense symmetric eigenspectrum",
            )
        return ConstrainedSystemDiagnostics(
            n_free=n,
            rank_dense=rank,
            min_eigenvalue=None,
            max_eigenvalue=None,
            condition_estimate=None,
            smallest_singular_value_dense=smin,
            note="dense SVD (non-symmetric)",
        )

    Ksp = K_ff.tocsr() if not issparse(K_ff) else K_ff

    def matvec(v: np.ndarray) -> np.ndarray:
        return Ksp @ v

    A = LinearOperator(Ksp.shape, matvec=matvec, dtype=float)
    nev = min(eigsh_nev, n - 1) if n > 1 else 1
    try:
        vals = eigsh(A, k=nev, which="SM", return_eigenvectors=False, tol=1e-8)
        lam_smallest = float(np.min(np.abs(vals)))
        note = f"sparse eigsh (k={nev}, smallest-magnitude eigenvalues)"
    except Exception as exc:  # pragma: no cover - ARPACK failures
        lam_smallest = float("nan")
        note = f"sparse eigsh failed: {exc}"
    return ConstrainedSystemDiagnostics(
        n_free=n,
        rank_dense=None,
        min_eigenvalue=lam_smallest,
        max_eigenvalue=None,
        condition_estimate=None,
        smallest_singular_value_dense=None,
        note=note,
    )


def residual_norm(K: csr_matrix, u: np.ndarray, f: np.ndarray) -> float:
    """Euclidean norm of the equilibrium residual ``K @ u - f``."""
    r = K @ u - f
    return float(np.linalg.norm(r))
