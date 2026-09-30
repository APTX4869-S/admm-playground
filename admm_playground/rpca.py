"""Robust PCA via ADMM (Candes, Li, Ma, Wright, 2011).

Problem:
    minimize  ||L||_* + lam * ||S||_1   s.t. M = L + S

The solver internally normalizes M to max|M|=1 for numerical stability,
then scales the result back. The ADMM penalty rho is chosen relative to
the normalized data, following the standard practice in the RPCA literature.
"""

from .base import Result, ConvergenceHistory, _soft_threshold, _stopping_criterion
import numpy as np


def _singular_value_threshold(M: np.ndarray, tau: float) -> np.ndarray:
    """Proximal operator of nuclear norm: SVT(M, tau) = U * S_tau(Sigma) * V^T."""
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    s_thresh = np.maximum(s - tau, 0.0)
    return U * s_thresh @ Vt


def solve(M: np.ndarray, lam: float = None,
          rho: float = None, max_iter: int = 500,
          tol_abs: float = 1e-4, tol_rel: float = 1e-2) -> Result:
    """Solve Robust PCA via ADMM.

    Parameters
    ----------
    M : (m, n) array – observed matrix (low-rank + sparse corruption)
    lam : float – sparsity weight; default 1/sqrt(max(m, n))
    rho : float – penalty parameter; default 1.0 (on normalized data)
    max_iter, tol_abs, tol_rel – stopping criteria

    Returns
    -------
    Result with x = low-rank component L, plus convergence history.
    Also stores the sparse component S in result.S.
    """
    m, n = M.shape
    if lam is None:
        lam = 1.0 / np.sqrt(max(m, n))
    if rho is None:
        rho = 1.0

    # Normalize for numerical stability
    scale = np.max(np.abs(M))
    if scale > 0:
        Mn = M / scale
    else:
        Mn = M.copy()

    L = np.zeros_like(Mn)
    S = np.zeros_like(Mn)
    Y = np.zeros_like(Mn)

    history = ConvergenceHistory()
    converged = False

    for k in range(max_iter):
        L_new = _singular_value_threshold(Mn - S - Y / rho, 1.0 / rho)
        S_new = _soft_threshold(Mn - L_new - Y / rho, lam / rho)
        r = L_new + S_new - Mn
        s = -rho * (L_new - L)
        Y = Y + rho * r

        history.primal_residual.append(np.linalg.norm(r, 2))
        history.dual_residual.append(np.linalg.norm(s, 2))
        obj = np.linalg.svd(L_new, compute_uv=False).sum() + lam * np.linalg.norm(S_new, 1)
        history.objective.append(obj)

        if _stopping_criterion(np.linalg.norm(r, 2), np.linalg.norm(s, 2),
                               np.linalg.norm(L_new, 'fro'), np.linalg.norm(S_new, 'fro'),
                               np.linalg.norm(Y, 'fro'),
                               m * n, tol_abs, tol_rel):
            converged = True
            L, S = L_new, S_new
            break

        L, S = L_new, S_new

    # Scale back
    L *= scale
    S *= scale

    from dataclasses import dataclass
    @dataclass
    class RPCAResult(Result):
        S: np.ndarray = None

    return RPCAResult(x=L, S=S, history=history, n_iter=k + 1,
                      converged=converged, rho=rho)
