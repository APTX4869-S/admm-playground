"""Matrix completion via ADMM.

Problem:
    minimize  (1/2) ||P_Omega(X) - P_Omega(M)||^2 + lam * ||X||_*
    where P_Omega is the projection onto observed entries (index set Omega).

ADMM form:
    minimize  (1/2) ||P_Omega(X) - P_Omega(M)||^2 + lam * ||Z||_*
    s.t.  X = Z

Updates:
    X^{k+1} = argmin  (1/2)||P_O(X) - P_O(M)||^2 + (rho/2)||X - Z + U/rho||^2
           = Z - U  on unobserved entries; (M + rho(Z - U)) / (1 + rho) on observed
    Z^{k+1} = SVT(X^{k+1} + U^k, lam / rho)
    U^{k+1} = U^k + X^{k+1} - Z^{k+1}
"""

from .base import Result, ConvergenceHistory, _stopping_criterion
import numpy as np


def _singular_value_threshold(M: np.ndarray, tau: float) -> np.ndarray:
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    s_thresh = np.maximum(s - tau, 0.0)
    return U * s_thresh @ Vt


def solve(M: np.ndarray, Omega: np.ndarray, lam: float = None,
          rho: float = 1.0, max_iter: int = 500,
          tol_abs: float = 1e-4, tol_rel: float = 1e-2) -> Result:
    """Complete a low-rank matrix from partial observations.

    Parameters
    ----------
    M : (m, n) array – observed matrix; entries NOT in Omega are ignored (can be 0)
    Omega : (m, n) bool array – True where M is observed
    lam : float – nuclear-norm weight; default 1/sqrt(max(m,n))
    rho : float – penalty parameter

    Returns
    -------
    Result with x = completed matrix (same shape as M).
    """
    m, n = M.shape
    if lam is None:
        lam = 1.0 / np.sqrt(max(m, n))
    if rho is None:
        rho = 1.0
    X = M.copy().astype(float)
    Z = np.zeros_like(M)
    U = np.zeros_like(M)

    history = ConvergenceHistory()
    converged = False

    M_obs = M[Omega]
    ones_obs = np.ones_like(M_obs)

    for k in range(max_iter):
        # X-update: closed form
        # On observed entries: (M + rho*(Z - U)) / (1 + rho)
        # On unobserved: Z - U
        X_new = Z - U
        X_new[Omega] = (M[Omega] + rho * (Z[Omega] - U[Omega])) / (1.0 + rho)

        # Z-update: SVT
        Z_new = _singular_value_threshold(X_new + U, lam / rho)

        r = X_new - Z_new          # primal residual
        s = rho * (Z_new - Z)      # dual residual
        U = U + (X_new - Z_new)

        history.primal_residual.append(np.linalg.norm(r, 2))
        history.dual_residual.append(np.linalg.norm(s, 2))
        nuc = np.linalg.svd(Z_new, compute_uv=False).sum()
        obj = 0.5 * np.linalg.norm(X_new[Omega] - M[Omega], 2) ** 2 + lam * nuc
        history.objective.append(obj)

        # Stopping criterion (Boyd et al., 2011, §3.3.1)
        r_norm = np.linalg.norm(r, 2)
        s_norm = np.linalg.norm(s, 2)
        if _stopping_criterion(r_norm, s_norm,
                               np.linalg.norm(X_new, 'fro'), np.linalg.norm(Z_new, 'fro'),
                               rho * np.linalg.norm(U, 'fro'),
                               m * n, tol_abs, tol_rel):
            converged = True
            Z = Z_new
            break

        Z = Z_new

    return Result(x=Z, history=history, n_iter=k + 1,
                  converged=converged, rho=rho)
