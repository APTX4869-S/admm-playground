"""LASSO via ADMM — minimizes  (1/2)||Ax - b||^2 + lambda * ||x||_1.

ADMM form (Boyd et al., 2011, §6.4):
    minimize  (1/2)||Ax - b||^2 + lambda ||z||_1   s.t. x = z

Updates:
    x^{k+1} = (A^T A + rho I)^{-1} (A^T b + rho z^k - y^k)
    z^{k+1} = S_{lambda/rho}(x^{k+1} + y^k / rho)
    y^{k+1} = y^k + rho (x^{k+1} - z^{k+1})
"""

from .base import Result, ConvergenceHistory, _soft_threshold, _stopping_criterion
import numpy as np


def solve(A: np.ndarray, b: np.ndarray, lam: float,
          rho: float = 1.0, max_iter: int = 500,
          tol_abs: float = 1e-4, tol_rel: float = 1e-2,
          adaptive_rho: bool = True, rho_min: float = 1e-6,
          rho_max: float = 1e6) -> Result:
    """Solve LASSO via ADMM.

    Parameters
    ----------
    A : (m, n) array
    b : (m,) array
    lam : float – L1 regularization strength
    rho : float – initial penalty parameter
    max_iter, tol_abs, tol_rel – stopping criteria
    adaptive_rho : bool – adjust rho based on residual balance (Boyd §3.4.1)

    Returns
    -------
    Result with x = solution (n,), plus convergence history.
    """
    m, n = A.shape
    x = np.zeros(n)
    z = np.zeros(n)
    u = np.zeros(n)  # scaled dual variable (y = rho * u in Boyd's notation)

    # Precompute factorization for x-update: (A^T A + rho I)^{-1} A^T b
    # For fixed rho we factor once; for adaptive rho we refactor on change.
    AtA = A.T @ A
    Atb = A.T @ b
    L = np.linalg.cholesky(AtA + rho * np.eye(n))
    lower = True

    def solve_x(z, u):
        rhs = Atb + rho * (z - u)
        return np.linalg.solve(L.T, np.linalg.solve(L, rhs))

    history = ConvergenceHistory()
    converged = False

    for k in range(max_iter):
        x = solve_x(z, u)
        z_new = _soft_threshold(x + u, lam / rho)
        r = x - z_new               # primal residual
        s = rho * (z_new - z)       # dual residual
        u = u + (x - z_new)

        history.primal_residual.append(np.linalg.norm(r, 2))
        history.dual_residual.append(np.linalg.norm(s, 2))
        # Objective: (1/2)||Ax - b||^2 + lam * ||z_new||_1
        obj = 0.5 * np.linalg.norm(A @ x - b, 2) ** 2 + lam * np.linalg.norm(z_new, 1)
        history.objective.append(obj)

        if _stopping_criterion(np.linalg.norm(r, 2), np.linalg.norm(s, 2),
                               np.linalg.norm(x, 2), np.linalg.norm(z_new, 2),
                               rho * np.linalg.norm(u, 2),
                               n, tol_abs, tol_rel):
            converged = True
            z = z_new
            break

        # Adaptive rho (Boyd §3.4.1)
        if adaptive_rho and k > 0 and k % 10 == 0:
            nr = history.primal_residual[-1]
            ns = history.dual_residual[-1]
            if nr > 10 * ns:
                rho = min(rho * 2, rho_max)
                L = np.linalg.cholesky(AtA + rho * np.eye(n))
            elif ns > 10 * nr:
                rho = max(rho / 2, rho_min)
                L = np.linalg.cholesky(AtA + rho * np.eye(n))
            u = u / (rho)  # rescale scaled dual (since rho changed)
            # Note: after rescaling, z stays the same; u absorbs the factor.

        z = z_new

    return Result(x=z, history=history, n_iter=k + 1,
                  converged=converged, rho=rho)
