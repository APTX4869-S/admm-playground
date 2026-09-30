"""Base classes shared by all ADMM solvers."""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class ConvergenceHistory:
    """Stores per-iteration residuals for convergence diagnostics."""
    primal_residual: list = field(default_factory=list)
    dual_residual: list = field(default_factory=list)
    objective: list = field(default_factory=list)

    def __len__(self):
        return len(self.primal_residual)


@dataclass
class Result:
    """Unified result container for all ADMM solvers.

    Attributes
    ----------
    x : np.ndarray            – primary variable (solution)
    history : ConvergenceHistory – per-iteration residual / objective logs
    n_iter : int              – number of iterations run
    converged : bool          – whether tolerance was met
    rho : float               – final penalty parameter (may differ from init)
    """
    x: np.ndarray
    history: ConvergenceHistory
    n_iter: int
    converged: bool
    rho: float

    def summary(self) -> str:
        h = self.history
        n = len(h)
        if n == 0:
            return f"Result(n_iter={self.n_iter}, converged={self.converged}, no history)"
        return (
            f"Result(n_iter={self.n_iter}, converged={self.converged}, "
            f"rho={self.rho:.4f}, "
            f"final_primal_res={h.primal_residual[-1]:.2e}, "
            f"final_dual_res={h.dual_residual[-1]:.2e})"
        )


def _soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    """Element-wise soft-thresholding (proximal operator of L1 norm)."""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def _stopping_criterion(r_norm: float, s_norm: float,
                        x_norm: float, z_norm: float, y_norm: float,
                        dim: int, tol_abs: float = 1e-4,
                        tol_rel: float = 1e-2) -> bool:
    """Standard ADMM stopping criterion (Boyd et al., 2011, §3.3.1).

    ||r||  <= eps_pri  = sqrt(dim) * tol_abs + tol_rel * max(||x||, ||z||)
    ||s||  <= eps_dual = sqrt(dim) * tol_abs + tol_rel * ||y||

    Parameters
    ----------
    r_norm : primal residual norm
    s_norm : dual residual norm
    x_norm, z_norm : norms of primal and auxiliary variables
    y_norm : norm of the (unscaled) dual variable
    dim : dimension of the primal variable
    """
    eps_pri = np.sqrt(dim) * tol_abs + tol_rel * max(x_norm, z_norm)
    eps_dual = np.sqrt(dim) * tol_abs + tol_rel * y_norm
    return r_norm <= eps_pri and s_norm <= eps_dual
