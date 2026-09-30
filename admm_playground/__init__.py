"""ADMM Playground — clean implementations of classic ADMM solvers.

Modules
-------
- lasso:    L1-regularized least squares via ADMM
- rpca:     Robust PCA (Candes et al., 2011) via ADMM
- completion: Matrix completion via ADMM

All solvers share a unified ``solve`` interface and return a ``Result``
object with convergence diagnostics.
"""

from .base import Result, ConvergenceHistory

__all__ = ["Result", "ConvergenceHistory"]
__version__ = "0.1.0"
