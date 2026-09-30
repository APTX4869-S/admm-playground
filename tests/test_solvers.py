"""Unit tests for admm-playground solvers.

Run with:  python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from admm_playground import lasso, rpca, completion
from admm_playground.base import _soft_threshold


def _rel_error(x, x_true):
    return np.linalg.norm(x - x_true) / np.linalg.norm(x_true)


# -----------------------------------------------------------------------
# soft threshold
# -----------------------------------------------------------------------

class TestSoftThreshold:
    def test_zero(self):
        assert np.allclose(_soft_threshold(np.array([0.0]), 0.5), [0.0])

    def test_positive(self):
        assert np.allclose(_soft_threshold(np.array([3.0]), 1.0), [2.0])

    def test_negative(self):
        assert np.allclose(_soft_threshold(np.array([-3.0]), 1.0), [-2.0])

    def test_shrink_to_zero(self):
        assert np.allclose(_soft_threshold(np.array([0.3]), 0.5), [0.0])

    def test_vector(self):
        x = np.array([-3, -0.5, 0, 0.5, 3])
        expected = np.array([-2, 0, 0, 0, 2])
        assert np.allclose(_soft_threshold(x, 1.0), expected)


# -----------------------------------------------------------------------
# LASSO
# -----------------------------------------------------------------------

class TestLASSO:
    def _make_problem(self, n=50, m=100, sparsity=5, seed=42):
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((m, n))
        x_true = np.zeros(n)
        idx = rng.choice(n, sparsity, replace=False)
        x_true[idx] = rng.standard_normal(sparsity) * 3
        b = A @ x_true + 0.01 * rng.standard_normal(m)
        return A, b, x_true

    def test_converges(self):
        A, b, x_true = self._make_problem()
        res = lasso.solve(A, b, lam=0.1, rho=1.0, max_iter=300)
        assert res.converged, f"LASSO did not converge in {res.n_iter} iters"

    def test_recovers_sparse(self):
        A, b, x_true = self._make_problem()
        res = lasso.solve(A, b, lam=0.1, rho=1.0, max_iter=500)
        assert _rel_error(res.x, x_true) < 0.1, f"rel_error={_rel_error(res.x, x_true):.3f}"

    def test_history_populated(self):
        A, b, _ = self._make_problem()
        res = lasso.solve(A, b, lam=0.1, max_iter=10)
        assert len(res.history) == res.n_iter
        assert len(res.history.primal_residual) > 0
        assert len(res.history.dual_residual) > 0


# -----------------------------------------------------------------------
# RPCA
# -----------------------------------------------------------------------

class TestRPCA:
    def _make_problem(self, m=40, n=40, rank=2, sparsity=0.05, seed=42):
        rng = np.random.default_rng(seed)
        L = rng.standard_normal((m, rank)) @ rng.standard_normal((rank, n))
        S = np.zeros((m, n))
        Omega = rng.random((m, n)) < sparsity
        S[Omega] = rng.standard_normal(Omega.sum()) * 5
        M = L + S
        return M, L, S

    def test_converges(self):
        M, L_true, S_true = self._make_problem()
        res = rpca.solve(M, max_iter=300)
        assert res.converged, "RPCA did not converge"

    def test_recover_low_rank(self):
        M, L_true, S_true = self._make_problem()
        res = rpca.solve(M, max_iter=500)
        err = _rel_error(res.x, L_true)
        assert err < 0.15, f"L recovery rel_error={err:.3f}"

    def test_recover_sparse(self):
        M, L_true, S_true = self._make_problem()
        res = rpca.solve(M, max_iter=500)
        err = _rel_error(res.S, S_true)
        assert err < 0.2, f"S recovery rel_error={err:.3f}"


# -----------------------------------------------------------------------
# Matrix Completion
# -----------------------------------------------------------------------

class TestCompletion:
    def _make_problem(self, m=30, n=30, rank=3, obs_ratio=0.5, seed=42):
        rng = np.random.default_rng(seed)
        L = rng.standard_normal((m, rank)) @ rng.standard_normal((rank, n))
        Omega = rng.random((m, n)) < obs_ratio
        M = np.zeros((m, n))
        M[Omega] = L[Omega]
        return M, Omega, L

    def test_converges(self):
        M, Omega, L_true = self._make_problem()
        res = completion.solve(M, Omega, max_iter=1000)
        assert res.converged, "Matrix completion did not converge"

    def test_recover(self):
        M, Omega, L_true = self._make_problem()
        res = completion.solve(M, Omega, max_iter=1000)
        # Check recovery on unobserved entries
        unobs = ~Omega
        err = np.linalg.norm(res.x[unobs] - L_true[unobs]) / np.linalg.norm(L_true[unobs])
        assert err < 0.3, f"unobserved rel_error={err:.3f}"
