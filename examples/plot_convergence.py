"""Example: run all three solvers and plot convergence curves.

Usage:
    python examples/plot_convergence.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from admm_playground import lasso, rpca, completion


def demo_lasso():
    rng = np.random.default_rng(42)
    m, n = 100, 50
    A = rng.standard_normal((m, n))
    x_true = np.zeros(n)
    x_true[rng.choice(n, 5, replace=False)] = rng.standard_normal(5) * 3
    b = A @ x_true + 0.01 * rng.standard_normal(m)
    return lasso.solve(A, b, lam=0.1, rho=1.0, max_iter=300), "LASSO"


def demo_rpca():
    rng = np.random.default_rng(42)
    m, n = 60, 60
    L = rng.standard_normal((m, 2)) @ rng.standard_normal((2, n))
    S = np.zeros((m, n))
    mask = rng.random((m, n)) < 0.05
    S[mask] = rng.standard_normal(mask.sum()) * 5
    M = L + S
    return rpca.solve(M, max_iter=300), "Robust PCA"


def demo_completion():
    rng = np.random.default_rng(42)
    m, n = 30, 30
    L = rng.standard_normal((m, 3)) @ rng.standard_normal((3, n))
    Omega = rng.random((m, n)) < 0.4
    M = np.zeros((m, n))
    M[Omega] = L[Omega]
    return completion.solve(M, Omega, max_iter=300), "Matrix Completion"


def main():
    results = [demo_lasso(), demo_rpca(), demo_completion()]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    for ax, (res, name) in zip(axes, results):
        iters = range(1, len(res.history.primal_residual) + 1)
        ax.semilogy(iters, res.history.primal_residual, label="primal", lw=1.5)
        ax.semilogy(iters, res.history.dual_residual, label="dual", lw=1.5, ls="--")
        ax.set_title(f"{name}\n({res.n_iter} iters, "
                      f"{'converged' if res.converged else 'max iter'})",
                      fontsize=10)
        ax.set_xlabel("iteration")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("residual (log scale)")
    fig.suptitle("ADMM Convergence: Primal vs Dual Residual", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig("examples/convergence.png", dpi=150, bbox_inches="tight")
    print("Saved: examples/convergence.png")

    for res, name in results:
        print(f"{name:20s}  iters={res.n_iter:4d}  "
              f"converged={res.converged}  "
              f"final_primal={res.history.primal_residual[-1]:.2e}  "
              f"final_dual={res.history.dual_residual[-1]:.2e}")


if __name__ == "__main__":
    main()
