# ADMM Playground

Clean, well-tested Python implementations of three classic **ADMM** (Alternating
Direction Method of Multipliers) problems with a unified solver interface and
built-in convergence diagnostics.

> This is a self-study project for understanding ADMM mechanics. The
> implementations follow Boyd et al. (2011) and Candès et al. (2011).

## Problems

| Solver | Problem | Key operation |
|--------|---------|---------------|
| **LASSO** | minimize (1/2)\|Ax−b\|² + λ\|x\|₁ | soft-thresholding |
| **Robust PCA** | minimize \|L\|∗ + λ\|S\|₁ s.t. M=L+S | singular value thresholding |
| **Matrix completion** | minimize (1/2)\|PΩ(X)−M\|² + λ\|X\|∗ | SVT + entry-wise blend |

## Install

```bash
pip install -e .
```

Requires: numpy, scipy, matplotlib (for examples).

## Quick start

```python
from admm_playground import lasso, rpca, completion

# LASSO
res = lasso.solve(A, b, lam=0.1)
print(res.x, res.converged, res.n_iter)

# Robust PCA
res = rpca.solve(M)          # M = L + S (unknown)
L, S = res.x, res.S           # recovered low-rank and sparse parts

# Matrix completion
res = completion.solve(M, Omega)   # Omega = boolean mask of observed entries
X = res.x                          # completed matrix
```

## Convergence diagnostics

Every solver returns a `Result` with per-iteration residuals:

```python
res.history.primal_residual   # list[float]
res.history.dual_residual     # list[float]
res.history.objective          # list[float]
```

Plot convergence curves:

```bash
python examples/plot_convergence.py   # saves examples/convergence.png
```

## Tests

```bash
python -m pytest tests/ -v
```

## Repository structure

```
admm-playground/
├── admm_playground/
│   ├── __init__.py
│   ├── base.py          # Result, ConvergenceHistory, soft-threshold, stopping rule
│   ├── lasso.py         # LASSO via ADMM (with adaptive rho)
│   ├── rpca.py          # Robust PCA via ADMM
│   └── completion.py    # Matrix completion via ADMM
├── tests/
│   └── test_solvers.py
├── examples/
│   └── plot_convergence.py
├── setup.py
└── README.md
```

## References

- Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2011).
  *Distributed Optimization and Statistical Learning via the Alternating
  Direction Method of Multipliers.* Foundations and Trends in Machine Learning.

- Candès, E. J., Li, X., Ma, Y., & Wright, J. (2011).
  *Robust Principal Component Analysis?* Journal of the ACM.

## License

MIT
