"""Optional predictor-weight (V) optimisation for the covariate SRC variant.

Zhu (2023)'s Basque application (Algorithm 3 / Table 5) does not stop at equal
predictor weights: it optimises a diagonal ``V`` over the matching rows to best
predict the held-out pre-treatment outcomes, exactly as Abadie's ``synth()``
chooses predictor importance. mlsynth exposes this as an explicit, seeded option
(``v_search="de"``) rather than the default, for a reason worth stating plainly:

The ``V`` optimum is not identified. A large manifold of ``V`` achieves
essentially the same pre-outcome fit while producing post-period effects that
range over a wide band, so the per-donor weights are not a stable function of the
data -- different optimisers / seeds land on different points of that manifold
(a global differential-evolution search reproduces the paper's *average* placebo
MSPE but not its per-region cells). The search is therefore driven by a fixed
seed so a given call is reproducible, and it is opt-in so the deterministic,
Cp-identified Algorithm 1 stays the default. Use it to reproduce the paper's
covariate specification; do not read the exact weights as identified quantities.

Mirrors the outer-search philosophy of the repository's MSCMT bilevel backend
(Becker & Kloessner 2018): a log-scaled global search over ``V`` with the SRC
inner solve, differing only in the (box, Cp) inner problem.
"""

from __future__ import annotations

import numpy as np

from .estimation import src_weights


def optimize_v(
    X: np.ndarray,
    y: np.ndarray,
    n_outcome_rows: int,
    *,
    ridge: float = 1e-3,
    seed: int = 0,
    maxiter: int = 60,
    popsize: int = 12,
) -> np.ndarray:
    """Global (differential-evolution) search for the predictor weights ``V``.

    Minimises the held-out pre-outcome fit ``||z - E_out w(V)||^2`` over
    ``log10(V)``, where the outcome rows are the last ``n_outcome_rows`` rows of
    the matching matrix. Returns a length-``n`` non-negative ``V`` normalised to
    sum to ``n`` (scale is immaterial to the inner solve).

    The search is seeded, so the returned ``V`` is reproducible for a given
    ``(X, y, seed)``; see the module docstring on why ``V`` is not identified.
    """
    from scipy.optimize import differential_evolution

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n = X.shape[0]
    if n_outcome_rows < 1 or n_outcome_rows > n:
        raise ValueError(
            f"n_outcome_rows must be in [1, {n}]; got {n_outcome_rows}."
        )
    z = y[-n_outcome_rows:]
    Z = X[-n_outcome_rows:, :]

    def upper(xlog: np.ndarray) -> float:
        V = 10.0 ** xlog
        V = n * V / V.sum()
        r = src_weights(X, y, ridge=ridge, V=V)
        return float(np.sum((z - r.bias - Z @ r.combined) ** 2))

    res = differential_evolution(
        upper, bounds=[(-2.0, 2.0)] * n, seed=seed, maxiter=maxiter,
        popsize=popsize, tol=1e-9, mutation=(0.5, 1.0), recombination=0.7,
        polish=True, init="sobol",
    )
    V = 10.0 ** res.x
    return n * V / V.sum()
