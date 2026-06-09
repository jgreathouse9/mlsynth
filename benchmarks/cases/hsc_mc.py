"""Path B benchmark: HSC regime adaptation (Liu & Xu, Appendix C.1).

Reproduces the paper's Monte Carlo headline: HSC's cross-validated allocation
rho adapts to the trend regime, so HSC tracks whichever fixed estimator is best
-- SC-on-levels (with intercept) under a *shared* stochastic trend, SC-on-first-
differences under an *idiosyncratic* one -- while each fixed estimator fails in
the other regime. The treated unit gets no effect, so post-period RMSE is pure
counterfactual error.

Provenance
----------
* DGP: :func:`mlsynth.utils.hsc_helpers.simulation.simulate_hsc_regime`
  (+ :func:`make_hsc_loadings`), the paper's Appendix-C.1 model with
  ``N0=20``, ``T0=100``, ``Tpost=10``, ``kappa=2``; ``rho_u=1`` is the shared
  regime, ``rho_u=0`` the idiosyncratic one.
* Headline: Liu & Xu MC -- common drift: SC-INT ~1.15 / SC-diff ~1.48 / HSC
  ~1.21 / rho_hat ~0.86; idiosyncratic: SC-INT ~10.6 / SC-diff ~6.11 / HSC ~6.46
  / rho_hat ~0.48. The paper uses more reps; we use 40 (bands absorb the gap).
"""
from __future__ import annotations

import warnings

import numpy as np

N0, T0, TPOST, R = 20, 100, 10, 40
KAPPA = 2.0


def _rmse(e) -> float:
    return float(np.sqrt(np.mean(np.asarray(e) ** 2)))


def _regime(rho_u: float):
    """Return (hsc_rmse, scint_rmse, scdiff_rmse, mean_rho) over R reps."""
    import cvxpy as cp
    import pandas as pd

    from mlsynth import HSC
    from mlsynth.utils.hsc_helpers.simulation import (
        make_hsc_loadings, simulate_hsc_regime,
    )

    def simplex(X, y):
        w = cp.Variable(X.shape[1])
        cp.Problem(cp.Minimize(cp.sum_squares(y - X @ w)),
                   [w >= 0, cp.sum(w) == 1]).solve(solver=cp.CLARABEL)
        return np.clip(w.value, 0, None)

    def sc_int(Xp, Yp, Xpost):
        w = simplex(Xp - Xp.mean(0), Yp - Yp.mean())
        return Xpost @ w + (Yp.mean() - Xp.mean(0) @ w)

    def sc_diff(Xp, Yp, Xpost):
        w = simplex(np.diff(Xp, axis=0), np.diff(Yp))
        return Yp[-1] + (Xpost - Xp[-1]) @ w

    def to_df(Y):
        return pd.DataFrame([{"unit": f"u{j:02d}", "time": t, "y": float(Y[j, t]),
                              "treat": int(j == 0 and t >= T0)}
                             for j in range(Y.shape[0]) for t in range(Y.shape[1])])

    loadings = make_hsc_loadings(N0, seed=0)
    ei, ed, eh, rhos = [], [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for r in range(R):
            rng = np.random.default_rng(1000 + r)
            Y = simulate_hsc_regime(rng, loadings, N0=N0, T0=T0, Tpost=TPOST,
                                    kappa=KAPPA, rho_u=rho_u)
            Yp, Xp = Y[0, :T0], Y[1:, :T0].T
            Xpost, Y0p = Y[1:, T0:].T, Y[0, T0:]
            ei.append(sc_int(Xp, Yp, Xpost) - Y0p)
            ed.append(sc_diff(Xp, Yp, Xpost) - Y0p)
            res = HSC({"df": to_df(Y), "outcome": "y", "unitid": "unit",
                       "time": "time", "treat": "treat"}).fit()
            eh.append(res.design.counterfactual_post - Y0p)
            rhos.append(res.selected_rho)
    return _rmse(eh), _rmse(ei), _rmse(ed), float(np.mean(rhos))


def run() -> dict:
    hsc_c, scint_c, scdiff_c, rho_c = _regime(1.0)   # shared / common drift
    hsc_i, scint_i, scdiff_i, rho_i = _regime(0.0)   # idiosyncratic
    return {
        "rho_common": rho_c,
        "rho_idio": rho_i,
        "hsc_common": hsc_c,
        "hsc_idio": hsc_i,
        "scint_idio": scint_i,
        # 1.0 iff the CV allocation leans to levels under a shared trend and to
        # differencing under an idiosyncratic one (rho adapts).
        "rho_adapts": float(rho_c > rho_i),
        # 1.0 iff HSC tracks the best fixed method (within 1.4x) in BOTH regimes.
        "hsc_tracks_best": float(
            hsc_c <= 1.4 * min(scint_c, scdiff_c)
            and hsc_i <= 1.4 * min(scint_i, scdiff_i)),
        # 1.0 iff SC-on-levels is regime-fragile: catastrophic when the trend is
        # idiosyncratic (>> its own shared-regime error, and worse than HSC).
        "scint_fragile": float(scint_i > 3.0 * scint_c and scint_i > hsc_i),
    }


# Deterministic (fixed loadings seed 0; per-rep seeds 1000+r). Binding facts:
# rho adapts across regimes, HSC tracks the oracle-best fixed method in both, and
# SC-on-levels is catastrophic in the idiosyncratic regime. Per-quantity values
# carry MC bands (40 reps vs the paper's larger M).
EXPECTED = {
    "rho_common": (0.86, 0.15),
    "rho_idio": (0.55, 0.22),
    "hsc_common": (1.15, 0.30),
    "hsc_idio": (6.67, 1.50),
    "scint_idio": (10.31, 2.50),
    "rho_adapts": (1.0, 0.0),
    "hsc_tracks_best": (1.0, 0.0),
    "scint_fragile": (1.0, 0.0),
}
