"""The six panels this study measures on, all drawn by the library's own DGPs.

Each adapter returns ``(YN, YI, T0)``: an untreated panel shaped ``(T, J)``,
the treated panel where the DGP defines one (``None`` otherwise), and the last
pre-period index. MAREX designs on the pre-period of ``YN``; the harness
applies the treatment to whichever units the design selects, taking the effect
from ``YI`` when the DGP supplies it and from the panel's own scale when it
does not.

The six are chosen to vary what makes a control market hard to reconstruct:
a correctly specified factor panel, a rank condition that holds and one that
breaks, serial correlation, a shared stochastic trend, and seasonality.
"""
from __future__ import annotations

import numpy as np

from mlsynth.utils.clustersc_helpers.simulation import simulate_rank_shift_panel
from mlsynth.utils.fdid_helpers.simulation import simulate_fdid_serial_sample
from mlsynth.utils.hsc_helpers.simulation import make_hsc_loadings, simulate_hsc_regime
from mlsynth.utils.marex_helpers.simulation import generate_marex_sample
from mlsynth.utils.pangeo_helpers.simulation import make_seasonal_sales_panel


def marex_native(seed: int):
    """Abadie and Zhao's own baseline design DGP (their Section 5).

    The only adapter that supplies ``Y_I``, so here the treatment effect is the
    paper's and not one this study imposes.
    """
    s = generate_marex_sample(J=15, R=7, F=11, T=30, T0=25, sigma=1.0,
                              rng=np.random.default_rng(seed))
    return s.Y_N.T, s.Y_I.T, s.T0


def _rank_shift(dormant: bool):
    def adapter(seed: int):
        r = simulate_rank_shift_panel(dormant_factor=dormant, N=12, T=60,
                                      T0=40, noise=0.3, seed=seed)
        return r.observed.T, None, r.T0
    return adapter


def fdid_ar1(seed: int):
    """Web Appendix E DGP 2 with an AR(1) residual at the fitted optimum."""
    s = simulate_fdid_serial_sample(rho=0.7, N=20, T1=40, T2=10,
                                    rng=np.random.default_rng(seed))
    Y = np.vstack([np.asarray(s.Y_treated)[None, :], np.asarray(s.Y_controls)])
    return Y.T, None, s.T1


def hsc_shared_trend(seed: int):
    """A shared stochastic trend (``rho_u=1``), which makes extrapolation hard."""
    rng = np.random.default_rng(seed)
    Y = simulate_hsc_regime(rng, make_hsc_loadings(N0=14, seed=seed),
                            N0=14, T0=40, Tpost=10, rho_u=1.0)
    return Y.T, None, 40


def pangeo_seasonal(seed: int):
    """Two years of weekly geo sales with a 52-week season."""
    df = make_seasonal_sales_panel(units_per_arm=7, arms=("A", "B"), T=104,
                                   season_period=52, noise=0.08, seed=seed)
    W = df.pivot(index="time", columns="unit", values="sales").to_numpy()
    return W, None, W.shape[0] - 10


DGPS = {
    "marex_native": marex_native,
    "rank_shift_ok": _rank_shift(False),
    "rank_shift_dormant": _rank_shift(True),
    "fdid_ar1_rho.7": fdid_ar1,
    "hsc_shared_trend": hsc_shared_trend,
    "pangeo_seasonal": pangeo_seasonal,
}
