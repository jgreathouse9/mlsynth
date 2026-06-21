"""Cross-unit SCPI aggregation for staggered VanillaSC (Section 4 of CFPT 2025).

The single-unit reduction of the aggregator must reproduce the scalar SCPI ATT
band exactly (it is the same algebra), and the in-sample aggregation must be the
weighted sum of the per-unit conic draws.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.bilevel import BilevelSCM
from mlsynth.utils.vanillasc_helpers.scpi import scpi_intervals
from mlsynth.utils.vanillasc_helpers.staggered_scpi import (
    _aggregate_band,
    unit_components,
)


class _Cfg:
    """Minimal stand-in for the VanillaSC config fields the helper reads."""
    scpi_sims = 400
    alpha = 0.05
    scpi_e_method = "gaussian"
    seed = 3


def _panel(seed=0, J=8, T0=22, T1=6, r=3):
    rng = np.random.default_rng(seed)
    T = T0 + T1
    F = rng.standard_normal((T, r))
    Lam = rng.standard_normal((J + 1, r))
    Y = F @ Lam.T + rng.standard_normal((T, J + 1)) * 0.3
    y = Y[:, 0].copy()
    Y0 = Y[:, 1:]
    y[T0:] += 2.0
    return y, Y0, T0


def _components(seed_panel, seed_scpi):
    y, Y0, pre = _panel(seed=seed_panel)
    W = BilevelSCM("outcome-only").fit(
        y[:pre], Y0[:pre], X1=None, X0=None,
        donor_names=[f"d{j}" for j in range(Y0.shape[1])], predictor_names=[]).W
    comp = unit_components(_Cfg, y, Y0, pre, W, seed=seed_scpi)
    return comp, W


def test_single_unit_reduces_to_scalar_att_band():
    """One unit, averaged predictand row -> exactly the scalar SCPI ATT band."""
    comp, _ = _components(seed_panel=1, seed_scpi=5)
    n = comp["n_post"]
    point, lower, upper = _aggregate_band([(comp, 1.0, n)], u_alpha=_Cfg.alpha)
    md = comp["result"].metadata
    assert point == pytest.approx(md["att"], abs=1e-9)
    assert lower == pytest.approx(md["att_lower"], abs=1e-9)
    assert upper == pytest.approx(md["att_upper"], abs=1e-9)


def test_single_unit_period_row_reduces_to_per_period_band():
    """One unit, a post-period row -> that period's scalar effect interval."""
    comp, _ = _components(seed_panel=2, seed_scpi=6)
    sc = comp["result"]
    for k in range(comp["n_post"]):
        point, lower, upper = _aggregate_band([(comp, 1.0, k)], u_alpha=_Cfg.alpha)
        assert point == pytest.approx(float(sc.tau[k]), abs=1e-9)
        assert lower == pytest.approx(float(sc.lower[k]), abs=1e-9)
        assert upper == pytest.approx(float(sc.upper[k]), abs=1e-9)


def test_insample_aggregate_is_weighted_sum_of_draws():
    """The aggregated in-sample band is the quantile of the weighted draw sum."""
    c1, _ = _components(seed_panel=3, seed_scpi=7)
    c2, _ = _components(seed_panel=4, seed_scpi=8)
    n1, n2 = c1["n_post"], c2["n_post"]
    w1, w2 = 0.6, 0.4
    _, lower, upper = _aggregate_band(
        [(c1, w1, n1), (c2, w2, n2)], u_alpha=_Cfg.alpha)
    agg_lo = w1 * c1["in_lo"][:, n1] + w2 * c2["in_lo"][:, n2]
    agg_hi = w1 * c1["in_hi"][:, n1] + w2 * c2["in_hi"][:, n2]
    m_in = float(np.nanquantile(agg_lo, _Cfg.alpha / 2.0))
    mbar_in = float(np.nanquantile(agg_hi, 1.0 - _Cfg.alpha / 2.0))
    # out-of-sample for the two averaged rows, independent quadrature
    mean = w1 * c1["e_mean"][n1] + w2 * c2["e_mean"][n2]
    half = np.sqrt((w1 * c1["e_half"][n1]) ** 2 + (w2 * c2["e_half"][n2]) ** 2)
    point = (w1 * float((c1["obs"] - c1["cf"]).mean())
             + w2 * float((c2["obs"] - c2["cf"]).mean()))
    assert lower == pytest.approx(point - mbar_in - (mean + half), abs=1e-9)
    assert upper == pytest.approx(point - m_in - (mean - half), abs=1e-9)


def test_general_convention_widens_relative_to_independent():
    """The general cross-unit rule is at least as wide as the independent one."""
    c1, _ = _components(seed_panel=5, seed_scpi=9)
    c2, _ = _components(seed_panel=6, seed_scpi=10)
    n1, n2 = c1["n_post"], c2["n_post"]
    _, lo_ind, hi_ind = _aggregate_band([(c1, 0.5, n1), (c2, 0.5, n2)],
                                        u_alpha=_Cfg.alpha, cross_unit="independent")
    _, lo_gen, hi_gen = _aggregate_band([(c1, 0.5, n1), (c2, 0.5, n2)],
                                        u_alpha=_Cfg.alpha, cross_unit="general")
    assert (hi_gen - lo_gen) >= (hi_ind - lo_ind) - 1e-9
