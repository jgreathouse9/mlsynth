"""Split-conformal prediction band for VanillaSC (``inference="conformal_split"``).

Adds the simple symmetric split-conformal band -- a constant half-width ``q``
around the synthetic counterfactual, where ``q`` is the
``ceil((n+1)(1-alpha))``-th order statistic of the absolute pre-period gaps --
matching the ``method="conformal"`` construction in Jens Hainmueller's R
``Synth`` (``synth_inference()``; j-hai/Synth 1.2.0). This is the split
(Chernozhukov-Wuthrich-Zhu) band, distinct from VanillaSC's existing
``inference="conformal"`` full test-inversion band, which widens over the
post-period.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlsynth import VanillaSC


def _panel(n_units: int, n_periods: int = 14, t0: int = 9, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    donor_base = rng.normal(10.0, 1.0, size=max(n_units - 1, 1))
    loads = rng.dirichlet(np.ones(max(n_units - 1, 1)))
    rows = []
    for t in range(n_periods):
        common = 0.4 * t + rng.normal(0.0, 0.2)
        donors = donor_base + common + rng.normal(0.0, 0.15, size=donor_base.size)
        treated = float(loads @ donors) + (4.0 if t >= t0 else 0.0)
        rows.append({"unit": "u0", "time": t, "y": treated, "treat": int(t >= t0)})
        for j, dv in enumerate(donors):
            rows.append({"unit": f"d{j}", "time": t, "y": float(dv), "treat": 0})
    return pd.DataFrame(rows)


_CFG = dict(outcome="y", treat="treat", unitid="unit", time="time",
            backend="outcome-only", display_graphs=False)


def _jhai_conformal_q(pre_gaps, alpha):
    """The exact j-hai/Synth synth_inference() split-conformal quantile
    (R: r <- sort(abs(effect[pre])); k <- ceiling((n+1)(1-alpha)); q <- r[k])."""
    r = np.sort(np.abs(np.asarray(pre_gaps, dtype=float)))
    n = r.size
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(r[k - 1]) if k <= n else np.inf


def test_helper_matches_jhai_synth_formula():
    from mlsynth.utils.inferutils import split_conformal_quantile

    rng = np.random.default_rng(3)
    for n in (8, 12, 19, 30):
        for alpha in (0.1, 0.05, 0.2):
            g = rng.normal(0, 2, size=n)
            assert split_conformal_quantile(g, alpha) == pytest.approx(
                _jhai_conformal_q(g, alpha)), (n, alpha)


def test_helper_uninformative_q_is_inf():
    from mlsynth.utils.inferutils import split_conformal_quantile
    # n < ceil(1/alpha) - 1 => k > n => q = inf
    assert np.isinf(split_conformal_quantile([1.0, 2.0, 3.0], alpha=0.05))
    assert np.isfinite(split_conformal_quantile(np.arange(1.0, 40.0), alpha=0.05))


def test_end_to_end_constant_width_band():
    # >= 19 pre-periods so the alpha=0.05 order statistic exists (finite q).
    df = _panel(8, n_periods=32, t0=22)
    res = VanillaSC({"df": df, "inference": "conformal_split", **_CFG}).fit()
    assert res.inference is not None
    assert "split" in res.inference.method.lower()
    lo = np.asarray(res.time_series.counterfactual_lower, dtype=float)
    hi = np.asarray(res.time_series.counterfactual_upper, dtype=float)
    cf = np.asarray(res.time_series.counterfactual_outcome, dtype=float)
    post = ~np.isnan(lo)
    q = res.inference.details["conformal_q"]
    assert np.isfinite(q) and q > 0
    widths = (hi - lo)[post]
    # constant half-width q at every post period (the split-conformal signature)
    assert np.allclose(widths, 2.0 * q)
    assert np.allclose(cf[post] - q, lo[post]) and np.allclose(cf[post] + q, hi[post])


def test_end_to_end_q_equals_reference_on_fit():
    df = _panel(8, n_periods=32, t0=22)
    res = VanillaSC({"df": df, "inference": "conformal_split", **_CFG}).fit()
    gap = np.asarray(res.time_series.estimated_gap, dtype=float)
    n_pre = int(res.additional_outputs["pre_periods"])
    q = res.inference.details["conformal_q"]
    assert np.isfinite(q)
    assert q == pytest.approx(_jhai_conformal_q(gap[:n_pre], 0.05))


def test_uninformative_band_warns():
    # 4 pre-periods < ceil(1/0.05)-1 = 19 => q = inf, band uninformative
    df = _panel(8, n_periods=6, t0=4)
    with pytest.warns(UserWarning, match="uninformative|inf"):
        res = VanillaSC({"df": df, "inference": "conformal_split", **_CFG}).fit()
    assert np.isinf(res.inference.details["conformal_q"])
