r"""The HAC variance for a measured TBRMM design (Li and Van den Bulte 2022).

Equation 6's noise term is :math:`T\sigma^2`, which prices the test window's
errors as independent. When they are not, the variance of their sum is larger by
:math:`1 + 2\sum_j (1 - j/T)\rho_j`, and the interval is too narrow by the square
root of that. Their Proposition 3.4 replaces both terms of the variance with
Newey-West truncated sums at bandwidth :math:`\ell = \lceil T_1^{1/4}\rceil`, and
their proof D.2 notes that the independent case is the special case where the
truncation keeps only the diagonal.

Ground truth here is an AR(1) residual process, where the inflation factor is
known in closed form, so the tests assert against a number the design fixes
rather than against the implementation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import TBRMM
from mlsynth.config_models import TBRMMConfig
from mlsynth.exceptions import MlsynthConfigError


def _panel(n_units: int = 8, n_pre: int = 60, n_post: int = 10,
           rho: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """A panel whose treated geo carries AR(1) deviations from the controls.

    ``rho`` is the serial correlation of the treated geo's own disturbance, which
    is what the fit's residual inherits and what the HAC term exists to price.
    """
    rng = np.random.default_rng(seed)
    T = n_pre + n_post
    factor = np.cumsum(rng.normal(size=T)) + 50.0
    level = rng.uniform(8.0, 20.0, n_units)
    loading = rng.uniform(0.9, 1.1, n_units)
    Y = level[None, :] + loading[None, :] * factor[:, None] + rng.normal(0, 0.4, (T, n_units))
    e = np.zeros(T)
    innov = rng.normal(0, 3.0, T)
    for t in range(1, T):
        e[t] = rho * e[t - 1] + innov[t]
    Y[:, 0] += e                                   # geo 0 gets the AR(1) noise
    units = [f"g{j:02d}" for j in range(n_units)]
    # Force geo 0 into treatment. Left free, the search avoids it precisely
    # because its disturbance is large, and the correction is never exercised.
    return pd.DataFrame([{"geo": units[j], "t": t, "sales": float(Y[t, j]),
                          "post": int(t >= n_pre),
                          "can_treat": int(j == 0),
                          "can_control": int(j != 0),
                          "can_exclude": 0}
                         for j in range(n_units) for t in range(T)])


def _cfg(df, **kw):
    base = dict(df=df, outcome="sales", unitid="geo", time="t",
                max_treatment_size=1, n_test=8, post_col="post",
                treatment_eligible_col="can_treat",
                control_eligible_col="can_control",
                unassigned_eligible_col="can_exclude")
    base.update(kw)
    return TBRMMConfig(**base)


# ---------------------------------------------------------------------------
# The default is untouched
# ---------------------------------------------------------------------------

def test_the_default_is_the_iid_variance():
    """No opt-in, no change: the shipped behaviour is equation 6."""
    res = TBRMM(_cfg(_panel())).fit()
    q = res.recommended.effect.posterior

    assert q.variance == "iid"
    assert q.bandwidth is None


def test_opting_in_does_not_move_the_point_estimate():
    """The variance choice prices uncertainty; it does not re-estimate anything."""
    base = TBRMM(_cfg(_panel())).fit().recommended.effect
    hac = TBRMM(_cfg(_panel(), variance="hac")).fit().recommended.effect

    assert hac.att == pytest.approx(base.att, rel=1e-12)
    assert hac.total_effect == pytest.approx(base.total_effect, rel=1e-12)
    for a, b in zip(base.market_effects, hac.market_effects):
        assert a.att == pytest.approx(b.att, rel=1e-12)


# ---------------------------------------------------------------------------
# What the correction does
# ---------------------------------------------------------------------------

def test_a_zero_bandwidth_is_the_heteroskedasticity_robust_sandwich():
    """Proof D.2 truncated at lag zero keeps only the diagonal, which is HC1.

    That is not equation 6. The two coincide only under homoskedasticity, since
    equation 6 scales ``(X'X)^-1`` by one sigma^2 while the diagonal sandwich
    weights each period by its own squared residual. The test computes HC1
    directly so it pins the identity rather than restating the implementation.
    """
    df = _panel(rho=0.6, n_pre=60, n_post=10)
    res = TBRMM(_cfg(df, variance="hac", hac_bandwidth=0)).fit()
    q = res.recommended.effect.posterior

    w = df.pivot(index="t", columns="geo", values="sales").sort_index()
    flag = df.groupby("t")["post"].max().sort_index().astype(int).values
    pre, post = w[flag == 0], w[flag == 1]
    y = pre[list(res.recommended.treatment_units)].sum(axis=1).to_numpy()
    x = pre[list(res.recommended.control_units)].mean(axis=1).to_numpy()
    xp = post[list(res.recommended.control_units)].mean(axis=1).to_numpy()
    X = np.column_stack([np.ones(x.size), x])
    u = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    dfc = x.size / (x.size - 2.0)
    meat = (X * u[:, None]).T @ (X * u[:, None]) * dfc
    V = np.linalg.inv(X.T @ X)
    c = np.array([float(xp.size), float(xp.sum())])
    expected = np.sqrt(float(c @ (V @ meat @ V) @ c)
                       + xp.size * float(u @ u) / x.size * dfc)

    assert q.scale == pytest.approx(expected, rel=1e-9)


def test_positively_correlated_residuals_widen_the_interval():
    """The case the correction exists for."""
    iid = TBRMM(_cfg(_panel(rho=0.7))).fit().recommended.effect.posterior
    hac = TBRMM(_cfg(_panel(rho=0.7), variance="hac")).fit().recommended.effect.posterior

    assert hac.scale > iid.scale
    assert hac.total_upper - hac.total_lower > iid.total_upper - iid.total_lower


def test_the_widening_tracks_the_known_ar1_inflation():
    """Ground truth: Var(sum) / (T sigma^2) = 1 + 2 sum_j (1 - j/T) rho^j.

    The realised factor is checked against the AR(1) value the panel was built
    with, loosely, because a finite pretest estimates rho with error and the
    Bartlett taper is deliberately conservative.
    """
    rho, T2 = 0.7, 10
    iid = TBRMM(_cfg(_panel(rho=rho))).fit().recommended.effect.posterior
    hac = TBRMM(_cfg(_panel(rho=rho), variance="hac")).fit().recommended.effect.posterior

    lags = np.arange(1, T2)
    expected = 1.0 + 2.0 * np.sum((1 - lags / T2) * rho ** lags)
    realised = (hac.scale / iid.scale) ** 2
    assert 1.2 < realised < expected * 1.6


def test_uncorrelated_residuals_leave_the_scale_close_to_the_iid_one():
    """With nothing to price, the correction should barely move."""
    iid = TBRMM(_cfg(_panel(rho=0.0))).fit().recommended.effect.posterior
    hac = TBRMM(_cfg(_panel(rho=0.0), variance="hac")).fit().recommended.effect.posterior

    assert hac.scale == pytest.approx(iid.scale, rel=0.45)


def test_the_correction_is_not_always_a_widening():
    """Negative serial correlation prices the other way; HAC is not a safety margin."""
    iid = TBRMM(_cfg(_panel(rho=-0.7))).fit().recommended.effect.posterior
    hac = TBRMM(_cfg(_panel(rho=-0.7), variance="hac")).fit().recommended.effect.posterior

    assert hac.scale < iid.scale


# ---------------------------------------------------------------------------
# The bandwidth
# ---------------------------------------------------------------------------

def test_the_default_bandwidth_is_the_papers():
    """l = ceil(T1 ** (1/4)), recorded on the posterior."""
    df = _panel(n_pre=60, n_post=10)
    q = TBRMM(_cfg(df, variance="hac")).fit().recommended.effect.posterior

    assert q.bandwidth == int(np.ceil(60 ** 0.25))
    assert q.variance == "hac"


def test_an_explicit_bandwidth_is_used_and_recorded():
    q = TBRMM(_cfg(_panel(rho=0.5), variance="hac",
                   hac_bandwidth=6)).fit().recommended.effect.posterior
    assert q.bandwidth == 6


def test_a_negative_bandwidth_is_refused():
    with pytest.raises(Exception):
        _cfg(_panel(), variance="hac", hac_bandwidth=-1)


def test_a_bandwidth_longer_than_the_pretest_is_refused():
    with pytest.raises((MlsynthConfigError, Exception)):
        TBRMM(_cfg(_panel(n_pre=60), variance="hac", hac_bandwidth=60)).fit()


def test_an_unknown_variance_name_is_refused():
    with pytest.raises(Exception):
        _cfg(_panel(), variance="sandwich")


# ---------------------------------------------------------------------------
# The group fit's parameters
# ---------------------------------------------------------------------------

def test_the_group_fit_reports_its_own_two_parameters():
    """delta1/delta2 for the summed treated series, so no caller has to refit."""
    df = _panel()
    res = TBRMM(_cfg(df)).fit()
    eff = res.recommended.effect

    w = df.pivot(index="t", columns="geo", values="sales").sort_index()
    flag = df.groupby("t")["post"].max().sort_index().astype(int)
    pre = w[flag.values == 0]
    y = pre[list(res.recommended.treatment_units)].sum(axis=1).to_numpy()
    x = pre[list(res.recommended.control_units)].mean(axis=1).to_numpy()
    coef = np.linalg.lstsq(np.column_stack([np.ones(x.size), x]), y, rcond=None)[0]

    assert eff.delta1 == pytest.approx(float(coef[0]), rel=1e-9)
    assert eff.delta2 == pytest.approx(float(coef[1]), rel=1e-9)


def test_a_single_treated_geo_shares_its_parameters_with_the_group():
    """One geo summed is that geo, so the two fits coincide."""
    res = TBRMM(_cfg(_panel(), max_treatment_size=1)).fit()
    eff = res.recommended.effect

    assert eff.delta1 == pytest.approx(eff.market_effects[0].delta1, rel=1e-9)
    assert eff.delta2 == pytest.approx(eff.market_effects[0].delta2, rel=1e-9)


# ---------------------------------------------------------------------------
# It reaches the report
# ---------------------------------------------------------------------------

def test_the_report_interval_follows_the_chosen_variance():
    iid = TBRMM(_cfg(_panel(rho=0.7))).fit()
    hac = TBRMM(_cfg(_panel(rho=0.7), variance="hac")).fit()

    wi = iid.report.inference.ci_upper - iid.report.inference.ci_lower
    wh = hac.report.inference.ci_upper - hac.report.inference.ci_lower
    assert wh > wi
    assert hac.report.inference.method == "tbr_posterior_hac"
    assert iid.report.inference.method == "tbr_posterior"


def test_every_market_gets_the_correction_too():
    res = TBRMM(_cfg(_panel(rho=0.7), variance="hac")).fit()
    for m in res.recommended.effect.market_effects:
        assert m.posterior.variance == "hac"
        assert m.posterior.bandwidth is not None
