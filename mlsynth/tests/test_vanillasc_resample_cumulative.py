"""VanillaSC's cumulative band, built by resampling instead of a split quantile.

``inference="conformal_cumulative"`` calibrates the band for a running total on
non-overlapping rolling-origin windows and takes a split-conformal order
statistic of the ``m`` window sums. That statistic is the constraint: it exists
only once ``m >= ceil(1/alpha) - 1``, so at the 5 percent level a panel needs
nineteen windows before the band is finite at all, and below that the estimator
correctly returns an infinite half-width and warns.

``conformal_method="resample"`` calibrates on the same windows but keeps them per
period, giving ``m * L`` values to draw from where the split band had ``m``. A band
then exists on pre-periods where the order statistic does not, which is the
practical reason to offer it.

What it does not change is the object: the same
:class:`~mlsynth.utils.conformal.structure.CumulativeConformalBand`, the same
point estimate, the same fields, only calibrated differently.

Levels: smoke, unit invariants, edge, failure.
"""
import numpy as np
import pandas as pd
import pytest

from mlsynth import VanillaSC
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError


def _panel(n_units=8, n_periods=40, t0=30, seed=0, effect=4.0, rho=0.0):
    rng = np.random.default_rng(seed)
    donor_base = rng.normal(10.0, 1.0, size=n_units - 1)
    loads = rng.dirichlet(np.ones(n_units - 1))
    common_e = rng.normal(0.0, 0.2, size=n_periods)
    # The treated unit needs its own idiosyncratic error, or it is an exact
    # convex combination of the donors and every calibration error is 1e-15.
    own_e = rng.normal(0.0, 0.5, size=n_periods)
    if rho:
        for t in range(1, n_periods):
            own_e[t] = rho * own_e[t - 1] + np.sqrt(1 - rho ** 2) * own_e[t]
    rows = []
    for t in range(n_periods):
        common = 0.4 * t + common_e[t]
        donors = donor_base + common + rng.normal(0.0, 0.15, size=donor_base.size)
        treated = float(loads @ donors) + own_e[t] + (effect if t >= t0 else 0.0)
        rows.append({"unit": "u0", "time": t, "y": treated, "treat": int(t >= t0)})
        for j, dv in enumerate(donors):
            rows.append({"unit": f"d{j}", "time": t, "y": float(dv), "treat": 0})
    return pd.DataFrame(rows)


_CFG = dict(outcome="y", treat="treat", unitid="unit", time="time",
            backend="outcome-only", display_graphs=False,
            inference="conformal_cumulative")


def _inference(df=None, **kw):
    cfg = dict(df=_panel() if df is None else df, **_CFG)
    cfg.update(kw)
    return VanillaSC(cfg).fit().inference


def _details(df=None, **kw):
    return _inference(df, **kw).details


# --------------------------------------------------------------------------- #
# smoke
# --------------------------------------------------------------------------- #
def test_it_produces_a_finite_band():
    d = _details(conformal_method="resample", conformal_n_sim=400)
    assert np.isfinite(d["conformal_q"])
    assert np.isfinite(d["cumulative_lower"]) and np.isfinite(d["cumulative_upper"])


# --------------------------------------------------------------------------- #
# unit invariants
# --------------------------------------------------------------------------- #
def test_the_construction_is_recorded():
    assert "resample" in _inference(conformal_method="resample",
                                    conformal_n_sim=400).method
    assert "resample" not in _inference().method


def test_the_point_estimate_does_not_depend_on_the_construction():
    """Only the calibration changes; the effect being bounded is the same."""
    split = _details()
    resampled = _details(conformal_method="resample", conformal_n_sim=400)
    assert resampled["cumulative_effect"] == pytest.approx(split["cumulative_effect"])
    assert resampled["horizon"] == split["horizon"]


def test_the_band_is_symmetric_about_the_point():
    d = _details(conformal_method="resample", conformal_n_sim=400)
    assert d["cumulative_effect"] - d["cumulative_lower"] == pytest.approx(
        d["conformal_q"])
    assert d["cumulative_upper"] - d["cumulative_effect"] == pytest.approx(
        d["conformal_q"])


def test_the_same_seed_gives_the_same_band():
    a = _details(conformal_method="resample", conformal_n_sim=400, conformal_seed=7)
    b = _details(conformal_method="resample", conformal_n_sim=400, conformal_seed=7)
    assert a["conformal_q"] == pytest.approx(b["conformal_q"])


def test_a_longer_block_widens_the_band_on_a_persistent_panel():
    df = _panel(rho=0.8, seed=5)
    one = _details(df=df, conformal_method="resample", conformal_block=1,
                   conformal_n_sim=4000)
    full = _details(df=df, conformal_method="resample", conformal_block=0,
                    conformal_n_sim=4000)
    assert full["conformal_q"] > one["conformal_q"]


def test_it_counts_the_periods_it_drew_from_not_the_windows():
    """m windows supply m*L values, which is the whole point of the change."""
    d = _details(conformal_method="resample", conformal_n_sim=400)
    split = _details()
    assert d["n_calibration_windows"] == split["n_calibration_windows"] * d["horizon"]


# --------------------------------------------------------------------------- #
# edge
# --------------------------------------------------------------------------- #
def test_it_is_finite_where_the_split_order_statistic_is_not():
    """The practical gain, on a pre-period too short for the split band.

    The block has to be short enough to draw from: one window supplies exactly
    ``horizon`` periods, and a whole-horizon block over a series that length is
    refused (see below). At ``block=1`` the m * L values are there and the band
    is real."""
    df = _panel(n_periods=22, t0=16, seed=2)
    split = _details(df=df, alpha=0.05)
    assert not np.isfinite(split["conformal_q"])
    resampled = _details(df=df, alpha=0.05, conformal_method="resample",
                         conformal_block=1, conformal_n_sim=400)
    assert np.isfinite(resampled["conformal_q"])
    assert resampled["conformal_q"] > 0.0


def test_a_block_as_long_as_the_calibration_series_is_refused():
    """One window supplies exactly ``horizon`` periods, so the default
    whole-horizon block has nothing to slide over: every circular block is a
    rotation of the whole series and every path sums to the same value. The band
    would report zero width, so the draw raises instead of returning it."""
    df = _panel(n_periods=22, t0=16, seed=2)
    with pytest.raises(MlsynthDataError, match="zero width|shorter block"):
        _details(df=df, alpha=0.05, conformal_method="resample",
                 conformal_n_sim=400)


# --------------------------------------------------------------------------- #
# failure
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", ["wheeler", "", "Resample", None, 1])
def test_an_unknown_construction_is_refused(bad):
    with pytest.raises(MlsynthConfigError, match="conformal_method"):
        _details(conformal_method=bad)


@pytest.mark.parametrize("bad", [-1, 2.5, "3", True])
def test_a_bad_block_is_refused(bad):
    with pytest.raises(MlsynthConfigError, match="conformal_block"):
        _details(conformal_method="resample", conformal_block=bad)


@pytest.mark.parametrize("bad", [0, 1, 2.5, "400"])
def test_a_bad_simulation_count_is_refused(bad):
    with pytest.raises(MlsynthConfigError, match="conformal_n_sim"):
        _details(conformal_method="resample", conformal_n_sim=bad)


def test_different_seeds_give_different_bands():
    """The draw is random, so the seed has to reach it. Single-period blocks,
    because at ``block=0`` a path is one contiguous window of the calibration
    series and there are only as many of those as there are periods, so the
    quantile lands on the same atom whatever the seed."""
    kw = dict(conformal_method="resample", conformal_block=1, conformal_n_sim=200)
    a = _details(conformal_seed=7, **kw)
    b = _details(conformal_seed=8, **kw)
    assert a["conformal_q"] != pytest.approx(b["conformal_q"], rel=1e-6)
    assert _details(conformal_seed=7, **kw)["conformal_q"] == pytest.approx(
        a["conformal_q"])


@pytest.mark.parametrize("alpha", [0.9, 0.5, 0.2, 0.05])
def test_the_half_width_is_a_magnitude_at_every_level(alpha):
    """The draws are sign-symmetric, so the (1-alpha) quantile of the signed
    totals goes negative once alpha passes a half. A half-width cannot: it is
    the quantile of the magnitudes, and the band brackets the point."""
    d = _details(conformal_method="resample", conformal_n_sim=400, alpha=alpha)
    assert d["conformal_q"] > 0.0
    assert d["cumulative_lower"] < d["cumulative_effect"] < d["cumulative_upper"]


def test_a_shorter_horizon_accumulates_only_that_horizon():
    df = _panel(n_periods=40, t0=30, effect=4.0)
    full = _details(df=df, conformal_method="resample", conformal_n_sim=400)
    part = _details(df=df, conformal_method="resample", conformal_n_sim=400,
                    conformal_horizon=4)
    assert full["horizon"] == 10 and part["horizon"] == 4
    # A constant post-period effect: the total over four periods is about four
    # fifths less than the total over ten.
    assert part["cumulative_effect"] == pytest.approx(
        full["cumulative_effect"] * 0.4, rel=0.25)
    assert part["spans_post_period"] is False
