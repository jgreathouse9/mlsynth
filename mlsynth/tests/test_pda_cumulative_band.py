"""A cumulative band for PDA, built from the bootstrap's own replicate paths.

PDA reports a per-period prediction interval for every post-treatment period
(Jiang, Li, Shen & Zhou 2025) and an asymptotic interval for the mean effect, but
nothing for the running total -- which is the number a reader of an event study
actually quotes ("the programme cost N lives over the first year"). PPSCM has had
``cumulative_band`` since #439; this is the same object for PDA, built the same
way and sharing the same sup-t machinery, so the two estimators cannot drift
apart in how they define it.

The construction is forced by what goes wrong otherwise. An interval for a
running total is not the running total of the period intervals: adding endpoints
treats every period's error as moving in lockstep, so the width grows like ``L``
whatever the data does, and on the Oregon opioid panel that inflates a
half-width of 0.9 to 3.8 against a point estimate of 7.6. Rescaling a single
period's interval by ``sqrt(L)`` makes the opposite assumption and is wrong
whenever the estimation error is persistent, which for a donor fit it is.

So the replicate paths are accumulated *first* and the standard error is taken
after. Whatever correlation the period errors have is then carried into the
band rather than assumed into it. The bootstrap supplies those paths already:
each draw's post-period prediction error is a path over horizons, and
``pda_prediction_intervals`` keeps them.

The band is simultaneous over horizons for the reason ``supt.py`` exists -- a
cumulative path is read as a path, and a pointwise band read that way covers at
much less than its nominal level.

Levels: smoke, unit invariants, edge, failure.
"""
import numpy as np
import pandas as pd
import pytest

from mlsynth import PDA
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.pda_helpers.inference import cumulative_supt_band


def _panel(n_donors=8, T0=30, T1=6, effect=1.0, seed=0, rho=0.0):
    """Factor panel with one treated unit adopting at ``T0``."""
    rng = np.random.default_rng(seed)
    T = T0 + T1
    f = rng.standard_normal((T, 2))
    load_d = rng.standard_normal((n_donors, 2)) * 0.5
    e = rng.standard_normal((n_donors + 1, T)) * 0.3
    if rho:
        for t in range(1, T):
            e[:, t] = rho * e[:, t - 1] + np.sqrt(1 - rho ** 2) * e[:, t]
    rec = []
    y = f @ load_d.mean(axis=0) + e[0]
    y[T0:] += effect
    rec += [{"unit": "treated", "t": t, "y": float(y[t]), "d": int(t >= T0)}
            for t in range(T)]
    for j in range(n_donors):
        s = f @ load_d[j] + e[j + 1]
        rec += [{"unit": f"d{j}", "t": t, "y": float(s[t]), "d": 0} for t in range(T)]
    return pd.DataFrame(rec)


def _fit(df=None, **kw):
    cfg = dict(df=_panel() if df is None else df, outcome="y", treat="d",
               unitid="unit", time="t", method="LASSO", display_graphs=False)
    cfg.update(kw)
    return PDA(cfg).fit()


def _band(res, method="lasso"):
    fit = res.fits[method] if isinstance(res.fits, dict) else res.fits[0]
    return fit.cumulative_band


# --------------------------------------------------------------------------- #
# smoke                                                                        #
# --------------------------------------------------------------------------- #
def test_asking_for_it_attaches_one_entry_per_post_period():
    b = _band(_fit(prediction_intervals=True, cumulative_band=True, pi_n_boot=200))
    assert b is not None
    for arr in (b.horizons, b.point, b.lower, b.upper, b.se):
        assert np.asarray(arr).shape == (6,)
    assert np.isfinite(np.asarray(b.point)).all()


def test_it_is_off_unless_asked_for():
    """Nothing changes for a caller who did not request it."""
    assert _band(_fit(prediction_intervals=True, pi_n_boot=200)) is None
    assert _band(_fit()) is None


# --------------------------------------------------------------------------- #
# unit invariants                                                              #
# --------------------------------------------------------------------------- #
def test_the_point_is_the_running_total_of_the_period_effects():
    res = _fit(prediction_intervals=True, cumulative_band=True, pi_n_boot=200)
    fit = res.fits["lasso"] if isinstance(res.fits, dict) else res.fits[0]
    gap = np.asarray(fit.gap, dtype=float)[-6:]
    np.testing.assert_allclose(_band(res).point, np.cumsum(gap), rtol=1e-10)


def test_the_band_brackets_its_point_at_every_horizon():
    b = _band(_fit(prediction_intervals=True, cumulative_band=True, pi_n_boot=200))
    assert (np.asarray(b.lower) <= np.asarray(b.point)).all()
    assert (np.asarray(b.point) <= np.asarray(b.upper)).all()


def test_horizons_are_one_through_H():
    b = _band(_fit(prediction_intervals=True, cumulative_band=True, pi_n_boot=200))
    np.testing.assert_array_equal(b.horizons, np.arange(1, 7))


def test_it_is_narrower_than_summing_the_period_interval_endpoints():
    """The failure mode the band exists to avoid.

    Adding per-period endpoints asserts perfectly correlated period errors. It is
    the widest thing anyone builds and it is never right unless the errors really
    do move in lockstep, which the replicates would then show.
    """
    res = _fit(prediction_intervals=True, cumulative_band=True, pi_n_boot=400)
    fit = res.fits["lasso"] if isinstance(res.fits, dict) else res.fits[0]
    e = fit.prediction_intervals["effect"]
    naive = np.cumsum((np.asarray(e["sy_upper"]) - np.asarray(e["sy_lower"])) / 2.0)
    b = _band(res)
    half = (np.asarray(b.upper) - np.asarray(b.lower)) / 2.0
    assert half[-1] < naive[-1]


def test_the_growth_is_measured_not_assumed():
    """The builder's whole purpose, tested where it can be stated exactly.

    Independent period errors accumulate like ``sqrt(L)``; errors that move
    together accumulate like ``L``. The same code must return both, because it
    reads the correlation off the paths rather than choosing one.
    """
    rng = np.random.default_rng(0)
    L, B = 8, 4000
    point = np.zeros(L)
    indep = rng.standard_normal((B, L))
    lock = np.repeat(rng.standard_normal((B, 1)), L, axis=1)
    se_i = np.asarray(cumulative_supt_band(point, indep, alpha=0.05).se)
    se_l = np.asarray(cumulative_supt_band(point, lock, alpha=0.05).se)
    assert se_i[-1] / se_i[0] == pytest.approx(np.sqrt(L), rel=0.10)
    assert se_l[-1] / se_l[0] == pytest.approx(L, rel=0.10)


def test_the_critical_value_is_simultaneous_not_pointwise():
    """One shared multiplier covering every horizon at once must exceed the
    pointwise normal quantile, and must grow as more horizons are covered."""
    from scipy.stats import norm
    rng = np.random.default_rng(1)
    small = cumulative_supt_band(np.zeros(3), rng.standard_normal((3000, 3)), alpha=0.05)
    big = cumulative_supt_band(np.zeros(12), rng.standard_normal((3000, 12)), alpha=0.05)
    assert small.critical_value > float(norm.ppf(0.975))
    assert big.critical_value > small.critical_value


def test_a_tighter_level_never_narrows_the_band():
    rng = np.random.default_rng(2)
    P = rng.standard_normal((3000, 6))
    tight = cumulative_supt_band(np.zeros(6), P, alpha=0.01)
    loose = cumulative_supt_band(np.zeros(6), P, alpha=0.20)
    assert tight.critical_value > loose.critical_value


def test_it_records_what_produced_it():
    b = _band(_fit(prediction_intervals=True, cumulative_band=True, pi_n_boot=200))
    assert b.alpha == pytest.approx(0.05)
    assert b.method == "bootstrap"
    assert b.n_replicates >= 2


def test_the_band_leaves_the_rest_of_the_fit_untouched():
    """It is additional. The ATT and the per-period intervals must not move."""
    base = _fit(prediction_intervals=True, pi_n_boot=200, pi_seed=0)
    with_b = _fit(prediction_intervals=True, cumulative_band=True, pi_n_boot=200, pi_seed=0)
    fb = base.fits["lasso"] if isinstance(base.fits, dict) else base.fits[0]
    fw = with_b.fits["lasso"] if isinstance(with_b.fits, dict) else with_b.fits[0]
    assert fw.att == pytest.approx(fb.att)
    assert fw.att_se == pytest.approx(fb.att_se)
    np.testing.assert_allclose(fw.counterfactual, fb.counterfactual)
    np.testing.assert_allclose(fw.prediction_intervals["effect"]["eq_lower"],
                               fb.prediction_intervals["effect"]["eq_lower"])


def test_the_bootstrap_hands_back_one_error_path_per_usable_draw():
    """The paths the band is built from, pinned at their source."""
    res = _fit(prediction_intervals=True, cumulative_band=True, pi_n_boot=200)
    fit = res.fits["lasso"] if isinstance(res.fits, dict) else res.fits[0]
    pi = fit.prediction_intervals
    paths = np.asarray(pi["error_paths"], dtype=float)
    assert paths.shape == (pi["n_boot_effective"], pi["post_periods"])
    assert np.isfinite(paths).all()


# --------------------------------------------------------------------------- #
# edge                                                                         #
# --------------------------------------------------------------------------- #
def test_a_single_post_period_is_a_band_of_one():
    b = _band(_fit(df=_panel(T1=1), prediction_intervals=True,
                   cumulative_band=True, pi_n_boot=200))
    assert np.asarray(b.point).shape == (1,)
    assert b.lower[0] <= b.point[0] <= b.upper[0]


def test_serially_correlated_errors_widen_it_relative_to_independent_ones():
    """Not a fixed number -- a direction. Persistent errors accumulate faster."""
    kw = dict(prediction_intervals=True, cumulative_band=True, pi_n_boot=300, pi_seed=0)
    flat = _band(_fit(df=_panel(rho=0.0, seed=3), **kw))
    slow = _band(_fit(df=_panel(rho=0.9, seed=3), **kw))
    ratio = lambda b: np.asarray(b.se)[-1] / np.asarray(b.se)[0]
    assert ratio(slow) > ratio(flat)


# --------------------------------------------------------------------------- #
# failure -- reported, not swallowed                                           #
# --------------------------------------------------------------------------- #
def test_asking_for_the_band_without_the_bootstrap_is_refused_by_name():
    """The band is built from the bootstrap's replicate paths. Without
    ``prediction_intervals`` there are none, and silently producing nothing would
    leave a caller reading a missing field as an absent effect."""
    with pytest.raises(MlsynthConfigError, match="cumulative_band"):
        PDA(dict(df=_panel(), outcome="y", treat="d", unitid="unit", time="t",
                 method="LASSO", cumulative_band=True, display_graphs=False))


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_an_invalid_level_is_refused(bad):
    rng = np.random.default_rng(4)
    with pytest.raises(MlsynthConfigError):
        cumulative_supt_band(np.zeros(4), rng.standard_normal((100, 4)), alpha=bad)


def test_mismatched_path_length_is_refused():
    rng = np.random.default_rng(5)
    with pytest.raises(MlsynthDataError):
        cumulative_supt_band(np.zeros(4), rng.standard_normal((100, 7)), alpha=0.05)


def test_too_few_replicates_is_refused():
    with pytest.raises(MlsynthDataError):
        cumulative_supt_band(np.zeros(3), np.ones((1, 3)), alpha=0.05)


def test_a_replicate_with_a_hole_is_dropped_not_counted_as_zero():
    """A refit that failed must be absent, not silently contribute a zero path
    that shrinks the standard error toward a band nobody earned."""
    rng = np.random.default_rng(6)
    P = rng.standard_normal((50, 4))
    P[7, 2] = np.nan
    b = cumulative_supt_band(np.zeros(4), P, alpha=0.05)
    assert b.n_replicates == 49
