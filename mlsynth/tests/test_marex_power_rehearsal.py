"""MAREX power by rehearsal (#947), and the pool's MDE baseline and horizon (#692).

The rehearsal prices a planned test by running it where nothing happened: placebo
designs on random subsets of the markets, launched ``horizon`` periods before the
real launch, each with the configured cumulative band on its pseudo-post periods.
A relative effect ``e`` shifts a design's total by ``e`` times its composite's
pseudo-post total, and the band shifts with it, so a placebo rejects at ``e``
exactly when its band, divided by that total, excludes ``-e``.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from mlsynth import MAREX
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.marex_helpers.inference import compute_inference
from mlsynth.utils.marex_helpers.pool import build_marex_pool
from mlsynth.utils.marex_helpers.rehearsal import rehearsal_mde, rehearsal_power
from mlsynth.utils.marex_helpers.structures import MAREXPowerRehearsal

N, T0, POST = 14, 60, 6
T = T0 + POST


def _panel(seed=0, level=100.0):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    common = np.sin(2 * np.pi * t / 26)
    region = np.repeat([0, 1], N // 2)
    shocks = np.cumsum(rng.normal(0, 0.02, (2, T)), axis=1) * 0.5
    Y = np.empty((N, T))
    for j in range(N):
        Y[j] = level * (1 + rng.uniform(0.05, 0.15) * common + shocks[region[j]]
                        + rng.normal(0, 0.01, T)) * rng.uniform(0.8, 1.2)
    return Y


def _frame(Y):
    units = [f"u{j:02d}" for j in range(N)]
    return pd.DataFrame({"unit": np.repeat(units, Y.shape[1]),
                         "t": np.tile(np.arange(Y.shape[1]), N), "y": Y.ravel()})


BASE = dict(outcome="y", unitid="unit", time="t", T0=T0, m_eq=2, design="penalized",
            lambda2=1e-3, backend="exact", display_graph=False)


def _fit(Y=None, **kw):
    Y = _panel() if Y is None else Y
    return MAREX(dict(df=_frame(Y), **{**BASE, **kw})).fit()


@pytest.fixture(scope="module")
def rehearsed():
    return _fit(power_placebos=10, power_seed=3)


# --------------------------------------------------------------------- the arithmetic

LOWER = np.array([-1.0, -2.0, -3.0, -4.0])
UPPER = np.array([1.0, 2.0, 3.0, 4.0])


def test_power_counts_the_placebos_whose_shifted_band_excludes_zero():
    # a drop of 2.5 shifts every band down by 2.5: [-3.5, -1.5] and [-4.5, -0.5]
    # exclude zero, [-5.5, 0.5] and [-6.5, 1.5] do not
    assert rehearsal_power(LOWER, UPPER, -2.5) == 0.5
    assert rehearsal_power(LOWER, UPPER, 2.5) == 0.5
    assert rehearsal_power(LOWER, UPPER, 0.0) == 0.0


def test_the_mde_is_the_smallest_effect_past_which_power_stays_on_target():
    assert rehearsal_mde(LOWER, UPPER, 0.75, "drop") == pytest.approx(3.0)
    assert rehearsal_mde(LOWER, UPPER, 0.75, "lift") == pytest.approx(3.0)
    assert rehearsal_power(LOWER, UPPER, -3.0) < 0.75 <= rehearsal_power(LOWER, UPPER, -3.0 - 1e-9)


def test_drop_and_lift_read_opposite_ends_of_the_band():
    lower, upper = np.array([-1.0, -1.0]), np.array([5.0, 5.0])
    assert rehearsal_mde(lower, upper, 0.5, "drop") == pytest.approx(5.0)
    assert rehearsal_mde(lower, upper, 0.5, "lift") == pytest.approx(1.0)


def test_a_band_that_excludes_zero_at_the_null_counts_as_a_rejection_there():
    lower, upper = np.array([-0.5, -0.5]), np.array([-0.1, 2.0])
    assert rehearsal_power(lower, upper, 0.0) == 0.5


def test_an_unbounded_band_never_rejects_and_can_leave_the_target_out_of_reach():
    lower = np.array([-1.0, -np.inf, -np.inf]); upper = np.array([1.0, np.inf, np.inf])
    assert rehearsal_power(lower, upper, -100.0) == pytest.approx(1 / 3)
    assert rehearsal_mde(lower, upper, 0.8, "drop") == np.inf


@pytest.mark.parametrize("lower, upper, target, direction, match", [
    (np.array([]), np.array([]), 0.8, "drop", "at least one placebo"),
    (np.array([1.0]), np.array([0.0]), 0.8, "drop", "lower end above"),
    (np.array([np.nan]), np.array([1.0]), 0.8, "drop", "NaN"),
    (LOWER, UPPER[:3], 0.8, "drop", "same length"),
    (LOWER, UPPER, 1.0, "drop", "strictly between 0 and 1"),
    (LOWER, UPPER, 0.8, "sideways", "'drop' or 'lift'"),
])
def test_malformed_bands_are_refused(lower, upper, target, direction, match):
    with pytest.raises(MlsynthDataError, match=match):
        rehearsal_mde(lower, upper, target, direction)


_bands = st.lists(st.tuples(st.floats(-50, 50), st.floats(0, 60)), min_size=1, max_size=25)


@settings(max_examples=150, deadline=None)
@given(_bands, st.floats(0.05, 0.95), st.sampled_from(["drop", "lift"]),
       st.floats(0, 200))
def test_property_power_holds_the_target_past_the_mde_and_misses_just_before(bands, target, direction, x):
    lower = np.array([a for a, _ in bands]); upper = lower + np.array([w for _, w in bands])
    mde = rehearsal_mde(lower, upper, target, direction)
    sign = -1.0 if direction == "drop" else 1.0
    assert np.isfinite(mde) and mde >= 0
    assert rehearsal_power(lower, upper, sign * (mde + 1e-9 + x)) >= target
    if mde > 0:
        # just below the MDE: the middle of the interval that ends at it
        ends = np.concatenate([[0.0], upper, lower, -upper, -lower])
        below = ends[(ends >= 0) & (ends < mde)].max()
        assert rehearsal_power(lower, upper, sign * (below + mde) / 2) < target


@settings(max_examples=100, deadline=None)
@given(_bands, st.floats(1.0, 100.0), st.sampled_from([-1.0, 1.0]))
def test_property_an_effect_past_every_band_is_detected_by_every_placebo(bands, margin, sign):
    lower = np.array([a for a, _ in bands]); upper = lower + np.array([w for _, w in bands])
    reach = float(np.max(np.abs(np.concatenate([lower, upper]))))
    assert rehearsal_power(lower, upper, sign * (reach + margin)) == 1.0


# ------------------------------------------------- the identity the method rests on

@pytest.mark.parametrize("method", ["inversion", "resample"])
def test_an_additive_effect_shifts_the_cumulative_band_by_its_total(method):
    Y = _panel(seed=4)
    w = np.zeros(N); w[:3] = 1 / 3
    v = np.zeros(N); v[3:] = 1 / (N - 3)
    yt, yc = w @ Y, v @ Y
    kw = dict(T0=T0, TcE=T0 - 25, Tb=25, cumulative_band=True, cumulative_method=method,
              control_outcomes=Y[3:])
    base = compute_inference(yt, yc, **kw).cumulative_band
    bump = np.zeros(T); bump[T0:] = 7.5
    moved = compute_inference(yt + bump, yc, **kw).cumulative_band
    assert moved.lower == pytest.approx(base.lower + 7.5 * POST, rel=1e-6, abs=1e-6)
    assert moved.upper == pytest.approx(base.upper + 7.5 * POST, rel=1e-6, abs=1e-6)


# ------------------------------------------------------------------ the estimator

def test_smoke_a_rehearsal_rides_on_the_result(rehearsed):
    r = rehearsed.power_rehearsal
    assert isinstance(r, MAREXPowerRehearsal)
    assert r.n_placebos == 10 and r.horizon == POST and r.n_units == round(0.85 * N)
    assert r.lower.shape == r.upper.shape == r.estimates.shape == (10,)
    assert np.all(r.lower <= r.estimates) and np.all(r.estimates <= r.upper)
    assert 0 < r.mde_drop < np.inf and 0 < r.mde_lift < np.inf
    assert r.cumulative_method == "resample" and r.alpha == 0.05 and r.power_target == 0.8


def test_each_placebo_is_the_design_resolved_on_its_markets_and_launched_early(rehearsed):
    # the first placebo, rebuilt by hand: the seed's first draw of markets, the
    # same design with the launch moved POST periods earlier, its cumulative band
    # over its own composite's pseudo-post total
    r = rehearsed.power_rehearsal
    Y = _panel()[:, :T0]
    sub = np.sort(np.random.default_rng(3).choice(N, r.n_units, replace=False))
    units = np.array([f"u{j:02d}" for j in range(N)])[sub]
    df = pd.DataFrame({"unit": np.repeat(units, T0), "t": np.tile(np.arange(T0), len(sub)),
                       "y": Y[sub].ravel()})
    g = MAREX(dict(df=df, **{**BASE, "T0": T0 - POST}, inference=True,
                   cumulative_band=True)).fit().globres
    band = g.inference.cumulative_band
    total = float(np.asarray(g.synthetic_treated)[T0 - POST:T0].sum())
    assert r.lower[0] == pytest.approx(band.lower / total)
    assert r.upper[0] == pytest.approx(band.upper / total)
    assert r.estimates[0] == pytest.approx(band.point / total)


def test_power_at_matches_the_arithmetic_and_meets_the_target_past_the_mde(rehearsed):
    r = rehearsed.power_rehearsal
    assert r.power_at(-r.mde_drop * 1.0001) >= r.power_target
    assert r.power_at(-0.3) == rehearsal_power(r.lower, r.upper, -0.3)


def test_no_rehearsal_unless_asked():
    assert _fit().power_rehearsal is None


def test_the_rehearsal_reads_only_the_weeks_before_launch(rehearsed):
    Y = _panel(); Y[:, T0:] *= 1.7
    other = _fit(Y, power_placebos=10, power_seed=3).power_rehearsal
    ref = rehearsed.power_rehearsal
    np.testing.assert_array_equal(other.lower, ref.lower)
    np.testing.assert_array_equal(other.upper, ref.upper)


def test_the_seed_draws_the_placebo_markets(rehearsed):
    other = _fit(power_placebos=10, power_seed=4).power_rehearsal
    assert not np.array_equal(other.lower, rehearsed.power_rehearsal.lower)


def test_the_rehearsal_runs_the_configured_band():
    r = _fit(power_placebos=10, inference=True, cumulative_band=True,
             cumulative_method="inversion").power_rehearsal
    assert r.cumulative_method == "inversion"


def test_a_pure_design_needs_a_horizon_and_takes_one():
    Y = _panel()[:, :T0]
    df = _frame(Y)
    with pytest.raises(MlsynthConfigError, match="power_horizon"):
        MAREX(dict(df=df, **BASE, power_placebos=10)).fit()
    r = MAREX(dict(df=df, **BASE, power_placebos=10, power_horizon=8)).fit().power_rehearsal
    assert r.horizon == 8


@pytest.mark.parametrize("kw, match", [
    (dict(power_placebos=5), "at least 10"),
    (dict(power_placebos=10, power_share=1.0), "power_share"),
    (dict(power_placebos=10, power_share=0.0), "power_share"),
    (dict(power_horizon=4), "power_placebos"),
    (dict(power_share=0.5), "power_placebos"),
    (dict(power_seed=2), "power_placebos"),
    (dict(power_placebos=10, power_horizon=0), "power_horizon"),
])
def test_config_refusals(kw, match):
    with pytest.raises(MlsynthConfigError, match=match):
        MAREX(dict(df=_frame(_panel()), **BASE, **kw))


def test_too_few_markets_per_placebo_is_refused():
    with pytest.raises(MlsynthConfigError, match="markets per placebo"):
        _fit(power_placebos=10, power_share=0.2)


def test_a_share_that_rounds_to_every_market_is_refused():
    # 0.97 of 14 markets rounds to 14: every placebo would be the same design
    with pytest.raises(MlsynthConfigError, match="draws all"):
        _fit(power_placebos=10, power_share=0.97)


def test_the_real_blank_window_is_rehearsed_and_must_fit_before_the_pseudo_launch():
    r = _fit(power_placebos=10, inference=True, blank_periods=30).power_rehearsal
    assert r.horizon == POST
    with pytest.raises(MlsynthConfigError, match="pseudo-launch"):
        _fit(power_placebos=10, inference=True, blank_periods=45)


def test_a_horizon_that_leaves_no_room_to_fit_is_refused():
    with pytest.raises(MlsynthConfigError, match="pseudo-launch"):
        _fit(power_placebos=10, power_horizon=T0 - 5)


def test_a_relative_effect_needs_a_positive_composite():
    with pytest.raises(MlsynthDataError, match="positive"):
        _fit(_panel() - 500.0, power_placebos=10)


# ------------------------------------------------------- the pool's MDE (#692, #947)

def _raw(w, v, Y_fit, Y_blank):
    return dict(df=pd.DataFrame(index=[f"u{j}" for j in range(len(w))]), w_agg=w, v_agg=v,
                z_opt=(np.asarray(w) > 0)[:, None].astype(float), Y_fit=Y_fit, Y_blank=Y_blank,
                objective=1.0)


def test_pool_baseline_is_the_designs_own_synthetic_treated_level():
    rng = np.random.default_rng(0)
    lev = np.array([100.0, 110.0, 120.0, 1000.0])
    Y = lev[:, None] * (1 + 0.02 * rng.normal(size=(4, 40)))
    w = np.array([0.0, 0.0, 0.9, 0.1]); v = np.array([0.5, 0.5, 0.0, 0.0])
    entry = build_marex_pool([_raw(w, v, Y[:, :30], Y[:, 30:])], n_post=6)[0]
    assert entry["power_curve"]["baseline"] == pytest.approx(float((w @ Y).mean()))


def test_pool_scores_a_test_longer_than_twelve_periods_at_its_own_length():
    rng = np.random.default_rng(1)
    Y = 100 * (1 + 0.02 * rng.normal(size=(4, 60)))
    w = np.array([0.5, 0.5, 0.0, 0.0]); v = np.array([0.0, 0.0, 0.5, 0.5])
    entry = build_marex_pool([_raw(w, v, Y[:, :40], Y[:, 40:])], n_post=15)[0]
    curve = entry["power_curve"]
    assert int(curve["horizons"][-1]) >= 15
    assert entry["mde_pct"] == pytest.approx(float(curve["mde_pct"][14]))
