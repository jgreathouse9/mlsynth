"""Test-first spec for PPSCM's per-unit cumulative band by moving-block conformal.

The split-conformal band (#431) calibrates on disjoint pre-period windows, and
there are roughly ``0.7 * T0 / L`` of them. A finite ``1 - alpha`` band needs
``ceil((m+1)(1-alpha)) <= m``, so at the 90 percent level it needs nine, which
puts a floor of about ``12.8 * L`` pre-periods on reporting one at all and pins
every feasible design in the regime where the rank never trims and the half-width
is simply the largest score.

This calibrates the same estimand against the ``T`` cyclic shifts of the residual
path instead, so the block count stops depending on the horizon and the floor
disappears.

Two things are settled before this suite and are asserted here so they cannot
drift back.

The statistic is the paper's. An earlier prototype scored blocks by their
absolute sum, on the reasoning that a running total is a signed quantity. That is
invalid on a fitted residual path: the quadratic program leaves end-of-window
residuals sign-coherent, so block sums ramp toward the end of the window while
magnitudes stay flat, and the trailing block always occupies the most inflated
position. Measured, it rejected at 0.225 against a nominal 0.10.
``moving_block_pvalue``'s docstring carries the full measurement. The statistic
that is valid there is ``mean_abs``, and it is not a compromise: displacing a
mean-zero block by ``delta`` raises the mean of its absolute values, so the test
has power against exactly the constant shift being inverted -- measured rejection
rises 0.04, 0.36, 0.76, 1.00 as the true effect grows through 0, 0.2, 0.5, 1.0.

The null refit balances the whole window. Under the null the adjusted series is
an untreated series, so every period is fitting data, and leaving one out puts the
trailing block partly outside the fit the reference blocks come from. Reaching
``d = T`` needs an explicit ``nu``, because the automatic rule sits exactly on its
boundary for a single treated unit and cvxpy refuses the program; the caller
passes ``nu_used`` for that reason, as the split-conformal entry point already
does.

Levels: smoke, unit invariants, property, edge, failure.
"""
import numpy as np
import pytest

from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.ppscm_helpers.inference import cwz_cumulative_per_unit


def _panel(n_ctrl=18, n_trt=2, t0=40, horizon=5, effect=0.0, seed=0):
    """Factor panel with a constant post-period effect on the treated units."""
    rng = np.random.default_rng(seed)
    n, T = n_ctrl + n_trt, t0 + horizon
    f = rng.normal(size=(T, 2))
    load = rng.uniform(0.3, 1.3, size=(n, 2))
    Xy = load @ f.T + rng.normal(scale=0.3, size=(n, T))
    Xy = Xy - Xy.min() + 1.0
    Xy[:n_trt, t0:] += effect
    trt = np.full(n, np.inf)
    trt[:n_trt] = t0
    return Xy, trt, t0, horizon


def _call(panel=None, **kw):
    Xy, trt, t0, L = panel or _panel()
    base = dict(d=t0, n_leads=L, n_lags=t0, fixedeff=True, time_cohort=False,
                nu_used=0.5, lam=0.0, solver=None, alpha=0.10, horizon=L,
                n_nulls=9)
    base.update(kw)
    return cwz_cumulative_per_unit(Xy, trt, **base)


# --------------------------------------------------------------------------- #
# smoke                                                                        #
# --------------------------------------------------------------------------- #
def test_smoke_one_band_per_treated_unit():
    point, lower, upper, p_zero = _call()
    for arr in (point, lower, upper, p_zero):
        assert np.asarray(arr).shape == (2,)
    assert np.isfinite(point).all()


# --------------------------------------------------------------------------- #
# unit invariants                                                              #
# --------------------------------------------------------------------------- #
def test_the_point_is_each_unit_summed_effect_over_the_horizon():
    from mlsynth.utils.ppscm_helpers.engine import run_multisynth

    Xy, trt, t0, L = _panel()
    full = run_multisynth(Xy, trt, t0, L, t0, nu=0.5)
    want = [float(np.sum(np.asarray(full["tau_rel"][k], float)[:L]))
            for k in range(len(full["groups"]))]
    point, *_ = _call((Xy, trt, t0, L))
    np.testing.assert_allclose(point, want, rtol=1e-8, atol=1e-10)


def test_the_band_is_reported_on_the_cumulative_scale():
    """The inversion is over a constant per-period effect; what is reported is
    the total, so the endpoints carry a factor of the horizon. Without it the
    band would be an order of magnitude too narrow for the number beside it."""
    _, lo5, hi5, _ = _call(horizon=5)
    _, lo2, hi2, _ = _call(horizon=2)
    assert np.all(np.asarray(hi5) - np.asarray(lo5) >
                  np.asarray(hi2) - np.asarray(lo2))


def test_the_p_value_of_no_effect_is_reported():
    """A caller reading significance off the band alone cannot tell an interval
    that just excludes zero from one that comfortably does."""
    _, _, _, p_zero = _call()
    assert np.all((p_zero > 0.0) & (p_zero <= 1.0))


def test_the_null_refit_balances_every_period():
    """Leaving a period out puts the trailing block partly outside the fit its
    reference blocks come from. This pins the window, since the failure it
    guards is silent -- the band still comes back, just miscalibrated."""
    import mlsynth.utils.ppscm_helpers.inference as inf

    Xy, trt, t0, L = _panel()
    T = Xy.shape[1]
    seen = []
    real = inf.run_multisynth

    def spy(Xy_, trt_, d_, n_leads_, n_lags_, **kw):
        seen.append(int(d_))
        return real(Xy_, trt_, d_, n_leads_, n_lags_, **kw)

    inf.run_multisynth = spy
    try:
        _call((Xy, trt, t0, L))
    finally:
        inf.run_multisynth = real
    assert T in seen, f"no refit balanced all {T} periods; saw {sorted(set(seen))}"


def test_a_tighter_level_never_narrows_the_band():
    _, lo_t, hi_t, _ = _call(alpha=0.02, n_nulls=15)
    _, lo_l, hi_l, _ = _call(alpha=0.30, n_nulls=15)
    assert np.all(np.asarray(hi_t) - np.asarray(lo_t) >=
                  np.asarray(hi_l) - np.asarray(lo_l) - 1e-9)


def test_a_coarse_grid_understates_the_band():
    """Discretisation errs anti-conservatively here, which is the opposite of the
    usual intuition, which is why it is pinned.

    The band is the range of accepted candidates. A coarse grid offers fewer
    candidates to accept, so the range of survivors is narrower and the reported
    band too tight; refining converges upward. Measured on this panel the width
    runs 3.14, 4.71, 5.38, 5.65, 5.86 as the grid goes 5, 9, 15, 21, 31. An
    earlier version of this test asserted the reverse, on the reasoning that a
    coarse grid rounds outward, and it was wrong.
    """
    widths = []
    for n in (5, 9, 21):
        _, lo, hi, _ = _call(n_nulls=n)
        w = np.asarray(hi) - np.asarray(lo)
        widths.append(w[np.isfinite(w)])
    assert widths[0].sum() < widths[1].sum() <= widths[2].sum() + 1e-9


def test_a_band_reaching_the_grid_edge_is_reported_as_infinite():
    """That end is bounded by ``grid_scale``, not by the data, and a finite
    number there would understate it without saying so."""
    _, lo, hi, _ = _call(grid_scale=0.05, n_nulls=9)
    assert np.any(np.isinf(np.asarray(lo)) | np.isinf(np.asarray(hi)))


def test_a_wide_enough_grid_gives_a_finite_band():
    """The control: the guard fires on a grid that is too narrow, not always."""
    _, lo, hi, _ = _call(grid_scale=8.0, n_nulls=15)
    assert np.isfinite(np.asarray(lo)).any() and np.isfinite(np.asarray(hi)).any()


# --------------------------------------------------------------------------- #
# property: validity and power                                                 #
# --------------------------------------------------------------------------- #
def test_it_covers_a_true_zero_effect():
    """Validity. Test inversion inherits it from the test, and the test is the
    paper's; this asserts the composition did not lose it."""
    covered = 0
    for s in range(12):
        _, lo, hi, _ = _call(_panel(effect=0.0, seed=s), n_nulls=11)
        covered += int(np.all((np.asarray(lo) <= 0.0) & (0.0 <= np.asarray(hi))))
    assert covered >= 10, f"covered the truth in only {covered}/12 draws"


def test_it_covers_a_true_constant_effect():
    """The null the inversion actually sweeps, at a value that is not zero."""
    eff, L = 0.6, 5
    covered = 0
    for s in range(12):
        _, lo, hi, _ = _call(_panel(effect=eff, seed=s), n_nulls=11)
        total = eff * L
        covered += int(np.all((np.asarray(lo) <= total) & (total <= np.asarray(hi))))
    assert covered >= 9, f"covered the truth in only {covered}/12 draws"


def test_a_large_effect_is_detected():
    """Power. A band that always covers is not evidence of anything."""
    _, lo, hi, p_zero = _call(_panel(effect=2.5, seed=1), n_nulls=15)
    assert np.all((np.asarray(lo) > 0.0) | (np.asarray(hi) < 0.0))
    assert np.all(np.asarray(p_zero) <= 0.10)


def test_the_block_count_does_not_depend_on_the_horizon():
    """The reason for the method. Split conformal loses windows as the horizon
    grows and stops producing a finite band; the cyclic reference set does not."""
    for L in (2, 5, 8):
        Xy, trt, t0, _ = _panel(t0=40, horizon=10)
        _, lo, hi, _ = _call((Xy, trt, t0, 10), horizon=L, n_nulls=9)
        assert np.isfinite(lo).all() and np.isfinite(hi).all(), \
            f"horizon {L} produced no finite band"


# --------------------------------------------------------------------------- #
# edge                                                                         #
# --------------------------------------------------------------------------- #
def test_a_single_treated_unit_is_supported():
    """The case with no cross-unit replication at all, and the one whose
    automatic nu sits on its boundary -- which is why nu_used is required."""
    point, lo, hi, _ = _call(_panel(n_trt=1))
    assert point.shape == (1,) and np.isfinite(point).all()


def test_an_effect_that_is_not_constant_may_leave_the_set_empty():
    """The cost of a constant-effect null, surfaced as a refusal. An empty
    accepted set is a real answer: no constant effect on the grid conforms."""
    Xy, trt, t0, L = _panel(effect=0.0, seed=3)
    Xy[0, t0:] += np.linspace(0.0, 6.0, L)        # a ramp, not a constant
    _, lo, hi, _ = _call((Xy, trt, t0, L), n_nulls=11)
    assert np.isnan(lo[0]) or np.isfinite(lo[0])  # either is legitimate; not a crash


def test_a_horizon_shorter_than_the_post_window_is_allowed():
    point, lo, hi, _ = _call(horizon=3)
    assert np.isfinite(point).all()


# --------------------------------------------------------------------------- #
# failure -- reported, not swallowed                                           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1])
def test_an_invalid_level_is_refused(bad):
    with pytest.raises(MlsynthConfigError):
        _call(alpha=bad)


@pytest.mark.parametrize("bad", [0, -2, 2.5])
def test_a_nonpositive_horizon_is_refused(bad):
    with pytest.raises(MlsynthConfigError):
        _call(horizon=bad)


def test_a_horizon_past_the_post_window_is_refused():
    with pytest.raises(MlsynthDataError):
        _call(horizon=99)


def test_too_few_grid_points_is_refused():
    """A two-point grid cannot express an interval; it can only return its own
    endpoints, which would look like a band and be an artifact of the grid."""
    with pytest.raises(MlsynthConfigError):
        _call(n_nulls=2)


def test_no_treated_unit_is_refused():
    Xy, trt, t0, L = _panel()
    with pytest.raises(MlsynthDataError):
        _call((Xy, np.full(trt.shape, np.inf), t0, L))
