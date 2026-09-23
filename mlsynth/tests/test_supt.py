"""Simultaneous (sup-t) bands from resampling draws.

A pointwise band covers each horizon with probability ``1 - alpha`` one at a
time. Read across a whole event-study path -- which is how anyone reads one --
it covers the entire path with much less, and the gap widens with the number of
horizons. A simultaneous band fixes the level for the path as a whole by
inflating every interval by one shared critical value, chosen so the *largest*
standardized deviation across horizons stays inside with probability
``1 - alpha``.

The construction is Montiel Olea and Plagborg-Moller (2019): standardize the
draws, estimate the correlation across horizons, and take the ``1 - alpha``
quantile of ``max_h |z_h|`` for ``z`` drawn from that correlation. Simulating
rather than reading the quantile off the draws themselves matters when the draws
are a delete-one jackknife: there are only as many of them as units, so an
empirical quantile at 0.95 is coarse, and jackknife deviations are on a
different scale from the estimator's sampling distribution.

What the tests pin:

* the critical value is at least the pointwise one, and equals it for a single
  horizon (there is nothing to correct for);
* perfectly correlated horizons return the pointwise value, since the path moves
  as one number;
* independent horizons cost the most, and the cost grows with their number;
* the cumulative path's standard error grows like ``sqrt(L)`` under independent
  per-period errors and like ``L`` under perfectly correlated ones -- which is
  the whole reason a cumulative band cannot be assembled by adding endpoints.
"""
import warnings

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from scipy.stats import norm

from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.supt import (
    cumulative_from_paths,
    jackknife_se,
    supt_critical_value,
)


def _draws(n, H, rho=0.0, seed=0):
    """``n`` draws over ``H`` horizons with equicorrelation ``rho``."""
    rng = np.random.default_rng(seed)
    common = rng.normal(size=(n, 1))
    idio = rng.normal(size=(n, H))
    return np.sqrt(rho) * common + np.sqrt(1.0 - rho) * idio


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------

def test_returns_a_finite_positive_scalar():
    c = supt_critical_value(_draws(40, 5), alpha=0.05, seed=0)
    assert np.isfinite(c) and c > 0.0


def test_is_deterministic_under_a_seed():
    kw = dict(alpha=0.05, seed=7)
    d = _draws(40, 5)
    assert supt_critical_value(d, **kw) == supt_critical_value(d, **kw)


# ---------------------------------------------------------------------------
# Unit: the invariants that make it a simultaneous band
# ---------------------------------------------------------------------------

def test_one_horizon_is_the_pointwise_critical_value():
    """With nothing to be simultaneous over, the correction must vanish."""
    c = supt_critical_value(_draws(200, 1), alpha=0.05, n_sims=200_000, seed=0)
    assert c == pytest.approx(norm.ppf(0.975), rel=0.02)


#: Simulation error of the tabulated quantile, as a tolerance. The critical value
#: is the ``1 - alpha`` sample quantile of ``max_h |z_h|`` over ``n_sims`` draws,
#: so it carries a Monte Carlo standard error of
#: ``sqrt(a (1-a) / n_sims) / f(q)``; at ``alpha = 0.05`` and the default
#: ``n_sims`` that is about 0.004. Asserting an exact inequality against the
#: analytic pointwise value would be asserting below the computation's own floor.
_MC_TOL = 0.02


def test_never_narrower_than_pointwise():
    """A simultaneous band that undercut the pointwise one would be a bug.

    The maximum of several standardized deviations is at least any one of them,
    so its quantile cannot be smaller -- up to the simulation error of the
    quantile itself, which is what ``_MC_TOL`` allows and nothing more.
    """
    z = norm.ppf(0.975)
    for H in (1, 2, 5, 12):
        c = supt_critical_value(_draws(200, H, rho=0.3), alpha=0.05, seed=1)
        assert c >= z - _MC_TOL


def test_the_simulation_error_is_smaller_than_the_correction_it_measures():
    """The tolerance above must not be wide enough to hide a missing correction.

    Eight independent horizons need a correction of roughly 0.7 over pointwise.
    If the simulation error were the same size, the test above would pass on an
    implementation that returned the pointwise value and did nothing else.
    """
    z = norm.ppf(0.975)
    c = supt_critical_value(_draws(400, 8, rho=0.0, seed=5), alpha=0.05,
                            n_sims=100_000, seed=0)
    assert c - z > 10.0 * _MC_TOL


def test_perfectly_correlated_horizons_cost_nothing():
    """If every horizon is the same number, there is only one thing to cover."""
    n = 300
    rng = np.random.default_rng(3)
    one = rng.normal(size=(n, 1))
    d = np.repeat(one, 6, axis=1)
    c = supt_critical_value(d, alpha=0.05, n_sims=200_000, seed=0)
    assert c == pytest.approx(norm.ppf(0.975), rel=0.05)


def test_more_independent_horizons_cost_more():
    """The correction grows with the number of things being covered at once."""
    cs = [supt_critical_value(_draws(400, H, rho=0.0, seed=H), alpha=0.05,
                              n_sims=100_000, seed=0)
          for H in (2, 4, 8, 16)]
    assert all(a < b for a, b in zip(cs, cs[1:]))


def test_correlation_reduces_the_correction():
    """Correlated horizons are closer to one horizon, so they cost less."""
    low = supt_critical_value(_draws(400, 8, rho=0.0, seed=5), alpha=0.05,
                              n_sims=100_000, seed=0)
    high = supt_critical_value(_draws(400, 8, rho=0.9, seed=5), alpha=0.05,
                               n_sims=100_000, seed=0)
    assert high < low


def test_a_smaller_alpha_never_gives_a_smaller_critical_value():
    d = _draws(200, 6, rho=0.4)
    wide = supt_critical_value(d, alpha=0.01, n_sims=100_000, seed=0)
    narrow = supt_critical_value(d, alpha=0.10, n_sims=100_000, seed=0)
    assert wide > narrow


def test_the_critical_value_ignores_the_scale_of_the_draws():
    """It is computed from the correlation, so rescaling a horizon changes nothing."""
    d = _draws(300, 6, rho=0.3)
    scaled = d * np.array([1.0, 10.0, 0.1, 5.0, 2.0, 100.0])
    a = supt_critical_value(d, alpha=0.05, n_sims=100_000, seed=0)
    b = supt_critical_value(scaled, alpha=0.05, n_sims=100_000, seed=0)
    assert a == pytest.approx(b, rel=0.02)


# ---------------------------------------------------------------------------
# Unit: the cumulative path, and why it is not a running sum of endpoints
# ---------------------------------------------------------------------------

def test_cumulative_is_the_running_sum_of_the_path():
    paths = np.array([[1.0, 2.0, 3.0], [0.0, -1.0, 4.0]])
    assert cumulative_from_paths(paths) == pytest.approx(
        np.array([[1.0, 3.0, 6.0], [0.0, -1.0, 3.0]])
    )


def test_independent_period_errors_grow_the_cumulative_se_like_sqrt_L():
    """The reason a cumulative band is not the running total of period bands.

    Adding period interval endpoints assumes the errors move together, which
    grows the width like ``L``. When they are independent it grows like
    ``sqrt(L)``, and the difference is a factor of ``sqrt(L)`` of width that a
    reader is either given or denied for no reason.
    """
    draws = _draws(4000, 9, rho=0.0, seed=2)
    se = jackknife_se(cumulative_from_paths(draws), jackknife=False)
    ratio = se / se[0]
    assert ratio == pytest.approx(np.sqrt(np.arange(1, 10)), rel=0.12)


def test_perfectly_correlated_period_errors_grow_it_like_L():
    rng = np.random.default_rng(11)
    one = rng.normal(size=(4000, 1))
    draws = np.repeat(one, 9, axis=1)
    se = jackknife_se(cumulative_from_paths(draws), jackknife=False)
    ratio = se / se[0]
    assert ratio == pytest.approx(np.arange(1, 10), rel=0.02)


def test_jackknife_se_uses_the_delete_one_inflation():
    """``(m-1)/m * sum (x - xbar)^2``, the delete-one jackknife variance.

    Not the sample variance: jackknife replicates differ from the estimate by
    O(1/m), so they need the inflation to estimate a sampling standard error.
    """
    x = np.array([[1.0], [2.0], [3.0], [6.0]])
    m = 4
    expected = np.sqrt((m - 1) / m * np.sum((x[:, 0] - x[:, 0].mean()) ** 2))
    assert jackknife_se(x)[0] == pytest.approx(expected)


def test_jackknife_se_skips_missing_replicates():
    """A leave-one-out fit that failed is absent, not zero."""
    x = np.array([[1.0], [np.nan], [3.0], [5.0]])
    got = jackknife_se(x)[0]
    kept = np.array([1.0, 3.0, 5.0])
    m = 3
    assert got == pytest.approx(np.sqrt((m - 1) / m * np.sum((kept - kept.mean()) ** 2)))


# ---------------------------------------------------------------------------
# Unit: the correlation route equals the covariance route
# ---------------------------------------------------------------------------

def _covariance_route(draws, *, alpha, n_sim=200_000, seed=0):
    """The same critical value, computed the other way, written independently.

    Montiel Olea and Plagborg-Moller can be implemented either by simulating
    ``N(0, Sigma)`` and dividing each coordinate by its own standard deviation,
    or by simulating ``N(0, R)`` from the correlation directly. The two are the
    same distribution, since dividing a mean-zero Gaussian coordinate-wise by
    its standard deviations is exactly the change of variables from ``Sigma`` to
    ``R``. :func:`supt_critical_value` takes the second route because it needs no
    scale from the draws; this takes the first, so a defect in the change of
    variables shows up as disagreement.
    """
    R = np.asarray(draws, float)
    R = R[np.isfinite(R).all(axis=1)]
    m, H = R.shape
    dev = R - R.mean(axis=0)
    Sigma = (m - 1) / m * (dev.T @ dev)
    sd = np.sqrt(np.clip(np.diag(Sigma), 0.0, None))
    sd_safe = np.where(sd > 0, sd, np.inf)
    rng = np.random.default_rng(seed)
    G = rng.multivariate_normal(np.zeros(H), Sigma, size=n_sim)
    return float(np.quantile(np.max(np.abs(G) / sd_safe, axis=1), 1.0 - alpha))


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_the_correlation_route_matches_the_covariance_route(seed):
    """Agreement to the simulation error of the two independent tabulations."""
    rng = np.random.default_rng(seed)
    n, H = int(rng.integers(12, 70)), int(rng.integers(3, 12))
    draws = (rng.normal(size=(n, 2)) @ rng.normal(size=(2, H))
             + 0.4 * rng.normal(size=(n, H)))
    # ``_covariance_route`` tabulates a normal maximum, so the comparison is
    # against the normal reference. The studentized one answers a different
    # question and is checked separately.
    mine = supt_critical_value(draws, alpha=0.10, n_sims=200_000, seed=0,
                               reference="normal")
    other = _covariance_route(draws, alpha=0.10, seed=0)
    assert mine == pytest.approx(other, rel=0.02)


def test_the_jackknife_se_matches_the_covariance_diagonal():
    """``jackknife_se`` is the square root of the jackknife covariance diagonal.

    Pinned to machine precision, because the two are the same arithmetic and any
    gap would mean one of them has picked up a stray degrees-of-freedom factor.
    """
    rng = np.random.default_rng(5)
    draws = rng.normal(size=(41, 7))
    dev = draws - draws.mean(axis=0)
    m = draws.shape[0]
    diag = np.sqrt(np.diag((m - 1) / m * (dev.T @ dev)))
    assert jackknife_se(draws) == pytest.approx(diag, rel=1e-12)


# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------

def test_a_constant_horizon_does_not_produce_a_nan_critical_value():
    """A horizon with no variation has no correlation; it must not poison the rest."""
    d = _draws(200, 4, rho=0.2)
    d[:, 2] = 5.0
    c = supt_critical_value(d, alpha=0.05, n_sims=50_000, seed=0)
    assert np.isfinite(c) and c >= norm.ppf(0.975) - 1e-6


def test_fewer_than_two_replicates_gives_nan_se_rather_than_zero():
    assert np.isnan(jackknife_se(np.array([[1.0]]))[0])


def test_all_missing_replicates_gives_nan_se():
    assert np.isnan(jackknife_se(np.full((4, 1), np.nan))[0])


# ---------------------------------------------------------------------------
# Failure
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5, "0.05", None])
def test_an_alpha_outside_the_open_unit_interval_is_rejected(bad):
    with pytest.raises(MlsynthConfigError, match="alpha"):
        supt_critical_value(_draws(20, 3), alpha=bad)


def test_a_one_dimensional_draw_matrix_is_rejected():
    with pytest.raises(MlsynthDataError, match="2-D"):
        supt_critical_value(np.zeros(10), alpha=0.05)


def test_too_few_draws_to_estimate_a_correlation_is_reported():
    with pytest.raises(MlsynthDataError, match="draws"):
        supt_critical_value(np.zeros((1, 4)), alpha=0.05)


def test_a_nonpositive_simulation_count_is_rejected():
    with pytest.raises(MlsynthConfigError, match="n_sims"):
        supt_critical_value(_draws(20, 3), alpha=0.05, n_sims=0)


# ---------------------------------------------------------------------------
# Choosing the estimator: simulated (default) or read off the draws
#
# The Gaussian tabulation above is the right instrument for a delete-one
# jackknife -- few draws, and on a different scale from the sampling
# distribution. It is a poorer one for a large bootstrap ensemble, where the
# draws are numerous and already on the estimator's scale, and where reducing
# them to a correlation matrix discards the thing that drives ``max_h``: draws
# that are large at every horizon at once. ``method="empirical"`` reads the
# quantile off the draws instead, which is free when they are already in hand.
# ---------------------------------------------------------------------------

def _short_support_blocks(e, H, n, rng):
    """Circular blocks of length ``H`` from a short fixed series, accumulated.

    What a block bootstrap over a rolling calibration pass produces. The support
    is the series, not the draw count: ``e.size`` distinct blocks times a sign, so
    a large ``n`` resamples a small set instead of exploring a continuum. That is
    the regime where the two estimators part company.
    """
    m = e.size
    starts = rng.integers(0, m, size=n)
    idx = (starts[:, None] + np.arange(H)[None, :]) % m
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n, 1))
    return np.cumsum(e[idx] * signs, axis=1)


def test_empirical_returns_a_finite_positive_scalar():
    rng = np.random.default_rng(0)
    c = supt_critical_value(rng.standard_normal((5000, 6)), alpha=0.10,
                            method="empirical")
    assert np.isfinite(c) and c > 0.0


def test_empirical_ignores_n_sims_and_seed():
    """No RNG is consulted, so the two knobs that steer the simulation cannot
    move it. That is the point: the answer is a statistic of the draws."""
    rng = np.random.default_rng(1)
    draws = rng.standard_normal((4000, 5))
    a = supt_critical_value(draws, alpha=0.10, method="empirical", n_sims=10, seed=0)
    b = supt_critical_value(draws, alpha=0.10, method="empirical", n_sims=999_999, seed=7)
    assert a == b


def test_empirical_one_horizon_is_the_pointwise_critical_value():
    rng = np.random.default_rng(2)
    c = supt_critical_value(rng.standard_normal((200_000, 1)), alpha=0.10,
                            method="empirical")
    from scipy.stats import norm
    assert c == pytest.approx(norm.ppf(0.95), abs=0.02)


def test_empirical_ignores_the_scale_of_the_draws():
    """Same invariance the simulated route has: only the standardized path
    matters, so rescaling a horizon leaves the multiplier alone."""
    rng = np.random.default_rng(3)
    draws = rng.standard_normal((5000, 4))
    scaled = draws * np.array([1.0, 100.0, 0.01, 7.0])
    assert supt_critical_value(draws, alpha=0.10, method="empirical") == pytest.approx(
        supt_critical_value(scaled, alpha=0.10, method="empirical"), rel=1e-12)


def test_empirical_agrees_with_the_simulation_on_gaussian_draws():
    """Where the Gaussian assumption holds, the two routes answer the same."""
    rng = np.random.default_rng(4)
    draws = rng.standard_normal((200_000, 8))
    g = supt_critical_value(draws, alpha=0.10, n_sims=200_000, seed=0)
    e = supt_critical_value(draws, alpha=0.10, method="empirical")
    assert e == pytest.approx(g, rel=0.03)


def test_the_two_methods_part_company_on_a_short_support_ensemble():
    """The case the choice exists for.

    A calibration series with one atypical stretch gives an ensemble whose
    extreme draws are the ones that catch it. The empirical quantile sees them;
    the simulated route cannot, because a correlation matrix does not record
    which blocks the series happens to contain. Neither answer is wrong -- the
    simulation trades that idiosyncrasy for stability -- but they differ enough
    that the caller should get to pick.
    """
    for seed in range(4):
        rng = np.random.default_rng(seed)
        e = np.concatenate([rng.standard_normal(11) * (6.0 if i == 0 else 1.0)
                            for i in range(7)])
        e = e - e.mean()
        draws = _short_support_blocks(e, 11, 100_000, np.random.default_rng(100 + seed))
        g = supt_critical_value(draws, alpha=0.10, n_sims=100_000, seed=0)
        emp = supt_critical_value(draws, alpha=0.10, method="empirical")
        assert emp > 1.15 * g, f"seed {seed}: empirical {emp:.4f}, gaussian {g:.4f}"


def test_empirical_is_never_narrower_than_pointwise():
    from scipy.stats import norm
    rng = np.random.default_rng(6)
    for H in (2, 5, 12):
        c = supt_critical_value(rng.standard_normal((50_000, H)), alpha=0.10,
                                method="empirical")
        assert c >= norm.ppf(0.95) - 0.02


@pytest.mark.parametrize("bad", ["bootstrap", "Gaussian", "empirical ", "", None, 1])
def test_an_unknown_method_is_refused(bad):
    """Anything outside the two names raises instead of falling through.

    Silently defaulting would be the worst outcome: the returned number is a
    perfectly plausible multiplier, so a caller who asked for the empirical one
    and got the simulated one has no way to notice.
    """
    rng = np.random.default_rng(7)
    with pytest.raises(MlsynthConfigError, match="method"):
        supt_critical_value(rng.standard_normal((100, 3)), alpha=0.10, method=bad)


def test_the_default_method_is_the_simulation():
    """The default must not move: every existing caller keeps its numbers."""
    rng = np.random.default_rng(8)
    draws = rng.standard_normal((2000, 5))
    assert supt_critical_value(draws, alpha=0.10) == supt_critical_value(
        draws, alpha=0.10, method="gaussian")


def test_the_simulation_cannot_distinguish_ensembles_sharing_one_correlation():
    """The root cause, stated as an invariant.

    ``method="gaussian"`` reduces the draws to their correlation across horizons
    and reads the multiplier from normals carrying it, so it is a functional of
    that matrix alone: two ensembles with the same correlation get the same
    answer however different their joint laws. The quantile being estimated is
    not such a functional. Where the draws really are normal the two routes
    agree; where they are not, the simulation reports the normal answer.
    """
    H, N = 11, 100_000
    rng = np.random.default_rng(2)
    e = np.concatenate([rng.standard_normal(11) * (6.0 if i == 0 else 1.0)
                        for i in range(7)])
    e -= e.mean()
    B = _short_support_blocks(e, H, N, np.random.default_rng(101))

    R = np.corrcoef(B, rowvar=False)
    w, V = np.linalg.eigh(R)
    L = V * np.sqrt(np.clip(w, 0.0, None))
    A = np.random.default_rng(7).standard_normal((N, H)) @ L.T   # same R, normal

    assert np.max(np.abs(np.corrcoef(A, rowvar=False) - R)) < 0.02

    gA = supt_critical_value(A, alpha=0.10, n_sims=N, seed=0)
    gB = supt_critical_value(B, alpha=0.10, n_sims=N, seed=0)
    assert gA == pytest.approx(gB, abs=0.02)          # blind to the difference

    eA = supt_critical_value(A, alpha=0.10, method="empirical")
    eB = supt_critical_value(B, alpha=0.10, method="empirical")
    assert eA == pytest.approx(gA, abs=0.05)          # normal: the routes agree
    assert eB > gB + 1.0                              # not normal: they do not


def test_the_two_methods_converge_as_the_calibration_support_grows():
    """Why the ensembles differ at all: a circular block bootstrap over ``m``
    periods draws from ``m`` distinct blocks, so a large ``n_sim`` resamples a
    small set instead of exploring a continuum. Lengthen the series and the
    empirical quantile stops inheriting that particular series' idiosyncrasy.
    """
    H, N = 11, 40_000
    spread = {}
    for m in (77, 1100):
        ratios = []
        for s in range(6):
            r = np.random.default_rng(s)
            e = r.standard_normal(m)
            e -= e.mean()
            d = _short_support_blocks(e, H, N, np.random.default_rng(500 + s))
            g = supt_critical_value(d, alpha=0.10, n_sims=N, seed=0)
            ratios.append(supt_critical_value(d, alpha=0.10, method="empirical") / g)
        spread[m] = float(np.std(ratios))
    assert spread[1100] < 0.5 * spread[77], spread


# ---------------------------------------------------------------------------
# Rung 3, as a property, not one worked case.
# ---------------------------------------------------------------------------

@given(
    n_h=st.integers(min_value=2, max_value=8),
    rho=st.floats(min_value=-0.4, max_value=0.95, allow_nan=False),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(max_examples=25, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
def test_the_simulation_depends_on_the_draws_only_through_their_correlation(
        n_h, rho, seed):
    """``method="gaussian"`` is a functional of ``R``.

    Replace an ensemble with normals carrying the same correlation and the
    simulated multiplier does not move. This is the root cause stated as an
    invariant: whatever else the draws contain, the estimator cannot see it.
    """
    rng = np.random.default_rng(seed)
    H, N = n_h, 20_000
    base = np.full((H, H), rho)
    np.fill_diagonal(base, 1.0)
    w, V = np.linalg.eigh(base)
    L = V * np.sqrt(np.clip(w, 0.0, None))

    # An ensemble that is emphatically not normal: a two-component mixture.
    heavy = rng.standard_normal((N, H)) @ L.T
    heavy *= np.where(rng.random((N, 1)) < 0.1, 6.0, 1.0)

    R = np.corrcoef(heavy, rowvar=False)
    w2, V2 = np.linalg.eigh(R)
    twin = rng.standard_normal((N, H)) @ (V2 * np.sqrt(np.clip(w2, 0.0, None))).T

    a = supt_critical_value(heavy, alpha=0.10, n_sims=20_000, seed=0)
    b = supt_critical_value(twin, alpha=0.10, n_sims=20_000, seed=0)
    assert a == pytest.approx(b, abs=0.06), (a, b, rho, H)


@given(
    method=st.sampled_from(["gaussian", "empirical"]),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    n_h=st.integers(min_value=2, max_value=8),
)
@settings(max_examples=25, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
def test_both_methods_are_invariant_to_rescaling_a_horizon(method, seed, n_h):
    """Neither route may read the units. The multiplier applies to ``se_h``,
    which carries the scale, so a horizon measured in thousands must not buy a
    different band from the same horizon measured in millions."""
    rng = np.random.default_rng(seed)
    draws = rng.standard_normal((4000, n_h))
    factors = np.exp(rng.uniform(-6.0, 6.0, size=n_h))
    plain = supt_critical_value(draws, alpha=0.10, method=method,
                                n_sims=20_000, seed=0)
    scaled = supt_critical_value(draws * factors, alpha=0.10, method=method,
                                 n_sims=20_000, seed=0)
    assert plain == pytest.approx(scaled, rel=1e-9)


# ---------------------------------------------------------------------------
# The correlation is estimated, and at few replicates it is estimated badly.
#
# A sample correlation over-fits: it implies more co-movement across horizons
# than the truth has, which lowers the effective number of independent
# directions the maximum runs over, which shrinks the multiplier. At n <= H the
# matrix is rank-deficient outright and the simulated path is confined to an
# (n-1)-dimensional subspace. The band then claims 1 - alpha and covers less.
# ---------------------------------------------------------------------------

def _c_from_correlation(R, n_sims=40_000, seed=0):
    """The multiplier the simulation would return for a *known* correlation."""
    H = R.shape[0]
    w, V = np.linalg.eigh(R)
    L = V * np.sqrt(np.clip(w, 0.0, None))
    z = np.random.default_rng(seed).standard_normal((n_sims, H)) @ L.T
    scale = np.sqrt(np.clip(np.einsum("ij,ij->i", L, L), 1e-300, None))
    return float(np.quantile(np.max(np.abs(z / scale), axis=1), 0.90))


def test_a_known_correlation_gives_the_right_multiplier():
    """Rung 2, the control. Handed the truth, the simulation is correct, so
    whatever goes wrong at few replicates is the estimate of R and not the
    simulation, the eigenvalue clip, or the quantile."""
    H = 12
    c = _c_from_correlation(np.eye(H))
    z = np.random.default_rng(1).standard_normal((200_000, H))
    assert c == pytest.approx(float(np.quantile(np.max(np.abs(z), axis=1), 0.90)),
                              abs=0.02)


def test_the_multiplier_falls_as_replicates_are_removed_from_one_truth():
    """The fault, as a monotone sweep.

    The truth is fixed -- twelve independent horizons -- so the correct
    multiplier is one number at every n. What changes is only how many
    replicates R is estimated from, and the multiplier falls as that shrinks.
    """
    H = 12
    target = _c_from_correlation(np.eye(H))
    got = {}
    for n in (5, 12, 50):
        cs = [supt_critical_value(
                  np.random.default_rng(3000 + r).standard_normal((n, H)),
                  alpha=0.10, n_sims=40_000, seed=0, reference="normal")
              for r in range(8)]
        got[n] = float(np.mean(cs))
    assert got[5] < got[12] < got[50] < target
    assert target - got[5] > 0.10, got      # severe where n is below H
    assert target - got[50] < 0.03, got     # nearly gone by n = 4H


def test_the_shortfall_is_governed_by_replicates_per_horizon():
    """Why ``n`` alone is the wrong thing to guard on.

    Two panels with the same ``n`` but different horizon counts are not equally
    exposed; two with the same ratio are. That is what a guard has to read.
    """
    def shortfall(n, H):
        target = _c_from_correlation(np.eye(H))
        cs = [supt_critical_value(
                  np.random.default_rng(4000 + r).standard_normal((n, H)),
                  alpha=0.10, n_sims=40_000, seed=0, reference="normal")
              for r in range(6)]
        return target - float(np.mean(cs))

    at_ratio_one = [shortfall(H, H) for H in (6, 12)]
    at_ratio_eight = [shortfall(8 * H, H) for H in (6, 12)]
    assert max(at_ratio_one) - min(at_ratio_one) < 0.06, at_ratio_one
    assert all(s < 0.02 for s in at_ratio_eight), at_ratio_eight
    assert min(at_ratio_one) > max(at_ratio_eight)


def test_ledoit_wolf_recovers_the_multiplier_at_few_replicates():
    """The corrective action, measured against the truth it should reach.

    Shrinking the sample correlation toward the identity by a data-chosen
    intensity undoes the over-fitting. Where the horizons really are
    independent it is nearly exact from a handful of replicates.
    """
    H = 12
    target = _c_from_correlation(np.eye(H))
    plain, shrunk = [], []
    for r in range(8):
        d = np.random.default_rng(5000 + r).standard_normal((5, H))
        plain.append(supt_critical_value(d, alpha=0.10, n_sims=40_000, seed=0,
                                         reference="normal"))
        shrunk.append(supt_critical_value(d, alpha=0.10, n_sims=40_000, seed=0,
                                          shrinkage="ledoit_wolf",
                                          reference="normal"))
    assert target - np.mean(plain) > 0.10
    assert abs(target - np.mean(shrunk)) < 0.05


def test_ledoit_wolf_errs_wide_not_narrow_on_correlated_horizons():
    """The cost, stated as a direction.

    Shrinking a genuinely correlated matrix toward the identity flattens it, so
    the multiplier comes out a little large. That is the safe way to be wrong:
    a band slightly too wide, never one that claims a level it does not have.
    """
    H = 12
    R = 0.8 ** np.abs(np.subtract.outer(np.arange(H), np.arange(H)))
    w, V = np.linalg.eigh(R)
    L = V * np.sqrt(np.clip(w, 0.0, None))
    target = _c_from_correlation(R)
    shrunk = [supt_critical_value(
                  np.random.default_rng(6000 + r).standard_normal((12, H)) @ L.T,
                  alpha=0.10, n_sims=40_000, seed=0, shrinkage="ledoit_wolf",
                  reference="normal")
              for r in range(8)]
    assert np.mean(shrunk) > target
    assert np.mean(shrunk) - target < 0.20


def test_few_replicates_per_horizon_warns_by_default():
    """The trap has to be visible. Below two replicates per horizon the
    unshrunk multiplier is materially short, so saying nothing would ship a
    band that claims a level it does not reach."""
    rng = np.random.default_rng(7)
    with pytest.warns(RuntimeWarning, match="replicates"):
        supt_critical_value(rng.standard_normal((8, 12)), alpha=0.10, n_sims=5000)


def test_enough_replicates_per_horizon_is_silent():
    rng = np.random.default_rng(8)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        supt_critical_value(rng.standard_normal((120, 12)), alpha=0.10, n_sims=5000)


def test_shrinkage_silences_the_warning():
    """Having taken the corrective action, the caller should not be nagged."""
    rng = np.random.default_rng(9)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        supt_critical_value(rng.standard_normal((8, 12)), alpha=0.10, n_sims=5000,
                            shrinkage="ledoit_wolf")


def test_the_empirical_route_neither_shrinks_nor_warns():
    """Shrinkage is a repair to an estimated correlation, and the empirical
    route estimates none, so the option does not apply to it."""
    rng = np.random.default_rng(10)
    draws = rng.standard_normal((8, 12))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        a = supt_critical_value(draws, alpha=0.10, method="empirical")
        b = supt_critical_value(draws, alpha=0.10, method="empirical",
                                shrinkage="ledoit_wolf")
    assert a == b


@pytest.mark.parametrize("bad", ["ledoit-wolf", "LedoitWolf", "shrink", None, 2])
def test_an_unknown_shrinkage_is_refused(bad):
    rng = np.random.default_rng(11)
    with pytest.raises(MlsynthConfigError, match="shrinkage"):
        supt_critical_value(rng.standard_normal((100, 3)), alpha=0.10,
                            shrinkage=bad)


# ---------------------------------------------------------------------------
# The reference distribution: what the multiplier multiplies
#
# A band is ``point +/- c * se``, and both factors are estimated. The
# multiplier has always been the ``1 - alpha`` quantile of ``max_h |z_h|`` for
# ``z`` normal, which is the right answer when ``se_h`` is the true standard
# error. It is estimated from the same replicates, so the ratio
# ``(theta_h - theta_h) / se_h`` is a studentized quantity, not a normal one,
# and at few replicates a normal quantile does not reach its tail.
#
# Measured on the model where both truths are closed form -- ``m`` units drawn
# ``N(0, Sigma)``, the estimate their mean, the replicates the ``m`` delete-one
# means -- supplying the true standard error restores the level at every
# replicate count while supplying the true multiplier moves coverage by at most
# 0.05. So the multiplier is the smaller of the two terms, and the fault is the
# reference it is taken from.
# ---------------------------------------------------------------------------

def _jackknife_panel(m, H, chol, rng):
    """One replication of the model: the estimate and its delete-one replicates."""
    X = rng.standard_normal((m, H)) @ chol.T
    return X.mean(axis=0), (X.sum(axis=0) - X) / (m - 1)


def _cumulative_corr(H):
    """corr(S_j, S_k) = sqrt(min / max): a running total of independent errors."""
    j = np.arange(1, H + 1)
    return np.sqrt(np.minimum(j[:, None], j[None, :])
                   / np.maximum(j[:, None], j[None, :]))


def test_the_default_reference_is_studentized():
    """Pinned so it cannot drift back without a test saying so."""
    d = _draws(9, 6, seed=1)
    assert supt_critical_value(d, alpha=0.10) == supt_critical_value(
        d, alpha=0.10, reference="studentized")


def test_one_horizon_is_the_student_t_quantile():
    """With a single horizon the maximum is one ratio, whose law is exactly t.

    This is the check with a closed form, so it has power the coverage tests do
    not: a reference that is merely wider than normal, but not a t on ``m - 1``
    degrees, fails here.
    """
    from scipy.stats import t as student_t
    for m in (5, 12, 40):
        c = supt_critical_value(np.random.default_rng(m).standard_normal((m, 1)),
                                alpha=0.10, n_sims=200_000, seed=0)
        assert c == pytest.approx(student_t.ppf(0.95, m - 1), rel=0.02), m


def test_perfectly_correlated_horizons_also_give_the_student_quantile():
    """A path that moves as one number has one ratio, whatever its length."""
    from scipy.stats import t as student_t
    m = 10
    base = np.random.default_rng(0).standard_normal((m, 1))
    c = supt_critical_value(np.repeat(base, 5, axis=1), alpha=0.10,
                            n_sims=200_000, seed=0)
    assert c == pytest.approx(student_t.ppf(0.95, m - 1), rel=0.05)


def test_the_studentized_reference_is_wider_and_converges_to_the_normal_one():
    """Wider where the standard error is poorly determined, equal once it is not."""
    ratios = []
    for m in (6, 12, 25, 60):
        d = _draws(m, 6, seed=100 + m)
        s = supt_critical_value(d, alpha=0.10, n_sims=40_000, seed=0,
                                reference="studentized")
        n = supt_critical_value(d, alpha=0.10, n_sims=40_000, seed=0,
                                reference="normal")
        assert s > n, f"m={m}: studentized {s} not wider than normal {n}"
        ratios.append(s / n)
    assert all(a > b for a, b in zip(ratios, ratios[1:])), ratios
    assert ratios[-1] < 1.10, ratios


def test_the_studentized_reference_holds_the_level_where_the_normal_one_does_not():
    """The claim the fix exists to make, measured and not merely asserted.

    Nine replicates over six horizons -- a fourteen-unit panel, which is an
    ordinary PPSCM panel -- under the correlation a running total carries.
    """
    m, H, alpha, reps = 9, 6, 0.10, 500
    chol = np.linalg.cholesky(_cumulative_corr(H) + 1e-12 * np.eye(H))
    hit = {"studentized": 0, "normal": 0}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rng = np.random.default_rng(20260826)
        for _ in range(reps):
            xbar, rep = _jackknife_panel(m, H, chol, rng)
            se = jackknife_se(rep, jackknife=True)
            stat = float(np.max(np.abs(xbar) / se))
            s = int(rng.integers(1 << 31))
            for ref in ("studentized", "normal"):
                hit[ref] += stat <= supt_critical_value(
                    rep, alpha=alpha, n_sims=8_000, seed=s, reference=ref)
    studentized, normal = hit["studentized"] / reps, hit["normal"] / reps
    assert studentized == pytest.approx(1 - alpha, abs=0.05), studentized
    assert normal < 0.85, normal
    assert studentized > normal + 0.05, (studentized, normal)


def test_shrinkage_still_applies_under_the_studentized_reference():
    """The two knobs are independent: one repairs R, the other the reference."""
    d = _draws(8, 12, seed=7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = supt_critical_value(d, alpha=0.10, n_sims=40_000, seed=0)
        shrunk = supt_critical_value(d, alpha=0.10, n_sims=40_000, seed=0,
                                     shrinkage="ledoit_wolf")
    assert shrunk != plain
    assert shrunk > plain


def test_the_reference_does_not_reach_the_empirical_method():
    """``empirical`` reads a quantile off the draws and simulates nothing."""
    d = _draws(30, 5, seed=3)
    a = supt_critical_value(d, alpha=0.10, method="empirical",
                            reference="studentized")
    b = supt_critical_value(d, alpha=0.10, method="empirical",
                            reference="normal")
    assert a == b


def test_the_studentized_reference_still_ignores_the_units():
    """Rescaling a horizon cannot move a scale-free quantity."""
    d = _draws(10, 4, seed=5)
    scaled = d.copy()
    scaled[:, 2] *= 1e5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = supt_critical_value(d, alpha=0.10, n_sims=40_000, seed=0)
        b = supt_critical_value(scaled, alpha=0.10, n_sims=40_000, seed=0)
    assert a == pytest.approx(b, rel=0.05)


def test_two_replicates_is_the_thinnest_ensemble_and_still_returns():
    """The minimum the function accepts; one degree of freedom, so very wide."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = supt_critical_value(_draws(2, 3, seed=0), alpha=0.10,
                                n_sims=20_000, seed=0)
    assert np.isfinite(c) and c > 6.0


@pytest.mark.parametrize("bad", ["student", "Normal", "t", "", None, 1])
def test_an_unknown_reference_is_refused(bad):
    """Silently defaulting would hand back a plausible number from the wrong law."""
    with pytest.raises(MlsynthConfigError):
        supt_critical_value(_draws(20, 4), alpha=0.10, reference=bad)
