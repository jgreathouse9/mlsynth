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
import numpy as np
import pytest
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
    mine = supt_critical_value(draws, alpha=0.10, n_sims=200_000, seed=0)
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
    a large ``n`` resamples a small set rather than exploring a continuum. That is
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
    """Anything outside the two names raises rather than falling through.

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
    small set rather than exploring a continuum. Lengthen the series and the
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
