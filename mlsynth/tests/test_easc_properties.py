"""Property tests for the EASC kernels (Spruk 2026, arXiv:2610.05535).

The kernels are pure NumPy functions in ``mlsynth.utils.easc_helpers``:
leverage and the silent-factor diagnostic (``diagnostics``), the cross-donor
exposure regression and the correction (``correction``), the exposure-fit
frontier (``frontier``), and the bias-aware intervals (``inference``). Most of
what the paper promises about them is an identity or a metamorphic relation,
which needs no oracle:

* Corollary 1(a): ``kappa_t^2 = (kappa^lambda_t)^2 + ell_t^2`` exactly, for
  every period and for every phase average.
* Proposition 1: the pre-period mean of ``kappa_t^2`` is the number of factors.
* Proposition 3: ``d_min`` is the distance to the donors' exposure range and no
  simplex weights do better; the frontier is non-increasing and convex.
* Proposition 4: the correction depends on the exposure index only through
  its information, so an affine rescaling of the index cannot move the EASC gap.
* Proposition 5: the interval covers at exactly ``1 - alpha`` when the bias
  sits at its bound.

The mutants for these modules are catalogued in ``tools/mutation/targets.toml``
under the ``easc-*`` targets.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from scipy.stats import norm

from mlsynth.utils.easc_helpers import correction as C
from mlsynth.utils.easc_helpers import diagnostics as D
from mlsynth.utils.easc_helpers import frontier as Fr
from mlsynth.utils.easc_helpers import inference as I

_SEEDS = st.integers(min_value=0, max_value=2**32 - 1)


def _rng(seed):
    return np.random.default_rng(seed)


def _pre(T, T0):
    return np.arange(T) < T0


def _proj(F):
    """Projection onto the column space of ``F``: the factors up to rotation and sign."""
    return F @ np.linalg.pinv(F)


# ---------------------------------------------------------------------------
# diagnostics: leverage and the silent factor
# ---------------------------------------------------------------------------

@given(seed=_SEEDS, k=st.integers(1, 3), T0=st.integers(8, 20), T1=st.integers(1, 10))
def test_leverage_splits_into_latent_and_partial_leverage(seed, k, T0, T1):
    """Corollary 1(a): kappa_t^2 = (kappa^lambda_t)^2 + ell_t^2 in every period."""
    rng = _rng(seed)
    T = T0 + T1
    pre = _pre(T, T0)
    F = rng.standard_normal((T, k))
    x = F @ rng.standard_normal(k) + rng.standard_normal(T)
    ell, _ = D.partial_leverage(x, F, pre)
    full = D.leverage(D.with_constant(F, x), pre)
    latent = D.leverage(D.with_constant(F), pre)
    np.testing.assert_allclose(full ** 2, latent ** 2 + ell ** 2, rtol=1e-8, atol=1e-8)


@given(seed=_SEEDS, k=st.integers(1, 3), T0=st.integers(8, 20), T1=st.integers(2, 10),
       start=st.integers(0, 9), length=st.integers(1, 10))
def test_phase_leverage_splits_the_same_way(seed, k, T0, T1, start, length):
    """Corollary 1(a) holds for the leverage of a phase-averaged factor vector,
    with the phase average of ell_t in place of ell_t. The average of the
    per-period leverages does not satisfy it."""
    rng = _rng(seed)
    T = T0 + T1
    pre = _pre(T, T0)
    mask = np.zeros(T, dtype=bool)
    lo = T0 + min(start, T1 - 1)
    mask[lo:min(T, lo + length)] = True
    F = rng.standard_normal((T, k))
    x = F @ rng.standard_normal(k) + rng.standard_normal(T)
    ell, _ = D.partial_leverage(x, F, pre)
    full = D.phase_leverage(D.with_constant(F, x), pre, mask)
    latent = D.phase_leverage(D.with_constant(F), pre, mask)
    assert full ** 2 == pytest.approx(latent ** 2 + ell[mask].mean() ** 2, rel=1e-8, abs=1e-8)


@given(seed=_SEEDS, p=st.integers(1, 4), T0=st.integers(8, 25), T1=st.integers(0, 8))
def test_pre_period_mean_squared_leverage_is_the_number_of_factors(seed, p, T0, T1):
    """Proposition 1: T0^{-1} sum_{t <= T0} kappa_t^2 = p, whatever happens after."""
    rng = _rng(seed)
    T = T0 + T1
    pre = _pre(T, T0)
    G = D.with_constant(rng.standard_normal((T, p - 1))) if p > 1 else np.ones((T, 1))
    G[~pre] *= 10.0  # the post-period is free to leave the pre-period span
    kappa = D.leverage(G, pre)
    assert np.mean(kappa[pre] ** 2) == pytest.approx(p, rel=1e-8)


@given(seed=_SEEDS, p=st.integers(1, 4), T0=st.integers(8, 20))
def test_leverage_does_not_depend_on_how_the_factors_are_rotated(seed, p, T0):
    """kappa_t is a Mahalanobis length: an invertible recombination of the factor
    vector cannot change it."""
    rng = _rng(seed)
    T = T0 + 6
    pre = _pre(T, T0)
    G = rng.standard_normal((T, p))
    A = rng.standard_normal((p, p)) + 3 * np.eye(p)
    assume(abs(np.linalg.det(A)) > 1e-3)
    np.testing.assert_allclose(D.leverage(G @ A, pre), D.leverage(G, pre), rtol=1e-7, atol=1e-9)


@given(seed=_SEEDS, a=st.floats(0.1, 10) | st.floats(-10, -0.1), b=st.floats(-50, 50))
def test_partial_leverage_is_affine_equivariant_in_the_shock(seed, a, b):
    """Changing the shock's units or origin flips at most the sign of ell_t and
    leaves the pre-period R^2 alone."""
    rng = _rng(seed)
    T, T0 = 24, 16
    pre = _pre(T, T0)
    F = rng.standard_normal((T, 2))
    x = F @ rng.standard_normal(2) + 0.5 * rng.standard_normal(T)
    ell, r2 = D.partial_leverage(x, F, pre)
    ell2, r22 = D.partial_leverage(a * x + b, F, pre)
    np.testing.assert_allclose(ell2, np.sign(a) * ell, rtol=1e-7, atol=1e-7)
    assert r22 == pytest.approx(r2, abs=1e-9)
    assert 0.0 <= r2 <= 1.0
    assert np.sqrt(np.mean(ell[pre] ** 2)) == pytest.approx(1.0, rel=1e-9)


def test_partial_leverage_of_an_in_span_shock_is_undefined_and_says_so():
    """S_r = 0 (the limiting case of Corollary 1): the shock is an exact linear
    function of the factors before treatment, so ell_t has no scale."""
    T, T0 = 20, 12
    pre = _pre(T, T0)
    F = np.linspace(-1, 1, T)[:, None]
    x = 2.0 * F[:, 0] + 1.0
    with pytest.warns(UserWarning, match="exact linear function"):
        ell, r2 = D.partial_leverage(x, F, pre)
    assert np.isnan(ell).all()
    assert r2 == 1.0


@given(seed=_SEEDS, J=st.integers(3, 12), T=st.integers(10, 30), k=st.integers(1, 2),
       scale=st.floats(-5, 5))
def test_exposure_orthogonal_factors_ignore_any_exposure_driven_component(seed, J, T, k, scale):
    """Removing each period's cross-sectional projection on the index makes the
    factors blind to any term of the form g_t s_j, which is exactly the part the
    exposure channel can contribute."""
    rng = _rng(seed)
    Y0 = rng.standard_normal((T, J)) + np.outer(rng.standard_normal(T), rng.standard_normal(J))
    s = rng.standard_normal(J)
    assume(s.std() > 1e-3)
    assume(k < min(T, J) - 1)
    g = scale * rng.standard_normal(T)
    F1 = D.donor_factors(Y0, k, s)
    F2 = D.donor_factors(Y0 + np.outer(g, s), k, s)
    assume(np.linalg.matrix_rank(F1) == k)
    np.testing.assert_allclose(_proj(F2), _proj(F1), atol=1e-6)


@given(seed=_SEEDS, J=st.integers(2, 10), s1=st.floats(-20, 20))
def test_d_min_is_the_distance_to_the_exposure_range(seed, J, s1):
    """Proposition 3(a): no simplex weights get closer to s_1 than d_min, and d_min
    is zero exactly when s_1 lies in [min s_j, max s_j]."""
    rng = _rng(seed)
    s0 = rng.uniform(-10, 10, J)
    dmin = D.min_imbalance(s1, s0)
    inside = s0.min() <= s1 <= s0.max()
    assert (dmin == 0.0) == inside
    if not inside:
        assert dmin == pytest.approx(min(abs(s1 - s0.min()), abs(s1 - s0.max())))
    w = rng.dirichlet(np.ones(J))
    assert abs(D.exposure_imbalance(s1, s0, w)) >= dmin - 1e-12
    nearest = np.zeros(J)
    nearest[np.argmin(np.abs(s0 - s1))] = 1.0
    if not inside:
        assert abs(D.exposure_imbalance(s1, s0, nearest)) == pytest.approx(dmin)


# ---------------------------------------------------------------------------
# correction: the exposure regression and the EASC gap
# ---------------------------------------------------------------------------

def _linear_panel(rng, T, T0, J):
    """Donor outcomes ``d_t + xi_j f_t + phi_t s_j + a_j`` with ``phi_t = 0`` before
    treatment: the growth controls span ``xi_j``, so the changes are exactly
    linear in ``(1, s_j, c_j)``."""
    s = rng.uniform(0, 30, J)
    xi = rng.standard_normal(J)
    f = np.cumsum(rng.standard_normal(T))
    d = rng.standard_normal(T)
    phi_true = np.where(np.arange(T) >= T0, rng.standard_normal(T), 0.0)
    Y0 = d[:, None] + np.outer(f, xi) + np.outer(phi_true, s) + rng.standard_normal(J)[None, :]
    return Y0, s, phi_true


@given(seed=_SEEDS, J=st.integers(8, 20), n_sub=st.integers(1, 3))
def test_exposure_regression_recovers_an_exact_linear_effect(seed, J, n_sub):
    """When donors' changes are exactly linear in (1, s_j, c_j), phi_t is the slope
    on s_j, period by period, with no error (Proposition 4(a), noiseless)."""
    rng = _rng(seed)
    T, T0 = 24, 15
    pre = _pre(T, T0)
    Y0, s, phi_true = _linear_panel(rng, T, T0, J)
    reg = C.exposure_effects(Y0, s, pre, T0 - 1, n_sub)
    post = ~pre
    np.testing.assert_allclose(reg.phi[post], phi_true[post], atol=1e-8)


@given(seed=_SEEDS, a=st.floats(0.05, 20) | st.floats(-20, -0.05), b=st.floats(-100, 100))
def test_easc_gap_ignores_the_units_and_origin_of_the_exposure_index(seed, a, b):
    """phi scales by 1/a and EI(w) by a, so the correction phi_t EI(w) -- and the
    EASC gap -- cannot depend on whether exposure is in percent or in shares."""
    rng = _rng(seed)
    T, T0, J = 20, 12, 10
    pre = _pre(T, T0)
    Y0 = rng.standard_normal((T, J)).cumsum(0)
    y1 = rng.standard_normal(T).cumsum()
    s0, s1 = rng.uniform(0, 20, J), float(rng.uniform(0, 20))
    w = rng.dirichlet(np.ones(J))
    phi = C.exposure_effects(Y0, s0, pre, T0 - 1, 2).phi
    phi2 = C.exposure_effects(Y0, a * s0 + b, pre, T0 - 1, 2).phi
    g, _ = C.easc_gap(y1, Y0, w, s1, s0, phi)
    g2, _ = C.easc_gap(y1, Y0, w, a * s1 + b, a * s0 + b, phi2)
    np.testing.assert_allclose(g2, g, rtol=1e-7, atol=1e-7)


@given(seed=_SEEDS)
def test_exposure_effects_ignore_period_shocks_and_unit_fixed_effects(seed):
    """Common period shocks are absorbed by the intercept and unit levels drop out
    of the changes, so neither can move phi or its standard error."""
    rng = _rng(seed)
    T, T0, J = 22, 14, 12
    pre = _pre(T, T0)
    Y0 = rng.standard_normal((T, J)).cumsum(0)
    s = rng.uniform(0, 20, J)
    base = C.exposure_effects(Y0, s, pre, T0 - 1, 3)
    shifted = Y0 + rng.standard_normal(T)[:, None] + rng.standard_normal(J)[None, :]
    other = C.exposure_effects(shifted, s, pre, T0 - 1, 3)
    np.testing.assert_allclose(other.phi, base.phi, atol=1e-9)
    np.testing.assert_allclose(other.se, base.se, atol=1e-9)


@given(seed=_SEEDS, c=st.floats(0.01, 100) | st.floats(-100, -0.01))
def test_exposure_effects_scale_with_the_outcome(seed, c):
    rng = _rng(seed)
    T, T0, J = 22, 14, 12
    pre = _pre(T, T0)
    Y0 = rng.standard_normal((T, J)).cumsum(0)
    s = rng.uniform(0, 20, J)
    base = C.exposure_effects(Y0, s, pre, T0 - 1, 3)
    scaled = C.exposure_effects(c * Y0, s, pre, T0 - 1, 3)
    np.testing.assert_allclose(scaled.phi, c * base.phi, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(scaled.se, abs(c) * base.se, rtol=1e-8, atol=1e-10)


@given(seed=_SEEDS, n_sub=st.integers(1, 3), t=st.integers(0, 30))
def test_growth_controls_never_read_the_base_period_or_anything_after_it(seed, n_sub, t):
    """The controls end before b so they share no noise with Y_jt - Y_jb."""
    rng = _rng(seed)
    T, T0, J = 24, 16, 6
    pre = _pre(T, T0)
    base = T0 - 1
    Y0 = rng.standard_normal((T, J))
    bumped = Y0.copy()
    period = t % T
    bumped[period] += 5.0
    same = np.allclose(C.growth_controls(bumped, pre, base, n_sub), C.growth_controls(Y0, pre, base, n_sub))
    if period >= base:
        assert same
    edges = {th[0] for th in np.array_split(np.arange(base), n_sub)} | {
        th[-1] for th in np.array_split(np.arange(base), n_sub)}
    if period in edges:
        assert not same


@given(seed=_SEEDS)
def test_easc_equals_sc_when_the_weights_balance_exposure(seed):
    """EI(w) = 0 switches the correction off, whatever phi is (Proposition 4(c))."""
    rng = _rng(seed)
    T, J = 15, 6
    Y0 = rng.standard_normal((T, J))
    y1 = rng.standard_normal(T)
    s0 = rng.uniform(0, 10, J)
    w = rng.dirichlet(np.ones(J))
    s1 = float(s0 @ w)
    phi = rng.standard_normal(T) * 100
    gap, ei = C.easc_gap(y1, Y0, w, s1, s0, phi)
    assert ei == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(gap, y1 - Y0 @ w, atol=1e-9)


@given(seed=_SEEDS, tau=st.floats(-5, 5))
def test_easc_removes_exactly_the_exposure_term(seed, tau):
    """If the treated path is the weighted donors plus phi_t EI(w) plus an effect,
    the EASC gap is the effect."""
    rng = _rng(seed)
    T, J = 15, 6
    Y0 = rng.standard_normal((T, J))
    s0 = rng.uniform(0, 10, J)
    s1 = float(rng.uniform(0, 30))
    w = rng.dirichlet(np.ones(J))
    phi = rng.standard_normal(T)
    y1 = Y0 @ w + phi * (s1 - s0 @ w) + tau
    gap, _ = C.easc_gap(y1, Y0, w, s1, s0, phi)
    np.testing.assert_allclose(gap, tau, atol=1e-9)


@given(seed=_SEEDS, J=st.integers(3, 15))
def test_jackknife_phase_se_of_a_mean_is_the_textbook_standard_error(seed, J):
    """For a sample mean the delete-one jackknife is exact: s / sqrt(J)."""
    rng = _rng(seed)
    v = rng.standard_normal(J)
    reps = np.array([[np.delete(v, j).mean()] for j in range(J)])
    se = C.jackknife_phase_se(reps, np.array([True]))
    assert se == pytest.approx(v.std(ddof=1) / np.sqrt(J), rel=1e-9)


@given(seed=_SEEDS, J=st.integers(7, 20))
def test_per_period_and_single_period_phase_standard_errors_agree(seed, J):
    """The per-period jackknife s.e. and the phase s.e. of a one-period phase are
    the same statistic computed two ways; EASC's noise scale uses both."""
    rng = _rng(seed)
    T, T0 = 20, 13
    pre = _pre(T, T0)
    reg = C.exposure_effects(rng.standard_normal((T, J)).cumsum(0), rng.uniform(0, 20, J), pre, T0 - 1, 2)
    for t in range(T):
        mask = np.zeros(T, dtype=bool)
        mask[t] = True
        assert reg.se[t] == pytest.approx(C.jackknife_phase_se(reg.replicates, mask), rel=1e-9, abs=1e-14)


@given(seed=_SEEDS, J=st.integers(7, 15), how=st.sampled_from(["varies", "constant", "control"]))
def test_exposure_is_identified_only_when_it_varies_beyond_the_controls(seed, J, how):
    rng = _rng(seed)
    T, T0 = 20, 13
    pre = _pre(T, T0)
    Y0 = rng.standard_normal((T, J)).cumsum(0)
    if how == "varies":
        s = rng.uniform(0, 20, J)
    elif how == "constant":
        s = np.full(J, 7.0)
    else:  # an affine function of the first growth control
        s = 3.0 * C.growth_controls(Y0, pre, T0 - 1, 2)[:, 0] + 1.0
    assert C.exposure_is_identified(Y0, s, pre, T0 - 1, 2) == (how == "varies")


@given(seed=_SEEDS, J=st.integers(7, 20), n_sub=st.integers(0, 3))
def test_hat_values_are_leverages_of_the_exposure_regression(seed, J, n_sub):
    rng = _rng(seed)
    T, T0 = 20, 13
    pre = _pre(T, T0)
    Y0 = rng.standard_normal((T, J)).cumsum(0)
    s = rng.uniform(0, 20, J)
    h = C.hat_values(Y0, s, pre, T0 - 1, n_sub)
    assert np.all(h >= -1e-12) and np.all(h <= 1 + 1e-12)
    assert h.sum() == pytest.approx(2 + n_sub, abs=1e-8)


# ---------------------------------------------------------------------------
# frontier
# ---------------------------------------------------------------------------

@settings(max_examples=25, deadline=None)
@given(seed=_SEEDS, J=st.integers(3, 8), outside=st.booleans())
def test_frontier_is_feasible_non_increasing_and_convex(seed, J, outside):
    """Proposition 3(b): R(d) is non-increasing and convex in d, every point meets
    its tolerance, the first point attains d_min and the last is synthetic control."""
    rng = _rng(seed)
    T0 = 12
    A = rng.standard_normal((T0, J)).cumsum(0)
    a = A @ rng.dirichlet(np.ones(J)) + 0.1 * rng.standard_normal(T0)
    s0 = rng.uniform(0, 10, J)
    s1 = 15.0 if outside else float(rng.uniform(s0.min(), s0.max()))
    w_sc = Fr.sc_weights(a, A)
    fr = Fr.exposure_fit_frontier(a, A, s1, s0, 9, w_sc)
    assert np.all(np.abs(fr.exposure_imbalance) <= fr.tolerance + 1e-6)
    assert abs(fr.exposure_imbalance[0]) == pytest.approx(D.min_imbalance(s1, s0), abs=1e-5)
    np.testing.assert_allclose(fr.weights[-1], w_sc)
    np.testing.assert_allclose(fr.weights.sum(1), 1.0, atol=1e-9)
    assert np.all(fr.weights >= 0)
    r2 = fr.rmspe ** 2
    assert np.all(np.diff(r2) <= 1e-7)
    # convexity of R(d) on an even grid: second differences are non-negative
    assert np.all(np.diff(fr.rmspe, 2) >= -1e-5)


@given(seed=_SEEDS, n=st.integers(2, 12))
def test_frontier_selection_takes_the_first_minimiser(seed, n):
    rng = _rng(seed)
    crit = rng.integers(0, 4, n).astype(float)
    assert Fr.first_minimiser(crit) == int(np.flatnonzero(crit == crit.min())[0])


@settings(max_examples=25, deadline=None)
@given(seed=_SEEDS)
def test_selection_criterion_prices_misfit_by_latent_leverage_and_imbalance_by_its_se(seed):
    """Equation (12): with no correction error the criterion is the extra misfit
    alone, zero at the synthetic-control end; the correction's variance adds
    se(phi_t)^2 EI(w)^2."""
    rng = _rng(seed)
    n, T = 6, 10
    post = np.arange(T) >= 6
    rmspe = np.sort(rng.uniform(0.01, 1, n))[::-1]
    ei = np.linspace(1, 5, n)
    kappa = rng.uniform(1, 5, T)
    crit = Fr.selection_criterion(rmspe, ei, kappa, np.zeros(T), post)
    assert crit[-1] == pytest.approx(0.0)
    expected = np.mean(kappa[post] ** 2) * (rmspe ** 2 - rmspe[-1] ** 2)
    np.testing.assert_allclose(crit, expected, rtol=1e-9)
    se = rng.uniform(0.001, 0.01, T)
    full = Fr.selection_criterion(rmspe, ei, kappa, se, post)
    np.testing.assert_allclose(full, expected + np.mean(se[post] ** 2) * ei ** 2, rtol=1e-9)


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------

_alpha = st.floats(0.01, 0.3)


@given(b=st.floats(0, 20), alpha=_alpha)
def test_bias_aware_interval_covers_exactly_at_the_bias_bound(b, alpha):
    """Proposition 5: P(|Z + b| <= cv(b)) = 1 - alpha, and cv sits between the
    one-sided and the additive bounds."""
    c = I.cv_bias_aware(b, alpha)
    coverage = norm.cdf(c - b) - norm.cdf(-c - b)
    assert coverage == pytest.approx(1 - alpha, abs=1e-8)
    assert c >= norm.ppf(1 - alpha / 2) - 1e-9
    assert c >= b + norm.ppf(1 - alpha) - 1e-7
    assert c <= b + norm.ppf(1 - alpha / 2) + 1e-7


@given(b1=st.floats(0, 10), b2=st.floats(0, 10), alpha=_alpha)
def test_bias_aware_critical_value_increases_with_the_bias(b1, b2, alpha):
    lo, hi = sorted((b1, b2))
    assert I.cv_bias_aware(lo, alpha) <= I.cv_bias_aware(hi, alpha) + 1e-9


def test_bias_aware_critical_value_without_bias_is_the_normal_quantile():
    assert I.cv_bias_aware(0.0, 0.05) == pytest.approx(norm.ppf(0.975))


@given(tau=st.floats(-1, 1), kappa=st.floats(0.5, 40), rmspe=st.floats(0.001, 0.1),
       sd=st.floats(0.01, 0.5), c=st.floats(0.1, 10))
def test_breakdown_value_is_unit_free(tau, kappa, rmspe, sd, c):
    """M* compares the estimate with a bias bound in the same units, so measuring
    the outcome in other units cannot change it."""
    m = I.breakdown(tau, kappa, rmspe, sd, 0.05)
    assert I.breakdown(c * tau, kappa, c * rmspe, c * sd, 0.05) == pytest.approx(m, abs=1e-8)


@given(tau=st.floats(-1, 1), kappa=st.floats(0.5, 40), rmspe=st.floats(0.001, 0.1),
       sd=st.floats(0.01, 0.5), c=st.floats(1.5, 10))
def test_breakdown_value_falls_in_proportion_to_leverage(tau, kappa, rmspe, sd, c):
    """The bias bound is M kappa RMSPE, so c times the leverage needs 1/c of the M."""
    m = I.breakdown(tau, kappa, rmspe, sd, 0.05)
    assume(0 < m < 4.9)
    assert I.breakdown(tau, c * kappa, rmspe, sd, 0.05) == pytest.approx(m / c, rel=1e-6)


@given(tau=st.floats(-1, 1), kappa=st.floats(0.5, 40), rmspe=st.floats(0.001, 0.1),
       sd=st.floats(0.01, 0.5))
def test_breakdown_value_is_where_the_interval_reaches_zero(tau, kappa, rmspe, sd):
    m = I.breakdown(tau, kappa, rmspe, sd, 0.05)
    hw = lambda M: I.bias_aware_halfwidth(M, kappa, rmspe, sd, 0.05)
    if abs(tau) <= hw(0.0):
        assert m == 0.0
    elif m < 5.0:
        assert hw(m) == pytest.approx(abs(tau), rel=1e-7)
        assert hw(0.9 * m) < abs(tau) < hw(1.1 * m)


@given(M1=st.floats(0, 3), M2=st.floats(0, 3), kappa=st.floats(0.5, 40),
       rmspe=st.floats(0.001, 0.1), sd=st.floats(0.01, 0.5))
def test_interval_widens_with_the_sensitivity_parameter(M1, M2, kappa, rmspe, sd):
    lo, hi = sorted((M1, M2))
    assert (I.bias_aware_halfwidth(lo, kappa, rmspe, sd, 0.05)
            <= I.bias_aware_halfwidth(hi, kappa, rmspe, sd, 0.05) + 1e-12)


@given(seed=_SEEDS, J=st.integers(3, 25), c=st.floats(0.01, 100))
def test_placebo_noise_scale_is_scale_equivariant(seed, J, c):
    rng = _rng(seed)
    means = rng.standard_normal(J)
    rmspe = rng.uniform(0.01, 1, J)
    sd, sdm = I.placebo_noise_scale(means, rmspe, 0.3)
    sd2, sdm2 = I.placebo_noise_scale(c * means, c * rmspe, c * 0.3)
    assert sd2 == pytest.approx(c * sd, rel=1e-9)
    assert sdm2 == pytest.approx(c * sdm, rel=1e-9)


@given(seed=_SEEDS, J=st.integers(3, 25), k=st.floats(-3, 3))
def test_placebos_with_proportional_gaps_carry_no_noise(seed, J, k):
    """Each placebo's phase average is standardised by its own pre-period RMSPE,
    so placebos whose gaps are a fixed multiple of their misfit agree exactly."""
    rng = _rng(seed)
    rmspe = rng.uniform(0.01, 1, J)
    sd, sdm = I.placebo_noise_scale(k * rmspe, rmspe, 0.5)
    assert sd == pytest.approx(0.0, abs=1e-12)
    assert sdm == pytest.approx(0.0, abs=1e-12)


@given(seed=_SEEDS, J=st.integers(2, 25))
def test_placebo_p_value_counts_the_treated_unit(seed, J):
    """The p-value is a rank among J + 1 units, so it is at least 1 / (J + 1) and
    equals it when the treated unit's ratio is the largest."""
    rng = _rng(seed)
    means = rng.standard_normal(J)
    rmspe = rng.uniform(0.5, 1, J)
    p = I.placebo_p_value(0.1, 1.0, means, rmspe)
    assert 1 / (J + 1) - 1e-12 <= p <= 1.0
    assert I.placebo_p_value(1e6, 1.0, means, rmspe) == pytest.approx(1 / (J + 1))
    assert I.placebo_p_value(0.0, 1.0, means, rmspe) == pytest.approx(1.0)
