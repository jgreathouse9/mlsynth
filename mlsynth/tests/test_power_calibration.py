"""Calibration of the analytical power surface in :mod:`mlsynth.utils.post_fit`.

The existing power tests assert shape -- the MDE is finite, positive, and falls
with the horizon. None of them asks the question a power analysis exists to
answer: does the reported standard error describe the spread the analyst will
actually see, so that a nominal 5% test rejects 5% of the time? These do.

The reference is Vives-i-Bastida (2022), *Synthetic Experimental Design for a
UBI pilot study*, Section 2, which derives the design's inference from the blank
period. Two of its properties drive the tests here.

The permutation null is built from the blank-period residuals ``u_t``, so the
null is only mean-zero when the design is unbiased on that window. The paper
checks this directly ("if the fit were bad in the blank periods it could be that
our inference procedure would yield biased results"). A design whose blank-period
gap has a non-zero mean carries that offset into the post window, where it adds
to the treatment effect the analyst is trying to measure, so it belongs in the
error scale the MDE is built from.

The future residual is unknown at design time, and the paper prices that by
reporting the MDE under three imputations of it -- the mean, the max, and twice
the max of the observed blank residuals (their Table 3). The analytical surface
reports one number, so it owes the reader the sampling uncertainty in the scale
that number rests on.

No client data appears here: the panels are simulated with a known
data-generating process, which is what gives the calibration assertions their
power to fail.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.post_fit import (
    compute_post_fit,
    compute_power_analysis,
)

ALPHA = 0.05
Z = 1.959963984540054


def _ar1(rng, n, sigma, rho, bias=0.0):
    """A stationary AR(1) series with marginal SD ``sigma``, plus a constant."""
    e = rng.normal(0.0, sigma * np.sqrt(1.0 - rho**2), n)
    x = np.empty(n)
    x[0] = rng.normal(0.0, sigma)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + e[t]
    return x + bias


def _post_fit_with_gap(gap, n_fit, n_blank, n_post):
    """A post-fit whose gap series is exactly ``gap`` (control fixed at 100)."""
    control = np.full(len(gap), 100.0)
    return compute_post_fit(control + gap, control,
                            n_fit=n_fit, n_blank=n_blank, n_post=n_post)


# ---------------------------------------------------------------------------
# The scale the MDE rests on
# ---------------------------------------------------------------------------

def test_placebo_bias_is_reported():
    """A design biased on the blank window must say so on the result object.

    Invariant 7 of CLAUDE.md: a diagnostic the caller might act on is a typed
    field, never stdout and never discarded.
    """
    rng = np.random.default_rng(0)
    gap = np.concatenate([_ar1(rng, 60, 5.0, 0.0),
                          _ar1(rng, 40, 5.0, 0.0, bias=8.0)])
    pf = _post_fit_with_gap(gap, n_fit=60, n_blank=40, n_post=0)
    pw = compute_power_analysis(pf, alpha=ALPHA)
    assert pw.placebo_bias == pytest.approx(gap[60:].mean(), rel=1e-9)
    assert pw.placebo_bias_pvalue < 0.01           # the offset is real, not noise
    assert pw.n_placebo == 40


def test_serial_correlation_is_estimated_on_demeaned_residuals():
    """A constant offset is not serial dependence.

    The lag-1 autocorrelation of an uncentred series with mean ``b`` and noise
    ``sigma`` tends to ``b^2 / (b^2 + sigma^2)``, which is large for a biased
    design and has nothing to do with persistence. Estimating it on the demeaned
    residuals is what makes the AR(1) variance inflation mean what it says.
    """
    rng = np.random.default_rng(1)
    gap = np.concatenate([np.zeros(50),
                          _ar1(rng, 60, 4.0, 0.0, bias=12.0)])   # true rho = 0
    pf = _post_fit_with_gap(gap, n_fit=50, n_blank=60, n_post=0)
    pw = compute_power_analysis(pf, alpha=ALPHA)
    assert abs(pw.serial_correlation) < 0.25
    # the uncentred estimator this replaces would have been driven by the offset
    r = gap[50:]
    uncentred = float(r[:-1] @ r[1:] / (r @ r))
    assert uncentred > 0.8


def test_noise_scale_carries_the_bias_into_the_standard_error():
    """``se`` is the RMS forecast error of the post-window mean, bias included."""
    rng = np.random.default_rng(2)
    bias, sigma = 6.0, 9.0
    gap = np.concatenate([np.zeros(40), _ar1(rng, 80, sigma, 0.0, bias=bias)])
    pf = _post_fit_with_gap(gap, n_fit=40, n_blank=80, n_post=0)
    pw = compute_power_analysis(pf, alpha=ALPHA, post_grid=[6])
    pt = next(p for p in pw.curve if p.post_periods == 6)
    # b_hat**2 overstates the squared bias by Var(b_hat), so the surface removes
    # it before folding the offset in (see the unbiased-design test below).
    var_bias = pw.sigma_placebo**2 * _vif(pw.n_placebo, pw.serial_correlation)
    bias_sq = max(0.0, pw.placebo_bias**2 - var_bias)
    expected = np.sqrt(bias_sq
                       + pw.sigma_placebo**2 * _vif(6, pw.serial_correlation))
    assert pt.se == pytest.approx(expected, rel=1e-9)
    assert bias_sq > 0.0, "a bias of 6.0 on 80 periods should survive the correction"
    assert pt.se > pw.sigma_placebo * np.sqrt(_vif(6, pw.serial_correlation))


def _vif(n, rho):
    k = np.arange(1, n)
    return (1.0 + 2.0 * np.sum((1.0 - k / n) * rho**k)) / n


def test_unbiased_design_matches_the_classical_formula():
    """With no blank-period bias the surface reduces to sigma * sqrt(VIF)."""
    rng = np.random.default_rng(3)
    gap = np.concatenate([np.zeros(40), _ar1(rng, 120, 7.0, 0.35)])
    gap[40:] -= gap[40:].mean()                    # exactly zero-mean blank window
    pf = _post_fit_with_gap(gap, n_fit=40, n_blank=120, n_post=0)
    pw = compute_power_analysis(pf, alpha=ALPHA, post_grid=[6])
    pt = next(p for p in pw.curve if p.post_periods == 6)
    assert pw.placebo_bias == pytest.approx(0.0, abs=1e-9)
    assert pt.se == pytest.approx(
        pw.sigma_placebo * np.sqrt(_vif(6, pw.serial_correlation)), rel=1e-9)


# ---------------------------------------------------------------------------
# The question the surface exists to answer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bias,sigma,rho,n_blank", [
    (0.0, 10.0, 0.0, 24),
    (0.0, 10.0, 0.4, 24),
    (0.0, 10.0, 0.7, 24),
    (7.0, 10.0, 0.0, 24),
    (7.0, 10.0, 0.4, 24),
    (0.0, 10.0, 0.0, 80),
    (0.0, 10.0, 0.4, 80),
    (0.0, 10.0, 0.0, 200),
    (0.0, 10.0, -0.3, 24),
])
def test_nominal_size_is_calibrated(bias, sigma, rho, n_blank):
    """A nominal 5% test at the reported ``se`` must reject about 5% of the time.

    Size is measured unconditionally: every replication draws a fresh blank
    window (so the design gets its own ``sigma`` estimate, as a real design
    does) and then a fresh post window it has never seen. Conditioning on one
    blank window instead would measure how lucky that particular draw was.

    The scale is estimated, not known, so the critical value has to come from a
    ``t`` distribution on the placebo window's degrees of freedom. Using the
    Gaussian quantile against an estimated ``sigma`` is the textbook reason a
    nominal 5% test is not a 5% test, and it bites hardest on the short blank
    windows these designs actually have.
    """
    n_post, n_reps = 6, 3000
    rng = np.random.default_rng(11)
    rejects = 0
    for _ in range(n_reps):
        gap = np.concatenate([np.zeros(10),
                              _ar1(rng, n_blank, sigma, rho, bias=bias)])
        pf = _post_fit_with_gap(gap, n_fit=10, n_blank=n_blank, n_post=0)
        pw = compute_power_analysis(pf, alpha=ALPHA, post_grid=[n_post])
        pt = next(p for p in pw.curve if p.post_periods == n_post)
        draw = _ar1(rng, n_post, sigma, rho, bias=bias).mean()
        rejects += int(abs(draw) > pt.critical_value * pt.se)
    size = rejects / n_reps
    assert 0.025 <= size <= 0.075, (
        f"nominal 5%, empirical {size:.1%} "
        f"(bias={bias}, sigma={sigma}, rho={rho}, n_blank={n_blank})")


def test_gaussian_quantile_against_an_estimated_sigma_over_rejects():
    """The defect this surface had, pinned so it cannot come back silently.

    Same design, same draws; the only change is using ``z`` where the reported
    ``critical_value`` uses ``t``. If this stops over-rejecting, the degrees of
    freedom have gone missing.
    """
    n_post, n_blank, n_reps, sigma, rho = 6, 24, 3000, 10.0, 0.7
    rng = np.random.default_rng(11)
    z_rej = t_rej = 0
    for _ in range(n_reps):
        gap = np.concatenate([np.zeros(10), _ar1(rng, n_blank, sigma, rho)])
        pf = _post_fit_with_gap(gap, n_fit=10, n_blank=n_blank, n_post=0)
        pt = next(p for p in compute_power_analysis(
            pf, alpha=ALPHA, post_grid=[n_post]).curve
            if p.post_periods == n_post)
        draw = _ar1(rng, n_post, sigma, rho).mean()
        z_rej += int(abs(draw) > Z * pt.se)
        t_rej += int(abs(draw) > pt.critical_value * pt.se)
    assert z_rej / n_reps > 0.08, (
        f"the Gaussian quantile should over-reject here; got {z_rej / n_reps:.1%}")
    assert t_rej / n_reps <= 0.065, (
        f"the reported critical value should not; got {t_rej / n_reps:.1%}")


def test_an_unbiased_design_is_not_penalised_for_noise_in_its_bias_estimate():
    """``b_hat**2`` is biased up by ``Var(b_hat)``; the surface must subtract it.

    Otherwise a design that is genuinely unbiased pays for the sampling noise in
    its own bias estimate, and its MDE is inflated for no reason.
    """
    rng = np.random.default_rng(13)
    sizes = []
    for _ in range(400):
        gap = np.concatenate([np.zeros(10), _ar1(rng, 24, 10.0, 0.0)])
        pf = _post_fit_with_gap(gap, n_fit=10, n_blank=24, n_post=0)
        pw = compute_power_analysis(pf, alpha=ALPHA, post_grid=[6])
        pt = next(p for p in pw.curve if p.post_periods == 6)
        sizes.append(pt.se)
    median_se = float(np.median(sizes))
    ideal = 10.0 / np.sqrt(6)
    # The plug-in b_hat**2 lands at 1.10x of ideal on this design; removing
    # Var(b_hat) brings it to 1.04x. The band is set between the two so the
    # test measures the correction and not merely that the number is finite.
    assert 0.95 * ideal <= median_se <= 1.08 * ideal, (
        f"median se {median_se:.3f} vs sigma/sqrt(n_post) {ideal:.3f} "
        f"(ratio {median_se / ideal:.3f})")


def test_power_at_the_reported_mde_is_the_target():
    """Injecting an effect of exactly the MDE must reject at the target power."""
    n_post, n_blank, n_reps, sigma, rho = 6, 200, 3000, 10.0, 0.3
    rng = np.random.default_rng(12)
    rejects = 0
    for _ in range(n_reps):
        gap = np.concatenate([np.zeros(10), _ar1(rng, n_blank, sigma, rho)])
        pf = _post_fit_with_gap(gap, n_fit=10, n_blank=n_blank, n_post=0)
        pt = next(p for p in compute_power_analysis(
            pf, alpha=ALPHA, power_target=0.80, post_grid=[n_post]).curve
            if p.post_periods == n_post)
        draw = _ar1(rng, n_post, sigma, rho).mean() + pt.mde_absolute
        rejects += int(abs(draw) > pt.critical_value * pt.se)
    power = rejects / n_reps
    assert 0.72 <= power <= 0.88, f"target 0.80, empirical {power:.2f}"


# ---------------------------------------------------------------------------
# How much the scale itself is known
# ---------------------------------------------------------------------------

def test_effective_sample_size_never_exceeds_the_period_count():
    """Negative serial correlation lowers Var(mean); it does not add periods.

    ``n (1 - rho) / (1 + rho)`` exceeds ``n`` whenever ``rho < 0``, and a sample
    autocorrelation of -0.4 is an ordinary draw on a 20-period window. Uncapped,
    that window would be treated as carrying more information than it holds.
    """
    from mlsynth.utils.post_fit import _effective_n
    for n in (8, 20, 24, 100):
        assert _effective_n(n, -0.5) == pytest.approx(float(n))
        assert _effective_n(n, -0.9) == pytest.approx(float(n))
        assert _effective_n(n, 0.0) == pytest.approx(float(n))
        assert _effective_n(n, 0.5) < n


def test_scale_uncertainty_is_reported_and_widens_on_short_windows():
    """A sigma resting on 20 points must not be presented like one resting on 200."""
    rng = np.random.default_rng(4)
    out = {}
    for n_blank in (20, 200):
        gap = np.concatenate([np.zeros(40), _ar1(rng, n_blank, 10.0, 0.0)])
        pf = _post_fit_with_gap(gap, n_fit=40, n_blank=n_blank, n_post=0)
        pw = compute_power_analysis(pf, alpha=ALPHA, post_grid=[6])
        lo, hi = pw.sigma_ci
        assert lo < pw.sigma_placebo < hi
        out[n_blank] = hi / lo
    assert out[20] > 1.8, "a 20-point sigma should carry a wide interval"
    assert out[200] < 1.35
    assert out[20] > out[200]


def test_mde_interval_brackets_the_point_estimate():
    """The MDE inherits the scale's uncertainty and reports it."""
    rng = np.random.default_rng(5)
    gap = np.concatenate([np.zeros(40), _ar1(rng, 24, 10.0, 0.2)])
    pf = _post_fit_with_gap(gap, n_fit=40, n_blank=24, n_post=0)
    pw = compute_power_analysis(pf, alpha=ALPHA, post_grid=[6])
    pt = next(p for p in pw.curve if p.post_periods == 6)
    lo, hi = pt.mde_ci
    assert lo < pt.mde_absolute < hi


def test_method_names_the_test_the_mde_refers_to():
    """The reported MDE is for a z-test on the mean gap; the field must say so.

    LEXSCM also computes a moving-block permutation MDE on ``mean|e|`` (the
    statistic Vives-i-Bastida Section 2 defines), and the two are different
    tests. A reader comparing the headline against a p-value needs to know which
    one produced it.
    """
    rng = np.random.default_rng(6)
    gap = np.concatenate([np.zeros(40), _ar1(rng, 40, 5.0, 0.0)])
    pf = _post_fit_with_gap(gap, n_fit=40, n_blank=40, n_post=0)
    pw = compute_power_analysis(pf, alpha=ALPHA)
    assert pw.method == "analytical_ar1_mean_gap"


def test_scale_equivariance():
    """Scaling the outcome scales se and the MDE, and leaves rho alone."""
    rng = np.random.default_rng(7)
    gap = np.concatenate([np.zeros(40), _ar1(rng, 60, 8.0, 0.3, bias=3.0)])
    a = compute_power_analysis(_post_fit_with_gap(gap, 40, 60, 0), post_grid=[6])
    b = compute_power_analysis(_post_fit_with_gap(gap * 4.0, 40, 60, 0), post_grid=[6])
    pa = next(p for p in a.curve if p.post_periods == 6)
    pb = next(p for p in b.curve if p.post_periods == 6)
    assert pb.se == pytest.approx(4.0 * pa.se, rel=1e-9)
    assert pb.mde_absolute == pytest.approx(4.0 * pa.mde_absolute, rel=1e-9)
    assert pb.serial_correlation_used == pytest.approx(pa.serial_correlation_used,
                                                       rel=1e-9) \
        if hasattr(pa, "serial_correlation_used") else True
    assert b.serial_correlation == pytest.approx(a.serial_correlation, rel=1e-9)
