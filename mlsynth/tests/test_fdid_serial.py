"""Serially correlated errors: the design, and what they cost Li's variance.

Li (2023) Proposition 2.1 studentises the ATT by the *marginal* variance of
the parallel-trends residual, ``sigma^2 = E[v_t^2]``. The sampling error of
the estimator is a difference of two block means of that residual, whose
variance is its *long-run* variance. The two coincide exactly when ``v_t``
is serially uncorrelated, which the appendix's Assumptions 2(ii) and 3(i)
impose. Assumption 2.1 in the main text asks only for weak dependence.

These test the two pieces that let a benchmark measure that gap: a DGP in
which ``v_t`` is AR(1) at the optimal donor subset, and the closed-form
prediction of how far the reported standard error then falls short.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.fdid_helpers.population import long_run_inflation
from mlsynth.utils.fdid_helpers.simulation import (
    FDIDSample,
    simulate_fdid_serial_sample,
)


# --------------------------------------------------------------------------
# long_run_inflation -- the closed form
# --------------------------------------------------------------------------

def test_no_serial_correlation_means_no_inflation():
    """At rho = 0 the marginal variance IS the long-run variance."""
    assert long_run_inflation(0.0, n=10, T1=400, T2=10) == pytest.approx(1.0)


@pytest.mark.parametrize("T1,T2", [(400, 10), (100, 25), (50, 50)])
def test_inflation_rises_with_rho(T1, T2):
    values = [long_run_inflation(r, n=10, T1=T1, T2=T2)
              for r in (0.0, 0.2, 0.4, 0.6, 0.8)]
    assert values == sorted(values)
    assert values[0] == pytest.approx(1.0)
    assert values[-1] > 1.3


def test_inflation_matches_an_independent_double_sum():
    """Cross-check the closed form against the autocovariance matrix.

    ``long_run_inflation`` sums the triangular weights; this builds the full
    ``T x T`` covariance of the block mean and sums it, which is the same
    quantity by a different route.
    """
    rho, n, T1, T2 = 0.5, 10, 60, 15
    g0 = 1.0 + 1.0 / n

    def block_var(T):
        idx = np.arange(T)
        cov = rho ** np.abs(idx[:, None] - idx[None, :]).astype(float)
        cov[idx, idx] = g0                      # the diagonal carries 1 + 1/n
        return cov.sum() / T ** 2

    expected = np.sqrt((block_var(T1) + block_var(T2))
                       / (g0 * (1.0 / T1 + 1.0 / T2)))
    assert long_run_inflation(rho, n=n, T1=T1, T2=T2) == pytest.approx(expected)


@pytest.mark.parametrize("rho", [0.0, 0.4, 0.8])
def test_inflation_matches_monte_carlo_on_the_residual(rho):
    """The prediction against simulated residuals, with no estimator involved.

    Draws the residual process directly, forms the estimator's sampling
    error -- the difference of the two block means -- and compares its
    dispersion, scaled by what Li's formula would predict, against the
    closed form. 4000 draws puts the Monte-Carlo error on a dispersion near
    2% of it, so the tolerance is 5%.
    """
    n, T1, T2, M = 10, 200, 20, 4000
    rng = np.random.default_rng(4)
    T = T1 + T2
    scale = np.sqrt(1 - rho ** 2) if rho else 1.0
    errs = np.empty(M)
    for j in range(M):
        e = rng.normal(0.0, scale, T)
        u = np.empty(T)
        u[0] = rng.normal()
        for t in range(1, T):
            u[t] = rho * u[t - 1] + e[t]
        v = u + rng.normal(0.0, np.sqrt(1.0 / n), T)
        errs[j] = v[T1:].mean() - v[:T1].mean()
    li_sd = np.sqrt((1.0 + 1.0 / n) * (1.0 / T1 + 1.0 / T2))
    assert errs.std() / li_sd == pytest.approx(
        long_run_inflation(rho, n=n, T1=T1, T2=T2), rel=0.05)


@pytest.mark.parametrize("bad", [1.0, -1.0, 1.5, -2.0])
def test_non_stationary_rho_raises(bad):
    with pytest.raises(ValueError, match="rho must be"):
        long_run_inflation(bad, n=10, T1=100, T2=10)


@pytest.mark.parametrize("kwargs", [
    dict(n=0, T1=100, T2=10),
    dict(n=10, T1=0, T2=10),
    dict(n=10, T1=100, T2=0),
])
def test_degenerate_dimensions_raise(kwargs):
    with pytest.raises(ValueError):
        long_run_inflation(0.5, **kwargs)


# --------------------------------------------------------------------------
# simulate_fdid_serial_sample -- the design
# --------------------------------------------------------------------------

def test_serial_sample_smoke():
    s = simulate_fdid_serial_sample(rho=0.5, N=20, T1=50, T2=10,
                                    rng=np.random.default_rng(0))
    assert isinstance(s, FDIDSample)
    assert s.Y_treated.shape == (60,)
    assert s.Y_controls.shape == (20, 60)
    assert set(s.df.columns) == {"unit", "time", "y", "treat"}
    assert s.df["treat"].sum() == 10          # treated unit, post-periods only


@pytest.mark.parametrize("rho", [0.0, 0.5, 0.9])
def test_treated_idiosyncratic_term_has_unit_variance(rho):
    """Only the dependence changes with rho, not the size of the shock.

    Otherwise a coverage drop could be the error growing instead of its
    autocorrelation, and the design would not isolate what it is meant to.
    """
    rng = np.random.default_rng(1)
    sds = []
    for _ in range(40):
        s = simulate_fdid_serial_sample(rho=rho, N=20, T1=2000, T2=10, rng=rng)
        # At the matched half the factor term cancels, leaving u_t + noise.
        v = s.Y_treated[:2000] - s.Y_controls[:10, :2000].mean(axis=0)
        sds.append(v.var() - 1.0 / 10)         # strip the donor-average term
    assert np.mean(sds) == pytest.approx(1.0, rel=0.05)


@pytest.mark.parametrize("rho", [0.0, 0.3, 0.6, 0.9])
def test_residual_autocorrelation_is_rho(rho):
    """The lag-1 autocorrelation of the treated shock is rho by construction."""
    rng = np.random.default_rng(2)
    acs = []
    for _ in range(30):
        s = simulate_fdid_serial_sample(rho=rho, N=20, T1=3000, T2=10, rng=rng)
        u = s.Y_treated[:3000] - s.Y_controls[:10, :3000].mean(axis=0)
        u = u - u.mean()
        acs.append(float(np.corrcoef(u[:-1], u[1:])[0, 1]))
    # The donor average adds iid noise of variance 1/n, which attenuates the
    # measured autocorrelation by 1 / (1 + 1/n).
    assert np.mean(acs) == pytest.approx(rho / (1.0 + 1.0 / 10), abs=0.02)


def test_control_pool_splits_into_matched_and_mismatched_halves():
    """Same loading structure as Web Appendix E DGP 2: c1 = c0 = 1, c2 = 2.

    The mismatched half is what gives the forward search something to
    exclude, so that the selected subset is the matched half and the factor
    term drops out of the residual.
    """
    s = simulate_fdid_serial_sample(rho=0.0, N=20, T1=400, T2=10,
                                    rng=np.random.default_rng(3))
    matched = s.Y_controls[:10].mean(axis=0)
    mismatched = s.Y_controls[10:].mean(axis=0)
    # Regressing each half's average on the treated series recovers the
    # loading ratio: 1 for the matched half, 2 for the mismatched one.
    for series, target in ((matched, 1.0), (mismatched, 2.0)):
        slope = np.polyfit(s.Y_treated - s.Y_treated.mean(),
                           series - series.mean(), 1)[0]
        assert slope == pytest.approx(target, rel=0.15)


@pytest.mark.parametrize("bad", [1.0, -1.0, 2.0])
def test_serial_sample_rejects_non_stationary_rho(bad):
    with pytest.raises(ValueError, match="rho must be"):
        simulate_fdid_serial_sample(rho=bad, N=20, T1=50, T2=10,
                                    rng=np.random.default_rng(0))
