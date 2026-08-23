"""The SBC simulation port, against what its DGPs are defined to produce.

``simulate_shi_xi_xie`` re-implements Shi-Xi-Xie's Models 1-3 in Python. The
Path B benchmark no longer calls it -- ``sbc_mc`` estimates on panels drawn in R
by the authors' own ``simulation_nonnegative.R`` -- so nothing else exercises it,
and these tests are what keeps it honest.

They pin the moments the DGPs are defined by, with the tolerance measured from
the draws' own spread instead of chosen: a statistic is accepted when it sits
within four standard errors of its analytic value, and the standard error comes
from the replications in the same test. That keeps the assertions sensitive to a
wrong DGP and indifferent to which draw came up.

The port is also checked against the authors' R directly. At 400 replications
each, R's construction and this one agree on the stationary factor variance and
on the mean squared increment to under 1%, both matching the analytic values
below. The 60 captured replications in ``benchmarks/reference/sbc_mc/`` sit two
to three standard errors under those values for Models 2 and 3, which is the
skew of a statistic built on squared factor loadings at that replication count.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.sbc_helpers.simulation import simulate_shi_xi_xie

NU, T0, H, PHI = 12, 100, 2, 0.5
REPS = 400


def _draws(model: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return [simulate_shi_xi_xie(model, T0, H=H, NU=NU, phi=PHI, rng=rng)
            for _ in range(REPS)]


def _within_four_se(sample: np.ndarray, analytic: float) -> bool:
    se = float(np.std(sample, ddof=1) / np.sqrt(sample.size))
    return abs(float(sample.mean()) - analytic) <= 4.0 * se


@pytest.mark.parametrize("model", [1, 2, 3])
def test_draw_shape_and_finiteness(model):
    """Smoke: an untreated ``(NU, T0 + H)`` panel, row 0 the treated unit."""
    Y = simulate_shi_xi_xie(model, T0, H=H, NU=NU, phi=PHI,
                            rng=np.random.default_rng(1))
    assert Y.shape == (NU, T0 + H)
    assert np.all(np.isfinite(Y))


@pytest.mark.parametrize("model", [1, 2, 3])
def test_draws_are_reproducible_from_the_generator(model):
    """Same generator state, same panel -- what makes a seeded study repeatable."""
    a = simulate_shi_xi_xie(model, T0, H=H, NU=NU, phi=PHI,
                            rng=np.random.default_rng(7))
    b = simulate_shi_xi_xie(model, T0, H=H, NU=NU, phi=PHI,
                            rng=np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_model1_increments_carry_the_drift_variance():
    """Model 1 is a random walk with drift mu_i ~ N(0, 1/4) and unit innovations,
    so an increment has mean square Var(mu) + 1 = 1.25."""
    stat = np.array([np.mean(np.diff(Y, axis=1) ** 2) for Y in _draws(1)])
    assert _within_four_se(stat, 0.25 + 1.0)


def test_model2_increments_carry_the_two_common_factors():
    """Model 2's increments are Lambda' f_t + u_t with two AR(phi) factors and
    standard normal loadings, so their mean square is 2 / (1 - phi^2) + 1."""
    stat = np.array([np.mean(np.diff(Y, axis=1) ** 2) for Y in _draws(2)])
    assert _within_four_se(stat, 2.0 / (1.0 - PHI ** 2) + 1.0)


def test_model2_units_share_their_short_run_movement():
    """The common factors are what SBC's cycle matching has to find: without
    them the cross-unit correlation of increments would be zero up to sampling
    error, and Model 2 would be Model 1 with extra steps."""
    corr = []
    for Y in _draws(2, seed=3)[:100]:
        C = np.corrcoef(np.diff(Y, axis=1))
        corr.append(np.mean(np.abs(C[~np.eye(NU, dtype=bool)])))
    assert np.mean(corr) > 0.25


def test_model3_splits_the_panel_into_two_structures():
    """Partial cointegration: the first half is a level factor plus a stationary
    component, the second half is Model 2's cumulated increments. Differencing a
    stationary component leaves a negative lag-one autocorrelation, cumulated
    increments do not, so the halves separate on that sign.
    """
    half = NU // 2
    first, second = [], []
    for Y in _draws(3, seed=5)[:100]:
        d = np.diff(Y, axis=1)
        d = d - d.mean(axis=1, keepdims=True)
        rho = np.array([np.corrcoef(row[:-1], row[1:])[0, 1] for row in d])
        first.append(rho[:half].mean())
        second.append(rho[half:].mean())
    assert np.mean(first) < -0.2
    assert np.mean(second) > np.mean(first)


def test_unknown_model_is_rejected():
    """A model number the DGP does not define raises, and does not return a
    panel drawn from whichever branch happened to fall through."""
    with pytest.raises(ValueError, match="model must be 1, 2 or 3"):
        simulate_shi_xi_xie(4, T0, H=H, NU=NU, phi=PHI,
                            rng=np.random.default_rng(0))
