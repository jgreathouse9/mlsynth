r"""PDA's cumulative band and the law its post-period errors are drawn from.

``pda_prediction_intervals`` implements Algorithm 2.1 of Jiang, Li, Shen & Zhou
(2025): a dependent wild bootstrap for the pre-period prediction error, and a
simple residual bootstrap for the out-of-sample error. That pairing is right for
what the paper targets. A per-period prediction interval needs each post
period's marginal, and an i.i.d. draw from the centred pre-period residuals
gives it.

mlsynth's cumulative band is an extension the paper does not make, and it reuses
those same paths. ``cumulative_supt_band`` accumulates them and takes the
standard error after, so the band inherits whatever correlation the period
errors carry. Drawn independently they carry none, and the running total's
uncertainty then grows like :math:`\sqrt{L}` whatever the series does.

The arithmetic that separates the two: a total of :math:`H` errors has variance

.. math::

   \sigma^2 H \quad\text{(independent)}, \qquad
   \sigma^2 \Bigl[H + 2\sum_{k=1}^{H-1}(H-k)\rho^k\Bigr] \quad\text{(AR}(\rho)\text{)},

so at :math:`\rho = 0.6` over six periods the honest standard error is 1.68
times the independent one, and at :math:`\rho = 0.8` it is 2.02 times.

These tests pin the post-period draw directly. A ``refit`` that returns the
fitted counterfactual whatever bootstrap sample it is handed makes the
extrapolation error identically zero, so ``error_paths[b]`` is exactly the
post-period draw and its accumulated spread is exactly what is under test --
nothing from the estimator is mixed in.

Levels: smoke, unit invariants, regression guard on the paper's own intervals,
edge, failure.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.inferutils import pda_prediction_intervals

T0, T1, P = 60, 6, 3
_BETA = np.array([1.0, -0.5, 0.3])


def _panel(rho: float, seed: int = 0):
    """Design, counterfactual and an AR(``rho``) error series of unit variance."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T0 + T1, P))
    e = rng.standard_normal(T0 + T1)
    if rho:
        for t in range(1, T0 + T1):
            e[t] = rho * e[t - 1] + np.sqrt(1.0 - rho ** 2) * e[t]
    cf = X @ _BETA
    return X, cf, cf + e, e


def _frozen_refit(cf):
    """A refit that ignores its argument, so the extrapolation error is zero."""
    def refit(y_boot):
        return cf.copy(), np.arange(P)
    return refit


def _paths(rho, *, seed=1, n_boot=4000, **kw):
    X, cf, y, e = _panel(rho)
    out = pda_prediction_intervals(
        y, X, T0, counterfactual=cf, support=np.arange(P),
        refit=_frozen_refit(cf), alpha=0.10, n_boot=n_boot, seed=seed, **kw)
    E = np.asarray(out["error_paths"], dtype=float)
    return out, E[np.isfinite(E).all(axis=1)], e


def _independent_total_sd(e):
    """The spread a running total would have if the periods were independent."""
    return float(np.std(e[:T0])) * np.sqrt(T1)


def _ar_inflation(rho, H=T1):
    v = H + 2.0 * sum((H - k) * rho ** k for k in range(1, H))
    return float(np.sqrt(v / H))


# --------------------------------------------------------------------------- #
# smoke                                                                        #
# --------------------------------------------------------------------------- #

def test_the_paths_still_have_one_column_per_post_period():
    _, E, _ = _paths(0.6, n_boot=200)
    assert E.shape[1] == T1
    assert E.shape[0] > 1
    assert np.isfinite(E).all()


# --------------------------------------------------------------------------- #
# unit invariants -- the fault                                                 #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("rho", [0.3, 0.6, 0.8])
def test_the_cumulative_spread_follows_the_series_autocorrelation(rho):
    """The running total's spread must grow with persistence, not with sqrt(H).

    This is the fault. Drawing the post-period errors independently leaves the
    accumulated spread at the independent value however persistent the series
    is, so a cumulative band built on them is too narrow by the AR inflation.
    """
    _, E, e = _paths(rho)
    measured = float(np.std(E.sum(axis=1))) / _independent_total_sd(e)
    expected = _ar_inflation(rho)
    assert measured == pytest.approx(expected, rel=0.15), (
        f"rho={rho}: accumulated spread is {measured:.3f} times the independent "
        f"value, expected about {expected:.3f}"
    )


def test_an_independent_series_is_not_widened():
    """The correction has to be adaptive: no persistence, no inflation."""
    _, E, e = _paths(0.0)
    measured = float(np.std(E.sum(axis=1))) / _independent_total_sd(e)
    assert measured == pytest.approx(1.0, rel=0.10)


def test_the_spread_is_monotone_in_persistence():
    ratios = []
    for rho in (0.0, 0.3, 0.6, 0.8):
        _, E, e = _paths(rho)
        ratios.append(float(np.std(E.sum(axis=1))) / _independent_total_sd(e))
    assert all(a < b for a, b in zip(ratios, ratios[1:])), ratios


def test_more_draws_do_not_repair_it():
    """The fault is the law being resampled, so n_boot cannot reach it."""
    small = _paths(0.8, n_boot=500)
    large = _paths(0.8, n_boot=8000)
    r_small = float(np.std(small[1].sum(axis=1))) / _independent_total_sd(small[2])
    r_large = float(np.std(large[1].sum(axis=1))) / _independent_total_sd(large[2])
    assert r_small == pytest.approx(r_large, rel=0.15)
    assert r_large == pytest.approx(_ar_inflation(0.8), rel=0.15)


# --------------------------------------------------------------------------- #
# the knob                                                                     #
# --------------------------------------------------------------------------- #

def test_a_block_of_one_reproduces_the_independent_draw():
    """``cumulative_block=1`` is the old behaviour, kept reachable on purpose."""
    _, E, e = _paths(0.8, cumulative_block=1)
    measured = float(np.std(E.sum(axis=1))) / _independent_total_sd(e)
    assert measured == pytest.approx(1.0, rel=0.10)


def test_a_block_longer_than_the_horizon_is_clamped():
    a = _paths(0.6, cumulative_block=T1)[1]
    b = _paths(0.6, cumulative_block=T1 * 5)[1]
    assert float(np.std(a.sum(axis=1))) == pytest.approx(
        float(np.std(b.sum(axis=1))), rel=0.10)


def test_zero_means_the_whole_horizon():
    a = _paths(0.6, cumulative_block=0)[1]
    b = _paths(0.6, cumulative_block=T1)[1]
    assert float(np.std(a.sum(axis=1))) == pytest.approx(
        float(np.std(b.sum(axis=1))), rel=0.10)


# --------------------------------------------------------------------------- #
# regression guard -- the paper's own intervals must not move                  #
# --------------------------------------------------------------------------- #

def test_the_per_period_intervals_are_untouched_by_the_post_draw():
    """Algorithm 2.1's studentized statistic keeps its i.i.d. residual draw.

    The cumulative band's input is a separate draw off the same refit, so the
    per-period intervals this repo validates against the paper cannot move when
    the block length changes.
    """
    base = _paths(0.6, cumulative_block=1)[0]
    wide = _paths(0.6, cumulative_block=0)[0]
    for block in ("effect", "counterfactual"):
        for key in ("point", "eq_lower", "eq_upper", "sy_lower", "sy_upper"):
            np.testing.assert_allclose(
                np.asarray(base[block][key], dtype=float),
                np.asarray(wide[block][key], dtype=float),
                rtol=0.0, atol=0.0,
                err_msg=f"{block}.{key} moved when cumulative_block changed")


def test_the_same_seed_still_gives_the_same_paths():
    a = _paths(0.6, seed=7, n_boot=300)[1]
    b = _paths(0.6, seed=7, n_boot=300)[1]
    np.testing.assert_allclose(a, b, rtol=0.0, atol=0.0)


# --------------------------------------------------------------------------- #
# edge                                                                         #
# --------------------------------------------------------------------------- #

def test_a_single_post_period_has_no_dependence_to_carry():
    """With one horizon a total is one draw, so every block length agrees."""
    rng = np.random.default_rng(3)
    X = rng.standard_normal((T0 + 1, P))
    cf = X @ _BETA
    y = cf + rng.standard_normal(T0 + 1)
    kw = dict(counterfactual=cf, support=np.arange(P), refit=_frozen_refit(cf),
              alpha=0.10, n_boot=400, seed=2)
    a = pda_prediction_intervals(y, X, T0, cumulative_block=0, **kw)
    b = pda_prediction_intervals(y, X, T0, cumulative_block=1, **kw)
    for out in (a, b):
        assert np.asarray(out["error_paths"], dtype=float).shape[1] == 1
    assert float(np.std(np.asarray(a["error_paths"], float))) == pytest.approx(
        float(np.std(np.asarray(b["error_paths"], float))), rel=0.15)


# --------------------------------------------------------------------------- #
# failure                                                                      #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("bad", [-1, -5, 1.5, "3", None, True])
def test_a_bad_block_length_is_refused(bad):
    X, cf, y, _ = _panel(0.3)
    with pytest.raises(MlsynthConfigError):
        pda_prediction_intervals(
            y, X, T0, counterfactual=cf, support=np.arange(P),
            refit=_frozen_refit(cf), alpha=0.10, n_boot=50, seed=0,
            cumulative_block=bad)
