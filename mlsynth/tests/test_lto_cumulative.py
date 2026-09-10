"""A cumulative interval whose calibration set does not involve the horizon.

Every cumulative band built on the time axis thins as the horizon grows: split
conformal needs ``ceil((m+1)(1-alpha)) <= m`` non-overlapping windows of length
``H`` from a fixed pre-period, so ``m`` falls as ``H`` rises, and a block
resample keeps reporting only because its period count hides a collapsing origin
count.

The leave-two-out reference set is ``C(J, 2)`` pairs of donor units. Lei &
Sudijono (2025) section 6.4 -- the theory "only relies on the uniform assignment
assumption but not the choice of summary statistics" -- is what lets the pair
statistic be the cumulative post-period total. The reference set is then counted
in donors and the horizon does not enter it.

One detail decides whether this is the right construction. The pair's spread has
to be the larger of the two left-out donors' *cumulative* absolute residuals,
``max(|sum r_a|, |sum r_b|)``. Accumulating the pointwise maxima instead --
``sum(max(|r_a|, |r_b|))`` -- fixes the cross-period correlation at one and is
the comonotone endpoint sum in a new costume. The two are pinned apart below.
"""
import numpy as np
import pytest

from mlsynth.utils.vanillasc_helpers import BilevelSCM
from mlsynth.utils.vanillasc_helpers.lto import lto_cumulative_interval, lto_interval


def _panel(seed=3, T=24, T0=16, J=7, effect=0.0):
    rng = np.random.default_rng(seed)
    Y0 = rng.normal(5, 1, size=(T, J))
    w = np.zeros(J)
    w[[0, 2, 5]] = [0.5, 0.3, 0.2]
    y = Y0 @ w + 0.05 * rng.normal(size=T)
    y[T0:] += effect
    return y, Y0, T0


def _run(*, effect=0.0, alpha=0.10, **kw):
    y, Y0, T0 = _panel(effect=effect, **kw)
    out = lto_cumulative_interval(BilevelSCM("outcome-only", seed=0), y, Y0, T0,
                                  alpha=alpha, seed=0)
    return out, y, T0


# ---- shape -----------------------------------------------------------------

def test_the_interval_is_a_pair_of_scalars_in_order():
    out, y, T0 = _run()
    for k in ("lower", "upper", "effect_lower", "effect_upper", "observed_total"):
        assert np.isscalar(out[k]) or np.ndim(out[k]) == 0, k
        assert np.isfinite(out[k]), k
    assert out["lower"] <= out["upper"]
    assert out["effect_lower"] <= out["effect_upper"]
    assert out["observed_total"] == pytest.approx(y[T0:].sum())


def test_the_effect_bounds_are_the_counterfactual_bounds_reflected():
    """effect = observed - counterfactual, so the bounds swap ends."""
    out, _, _ = _run()
    assert out["effect_lower"] == pytest.approx(out["observed_total"] - out["upper"])
    assert out["effect_upper"] == pytest.approx(out["observed_total"] - out["lower"])


# ---- the property the construction exists for ------------------------------

@pytest.mark.parametrize("T", [20, 26, 34])
def test_the_reference_set_does_not_involve_the_horizon(T):
    """4, 10 and 18 post periods; the same 21 pairs calibrate all three."""
    out, _, _ = _run(T=T)
    assert out["n_pairs"] == 7 * 6 // 2
    assert out["horizon"] == T - 16


# ---- the construction ------------------------------------------------------

def test_the_spread_accumulates_before_it_takes_the_maximum():
    """``max(|sum r|)``, not ``sum(max|r|)``.

    Taking the maximum period by period and then accumulating would assume the
    two donors' errors line up in sign across every period -- the comonotone
    assumption. The faithful statistic accumulates each donor first.
    """
    out, _, _ = _run()
    cum_a = out["pair_resid_i"].sum(axis=1)
    cum_b = out["pair_resid_j"].sum(axis=1)
    assert np.allclose(out["pair_spreads"],
                       np.maximum(np.abs(cum_a), np.abs(cum_b)))

    comonotone = np.maximum(np.abs(out["pair_resid_i"]),
                            np.abs(out["pair_resid_j"])).sum(axis=1)
    assert not np.allclose(out["pair_spreads"], comonotone), (
        "the fixture is degenerate: the two constructions agree here, so this "
        "test would pass on either")
    assert (out["pair_spreads"] <= comonotone + 1e-9).all()


def test_at_a_one_period_horizon_it_is_the_pointwise_interval():
    """A cumulative total over one period is that period."""
    y, Y0, _ = _panel(T=24)
    eng = BilevelSCM("outcome-only", seed=0)
    pre = 23                                    # a single post period
    cum = lto_cumulative_interval(eng, y, Y0, pre, alpha=0.10, seed=0)
    pt = lto_interval(eng, y, Y0, pre, alpha=0.10, seed=0)
    assert cum["lower"] == pytest.approx(pt["lower"][-1])
    assert cum["upper"] == pytest.approx(pt["upper"][-1])


def test_the_quantile_is_an_order_statistic():
    out, _, _ = _run()
    hi = out["pair_centres"] + out["pair_spreads"]
    lo = out["pair_centres"] - out["pair_spreads"]
    assert out["upper"] in hi
    assert out["lower"] in lo


# ---- behaviour under a known effect ----------------------------------------

def test_a_planted_effect_moves_the_interval_off_zero():
    null_out, _, _ = _run(effect=0.0)
    hit_out, _, _ = _run(effect=-8.0)
    assert null_out["effect_lower"] <= 0.0 <= null_out["effect_upper"]
    assert hit_out["effect_upper"] < null_out["effect_upper"]


# ---- failures --------------------------------------------------------------

def test_no_post_periods_is_refused():
    y, Y0, _ = _panel()
    with pytest.raises(ValueError, match="post"):
        lto_cumulative_interval(BilevelSCM("outcome-only", seed=0), y, Y0, len(y))


def test_an_alpha_finer_than_the_pair_set_can_resolve_is_refused():
    y, Y0, T0 = _panel()
    with pytest.raises(ValueError, match="alpha"):
        lto_cumulative_interval(BilevelSCM("outcome-only", seed=0), y, Y0, T0,
                                alpha=0.001)


def test_fewer_than_three_donors_is_refused():
    rng = np.random.default_rng(0)
    Y0 = rng.normal(5, 1, size=(12, 2))
    y = Y0 @ np.array([0.6, 0.4]) + 0.05 * rng.normal(size=12)
    with pytest.raises(ValueError, match="donor"):
        lto_cumulative_interval(BilevelSCM("outcome-only"), y, Y0, 8)


def test_an_alpha_outside_the_unit_interval_is_refused():
    y, Y0, T0 = _panel()
    eng = BilevelSCM("outcome-only", seed=0)
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="alpha must lie"):
            lto_cumulative_interval(eng, y, Y0, T0, alpha=bad)
