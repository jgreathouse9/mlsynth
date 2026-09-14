"""Pointwise confidence intervals from the leave-two-out reference set.

Lei & Sudijono (2025). The reference implementation
(``tsudijon/LeaveTwoOutSCI``) builds the interval per period from the pair set,
in ``basque_analysis/slurm/basque_ltojk_poweranalysis_slurm.R``:

    LTO.residuals[pair,] = pmax(abs(res1), abs(res2))
    ci.upper[t] = quantile(LTO.regression[,t] + LTO.residuals[,t], 1-alpha, type=1)
    ci.lower[t] = quantile(LTO.regression[,t] - LTO.residuals[,t],   alpha, type=1)

Each pair contributes its own counterfactual for the treated unit -- the pool
differs pair to pair -- so the interval is a quantile over pair-specific centres
and not a fixed centre with a spread around it.

What matters for this port, and what these tests hold:

  * the reference set is ``C(J, 2)`` pairs of donor units, so its size does not
    involve the post-period length. That is the property that makes this usable
    where a calibration set drawn from the time axis runs out.
  * ``quantile(type = 1)`` is R's inverse-ECDF, which returns an order statistic
    and not an interpolation between two. NumPy's default interpolates, so the
    two disagree on small pair sets -- exactly the regime this method is for.
"""
import numpy as np
import pytest

from mlsynth.utils.vanillasc_helpers import BilevelSCM
from mlsynth.utils.vanillasc_helpers.lto import lto_interval


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
    return lto_interval(BilevelSCM("outcome-only", seed=0), y, Y0, T0,
                        alpha=alpha, seed=0), y, T0


# ---- shape and orientation -------------------------------------------------

def test_interval_spans_the_panel_and_is_ordered():
    out, y, _ = _run()
    assert out["lower"].shape == out["upper"].shape == y.shape
    assert np.isfinite(out["lower"]).all() and np.isfinite(out["upper"]).all()
    assert (out["lower"] <= out["upper"]).all()


def test_the_reference_set_is_every_pair_of_donors():
    out, _, _ = _run()
    J = 7
    assert out["n_pairs"] == J * (J - 1) // 2
    assert out["N"] == J + 1


# ---- the property the method is being adopted for --------------------------

@pytest.mark.parametrize("T", [20, 26, 34])
def test_the_reference_set_does_not_shrink_as_the_post_period_grows(T):
    """C(J, 2) pairs at every horizon.

    A calibration set cut from the time axis loses windows as the horizon grows;
    this one is counted in donors, so it does not. Same pre-period, same donors,
    a post-period from 4 to 18 periods long.
    """
    out, _, _ = _run(T=T)
    assert out["n_pairs"] == 7 * 6 // 2
    assert out["lower"].shape == (T,)


# ---- the recipe ------------------------------------------------------------

def test_the_quantile_is_an_order_statistic_not_an_interpolation():
    """R's ``type = 1`` against NumPy's default, on the shipped result.

    Every reported bound has to be a value the pair set actually produced. With
    21 pairs at alpha = 0.10 the two conventions differ, so this fails if the
    port silently uses the interpolating default.
    """
    out, _, _ = _run()
    centres, spreads = out["pair_centres"], out["pair_spreads"]
    for t in range(centres.shape[1]):
        hi = centres[:, t] + spreads[:, t]
        lo = centres[:, t] - spreads[:, t]
        assert out["upper"][t] in hi, f"upper[{t}] is not one of the pair values"
        assert out["lower"][t] in lo, f"lower[{t}] is not one of the pair values"


def test_the_spread_is_the_larger_of_the_two_left_out_donors():
    """``pmax(abs(res1), abs(res2))``, per period, per pair."""
    out, _, _ = _run()
    assert (out["pair_spreads"] >= 0).all()
    assert np.array_equal(
        out["pair_spreads"],
        np.maximum(np.abs(out["pair_resid_i"]), np.abs(out["pair_resid_j"])))


# ---- behaviour under a known effect ----------------------------------------

def test_a_null_panel_is_mostly_covered_and_a_large_effect_is_not():
    """Direction, not a coverage rate: one panel cannot measure coverage."""
    null_out, y_null, T0 = _run(effect=0.0)
    inside_null = ((y_null >= null_out["lower"]) & (y_null <= null_out["upper"]))

    hit_out, y_hit, _ = _run(effect=-8.0)
    inside_hit = ((y_hit >= hit_out["lower"]) & (y_hit <= hit_out["upper"]))

    assert inside_null[T0:].mean() > inside_hit[T0:].mean()


# ---- failures --------------------------------------------------------------

def test_fewer_than_three_donors_is_refused():
    rng = np.random.default_rng(0)
    Y0 = rng.normal(5, 1, size=(12, 2))
    y = Y0 @ np.array([0.6, 0.4]) + 0.05 * rng.normal(size=12)
    with pytest.raises(ValueError, match="donor"):
        lto_interval(BilevelSCM("outcome-only"), y, Y0, 8)


def test_an_alpha_finer_than_the_pair_set_can_resolve_is_refused():
    """21 pairs cannot place a 1% bound: the order statistic does not exist."""
    y, Y0, T0 = _panel()
    with pytest.raises(ValueError, match="alpha"):
        lto_interval(BilevelSCM("outcome-only", seed=0), y, Y0, T0, alpha=0.001)


def test_an_alpha_outside_the_unit_interval_is_refused():
    y, Y0, T0 = _panel()
    eng = BilevelSCM("outcome-only", seed=0)
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="alpha must lie"):
            lto_interval(eng, y, Y0, T0, alpha=bad)


def test_capping_the_pair_count_subsamples_deterministically():
    """``max_pairs`` trades reference set for cost, and says that it did.

    The cap changes the reference set, so it changes the guarantee -- the flag is
    how a caller finds out. Two runs at one seed agree; a different seed draws a
    different subsample.
    """
    y, Y0, T0 = _panel()
    eng = BilevelSCM("outcome-only", seed=0)
    a = lto_interval(eng, y, Y0, T0, alpha=0.10, max_pairs=12, seed=0)
    b = lto_interval(eng, y, Y0, T0, alpha=0.10, max_pairs=12, seed=0)
    c = lto_interval(eng, y, Y0, T0, alpha=0.10, max_pairs=12, seed=1)

    assert a["n_pairs"] == 12 and a["subsampled"] is True
    assert np.array_equal(a["lower"], b["lower"])
    assert not np.array_equal(a["pair_centres"], c["pair_centres"])
    full = lto_interval(eng, y, Y0, T0, alpha=0.10)
    assert full["subsampled"] is False and full["n_pairs"] == 21


def test_the_cold_path_reaches_the_same_pairs():
    """``warm_start`` picks where the active set starts, not where it lands."""
    y, Y0, T0 = _panel()
    eng = BilevelSCM("outcome-only", seed=0)
    warm = lto_interval(eng, y, Y0, T0, alpha=0.10, warm_start=True)
    cold = lto_interval(eng, y, Y0, T0, alpha=0.10, warm_start=False)
    assert warm["n_pairs"] == cold["n_pairs"]
    assert np.allclose(warm["lower"], cold["lower"], atol=1e-6)
    assert np.allclose(warm["upper"], cold["upper"], atol=1e-6)


def test_capping_the_pairs_raises_the_level_the_interval_can_reach():
    """The cap is not free: fewer pairs is a coarser reachable alpha.

    21 pairs resolve alpha = 0.10; 8 do not, since 1/8 exceeds it. The refusal
    names the reachable level so a caller can choose between the cap and the
    level instead of guessing.
    """
    y, Y0, T0 = _panel()
    eng = BilevelSCM("outcome-only", seed=0)
    assert lto_interval(eng, y, Y0, T0, alpha=0.10)["n_pairs"] == 21
    with pytest.raises(ValueError, match=r"tightest reachable level here is alpha=0\.1250"):
        lto_interval(eng, y, Y0, T0, alpha=0.10, max_pairs=8)
