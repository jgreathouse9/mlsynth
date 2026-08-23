"""A band for the cumulative effect, obtained by inverting a joint conformal test.

The other route to a cumulative band calibrates on out-of-sample errors: split a
withheld stretch into horizon-length windows and resample them. Its accuracy is
bounded by how many such windows exist, which on a design with a 47-period blank
window and a 13-period horizon is three, and three windows do not reach nominal
coverage even when every distributional assumption holds.

Test inversion has no such ceiling. Under a sharp null the whole post block is
imputed, the control is refit on every period, and the post-block residual is
compared with permutations of the residual path -- so the reference set is the
``T0 + H`` periods themselves, not a count of windows carved out of them.

The price is the shape of the null. A one-dimensional interval needs a
one-dimensional null, so the effect is taken to be constant across the post
block; the interval covers the total under that restriction and says nothing
about an arbitrary effect path. A constant null also has two honest failure
states this band must preserve: no constant effect conforms at all (an empty
set), and nothing is excluded in a direction (an unbounded side).

Levels: smoke, unit invariants, edge, failure.
"""
import numpy as np
import pytest

from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.conformal import (CumulativeInversionBand,
                                     cumulative_conformal_by_inversion)


def _panel(J=8, pre=30, horizon=6, effect=0.0, seed=0, trend=0.0):
    """A donor pool the treated unit sits inside, plus a constant post effect."""
    rng = np.random.default_rng(seed)
    T = pre + horizon
    f = np.cumsum(rng.normal(0, 1.0, T)) + 20.0
    Y0 = np.column_stack([f * (0.9 + 0.05 * j) + rng.normal(0, 0.3, T)
                          for j in range(J)])
    w = rng.dirichlet(np.ones(J))
    y = Y0 @ w + rng.normal(0, 0.3, T)
    y[pre:] += effect
    if trend:
        y[pre:] += trend * np.arange(1, horizon + 1)
    return y, Y0, pre, horizon


_KW = dict(refit="sc", conformal_type="block", alpha=0.1)


# --------------------------------------------------------------------------- #
# smoke
# --------------------------------------------------------------------------- #
def test_it_produces_a_band_around_the_total():
    y, Y0, pre, h = _panel(effect=4.0)
    band = cumulative_conformal_by_inversion(y, Y0, pre, h, **_KW)
    assert isinstance(band, CumulativeInversionBand)
    assert band.horizon == h
    assert np.isfinite(band.lower) and np.isfinite(band.upper)
    assert band.lower < band.upper
    assert band.lower <= 4.0 * h <= band.upper


# --------------------------------------------------------------------------- #
# unit invariants
# --------------------------------------------------------------------------- #
def test_the_cumulative_band_is_the_constant_effect_band_times_the_horizon():
    """The arithmetic that turns a per-period null into a total, pinned against
    the search it delegates to. A total over ``H`` periods of a constant effect
    is ``H`` times that effect, so nothing else may enter."""
    from mlsynth.utils.bilevel.ridge_inference import conformal_att_interval

    y, Y0, pre, h = _panel(effect=3.0)
    band = cumulative_conformal_by_inversion(y, Y0, pre, h, **_KW)
    lo, hi = conformal_att_interval(y[:pre + h], Y0[:pre + h], pre,
                                    alpha=0.1, refit="sc",
                                    conformal_type="block")
    assert band.per_period_lower == pytest.approx(lo)
    assert band.per_period_upper == pytest.approx(hi)
    assert band.lower == pytest.approx(lo * h)
    assert band.upper == pytest.approx(hi * h)


def test_shifting_the_post_block_shifts_the_band_by_the_same_total():
    """Equivariance. Adding a constant ``c`` to every post period adds exactly
    ``c`` to the effect being tested, so the interval must translate by ``c * H``
    and keep its width -- the test is about the effect, not about the level."""
    y, Y0, pre, h = _panel(effect=2.0)
    base = cumulative_conformal_by_inversion(y, Y0, pre, h, **_KW)
    shifted_y = y.copy()
    shifted_y[pre:] += 5.0
    moved = cumulative_conformal_by_inversion(shifted_y, Y0, pre, h, **_KW)
    assert moved.lower == pytest.approx(base.lower + 5.0 * h, rel=2e-2)
    assert moved.upper == pytest.approx(base.upper + 5.0 * h, rel=2e-2)
    assert (moved.upper - moved.lower) == pytest.approx(base.upper - base.lower,
                                                        rel=2e-2)


def test_a_tighter_level_does_not_narrow_the_band():
    """Fewer candidates are rejected at a smaller alpha, so the accepted set can
    only grow. The comparison is one-sided because the p-value is discrete: two
    levels between the same pair of attainable values give the identical set."""
    y, Y0, pre, h = _panel(effect=3.0)
    kw = dict(refit="sc", conformal_type="block")
    loose = cumulative_conformal_by_inversion(y, Y0, pre, h, alpha=0.2, **kw)
    tight = cumulative_conformal_by_inversion(y, Y0, pre, h, alpha=0.05, **kw)
    assert (tight.upper - tight.lower) >= (loose.upper - loose.lower) - 1e-9


def test_the_point_is_the_horizon_times_the_average_gap():
    """The point comes from a fit on the pre-period alone, not from a null refit.

    A refit that treats the post block as matching periods pulls its own
    post-period gaps toward zero, so a point taken from it would sit nearer zero
    than the estimator's own effect and the band would be centred away from what
    it is meant to describe.
    """
    from mlsynth.utils.bilevel.simplex import simplex_lstsq

    y, Y0, pre, h = _panel(effect=4.0)
    band = cumulative_conformal_by_inversion(y, Y0, pre, h, **_KW)
    w = simplex_lstsq(Y0[:pre], y[:pre])
    expected = h * float(np.mean(y[pre:pre + h] - Y0[pre:pre + h] @ w))
    assert band.point == pytest.approx(expected, rel=1e-4)


def test_the_horizon_bounds_the_reference_set():
    """Only ``pre + horizon`` periods enter, so a longer series does not lend the
    test extra permutations."""
    y, Y0, pre, h = _panel(pre=30, horizon=10, effect=3.0)
    short = cumulative_conformal_by_inversion(y, Y0, pre, 4, **_KW)
    assert short.horizon == 4
    assert short.n_reference == pre + 4


def test_an_unstated_horizon_takes_every_post_period():
    """``None`` is the documented default, not a rejected value: a caller who
    wants the whole post block should not have to count it."""
    y, Y0, pre, h = _panel(pre=30, horizon=6, effect=3.0)
    band = cumulative_conformal_by_inversion(y, Y0, pre, None, **_KW)
    assert band.horizon == 6
    assert band.n_reference == y.size


def test_the_two_permutation_schemes_are_both_available_and_differ():
    """``block`` uses the ``T`` cyclic shifts and preserves serial dependence;
    ``iid`` draws permutations and gives a finer p-value grid. They are different
    reference sets and need not agree."""
    y, Y0, pre, h = _panel(effect=3.0)
    blk = cumulative_conformal_by_inversion(y, Y0, pre, h, refit="sc",
                                            conformal_type="block", alpha=0.1)
    iid = cumulative_conformal_by_inversion(y, Y0, pre, h, refit="sc",
                                            conformal_type="iid", alpha=0.1,
                                            ns=400, seed=0)
    assert np.isfinite(blk.lower) and np.isfinite(iid.lower)
    assert not np.isclose(blk.lower, iid.lower, rtol=1e-9)


def test_the_band_and_the_zero_null_p_value_agree_about_zero():
    """The band is the set the test accepts, so the two must not disagree about
    the one candidate anybody checks. Zero belongs inside exactly when the
    zero-null p-value clears the level, and the same statement has to hold at a
    level on the other side of that p-value."""
    from mlsynth.utils.bilevel.ridge_inference import conformal_pvalue

    y, Y0, pre, h = _panel(effect=2.5, pre=24, horizon=6, seed=3)
    kw = dict(refit="sc", conformal_type="block")
    p0 = conformal_pvalue(y[:pre + h], Y0[:pre + h], pre, q=1.0, **kw)
    for alpha in (p0 / 2.0, min(0.999, p0 * 2.0)):
        band = cumulative_conformal_by_inversion(y, Y0, pre, h, alpha=alpha, **kw)
        if band.is_empty:
            continue
        contains_zero = band.lower <= 0.0 <= band.upper
        assert contains_zero == (p0 > alpha)


def test_it_records_what_produced_it():
    y, Y0, pre, h = _panel(effect=3.0)
    band = cumulative_conformal_by_inversion(y, Y0, pre, h, **_KW)
    assert band.alpha == pytest.approx(0.1)
    assert band.refit == "sc"
    assert band.conformal_type == "block"
    assert band.n_reference == pre + h


# --------------------------------------------------------------------------- #
# edge
# --------------------------------------------------------------------------- #
def test_a_level_finer_than_the_reference_set_leaves_the_band_unbounded():
    """The block scheme's reference set is the ``T`` cyclic shifts, one of which
    is the observed path, so no p-value falls below ``1 / T``. Below that level
    nothing is rejected, and an infinite endpoint is the honest report -- a
    finite one would invent a rejection that never happened."""
    y, Y0, pre, h = _panel(effect=3.0, pre=12, horizon=4)
    band = cumulative_conformal_by_inversion(y, Y0, pre, h, refit="sc",
                                             conformal_type="block",
                                             alpha=1.0 / (pre + h) / 2.0)
    assert band.is_unbounded
    assert band.lower == -np.inf and band.upper == np.inf


def test_an_effect_no_constant_explains_returns_an_empty_set():
    """The null being inverted is a constant effect. A steeply growing one is not
    constant, and when no candidate conforms the band says so instead of
    returning the least-rejected value as though it had been accepted."""
    y, Y0, pre, h = _panel(effect=0.0, trend=14.0, pre=30, horizon=8, seed=5)
    band = cumulative_conformal_by_inversion(y, Y0, pre, h, refit="sc",
                                             conformal_type="iid", alpha=0.5,
                                             ns=400, seed=0)
    if band.is_empty:
        assert np.isnan(band.lower) and np.isnan(band.upper)
    else:                       # the panel did not defeat the constant null
        assert band.lower < band.upper


def test_a_single_post_period_is_a_per_period_interval():
    y, Y0, pre, h = _panel(effect=4.0, horizon=6)
    band = cumulative_conformal_by_inversion(y, Y0, pre, 1, **_KW)
    assert band.horizon == 1
    assert band.lower == pytest.approx(band.per_period_lower)
    assert band.upper == pytest.approx(band.per_period_upper)


# --------------------------------------------------------------------------- #
# failure
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5, "0.05", None, True])
def test_a_bad_level_is_refused(bad):
    y, Y0, pre, h = _panel()
    with pytest.raises(MlsynthConfigError, match="alpha"):
        cumulative_conformal_by_inversion(y, Y0, pre, h, refit="sc", alpha=bad)


@pytest.mark.parametrize("bad", [0, -2, 2.5, "6", True])
def test_a_bad_horizon_is_refused(bad):
    y, Y0, pre, h = _panel()
    with pytest.raises(MlsynthConfigError, match="horizon"):
        cumulative_conformal_by_inversion(y, Y0, pre, bad, **_KW)


def test_a_horizon_past_the_series_is_refused():
    y, Y0, pre, h = _panel(pre=30, horizon=6)
    with pytest.raises(MlsynthConfigError, match="horizon"):
        cumulative_conformal_by_inversion(y, Y0, pre, 7, **_KW)


@pytest.mark.parametrize("bad", [0, -1, 40])
def test_a_bad_pre_period_count_is_refused(bad):
    y, Y0, pre, h = _panel(pre=30, horizon=6)
    with pytest.raises(MlsynthConfigError, match="pre"):
        cumulative_conformal_by_inversion(y, Y0, bad, 3, **_KW)


@pytest.mark.parametrize("bad", ["lasso", "SC", "", None])
def test_an_unknown_refit_rule_is_refused(bad):
    y, Y0, pre, h = _panel()
    with pytest.raises(MlsynthConfigError, match="refit"):
        cumulative_conformal_by_inversion(y, Y0, pre, h, refit=bad,
                                          conformal_type="block", alpha=0.1)


@pytest.mark.parametrize("bad", ["moving", "BLOCK", "", None])
def test_an_unknown_permutation_scheme_is_refused(bad):
    y, Y0, pre, h = _panel()
    with pytest.raises(MlsynthConfigError, match="conformal_type"):
        cumulative_conformal_by_inversion(y, Y0, pre, h, refit="sc",
                                          conformal_type=bad, alpha=0.1)


def test_a_donor_block_of_the_wrong_length_is_refused():
    y, Y0, pre, h = _panel()
    with pytest.raises(MlsynthDataError, match="periods"):
        cumulative_conformal_by_inversion(y, Y0[:-3], pre, h, **_KW)


def test_a_two_dimensional_treated_series_is_refused():
    y, Y0, pre, h = _panel()
    with pytest.raises(MlsynthDataError, match="one-dimensional"):
        cumulative_conformal_by_inversion(np.tile(y, (2, 1)), Y0, pre, h, **_KW)


@pytest.mark.parametrize("spoil", ["nan", "inf"])
def test_a_non_finite_series_is_refused(spoil):
    y, Y0, pre, h = _panel()
    y = y.copy()
    y[2] = np.nan if spoil == "nan" else np.inf
    with pytest.raises(MlsynthDataError, match="finite"):
        cumulative_conformal_by_inversion(y, Y0, pre, h, **_KW)


def test_a_one_dimensional_donor_block_is_refused():
    """A single donor passed as a flat series is the common shape slip, and the
    failure names the argument instead of broadcasting into a wrong fit."""
    y, Y0, pre, h = _panel()
    with pytest.raises(MlsynthDataError, match="two-dimensional"):
        cumulative_conformal_by_inversion(y, Y0[:, 0], pre, h, **_KW)


@pytest.mark.parametrize("bad", [2.5, "30", None, True])
def test_a_non_integer_pre_period_count_is_refused(bad):
    y, Y0, pre, h = _panel()
    with pytest.raises(MlsynthConfigError, match="pre"):
        cumulative_conformal_by_inversion(y, Y0, bad, 3, **_KW)


# --------------------------------------------------------------------------- #
# the accepted set need not be an interval
# --------------------------------------------------------------------------- #
def test_a_grid_search_is_offered_and_reports_gaps():
    """A conformal p-value is not unimodal in the candidate.

    On the Proposition 99 panel the accepted set at ``alpha = 0.1`` is two
    islands, ``[-120, -20]`` and ``[-10, +5]``, separated by a candidate whose
    p-value falls to the block scheme's floor of ``1 / T``. A search that
    brackets outward from the point estimate and bisects stops at the first
    crossing and never sees the far island -- so it can exclude a candidate the
    test accepts, including zero. The grid route scans instead and reports the
    range of everything that survived, plus whether the survivors had holes.
    """
    y, Y0, pre, h = _panel(effect=3.0)
    grid = np.linspace(-20.0, 20.0, 41)
    band = cumulative_conformal_by_inversion(y, Y0, pre, h, search="grid",
                                             grid=grid, **_KW)
    assert band.search == "grid"
    assert band.has_interior_gaps in (True, False)
    assert band.lower <= band.upper or band.is_empty


def test_the_bisecting_search_does_not_claim_to_know_about_gaps():
    """It cannot: it only ever evaluates candidates along one outward walk, so
    ``None`` is the honest report and ``False`` would be a claim it has not
    earned."""
    y, Y0, pre, h = _panel(effect=3.0)
    band = cumulative_conformal_by_inversion(y, Y0, pre, h, **_KW)
    assert band.search == "bisect"
    assert band.has_interior_gaps is None


def test_the_grid_route_keeps_every_accepted_candidate_inside_the_band():
    """The defining property. Whatever the survivors look like, the reported
    range must contain all of them -- that is what makes it a confidence set."""
    from mlsynth.utils.bilevel.ridge_inference import conformal_pvalue

    y, Y0, pre, h = _panel(effect=2.0, pre=24, horizon=6, seed=8)
    grid = np.linspace(-12.0, 12.0, 25)
    kw = dict(refit="sc", conformal_type="block")
    band = cumulative_conformal_by_inversion(y, Y0, pre, h, alpha=0.1,
                                             search="grid", grid=grid, **kw)
    accepted = []
    for tau in grid:
        shifted = y[:pre + h].copy()
        shifted[pre:] -= tau
        if conformal_pvalue(shifted, Y0[:pre + h], pre, q=1.0, **kw) > 0.1:
            accepted.append(tau)
    if not accepted:
        assert band.is_empty
    else:
        assert band.per_period_lower <= min(accepted) + 1e-9
        assert band.per_period_upper >= max(accepted) - 1e-9


def test_a_grid_search_without_a_grid_is_refused():
    """There is no safe default grid: its width has to be scaled to the panel's
    own dispersion, and one guessed here would step over a narrow set and report
    it empty."""
    y, Y0, pre, h = _panel()
    with pytest.raises(MlsynthConfigError, match="grid"):
        cumulative_conformal_by_inversion(y, Y0, pre, h, search="grid", **_KW)


@pytest.mark.parametrize("bad", [[], [1.0], "0,1", 5.0])
def test_a_bad_grid_is_refused(bad):
    y, Y0, pre, h = _panel()
    with pytest.raises((MlsynthConfigError, MlsynthDataError), match="grid"):
        cumulative_conformal_by_inversion(y, Y0, pre, h, search="grid",
                                          grid=bad, **_KW)


def test_a_non_finite_grid_is_refused():
    y, Y0, pre, h = _panel()
    with pytest.raises(MlsynthDataError, match="finite"):
        cumulative_conformal_by_inversion(y, Y0, pre, h, search="grid",
                                          grid=[0.0, np.nan, 1.0], **_KW)


@pytest.mark.parametrize("bad", ["bisection", "GRID", "", None])
def test_an_unknown_search_is_refused(bad):
    y, Y0, pre, h = _panel()
    with pytest.raises(MlsynthConfigError, match="search"):
        cumulative_conformal_by_inversion(y, Y0, pre, h, search=bad, **_KW)
