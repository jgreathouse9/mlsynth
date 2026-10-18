"""The signed statistic and the reference window of the CWZ test inversion.

Two options, both for a band on a total:

``statistic="signed"``
    Chernozhukov, Wuthrich and Zhu's Remark 1 statistic, ``|sum_t u_t| /
    sqrt(T*)``. Under a constant-effect null the observed value depends on the
    effect path only through its total, so the band covers the total under paths
    the constant null does not describe. The default ``S_q`` statistic sums the
    absolute residuals and grows with every departure from a constant path.

``reference_start``
    The first period whose residual enters the permutation reference. The refit
    still matches on every period. A design chosen on an early window leaves
    smaller residuals there, and a reference that includes them understates the
    post sum's spread; starting the reference after that window keeps it out.

Levels: smoke, unit invariants, edge, failure. The properties are in
``test_conformal_signed_reference_properties.py``.
"""
import numpy as np
import pytest

from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.conformal import (CumulativeInversionBand,
                                     cumulative_conformal_by_inversion)
from mlsynth.utils.conformal.refit import conformal_refit_gaps
from mlsynth.utils.conformal.ridge_inference import (CONFORMAL_STATISTICS,
                                                     _reference_stats, _stat,
                                                     conformal_att_interval,
                                                     conformal_pvalue)


def _panel(J=8, pre=40, horizon=6, effect=0.0, seed=0, early=0, early_sd=1.0,
           noise=0.3):
    """Donors the treated unit sits inside; the first ``early`` periods can be quieter."""
    rng = np.random.default_rng(seed)
    T = pre + horizon
    f = np.cumsum(rng.normal(0, 1.0, T)) + 20.0
    Y0 = np.column_stack([f * (0.9 + 0.05 * j) + rng.normal(0, 0.3, T)
                          for j in range(J)])
    w = rng.dirichlet(np.ones(J))
    e = rng.normal(0, noise, T)
    e[:early] *= early_sd
    y = Y0 @ w + e
    y[pre:] += effect
    return y, Y0, pre, horizon


_KW = dict(refit="sc", conformal_type="block")


# --------------------------------------------------------------------------- #
# smoke
# --------------------------------------------------------------------------- #
def test_a_signed_band_with_a_reference_window_is_a_finite_interval():
    """One draw, so the band's coverage is not asserted here (a 90% band misses
    one draw in ten); the level is held in the size test below."""
    y, Y0, pre, h = _panel(effect=4.0)
    band = cumulative_conformal_by_inversion(
        y, Y0, pre, h, alpha=0.1, statistic="signed", reference_start=10, **_KW)
    assert isinstance(band, CumulativeInversionBand)
    assert np.isfinite(band.lower) and np.isfinite(band.upper)
    assert band.lower < band.upper
    assert band.lower * 0.5 < 4.0 * h < band.upper * 2.0
    assert band.statistic == "signed"
    assert band.reference_start == 10
    assert band.n_reference == pre + h - 10


def test_the_defaults_are_recorded_on_the_band():
    y, Y0, pre, h = _panel(effect=4.0)
    band = cumulative_conformal_by_inversion(y, Y0, pre, h, alpha=0.1, **_KW)
    assert band.statistic == "sq"
    assert band.reference_start == 0
    assert band.n_reference == pre + h


# --------------------------------------------------------------------------- #
# unit invariants
# --------------------------------------------------------------------------- #
def test_the_statistics_on_offer_are_named():
    assert CONFORMAL_STATISTICS == ("sq", "signed")


def test_the_signed_statistic_is_the_scaled_absolute_sum():
    x = np.array([1.0, -2.0, 0.5, 3.0])
    assert _stat(x, statistic="signed") == pytest.approx(abs(x.sum()) / 2.0)
    assert _stat(-x, statistic="signed") == _stat(x, statistic="signed")
    # residuals of mixed sign cancel in the signed sum and not in S_1
    assert _stat(x, statistic="signed") < _stat(x, 1.0)


def test_the_default_statistic_is_unchanged():
    x = np.array([1.0, -2.0, 0.5, 3.0])
    assert _stat(x) == _stat(x, 1.0, "sq") == pytest.approx(6.5 / 2.0)


def test_the_signed_block_reference_starts_with_the_observed_statistic():
    """Shift zero is the observed path, bit for bit: the p-value's floor of
    ``1 / T`` rests on the observed statistic reaching itself exactly."""
    r = np.random.default_rng(3).normal(size=37)
    ref = _reference_stats(r, slice(30, None), 1.0, conformal_type="block",
                           ns=0, seed=0, statistic="signed")
    assert ref.shape == (37,)
    assert ref[0] == _stat(r[30:], 1.0, "signed")


def test_the_signed_block_reference_is_the_statistic_of_every_shift():
    r = np.random.default_rng(4).normal(size=23)
    ref = _reference_stats(r, slice(17, None), 1.0, conformal_type="block",
                           ns=0, seed=0, statistic="signed")
    loop = np.array([_stat(np.roll(r, s)[17:], 1.0, "signed") for s in range(23)])
    np.testing.assert_array_equal(ref, loop)


def test_the_signed_iid_reference_sums_the_same_draws():
    r = np.random.default_rng(5).normal(size=20)
    ref = _reference_stats(r, slice(15, None), 1.0, conformal_type="iid",
                           ns=50, seed=11, statistic="signed")
    rng = np.random.default_rng(11)
    loop = np.array([_stat(rng.permutation(r)[15:], 1.0, "signed")
                     for _ in range(50)])
    np.testing.assert_allclose(ref, loop, rtol=0, atol=1e-12)


@pytest.mark.parametrize("scheme", ["block", "iid"])
@pytest.mark.parametrize("statistic", ["sq", "signed"])
def test_a_reference_window_from_the_start_is_the_default_test(scheme, statistic):
    y, Y0, pre, h = _panel(effect=1.0)
    kw = dict(refit="sc", conformal_type=scheme, statistic=statistic, seed=2)
    assert conformal_pvalue(y, Y0, pre, reference_start=0, **kw) == \
        conformal_pvalue(y, Y0, pre, **kw)


def test_the_window_drops_early_residuals_and_the_refit_keeps_them():
    """The p-value is the full-window refit's residuals, shifted over the
    reference window only, with the post block rebased into that window."""
    y, Y0, pre, h = _panel(effect=1.5, seed=7)
    k = 12
    p = conformal_pvalue(y, Y0, pre, refit="sc", conformal_type="block",
                         statistic="signed", reference_start=k)
    r = conformal_refit_gaps(y, Y0, rule="sc")[k:]
    n = r.size
    obs = abs(r[-h:].sum())
    shifts = [abs(np.roll(r, s)[-h:].sum()) for s in range(n)]
    assert p == pytest.approx(np.mean(obs <= np.array(shifts)))


def test_the_block_floor_is_one_over_the_reference_window():
    y, Y0, pre, h = _panel(effect=500.0)
    k = 15
    p = conformal_pvalue(y, Y0, pre, refit="sc", conformal_type="block",
                         statistic="signed", reference_start=k)
    assert p == pytest.approx(1.0 / (pre + h - k))


@pytest.mark.parametrize("scheme", ["block", "iid"])
def test_one_post_period_makes_the_two_statistics_the_same_test(scheme):
    y, Y0, pre, _ = _panel(horizon=1, effect=0.8)
    kw = dict(refit="sc", conformal_type=scheme, seed=4, reference_start=5)
    assert conformal_pvalue(y, Y0, pre, statistic="signed", **kw) == \
        conformal_pvalue(y, Y0, pre, statistic="sq", **kw)


def test_the_band_is_the_signed_interval_times_the_horizon():
    """The band hands both options to the search it delegates to; recording
    them on the band is not enough."""
    y, Y0, pre, h = _panel(effect=3.0, seed=2)
    band = cumulative_conformal_by_inversion(
        y, Y0, pre, h, alpha=0.1, statistic="signed", reference_start=12, **_KW)
    lo, hi = conformal_att_interval(y[:pre + h], Y0[:pre + h], pre, alpha=0.1,
                                    refit="sc", conformal_type="block",
                                    statistic="signed", reference_start=12)
    assert band.lower == pytest.approx(lo * h)
    assert band.upper == pytest.approx(hi * h)
    plain = cumulative_conformal_by_inversion(y, Y0, pre, h, alpha=0.1, **_KW)
    assert (plain.lower, plain.upper) != pytest.approx((band.lower, band.upper))


def test_the_interval_search_runs_on_the_signed_statistic():
    y, Y0, pre, h = _panel(effect=2.0, seed=9)
    lo, hi = conformal_att_interval(y, Y0, pre, alpha=0.1, refit="sc",
                                    statistic="signed", reference_start=8)
    assert lo <= 2.0 <= hi
    # every candidate inside is accepted and the endpoints' outer neighbours not
    for c in np.linspace(lo, hi, 5)[1:-1]:
        shifted = y.copy()
        shifted[pre:] -= c
        assert conformal_pvalue(shifted, Y0, pre, refit="sc", conformal_type="block",
                                statistic="signed", reference_start=8) > 0.1


def _rejection_rate(reference_start, draws=200, alpha=0.2):
    """No effect; the first 30 periods carry a fifth of the later noise."""
    hits = 0
    for s in range(draws):
        y, Y0, pre, h = _panel(J=6, pre=50, horizon=8, seed=100 + s, early=30,
                               early_sd=0.2, noise=1.0)
        p = conformal_pvalue(y, Y0, pre, refit="sc", conformal_type="block",
                             statistic="signed", reference_start=reference_start)
        hits += p <= alpha
    return hits / draws


def test_a_quiet_early_stretch_in_the_reference_inflates_the_size():
    """The failure the window exists for, and its repair: residuals smaller in
    an early stretch understate the post sum's spread when they are referenced,
    and the test holds its level once they are not. At ``alpha = 0.2`` over 200
    draws the standard error is 0.028."""
    leaky = _rejection_rate(0)
    held = _rejection_rate(30)
    assert leaky > 0.30
    assert held <= 0.2 + 3 * 0.028


# --------------------------------------------------------------------------- #
# edge
# --------------------------------------------------------------------------- #
def test_a_window_too_short_for_the_level_leaves_the_band_unbounded():
    y, Y0, pre, h = _panel(effect=1.0)
    band = cumulative_conformal_by_inversion(
        y, Y0, pre, h, alpha=0.05, statistic="signed", reference_start=pre - 4,
        **_KW)
    assert band.n_reference == 4 + h
    assert band.is_unbounded


def test_the_iid_scheme_permutes_inside_the_window():
    y, Y0, pre, h = _panel(effect=0.5)
    p = conformal_pvalue(y, Y0, pre, refit="sc", conformal_type="iid", ns=200,
                         seed=3, statistic="signed", reference_start=20,
                         finite_sample=True)
    assert 1.0 / 201.0 <= p <= 1.0


# --------------------------------------------------------------------------- #
# failure
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", ["Signed", "", None, 1, "absolute"])
def test_an_unknown_statistic_is_refused(bad):
    y, Y0, pre, h = _panel()
    with pytest.raises(MlsynthConfigError, match="statistic"):
        conformal_pvalue(y, Y0, pre, refit="sc", statistic=bad)
    with pytest.raises(MlsynthConfigError, match="statistic"):
        cumulative_conformal_by_inversion(y, Y0, pre, h, statistic=bad, **_KW)


def test_a_norm_with_the_signed_statistic_is_refused():
    y, Y0, pre, h = _panel()
    with pytest.raises(MlsynthConfigError, match="q"):
        conformal_pvalue(y, Y0, pre, refit="sc", statistic="signed", q=2.0)


@pytest.mark.parametrize("bad", [-1, 40, 41, 2.5, True, "3", None])
def test_a_bad_reference_start_is_refused(bad):
    y, Y0, pre, h = _panel()          # pre = 40
    with pytest.raises(MlsynthConfigError, match="reference_start"):
        conformal_pvalue(y, Y0, pre, refit="sc", reference_start=bad)
    with pytest.raises(MlsynthConfigError, match="reference_start"):
        cumulative_conformal_by_inversion(y, Y0, pre, h, reference_start=bad, **_KW)
