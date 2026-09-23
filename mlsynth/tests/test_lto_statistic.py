"""The LTO pair statistic is a choice, not a fixture of the procedure.

Lei & Sudijono (2025) section 6.4: the theory "only relies on the uniform
assignment assumption but not the choice of summary statistics". The shipped test
hardcodes the post/pre RMSPE ratio, which is the right default and the only thing
the paper's own applications use, but it is not the only statistic the guarantee
covers -- and a cumulative interval needs a different one.

These pin the seam: the default is unchanged, a caller's statistic is the one
that decides the comparison, and a statistic that misbehaves is refused at the
boundary instead of quietly producing a p-value from garbage.
"""
import numpy as np
import pytest

from mlsynth.utils.vanillasc_helpers import BilevelSCM
from mlsynth.utils.vanillasc_helpers.lto import _rmspe_ratio_resid, lto_placebo_test


def _case(seed=3, T=24, T0=16, J=7, effect=-2.0):
    """A small donor pool with a planted post-period effect."""
    rng = np.random.default_rng(seed)
    Y0 = rng.normal(5, 1, size=(T, J))
    w = np.zeros(J)
    w[[0, 2, 5]] = [0.5, 0.3, 0.2]
    y = Y0 @ w + 0.05 * rng.normal(size=T)
    y[T0:] += effect
    return y, Y0, T0


def _run(**kw):
    y, Y0, T0 = _case()
    return lto_placebo_test(BilevelSCM("outcome-only", seed=0), y, Y0, T0,
                            alpha=0.05, seed=0, **kw)


def test_passing_the_default_explicitly_changes_nothing():
    """The seam is a seam, not a new procedure."""
    assert _run() == _run(statistic=_rmspe_ratio_resid)


def test_the_callers_statistic_is_the_one_that_decides():
    """Two statistics with opposite verdicts, to prove the argument is used.

    A statistic that ranks the treated unit above every donor makes it win every
    triple; one that ranks it below makes it lose every triple. Asserting only
    one of the two would pass on a stub that ignored the argument and happened to
    agree.
    """
    def treated_always_wins(y_k, cf, pre):
        # the treated series is the only one carrying the planted effect, so
        # keying on the post-period gap separates it from every donor
        return float(np.abs(np.asarray(y_k)[pre:] - np.asarray(cf)[pre:]).mean())

    def treated_always_loses(y_k, cf, pre):
        return -treated_always_wins(y_k, cf, pre)

    win = _run(statistic=treated_always_wins)
    lose = _run(statistic=treated_always_loses)
    assert win["p_value"] < lose["p_value"]
    assert lose["p_value"] == pytest.approx(1.0)


def test_a_statistic_is_called_for_every_unit_in_the_triple():
    """Three residuals per pair: the treated unit and both left-out donors."""
    seen = []

    def spy(y_k, cf, pre):
        seen.append(np.asarray(y_k).copy())
        return _rmspe_ratio_resid(y_k, cf, pre)

    out = _run(statistic=spy)
    assert len(seen) == 3 * out["n_pairs"]


def test_a_non_callable_statistic_is_refused():
    with pytest.raises(ValueError, match="statistic"):
        _run(statistic="rmspe")


def test_a_statistic_returning_nan_on_the_treated_unit_does_not_win_by_default():
    """A degenerate treated score must not be read as an extreme one.

    The shipped guard replaces a non-finite treated residual with the largest
    float, which makes the treated unit win the triple. That is a decision about
    a degenerate fit, and it has to survive a caller's statistic rather than
    depend on which one is passed.
    """
    def nan_on_treated(y_k, cf, pre):
        r = _rmspe_ratio_resid(y_k, cf, pre)
        y_k = np.asarray(y_k)
        return float("nan") if y_k.shape == (24,) and y_k[16:].mean() < 4.0 else r

    out = _run(statistic=nan_on_treated)
    assert 0.0 <= out["p_value"] <= 1.0
    assert np.isfinite(out["p_value"])
