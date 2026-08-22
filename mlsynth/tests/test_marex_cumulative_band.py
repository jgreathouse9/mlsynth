"""MAREX's cumulative-effect band, drawn from the blank-period placebo effects.

MAREX already reports a pointwise band: a constant half-width ``q``, the
``1 - alpha`` quantile of the absolute blank-period effects, laid around each
post period's effect. That answers "how big was the effect in week 7". It does not
answer "how much did the intervention add in total", and the two are not related
by arithmetic on the endpoints -- summing them assumes the period errors move in
lockstep, dividing one by the horizon assumes they are independent.

The blank window makes the total cheap to calibrate. It is a contiguous stretch of
periods the fit never saw, so the effects there are an out-of-sample error series
already: no refits, where VanillaSC and PDA each had to slide an origin across the
pre-period to manufacture one. ``cumulative_band=True`` splits that window into
horizon-length sub-windows and draws each path as one sub-window's level plus blocks
of the residuals about it, then reads the half-width off the accumulated totals. The
level is the term that sizes a total -- a total over ``H`` periods is ``H`` times the
mean error over them -- so a draw that could not see it between windows under-covered
badly, which is why the sub-windows exist.

Levels: smoke, unit invariants, edge, failure.
"""
import numpy as np
import pandas as pd
import pytest

from mlsynth import MAREX
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.conformal import CumulativeConformalBand


def _panel(J=10, T=48, seed=0, rho=0.0):
    """A factor panel with an autocorrelated idiosyncratic part.

    ``rho`` drives the serial correlation the block length exists to carry; at
    ``rho=0`` a whole-horizon block and single periods should agree.
    """
    rng = np.random.default_rng(seed)
    F = rng.normal(0, 1, (T, 2))
    lam = rng.normal(0, 1, (J, 2))
    e = rng.standard_normal((J, T))
    if rho:
        for t in range(1, T):
            e[:, t] = rho * e[:, t - 1] + np.sqrt(1 - rho ** 2) * e[:, t]
    Y = lam @ F.T + 0.4 * e
    rows = [{"unit": f"u{j:02d}", "time": t, "y": float(Y[j, t])}
            for j in range(J) for t in range(T)]
    return pd.DataFrame(rows)


# The regime the band is built for. The blank window is split into Tb // H
# horizon-length sub-windows, and two are needed before a level can be identified at
# all, so Tb >= 2H either way. A 120-week pre-period with a 30 percent blank tail and
# a three-month experiment is Tb ~ 36 against H ~ 13, which clears it; this is the
# same shape, scaled down so the MIQP solves in test time.
_CFG = dict(outcome="y", unitid="unit", time="time", T0=40, m_eq=2,
            inference=True, blank_periods=18, T_post=8)


def _inf(df=None, **kw):
    cfg = dict(df=_panel() if df is None else df, **_CFG)
    cfg.update(kw)
    return MAREX(cfg).fit().globres.inference


def _direct(alpha=0.05, Tb=16, T_post=6, T0=28, rho=0.0, seed=1, **kw):
    """``compute_inference`` on a synthetic contrast, with no design solve.

    The level is exercised here and not through ``fit()`` on purpose:
    ``MAREXConfig.alpha`` is documented as the level of the MDE power curve, and
    the estimator does not route it to ``compute_inference``, which keeps its own
    default. Driving alpha end-to-end would be asserting a plumbing path that does
    not exist.
    """
    from mlsynth.utils.marex_helpers.inference import compute_inference

    rng = np.random.default_rng(seed)
    T = T0 + T_post
    e = rng.standard_normal(T)
    if rho:
        for t in range(1, T):
            e[t] = rho * e[t - 1] + np.sqrt(1 - rho ** 2) * e[t]
    base = np.cumsum(rng.normal(0, 0.3, T)) + 10.0
    y_c = base
    y_t = base + e
    y_t[T0:] += 1.5
    return compute_inference(y_t, y_c, T0, T0 - Tb, Tb, alpha=alpha,
                             random_state=0, **kw)


# --------------------------------------------------------------------------- #
# smoke
# --------------------------------------------------------------------------- #
def test_it_produces_a_finite_band():
    band = _inf(cumulative_band=True, cumulative_n_sim=400).cumulative_band
    assert isinstance(band, CumulativeConformalBand)
    assert np.isfinite(band.half_width) and band.half_width > 0
    assert np.isfinite(band.lower) and np.isfinite(band.upper)


# --------------------------------------------------------------------------- #
# unit invariants
# --------------------------------------------------------------------------- #
def test_it_is_absent_unless_asked_for():
    """Off by default: no existing MAREX caller sees a new populated field."""
    assert _inf().cumulative_band is None


def test_the_pointwise_band_is_untouched():
    """The new band sits beside the old one; it does not restate it."""
    plain = _inf()
    with_band = _inf(cumulative_band=True, cumulative_n_sim=400)
    assert np.allclose(plain.ci, with_band.ci, equal_nan=True)
    assert plain.global_p_value == pytest.approx(with_band.global_p_value)
    assert np.allclose(plain.per_period_pvals, with_band.per_period_pvals)


def test_the_point_is_the_total_effect_over_the_post_period():
    inf = _inf(cumulative_band=True, cumulative_n_sim=400)
    assert inf.cumulative_band.point == pytest.approx(float(inf.treated_effects.sum()))
    assert inf.cumulative_band.horizon == inf.treated_effects.size


def test_the_band_is_symmetric_about_the_point():
    b = _inf(cumulative_band=True, cumulative_n_sim=400).cumulative_band
    assert b.point - b.lower == pytest.approx(b.half_width)
    assert b.upper - b.point == pytest.approx(b.half_width)


def test_it_records_the_blank_periods_it_drew_from():
    """``n_scores`` is what makes a recycled short window visible to a reader."""
    inf = _inf(cumulative_band=True, cumulative_n_sim=400)
    assert inf.cumulative_band.n_scores == inf.placebo_effects.size


def test_it_records_the_level():
    b = _direct(alpha=0.1, cumulative_band=True, cumulative_n_sim=400).cumulative_band
    assert b.alpha == pytest.approx(0.1)


def test_the_same_seed_gives_the_same_band():
    kw = dict(cumulative_band=True, cumulative_block=1, cumulative_n_sim=300)
    a = _inf(cumulative_seed=4, **kw).cumulative_band
    b = _inf(cumulative_seed=4, **kw).cumulative_band
    assert a.half_width == pytest.approx(b.half_width)


def test_different_seeds_give_different_bands():
    """Single-period blocks, because the level is drawn from only Tb // H
    sub-window means and a whole-horizon block leaves the fluctuation little room to
    vary, so two seeds can land on the same pair of atoms."""
    kw = dict(cumulative_band=True, cumulative_block=1, cumulative_n_sim=300)
    a = _inf(cumulative_seed=4, **kw).cumulative_band
    b = _inf(cumulative_seed=5, **kw).cumulative_band
    assert a.half_width != pytest.approx(b.half_width, rel=1e-6)


@pytest.mark.parametrize("alpha", [0.9, 0.5, 0.2, 0.05])
def test_the_half_width_is_a_magnitude_at_every_level(alpha):
    """The draws are sign-symmetric, so the ``1 - alpha`` quantile of the signed
    totals goes negative once alpha passes a half. A half-width cannot, and the
    band brackets its point."""
    b = _direct(alpha=alpha, cumulative_band=True,
                cumulative_n_sim=400).cumulative_band
    assert b.half_width > 0.0
    assert b.lower < b.point < b.upper


def test_a_tighter_level_gives_a_wider_band():
    kw = dict(cumulative_band=True, cumulative_n_sim=4000, cumulative_block=1)
    assert (_direct(alpha=0.01, **kw).cumulative_band.half_width
            > _direct(alpha=0.20, **kw).cumulative_band.half_width)


def test_a_longer_block_widens_the_band_on_a_persistent_panel():
    """The generalisation doing its work: a total accumulated from positively
    autocorrelated periods is more variable than independent draws imply."""
    df = _panel(rho=0.85, seed=3)
    kw = dict(df=df, cumulative_band=True, cumulative_n_sim=4000)
    one = _inf(cumulative_block=1, **kw).cumulative_band
    full = _inf(cumulative_block=0, **kw).cumulative_band
    assert full.half_width > one.half_width


def test_the_band_is_not_the_pointwise_band_times_the_horizon():
    """Stacking the pointwise endpoints is the lockstep assumption; this is the
    construction that replaces it, so the two must not coincide."""
    inf = _inf(cumulative_band=True, cumulative_n_sim=4000)
    q = float(np.quantile(np.abs(inf.placebo_effects), 0.95))
    stacked = q * inf.treated_effects.size
    assert inf.cumulative_band.half_width < stacked


# --------------------------------------------------------------------------- #
# edge
# --------------------------------------------------------------------------- #
def test_a_horizon_longer_than_the_blank_window_still_bands():
    """H > Tb is drawable: a path is assembled from several blocks, so only the
    BLOCK has to fit inside the window. It recycles a short series, which is why
    n_scores reports the window it came from."""
    # The horizon is the post-period the panel actually has (len(y) - T0 = 16),
    # against a five-period blank window.
    inf = _inf(df=_panel(T=50), T0=34, blank_periods=5, T_post=16,
               cumulative_band=True, cumulative_block=2, cumulative_n_sim=400)
    assert inf.cumulative_band.horizon == 16
    assert inf.cumulative_band.n_scores == 5
    assert np.isfinite(inf.cumulative_band.half_width)
    assert inf.cumulative_band.half_width > 0.0


def test_a_whole_horizon_block_longer_than_the_blank_window_is_refused():
    """The default block is the horizon, so a horizon longer than the blank window
    asks for a block past half the window. Over a centred series such a block has
    the spread of its complement, not of itself, so it is refused."""
    with pytest.raises(MlsynthDataError, match="half"):
        _inf(df=_panel(T=50), T0=34, blank_periods=5, T_post=16,
             cumulative_band=True, cumulative_n_sim=400)


def test_a_single_blank_period_cannot_band():
    """One blank period is one error, and centring it leaves exactly zero. There
    is no spread to resample and no shorter block to retreat to, so the estimator
    says so instead of reporting a band of no width."""
    with pytest.raises(MlsynthDataError, match="single calibration error"):
        _inf(blank_periods=1, cumulative_band=True, cumulative_n_sim=400)


def test_asking_for_it_without_inference_leaves_it_absent():
    """No blank window means no calibration series; there is no band to build and
    nothing is silently substituted for one."""
    res = MAREX({"df": _panel(), "outcome": "y", "unitid": "unit", "time": "time",
                 "T0": 18, "m_eq": 2, "cumulative_band": True}).fit()
    assert res.globres.inference is None


# --------------------------------------------------------------------------- #
# failure
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [-1, 2.5, "3", True, None])
def test_a_bad_block_is_refused(bad):
    with pytest.raises(MlsynthConfigError, match="cumulative_block"):
        _inf(cumulative_band=True, cumulative_block=bad)


@pytest.mark.parametrize("bad", [0, 1, 2.5, "400", None])
def test_a_bad_simulation_count_is_refused(bad):
    with pytest.raises(MlsynthConfigError, match="cumulative_n_sim"):
        _inf(cumulative_band=True, cumulative_n_sim=bad)


@pytest.mark.parametrize("bad", [2.5, "7", None])
def test_a_bad_seed_is_refused(bad):
    with pytest.raises(MlsynthConfigError, match="cumulative_seed"):
        _inf(cumulative_band=True, cumulative_seed=bad)


@pytest.mark.parametrize("bad", ["yes", 1, None])
def test_a_non_boolean_request_is_refused(bad):
    with pytest.raises(MlsynthConfigError, match="cumulative_band"):
        _inf(cumulative_band=bad)
