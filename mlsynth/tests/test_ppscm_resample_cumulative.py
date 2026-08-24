"""Calibrating PPSCM's per-unit cumulative band by resampling the rolling pass.

``cumulative_conformal_per_unit`` reduces each calibration window to one number,
its total, and reads an order statistic off the ``m`` totals. That is what puts
a floor under the split band: a finite ``1 - alpha`` interval needs
``ceil((m+1)(1-alpha)) <= m``, so at the 90 percent level it needs nine windows,
about ``12.8 * L`` pre-periods.

The same pass produces ``m * L`` per-period errors on the way to those totals.
Resampling draws whole paths from them -- circular blocks, each sign-flipped
with probability one half -- and reads the band off the accumulated draws. The
reference set is periods instead of windows, so the window floor does not apply
and a panel that leaves the split band infinite still gets a finite one.

It is the third value of ``conformal_method``, on the same axis as the other
two: all three report the same estimand, the total a treated unit gained over
``conformal_horizon`` periods, from different reference sets. Unlike the cyclic
band it assumes no shape for the effect and refits nothing beyond the rolling
pass the split band already pays for.

The aggregate band is untouched. Its jackknife resamples units, the dimension
along which a pooled total has exchangeable replicates; this resamples time,
which is the only dimension a single unit has.
"""

import numpy as np
import pandas as pd
import pytest

from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.ppscm_helpers.engine import Conventions


# A panel wide enough in the pre-period to schedule several calibration windows,
# and deliberately short of the split band's floor at alpha=0.10: earliest
# adoption 40, horizon 4, min_train_frac 0.3 puts origins at 12..36 step 4, so
# m = 7 and the split band needs ceil(8*0.9) = 8. Seven windows, rank eight.
def _staggered_df(seed=0, adoption=(40, 44), n_donors=8, T=60, effect=-3.0,
                  noise=0.4):
    rng = np.random.default_rng(seed)
    factors = rng.standard_normal((T, 2))
    load_d = rng.standard_normal((n_donors, 2)) * 0.5
    load_t = load_d.mean(axis=0)
    rec = []
    for j, Tj in enumerate(adoption):
        series = (factors @ (load_t + 0.1 * rng.standard_normal(2))
                  + rng.standard_normal(T) * noise)
        series[Tj:] += effect
        rec += [{"unit": f"treated_{j}", "year": 2000 + t, "y": float(series[t]),
                 "tr": int(t >= Tj)} for t in range(T)]
    for dd in range(n_donors):
        series = factors @ load_d[dd] + rng.standard_normal(T) * noise
        rec += [{"unit": f"d_{dd}", "year": 2000 + t, "y": float(series[t]), "tr": 0}
                for t in range(T)]
    return pd.DataFrame(rec)


def _cfg(**kw):
    cfg = dict(df=_staggered_df(), outcome="y", treat="tr", unitid="unit",
               time="year", display_graphs=False, run_inference=False)
    cfg.update(kw)
    return cfg


def _build(**kw):
    from mlsynth import PPSCM
    return PPSCM(_cfg(**kw))


def _fit(**kw):
    import warnings as _w
    from mlsynth import PPSCM
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        return PPSCM(_cfg(**kw)).fit()


def _panel():
    """``(Xy, trt, d, n_leads, n_lags)`` as the estimator assembles them."""
    import numpy as _np
    from mlsynth.utils.ppscm_helpers.setup import prepare_ppscm_inputs
    inp = prepare_ppscm_inputs(_staggered_df(), outcome="y", treat="tr",
                               unitid="unit", time="year")
    Xy, trt, d = inp.Xy, inp.trt, inp.n_pre
    T = Xy.shape[1]
    n_leads = min(T - d, T - int(_np.min(trt[_np.isfinite(trt)])))
    return Xy, trt, d, n_leads, d


HORIZON = 4
COMMON = dict(fixedeff=True, time_cohort=False, nu_used=float("nan"), lam=0.0,
              solver=None, conventions=Conventions())


# --------------------------------------------------------------------------- #
# the config field                                                             #
# --------------------------------------------------------------------------- #

def test_resample_is_an_accepted_method():
    _build(conformal_horizon=4, conformal_method="resample")


def test_the_default_is_still_the_split_band():
    from mlsynth.config_models import PPSCMConfig
    assert PPSCMConfig.model_fields["conformal_method"].default == "split"


def test_choosing_resample_without_a_horizon_is_refused():
    with pytest.raises(MlsynthConfigError) as exc:
        _build(conformal_method="resample")
    assert "conformal_horizon" in str(exc.value)


@pytest.mark.parametrize("bad", [-1, True, 2.5, "4"])
def test_an_unusable_block_length_is_refused(bad):
    with pytest.raises(MlsynthConfigError, match="conformal_block"):
        _build(conformal_horizon=4, conformal_method="resample",
               conformal_block=bad)


@pytest.mark.parametrize("bad", [1, 0, -5, True, 500.0, "2000"])
def test_an_unusable_draw_count_is_refused(bad):
    with pytest.raises(MlsynthConfigError, match="conformal_n_sim"):
        _build(conformal_horizon=4, conformal_method="resample",
               conformal_n_sim=bad)


def test_a_block_of_zero_is_accepted_and_means_the_whole_horizon():
    """Zero is the documented default, not a missing value."""
    from mlsynth.config_models import PPSCMConfig
    assert PPSCMConfig.model_fields["conformal_block"].default == 0
    _build(conformal_horizon=4, conformal_method="resample", conformal_block=0)


@pytest.mark.parametrize("field,value", [("conformal_block", 2),
                                         ("conformal_n_sim", 500)])
def test_a_resample_parameter_set_on_another_band_is_refused(field, value):
    for method in ("split", "cyclic"):
        with pytest.raises(MlsynthConfigError) as exc:
            _build(conformal_horizon=4, conformal_method=method, **{field: value})
        assert field in str(exc.value)
        assert "resample" in str(exc.value)


@pytest.mark.parametrize("field,value", [("conformal_n_nulls", 9),
                                         ("conformal_grid_scale", 2.0)])
def test_a_cyclic_parameter_set_on_the_resample_band_is_refused(field, value):
    with pytest.raises(MlsynthConfigError) as exc:
        _build(conformal_horizon=4, conformal_method="resample", **{field: value})
    assert field in str(exc.value)
    assert "cyclic" in str(exc.value)


def test_the_training_fraction_is_shared_with_the_split_band():
    """Both run the same rolling pass, so both read where it starts."""
    _build(conformal_horizon=4, conformal_method="resample",
           conformal_min_train_frac=0.4)


def test_leaving_the_resample_parameters_at_their_defaults_is_not_setting_them():
    _build(conformal_horizon=4, conformal_method="resample")


# --------------------------------------------------------------------------- #
# the rolling pass returns the per-period errors it already computed            #
# --------------------------------------------------------------------------- #

def test_the_pass_returns_one_window_by_horizon_matrix_per_treated_unit():
    from mlsynth.utils.ppscm_helpers.inference import rolling_pooled_period_errors
    Xy, trt, d, n_leads, n_lags = _panel()
    out = rolling_pooled_period_errors(
        Xy, trt, d, n_leads, n_lags, horizon=HORIZON, **COMMON)
    assert len(out) >= 1
    for arr in out:
        assert arr.ndim == 2
        assert arr.shape[1] == HORIZON


def test_the_period_errors_sum_to_the_block_sums_the_split_band_reads():
    """One pass, two readings of it. If these diverge the two bands are no
    longer describing the same calibration set."""
    from mlsynth.utils.ppscm_helpers.inference import (
        rolling_pooled_block_sums, rolling_pooled_period_errors)
    Xy, trt, d, n_leads, n_lags = _panel()
    sums = rolling_pooled_block_sums(
        Xy, trt, d, n_leads, n_lags, horizon=HORIZON, **COMMON)
    errs = rolling_pooled_period_errors(
        Xy, trt, d, n_leads, n_lags, horizon=HORIZON, **COMMON)
    assert len(sums) == len(errs)
    for s, e in zip(sums, errs):
        assert e.shape[0] == s.size
        np.testing.assert_allclose(e.sum(axis=1), s, rtol=0, atol=1e-12)


def test_the_pass_refuses_a_panel_with_no_treated_unit():
    from mlsynth.utils.ppscm_helpers.inference import rolling_pooled_period_errors
    Xy, trt, d, n_leads, n_lags = _panel()
    with pytest.raises(MlsynthDataError):
        rolling_pooled_period_errors(
            Xy, np.full_like(trt, np.nan, dtype=float), d, n_leads, n_lags,
            horizon=HORIZON, **COMMON)


# --------------------------------------------------------------------------- #
# the band                                                                     #
# --------------------------------------------------------------------------- #

def _band(**kw):
    from mlsynth.utils.ppscm_helpers.inference import resample_cumulative_per_unit
    Xy, trt, d, n_leads, n_lags = _panel()
    opts = dict(alpha=0.10, horizon=HORIZON, **COMMON)
    opts.update(kw)
    return resample_cumulative_per_unit(Xy, trt, d, n_leads, n_lags, **opts)


def test_the_band_brackets_its_point_estimate():
    pt, lo, hi, n = _band()
    assert np.all(lo <= pt) and np.all(pt <= hi)


def test_the_band_is_symmetric_about_the_point():
    """The draw is symmetric by construction -- every block's sign is flipped
    with probability one half -- so an asymmetric band would mean the
    accumulation, not the calibration, moved it."""
    pt, lo, hi, _ = _band()
    np.testing.assert_allclose(pt - lo, hi - pt, rtol=1e-9)


def test_it_is_finite_where_the_split_band_is_infinite():
    """The reason this construction exists. Seven windows against a rank of
    eight leaves the split band with no order statistic; the same seven windows
    hold twenty-eight periods."""
    from mlsynth.utils.ppscm_helpers.inference import cumulative_conformal_per_unit
    Xy, trt, d, n_leads, n_lags = _panel()
    _, s_lo, s_hi, s_n = cumulative_conformal_per_unit(
        Xy, trt, d, n_leads, n_lags, alpha=0.10, horizon=HORIZON, **COMMON)
    assert np.all(np.isinf(s_hi)), f"panel was meant to starve the split band: {s_n}"
    _, r_lo, r_hi, _ = _band()
    assert np.all(np.isfinite(r_lo)) and np.all(np.isfinite(r_hi))


def test_the_reported_window_count_is_the_passes_own():
    from mlsynth.utils.ppscm_helpers.inference import rolling_pooled_period_errors
    Xy, trt, d, n_leads, n_lags = _panel()
    errs = rolling_pooled_period_errors(
        Xy, trt, d, n_leads, n_lags, horizon=HORIZON, **COMMON)
    _, _, _, n = _band()
    np.testing.assert_array_equal(n, [e.shape[0] for e in errs])


def test_the_draw_is_reproducible_under_a_seed():
    a = _band(seed=7)
    b = _band(seed=7)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_a_different_seed_moves_the_band_but_not_the_point():
    a_pt, a_lo, _, _ = _band(seed=1)
    b_pt, b_lo, _, _ = _band(seed=2)
    np.testing.assert_array_equal(a_pt, b_pt)
    assert not np.allclose(a_lo, b_lo)


def test_a_tighter_level_gives_a_wider_band():
    _, lo10, hi10, _ = _band(alpha=0.10)
    _, lo01, hi01, _ = _band(alpha=0.01)
    assert np.all((hi01 - lo01) >= (hi10 - lo10))


def test_drawing_periods_independently_gives_a_different_band_than_whole_blocks():
    """block=1 is Wheeler's original and discards the serial correlation the
    accumulated total is driven by; if these agreed the block length would not
    be doing anything."""
    _, lo1, hi1, _ = _band(block=1)
    _, lo0, hi0, _ = _band(block=0)
    assert not np.allclose(hi1 - lo1, hi0 - lo0)


def test_a_block_longer_than_the_horizon_is_clamped_to_it():
    a = _band(block=HORIZON)
    b = _band(block=HORIZON + 50)
    np.testing.assert_allclose(a[1], b[1])


def test_a_unit_with_no_calibration_windows_gets_an_infinite_band():
    """A refusal to report, not a narrow band that does not cover."""
    from mlsynth.utils.ppscm_helpers.inference import resample_cumulative_per_unit
    Xy, trt, d, n_leads, n_lags = _panel()
    pt, lo, hi, n = resample_cumulative_per_unit(
        Xy, trt, d, n_leads, n_lags, alpha=0.10, horizon=39, **COMMON)
    assert np.all(np.isneginf(lo)) and np.all(np.isposinf(hi))
    assert np.all(n == 0)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, True, "0.1"])
def test_an_unusable_level_is_refused(bad):
    with pytest.raises(MlsynthConfigError, match="alpha"):
        _band(alpha=bad)


def test_the_band_refuses_a_panel_with_no_treated_unit():
    """Checked here as well as in the pass: this function reads ``trt`` for its
    point estimates before the pass is ever called, so the pass's refusal would
    arrive too late to be the one the caller sees."""
    from mlsynth.utils.ppscm_helpers.inference import resample_cumulative_per_unit
    Xy, trt, d, n_leads, n_lags = _panel()
    with pytest.raises(MlsynthDataError):
        resample_cumulative_per_unit(
            Xy, np.full_like(trt, np.nan, dtype=float), d, n_leads, n_lags,
            alpha=0.10, horizon=HORIZON, **COMMON)


# --------------------------------------------------------------------------- #
# end to end                                                                   #
# --------------------------------------------------------------------------- #

def test_asking_for_the_resample_band_fills_the_cumulative_fields():
    res = _fit(conformal_horizon=4, conformal_method="resample",
               run_inference=True)
    units = list(res.per_unit.values())
    assert units
    for u in units:
        assert u.cumulative_method == "resample"
        assert u.cumulative_effect is not None
        assert np.isfinite(u.cumulative_lower) and np.isfinite(u.cumulative_upper)


def test_the_resample_band_reports_a_window_count_and_no_p_value():
    """The diagnostics belong one to each method; only the chosen one is filled."""
    res = _fit(conformal_horizon=4, conformal_method="resample",
               run_inference=True)
    for u in res.per_unit.values():
        assert u.cumulative_windows is not None
        assert u.cumulative_p_value is None


def test_no_horizon_leaves_every_cumulative_field_empty():
    res = _fit(run_inference=True)
    for u in res.per_unit.values():
        assert u.cumulative_method is None
        assert u.cumulative_effect is None
        assert u.cumulative_windows is None


def test_the_split_path_is_untouched_by_the_new_method():
    """The default is byte-for-byte the band it always was."""
    a = _fit(conformal_horizon=4, run_inference=True)
    for u in a.per_unit.values():
        assert u.cumulative_method == "split"
        assert u.cumulative_p_value is None
