"""Choosing which cumulative conformal band PPSCM reports.

``cwz_cumulative_per_unit`` calibrates against the cyclic shifts of the residual
path; ``cumulative_conformal_per_unit`` calibrates against a disjoint split of
the pre-period. They report the same estimand -- the total a treated unit gained
over ``conformal_horizon`` periods -- from different reference sets, so
``conformal_method`` selects between them the way ``inference_method`` selects
the bootstrap or the jackknife behind the ATT.

The split band's reference set is the number of non-overlapping calibration
windows, so a finite ``1 - alpha`` band needs ``ceil((m+1)(1-alpha)) <= m`` and
there is a floor of roughly ``12.8 * L`` pre-periods before one exists. The
cyclic reference set does not depend on the horizon, so neither the floor nor
the regime past it applies. The price is a shape assumption: the cyclic band
inverts a test against a constant per-period effect, and an effect that ramps is
outside that null family, which comes back as an empty accepted set.

The two carry different diagnostics -- a window count for the split, a
permutation p-value for the cyclic -- so each fills its own field and
``cumulative_method`` says which one produced the band.
"""

import numpy as np
import pandas as pd
import pytest

from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.ppscm_helpers.engine import Conventions
from mlsynth.utils.ppscm_helpers.inference import cwz_cumulative_per_unit


def _staggered_df(seed=0, adoption=(14, 18), n_donors=8, T=34, effect=-3.0, noise=0.4):
    """Staggered panel with enough pre-period for several calibration windows."""
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


def _fit(**kw):
    import warnings as _w
    from mlsynth import PPSCM
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        return PPSCM(_cfg(**kw)).fit()


def _build(**kw):
    from mlsynth import PPSCM
    return PPSCM(_cfg(**kw))


# --------------------------------------------------------------------------- #
# the config field                                                             #
# --------------------------------------------------------------------------- #

def test_the_default_is_the_split_band():
    from mlsynth.config_models import PPSCMConfig
    assert PPSCMConfig.model_fields["conformal_method"].default == "split"


@pytest.mark.parametrize("bad", ["cwz", "Split", "", None, 3])
def test_an_unknown_method_is_refused_by_name(bad):
    with pytest.raises(MlsynthConfigError, match="conformal_method"):
        _build(conformal_horizon=3, conformal_method=bad)


def test_choosing_a_method_without_a_horizon_is_refused():
    """The horizon is what turns the band on; a method with no band is a typo."""
    with pytest.raises(MlsynthConfigError) as exc:
        _build(conformal_method="cyclic")
    assert "conformal_horizon" in str(exc.value)
    assert "conformal_method" in str(exc.value)


@pytest.mark.parametrize("field,value", [("conformal_n_nulls", 9),
                                         ("conformal_grid_scale", 2.0)])
def test_a_cyclic_parameter_set_on_the_split_band_is_refused(field, value):
    with pytest.raises(MlsynthConfigError) as exc:
        _build(conformal_horizon=3, **{field: value})
    assert field in str(exc.value)
    assert "cyclic" in str(exc.value)


def test_the_split_parameter_set_on_the_cyclic_band_is_refused():
    with pytest.raises(MlsynthConfigError) as exc:
        _build(conformal_horizon=3, conformal_method="cyclic",
               conformal_min_train_frac=0.4)
    assert "conformal_min_train_frac" in str(exc.value)
    assert "split" in str(exc.value)


def test_leaving_a_parameter_at_its_default_is_not_setting_it():
    """The refusal is about what the caller asked for, not what the field holds."""
    _build(conformal_horizon=3)                                    # split defaults
    _build(conformal_horizon=3, conformal_method="cyclic")         # cyclic defaults


@pytest.mark.parametrize("bad", [2, 0, -1, True, 5.0, "9"])
def test_an_unusable_null_grid_size_is_refused(bad):
    with pytest.raises(MlsynthConfigError, match="conformal_n_nulls"):
        _build(conformal_horizon=3, conformal_method="cyclic", conformal_n_nulls=bad)


@pytest.mark.parametrize("bad", [0.0, -1.0, True, "3"])
def test_an_unusable_grid_scale_is_refused(bad):
    with pytest.raises(MlsynthConfigError, match="conformal_grid_scale"):
        _build(conformal_horizon=3, conformal_method="cyclic", conformal_grid_scale=bad)


# --------------------------------------------------------------------------- #
# routing                                                                      #
# --------------------------------------------------------------------------- #

def test_no_horizon_leaves_every_cumulative_field_empty():
    res = _fit()
    assert res.per_unit
    for unit in res.per_unit.values():
        assert unit.cumulative_effect is None
        assert unit.cumulative_lower is None
        assert unit.cumulative_upper is None
        assert unit.cumulative_windows is None
        assert unit.cumulative_p_value is None
        assert unit.cumulative_method is None


def test_the_split_band_reports_its_window_count_and_no_p_value():
    res = _fit(conformal_horizon=3, alpha=0.2)
    assert res.per_unit
    for unit in res.per_unit.values():
        assert unit.cumulative_method == "split"
        assert unit.cumulative_windows is not None and unit.cumulative_windows >= 1
        assert unit.cumulative_p_value is None
        assert unit.cumulative_lower <= unit.cumulative_effect <= unit.cumulative_upper


def test_the_cyclic_band_reports_its_p_value_and_no_window_count():
    res = _fit(conformal_horizon=3, alpha=0.2, conformal_method="cyclic",
               conformal_n_nulls=5)
    assert res.per_unit
    for unit in res.per_unit.values():
        assert unit.cumulative_method == "cyclic"
        assert unit.cumulative_windows is None
        assert 0.0 <= unit.cumulative_p_value <= 1.0
        assert unit.cumulative_effect is not None


def test_naming_the_default_method_changes_nothing():
    """The pin on existing behaviour: ``split`` is what a caller already got."""
    implicit = _fit(conformal_horizon=3, alpha=0.2)
    explicit = _fit(conformal_horizon=3, alpha=0.2, conformal_method="split")
    for key, unit in implicit.per_unit.items():
        other = explicit.per_unit[key]
        assert other.cumulative_effect == pytest.approx(unit.cumulative_effect)
        assert other.cumulative_lower == pytest.approx(unit.cumulative_lower)
        assert other.cumulative_upper == pytest.approx(unit.cumulative_upper)
        assert other.cumulative_windows == unit.cumulative_windows


def test_the_two_methods_agree_on_the_point_and_differ_on_the_band():
    """Same estimand, different reference set."""
    split = _fit(conformal_horizon=3, alpha=0.2)
    cyclic = _fit(conformal_horizon=3, alpha=0.2, conformal_method="cyclic",
                  conformal_n_nulls=5)
    widths = []
    for key, unit in split.per_unit.items():
        other = cyclic.per_unit[key]
        assert other.cumulative_effect == pytest.approx(unit.cumulative_effect)
        widths.append((unit.cumulative_upper - unit.cumulative_lower,
                       other.cumulative_upper - other.cumulative_lower))
    assert any(a != b for a, b in widths)


def test_the_band_is_additional_and_does_not_disturb_the_att():
    base = _fit(run_inference=True, alpha=0.2)
    with_band = _fit(run_inference=True, alpha=0.2, conformal_horizon=3,
                     conformal_method="cyclic", conformal_n_nulls=5)
    for key, unit in base.per_unit.items():
        other = with_band.per_unit[key]
        assert other.att == pytest.approx(unit.att)
        np.testing.assert_allclose(other.tau, unit.tau, equal_nan=True)


# --------------------------------------------------------------------------- #
# edges                                                                        #
# --------------------------------------------------------------------------- #

def _single_treated_df(seed=0, Tj=18, n_donors=8, T=34, effect=-3.0, noise=0.4):
    rng = np.random.default_rng(seed)
    factors = rng.standard_normal((T, 2))
    load_d = rng.standard_normal((n_donors, 2)) * 0.5
    series = factors @ load_d.mean(axis=0) + rng.standard_normal(T) * noise
    series[Tj:] += effect
    rec = [{"unit": "treated_0", "year": 2000 + t, "y": float(series[t]),
            "tr": int(t >= Tj)} for t in range(T)]
    for dd in range(n_donors):
        z = factors @ load_d[dd] + rng.standard_normal(T) * noise
        rec += [{"unit": f"d_{dd}", "year": 2000 + t, "y": float(z[t]), "tr": 0}
                for t in range(T)]
    return pd.DataFrame(rec)


def test_a_single_treated_unit_gets_a_cyclic_band():
    """The automatic pooling rule sits on the program's boundary at one treated
    unit, so the null refits are given the fit's own ``nu`` instead of asking
    for it again. Without that they would be infeasible and the band would
    never be computed."""
    res = _fit(df=_single_treated_df(), conformal_horizon=3, alpha=0.2,
               conformal_method="cyclic", conformal_n_nulls=5)
    assert len(res.per_unit) == 1
    unit = next(iter(res.per_unit.values()))
    assert unit.cumulative_method == "cyclic"
    assert np.isfinite(unit.cumulative_effect)
    assert 0.0 <= unit.cumulative_p_value <= 1.0


def test_the_null_refits_are_given_the_fits_own_pooling_level():
    from mlsynth.estimators import ppscm as mod
    seen = {}
    original = mod.cwz_cumulative_per_unit

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return original(*args, **kwargs)

    mod.cwz_cumulative_per_unit = spy
    try:
        _fit(conformal_horizon=3, alpha=0.2, conformal_method="cyclic",
             conformal_n_nulls=5)
    finally:
        mod.cwz_cumulative_per_unit = original
    assert np.isfinite(seen["nu_used"])
    assert seen["n_nulls"] == 5
    assert seen["conventions"] is not None


@pytest.mark.parametrize("method", ["split", "cyclic"])
def test_a_horizon_past_the_post_period_is_reported_and_not_swallowed(method):
    from mlsynth.exceptions import MlsynthDataError
    kw = {"conformal_n_nulls": 5} if method == "cyclic" else {}
    with pytest.raises(MlsynthDataError):
        _fit(conformal_horizon=99, alpha=0.2, conformal_method=method, **kw)


# --------------------------------------------------------------------------- #
# what the cyclic band reports when it has nothing to report                   #
# --------------------------------------------------------------------------- #

def test_an_empty_accepted_set_is_carried_as_nan_and_not_as_absent():
    """``None`` means no band was asked for; ``nan`` means none was accepted.

    Collapsing the two would make an effect outside the constant null family
    indistinguishable from a fit that never computed a band.
    """
    from mlsynth.estimators import ppscm as mod
    nan = np.array([np.nan, np.nan])
    stub = (np.array([1.0, 2.0]), nan, nan, np.array([0.5, 0.5]))
    original = mod.cwz_cumulative_per_unit
    mod.cwz_cumulative_per_unit = lambda *a, **k: stub
    try:
        res = _fit(conformal_horizon=3, alpha=0.2, conformal_method="cyclic",
                   conformal_n_nulls=5)
    finally:
        mod.cwz_cumulative_per_unit = original
    for unit in res.per_unit.values():
        assert unit.cumulative_method == "cyclic"
        assert unit.cumulative_effect is not None
        assert np.isnan(unit.cumulative_lower)
        assert np.isnan(unit.cumulative_upper)


# --------------------------------------------------------------------------- #
# the conventions the wiring forces through the helper                         #
# --------------------------------------------------------------------------- #

def _panel(n_ctrl=10, n_trt=2, t0=24, horizon=4, seed=3):
    rng = np.random.default_rng(seed)
    n, T = n_ctrl + n_trt, t0 + horizon
    f = rng.normal(size=(T, 2))
    load = rng.uniform(0.3, 1.3, size=(n, 2))
    Xy = load @ f.T + rng.normal(scale=0.3, size=(n, T))
    Xy = Xy - Xy.min() + 1.0
    trt = np.full(n, np.inf)
    # Two cohorts far enough apart that the later one is inside the earlier
    # one's estimation window: "window" admits it as a donor, "never_treated"
    # does not, so the two conventions are different estimators here.
    trt[0] = t0 - 8
    trt[1] = t0
    return Xy, trt, t0, horizon


def _cwz(**kw):
    Xy, trt, t0, L = _panel()
    base = dict(d=t0, n_leads=L, n_lags=t0, fixedeff=True, time_cohort=False,
                nu_used=0.5, lam=0.0, solver=None, alpha=0.10, horizon=L,
                n_nulls=5)
    base.update(kw)
    return cwz_cumulative_per_unit(Xy, trt, **base)


def test_the_default_conventions_reproduce_the_call_without_them():
    """#511's behaviour is the ``Conventions()`` default, unchanged."""
    a = _cwz()
    b = _cwz(conventions=Conventions())
    for x, y in zip(a, b):
        np.testing.assert_allclose(x, y, equal_nan=True)


def test_a_different_estimator_moves_what_the_cyclic_band_reports():
    """Without this the band would answer for augsynth's estimator, not the fit's."""
    a = _cwz()
    b = _cwz(conventions=Conventions(donor_weights="uniform"))
    assert not np.allclose(a[0], b[0], equal_nan=True)


def test_every_refit_the_cyclic_band_makes_uses_the_configured_estimator():
    """The observed fit and each null refit alike.

    A null refit left on the defaults would calibrate the band against an
    estimator the caller did not configure, and the point estimate would still
    look right, so the point alone cannot pin this.
    """
    import mlsynth.utils.ppscm_helpers.inference as inf
    conv = Conventions(donor_weights="uniform")
    seen, original = [], inf.run_multisynth

    def spy(*args, **kwargs):
        seen.append(kwargs.get("conventions"))
        return original(*args, **kwargs)

    inf.run_multisynth = spy
    try:
        _cwz(conventions=conv, n_nulls=3)
    finally:
        inf.run_multisynth = original
    assert len(seen) > 1                       # the observed fit, then the nulls
    assert all(c == conv for c in seen)
