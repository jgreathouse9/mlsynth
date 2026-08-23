"""The resampled cumulative band for PDA: Wheeler's construction, wired in.

``cumulative_band`` today reuses the paths ``pda_prediction_intervals`` produced,
so it needs ``prediction_intervals=True`` and costs ``pi_n_boot`` refits -- 999 at
the default. ``cumulative_method="resample"`` builds the same object from a
rolling-origin calibration pass instead: one refit per origin, roughly ten, and no
bootstrap at all.

Two things follow from that and are pinned here. The band no longer depends on the
prediction intervals, so the config's ``cumulative_band needs prediction_intervals``
rule must not apply to it. And the object handed back is the same
``PDACumulativeBand`` the bootstrap path returns, so every reader downstream is
untouched -- only ``method`` records which construction produced it.

Levels: smoke, unit invariants, edge, failure.
"""
import numpy as np
import pandas as pd
import pytest

from mlsynth import PDA
from pydantic import ValidationError

from mlsynth.exceptions import MlsynthConfigError


def _panel(n_donors=8, T0=40, T1=6, effect=1.0, seed=0, rho=0.0):
    rng = np.random.default_rng(seed)
    T = T0 + T1
    f = rng.standard_normal((T, 2))
    load_d = rng.standard_normal((n_donors, 2)) * 0.5
    e = rng.standard_normal((n_donors + 1, T)) * 0.3
    if rho:
        for t in range(1, T):
            e[:, t] = rho * e[:, t - 1] + np.sqrt(1 - rho ** 2) * e[:, t]
    rec = []
    y = f @ load_d.mean(axis=0) + e[0]
    y[T0:] += effect
    rec += [{"unit": "treated", "t": t, "y": float(y[t]), "d": int(t >= T0)}
            for t in range(T)]
    for j in range(n_donors):
        s = f @ load_d[j] + e[j + 1]
        rec += [{"unit": f"d{j}", "t": t, "y": float(s[t]), "d": 0} for t in range(T)]
    return pd.DataFrame(rec)


def _fit(df=None, **kw):
    cfg = dict(df=_panel() if df is None else df, outcome="y", treat="d",
               unitid="unit", time="t", method="LASSO", display_graphs=False)
    cfg.update(kw)
    return PDA(cfg).fit()


def _fit_obj(res, method="lasso"):
    return res.fits[method] if isinstance(res.fits, dict) else res.fits[0]


def _resampled(**kw):
    base = dict(cumulative_band=True, cumulative_method="resample",
                cumulative_n_sim=400)
    base.update(kw)
    return _fit(**base)


# --------------------------------------------------------------------------- #
# smoke
# --------------------------------------------------------------------------- #
def test_it_attaches_a_band_without_any_bootstrap():
    res = _resampled()
    fit = _fit_obj(res)
    assert fit.prediction_intervals is None
    b = fit.cumulative_band
    assert b is not None
    for arr in (b.horizons, b.point, b.lower, b.upper, b.se):
        assert np.asarray(arr).shape == (6,)
    assert np.isfinite(np.asarray(b.point)).all()


# --------------------------------------------------------------------------- #
# unit invariants
# --------------------------------------------------------------------------- #
def test_the_construction_is_recorded_on_the_band():
    assert _fit_obj(_resampled()).cumulative_band.method == "resample"
    boot = _fit(prediction_intervals=True, cumulative_band=True, pi_n_boot=200)
    assert _fit_obj(boot).cumulative_band.method == "bootstrap"


def test_it_draws_the_number_of_paths_it_was_asked_for():
    assert _fit_obj(_resampled(cumulative_n_sim=250)).cumulative_band.n_replicates == 250


def test_the_point_is_the_running_total_of_the_period_effects():
    res = _resampled()
    gap = np.asarray(_fit_obj(res).gap, dtype=float)[-6:]
    np.testing.assert_allclose(_fit_obj(res).cumulative_band.point, np.cumsum(gap),
                               rtol=1e-10)


def test_the_band_brackets_its_point_at_every_horizon():
    b = _fit_obj(_resampled()).cumulative_band
    assert (np.asarray(b.lower) <= np.asarray(b.point)).all()
    assert (np.asarray(b.point) <= np.asarray(b.upper)).all()


def test_the_same_seed_gives_the_same_band():
    a = _fit_obj(_resampled(pi_seed=7)).cumulative_band
    b = _fit_obj(_resampled(pi_seed=7)).cumulative_band
    np.testing.assert_allclose(a.lower, b.lower)
    np.testing.assert_allclose(a.upper, b.upper)


def test_a_longer_block_widens_the_band_on_a_persistent_panel():
    """Blocks carry serial correlation an independent draw discards."""
    df = _panel(rho=0.7, seed=3)
    one = _fit_obj(_resampled(df=df, cumulative_block=1)).cumulative_band
    full = _fit_obj(_resampled(df=df, cumulative_block=0)).cumulative_band
    assert full.se[-1] > one.se[-1]


def test_the_width_does_not_depend_on_the_size_of_the_effect():
    """Calibration is pre-period only, so the treatment cannot widen the band.

    A schedule that ran past T0 would score windows containing the effect, and the
    band would grow with it -- calibrating on the thing being measured.
    """
    quiet = _fit_obj(_resampled(df=_panel(effect=0.0, seed=5))).cumulative_band
    loud = _fit_obj(_resampled(df=_panel(effect=50.0, seed=5))).cumulative_band
    assert loud.se[-1] == pytest.approx(quiet.se[-1], rel=0.02)


def test_the_lasso_penalty_is_pinned_before_the_origins_are_refit():
    """Every calibration window must be the same estimator as the reported fit.

    With no penalty pinned, fit_lasso re-tunes by cross-validation at each origin,
    so the windows calibrate a different estimator at each one. The penalty
    actually used is recorded, which is what this asserts.
    """
    fit = _fit_obj(_resampled())
    assert fit.metadata.get("alpha") is not None
    assert np.isfinite(float(fit.metadata["alpha"]))


def test_asking_for_prediction_intervals_as_well_still_gives_both():
    res = _resampled(prediction_intervals=True, pi_n_boot=200)
    fit = _fit_obj(res)
    assert fit.prediction_intervals is not None
    assert fit.cumulative_band.method == "resample"


# --------------------------------------------------------------------------- #
# edge
# --------------------------------------------------------------------------- #
def test_it_is_off_unless_asked_for():
    assert _fit_obj(_fit(cumulative_method="resample")).cumulative_band is None


def test_every_pda_variant_can_produce_one():
    """The reader asks for a counterfactual, which all four variants return."""
    for m in ("LASSO", "l2", "fs", "hcw"):
        b = _fit_obj(_resampled(method=m), method=m.lower()).cumulative_band
        assert b is not None and b.method == "resample"


# --------------------------------------------------------------------------- #
# failure
# --------------------------------------------------------------------------- #
def test_the_bootstrap_band_still_needs_prediction_intervals():
    with pytest.raises(MlsynthConfigError, match="prediction_intervals"):
        _fit(cumulative_band=True)


def test_the_resampled_band_does_not_need_them():
    assert _fit_obj(_resampled()).cumulative_band is not None


@pytest.mark.parametrize("bad", ["wheeler", "", "Resample", None, 1])
def test_an_unknown_construction_is_refused(bad):
    with pytest.raises(MlsynthConfigError):
        _fit(cumulative_band=True, cumulative_method=bad)


@pytest.mark.parametrize("bad", [2.5, "3", True, None])
def test_a_non_integer_block_is_refused(bad):
    """A rounded block length is a silent change to how much correlation is kept."""
    with pytest.raises(MlsynthConfigError, match="cumulative_block"):
        _resampled(cumulative_block=bad)


@pytest.mark.parametrize("bad", [-1, -8])
def test_a_negative_block_is_refused(bad):
    with pytest.raises(ValidationError):
        _resampled(cumulative_block=bad)


@pytest.mark.parametrize("bad", [2.5, "400", True, None])
def test_a_non_integer_simulation_count_is_refused(bad):
    with pytest.raises(MlsynthConfigError, match="cumulative_n_sim"):
        _resampled(cumulative_n_sim=bad)


@pytest.mark.parametrize("bad", [0, 1, -5])
def test_too_few_simulations_to_take_a_quantile_is_refused(bad):
    with pytest.raises(ValidationError):
        _resampled(cumulative_n_sim=bad)
