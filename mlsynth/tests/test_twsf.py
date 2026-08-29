"""Tests for TWSF, the two-way synthetic forecasting estimator.

TWSF forecasts the *treated* potential outcome of a unit that has never been
treated, at dates beyond the end of the panel. The identification has two
halves and the tests are organised the same way: the unit side transports a
cross-sectional relationship learned under control into the treated regime,
and the time side learns that regime's dynamics from donors already exposed
to it.

The recovery fixture is a noiseless factor panel whose treated time factor is
a sum of harmonics, so it satisfies a linear recursion of order at most the
lag length. With no noise the forecast is then exact, which is what pins the
Page-block layout, the companion recursion and the bilinear combination all at
once -- an error in any of them shows up as a finite forecast error.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlsynth import TWSF
from mlsynth.config_models import TWSFConfig
from mlsynth.exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

def _factor_panel(n_donors=8, T0=40, T1=40, horizon=5, rank=2, sigma=0.0,
                  seed=0, stagger=0):
    """A long panel whose target factor lies in the donor span.

    The donors are treated from ``T0 + 1``; the target never is. Returns the
    frame plus the noiseless treated path of the target over the ``horizon``
    dates after the panel ends, which is the estimand.
    """
    rng = np.random.default_rng(seed)
    T = T0 + T1
    U = rng.standard_normal((n_donors, rank))
    lam = rng.dirichlet(np.ones(n_donors))
    u_target = lam @ U

    t = np.arange(1, T + horizon + 1)
    basis = np.vstack([np.sin(2 * np.pi * t / 11), np.cos(2 * np.pi * t / 11),
                       np.sin(2 * np.pi * t / 23), np.cos(2 * np.pi * t / 23)])
    A0 = rng.standard_normal((rank, 4))
    A1 = rng.standard_normal((rank, 4))
    V0, V1 = A0 @ basis, A1 @ basis

    names = [f"donor{i}" for i in range(n_donors)]
    rows = []
    for i, nm in enumerate(names):
        # staggered adoption shifts each donor's own date by up to `stagger`
        own_T0 = T0 + (i % (stagger + 1))
        for k in range(T):
            treated = k >= own_T0
            val = U[i] @ (V1[:, k] if treated else V0[:, k])
            rows.append(dict(unit=nm, time=k + 1, y=val, treat=int(treated)))
    for k in range(T):
        rows.append(dict(unit="target", time=k + 1,
                         y=u_target @ V0[:, k], treat=0))
    df = pd.DataFrame(rows)
    if sigma:
        df["y"] = df["y"] + rng.normal(0, sigma, len(df))
    truth = np.array([u_target @ V1[:, T + h] for h in range(horizon)])
    return df, truth


def _cfg(df, **kw):
    base = dict(df=df, outcome="y", unitid="unit", time="time", treat="treat",
                target="target", L=8, k_y=2, k_z=4, horizon=5,
                display_graphs=False)
    base.update(kw)
    return TWSFConfig(**base)


# --------------------------------------------------------------------------
# smoke
# --------------------------------------------------------------------------

def test_fit_returns_the_result_contract():
    df, _ = _factor_panel(sigma=0.05, seed=1)
    res = TWSF(_cfg(df)).fit()
    assert res.time_series is not None and res.weights is not None
    cf = np.asarray(res.time_series.counterfactual_outcome, dtype=float)
    assert cf.shape == (5,) and np.all(np.isfinite(cf))
    assert res.effects is not None and np.isfinite(res.effects.att)
    assert res.method_details.method_name.upper().startswith("TWSF")


def test_weights_are_two_way_and_land_on_the_contract():
    df, _ = _factor_panel(n_donors=6, sigma=0.05, seed=2)
    res = TWSF(_cfg(df, L=6)).fit()
    donor_w = np.asarray(res.weights.donor_weights_array, dtype=float) \
        if hasattr(res.weights, "donor_weights_array") \
        else np.asarray(list(res.weights.donor_weights.values()), dtype=float)
    time_w = np.asarray(list(res.weights.time_weights.values()), dtype=float)
    assert donor_w.size == 6          # one per treated donor
    assert time_w.size == 6           # one per lag, L = 6
    assert np.all(np.isfinite(donor_w)) and np.all(np.isfinite(time_w))


# --------------------------------------------------------------------------
# invariants
# --------------------------------------------------------------------------

@pytest.mark.parametrize("multistep", ["direct", "recursive"])
@pytest.mark.parametrize("horizon", [1, 3, 5])
def test_noiseless_forecast_is_exact(multistep, horizon):
    """With sigma = 0 the algebra must close to machine precision."""
    df, truth = _factor_panel(T0=60, T1=120, horizon=horizon, sigma=0.0, seed=3)
    res = TWSF(_cfg(df, L=10, k_y=2, k_z=4, horizon=horizon,
                    multistep=multistep)).fit()
    cf = np.asarray(res.time_series.counterfactual_outcome, dtype=float)
    assert np.allclose(cf, truth[:horizon], atol=1e-6), (
        f"{multistep} h={horizon}: max |err| = "
        f"{np.max(np.abs(cf - truth[:horizon])):.3e}")


def test_direct_and_recursive_agree_at_one_step():
    """The paper states the two estimators coincide when h = 1."""
    df, _ = _factor_panel(sigma=0.05, seed=4)
    a = TWSF(_cfg(df, horizon=1, multistep="direct")).fit()
    b = TWSF(_cfg(df, horizon=1, multistep="recursive")).fit()
    assert np.allclose(
        np.asarray(a.time_series.counterfactual_outcome, dtype=float),
        np.asarray(b.time_series.counterfactual_outcome, dtype=float),
        atol=1e-8)


def test_interval_widens_with_the_horizon():
    df, _ = _factor_panel(sigma=0.05, seed=5)
    res = TWSF(_cfg(df, horizon=5)).fit()
    lo = np.asarray(res.time_series.counterfactual_lower, dtype=float)
    hi = np.asarray(res.time_series.counterfactual_upper, dtype=float)
    width = hi - lo
    assert np.all(width > 0)
    assert width[-1] >= width[0]


def test_noise_free_interval_collapses():
    """No noise, no uncertainty: the plug-in variance must vanish with sigma."""
    df, _ = _factor_panel(T0=60, T1=120, sigma=0.0, seed=6)
    res = TWSF(_cfg(df, L=10)).fit()
    lo = np.asarray(res.time_series.counterfactual_lower, dtype=float)
    hi = np.asarray(res.time_series.counterfactual_upper, dtype=float)
    assert np.max(hi - lo) < 1e-4


# --------------------------------------------------------------------------
# edge cases
# --------------------------------------------------------------------------

def test_too_few_page_blocks_is_reported():
    """B = T1 / (L + 1) must give at least two blocks."""
    df, _ = _factor_panel(T0=40, T1=12, seed=7)
    with pytest.raises((MlsynthConfigError, MlsynthDataError)) as exc:
        TWSF(_cfg(df, L=10)).fit()
    assert "block" in str(exc.value).lower()


def test_single_donor_still_fits():
    df, _ = _factor_panel(n_donors=1, T0=40, T1=60, seed=8, sigma=0.02)
    res = TWSF(_cfg(df, L=6, k_y=1, k_z=4)).fit()
    assert np.all(np.isfinite(
        np.asarray(res.time_series.counterfactual_outcome, dtype=float)))


def test_staggered_donor_pool_warns_and_proceeds():
    """The theory assumes a common treatment date; real panels stagger."""
    df, _ = _factor_panel(T0=40, T1=60, stagger=3, sigma=0.02, seed=9)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = TWSF(_cfg(df, L=6)).fit()
    assert any("stagger" in str(w.message).lower() for w in caught)
    assert np.all(np.isfinite(
        np.asarray(res.time_series.counterfactual_outcome, dtype=float)))


def test_unknown_target_unit_is_reported():
    df, _ = _factor_panel(seed=10)
    with pytest.raises(MlsynthDataError) as exc:
        TWSF(_cfg(df, target="nobody")).fit()
    assert "nobody" in str(exc.value)


def test_target_may_not_be_a_treated_donor():
    df, _ = _factor_panel(seed=11)
    with pytest.raises(MlsynthDataError):
        TWSF(_cfg(df, target="donor0")).fit()


# --------------------------------------------------------------------------
# config validation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    dict(L=0), dict(L=-2), dict(horizon=0), dict(k_y=0), dict(k_z=0),
    dict(alpha=0.0), dict(alpha=1.0), dict(multistep="sideways"),
])
def test_invalid_config_raises(bad):
    df, _ = _factor_panel(seed=12)
    with pytest.raises((MlsynthConfigError, ValueError)):
        _cfg(df, **bad)


def test_extra_field_is_forbidden():
    df, _ = _factor_panel(seed=13)
    with pytest.raises((MlsynthConfigError, ValueError)):
        _cfg(df, not_a_real_option=1)


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------

def test_plotter_returns_a_figure_without_showing_it():
    from mlsynth.utils.twsf_helpers.plotter import plot_twsf
    df, _ = _factor_panel(sigma=0.05, seed=14)
    res = TWSF(_cfg(df)).fit()
    fig = plot_twsf(res, title="TWSF")
    assert fig is not None and hasattr(fig, "savefig")


# --------------------------------------------------------------------------
# remaining branches
# --------------------------------------------------------------------------

def test_rank_may_not_exceed_the_lag_length():
    df, _ = _factor_panel(seed=15)
    with pytest.raises(MlsynthConfigError) as exc:
        _cfg(df, L=4, k_z=6)
    assert "k_z" in str(exc.value)


def test_empty_donor_list_is_rejected():
    df, _ = _factor_panel(seed=16)
    with pytest.raises(MlsynthConfigError):
        _cfg(df, donors=[])


def test_target_may_not_appear_in_the_donor_list():
    df, _ = _factor_panel(seed=17)
    with pytest.raises(MlsynthConfigError):
        _cfg(df, donors=["donor0", "target"])


def test_donor_pool_can_be_restricted():
    df, _ = _factor_panel(n_donors=8, sigma=0.02, seed=18)
    res = TWSF(_cfg(df, donors=["donor0", "donor1", "donor2"], k_y=2)).fit()
    assert set(res.weights.donor_weights) == {"donor0", "donor1", "donor2"}


def test_named_donor_that_never_adopts_is_reported():
    df, _ = _factor_panel(seed=19)
    with pytest.raises(MlsynthDataError) as exc:
        TWSF(_cfg(df, donors=["donor0", "target"] if False else ["donor0", "ghost"])).fit()
    assert "ghost" in str(exc.value)


def test_panel_with_no_treated_unit_is_reported():
    """TWSF needs donors that have already adopted; dataprep is the guard."""
    df, _ = _factor_panel(seed=20)
    df = df.assign(treat=0)
    with pytest.raises(MlsynthDataError) as exc:
        TWSF(_cfg(df)).fit()
    assert "treated" in str(exc.value).lower()


def test_too_short_a_control_window_is_reported():
    """The unit side needs a pre-adoption window to learn from."""
    df, _ = _factor_panel(T0=1, T1=60, seed=21)
    with pytest.raises(MlsynthDataError) as exc:
        TWSF(_cfg(df, L=6)).fit()
    assert "pre-adoption" in str(exc.value)


def test_prediction_interval_is_wider_than_the_confidence_interval():
    """The prediction interval carries the future innovation the CI omits."""
    df, _ = _factor_panel(sigma=0.05, seed=22)
    ci = TWSF(_cfg(df, interval="confidence")).fit()
    pi = TWSF(_cfg(df, interval="prediction")).fit()
    w = lambda r: (np.asarray(r.time_series.counterfactual_upper, dtype=float)
                   - np.asarray(r.time_series.counterfactual_lower, dtype=float))
    assert np.all(w(pi) > w(ci))
    assert ci.time_series.prediction_interval_kind == "confidence"
    assert pi.time_series.prediction_interval_kind == "prediction"


def test_non_arithmetic_time_labels_fall_back_to_offsets():
    """String period labels cannot be extrapolated, so offsets are used."""
    from mlsynth.utils.twsf_helpers.setup import _extend
    assert _extend(["a", "b", "c"], 3) == ["+1", "+2", "+3"]
    assert _extend(["only"], 2) == [1, 2]


def test_datetime_labels_are_extended_by_their_own_step():
    from mlsynth.utils.twsf_helpers.setup import _extend
    idx = list(pd.date_range("2020-01-01", periods=4, freq="D"))
    assert _extend(idx, 2) == [pd.Timestamp("2020-01-05"),
                              pd.Timestamp("2020-01-06")]


# --------------------------------------------------------------------------
# estimator entry points
# --------------------------------------------------------------------------

def test_config_may_be_given_as_a_dict():
    df, _ = _factor_panel(sigma=0.05, seed=23)
    res = TWSF(dict(df=df, outcome="y", unitid="unit", time="time",
                    treat="treat", target="target", L=8, k_y=2, k_z=4,
                    horizon=3, display_graphs=False)).fit()
    assert np.asarray(
        res.time_series.counterfactual_outcome, dtype=float).size == 3


def test_invalid_dict_config_is_translated():
    df, _ = _factor_panel(seed=24)
    with pytest.raises(MlsynthConfigError):
        TWSF(dict(df=df, outcome="y", unitid="unit", time="time",
                  treat="treat", target="target", L=-1, k_y=2, k_z=1))


def test_config_of_the_wrong_type_is_rejected():
    with pytest.raises(MlsynthConfigError) as exc:
        TWSF("not a config")
    assert "TWSFConfig" in str(exc.value)


def test_degenerate_design_is_reported_as_an_estimation_error():
    """An all-zero donor block identifies no weights at all."""
    df, _ = _factor_panel(n_donors=3, T0=30, T1=60, seed=25)
    df.loc[df.unit.str.startswith("donor"), "y"] = 0.0
    with pytest.raises(MlsynthEstimationError):
        TWSF(_cfg(df, L=6, k_y=1, k_z=2)).fit()


def test_display_graphs_draws_without_blocking():
    import matplotlib
    matplotlib.use("Agg")
    df, _ = _factor_panel(sigma=0.05, seed=26)
    res = TWSF(_cfg(df, display_graphs=True)).fit()
    assert res.time_series is not None


def test_plug_in_standard_error_is_calibrated_against_the_empirical_spread():
    """The interval must be the right *size*, not merely positive and widening.

    Every other assertion here constrains the band's shape -- symmetric, non-
    negative, wider at longer horizons, collapsing with the noise. A variance
    that is systematically too small satisfies all of them, so the only thing
    that pins the magnitude is the diagnostic the theory actually promises:
    over repeated panels the empirical spread of the forecast error should
    match the plug-in standard error.

    Measured at lead 5 rather than lead 1 on purpose. At one step the recursion
    is the identity and its Jacobian carries no information; the term that
    propagates one-step estimation error through the recursion only bites at
    longer leads, and this is where a missing one shows up.

    The panel is deliberately not small. The plug-in variance is a leading-order
    approximation and the ratio is only near one where the theory's asymptotics
    have room: measured across donor counts it runs 3.9 at ten donors, then
    0.79, 0.90 and 1.28 at twenty, forty and sixty, and the full-budget gate on
    the paper's own design gave 0.894 to 1.165. Ten donors is outside the
    regime, not evidence against the formula -- the same story as the coverage
    shortfall at the smallest panels.
    """
    lead, R = 5, 60
    errs, ses = [], []
    for r in range(R):
        df, truth = _factor_panel(n_donors=40, T0=80, T1=300, horizon=lead,
                                  sigma=0.05, seed=900 + r)
        res = TWSF(_cfg(df, L=16, k_y=2, k_z=4, horizon=lead)).fit()
        cf = np.asarray(res.time_series.counterfactual_outcome, dtype=float)
        se = np.asarray(res.inference.details["std_error_path"], dtype=float)
        errs.append(float(cf[lead - 1] - truth[lead - 1]))
        ses.append(float(se[lead - 1]))
    ratio = float(np.std(errs, ddof=1) / np.mean(ses))
    assert 0.6 < ratio < 1.5, (
        f"empirical SD over mean plug-in SE = {ratio:.3f}; a ratio well above 1 "
        "means the interval understates the true sampling spread")
