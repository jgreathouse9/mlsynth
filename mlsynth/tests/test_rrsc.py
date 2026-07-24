"""Tests for the RRSC estimator (config, setup; pipeline/estimator/plotting
added alongside those phases). Value-for-value fidelity to the reference R on
the Beijing / Middle East panels is pinned in the golden benchmark.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth.config_models import RRSCConfig
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.rrsc_helpers.setup import prepare_rrsc_inputs


def make_panel(N=12, T=40, T0=30, r=2, interfered=(1, 2), effect=8.0,
               interference=5.0, seed=0):
    """Synthetic panel drawn from the method's untreated model ``Y_it(0) =
    lambda_i' f_t``: time-varying factors (one a smooth trend carrying the
    level, which survives the large-N mean-centering) with no separate
    idiosyncratic intercept, a direct effect on the treated unit (row 0), and
    interference planted on a few controls."""
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(T, r))
    F[:, 0] = np.cumsum(rng.normal(size=T)) / 3.0          # smooth trend factor
    L = rng.normal(size=(N, r))
    Y = (F @ L.T).T + 0.3 * rng.normal(size=(N, T))
    Y[0, T0:] += effect                                   # direct effect
    for j in interfered:
        Y[j, T0:] += interference                         # interference effects
    rows = [dict(u=f"unit{i}", t=t, y=Y[i, t], d=int(i == 0 and t >= T0))
            for i in range(N) for t in range(T)]
    return pd.DataFrame(rows), Y


@pytest.fixture(scope="module")
def panel_df():
    df, _ = make_panel()
    return df


def _cfg(df, **kw):
    return RRSCConfig(df=df, outcome="y", treat="d", unitid="u", time="t", **kw)


# ------------------------------------------------------------------ config
def test_config_defaults(panel_df):
    c = _cfg(panel_df)
    assert c.regime == "auto" and c.inference == "cbb"
    assert c.loss == "huber" and c.lts_alpha == 0.5 and c.update_dif is True


def test_config_rejects_nfactors_over_max(panel_df):
    with pytest.raises(MlsynthConfigError):
        _cfg(panel_df, n_factors=20, max_factors=10)


def test_config_forbids_extra_fields(panel_df):
    with pytest.raises(Exception):
        _cfg(panel_df, not_a_field=1)


@pytest.mark.parametrize("field,bad", [
    ("regime", "bogus"), ("inference", "bogus"), ("loss", "bogus"),
])
def test_config_rejects_bad_literals(panel_df, field, bad):
    with pytest.raises(Exception):
        _cfg(panel_df, **{field: bad})


# ------------------------------------------------------------------- setup
def test_setup_builds_full_panel(panel_df):
    inp = prepare_rrsc_inputs(panel_df, "y", "d", "u", "t", regime="auto")
    assert inp.Y.shape == (12, 40)
    assert inp.N == 12 and inp.T0 == 30 and inp.n_post == 10
    assert inp.treated_name == "unit0"
    assert inp.X.sum() == 10 and (inp.X[:30] == 0).all()
    # treated unit is row 0
    assert np.allclose(inp.Y[0], panel_df[panel_df.u == "unit0"].sort_values("t")["y"].to_numpy())


def test_setup_regime_auto():
    # T0 >= 5N -> fixed_n ; else large_n
    df_wide, _ = make_panel(N=5, T=40, T0=30)      # T0=30 >= 5*5=25 -> fixed_n
    assert prepare_rrsc_inputs(df_wide, "y", "d", "u", "t", "auto").regime == "fixed_n"
    df_tall, _ = make_panel(N=20, T=40, T0=30)     # T0=30 < 5*20=100 -> large_n
    assert prepare_rrsc_inputs(df_tall, "y", "d", "u", "t", "auto").regime == "large_n"


def test_setup_regime_explicit(panel_df):
    assert prepare_rrsc_inputs(panel_df, "y", "d", "u", "t", "fixed_n").regime == "fixed_n"
    assert prepare_rrsc_inputs(panel_df, "y", "d", "u", "t", "large_n").regime == "large_n"


def test_setup_rejects_bad_regime(panel_df):
    with pytest.raises(MlsynthConfigError):
        prepare_rrsc_inputs(panel_df, "y", "d", "u", "t", regime="bogus")


def test_setup_rejects_missing_data(panel_df):
    df = panel_df.copy()
    df.loc[df.index[5], "y"] = np.nan
    with pytest.raises(MlsynthDataError):
        prepare_rrsc_inputs(df, "y", "d", "u", "t")


def test_setup_rejects_too_few_controls():
    df, _ = make_panel(N=2, T=20, T0=15, interfered=())   # only 1 control
    with pytest.raises(MlsynthDataError):
        prepare_rrsc_inputs(df, "y", "d", "u", "t")


# --------------------------------------------------------------- estimator
import matplotlib                                            # noqa: E402
matplotlib.use("Agg")
from mlsynth import RRSC                                     # noqa: E402
from mlsynth.config_models import BaseEstimatorResults      # noqa: E402


@pytest.mark.parametrize("regime", ["large_n", "fixed_n"])
def test_recovers_direct_and_interference_effects(regime):
    df, _ = make_panel(N=12, T=40, T0=30, interfered=(1, 2),
                       effect=8.0, interference=5.0)
    r = RRSC(_cfg(df, regime=regime, n_factors=2, inference="none")).fit()
    assert r.regime == regime and r.n_factors == 2
    assert r.att == pytest.approx(8.0, abs=2.0)             # direct effect
    assert r.interference_effects["unit1"] == pytest.approx(5.0, abs=2.0)
    assert r.interference_effects["unit2"] == pytest.approx(5.0, abs=2.0)
    assert abs(r.interference_effects["unit5"]) < 2.0       # clean control ~ 0


def test_result_contract_surface():
    df, _ = make_panel()
    r = RRSC(_cfg(df, regime="large_n", n_factors=2, inference="none")).fit()
    assert isinstance(r, BaseEstimatorResults)              # EffectResult
    assert np.isfinite(r.att)
    assert r.counterfactual.shape == (40,)
    assert r.gap.shape == (40,)
    assert np.isfinite(r.pre_rmse)
    # interference map covers every unit; treated entry == direct effect (att)
    assert set(r.interference_effects) == {f"unit{i}" for i in range(12)}
    assert r.direct_effect == pytest.approx(r.att)
    assert r.counterfactual_panel.shape == (12, 40)


def test_weights_submodel_populated():
    """RRSC exposes an implied, non-unique donor-weights submodel (factor-model
    convention, as MC-NNM): weights over the control units only, with a note."""
    df, _ = make_panel(N=12, T=40, T0=30)
    r = RRSC(_cfg(df, regime="large_n", n_factors=2, inference="none")).fit()
    assert r.weights is not None
    dw = r.donor_weights
    assert isinstance(dw, dict)
    # keyed over the controls (unit1..unit11), never the treated unit
    assert set(dw) == {f"unit{i}" for i in range(1, 12)}
    assert "unit0" not in dw
    assert r.weights.summary_stats.get("weights_are") == "implied_non_unique"
    assert "note" in r.weights.summary_stats


def test_cbb_inference_flags_interfered_units():
    df, _ = make_panel(N=12, T=44, T0=32, interfered=(1, 2),
                       effect=10.0, interference=8.0)
    r = RRSC(_cfg(df, regime="large_n", n_factors=2, inference="cbb",
                  cbb_reps=120, random_state=0)).fit()
    assert r.att_ci is not None and np.isfinite(r.inference.p_value)
    sig = r.significant_units(0.10)
    assert "unit1" in sig and "unit2" in sig                # interfered flagged
    assert "unit5" not in sig                               # clean not flagged


def test_cbb_non_separated_sampling():
    df, _ = make_panel(N=10, T=36, T0=26, interfered=(1,))
    r = RRSC(_cfg(df, regime="large_n", n_factors=2, inference="cbb",
                  cbb_reps=60, cbb_separation=False, random_state=0)).fit()
    assert np.isfinite(r.inference.p_value)


def test_setup_rejects_too_few_pre_periods():
    # treated adopts at t=1 -> only 1 pre-period
    df, _ = make_panel(N=6, T=20, T0=1, interfered=())
    with pytest.raises(MlsynthDataError):
        prepare_rrsc_inputs(df, "y", "d", "u", "t")


def test_conformal_inference_runs():
    df, _ = make_panel()
    r = RRSC(_cfg(df, regime="large_n", n_factors=2, inference="conformal")).fit()
    assert all(0.0 <= p <= 1.0 for p in r.interference_pvalue.values())
    # conformal reports p-values but no CI
    assert r.interference_ci["unit0"] == (None, None)


def test_auto_regime_and_factor_selection():
    df, _ = make_panel(N=6, T=40, T0=32)                    # T0=32 >= 5*6=30 -> fixed_n
    r = RRSC(_cfg(df, regime="auto", n_factors=2, inference="none")).fit()
    assert r.regime == "fixed_n"


def test_communality_diagnostic_present():
    df, _ = make_panel()
    r = RRSC(_cfg(df, regime="large_n", n_factors=2, inference="none")).fit()
    assert set(r.communality) == set(r.interference_effects)
    assert all(0.0 <= c <= 1.0 for c in r.communality.values())


def test_plotting_smoke():
    df, _ = make_panel()
    r = RRSC(_cfg(df, regime="large_n", n_factors=2, inference="none",
                  display_graphs=False)).fit()
    from mlsynth.utils.rrsc_helpers.plotter import plot_rrsc
    fig = plot_rrsc(r, show=False)
    assert len(fig.axes) == 2


def test_estimator_rejects_bad_config():
    df, _ = make_panel()
    with pytest.raises(MlsynthConfigError):
        RRSC(dict(df=df, outcome="y", treat="d", unitid="u", time="t",
                  n_factors=99, max_factors=5))


def test_estimator_wraps_pydantic_validation_error():
    df, _ = make_panel()
    with pytest.raises(MlsynthConfigError):        # bad type -> ValidationError -> translated
        RRSC(dict(df=df, outcome="y", treat="d", unitid="u", time="t", alpha="oops"))


def test_pipeline_rejects_r_ge_N():
    from mlsynth.exceptions import MlsynthEstimationError
    df, _ = make_panel(N=6, T=30, T0=22)
    with pytest.raises(MlsynthEstimationError):     # r >= N
        RRSC(_cfg(df, regime="large_n", n_factors=6, max_factors=8,
                  inference="none")).fit()


def test_auto_factor_selection_runs():
    df, _ = make_panel(N=25, T=45, T0=35, interfered=(1, 2))
    r = RRSC(_cfg(df, regime="large_n", inference="none")).fit()  # n_factors=None -> Bai-Ng
    assert r.n_factors >= 1 and np.isfinite(r.att)


def test_inputs_expose_donor_names():
    df, _ = make_panel()
    inp = prepare_rrsc_inputs(df, "y", "d", "u", "t")
    assert set(inp.donor_names) == {f"unit{i}" for i in range(1, 12)}
    assert "unit0" not in set(inp.donor_names)         # treated excluded


def test_setup_rejects_multiple_cohorts():
    # two treated units adopting at different times -> staggered cohorts
    df, _ = make_panel(N=8, T=30, T0=20, interfered=())
    extra = df[df.u == "unit1"].copy()
    df.loc[df.u == "unit1", "d"] = (df.loc[df.u == "unit1", "t"] >= 24).astype(int)
    with pytest.raises(MlsynthDataError):
        prepare_rrsc_inputs(df, "y", "d", "u", "t")


def test_display_graphs_path(monkeypatch):
    df, _ = make_panel()
    r = RRSC(_cfg(df, regime="large_n", n_factors=2, inference="none",
                  display_graphs=True)).fit()
    assert np.isfinite(r.att)
