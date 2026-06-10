"""Exhaustive tests for the Panel Data Approach (PDA) estimator.

Covers every line of the estimator and its ``pda_helpers`` package:

* ``estimators/pda.py``                         -- PDA orchestration class
* ``pda_helpers/structures.py``                 -- PDAInputs / PDAMethodFit / PDAResults
* ``pda_helpers/setup.py``                      -- DataFrame -> NumPy boundary
* ``pda_helpers/inference.py``                  -- shared HAC machinery
* ``pda_helpers/orchestration.py``              -- resolve_methods / run_pda
* ``pda_helpers/results_assembly.py``           -- assemble_pda_results
* ``pda_helpers/plotter.py``                    -- plot_pda
* ``pda_helpers/{l2,lasso,fs}/{estimation,inference}.py`` -- the three variants

Smoke tests, every utility, and deliberately pathological / impossible inputs
that should raise.
"""

from __future__ import annotations

from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")  # headless; plotter must never pop a window

import numpy as np
import pandas as pd
import pytest

from mlsynth.estimators.pda import PDA
from mlsynth.config_models import PDAConfig
from mlsynth.exceptions import MlsynthDataError, MlsynthEstimationError

from mlsynth.utils.fast_scm_helpers.structure import IndexSet
from mlsynth.utils.pda_helpers import (
    FS, L2, LASSO,
    PDAInputs, PDAMethodFit, PDAResults,
    derive_treatment, prepare_pda_inputs,
    hac_lrv, newey_west_lag, normal_test,
    resolve_methods, run_pda, assemble_pda_results, plot_pda,
)
from mlsynth.utils.pda_helpers.orchestration import _weights_dict
from mlsynth.utils.pda_helpers.l2 import cross_validate_tau, fit_l2, l2_relax, l2_ate_inference
from mlsynth.utils.pda_helpers.l2 import estimation as l2_estimation
from mlsynth.utils.pda_helpers.lasso import fit_lasso, lasso_ate_inference
from mlsynth.utils.pda_helpers.fs import forward_select, fs_ate_inference
from mlsynth.utils.pda_helpers.fs.estimation import _ols_sigma2


# ======================================================================
# Data helpers
# ======================================================================
def make_panel(
    n_units: int = 8,
    n_periods: int = 40,
    treatment_start: int = 30,
    effect: float = 5.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Long-format panel; unit 1 is treated from ``treatment_start`` on.

    The treated unit's untreated path is a noisy linear combination of the
    donors, so every variant can recover a sensible counterfactual.
    """
    rng = np.random.default_rng(seed)
    periods = np.arange(1, n_periods + 1)
    n_donors = n_units - 1
    donor_paths = {}
    for j in range(n_donors):
        base = rng.normal(20 + 4 * j, 3)
        trend = np.linspace(0, 6 + j, n_periods)
        donor_paths[j] = base + trend + rng.normal(0, 0.6, n_periods)

    weights = rng.uniform(0.2, 1.0, n_donors)
    weights /= weights.sum()
    treated_untreated = sum(weights[j] * donor_paths[j] for j in range(n_donors))
    treated_untreated = treated_untreated + rng.normal(0, 0.3, n_periods)

    rows = []
    for t_idx, t in enumerate(periods):
        treated_val = treated_untreated[t_idx]
        if t >= treatment_start:
            treated_val += effect
        rows.append({"ID": 1, "Period": int(t), "Value": float(treated_val),
                     "IsTreated": int(t >= treatment_start)})
        for j in range(n_donors):
            rows.append({"ID": j + 2, "Period": int(t),
                         "Value": float(donor_paths[j][t_idx]), "IsTreated": 0})
    return pd.DataFrame(rows)


@pytest.fixture
def panel() -> pd.DataFrame:
    return make_panel()


def base_config(df: pd.DataFrame, **overrides) -> Dict[str, Any]:
    cfg = {
        "df": df, "treat": "IsTreated", "time": "Period",
        "outcome": "Value", "unitid": "ID", "display_graphs": False,
    }
    cfg.update(overrides)
    return cfg


# ======================================================================
# PDA estimator: construction + smoke
# ======================================================================
@pytest.mark.parametrize("method", ["fs", "LASSO", "l2"])
def test_fit_smoke_each_method(panel, method):
    res = PDA(PDAConfig(**base_config(panel, method=method))).fit()
    assert isinstance(res, PDAResults)
    assert len(res.fits) == 1
    fit = next(iter(res.fits.values()))
    assert fit.counterfactual.shape == (panel["Period"].nunique(),)
    assert np.isfinite(res.att)
    # treatment effect of +5 should be roughly recovered
    assert res.att == pytest.approx(5.0, abs=2.0)


def test_init_accepts_dict():
    """PDA(config_dict) should build a PDAConfig internally."""
    df = make_panel()
    est = PDA(base_config(df, method="fs"))
    assert isinstance(est.config, PDAConfig)
    assert est.outcome == "Value" and est.treat == "IsTreated"


def test_init_accepts_config_object(panel):
    cfg = PDAConfig(**base_config(panel))
    est = PDA(cfg)
    assert est.config is cfg


def test_fit_multiple_methods_at_once(panel):
    res = PDA(PDAConfig(**base_config(panel, methods=["fs", "l2", "LASSO"]))).fit()
    assert set(res.fits) == {FS, L2, LASSO}
    assert res.selected_variant == FS  # first requested
    att_map = res.att_by_method()
    se_map = res.se_by_method()
    assert set(att_map) == {FS, L2, LASSO}
    assert set(se_map) == {FS, L2, LASSO}
    assert all(np.isfinite(v) for v in att_map.values())


def test_fit_l2_with_explicit_tau(panel):
    res = PDA(PDAConfig(**base_config(panel, method="l2", tau=0.5))).fit()
    assert res.fits[L2].metadata["tau"] == 0.5


def test_fit_with_display_graphs_saves(panel, tmp_path):
    out = tmp_path / "pda.png"
    PDA(PDAConfig(**base_config(panel, method="fs", display_graphs=True,
                                save=str(out)))).fit()
    assert out.exists()


def test_fit_with_display_graphs_show(panel):
    """Display_graphs=True with save=False exercises the plt.show() branch."""
    PDA(PDAConfig(**base_config(panel, method="fs", display_graphs=True,
                                save=False))).fit()


# ======================================================================
# setup.py : derive_treatment / prepare_pda_inputs
# ======================================================================
def test_derive_treatment_normal(panel):
    unit, t0 = derive_treatment(panel, "ID", "Period", "IsTreated")
    assert unit == 1
    assert t0 == 30


def test_derive_treatment_no_treated_rows():
    df = make_panel()
    df["IsTreated"] = 0
    with pytest.raises(MlsynthDataError, match="No treated rows"):
        derive_treatment(df, "ID", "Period", "IsTreated")


def test_derive_treatment_multiple_treated_units(panel):
    panel.loc[(panel["ID"] == 2) & (panel["Period"] >= 30), "IsTreated"] = 1
    with pytest.raises(MlsynthDataError, match="exactly one treated unit"):
        derive_treatment(panel, "ID", "Period", "IsTreated")


def test_prepare_inputs_shapes(panel):
    inp = prepare_pda_inputs(panel, unitid="ID", time="Period",
                             outcome="Value", treat="IsTreated")
    assert isinstance(inp, PDAInputs)
    assert inp.T == 40 and inp.T0 == 29 and inp.T2 == 11
    assert inp.n_donors == 7
    assert inp.X.shape == (40, 7)
    assert inp.treated_label == 1
    assert inp.metadata["intervention_time"] == 30


def test_prepare_inputs_incomplete_panel_raises():
    df = make_panel()
    df = df.drop(df.index[5])  # puncture the panel -> NaN after pivot
    with pytest.raises(MlsynthDataError, match="complete outcome panel"):
        prepare_pda_inputs(df, unitid="ID", time="Period",
                           outcome="Value", treat="IsTreated")


def test_prepare_inputs_too_few_pre_periods_raises():
    # treatment at period 2 -> only one pre-period
    df = make_panel(n_periods=10, treatment_start=2)
    with pytest.raises(MlsynthDataError, match="at least two pre-treatment"):
        prepare_pda_inputs(df, unitid="ID", time="Period",
                           outcome="Value", treat="IsTreated")


# ======================================================================
# inference.py : newey_west_lag / hac_lrv / normal_test
# ======================================================================
def test_newey_west_lag_values():
    assert newey_west_lag(100) == 4
    assert newey_west_lag(1) == 0
    assert newey_west_lag(0) == 0
    assert newey_west_lag(500) > newey_west_lag(50)


def test_fspda_lrvar_lag_rule_and_cap():
    from mlsynth.utils.pda_helpers.inference import fspda_lrvar_lag
    # default rule: floor(T2 ** 1/4)
    assert fspda_lrvar_lag(36) == 2          # floor(36**.25)=2
    assert fspda_lrvar_lag(100) == 3         # floor(100**.25)=3
    # explicit lag within the floor(sqrt(T2)) cap is accepted
    assert fspda_lrvar_lag(100, 5) == 5
    # over the cap or negative -> error
    with pytest.raises(ValueError):
        fspda_lrvar_lag(36, 99)
    with pytest.raises(ValueError):
        fspda_lrvar_lag(36, -1)


def test_prewhitened_nw_lrvar_handles_negative_autocorr():
    from mlsynth.utils.pda_helpers.inference import lrvar_prewhite_nw, hac_lrv
    # Strongly mean-reverting (lag-1 autocorr ~ -1) series: prewhitening should
    # shrink the long-run variance of the mean well below the iid 1/n level.
    z = np.tile([1.0, -1.0], 30)
    v_pw = lrvar_prewhite_nw(z)
    assert v_pw >= 0 and np.isfinite(v_pw)
    assert v_pw < np.var(z) / z.size          # below the naive var(mean)
    # short series falls back gracefully
    assert np.isfinite(lrvar_prewhite_nw(np.array([1.0, 2.0, 3.0])))


def test_fs_lrvar_lag_override_changes_se(rng_panel_factory=None):
    # Default fs (prewhitened NW) vs the fixed-lag Bartlett override should give
    # different standard errors on a serially-dependent effect series.
    from mlsynth.utils.pda_helpers.fs.inference import fs_ate_inference
    rng = np.random.default_rng(0)
    T0, T2 = 30, 24
    y = np.r_[rng.normal(size=T0), 2.0 + np.tile([1.0, -1.0], T2 // 2)]
    cf = np.r_[y[:T0], np.zeros(T2)]
    _, se_default, _, _ = fs_ate_inference(y, cf, T0)
    _, se_bartlett, _, _ = fs_ate_inference(y, cf, T0, lrvar_lag=2)
    assert se_default > 0 and se_bartlett > 0
    assert not np.isclose(se_default, se_bartlett)


def test_hac_lrv_empty_is_nan():
    assert np.isnan(hac_lrv(np.array([])))


def test_hac_lrv_iid_positive():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 1, 200)
    lrv = hac_lrv(z)
    assert lrv > 0 and np.isfinite(lrv)


def test_hac_lrv_explicit_lag_bartlett():
    z = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    # bartlett with L=1: w = 1 - 1/2 = 0.5 -> gamma0 + 2*0.5*gamma1 = 1 + (-1) = 0
    lrv_bart = hac_lrv(z, lag=1, kernel="bartlett")
    assert lrv_bart == pytest.approx(0.0, abs=1e-9)


def test_hac_lrv_negative_is_clamped_to_zero():
    """Strong negative autocorrelation + uniform kernel -> raw LRV<0 -> clamp 0."""
    z = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    lrv = hac_lrv(z, lag=1, kernel="uniform")
    assert lrv == 0.0


def test_normal_test_finite():
    p, (lo, hi) = normal_test(2.0, 1.0, alpha=0.05)
    assert 0.0 <= p <= 1.0
    assert lo < 2.0 < hi
    assert hi - lo == pytest.approx(2 * 1.959963984540054, rel=1e-6)


def test_normal_test_zero_se_is_nan():
    p, (lo, hi) = normal_test(1.0, 0.0)
    assert np.isnan(p) and np.isnan(lo) and np.isnan(hi)


def test_normal_test_nonfinite_se_is_nan():
    p, ci = normal_test(1.0, np.inf)
    assert np.isnan(p) and all(np.isnan(c) for c in ci)


# ======================================================================
# orchestration.py : resolve_methods / _weights_dict / run_pda
# ======================================================================
def test_resolve_methods_single_and_normalization():
    assert resolve_methods("fs", None) == [FS]
    assert resolve_methods("LASSO", None) == [LASSO]
    assert resolve_methods("l2", None) == [L2]


def test_resolve_methods_list_wins():
    assert resolve_methods("fs", ["l2", "LASSO"]) == [L2, LASSO]


def test_resolve_methods_unknown_passthrough():
    # unknown keys are passed through verbatim (run_pda is what rejects them)
    assert resolve_methods("bogus", None) == ["bogus"]


def test_weights_dict_thresholds_small_coeffs():
    labels = np.array(["a", "b", "c"])
    beta = np.array([0.5, 1e-12, -0.3])
    wd = _weights_dict(beta, labels)
    assert set(wd) == {"a", "c"}      # near-zero 'b' dropped
    assert wd["a"] == 0.5 and wd["c"] == -0.3


def test_run_pda_unknown_method_raises(panel):
    inp = prepare_pda_inputs(panel, unitid="ID", time="Period",
                             outcome="Value", treat="IsTreated")
    with pytest.raises(ValueError, match="Unknown PDA method"):
        run_pda(inp, ["not_a_method"], tau=None, alpha=0.05)


def test_run_pda_all_three_populate_fits(panel):
    inp = prepare_pda_inputs(panel, unitid="ID", time="Period",
                             outcome="Value", treat="IsTreated")
    fits = run_pda(inp, [L2, LASSO, FS], tau=None, alpha=0.05)
    assert set(fits) == {L2, LASSO, FS}
    for f in fits.values():
        assert isinstance(f, PDAMethodFit)
        assert f.counterfactual.shape == (inp.T,)
        assert f.gap.shape == (inp.T,)
        assert np.isfinite(f.att)
    # lasso/fs expose selected_donors, l2 does not
    assert fits[L2].selected_donors is None
    assert fits[LASSO].selected_donors is not None
    assert fits[FS].selected_donors is not None


# ======================================================================
# results_assembly.py : assemble_pda_results
# ======================================================================
def _toy_inputs() -> PDAInputs:
    return PDAInputs(
        unit_index=IndexSet.from_labels(["d1", "d2"]),
        time_index=IndexSet.from_labels([1, 2, 3, 4]),
        y=np.array([1.0, 2.0, 3.0, 4.0]),
        X=np.array([[1.0, 0.0], [2.0, 1.0], [3.0, 2.0], [4.0, 3.0]]),
        T0=2, treated_label="t",
    )


def _toy_fit(name: str, att: float = 1.0) -> PDAMethodFit:
    return PDAMethodFit(
        name=name, beta=np.array([0.5, 0.5]), intercept=0.0,
        counterfactual=np.array([1.0, 2.0, 2.5, 3.5]),
        gap=np.array([0.0, 0.0, 0.5, 0.5]), att=att, att_se=0.1,
        ci=(0.8, 1.2), p_value=0.01, donor_weights={"d1": 0.5, "d2": 0.5},
    )


def test_assemble_keeps_valid_selected_variant():
    res = assemble_pda_results(_toy_inputs(), {FS: _toy_fit(FS)}, selected_variant=FS)
    assert res.selected_variant == FS


def test_assemble_falls_back_when_selected_missing():
    res = assemble_pda_results(_toy_inputs(), {L2: _toy_fit(L2)}, selected_variant="zzz")
    assert res.selected_variant == L2   # fell back to the only present fit


def test_assemble_empty_fits_keeps_label():
    res = assemble_pda_results(_toy_inputs(), {}, selected_variant=FS)
    assert res.fits == {} and res.selected_variant == FS


# ======================================================================
# structures.py : PDAResults aliases + property fallbacks
# ======================================================================
def test_pdaresults_aliases_forward_to_primary():
    fit = _toy_fit(FS, att=2.5)
    res = PDAResults(inputs=_toy_inputs(), fits={FS: fit}, selected_variant=FS)
    assert res.att == 2.5
    assert res.att_se == fit.att_se
    np.testing.assert_array_equal(res.counterfactual, fit.counterfactual)
    np.testing.assert_array_equal(res.gap, fit.gap)
    assert res.donor_weights == fit.donor_weights


def test_pdaresults_primary_fallback_on_bad_variant():
    fit = _toy_fit(L2, att=9.0)
    res = PDAResults(inputs=_toy_inputs(), fits={L2: fit}, selected_variant="missing")
    # _primary cannot find 'missing' -> falls back to first fit
    assert res.att == 9.0


def test_pdainputs_properties():
    inp = _toy_inputs()
    assert inp.T == 4 and inp.T0 == 2 and inp.T2 == 2 and inp.n_donors == 2
    np.testing.assert_array_equal(inp.donor_labels, np.array(["d1", "d2"]))


# ======================================================================
# l2 estimation + inference
# ======================================================================
def test_l2_relax_returns_coeffs_and_intercept():
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, (30, 4))
    y = X @ np.array([1.0, 0.5, -0.3, 0.2]) + rng.normal(0, 0.1, 30)
    beta, intercept = l2_relax(y, X, tau=0.1)
    assert beta.shape == (4,) and np.isfinite(intercept)


def test_l2_relax_all_solvers_fail_raises(monkeypatch):
    """If every solver leaves beta.value None, raise MlsynthEstimationError."""
    def boom(self, *a, **k):
        raise RuntimeError("solver down")
    monkeypatch.setattr(l2_estimation.cp.Problem, "solve", boom)
    with pytest.raises(MlsynthEstimationError, match="all solvers diverged"):
        l2_relax(np.arange(10.0), np.random.default_rng(0).normal(0, 1, (10, 2)), tau=0.1)


def test_l2_relax_solver_returns_none_value_raises(monkeypatch):
    """Solver runs without error but leaves beta.value None -> try next, then raise."""
    monkeypatch.setattr(l2_estimation.cp.Problem, "solve", lambda self, **k: None)
    with pytest.raises(MlsynthEstimationError, match="all solvers diverged"):
        l2_relax(np.arange(10.0), np.random.default_rng(0).normal(0, 1, (10, 2)), tau=0.1)


def test_cross_validate_tau_returns_float():
    rng = np.random.default_rng(3)
    X = rng.normal(0, 1, (40, 3))
    y = X @ np.array([1.0, -0.5, 0.2]) + rng.normal(0, 0.2, 40)
    tau = cross_validate_tau(y, X)
    assert isinstance(tau, float) and tau >= 0.0


def test_cross_validate_tau_handles_solver_failures(monkeypatch):
    """If l2_relax always fails, val_mse -> inf for every grid point (still returns)."""
    def always_fail(*a, **k):
        raise MlsynthEstimationError("nope")
    monkeypatch.setattr(l2_estimation, "l2_relax", always_fail)
    rng = np.random.default_rng(4)
    tau = cross_validate_tau(rng.normal(0, 1, 30), rng.normal(0, 1, (30, 2)))
    assert isinstance(tau, float)


def test_fit_l2_autotune_vs_fixed_tau():
    rng = np.random.default_rng(5)
    X = rng.normal(0, 1, (40, 3))
    y = X @ np.array([1.0, 0.0, 0.5]) + rng.normal(0, 0.2, 40)
    y = np.concatenate([y, y[:10] + 3.0])           # 10 "post" periods w/ effect
    X = np.vstack([X, rng.normal(0, 1, (10, 3))])
    b1, a1, cf1, tau1 = fit_l2(y, X, T0=40, tau=None)   # auto-tuned
    b2, a2, cf2, tau2 = fit_l2(y, X, T0=40, tau=0.3)    # fixed
    assert tau2 == 0.3 and tau1 >= 0.0
    assert cf1.shape == (50,) and cf2.shape == (50,)


def test_l2_standardize_default_and_effect():
    # Standardisation is the default and changes the fit on scale-heterogeneous
    # controls; the coefficients map back to the original scale either way.
    from mlsynth.config_models import PDAConfig
    rng = np.random.default_rng(0)
    scales = np.array([1.0, 10.0, 100.0, 0.1, 5.0])
    X = rng.normal(0, 1, (40, 5)) * scales
    y = X @ np.array([0.5, 0.0, 0.0, 1.0, 0.0]) + rng.normal(0, 0.5, 40)
    yf = np.r_[y, y[:8] + 2.0]; Xf = np.vstack([X, rng.normal(0, 1, (8, 5)) * scales])
    b_std, a_std, _, _ = fit_l2(yf, Xf, T0=40, tau=0.05, standardize=True)
    b_raw, a_raw, _, _ = fit_l2(yf, Xf, T0=40, tau=0.05, standardize=False)
    assert not np.allclose(b_std, b_raw)
    from mlsynth.config_models import PDAConfig as _Cfg
    assert _Cfg.model_fields["l2_standardize"].default is True


def test_cross_validate_tau_grid_is_log_spaced_small_tau():
    # The auto grid must reach down to ~1e-4 * max|eta| so a tiny optimal tau is
    # representable (the China-PPI regime); a fixed tau_grid is honoured.
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (60, 4)); y = X @ np.array([1.0, 0, 0, 0]) + rng.normal(0, 0.3, 60)
    tau = cross_validate_tau(y, X, tau_grid=np.array([1e-4, 1e-3, 1e-2]))
    assert tau in (1e-4, 1e-3, 1e-2)


def test_l2_ate_inference_outputs():
    rng = np.random.default_rng(20)
    T0, T2 = 20, 10
    gap = np.concatenate([rng.normal(0, 0.5, T0),       # nonzero pre residuals
                          2.0 + rng.normal(0, 0.5, T2)])  # +2 effect, with variation
    y = np.linspace(0, 10, T0 + T2)
    cf = y - gap
    att, se, ci, p = l2_ate_inference(y, cf, T0=T0, alpha=0.05)
    assert att == pytest.approx(float(gap[T0:].mean()), abs=1e-9)
    assert se > 0.0 and ci[0] < ci[1]                    # both LRV terms contribute
    assert 0.0 <= p <= 1.0


# ======================================================================
# lasso estimation + inference
# ======================================================================
def test_fit_lasso_basic():
    rng = np.random.default_rng(6)
    X = rng.normal(0, 1, (40, 5))
    y = X @ np.array([2.0, 0.0, 0.0, -1.0, 0.0]) + rng.normal(0, 0.1, 40)
    beta, intercept, cf, support = fit_lasso(y, X, T0=40)
    assert beta.shape == (5,) and support.dtype == bool
    assert cf.shape == (40,)


def test_fit_lasso_small_pre_period_cv_floor():
    """T0=3 -> n_splits clamps to >=2; should still fit without error."""
    rng = np.random.default_rng(7)
    X = rng.normal(0, 1, (6, 2))
    y = X @ np.array([1.0, -1.0]) + rng.normal(0, 0.1, 6)
    beta, intercept, cf, support = fit_lasso(y, X, T0=3)
    assert beta.shape == (2,) and cf.shape == (6,)


def test_lasso_inference_with_active_support():
    rng = np.random.default_rng(8)
    X = rng.normal(0, 1, (40, 3))
    y = np.linspace(0, 5, 40)
    cf = y.copy(); cf[30:] -= 1.5
    support = np.array([True, False, True])           # active, T0 > |S|+1
    att, se, ci, p = lasso_ate_inference(y, X, cf, support, T0=30, alpha=0.05)
    assert att == pytest.approx(1.5, abs=1e-9)
    assert se > 0.0                                    # first-stage term contributes


def test_lasso_inference_empty_support():
    y = np.linspace(0, 5, 30)
    cf = y.copy(); cf[20:] -= 1.0
    X = np.random.default_rng(9).normal(0, 1, (30, 4))
    support = np.zeros(4, dtype=bool)                  # no donors selected
    att, se, ci, p = lasso_ate_inference(y, X, cf, support, T0=20)
    assert att == pytest.approx(1.0, abs=1e-9)
    assert se >= 0.0


def test_lasso_inference_support_too_large_skips_firststage():
    """|S|+1 >= T0 -> first-stage variance skipped (only post variance)."""
    y = np.linspace(0, 5, 10)
    cf = y.copy(); cf[6:] -= 1.0
    X = np.random.default_rng(10).normal(0, 1, (10, 8))
    support = np.ones(8, dtype=bool)                   # |S|=8, T0=6 -> 6 <= 9
    att, se, ci, p = lasso_ate_inference(y, X, cf, support, T0=6)
    assert np.isfinite(att) and se >= 0.0


# ======================================================================
# fs estimation + inference
# ======================================================================
def test_ols_sigma2_zero_for_perfect_fit():
    Z = np.column_stack([np.ones(5), np.arange(5.0)])
    y = 3.0 + 2.0 * np.arange(5.0)
    assert _ols_sigma2(y, Z) == pytest.approx(0.0, abs=1e-18)


def test_forward_select_picks_informative_donors():
    rng = np.random.default_rng(11)
    X = rng.normal(0, 1, (40, 6))
    y = 2.0 * X[:, 0] - 1.5 * X[:, 3] + rng.normal(0, 0.1, 40)
    sel, beta, intercept, cf = forward_select(y, X, T0=40)
    assert len(sel) >= 1
    assert 0 in sel and 3 in sel                       # the two real donors
    assert beta.shape == (6,) and cf.shape == (40,)
    # zeros off-support
    off = [j for j in range(6) if j not in sel]
    assert np.allclose(beta[off], 0.0)


def test_forward_select_exhausts_all_donors():
    """N < T0 with every donor informative -> selection empties `remaining`."""
    rng = np.random.default_rng(12)
    X = rng.normal(0, 1, (30, 2))
    y = X[:, 0] + X[:, 1] + rng.normal(0, 0.01, 30)    # both strongly predictive
    sel, beta, intercept, cf = forward_select(y, X, T0=30)
    assert set(sel) == {0, 1}                           # both selected, remaining drained


def test_forward_select_degenerate_no_selection(monkeypatch):
    """If no donor ever lowers the IC, fall back appropriately per intercept mode."""
    import mlsynth.utils.pda_helpers.fs.estimation as fs_est
    monkeypatch.setattr(fs_est, "_ols_sigma2", lambda y, Z: 1e9)  # never improves
    rng = np.random.default_rng(13)
    X = rng.normal(0, 1, (20, 4))
    y = rng.normal(5.0, 1.0, 20)

    # default (no intercept): degenerate -> zero counterfactual
    sel, beta, intercept, cf = forward_select(y, X, T0=20)
    assert sel == [] and np.allclose(beta, 0.0)
    assert np.allclose(cf, 0.0)

    # intercept mode: degenerate -> flat counterfactual at the pre-period mean
    sel2, beta2, icpt2, cf2 = forward_select(y, X, T0=20, intercept=True)
    assert sel2 == [] and np.allclose(beta2, 0.0)
    assert np.allclose(cf2, np.mean(y[:20]))


def test_forward_select_runs_full_t0_iterations(monkeypatch):
    """Every step keeps improving the IC -> the range(T0) loop cap is the exit."""
    import mlsynth.utils.pda_helpers.fs.estimation as fs_est
    # sigma^2 strictly shrinks as more columns are added, so IC always improves
    monkeypatch.setattr(fs_est, "_ols_sigma2", lambda y, Z: 1.0 / Z.shape[1])
    rng = np.random.default_rng(14)
    X = rng.normal(0, 1, (4, 6))                       # N=6 > T0=4
    y = rng.normal(0, 1, 4)
    sel, beta, intercept, cf = forward_select(y, X, T0=4, intercept=True)
    assert len(sel) == 4                                # selected once per iteration
    assert cf.shape == (4,)


def test_fs_ate_inference_outputs():
    rng = np.random.default_rng(21)
    T0, T2 = 24, 8
    gap = np.concatenate([np.zeros(T0), 3.0 + rng.normal(0, 0.4, T2)])
    y = np.linspace(0, 8, T0 + T2)
    cf = y - gap
    att, se, ci, p = fs_ate_inference(y, cf, T0=T0, alpha=0.05)
    assert att == pytest.approx(float(gap[T0:].mean()), abs=1e-9)
    assert se > 0.0 and ci[0] < ci[1] and 0.0 <= p <= 1.0


# ======================================================================
# plotter.py
# ======================================================================
def _toy_results_for_plot(t0=2, n_t=4):
    inp = PDAInputs(
        unit_index=IndexSet.from_labels(["d1", "d2"]),
        time_index=IndexSet.from_labels(list(range(1, n_t + 1))),
        y=np.linspace(0, 1, n_t),
        X=np.zeros((n_t, 2)), T0=t0, treated_label="t",
    )
    fit = PDAMethodFit(
        name=FS, beta=np.zeros(2), intercept=0.0,
        counterfactual=np.linspace(0, 1, n_t) + 0.1,
        gap=np.full(n_t, -0.1), att=-0.1, att_se=0.05,
        ci=(-0.2, 0.0), p_value=0.5, donor_weights={},
    )
    return PDAResults(inputs=inp, fits={FS: fit}, selected_variant=FS)


def test_plot_pda_save_to_path(tmp_path):
    out = tmp_path / "fig.png"
    plot_pda(_toy_results_for_plot(), outcome="Value", time="Period", save=str(out))
    assert out.exists()


def test_plot_pda_save_true_default_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    plot_pda(_toy_results_for_plot(), outcome="Value", time="Period", save=True)
    assert (tmp_path / "pda_estimates.png").exists()


def test_plot_pda_show_branch_with_color_list():
    # save=False -> plt.show() branch; list of colors exercises the cycling path
    plot_pda(_toy_results_for_plot(), outcome="Value", time="Period",
             counterfactual_color=["red", "blue"], save=False)


def test_plot_pda_no_post_period_skips_axvline():
    # T0 == len(years) -> the axvline branch is skipped
    plot_pda(_toy_results_for_plot(t0=4, n_t=4), outcome="Value", time="Period",
             save=False)


def test_plot_pda_failure_warns():
    """A broken color spec makes the body raise; it must be swallowed as a warning."""
    with pytest.warns(UserWarning, match="PDA plotting failed"):
        plot_pda(_toy_results_for_plot(), outcome="Value", time="Period",
                 counterfactual_color=[], save=False)   # empty list -> modulo-by-zero
