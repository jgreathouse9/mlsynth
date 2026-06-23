"""Line/branch-coverage tests for SPILLSYNTH.

Drives the remaining uncovered paths in ``mlsynth.estimators.spillsynth``
and ``mlsynth.utils.spillsynth_helpers`` -- the shared plotter (every
layout branch), the Grossi covariate-bootstrap inference branches, the
ISCM no-clean-controls path, the structures aliases / ``_active`` error
branches, and assorted setup/scm_core validation branches.

Plotting tests use the non-interactive Agg backend and close every
figure they create.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from mlsynth import SPILLSYNTH  # noqa: E402
from mlsynth.utils.spillsynth_helpers import (  # noqa: E402
    CDFit,
    GrossiFit,
    ISCMFit,
    SpillSynthResults,
    prepare_spillsynth_inputs,
    run_cd,
    run_grossi,
    run_iscm,
)
from mlsynth.utils.spillsynth_helpers import build_A_distance_decay  # noqa: E402
from mlsynth.utils.spillsynth_helpers.cd.estimation import vanilla_scm_path  # noqa: E402
from mlsynth.utils.spillsynth_helpers.cd.scm_core import (  # noqa: E402
    fit_demeaned_sc,
    fit_leave_one_out_sc,
)
from mlsynth.utils.spillsynth_helpers import p_test  # noqa: E402
from mlsynth.utils.spillsynth_helpers.cd.inference import (  # noqa: E402
    G_matrix,
    compute_pre_residuals,
)
from mlsynth.exceptions import MlsynthDataError  # noqa: E402
from mlsynth.utils.spillsynth_helpers.plotter import plot_spillsynth  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic-panel builders
# --------------------------------------------------------------------------- #
def _panel(*, N=8, T=40, T0=30, treatment=-3.0, spillover=1.5,
           spillover_idx=1, seed=0):
    rng = np.random.default_rng(seed)
    loadings = rng.uniform(0.5, 1.5, size=N)
    f = np.cumsum(rng.standard_normal(T)) * 0.4 + 0.05 * np.arange(T)
    intercept = rng.uniform(-1, 1, size=N)
    Y = intercept[:, None] + np.outer(loadings, f) + 0.1 * rng.standard_normal((N, T))
    Y[0, T0:] += treatment
    if spillover_idx is not None:
        Y[spillover_idx, T0:] += spillover
    D = np.zeros((N, T))
    D[0, T0:] = 1
    rows = [
        {"unit": f"u{i}", "year": t, "y": float(Y[i, t]),
         "x1": float(Y[i, t] * 0.3 + rng.normal()), "treat": int(D[i, t])}
        for i in range(N) for t in range(T)
    ]
    return pd.DataFrame(rows)


def _multi_treated_panel(*, N=8, T=36, T0=28, seed=3):
    rng = np.random.default_rng(seed)
    loadings = rng.uniform(0.5, 1.5, size=N)
    f = np.cumsum(rng.standard_normal(T)) * 0.4 + 0.05 * np.arange(T)
    intercept = rng.uniform(-1, 1, size=N)
    Y = intercept[:, None] + np.outer(loadings, f) + 0.1 * rng.standard_normal((N, T))
    Y[0, T0:] += -3.0
    Y[1, T0:] += -2.0
    Y[2, T0:] += 1.0          # spillover on an affected unit
    D = np.zeros((N, T))
    D[0, T0:] = 1
    D[1, T0:] = 1
    rows = [
        {"unit": f"u{i}", "year": t, "y": float(Y[i, t]), "treat": int(D[i, t])}
        for i in range(N) for t in range(T)
    ]
    return pd.DataFrame(rows)


def _cfg(df, **kw):
    base = dict(df=df, outcome="y", treat="treat", unitid="unit", time="year",
                display_graphs=False)
    base.update(kw)
    return base


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


# --------------------------------------------------------------------------- #
# Plotter: every layout branch
# --------------------------------------------------------------------------- #
class TestPlotter:
    def test_single_treated_cd_with_affected_save(self, tmp_path):
        res = SPILLSYNTH(_cfg(_panel(), affected_units=["u1"])).fit()
        out = tmp_path / "cd.png"
        plot_spillsynth(res, save=str(out))
        assert out.exists()

    def test_single_treated_cd_no_affected_show(self, monkeypatch):
        # p == 0 -> two-panel layout; save=None -> plt.show() branch.
        res = SPILLSYNTH(_cfg(_panel(spillover_idx=None))).fit()
        called = {}
        monkeypatch.setattr(plt, "show", lambda *a, **k: called.setdefault("y", True))
        plot_spillsynth(res)
        assert called.get("y")

    def test_display_graphs_through_estimator(self, monkeypatch, tmp_path):
        monkeypatch.setattr(plt, "show", lambda *a, **k: None)
        SPILLSYNTH(_cfg(_panel(), affected_units=["u1"],
                        display_graphs=True)).fit()

    def test_display_graphs_with_save_and_color_list(self, tmp_path):
        out = tmp_path / "g.png"
        SPILLSYNTH(_cfg(_panel(), affected_units=["u1"], display_graphs=True,
                        save=str(out), counterfactual_color=["green"])).fit()
        assert out.exists()

    def test_iscm_plot(self, tmp_path):
        res = SPILLSYNTH(_cfg(_panel(), method="iscm",
                              affected_units=["u1"])).fit()
        out = tmp_path / "iscm.png"
        plot_spillsynth(res, save=str(out))
        assert out.exists()

    def test_grossi_plot(self, tmp_path):
        res = SPILLSYNTH(_cfg(_panel(), method="grossi",
                              affected_units=["u1"])).fit()
        out = tmp_path / "grossi.png"
        plot_spillsynth(res, save=str(out))
        assert out.exists()

    def test_event_study_multi_treated_with_affected_save(self, tmp_path):
        res = SPILLSYNTH(_cfg(_multi_treated_panel(),
                              affected_units=["u2"])).fit()
        assert res.inputs.n_treated == 2
        out = tmp_path / "es.png"
        plot_spillsynth(res, save=str(out))
        assert out.exists()

    def test_event_study_multi_treated_no_affected_show(self, monkeypatch):
        res = SPILLSYNTH(_cfg(_multi_treated_panel())).fit()
        assert res.inputs.n_treated == 2
        called = {}
        monkeypatch.setattr(plt, "show", lambda *a, **k: called.setdefault("y", True))
        plot_spillsynth(res)
        assert called.get("y")


# --------------------------------------------------------------------------- #
# Grossi covariate + bootstrap inference branches (inference.py 33-50, 81-103)
# --------------------------------------------------------------------------- #
class TestGrossiInference:
    def test_covariate_bootstrap_penalized(self):
        # covariates + n_boot exercises _std_cov, the covariate _fit_weights
        # branch, the covariate bootstrap clean-fit branch, and (with
        # bias_correct) bias_corrected_gaps.
        df = _panel(N=9)
        res = SPILLSYNTH(_cfg(df, method="grossi", affected_units=["u1"],
                              covariates=["x1"], bilevel_solver="penalized",
                              bias_correct=True, n_boot=8, seed=1)).fit()
        f = res.grossi
        assert f.direct_ci is not None
        assert f.direct_ci.shape == (f.gap.shape[0], 2)

    def test_covariate_bootstrap_no_bias_correct(self):
        df = _panel(N=9)
        res = SPILLSYNTH(_cfg(df, method="grossi", affected_units=["u1"],
                              covariates=["x1"], bilevel_solver="penalized",
                              bias_correct=False, n_boot=8, seed=2)).fit()
        assert res.grossi.avg_spillover_ci is not None

    def test_grossi_too_few_clean_controls_raises(self):
        # N=3: treated + 1 affected + 1 clean -> only 1 clean control (< 2).
        df = _panel(N=3)
        with pytest.raises(MlsynthDataError, match=">= 2 clean controls"):
            SPILLSYNTH(_cfg(df, method="grossi", affected_units=["u1"])).fit()


# --------------------------------------------------------------------------- #
# structures.py: SP-dialect aliases + _active error branches
# --------------------------------------------------------------------------- #
class TestStructures:
    def test_iscm_sp_aliases(self):
        res = SPILLSYNTH(_cfg(_panel(), method="iscm",
                              affected_units=["u1"])).fit()
        f = res.iscm
        assert f.att_sp == f.att
        np.testing.assert_array_equal(f.gap_sp, f.gap)
        np.testing.assert_array_equal(f.counterfactual_sp, f.counterfactual)

    def test_grossi_sp_aliases(self):
        res = SPILLSYNTH(_cfg(_panel(), method="grossi",
                              affected_units=["u1"])).fit()
        f = res.grossi
        assert f.att_sp == f.direct_att
        np.testing.assert_array_equal(f.gap_sp, f.gap)
        np.testing.assert_array_equal(f.counterfactual_sp, f.counterfactual)

    def test_active_raises_when_cd_missing(self):
        inp = SPILLSYNTH(_cfg(_panel(), affected_units=["u1"])).fit().inputs
        res = SpillSynthResults(inputs=inp, method="cd", cd=None)
        with pytest.raises(AttributeError, match="no Cao-Dowd fit"):
            _ = res.att

    def test_active_raises_when_iscm_missing(self):
        inp = SPILLSYNTH(_cfg(_panel(), affected_units=["u1"])).fit().inputs
        res = SpillSynthResults(inputs=inp, method="iscm", iscm=None)
        with pytest.raises(AttributeError, match="no inclusive-SCM fit"):
            _ = res.att

    def test_active_raises_when_grossi_missing(self):
        inp = SPILLSYNTH(_cfg(_panel(), affected_units=["u1"])).fit().inputs
        res = SpillSynthResults(inputs=inp, method="grossi", grossi=None)
        with pytest.raises(AttributeError, match="no partial-interference fit"):
            _ = res.att

    def test_active_raises_on_unknown_method(self):
        inp = SPILLSYNTH(_cfg(_panel(), affected_units=["u1"])).fit().inputs
        res = SpillSynthResults(inputs=inp, method="bogus")
        with pytest.raises(AttributeError, match="Unknown SPILLSYNTH method"):
            _ = res.att

    def test_active_raises_when_sar_missing(self):
        inp = SPILLSYNTH(_cfg(_panel(), affected_units=["u1"])).fit().inputs
        res = SpillSynthResults(inputs=inp, method="sar", sar=None)
        with pytest.raises(AttributeError, match="no SAR spillover fit"):
            _ = res.att

    def test_active_raises_when_iterative_missing(self):
        inp = SPILLSYNTH(_cfg(_panel(), affected_units=["u1"])).fit().inputs
        res = SpillSynthResults(inputs=inp, method="iterative", iterative=None)
        with pytest.raises(AttributeError, match="no Iterative-SCM fit"):
            _ = res.att

    def test_results_accessors_route_to_active(self):
        res = SPILLSYNTH(_cfg(_panel(), affected_units=["u1"])).fit()
        assert isinstance(res.att, float)
        assert isinstance(res.att_scm, float)
        assert res.gap.ndim == 1
        assert res.gap_scm.ndim == 1
        assert res.counterfactual.ndim == 1
        assert res.counterfactual_scm.ndim == 1
        assert "u1" in res.spillover_effects


# --------------------------------------------------------------------------- #
# scm_core.py: bad dims (47) and degenerate solver value (70)
# --------------------------------------------------------------------------- #
class TestScmCore:
    def test_fit_demeaned_sc_rejects_one_row(self):
        from mlsynth.exceptions import MlsynthEstimationError
        with pytest.raises(MlsynthEstimationError, match="at least 2 rows"):
            fit_demeaned_sc(np.ones((1, 5)))

    def test_fit_demeaned_sc_rejects_1d(self):
        from mlsynth.exceptions import MlsynthEstimationError
        with pytest.raises(MlsynthEstimationError, match="at least 2 rows"):
            fit_demeaned_sc(np.ones(5))


# --------------------------------------------------------------------------- #
# estimation.py: vanilla_scm_path helper (line 90)
# --------------------------------------------------------------------------- #
def test_p_test_with_explicit_d_and_WT():
    # Exercises the non-default branches of p_test (d and W_T supplied).
    inputs = prepare_spillsynth_inputs(
        df=_panel(), outcome="y", treat="treat", unitid="unit", time="year",
        affected_units=["u1"])
    fit = run_cd(inputs)
    U_pre = compute_pre_residuals(inputs.Y_pre, fit.a, fit.B)
    G_hat = G_matrix(inputs.A, fit.B)
    C = np.zeros((1, inputs.N)); C[0, 0] = 1.0
    res = p_test(alpha_hat=fit.alpha, U_pre=U_pre, G_hat=G_hat, C=C,
                 d=np.zeros(1), W_T=np.eye(1))
    assert res.P_post.shape == (inputs.T1,)
    assert res.p_value.shape == (inputs.T1,)


def test_vanilla_scm_path_matches_a_plus_B():
    inputs = prepare_spillsynth_inputs(
        df=_panel(), outcome="y", treat="treat", unitid="unit", time="year",
        affected_units=["u1"])
    fit = run_cd(inputs)
    cf = vanilla_scm_path(inputs.Y_post, a=fit.a, B=fit.B)
    np.testing.assert_allclose(cf, fit.a[0] + fit.B[0] @ inputs.Y_post)


# --------------------------------------------------------------------------- #
# setup.py validation branches
# --------------------------------------------------------------------------- #
class TestSetupValidation:
    def _inp(self, df, **kw):
        base = dict(outcome="y", treat="treat", unitid="unit", time="year")
        base.update(kw)
        return prepare_spillsynth_inputs(df=df, **base)

    def test_homogeneous_runs(self):
        inp = self._inp(_panel(), affected_units=["u1", "u2"],
                        spillover_structure="homogeneous")
        assert inp.A.shape[1] == 2

    def test_distance_decay_with_distances(self):
        inp = self._inp(_panel(), spillover_structure="distance_decay",
                        unit_distances={"u1": 0.5, "u2": 1.0})
        assert inp.spillover_structure == "distance_decay"

    def test_build_A_distance_decay_nonfinite_raises(self):
        with pytest.raises(MlsynthDataError, match="finite"):
            build_A_distance_decay(np.array([1.0, np.inf]))

    def test_build_A_distance_decay_negative_raises(self):
        with pytest.raises(MlsynthDataError, match="non-negative"):
            build_A_distance_decay(np.array([1.0, -0.5]))

    def test_build_A_distance_decay_bad_n_treated_raises(self):
        with pytest.raises(MlsynthDataError, match="n_treated must be"):
            build_A_distance_decay(np.array([1.0, 0.5]), n_treated=0)

    def test_covariate_empty_window_raises(self):
        # A window with no overlapping years -> covariate has no observations.
        with pytest.raises(MlsynthDataError, match="no observations in its"):
            self._inp(_panel(), affected_units=["u1"], covariates=["x1"],
                      covariate_windows={"x1": (9000, 9001)})

    def test_nan_cells_after_pivot_raises(self):
        # Ragged panel (one affected unit missing a period) bypasses the
        # estimator's balance() check when prepare is called directly, and
        # produces NaN cells after the pivot.
        df = _panel()
        df = df[~((df["unit"] == "u2") & (df["year"] == 5))].copy()
        with pytest.raises(MlsynthDataError, match="missing cells after pivot"):
            self._inp(df, affected_units=["u1"])

    def test_insufficient_pre_periods_raises(self):
        # Only one pre-period -> T0 < 2.
        df = _panel(T=11, T0=1)
        with pytest.raises(MlsynthDataError, match="T0 >= 2"):
            self._inp(df, affected_units=["u1"])


# --------------------------------------------------------------------------- #
# scm_core: degenerate (all-zero) solver value branch (line 70/76 guards)
# --------------------------------------------------------------------------- #
def test_fit_demeaned_sc_no_solution_raises(monkeypatch):
    # When the QP solver returns no value, surface a clear estimation error
    # (scm_core.py line 70).
    import cvxpy as cp
    from mlsynth.exceptions import MlsynthEstimationError

    def _fake_solve(self, *a, **k):
        # Leave self.variables()[0].value as None and report a failure status.
        self._status = "infeasible"
        return None

    monkeypatch.setattr(cp.Problem, "solve", _fake_solve, raising=True)
    monkeypatch.setattr(cp.Problem, "status", "infeasible", raising=False)
    with pytest.raises(MlsynthEstimationError, match="no solution"):
        fit_demeaned_sc(np.random.default_rng(0).standard_normal((4, 10)))


def test_fit_leave_one_out_runs_with_distinct_donors():
    # Exercises the i==0 and i>0 swap/re-permute branches of
    # fit_leave_one_out_sc with a clean panel.
    rng = np.random.default_rng(0)
    Y_pre = rng.standard_normal((4, 12))
    a, B = fit_leave_one_out_sc(Y_pre)
    assert B.shape == (4, 4)
    assert np.allclose(np.diag(B), 0.0)
    assert np.allclose(B.sum(axis=1), 1.0)


# --------------------------------------------------------------------------- #
# Estimator exception-translation branches (spillsynth.py 116-117, 151, 172-175)
# --------------------------------------------------------------------------- #
class TestEstimatorErrorTranslation:
    def test_prepare_mlsynthdataerror_reraised(self):
        # A missing covariate column makes prepare raise MlsynthDataError,
        # which fit() must re-raise unchanged (lines 114-115).
        from mlsynth.config_models import SPILLSYNTHConfig
        cfg = SPILLSYNTHConfig(**_cfg(_panel(), affected_units=["u1"],
                                      covariates=["x1"]))
        object.__setattr__(cfg, "df", cfg.df.drop(columns=["x1"]))
        est = SPILLSYNTH(cfg)
        est.covariates = ["x1"]
        with pytest.raises(MlsynthDataError, match="required column"):
            est.fit()

    def test_prepare_generic_error_wrapped(self, monkeypatch):
        # A non-Mlsynth error from the panel-prep boundary is wrapped into
        # MlsynthDataError (lines 116-119).
        import mlsynth.estimators.spillsynth as mod
        est = SPILLSYNTH(_cfg(_panel(), affected_units=["u1"]))
        monkeypatch.setattr(
            mod, "prepare_spillsynth_inputs",
            lambda **k: (_ for _ in ()).throw(RuntimeError("prep boom")))
        with pytest.raises(MlsynthDataError, match="failed to prepare panel"):
            est.fit()

    def test_estimation_error_wrapped(self, monkeypatch):
        # Force the cd backend to raise a generic error -> wrapped as
        # MlsynthEstimationError (lines 152-155), and a MlsynthConfigError
        # propagates unchanged (line 150-151).
        from mlsynth.exceptions import MlsynthConfigError, MlsynthEstimationError
        import mlsynth.estimators.spillsynth as mod

        est = SPILLSYNTH(_cfg(_panel(), affected_units=["u1"]))

        monkeypatch.setattr(mod, "run_cd",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        with pytest.raises(MlsynthEstimationError, match="estimation failed"):
            est.fit()

        monkeypatch.setattr(
            mod, "run_cd",
            lambda *a, **k: (_ for _ in ()).throw(MlsynthConfigError("cfg")))
        with pytest.raises(MlsynthConfigError):
            est.fit()

    def test_plotting_error_reraised_and_wrapped(self, monkeypatch):
        from mlsynth.exceptions import MlsynthPlottingError
        import mlsynth.estimators.spillsynth as mod

        # MlsynthPlottingError propagates unchanged (lines 172-173).
        est = SPILLSYNTH(_cfg(_panel(), affected_units=["u1"],
                              display_graphs=True))
        monkeypatch.setattr(
            mod, "plot_spillsynth",
            lambda *a, **k: (_ for _ in ()).throw(MlsynthPlottingError("plot")))
        with pytest.raises(MlsynthPlottingError, match="plot"):
            est.fit()

        # A generic error in plotting is wrapped (lines 174-177).
        monkeypatch.setattr(
            mod, "plot_spillsynth",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("kaboom")))
        with pytest.raises(MlsynthPlottingError, match="plotting failed"):
            est.fit()
