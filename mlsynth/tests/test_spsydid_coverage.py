"""Line/branch coverage tests for the SpSyDiD subsystem.

Drives the remaining uncovered branches in
``mlsynth/estimators/spsydid.py`` and ``mlsynth/utils/spsydid_helpers/**``
directly with small synthetic panels: every validation branch in the
weight QPs and spatial builders, the plotter (Agg backend), the setup
partition edge cases, and the pipeline solver-failure paths.
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")  # noqa: E402  (must precede pyplot import)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from mlsynth import SpSyDiD  # noqa: E402
from mlsynth.config_models import SpSyDiDConfig  # noqa: E402
from mlsynth.exceptions import (  # noqa: E402
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)
from mlsynth.utils.spsydid_helpers import (  # noqa: E402
    SpSyDiDInputs,
    compute_regularization,
    contiguity_weights,
    fit_time_weights,
    fit_unit_weights,
    inverse_distance_weights,
    knn_weights,
    prepare_spsydid_inputs,
    row_standardize,
    validate_spatial_matrix,
)
from mlsynth.utils.spsydid_helpers import pipeline as pipeline_mod  # noqa: E402
from mlsynth.utils.spsydid_helpers import weights as weights_mod  # noqa: E402
from mlsynth.utils.spsydid_helpers.plotter import plot_spsydid  # noqa: E402

import cvxpy as cp  # noqa: E402


# ----------------------------------------------------------------------
# Shared small-panel builder
# ----------------------------------------------------------------------
def _small_panel(N=6, T=8, T_pre=5, treated=(0,), seed=0):
    """A tiny line-graph panel with a clean treatment effect."""
    rng = np.random.default_rng(seed)
    unit_fe = rng.standard_normal(N) * 0.3
    time_fe = np.linspace(0.0, 1.0, T)
    Y = unit_fe[:, None] + time_fe[None, :] + rng.standard_normal((N, T)) * 0.05
    D = np.zeros((N, T))
    for u in treated:
        D[u, T_pre:] = 1.0
    Y = Y + 2.0 * D
    rows = [
        {"unit": i, "time": t, "y": float(Y[i, t]), "D": float(D[i, t])}
        for i in range(N) for t in range(T)
    ]
    return pd.DataFrame(rows)


def _line_W(N=6):
    """Row-standardised line-graph contiguity (each interior unit has 2)."""
    W = np.zeros((N, N))
    for i in range(N - 1):
        W[i, i + 1] = 1.0
        W[i + 1, i] = 1.0
    return row_standardize(W)


# ======================================================================
# weights.py -- every validation branch + happy path
# ======================================================================
class TestFitTimeWeights:
    def test_happy_path(self):
        pre = np.linspace(0, 1, 5)[:, None] * np.ones((5, 3))
        post = np.array([1.1, 1.2, 1.3])
        b0, lam = fit_time_weights(pre, post)
        assert lam is not None
        assert abs(lam.sum() - 1.0) < 1e-6

    def test_pre_not_2d(self):
        with pytest.raises(MlsynthDataError, match="2-D"):
            fit_time_weights(np.zeros(5), np.zeros(3))

    def test_post_not_1d(self):
        with pytest.raises(MlsynthDataError, match="1-D"):
            fit_time_weights(np.zeros((5, 3)), np.zeros((3, 2)))

    def test_empty_pre(self):
        with pytest.raises(MlsynthDataError, match="Empty"):
            fit_time_weights(np.zeros((0, 3)), np.zeros(3))

    def test_empty_donors(self):
        with pytest.raises(MlsynthDataError, match="Empty"):
            fit_time_weights(np.zeros((5, 0)), np.zeros(0))

    def test_donor_count_mismatch(self):
        with pytest.raises(MlsynthDataError, match="mismatch"):
            fit_time_weights(np.zeros((5, 3)), np.zeros(4))


class TestComputeRegularization:
    def test_happy_path(self):
        pre = np.cumsum(np.ones((6, 3)), axis=0).astype(float)
        z = compute_regularization(pre, num_post_periods=4)
        assert z >= 0.0 and np.isfinite(z)

    def test_not_2d(self):
        with pytest.raises(MlsynthDataError, match="2-D"):
            compute_regularization(np.zeros(5), num_post_periods=2)

    def test_negative_post(self):
        with pytest.raises(MlsynthDataError, match="non-negative"):
            compute_regularization(np.zeros((5, 3)), num_post_periods=-1)

    def test_single_row_falls_back_to_unit_sd(self):
        # < 2 rows -> sd_diff defaults to 1.0 branch.
        z = compute_regularization(np.zeros((1, 3)), num_post_periods=16)
        assert z == pytest.approx(16 ** 0.25)

    def test_zero_columns_falls_back(self):
        z = compute_regularization(np.zeros((5, 0)), num_post_periods=16)
        assert z == pytest.approx(16 ** 0.25)

    def test_constant_panel_zero_sd_falls_back(self):
        # diffs all zero -> sd_diff <= 0 branch -> 1.0
        pre = np.ones((6, 3))
        z = compute_regularization(pre, num_post_periods=16)
        assert z == pytest.approx(16 ** 0.25)

    def test_two_rows_single_diff_falls_back(self):
        # exactly 2 rows, 1 donor -> diffs.size == 1 -> not > 1 -> 1.0
        pre = np.array([[0.0], [5.0]])
        z = compute_regularization(pre, num_post_periods=16)
        assert z == pytest.approx(16 ** 0.25)


class TestFitUnitWeights:
    def test_happy_path(self):
        pre = np.linspace(0, 1, 5)[:, None] * np.ones((5, 3))
        target = np.linspace(0, 1, 5)
        b0, om = fit_unit_weights(pre, target, zeta=0.1)
        assert om is not None
        assert abs(om.sum() - 1.0) < 1e-6

    def test_pre_not_2d(self):
        with pytest.raises(MlsynthDataError, match="2-D"):
            fit_unit_weights(np.zeros(5), np.zeros(5), zeta=0.1)

    def test_target_not_1d(self):
        with pytest.raises(MlsynthDataError, match="1-D"):
            fit_unit_weights(np.zeros((5, 3)), np.zeros((5, 2)), zeta=0.1)

    def test_negative_zeta(self):
        with pytest.raises(MlsynthDataError, match="non-negative"):
            fit_unit_weights(np.zeros((5, 3)), np.zeros(5), zeta=-1.0)

    def test_empty_pre(self):
        with pytest.raises(MlsynthDataError, match="Empty pre-period"):
            fit_unit_weights(np.zeros((0, 3)), np.zeros(0), zeta=0.1)

    def test_no_donors(self):
        with pytest.raises(MlsynthDataError, match="at least one donor"):
            fit_unit_weights(np.zeros((5, 0)), np.zeros(5), zeta=0.1)

    def test_pre_length_mismatch(self):
        with pytest.raises(MlsynthDataError, match="length mismatch"):
            fit_unit_weights(np.zeros((5, 3)), np.zeros(4), zeta=0.1)


class _StatusProblem:
    """Stub cp.Problem that reports a non-optimal status."""

    def __init__(self, *a, **k):
        self.status = "infeasible"

    def solve(self, *a, **k):
        return None


class _RaisingProblem:
    """Stub cp.Problem whose solve raises a cvxpy SolverError."""

    def __init__(self, *a, **k):
        self.status = "optimal"

    def solve(self, *a, **k):
        raise cp.error.SolverError("synthetic solver failure")


class TestWeightSolverPaths:
    """Cover the non-optimal-status (return None) and SolverError branches."""

    def test_time_weights_nonoptimal_returns_none(self, monkeypatch):
        monkeypatch.setattr(weights_mod.cp, "Problem", _StatusProblem)
        b0, lam = fit_time_weights(np.zeros((5, 3)), np.zeros(3))
        assert b0 is None and lam is None

    def test_time_weights_solver_error(self, monkeypatch):
        monkeypatch.setattr(weights_mod.cp, "Problem", _RaisingProblem)
        with pytest.raises(MlsynthEstimationError, match="fit_time_weights"):
            fit_time_weights(np.zeros((5, 3)), np.zeros(3))

    def test_unit_weights_nonoptimal_returns_none(self, monkeypatch):
        monkeypatch.setattr(weights_mod.cp, "Problem", _StatusProblem)
        b0, om = fit_unit_weights(np.zeros((5, 3)), np.zeros(5), zeta=0.1)
        assert b0 is None and om is None

    def test_unit_weights_solver_error(self, monkeypatch):
        monkeypatch.setattr(weights_mod.cp, "Problem", _RaisingProblem)
        with pytest.raises(MlsynthEstimationError, match="fit_unit_weights"):
            fit_unit_weights(np.zeros((5, 3)), np.zeros(5), zeta=0.1)


# ======================================================================
# spatial.py -- builders + validation branches
# ======================================================================
class TestSpatialBranches:
    def test_validate_coerces_non_ndarray(self):
        # list input -> np.asarray branch (line 30-31).
        W = [[0.0, 1.0], [1.0, 0.0]]
        out = validate_spatial_matrix(W, n_units=2)
        assert isinstance(out, np.ndarray)
        assert out.dtype == float

    def test_validate_rejects_nan(self):
        W = np.zeros((3, 3))
        W[0, 1] = np.nan
        with pytest.raises(MlsynthDataError, match="NaN or Inf"):
            validate_spatial_matrix(W, n_units=3)

    def test_validate_rejects_inf(self):
        W = np.zeros((3, 3))
        W[0, 1] = np.inf
        with pytest.raises(MlsynthDataError, match="NaN or Inf"):
            validate_spatial_matrix(W, n_units=3)

    def test_validate_rejects_1d(self):
        with pytest.raises(MlsynthDataError, match="shape"):
            validate_spatial_matrix(np.zeros(5), n_units=5)

    def test_row_standardize_isolated_row_stays_zero(self):
        W = np.array([[0.0, 2.0], [0.0, 0.0]])
        S = row_standardize(W)
        np.testing.assert_allclose(S[0], [0.0, 1.0])
        np.testing.assert_allclose(S[1], [0.0, 0.0])

    def test_knn_coords_not_2d(self):
        with pytest.raises(MlsynthDataError, match="2-D"):
            knn_weights(np.zeros(5), k=2)

    def test_knn_k_too_small(self):
        with pytest.raises(MlsynthDataError, match=r"k must lie"):
            knn_weights(np.zeros((5, 2)), k=0)

    def test_knn_k_too_large(self):
        with pytest.raises(MlsynthDataError, match=r"k must lie"):
            knn_weights(np.zeros((5, 2)), k=5)

    def test_knn_non_standardized(self):
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        W = knn_weights(coords, k=1, row_standardized=False)
        assert W.sum(axis=1).max() == 1.0  # raw 0/1 edges

    def test_knn_nonfinite_coords(self):
        coords = np.array([[0.0, 0.0], [np.nan, 0.0], [2.0, 0.0]])
        with pytest.raises(MlsynthDataError, match="NaN or Inf"):
            knn_weights(coords, k=1)

    def test_inverse_distance_coords_not_2d(self):
        with pytest.raises(MlsynthDataError, match="2-D"):
            inverse_distance_weights(np.zeros(5))

    def test_inverse_distance_bad_power(self):
        with pytest.raises(MlsynthDataError, match="power"):
            inverse_distance_weights(np.zeros((3, 2)), power=0.0)

    def test_inverse_distance_bad_cutoff(self):
        with pytest.raises(MlsynthDataError, match="cutoff"):
            inverse_distance_weights(np.zeros((3, 2)), cutoff=-1.0)

    def test_inverse_distance_nonfinite(self):
        coords = np.array([[0.0, 0.0], [np.inf, 0.0]])
        with pytest.raises(MlsynthDataError, match="NaN or Inf"):
            inverse_distance_weights(coords)

    def test_inverse_distance_non_standardized_and_power(self):
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        W = inverse_distance_weights(coords, power=2.0, row_standardized=False)
        assert W[0, 1] == pytest.approx(1.0)
        assert W[0, 2] == pytest.approx(0.25)

    def test_contiguity_skips_unknown_unit(self):
        # adjacency contains a unit id not in unit_order -> `continue` branch.
        adj = {0: [1], 1: [0], 99: [0, 1]}
        W = contiguity_weights(adj, unit_order=[0, 1], row_standardized=False)
        np.testing.assert_allclose(W, [[0.0, 1.0], [1.0, 0.0]])

    def test_contiguity_skips_self_and_unknown_neighbour(self):
        # neighbour list has self (v == u) and an unknown id -> both skipped.
        adj = {0: [0, 1, 99], 1: [0]}
        W = contiguity_weights(adj, unit_order=[0, 1], row_standardized=False)
        np.testing.assert_allclose(W, [[0.0, 1.0], [1.0, 0.0]])


# ======================================================================
# setup.py -- partition edge cases & validation
# ======================================================================
class TestSetupBranches:
    def test_missing_column(self):
        df = _small_panel()
        with pytest.raises(MlsynthDataError, match="missing"):
            prepare_spsydid_inputs(
                df=df, outcome="nope", treat="D", unitid="unit",
                time="time", spatial_matrix=_line_W(),
            )

    def test_outcome_nan(self):
        df = _small_panel()
        df.loc[0, "y"] = np.nan
        with pytest.raises(MlsynthDataError, match="Outcome column contains NaN"):
            prepare_spsydid_inputs(
                df=df, outcome="y", treat="D", unitid="unit",
                time="time", spatial_matrix=_line_W(),
            )

    def test_explicit_unit_order(self):
        df = _small_panel()
        order = list(range(6))
        inp = prepare_spsydid_inputs(
            df=df, outcome="y", treat="D", unitid="unit", time="time",
            spatial_matrix=_line_W(), unit_order=order,
        )
        assert inp.unit_names == order

    def test_unit_order_mismatch(self):
        df = _small_panel()
        with pytest.raises(MlsynthDataError, match="unit_order does not match"):
            prepare_spsydid_inputs(
                df=df, outcome="y", treat="D", unitid="unit", time="time",
                spatial_matrix=_line_W(), unit_order=[0, 1, 2, 3, 4, 999],
            )

    def test_unbalanced_panel(self):
        df = _small_panel()
        df = df.drop(df.index[0])  # remove one (unit, time) cell
        with pytest.raises(MlsynthDataError, match="not balanced"):
            prepare_spsydid_inputs(
                df=df, outcome="y", treat="D", unitid="unit", time="time",
                spatial_matrix=_line_W(),
            )

    def test_no_row_standardize_branch(self):
        df = _small_panel()
        W = _line_W()  # already standardised
        inp = prepare_spsydid_inputs(
            df=df, outcome="y", treat="D", unitid="unit", time="time",
            spatial_matrix=W, row_standardize_spatial=False,
        )
        np.testing.assert_allclose(inp.spatial_matrix, W)

    def test_no_directly_treated(self):
        df = _small_panel(treated=())  # nobody treated
        with pytest.raises(MlsynthDataError, match="No directly treated"):
            prepare_spsydid_inputs(
                df=df, outcome="y", treat="D", unitid="unit", time="time",
                spatial_matrix=_line_W(),
            )

    def test_treated_at_earliest_period(self):
        df = _small_panel(treated=())
        # make unit 0 treated at the very first period -> no pre-period.
        df.loc[(df["unit"] == 0), "D"] = 1.0
        with pytest.raises(MlsynthDataError, match="earliest period"):
            prepare_spsydid_inputs(
                df=df, outcome="y", treat="D", unitid="unit", time="time",
                spatial_matrix=_line_W(),
            )

    def test_minimal_two_period_panel(self):
        # Smallest valid panel: T0 == 1, T_post == 1. (The T0 < 1 and
        # T - T0 < 1 guards in setup.py are defensively unreachable because
        # the "earliest period" and "no directly treated" guards fire first;
        # documented with pragma: no cover.)
        N, T = 6, 2
        Y = np.random.default_rng(0).standard_normal((N, T))
        D = np.zeros((N, T))
        D[0, 1] = 1.0  # treated at t=1 only -> T0=1, T_post=1 (valid)
        rows = [
            {"unit": i, "time": t, "y": float(Y[i, t]), "D": float(D[i, t])}
            for i in range(N) for t in range(T)
        ]
        df = pd.DataFrame(rows)
        # This one is valid (T0=1, T_post=1); assert it constructs.
        inp = prepare_spsydid_inputs(
            df=df, outcome="y", treat="D", unitid="unit", time="time",
            spatial_matrix=_line_W(),
        )
        assert inp.T0 == 1 and inp.T - inp.T0 == 1


# ======================================================================
# structures.py -- the N property (line 73)
# ======================================================================
def test_inputs_N_property():
    df = _small_panel()
    inp = prepare_spsydid_inputs(
        df=df, outcome="y", treat="D", unitid="unit", time="time",
        spatial_matrix=_line_W(),
    )
    assert inp.N == 6
    assert inp.N == len(inp.unit_names)


# ======================================================================
# plotter.py -- full render path on Agg backend
# ======================================================================
class TestPlotter:
    def test_plot_runs_with_str_color(self):
        df = _small_panel()
        res = SpSyDiD({
            "df": df, "outcome": "y", "treat": "D", "unitid": "unit",
            "time": "time", "spatial_matrix": _line_W(),
            "display_graphs": False,
        }).fit()
        plot_spsydid(res, counterfactual_color="red")
        plt.close("all")

    def test_plot_runs_with_list_color(self):
        df = _small_panel()
        res = SpSyDiD({
            "df": df, "outcome": "y", "treat": "D", "unitid": "unit",
            "time": "time", "spatial_matrix": _line_W(),
            "display_graphs": False,
        }).fit()
        plot_spsydid(res, counterfactual_color=["green"])
        plt.close("all")

    def test_fit_with_display_graphs(self):
        # Exercises the display_graphs=True branch in SpSyDiD.fit.
        df = _small_panel()
        SpSyDiD({
            "df": df, "outcome": "y", "treat": "D", "unitid": "unit",
            "time": "time", "spatial_matrix": _line_W(),
            "display_graphs": True,
        }).fit()
        plt.close("all")


# ======================================================================
# estimator wrapper -- config dict error & exception wrapping
# ======================================================================
class TestEstimatorWrapper:
    def test_invalid_config_dict_raises_config_error(self):
        with pytest.raises(MlsynthConfigError, match="Invalid SpSyDiD"):
            SpSyDiD({"outcome": "y"})  # missing required fields

    def test_accepts_config_object(self):
        df = _small_panel()
        cfg = SpSyDiDConfig(
            df=df, outcome="y", treat="D", unitid="unit", time="time",
            spatial_matrix=_line_W(), display_graphs=False,
        )
        res = SpSyDiD(cfg).fit()
        assert res.att == res.tau

    def test_data_error_propagates_unwrapped(self):
        df = _small_panel()
        bad_W = np.zeros((3, 3))  # wrong shape
        with pytest.raises(MlsynthDataError):
            SpSyDiD({
                "df": df, "outcome": "y", "treat": "D", "unitid": "unit",
                "time": "time", "spatial_matrix": bad_W,
                "display_graphs": False,
            }).fit()

    def test_unexpected_exception_wrapped(self, monkeypatch):
        # Force run_spsydid to raise a generic exception -> wrapped as
        # MlsynthEstimationError by the estimator's catch-all.
        df = _small_panel()

        def _boom(_inputs):
            raise RuntimeError("synthetic boom")

        monkeypatch.setattr(
            "mlsynth.estimators.spsydid.run_spsydid", _boom
        )
        with pytest.raises(MlsynthEstimationError, match="estimation failed"):
            SpSyDiD({
                "df": df, "outcome": "y", "treat": "D", "unitid": "unit",
                "time": "time", "spatial_matrix": _line_W(),
                "display_graphs": False,
            }).fit()


# ======================================================================
# pipeline.py -- solver-failure paths (78, 85) and LinAlgError (139-140)
# ======================================================================
class TestPipelineFailures:
    def _inputs(self):
        df = _small_panel()
        return prepare_spsydid_inputs(
            df=df, outcome="y", treat="D", unitid="unit", time="time",
            spatial_matrix=_line_W(),
        )

    def test_unit_weight_qp_none_raises(self, monkeypatch):
        inp = self._inputs()
        monkeypatch.setattr(
            pipeline_mod, "fit_unit_weights", lambda **kw: (None, None)
        )
        with pytest.raises(MlsynthEstimationError, match="unit-weight QP failed"):
            pipeline_mod.run_spsydid(inp)

    def test_time_weight_qp_none_raises(self, monkeypatch):
        inp = self._inputs()
        # Let unit weights succeed but force time weights to fail.
        real_unit = pipeline_mod.fit_unit_weights
        monkeypatch.setattr(pipeline_mod, "fit_unit_weights", real_unit)
        monkeypatch.setattr(
            pipeline_mod, "fit_time_weights", lambda **kw: (None, None)
        )
        with pytest.raises(MlsynthEstimationError, match="time-weight QP failed"):
            pipeline_mod.run_spsydid(inp)

    def test_lstsq_linalgerror_wrapped(self, monkeypatch):
        inp = self._inputs()

        def _bad_lstsq(*a, **k):
            raise np.linalg.LinAlgError("synthetic SVD failure")

        monkeypatch.setattr(np.linalg, "lstsq", _bad_lstsq)
        with pytest.raises(MlsynthEstimationError, match="final WLS failed"):
            pipeline_mod.run_spsydid(inp)
