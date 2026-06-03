"""Robustness / failure-mode tests for the SpSyDiD subsystem.

Where ``test_spsydid.py`` and ``test_spsydid_coverage.py`` prove every line
runs on well-formed input, this suite proves SpSyDiD *fails loudly and
correctly* on malformed, degenerate, or pathological input:

* :class:`MlsynthDataError` at the data/setup boundary -- bad spatial-matrix
  shape / finiteness / non-negativity / zero diagonal, NaN outcome or
  treatment, non-binary treatment, unbalanced panel, no pure controls,
  no directly-treated units, non-finite coordinates in the W builders;
* ``ValueError``-style guards (raised as :class:`MlsynthDataError` here,
  matching the package's convention) for nonsensical builder params
  (bad ``k``, non-positive ``power`` / ``cutoff``, negative ``zeta``);
* ``RuntimeWarning`` diagnostics for previously-silent suspect-but-ran
  modes -- units with no spatial neighbours (isolated rows) and a
  rank-deficient final WLS design (tau / tau_s not separately identified).

A :func:`warnings_as_errors` context manager asserts the happy path emits
*no* warning.
"""

from __future__ import annotations

import contextlib
import warnings as _warnings

import numpy as np
import pandas as pd
import pytest

from mlsynth import SpSyDiD
from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.spsydid_helpers import (
    inverse_distance_weights,
    knn_weights,
    prepare_spsydid_inputs,
    row_standardize,
)
from mlsynth.utils.spsydid_helpers import pipeline as pipeline_mod


# ----------------------------------------------------------------------
# context manager: assert *no* warning is emitted on the happy path
# ----------------------------------------------------------------------
@contextlib.contextmanager
def warnings_as_errors():
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        yield


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------
def _panel(N=8, T=10, T_pre=6, treated=(0,), seed=0, tau=2.0, tau_s=1.0,
           W=None):
    rng = np.random.default_rng(seed)
    unit_fe = rng.standard_normal(N) * 0.3
    time_fe = np.linspace(0.0, 1.0, T)
    Y0 = unit_fe[:, None] + time_fe[None, :] + rng.standard_normal((N, T)) * 0.05
    D = np.zeros((N, T))
    for u in treated:
        D[u, T_pre:] = 1.0
    if W is None:
        W = _line_W(N)
    Y = Y0 + tau * D + tau_s * (W @ D)
    rows = [
        {"unit": i, "time": t, "y": float(Y[i, t]), "D": float(D[i, t])}
        for i in range(N) for t in range(T)
    ]
    return pd.DataFrame(rows), W


def _line_W(N=8):
    W = np.zeros((N, N))
    for i in range(N - 1):
        W[i, i + 1] = 1.0
        W[i + 1, i] = 1.0
    return row_standardize(W)


# ======================================================================
# MlsynthDataError -- spatial-matrix validation at the boundary
# ======================================================================
class TestSpatialMatrixGuards:
    def test_wrong_shape(self):
        df, _ = _panel()
        with pytest.raises(MlsynthDataError, match="shape"):
            SpSyDiD({"df": df, "outcome": "y", "treat": "D", "unitid": "unit",
                     "time": "time", "spatial_matrix": np.zeros((3, 3)),
                     "display_graphs": False}).fit()

    def test_negative_entries(self):
        df, _ = _panel()
        W = -np.ones((8, 8))
        np.fill_diagonal(W, 0.0)
        with pytest.raises(MlsynthDataError, match="non-negative"):
            SpSyDiD({"df": df, "outcome": "y", "treat": "D", "unitid": "unit",
                     "time": "time", "spatial_matrix": W,
                     "display_graphs": False}).fit()

    def test_nonzero_diagonal(self):
        df, _ = _panel()
        W = _line_W()
        W = W + np.eye(8)
        with pytest.raises(MlsynthDataError, match="zero diagonal"):
            SpSyDiD({"df": df, "outcome": "y", "treat": "D", "unitid": "unit",
                     "time": "time", "spatial_matrix": W,
                     "display_graphs": False}).fit()

    def test_nan_entry(self):
        df, _ = _panel()
        W = _line_W()
        W[0, 1] = np.nan
        with pytest.raises(MlsynthDataError, match="NaN or Inf"):
            SpSyDiD({"df": df, "outcome": "y", "treat": "D", "unitid": "unit",
                     "time": "time", "spatial_matrix": W,
                     "display_graphs": False}).fit()


# ======================================================================
# MlsynthDataError -- panel validation
# ======================================================================
class TestPanelGuards:
    def test_outcome_nan(self):
        df, W = _panel()
        df.loc[0, "y"] = np.nan
        with pytest.raises(MlsynthDataError, match="Outcome column contains NaN"):
            prepare_spsydid_inputs(df=df, outcome="y", treat="D",
                                   unitid="unit", time="time",
                                   spatial_matrix=W)

    def test_treatment_nan(self):
        df, W = _panel()
        df.loc[0, "D"] = np.nan
        with pytest.raises(MlsynthDataError, match="Treatment column contains NaN"):
            prepare_spsydid_inputs(df=df, outcome="y", treat="D",
                                   unitid="unit", time="time",
                                   spatial_matrix=W)

    def test_non_binary_treatment(self):
        df, W = _panel()
        # set a treated cell to 2.0 -> non-binary
        idx = df.index[(df["unit"] == 0) & (df["time"] == 9)]
        df.loc[idx, "D"] = 2.0
        with pytest.raises(MlsynthDataError, match="binary 0/1 indicator"):
            prepare_spsydid_inputs(df=df, outcome="y", treat="D",
                                   unitid="unit", time="time",
                                   spatial_matrix=W)

    def test_unbalanced_panel(self):
        df, W = _panel()
        df = df.drop(df.index[5])  # drop one (unit, time) cell
        with pytest.raises(MlsynthDataError, match="not balanced"):
            prepare_spsydid_inputs(df=df, outcome="y", treat="D",
                                   unitid="unit", time="time",
                                   spatial_matrix=W)

    def test_no_pure_controls(self):
        # fully-connected W + one treated unit -> every donor is exposed.
        N, T, T_pre = 6, 8, 5
        W = row_standardize(np.ones((N, N)) - np.eye(N))
        df, _ = _panel(N=N, T=T, T_pre=T_pre, treated=(0,), W=W, tau_s=0.0)
        with pytest.raises(MlsynthDataError, match="No pure controls"):
            prepare_spsydid_inputs(df=df, outcome="y", treat="D",
                                   unitid="unit", time="time",
                                   spatial_matrix=W)

    def test_no_directly_treated(self):
        df, W = _panel(treated=())
        with pytest.raises(MlsynthDataError, match="No directly treated"):
            prepare_spsydid_inputs(df=df, outcome="y", treat="D",
                                   unitid="unit", time="time",
                                   spatial_matrix=W)

    def test_treated_at_earliest_period(self):
        df, W = _panel(treated=())
        df.loc[df["unit"] == 0, "D"] = 1.0  # treated everywhere incl. t=0
        with pytest.raises(MlsynthDataError, match="earliest period"):
            prepare_spsydid_inputs(df=df, outcome="y", treat="D",
                                   unitid="unit", time="time",
                                   spatial_matrix=W)


# ======================================================================
# MlsynthDataError -- spatial builder param guards
# ======================================================================
class TestBuilderGuards:
    def test_knn_bad_k(self):
        with pytest.raises(MlsynthDataError, match="k must lie"):
            knn_weights(np.zeros((5, 2)), k=10)

    def test_knn_nonfinite_coords(self):
        coords = np.array([[0.0, 0.0], [np.nan, 1.0], [2.0, 2.0]])
        with pytest.raises(MlsynthDataError, match="NaN or Inf"):
            knn_weights(coords, k=1)

    def test_inverse_distance_bad_power(self):
        with pytest.raises(MlsynthDataError, match="power must be positive"):
            inverse_distance_weights(np.zeros((4, 2)), power=-1.0)

    def test_inverse_distance_bad_cutoff(self):
        with pytest.raises(MlsynthDataError, match="cutoff must be positive"):
            inverse_distance_weights(np.zeros((4, 2)), cutoff=0.0)

    def test_inverse_distance_nonfinite_coords(self):
        coords = np.array([[0.0, 0.0], [np.inf, 1.0]])
        with pytest.raises(MlsynthDataError, match="NaN or Inf"):
            inverse_distance_weights(coords)


# ======================================================================
# RuntimeWarning -- previously-silent suspect-but-ran modes
# ======================================================================
class TestRuntimeWarnings:
    def test_isolated_unit_warns(self):
        # Build a W where one PURE-CONTROL unit has no neighbours. Treat
        # unit 0 on a line graph; isolate unit 6 (far from everyone).
        N = 8
        W = np.zeros((N, N))
        for i in range(N - 2):  # connect 0..5 in a line; leave 6, 7 isolated-ish
            W[i, i + 1] = 1.0
            W[i + 1, i] = 1.0
        # unit 7 has neighbour 6 only; make unit 6 isolated by removing its edges
        W[5, 6] = W[6, 5] = 0.0
        W[6, 7] = W[7, 6] = 0.0  # unit 6 fully isolated
        with pytest.warns(RuntimeWarning, match="no neighbours"):
            row_standardize(W, warn_isolated=True)

    def test_isolated_unit_warns_through_setup(self):
        N = 8
        W = np.zeros((N, N))
        for i in range(1, N - 1):
            W[i, i + 1] = 1.0
            W[i + 1, i] = 1.0
        # unit 0 is treated and has no neighbours -> isolated row.
        df, _ = _panel(N=N, treated=(0,), W=row_standardize(W), tau_s=0.0)
        with pytest.warns(RuntimeWarning, match="no neighbours"):
            prepare_spsydid_inputs(df=df, outcome="y", treat="D",
                                   unitid="unit", time="time",
                                   spatial_matrix=W)

    def test_rank_deficient_wls_warns(self):
        # W == 0 -> WD column is all-zero -> design rank-deficient -> warn.
        N = 8
        df, _ = _panel(N=N, W=np.zeros((N, N)), tau_s=0.0)
        inp = prepare_spsydid_inputs(
            df=df, outcome="y", treat="D", unitid="unit", time="time",
            spatial_matrix=np.zeros((N, N)),
        )
        with pytest.warns(RuntimeWarning, match="rank-deficient"):
            pipeline_mod.run_spsydid(inp)


# ======================================================================
# happy path emits NO warning
# ======================================================================
class TestHappyPathNoWarning:
    def test_fit_emits_no_warning(self):
        df, W = _panel(N=10, T=12, T_pre=8, treated=(0, 9))
        with warnings_as_errors():
            res = SpSyDiD({"df": df, "outcome": "y", "treat": "D",
                           "unitid": "unit", "time": "time",
                           "spatial_matrix": W,
                           "display_graphs": False}).fit()
        assert np.isfinite(res.att)
        assert np.isfinite(res.aite)

    def test_prepare_inputs_emits_no_warning(self):
        df, W = _panel(N=10, T=12, T_pre=8, treated=(0, 9))
        with warnings_as_errors():
            prepare_spsydid_inputs(df=df, outcome="y", treat="D",
                                   unitid="unit", time="time",
                                   spatial_matrix=W)
