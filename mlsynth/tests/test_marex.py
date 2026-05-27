"""Tests for the MAREX (synthetic experimental design) estimator.

Covers config validation, the design optimizers, clustering, cost/budget and
cardinality constraints, blank-period inference, the relaxed solver, and
plotting.

Reference: Abadie & Zhao (2026), "Synthetic Controls for Experimental Design."
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as _plt

from mlsynth import MAREX
from mlsynth.config_models import MAREXConfig
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.marex_helpers.optimization import solve_design, solve_design_relaxed
from mlsynth.utils.marex_helpers.plotter import plot_marex
from mlsynth.utils.marex_helpers.structures import MAREXInference, MAREXResults


def _panel(J=8, T=14, clusters=False, seed=0):
    rng = np.random.default_rng(seed)
    F = rng.normal(0, 1, (T, 2))
    lam = rng.normal(0, 1, (J, 2))
    Y = lam @ F.T + 0.2 * rng.standard_normal((J, T))
    grp = np.repeat([0, 1], J // 2)
    rows = []
    for j in range(J):
        for t in range(T):
            row = {"unit": f"u{j:02d}", "time": t, "y": float(Y[j, t])}
            if clusters:
                row["grp"] = int(grp[j])
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def panel():
    return _panel()


# ----------------------------------------------------------------------
# Config validation
# ----------------------------------------------------------------------

class TestMAREXConfig:
    def test_defaults(self, panel):
        cfg = MAREXConfig(df=panel, outcome="y", unitid="unit", time="time")
        assert cfg.design == "base" and cfg.exclusive is True and cfg.relaxed is False

    def test_bad_design_rejected(self, panel):
        with pytest.raises(MlsynthDataError):
            MAREXConfig(df=panel, outcome="y", unitid="unit", time="time", design="nope")

    def test_bad_T0_rejected(self, panel):
        with pytest.raises(MlsynthDataError):
            MAREXConfig(df=panel, outcome="y", unitid="unit", time="time", T0=999)

    def test_missing_column_rejected(self, panel):
        with pytest.raises(MlsynthDataError):
            MAREXConfig(df=panel, outcome="nope", unitid="unit", time="time")

    def test_bad_cluster_column(self, panel):
        with pytest.raises(MlsynthDataError):
            MAREX({"df": panel, "outcome": "y", "unitid": "unit", "time": "time",
                   "cluster": "missing", "m_eq": 1})


# ----------------------------------------------------------------------
# Estimation
# ----------------------------------------------------------------------

class TestEstimator:
    def test_fit_base(self, panel):
        res = MAREX({"df": panel, "outcome": "y", "unitid": "unit", "time": "time",
                     "T0": 10, "m_eq": 2}).fit()
        assert isinstance(res, MAREXResults) and res.mode == "marex"
        assert res.synthetic_treated.shape == (14,)
        assert len(res.treated_units) == 2
        g = res.globres
        # treated and control weights are disjoint
        assert np.all(g.treated_weights_agg * g.control_weights_agg == 0)

    @pytest.mark.parametrize("design,kw", [
        ("base", {}),
        ("weak", {"beta": 0.1}),
        ("eq11", {"lambda1": 0.1, "lambda2": 0.1}),
        ("unit", {"lambda1_unit": 0.1}),
    ])
    def test_designs_run(self, panel, design, kw):
        res = MAREX({"df": panel, "outcome": "y", "unitid": "unit", "time": "time",
                     "T0": 10, "m_eq": 2, "design": design, **kw}).fit()
        assert res.study.design == design
        assert np.isfinite(res.clusters["0"].rmse)

    def test_m_min_max_range(self, panel):
        res = MAREX({"df": panel, "outcome": "y", "unitid": "unit", "time": "time",
                     "T0": 10, "m_min": 1, "m_max": 3}).fit()
        assert 1 <= len(res.treated_units) <= 3

    def test_conflicting_m_eq_and_range(self, panel):
        with pytest.raises(MlsynthConfigError):
            MAREX({"df": panel, "outcome": "y", "unitid": "unit", "time": "time",
                   "T0": 10, "m_eq": 1, "m_min": 1}).fit()

    def test_missing_cardinality(self, panel):
        with pytest.raises(MlsynthConfigError):
            MAREX({"df": panel, "outcome": "y", "unitid": "unit", "time": "time",
                   "T0": 10}).fit()

    def test_clustered_one_treated_each(self):
        df = _panel(J=10, T=14, clusters=True)
        res = MAREX({"df": df, "outcome": "y", "unitid": "unit", "time": "time",
                     "cluster": "grp", "T0": 10, "m_eq": 1}).fit()
        assert set(res.clusters) == {"0", "1"}
        for c in res.clusters.values():
            assert len(c.unit_weight_map["Treated"]) == 1

    def test_costs_budget(self, panel):
        res = MAREX({"df": panel, "outcome": "y", "unitid": "unit", "time": "time",
                     "T0": 10, "m_eq": 2, "costs": [1.0] * 8, "budget": 10.0}).fit()
        assert len(res.treated_units) == 2

    def test_relaxed_runs(self, panel):
        res = MAREX({"df": panel, "outcome": "y", "unitid": "unit", "time": "time",
                     "T0": 10, "m_eq": 2, "relaxed": True}).fit()
        assert len(res.treated_units) == 2


# ----------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------

class TestInference:
    def test_blank_period_inference(self):
        df = _panel(J=10, T=18)
        res = MAREX({"df": df, "outcome": "y", "unitid": "unit", "time": "time",
                     "T0": 14, "m_eq": 2, "inference": True,
                     "blank_periods": 3, "T_post": 3}).fit()
        inf = res.globres.inference
        assert isinstance(inf, MAREXInference)
        assert 0.0 <= inf.global_p_value <= 1.0
        assert inf.ci.shape == (18, 2)
        assert np.isnan(inf.ci[:res.study.T0]).all()

    def test_no_inference_by_default(self, panel):
        res = MAREX({"df": panel, "outcome": "y", "unitid": "unit", "time": "time",
                     "T0": 10, "m_eq": 2}).fit()
        assert res.globres.inference is None


# ----------------------------------------------------------------------
# Helper-level checks
# ----------------------------------------------------------------------

class TestHelpers:
    def test_solve_design_disjoint(self):
        rng = np.random.default_rng(3)
        Y = rng.normal(0, 1, (8, 12))
        raw = solve_design(Y, T0=10, clusters=np.zeros(8, dtype=int), m_eq=2)
        w, v = raw["w_opt"][:, 0], raw["v_opt"][:, 0]
        assert np.all(w * v < 1e-8)
        assert abs(w.sum() - 1) < 1e-6 and abs(v.sum() - 1) < 1e-6

    def test_relaxed_design_discretizes(self):
        rng = np.random.default_rng(4)
        Y = rng.normal(0, 1, (8, 12))
        raw = solve_design_relaxed(Y, T0=10, clusters=np.zeros(8, dtype=int), m_eq=2)
        assert len(raw["selected_treated"][0]) == 2


class TestPlot:
    def test_plot_runs(self, panel):
        res = MAREX({"df": panel, "outcome": "y", "unitid": "unit", "time": "time",
                     "T0": 10, "m_eq": 2}).fit()
        plot_marex(res, plot_type="treatment")
        plot_marex(res, plot_type="prediction")
        _plt.close("all")
