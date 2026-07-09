"""Auxiliary-covariate support for PPSCM (augsynth::multisynth Sec 5.2).

Differential TDD against a live ``augsynth 0.2.0`` run on the Paglayan (2018)
collective-bargaining panel (captured in
``benchmarks/reference/ppscm_paglayan_covs/reference.out``): with covariates
``perinc_1959 + studteachratio_1959`` the reference gives nu=0.2244314,
ATT=-0.018749, global L2=0.003884, individual L2=0.043498, and a pinned
event-study path.

Covariates enter the QP as augsynth does (Sec 5.2 / multi_synth_qp.R): each
covariate is z-scored against the never-treated controls and rescaled by the
control-outcome standard deviation, then its imbalance is stacked with the
outcome imbalance in both the pooled and separate terms (normalized by the
number of covariates).
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from mlsynth import PPSCM
from mlsynth.config_models import PPSCMConfig
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.ppscm_helpers.setup import prepare_ppscm_inputs

BASEDATA = pathlib.Path(__file__).resolve().parents[2] / "basedata"
PAGLAYAN = BASEDATA / "Teachingaugsynth.scv"


def _paglayan_covs() -> pd.DataFrame:
    """Reproduce the reference.R covariate panel exactly."""
    d = pd.read_csv(PAGLAYAN)
    d = d[~d.State.isin(["DC", "WI", "AK", "HI"])]
    d = d[(d.year >= 1959) & (d.year <= 1997)].copy()
    d["cbr"] = (d.year >= d.YearCBrequired.fillna(np.inf)).astype(int)
    snap = (d[d.year == 1959][["State", "perinc", "studteachratio"]]
            .rename(columns={"perinc": "perinc_1959",
                             "studteachratio": "studteachratio_1959"}))
    return d.merge(snap, on="State")


# ---------------------------------------------------------------------------
# small synthetic staggered panel with two unit-level covariates
# ---------------------------------------------------------------------------

def _toy_panel(seed: int = 0):
    rng = np.random.default_rng(seed)
    n, T = 8, 12
    adopt = {0: 9, 1: 10}                       # two treated units, staggered
    x1 = rng.normal(size=n)                     # unit-level covariates
    x2 = rng.normal(size=n)
    f = rng.normal(size=T)
    rows = []
    for i in range(n):
        base = 5 + 0.5 * x1[i] - 0.3 * x2[i]
        for t in range(T):
            y = base + f[t] + 0.1 * rng.normal()
            treated = int(i in adopt and t >= adopt[i])
            rows.append({"unit": i, "time": t, "y": y, "D": treated,
                         "x1": float(x1[i]), "x2": float(x2[i])})
    return pd.DataFrame(rows)


# ======================================================================
# Layer 2: setup / config
# ======================================================================

class TestSetupAndConfig:
    def test_covariate_matrix_built_pre_mean(self):
        df = _toy_panel()
        inp = prepare_ppscm_inputs(df, outcome="y", treat="D",
                                   unitid="unit", time="time",
                                   covariates=["x1", "x2"])
        assert inp.Z is not None
        assert inp.Z.shape == (8, 2)            # (n_units, n_covariates)
        assert list(inp.cov_names) == ["x1", "x2"]
        # time-invariant covariate: pre-mean equals the constant value
        x1_by_unit = df.groupby("unit")["x1"].first().to_numpy()
        assert np.allclose(inp.Z[:, 0], x1_by_unit)

    def test_no_covariates_leaves_Z_none(self):
        inp = prepare_ppscm_inputs(_toy_panel(), outcome="y", treat="D",
                                   unitid="unit", time="time")
        assert inp.Z is None

    def test_missing_covariate_column_rejected(self):
        with pytest.raises((MlsynthConfigError, MlsynthDataError, KeyError)):
            prepare_ppscm_inputs(_toy_panel(), outcome="y", treat="D",
                                 unitid="unit", time="time",
                                 covariates=["nope"])

    def test_covariate_with_missing_values_rejected(self):
        df = _toy_panel()
        df.loc[df.unit == 3, "x1"] = np.nan          # one unit missing a covariate
        with pytest.raises(MlsynthDataError):
            prepare_ppscm_inputs(df, outcome="y", treat="D",
                                 unitid="unit", time="time", covariates=["x1"])

    def test_config_accepts_covariates(self):
        cfg = PPSCMConfig(df=_toy_panel(), outcome="y", treat="D",
                          unitid="unit", time="time", covariates=["x1", "x2"],
                          display_graphs=False)
        assert cfg.covariates == ["x1", "x2"]


# ======================================================================
# Layer 3: estimator integration on the toy panel
# ======================================================================

class TestToyIntegration:
    def test_fit_with_covariates_runs(self):
        res = PPSCM({"df": _toy_panel(), "outcome": "y", "treat": "D",
                     "unitid": "unit", "time": "time",
                     "covariates": ["x1", "x2"], "run_inference": False,
                     "display_graphs": False}).fit()
        assert np.isfinite(res.att)
        assert 0.0 <= res.nu <= 1.0

    def test_covariates_change_the_fit(self):
        base = {"df": _toy_panel(), "outcome": "y", "treat": "D",
                "unitid": "unit", "time": "time", "run_inference": False,
                "display_graphs": False}
        no_cov = PPSCM(base).fit()
        with_cov = PPSCM({**base, "covariates": ["x1", "x2"]}).fit()
        # balancing covariates moves the pooling and the estimate
        assert abs(with_cov.nu - no_cov.nu) > 1e-3 or abs(with_cov.att - no_cov.att) > 1e-3


# ======================================================================
# Layer 3b: differential cross-validation vs live augsynth 0.2.0
# ======================================================================

_COVS = ["perinc_1959", "studteachratio_1959"]


@pytest.fixture(scope="module")
def fit():
    return PPSCM({"df": _paglayan_covs(), "outcome": "lnppexpend",
                  "treat": "cbr", "unitid": "State", "time": "year",
                  "covariates": _COVS, "run_inference": False,
                  "display_graphs": False}).fit()


@pytest.mark.skipif(not PAGLAYAN.exists(), reason="Paglayan data absent")
class TestAugsynthReference:

    def test_nu_matches(self, fit):
        assert fit.nu == pytest.approx(0.2244314, abs=2e-3)

    def test_att_matches(self, fit):
        assert fit.att == pytest.approx(-0.018749, abs=2e-3)

    def test_global_l2_matches(self, fit):
        assert fit.design.global_l2 == pytest.approx(0.003884, abs=5e-4)

    def test_individual_l2_matches(self, fit):
        assert fit.design.ind_l2 == pytest.approx(0.043498, abs=3e-3)

    def test_event_study_path_matches(self, fit):
        ref = [-0.000262, -0.015646, 0.006937, -0.010612, -0.019425,
               -0.020914, -0.021253, -0.027610, -0.027845, -0.035497, -0.034108]
        got = np.asarray(fit.event_study.tau[:11], float)
        assert np.allclose(got, ref, atol=3e-3)
