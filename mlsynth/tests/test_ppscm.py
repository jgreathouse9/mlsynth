"""Tests for the Partially Pooled SCM estimator.

Reference: Ben-Michael, Feller & Rothstein (2022), *JRSS-B* 84(2):351-381.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import PPSCM
from mlsynth.config_models import PPSCMConfig
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.ppscm_helpers.imbalance import (
    compute_q_pool,
    compute_q_sep,
    residuals,
)
from mlsynth.utils.ppscm_helpers.optimization import solve_ppscm
from mlsynth.utils.ppscm_helpers.setup import prepare_ppscm_inputs
from mlsynth.utils.ppscm_helpers.structures import (
    PPSCMDesign,
    PPSCMEventStudy,
    PPSCMInference,
    PPSCMInputs,
    PPSCMResults,
)


def _staggered_panel(
    *, seed: int = 0,
    adoption_offsets=(10, 15, 20),
    N_donors: int = 8,
    T: int = 30,
    true_effect: float = -3.0,
    noise: float = 0.4,
) -> pd.DataFrame:
    """Staggered panel from the paper's linear factor DGP."""
    rng = np.random.default_rng(seed)
    factors = rng.standard_normal((T, 2))
    loadings_donors = rng.standard_normal((N_donors, 2)) * 0.5
    loadings_treated = loadings_donors.mean(axis=0)
    records = []
    for j, T_j in enumerate(adoption_offsets):
        base_load = loadings_treated + 0.1 * rng.standard_normal(2)
        series = factors @ base_load + rng.standard_normal(T) * noise
        series[T_j:] += true_effect
        for t in range(T):
            records.append({
                "unit": f"treated_{j}", "year": 2000 + t,
                "y": float(series[t]), "tr": int(t >= T_j),
            })
    for d in range(N_donors):
        series = factors @ loadings_donors[d] + rng.standard_normal(T) * noise
        for t in range(T):
            records.append({
                "unit": f"d_{d}", "year": 2000 + t,
                "y": float(series[t]), "tr": 0,
            })
    return pd.DataFrame(records)


@pytest.fixture(scope="module")
def staggered_panel() -> pd.DataFrame:
    return _staggered_panel()


# ---------------------------------------------------------------------------
# Layer 1: imbalance helpers
# ---------------------------------------------------------------------------

class TestImbalance:
    def test_residuals_shape(self):
        rng = np.random.default_rng(0)
        L, N, J = 5, 4, 3
        Yt = rng.standard_normal((L, J))
        Yd = rng.standard_normal((L, N, J))
        Gamma = np.full((N, J), 1.0 / N)
        r = residuals(Yt, Yd, Gamma)
        assert r.shape == (L, J)

    def test_q_sep_nonnegative(self):
        rng = np.random.default_rng(1)
        Yt = rng.standard_normal((6, 4))
        Yd = rng.standard_normal((6, 5, 4))
        Gamma = rng.uniform(size=(5, 4))
        Gamma /= Gamma.sum(axis=0, keepdims=True)
        assert compute_q_sep(Yt, Yd, Gamma) >= 0
        assert compute_q_pool(Yt, Yd, Gamma) >= 0

    def test_perfect_fit_yields_zero(self):
        # If Y_treated_pre = Y_donors_pre @ Gamma exactly, both imbalances are 0.
        L, N, J = 5, 3, 2
        rng = np.random.default_rng(2)
        Yd = rng.standard_normal((L, N, J))
        Gamma = rng.uniform(size=(N, J))
        Gamma /= Gamma.sum(axis=0, keepdims=True)
        Yt = np.einsum("lij,ij->lj", Yd, Gamma)
        assert compute_q_sep(Yt, Yd, Gamma) == pytest.approx(0.0, abs=1e-10)
        assert compute_q_pool(Yt, Yd, Gamma) == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Layer 2: setup
# ---------------------------------------------------------------------------

class TestSetup:
    def test_default_L_and_K(self, staggered_panel):
        inputs = prepare_ppscm_inputs(
            df=staggered_panel, outcome="y", treat="tr",
            unitid="unit", time="year",
        )
        assert inputs.J == 3
        assert inputs.N == 8
        # Earliest adoption at offset 10 (1-based period 11) -> L = 10.
        assert inputs.L == 10
        # Latest adoption at offset 20 (1-based period 21), T = 30 -> K = 9.
        assert inputs.K == 9
        assert inputs.Y_treated_pre.shape == (inputs.L, inputs.J)
        assert inputs.Y_donors_pre.shape == (inputs.L, inputs.N, inputs.J)
        assert inputs.Y_treated_post.shape == (inputs.K + 1, inputs.J)
        assert inputs.Y_donors_post.shape == (inputs.K + 1, inputs.N, inputs.J)

    def test_no_treated_unit_rejected(self, staggered_panel):
        df = staggered_panel.copy()
        df["tr"] = 0
        with pytest.raises(MlsynthDataError):
            prepare_ppscm_inputs(
                df=df, outcome="y", treat="tr",
                unitid="unit", time="year",
            )

    def test_excessive_L_rejected(self, staggered_panel):
        # Requesting L = 50 but no treated unit has 50 pre-periods.
        with pytest.raises(MlsynthDataError):
            prepare_ppscm_inputs(
                df=staggered_panel, outcome="y", treat="tr",
                unitid="unit", time="year",
                L=50,
            )


# ---------------------------------------------------------------------------
# Layer 3: integration
# ---------------------------------------------------------------------------

class TestSyntheticRecovery:
    """PPSCM recovers the true ATT under the paper's factor DGP."""

    def test_auto_nu_recovers_true_effect(self, staggered_panel):
        res = PPSCM({
            "df": staggered_panel, "outcome": "y", "treat": "tr",
            "unitid": "unit", "time": "year",
            "display_graphs": False,
        }).fit()
        assert res.inference.att == pytest.approx(-3.0, abs=0.4)
        # Auto-nu should pick a genuinely intermediate value, not 0 or 1.
        assert 0.0 < res.design.nu_used < 1.0

    def test_separate_scm_at_nu_zero(self, staggered_panel):
        res = PPSCM({
            "df": staggered_panel, "outcome": "y", "treat": "tr",
            "unitid": "unit", "time": "year",
            "nu": 0.0, "run_inference": False, "display_graphs": False,
        }).fit()
        # At nu = 0, q_sep should match its baseline.
        assert res.design.q_sep == pytest.approx(
            res.design.q_sep_baseline, abs=1e-6
        )
        # The ATT is still well-defined.
        assert res.inference.att == pytest.approx(-3.0, abs=0.5)

    def test_pooled_scm_at_nu_one(self, staggered_panel):
        res = PPSCM({
            "df": staggered_panel, "outcome": "y", "treat": "tr",
            "unitid": "unit", "time": "year",
            "nu": 1.0, "run_inference": False, "display_graphs": False,
        }).fit()
        # At nu = 1, q_pool should be no worse than its baseline.
        assert res.design.q_pool <= res.design.q_pool_baseline + 1e-6

    def test_simplex_constraint_satisfied(self, staggered_panel):
        res = PPSCM({
            "df": staggered_panel, "outcome": "y", "treat": "tr",
            "unitid": "unit", "time": "year",
            "nu": 0.5, "run_inference": False, "display_graphs": False,
        }).fit()
        col_sums = res.design.Gamma.sum(axis=0)
        assert np.allclose(col_sums, 1.0, atol=1e-6)
        assert (res.design.Gamma >= -1e-8).all()


class TestJackknife:
    def test_jackknife_yields_finite_se(self, staggered_panel):
        res = PPSCM({
            "df": staggered_panel, "outcome": "y", "treat": "tr",
            "unitid": "unit", "time": "year",
            "run_inference": True, "display_graphs": False,
        }).fit()
        assert res.inference.method == "jackknife"
        assert np.isfinite(res.inference.se)
        assert res.inference.se >= 0
        lo, hi = res.inference.ci
        assert lo <= res.inference.att <= hi

    def test_inference_can_be_disabled(self, staggered_panel):
        res = PPSCM({
            "df": staggered_panel, "outcome": "y", "treat": "tr",
            "unitid": "unit", "time": "year",
            "run_inference": False, "display_graphs": False,
        }).fit()
        assert res.inference.method == "none"
        assert np.isnan(res.inference.se)


# ---------------------------------------------------------------------------
# Layer 4: public API
# ---------------------------------------------------------------------------

class TestPublicAPI:
    def test_import(self):
        from mlsynth import PPSCM as Imported  # noqa: F401
        assert Imported is PPSCM

    def test_results_object_types(self, staggered_panel):
        res = PPSCM({
            "df": staggered_panel, "outcome": "y", "treat": "tr",
            "unitid": "unit", "time": "year",
            "run_inference": False, "display_graphs": False,
        }).fit()
        assert isinstance(res, PPSCMResults)
        assert isinstance(res.inputs, PPSCMInputs)
        assert isinstance(res.design, PPSCMDesign)
        assert isinstance(res.event_study, PPSCMEventStudy)
        assert isinstance(res.inference, PPSCMInference)

    def test_donor_weights_aligned(self, staggered_panel):
        res = PPSCM({
            "df": staggered_panel, "outcome": "y", "treat": "tr",
            "unitid": "unit", "time": "year",
            "run_inference": False, "display_graphs": False,
        }).fit()
        # One outer entry per treated unit.
        assert len(res.donor_weights) == res.inputs.J
        for treated_name, inner in res.donor_weights.items():
            assert set(inner.keys()) == set(
                str(n) for n in res.inputs.donor_names
            )

    def test_dict_vs_config_object(self, staggered_panel):
        cfg_dict = {
            "df": staggered_panel, "outcome": "y", "treat": "tr",
            "unitid": "unit", "time": "year",
            "nu": 0.5, "run_inference": False, "display_graphs": False,
        }
        cfg_obj = PPSCMConfig(**cfg_dict)
        r1 = PPSCM(cfg_dict).fit()
        r2 = PPSCM(cfg_obj).fit()
        assert r1.inference.att == pytest.approx(r2.inference.att)

    def test_event_study_shapes(self, staggered_panel):
        res = PPSCM({
            "df": staggered_panel, "outcome": "y", "treat": "tr",
            "unitid": "unit", "time": "year",
            "run_inference": True, "display_graphs": False,
        }).fit()
        K_plus_1 = res.inputs.K + 1
        assert res.event_study.tau.shape == (K_plus_1,)
        assert res.event_study.ci.shape == (K_plus_1, 2)
