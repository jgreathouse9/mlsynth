"""Tests for the MC-NNM estimator (Athey et al. 2021).

Layered per agents/agents_tests.md:

* Layer 1 (numerical helpers): the SOFT-IMPUTE engine recovers a low-rank
  matrix; nuclear-norm shrinkage; fixed-effects fitting.
* Layer 2 (data utilities): panel ingestion + mask construction.
* Layer 3 (estimator integration): recovery of a planted effect on a
  low-rank panel; Prop 99 lands in the ADH range; jackknife inference.
* Layer 4 (public API contracts): import, frozen results, config.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import MCNNM
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.mcnnm_helpers import (
    MCNNMInputs,
    MCNNMResults,
    mcnnm_cv,
    mcnnm_fit,
    prepare_mcnnm_inputs,
)
from mlsynth.utils.mcnnm_helpers.completion import _shrink


def _low_rank_panel(n_co=25, n_tr=5, T=40, T0=30, r=3, effect=2.0,
                    noise=0.05, seed=1):
    rng = np.random.default_rng(seed)
    N = n_co + n_tr
    A = rng.standard_normal((N, r)) @ rng.standard_normal((r, T))
    Y = A + rng.standard_normal((N, T)) * noise
    D = np.zeros((N, T))
    D[n_co:, T0:] = 1
    Y[n_co:, T0:] += effect
    rows = [{"unit": f"u{i}", "time": t, "y": Y[i, t], "D": int(D[i, t])}
            for i in range(N) for t in range(T)]
    return pd.DataFrame(rows), effect


# ----------------------------------------------------------------------
# Layer 1: SOFT-IMPUTE engine
# ----------------------------------------------------------------------

class TestEngine:
    def test_shrink_soft_thresholds_singular_values(self):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((10, 8))
        s_before = np.linalg.svd(A, compute_uv=False)
        B = _shrink(A, thr=s_before[0])     # threshold at the largest sv
        # All singular values shrunk to (near) zero -> B is ~ zero matrix.
        assert np.linalg.norm(B) < 1e-8

    def test_recovers_low_rank_block_missingness(self):
        rng = np.random.default_rng(0)
        N, T, r = 30, 40, 3
        A = rng.standard_normal((N, r)) @ rng.standard_normal((r, T))
        Y = A + rng.standard_normal((N, T)) * 0.02
        mask = np.ones((N, T))
        mask[25:, 30:] = 0                  # block-missing
        fit = mcnnm_cv(Y, mask, est_u=False, est_v=False, n_lam=20, n_folds=4)
        miss = mask == 0
        err = np.abs(fit["completed"][miss] - A[miss]).mean()
        assert err < 0.3                    # recovers the low-rank signal

    def test_fixed_effects_absorb_additive_structure(self):
        # Pure two-way FE (rank-2 additive) -> FE should fit it; L ~ 0.
        N, T = 12, 15
        gamma = np.arange(N) * 1.0
        delta = np.arange(T) * 0.5
        Y = gamma[:, None] + delta[None, :]
        mask = np.ones((N, T))
        fit = mcnnm_fit(Y, mask, thr=1.0, est_u=True, est_v=True)
        recon = fit["completed"]
        np.testing.assert_allclose(recon, Y, atol=1e-3)


# ----------------------------------------------------------------------
# Layer 2: data utilities
# ----------------------------------------------------------------------

class TestSetup:
    def test_prepare_inputs_mask(self):
        df, _ = _low_rank_panel()
        inp = prepare_mcnnm_inputs(df, "y", "D", "unit", "time")
        assert isinstance(inp, MCNNMInputs)
        assert inp.N == 30 and inp.T == 40 and inp.T0 == 30
        # Missing cells == treated post cells.
        assert int((1 - inp.mask).sum()) == 5 * 10
        # Observed mask is the complement of treatment.
        np.testing.assert_array_equal(inp.mask, (inp.D == 0).astype(float))

    def test_all_control_rejected(self):
        df, _ = _low_rank_panel()
        df = df.copy()
        df["D"] = 0
        with pytest.raises(MlsynthDataError):
            prepare_mcnnm_inputs(df, "y", "D", "unit", "time")


# ----------------------------------------------------------------------
# Layer 3: estimator integration
# ----------------------------------------------------------------------

class TestEstimator:
    def test_recovers_planted_effect(self):
        df, effect = _low_rank_panel()
        res = MCNNM({"df": df, "outcome": "y", "treat": "D",
                     "unitid": "unit", "time": "time",
                     "display_graphs": False}).fit()
        assert isinstance(res, MCNNMResults)
        assert res.att == pytest.approx(effect, abs=0.25)

    def test_per_period_effects(self):
        df, effect = _low_rank_panel()
        res = MCNNM({"df": df, "outcome": "y", "treat": "D",
                     "unitid": "unit", "time": "time",
                     "display_graphs": False}).fit()
        assert len(res.att_by_period) == 10
        for v in res.att_by_period.values():
            assert abs(v - effect) < 0.6

    def test_prop99_in_adh_range(self):
        """MC-NNM on the canonical Prop 99 panel lands in the ADH range."""
        df = pd.read_csv("basedata/smoking_data.csv")
        res = MCNNM({"df": df, "outcome": "cigsale", "treat": "Proposition 99",
                     "unitid": "state", "time": "year",
                     "display_graphs": False}).fit()
        # Average post-period gap negative and in the -15 .. -27 range.
        assert -27.0 < res.att < -12.0
        # California pre-treatment fit is tight (credible counterfactual).
        ca = res.inputs.unit_names.index("California")
        T0 = res.inputs.T0
        pre_rmse = float(np.sqrt(np.mean(
            (res.inputs.Y[ca, :T0] - res.counterfactual_matrix[ca, :T0]) ** 2)))
        assert pre_rmse < 3.0

    def test_exposes_factors_and_implied_weights(self):
        """MC-NNM exposes its factor decomposition and implied (non-unique)
        donor weights flagged as such.
        """
        df, _ = _low_rank_panel()
        res = MCNNM({"df": df, "outcome": "y", "treat": "D",
                     "unitid": "unit", "time": "time",
                     "display_graphs": False}).fit()
        assert res.unit_factors.shape[0] == res.inputs.N
        assert res.time_factors.shape[0] == res.inputs.T
        assert res.unit_factors.shape[1] == res.time_factors.shape[1]
        assert res.weights is not None
        assert res.weights.summary_stats["weights_are"] == "implied_non_unique"

    def test_staggered_adoption_cohort_and_event_study(self):
        """Three-cohort staggered design: overall ATT, per-cohort ATTs, and
        the event-study curve all recover the constant true effect; the
        pre-adoption event times are ~0 (no pre-trend).
        """
        rng = np.random.default_rng(0)
        N, T, r, effect = 30, 40, 3, 2.0
        Y0 = (rng.standard_normal((N, r)) @ rng.standard_normal((r, T))
              + rng.standard_normal(N)[:, None] * 0.5
              + np.linspace(0, 1, T)[None, :]
              + rng.standard_normal((N, T)) * 0.05)
        D = np.zeros((N, T))
        adopt = {**{i: 20 for i in range(5)}, **{i: 28 for i in range(5, 10)},
                 **{i: 34 for i in range(10, 15)}}
        for i, t0 in adopt.items():
            D[i, t0:] = 1
        Y = Y0 + effect * D
        df = pd.DataFrame([{"unit": f"u{i}", "time": t, "y": Y[i, t],
                            "D": int(D[i, t])}
                           for i in range(N) for t in range(T)])
        res = MCNNM({"df": df, "outcome": "y", "treat": "D", "unitid": "unit",
                     "time": "time", "display_graphs": False}).fit()
        assert res.att == pytest.approx(effect, abs=0.2)
        # Three cohorts, each recovered.
        assert len(res.cohort_att) == 3
        for v in res.cohort_att.values():
            assert abs(v - effect) < 0.25
        # Event study: post effects ~ effect, pre effects ~ 0.
        assert abs(res.event_study[0] - effect) < 0.3
        assert abs(res.event_study[5] - effect) < 0.3
        assert abs(res.event_study[-3]) < 0.2

    def test_jackknife_inference(self):
        df, _ = _low_rank_panel(n_co=12)
        res = MCNNM({"df": df, "outcome": "y", "treat": "D", "unitid": "unit",
                     "time": "time", "inference": True,
                     "display_graphs": False}).fit()
        # Raw jackknife object preserved under inference_jackknife; the
        # standardized InferenceResults is mirrored into the `inference` slot.
        assert res.inference_jackknife is not None
        assert res.inference_jackknife.method == "jackknife"
        lo, hi = res.inference_jackknife.ci
        assert lo <= res.att <= hi
        assert res.inference is not None
        assert res.att_ci == pytest.approx((lo, hi))


# ----------------------------------------------------------------------
# Layer 4: public API contracts
# ----------------------------------------------------------------------

class TestPublicAPI:
    def test_top_level_import(self):
        from mlsynth import MCNNM as _M
        assert _M is MCNNM

    def test_results_frozen(self):
        df, _ = _low_rank_panel()
        res = MCNNM({"df": df, "outcome": "y", "treat": "D",
                     "unitid": "unit", "time": "time",
                     "display_graphs": False}).fit()
        with pytest.raises(Exception):
            res.att = 0.0

    def test_invalid_config_rejected(self):
        df, _ = _low_rank_panel()
        with pytest.raises(MlsynthConfigError):
            MCNNM({"df": df, "outcome": "y", "treat": "D", "unitid": "unit",
                   "time": "time", "alpha": 1.5, "display_graphs": False})
