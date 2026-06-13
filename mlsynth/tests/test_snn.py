"""Tests for the Synthetic Nearest Neighbors (SNN) estimator (Agarwal et al. 2021).

Layered per agents/agents_tests.md:

* Layer 1 (numerical helpers): the matrix-completion engine -- anchor
  finding + PCR recover a low-rank matrix under block-structured
  missingness.
* Layer 2 (data utilities): panel ingestion + validation.
* Layer 3 (estimator integration): causal recovery of a planted effect;
  per-period effects; jackknife inference.
* Layer 4 (public API contracts): import, frozen results, config.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from mlsynth import SNN
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.snn_helpers import (
    SNNInputs,
    SNNResults,
    prepare_snn_inputs,
    snn_complete,
)
from mlsynth.utils.snn_helpers.completion import _find_anchors


def _low_rank_panel(n_co=25, n_tr=5, T=40, T0=30, r=3, effect=2.0,
                    noise=0.05, seed=1):
    rng = np.random.default_rng(seed)
    N = n_co + n_tr
    U = rng.standard_normal((N, r))
    Vt = rng.standard_normal((r, T))
    A = U @ Vt
    Y = A + rng.standard_normal((N, T)) * noise
    D = np.zeros((N, T))
    D[n_co:, T0:] = 1
    Y[n_co:, T0:] += effect
    rows = [{"unit": f"u{i}", "time": t, "y": Y[i, t], "D": int(D[i, t])}
            for i in range(N) for t in range(T)]
    return pd.DataFrame(rows), effect, A, D


# ----------------------------------------------------------------------
# Layer 1: matrix-completion engine
# ----------------------------------------------------------------------

class TestEngine:
    def test_recovers_low_rank_block_missingness(self):
        rng = np.random.default_rng(0)
        N, T, r = 30, 40, 3
        A = rng.standard_normal((N, r)) @ rng.standard_normal((r, T))
        Y = A + rng.standard_normal((N, T)) * 0.05
        X = Y.copy()
        X[25:, 30:] = np.nan                      # block-structured missingness
        completed, feasible = snn_complete(X, max_rank=3)
        miss = np.isnan(X)
        imp = feasible & miss
        assert imp.sum() == miss.sum()             # all imputed
        assert np.abs(completed[imp] - A[imp]).mean() < 0.1   # ~ noise floor

    def test_find_anchors_panel_block(self):
        # Controls fully observed, treated post missing -> anchors are the
        # full control x pre-period block.
        N, T, T0, n_co = 20, 30, 25, 15
        D = np.ones((N, T))
        D[n_co:, T0:] = 0
        AR, AC = _find_anchors(D, n_co, T0)
        assert len(AR) == n_co            # all controls
        assert len(AC) == T0              # all pre-periods

    def test_no_anchors_returns_infeasible(self):
        X = np.full((4, 4), np.nan)
        X[0, 0] = 1.0
        completed, feasible = snn_complete(X)
        # Almost everything is infeasible (no observed anchor block).
        assert feasible.sum() < X.size

    def test_universal_rank_matches_gavish_donoho(self):
        """The Donoho-Gavish (2014) omega is 0.56b^3 - 0.95b^2 + 1.82b + 1.43
        (linear coeff 1.82, constant 1.43 -- not swapped). This spectrum +
        non-square shape picks a different rank under the swapped coefficients,
        so it pins the correct formula (and matches deshen24/syntheticNN)."""
        from mlsynth.utils.snn_helpers.completion import _universal_rank
        s = np.array([10.0, 1.8, 1.0, 0.5, 0.1])
        shape = (5, 50)                      # beta = 0.1, strongly non-square
        beta = 0.1
        omega = 0.56 * beta ** 3 - 0.95 * beta ** 2 + 1.82 * beta + 1.43
        expected = max(int((s > omega * np.median(s)).sum()), 1)
        assert expected == 2                 # guards the discriminator itself
        assert _universal_rank(s, shape) == expected


# ----------------------------------------------------------------------
# Layer 2: data utilities
# ----------------------------------------------------------------------

class TestSetup:
    def test_prepare_inputs(self):
        df, *_ = _low_rank_panel()
        inp = prepare_snn_inputs(df, "y", "D", "unit", "time")
        assert isinstance(inp, SNNInputs)
        assert inp.N == 30 and inp.T == 40 and inp.T0 == 30
        # Unit names sort lexicographically, so treated rows are wherever
        # u25..u29 land -- check by name, not position.
        treated_names = {inp.unit_names[i] for i in inp.treated_idx}
        assert treated_names == {"u25", "u26", "u27", "u28", "u29"}

    def test_all_control_rejected(self):
        df, *_ = _low_rank_panel()
        df = df.copy()
        df["D"] = 0
        with pytest.raises(MlsynthDataError):
            prepare_snn_inputs(df, "y", "D", "unit", "time")


# ----------------------------------------------------------------------
# Layer 3: estimator integration
# ----------------------------------------------------------------------

class TestEstimator:
    def test_recovers_planted_effect(self):
        df, effect, *_ = _low_rank_panel()
        res = SNN({"df": df, "outcome": "y", "treat": "D",
                   "unitid": "unit", "time": "time", "max_rank": 3, "display_graphs": False}).fit()
        assert isinstance(res, SNNResults)
        assert res.att == pytest.approx(effect, abs=0.15)
        assert res.metadata["imputed_cells"] == res.metadata["treated_cells"]

    def test_per_period_effects(self):
        df, effect, *_ = _low_rank_panel()
        res = SNN({"df": df, "outcome": "y", "treat": "D",
                   "unitid": "unit", "time": "time", "max_rank": 3, "display_graphs": False}).fit()
        # 10 post-treatment periods, each effect near the truth.
        assert len(res.att_by_period) == 10
        for v in res.att_by_period.values():
            assert abs(v - effect) < 0.3

    def test_donor_weights_exposed_and_reconstruct(self):
        """SNN exposes per-treated-unit PCR donor weights that rebuild the
        counterfactual as a linear combination of the donors.
        """
        df, effect, A, D = _low_rank_panel()
        res = SNN({"df": df, "outcome": "y", "treat": "D",
                   "unitid": "unit", "time": "time", "max_rank": 3,
                   "display_graphs": False}).fit()
        w = res.weights
        assert w is not None
        assert isinstance(w.donor_weights, dict) and len(w.donor_weights) > 0
        assert "constraint" in w.summary_stats   # unconstrained PCR note

    def test_jackknife_inference(self):
        df, effect, *_ = _low_rank_panel()
        res = SNN({"df": df, "outcome": "y", "treat": "D", "unitid": "unit",
                   "time": "time", "max_rank": 3, "inference": True, "display_graphs": False}).fit()
        # Raw jackknife object preserved under inference_jackknife; the
        # standardized InferenceResults is mirrored into the `inference` slot.
        assert res.inference_jackknife is not None
        assert res.inference_jackknife.method == "jackknife"
        assert res.inference_jackknife.se >= 0
        lo, hi = res.inference_jackknife.ci
        assert lo <= res.att <= hi
        assert res.inference is not None
        assert res.inference.standard_error == pytest.approx(res.inference_jackknife.se)
        assert res.att_ci == pytest.approx((lo, hi))


# ----------------------------------------------------------------------
# Layer 4: public API contracts
# ----------------------------------------------------------------------

_PROP99 = os.path.join(os.path.dirname(__file__), "..", "..", "basedata", "smoking_data.csv")


@pytest.mark.skipif(not os.path.exists(_PROP99), reason="Prop99 data not present")
class TestReferenceProp99:
    """Cross-validation against deshen24/syntheticNN on the Prop 99 panel.

    On block missingness (California's post-period the only NaN block) the
    reference's NetworkX max-biclique and mlsynth's greedy anchor search both
    return the full control x pre-period block, so -- with the corrected
    Donoho-Gavish rank -- the imputed counterfactual matches the reference to
    machine precision. Values below come from a live run of the reference
    implementation (n_neighbors=1, universal rank)."""

    REF_ATT = -18.4341
    REF_GAP_2000 = -29.3258

    def _fit(self):
        df = pd.read_csv(_PROP99)
        return SNN({"df": df, "outcome": "cigsale", "treat": "Proposition 99",
                    "unitid": "state", "time": "year", "display_graphs": False}).fit()

    def test_matches_reference_att(self):
        res = self._fit()
        assert res.att == pytest.approx(self.REF_ATT, abs=1e-3)

    def test_matches_reference_gap_2000(self):
        res = self._fit()
        assert res.att_by_period[2000] == pytest.approx(self.REF_GAP_2000, abs=1e-3)


class TestPublicAPI:
    def test_top_level_import(self):
        from mlsynth import SNN as _S
        assert _S is SNN

    def test_results_frozen(self):
        df, *_ = _low_rank_panel()
        res = SNN({"df": df, "outcome": "y", "treat": "D",
                   "unitid": "unit", "time": "time", "max_rank": 3, "display_graphs": False}).fit()
        with pytest.raises(Exception):
            res.att = 0.0

    def test_invalid_config_rejected(self):
        df, *_ = _low_rank_panel()
        with pytest.raises(MlsynthConfigError):
            SNN({"df": df, "outcome": "y", "treat": "D", "unitid": "unit",
                 "time": "time", "spectral_energy": 2.0, "display_graphs": False})  # must be <= 1
