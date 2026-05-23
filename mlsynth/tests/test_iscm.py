"""Tests for the Imperfect Synthetic Controls (ISCM) estimator (Powell 2026).

Layered per agents/agents_tests.md:

* Layer 1 (numerical helpers): all-units weights, fit metric, WLS effect.
* Layer 2 (data utilities): panel ingestion + validation.
* Layer 3 (estimator integration): the headline claim -- identification
  when the treated unit is OUTSIDE the convex hull; convergence to truth;
  treated unit excluded by the fit metric; beats plain SCM; inference.
* Layer 4 (public API contracts): import, frozen results, config.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import ISCM
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.iscm_helpers import (
    ISCMInputs,
    ISCMResults,
    all_units_weights,
    fit_metric,
    prepare_iscm_inputs,
    residuals_and_exposure,
    weighted_att,
)


def _factor_panel(
    N: int = 8, T: int = 60, T0: int = 48, true_alpha: float = 3.0,
    noise: float = 0.05, seed: int = 0,
):
    """One-factor panel; unit 0 (treated) has the MAX loading, so it lies
    OUTSIDE the convex hull of the controls but is used as a donor by the
    next-highest controls -- ISCM's identifying mechanism."""
    rng = np.random.default_rng(seed)
    loadings = np.linspace(2.0, -1.5, N)        # unit 0 = max loading
    f = np.cumsum(rng.standard_normal(T)) * 0.3 + np.linspace(0, 2, T)
    Y = np.outer(loadings, f) + rng.standard_normal((N, T)) * noise
    D = np.zeros((N, T))
    Y[0, T0:] += true_alpha
    D[0, T0:] = 1
    rows = [{"unit": f"u{i}", "time": t, "y": Y[i, t], "D": int(D[i, t])}
            for i in range(N) for t in range(T)]
    return pd.DataFrame(rows), true_alpha


@pytest.fixture
def panel():
    return _factor_panel()


# ----------------------------------------------------------------------
# Layer 1: numerical helpers
# ----------------------------------------------------------------------

class TestHelpers:
    def test_all_units_weights_simplex(self, panel):
        df, _ = panel
        inp = prepare_iscm_inputs(df, "y", "D", "unit", "time")
        W = all_units_weights(inp.Y, inp.T0)
        assert W.shape == (inp.N, inp.N)
        np.testing.assert_allclose(np.diag(W), 0.0, atol=1e-9)   # no self-weight
        np.testing.assert_allclose(W.sum(axis=1), 1.0, atol=1e-5)  # rows -> simplex
        assert (W >= -1e-6).all()

    def test_fit_metric_range(self, panel):
        df, _ = panel
        inp = prepare_iscm_inputs(df, "y", "D", "unit", "time")
        W = all_units_weights(inp.Y, inp.T0)
        R, _ = residuals_and_exposure(inp.Y, inp.D, W)
        a = fit_metric(R, inp.Y, inp.T0)
        assert a.shape == (inp.N,)
        assert (a >= 0).all() and (a <= 1 + 1e-9).all()
        assert np.isclose(a.max(), 1.0)        # best-fitting unit -> 1

    def test_weighted_att_decomposition(self, panel):
        df, true_alpha = panel
        inp = prepare_iscm_inputs(df, "y", "D", "unit", "time")
        W = all_units_weights(inp.Y, inp.T0)
        R, E = residuals_and_exposure(inp.Y, inp.D, W)
        a = fit_metric(R, inp.Y, inp.T0)
        att, unit_att, contribution = weighted_att(R, E, a, inp.T0)
        # v_i sums to one and reconstructs the aggregate from per-unit estimates.
        np.testing.assert_allclose(contribution.sum(), 1.0, atol=1e-6)
        fin = np.isfinite(unit_att)
        recon = float(np.sum(contribution[fin] * unit_att[fin]))
        assert recon == pytest.approx(att, abs=1e-6)


# ----------------------------------------------------------------------
# Layer 2: data utilities
# ----------------------------------------------------------------------

class TestSetup:
    def test_prepare_inputs(self, panel):
        df, _ = panel
        inp = prepare_iscm_inputs(df, "y", "D", "unit", "time")
        assert isinstance(inp, ISCMInputs)
        assert inp.N == 8 and inp.T == 60 and inp.T0 == 48
        assert list(inp.treated_idx) == [0]

    def test_too_few_units_rejected(self):
        rows = [{"unit": f"u{i}", "time": t, "y": float(i + t),
                 "D": int(i == 0 and t >= 3)}
                for i in range(2) for t in range(5)]
        with pytest.raises(MlsynthDataError, match="at least 3 units"):
            prepare_iscm_inputs(pd.DataFrame(rows), "y", "D", "unit", "time")

    def test_treated_at_first_period_rejected(self, panel):
        df, _ = panel
        df = df.copy()
        df.loc[(df["unit"] == "u0"), "D"] = 1   # treated from the start
        with pytest.raises(MlsynthDataError):
            prepare_iscm_inputs(df, "y", "D", "unit", "time")


# ----------------------------------------------------------------------
# Layer 3: estimator integration
# ----------------------------------------------------------------------

class TestEstimator:
    def test_treated_outside_hull_excluded(self, panel):
        """Treated unit (max loading) is outside the hull -> tiny fit metric,
        negligible contribution to the aggregate."""
        df, _ = panel
        res = ISCM({"df": df, "outcome": "y", "treat": "D",
                    "unitid": "unit", "time": "time", "display_graphs": False, "inference": False}).fit()
        assert res.fit_metric[0] < 0.05
        assert res.contribution[0] < 0.05

    def test_identification_outside_hull(self, panel):
        """The effect is identified via control units that use the treated
        unit as a donor, despite the treated unit being outside the hull."""
        df, true_alpha = panel
        res = ISCM({"df": df, "outcome": "y", "treat": "D",
                    "unitid": "unit", "time": "time", "display_graphs": False, "inference": False}).fit()
        assert res.att == pytest.approx(true_alpha, abs=0.4)

    def test_converges_to_truth_low_noise(self):
        df, true_alpha = _factor_panel(noise=0.01, T0=100, T=120)
        res = ISCM({"df": df, "outcome": "y", "treat": "D",
                    "unitid": "unit", "time": "time", "display_graphs": False, "inference": False}).fit()
        assert res.att == pytest.approx(true_alpha, abs=0.05)

    def test_beats_plain_scm(self, panel):
        """ISCM is less biased than a plain SCM on the (outside-hull) treated."""
        from mlsynth.utils.iscm_helpers.weights import _one_unit_weights
        df, true_alpha = panel
        inp = prepare_iscm_inputs(df, "y", "D", "unit", "time")
        res = ISCM({"df": df, "outcome": "y", "treat": "D",
                    "unitid": "unit", "time": "time", "display_graphs": False, "inference": False}).fit()
        others = list(range(1, inp.N))
        w = _one_unit_weights(inp.Y[others][:, :inp.T0].T, inp.Y[0, :inp.T0])
        scm_att = float((inp.Y[0, inp.T0:] - (w @ inp.Y[others])[inp.T0:]).mean())
        assert abs(res.att - true_alpha) < abs(scm_att - true_alpha)

    def test_inference_present(self, panel):
        df, _ = panel
        res = ISCM({"df": df, "outcome": "y", "treat": "D", "unitid": "unit",
                    "time": "time", "display_graphs": False, "inference": True, "n_draws": 2000}).fit()
        assert res.inference is not None
        inf = res.inference
        assert inf.method == "ibragimov_muller"
        assert 0.0 <= inf.p_value <= 1.0
        assert inf.ci[0] <= res.att <= inf.ci[1]
        assert inf.n_contributing >= 1


# ----------------------------------------------------------------------
# Layer 4: public API contracts
# ----------------------------------------------------------------------

class TestPublicAPI:
    def test_top_level_import(self):
        from mlsynth import ISCM as _I
        assert _I is ISCM

    def test_results_frozen(self, panel):
        df, _ = panel
        res = ISCM({"df": df, "outcome": "y", "treat": "D", "unitid": "unit",
                    "time": "time", "display_graphs": False, "inference": False}).fit()
        with pytest.raises(Exception):
            res.att = 0.0

    def test_invalid_config_rejected(self, panel):
        df, _ = panel
        with pytest.raises(MlsynthConfigError):
            ISCM({"df": df, "outcome": "y", "treat": "D", "unitid": "unit",
                  "time": "time", "display_graphs": False, "alpha": 1.5})   # alpha out of (0,1)


def test_weights_results_exposed(panel):
    """ISCM exposes standardized WeightsResults (treated SC weights) plus the
    full all-units weight matrix."""
    from mlsynth.config_models import WeightsResults
    df, _ = panel
    res = ISCM({"df": df, "outcome": "y", "treat": "D", "unitid": "unit",
                "time": "time", "display_graphs": False, "inference": False}).fit()
    assert isinstance(res.weights, WeightsResults)
    assert res.unit_weight_matrix.shape == (res.inputs.N, res.inputs.N)
