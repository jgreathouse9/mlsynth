import numpy as np
import pandas as pd

from mlsynth.utils.geolift_helpers.marketselect.helpers.design import (
    design_fit,
    CandidateDesign,
)
from mlsynth.config_models import (
    WeightsResults,
    TimeSeriesResults,
    FitDiagnosticsResults,
)


def _wide(T=12, seed=5):
    rng = np.random.default_rng(seed)
    base = np.arange(T) * 0.3
    cols = {u: base + rng.normal(scale=1.0, size=T) + i for i, u in enumerate("ABCDE")}
    Ywide = pd.DataFrame(cols, index=pd.Index(range(T), name="time"))
    Ywide.columns.name = "unit"
    return Ywide


def test_design_fit_types_and_candidate():
    cd = design_fit(_wide(), frozenset({"A", "B"}), how="mean", augment="ridge")
    assert isinstance(cd, CandidateDesign)
    assert isinstance(cd.weights, WeightsResults)
    assert isinstance(cd.time_series, TimeSeriesResults)
    assert isinstance(cd.fit_diagnostics, FitDiagnosticsResults)
    assert isinstance(cd.intercept, float)
    assert cd.candidate == frozenset({"A", "B"})


def test_design_fit_donor_weights_are_controls_only():
    cd = design_fit(_wide(), frozenset({"A", "B"}), how="mean", augment="ridge")
    dw = cd.weights.donor_weights
    assert set(dw).issubset({"C", "D", "E"})
    assert "A" not in dw and "B" not in dw
    assert all(isinstance(v, float) for v in dw.values())


def test_design_fit_intercept_is_sibling_zero_for_ridge():
    cd = design_fit(_wide(), frozenset({"A", "B"}), augment="ridge")
    assert cd.intercept == 0.0          # sibling to weights; ridge centers internally


def test_design_fit_simplex_intercept_and_simplex_weights():
    cd = design_fit(_wide(), frozenset({"A", "B"}), augment=None)
    assert abs(sum(cd.weights.donor_weights.values()) - 1.0) < 1e-6
    assert isinstance(cd.intercept, float)
    assert cd.augment is None


def test_design_fit_time_series_reconstruction():
    Y = _wide()
    cd = design_fit(Y, frozenset({"A", "B"}), how="mean", augment="ridge")
    ts = cd.time_series
    assert ts.observed_outcome.shape == (12,)
    assert ts.counterfactual_outcome.shape == (12,)
    np.testing.assert_allclose(ts.estimated_gap,
                               ts.observed_outcome - ts.counterfactual_outcome)
    assert ts.time_periods.shape == (12,)
    assert ts.intervention_time is None


def test_design_fit_diagnostics_carry_scaled_l2_and_rmse():
    cd = design_fit(_wide(), frozenset({"A", "B"}), augment="ridge")
    assert cd.fit_diagnostics.rmse_pre is not None
    assert "scaled_l2" in cd.fit_diagnostics.additional_metrics


def test_design_fit_metric_slots_default_none():
    cd = design_fit(_wide(), frozenset({"A", "B"}), augment="ridge")
    assert cd.mde is None and cd.power is None and cd.rank is None
