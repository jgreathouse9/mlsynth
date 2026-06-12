import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from mlsynth.config_models import (
    BaseEstimatorResults, EffectsResults, FitDiagnosticsResults,
    InferenceResults, TimeSeriesResults, WeightsResults,
)
from mlsynth.utils.geolift_helpers.marketselect.helpers.design import CandidateDesign
from mlsynth.utils.geolift_helpers.marketselect.orchestration import (
    GEOLIFTResults, MarketSelectSearch,
)
from mlsynth.utils.geolift_helpers.marketselect.plotter import plot_design


def _result(T=10):
    obs = np.arange(T, dtype=float) + 5.0
    cf = obs - 0.2
    cd = CandidateDesign(
        candidate=frozenset({"A", "B"}),
        weights=WeightsResults(donor_weights={"C": 0.5, "D": 0.5}),
        intercept=0.0,
        time_series=TimeSeriesResults(observed_outcome=obs, counterfactual_outcome=cf,
                                      estimated_gap=obs - cf, time_periods=np.arange(T)),
        fit_diagnostics=FitDiagnosticsResults(rmse_pre=0.2, additional_metrics={"scaled_l2": 0.05}),
        mde=0.1, power=1.0, rank=1.0,
    )
    search = MarketSelectSearch(shortlist=pd.DataFrame(), candidates=[cd], winner=cd)
    return GEOLIFTResults(selected_units=["A", "B"], search=search)


def test_plot_design_phase_single_panel():
    fig = plot_design(_result())
    assert len(fig.axes) == 1


def test_plot_post_phase_two_panels_with_conformal_band():
    T, pre = 12, 8
    obs = np.arange(T, dtype=float) + 5.0
    cf = obs - 0.1
    gap = obs - cf
    post = list(range(pre, T))
    rep = BaseEstimatorResults(
        effects=EffectsResults(att=0.1),
        time_series=TimeSeriesResults(observed_outcome=obs, counterfactual_outcome=cf,
                                      estimated_gap=gap, time_periods=np.arange(T),
                                      intervention_time=pre),
        inference=InferenceResults(p_value=0.6, method="conformal",
                                   details={"periods": post, "att": gap[pre:],
                                            "lower": gap[pre:] - 1, "upper": gap[pre:] + 1}),
    )
    fig = plot_design(_result(T=T), report=rep)
    assert len(fig.axes) == 2
    # the effect panel plots the gap
    ax2 = fig.axes[1]
    line = ax2.get_lines()[0]
    np.testing.assert_allclose(line.get_ydata(), gap)


def test_plot_design_saves_file(tmp_path):
    out = tmp_path / "design.png"
    plot_design(_result(), save_path=str(out))
    assert out.exists()


import pytest


def test_plot_design_no_winner_raises():
    search = MarketSelectSearch(shortlist=pd.DataFrame(), candidates=[], winner=None)
    result = GEOLIFTResults(selected_units=None, search=search)
    with pytest.raises(ValueError, match="no winning design"):
        plot_design(result)


def test_plot_design_show_true_is_noop_under_agg():
    plot_design(_result(), show=True)   # Agg backend -> plt.show() is a no-op
