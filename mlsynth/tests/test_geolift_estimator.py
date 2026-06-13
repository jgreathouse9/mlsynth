import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

from mlsynth import GEOLIFT
from mlsynth.utils.geolift_helpers.marketselect.orchestration import GEOLIFTResults
from mlsynth.config_models import BaseEstimatorResults
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError


def _panel(n_units=6, T=26, seed=11):
    rng = np.random.default_rng(seed)
    base = np.arange(T) * 0.3
    post = {t: int(t >= T - 6) for t in range(T)}      # last 6 periods are post
    rows = []
    for i in range(n_units):
        series = base + rng.normal(scale=0.5, size=T) + i
        for t in range(T):
            rows.append({"unit": f"u{i}", "time": t, "Y": float(series[t]), "post": post[t]})
    return pd.DataFrame(rows)


def _cfg(**over):
    base = dict(df=_panel(), outcome="Y", unitid="unit", time="time",
                treatment_size=2, durations=[4], effect_sizes=[0.0, 0.5],
                lookback_window=2, ns=25, seed=0, display_graphs=False)
    base.update(over)
    return base


def test_geolift_only_public_method_is_fit():
    # the design surface is a single method; no realize/plot ceremony.
    assert not hasattr(GEOLIFT, "realize")
    assert not hasattr(GEOLIFT, "plot")
    assert hasattr(GEOLIFT, "fit")


def test_geolift_fit_design_only_when_no_post_col():
    res = GEOLIFT(_cfg()).fit()                          # no post_col -> design only
    assert isinstance(res, GEOLIFTResults)
    assert res.report is None                            # not realized
    assert res.search is not None and len(res.search.candidates) >= 1


def test_geolift_fit_auto_realizes_with_post_col():
    res = GEOLIFT(_cfg(post_col="post")).fit()           # post window -> realized under the hood
    if res.search.winner is None:
        pytest.skip("no detectable design in this tiny synthetic")
    assert isinstance(res.report, BaseEstimatorResults)
    assert res.report.inference.method == "conformal"
    assert res.report.time_series.intervention_time is not None


def test_geolift_weight_vector_matches_donor_dict():
    """The numpy control-weight vector is the donor_weights values, in order."""
    res = GEOLIFT(_cfg(post_col="post")).fit()
    if res.search.winner is None:
        pytest.skip("no detectable design in this tiny synthetic")
    w = res.report.weights
    vec = w.weight_vector
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (len(w.donor_weights),)
    assert np.allclose(vec, list(w.donor_weights.values()))
    # the flat accessor on the EffectResult agrees
    assert np.allclose(res.report.weight_vector, vec)


def test_geolift_design_weights_not_duplicated_after_realize():
    """After realization the design-weights front door and the report share one
    weights object (no second identical donor dict)."""
    res = GEOLIFT(_cfg(post_col="post")).fit()
    if res.search.winner is None:
        pytest.skip("no detectable design in this tiny synthetic")
    assert res.design_weights is res.report.weights


def test_geolift_display_graphs_runs_without_error():
    # display_graphs=True plots under the hood during fit (Agg -> no-op show)
    res = GEOLIFT(_cfg(display_graphs=True)).fit()
    assert isinstance(res, GEOLIFTResults)


def test_geolift_bad_config_raises():
    with pytest.raises(MlsynthConfigError):
        GEOLIFT("not a config")


def test_geolift_fit_no_winner_returns_design_only():
    # only es=0 -> nothing clears the power threshold -> no winner, no realize/plot
    res = GEOLIFT(_cfg(effect_sizes=[0.0], display_graphs=True)).fit()
    assert res.search.winner is None
    assert res.report is None


def test_geolift_fit_propagates_data_error():
    # unbalanced panel passes config validation but trips geoex_dataprep's balance
    df = _panel().iloc[1:]
    with pytest.raises(MlsynthDataError):
        GEOLIFT(_cfg(df=df)).fit()
