"""Equal weight per outcome in the SCMO matching objective.

Tian, Lee & Panchenko (2026) weight their COVID application's matching
variables so that "the outcomes in the domain are equally weighted, the values
of the outcomes observed in different pretreatment periods are equally weighted
within each outcome" (Online Appendix B.3.2). Their ``fn_W`` carries that as the
diagonal metric ``V`` with entry ``1 / #pre_k`` on every column of outcome
``k``.

It matters whenever the outcomes are observed at different frequencies: without
it, a daily series contributes a hundred columns and a quarterly series four,
so the daily series decides the weights on its own.
"""

import numpy as np
import pandas as pd
import pytest

from mlsynth.estimators.scmo import SCMO
from mlsynth.utils.scmo_helpers import CONCATENATED, prepare_scmo_inputs
from mlsynth.utils.scmo_helpers.estimation import col_scale_for, outcome_column_scale

PRE_YEARS = [1960, 1961, 1962, 1963, 1964]


@pytest.fixture
def panel() -> pd.DataFrame:
    """Seven units. ``y1`` moves over time; ``p`` is a unit's own constant."""
    rng = np.random.default_rng(19)
    rows = []
    for i in range(7):
        predictor = rng.uniform(-1, 1)
        level = rng.uniform(-2, 2)
        for t in range(7):
            year = 1960 + t
            rows.append({"unit": f"u{i}", "time": year, "p": predictor,
                         "y1": level + rng.normal(), "treat": int(i == 0 and year >= 1965)})
    return pd.DataFrame(rows)


def test_outcome_column_scale_splits_a_block_evenly():
    scale = outcome_column_scale(["a@1", "a@2", "a@3", "a@4", "b"])
    np.testing.assert_allclose(scale[:4], np.sqrt(1 / 4))
    np.testing.assert_allclose(scale[4], 1.0)


def test_outcome_column_scale_is_one_when_every_outcome_has_one_column():
    np.testing.assert_allclose(outcome_column_scale(["a", "b", "c"]), 1.0)


def test_col_scale_is_none_under_the_default(panel):
    inp = prepare_scmo_inputs(panel, unitid="unit", time="time", outcome="y1",
                              spec={"year": PRE_YEARS, "vars": {"a": "y1"}},
                              treated_unit="u0", intervention_time=1965)
    assert col_scale_for(inp, CONCATENATED, "simplex", None) is None
    assert col_scale_for(inp, CONCATENATED, "simplex", None,
                         metric_weighting="outcome") is not None


def _weights(panel: pd.DataFrame, spec: dict, weighting: str) -> np.ndarray:
    return SCMO({"df": panel, "outcome": "y1", "treat": "treat", "unitid": "unit",
                 "time": "time", "spec": spec, "schemes": [CONCATENATED],
                 "metric_weighting": weighting, "display_graphs": False,
                 }).fit()._primary.weights


# One predictor read once, against the same predictor stacked at every period.
# It is constant over time, so the two specs carry the same information.
_SPEC_ONCE = {"year": PRE_YEARS,
              "vars": {"a": "y1", "p": {"column": "p", "year": 1960}}}
_SPEC_STACKED = {"year": PRE_YEARS, "vars": {"a": "y1", "p": "p"}}


def test_outcome_weighting_ignores_how_often_a_predictor_is_stacked(panel):
    """Equal weight per outcome: repeating a time-invariant predictor at every
    period leaves the fit where it was."""
    once = _weights(panel, _SPEC_ONCE, "outcome")
    stacked = _weights(panel, _SPEC_STACKED, "outcome")
    np.testing.assert_allclose(once, stacked, atol=1e-6)


def test_column_weighting_lets_the_stacking_decide(panel):
    """Equal weight per column, the default: the same repetition multiplies the
    predictor's weight by five and moves the fit."""
    once = _weights(panel, _SPEC_ONCE, "column")
    stacked = _weights(panel, _SPEC_STACKED, "column")
    assert not np.allclose(once, stacked, atol=1e-4)


def test_column_weighting_is_the_default(panel):
    explicit = _weights(panel, _SPEC_STACKED, "column")
    default = SCMO({"df": panel, "outcome": "y1", "treat": "treat", "unitid": "unit",
                    "time": "time", "spec": _SPEC_STACKED, "schemes": [CONCATENATED],
                    "display_graphs": False}).fit()._primary.weights
    np.testing.assert_allclose(explicit, default, atol=1e-10)


def test_the_two_weightings_disagree_on_a_lopsided_spec(panel):
    """With five columns of one outcome against one of another, the choice of
    metric changes the answer -- which is why the paper states its metric."""
    assert not np.allclose(_weights(panel, _SPEC_ONCE, "outcome"),
                           _weights(panel, _SPEC_ONCE, "column"), atol=1e-4)


def test_the_two_weightings_agree_when_the_counts_match(panel):
    """Both outcomes observed the same number of times: the outcome metric is
    then one number on every column, and scaling an objective uniformly leaves
    its minimizer alone."""
    np.testing.assert_allclose(_weights(panel, _SPEC_STACKED, "outcome"),
                               _weights(panel, _SPEC_STACKED, "column"), atol=1e-6)


def test_outcome_weighting_is_recorded_on_the_fit(panel):
    fit = SCMO({"df": panel, "outcome": "y1", "treat": "treat", "unitid": "unit",
                "time": "time", "spec": _SPEC_STACKED, "schemes": [CONCATENATED],
                "metric_weighting": "outcome", "display_graphs": False}).fit()._primary
    assert fit.metadata["metric_weighting"] == "outcome"


def test_unknown_metric_weighting_is_rejected(panel):
    with pytest.raises(Exception) as excinfo:
        SCMO({"df": panel, "outcome": "y1", "treat": "treat", "unitid": "unit",
              "time": "time", "spec": _SPEC_STACKED, "schemes": [CONCATENATED],
              "metric_weighting": "inverse-variance", "display_graphs": False}).fit()
    assert "metric_weighting" in str(excinfo.value)
