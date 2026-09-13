"""Fitting against a restricted donor pool.

Leave-one-out robustness refits the synthetic control with one donor barred and
asks whether the answer moves (Tian, Lee & Panchenko 2026, Online Appendix
B.3.4.2). The donor leaves the optimization, not the panel: the matching matrix
is still built and standardized on every unit, so the refits are comparable to
each other and to the full fit.

Dropping the unit from the panel instead would rescale every column by a
different cross-unit SD, which changes the objective instead of the choice set.

Levels: smoke, unit invariants, edge, failure.
"""

import numpy as np
import pandas as pd
import pytest

from mlsynth.estimators.scmo import SCMO
from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.scmo_helpers import CONCATENATED, prepare_scmo_inputs
from mlsynth.utils.scmo_helpers.inference import permutation_inference

PRE_YEARS = list(range(1960, 1966))
SPEC = {"year": PRE_YEARS, "vars": {"a": "y1", "b": "y2"}}
UNITS = [f"u{i}" for i in range(6)]
DONORS = UNITS[1:]


@pytest.fixture
def panel() -> pd.DataFrame:
    rng = np.random.default_rng(23)
    factors = np.cumsum(rng.normal(size=(8, 2)), axis=0)
    rows = []
    for i, unit in enumerate(UNITS):
        load = rng.uniform(0.5, 1.5, size=2)
        for t in range(8):
            year = 1960 + t
            base = 20.0 + factors[t] @ load
            rows.append({"unit": unit, "time": year,
                         "treat": int(i == 0 and year >= 1966),
                         "y1": base + rng.normal(scale=0.2),
                         "y2": 1.1 * base + rng.normal(scale=0.2)})
    return pd.DataFrame(rows)


def _inputs(panel, donors=None):
    return prepare_scmo_inputs(panel, unitid="unit", time="time", outcome="y1",
                               spec=SPEC, treated_unit="u0", intervention_time=1966,
                               donors=donors)


def _fit(panel, donors=None):
    return SCMO({"df": panel, "outcome": "y1", "treat": "treat", "unitid": "unit",
                 "time": "time", "spec": SPEC, "schemes": [CONCATENATED],
                 "donors": donors, "display_graphs": False}).fit()._primary


# --------------------------------------------------------------------------- smoke

def test_a_restricted_pool_is_the_only_one_weighted(panel):
    fit = _fit(panel, donors=["u1", "u2", "u3"])
    assert set(fit.donor_weights) == {"u1", "u2", "u3"}
    assert abs(float(fit.weights.sum()) - 1.0) < 1e-6


# ------------------------------------------------------------------ unit invariants

def test_the_matching_matrix_does_not_notice(panel):
    """The restriction is a choice set, not a change of the data: the matrix and
    its column scaling are built on every unit either way."""
    full, restricted = _inputs(panel), _inputs(panel, donors=["u1", "u2"])
    np.testing.assert_allclose(full.Z, restricted.Z, atol=1e-12)
    assert full.Y.shape == restricted.Y.shape
    assert list(full.unit_index.labels) == list(restricted.unit_index.labels)


def test_naming_every_donor_is_the_default(panel):
    np.testing.assert_allclose(_fit(panel).weights, _fit(panel, donors=DONORS).weights,
                               atol=1e-8)


def test_dropping_a_donor_that_carried_weight_moves_the_fit(panel):
    base = _fit(panel)
    carrier = max(base.donor_weights, key=lambda k: base.donor_weights[k])
    without = _fit(panel, donors=[d for d in DONORS if d != carrier])
    assert carrier not in without.donor_weights
    # The pools have different sizes, so the comparable object is what the
    # weights produce: the counterfactual has to move.
    assert not np.allclose(base.counterfactual, without.counterfactual, atol=1e-4)


def test_dropping_a_donor_that_carried_nothing_leaves_it_alone(panel):
    """A donor at zero weight is not in the solution, so barring it cannot move
    the counterfactual."""
    base = _fit(panel)
    idle = [d for d, w in base.donor_weights.items() if abs(w) < 1e-6]
    if not idle:                                     # pragma: no cover - fixture has some
        pytest.skip("every donor carries weight on this panel")
    without = _fit(panel, donors=[d for d in DONORS if d != idle[0]])
    np.testing.assert_allclose(base.counterfactual, without.counterfactual, atol=1e-4)


def test_the_permutation_test_draws_from_the_same_pool(panel):
    """A placebo unit takes the treated seat and matches on the pool it is left
    with, so the ranking stays inside the restricted universe."""
    res = permutation_inference(_inputs(panel, donors=["u1", "u2", "u3"]), CONCATENATED)
    assert res.ratios.shape == (4,)                  # the treated unit and its pool
    assert 0.0 < res.p_value <= 1.0
    assert res.p_value * 4 == pytest.approx(round(res.p_value * 4))


# ----------------------------------------------------------------------- edge

def test_a_single_donor_is_allowed(panel):
    fit = _fit(panel, donors=["u1"])
    assert fit.donor_weights == {"u1": 1.0}


# ----------------------------------------------------------------------- failure

def test_an_unknown_donor_raises(panel):
    with pytest.raises(MlsynthDataError, match="atlantis"):
        _fit(panel, donors=["u1", "atlantis"])


def test_naming_the_treated_unit_raises(panel):
    with pytest.raises(MlsynthDataError, match="treated"):
        _fit(panel, donors=["u0", "u1"])


def test_an_empty_pool_raises(panel):
    with pytest.raises(MlsynthDataError, match="at least one donor"):
        _fit(panel, donors=[])
