"""Demeaned matching and time-invariant predictors in the SCMO matching matrix.

Two pieces of Tian, Lee & Panchenko (2026) Online Appendix B.1.1 ("Adjusting for
differences in levels"), which the multi-outcome schemes need to reproduce the
appendix simulation:

* ``demean=True`` centers each outcome's block of matching columns on that
  unit's own pre-treatment mean before the columns are standardized, so a stable
  level difference between units stops driving the weights. Until this existed,
  ``demean`` only shifted the counterfactual, which left the concatenated and
  averaged schemes matching on levels.
* a spec rule may pin its own period, so a time-invariant predictor enters the
  matching matrix once instead of being repeated at every stacked period.
"""

import numpy as np
import pandas as pd
import pytest

from mlsynth.estimators.scmo import SCMO
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.fast_scm_helpers.structure import IndexSet
from mlsynth.utils.scmo_helpers import CONCATENATED, build_matching_matrix, prepare_scmo_inputs


PRE_YEARS = [1960, 1961, 1962, 1963, 1964]
SPEC = {"year": PRE_YEARS, "vars": {"a": "y1", "b": "y2"}}


@pytest.fixture
def panel() -> pd.DataFrame:
    """Six units, two outcomes, five pre-periods and two post; unit u0 treated.

    Every unit carries a time-invariant predictor ``p`` and a level offset, so
    the level-shift invariance of demeaned matching is visible.
    """
    rng = np.random.default_rng(11)
    rows = []
    for i in range(6):
        level = 10.0 * i
        predictor = rng.uniform(-1, 1)
        for t in range(7):
            year = 1960 + t
            rows.append({
                "unit": f"u{i}", "time": year, "p": predictor,
                "y1": level + rng.normal(), "y2": level + rng.normal(),
                "treat": int(i == 0 and year >= 1965),
            })
    return pd.DataFrame(rows)


def _build(panel: pd.DataFrame, spec: dict, demean: bool):
    unit_index = IndexSet.from_labels(list(pd.unique(panel["unit"])))
    return build_matching_matrix(panel, unitid="unit", time="time", spec=spec,
                                 unit_index=unit_index, demean=demean)


def _shift_levels(panel: pd.DataFrame) -> pd.DataFrame:
    """Add a unit-specific constant to both outcomes in every period."""
    out = panel.copy()
    bump = {f"u{i}": 100.0 * (i + 1) for i in range(6)}
    add = out["unit"].map(bump)
    out["y1"] = out["y1"] + add
    out["y2"] = out["y2"] + add
    return out


# --- demeaned matching -----------------------------------------------------

def test_demean_is_invariant_to_unit_level_shifts(panel):
    """The point of B.1.1: a stable level difference leaves the matrix alone."""
    Z, _labels, _cp = _build(panel, SPEC, demean=True)
    Z_shift, _l, _c = _build(_shift_levels(panel), SPEC, demean=True)
    np.testing.assert_allclose(Z, Z_shift, atol=1e-10)


def test_levels_matching_is_not_invariant_to_shifts(panel):
    """Without demeaning the same shift moves the matrix (the behavior it fixes)."""
    Z, _labels, _cp = _build(panel, SPEC, demean=False)
    Z_shift, _l, _c = _build(_shift_levels(panel), SPEC, demean=False)
    assert not np.allclose(Z, Z_shift, atol=1e-6)


def _expected_block(panel: pd.DataFrame, column: str, center_first: bool) -> np.ndarray:
    """The appendix's own construction: center each unit over the block, then
    divide each column by its cross-unit SD (or the two steps swapped)."""
    units = list(pd.unique(panel["unit"]))
    raw = np.column_stack([
        panel[panel["time"] == yr].set_index("unit")[column].reindex(units).to_numpy(float)
        for yr in PRE_YEARS])
    if center_first:
        raw = raw - raw.mean(axis=1, keepdims=True)
        return raw / raw.std(axis=0, ddof=1)
    scaled = raw / raw.std(axis=0, ddof=1)
    return scaled - scaled.mean(axis=1, keepdims=True)


def test_demean_centers_each_block_before_standardizing(panel):
    """Each outcome block is centered per unit and the columns are scaled after,
    the order Tian-Lee-Panchenko's own code uses. Scaling first gives a
    different matrix, so the order is pinned, not incidental."""
    Z, labels, _cp = _build(panel, SPEC, demean=True)
    for var, column in (("a", "y1"), ("b", "y2")):
        cols = [j for j, l in enumerate(labels) if str(l).split("@")[0] == var]
        assert len(cols) == len(PRE_YEARS)
        np.testing.assert_allclose(
            Z[:, cols], _expected_block(panel, column, center_first=True), atol=1e-10)
        assert not np.allclose(
            Z[:, cols], _expected_block(panel, column, center_first=False), atol=1e-6)


def test_demean_leaves_a_single_column_block_alone(panel):
    """A block with one column has no within-block mean to remove; centering it
    would zero it out, so it is passed through (the observed predictors of the
    appendix simulation are matched in levels)."""
    spec = {"year": 1960, "vars": {"a": "y1", "p": "p"}}
    Z_plain, labels, _cp = _build(panel, spec, demean=False)
    Z_demean, labels_d, _c = _build(panel, spec, demean=True)
    assert labels == labels_d
    np.testing.assert_allclose(Z_plain, Z_demean, atol=1e-12)


def test_demean_flows_from_the_config(panel):
    """``demean=True`` reaches the matching matrix, not only the counterfactual."""
    cfg = {"df": panel, "outcome": "y1", "treat": "treat", "unitid": "unit",
           "time": "time", "spec": SPEC, "schemes": [CONCATENATED],
           "display_graphs": False}
    plain = SCMO({**cfg}).fit()._primary
    demeaned = SCMO({**cfg, "demean": True}).fit()._primary
    assert not np.allclose(plain.weights, demeaned.weights, atol=1e-6)
    assert demeaned.metadata["demean"] is True


def test_demeaned_fit_is_invariant_to_unit_level_shifts(panel):
    """End to end: demeaned matching plus the intercept-shifted counterfactual
    leave both the weights and the ATT unchanged under a level shift."""
    cfg = {"outcome": "y1", "treat": "treat", "unitid": "unit", "time": "time",
           "spec": SPEC, "schemes": [CONCATENATED], "demean": True,
           "display_graphs": False}
    base = SCMO({**cfg, "df": panel}).fit()._primary
    shifted = SCMO({**cfg, "df": _shift_levels(panel)}).fit()._primary
    np.testing.assert_allclose(base.weights, shifted.weights, atol=1e-6)
    assert abs(base.att - shifted.att) < 1e-6


# --- period-pinned (time-invariant) predictors -----------------------------

def test_pinned_rule_enters_once(panel):
    """A rule that names its own period is read there and stacked once."""
    spec = {"year": PRE_YEARS,
            "vars": {"a": "y1", "p": {"column": "p", "year": 1960}}}
    Z, labels, col_period = _build(panel, spec, demean=False)
    assert labels.count("p") == 1
    assert len([l for l in labels if str(l).startswith("a@")]) == len(PRE_YEARS)
    assert Z.shape == (6, len(PRE_YEARS) + 1)
    assert col_period[labels.index("p")] == 1960


def test_pinned_rule_reads_its_own_period(panel):
    """The pinned column is that period's values, up to the column scaling."""
    spec = {"year": PRE_YEARS,
            "vars": {"a": "y1", "p": {"column": "p", "year": 1960}}}
    Z, labels, _cp = _build(panel, spec, demean=False)
    units = list(pd.unique(panel["unit"]))
    raw = (panel[panel["time"] == 1960].set_index("unit")["p"]
           .reindex(units).to_numpy(dtype=float))
    col = Z[:, labels.index("p")]
    np.testing.assert_allclose(col * raw.std(ddof=1), raw, atol=1e-10)


def test_pinned_rule_supports_an_operation(panel):
    """``op`` still applies to a pinned rule."""
    spec = {"year": PRE_YEARS,
            "vars": {"a": "y1", "s": {"column": "y2", "op": "level", "year": 1961}}}
    _Z, labels, col_period = _build(panel, spec, demean=False)
    assert col_period[labels.index("s")] == 1961


def test_pinned_rule_without_a_column_raises(panel):
    spec = {"year": PRE_YEARS, "vars": {"a": "y1", "p": {"year": 1960}}}
    with pytest.raises(MlsynthConfigError, match="column"):
        _build(panel, spec, demean=False)


def test_pinned_rule_with_an_absent_period_raises(panel):
    spec = {"year": PRE_YEARS, "vars": {"p": {"column": "p", "year": 1999}}}
    with pytest.raises(MlsynthConfigError, match="1999"):
        _build(panel, spec, demean=False)


def test_pinned_rule_reaches_prepare_inputs(panel):
    """The predictor survives the DataFrame -> NumPy boundary with its label."""
    spec = {"year": PRE_YEARS,
            "vars": {"a": "y1", "p": {"column": "p", "year": 1960}}}
    inp = prepare_scmo_inputs(panel, unitid="unit", time="time", outcome="y1",
                              spec=spec, treated_unit="u0", intervention_time=1965)
    assert "p" in list(inp.predictor_labels)
    assert inp.Z.shape[1] == len(PRE_YEARS) + 1


# --- outcomes observed on part of the panel's period axis -------------------

def _ragged(panel: pd.DataFrame) -> pd.DataFrame:
    """``y2`` is observed at every period; ``y1`` only at even ones -- the
    multi-frequency case of the COVID application, where a daily series and a
    quarterly series share one panel."""
    out = panel.copy()
    out.loc[out["time"] % 2 == 1, "y1"] = np.nan
    return out


def test_a_period_no_unit_observes_is_dropped_from_the_outcome_panel(panel):
    """The matching matrix still spans every period; the outcome panel covers
    the periods that outcome is observed in."""
    inp = prepare_scmo_inputs(_ragged(panel), unitid="unit", time="time",
                              outcome="y1", spec=SPEC, treated_unit="u0",
                              intervention_time=1965)
    assert inp.Y.shape == (6, 4)                       # 1960, 1962, 1964, 1966
    assert list(inp.time_index.labels) == [1960, 1962, 1964, 1966]
    assert inp.T0 == 3
    # The matching matrix keeps each outcome's own observed periods: all five
    # for y2, the three even ones for y1 (the rest are missing for every unit
    # and drop out as incomplete columns).
    assert inp.Z.shape == (6, 3 + len(PRE_YEARS))
    assert [l for l in inp.predictor_labels if str(l).startswith("a@")] == \
        ["a@1960", "a@1962", "a@1964"]
    assert inp.metadata["dropped_outcome_periods"] == [1961, 1963, 1965]


def test_a_partially_observed_period_still_raises(panel):
    """One unit missing where others are not is a broken panel, not a frequency
    difference, and the fit says so."""
    broken = panel.copy()
    broken.loc[(broken["unit"] == "u3") & (broken["time"] == 1962), "y1"] = np.nan
    with pytest.raises(MlsynthDataError, match="complete"):
        prepare_scmo_inputs(broken, unitid="unit", time="time", outcome="y1",
                            spec=SPEC, treated_unit="u0", intervention_time=1965)


def test_two_outcomes_on_one_panel_share_their_weights(panel):
    """Reading a different outcome off the same domain does not move the
    synthetic control: the weights come from the matching matrix, which both
    fits share."""
    df = _ragged(panel)
    cfg = {"df": df, "treat": "treat", "unitid": "unit", "time": "time",
           "spec": SPEC, "schemes": [CONCATENATED], "display_graphs": False}
    sparse = SCMO({**cfg, "outcome": "y1"}).fit()._primary
    dense = SCMO({**cfg, "outcome": "y2"}).fit()._primary
    np.testing.assert_allclose(sparse.weights, dense.weights, atol=1e-6)
    assert sparse.counterfactual.shape != dense.counterfactual.shape
