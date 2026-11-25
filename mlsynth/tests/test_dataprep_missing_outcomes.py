"""A panel with a missing outcome is refused at ingestion.

``dataprep`` and ``geoex_dataprep`` pivot the outcome into the wide matrix every estimator works from. A missing
(NaN) or non-finite outcome cell used to pass through: estimators with their own check (CAST, CFM, ClusterSC, DSC,
FMA, FSC) refused it in their setup, and the rest computed with it. FDID raised on a missing treated outcome only
through a numpy error inside its search, and on a missing donor outcome returned a result, its forward search ending
by re-selecting a chosen donor at an R^2 of minus infinity. mlsynth stops when outcome values are missing, so the
refusal sits where every estimator's data comes in.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.datautils import dataprep, geoex_dataprep


def _panel(n_units=4, n_periods=6, treated=("a",), start=4):
    rows = [{"id": u, "time": t, "y": float(10 * i + t) + 0.5 * t * t,
             "d": int(u in treated and t >= start)}
            for i, u in enumerate(chr(97 + j) for j in range(n_units))
            for t in range(1, n_periods + 1)]
    return pd.DataFrame(rows)


def _prep(df):
    return dataprep(df, unit_id_column_name="id", time_period_column_name="time",
                    outcome_column_name="y", treatment_indicator_column_name="d")


def _geo(df):
    return geoex_dataprep(df, unit_id_column_name="id", time_period_column_name="time", outcome_column_name="y")


# --- smoke: a fully observed panel is unchanged ---

def test_a_fully_observed_panel_ingests():
    prep = _prep(_panel())
    assert prep["Ywide"].shape == (6, 4) and np.isfinite(prep["Ywide"].to_numpy(dtype=float)).all()
    assert _geo(_panel())["Ywide"].shape == (6, 4)


# --- failure: every placement of a missing outcome is refused, and reported ---

@pytest.mark.parametrize("unit, time", [("a", 2), ("a", 5), ("c", 1), ("d", 6)])
def test_a_missing_outcome_is_refused(unit, time):
    """Treated or donor, pre-period or post: one missing cell stops ingestion."""
    df = _panel()
    df.loc[(df.id == unit) & (df.time == time), "y"] = np.nan
    with pytest.raises(MlsynthDataError, match="Missing outcome"):
        _prep(df)


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_a_non_finite_outcome_is_refused(value):
    df = _panel()
    df.loc[3, "y"] = value
    with pytest.raises(MlsynthDataError, match="not finite"):
        _prep(df)


def test_the_message_names_the_column_the_count_and_the_first_cell():
    df = _panel()
    df.loc[(df.id == "c") & (df.time == 2), "y"] = np.nan
    df.loc[(df.id == "b") & (df.time == 5), "y"] = np.nan
    with pytest.raises(MlsynthDataError) as excinfo:
        _prep(df)
    msg = str(excinfo.value)
    assert "'y'" in msg and "2 of 24" in msg
    assert "unit 'c', time 2" in msg            # the first missing cell in time order, then unit order


def test_the_message_counts_absent_rows_nan_and_infinite_cells_apart():
    """An absent unit-period row and a NaN in a present row leave the same missing cell after the pivot; the
    message tells them apart, since the first is an unbalanced panel and the second a missing value."""
    df = _panel()
    df = df[~((df.id == "d") & (df.time == 6))].copy()
    df.loc[(df.id == "b") & (df.time == 3), "y"] = np.nan
    df.loc[(df.id == "a") & (df.time == 4), "y"] = np.inf
    with pytest.raises(MlsynthDataError) as excinfo:
        _prep(df)
    msg = str(excinfo.value)
    assert "3 of 24" in msg
    assert "1 absent unit-period row(s), so the panel is not balanced, 1 NaN, 1 infinite" in msg
    assert "unit 'b', time 3" in msg


def test_only_the_kinds_present_are_counted():
    df = _panel()
    df.loc[(df.id == "c") & (df.time == 1), "y"] = -np.inf
    with pytest.raises(MlsynthDataError) as excinfo:
        _prep(df)
    msg = str(excinfo.value)
    assert "(1 infinite)" in msg and "NaN" not in msg and "absent" not in msg


def test_an_object_column_of_numbers_with_a_none_is_refused():
    """``None`` in an object-dtype outcome column is a missing value once the column converts to float."""
    df = _panel()
    df["y"] = df["y"].astype(object)
    df.loc[5, "y"] = None
    with pytest.raises(MlsynthDataError, match="Missing outcome"):
        _prep(df)


def test_an_outcome_that_is_not_numeric_is_not_this_checks_concern():
    """A text outcome does not convert to float; estimators that require a numeric outcome report that
    themselves, so ``dataprep`` passes it through unchanged."""
    df = _panel()
    df["y"] = [f"v{i}" for i in range(len(df))]
    assert _prep(df)["Ywide"].shape == (6, 4)


def test_proximal_stops_on_a_missing_outcome():
    """PROXIMAL used to fit with a NaN outcome cell; a missing proxy value still does not stop it."""
    from mlsynth import PROXIMAL
    rng = np.random.default_rng(0)
    rows = [{"id": u, "time": t, "y": 5.0 + u + 0.3 * t + rng.normal(), "p": 1.0 + u + rng.normal(),
             "d": int(u == 0 and t >= 16)} for u in range(5) for t in range(20)]
    df = pd.DataFrame(rows)
    df.loc[(df.id == 2) & (df.time == 0), "y"] = np.nan
    with pytest.raises(MlsynthDataError, match="Missing outcome"):
        PROXIMAL(dict(df=df, unitid="id", time="time", outcome="y", treat="d", methods=["PI"],
                      donors=[1, 2, 3, 4], vars={"donorproxies": ["p"]}, display_graphs=False)).fit()


def test_the_cohort_branch_refuses_too():
    """Two treated units (cohort mode) take the other branch of dataprep; it refuses the same way."""
    df = _panel(n_units=5, treated=("a", "b"))
    df.loc[(df.id == "e") & (df.time == 3), "y"] = np.nan
    with pytest.raises(MlsynthDataError, match="Missing outcome"):
        _prep(df)


def test_geoex_dataprep_refuses_too():
    """The treatment-agnostic ingestion of the design tools pivots the same outcome and refuses the same way."""
    df = _panel()
    df.loc[7, "y"] = np.nan
    with pytest.raises(MlsynthDataError, match="Missing outcome"):
        _geo(df)


def test_fdid_stops_on_a_donors_missing_outcome():
    """End to end: FDID used to return a result here, its search ending on a re-selected donor at R^2 = -inf."""
    from mlsynth import FDID
    df = _panel(n_units=6, n_periods=12, start=9)
    df.loc[(df.id == "c") & (df.time == 3), "y"] = np.nan
    with pytest.raises(MlsynthDataError, match="Missing outcome"):
        FDID(dict(df=df, unitid="id", time="time", outcome="y", treat="d", display_graphs=False)).fit()


# --- edge: what is not an outcome is not checked here ---

def test_a_missing_treatment_value_is_not_an_outcome():
    """A missing treatment value has its own refusal (test_dataprep_missing_treatment.py); it is not counted
    against the outcome column."""
    df = _panel()
    df.loc[(df.id == "b") & (df.time == 2), "d"] = np.nan
    with pytest.raises(MlsynthDataError) as excinfo:
        _prep(df)
    assert "Missing treatment values" in str(excinfo.value)
    assert "Missing outcome" not in str(excinfo.value)


def test_a_missing_value_in_another_column_is_not_checked():
    df = _panel()
    df["x"] = 1.0
    df.loc[2, "x"] = np.nan
    assert _prep(df)["Ywide"].shape == (6, 4)


# --- property: the refusal is exactly the presence of a non-finite outcome ---

@settings(max_examples=60, deadline=None)
@given(n_units=st.integers(2, 6), n_periods=st.integers(4, 9),
       cells=st.lists(st.tuples(st.integers(0, 5), st.integers(0, 8),
                                st.sampled_from([np.nan, np.inf, -np.inf])), max_size=3))
def test_property_ingestion_refuses_exactly_the_panels_with_a_non_finite_outcome(n_units, n_periods, cells):
    """Property: over panel shapes and any set of missing or infinite cells (none included), ingestion succeeds
    exactly when every outcome is finite, and the wide matrix it returns is then finite."""
    df = _panel(n_units=n_units, n_periods=n_periods, start=n_periods - 1)
    placed = 0
    for u, t, v in cells:
        if u < n_units and t < n_periods:
            df.loc[(df.id == chr(97 + u)) & (df.time == t + 1), "y"] = v
            placed += 1
    if placed:
        with pytest.raises(MlsynthDataError):
            _prep(df)
        with pytest.raises(MlsynthDataError):
            _geo(df)
    else:
        assert np.isfinite(_prep(df)["Ywide"].to_numpy(dtype=float)).all()
        assert np.isfinite(_geo(df)["Ywide"].to_numpy(dtype=float)).all()
