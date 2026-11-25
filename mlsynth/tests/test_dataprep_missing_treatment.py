"""A panel with a missing treatment value is refused at ingestion.

``logictreat`` validated the treatment indicator as binary after dropping NaNs from the check, a skip that dates from
when an absent unit-period pivoted to NaN. ``balance`` and the outcome refusal now refuse absent rows, so the only NaN
left in the treatment pivot is a missing value in the column, and the skip read it as untreated: on a 5-unit,
8-period panel with unit 0 treated from period 5, a NaN treatment at a control unit or in the treated unit's
pre-period was accepted, and one after adoption was refused as "Treatment is not sustained". A unit-period whose
treatment status is unknown is the same case as an absent row, which the library already refuses, so ``dataprep``
refuses it and names the unit and period.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.datautils import dataprep, logictreat


def _panel(n_units=5, n_periods=8, treated=(0,), start=5, ids=None):
    """Unit ``u`` in ``treated`` adopts at ``start`` (a dict gives each unit its own start)."""
    ids = list(range(n_units)) if ids is None else list(ids)
    starts = treated if isinstance(treated, dict) else {u: start for u in treated}
    rng = np.random.default_rng(0)
    rows = [{"id": ids[u], "time": t, "y": 10.0 + u + 0.5 * t + rng.normal(),
             "d": int(u in starts and t >= starts[u])}
            for u in range(n_units) for t in range(n_periods)]
    return pd.DataFrame(rows)


def _blank(df, cells):
    """Set the treatment of each (unit, time) in ``cells`` to NaN."""
    df = df.copy()
    df["d"] = df["d"].astype(float)
    for unit, time in cells:
        df.loc[(df.id == unit) & (df.time == time), "d"] = np.nan
    return df


def _prep(df, **kw):
    return dataprep(df, unit_id_column_name="id", time_period_column_name="time",
                    outcome_column_name="y", treatment_indicator_column_name="d", **kw)


def _refusal(df, **kw):
    with pytest.raises(MlsynthDataError) as excinfo:
        _prep(df, **kw)
    return str(excinfo.value)


def _assert_same(a, b):
    """Two dataprep results agree key by key, arrays and frames by value."""
    if isinstance(a, pd.DataFrame):
        pd.testing.assert_frame_equal(a, b)
    elif isinstance(a, pd.Index):
        pd.testing.assert_index_equal(a, b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif hasattr(a, "keys") and hasattr(a, "__getitem__"):
        assert list(a.keys()) == list(b.keys())
        for k in a.keys():
            _assert_same(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _assert_same(x, y)
    else:
        assert a == b


# --- smoke: the panel with no missing cell is unchanged ---

def test_a_fully_observed_panel_ingests_unchanged():
    """A float treatment column with no NaN gives the same panel as the integer column it came from."""
    df = _panel()
    prep = _prep(df)
    assert prep["treated_unit_name"] == 0 and prep["pre_periods"] == 5 and prep["post_periods"] == 3
    _assert_same(_prep(_blank(df, [])), prep)


# --- failure: the three placements from the defect, each refused and reported ---

@pytest.mark.parametrize("unit, time", [(2, 3), (0, 2), (0, 6)],
                         ids=["control", "treated-pre-period", "treated-post-period"])
def test_a_missing_treatment_is_refused_and_named(unit, time):
    """A control unit, the treated unit before adoption, the treated unit after it: each was read as untreated
    (the last then refused as unsustained treatment); each is now refused as missing data, naming the cell."""
    msg = _refusal(_blank(_panel(), [(unit, time)]))
    assert msg.startswith("Missing treatment values")
    assert "'d'" in msg and "1 of 40" in msg
    assert f"unit '{unit}', time {time}" in msg
    assert "not sustained" not in msg


def test_the_message_counts_the_cells_and_names_the_first_in_time_then_unit_order():
    """The first missing cell is the first in time order, then unit order, as in the outcome refusal."""
    msg = _refusal(_blank(_panel(), [(4, 1), (1, 6), (3, 1)]))
    assert "3 of 40" in msg
    assert "unit '3', time 1" in msg            # unit order first would name unit '1', time 6


def test_string_unit_ids_are_named():
    ids = ["alpha", "bravo", "charlie", "delta", "echo"]
    msg = _refusal(_blank(_panel(ids=ids), [("charlie", 3)]))
    assert "unit 'charlie', time 3" in msg


@pytest.mark.parametrize("unit, time", [(4, 0), (0, 3), (1, 7)])
def test_the_cohort_branch_refuses_too(unit, time):
    """Two units adopting at different periods take the cohort branch of dataprep; it refuses the same way."""
    df = _panel(treated={0: 5, 1: 3})
    assert set(_prep(df)["cohorts"]) == {3, 5}
    msg = _refusal(_blank(df, [(unit, time)]))
    assert msg.startswith("Missing treatment values") and f"unit '{unit}', time {time}" in msg


def test_a_none_in_an_object_treatment_column_is_refused():
    df = _panel()
    df["d"] = df["d"].astype(object)
    df.loc[(df.id == 3) & (df.time == 4), "d"] = None
    msg = _refusal(df)
    assert msg.startswith("Missing treatment values") and "unit '3', time 4" in msg


def test_a_missing_treatment_is_not_reported_as_a_missing_outcome():
    """The two refusals are distinct: a NaN treatment is not counted against the outcome column."""
    msg = _refusal(_blank(_panel(), [(2, 3)]))
    assert "Missing outcome" not in msg


# --- edge: degenerate panels ---

def test_no_donors_does_not_excuse_a_missing_treatment():
    df = _panel(n_units=1)
    assert _prep(df, allow_no_donors=True)["donor_matrix"].shape == (8, 0)
    assert "unit '0', time 1" in _refusal(_blank(df, [(0, 1)]), allow_no_donors=True)


def test_a_single_donor_panel_refuses_at_the_donor():
    assert "unit '1', time 0" in _refusal(_blank(_panel(n_units=2), [(1, 0)]))


def test_a_missing_treatment_in_the_first_period_is_refused():
    """t = 0, where a unit treated from the start would adopt."""
    assert "unit '3', time 0" in _refusal(_blank(_panel(), [(3, 0)]))


def test_every_treatment_missing_is_refused_as_missing_not_as_no_treated_units():
    df = _panel(n_units=3, n_periods=4, start=2)
    msg = _refusal(_blank(df, [(u, t) for u in range(3) for t in range(4)]))
    assert "12 of 12" in msg and "No treated units" not in msg


def test_an_absent_row_is_still_reported_by_the_outcome_refusal():
    """An absent unit-period row also has no treatment status; it is reported as the panel being unbalanced, as
    before, and not as a missing treatment value."""
    df = _panel()
    df = df[~((df.id == 3) & (df.time == 6))]
    msg = _refusal(df)
    assert "Missing outcome" in msg and "1 absent unit-period row(s)" in msg


def test_an_absent_row_after_adoption_is_reported_as_absent_not_as_unsustained():
    df = _panel()
    df = df[~((df.id == 0) & (df.time == 6))]
    msg = _refusal(df)
    assert "absent unit-period row(s)" in msg and "not sustained" not in msg


def test_logictreat_refuses_a_nan_cell():
    """``logictreat`` takes a matrix from any caller; a NaN cell there is a missing treatment, not a zero."""
    matrix = np.zeros((6, 3))
    matrix[4:, 0] = 1.0
    matrix[2, 1] = np.nan
    with pytest.raises(MlsynthDataError, match=r"Missing treatment values: 1 of 18 cells .* row 2, column 1"):
        logictreat(matrix)


def test_logictreat_still_refuses_a_non_binary_value():
    matrix = np.zeros((6, 3))
    matrix[4:, 0] = 1.0
    matrix[1, 2] = 2.0
    with pytest.raises(MlsynthDataError, match="binary"):
        logictreat(matrix)


# --- failure through fit(): an MlsynthDataError, not a pandas or numpy error ---

def test_fdid_fit_refuses_a_missing_treatment():
    from mlsynth import FDID
    df = _blank(_panel(n_units=6, n_periods=12, start=9), [(3, 4)])
    with pytest.raises(MlsynthDataError, match=r"Missing treatment values.*unit '3', time 4"):
        FDID(dict(df=df, unitid="id", time="time", outcome="y", treat="d", display_graphs=False)).fit()


def test_sdid_cohort_fit_refuses_a_missing_treatment():
    from mlsynth import SDID
    df = _blank(_panel(n_units=8, n_periods=12, treated={0: 8, 1: 8}), [(5, 2)])
    with pytest.raises(MlsynthDataError, match=r"Missing treatment values.*unit '5', time 2"):
        SDID(dict(df=df, unitid="id", time="time", outcome="y", treat="d", display_graphs=False)).fit()


# --- property: the refusal is exactly the presence of a missing treatment cell ---

@settings(max_examples=80, deadline=None)
@given(data=st.data(), n_units=st.integers(2, 6), n_periods=st.integers(3, 9),
       string_ids=st.booleans())
def test_property_refused_exactly_when_a_treatment_cell_is_missing(data, n_units, n_periods, string_ids):
    """Over panel shapes, adoption patterns (single or cohort) and any set of missing treatment cells, ingestion
    refuses exactly when a cell is missing, counting the cells and naming the first in time-then-unit order; with
    none missing, the float column gives the result of the unmodified integer column."""
    n_treated = data.draw(st.integers(1, n_units - 1))
    starts = {u: data.draw(st.integers(1, n_periods - 1)) for u in range(n_treated)}
    ids = [f"u{u}" for u in range(n_units)] if string_ids else None
    labels = ids or list(range(n_units))
    df = _panel(n_units=n_units, n_periods=n_periods, treated=starts, ids=ids)
    cells = data.draw(st.sets(st.tuples(st.integers(0, n_units - 1), st.integers(0, n_periods - 1)), max_size=4))
    blanked = _blank(df, [(labels[u], t) for u, t in cells])
    if cells:
        msg = _refusal(blanked)
        assert msg.startswith("Missing treatment values")
        assert f"{len(cells)} of {n_units * n_periods}" in msg
        u, t = min(cells, key=lambda c: (c[1], c[0]))
        assert f"unit '{labels[u]}', time {t}" in msg
    else:
        _assert_same(_prep(blanked), _prep(df))
