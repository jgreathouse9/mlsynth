"""Property tests for ``balance``, the validator every estimator runs first.

Thirty-five estimator modules call ``mlsynth.utils.datautils.balance`` on the
caller's frame before anything else touches it, so its verdict is the library's
front door. It answers three questions -- are any ``(unit, time)`` pairs
repeated, does every unit carry every period, and do all units carry the same
number of rows -- and which of them fires first is observable, because each
raises a different message that callers and tests match on.

Reimplementing it on the panel's key codes is a change to *how* it decides, so
the property under test is differential: for any frame at all, the shipped
implementation and a verbatim copy of the previous one must reach the same
outcome, including the exception's message. Generation earns its place here
because the interesting frames are the malformed ones, and the space of ways a
panel can be malformed is larger than a fixture list. The strategy therefore
builds a rectangular panel and then damages it -- dropping rows, repeating rows,
blanking keys, permuting order, retyping columns -- instead of only generating
well-formed input.

Two invariants beyond the differential one are worth generating over on their
own. The verdict cannot depend on row order, since a panel is a set of
observations. And it cannot depend on the labels used for units or periods, only
on the incidence pattern between them, so relabelling either key leaves the
answer alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.datautils import balance


# --------------------------------------------------------------------------- #
# the oracle -- the implementation as it stood before the key-code rewrite
# --------------------------------------------------------------------------- #
def _balance_oracle(df, unit_id_column_name, time_period_column_name):
    if df.duplicated([unit_id_column_name, time_period_column_name]).any():
        raise MlsynthDataError(
            "Duplicate observations found. Ensure each combination of unit and "
            "time is unique."
        )
    total_unique_time_periods = df[time_period_column_name].nunique()
    observations_per_unit = (
        df.groupby(unit_id_column_name)[time_period_column_name].nunique())
    if not (observations_per_unit == total_unique_time_periods).all():
        raise MlsynthDataError(
            "The panel is not strongly balanced. Not all units have observations "
            "for all unique time periods in the dataset."
        )
    if observations_per_unit.nunique() != 1:
        raise MlsynthDataError(
            "The panel is not strongly balanced. Units have different numbers "
            "of observations."
        )


def _outcome(fn, df):
    """``None`` if the frame is accepted, else ``(exception type, message)``."""
    try:
        fn(df, "id", "time")
        return None
    except Exception as exc:                     # noqa: BLE001 - comparing behaviour
        return type(exc).__name__, str(exc)


# --------------------------------------------------------------------------- #
# strategies
# --------------------------------------------------------------------------- #
@st.composite
def _rectangular(draw, max_units=6, max_periods=6):
    n_units = draw(st.integers(min_value=1, max_value=max_units))
    n_periods = draw(st.integers(min_value=1, max_value=max_periods))
    return pd.DataFrame({
        "id": np.repeat([f"u{i}" for i in range(n_units)], n_periods),
        "time": np.tile(np.arange(1, n_periods + 1), n_units),
        "y": np.arange(n_units * n_periods, dtype=float),
    })


@st.composite
def _panel(draw):
    """A rectangular panel, then zero or more kinds of damage applied to it.

    The damages are the ways real frames arrive broken: a missing observation,
    a repeated one, a blank key, and reordering. They compose, so a frame can be
    duplicated *and* short, which is what makes the precedence between the three
    errors testable.
    """
    df = draw(_rectangular())
    for _ in range(draw(st.integers(min_value=0, max_value=3))):
        kind = draw(st.sampled_from(["drop", "repeat", "blank_unit",
                                     "blank_time", "shuffle", "retype_time"]))
        if kind == "drop" and len(df) > 1:
            i = draw(st.integers(min_value=0, max_value=len(df) - 1))
            df = df.drop(index=df.index[i]).reset_index(drop=True)
        elif kind == "repeat" and len(df):
            i = draw(st.integers(min_value=0, max_value=len(df) - 1))
            df = pd.concat([df, df.iloc[[i]]], ignore_index=True)
        elif kind == "blank_unit" and len(df):
            i = draw(st.integers(min_value=0, max_value=len(df) - 1))
            df = df.copy()
            df.loc[df.index[i], "id"] = None
        elif kind == "blank_time" and len(df):
            i = draw(st.integers(min_value=0, max_value=len(df) - 1))
            df = df.copy()
            df["time"] = df["time"].astype(object)
            df.loc[df.index[i], "time"] = None
        elif kind == "shuffle" and len(df):
            order = draw(st.permutations(list(range(len(df)))))
            df = df.iloc[list(order)].reset_index(drop=True)
        elif kind == "retype_time":
            df = df.copy()
            df["time"] = df["time"].astype(str)
    return df


SETTINGS = settings(max_examples=250, deadline=None,
                    suppress_health_check=[HealthCheck.too_slow,
                                           HealthCheck.data_too_large])


# --------------------------------------------------------------------------- #
# 1. the differential property -- the acceptance bar, generated
# --------------------------------------------------------------------------- #
@SETTINGS
@given(df=_panel())
def test_same_verdict_as_the_previous_implementation(df):
    assert _outcome(balance, df) == _outcome(_balance_oracle, df)


@SETTINGS
@given(df=_rectangular())
def test_a_rectangular_panel_is_always_accepted(df):
    assert balance(df, "id", "time") is None


# --------------------------------------------------------------------------- #
# 2. invariants of the verdict itself
# --------------------------------------------------------------------------- #
@SETTINGS
@given(df=_panel(), data=st.data())
def test_verdict_is_independent_of_row_order(df, data):
    """A panel is a set of observations, so shuffling the frame cannot change
    whether it is well formed or which fault it has."""
    assume(len(df) > 1)
    order = data.draw(st.permutations(list(range(len(df)))))
    shuffled = df.iloc[list(order)].reset_index(drop=True)
    assert _outcome(balance, df) == _outcome(balance, shuffled)


@SETTINGS
@given(df=_panel())
def test_verdict_is_independent_of_unit_labels(df):
    """Only the incidence pattern between units and periods matters, so
    renaming units cannot change the answer."""
    relabelled = df.copy()
    relabelled["id"] = relabelled["id"].map(
        lambda v: v if v is None or pd.isna(v) else f"z-{v}-z")
    assert _outcome(balance, df) == _outcome(balance, relabelled)


@SETTINGS
@given(df=_panel())
def test_verdict_is_independent_of_period_labels(df):
    """Likewise for periods: a monotone relabelling of the time axis leaves the
    incidence pattern intact."""
    relabelled = df.copy()
    relabelled["time"] = relabelled["time"].map(
        lambda v: v if v is None or pd.isna(v) else f"t{v}")
    assert _outcome(balance, df) == _outcome(balance, relabelled)


@SETTINGS
@given(df=_panel())
def test_accepting_means_every_unit_carries_every_observed_period(df):
    """The positive verdict has content: in an accepted frame every unit has
    exactly one row per distinct observed period.

    Stated over *observed* periods, not over rows, because a missing time key is
    excluded from the counts on both sides -- see
    ``test_a_panel_whose_periods_are_all_missing_is_accepted`` for the corner
    that makes that distinction visible.
    """
    if _outcome(balance, df) is not None:
        return
    n_periods = df["time"].nunique()
    per_unit = df.dropna(subset=["id"]).groupby("id")["time"].nunique()
    assert set(per_unit.unique()) <= {n_periods}


def test_a_panel_whose_periods_are_all_missing_is_accepted():
    """A corner of the shipped contract, characterised rather than corrected.

    ``nunique`` excludes missing values on both sides of the comparison, so a
    panel where every unit is missing the same period passes: with one unit and
    times ``[1, NaN]`` there is one distinct period and the unit is observed in
    one, so the check is satisfied. It is preserved here because 35 estimator
    modules depend on this verdict, and changing which frames are admitted is a
    behaviour change and not a speed one.
    """
    accepted = pd.DataFrame({"id": ["a", "a", "b", "b"],
                             "time": [1, None, 1, None], "y": 1.0})
    assert balance(accepted, "id", "time") is None

    # Asymmetric, so the counts disagree and it is rejected.
    rejected = pd.DataFrame({"id": ["a", "a", "b", "b"],
                             "time": [1, None, 1, 2], "y": 1.0})
    with pytest.raises(MlsynthDataError,
                       match="Not all units have observations"):
        balance(rejected, "id", "time")


def test_a_row_whose_unit_key_is_missing_is_invisible_to_the_check():
    """The same corner on the other axis, and the sharper one.

    ``groupby`` forms no group for a missing unit key, so rows carrying one are
    not counted, not compared, and not reported -- the frame is accepted and the
    row survives into ``dataprep``. Characterised, not corrected: admitting
    fewer frames is a behaviour change across 35 estimator modules and belongs
    with the validator's semantics, not with its cost.
    """
    df = pd.DataFrame({"id": ["a", None], "time": [1, 1], "y": 1.0})
    assert balance(df, "id", "time") is None


@SETTINGS
@given(df=_rectangular(), data=st.data())
def test_repeating_any_row_is_always_reported_as_a_duplicate(df, data):
    """The first of the three checks, generated: repeating an observation is
    reported as a duplicate whatever else is true of the panel."""
    i = data.draw(st.integers(min_value=0, max_value=len(df) - 1))
    damaged = pd.concat([df, df.iloc[[i]]], ignore_index=True)
    with pytest.raises(MlsynthDataError, match="Duplicate observations found"):
        balance(damaged, "id", "time")


@SETTINGS
@given(df=_rectangular(), data=st.data())
def test_dropping_any_row_is_always_reported_as_unbalanced(df, data):
    """And the second: a panel with a hole in it is never accepted.

    Both axes must have extent for a dropped row to leave a hole. Removing the
    only unit's row for some period, or the only period's row for some unit,
    removes that period or unit from the panel entirely and leaves a smaller
    rectangle -- correctly accepted.
    """
    assume(df["id"].nunique() > 1 and df["time"].nunique() > 1)
    i = data.draw(st.integers(min_value=0, max_value=len(df) - 1))
    damaged = df.drop(index=df.index[i]).reset_index(drop=True)
    with pytest.raises(MlsynthDataError,
                       match="The panel is not strongly balanced"):
        balance(damaged, "id", "time")


# --------------------------------------------------------------------------- #
# 3. the work bound, generated over shapes
# --------------------------------------------------------------------------- #
@settings(max_examples=60, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(n_units=st.integers(min_value=1, max_value=60),
       n_periods=st.integers(min_value=1, max_value=60))
def test_cell_count_never_exceeds_the_row_budget(n_units, n_periods):
    """The rejection path stays bounded whatever the shape.

    A count over the ``(units + 1) x (periods + 1)`` grid is only taken where
    that grid is within twice the row count; otherwise the cells are hashed.
    Without the bound, a panel of singleton units asks for a count over a grid
    quadratic in the number of rows.
    """
    df = pd.DataFrame({
        "id": np.repeat([f"u{i}" for i in range(n_units)], n_periods),
        "time": np.tile(np.arange(n_periods), n_units),
        "y": 0.0,
    })
    budget = 4 * max(len(df), 1)
    real = np.bincount
    seen = []

    def guarded(x, weights=None, minlength=0):
        seen.append(minlength)
        return real(x, weights=weights, minlength=minlength)

    np_bincount = np.bincount
    np.bincount = guarded
    try:
        balance(df, "id", "time")
    finally:
        np.bincount = np_bincount
    assert all(m <= budget for m in seen), (
        f"counted over {max(seen)} cells for {len(df)} rows")
