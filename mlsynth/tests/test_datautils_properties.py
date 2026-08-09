"""Property tests for ``dataprep``, the ingestion contract every estimator uses.

Architecture invariant 2 in ``CLAUDE.md`` routes every estimator through
``mlsynth.utils.datautils.dataprep``, so its output contract -- ``y``,
``donor_matrix``, ``pre_periods``, ``post_periods``, ``Ywide`` -- is the one
piece of shared state the whole library depends on. A defect here misreports
every estimator at once, and it would do so silently: a donor column in the
wrong order or an off-by-one in ``pre_periods`` produces numbers, not errors.

What the contract promises, and what is asserted below over generated panels:

* the period counts partition the panel, ``pre + post == total``;
* ``y`` is the treated unit's own series, and ``donor_matrix`` is the wide
  frame with exactly that column removed, in column order matching
  ``donor_names``;
* nothing is invented or dropped -- every value round-trips back to the long
  input frame;
* ingestion is a set operation on rows, so shuffling the input frame cannot
  move a number, and relabelling the units permutes the donor columns instead
  of changing their contents;
* the two documented failures raise ``MlsynthDataError``.

The row-order and relabelling properties are the ones generation earns most.
Both are the kind of assumption a pivot silently violates, and neither is
visible in an estimator's output until two runs of the same data disagree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.datautils import dataprep

_VALUE = st.floats(min_value=-1e4, max_value=1e4,
                   allow_nan=False, allow_infinity=False, width=64)


@st.composite
def panels(draw, min_units: int = 2, max_units: int = 6):
    """A balanced long panel with one absorbing-treated unit.

    Returns ``(df, n_units, n_periods, treat_start)`` where ``treat_start`` is
    the index of the first treated period, so ``pre_periods`` should equal it.
    """
    n_units = draw(st.integers(min_value=min_units, max_value=max_units))
    n_periods = draw(st.integers(min_value=3, max_value=18))
    treat_start = draw(st.integers(min_value=1, max_value=n_periods - 1))
    values = draw(st.lists(_VALUE, min_size=n_units * n_periods,
                           max_size=n_units * n_periods))
    grid = np.asarray(values).reshape(n_units, n_periods)

    rows = [
        {"unit": f"u{u}", "time": 2000 + t, "y": float(grid[u, t]),
         "treated": int(u == 0 and t >= treat_start)}
        for u in range(n_units) for t in range(n_periods)
    ]
    return pd.DataFrame(rows), n_units, n_periods, treat_start


def _prep(df):
    return dataprep(df, "unit", "time", "y", "treated")


# ---------------------------------------------------------------------------
# The shape contract
# ---------------------------------------------------------------------------

@given(panel=panels())
@settings(max_examples=150)
def test_period_counts_partition_the_panel(panel):
    df, n_units, n_periods, treat_start = panel
    out = _prep(df)
    assert out["pre_periods"] + out["post_periods"] == out["total_periods"]
    assert out["total_periods"] == n_periods
    assert out["pre_periods"] == treat_start


@given(panel=panels())
@settings(max_examples=150)
def test_donor_matrix_is_the_wide_frame_without_the_treated_column(panel):
    df, n_units, n_periods, _ = panel
    out = _prep(df)

    assert out["treated_unit_name"] == "u0"
    assert out["donor_matrix"].shape == (n_periods, n_units - 1)
    assert list(out["donor_names"]) == [f"u{u}" for u in range(1, n_units)]

    wide = out["Ywide"]
    np.testing.assert_allclose(out["y"], wide["u0"].to_numpy(), atol=0)
    np.testing.assert_allclose(
        out["donor_matrix"], wide.drop(columns=["u0"]).to_numpy(), atol=0
    )


@given(panel=panels())
@settings(max_examples=150)
def test_every_value_round_trips_back_to_the_long_frame(panel):
    """Nothing is invented, dropped or transposed by the pivot."""
    df, _, _, _ = panel
    out = _prep(df)
    wide = out["Ywide"]

    for _, row in df.iterrows():
        assert wide.loc[row["time"], row["unit"]] == pytest.approx(row["y"], abs=0)


# ---------------------------------------------------------------------------
# Invariances
# ---------------------------------------------------------------------------

@given(panel=panels(), seed=st.integers(0, 2**31 - 1))
@settings(max_examples=100)
def test_row_order_of_the_long_frame_is_irrelevant(panel, seed):
    """A long frame is a set of rows; the pivot must not depend on their order."""
    df, _, _, _ = panel
    shuffled = df.sample(frac=1.0, random_state=seed % (2**31)).reset_index(drop=True)

    base, other = _prep(df), _prep(shuffled)
    np.testing.assert_allclose(other["y"], base["y"], atol=0)
    np.testing.assert_allclose(other["donor_matrix"], base["donor_matrix"], atol=0)
    assert other["pre_periods"] == base["pre_periods"]
    assert other["treated_unit_name"] == base["treated_unit_name"]


@given(panel=panels(min_units=3))
@settings(max_examples=100)
def test_relabelling_donors_permutes_the_columns_and_keeps_their_contents(panel):
    """Unit labels are identifiers, not data.

    Renaming the donors so their sort order reverses must reorder the donor
    columns to match ``donor_names`` and leave the set of series unchanged.
    """
    df, n_units, _, _ = panel
    base = _prep(df)

    # u1..u{n-1} -> reversed alphabetical order, treated unit untouched.
    mapping = {f"u{u}": f"z{n_units - u}" for u in range(1, n_units)}
    renamed = df.copy()
    renamed["unit"] = renamed["unit"].map(lambda u: mapping.get(u, u))
    other = _prep(renamed)

    assert list(other["donor_names"]) == sorted(mapping.values())
    for label, original in zip(base["donor_names"], base["donor_matrix"].T):
        column = other["donor_matrix"][:, list(other["donor_names"]).index(mapping[label])]
        np.testing.assert_allclose(column, original, atol=0)


@given(panel=panels(), shift=st.floats(min_value=-1e3, max_value=1e3,
                                       allow_nan=False, allow_infinity=False))
@settings(max_examples=100)
def test_shifting_the_outcome_shifts_every_series_and_nothing_else(panel, shift):
    """A units change touches the outcome arrays and no structural field."""
    df, _, _, _ = panel
    shifted = df.copy()
    shifted["y"] += shift

    base, other = _prep(df), _prep(shifted)
    np.testing.assert_allclose(other["y"], base["y"] + shift, rtol=1e-9, atol=1e-6)
    np.testing.assert_allclose(other["donor_matrix"], base["donor_matrix"] + shift,
                               rtol=1e-9, atol=1e-6)
    assert other["pre_periods"] == base["pre_periods"]
    assert other["post_periods"] == base["post_periods"]


# ---------------------------------------------------------------------------
# Documented failures
# ---------------------------------------------------------------------------

@given(panel=panels())
@settings(max_examples=50)
def test_a_panel_with_no_donors_is_reported(panel):
    df, _, _, _ = panel
    only_treated = df[df["unit"] == "u0"]
    with pytest.raises(MlsynthDataError, match="donor"):
        _prep(only_treated)


@given(panel=panels())
@settings(max_examples=50)
def test_treatment_from_the_first_period_leaves_no_pre_period_and_is_reported(panel):
    """Zero pre-periods is unusable for a synthetic control and must raise."""
    df, _, _, _ = panel
    always_treated = df.copy()
    always_treated["treated"] = (always_treated["unit"] == "u0").astype(int)
    with pytest.raises(MlsynthDataError, match="pre-treatment"):
        _prep(always_treated)


@given(panel=panels())
@settings(max_examples=50)
def test_no_donors_is_allowed_when_the_caller_opts_in(panel):
    """``allow_no_donors`` exists for estimators with no control pool (SHC)."""
    df, _, _, _ = panel
    only_treated = df[df["unit"] == "u0"]
    out = dataprep(only_treated, "unit", "time", "y", "treated",
                   allow_no_donors=True)
    assert out["donor_matrix"].shape[1] == 0
    assert out["pre_periods"] >= 1
