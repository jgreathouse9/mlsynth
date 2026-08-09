"""Worst-case boundary testing for ``dataprep``.

``agents/agents_tests.md`` records two gaps as open. This file closes one of
them for the ingestion layer.

The four test levels there -- smoke, unit, edge, failure -- are all
single-fault: each varies one input to a boundary and holds the others
nominal. That is what makes them affordable (``4n + 1`` cases for ``n`` inputs
instead of ``5^n``), and it is sound when failures arise from one variable
being extreme. ``dataprep``'s inputs are not independent. The number of units,
the number of periods and the treatment date interact: one unit means no
donors, a treatment date of zero means no pre-periods, and a panel can be both
at once. Which error a caller sees then depends on a precedence between checks
that no single-fault test exercises, because no single-fault test ever makes
two conditions true together.

Worst-case testing is the classical answer (Jorgensen §5.4): take the
boundary values of each input and test their cross product, not one axis at
a time. The grid below is ``min``, ``min+``, ``nom``, ``max-``,
``max`` on each of the three panel-shape inputs.

The expected outcome is a decision table, and its content is the precedence:

===========================  ===========================================
condition (first that holds)  outcome
===========================  ===========================================
no treated observations       ``MlsynthDataError`` -- "No treated units"
exactly one unit              ``MlsynthDataError`` -- "No donor units"
treatment from period zero    ``MlsynthDataError`` -- "pre-treatment"
otherwise                     a valid contract, ``pre == treat_start``
===========================  ===========================================

The ordering is the finding. A panel with one unit treated from the first
period trips the donor rule and the pre-period rule simultaneously, and
reports the donor one; a caller who fixes only what the message names is sent
back for a second round. That is a property of the code as written, pinned
here so a reordering of the checks is a visible change and not a silent one.
"""

from __future__ import annotations

import itertools
from typing import Optional

import pandas as pd
import pytest

from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.datautils import dataprep

# Boundary values per input: min, min+, nom, max-, max.
_UNIT_COUNTS = (1, 2, 4, 7, 8)
_PERIOD_COUNTS = (1, 2, 6, 11, 12)
# Treatment dates are expressed relative to the panel length and resolved per
# case, since the upper boundary depends on the number of periods.
_TREAT_POSITIONS = ("none", "first", "second", "penultimate", "last")


def _panel(n_units: int, n_periods: int, treat_start: Optional[int]) -> pd.DataFrame:
    return pd.DataFrame([
        {"unit": f"u{u}", "time": 2000 + t, "y": float(10 * u + t),
         "treated": int(u == 0 and treat_start is not None and t >= treat_start)}
        for u in range(n_units) for t in range(n_periods)
    ])


def _resolve(position: str, n_periods: int) -> Optional[int]:
    """Map a boundary position onto an index, or ``None`` for no treatment."""
    return {
        "none": None,
        "first": 0,
        "second": min(1, n_periods - 1),
        "penultimate": max(n_periods - 2, 0),
        "last": n_periods - 1,
    }[position]


def _expected(n_units: int, n_periods: int, treat_start: Optional[int]) -> Optional[str]:
    """The decision table: the message fragment expected, or ``None`` if valid."""
    if treat_start is None:
        return "No treated units"
    if n_units == 1:
        return "No donor units"
    if treat_start == 0:
        return "pre-treatment"
    return None


_GRID = list(itertools.product(_UNIT_COUNTS, _PERIOD_COUNTS, _TREAT_POSITIONS))


@pytest.mark.parametrize("n_units,n_periods,position", _GRID)
def test_worst_case_panel_shapes(n_units, n_periods, position):
    """Every corner of the boundary cross product, against the decision table."""
    treat_start = _resolve(position, n_periods)
    df = _panel(n_units, n_periods, treat_start)
    expected = _expected(n_units, n_periods, treat_start)

    if expected is not None:
        with pytest.raises(MlsynthDataError, match=expected):
            dataprep(df, "unit", "time", "y", "treated")
        return

    out = dataprep(df, "unit", "time", "y", "treated")
    assert out["pre_periods"] == treat_start
    assert out["post_periods"] == n_periods - treat_start
    assert out["total_periods"] == n_periods
    assert out["donor_matrix"].shape == (n_periods, n_units - 1)


def test_the_grid_covers_every_row_of_the_decision_table():
    """A cross product earns its cost only by reaching every outcome.

    Guards against the grid drifting until some rule is never exercised --
    which is how a worst-case suite decays into an expensive smoke test.
    """
    reached = {
        _expected(nu, npd, _resolve(pos, npd)) for nu, npd, pos in _GRID
    }
    assert reached == {None, "No treated units", "No donor units", "pre-treatment"}


# ---------------------------------------------------------------------------
# The multiple-fault cases, named
# ---------------------------------------------------------------------------

def test_one_unit_treated_from_the_first_period_reports_the_donor_rule_first():
    """Both the donor rule and the pre-period rule are violated at once.

    Single-fault testing cannot reach this: it holds every other input
    nominal, so the two conditions never hold together and the precedence is
    never observed. The donor check runs first.
    """
    df = _panel(n_units=1, n_periods=4, treat_start=0)
    with pytest.raises(MlsynthDataError, match="No donor units"):
        dataprep(df, "unit", "time", "y", "treated")


def test_one_unit_with_no_treatment_reports_the_treated_rule_first():
    """Three-way: no treated unit, no donors, and nothing to split on."""
    df = _panel(n_units=1, n_periods=4, treat_start=None)
    with pytest.raises(MlsynthDataError, match="No treated units"):
        dataprep(df, "unit", "time", "y", "treated")


def test_allowing_no_donors_does_not_also_waive_the_pre_period_rule():
    """``allow_no_donors`` waives exactly one rule of the table.

    With the donor rule opted out, a panel that also has no pre-periods must
    still fail -- on the next rule down. An opt-out that swallowed both would
    return a contract with nothing to fit on.
    """
    df = _panel(n_units=1, n_periods=4, treat_start=0)
    with pytest.raises(MlsynthDataError, match="pre-treatment"):
        dataprep(df, "unit", "time", "y", "treated", allow_no_donors=True)


def test_treatment_in_the_final_period_leaves_exactly_one_post_period():
    """The upper boundary of the treatment date, where ``post_periods == 1``."""
    df = _panel(n_units=3, n_periods=6, treat_start=5)
    out = dataprep(df, "unit", "time", "y", "treated")
    assert out["pre_periods"] == 5
    assert out["post_periods"] == 1
