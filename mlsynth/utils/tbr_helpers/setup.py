"""Ingestion for TBR: validate the design columns, then aggregate to two series.

Section 3.1 is the step that distinguishes TBR from geo-based regression: the
panel is aggregated across geos, not across time, giving one observation per
period for each of the treatment and control groups. That aggregate pair is what
``dataprep`` then ingests, so ingestion stays on the canonical path and the
group sums are a setup step on top of it and not a replacement for it.

A geo experiment has three groups. ``treat`` names the treated units and when
they were treated; ``control_col`` names which of the rest form the control
group. Anything neither treated nor flagged is unassigned and enters neither
sum, which is how a geo is held out of the experiment without being dropped
from the panel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..datautils import dataprep

MIN_PRETEST = 3
_GROUP_TREAT, _GROUP_CONTROL = "__tbr_treatment__", "__tbr_control__"


@dataclass(frozen=True)
class TBRInputs:
    """The two aggregated series and everything downstream reads off them."""

    y: np.ndarray                 # treatment group aggregate, all periods
    x: np.ndarray                 # control group aggregate, all periods
    cost_y: Optional[np.ndarray]
    cost_x: Optional[np.ndarray]
    n_pre: int
    n_test: int                   # periods credited to the test, cooldown included
    n_cooldown: int
    time_labels: np.ndarray
    treated_units: List
    control_units: List
    unassigned_units: List
    filled_cells: int
    intervention_time: object


def _binary_unit_flag(df: pd.DataFrame, unit: str, col: str) -> pd.Series:
    """A 0/1 flag that must be constant within unit, returned one row per unit."""
    values = pd.unique(df[col].dropna())
    allowed = {0, 1, True, False}
    bad = [v for v in values if v not in allowed]
    if bad:
        raise MlsynthDataError(
            f"{col!r} must be boolean or 0/1; it also takes {bad[:5]}")
    per_unit = df.groupby(unit, observed=True)[col].nunique()
    varying = per_unit[per_unit > 1].index.tolist()
    if varying:
        raise MlsynthDataError(
            f"{col!r} must be constant within unit; it varies for "
            f"{varying[:5]}")
    return df.groupby(unit, observed=True)[col].first().astype(int)


def _block_flag_start(df: pd.DataFrame, time: str, col: str) -> Optional[int]:
    """Validate a block-assigned period flag; return where it turns on.

    The flag is a property of the period, so it is constant across units within
    a period, and it is block assigned, so once it turns 1 it stays 1. Both the
    cooldown flag and the design-mode post flag are of this shape.
    """
    values = pd.unique(df[col].dropna())
    bad = [v for v in values if v not in {0, 1, True, False}]
    if bad:
        raise MlsynthDataError(
            f"{col!r} must be boolean or 0/1; it also takes {bad[:5]}")
    per_period = df.groupby(time, observed=True)[col].nunique()
    split = per_period[per_period > 1].index.tolist()
    if split:
        raise MlsynthDataError(
            f"{col!r} is block assigned, so it must be constant across units "
            f"within a period; it differs across units at {split[:5]}")
    flag = df.groupby(time, observed=True)[col].first().astype(int).sort_index()
    if not flag.any():
        return None
    if not flag.is_monotonic_increasing:
        raise MlsynthDataError(
            f"{col!r} must be sustained: once the cooldown begins the flag "
            f"stays 1, and this one turns back off")
    return int(np.argmax(flag.to_numpy() == 1))


def _complete_grid(df: pd.DataFrame, unit: str, time: str, config,
                   value_cols) -> "tuple[pd.DataFrame, int]":
    """Complete the unit-by-period grid, filling absent cells with zero.

    TBR sums across geos, so an absent geo-period enters the group total as
    nothing at all, which is arithmetically a zero. Making the fill explicit
    keeps the sum defined and lets the count be reported, because it is an
    assumption about why a cell is absent and not a detail.

    The design columns are reconstructed from what they are and never carried
    from a neighbouring row. A group flag is a property of the unit and a window
    flag a property of the period, so filling either along the wrong axis
    invents a value: a geo absent on the first cooldown day would inherit the
    previous day's 0 while every other geo reads 1. ``treat`` is a property of
    both, so it is rebuilt as "this unit is treated" and "this period is
    post", each read off the rows that are present.
    """
    units = sorted(df[unit].unique())
    periods = sorted(df[time].unique())
    grid = pd.MultiIndex.from_product([units, periods], names=[unit, time])
    indexed = df.set_index([unit, time])
    if indexed.index.has_duplicates:
        dupes = indexed.index[indexed.index.duplicated()].tolist()
        raise MlsynthDataError(
            f"the panel repeats {len(dupes)} unit-period cell(s), the first "
            f"being {dupes[:3]}. TBR sums across units, so a repeated cell is "
            f"either two observations to add or a duplicated row, and the two "
            f"give different group totals; de-duplicate or aggregate the panel "
            f"before fitting")
    missing = int(len(grid) - len(indexed))
    if missing == 0:
        return df, 0

    unit_level = [c for c in (config.control_col, config.treatment_col)
                  if c is not None]
    period_level = [c for c in (config.cooldown_col, config.post_col)
                    if c is not None]
    per_unit = {c: df.groupby(unit, observed=True)[c].max() for c in unit_level}
    per_period = {c: df.groupby(time, observed=True)[c].max() for c in period_level}
    treated_unit = post_period = None
    if config.treat is not None:
        treated_unit = df.groupby(unit, observed=True)[config.treat].max()
        post_period = df.groupby(time, observed=True)[config.treat].max()

    out = indexed.reindex(grid).reset_index()
    for c in value_cols:
        if c in out.columns:
            out[c] = out[c].fillna(0.0)
    for c in unit_level:
        out[c] = out[unit].map(per_unit[c])
    for c in period_level:
        out[c] = out[time].map(per_period[c])
    if config.treat is not None:
        out[config.treat] = (out[unit].map(treated_unit).astype(int)
                             & out[time].map(post_period).astype(int))
    # anything else the caller carried along is a passenger; fill it within the
    # unit, which is where a per-unit attribute lives
    handled = set(value_cols) | set(unit_level) | set(period_level) | {
        unit, time, config.treat}
    for c in [c for c in out.columns if c not in handled]:
        out[c] = out.groupby(unit, observed=True)[c].transform(
            lambda col: col.ffill().bfill())
    return out, missing


def build_inputs(config) -> TBRInputs:
    """Validate, fill, aggregate, and hand the aggregate pair to ``dataprep``."""
    unit, time = config.unitid, config.time
    outcome, treat = config.outcome, config.treat
    df = config.df.copy()

    value_cols = [outcome] + ([config.cost_col] if config.cost_col else [])
    df, filled = _complete_grid(df, unit, time, config, value_cols)

    is_control = _binary_unit_flag(df, unit, config.control_col)
    if treat is not None:
        df[treat] = df[treat].astype(int)
        treated_flag = df.groupby(unit, observed=True)[treat].max().astype(int)
    else:
        treated_flag = _binary_unit_flag(df, unit, config.treatment_col)
    treated = treated_flag[treated_flag == 1].index.tolist()
    controls = [u for u in is_control[is_control == 1].index if u not in treated]
    both = sorted(set(treated) & set(is_control[is_control == 1].index))
    if both:
        raise MlsynthDataError(
            f"unit(s) {both[:5]} are treated and also flagged as control by "
            f"{config.control_col!r}; a geo belongs to one group")
    if not treated:
        source = treat if treat is not None else config.treatment_col
        raise MlsynthDataError(
            f"the treatment group is empty: {source!r} marks no unit")
    if not controls:
        raise MlsynthDataError(
            f"no unit is in the control group: {config.control_col!r} flags "
            f"none of the untreated units, so there is nothing to regress on")
    unassigned = [u for u in df[unit].unique()
                  if u not in treated and u not in controls]

    cooldown_at = (None if config.cooldown_col is None
                   else _block_flag_start(df, time, config.cooldown_col))

    # Section 3.1's aggregation, and the treatment indicator carried with it.
    def _aggregate(column):
        wide = df.pivot_table(index=time, columns=unit, values=column,
                              aggfunc="sum").sort_index()
        return (wide[treated].sum(axis=1).to_numpy(),
                wide[controls].sum(axis=1).to_numpy())

    y, x = _aggregate(outcome)
    periods = np.asarray(sorted(df[time].unique()))
    if treat is not None:
        d_treat = (df[df[unit].isin(treated)].groupby(time, observed=True)[treat].max()
                   .astype(int).sort_index().to_numpy())
    else:
        # Design mode: the window is named by the post flag, so synthesise the
        # indicator the aggregate pair carries into ``dataprep``. Ingestion then
        # checks the window the same way in both modes.
        post_at = _block_flag_start(df, time, config.post_col)
        if post_at is None:
            raise MlsynthDataError(
                f"{config.post_col!r} never turns 1, so the panel has no "
                f"post-treatment window to report an effect over")
        if post_at == 0:
            raise MlsynthDataError(
                f"{config.post_col!r} is 1 from the first period, so there is "
                f"no pretest period to fit the relation on")
        d_treat = (periods >= periods[post_at]).astype(int)

    two_unit = pd.DataFrame({
        unit: np.repeat([_GROUP_TREAT, _GROUP_CONTROL], periods.size),
        time: np.tile(periods, 2),
        outcome: np.concatenate([y, x]),
        treat: np.concatenate([d_treat, np.zeros(periods.size, dtype=int)]),
    })
    prep = dataprep(two_unit, unit, time, outcome, treat)
    n_pre = int(prep["pre_periods"])
    if n_pre < MIN_PRETEST:
        raise MlsynthDataError(
            f"TBR needs at least {MIN_PRETEST} pretest periods to leave a "
            f"degree of freedom for the posterior (df = n - 2); this panel has "
            f"{n_pre}")

    if cooldown_at is not None and cooldown_at < n_pre:
        raise MlsynthDataError(
            f"the cooldown begins at period index {cooldown_at}, before "
            f"treatment starts at {n_pre}; a cooldown follows an intervention")
    # Section 3.2 credits the cooldown to the effect, as the reference's
    # ``use_cooldown=True`` default does, so the window is always the whole post
    # period and the flag only says where the intervention stopped inside it.
    n_test = periods.size - n_pre
    n_cooldown = 0 if cooldown_at is None else periods.size - cooldown_at
    if n_test < 1:  # pragma: no cover - dataprep already refuses a panel with
        # no post-treatment period, so n_pre is always strictly less than the
        # number of periods by the time this runs. Kept as an assertion of the
        # invariant this function relies on.
        raise MlsynthDataError(
            "no post-treatment periods; there is nothing to report an effect "
            "over")

    cost_y = cost_x = None
    if config.cost_col:
        cost_y, cost_x = _aggregate(config.cost_col)

    return TBRInputs(
        y=y, x=x, cost_y=cost_y, cost_x=cost_x, n_pre=n_pre, n_test=n_test,
        n_cooldown=n_cooldown, time_labels=periods, treated_units=treated,
        control_units=controls, unassigned_units=unassigned,
        filled_cells=filled,
        intervention_time=(periods[n_pre] if n_pre < periods.size else None),
    )
