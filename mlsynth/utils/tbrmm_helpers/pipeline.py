"""Ingestion and result assembly for TBRMM.

The panel carries no treatment, so ingestion has one job: produce the scoring
window as a periods-by-geos matrix, the geo labels in a fixed order, and each
geo's ``A_i``. The eligibility and post-period validators are TBR's, imported and
not reimplemented, so a column shaped wrong is refused the same way at both
stages.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..tbr_helpers.setup import _binary_unit_flag, _block_flag_start
from .config import TBRMMConfig
from ..post_fit import compute_post_fit_tbrmm, to_effect_result
from .estimate import measure_design
from .search import CONTROL, TREATMENT, UNASSIGNED, greedy_search
from .structures import TBRMMDesign, TBRMMResults


def _scoring_window(config: TBRMMConfig) -> pd.DataFrame:
    """The rows the objective is computed over.

    Without ``post_col`` that is the whole panel. With it, the periods before the
    flag turns on: a panel that already carries a later window is scored on its
    history alone, which is what lets a design be chosen on a panel whose
    intervention has already happened elsewhere.
    """
    df = config.df
    if config.post_col is None:
        return df
    _block_flag_start(df, config.time, config.post_col)
    return df[df[config.post_col].astype(int) == 0]


def _post_window(config: TBRMMConfig) -> Optional[pd.DataFrame]:
    """The realized rows, or ``None`` when the panel carries none.

    The mirror of :func:`_scoring_window`: the periods the design was *not*
    scored on are the periods it is measured on. A ``post_col`` that marks
    nothing returns ``None``, so a column of zeros is a design-only run and not
    an effect of zero.
    """
    if config.post_col is None:
        return None
    df = config.df
    _block_flag_start(df, config.time, config.post_col)
    post = df[df[config.post_col].astype(int) == 1]
    return None if post.empty else post


def _wide(window: pd.DataFrame, config: TBRMMConfig) -> pd.DataFrame:
    """Periods by geos, geos in sorted label order.

    Cell uniqueness is the base configuration's invariant and is not rechecked
    here. Completeness is checked here, because it is not enforced anywhere
    above: a group aggregate sums across units, so a gap drops that geo from
    whichever aggregate it lands in for those periods and shortens no series
    while doing it.
    """
    unit, time, outcome = config.unitid, config.time, config.outcome
    wide = window.pivot(index=time, columns=unit, values=outcome).sort_index()
    wide = wide[sorted(wide.columns)]
    missing = int(wide.isna().to_numpy().sum())
    if missing:
        raise MlsynthDataError(
            f"the scoring window is missing {missing} unit-period outcome "
            f"value(s); a group aggregate sums across units, so a gap silently "
            f"drops that geo from the aggregate for those periods.")
    return wide


def _eligibility(window: pd.DataFrame, config: TBRMMConfig,
                 units: List) -> List[FrozenSet[str]]:
    """Au's ``A_i`` per geo, from the three optional columns.

    A column left unset admits every geo to that role, so the default is the
    unconstrained problem.
    """
    unit = config.unitid
    flags: Dict[str, Optional[pd.Series]] = {}
    for role, col in ((TREATMENT, config.treatment_eligible_col),
                      (CONTROL, config.control_eligible_col),
                      (UNASSIGNED, config.unassigned_eligible_col)):
        flags[role] = None if col is None else _binary_unit_flag(window, unit, col)
    out = []
    for geo in units:
        roles = {role for role, flag in flags.items()
                 if flag is None or int(flag.loc[geo]) == 1}
        out.append(frozenset(roles))
    return out


def build_inputs(config: TBRMMConfig) -> Tuple[np.ndarray, List, List[FrozenSet[str]]]:
    """The scoring matrix, the geo labels and each geo's eligibility."""
    window = _scoring_window(config)
    wide = _wide(window, config)
    units = list(wide.columns)
    return wide.to_numpy(dtype=float), units, _eligibility(window, config, units)


def run(config: TBRMMConfig) -> TBRMMResults:
    """Search the partitions and assemble the design result."""
    y_matrix, units, eligibility = build_inputs(config)
    outcomes = greedy_search(
        y_matrix, eligibility, max_treatment_size=config.max_treatment_size,
        n_test=config.n_test, objective=config.objective,
        control_start=config.control_start)

    # When the panel carries realized periods, every candidate is measured on
    # them -- not only the recommendation -- so the menu can be compared after
    # the fact as well as before it. The post matrix is built on the same geo
    # order as the scoring matrix so a column index means the same geo in both.
    post_window = _post_window(config)
    post_matrix = (_wide(post_window, config).reindex(columns=units).to_numpy(dtype=float)
                   if post_window is not None else None)

    designs = []
    measured = []
    for o in outcomes:
        treatment_units = [units[j] for j in o.treatment]
        control_units = [units[j] for j in o.control]
        effect, treated_path, control_path = (
            measure_design(y_matrix, post_matrix, units,
                           treatment_units, control_units)
            if post_matrix is not None else (None, None, None))
        measured.append((treated_path, control_path))
        designs.append(TBRMMDesign(
            k=o.k,
            treatment_units=treatment_units,
            control_units=control_units,
            unassigned_units=[units[j] for j in o.unassigned],
            objective_value=float(o.score.value),
            detail=dict(o.score.detail),
            matching_trace=[tuple(float(v) for v in key) for key in o.trace],
            matching_converged=o.converged,
            n_candidates_evaluated=o.evaluations,
            effect=effect,
        ))
    best = max(range(len(outcomes)), key=lambda i: outcomes[i].score.key)
    recommended = designs[best]

    report = None
    if post_matrix is not None:
        treated_path, control_path = measured[best]
        report = to_effect_result(
            compute_post_fit_tbrmm(
                treated_path, control_path,
                n_fit=int(y_matrix.shape[0]),
                n_post=int(post_matrix.shape[0]),
                n_treated_units=len(recommended.treatment_units)),
            method_name="TBRMM")

    return TBRMMResults(
        report=report,
        designs=designs,
        recommended=recommended,
        objective=config.objective,
        forced_treatment_units=[units[j] for j, roles in enumerate(eligibility)
                                if roles == frozenset({TREATMENT})],
        eligible_treatment_units=[units[j] for j, roles in enumerate(eligibility)
                                  if TREATMENT in roles],
        eligible_control_units=[units[j] for j, roles in enumerate(eligibility)
                                if CONTROL in roles],
        n_periods_scored=int(y_matrix.shape[0]),
        selected_units=list(recommended.treatment_units),
        power=pd.DataFrame([
            {"k": d.k, "objective_value": d.objective_value,
             **{key: d.detail[key] for key in sorted(d.detail)}}
            for d in designs
        ]),
    )
