"""Ingestion and result assembly for TBRMM.

The panel carries no treatment, so ingestion has one job: produce the scoring
window as a periods-by-geos matrix, the geo labels in a fixed order, and each
geo's ``A_i``. The eligibility and post-period validators are TBR's, imported and
not reimplemented, so a column shaped wrong is refused the same way at both
stages.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np
import pandas as pd

from ....exceptions import MlsynthDataError
from ...syndes_helpers.eligibility import unit_attribute_map
from ...syndes_helpers.restrictions import build_restrictions
from ..setup import _binary_unit_flag, _block_flag_start
from .config import TBRConfig
from ...post_fit import compute_post_fit_tbrmm, to_effect_result
from ..calibration import calibrate, design_end, resolve_blank_periods
from ..posterior import fit_pretest
from .estimate import measure_design
from .local import local_search
from .objective import required_impact
from .rules import DesignRules
from .search import CONTROL, TREATMENT, UNASSIGNED, greedy_search
from .structures import (TBRDesignConstraints, TBRResults, TBRMMDesign,
                         TBRStratumQuota)


def _scoring_window(config: TBRConfig) -> pd.DataFrame:
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


def _post_window(config: TBRConfig) -> Optional[pd.DataFrame]:
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


def _wide(window: pd.DataFrame, config: TBRConfig) -> pd.DataFrame:
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


def _eligibility(window: pd.DataFrame, config: TBRConfig,
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


def build_inputs(config: TBRConfig) -> Tuple[np.ndarray, List, List[FrozenSet[str]]]:
    """The scoring matrix, the geo labels and each geo's eligibility."""
    window = _scoring_window(config)
    wide = _wide(window, config)
    units = list(wide.columns)
    return wide.to_numpy(dtype=float), units, _eligibility(window, config, units)


def design_rules(config: TBRConfig, units: List,
                 eligibility: List[FrozenSet[str]]
                 ) -> Tuple[List[FrozenSet[str]], Optional[DesignRules],
                            Optional[TBRDesignConstraints]]:
    """The business constraints, resolved against the geo order of the search.

    Returns the eligibility with forced and forbidden geos folded in, the
    group-level rules (None when no group-level rule is set, so the search is
    the one without rules) and the record the result carries (None when no
    constraint field is set).

    Forced geos, forbidden geos and the size band speak about one geo each, so
    they become roles: a forced geo is eligible for treatment alone, which is
    what the search already reads as forced, and a forbidden or out-of-band geo
    loses the treatment role and keeps the others. The labels are resolved by
    the SYNDES builder, so the vocabulary and its checks are the ones SYNDES,
    MAREX and LEXSCM share.
    """
    if not config.design_constraint_fields:
        return eligibility, None, None
    df, unit = config.df, config.unitid
    restrictions = build_restrictions(
        df, unit, SimpleNamespace(labels=np.asarray(units)),
        to_be_treated=config.to_be_treated,
        not_to_be_treated=config.not_to_be_treated,
        cluster_col=config.cluster_col, adjacency=config.adjacency,
        spillover_threshold=config.spillover_threshold,
        stratum_col=config.stratum_col, min_per_stratum=config.min_per_stratum,
        max_per_stratum=config.max_per_stratum, size_col=config.size_col,
        min_size=config.min_size, max_size=config.max_size)

    roles = list(eligibility)
    for j in restrictions.forced_in:
        if TREATMENT not in roles[j]:
            raise MlsynthDataError(
                f"{units[j]!r} is in to_be_treated but treatment_eligible_col "
                f"does not allow it in the treatment group.")
        roles[j] = frozenset({TREATMENT})
    for j in restrictions.forbidden:
        roles[j] = roles[j] - {TREATMENT}

    costs = None
    if config.market_cost_col is not None:
        by_unit = unit_attribute_map(df, unit, config.market_cost_col)
        costs = [float(by_unit[u]) for u in units]

    group_level = (restrictions.conflict_pairs or restrictions.strata
                   or costs is not None or config.exclude_bordering_donors
                   or config.min_controls is not None
                   or config.max_controls is not None)
    rules = DesignRules.build(
        roles, conflict_pairs=restrictions.conflict_pairs,
        strata=restrictions.strata, costs=costs, budget=config.max_total_cost,
        exclude_bordering_donors=config.exclude_bordering_donors,
        min_controls=config.min_controls, max_controls=config.max_controls)

    stratum_of = (unit_attribute_map(df, unit, config.stratum_col)
                  if config.stratum_col is not None else {})
    forbidden_by_name = set(config.not_to_be_treated or [])
    record = TBRDesignConstraints(
        to_be_treated=list(config.to_be_treated or []),
        not_to_be_treated=list(config.not_to_be_treated or []),
        size_ineligible_units=[units[j] for j in restrictions.forbidden
                               if units[j] not in forbidden_by_name],
        conflict_pairs=[(units[i], units[j])
                        for i, j in restrictions.conflict_pairs],
        strata=[TBRStratumQuota(stratum=stratum_of[units[min(members)]],
                                members=[units[j] for j in sorted(members)],
                                min_treated=lo, max_treated=hi)
                for members, lo, hi in rules.strata],
        market_costs=({} if costs is None else dict(zip(units, costs))),
        max_total_cost=config.max_total_cost,
        exclude_bordering_donors=config.exclude_bordering_donors,
        min_controls=config.min_controls, max_controls=config.max_controls,
        rules=list(rules.active))
    return roles, (rules if group_level else None), record


def _reference_impact(search_matrix: np.ndarray, treatment: List[int],
                      control: List[int], n_test: int) -> float:
    """The reference's required impact for a design, on its search window.

    :func:`~.objective.required_impact` from the design's own fit on the window
    it was chosen on, so under the reference objective it is the
    ``required_impact`` the search scored, and under ``"paper"`` the same
    formula the reference would have.
    """
    fit = fit_pretest(search_matrix[:, treatment].sum(axis=1),
                      search_matrix[:, control].sum(axis=1))
    return required_impact(float(np.sqrt(fit.sigma_sq)), fit.n_pretest, n_test)


def run(config: TBRConfig) -> TBRResults:
    """Search the partitions and assemble the design result."""
    y_matrix, units, eligibility = build_inputs(config)
    eligibility, rules, constraints = design_rules(config, units, eligibility)

    # When the panel carries realized periods, every candidate is measured on
    # them -- not only the recommendation -- so the menu can be compared after
    # the fact as well as before it. The post matrix is built on the same geo
    # order as the scoring matrix so a column index means the same geo in both.
    post_window = _post_window(config)
    post_wide = (_wide(post_window, config).reindex(columns=units)
                 if post_window is not None else None)
    post_matrix = post_wide.to_numpy(dtype=float) if post_wide is not None else None

    # Calibrated inference holds the last B pre-periods back from the search,
    # so the blank windows its interval is calibrated on are ones no design
    # choice saw. The fits that report on a design still use the whole
    # pre-period.
    calibrated = config.inference == "calibrated"
    search_matrix = y_matrix
    if calibrated:
        test_length = (config.n_test if post_matrix is None
                       else int(post_matrix.shape[0]))
        blank_periods = resolve_blank_periods(config.blank_periods, test_length)
        search_matrix = y_matrix[:design_end(y_matrix.shape[0], blank_periods)]

    outcomes = greedy_search(
        search_matrix, eligibility, max_treatment_size=config.max_treatment_size,
        n_test=config.n_test, objective=config.objective,
        control_start=config.control_start, rules=rules)
    greedy_keys = [None] * len(outcomes)
    if config.search == "local":
        greedy_keys = [o.score.key for o in outcomes]
        outcomes = [local_search(search_matrix, eligibility, o,
                                 n_test=config.n_test,
                                 restarts=config.local_restarts,
                                 seed=config.local_seed + o.k, rules=rules)
                    for o in outcomes]
    best = max(range(len(outcomes)), key=lambda i: outcomes[i].score.key)

    blank = None
    if calibrated:
        blank = calibrate(
            y_matrix, outcomes[best].treatment, outcomes[best].control,
            blank_periods=blank_periods,
            horizon=max(test_length, config.n_test), eligibility=eligibility,
            rules=rules, objective=config.objective, n_test=config.n_test,
            n_placebos=config.n_placebos, seed=config.calibration_seed)
        k_test = blank.k[config.n_test - 1]

    designs = []
    measured = []
    for o, greedy_key in zip(outcomes, greedy_keys):
        treatment_units = [units[j] for j in o.treatment]
        control_units = [units[j] for j in o.control]
        effect, treated_path, control_path = (
            measure_design(y_matrix, post_matrix, units,
                           treatment_units, control_units,
                           level=config.level, variance=config.variance,
                           hac_bandwidth=config.hac_bandwidth)
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
            search=config.search,
            greedy_key=(None if greedy_key is None
                        else tuple(float(v) for v in greedy_key)),
            abandoned_restarts=o.abandoned_restarts,
            calibrated_required_impact=(
                k_test * _reference_impact(search_matrix, o.treatment,
                                           o.control, config.n_test)
                if calibrated else None),
            effect=effect,
        ))
    recommended = designs[best]

    # The report is the estimate on the design the search recommends, built by
    # the same code the named mode uses. Before the merge this assembled a
    # generic effect result from the group posterior, so report.cumulative and
    # report.tbr_fit were present in one mode and absent in the other.
    from ..pipeline import calibration_record, estimate
    reference = (None if blank is None
                 else _reference_impact(search_matrix, outcomes[best].treatment,
                                        outcomes[best].control, config.n_test))
    report = None
    if post_matrix is not None:
        report = estimate(config,
                          treated=list(recommended.treatment_units),
                          controls=list(recommended.control_units),
                          blank=blank, required_impact=reference)
    calibration = (None if blank is None
                   else report.calibration if report is not None
                   else calibration_record(config, blank, units, honest=True,
                                           required_impact=reference))

    return TBRResults(
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
        n_periods_scored=int(search_matrix.shape[0]),
        calibration=calibration,
        design_constraints=constraints,
        selected_units=list(recommended.treatment_units),
        power=pd.DataFrame([
            {"k": d.k, "objective_value": d.objective_value,
             **{key: d.detail[key] for key in sorted(d.detail)},
             **({} if blank is None else
                {"calibrated_required_impact": d.calibrated_required_impact})}
            for d in designs
        ]),
    )
