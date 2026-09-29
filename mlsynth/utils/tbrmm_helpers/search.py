"""Algorithm 1: the matched-markets hill climb (Au 2018, section 4).

Three groups, and why. TBR sums outcomes within each group and regresses one
aggregate on the other, so a geo's weight is its membership: one inside a group,
zero outside. A geo whose series moves independently of the treatment group
cannot be downweighted the way a synthetic control downweights a poor donor. It
enters the control aggregate at full weight and inflates the residual scale every
candidate's detectable effect depends on. Holding it out of the experiment is the
only zero weight the method has, which is what the third group is for.

That makes this a partition problem over 3^n labellings, some 5e47 at a hundred
geos, so the search is a heuristic and carries no optimality guarantee. Algorithm
1 alternates two routines:

* matching -- treatment group fixed, toggle the single control membership that
  most improves the objective, stopping when no single toggle does. The result is
  a local optimum with respect to one-geo moves, so a pair of geos that helps
  only when swapped together is not reachable from here;
* augmentation -- control group fixed, add the treatment-eligible geo that most
  improves the objective.

One design is recorded per treatment size k, because the objective is not
monotone in k: a geo added to the treatment group brings its volume to the
treatment aggregate and takes it out of the pool available to the control
aggregate. Which size to run is the advertiser's call, so every size is reported.

Au's pseudocode has a gap. ``R_uad`` excludes ``i in G*_trt``, so a geo promoted
to treatment while still in ``G_ctl`` can never be toggled out of the control
group, and it then enters both aggregates. That shares its variance between the
two series, pulls the slope toward one and shrinks the residual scale, so the
split scores well because it is double counting. Treatment membership wins here:
a geo entering the treatment group leaves the control group in the same step.

Eligibility is Au's ``A_i``, a subset of the three roles per geo. It shapes the
search space instead of filtering its output. For a geo outside the treatment
group, control eligibility with unassigned eligibility makes it free to toggle;
control eligibility without unassigned eligibility pins it into the control
group; no control eligibility pins it out.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, FrozenSet, List, Sequence, Set, Tuple

import numpy as np

from ...exceptions import MlsynthDataError
from .objective import SplitScore, score_split

TREATMENT = "treatment"
CONTROL = "control"
UNASSIGNED = "unassigned"
ROLES: Tuple[str, str, str] = (TREATMENT, CONTROL, UNASSIGNED)

#: Two pretest periods leave no degrees of freedom for a residual scale, and
#: every gate downstream divides by it.
MIN_SCORING_PERIODS = 3


@dataclass(frozen=True)
class SearchOutcome:
    """One treatment size's recommended partition and the climb that found it."""

    k: int
    treatment: List[int]
    control: List[int]
    unassigned: List[int]
    score: SplitScore
    trace: List[Tuple[float, ...]]
    converged: bool
    evaluations: int


class _Scorer:
    """Scores a partition, memoised on the two memberships.

    The climb revisits partitions -- the step that fails to improve rescores the
    current one, and augmentation reaches states matching already visited. The
    memo is keyed on the aggregates' membership and not on the walk, so it is
    exact.
    """

    def __init__(self, y_matrix: np.ndarray, objective: str, n_test: int) -> None:
        self._y = y_matrix
        self._objective = objective
        self._n_test = n_test
        self._memo: Dict[Tuple[FrozenSet[int], FrozenSet[int]], SplitScore] = {}
        self.evaluations = 0

    def __call__(self, treatment: Sequence[int], control: Sequence[int]) -> SplitScore:
        key = (frozenset(treatment), frozenset(control))
        hit = self._memo.get(key)
        if hit is not None:
            return hit
        score = score_split(
            self._y[:, sorted(key[0])].sum(axis=1),
            self._y[:, sorted(key[1])].sum(axis=1),
            objective=self._objective, n_test=self._n_test)
        self._memo[key] = score
        self.evaluations += 1
        return score


def _read_eligibility(eligibility: Sequence[FrozenSet[str]]) -> Dict[str, List[int]]:
    """The role pools, with the geos no role admits reported.

    A geo eligible for nothing has no place in the partition, and silently
    dropping it would answer a question the advertiser did not ask.
    """
    unusable = [j for j, roles in enumerate(eligibility) if not roles]
    if unusable:
        raise MlsynthDataError(
            f"geo(s) at column(s) {unusable} are eligible for no group; every "
            f"geo has to be allowed in at least one of {list(ROLES)}.")
    unknown = sorted({r for roles in eligibility for r in roles} - set(ROLES))
    if unknown:
        raise MlsynthDataError(
            f"eligibility names role(s) {unknown}; the roles are {list(ROLES)}.")
    return {
        "forced": [j for j, r in enumerate(eligibility) if r == frozenset({TREATMENT})],
        "treatment": [j for j, r in enumerate(eligibility) if TREATMENT in r],
        "control": [j for j, r in enumerate(eligibility) if CONTROL in r],
    }


def _check_feasible(pools: Dict[str, List[int]], max_treatment_size: int) -> None:
    """Refuse a K no partition can satisfy, before any climbing.

    The condition is exact, which is what lets the augmentation routine treat
    running out of admissible candidates as unreachable. A treatment group of K
    exists alongside a non-empty control group when K geos are treatment eligible
    and some control-eligible geo can stay outside them. When a control-eligible
    geo is not treatment eligible it is always available to stay outside, so K
    may use the whole treatment pool; when every control-eligible geo is also
    treatment eligible, one of them has to be held back.
    """
    n_treatment = len(pools["treatment"])
    forced = len(pools["forced"])
    if not pools["control"]:
        raise MlsynthDataError(
            "no geo is eligible for the control group, so no design has a "
            "control aggregate to regress on.")
    if max_treatment_size < forced:
        raise MlsynthDataError(
            f"max_treatment_size is {max_treatment_size} but {forced} geo(s) are "
            f"eligible for treatment alone and so are forced into every "
            f"treatment group; K cannot be below that count.")
    spare = set(pools["control"]) - set(pools["treatment"])
    room = n_treatment if spare else n_treatment - 1
    if max_treatment_size > room:
        raise MlsynthDataError(
            f"a treatment group of {max_treatment_size} geos is not attainable: "
            f"{n_treatment} geo(s) are eligible for treatment and at least one "
            f"control-eligible geo has to stay out of it, which leaves room for "
            f"{room}.")


def _match(scorer: _Scorer, treatment: Sequence[int], control: Set[int],
           toggleable: Sequence[int]) -> Tuple[Set[int], List[Tuple[float, ...]], bool]:
    """Toggle one control membership at a time while the objective improves.

    The trace opens with the score of the partition as handed over, so a climb
    that improves nothing still records where it stood, and the returned trace is
    never empty.
    """
    control = set(control)
    trace = [scorer(treatment, control).key]
    while True:
        current = scorer(treatment, control)
        best_score, best_geo = None, None
        for geo in toggleable:
            candidate = control ^ {geo}
            if not candidate:                     # never empty the control group
                continue
            score = scorer(treatment, candidate)
            if best_score is None or score.key > best_score.key:
                best_score, best_geo = score, geo
        if best_geo is None or best_score.key <= current.key:
            return control, trace, True
        control = control ^ {best_geo}
        trace.append(best_score.key)


def _augment(scorer: _Scorer, treatment: Sequence[int], control: Set[int],
             pools: Dict[str, List[int]], *, from_pool: bool = False
             ) -> Tuple[List[int], Set[int]]:
    """Add the treatment-eligible geo that most improves the objective.

    A candidate already sitting in the control group leaves it in the same step,
    which is the resolution of Au's ``R_uad`` gap. When that empties the control
    group the pool is re-derived from every control-eligible geo the new
    treatment group leaves free, so a climb that pruned the control group down to
    one geo does not block the next size.

    ``from_pool`` scores every candidate against the whole control pool instead
    of against the incumbent control group. See :func:`greedy_search`.
    """
    control_pool = set(pools["control"])
    best: Tuple[SplitScore, int, Set[int]] | None = None
    for geo in pools["treatment"]:
        if geo in treatment:
            continue
        if from_pool:
            candidate_control = control_pool - set(treatment) - {geo}
        else:
            candidate_control = set(control) - {geo}
        if not candidate_control:
            candidate_control = control_pool - set(treatment) - {geo}
        if not candidate_control:
            continue
        candidate_treatment = list(treatment) + [geo]
        score = scorer(candidate_treatment, candidate_control)
        if best is None or score.key > best[0].key:
            best = (score, geo, candidate_control)
    if best is None:   # pragma: no cover - _check_feasible is exact, so some
        raise MlsynthDataError(          # candidate always leaves a control geo
            f"no treatment-eligible geo can join a treatment group of "
            f"{len(treatment)} while leaving a control group behind.")
    _, geo, candidate_control = best
    return list(treatment) + [geo], candidate_control


def greedy_search(y_matrix: np.ndarray, eligibility: Sequence[FrozenSet[str]], *,
                  max_treatment_size: int, n_test: int, objective: str,
                  control_start: str = "carried") -> List[SearchOutcome]:
    """Algorithm 1 over the columns of ``y_matrix``, one outcome per size.

    ``y_matrix`` is periods by geos and holds the scoring window only. Columns
    index geos, and ``eligibility[j]`` is geo ``j``'s ``A_i``. Geos are visited in
    column order and a toggle has to improve strictly, so the walk is
    deterministic.

    ``control_start`` decides where the control group search begins at each
    treatment size. ``"carried"`` hands the previous size's matched group to the
    next one, which is Algorithm 1 and what the reference implementation does.
    ``"pool"`` re-derives it from every control-eligible geo the treatment group
    leaves free. Matching is a single-toggle climb, so its answer depends on where
    it starts: on the GeoLift panel the same treatment group reaches a different
    local optimum from a different start in 16 of 19 cases, and the two answers
    can straddle a rounded-correlation boundary, which is the element of the key
    above the detectable impact.
    """
    y_matrix = np.asarray(y_matrix, dtype=float)
    if y_matrix.ndim != 2:
        raise MlsynthDataError(
            f"the scoring window has to be periods by geos; got shape "
            f"{y_matrix.shape}.")
    n_periods, n_units = y_matrix.shape
    if len(eligibility) != n_units:
        raise MlsynthDataError(
            f"eligibility covers {len(eligibility)} geo(s) but the panel has "
            f"{n_units}.")
    if control_start not in ("carried", "pool", "best"):
        raise MlsynthDataError(
            f"control_start is {control_start!r}; it has to be 'carried', "
            f"'pool' or 'best'.")
    if control_start == "best":
        runs = [greedy_search(y_matrix, eligibility,
                              max_treatment_size=max_treatment_size,
                              n_test=n_test, objective=objective,
                              control_start=start)
                for start in ("carried", "pool")]
        # ``max`` keeps the first argument on a tie, so an even contest returns
        # the reference walk's design and the option costs nothing but time.
        return [replace(max(a, b, key=lambda o: o.score.key),
                        evaluations=a.evaluations + b.evaluations)
                for a, b in zip(*runs)]
    if n_periods < MIN_SCORING_PERIODS:
        raise MlsynthDataError(
            f"scoring a candidate split needs at least {MIN_SCORING_PERIODS} "
            f"periods to leave degrees of freedom for a residual scale; the "
            f"window has {n_periods}.")

    pools = _read_eligibility(eligibility)
    _check_feasible(pools, max_treatment_size)

    scorer = _Scorer(y_matrix, objective, n_test)
    treatment: List[int] = list(pools["forced"])
    control: Set[int] = set(pools["control"]) - set(treatment)
    outcomes: List[SearchOutcome] = []

    from_pool = control_start == "pool"
    for k in range(max(len(pools["forced"]), 1), max_treatment_size + 1):
        spent = scorer.evaluations
        while len(treatment) < k:
            treatment, control = _augment(scorer, treatment, control, pools,
                                          from_pool=from_pool)
        toggleable = [j for j in pools["control"]
                      if j not in treatment and UNASSIGNED in eligibility[j]]
        start = (set(pools["control"]) - set(treatment)) if from_pool else control
        control, trace, converged = _match(scorer, treatment, start, toggleable)
        assigned = set(treatment) | control
        outcomes.append(SearchOutcome(
            k=k, treatment=list(treatment), control=sorted(control),
            unassigned=sorted(set(range(n_units)) - assigned),
            score=scorer(treatment, control), trace=trace,
            converged=converged, evaluations=scorer.evaluations - spent))
    return outcomes
