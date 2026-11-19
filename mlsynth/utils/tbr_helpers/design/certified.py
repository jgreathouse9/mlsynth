"""TBR's certified design search, ``search="exact"`` (#965).

:func:`exact_search` runs :func:`~mlsynth.utils.tbr_helpers.design.exact.treatment.certify_treatment`
at every treatment size from the forced geos' count (at least one) to
``max_treatment_size``, each under its own time limit, and returns each size's
design as the :class:`~mlsynth.utils.tbr_helpers.design.search.SearchOutcome` the
greedy returns, scored by the library's own
:func:`~mlsynth.utils.tbr_helpers.design.objective.score_split`, beside the
certificate that says how far it is proven from the best design the rules allow.

The sizes follow the greedy's: a size the rules make impossible (a stratum's
minimum needs more treated geos than it holds) has no design and is not reported,
nor is a size where the search finds none to rank: the rules rule out every
treatment group of the size, or every design they allow fits the treated aggregate
exactly, which has no residual scale and so no detectable impact. A problem with no
size left raises.
"""
from __future__ import annotations

import os
from typing import FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from ....exceptions import MlsynthDataError
from .exact.treatment import TreatmentCertificate, certify_treatment
from .objective import score_split
from .rules import DesignRules
from .search import SearchOutcome, _check_feasible, _read_eligibility


def exact_search(y_matrix: np.ndarray, eligibility: Sequence[FrozenSet[str]], *,
                 max_treatment_size: int, n_test: int,
                 rules: Optional[DesignRules] = None, gap: float = 0.0,
                 time_limit: Optional[float] = 120.0,
                 n_jobs: Optional[int] = None,
                 ) -> List[Tuple[SearchOutcome, TreatmentCertificate]]:
    """The certified design at each treatment size, over the columns of
    ``y_matrix`` (the scoring window, periods by geos).

    ``gap``, ``time_limit`` and ``n_jobs`` are ``certify_treatment``'s, the last
    defaulting to every core. The search scores the whole of ``y_matrix``: the
    caller holds back any blank periods before calling.
    """
    y_matrix = np.asarray(y_matrix, dtype=float)
    pools = _read_eligibility(eligibility)
    _check_feasible(pools, max_treatment_size)
    forced = pools["forced"]
    if rules is not None:
        rules.audit(forced, max_treatment_size)
    jobs = (os.cpu_count() or 1) if n_jobs is None else int(n_jobs)

    out, empty = [], []
    for k in range(max(len(forced), 1), max_treatment_size + 1):
        if rules is not None:
            try:
                rules.audit(forced, k)
            except MlsynthDataError:
                continue            # the rules need more treated geos than k
        cert = certify_treatment(y_matrix, k, n_test=n_test, gap=gap, n_jobs=jobs,
                                 blank_periods=0, time_limit=time_limit,
                                 eligibility=list(eligibility), rules=rules)
        if cert.treatment is None:
            # No design to rank at k: the rules rule out every treatment group, or
            # every design they allow fits exactly (no residual scale). The
            # certificate does not say which, so neither does the error below.
            empty.append(k)
            continue
        T, C = sorted(cert.treatment), sorted(cert.controls)
        score = score_split(y_matrix[:, T].sum(axis=1), y_matrix[:, C].sum(axis=1),
                            objective="relative", n_test=n_test)
        U = sorted(set(range(y_matrix.shape[1])) - set(T) - set(C))
        out.append((SearchOutcome(k=k, treatment=T, control=C, unassigned=U,
                                  score=score, trace=[score.key],
                                  converged=cert.certified, evaluations=cert.groups),
                    cert))
    # Size max_treatment_size passed the audit above, so it lands in out or in
    # empty; a problem whose rules admit no design at any size raises earlier,
    # inside certify_treatment's start, naming the rule.
    if not out:
        raise MlsynthDataError(
            f"no treatment size has a design to rank: at size(s) {empty} the "
            f"rules allow no design, or every design they allow fits the treated "
            f"aggregate exactly and so has no residual scale or smallest "
            f"detectable impact (the panel's geos move in proportion to one "
            f"another on the scoring window).")
    return out
