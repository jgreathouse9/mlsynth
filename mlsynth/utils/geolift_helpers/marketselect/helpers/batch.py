"""Batch driver for GeoLift market-selection scoring.

``run_simulations`` loops the single-placement :func:`simulate_lookback` over the
full ``candidates x durations x lookback-windows`` grid and stacks the rows into
one long table (the p-value cube), ready for vectorized power -> MDE -> rank
aggregation.
"""

from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from mlsynth.exceptions import MlsynthConfigError

from .constraints import ConflictGraph, conflict_neighbors
from .shaping import aggregate_treated, donor_matrix
from .simulate import simulate_lookback

_COLUMNS = [
    "candidate", "duration", "sim", "effect_size",
    "p_value", "placebo_mean_effect", "detected_lift", "scaled_l2", "pre_rmspe",
    "investment",
]


def _simulate_candidate(
    candidate, Ywide, durations, lookback_window, effect_sizes, *,
    fit_how, n_periods, augment, q, ns, seed, conformal_type, fixed_effects,
    cpic, conflict,
):
    """All rows for one candidate (pure + deterministic given a fixed ``seed``).

    Factored out as a module-level function so it is picklable for the joblib
    parallel backend; the serial path calls it directly.
    """
    treated = aggregate_treated(Ywide, candidate, how=fit_how)
    # CPIC investment uses the *summed* treated volume (GeoLift's sum(Y[D==1])),
    # independent of the fit aggregation.
    treated_total = aggregate_treated(Ywide, candidate, how="sum").to_numpy()
    exclude = conflict_neighbors(candidate, conflict) if conflict else None
    donors = donor_matrix(Ywide, candidate, exclude=exclude)
    rows: List[dict] = []
    for duration in durations:
        for sim in range(1, int(lookback_window) + 1):
            for row in simulate_lookback(
                treated, donors, n_periods, duration, sim, effect_sizes,
                augment=augment, q=q, ns=ns, seed=seed, conformal_type=conformal_type,
                fixed_effects=fixed_effects, cpic=cpic, treated_total=treated_total,
            ):
                row["candidate"] = candidate
                rows.append(row)
    return rows


def run_simulations(
    Ywide: pd.DataFrame,
    candidates: Iterable[frozenset],
    durations: Iterable[int],
    lookback_window: int,
    effect_sizes: Iterable[float],
    *,
    how: str = "sum",
    augment: Optional[str] = "ridge",
    q: float = 1.0,
    ns: int = 1000,
    seed: int = 0,
    conformal_type: str = "iid",
    fixed_effects: bool = False,
    cpic: Optional[float] = None,
    conflict: Optional[ConflictGraph] = None,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Run the simulation grid and stack the results into one long table.

    For each candidate test-market set, each treatment duration, and each
    lookback placement ``sim = 1 .. lookback_window``, fit once and sweep the
    effect sizes (:func:`simulate_lookback`), tagging every row with its
    candidate.

    Parameters
    ----------
    Ywide : pd.DataFrame
        The balanced wide panel (``geoex_dataprep(...)["Ywide"]``).
    candidates : iterable of frozenset
        Candidate test-market sets (from ``generate_candidate_markets``).
    durations : iterable of int
        Treatment durations to scan.
    lookback_window : int
        Number of backward placements per (candidate, duration).
    effect_sizes : iterable of float
        Effect sizes to inject.
    how, augment, q, ns, seed
        Forwarded to the shaping / fit / inference layers.
    conflict : ConflictGraph, optional
        Market interference graph; when given, each candidate's conflict-
        neighbours are excluded from its donor pool (spillover exclusion).

    Returns
    -------
    pd.DataFrame
        Long-form table with columns ``candidate``, ``duration``, ``sim``,
        ``effect_size``, ``p_value``, ``placebo_mean_effect``, ``scaled_l2``, ``pre_rmspe`` --
        one row per (candidate, duration, sim, effect size).

    Raises
    ------
    MlsynthConfigError
        If ``lookback_window`` is not a positive integer, or a placement runs
        off the start of the panel (propagated from :func:`simulate_lookback`).
    """
    if (
        isinstance(lookback_window, bool)
        or not isinstance(lookback_window, (int, np.integer))
        or lookback_window < 1
    ):
        raise MlsynthConfigError(
            f"lookback_window must be a positive integer; got {lookback_window!r}."
        )

    # augsynth fixedeff fits the mean of the treated units; keep the simulation
    # consistent with the realized report so power/MDE reflect the same model.
    fit_how = "mean" if fixed_effects else how
    n_periods = Ywide.shape[0]
    candidates = list(candidates)
    work = dict(
        fit_how=fit_how, n_periods=n_periods, augment=augment, q=q, ns=ns,
        seed=seed, conformal_type=conformal_type, fixed_effects=fixed_effects,
        cpic=cpic, conflict=conflict,
    )
    if n_jobs == 1:
        per_candidate = [
            _simulate_candidate(c, Ywide, durations, lookback_window,
                                effect_sizes, **work)
            for c in candidates
        ]
    else:
        # Embarrassingly parallel over candidates; joblib preserves input order,
        # and each candidate is pure + uses the fixed ``seed``, so the stacked
        # table is identical to the serial run -- only faster.
        from joblib import Parallel, delayed

        per_candidate = Parallel(n_jobs=n_jobs)(
            delayed(_simulate_candidate)(c, Ywide, durations, lookback_window,
                                         effect_sizes, **work)
            for c in candidates
        )
    records = [row for rows in per_candidate for row in rows]
    if not records:
        return pd.DataFrame(columns=_COLUMNS)
    return pd.DataFrame(records)[_COLUMNS]
