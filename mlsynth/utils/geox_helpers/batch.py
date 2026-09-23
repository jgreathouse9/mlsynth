"""Batch driver for GEOX market-selection scoring.

Loops the single-backtest :func:`~mlsynth.utils.geox_helpers.simulate.simulate_backtest`
over the full ``candidates x durations x backtests`` grid and stacks the
rows into one long table, ready for the power -> MDE -> rank aggregation.
"""

from typing import Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd

from ...exceptions import MlsynthConfigError
from .shaping import aggregate_treated, donor_matrix
from .simulate import simulate_backtest

_COLUMNS = [
    "candidate", "duration", "sim", "effect_size",
    "p_value", "placebo_mean_effect", "detected_lift", "att_error",
    "placebo_sigma", "scaled_l2", "pre_rmspe",
    "pre_rmspe_lambda", "boundary_up", "boundary_down", "investment",
]


def _simulate_candidate(candidate, Ywide, durations, n_backtests,
                        effect_sizes, *, n_periods, n_draws, seed, cpic,
                        exclude=None, engine="sdid", engine_kwargs=None,
                        alpha=0.1):
    """Every row for one candidate.

    Pure and deterministic given a fixed ``seed``, and defined at module level so
    it pickles for the joblib backend; the serial path calls it directly.

    SDID fits the *mean* of the treated units, so the candidate is aggregated
    that way for the fit. The investment volume stays a sum, since cost scales
    with total treated conversions and not with the per-market average.

    ``exclude`` drops the candidate's conflict-neighbours from its donor pool,
    so the scoring fit sees the same pool the deployed design will.
    """
    treated = aggregate_treated(Ywide, candidate, how="mean").to_numpy()
    treated_total = aggregate_treated(Ywide, candidate, how="sum").to_numpy()
    donors = donor_matrix(Ywide, candidate, exclude=exclude).to_numpy()
    n_tr = len(candidate)
    rows: List[dict] = []
    for duration in durations:
        for sim in range(1, int(n_backtests) + 1):
            for row in simulate_backtest(
                treated, donors, n_periods, duration, sim, effect_sizes,
                n_draws=n_draws, n_tr=n_tr, seed=seed, cpic=cpic,
                treated_total=treated_total, engine=engine,
                engine_kwargs=engine_kwargs, alpha=alpha,
            ):
                row["candidate"] = candidate
                rows.append(row)
    return rows


def run_simulations(
    Ywide: pd.DataFrame,
    candidates: Iterable[frozenset],
    durations: Iterable[int],
    n_backtests: int,
    effect_sizes: Iterable[float],
    *,
    n_draws: int = 200,
    seed: int = 0,
    cpic: Optional[float] = None,
    n_jobs: int = 1,
    excluded: Optional[Mapping[frozenset, Iterable]] = None,
    engine: str = "sdid",
    engine_kwargs: Optional[Mapping[str, object]] = None,
    alpha: float = 0.1,
) -> pd.DataFrame:
    """Run the simulation grid and stack the results into one long table.

    For each candidate test region, each treatment duration, and each backtest
    backtest ``sim = 1 .. n_backtests``, fit SDID once and sweep the effect
    sizes, tagging every row with its candidate.

    ``excluded`` maps a candidate to the markets barred from its donor pool (its
    spillover conflict-neighbours). Absent, every candidate keeps the full
    complement as donors.

    Returns
    -------
    pd.DataFrame
        Long-form table with one row per (candidate, duration, sim, effect
        size). Empty (with the right columns) when there are no candidates.

    Raises
    ------
    MlsynthConfigError
        If ``n_backtests`` is not a positive integer, or a backtest runs
        off the start of the panel.
    """
    if (isinstance(n_backtests, bool)
            or not isinstance(n_backtests, (int, np.integer))
            or n_backtests < 1):
        raise MlsynthConfigError(
            f"n_backtests must be a positive integer; got "
            f"{n_backtests!r}.")

    n_periods = Ywide.shape[0]
    candidates = list(candidates)
    work = dict(n_periods=n_periods, n_draws=n_draws, seed=seed, cpic=cpic,
                engine=engine, engine_kwargs=dict(engine_kwargs or {}),
                alpha=alpha)
    excluded = excluded or {}

    if n_jobs == 1 or len(candidates) <= 1:
        per_candidate = [
            _simulate_candidate(c, Ywide, durations, n_backtests,
                                effect_sizes, exclude=excluded.get(c), **work)
            for c in candidates
        ]
    else:
        # Embarrassingly parallel over candidates. joblib preserves input order
        # and each candidate is pure at the fixed seed, so any worker count
        # returns the identical table.
        from joblib import Parallel, delayed

        per_candidate = Parallel(n_jobs=n_jobs)(
            delayed(_simulate_candidate)(c, Ywide, durations, n_backtests,
                                         effect_sizes, exclude=excluded.get(c),
                                         **work)
            for c in candidates
        )

    records = [row for rows in per_candidate for row in rows]
    if not records:
        return pd.DataFrame(columns=_COLUMNS)
    return pd.DataFrame(records)[_COLUMNS]
