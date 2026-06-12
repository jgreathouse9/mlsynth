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

from .shaping import aggregate_treated, donor_matrix
from .simulate import simulate_lookback

_COLUMNS = [
    "candidate", "duration", "sim", "effect_size",
    "p_value", "att", "scaled_l2", "pre_rmspe",
]


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

    Returns
    -------
    pd.DataFrame
        Long-form table with columns ``candidate``, ``duration``, ``sim``,
        ``effect_size``, ``p_value``, ``att``, ``scaled_l2``, ``pre_rmspe`` --
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

    n_periods = Ywide.shape[0]
    records: List[dict] = []
    for candidate in candidates:
        treated = aggregate_treated(Ywide, candidate, how=how)
        donors = donor_matrix(Ywide, candidate)
        for duration in durations:
            for sim in range(1, int(lookback_window) + 1):
                for row in simulate_lookback(
                    treated, donors, n_periods, duration, sim, effect_sizes,
                    augment=augment, q=q, ns=ns, seed=seed,
                ):
                    row["candidate"] = candidate
                    records.append(row)

    if not records:
        return pd.DataFrame(columns=_COLUMNS)
    return pd.DataFrame(records)[_COLUMNS]
