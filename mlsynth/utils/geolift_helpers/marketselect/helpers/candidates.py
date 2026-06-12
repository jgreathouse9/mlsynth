"""Candidate test-market generation for GeoLift market selection.

Port of GeoLift's ``stochastic_market_selector``: turn the ranked-neighbor
table (Function 1, :func:`rank_markets_by_correlation`) into candidate
test-market *sets* of a chosen size, ready for the power-analysis scoring
stage.

The trick that keeps this tractable is GeoLift's: rather than enumerating the
astronomically many ``choose(N, k)`` subsets, anchor a candidate at each unit
and take that unit plus its nearest correlated neighbors. With ``L`` units that
yields ``L`` candidate sets instead of ``choose(L, N)``.

Each candidate set is returned as a :class:`frozenset` of unit labels — order
is irrelevant for a test region, and immutability makes whole-batch
de-duplication a one-liner.
"""

from typing import List, Optional, Union

import numpy as np
import pandas as pd

from mlsynth.exceptions import MlsynthConfigError


def _as_generator(
    rng: Optional[Union[int, np.random.Generator]]
) -> np.random.Generator:
    """Coerce ``None`` / an int seed / a Generator into a Generator."""
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def generate_candidate_markets(
    ranked: pd.DataFrame,
    treatment_size: int,
    *,
    run_stochastic: bool = False,
    stochastic_mode: str = "global",
    rng: Optional[Union[int, np.random.Generator]] = None,
) -> List[frozenset]:
    """Generate candidate test-market sets from a ranked-neighbor table.

    Parameters
    ----------
    ranked : pd.DataFrame
        Output of :func:`rank_markets_by_correlation`: index is the anchor unit
        :class:`~pandas.Index`, columns are integer similarity ranks, and each
        cell holds a neighbor unit id (self excluded, descending correlation).
    treatment_size : int
        Number of markets ``N`` in each candidate test set (anchor included).
    run_stochastic : bool, default False
        If ``False`` (deterministic), each anchor's set is the anchor plus its
        top ``N - 1`` neighbors. If ``True``, the set is built by GeoLift's
        paired-jitter walk over the ranked positions ``{0,1}, {2,3}, ...``,
        picking one position per pair.
    stochastic_mode : {"global", "per_anchor"}, default "global"
        Only used when ``run_stochastic`` is ``True``.

        - ``"global"`` (faithful to GeoLift): one coin per rank-tier, applied to
          every anchor at once.
        - ``"per_anchor"`` (corrected): an independent coin per (anchor, tier),
          so anchors jitter independently. GeoLift's global coin is effectively
          a bug; this is the intended behavior, kept opt-in for now.
    rng : int, numpy.random.Generator, or None, optional
        Seed or generator for the stochastic draws (reproducibility).

    Returns
    -------
    list of frozenset
        De-duplicated candidate test-market sets, each a ``frozenset`` of
        ``treatment_size`` unit labels.

    Raises
    ------
    MlsynthConfigError
        If ``treatment_size < 1``; if it exceeds the number of units
        (deterministic) or half the number of units (stochastic, whose paired
        walk indexes up to position ``2N - 1``); or if ``stochastic_mode`` is
        not recognized.
    """
    anchors = ranked.index
    num_units = len(anchors)
    n = treatment_size

    if n < 1:
        raise MlsynthConfigError(f"treatment_size must be >= 1; got {n}.")

    # Reconstruct GeoLift's per-anchor "similarity_matrix" row: the anchor at
    # position 0, then its ranked neighbors. Shape (num_units, num_units).
    augmented = np.column_stack([anchors.to_numpy(), ranked.to_numpy()])

    if not run_stochastic:
        if n > num_units:
            raise MlsynthConfigError(
                f"treatment_size ({n}) cannot exceed the number of units "
                f"({num_units}) in deterministic mode."
            )
        # Anchor + top (N-1) neighbors: the first N consecutive positions.
        selected = augmented[:, :n]
    else:
        if stochastic_mode not in ("global", "per_anchor"):
            raise MlsynthConfigError(
                "stochastic_mode must be 'global' or 'per_anchor'; "
                f"got {stochastic_mode!r}."
            )
        # The paired walk indexes positions up to 2N - 1.
        if 2 * n > num_units:
            raise MlsynthConfigError(
                f"treatment_size ({n}) must be <= half the number of units "
                f"({num_units}) in stochastic mode."
            )
        generator = _as_generator(rng)
        tier_base = 2 * np.arange(n)  # positions {0,2,4,...}; pair is base+{0,1}
        if stochastic_mode == "global":
            columns = tier_base + generator.integers(0, 2, size=n)
            selected = augmented[:, columns]
        else:  # per_anchor
            coins = generator.integers(0, 2, size=(num_units, n))
            columns = tier_base[None, :] + coins
            selected = np.take_along_axis(augmented, columns, axis=1)

    # Canonicalize each row into an order-independent set and de-duplicate,
    # preserving first-seen order for a stable, reproducible result.
    seen: set = set()
    candidates: List[frozenset] = []
    for row in selected:
        candidate = frozenset(row.tolist())
        if candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)
    return candidates
