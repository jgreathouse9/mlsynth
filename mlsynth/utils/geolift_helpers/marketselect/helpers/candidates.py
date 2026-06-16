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

from typing import Iterable, List, Optional, Union

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


def _generate_from_ranked(
    ranked: pd.DataFrame,
    n: int,
    *,
    run_stochastic: bool,
    stochastic_mode: str,
    rng: Optional[Union[int, np.random.Generator]],
) -> List[frozenset]:
    """Core generator: anchor + neighbours from a ranked table, no constraints.

    Assumes ``n >= 1``. Used directly when there are no forced units, and on the
    free-pool ranked table when there are.
    """
    anchors = ranked.index
    num_units = len(anchors)

    augmented = np.column_stack([anchors.to_numpy(), ranked.to_numpy()])

    if not run_stochastic:
        if n > num_units:
            raise MlsynthConfigError(
                f"treatment_size ({n}) cannot exceed the number of units "
                f"({num_units}) in deterministic mode."
            )
        selected = augmented[:, :n]
    else:
        if stochastic_mode not in ("global", "per_anchor"):
            raise MlsynthConfigError(
                "stochastic_mode must be 'global' or 'per_anchor'; "
                f"got {stochastic_mode!r}."
            )
        if 2 * n > num_units:
            raise MlsynthConfigError(
                f"treatment_size ({n}) must be <= half the number of units "
                f"({num_units}) in stochastic mode."
            )
        generator = _as_generator(rng)
        tier_base = 2 * np.arange(n)
        if stochastic_mode == "global":
            columns = tier_base + generator.integers(0, 2, size=n)
            selected = augmented[:, columns]
        else:  # per_anchor
            coins = generator.integers(0, 2, size=(num_units, n))
            columns = tier_base[None, :] + coins
            selected = np.take_along_axis(augmented, columns, axis=1)

    seen: set = set()
    candidates: List[frozenset] = []
    for row in selected:
        candidate = frozenset(row.tolist())
        if candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)
    return candidates


def generate_candidate_markets(
    ranked: pd.DataFrame,
    treatment_size: int,
    *,
    to_be_treated: Optional[Iterable] = None,
    not_to_be_treated: Optional[Iterable] = None,
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
    to_be_treated : iterable, optional
        Units forced into **every** candidate test set (GeoLift's "include"
        markets). The remaining ``N - len(to_be_treated)`` slots are filled from
        the free pool. Must not exceed ``treatment_size``.
    not_to_be_treated : iterable, optional
        Units forbidden from **any** candidate (GeoLift's "exclude" markets);
        they remain available as donors. Must be disjoint from ``to_be_treated``.
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
    n = treatment_size
    if n < 1:
        raise MlsynthConfigError(f"treatment_size must be >= 1; got {n}.")

    units = set(ranked.index)
    forced_in = frozenset(to_be_treated or ())
    forced_out = frozenset(not_to_be_treated or ())

    unknown = (forced_in | forced_out) - units
    if unknown:
        raise MlsynthConfigError(
            "to_be_treated/not_to_be_treated contain units not in the panel: "
            f"{sorted(map(str, unknown))}."
        )
    overlap = forced_in & forced_out
    if overlap:
        raise MlsynthConfigError(
            "units cannot be both to_be_treated and not_to_be_treated: "
            f"{sorted(map(str, overlap))}."
        )
    if len(forced_in) > n:
        raise MlsynthConfigError(
            f"to_be_treated ({len(forced_in)}) cannot exceed treatment_size ({n})."
        )

    # No constraints: the plain anchor + neighbours generator.
    if not (forced_in or forced_out):
        return _generate_from_ranked(
            ranked, n, run_stochastic=run_stochastic,
            stochastic_mode=stochastic_mode, rng=rng,
        )

    # Forced-in units fully specify the test set: use it directly.
    if forced_in and len(forced_in) == n:
        return [frozenset(forced_in)]

    # GeoLift's generate-then-filter (``pre_test_power.R``): generate candidates
    # from the full ranked table with only the *forbidden* (``not_to_be_treated``)
    # units removed -- those stay donors -- then keep the candidates that already
    # contain every forced-in (``include``) unit. GeoLift runs
    # ``stochastic_market_selector`` ignoring ``include_markets`` and filters the
    # result afterwards, so a forced unit is never welded onto an anchor it is
    # uncorrelated with: only correlation-natural sets survive. (Re-attaching the
    # forced unit to every anchor's neighbourhood -- the previous approach --
    # manufactured low-correlation candidates GeoLift never forms.)
    pool = [u for u in ranked.index if u not in forced_out]
    if n > len(pool):
        raise MlsynthConfigError(
            f"free pool ({len(pool)}) too small for {n} market(s) "
            f"(treatment_size {n} after removing "
            f"{len(forced_out)} not_to_be_treated)."
        )

    # Restrict each anchor's neighbour list to the pool (uniform length, since
    # every unit appears in every original row).
    pool_set = set(pool)
    rows = [
        [neighbor for neighbor in ranked.loc[anchor].tolist()
         if neighbor in pool_set]
        for anchor in pool
    ]
    filtered_ranked = pd.DataFrame(
        rows, index=pd.Index(pool, name=ranked.index.name)
    )
    filtered_ranked.columns = range(1, filtered_ranked.shape[1] + 1)

    generated = _generate_from_ranked(
        filtered_ranked, n, run_stochastic=run_stochastic,
        stochastic_mode=stochastic_mode, rng=rng,
    )
    if not forced_in:                       # only not_to_be_treated given
        return generated

    kept = [c for c in generated if forced_in <= c]
    if not kept:
        raise MlsynthConfigError(
            "no correlation-natural candidate contains all to_be_treated units "
            f"{sorted(map(str, forced_in))} at treatment_size {n}; increase "
            "treatment_size or reduce to_be_treated."
        )
    return kept
