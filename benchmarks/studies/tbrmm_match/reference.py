"""Named wrappers over google/matched_markets.

Every function here is one statement about the method, so the comparison code
reads as econometrics and not as that package's API. The reference is Apache 2.0
and is not vendored; point ``MLSYNTH_MATCHED_MARKETS`` at a checkout.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Sequence, Set, Tuple

import pandas as pd

ENV_VAR = "MLSYNTH_MATCHED_MARKETS"
DEFAULT_PATH = "/home/user/google/matched_markets"

Key = Tuple[float, ...]


def checkout() -> str:
    """Where the reference lives, from the environment or the default path."""
    return os.environ.get(ENV_VAR, DEFAULT_PATH)


def available() -> bool:
    """True when the reference can be imported from :func:`checkout`."""
    try:
        _modules()
    except Exception:
        return False
    return True


def _modules():
    """The four reference modules this study drives, imported once."""
    path = checkout()
    if path not in sys.path:
        sys.path.insert(0, path)
    from matched_markets.methodology import (  # noqa: E402
        tbrmatchedmarkets, tbrmmdata, tbrmmdesignparameters, tbrmmdiagnostics,
        tbrmmscore)
    return (tbrmmdata.TBRMMData, tbrmmdesignparameters.TBRMMDesignParameters,
            tbrmatchedmarkets.TBRMatchedMarkets, tbrmmdiagnostics.TBRMMDiagnostics,
            tbrmmscore.TBRMMScore)


def long_panel(csv_path: str) -> pd.DataFrame:
    """The shipped GeoLift panel under the column names the reference wants."""
    return pd.read_csv(csv_path).rename(columns={"location": "geo"})


def engine(panel: pd.DataFrame, *, n_test: int, k_max: int, n_pretest: int):
    """A configured ``TBRMatchedMarkets`` over ``panel``.

    ``n_pretest`` is the reference's ``n_pretest_max``, which truncates the panel
    to its last periods inside the constructor. Reproducing a run means applying
    the same truncation on the mlsynth side.
    """
    data_cls, params_cls, markets_cls, _, _ = _modules()
    params = params_cls(n_test=n_test, iroas=1.0, treatment_geos_range=(1, k_max),
                        n_pretest_max=n_pretest, n_designs=k_max + 1)
    return markets_cls(data_cls(panel, response_column="Y"), params)


def geo_order(eng) -> List[str]:
    """The geo labels in the reference's own order, largest volume first.

    The reference indexes geos by position in this list, so every set it
    exposes is a set of integers. The functions below translate, and nothing
    outside this module handles an index.
    """
    return list(eng.data.geo_index)


def to_indices(eng, geos: Sequence[str]) -> Set[int]:
    """Geo labels as the positional indices the reference works in."""
    order = {geo: i for i, geo in enumerate(geo_order(eng))}
    return {order[geo] for geo in geos}


def to_labels(eng, indices) -> List[str]:
    """Positional indices back to sorted geo labels."""
    order = geo_order(eng)
    return sorted(order[i] for i in indices)


def treatment_pool(eng) -> Set[str]:
    """Geos the reference will consider treating."""
    return set(to_labels(eng, eng.geo_assignments.t))


def control_pool(eng) -> Set[str]:
    """Geos the reference will consider as controls."""
    return set(to_labels(eng, eng.geo_assignments.c))


def unassigned_pool(eng) -> Set[str]:
    """Geos the reference will consider holding out of the experiment."""
    return set(to_labels(eng, eng.geo_assignments.x))


def score(eng, treatment: Sequence[str], control: Sequence[str]) -> Key:
    """The reference's own score for one split, as a plain tuple.

    This is the reference's ``TBRMMScore`` over its ``TBRMMDiagnostics``, so it
    carries the reference's aggregation and its truncation, not a reimplementation
    of either.
    """
    _, _, _, diagnostics_cls, score_cls = _modules()
    diag = diagnostics_cls(
        eng.data.aggregate_time_series(to_indices(eng, treatment)),
        eng.parameters)
    diag.x = eng.data.aggregate_time_series(to_indices(eng, control))
    return tuple(float(v) for v in score_cls(diag).score)


def designs(eng) -> Dict[int, Tuple[List[str], List[str], Key]]:
    """One recommended design per treatment size, keyed by size."""
    out: Dict[int, Tuple[List[str], List[str], Key]] = {}
    for design in eng.greedy_search():
        treatment = sorted(design.treatment_geos)
        out[len(treatment)] = (treatment, sorted(design.control_geos),
                               tuple(float(v) for v in design.score.score))
    return out
