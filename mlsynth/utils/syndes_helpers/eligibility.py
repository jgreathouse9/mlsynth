"""Per-unit attribute and size-band eligibility primitives for SYNDES.

Two pure, label-based helpers used by :mod:`~mlsynth.utils.syndes_helpers.restrictions`
to translate the SYNDES design vocabulary (cluster / stratum / size restrictions)
into constraints on the assignment vector ``D``:

* :func:`unit_attribute_map` resolves a per-unit-constant attribute column (a
  market's region, tier, or size) to a ``{unit: value}`` map;
* :func:`eligible_by_size` returns the units whose size falls inside a treatment
  band ``[min_size, max_size]``.

Both are self-contained (they touch only the long panel labels) so the SYNDES
constraint layer owns its eligibility primitives outright.
"""
from __future__ import annotations

from typing import Dict, Hashable, Iterable, Mapping, Optional

import pandas as pd

from ...exceptions import MlsynthConfigError, MlsynthDataError


def unit_attribute_map(
    df: pd.DataFrame, unit_id_column_name: str, attr_col: str
) -> Dict[Hashable, Hashable]:
    """Resolve a per-unit-constant attribute column to a ``{unit: value}`` map.

    Cluster / stratum / size constraints all read a per-unit attribute (a market's
    region, tier, or size) from a column of the long panel. The attribute is a
    property of the market, so it must be constant over that market's rows.

    Raises
    ------
    MlsynthConfigError
        If ``attr_col`` is not a column of ``df``.
    MlsynthDataError
        If the attribute varies within any unit.
    """
    if attr_col not in df.columns:
        raise MlsynthConfigError(f"column {attr_col!r} not found in df.")
    grp = df.groupby(unit_id_column_name)[attr_col]
    varying = grp.nunique(dropna=False)
    if (varying > 1).any():
        bad = list(varying[varying > 1].index)[:5]
        raise MlsynthDataError(
            f"column {attr_col!r} must be constant per unit; it varies for "
            f"units {bad}."
        )
    return {unit: value for unit, value in grp.first().items()}


def eligible_by_size(
    units: Iterable[Hashable],
    size_map: Mapping[Hashable, float],
    *,
    min_size: Optional[float] = None,
    max_size: Optional[float] = None,
) -> frozenset:
    """The units eligible for treatment under a size band ``[min, max]``.

    A treated-unit size band (both bounds inclusive): the floor is a power /
    operational minimum, the ceiling encodes synthesizability (a market far
    larger than the donors cannot sit inside their convex hull -- the scaled-L2
    imbalance would blow up). Units outside the band stay available as donors;
    only their treatment eligibility is removed. A unit with no recorded size is
    treated as ineligible.
    """
    out: set = set()
    for u in units:
        if u not in size_map:
            continue
        s = size_map[u]
        if min_size is not None and s < min_size:
            continue
        if max_size is not None and s > max_size:
            continue
        out.add(u)
    return frozenset(out)
