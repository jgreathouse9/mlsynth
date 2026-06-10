"""Coverage / stratification quotas for LEXSCM treated-unit selection.

A per-unit **stratum** label (region / tier / segment) plus optional quotas on
the treated set:

* ``min_per_stratum`` -- the treated ``m``-tuple must contain at least this many
  units from **every stratum that has a candidate** (coverage: "test in every
  region");
* ``max_per_stratum`` -- at most this many treated units from any one stratum.

Like the spillover conflict graph, the stratum array is **aligned to the
IndexSet** (the single source of truth for unit identity), and the quotas restrict
the admissible treated supports -- they live on the same Stage-1 combinatorial
layer as the cardinality ``||w||_0 = m`` and never enter the inner weight QP.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ...exceptions import MlsynthConfigError


def build_strata(unit_index: Any,
                 stratum_of: Optional[Mapping[Any, Any]]) -> Optional[np.ndarray]:
    """Integer stratum code per unit, in IndexSet order.

    Units with a missing / ``NaN`` stratum get code ``-1`` and are excluded from
    every quota. Returns ``None`` when ``stratum_of`` is ``None`` (unstratified).
    """
    if stratum_of is None:
        return None
    labels = unit_index.labels
    codes = np.full(len(labels), -1, dtype=int)
    seen: dict = {}
    for i, lab in enumerate(labels):
        s = stratum_of.get(lab, None)
        if s is None or (isinstance(s, float) and np.isnan(s)):
            continue
        if s not in seen:
            seen[s] = len(seen)
        codes[i] = seen[s]
    return codes


def required_codes(codes: np.ndarray, cand: Sequence[int]) -> np.ndarray:
    """Distinct stratum codes present among the candidate units (coverage targets)."""
    c = codes[np.asarray(cand, dtype=int)]
    return np.unique(c[c >= 0])


def within_max(codes: Optional[np.ndarray], S: Sequence[int],
               max_per: Optional[int]) -> bool:
    """Partial-safe: no stratum exceeds ``max_per`` in ``S`` (True if no quota)."""
    if codes is None or max_per is None:
        return True
    cnt = Counter(int(codes[i]) for i in S if codes[i] >= 0)
    return all(v <= max_per for v in cnt.values())


def satisfies(codes: Optional[np.ndarray], S: Sequence[int],
              min_per: Optional[int], max_per: Optional[int],
              required: Sequence[int]) -> bool:
    """Full quota check for a size-``m`` tuple: max quota AND min coverage."""
    if codes is None:
        return True
    cnt = Counter(int(codes[i]) for i in S if codes[i] >= 0)
    if max_per is not None and any(v > max_per for v in cnt.values()):
        return False
    if min_per is not None:
        for k in required:
            if cnt.get(int(k), 0) < min_per:
                return False
    return True


def satisfies_many(codes: Optional[np.ndarray], combs: np.ndarray,
                   min_per: Optional[int], max_per: Optional[int],
                   required: Sequence[int]) -> np.ndarray:
    """Vectorised :func:`satisfies` over an ``(N, m)`` array of tuples."""
    n = len(combs)
    if codes is None or (min_per is None and max_per is None):
        return np.ones(n, dtype=bool)
    K = int(codes.max()) + 1 if codes.size and codes.max() >= 0 else 0
    cc = codes[combs]                                  # (N, m) in [-1, K-1]
    counts = np.zeros((n, max(K, 1)), dtype=int)
    for k in range(K):
        counts[:, k] = (cc == k).sum(1)
    ok = np.ones(n, dtype=bool)
    if max_per is not None:
        ok &= (counts <= max_per).all(1)
    if min_per is not None and len(required):
        ok &= (counts[:, np.asarray(required, dtype=int)] >= min_per).all(1)
    return ok


def check_feasible(codes: Optional[np.ndarray], cand: Sequence[int], m: int,
                   min_per: Optional[int], max_per: Optional[int]) -> None:
    """Raise :class:`MlsynthConfigError` if the quotas admit no size-``m`` tuple."""
    if codes is None:
        return
    cand = np.asarray(cand, dtype=int)
    req = required_codes(codes, cand)
    if min_per is not None:
        if min_per * len(req) > m:
            raise MlsynthConfigError(
                f"min_per_stratum={min_per} across {len(req)} candidate strata "
                f"needs at least {min_per * len(req)} treated units, but m={m}."
            )
        for k in req:
            avail = int((codes[cand] == k).sum())
            if avail < min_per:
                raise MlsynthConfigError(
                    f"A candidate stratum has only {avail} unit(s) but "
                    f"min_per_stratum={min_per}."
                )
    if max_per is not None:
        cap = int((codes[cand] < 0).sum())             # unstratified: uncapped
        for k in req:
            cap += min(int((codes[cand] == k).sum()), max_per)
        if cap < m:
            raise MlsynthConfigError(
                f"max_per_stratum={max_per} caps the treatable capacity at "
                f"{cap} < m={m}."
            )
