"""The library's enumeration with its incumbent seeded, or its bound two-sided.

:func:`run_search` calls :func:`mlsynth.utils.marex_helpers.exact.search_cluster`
itself, with two choices opened up by swapping one of its module globals for
the duration of the call:

- ``seed``: the search's ``_TopK`` starts holding the given design, so its
  cutoff is that design's objective from the first candidate on. Without a
  seed the cutoff starts at infinity, as in the library.
- ``two_sided``: every candidate's bound gains the closed-form bound of its
  complement. The library prices a candidate by the closed form of its treated
  fit plus a floor under every control fit (zero for the base design, whose
  target the whole cluster reproduces). The control fit of the base design is
  the same fit over the complement, so the same closed form, applied to the
  complement, bounds it from below; the floor is the larger of the two. Donor
  exclusions only shrink a control pool, so the bound would stay valid under
  them too; this study runs without restrictions.

Everything else, the pricing, the buffer, the second pass, the fits, the
margin and the report, is the library's own code. The search reports how many
candidates reached a treated fit (``walked``) and how many were fitted to the
end (``fitted``, the designs whose control fit was computed).
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Dict, Optional, Sequence

import numpy as np

import mlsynth.utils.marex_helpers.exact as ex
from mlsynth.utils.marex_helpers.diagnostics import SearchReport
from mlsynth.utils.syndes_helpers.exact import SubsetFilter

#: The library's objects, restored after every call.
LIBRARY_TOPK = ex._TopK
LIBRARY_BOUND = ex.treated_fit_bound
LIBRARY_FLOOR = ex.control_fit_floor


def complements(sets: np.ndarray, n: int) -> np.ndarray:
    """Each row's complement in ``range(n)``, in increasing order."""
    sets = np.asarray(sets, dtype=np.int64)
    keep = np.ones((len(sets), n), dtype=bool)
    keep[np.arange(len(sets))[:, None], sets] = False
    return np.nonzero(keep)[1].reshape(len(sets), n - sets.shape[1])


def two_sided_bound(B: np.ndarray, A: np.ndarray, sets: np.ndarray,
                    linear: Optional[np.ndarray] = None, report: Any = None, *,
                    rigorous: bool = True) -> np.ndarray:
    """A lower bound on ``g(S) + h(S^c)`` for every row ``S`` of ``sets``.

    The library's treated bound on ``S``, plus the larger of the control floor
    and the library's closed-form bound on the complement. Takes
    :func:`~mlsynth.utils.marex_helpers.exact.treated_fit_bound`'s arguments.
    """
    sets = np.asarray(sets, dtype=np.int64)
    n = B.shape[1]
    out = LIBRARY_BOUND(B, A, sets, linear, report=report, rigorous=rigorous)
    floor = LIBRARY_FLOOR(B, A)
    if sets.shape[1] < n:
        control = LIBRARY_BOUND(B, A, complements(sets, n), None, rigorous=rigorous)
        out = out + np.maximum(control, floor)
    else:                                             # pragma: no cover - no control pool
        out = out + floor
    return out


@contextmanager
def _patched(seed: Optional[ex.ClusterDesign], two_sided: bool):
    class Seeded(LIBRARY_TOPK):
        def __init__(self, k: int):
            super().__init__(k)
            if seed is not None:
                self.items = [seed]

    ex._TopK = Seeded
    if two_sided:
        ex.treated_fit_bound = two_sided_bound
        ex.control_fit_floor = lambda B, A, linear=None: 0.0   # inside the bound now
    try:
        yield
    finally:
        ex._TopK = LIBRARY_TOPK
        ex.treated_fit_bound = LIBRARY_BOUND
        ex.control_fit_floor = LIBRARY_FLOOR


def run_search(B: np.ndarray, A: np.ndarray, sizes: Sequence[int], *,
               seed: Optional[ex.ClusterDesign] = None,
               two_sided: bool = False) -> Dict[str, Any]:
    """One cluster's search; its design, what it fitted and its time.

    ``walked`` counts the candidates that reached a treated fit, ``fitted`` the
    ones whose control fit was computed too.
    """
    report = SearchReport()
    with _patched(seed, two_sided):
        clock = time.perf_counter()
        design = ex.search_cluster(B, A, list(sizes), filt=SubsetFilter(),
                                   report=report)[0]
        seconds = time.perf_counter() - clock
    c = report.freeze("enumeration").counts
    return {"value": float(design.value), "treated": [int(j) for j in design.treated],
            "walked": int(c.fitted + c.cleared_by_treated_fit + c.unweighted
                          + c.without_control),
            "fitted": int(c.fitted), "second_pass": bool(c.second_pass),
            "seconds": seconds}
