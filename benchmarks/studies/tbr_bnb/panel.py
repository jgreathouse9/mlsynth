"""The study's panels: the GeoLift markets' scoring window and its treatment groups.

The window is the tbrmm benchmark's (``benchmarks/cases/tbrmm.py``), the
pre-period that Au (2018), Algorithm 1, scores designs on. Each geo's series is
centred over the window, as the regression of the treatment aggregate on the
control aggregate with an intercept (Kerman, Wang and Vaver 2017, eq. (1))
leaves it: with an intercept, the residual sum of squares is that of the
centred series.

Au, T. C. (2018). A time-based regression matched markets approach for designing
    geo experiments. Google technical report.
Kerman, J., Wang, P. and Vaver, J. (2017). Estimating ad effectiveness using geo
    experiments in a time-based regression framework. Google technical report.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

#: The four treatment groups of the measurements in #965, one to four geos.
GROUPS: Tuple[Tuple[str, ...], ...] = (
    ("dallas",),
    ("dallas", "denver"),
    ("cincinnati", "dallas", "denver"),
    ("chicago", "cincinnati", "dallas", "denver"),
)


def centred_window() -> Tuple[List[str], np.ndarray]:
    """The geos and the window's series, centred per geo (periods by geos)."""
    from benchmarks.cases.tbrmm import _scoring_window

    wide = _scoring_window().pivot(index="date", columns="geo", values="Y").sort_index()
    Y = wide.to_numpy(float)
    return list(wide.columns), Y - Y.mean(axis=0)


def group(names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """The treatment aggregate ``y``, the candidate controls ``X`` and their names."""
    geos, M = centred_window()
    T = [geos.index(g) for g in names]
    pool = [j for j in range(len(geos)) if j not in T]
    return M[:, T].sum(axis=1), M[:, pool], [geos[j] for j in pool]
