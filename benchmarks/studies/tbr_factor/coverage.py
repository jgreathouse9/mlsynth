r"""Coverage of TBR's posterior as the factor count rises.

The panel carries no treatment, so the cumulative effect is zero in every
period and coverage is the share of replications whose interval contains zero.
The estimator is ``mlsynth.TBR`` as shipped, reached through ``dataprep``, so
what is measured is the library and not a port of the algebra.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Sequence

import numpy as np
from scipy import stats

from mlsynth import TBR
from mlsynth.config_models import TBRConfig

from . import dgp

#: Factor counts swept. One is the case the method is stated under.
RS = (1, 2, 3, 5)
#: Treated-group sizes. Aggregating more geos does not restore the relation,
#: so this is swept to show the failure is not a small-group artefact.
KS = (2, 5, 25)
#: Kerman (2011)'s neutral prior, for an interval on a coverage rate.
NEUTRAL_PRIOR = 1.0 / 3.0


def one_replication(seed: int, r: int, n_treated: int, *, level: float = 0.90,
                    variance: str = "iid") -> Dict:
    """One panel, one fit; whether the interval covered the true zero."""
    frame = dgp.panel(seed, r, n_treated)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = TBR(TBRConfig(
            df=frame, outcome="y", unitid="geo", time="t",
            treatment_col="is_treat", control_col="is_ctrl", post_col="post",
            level=level, variance=variance)).fit().report
    lower = float(report.cumulative.lower[-1])
    upper = float(report.cumulative.upper[-1])
    return {
        "r": r, "n_treated": n_treated, "seed": seed,
        "covered": bool(lower <= 0.0 <= upper),
        "width": upper - lower,
        "estimate": float(report.cumulative.estimate[-1]),
        "delta2": float(report.tbr_fit.beta),
        "collinearity": float(frame.attrs["collinearity"]),
    }


def run_sweep(reps: int, *, rs: Sequence[int] = RS, ks: Sequence[int] = KS,
              level: float = 0.90, variance: str = "iid",
              seed: int = 0) -> List[Dict]:
    """Every (r, n_treated) cell, each on its own replication stream.

    A cell's draws are derived from its own identity, so its result does not
    depend on which cells preceded it in the request.
    """
    rows = []
    for r in rs:
        for k in ks:
            for i in range(reps):
                rows.append(one_replication(seed * 1_000_003 + i, r, k,
                                            level=level, variance=variance))
    return rows


def beta_interval(n_covered: int, n_draws: int, level: float = 0.99):
    """An interval on a coverage rate under the neutral prior."""
    a = NEUTRAL_PRIOR + n_covered
    b = NEUTRAL_PRIOR + n_draws - n_covered
    tail = 0.5 * (1.0 - level)
    return (float(stats.beta.ppf(tail, a, b)),
            float(stats.beta.ppf(1.0 - tail, a, b)))


def summarise(rows: Sequence[Dict], *, level: float = 0.99) -> Dict:
    """Coverage per factor count, pooled over treated-group sizes."""
    out: Dict[str, float] = {}
    by_r: Dict[int, List[Dict]] = {}
    for row in rows:
        by_r.setdefault(row["r"], []).append(row)
    for r, cells in sorted(by_r.items()):
        hits = sum(c["covered"] for c in cells)
        n = len(cells)
        lo, hi = beta_interval(hits, n, level)
        out[f"cov_r{r}"] = hits / n
        out[f"cov_r{r}_lo"] = lo
        out[f"cov_r{r}_hi"] = hi
        out[f"width_r{r}"] = float(np.mean([c["width"] for c in cells]))
        out[f"collinearity_r{r}"] = float(np.mean([c["collinearity"]
                                                   for c in cells]))
    # The spread across treated-group sizes at one factor, which is the check
    # that aggregating more geos neither causes nor cures the failure.
    by_k = {}
    for row in rows:
        if row["r"] > 1:
            by_k.setdefault(row["n_treated"], []).append(row["covered"])
    if by_k:
        rates = [sum(v) / len(v) for v in by_k.values()]
        out["multifactor_cov_max_over_k"] = float(max(rates))
    out["n_per_r"] = float(len(next(iter(by_r.values()))))
    return out
