"""MAREX's branch and bound and its persp+N bound, measured at scale.

:func:`root_bound` times the bound at the root of a cluster, every market free,
which is the largest bound a search computes. :func:`search` runs
:func:`mlsynth.utils.marex_helpers.branch.branch_and_bound` under a work limit
and records what it did; a search stopped at the limit reports the bracket its
certificate carries. The runs of #842 capped the search at 30 to 40 bounds;
#843 moved the cap to the cost model's seconds, and every one of those runs
closed, so its rows stand.
"""
from __future__ import annotations

import time

import numpy as np

from mlsynth.utils.marex_helpers import branch
from mlsynth.utils.marex_helpers.certify import ExactData
from mlsynth.utils.marex_helpers.exact import SearchNotClosed
from mlsynth.utils.syndes_helpers.exact import SubsetFilter

_ORIGINAL_NODE_BOUND = branch.node_bound


def marex_panel(Y: np.ndarray):
    """One cluster as MAREX's exact backend sees it: the markets as the columns
    of ``B`` and their mean as the target ``A``, as in ``marex_bnb.panels``."""
    B = np.asarray(Y, float)
    return B, B.mean(axis=1)


def root_bound(Y: np.ndarray, m: int) -> dict:
    """The root's bound with ``m`` treated, and its seconds."""
    B, A = marex_panel(Y)
    data = ExactData(B, A)
    n = data.n
    t0 = time.process_time()
    nb = branch.node_bound(data, m, (), (), tuple(range(n)))
    seconds = time.process_time() - t0
    return {"markets": n, "m": m, "seconds": seconds,
            "modelled": branch.lift_seconds(n),
            "lower": None if nb.lower is None else float(nb.lower),
            "relaxation": nb.relaxation}


def search(Y: np.ndarray, m: int, *, work_limit: float = branch.WORK_LIMIT,
           price_seconds: float = branch.PRICE_SECONDS) -> dict:
    """One search with ``m`` treated under ``work_limit``, with the study's
    columns."""
    B, A = marex_panel(Y)
    tally = {"lifts": 0, "bound_seconds": 0.0, "slowest": 0.0}

    def node_bound(*args, **kwargs):
        t0 = time.process_time()
        out = _ORIGINAL_NODE_BOUND(*args, **kwargs)
        dt = time.process_time() - t0
        tally["lifts"] += 1
        tally["bound_seconds"] += dt
        tally["slowest"] = max(tally["slowest"], dt)
        return out

    row = {"markets": B.shape[1], "m": m, "work_limit": work_limit}
    branch.node_bound = node_bound
    c0 = time.process_time()
    try:
        res = branch.branch_and_bound(B, A, [m], filt=SubsetFilter(),
                                      work_limit=work_limit,
                                      price_seconds=price_seconds)
    except SearchNotClosed as exc:
        row.update(closed=False, stopped=str(exc), treated=None, priced=None,
                   lower=float("nan"), upper=float("nan"))
    else:
        c = res.certificate
        row.update(closed=res.closed,
                   stopped="" if res.closed else f"{res.open} node(s) open at the limit",
                   treated=list(res.designs[0].treated), priced=res.priced,
                   lower=float(c.lower), upper=float(c.upper), fitted=c.fitted,
                   unresolved=len(c.unresolved), work=res.work)
    finally:
        branch.node_bound = _ORIGINAL_NODE_BOUND
    row.update(cpu_seconds=time.process_time() - c0, **tally)
    row["gap"] = ((row["upper"] - row["lower"]) / row["upper"]
                  if np.isfinite(row["upper"]) and row["upper"] else float("nan"))
    return row
