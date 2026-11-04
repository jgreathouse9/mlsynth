"""The reference key's four gates on the slope-free search, handled lazily (#965).

The reference implementation scores a design by the key (corr_test, aa_test,
bb_test, dw_test, round(corr, 2), 1 / required_impact), lexicographically
(mlsynth.utils.tbr_helpers.design.objective.score_split). For a fixed treatment
group, RSS = y'y (1 - corr^2) on the centred window, so the design of least RSS
among those that pass all four gates also has the greatest correlation among them,
and so the greatest rounded correlation and, among designs that share it, the
least required impact, which rises with the residual scale (derived here from
required_impact's formula). It is therefore the key's optimum among passing designs.

The gates are tests on the fitted residuals, not constraints on u that a relaxation
can carry, so they are handled lazily: certify the RSS-best design not yet excluded
(slopefree.certify_slope_free), test it with the library's own gates, and when it
fails exclude it with a no-good cut and certify again. Each round's design is the
next by RSS, so the first that passes is the passing optimum. The correlation gate
ends the search at once when it fails: the RSS-best remaining design has the
greatest correlation of all remaining ones, so none of them reaches 0.8 either.
A caller's stop rule ends it the same way: it is handed each round's RSS before
the gates are scored, and every remaining design has at least that RSS, so a
caller that needs nothing at or above it (treatment.py, against an incumbent from
another treatment group) is answered, certified, without that round's design.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from fractions import Fraction as Q
from typing import Callable, List, Optional

import numpy as np

from mlsynth.utils.tbr_helpers.design.objective import score_split

from .slopefree import NoDesignLeft, certify_slope_free

GATES = ("corr_test", "aa_test", "bb_test", "dw_test")


@dataclass
class GatedCertificate:
    """The outcome of :func:`certify_gated` for one treatment group."""

    controls: Optional[List[int]]          # the passing optimum, or None when no design passes
    rss: Optional[Q]
    key: Optional[tuple]                   # the reference key at it
    passing: bool                          # a design passes all four gates
    certified: bool                        # the answer is proved, not cut short by max_rounds
    stopped: bool = False                  # the caller's stop rule ended the search
    rounds: List[dict] = field(default_factory=list)
    seconds: float = 0.0


def gates_of(y, X, controls, n_test):
    """The four gates and the reference key at a design."""
    s = score_split(y, X[:, sorted(controls)].sum(axis=1), objective="reference", n_test=n_test)
    return tuple(bool(s.detail[g]) for g in GATES), s.key


def certify_gated(y: np.ndarray, X: np.ndarray, *, n_test: int, restarts: int = 50,
                  max_rounds: int = 50, stop: Optional[Callable[[Q], bool]] = None,
                  **kw) -> GatedCertificate:
    """The least-RSS positive-slope control group that passes the reference key's
    four gates, certified, or a certificate that none does (module docstring)."""
    t0 = time.perf_counter()
    excluded: List[set] = []
    out = GatedCertificate(controls=None, rss=None, key=None, passing=False, certified=False)
    while len(out.rounds) < max_rounds:
        try:
            res = certify_slope_free(y, X, restarts=restarts, exclude=excluded, **kw)
        except NoDesignLeft:
            out.certified = True               # no positive-slope design is left
            break
        if not res.certified:                  # pragma: no cover - the GeoLift groups and the test panels close
            break
        if stop is not None and stop(res.incumbent):
            out.certified = out.stopped = True # every remaining design has at least this RSS
            break
        gates, key = gates_of(y, X, res.controls, n_test)
        out.rounds.append(dict(controls=res.controls, rss=res.incumbent, gates=gates, seconds=res.seconds))
        if all(gates):
            out.controls, out.rss, out.key, out.passing, out.certified = res.controls, res.incumbent, key, True, True
            break
        if not gates[0]:
            out.certified = True               # no remaining design reaches the correlation floor
            break
        excluded.append(set(res.controls))
    out.seconds = time.perf_counter() - t0
    return out
