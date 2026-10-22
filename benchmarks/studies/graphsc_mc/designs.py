"""Four simulation families, drawn by the repository's own simulators.

``cwz``  Chernozhukov, Wuthrich & Zhu's application-calibrated t-test design
         (``benchmarks/cwz_common.py``): 14 donors, ``T0 = 30``, ``T1 = 16``,
         nine DGPs from a correctly specified factor model to donor-specific
         trends. The calibration is the reference run's
         (``benchmarks/reference/cwz_ttest_mc``).
``lsz``  Liao, Shi & Zheng's latent-group factor design
         (``mlsynth.utils.laxscm_helpers.simulation``), exact and perturbed
         groups, ``T1 = 50``, ``J`` in {50, 100, 200}, ``T0`` in {J/2, J, 2J}.
         The oracle weights are equal within each group, the structure a donor
         graph is meant to encode.
``hz``   Hsiao & Zhou's DGP6 as Zhang, Wang & Zhang use it: two factors, MA(1)
         cross-sectional shocks, ``J`` in {30, 50}, ``T0`` in {50, 100, 200,
         400}, ``T1 = 10``. Each draw carries the exact post-period risk
         ``R(w)`` of Zhang-Wang-Zhang's eq. (simu:R) and its minimum over the
         simplex, so weight quality is measured without simulation noise.
``gsc``  The Graph-SC paper's cluster design
         (``benchmarks/studies/graphsc_replicate/simulate.py``): 200 donors in
         10 clusters, ``T0 = 50``, ``T1 = 20``, effect 2.

No family but ``gsc`` has a treatment effect: the post-period of the treated
unit is its own counterfactual, and an estimated ATT is its own error.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Optional

import numpy as np

from ..graphsc_replicate import oracle as G

HERE = os.path.dirname(os.path.abspath(__file__))
CWZ_REF = os.path.join(HERE, "..", "..", "reference", "cwz_ttest_mc")
CWZ_T0, CWZ_T1 = 30, 16
#: The intercept ``ttest_dgp_spec`` adds to the treated unit, by DGP.
CWZ_CONST = {3: 2.0, 4: 2.0, 5: 2.0}
HZ_B = 1.0


@dataclass(frozen=True)
class Draw:
    """One panel and what is true about it.

    ``target`` is the counterfactual an estimate is scored against where the
    design defines one (the systematic part, or the oracle synthetic control);
    ``w_true`` the weights the oracle t-test uses; ``risk`` and ``risk_min``
    the exact risk of a weight vector and its simplex minimum (``hz`` only).
    """

    y: np.ndarray
    Y0: np.ndarray
    T0: int
    tau: float
    target: Optional[np.ndarray] = None
    w_true: Optional[np.ndarray] = None
    risk: Optional[Callable[[np.ndarray], float]] = None
    risk_min: Optional[float] = None

    @property
    def T1(self) -> int:
        return self.y.shape[0] - self.T0


# ---------------------------------------------------------------- cwz


@lru_cache(maxsize=1)
def cwz_calibration() -> dict:
    from benchmarks.cwz_common import load_ttest_calibration
    from benchmarks.reference import reference_value

    return load_ttest_calibration(
        CWZ_REF, rho_u=reference_value("cwz_ttest_mc", "calib_rho_u"),
        var_u=reference_value("cwz_ttest_mc", "calib_var_u"))


def draw_cwz(dgp: int, rng) -> Draw:
    from benchmarks.cwz_common import simulate_ttest_panel

    y, Y0, w = simulate_ttest_panel(dgp, cwz_calibration(), pre_periods=CWZ_T0,
                                    post_periods=CWZ_T1, rng=rng)
    return Draw(y, Y0, CWZ_T0, 0.0, CWZ_CONST.get(dgp, 0.0) + Y0 @ w, w)


# ---------------------------------------------------------------- lsz


def draw_lsz(J: int, T0: int, approximate: bool, rng, T1: int = 50) -> Draw:
    from mlsynth.utils.laxscm_helpers.simulation import simulate_relaxation_groups_design

    d = simulate_relaxation_groups_design(rng, J=J, T0=T0, T1=T1, approximate=approximate)
    return Draw(d.y0, d.Yc.T, T0, 0.0, d.oracle_cf, d.w_star)


# ---------------------------------------------------------------- hz


def hz_design(J: int, T0: int, T1: int, rng) -> dict:
    """DGP6, drawn in the order of ``hsiao_zhou_counterfactuals.methods.simulate``, keeping the design.

    ``mu`` is the factor part of every unit, treated first; ``sigma2`` the
    variances of the shocks behind the MA(1) cross-sectional errors.
    """
    T, N = T0 + T1, J + 1
    f = rng.standard_normal((T, 2))
    loadings = rng.standard_normal((N, 2))
    sigma2 = 0.5 * (rng.chisquare(1, N) + 1.0)
    v = rng.standard_normal((T, N)) * np.sqrt(sigma2)
    pad = np.zeros((T, 1))
    u = (1 + HZ_B ** 2) * v + HZ_B * np.hstack([v[:, 1:], pad]) + HZ_B * np.hstack([pad, v[:, :-1]])
    mu = f @ loadings.T
    Y = mu + u
    return {"y": Y[:, 0], "Y0": Y[:, 1:], "mu": mu, "sigma2": sigma2}


def hz_sigma(sigma2, b: float = HZ_B) -> np.ndarray:
    """Covariance of ``u = A v``: ``A`` has ``1 + b^2`` on the diagonal and ``b`` beside it, no wrap."""
    n = len(sigma2)
    A = (1 + b * b) * np.eye(n) + b * np.eye(n, k=1) + b * np.eye(n, k=-1)
    return A @ np.diag(sigma2) @ A.T


def hz_risk(w, mu_post, S) -> float:
    """Zhang-Wang-Zhang's ``R_{T1}(w)``: mean squared post-period bias plus ``a' S a``, ``a = (1, -w)``."""
    a = np.r_[1.0, -np.asarray(w, dtype=float)]
    return float(np.mean((mu_post @ a) ** 2) + a @ S @ a)


def hz_min_risk(mu_post, S):
    """``(w_opt, R_min)``: the risk's minimum over the simplex, as a simplex least-squares problem.

    ``R(w) = c0 - 2 q'w + w'Qw`` with ``Q`` positive definite; with ``Q = CC'``
    this is ``||C'w - C^{-1} q||^2`` plus a constant, which ``simplex_sc`` solves.
    """
    T1 = mu_post.shape[0]
    mc, m0 = mu_post[:, 1:], mu_post[:, 0]
    Q = mc.T @ mc / T1 + S[1:, 1:]
    q = mc.T @ m0 / T1 + S[1:, 0]
    C = np.linalg.cholesky(Q)
    w = G.simplex_sc(np.linalg.solve(C, q), C.T)
    return w, hz_risk(w, mu_post, S)


def draw_hz(J: int, T0: int, rng, T1: int = 10) -> Draw:
    d = hz_design(J, T0, T1, rng)
    S, mu_post = hz_sigma(d["sigma2"]), d["mu"][T0:]
    w_opt, r_min = hz_min_risk(mu_post, S)
    return Draw(d["y"], d["Y0"], T0, 0.0, None, w_opt,
                risk=lambda w: hz_risk(w, mu_post, S), risk_min=r_min)


# ---------------------------------------------------------------- gsc


def draw_gsc(rng) -> Draw:
    from ..graphsc_replicate import simulate as GS

    d = GS.draw(rng)
    return Draw(d["y"], d["X"], GS.T0, GS.TAU, d["noiseless_cf"], d["w_true"])


# ---------------------------------------------------------------- the grid

_LSZ = [(50, 25), (50, 50), (50, 100), (100, 50), (100, 100), (100, 200)]

FAMILIES = {
    "cwz": {"label": "Chernozhukov-Wuthrich-Zhu calibrated design",
            "cells": [{"dgp": d} for d in range(1, 10)], "reps": 1000, "K": 3},
    "lsz": {"label": "Liao-Shi-Zheng latent groups",
            "cells": [{"J": J, "T0": T0, "approximate": a} for J, T0 in _LSZ for a in (False, True)]
                     + [{"J": 200, "T0": T0, "approximate": False} for T0 in (100, 200, 400)],
            "reps": 500, "K": 5},
    "hz": {"label": "Hsiao-Zhou DGP6 (Zhang-Wang-Zhang)",
           "cells": [{"J": J, "T0": T0} for J in (30, 50) for T0 in (50, 100, 200, 400)],
           "reps": 500, "K": 5},
    "gsc": {"label": "Graph-SC paper's cluster design", "cells": [{}], "reps": 500, "K": 5},
}


def draw(family: str, cell: dict, rng) -> Draw:
    if family == "cwz":
        return draw_cwz(cell["dgp"], rng)
    if family == "lsz":
        return draw_lsz(cell["J"], cell["T0"], cell["approximate"], rng)
    if family == "hz":
        return draw_hz(cell["J"], cell["T0"], rng)
    if family == "gsc":
        return draw_gsc(rng)
    raise ValueError(f"unknown family {family!r}; choose from {sorted(FAMILIES)}")
