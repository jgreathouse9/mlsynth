"""CWZ debiased SC t-test (arXiv:1812.10820): Section 6 Monte Carlo (Path B).

Reproduces, through the public API (``VanillaSC(inference="ttest")``), the
defining behaviour of the paper's Table 3 application-based simulations,
calibrated to the Andersson (2019) carbon-tax data: a 4-factor model fit to the
detrended controls plus an AR(1) for the SC prediction errors (we recover the
paper's rho_u ~ 0.31). Treatment effect is zero, so we measure the t-DISCo
estimator's bias and the CI's coverage of the truth.

We pin a representative subset (the full 9-DGP x {K=3,4} sweep at 2000 reps
matches every cell of Table 3 to <= 0.03; this trimmed run uses fewer reps and
the most diagnostic DGPs to stay fast):

* DGP1  stationary, SC weights        -> nominal coverage, tiny length.
* DGP3  stationary, *misspecified*     -> cross-fitting still debiases (bias~0),
  but the CI is much wider than the oracle's (variance inflation, not bias).
* DGP6  common trend + one deviation   -> the documented mild undercoverage.
* DGP8  heterogeneous trends (not covered by the theory) -> strong undercoverage
  at T0=30 (the paper's stress case).

The "Oracle" column (known weights, no estimation) is reproduced via the
``oracle_weights`` public-API option.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from benchmarks.compare import BenchmarkSkipped

_BASE = os.path.join(os.path.dirname(__file__), "..", "..", "basedata")
_REPS = 400
_T0, _T1 = 30, 16


def _calibrate():
    p = os.path.abspath(os.path.join(_BASE, "carbontax_data.dta"))
    if not os.path.exists(p):
        raise BenchmarkSkipped("carbontax_data.dta not available")
    import cvxpy as cp

    d = pd.read_stata(p)
    piv = d.pivot(index="year", columns="country",
                  values="CO2_transport_capita").sort_index()
    sw = "Sweden"
    controls = [c for c in piv.columns if c != sw]
    Y = piv[controls].to_numpy()
    ys = piv[sw].to_numpy()
    T, N = Y.shape
    tt = np.arange(T)
    Z = np.column_stack([np.ones(T), tt])
    det = np.column_stack(
        [Y[:, i] - Z @ np.linalg.lstsq(Z, Y[:, i], rcond=None)[0] for i in range(N)])
    U, S, Vt = np.linalg.svd(det - det.mean(0), full_matrices=False)
    r = 4
    F = U[:, :r] * S[:r]
    L = Vt[:r].T
    sigF = F.std(0)
    eta = det - det.mean(0) - F @ L.T
    rho_i = np.array([np.clip(np.dot(eta[1:, i], eta[:-1, i])
                              / max(np.dot(eta[:-1, i], eta[:-1, i]), 1e-12), -.99, .99)
                      for i in range(N)])
    sig_e = np.array([np.std(eta[1:, i] - rho_i[i] * eta[:-1, i]) for i in range(N)])
    w = cp.Variable(N)
    cp.Problem(cp.Minimize(cp.sum_squares(Y[:_T0] @ w - ys[:_T0])),
               [cp.sum(w) == 1, w >= 0]).solve(solver=cp.OSQP, eps_abs=1e-9,
                                               eps_rel=1e-9, max_iter=200000)
    wSC = np.asarray(w.value).ravel()
    u = ys[:_T0] - Y[:_T0] @ wSC
    rho_u = float(np.dot(u[1:], u[:-1]) / np.dot(u[:-1], u[:-1]))
    sig_v = float(np.std(u[1:] - rho_u * u[:-1]))
    wMIS = np.zeros(N); wMIS[:3] = [-3, 3, 1]
    return dict(controls=controls, T=T, N=N, tt=tt, L=L, sigF=sigF, r=r,
                rho_i=rho_i, sig_e=sig_e, rho_u=rho_u, sig_v=sig_v, wSC=wSC, wMIS=wMIS)


def _ar1(rho, sd, T, rng):
    x = np.empty(T)
    x[0] = rng.normal(0, sd / np.sqrt(max(1 - rho ** 2, 1e-6)))
    for t in range(1, T):
        x[t] = rho * x[t - 1] + rng.normal(0, sd)
    return x


def _theta(dgp, cal, rng):
    T, N, tt = cal["T"], cal["N"], cal["tt"]
    th = np.zeros((T, N)); tc = tt[:, None]
    if dgp == "DGP6":
        th = tc * np.ones((1, N)); th[:, 0] += tt
    if dgp == "DGP8":
        a = np.arange(1, N + 1)[None, :]; th = a + a * tc
    return th


def _spec(dgp, cal):
    return {"DGP1": (0, cal["wSC"]), "DGP3": (2, cal["wMIS"]),
            "DGP6": (0, cal["wSC"]), "DGP8": (0, cal["wMIS"])}[dgp]


def _cell(dgp, col, cal, reps, seed):
    from mlsynth import VanillaSC

    T, N, r = cal["T"], cal["N"], cal["r"]
    units = ["Sweden"] + cal["controls"]
    mu, w = _spec(dgp, cal)
    rng = np.random.default_rng(seed)
    cov = np.empty(reps); ln = np.empty(reps); att = np.empty(reps)
    for j in range(reps):
        Fr = rng.normal(size=(T, r)) * cal["sigF"]
        etar = np.column_stack([_ar1(cal["rho_i"][i], cal["sig_e"][i], T, rng)
                                for i in range(N)])
        Y0 = _theta(dgp, cal, rng) + Fr @ cal["L"].T + etar
        y = mu + Y0 @ w + _ar1(cal["rho_u"], cal["sig_v"], T, rng)
        M = np.column_stack([y, Y0])
        df = pd.DataFrame({"unit": np.repeat(units, T),
                           "year": np.tile(cal["tt"], N + 1),
                           "yv": M.T.ravel(), "treated": 0})
        df.loc[(df.unit == "Sweden") & (df.year >= _T0), "treated"] = 1
        cfg = dict(df=df, outcome="yv", treat="treated", unitid="unit", time="year",
                   backend="outcome-only", inference="ttest", ttest_K=3, alpha=0.1,
                   display_graphs=False)
        if col == "oracle":
            cfg["oracle_weights"] = {c: float(w[i]) for i, c in enumerate(cal["controls"])}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            inf = VanillaSC(cfg).fit().inference
        att[j] = inf.details["att_debiased"]
        cov[j] = inf.ci_lower <= 0 <= inf.ci_upper
        ln[j] = inf.ci_upper - inf.ci_lower
    return 10 * att.mean(), cov.mean(), ln.mean()


def run() -> dict:
    cal = _calibrate()
    b1, c1, _ = _cell("DGP1", "tdisco", cal, _REPS, 11)
    b3, c3, l3 = _cell("DGP3", "tdisco", cal, _REPS, 33)
    _, oc3, ol3 = _cell("DGP3", "oracle", cal, _REPS, 34)
    _, c6, _ = _cell("DGP6", "tdisco", cal, _REPS, 66)
    _, c8, _ = _cell("DGP8", "tdisco", cal, _REPS, 88)
    return {
        "rho_u": cal["rho_u"],
        "tD_bias10_dgp1": b1, "tD_cov_dgp1": c1,
        "tD_bias10_dgp3": b3, "tD_cov_dgp3": c3, "tD_len_dgp3": l3,
        "oracle_cov_dgp3": oc3, "oracle_len_dgp3": ol3,
        "tD_cov_dgp6": c6, "tD_cov_dgp8": c8,
    }


# Seeded; ~400 reps -> coverage MC error ~1.5pp. Values track CWZ Table 3.
EXPECTED = {
    "rho_u": (0.31, 0.05),               # calibration recovers the paper's 0.31
    "tD_bias10_dgp1": (0.00, 0.10),      # unbiased, stationary
    "tD_cov_dgp1": (0.91, 0.05),         # nominal coverage
    "tD_bias10_dgp3": (0.00, 0.15),      # cross-fitting debiases under misspec
    "tD_cov_dgp3": (0.90, 0.05),         # ...with nominal coverage
    "tD_len_dgp3": (0.68, 0.12),         # but a wide CI (variance inflation)
    "oracle_cov_dgp3": (0.89, 0.05),     # oracle (known weights) nominal
    "oracle_len_dgp3": (0.07, 0.03),     # ...and tight: ~10x shorter than t-DISCo
    "tD_cov_dgp6": (0.81, 0.05),         # documented mild undercoverage
    "tD_cov_dgp8": (0.59, 0.07),         # uncovered DGP -> strong undercoverage
}
