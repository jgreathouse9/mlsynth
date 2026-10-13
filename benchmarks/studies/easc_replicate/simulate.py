"""The Monte Carlo of Spruk (2026), Section 5.1, ported from ``code/mc.py``.

Untreated outcomes, 1996-2024, treatment from 2012:

    Y_jt = a_j + d_t + theta_j x_t + mu_j f_t + nu_j h_t + e_jt

``x_t`` is the actual log Brent price centred on its 1996-2011 mean, ``f_t`` a
latent trend, ``h_t`` a latent random walk redrawn every replication, and
``e_jt`` AR(1) noise. Exposure is ``theta_j = gamma s_j`` with ``s_j`` the
donors' actual 1996-2011 oil rents, and ``mu_j = xi_j - gamma b s_j`` makes the
composite trend loading independent of exposure, so pre-treatment fit carries
no information about it (the situation of Corollary 1). The treated unit has
Iran's exposure, 22.87, and the effect is -0.10 in 2012-15 and -0.05 after.

``draw`` reproduces ``mc.draw`` call for call, so a seed gives the reference's
panel bit for bit; seeds follow ``mc.py``: ``20261002 + 1000 * (design index) +
replication``. ``one_rep`` reproduces the part of ``mc.one_rep`` that concerns
SC, EBSC-min, EASC and EASC*; the other estimators in the paper's tables belong
to their own benchmarks.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from . import oracle as O

YEARS = np.arange(1996, 2025)
T = len(YEARS)
PRE = YEARS < 2012
POST = ~PRE
BASEIDX = int(np.where(YEARS == 2011)[0][0])
PHASES = {"shock": (2012, 2015), "relief": (2016, 2017), "siege": (2018, 2024)}
PHASE_MASKS = {k: (YEARS >= a) & (YEARS <= b) for k, (a, b) in PHASES.items()}
TAU = np.where(YEARS < 2012, 0.0, np.where(YEARS <= 2015, -0.10, -0.05))
# Donor order matters: the draws are assigned to donors in this order.
BASE = ["KAZ", "NGA", "ECU", "MEX", "COL", "JOR", "ROU", "BGR", "VNM", "ZAF", "PHL", "THA", "KEN", "BRA",
        "CHL", "PER", "CRI", "DOM", "PAN"]
GAMMA = 0.008
DESIGNS = ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11")
MS = (0.0, 0.5, 1.0)
ESTIMATORS = ("SC", "EBSC-min", "EASC", "EASC*")


@dataclass(frozen=True)
class Calibration:
    """The fixed ingredients of the DGP, computed once from the application's data."""

    x_out: np.ndarray       # actual log Brent, centred on 1996-2011
    x_in: np.ndarray        # S1: post-2012 log Brent on its pre-2012 trend in f_t
    f: np.ndarray           # latent trend, decades from 2003.5
    slope: float            # pre-2012 slope of x on f
    s_base: np.ndarray      # 19 normalisers' exposure, in BASE order
    s_ext: np.ndarray       # plus Azerbaijan and Angola
    s_treated: float        # Iran


def calibrate(panel: Path) -> Calibration:
    df = pd.read_csv(panel)
    brent = df.drop_duplicates("year").set_index("year")["brent"]
    xr = np.log(brent.loc[YEARS].values)
    xr = xr - xr[PRE].mean()
    f = (YEARS - 2003.5) / 10.0
    b = np.polyfit(f[PRE], xr[PRE], 1)
    s_all = df[df.year.between(1996, 2011)].groupby("iso")["oilrent"].mean()
    return Calibration(
        x_out=xr, x_in=np.where(PRE, xr, np.polyval(b, f)), f=f, slope=float(b[0]),
        s_base=s_all.reindex(BASE).fillna(0.0).values,
        s_ext=s_all.reindex(BASE + ["AZE", "AGO"]).fillna(0.0).values,
        s_treated=float(s_all["IRN"]),
    )


def design(cal: Calibration, scn: str) -> Dict:
    """The parameters of design ``scn`` (``mc.SCEN``)."""
    g, xo, sb, se = GAMMA, cal.x_out, cal.s_base, cal.s_ext
    table = {
        "S0": dict(gamma=0.0, x=xo, s0=sb),
        "S1": dict(gamma=g, x=cal.x_in, s0=sb),
        "S2": dict(gamma=g, x=xo, s0=sb),
        "S3": dict(gamma=g, x=xo, s0=se),
        "S4": dict(gamma=g, x=xo, s0=sb, noise=0.3),
        "S5": dict(gamma=g, x=xo, s0=sb, concave=True),
        "S6": dict(gamma=g, x=xo, s0=se, concave=True),
        "S7": dict(gamma=g, x=xo, s0=np.r_[sb, sb]),
        "S8": dict(gamma=g, x=xo, s0=se, corr=True),
        "S9": dict(gamma=g, x=xo, s0=se, latent_break=True),
        "S10": dict(gamma=g, x=xo, s0=se, corr=True, latent_break=True, treated_out=True),
        "S11": dict(gamma=g, x=xo, s0=se, hidden=True),
    }
    p = dict(noise=0.0, concave=False)
    p.update(table[scn])
    return p


def draw(cal: Calibration, scn: str, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """One panel (column 0 treated) and the observed exposure. Ports ``mc.draw``."""
    p = design(cal, scn)
    s0 = p["s0"]
    J = len(s0)
    x = p["x"]
    s = np.r_[cal.s_treated, s0]
    theta = p["gamma"] * (20 * (1 - np.exp(-s / 20)) if p["concave"] else s)
    xi = np.r_[0.0, rng.normal(0, 0.08, J)]
    if p.get("corr"):
        xi = xi + 0.004 * (s - s.mean())
    if p.get("treated_out"):
        xi[0] = 0.16
    mu = xi - p["gamma"] * cal.slope * s if not p["concave"] else xi - cal.slope * theta
    nu = np.r_[0.0, rng.normal(0, 1.0, J)]
    a = np.r_[0.0, rng.normal(0, 0.3, J)]
    sd_h = np.where(YEARS >= 2015, 0.3, 0.03) if p.get("latent_break") else np.full(T, 0.03)
    h = np.cumsum(rng.normal(0, 1.0, T) * sd_h)
    h -= h[PRE].mean()
    e = np.zeros((T, J + 1))
    sd_u = 0.015 * np.sqrt(1 - 0.25)
    e[0] = rng.normal(0, 0.015, J + 1)
    for t in range(1, T):
        e[t] = 0.5 * e[t - 1] + rng.normal(0, sd_u, J + 1)
    d = 0.02 * (YEARS - 1996)
    Y = a[None, :] + d[:, None] + np.outer(x, theta) + np.outer(cal.f, mu) + np.outer(h, nu) + e
    if p.get("hidden"):
        z = np.where(YEARS >= 2015, -(YEARS - 2014) / 10.0, 0.0)
        zeta = 0.004 * s + rng.normal(0, 0.03, J + 1)
        zeta[0] = 0.004 * s[0] + 0.06
        Y = Y + np.outer(z, zeta)
    Y[:, 0] += TAU
    s_obs = s * np.exp(rng.normal(0, p["noise"], J + 1)) if p["noise"] > 0 else s.copy()
    return Y, s_obs


def seed(scn: str, rep: int) -> int:
    return 20261002 + 1000 * DESIGNS.index(scn) + rep


def one_rep(cal: Calibration, scn: str, rep: int) -> Dict:
    """Errors, interval coverage and diagnostics for one replication (``mc.one_rep``)."""
    Y, s = draw(cal, scn, np.random.default_rng(seed(scn, rep)))
    y1, Y0, s1, s0 = Y[:, 0], Y[:, 1:], float(s[0]), s[1:]
    J = Y0.shape[1]
    x = design(cal, scn)["x"]
    tr = O._unit(y1, Y0, PRE, s1, s0, BASEIDX)
    kap_l = O.leverage(O.with_constant(tr["Fl"]), PRE)
    path = O.frontier(y1[PRE], Y0[PRE], s1, s0, n=21, w_sc=tr["w"])
    best = O.select_frontier_point(path, kap_l, tr["se"], POST)
    g_star, ei_star = O.easc_gap(y1, Y0, best["w"], s1, s0, tr["phi"])
    gaps = {"SC": tr["g"], "EASC": tr["ge"], "EBSC-min": y1 - Y0 @ path[0]["w"], "EASC*": g_star}
    err = {e: {k: float(gaps[e][m].mean() - TAU[m].mean()) for k, m in PHASE_MASKS.items()} for e in ESTIMATORS}

    plac = [O._unit(Y0[:, j], np.delete(Y0, j, 1), PRE, float(s0[j]), np.delete(s0, j), BASEIDX) for j in range(J)]
    ell, r2 = O.partial_leverage(x, tr["Fl"], PRE)
    inf = {}
    for est, key in (("SC", "g"), ("EASC", "ge")):
        for ph, m in PHASE_MASKS.items():
            tau_hat = float(tr[key][m].mean())
            truth = float(TAU[m].mean())
            z = np.array([p[key][m].mean() / p["rmspe"] for p in plac])
            sd = float(z.std(ddof=1) * tr["rmspe"])
            if est == "EASC":
                sd = float(np.sqrt(sd ** 2 + (abs(tr["ei"]) * tr["se"][m].mean()) ** 2))
                kb = float(np.sqrt(O.phase_leverage(O.with_constant(tr["Fl"]), PRE, m) ** 2 + ell[BASEIDX] ** 2))
            else:
                kb = O.phase_leverage(O.with_constant(tr["Fl"], x), PRE, m)
            for M in MS:
                hw = O.bias_aware_halfwidth(M, kb, tr["rmspe"], sd)
                inf[f"{est}|{ph}|{M}"] = [float(abs(tau_hat - truth) <= hw), 2 * hw]
            inf[f"{est}|{ph}|kbar"] = kb
            inf[f"{est}|{ph}|sd"] = sd
            stat = abs(tau_hat) / tr["rmspe"]
            inf[f"{est}|{ph}|rank"] = (1 + sum(abs(p[key][m].mean()) / p["rmspe"] >= stat for p in plac)) / (J + 1)
    diag = dict(ei_sc=tr["ei"], ei_min=path[0]["ei"], ei_star=ei_star, rmspe=tr["rmspe"],
                rmspe_min=path[0]["rmspe"], frac=best["frac"], r2=r2,
                plmax=float(np.max(np.abs(ell[POST]))), dmin=O.min_imbalance(s1, s0))
    return dict(scn=scn, rep=rep, err=err, inf=inf, diag=diag)


def summarise(reps) -> Dict:
    """Bias, RMSE, coverage and average length by design, as ``code/mc_summary.py``."""
    out = {"bias": {}, "rmse": {}, "coverage": {}, "length": {}}
    by = {}
    for r in reps:
        by.setdefault(r["scn"], []).append(r)
    for scn, rs in by.items():
        for e in ESTIMATORS:
            for ph in PHASES:
                v = np.array([r["err"][e][ph] for r in rs])
                out["bias"][f"{scn}|{ph}|{e}"] = float(v.mean())
                out["rmse"][f"{scn}|{ph}|{e}"] = float(np.sqrt(np.mean(v ** 2)))
        for e in ("SC", "EASC"):
            for ph in PHASES:
                for M in MS:
                    c = np.array([r["inf"][f"{e}|{ph}|{M}"] for r in rs])
                    out["coverage"][f"{scn}|{ph}|{e}|{M}"] = float(c[:, 0].mean())
                    out["length"][f"{scn}|{ph}|{e}|{M}"] = float(c[:, 1].mean())
    return out
