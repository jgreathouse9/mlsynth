"""PANGEO Path-B: trajectory-matched supergeo design vs scalar matching.

Validates mlsynth's ``PANGEO`` geo-experiment design on the paper's motivating
scenario (Chen, Doudchenko, Jiang, Stein & Ying 2023): six geos form three
parallel pairs (up-trend / down-trend / cycle), each **demeaned over the
pre-period so all baseline means are identical**. A scalar (Google-style) match on
the baseline mean is therefore blind to the shape and pairs geos essentially at
random; PANGEO matches on the full pre-treatment trajectory and recovers the
three parallel pairs.

Injecting a true effect ``tau = 4`` and running the downstream
difference-in-differences, PANGEO estimates the effect **~30x more precisely**
than the scalar match:

  =========================  ===============
  design                     DiD RMSE
  =========================  ===============
  PANGEO (trajectory)        ~0.2
  scalar matched-pairs       ~6
  =========================  ===============

Path B (the paper's design scenario): the case asserts PANGEO's RMSE is far below
the scalar match's (an order of magnitude or more) -- the supergeo idea carried
into the panel world (match on *how series move*, not a single number).
Deterministic (seeded); the six-geo MIP is instantaneous.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

R = 30
_TAU = 4.0


def _make_panel(rng, T_pre=20, S=6, level=100.0, noise=0.6):
    T = T_pre + S
    t = np.arange(T)
    up = (t - t.mean()) / t.std()
    cyc = np.sin(2 * np.pi * t / 5.0)
    shapes = [5 * up, 5 * up, -5 * up, -5 * up, 5 * cyc, 5 * cyc]
    cols = []
    for s in shapes:
        s = s - s[:T_pre].mean()                 # equal pre-means => scalar-blind
        cols.append(level + s + rng.normal(0, noise, T))
    return np.column_stack(cols), T_pre


def _did(Y, T_pre, treated, control):
    Yo = Y.copy()
    Yo[T_pre:, treated] += _TAU
    t_eff = Yo[T_pre:, treated].mean() - Yo[:T_pre, treated].mean()
    c_eff = Yo[T_pre:, control].mean() - Yo[:T_pre, control].mean()
    return (t_eff - c_eff) - _TAU


def _pangeo_design(Y, T_pre):
    from mlsynth import PANGEO

    df = pd.DataFrame(
        {"geo": f"g{g}", "t": int(t), "y": Y[t, g], "arm": "A"}
        for g in range(6) for t in range(T_pre)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = PANGEO({"df": df, "outcome": "y", "unitid": "geo", "time": "t",
                    "arm": "arm", "max_supergeo_size": 1,
                    "compute_power": False, "display_graphs": False}).fit().assignment
    T = [int(g[1:]) for g, v in a.items() if v == "treatment"]
    C = [int(g[1:]) for g, v in a.items() if v == "control"]
    return T, C


def _scalar_match(Y, T_pre, rng):
    order = np.argsort(Y[:T_pre].mean(0))
    T, C = [], []
    for k in range(0, 6, 2):
        a, b = order[k], order[k + 1]
        if rng.random() < 0.5:
            T.append(a); C.append(b)
        else:
            T.append(b); C.append(a)
    return T, C


def run() -> dict:
    rng = np.random.default_rng(1)
    pe, se = np.empty(R), np.empty(R)
    for i in range(R):
        Y, T_pre = _make_panel(rng)
        Tp, Cp = _pangeo_design(Y, T_pre)
        pe[i] = _did(Y, T_pre, Tp, Cp)
        Ts, Cs = _scalar_match(Y, T_pre, rng)
        se[i] = _did(Y, T_pre, Ts, Cs)
    rmse_p = float(np.sqrt((pe ** 2).mean()))
    rmse_s = float(np.sqrt((se ** 2).mean()))
    return {
        "rmse_pangeo": rmse_p,
        "rmse_scalar": rmse_s,
        "precision_ratio": rmse_s / rmse_p,
        "pangeo_beats_scalar_10x": float(rmse_p < 0.1 * rmse_s),
    }


# Deterministic (seeded). Tolerances absorb the Monte Carlo noise at R=30.
# Reproduces the paper's design scenario: PANGEO's trajectory match recovers the
# parallel pairs and estimates the effect an order of magnitude (~30x) more
# precisely than a scalar baseline match, which is blind to the equal-mean shapes.
EXPECTED = {
    "rmse_pangeo": (0.22, 0.18),
    "rmse_scalar": (6.1, 2.5),
    "precision_ratio": (28.0, 22.0),       # ~30x; assert well above 10x
    "pangeo_beats_scalar_10x": (1.0, 0.0),
}
