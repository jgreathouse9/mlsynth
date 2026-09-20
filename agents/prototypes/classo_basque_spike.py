"""SPIKE CODE -- not library code. See ``agents/scope_classo.md``.

Reproduces the two findings in that scope's Basque section. Run from the
repository root with mlsynth importable and ``cvxpy`` available::

    python agents/prototypes/classo_basque_spike.py

Finding 1 -- SSP's information criterion cannot select ``K`` at this sample
size: for the dynamic specifications the per-group penalty ``rho * p`` exceeds
the entire post-Lasso MSE, so ``K = 1`` is arithmetic and not evidence.

Finding 2 -- at a forced ``K``, restricting the donor pool worsens the
pre-treatment fit (mechanically: a subset's simplex is contained in the full
simplex) and improves held-out accuracy, scored by placebo-in-time where the
true effect is zero.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classo_pls_spike import classo_pls, demean, info_criterion  # noqa: E402

from mlsynth.estimators.vanillasc import VanillaSC  # noqa: E402

TREATED = "Basque Country (Pais Vasco)"
TREAT_YEAR = 1970
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "..", "basedata", "basque_data.csv")

_DF = pd.read_csv(os.path.abspath(DATA))
_PRE = _DF[_DF.year < TREAT_YEAR].sort_values(["regionname", "year"])
UNITS = sorted(_PRE.regionname.unique())
N = len(UNITS)


def build(spec: str):
    """Stack the panel unit-major, standardised per unit as SSP's ``master.m``."""
    ys, Xs = [], []
    for u in UNITS:
        g = _PRE[_PRE.regionname == u]
        y_ = g.gdpcap.to_numpy(float)
        iv = g.invest.to_numpy(float)
        sh = g["school.high"].to_numpy(float)
        si = g["school.illit"].to_numpy(float)
        if spec == "lag":
            yi, xi = y_[1:], y_[:-1, None]
        elif spec == "lag+invest":
            yi, xi = y_[1:], np.column_stack([y_[:-1], iv[1:]])
        elif spec == "lag+sch":
            yi, xi = y_[1:], np.column_stack([y_[:-1], sh[1:], si[1:]])
        elif spec == "invest+sch":
            yi, xi = y_, np.column_stack([iv, sh, si])
        elif spec == "growth":
            gr = np.diff(np.log(y_))
            yi, xi = gr[1:], np.column_stack([gr[:-1], iv[2:]])
        else:
            raise ValueError(f"unknown spec {spec!r}")
        ys.append((yi - yi.mean()) / yi.std())            # std(., 1)
        Xs.append((xi - xi.mean(0)) / xi.std(0))
    T = len(ys[0])
    y, X = np.concatenate(ys), np.vstack(Xs)
    return demean(y[:, None], N, T).ravel(), demean(X, N, T), T


def _pool(spec: str, K: int, c: float = 0.5):
    y, X, T = build(spec)
    lam = c * np.var(y, ddof=1) / T ** (1 / 3)
    labels, _alpha, _beta, _conv = classo_pls(y, X, N, T, K, lam)
    s = pd.Series(labels, index=UNITS)
    return [u for u in UNITS if s[u] == s[TREATED] and u != TREATED]


def _sc(frame: pd.DataFrame, treat_year: int):
    d = frame.copy()
    d["treat"] = ((d.regionname == TREATED) & (d.year >= treat_year)).astype(int)
    return VanillaSC(dict(df=d, outcome="gdpcap", treat="treat",
                          unitid="regionname", time="year",
                          display_graphs=False)).fit()


def finding_one() -> None:
    print("Finding 1 -- IC(K) = MSE(K) + rho*p*K,  rho = (2/3)(NT)^(-1/2)\n")
    print(f"{'spec':14s} {'p':>2s} {'MSE(K=1)':>9s} {'rho*p':>8s}   verdict")
    for spec in ("lag", "lag+invest", "lag+sch", "invest+sch", "growth"):
        y, X, T = build(spec)
        p = X.shape[1]
        lam = 0.5 * np.var(y, ddof=1) / T ** (1 / 3)
        labels, alpha, _b, _c = classo_pls(y, X, N, T, 1, lam)
        yb, Xb = y.reshape(N, T), X.reshape(N, T, p)
        mse = float(np.mean(np.concatenate(
            [yb[i] - Xb[i] @ alpha[labels[i]] for i in range(N)]) ** 2))
        rp = (2 / 3) * (N * T) ** -0.5 * p
        verdict = (f"needs a {100 * rp / mse:.0f}% MSE drop per group"
                   if rp < mse else "IMPOSSIBLE: penalty exceeds the whole MSE")
        print(f"{spec:14s} {p:2d} {mse:9.4f} {rp:8.4f}   {verdict}")


def finding_two(spec: str = "lag+invest") -> None:
    print("\nFinding 2a -- in-sample fit, full pool vs forced-K C-Lasso pools\n")
    for tag, frame in [("all 16", _DF)] + [
            (f"C-Lasso K={K}", _DF[_DF.regionname.isin(_pool(spec, K) + [TREATED])])
            for K in (2, 3, 4)]:
        r = _sc(frame, TREAT_YEAR)
        w = {str(k): float(v) for k, v in r.donor_weights.items()
             if abs(float(v)) > 1e-4}
        print(f"  {tag:14s} ATT {float(r.att):+.4f}  pre-RMSE "
              f"{float(r.fit_diagnostics.rmse_pre):.4f}  "
              + ", ".join(f"{k.split(' (')[0]} {v:.3f}"
                          for k, v in sorted(w.items(), key=lambda z: -z[1])))

    dates = list(range(1962, 1968))
    print("\nFinding 2b -- placebo in time: fit to the date, score the remaining "
          "untreated years\n")
    print(f"  {'donor pool':18s} " + " ".join(f"{d:>7d}" for d in dates) + "     mean")

    def held_out(frame: pd.DataFrame) -> list:
        out = []
        for placebo in dates:
            r = _sc(frame[frame.year < TREAT_YEAR], placebo)
            obs = np.asarray(r.time_series.observed_outcome, float)
            cf = np.asarray(r.time_series.counterfactual_outcome, float)
            k = TREAT_YEAR - placebo
            out.append(float(np.sqrt(np.mean((obs[-k:] - cf[-k:]) ** 2))))
        return out

    rows = {"all 16": held_out(_DF)}
    for K in (2, 3, 4, 5):
        pool = _pool(spec, K)
        if pool:
            rows[f"C-Lasso K={K} ({len(pool)})"] = held_out(
                _DF[_DF.regionname.isin(pool + [TREATED])])
    for name, v in rows.items():
        v = np.asarray(v)
        print(f"  {name:18s} " + " ".join(f"{x:7.4f}" for x in v) + f"  {v.mean():7.4f}")


if __name__ == "__main__":
    finding_one()
    finding_two()
