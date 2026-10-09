"""What detection error costs the repair, and how RRSC compares.

Every earlier arm assumed the contaminated market was known. This one does not.
:class:`~mlsynth.SPOTSYNTH` forecasts each donor's post-intervention values from
pre-intervention donor data alone and returns the donors the forecast misses on
``screen.excluded_idx``. The repair then runs on whatever that screen returns,
and the cost of its mistakes is the quantity of interest.

Two repairs are compared on the same replication: one told which market was
contaminated, and one given only the screen's answer. A screen that misses the
contaminated market leaves the whole bias in place; a screen that over-flags
rebuilds clean markets for no reason and spends their contribution to the fit.

:class:`~mlsynth.RRSC` is the other comparison. It needs neither a nominated
market nor a clean pool, only a majority of controls unaffected without knowing
which. It replaces the design's weighting, so it cannot preserve a committed
``w`` and ``v``; the comparison is to see what the repair buys over an estimator
that asks less of the analyst.

RRSC is reported only where it passes an applicability gate: on a clean panel of
the same shape it must recover a known effect to within a stated tolerance.
Without that gate a misconfiguration reads as a finding about the method.

    python detection.py 40 results/detection.csv
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

from mlsynth import RRSC, SPOTSYNTH
from mlsynth.utils.rrsc_helpers.config import RRSCConfig
from mlsynth.utils.solvers.simplex import simplex_lstsq
from mlsynth.utils.spotsynth_helpers.config import SPOTSYNTHConfig

from dgps import DGPS
from run import design

RATIOS = (0.0, 0.5, 1.0, 2.0, 4.0)
GATE_TOL = 0.5          # clean-panel relative error RRSC must beat to be reported


def _single_treated_panel(Y, w, controls, T0):
    """Row 0 is the w-weighted synthetic treated aggregate; the rest are controls."""
    T = Y.shape[0]
    agg = Y @ w
    block = np.column_stack([agg, Y[:, controls]])
    J = block.shape[1]
    d = np.zeros((T, J), dtype=int)
    d[T0:, 0] = 1
    return pd.DataFrame({"unit": np.repeat(np.arange(J), T),
                         "time": np.tile(np.arange(T), J),
                         "y": block.T.reshape(-1), "d": d.T.reshape(-1)})


def _waterfall(Yc, controls, flagged_local, T0):
    """Rebuild each flagged control from the unflagged ones, post-period only."""
    out = Yc.copy()
    keep = [i for i in range(len(controls)) if i not in flagged_local]
    if len(keep) < 2:
        raise ValueError("screen left under two unflagged controls")
    donors = controls[keep]
    for i in sorted(flagged_local):
        tgt = controls[i]
        fit = simplex_lstsq(out[:T0, donors], out[:T0, tgt])
        out[T0:, tgt] = out[T0:, donors] @ fit
    return out


def rrsc_att(df, regime, n_factors):
    cfg = RRSCConfig(df=df, outcome="y", treat="d", unitid="unit", time="time",
                     regime=regime, n_factors=n_factors,
                     update_dif=(regime == "fixed_n"), inference="none")
    return float(RRSC(cfg).fit().effects.att)


def replication(name, seed):
    YN, _, T0 = DGPS[name](seed)
    T, J = YN.shape
    w, v = design(YN, T0, max(2, J // 6))
    treated = np.flatnonzero(w > 1e-8)
    controls = np.flatnonzero(v > 1e-12)
    kstar = int(np.argmax(v))
    kstar_local = int(np.flatnonzero(controls == kstar)[0])
    scale = float(np.median(YN[:T0].std(axis=0)))
    regime = "fixed_n" if J <= 12 else "large_n"
    n_factors = min(3, max(1, J // 5))

    post = slice(T0, T)
    Yt = YN.copy()
    Yt[post, treated] += scale
    clean = np.array([j for j in controls if j != kstar])

    # applicability gate: can RRSC recover a known effect on this clean panel?
    gate_df = _single_treated_panel(Yt, w, controls, T0)
    try:
        gate_est = rrsc_att(gate_df, regime, n_factors)
        gate_ok = bool(abs(gate_est - scale) <= GATE_TOL * abs(scale))
    except Exception:
        gate_est, gate_ok = float("nan"), False

    rows = []
    for ratio in RATIOS:
        pi = ratio * scale
        Yc = Yt.copy()
        Yc[post, kstar] += pi
        oracle = float(np.mean(Yt[post] @ w - Yt[post] @ v))
        naive = float(np.mean(Yc[post] @ w - Yc[post] @ v))

        # repair told which market is contaminated
        fit_known = simplex_lstsq(Yc[:T0, clean], Yc[:T0, kstar])
        Yk = Yc.copy()
        Yk[post, kstar] = Yc[post][:, clean] @ fit_known
        known = float(np.mean(Yk[post] @ w - Yk[post] @ v))

        # repair given only the screen
        sdf = _single_treated_panel(Yc, w, controls, T0)
        try:
            scr = SPOTSYNTH(SPOTSYNTHConfig(df=sdf, outcome="y", treat="d",
                                            unitid="unit", time="time",
                                            inference="frequentist")).fit().screen
            flagged = set(int(i) for i in scr.excluded_idx)
            found = kstar_local in flagged
            n_flag, n_fp = len(flagged), len(flagged - {kstar_local})
            try:
                Yd = _waterfall(Yc, controls, flagged, T0)
                detected = float(np.mean(Yd[post] @ w - Yd[post] @ v))
            except ValueError:
                detected = float("nan")
        except Exception:
            found, n_flag, n_fp, detected = False, -1, -1, float("nan")

        try:
            rrsc = rrsc_att(_single_treated_panel(Yc, w, controls, T0), regime, n_factors)
        except Exception:
            rrsc = float("nan")

        rows.append(dict(dgp=name, seed=seed, ratio=ratio, pi=pi, J=J, vk=float(v[kstar]),
                         n_controls=len(controls), tau=scale, oracle=oracle, naive=naive,
                         known=known, detected=detected, rrsc=rrsc,
                         screen_found_kstar=found, n_flagged=n_flag, n_false_pos=n_fp,
                         rrsc_gate_ok=gate_ok, rrsc_gate_est=gate_est, regime=regime))
    return rows


def main(reps, out):
    warnings.filterwarnings("ignore")
    rows, fails = [], {}
    for name in DGPS:
        for seed in range(reps):
            try:
                rows += replication(name, seed)
            except Exception as exc:                       # noqa: BLE001
                fails[name] = fails.get(name, 0) + 1
                if fails[name] <= 2:
                    print(f"{name} seed={seed}: {type(exc).__name__}: {exc}", flush=True)
        print(f"done {name} ({fails.get(name, 0)} failures)", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {len(rows)} rows to {out}; failures={fails}")


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2])
