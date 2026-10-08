"""A negative result: seasonality is not why the in-sample threshold fails.

The first explanation for the in-sample threshold understating the post-window
error was phase mismatch -- the post window sits on one seasonal phase while
consecutive pre-period blocks average across phases. It is wrong. Blocks drawn
at the same seasonal phase as the post window do no better, and the
understatement survives with the season amplitude set to zero. The cause is
that the pre-period residual is in-sample, which `threshold_calibration.py`
measures and the blank window fixes.

Keeping this arm stops the phase-matching explanation being rediscovered.

    python seasonality_negative.py
"""
from __future__ import annotations

import warnings

import numpy as np

from mlsynth.utils.pangeo_helpers.simulation import make_seasonal_sales_panel
from mlsynth.utils.solvers.simplex import simplex_lstsq

TPOST, SEASON, REPS = 10, 52, 300


def one(seed: int, T: int, season_amp: float):
    df = make_seasonal_sales_panel(units_per_arm=7, arms=("A", "B"), T=T,
                                   season_period=SEASON, noise=0.08, seed=seed,
                                   season_amp=season_amp)
    W = df.pivot(index="time", columns="unit", values="sales").to_numpy()
    T0 = T - TPOST
    donors = np.arange(1, W.shape[1])
    weights = simplex_lstsq(W[:T0, donors], W[:T0, 0])
    err = W[:, 0] - W[:, donors] @ weights
    pre = err[:T0]

    per_period = np.sqrt(np.mean(pre ** 2))
    nb = T0 // TPOST
    blocks = np.sqrt(np.mean(pre[:nb * TPOST].reshape(nb, TPOST).mean(axis=1) ** 2))

    phases, s = [], 1
    while T0 - s * SEASON >= 0:
        a = T0 - s * SEASON
        phases.append(pre[a:a + TPOST].mean())
        s += 1
    phase = np.sqrt(np.mean(np.array(phases) ** 2)) if phases else np.nan
    return per_period, blocks, phase, err[T0:].mean(), len(phases)


def main() -> None:
    warnings.filterwarnings("ignore")
    rms = lambda x: float(np.sqrt(np.nanmean(x ** 2)))
    for T, amp in ((104, 1.0), (208, 1.0), (208, 2.0), (208, 0.0)):
        pp, bl, ph, tr, nph = map(np.array,
                                  zip(*[one(s, T, amp) for s in range(REPS)]))
        truth = rms(tr)
        print(f"T={T:3}  season_amp={amp}  phase blocks={int(nph[0])}  "
              f"true threshold={truth:.4f}")
        for label, series in (("per_period", pp), ("blocks", bl),
                              ("phase_matched", ph)):
            print(f"    {label:14} {rms(series):.4f}   "
                  f"ratio to truth {rms(series) / truth:5.2f}")


if __name__ == "__main__":
    main()
