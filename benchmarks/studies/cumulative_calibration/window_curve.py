"""Coverage of the cumulative band against the number of calibration windows.

The finer-grained version of the durable case in
``benchmarks/cases/conformal_window_count.py``. Everything the time-series
conformal literature warns about is granted: the level and fluctuation are
Gaussian, independent, stationary and exchangeable across windows, with variances
measured off the MAREX arm of this study. Only the window count varies.

    python window_curve.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# run from anywhere: the repo root supplies mlsynth, this directory supplies panel
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from mlsynth.utils.conformal import block_error_paths

H, ALPHA = 13, 0.05
SD_LEVEL, SD_WITHIN = 0.01569, 0.00986
SD_E = SD_WITHIN * np.sqrt(H)
GRID = (2, 3, 4, 5, 6, 8, 10, 13, 19, 26, 40, 80)


def coverage(m, n_draws=4000, n_sim=2000, seed=0):
    rng = np.random.default_rng(seed)
    hit = 0
    for d in range(n_draws):
        level = rng.normal(0.0, SD_LEVEL, m + 1)
        cal = (level[:m, None] + rng.normal(0.0, SD_E, (m, H))).ravel()
        held = float((level[m] + rng.normal(0.0, SD_E, H)).sum())
        paths = block_error_paths(cal, horizon=H, block=0, n_sim=n_sim,
                                  rng=np.random.default_rng(seed * 100_003 + d),
                                  n_windows=m)
        hit += abs(held) <= np.quantile(np.abs(paths.sum(axis=1)), 1 - ALPHA)
    p = hit / n_draws
    return p, float(np.sqrt(p * (1 - p) / n_draws))


if __name__ == "__main__":
    print(f"nominal {1 - ALPHA:.2f}, H = {H}")
    for m in GRID:
        p, se = coverage(m)
        print(f"  m = {m:3d}   coverage {p:.4f} +/- {se:.4f}")
