"""Is the panel the paper models a count panel, and does it satisfy the paper's
conditional-independence assumption?

The case for a Poisson likelihood is that the outcomes are counts: "the discrete
case counts violate PPCA's main assumption that the data are Gaussian-distributed"
(Section 4.3.1), and a Poisson assumption "is arguably more appropriate for bursty
COVID case counts" (Section 5). The panels the authors ship are *cumulative* case
counts, so both claims are measured here, not assumed.

Two quantities, each chosen so it isolates what the paper's assumptions need:

* Pearson dispersion of the data around a fitted rank-K Poisson factor model,
  ``(y - mu)^2 / mu``. Poisson requires 1. This is the right statistic because a
  raw variance-to-mean ratio over pooled cells mostly measures how much the rate
  varies across units and periods, which the factor model is there to explain;
  only the residual part speaks to the likelihood. It is evaluated on held-out
  cells, because a rank-10 fit to a panel with 17 columns interpolates and would
  report a dispersion near 1 whatever the data did; and summarised by the median
  over cells with a fitted rate of at least 1, because the ratio is unbounded
  where the fit sends a rate toward zero and a handful of such cells otherwise
  set the whole statistic.
* Lag-1 autocorrelation of the Pearson residuals, by unit. Equations 12 and 19
  require the latent factors to render a unit's own outcomes conditionally
  independent, so it is the correlation left *after* conditioning that matters,
  not the raw correlation of a cumulative series (which is 1 by construction).

Both are reported for the panel as shipped and for its first difference, which is
the daily count the paper's prose describes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF

LATENT_DIM = 10


def diagnostics(matrix: np.ndarray, label: str, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    mask = rng.random(matrix.shape) > 0.10          # True where used for fitting
    held = ~mask

    train = np.where(mask, np.clip(matrix, 0, None), 0.0)
    nmf = NMF(n_components=LATENT_DIM, init="random", random_state=seed, max_iter=8000)
    W = nmf.fit_transform(train)
    mu = np.clip(W @ nmf.components_, 1e-8, None)

    usable = held & (mu >= 1.0)
    ratio = (matrix[usable] - mu[usable]) ** 2 / mu[usable]
    dispersion = float(np.median(ratio)) if ratio.size else float("nan")

    pearson = (matrix - mu) / np.sqrt(mu)
    lag1, raw_lag1 = [], []
    for n in range(matrix.shape[1]):
        if pearson[:, n].std() > 1e-12:
            lag1.append(float(np.corrcoef(pearson[:-1, n], pearson[1:, n])[0, 1]))
        if matrix[:, n].std() > 1e-12:
            raw_lag1.append(float(np.corrcoef(matrix[:-1, n], matrix[1:, n])[0, 1]))

    return {
        "series": label,
        "min": float(matrix.min()),
        "max": float(matrix.max()),
        "heldout_median_pearson_dispersion": dispersion,
        "n_heldout_cells_scored": int(usable.sum()),
        "mean_residual_lag1_autocorrelation": float(np.mean(lag1)) if lag1 else float("nan"),
        "mean_raw_lag1_autocorrelation": float(np.mean(raw_lag1)) if raw_lag1 else float("nan"),
        "fraction_of_steps_non_decreasing": float(np.mean(np.diff(matrix, axis=0) >= 0)),
        "integer_valued": bool(np.all(matrix == np.round(matrix))),
    }


def main() -> dict:
    here = Path(__file__).parent
    out = {}
    for team in ("indianapolis", "baltimore"):
        long = pd.read_parquet(here / "panels" / f"{team}.parquet")
        wide = long.pivot(index="date", columns="county", values="cases").to_numpy(float)
        out[team] = {
            "shape": list(wide.shape),
            "as_shipped": diagnostics(wide, "cumulative (as shipped)"),
            "differenced": diagnostics(np.clip(np.diff(wide, axis=0), 0, None), "daily (difference)"),
        }

    print("Poisson requires a Pearson dispersion of 1. Equations 12 and 19 require the")
    print("residual autocorrelation to be near 0 once the factors are conditioned on.\n")
    for team, block in out.items():
        print(f"{team}  {block['shape'][0]} periods x {block['shape'][1]} counties")
        for key in ("as_shipped", "differenced"):
            d = block[key]
            print(
                f"  {d['series']:<26} dispersion={d['heldout_median_pearson_dispersion']:>10,.1f}"
                f"  residual lag1={d['mean_residual_lag1_autocorrelation']:>6.3f}"
                f"  (raw lag1={d['mean_raw_lag1_autocorrelation']:.3f})"
            )
        print()

    (here / "outcome_scale.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


if __name__ == "__main__":
    main()
