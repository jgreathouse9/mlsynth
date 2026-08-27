"""The spike's decision experiment.

Question this answers, fixed before the run: does the gamma-Poisson arm's
advantage over the Gaussian arms survive dropping the undocumented lagged-outcome
regressor, and does it hold at a short pre-period as well as a long one?

Three methods on identical semi-synthetic panels:

* ``GAP``  -- gamma-Poisson factorisation, the paper's proposal
* ``PPCA`` -- Gaussian factorisation, the paper's own control for inference style
* ``rSC``  -- robust synthetic control through ``mlsynth.CLUSTERSC``, the estimator
  a user of this library would actually reach for. Upstream calls ``tslib``'s
  ``RobustSyntheticControl``; the equivalent configuration is the PCR family with
  clustering off, an OLS objective and the rank pinned to K, which is the same
  procedure -- truncate the pre-period donor block to rank K, regress the treated
  pre-period on it, apply the weights to the raw post-period donors. Going through
  the estimator, not its kernel, also exercises ``dataprep`` ingestion and
  the result contract, so the baseline is the path this library ships.

GAP and PPCA run twice, with the lagged outcome in the design and without. rSC
has no equivalent term, so it runs once and is scored against both.

Metric is the paper's mean relative error at each post-period t, against the
untreated potential outcome of the treated unit.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from dgp import create_semi_synthetic_matrix
from mosc_port import counterfactual_from_regression, gap_gibbs, ppca_em

from mlsynth import CLUSTERSC

HORIZON = 30


def load_panel(path: Path) -> tuple[np.ndarray, int]:
    long = pd.read_parquet(path)
    wide = long.pivot(index="date", columns="county", values="cases")
    wide = wide[[c for c in wide.columns if c != "Stadium_County"] + ["Stadium_County"]]
    treated_post = set(long.loc[long["stadium_open"] == 1, "date"])
    intervention_t = int(len(wide) - len(treated_post))
    return wide.to_numpy(dtype=float), intervention_t


def to_long(panel: np.ndarray, intervention_t: int) -> pd.DataFrame:
    """Long form for dataprep; the last column is the single treated unit."""
    n_time, n_unit = panel.shape
    names = [f"county_{i}" for i in range(n_unit - 1)] + ["Stadium_County"]
    frame = pd.DataFrame(panel, columns=names)
    frame["period"] = np.arange(n_time)
    long = frame.melt(id_vars="period", var_name="unit", value_name="cases")
    long["treated"] = (
        (long["unit"] == "Stadium_County") & (long["period"] >= intervention_t)
    ).astype(int)
    return long.sort_values(["unit", "period"], kind="stable").reset_index(drop=True)


def rsc_counterfactual(panel: np.ndarray, intervention_t: int, rank: int) -> np.ndarray:
    """Robust synthetic control through CLUSTERSC's PCR family."""
    result = CLUSTERSC(
        {
            "df": to_long(panel, intervention_t),
            "outcome": "cases",
            "treat": "treated",
            "unitid": "unit",
            "time": "period",
            "method": "pcr",
            "clustering": False,
            "pcr_objective": "OLS",
            "rank_method": "fixed",
            "rank": rank,
            "compute_shen_ci": False,
            "display_graphs": False,
        }
    ).fit()
    counterfactual = np.asarray(result.counterfactual, dtype=float).ravel()
    return counterfactual[intervention_t:]


def factor_counterfactual(
    panel: np.ndarray,
    intervention_t: int,
    latent_dim: int,
    model: str,
    include_previous_outcome: bool,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    """Posterior-mean counterfactual, the aggregation upstream uses in Figure 8."""
    train = panel[:intervention_t]
    if model == "GAP":
        posterior = gap_gibbs(train, latent_dim, n_samples=n_samples, warmup=n_samples, seed=seed)
    elif model == "PPCA":
        posterior = ppca_em(train, latent_dim, n_samples=n_samples, seed=seed)
    else:
        raise ValueError(model)

    draws = [
        counterfactual_from_regression(
            posterior.Z[s], panel, intervention_t, include_previous_outcome
        )
        for s in range(posterior.n_samples)
    ]
    return np.mean(draws, axis=0)


def mre(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    horizon = min(HORIZON, len(truth), len(prediction))
    p, t = prediction[:horizon], truth[:horizon]
    return np.abs(p - t) / np.maximum(t, 1e-9)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panels", type=Path, default=Path(__file__).parent / "panels")
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "bakeoff.csv")
    p.add_argument("--latent-dim", type=int, default=10)
    p.add_argument("--posterior-samples", type=int, default=150)
    p.add_argument("--teams", nargs="+", default=["indianapolis", "baltimore"])
    p.add_argument("--n-pre", nargs="+", type=int, default=[25, 100])
    p.add_argument("--mismatch", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    p.add_argument("--alpha-code", nargs="+", type=float, default=[250.0, 100000.0])
    p.add_argument("--data-seeds", nargs="+", type=int, default=[617, 781])
    args = p.parse_args()

    rows: list[dict] = []
    started = time.time()
    grid = list(
        itertools.product(args.teams, args.n_pre, args.mismatch, args.alpha_code, args.data_seeds)
    )
    for i, (team, n_pre, mismatch, alpha, seed) in enumerate(grid, start=1):
        panel, intervention_t = load_panel(args.panels / f"{team}.parquet")
        if n_pre > intervention_t:
            continue
        # Upstream truncates the pre-period from the left, keeping the post block.
        window = panel[intervention_t - n_pre :]
        synthetic = create_semi_synthetic_matrix(
            window,
            intervention_t=n_pre,
            alpha=alpha,
            latent_dim=args.latent_dim,
            seed=seed,
            model_mismatch_p=mismatch,
            effect_form="code",
        )
        observed, truth = synthetic.observed, synthetic.truth

        base = dict(
            team=team, n_pre=n_pre, mismatch=mismatch, alpha_code=alpha,
            alpha_paper=1000.0 / alpha, data_seed=seed,
        )

        try:
            rsc = rsc_counterfactual(observed, n_pre, rank=args.latent_dim)
            for lag in (False, True):
                for t, err in enumerate(mre(rsc, truth), start=1):
                    rows.append({**base, "model": "rSC", "include_previous_outcome": lag,
                                 "day": t, "relative_error": float(err)})
        except Exception as exc:  # pragma: no cover - recorded, not swallowed
            rows.append({**base, "model": "rSC", "include_previous_outcome": None,
                         "day": None, "relative_error": None, "error": str(exc)})

        for model in ("GAP", "PPCA"):
            for lag in (False, True):
                prediction = factor_counterfactual(
                    observed, n_pre, args.latent_dim, model, lag,
                    n_samples=args.posterior_samples, seed=seed,
                )
                for t, err in enumerate(mre(prediction, truth), start=1):
                    rows.append({**base, "model": model, "include_previous_outcome": lag,
                                 "day": t, "relative_error": float(err)})

        print(
            f"[{i:>3}/{len(grid)}] {team:<13} n_pre={n_pre:<4} mismatch={mismatch:<4} "
            f"alpha={alpha:<9.0f} seed={seed}  ({time.time() - started:6.1f}s)",
            flush=True,
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(frame)} rows, {time.time() - started:.1f}s)")


if __name__ == "__main__":
    main()
