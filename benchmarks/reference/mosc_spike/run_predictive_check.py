"""Run the paper's holdout predictive check on the real panels.

Equations 35-36. A randomly chosen fraction of cells is masked, the factor model
is fit to the rest, and the held-out cells are scored against replicates drawn
from the fitted model. The paper rejects a model unless ``p_pop`` lands inside
``[alpha/2, 1 - alpha/2]`` at ``alpha = 0.05``, and reports that GaP passes for
almost all of the 31 teams while PPCA returns 1.0 nearly everywhere.

The check is the gate that licenses the whole procedure -- Section 3.4 makes model
criticism what permits ``Z`` to stand in for the unobserved confounder -- so what
it does on these panels decides how much the licence is worth.

Upstream's ``create_mask`` holds out 1% under ``speckled`` where the paper says
10%, and its ``plaid`` and ``random`` branches compare a string against a list and
so hold out nothing at all. The paper's 10% is used here; ``--holdout`` varies it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mosc_port import gap_gibbs, population_predictive_check, ppca_em


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panels", type=Path, default=Path(__file__).parent / "panels")
    p.add_argument("--teams", nargs="+", default=["indianapolis", "baltimore"])
    p.add_argument("--latent-dims", nargs="+", type=int, default=[5, 10, 15, 25])
    p.add_argument("--holdout", type=float, default=0.10)
    p.add_argument("--masks", type=int, default=3)
    p.add_argument("--posterior-samples", type=int, default=150)
    p.add_argument("--scale", choices=["cumulative", "daily"], default="cumulative")
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "predictive_check.csv")
    args = p.parse_args()

    rows = []
    for team in args.teams:
        long = pd.read_parquet(args.panels / f"{team}.parquet")
        wide = long.pivot(index="date", columns="county", values="cases").to_numpy(float)
        treated_post = long.loc[long["stadium_open"] == 1, "date"].nunique()
        pre = wide[: len(wide) - treated_post]
        if args.scale == "daily":
            pre = np.clip(np.diff(pre, axis=0), 0, None)

        for mask_seed in range(args.masks):
            rng = np.random.default_rng(1000 + mask_seed)
            mask = rng.random(pre.shape) > args.holdout
            for K in args.latent_dims:
                gap = gap_gibbs(pre, K, mask=mask, n_samples=args.posterior_samples,
                                warmup=args.posterior_samples, seed=mask_seed)
                p_gap = population_predictive_check(pre, mask, gap, "poisson", seed=mask_seed)

                ppca = ppca_em(pre, K, mask=mask, n_samples=args.posterior_samples, seed=mask_seed)
                p_ppca = population_predictive_check(pre, mask, ppca, "gaussian", seed=mask_seed)

                for model, p_pop in (("GAP", p_gap), ("PPCA", p_ppca)):
                    rows.append({
                        "team": team, "scale": args.scale, "mask_seed": mask_seed,
                        "latent_dim": K, "model": model, "p_pop": p_pop,
                        "passes": bool(0.025 <= p_pop <= 0.975),
                    })
                print(f"  {team:<13} {args.scale:<10} mask={mask_seed} K={K:<3} "
                      f"GAP p_pop={p_gap:.3f} {'pass' if 0.025 <= p_gap <= 0.975 else 'REJECT':<6} "
                      f"PPCA p_pop={p_ppca:.3f} {'pass' if 0.025 <= p_ppca <= 0.975 else 'REJECT'}",
                      flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(args.out, index=False)
    print("\npass rate by model:")
    print(frame.groupby(["scale", "model"]).passes.mean().round(3).to_string())
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
