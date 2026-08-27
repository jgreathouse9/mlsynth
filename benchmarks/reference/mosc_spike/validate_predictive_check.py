"""Calibrate the holdout predictive check before trusting what it rejects.

The check rejected every model at every rank on the authors' panels. A screen
that never accepts is indistinguishable from a broken screen, so this establishes
the other half: on data drawn from the model being checked, ``p_pop`` must land
inside the paper's acceptance region. Section 4.3.1 states the property that makes
the rule a test -- "If the model is well-specified ``p_pop`` should be uniformly
distributed. Therefore, a policy that rejects a model unless it has ``p_pop`` in
``[alpha/2, 1 - alpha/2]`` will have a false rejection rate of ``alpha``."

It does not hold, and the reason is the aggregation, not the models.
Equation 36 sums the discrepancy over every held-out cell::

    d(Y, Z, Theta) = -log P(Y | Z, Theta)

and equation 35 compares that sum for a replicate against the same sum for the
real data, both scored at the same posterior draw. The replicate is drawn at the
fitted rate, so it is matched to that rate exactly, while the real data carries
the rate's estimation error. Per cell that leaves a small systematic gap in the
replicate's favour. Summed over ``n`` held-out cells the gap grows like ``n``
while its spread grows like ``sqrt(n)``, so the comparison becomes deterministic
and ``p_pop`` collapses onto 0 or 1.

This sweeps the number of held-out cells to show the collapse, and reports the
held-out predictive log density -- a score, with no calibration claim -- as the
model comparison the check was reached for.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mosc_port import _poisson_logpmf, gap_gibbs, population_predictive_check

LATENT_DIM = 4
SAMPLES = 120


def heldout_log_density(data, mask, posterior, likelihood="poisson") -> float:
    """Mean log predictive density per held-out cell. Higher is better."""
    held = ~mask.astype(bool)
    totals = []
    for s in range(posterior.n_samples):
        rate = np.clip(posterior.rate(s), 1e-10, None)
        if likelihood == "poisson":
            lp = _poisson_logpmf(np.rint(data[held]).astype(np.int64), rate[held])
        else:
            resid = data[mask.astype(bool)] - rate[mask.astype(bool)]
            scale = max(float(np.std(resid)), 1e-8)
            lp = -0.5 * np.log(2 * np.pi * scale**2) - 0.5 * ((data[held] - rate[held]) / scale) ** 2
        totals.append(float(np.mean(lp)))
    return float(np.mean(totals))


def main() -> dict:
    rng = np.random.default_rng(4242)
    n_time, n_unit = 60, 30
    H = rng.gamma(2.0, 2.0, size=(n_time, LATENT_DIM))
    Z = rng.gamma(2.0, 2.0, size=(LATENT_DIM, n_unit))
    data = rng.poisson(H @ Z).astype(float)  # drawn from the model being checked

    print("Data drawn from the gamma-Poisson model, checked against that model.")
    print("A calibrated screen accepts this. p_pop should sit inside [0.025, 0.975].\n")
    print("  held-out cells   p_pop    verdict")
    sweep = []
    for fraction in (0.005, 0.02, 0.05, 0.10, 0.25):
        seed = int(fraction * 10000)
        mask = np.random.default_rng(seed).random(data.shape) > fraction
        n_held = int((~mask).sum())
        posterior = gap_gibbs(data, LATENT_DIM, mask=mask, n_samples=SAMPLES, warmup=SAMPLES, seed=seed)
        p_pop = population_predictive_check(data, mask, posterior, "poisson", seed=seed)
        accepted = bool(0.025 <= p_pop <= 0.975)
        sweep.append({"holdout_fraction": fraction, "n_heldout_cells": n_held,
                      "p_pop": p_pop, "accepted": accepted})
        print(f"  {n_held:>14}   {p_pop:.3f}    {'accept' if accepted else 'REJECT'}")

    false_rejection_rate = 1.0 - np.mean([r["accepted"] for r in sweep])
    print(f"\n  false rejection rate on a correctly specified model: {false_rejection_rate:.2f}")
    print("  the paper's stated rate: 0.05")

    # The score the check was reached for, on the authors' own panel.
    here = Path(__file__).parent
    long = pd.read_parquet(here / "panels" / "indianapolis.parquet")
    wide = long.pivot(index="date", columns="county", values="cases").to_numpy(float)
    pre = wide[:198]
    scores = {}
    print("\nHeld-out predictive log density per cell on the authors' Indianapolis")
    print("pre-period panel, 10% held out, K=10 (higher is better):\n")
    for label, panel in (("cumulative (as shipped)", pre),
                         ("daily (first difference)", np.clip(np.diff(pre, axis=0), 0, None))):
        mask = np.random.default_rng(11).random(panel.shape) > 0.10
        gap = gap_gibbs(panel, 10, mask=mask, n_samples=SAMPLES, warmup=SAMPLES, seed=11)
        from mosc_port import ppca_em

        ppca = ppca_em(panel, 10, mask=mask, n_samples=SAMPLES, seed=11)
        s_gap = heldout_log_density(panel, mask, gap, "poisson")
        s_ppca = heldout_log_density(panel, mask, ppca, "gaussian")
        scores[label] = {"GAP": s_gap, "PPCA": s_ppca}
        print(f"  {label:<26} GAP {s_gap:>12.3f}   PPCA {s_ppca:>10.3f}   "
              f"{'GAP' if s_gap > s_ppca else 'PPCA'} wins")

    out = {
        "sweep": sweep,
        "false_rejection_rate_on_correct_model": float(false_rejection_rate),
        "paper_stated_false_rejection_rate": 0.05,
        "heldout_log_density": scores,
    }
    (here / "predictive_check_calibration.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


if __name__ == "__main__":
    main()
