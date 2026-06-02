"""Replications of O'Riordan & Gilligan-Lee (2025).

* **Path B** -- the paper's simulation study (Section 4.1 / Figure 2). On the
  Appendix B data-generating process, a synthetic control built on *all* donors
  is biased by the spillover-contaminated ones (bias ~1.6), one built on the
  *valid* donors is unbiased, and the ``S1`` / ``S2`` donor-selection procedures
  recover most of that gap, degrading as the donor noise approaches the
  spillover magnitude.
* **Path A (semi-synthetic)** -- the real-data demonstrations (Section 4.2 /
  Figure 6) on California tobacco control and German reunification. A
  semi-synthetic donor that is a noisy proxy of the treated unit grabs a large
  synthetic-control weight and biases the effect toward zero; the screen flags
  and excludes it, restoring the canonical effect.

Every reproduction is driven through the public :class:`mlsynth.SPOTSYNTH`
estimator (``.fit()``), not the internal helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from .simulation import simulate_spillover_panel

PROP99_URL = ("https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
              "main/basedata/P99data.csv")
GERMANY_URL = ("https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
               "main/basedata/german_reunification.csv")


# ---------------------------------------------------------------------------
# Path B: the simulation study (Figure 2)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpotSimConfig:
    """Parameters for the SPOTSYNTH simulation study (paper Section 4.1)."""

    n_donors: int = 120
    n_latent: int = 10
    T0: int = 80
    n_post: int = 20
    frac_invalid: float = 0.8
    tau: float = 2.0
    spillover: float = -2.0
    n_keep: int = 15
    ppi: float = 0.8
    n_factors: int = 5
    n_reps: int = 20
    noise_levels: tuple = (0.1, 0.5, 1.0)


PAPER = SpotSimConfig(n_donors=1000, T0=100, n_post=30, n_keep=10, n_reps=2000)
DEMO = SpotSimConfig()


def _att(df: pd.DataFrame, *, selection: str, forecast: str = "lag",
         n_donors=None, ppi: float = 0.8, n_factors: int = 5) -> float:
    """Fit the public SPOTSYNTH estimator on ``df`` and return its ATT."""
    from ...estimators.spotsynth import SPOTSYNTH

    res = SPOTSYNTH({
        "df": df, "outcome": "Y", "treat": "treated",
        "unitid": "unit", "time": "time",
        "selection": selection, "forecast": forecast,
        "n_donors": n_donors, "ppi": ppi, "n_factors": n_factors,
        "inference": "frequentist",   # fast point estimate for the many-rep study
        "display_graphs": False,
    }).fit()
    return res.att


def run_spotsynth_simulation(cfg: SpotSimConfig = DEMO, *, seed: int = 0,
                             verbose: bool = True) -> Dict[float, Dict[str, float]]:
    """Reproduce the Figure 2 bias finding through ``SPOTSYNTH.fit()``.

    For each donor-noise level returns the mean bias ``E[tau_hat] - tau`` of four
    strategies: ``All`` (every donor), ``Valid`` (oracle -- only the truly valid
    donors), ``S1`` (smallest forecast error), and ``S2`` (inside the PPI).

    Returns
    -------
    dict
        ``{sigma_x: {"All": bias, "Valid": bias, "S1": bias, "S2": bias}}``.
    """
    out: Dict[float, Dict[str, float]] = {}
    if verbose:
        print(f"SPOTSYNTH simulation (Figure 2), {cfg.n_reps} reps, "
              f"{cfg.n_donors} donors, {int(cfg.frac_invalid*100)}% invalid:")
        print(f"{'noise':>7} {'All':>8} {'Valid':>8} {'S1':>8} {'S2':>8}")
    for sigma_x in cfg.noise_levels:
        acc = {k: [] for k in ("All", "Valid", "S1", "S2")}
        for rep in range(cfg.n_reps):
            df, valid_mask = simulate_spillover_panel(
                n_donors=cfg.n_donors, n_latent=cfg.n_latent, T0=cfg.T0,
                n_post=cfg.n_post, sigma_x=sigma_x, frac_invalid=cfg.frac_invalid,
                tau=cfg.tau, spillover=cfg.spillover, seed=seed + rep,
            )
            valid_names = ["target"] + [
                n for n, ok in zip(sorted(df.loc[df["unit"] != "target", "unit"].unique()),
                                   valid_mask) if ok]
            df_valid = df[df["unit"].isin(valid_names)]
            acc["All"].append(_att(df, selection="all", n_factors=cfg.n_factors))
            acc["Valid"].append(_att(df_valid, selection="all", n_factors=cfg.n_factors))
            acc["S1"].append(_att(df, selection="S1", n_donors=cfg.n_keep,
                                  n_factors=cfg.n_factors))
            acc["S2"].append(_att(df, selection="S2", ppi=cfg.ppi,
                                  n_factors=cfg.n_factors))
        bias = {k: float(np.mean(v) - cfg.tau) for k, v in acc.items()}
        out[sigma_x] = bias
        if verbose:
            print(f"{sigma_x:>7} {bias['All']:>+8.2f} {bias['Valid']:>+8.2f} "
                  f"{bias['S1']:>+8.2f} {bias['S2']:>+8.2f}")
    return out


# ---------------------------------------------------------------------------
# Path A (semi-synthetic): real-data demonstrations (Figure 6)
# ---------------------------------------------------------------------------

def _semi_synthetic_demo(df_wide_long: pd.DataFrame, *, unit: str, time: str,
                         outcome: str, treated_unit, t0, n_keep: int,
                         forecast: str, sigma: float, seed: int, label: str,
                         verbose: bool):
    """Plant a noisy proxy of the treated unit, then screen + fit SPOTSYNTH."""
    from ...estimators.spotsynth import SPOTSYNTH

    d = df_wide_long.copy()
    d = d.rename(columns={unit: "unit", time: "time", outcome: "Y"})
    d["treated"] = ((d["unit"] == treated_unit) & (d["time"] >= t0)).astype(int)

    # Oracle: no contamination (fast frequentist baseline).
    oracle = SPOTSYNTH({"df": d, "outcome": "Y", "treat": "treated",
                        "unitid": "unit", "time": "time", "selection": "all",
                        "inference": "frequentist",
                        "display_graphs": False}).fit().att

    # Semi-synthetic invalid donor: a noisy proxy of the treated series.
    rng = np.random.default_rng(seed)
    tgt = d[d["unit"] == treated_unit].sort_values("time")
    synth = tgt.copy()
    synth["unit"] = "__synthetic__"
    synth["Y"] = tgt["Y"].to_numpy() + rng.normal(0.0, sigma, len(tgt))
    synth["treated"] = 0
    d_contam = pd.concat([d, synth], ignore_index=True)

    contam = SPOTSYNTH({"df": d_contam, "outcome": "Y", "treat": "treated",
                        "unitid": "unit", "time": "time", "selection": "all",
                        "inference": "frequentist",
                        "display_graphs": False}).fit().att

    # Screened fit: the authors' Bayesian Dirichlet SC, so we also get the
    # 95% posterior-predictive credible interval shown in Figure 6.
    res = SPOTSYNTH({"df": d_contam, "outcome": "Y", "treat": "treated",
                     "unitid": "unit", "time": "time", "selection": "S1",
                     "n_donors": n_keep, "forecast": forecast,
                     "inference": "bayes", "seed": seed,
                     "display_graphs": False}).fit()
    excluded = "__synthetic__" in res.screen.excluded_names
    if verbose:
        ci = res.att_ci
        ci_str = f"  95% CrI=({ci[0]:+.2f}, {ci[1]:+.2f})" if ci else ""
        print(f"{label}: oracle ATT={oracle:+.2f}  contaminated={contam:+.2f}  "
              f"screened={res.att:+.2f}{ci_str}  synthetic-donor excluded={excluded}")
    return {"oracle": oracle, "contaminated": contam, "screened": res.att,
            "excluded": excluded, "results": res}


def replicate_prop99_spillover(data: Union[str, pd.DataFrame, None] = None, *,
                               n_keep: int = 30, sigma: float = 0.5,
                               seed: int = 0, verbose: bool = True):
    """California tobacco control with a planted spillover donor (Figure 6b).

    Loads the 39-state Abadie tobacco panel, plants a semi-synthetic donor that
    is a noisy proxy of California, and runs ``SPOTSYNTH`` ``S1`` (keep 30). The
    invalid donor is flagged and excluded, restoring the canonical ~-20 effect.
    """
    if data is None:
        data = PROP99_URL
    df = pd.read_csv(data) if isinstance(data, str) else data.copy()
    return _semi_synthetic_demo(
        df[["state", "year", "cigsale"]], unit="state", time="year",
        outcome="cigsale", treated_unit="California", t0=1989, n_keep=n_keep,
        forecast="lag", sigma=sigma, seed=seed, label="Prop 99 (California)",
        verbose=verbose)


def replicate_germany_spillover(data: Union[str, pd.DataFrame, None] = None, *,
                                n_keep: int = 12, sigma: float = 20.0,
                                seed: int = 0, verbose: bool = True):
    """German reunification with a planted spillover donor (Figure 6a).

    Loads the 17-country OECD panel, plants a semi-synthetic donor that is a
    noisy proxy of West Germany, and runs ``SPOTSYNTH`` ``S1`` (keep 10) with the
    leave-one-out forecast (the reunification effect builds slowly). The invalid
    donor is excluded, restoring the large negative per-capita-GDP effect.
    """
    if data is None:
        data = GERMANY_URL
    df = pd.read_csv(data) if isinstance(data, str) else data.copy()
    return _semi_synthetic_demo(
        df[["country", "year", "gdp"]], unit="country", time="year",
        outcome="gdp", treated_unit="West Germany", t0=1990, n_keep=n_keep,
        forecast="loo", sigma=sigma, seed=seed, label="Reunification (Germany)",
        verbose=verbose)


if __name__ == "__main__":
    run_spotsynth_simulation(DEMO)
    replicate_prop99_spillover("../../../basedata/P99data.csv")
    replicate_germany_spillover("../../../basedata/german_reunification.csv")
