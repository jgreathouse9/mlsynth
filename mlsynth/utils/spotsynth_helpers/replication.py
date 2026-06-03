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
BASQUE_URL = ("https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
              "main/basedata/basque_data.csv")


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
    """Fit the public SPOTSYNTH estimator on ``df`` and return its ATT.

    The Figure 2 study is the paper's *mostly-invalid* regime (80% of donors
    contaminated), so it pins ``forecast="lag"`` -- the leave-one-out anchor
    (the package default) provably inverts when the contaminated donors form the
    majority, since they define the "consensus" the screen forecasts against.
    """
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


def _auc(score: np.ndarray, is_invalid: np.ndarray) -> float:
    """AUC: P(an invalid donor scores more anomalous than a valid one)."""
    from scipy.stats import rankdata
    r = rankdata(score)
    n1 = int(is_invalid.sum())
    n0 = int((~is_invalid).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[is_invalid].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def run_forecast_power_analysis(
    *, n_donors: int = 60, T0: int = 60, n_post: int = 30, sigma_x: float = 0.5,
    invalid_fracs=(0.3, 0.8), ramps=(1, 6, 24), n_factors: int = 8,
    n_reps: int = 30, seed: int = 0, verbose: bool = True,
) -> Dict:
    """Detection power (AUC) of the ``lag`` vs ``loo`` anchors vs onset speed.

    Reproduces the analysis that motivates ``loo`` as the default: for each
    contamination fraction and spillover *onset speed* (``ramp``; 1 = sharp,
    larger = gradual), the detection AUC of the two shipped forecast anchors --
    where AUC is the probability an invalid donor's forecast statistic is more
    anomalous than a valid donor's (1 = perfect, 0.5 = none, < 0.5 = inverted).

    The headline findings it reproduces:

    * **Valid majority (e.g. 30% invalid):** ``loo`` is near-perfect and
      onset-robust (AUC ~0.95+ for sharp *and* gradual); ``lag`` (first
      post-point) only has power for sharp onsets and decays to chance as the
      onset becomes gradual.
    * **Invalid majority (80%, the paper's regime):** ``loo`` *inverts*
      (AUC ~0, it flags the valid donors); ``lag`` is the only anchor with power,
      and only for sharp onsets.
    * **Gradual onset + invalid majority:** neither anchor has power -- the
      honest limit of forecast-based spillover detection.

    Returns
    -------
    dict
        ``{(frac_invalid, ramp): {"lag": auc, "loo": auc}}``.
    """
    from .screen import spillover_screen

    out: Dict = {}
    if verbose:
        print("Detection AUC (1=perfect, 0.5=none, <0.5=inverted), "
              f"{n_reps} reps, final spillover -2:")
        print(f"{'invalid%':>8} {'onset':>8} {'lag':>8} {'loo':>8}")
    for frac in invalid_fracs:
        for ramp in ramps:
            la, lo = [], []
            for rep in range(n_reps):
                df, valid = simulate_spillover_panel(
                    n_donors=n_donors, T0=T0, n_post=n_post, sigma_x=sigma_x,
                    frac_invalid=frac, spillover_ramp=ramp, seed=seed + rep)
                # rebuild the donor matrix in mlsynth's sorted-name order
                w = df.pivot(index="time", columns="unit", values="Y")
                donors = [c for c in w.columns if c != "target"]
                D = w[donors].to_numpy()
                inv = ~np.asarray(valid)
                for anchor, acc in (("lag", la), ("loo", lo)):
                    sc = spillover_screen(D, T0, donors, selection="all",
                                          forecast=anchor, n_factors=n_factors)
                    acc.append(_auc(sc.forecast_error, inv))
            res = {"lag": float(np.nanmean(la)), "loo": float(np.nanmean(lo))}
            out[(frac, ramp)] = res
            if verbose:
                lab = "sharp" if ramp == 1 else ("gradual" if ramp >= 24 else "")
                print(f"{int(frac*100):>7}% {str(ramp)+' '+lab:>8} "
                      f"{res['lag']:>8.3f} {res['loo']:>8.3f}")
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

    # Screened fit: the authors' Bayesian Dirichlet SC (NumPyro NUTS), so we
    # also get the 95% posterior-predictive credible interval.
    res = SPOTSYNTH({"df": d_contam, "outcome": "Y", "treat": "treated",
                     "unitid": "unit", "time": "time", "selection": "S1",
                     "n_donors": n_keep, "forecast": forecast,
                     "inference": "bayes", "seed": seed,
                     "display_graphs": False}).fit()
    excluded = "__synthetic__" in res.screen.excluded_names

    # Standard diagnostics off the screened fit.
    T0 = res.inputs.T0
    pre_rmse = float(np.sqrt(np.mean(
        (res.inputs.y[:T0] - res.counterfactual[:T0]) ** 2)))
    weights = {k: round(v, 4) for k, v in
               sorted(res.donor_weights.items(), key=lambda kv: -kv[1])
               if v > 1e-4}

    out = {
        "label": label,
        "oracle_att": oracle,
        "contaminated_att": contam,
        "screened_att": res.att,
        "att_ci": res.att_ci,
        "pre_rmse": pre_rmse,
        "donor_weights": weights,
        "selected_donors": res.screen.selected_names,
        "excluded_donors": res.screen.excluded_names,
        "synthetic_donor_excluded": excluded,
        "n_selected": res.metadata["n_selected"],
        "n_excluded": res.metadata["n_excluded"],
        "results": res,
    }
    if verbose:
        ci = res.att_ci
        ci_str = f"  95% CrI=({ci[0]:+.2f}, {ci[1]:+.2f})" if ci else ""
        top = "  ".join(f"{k}={v:.2f}" for k, v in list(weights.items())[:3])
        print(f"{label}:")
        print(f"  oracle ATT={oracle:+.3f}  contaminated={contam:+.3f}  "
              f"screened ATT={res.att:+.3f}{ci_str}")
        print(f"  pre-treatment RMSE={pre_rmse:.3f}  donors kept={out['n_selected']}"
              f"/{out['n_selected']+out['n_excluded']}  "
              f"synthetic donor excluded={excluded}")
        print(f"  top SC weights: {top}")
    return out


def replicate_prop99_spillover(data: Union[str, pd.DataFrame, None] = None, *,
                               n_keep: int = 30, sigma: float = 0.5,
                               seed: int = 0, verbose: bool = True):
    """California tobacco control with a planted spillover donor (Figure 6b).

    Loads the 39-state Abadie tobacco panel, plants a semi-synthetic donor that
    is a noisy proxy of California, and runs ``SPOTSYNTH`` ``S1`` (keep 30) with
    the default leave-one-out (``loo``) forecast anchor. The invalid donor is
    flagged and excluded, restoring the canonical ~-20 effect.
    """
    if data is None:
        data = PROP99_URL
    df = pd.read_csv(data) if isinstance(data, str) else data.copy()
    return _semi_synthetic_demo(
        df[["state", "year", "cigsale"]], unit="state", time="year",
        outcome="cigsale", treated_unit="California", t0=1989, n_keep=n_keep,
        forecast="loo", sigma=sigma, seed=seed, label="Prop 99 (California)",
        verbose=verbose)


def replicate_germany_spillover(data: Union[str, pd.DataFrame, None] = None, *,
                                n_keep: int = 12, sigma: float = 20.0,
                                seed: int = 0, verbose: bool = True):
    """German reunification with a planted spillover donor (Figure 6a).

    Loads the 17-country OECD panel, plants a semi-synthetic donor that is a
    noisy proxy of West Germany, and runs ``SPOTSYNTH`` ``S1`` (keep 12) with the
    default leave-one-out forecast (the reunification effect builds slowly). The
    invalid donor is excluded, restoring the large negative per-capita-GDP effect.
    """
    if data is None:
        data = GERMANY_URL
    df = pd.read_csv(data) if isinstance(data, str) else data.copy()
    return _semi_synthetic_demo(
        df[["country", "year", "gdp"]], unit="country", time="year",
        outcome="gdp", treated_unit="West Germany", t0=1990, n_keep=n_keep,
        forecast="loo", sigma=sigma, seed=seed, label="Reunification (Germany)",
        verbose=verbose)


def replicate_basque_spillover(data: Union[str, pd.DataFrame, None] = None, *,
                               n_keep: int = 12, sigma: float = 0.1,
                               seed: int = 0, verbose: bool = True):
    """Basque Country (ETA terrorism, 1975) with a planted spillover donor.

    A third canonical SC panel (Abadie & Gardeazabal 2003), *not* in the
    O'Riordan & Gilligan-Lee paper -- an additional robustness check. Loads the
    17-region Spanish panel, plants a semi-synthetic donor that is a noisy proxy
    of the Basque Country, and runs ``SPOTSYNTH`` ``S1`` (keep 12) with the
    default leave-one-out forecast. Because the ETA effect builds gradually, this
    is exactly the case the ``loo`` anchor is designed for; the invalid donor is
    excluded and the ~-0.7 (thousand 1986 USD) per-capita-GDP effect restored.
    """
    if data is None:
        data = BASQUE_URL
    df = pd.read_csv(data) if isinstance(data, str) else data.copy()
    return _semi_synthetic_demo(
        df[["regionname", "year", "gdpcap"]], unit="regionname", time="year",
        outcome="gdpcap", treated_unit="Basque Country (Pais Vasco)", t0=1975,
        n_keep=n_keep, forecast="loo", sigma=sigma, seed=seed,
        label="Basque Country (ETA)", verbose=verbose)


def replicate_all_spillover(*, verbose: bool = True) -> Dict[str, Dict]:
    """Run all three real-data spillover demonstrations with the ``loo`` default.

    Reproduces, end-to-end through ``SPOTSYNTH.fit()``, the semi-synthetic
    contamination-and-recovery on California tobacco control, German
    reunification, and the Basque Country -- each returning the oracle /
    contaminated / screened ATTs, the screened 95% credible interval, the
    pre-treatment RMSE, the synthetic-control donor weights, and the selected /
    excluded donor sets.

    Returns
    -------
    dict
        ``{"prop99": ..., "germany": ..., "basque": ...}``, each the rich result
        dict from the corresponding ``replicate_*`` function.
    """
    out = {
        "prop99": replicate_prop99_spillover(verbose=verbose),
        "germany": replicate_germany_spillover(verbose=verbose),
        "basque": replicate_basque_spillover(verbose=verbose),
    }
    if verbose:
        print("\nSummary (loo forecast, Bayesian Dirichlet SC):")
        print(f"{'panel':24} {'oracle':>9} {'contam':>9} {'screened':>10} "
              f"{'pre-RMSE':>9}  synth excl")
        for r in out.values():
            print(f"{r['label']:24} {r['oracle_att']:>+9.3f} "
                  f"{r['contaminated_att']:>+9.3f} {r['screened_att']:>+10.3f} "
                  f"{r['pre_rmse']:>9.3f}  {r['synthetic_donor_excluded']}")
    return out


if __name__ == "__main__":  # pragma: no cover - manual script entry point
    run_spotsynth_simulation(DEMO)
    replicate_all_spillover()
