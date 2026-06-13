"""Benchmark: SPOTSYNTH vs O'Riordan & Gilligan-Lee (2025).

Reproduces the donor-spillover-detection results of

    O'Riordan & Gilligan-Lee (2025). "Spillover detection for donor selection in
    synthetic control models." J. Causal Inference 13:20240036.

SPOTSYNTH screens every candidate donor for spillover contamination -- a valid
donor's post-intervention value is forecastable from the other donors'
pre-intervention data (Theorem 3.1); a forecast failure flags an invalid donor --
then builds a simplex synthetic control on the donors judged valid.

Two faithful reproductions:

* **Real data (Figure 6), Path A semi-synthetic.** On the German Reunification,
  California Tobacco Control, *and* Basque Country / ETA panels, plant a
  *semi-synthetic* invalid donor: a noisy proxy of the target,
  ``x_syn ~ N(y, sigma)``. Because it tracks the target it earns a large SC weight
  and biases the unscreened effect toward zero; **both** screening rules -- ``S1``
  (keep the ``n`` best-forecast donors) and ``S2`` (drop donors whose realised
  post value falls outside the forecast PPI) -- flag and exclude it, recovering
  the effect consistent with Abadie et al. The paper keeps 10 donors (Germany) /
  30 (California) under S1.
* **Simulation (Figure 2 / power), Path B.** Detection AUC (probability an
  invalid donor scores more anomalous than a valid one): the leave-one-out screen
  detects under a *valid majority* and -- a documented limit -- inverts under an
  *invalid majority*.

Provenance
----------
* Data: ``basedata/german_reunification.csv`` (16 donors),
  ``basedata/smoking_data.csv`` (38 donors, California), and
  ``basedata/basque_data.csv`` (16 regions, Basque Country target) -- the canonical
  Abadie synthetic-control datasets (refs [11,12,13]).
* Headline: with screening the effect is recovered (California ATT ~ -22, Basque
  ~ -1.2, the canonical ADH/ABG magnitudes) and the planted proxy is excluded by
  both S1 and S2; without screening the proxy biases the effect toward zero
  (Figure 6, top vs bottom panel).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_BASE = Path(__file__).resolve().parents[2] / "basedata"
_FRAC = 0.02      # noisy-proxy std as a fraction of the target's std
_SEED = 0


def _contaminated(df, outcome, treat, unit, time, target):
    """Plant a noisy-proxy invalid donor (a noisy copy of the target)."""
    rng = np.random.default_rng(_SEED)
    tgt = df[df[unit] == target].sort_values(time)
    proxy = tgt.copy()
    proxy[unit] = "proxy_invalid"
    proxy[treat] = False
    proxy[outcome] = tgt[outcome].to_numpy() + rng.normal(
        0, _FRAC * tgt[outcome].std(), len(tgt))
    return pd.concat([df, proxy], ignore_index=True)


def _screen(dc, outcome, treat, unit, time, **selection_kwargs):
    """Run SPOTSYNTH; return (proxy_excluded, screened ATT, unscreened ATT)."""
    from mlsynth import SPOTSYNTH

    res = SPOTSYNTH({
        "df": dc, "outcome": outcome, "treat": treat, "unitid": unit,
        "time": time, "display_graphs": False, "forecast": "loo",
        **selection_kwargs,
    }).fit()
    excluded = {res.screen.donor_names[i] for i in np.asarray(res.screen.excluded_idx)}
    return float("proxy_invalid" in excluded), float(res.att), float(res.att_unscreened)


def _debias_eiv(n_trials=30):
    """Sensitivity analysis (Section 3.3 / Figure 4): with errors-in-variables
    (donors are *noisy* proxies of the latents) even a perfect valid-donor SC is
    attenuation-biased; the proximal two-stage debias (eq. 5) -- regressing the
    kept donors on the screen-*excluded* donors -- reduces that bias. Returns the
    mean |bias| of the naive SC and of the debiased estimate over the paper's
    errors-in-variables DGP (no treatment effect, so the truth is 0)."""
    from mlsynth.utils.spotsynth_helpers import proximal_debias

    naive_bias, debiased_bias = [], []
    for seed in range(n_trials):
        rng = np.random.default_rng(seed)
        T, T0, k = 50, 40, 3
        sig = np.cumsum(rng.normal(0, 1, (T, k)), axis=0)         # latent factors
        beta = np.array([0.5, 0.3, 0.2])
        y = sig @ beta + rng.normal(0, 0.1, T)                    # target, no effect
        X = sig + rng.normal(0, 1.0, (T, k))                      # kept donors (noisy proxies)
        Z = np.column_stack([sig + rng.normal(0, 1.0, (T, k)) for _ in range(2)])  # excluded proxies
        Xc = np.column_stack([np.ones(T0), X[:T0]])
        b = np.linalg.lstsq(Xc, y[:T0], rcond=None)[0]
        naive_bias.append(abs(float(np.mean((y - (b[0] + X @ b[1:]))[np.arange(T) >= T0]))))
        debiased_bias.append(abs(float(proximal_debias(y, X, Z, T0).att)))
    return float(np.mean(naive_bias)), float(np.mean(debiased_bias))


def run() -> dict:
    from mlsynth.utils.spotsynth_helpers import run_forecast_power_analysis

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ger = _contaminated(pd.read_csv(_BASE / "german_reunification.csv"),
                            "gdp", "Reunification", "country", "year", "West Germany")
        gcols = ("gdp", "Reunification", "country", "year")
        g1_excl, g_scr, g_unscr = _screen(ger, *gcols, selection="S1", n_donors=10)
        g2_excl, _, _ = _screen(ger, *gcols, selection="S2", ppi=0.95)

        cal = _contaminated(pd.read_csv(_BASE / "smoking_data.csv"),
                            "cigsale", "Proposition 99", "state", "year", "California")
        ccols = ("cigsale", "Proposition 99", "state", "year")
        c1_excl, c_scr, c_unscr = _screen(cal, *ccols, selection="S1", n_donors=30)
        c2_excl, _, _ = _screen(cal, *ccols, selection="S2", ppi=0.95)

        # Basque Country / ETA terrorism (Abadie-Gardeazabal 2003); build the
        # treatment flag (pre-period 1955-1969, treated from 1970).
        bsq = pd.read_csv(_BASE / "basque_data.csv")[["regionname", "year", "gdpcap"]].dropna()
        bsq["treat"] = ((bsq.regionname == "Basque Country (Pais Vasco)") &
                        (bsq.year >= 1970)).astype(int)
        bsq = _contaminated(bsq, "gdpcap", "treat", "regionname", "year",
                            "Basque Country (Pais Vasco)")
        bcols = ("gdpcap", "treat", "regionname", "year")
        b1_excl, b_scr, b_unscr = _screen(bsq, *bcols, selection="S1", n_donors=10)
        b2_excl, _, _ = _screen(bsq, *bcols, selection="S2", ppi=0.95)

        # Simulation: leave-one-out detection AUC (valid vs invalid majority).
        power = run_forecast_power_analysis(
            n_donors=40, T0=50, n_post=24, invalid_fracs=(0.3, 0.8),
            ramps=(1, 24), n_reps=20, n_factors=8, verbose=False)

        # Sensitivity analysis (Figure 4): proximal debias reduces EIV bias.
        naive_bias, debiased_bias = _debias_eiv()

    return {
        "ger_proxy_excluded_s1": g1_excl,
        "ger_proxy_excluded_s2": g2_excl,
        "ger_att_screened": g_scr,
        "ger_att_unscreened": g_unscr,
        "ger_screening_recovers": float(abs(g_scr) > abs(g_unscr)),
        "cal_proxy_excluded_s1": c1_excl,
        "cal_proxy_excluded_s2": c2_excl,
        "cal_att_screened": c_scr,
        "cal_att_unscreened": c_unscr,
        "cal_screening_recovers": float(abs(c_scr) > abs(c_unscr)),
        "bsq_proxy_excluded_s1": b1_excl,
        "bsq_proxy_excluded_s2": b2_excl,
        "bsq_att_screened": b_scr,
        "bsq_att_unscreened": b_unscr,
        "bsq_screening_recovers": float(abs(b_scr) > abs(b_unscr)),
        "sim_loo_auc_valid_sharp": float(power[(0.3, 1)]["loo"]),
        "sim_loo_auc_valid_gradual": float(power[(0.3, 24)]["loo"]),
        "sim_loo_auc_invalid_sharp": float(power[(0.8, 1)]["loo"]),
        "debias_naive_bias": naive_bias,
        "debias_proximal_bias": debiased_bias,
        "debias_reduces_eiv_bias": float(debiased_bias < naive_bias),
    }


# The screen excludes the planted proxy on both real panels and recovers the
# canonical effect (California ATT ~ -22), while the unscreened estimate is biased
# toward zero; the simulation reproduces loo's detection AUC (high under a valid
# majority, inverted under an invalid majority -- the paper's documented regime).
EXPECTED = {
    # German reunification: both screens exclude the proxy and recover the effect.
    "ger_proxy_excluded_s1": (1.0, 0.0),
    "ger_proxy_excluded_s2": (1.0, 0.0),
    "ger_screening_recovers": (1.0, 0.0),
    # California tobacco control (canonical ADH magnitude, recovered).
    "cal_proxy_excluded_s1": (1.0, 0.0),
    "cal_proxy_excluded_s2": (1.0, 0.0),
    "cal_screening_recovers": (1.0, 0.0),
    "cal_att_screened": (-22.0, 5.0),
    # Basque Country / ETA terrorism (Abadie-Gardeazabal magnitude, recovered).
    "bsq_proxy_excluded_s1": (1.0, 0.0),
    "bsq_proxy_excluded_s2": (1.0, 0.0),
    "bsq_screening_recovers": (1.0, 0.0),
    "bsq_att_screened": (-1.2, 0.5),
    # Simulation: loo detection AUC -- high under a valid majority, inverted
    # under an invalid majority (the paper's documented regime).
    "sim_loo_auc_valid_sharp": (0.96, 0.15),
    "sim_loo_auc_valid_gradual": (0.92, 0.18),
    "sim_loo_auc_invalid_sharp": (0.0, 0.15),
    # Sensitivity analysis (Figure 4): the proximal debias reduces the
    # errors-in-variables attenuation bias that survives a perfect selection.
    "debias_reduces_eiv_bias": (1.0, 0.0),
}
