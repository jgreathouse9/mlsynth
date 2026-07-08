"""SMC cross-validation + Path A: the Basque Country / ETA-terrorism study.

Zhu, Rong J. B. (2023), *"Synthetic Matching Control Method,"* arXiv:2306.02584,
re-examines the Abadie & Gardeazabal (2003) study of the per-capita GDP cost of
ETA terrorism (17 Spanish regions, 1955-1997). SMC matches each donor to the
Basque Country by a univariate regression, then synthesises the matched controls
with box-``[0, 1]`` weights chosen by a Mallows/Cp unbiased-risk criterion.

Primary validation is a **cross-validation against the author's reference R
implementation** (``Code_SMC.R``): on the identical Basque matching matrix, the
mlsynth weight computation matches the reference ``SMCV`` (per-donor OLS + the
Cp box-QP solved by ``solve.QP``) to machine precision -- ``theta``, the box
weights, the combined coefficients, ``bias`` and ``sigma^2`` all agree to
``< 2e-13`` (see ``docs/replications/smc.rst`` for the harness).

This case runs two arms through the public estimator on
``basedata/basque_data.csv``:

* the deterministic Algorithm 1 (outcome-only matching) -- exact re-runs, the
  ``Cp`` penalty identifies the weights;
* the paper's Algorithm 3 spec (covariate matching + ``fit_window`` +
  ``v_search="de"``), which rebuilds the paper's matching matrix and reproduces
  its Table 5 / Figure 1 donor structure (Rioja dominant, then Madrid) and ATT
  magnitude. The ``V`` search is seeded, so the arm is reproducible; the exact
  split among the top donors is a draw from the non-identified ``V`` manifold and
  is asserted only structurally.

  =========================  ==============  ==============================
  Quantity                   mlsynth SMC     note
  =========================  ==============  ==============================
  pre-period RMSE            ~0.048          tight pre-treatment fit
  mean post-1969 ATT         ~-0.858         terrorism depresses GDP/capita
  1997 gap                   ~-0.848
  donors (combined coef)     Murcia 0.63,    box-weighted matched controls
                             Madrid 0.37,
                             Castilla y Leon 0.24
  =========================  ==============  ==============================

Provenance: Zhu (2023) Section 5.2; the reference R ``Code_SMC.R``; Abadie &
Gardeazabal (2003) for the original study.
"""
from __future__ import annotations

import os

import pandas as pd

_DATA = os.path.join(
    os.path.dirname(__file__), "..", "..", "basedata", "basque_data.csv")
_TREATED = "Basque Country (Pais Vasco)"


_COVS = ["school.illit", "school.prim", "school.med", "school.high", "invest",
         "sec.agriculture", "sec.energy", "sec.industry", "sec.construction",
         "sec.services.venta", "sec.services.nonventa", "popdens"]
_WINS = {**{c: (1964, 1969) for c in
            ["school.illit", "school.prim", "school.med", "school.high", "invest"]},
         **{c: (1961, 1969) for c in
            ["sec.agriculture", "sec.energy", "sec.industry", "sec.construction",
             "sec.services.venta", "sec.services.nonventa", "popdens"]}}


def run() -> dict:
    from mlsynth import SMC

    df = pd.read_csv(os.path.abspath(_DATA))
    df["treat"] = ((df["regionname"] == _TREATED) & (df["year"] >= 1970)).astype(int)
    base = dict(df=df, outcome="gdpcap", treat="treat", unitid="regionname",
                time="year", display_graphs=False)

    # Arm 1: deterministic Algorithm 1 (outcome-only).
    res = SMC(base).fit()
    w = {str(k): float(v) for k, v in res.donor_weights.items()}
    gap = res.time_series.observed_outcome - res.time_series.counterfactual_outcome

    # Arm 2: the paper's Algorithm 3 spec (covariates + fit window + seeded V search).
    pap = SMC({**base, "covariates": _COVS, "covariate_windows": _WINS,
               "fit_window": (1960, 1969), "v_search": "de", "v_seed": 0}).fit()
    pw = {str(k): float(v) for k, v in pap.donor_weights.items()}
    top = max(pw, key=lambda k: abs(pw[k]))

    return {
        "att": float(res.att),
        "pre_rmse": float(res.fit_diagnostics.rmse_pre),
        "gap_1997": float(gap[-1]),
        "murcia_coef": w.get("Murcia (Region de)", 0.0),
        "madrid_coef": w.get("Madrid (Comunidad De)", 0.0),
        # Paper (Table 5 / Fig 1) arm:
        "paper_att": float(pap.att),
        "paper_rioja_coef": pw.get("Rioja (La)", 0.0),
        "paper_top_is_rioja": 1.0 if top == "Rioja (La)" else 0.0,
    }


# Arm 1 is deterministic (Cp-identified, no V search) -- exact re-runs. Arm 2 is
# the paper's covariate + seeded-V-search spec: it reproduces the Table 5 donor
# structure (Rioja dominant) and the Fig 1 ATT magnitude; the exact Rioja weight
# is a draw from the non-identified V manifold, so it is bounded, not pinned.
EXPECTED = {
    "att": (-0.8575, 0.02),
    "pre_rmse": (0.0481, 0.01),
    "gap_1997": (-0.8484, 0.02),
    "murcia_coef": (0.625, 0.05),
    "madrid_coef": (0.370, 0.05),
    "paper_att": (-1.37, 0.35),                 # paper Fig 1 mean ATT ~ -1.4
    "paper_rioja_coef": (0.70, 0.30),           # Rioja dominant (paper 0.567)
    "paper_top_is_rioja": (1.0, 0.0),
}
