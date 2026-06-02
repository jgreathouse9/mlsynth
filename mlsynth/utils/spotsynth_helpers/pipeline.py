"""Orchestration for the SPOTSYNTH estimator (O'Riordan & Gilligan-Lee 2025)."""

from __future__ import annotations

import numpy as np

from .bayes import bayesian_simplex_sc
from .debias import proximal_debias
from .sc import simplex_weights
from .screen import spillover_screen
from .structures import SpotSynthInputs, SpotSynthResults


def run_spotsynth(
    inputs: SpotSynthInputs,
    *,
    selection: str = "S1",
    forecast: str = "lag",
    n_donors=None,
    ppi: float = 0.8,
    n_factors: int = 5,
    time_average=None,
    inference: str = "bayes",
    dirichlet_alpha: float = 0.4,
    ci_level: float = 0.95,
    n_samples: int = 4000,
    n_warmup: int = 2000,
    debias: bool = False,
    seed: int = 0,
) -> SpotSynthResults:
    """Screen donors for spillover, then fit a synthetic control on the valid set.

    Stages
    ------
    1. Spillover screen (Algorithm 1) over the donor pool -> valid-donor subset.
    2. Synthetic control on the selected donors: the authors' Bayesian Dirichlet
       simplex SC (``inference="bayes"``, with credible intervals) or a fast
       frequentist simplex (``inference="frequentist"``).
    3. ATT as the mean post-intervention gap; an unscreened (``All``) ATT is
       reported alongside for comparison.
    4. Optionally, the proximal (two-stage / GMM) debiased ATT using the
       excluded donors (``debias=True``).
    """
    y, D, T0 = inputs.y, inputs.D, inputs.T0
    post = np.arange(inputs.T) >= T0

    screen = spillover_screen(
        D, T0, inputs.donor_names, selection=selection, forecast=forecast,
        n_donors=n_donors, ppi=ppi, n_factors=n_factors, time_average=time_average,
    )
    sel = screen.selected_idx
    Dsel = D[:, sel]

    att_ci = cf_lower = cf_upper = None
    if inference == "bayes":
        fit = bayesian_simplex_sc(
            y, Dsel, T0, alpha=dirichlet_alpha, n_samples=n_samples,
            n_warmup=n_warmup, ci_level=ci_level, seed=seed,
        )
        weights = fit.weights
        counterfactual = fit.counterfactual
        att = fit.att
        att_ci = fit.att_ci
        cf_lower, cf_upper = fit.cf_lower, fit.cf_upper
    else:
        weights, counterfactual = simplex_weights(y, Dsel, T0)
        att = float(np.mean((y - counterfactual)[post]))

    gap = y - counterfactual

    # Unscreened ("All") baseline -- always a fast frequentist simplex.
    _, cf_all = simplex_weights(y, D, T0)
    att_all = float(np.mean((y - cf_all)[post]))

    # Optional proximal (GMM) debiasing using the excluded donors.
    att_debiased = None
    if debias and screen.excluded_idx.size > 0:
        deb = proximal_debias(y, Dsel, D[:, screen.excluded_idx], T0)
        att_debiased = deb.att

    att_by_period = {
        inputs.time_labels[t]: float(gap[t]) for t in range(T0, inputs.T)
    }
    donor_weights = {
        screen.donor_names[sel[j]]: float(weights[j]) for j in range(sel.size)
    }
    metadata = {
        "estimator": "SPOTSYNTH", "T": int(inputs.T), "T0": int(T0),
        "n_donors": int(inputs.n_donors), "n_selected": int(sel.size),
        "n_excluded": int(screen.excluded_idx.size),
        "selection": selection, "forecast": forecast, "inference": inference,
        "dirichlet_alpha": float(dirichlet_alpha) if inference == "bayes" else None,
        "treated_name": inputs.treated_name,
    }
    return SpotSynthResults(
        inputs=inputs, screen=screen, att=att, counterfactual=counterfactual,
        gap=gap, att_by_period=att_by_period, donor_weights=donor_weights,
        att_unscreened=att_all, inference=inference, att_ci=att_ci,
        counterfactual_lower=cf_lower, counterfactual_upper=cf_upper,
        att_debiased=att_debiased, metadata=metadata,
    )
