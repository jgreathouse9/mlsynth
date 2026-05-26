"""Ad-holdout simulation for MicroSynth: ITT vs. as-treated vs. CACE.

A self-contained data generator for the canonical online-advertising
holdout problem with **contamination** (some holdout users see ads
anyway) and an **unobserved confounder** (latent purchase intent that
drives both who gets exposed and how much they buy).

The point of the generator is to make three estimands separable and to
expose the trap that covariate balancing alone cannot escape:

* **As-treated** -- regroup users by what they *received* (exposed vs.
  not) and balance on observed covariates. Biased, because exposure is
  selected on the *unobserved* intent that also lifts sales; balancing
  age / income only removes the part of that selection the covariates
  happen to explain.
* **Intent-to-treat (ITT)** -- keep users in their randomized arm and
  balance on observed covariates for precision. Unbiased for the
  campaign effect, because randomization balances the unobserved intent
  across arms regardless of what balancing does.
* **CACE / per-exposure** -- the ITT effect divided by the compliance
  gap (a covariate-balanced Wald ratio), which recovers the true
  per-exposure effect while preserving randomization.

Treatment is encoded two ways so the same panel can be fed to MicroSynth
under either labelling:

* ``D_itt`` -- 1 in the post-period for users *assigned* to the ad arm.
* ``D_att`` -- 1 in the post-period for users *actually exposed* to ads
  (assigned-arm compliers plus contaminated holdout users).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit


def simulate_ad_holdout(
    *,
    n_per_arm: int = 5000,
    delta: float = 1.0,
    intent_on_exposure: float = 1.2,
    intent_on_outcome: float = 3.0,
    treat_arm_logit: float = 3.0,
    holdout_arm_logit: float = -2.2,
    seed: int = 0,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Generate a contaminated ad-holdout panel with a known effect.

    Parameters
    ----------
    n_per_arm : int
        Users assigned to each of the ad arm and the holdout arm.
    delta : float
        True homogeneous **per-exposure** effect on post-period sales
        (the estimand the CACE/Wald ratio targets).
    intent_on_exposure : float
        Strength with which the latent intent ``U`` drives exposure.
        Larger values mean the contaminated holdout users (and the
        treatment-arm compliers) are more strongly self-selected on
        intent -- i.e. more as-treated bias.
    intent_on_outcome : float
        Coefficient on the latent intent ``U`` in the sales equation.
        This is the unobserved confounding that covariate balancing on
        age / income cannot fully remove.
    treat_arm_logit, holdout_arm_logit : float
        Baseline exposure log-odds in the ad arm and holdout arm. The
        gap between the realized exposure rates is the compliance gap
        the CACE divides by.
    seed : int
        RNG seed.

    Returns
    -------
    df : pandas.DataFrame
        Long panel with columns ``user_id``, ``time`` (0 = pre,
        1 = post), ``sales``, ``age``, ``income``, ``assigned``,
        ``exposed``, ``D_itt``, ``D_att``.
    truth : dict
        Ground-truth quantities: ``delta_per_exposure``,
        ``p_expose_treat``, ``p_expose_holdout``, ``compliance_gap``,
        and ``itt_effect`` (``delta * compliance_gap``).
    """
    rng = np.random.default_rng(seed)
    n = 2 * n_per_arm

    # Latent purchase intent -- UNOBSERVED by the analyst.
    U = rng.standard_normal(n)

    # Observed covariates: correlated with intent but noisy, so balancing
    # on them captures only part of the intent-driven selection.
    age = 40.0 + 7.0 * U + 5.0 * rng.standard_normal(n)
    income = 55.0 + 9.0 * U + 8.0 * rng.standard_normal(n)

    # Randomized assignment (independent of intent and covariates).
    assigned = np.zeros(n, dtype=int)
    assigned[rng.permutation(n)[:n_per_arm]] = 1

    # Exposure: assigned arm complies at a high rate; the holdout leaks,
    # and the leak is selected on the unobserved intent.
    base_logit = np.where(assigned == 1, treat_arm_logit, holdout_arm_logit)
    p_expose = expit(base_logit + intent_on_exposure * U)
    exposed = rng.binomial(1, p_expose)

    # Sales. Baseline depends on observed covariates AND latent intent;
    # only actual exposure carries the causal effect ``delta``.
    baseline = 10.0 + 0.10 * age + 0.05 * income + intent_on_outcome * U
    sales_pre = baseline + rng.standard_normal(n)
    sales_post = baseline + delta * exposed + rng.standard_normal(n)

    rows = []
    for i in range(n):
        common = dict(
            user_id=f"u{i:06d}",
            age=float(age[i]),
            income=float(income[i]),
            assigned=int(assigned[i]),
            exposed=int(exposed[i]),
        )
        rows.append({
            **common, "time": 0, "sales": float(sales_pre[i]),
            "D_itt": 0, "D_att": 0,
        })
        rows.append({
            **common, "time": 1, "sales": float(sales_post[i]),
            "D_itt": int(assigned[i]), "D_att": int(exposed[i]),
        })
    df = pd.DataFrame(rows)

    p1 = float(exposed[assigned == 1].mean())
    p0 = float(exposed[assigned == 0].mean())
    truth = {
        "delta_per_exposure": float(delta),
        "p_expose_treat": p1,
        "p_expose_holdout": p0,
        "compliance_gap": p1 - p0,
        "itt_effect": float(delta) * (p1 - p0),
    }
    return df, truth
