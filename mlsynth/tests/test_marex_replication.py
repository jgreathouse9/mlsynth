"""Self-contained Path-B replication of Abadie & Zhao (2026), Section 5.

Reproduces the simulation study's qualitative conclusions using only mlsynth
code (the linear-factor DGP is reimplemented in
:mod:`mlsynth.utils.marex_helpers.simulation`):

* the synthetic-control design recovers the true average treatment effect with
  error far below the effect's own scale (the core finding of Table 2), for
  both the Unconstrained and single-treated Constrained designs.

The paper's finer ordering (more treated units => smaller error) holds only on
average over its 1000 simulations and with the weight-swapping tie-break; it is
not asserted in this fast smoke test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlsynth import MAREX
from mlsynth.utils.marex_helpers.simulation import generate_marex_sample


def _design_mae(sample, m_eq=None, m_min=None, m_max=None):
    """Run a MAREX design on the pre-period and score |tau_hat - tau| post."""
    J, T = sample.Y_N.shape
    T0 = sample.T0
    df = pd.DataFrame([
        {"unit": f"u{j}", "time": t, "y": float(sample.Y_N[j, t])}
        for j in range(J) for t in range(T)
    ])
    cfg = {"df": df, "outcome": "y", "unitid": "unit", "time": "time", "T0": T0}
    if m_eq is not None:
        cfg["m_eq"] = m_eq
    else:
        cfg["m_min"], cfg["m_max"] = m_min, m_max
    res = MAREX(cfg).fit()

    w = res.globres.treated_weights_agg
    v = res.globres.control_weights_agg
    treated = np.where(w > 1e-8)[0]
    # experimental outcomes: treated units realise Y^I post-treatment
    Y_obs = sample.Y_N.copy()
    Y_obs[treated, T0:] = sample.Y_I[treated, T0:]
    tau_hat = w @ Y_obs[:, T0:] - v @ Y_obs[:, T0:]
    tau_true = sample.tau[T0:]
    return float(np.mean(np.abs(tau_hat - tau_true)))


def test_marex_recovers_ate():
    rng = np.random.default_rng(0)
    mae_unc, mae_m1, scale = [], [], []
    for _ in range(5):
        s = generate_marex_sample(J=12, T=30, T0=25, rng=rng)
        mae_unc.append(_design_mae(s, m_min=1, m_max=11))
        mae_m1.append(_design_mae(s, m_eq=1))
        scale.append(float(np.mean(np.abs(s.tau[s.T0:]))))

    mae_unc, mae_m1, scale = np.mean(mae_unc), np.mean(mae_m1), np.mean(scale)
    # both designs recover the effect to well within its own scale (Table 2)
    assert mae_unc < 0.6 * scale
    assert mae_m1 < 0.6 * scale
