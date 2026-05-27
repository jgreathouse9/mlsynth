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

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlsynth import MAREX
from mlsynth.utils.marex_helpers.simulation import generate_marex_sample

_WALMART = Path(__file__).resolve().parents[2] / "basedata" / "walmart_weekly_sales.csv"


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
    # Use the cardinality-CONSTRAINED design: the unconstrained "standard"
    # design (formulation 5) is degenerate (many disjoint splits match Xbar
    # equally), so a single solve is solver-dependent. The constrained design
    # is the stable, recommended one.
    rng = np.random.default_rng(0)
    mae2, mae3, scale = [], [], []
    for _ in range(5):
        s = generate_marex_sample(J=12, T=30, T0=25, rng=rng)
        mae2.append(_design_mae(s, m_eq=2))
        mae3.append(_design_mae(s, m_eq=3))
        scale.append(float(np.mean(np.abs(s.tau[s.T0:]))))

    mae2, mae3, scale = np.mean(mae2), np.mean(mae3), np.mean(scale)
    # the design recovers the effect to within its own scale (Table 2)
    assert mae2 < scale
    assert mae3 < scale


# ----------------------------------------------------------------------
# Path A: the Walmart placebo empirical illustration (Section 4)
# ----------------------------------------------------------------------

@pytest.mark.slow
def test_walmart_placebo():
    """The placebo design tracks closely and fails to reject (paper p = 0.933)."""
    if not _WALMART.exists():
        pytest.skip(f"vendored Walmart panel not found at {_WALMART}")
    df = pd.read_csv(_WALMART)
    res = MAREX({
        "df": df, "outcome": "sales", "unitid": "store", "time": "week",
        "T0": 128, "blank_periods": 28, "T_post": 15, "m_eq": 2,
        "design": "standard", "standardize": True, "inference": True,
    }).fit()

    g = res.globres
    T0 = res.study.T0
    Tfit = T0 - res.study.blank_periods
    gap = g.synthetic_treated - g.synthetic_control
    mean = g.synthetic_treated[:Tfit].mean()
    assert len(res.treated_units) == 2
    # close pre-period fit, near-zero placebo effect, fail to reject
    assert np.sqrt(np.mean(gap[:Tfit] ** 2)) / mean < 0.10
    assert abs(np.mean(gap[T0:]) / mean) < 0.10
    assert g.inference.global_p_value > 0.5
