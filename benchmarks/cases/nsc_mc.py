"""NSC Path-B: the nonlinear-outcome Monte Carlo (Tian 2023, Table 1).

Path B (Monte Carlo, scenario: paper provides the full DGP). Reproduces the
qualitative findings of Tian (2023) Table 1 on the nonlinear-outcome design
(``r = 2``), where the standard SC estimator is biased:

  * the **95% CI covers near the nominal rate** (the paper reports ~0.94-0.95
    for NSC across settings), and
  * **estimation error shrinks as the donor pool grows** (J: 25 -> 50), the
    paper's "more donors are unambiguously better in the nonlinear case".

The DGP is the faithful port in
:func:`mlsynth.utils.nsc_helpers.simulate.simulate_nsc_panel` (two observed +
four unobserved predictors, latent outcome rescaled to [0,1] and raised to the
power ``r``, treated unit 0 with the ramped effect 0.02..0.20). NSC's own
cross-validation selects the penalty on each draw, as in the paper.

Note on metrics: Table 1's *signed* bias column needs the paper's 5000
simulations to estimate at its ~0.01 magnitude, which is out of reach for a CI
benchmark; the *coverage* and the *error-shrinks-with-J* geometry are robust at
a small draw count, so those are asserted here (the mean absolute error is
reported for transparency). Determinism: per-draw seeds ``500 + m``.

Provenance: Tian (2023), arXiv:2306.01967v1, Section 4, Table 1 (NSC column,
nonlinear panel).
"""
from __future__ import annotations

import warnings

import numpy as np

_T0 = 30
_R = 2                              # nonlinear outcome
_SETTINGS = {"J25": (25, 40), "J50": (50, 18)}   # (J, n_draws)


def _fit_draw(J, seed):
    from mlsynth import NSC
    from mlsynth.utils.nsc_helpers.simulate import simulate_nsc_panel

    df, tau = simulate_nsc_panel(J=J, T0=_T0, r=_R, seed=seed)
    res = NSC({
        "df": df, "outcome": "y", "treat": "D", "unitid": "unit",
        "time": "time", "run_inference": True, "display_graphs": False,
    }).fit()
    inf = res.inference_detail
    gap = np.asarray(inf.gap)[_T0:]
    lo = np.asarray(inf.gap_lower)[_T0:]
    hi = np.asarray(inf.gap_upper)[_T0:]
    abs_err = np.abs(gap - tau)                      # per post-period
    covered = (lo <= tau) & (tau <= hi)
    return abs_err, covered


def run() -> dict:
    out = {}
    mae_by_J = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tag, (J, n_draws) in _SETTINGS.items():
            errs, covs = [], []
            for m in range(n_draws):
                ae, cv = _fit_draw(J, 500 + m)
                errs.append(ae)
                covs.append(cv)
            mae = float(np.mean(errs)) * 100.0     # mean abs error, x100
            cov = float(np.mean(covs))
            mae_by_J[J] = mae
            out[f"coverage_{tag}"] = cov
            out[f"mae_{tag}"] = mae
    # Headline geometry: error shrinks as the donor pool grows.
    out["mae_ratio_J50_over_J25"] = mae_by_J[50] / mae_by_J[25]
    return out


# Deterministic (per-draw seeds 500+m). Coverage lands near the paper's nominal
# ~0.94 in both settings, and the mean absolute error falls as J doubles
# (25 -> 50), reproducing Table 1's two robust findings for the nonlinear panel.
# Tolerances are wide enough to absorb the small draw count (coverage SE ~
# sqrt(p(1-p)/(n*10)) and the MAE Monte-Carlo noise) yet still fail if coverage
# collapses or the error stops shrinking with J.
EXPECTED = {
    "coverage_J25": (0.928, 0.08),             # near nominal (paper ~0.935)
    "coverage_J50": (0.967, 0.08),             # near nominal (paper ~0.950)
    "mae_J25": (0.89, 0.35),                   # x100; paper NSC bias 0.87
    "mae_J50": (0.67, 0.30),                   # x100; paper NSC bias 0.68
    "mae_ratio_J50_over_J25": (0.75, 0.20),    # < 1: error shrinks with J
}
