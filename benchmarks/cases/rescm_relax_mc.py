"""Path B: the latent-group Monte Carlo of Liao, Shi & Zheng (2026, Section 5).

Under the paper's latent-group factor DGP with many more donors than pre-periods
(``J >> T0``), classical SCM overfits the (non-unique) exact-balance weights and
predicts the treated unit's counterfactual poorly out of sample, while the
L2-SCM-relaxation exploits the group structure (dense, group-diversified weights)
and does markedly better. The paper's Table 1 reports post-treatment
prediction-error ratios of the relaxation vs SCM well below 1 (~0.15-0.53); it
calls L2-relaxation "the best performer" and recommends it in practice.

We reproduce that **ordering and magnitude** for the L2 relaxation as a
directional guard. Three calibration points matter (full derivation in
``agents/future_integrations.md``):

* **Target = the oracle counterfactual, not the noisy realization.** Error is
  measured against ``oracle_cf`` (the group-equal oracle synthetic control -- the
  estimand the relaxation targets), *not* the realized ``y0``. ``y0`` carries the
  treated unit's idiosyncratic noise, an irreducible floor added to every
  method's error equally that compresses all ratios toward 1 (median ~0.87 vs
  ~0.43 against the oracle).
* **Median, not mean.** ``tau`` is cross-validated on a coarse grid, so an
  occasional rep selects a loose ``tau`` (weights collapse toward uniform) and
  blows up its ratio; the median rep shows the paper's effect while a few
  outliers drag the mean to ~1.
* **L2 only, for tractability.** The entropy/EL relaxations are exp-cone; their
  DPP solve carries a J-by-J Gram parameter, which is inefficient for DPP at the
  large ``J`` this regime needs (cvxpy warns), so a CV-``tau`` multi-rep MC over
  them is impractically slow. L2 is the paper's recommended method and is fast
  via the OSQP path; entropy/EL are covered by the engine cross-validation
  (``rescm_relax_ref``) and the solver speedup. A full multi-method MC is a
  follow-up gated on an efficient large-``J`` entropy/EL solve.

Durable benchmark (run via ``run_benchmarks.py``, not the pytest CI gate). It
pins the directional result, not the paper's 4-decimal ``B = 1000`` cells.
"""
import numpy as np

SEED = 7
B = 30                      # replications (paper uses 1000)
J, T0, T1 = 120, 40, 20     # J >> T0: the regime where SCM overfits
N_TAUS, N_SPLITS = 15, 2    # CV grid for tau selection


def run() -> dict:
    from mlsynth import RESCM
    from mlsynth.utils.laxscm_helpers.simulation import (
        simulate_relaxation_groups,
        to_panel,
    )

    rng = np.random.default_rng(SEED)
    ratios = []
    for _ in range(B):
        Yc, y0, oracle_cf, t0 = simulate_relaxation_groups(rng, J=J, T0=T0, T1=T1)
        df = to_panel(Yc, y0, t0)
        res = RESCM({
            "df": df, "outcome": "y", "treat": "treat", "unitid": "unit",
            "time": "time", "methods": ["SC", "RELAX_L2"], "tau": None,
            "n_taus": N_TAUS, "n_splits": N_SPLITS,
            "standardize": False, "display_graphs": False,
        }).fit()
        post = slice(t0, None)
        # Out-of-sample error against the oracle counterfactual (the estimand).
        e_sc = float(np.sum((np.asarray(res.fits["SC"].counterfactual)[post] - oracle_cf[post]) ** 2))
        e_l2 = float(np.sum((np.asarray(res.fits["RELAX_L2"].counterfactual)[post] - oracle_cf[post]) ** 2))
        if e_sc > 0:
            ratios.append(e_l2 / e_sc)

    a = np.array(ratios, dtype=float)
    return {
        "relax_l2_median_ratio": float(np.median(a)),
        "relax_l2_frac_beats_sc": float(np.mean(a < 1.0)),
    }


# L2-relaxation should beat SCM out-of-sample under latent groups: a median error
# ratio well below 1 (paper ~0.15-0.53; band 0.05-0.85 fails if it regresses
# toward the no-effect ~1.0) and a majority of reps below 1.
EXPECTED = {
    "relax_l2_median_ratio": (0.45, 0.40),
    "relax_l2_frac_beats_sc": (1.0, 0.49),   # > 0.5: more often than not
}
