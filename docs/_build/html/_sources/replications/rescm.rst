.. _replication-rescm:

RESCM — SCM-relaxation (Liao, Shi & Zheng 2026)
===============================================

:Estimator: :doc:`../rescm` — :class:`mlsynth.RESCM` (relaxation branch:
   ``RELAX_L2`` / ``RELAX_ENTROPY`` / ``RELAX_EL``)
:Source: Liao, Chengwang, Zhentao Shi, and Yapeng Zheng (2026),
   *"A Relaxation Approach to Synthetic Control,"* arXiv:2508.01793 [RelaxSC]_.
:Replication type: **Path A** — the paper's Brexit / UK real-GDP empirical —
   **and cross-validation** — mlsynth's L2 relaxation matched cell-by-cell
   against the authors' ``scmrelax`` package.
:Status: **Verified** — empirical reproduced and engine cross-validated.

Validation strategy
-------------------

SCM-relaxation keeps the simplex but *relaxes* the exact balance first-order
condition to an :math:`\ell_\infty` tolerance :math:`\eta`, then minimizes an
information-theoretic divergence (squared :math:`\ell_2`, entropy, or empirical
likelihood) over the relaxed feasible set. The relaxed-balance program is
**scale-sensitive**, so mlsynth exposes a ``standardize`` flag
(:class:`mlsynth.config_models.RESCMConfig`): ``standardize=False`` solves on the
raw series, matching the authors' reference; ``True`` (the historical default)
standardizes the donor columns first.

The authors ship two public repositories, both linked from the estimator page:
the ``scmrelax`` package (https://github.com/metricshilab/scmrelax) and the
Brexit application (https://github.com/YapengZheng/Relaxed_SC). We use both.

Path A — Brexit / UK real GDP
-----------------------------

On the authors' ``balanced_GDP_data.csv`` (United Kingdom + 57 donor economies,
quarterly real GDP 2002Q4-2024Q2; treatment 2016Q3), the outcome is
year-over-year GDP growth and the cumulative level effect is rebuilt by
compounding the predicted no-Brexit growth path. With ``standardize=False`` the
L2 relaxation gives a **cumulative UK GDP loss of about 4.0%** (paper: 3.85%),
and reproduces the paper's weight contrast: the relaxation is **dense** and
weights the major EU economies (Germany, France, Italy) — exactly the donors the
**sparse classic SC drops** while concentrating ~0.30-0.36 on the United States.

A subtlety the paper itself stresses: with :math:`J = 57` donors and only
:math:`T_0 = 51` pre-periods, the classic SC weight vector is **non-unique**
(different solvers select different optimal vertices, all with near-perfect
pre-fit), so its cumulative loss is solver-dependent. The relaxation is unique
and stable — which is the point of the method — so the benchmark asserts the
well-posed relaxation result and the robust sparse-vs-dense weight contrast.
Durable case: ``rescm_brexit``.

Cross-validation — mlsynth vs ``scmrelax``
------------------------------------------

Because the authors' package is public, the strongest available evidence is a
cell-by-cell match. On a shared panel at a matched relaxation level
:math:`\tau`, mlsynth's L2 relaxation (through the public
``RESCM(methods=["RELAX_L2"], tau=tau, standardize=False)``) agrees with
``scmrelax.L2RelaxationCV`` to **solver precision** — donor-weight
:math:`L_1` distance :math:`\approx 0.0014`, maximum absolute difference
:math:`\approx 3.6\times10^{-4}` — as expected for two independent
implementations of the same unique QP. The case routes ``scmrelax``'s hardcoded
MOSEK dependency to an open solver (the L2 program is a QP, so the optimum is
solver-invariant) and skips gracefully when ``scmrelax`` is unavailable.
Durable case: ``rescm_relax_ref``.

Path B — the latent-group Monte Carlo
-------------------------------------

The paper's Section-5 simulation is where the relaxation's advantage shows: under
a latent-group factor DGP with **many more donors than pre-periods**
(:math:`J \gg T_0`), classic SC overfits its non-unique exact-balance weights and
predicts poorly out of sample, while the L2 relaxation recovers the dense,
group-diversified oracle weighting. The paper's Table 1 reports out-of-sample
prediction-error ratios of the relaxation vs SC of :math:`\approx 0.15`–:math:`0.53`.

mlsynth reproduces this: at :math:`J = 120`, :math:`T_0 = 40`, with
cross-validated :math:`\tau`, the L2 relaxation's post-period error against the
**oracle counterfactual** is :math:`\approx 0.43` of classic SC's (median over
reps), and it beats SC in :math:`\approx 73\%` of reps.

Two calibration points are essential and worth remembering:

* **Measure against the oracle counterfactual, not the noisy realization.** The
  realized :math:`y_0` carries the treated unit's own idiosyncratic noise, an
  irreducible error floor added to *every* method equally; it compresses all
  ratios toward 1 (median :math:`\approx 0.87` against :math:`y_0` vs
  :math:`\approx 0.43` against the oracle) and hides the effect.
* **Use the median, not the mean.** With a cross-validated :math:`\tau` on a
  coarse grid, an occasional rep selects a loose :math:`\tau` (weights collapse
  toward uniform) and inflates its ratio; the median rep shows the paper's
  effect while those outliers drag the mean to :math:`\approx 1`.

The case pins L2 (the paper's recommended method, fast via the OSQP path); a full
multi-method MC over the exp-cone entropy/EL relaxations is a follow-up, gated on
an efficient large-:math:`J` solve for those (their DPP form carries a
:math:`J\times J` Gram parameter that is inefficient at the large :math:`J` this
regime needs). Durable case: ``rescm_relax_mc``.
