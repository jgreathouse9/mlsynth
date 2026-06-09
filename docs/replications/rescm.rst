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
