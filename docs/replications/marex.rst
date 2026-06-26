MAREX — Abadie & Zhao (2026) Walmart design (live R cross-validation)
=====================================================================

.. currentmodule:: mlsynth

Live, commit-stamped cross-validation of **MAREX** -- mlsynth's port of Abadie &
Zhao's synthetic-control experimental-design estimator -- against the authors' own
R code on the paper's Section 4 Walmart application. The reference is
`jinglongzhao2/SCDesign <https://github.com/jinglongzhao2/SCDesign>`_, run live
(no commercial solver) and captured under ``benchmarks/reference/marex_walmart/``.

This complements :doc:`lexscm`, which validates the *same* Abadie-Zhao design on
the same Walmart panel through the lexicographic LEXSCM solver; here the check
runs MAREX's own mixed-integer optimizer.

Validated without Gurobi
------------------------

SCDesign's published design is a Gurobi non-convex MIQP, which is licence-gated.
Its *constrained* (cardinality-:math:`K`) routine, however, is fully open:
``Synthetic_Experiment_Cardinality_Constraint`` enumerates every partition of size
:math:`\le K`, solves each treated/control synthetic control through
``quadprog::solve.QP``, and keeps the min-loss partition. That is exactly the
design MAREX's ``m_eq`` solves, and it runs with no commercial solver. The
reference script (``benchmarks/reference/marex_walmart/reference.R``) reproduces it
verbatim, together with the authors' permutation test and conformal interval, on
the full 45-store panel.

Data
----

``basedata/walmart_weekly_sales_covariates.csv`` -- the full SCDesign Walmart panel
(45 stores x 143 weeks; sales value-identical to ``Walmart.csv``) with the four
store-level covariates (temperature, fuel price, CPI, unemployment). The design
matches on the pre-period sales and these covariates -- the R code's "few
covariates" configuration. Windows follow the paper: fit weeks 1-100, blank
101-128, experimental 129-143 (``T0 = 128``, ``blank_periods = 28``,
``T_post = 15``), ``m_eq = 2``, uniform population weights, per-predictor
standardisation.

Result — placebo design (Sec. 4)
--------------------------------

A placebo intervention (week 129, no real effect) must yield a design whose
synthetic treated and control units track closely pre-period and whose estimated
effect is indistinguishable from zero. MAREX and SCDesign agree cell-by-cell:

.. list-table::
   :header-rows: 1
   :widths: 34 18 22

   * - Quantity
     - MAREX
     - SCDesign (quadprog)
   * - Treated stores selected
     - 15, 31
     - 15, 31
   * - Treated weights
     - 0.461 / 0.539
     - 0.461 / 0.539
   * - Pre-fit RMSE (% mean sales)
     - 2.90%
     - 2.84%
   * - Placebo effect (% mean)
     - 2.73%
     - 2.74%
   * - Placebo permutation p
     - 0.125
     - 0.109
   * - CI covers zero
     - yes
     - yes

The two implementations select the same two treated stores, with treated weights
agreeing to :math:`2\times10^{-4}` and the placebo effect to :math:`10^{-4}` of
mean sales -- the paper's "no spurious effect" result, reproduced across languages
and solvers without Gurobi. (The permutation p differs by ~0.02 because the two
Monte-Carlo permutation samples differ; both fail to reject. Abadie & Zhao's
headline :math:`p = 0.933` is the no-covariate special case -- matching on the
covariates tightens the design and lowers the placebo p, as in the authors' own
covariate runs.)

.. note::

   Exact MIQP, not the relaxation. MAREX uses its exact MIQP (free SCIP backend).
   The relaxed continuous-``z`` mode shares A&Z's objective (``build_objective`` is
   common to both) but drops the integrality that *defines* the selection; for a
   small treated count the relaxed optimum is degenerate, so its top-``m`` rounding
   is lossy and non-deterministic -- unfaithful to the paper's exact design.

Reproduce
---------

.. code-block:: bash

   # regenerate the captured SCDesign reference (needs R + quadprog + Matrix)
   python benchmarks/reference/generate.py marex_walmart
   # run the cross-validation
   python benchmarks/run_benchmarks.py marex_walmart
