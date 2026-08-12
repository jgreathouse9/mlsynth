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

Path B — the Section 5 simulation (Table 2)
--------------------------------------------

The Walmart check above is Path A on the authors' empirical application. Their
Section 5 simulation is a separate target, and it is the one that exercises the
cardinality constraint across its range and the design family beyond
``standard``.

The data-generating process is the linear factor model of Assumption 1:
:math:`J = 15` units, :math:`R = 7` observed and :math:`F = 11` unobserved
covariates, :math:`T = 30` periods with :math:`T_0 = 25`, weights estimated on
the first :math:`T_E = 20` periods and 21-25 left blank. The intercept series
:math:`\delta_t` and :math:`\upsilon_t` are small-to-large rearrangements of
Uniform(0, 20) draws, :math:`Z_j` and :math:`\mu_j` are Uniform(0, 1), the
coefficient vectors Uniform(0, 10), and the errors :math:`N(0, 1)`. The same
process is the generation block of the authors' ``SCdesign_LazyRun.R``.

R's Mersenne-Twister stream cannot be reproduced by numpy's PCG64, so these are
not the authors' draws. The port is checked against the effect path instead,
which has a closed form: both intercept series are order statistics of
Uniform(0, 20) draws and the covariate terms share their distributions across
the treated and control processes, so they cancel in expectation and

.. math::

   \tau_{25+k} \;=\; \frac{20k}{6} \;-\; \frac{20(25+k)}{31},
   \qquad k = 1, \ldots, 5,

giving :math:`-13.44, -10.75, -8.06, -5.38, -2.69`. The paper's Table 2 reports
:math:`-13.58, -10.99, -8.35, -5.00, -2.50`, within 1.3 standard errors of that
closed form at its :math:`M = 1000` (the per-period spread across simulations is
about 9.4, so their standard error is roughly 0.30). The port lands in the same
place, which is what licenses using the closed form as the target instead of
their printed values.

Estimates use the paper's equation (8),
:math:`\hat\tau_t = \mathbf{w}'\mathbf{Y}_{I,t} - \mathbf{v}'\mathbf{Y}_{N,t}`,
scored against the realized :math:`\tau_t` under uniform population weights. At
:math:`M = 20` simulations against the paper's 1000:

.. list-table::
   :header-rows: 1
   :widths: 22 16 16 16 16 16

   * - design
     - MAE
     - paper
     - RMSE
     - paper
     - :math:`\|w\|_0`
   * - Constrained :math:`m = 1`
     - 3.02
     - 2.93
     - 3.66
     - 3.45
     - 1.0
   * - Constrained :math:`m = 3`
     - 1.33
     - 1.26
     - 1.56
     - 1.49
     - 3.0
   * - Unconstrained
     - 0.75
     - 0.83
     - 0.90
     - 0.97
     - 6.80 (paper 6.76)

The paper's headline for Table 2 reproduces: accuracy improves monotonically as
the cardinality constraint relaxes, and the unconstrained design selects about
seven of the fifteen units without being told to.

The weakly targeted design
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Formulation (9) matches the treated synthetic to the population predictor vector
and ties the control synthetic to it with weight :math:`\beta`, so :math:`\beta`
trades between estimating :math:`\tau_t` and the effect on the treated,
:math:`\tau^T_t`. The paper reports that a large :math:`\beta` gives high
relative accuracy for the latter.

Measured on the same panels at :math:`m = 3`, root mean square error against
:math:`\tau^T_t` is 1.60 for ``design="standard"`` and 1.56 for
``design="weakly_targeted"`` with :math:`\beta = 20`. The direction agrees with
the paper, and at :math:`M = 25` the gap widens to 1.62 against 1.50. It is
about 0.9 combined standard errors either way, so the case pins the two levels
and does not assert the ordering: at any :math:`M` this benchmark can afford,
an ordering indicator would be a coin flip. The levels still catch a regression
in either design, and this is the only coverage ``weakly_targeted`` has outside
unit tests.

Durable case
~~~~~~~~~~~~

`benchmarks/cases/marex_section5_mc.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/marex_section5_mc.py>`_

.. code-block:: bash

   python benchmarks/run_benchmarks.py --case marex_section5_mc

Thirteen pinned quantities, seeded and reproducing bit-identically. Runtime is
about 280 seconds, almost all of it the mixed-integer solves.
