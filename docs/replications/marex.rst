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

Panels are read, not regenerated. ``benchmarks/reference/marex_scdesign_sim/``
captures a live run of the authors' own ``Generate_Model_Primitives``
("2. Many Simulations (...)/Main_LazyRun.R"), under the seeding their driver
uses: ``R_job.qsub`` submits an SGE array job (``#$ -t 1-1000``) and the script
reads ``repetition.RANDOM.SEED = Sys.getenv("SGE_TASK_ID")``, so simulation
:math:`i` is ``set.seed(i)`` --- a thousand independent streams, not one
advancing stream. Reproducing R's generator in numpy is neither possible nor
needed once the draws are captured.

Under that convention their DGP reproduces Table 2's effect path exactly:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 15 15 15

   * -
     - :math:`t=26`
     - :math:`t=27`
     - :math:`t=28`
     - :math:`t=29`
     - :math:`t=30`
   * - paper, Table 2
     - -13.58
     - -10.99
     - -8.35
     - -5.00
     - -2.50
   * - their code, their seeds
     - -13.5763
     - -10.9889
     - -8.3521
     - -4.9981
     - -2.4999

The largest gap is 0.004, which is the table's own display rounding. No design is
solved to produce this, so it isolates the data-generating process.

Estimates use the paper's equation (8),
:math:`\hat\tau_t = \mathbf{w}'\mathbf{Y}_{I,t} - \mathbf{v}'\mathbf{Y}_{N,t}`,
scored against the realized :math:`\tau_t` under uniform population weights. On
12 captured panels against the paper's 1000 simulations:

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
     - 3.03
     - 2.93
     - 3.49
     - 3.45
     - 1.0
   * - Constrained :math:`m = 3`
     - 1.23
     - 1.26
     - 1.41
     - 1.49
     - 3.0
   * - Unconstrained
     - 0.57
     - 0.83
     - 0.69
     - 0.97
     - 6.83 (paper 6.76)

The paper's headline for Table 2 reproduces: accuracy improves monotonically as
the cardinality constraint relaxes, and the unconstrained design selects about
seven of the fifteen units without being told to. The unconstrained cell runs
better than the paper's at this panel count, which is the Monte-Carlo error of 12
draws against 1000 and is what the tolerances absorb.

The weakly targeted design
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Formulation (9) matches the treated synthetic to the population predictor vector
and ties the control synthetic to it with weight :math:`\beta`, so :math:`\beta`
trades between estimating :math:`\tau_t` and the effect on the treated,
:math:`\tau^T_t`. The paper reports that a large :math:`\beta` gives high
relative accuracy for the latter.

The measurement does not settle it. Root mean square error against
:math:`\tau^T_t` at :math:`m = 3` is 1.46 for ``design="standard"`` against 1.66
for ``design="weakly_targeted"`` with :math:`\beta = 20` on these 12 panels ---
standard ahead. On 20 panels drawn from an independent stream the order was
reversed, and at 25 the weakly targeted design led by about 0.9 combined standard
errors. The ordering sits inside the noise at any panel count a benchmark can
afford, so the case pins the two levels and does not assert it. The levels still
catch a regression in either design, and this is the only coverage
``weakly_targeted`` has outside unit tests.

Durable case
~~~~~~~~~~~~

`benchmarks/cases/marex_section5_mc.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/marex_section5_mc.py>`_

.. code-block:: bash

   python benchmarks/run_benchmarks.py --case marex_section5_mc

Fourteen pinned quantities. Runtime is about 180 seconds, almost all of it the
mixed-integer solves.

Cross-validation against SCDesign on the same panels
------------------------------------------------------

Table 2 compares averages. This compares designs: the authors' own R routine and
MAREX solve the same program on the same panel, checked unit by unit and weight
by weight.

The same captured bundle carries, for each panel, the design SCDesign's
``Synthetic_Experiment_Cardinality_Constraint`` selected. That routine is exact by
construction --- it enumerates every partition of size :math:`1 \le p \le K`,
solves the treated and control weights for each by ``quadprog::solve.QP``, and
keeps the minimum --- so its answer is the optimum, and MAREX's mixed-integer
solve has to reach it. It needs no commercial solver; SCDesign's headline design
is a Gurobi MIQP and is not used. It is the design MAREX solves with
``m_min = 1``, ``m_max = K``.

Matching is on the 20 fitting periods and the 7 observed covariates with
per-predictor standardisation, which is the routine's row rescaling.

Across 12 panels at :math:`K = 2` the two select the same treated units every
time and agree on the treated weights to 7.0e-05 --- the two solvers' numerical
tolerance, R working on a ``nearPD``-repaired Hessian and rounding to six
decimals.

`benchmarks/cases/marex_scdesign_sim.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/marex_scdesign_sim.py>`_

.. code-block:: bash

   # refresh the captured SCDesign run (needs R + quadprog + Matrix)
   python benchmarks/reference/generate.py marex_scdesign_sim
   # run the cross-validation (no R needed; reads the committed bundle)
   python benchmarks/run_benchmarks.py --case marex_scdesign_sim

Path B — Table 3, against the randomized alternatives
--------------------------------------------------------

Table 3 sets the design against five randomized alternatives at every
cardinality: randomized assignment with difference in means (RND), stratified
randomization (STR), regression adjustment on the covariates (REG), and
nearest-neighbour matching at one and five neighbours.

The SC column is computed here, not copied. Table 3's note defines SC as the
Constrained formulation, which in ``Main_LazyRun.R`` is
``Synthetic_Experiment_Cardinality_Constraint`` --- the quadprog routine, with no
Gurobi call anywhere in that section --- and the cross-validation above shows
MAREX reaches the same design as that routine, unit for unit. The comparator
columns are quoted from the table: they are properties of the authors'
randomized designs, not of ``mlsynth``, so re-implementing them would test a
transcription. What the library has to clear is the ordering they establish.

On the authors' panels, MAREX against the strongest published alternative at
each cardinality:

.. list-table::
   :header-rows: 1
   :widths: 12 20 18 30

   * - :math:`m`
     - MAREX SC
     - paper SC
     - best alternative
   * - 1
     - 3.46
     - 3.45
     - 4.40 (5-NN)
   * - 2
     - 1.68
     - 2.00
     - 3.20 (5-NN)
   * - 3
     - 1.30
     - 1.49
     - 2.66 (5-NN)
   * - 4
     - 1.03
     - 1.25
     - 2.40 (5-NN)
   * - 5
     - 0.83
     - 1.09
     - 2.07 (STR)
   * - 6
     - 0.84
     - 1.02
     - 1.95 (STR)
   * - 7
     - 0.82
     - 0.97
     - 1.85 (STR)

The paper's headline holds at every cardinality, and by a wide margin: the
design's error is between a half and a quarter of the best randomized
alternative. Clearing the strongest of the five clears the whole row.

Monotonicity is pinned only where twelve panels can resolve it. The published
column falls steeply at first --- 3.45, 2.00, 1.49 --- then by 0.07 and 0.05
between :math:`m = 5, 6, 7`. Those last gaps sit inside the Monte-Carlo error of
twelve panels against a thousand, and this run duly puts :math:`m = 6` (0.843) a
hair above :math:`m = 5` (0.832). The case pins the steep range and separately
pins that :math:`m = 7` lands below half of :math:`m = 1`; asserting the full
ordering would be asserting noise.

`benchmarks/cases/marex_table3.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/marex_table3.py>`_

.. code-block:: bash

   python benchmarks/run_benchmarks.py --case marex_table3
