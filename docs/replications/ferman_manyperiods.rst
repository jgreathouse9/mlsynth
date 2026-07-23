.. _replication-ferman-manyperiods:

VanillaSC recovers the factor structure with many periods and many controls
===========================================================================

:Estimator: :doc:`../vanillasc` -- :class:`mlsynth.VanillaSC` (``backend="outcome-only"``)
:Source: Ferman, B. (2021), "On the Properties of the Synthetic Control
   Estimator with Many Periods and Many Controls", *Journal of the American
   Statistical Association* 116(536):1764-1772,
   `doi:10.1080/01621459.2021.1965613
   <https://doi.org/10.1080/01621459.2021.1965613>`_.
:Replication type: Path B -- the paper's Monte Carlo (Table 1), the SC-estimator
   columns.
:Status: verified -- mlsynth reproduces the Table 1 SC columns within
   Monte-Carlo error, and matches the paper's own ``solve.QP`` solver
   value-for-value on a fixed draw.
:Benchmark: ``benchmarks/cases/ferman_manyperiods.py``
   (`source <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/ferman_manyperiods.py>`__).

Why this case exists
--------------------

The other VanillaSC cases pin the estimator on empirical panels (Prop 99, the
Iberian electricity market, the Swedish carbon tax). This one pins a
*theoretical* property: what the synthetic control does as the panel gets large
in both dimensions at once.

Abadie, Diamond and Hainmueller (2010) bounded the SC bias under a perfect
pre-treatment fit as the number of pre-periods grows with a fixed donor pool.
Ferman (2021) asks the harder question -- what happens when the pre-treatment fit
is *imperfect* and the number of donors :math:`J` grows together with the number
of pre-periods :math:`T_0`. His answer: if outcomes follow a linear factor model,
the SC unit still recovers the treated unit's factor structure, because weights
diluted across a growing donor pool can reconstruct that structure even when no
finite-donor combination fits the pre-period exactly. This is the regularity that
justifies plain synthetic control in wide panels, and it is the behaviour this
benchmark reproduces with mlsynth's solver.

The setup
---------

Potential outcomes follow :math:`y_t = \boldsymbol{\mu} F_t + \varepsilon_t` with
:math:`K = 2` Gaussian AR(1) factors (:math:`\rho = 0.5`) and unit-variance
idiosyncratic shocks. The treated unit and the first half of the donors load only
on factor 1; the second half of the donors load only on factor 2. To reconstruct
the treated unit's factor structure the synthetic control must place
(asymptotically) all its weight on the first-group donors. Treatment has no
effect, so the estimated one-period-ahead effect *is* the bias, and
:math:`\mathrm{se}(\widehat{\alpha})` is its Monte-Carlo standard deviation.

Two summaries track the recovery: :math:`E[\widehat{\mu}_{01}]`, the total SC
weight on the treated unit's own factor group (heading to 1), and
:math:`E[\widehat{\mu}_{02}]`, the weight on the other group (heading to 0).

Factor recovery
---------------

Running mlsynth's outcome-only solver over the paper's grid (200 replications per
cell, seed 47) reproduces the Table 1 SC columns. The weight on the treated
unit's own factor group, :math:`E[\widehat{\mu}_{01}]`, climbs toward 1 as
:math:`J` and :math:`T_0` grow together (Panel B, :math:`T_0 = 2J`):

=========  ==========  ===============
:math:`J`  mlsynth     Ferman Table 1
=========  ==========  ===============
4          0.769       0.753
10         0.814       0.831
50         0.922       0.922
100        0.944       0.944
=========  ==========  ===============

Panel A (:math:`T_0 = J + 5`, so donors and pre-periods stay roughly matched)
tells the same story: :math:`E[\widehat{\mu}_{01}]` runs 0.777 / 0.806 / 0.894 /
0.937 against the paper's 0.760 / 0.817 / 0.905 / 0.929. Across both panels the
recovery is strictly increasing in :math:`J`, and the misallocated weight
:math:`E[\widehat{\mu}_{02}]` falls to about 0.06 at :math:`J = 100` -- the
paper's Proposition 3.1 in the simulation.

Precision, and why the constraint matters
-----------------------------------------

The paper's foil is unconstrained OLS, which also drives
:math:`E[\widehat{\mu}_{01}] \to 1` but pays for it in variance. As :math:`J`
grows with :math:`T_0` (Panel A), the SC effect's standard deviation *shrinks*
(from :math:`\approx 1.27` at :math:`J = 4` to :math:`\approx 1.10` at
:math:`J = 100`) while the OLS standard deviation *grows* about threefold
(:math:`1.62 \to 4.98`) -- so SC's is roughly a fifth of OLS's at
:math:`J = 100`. The simplex constraint (non-negative weights summing to one) is
what buys synthetic control that precision; it is the practical reason to prefer
SC over an unconstrained regression on the donors in a wide panel.

Same estimator as the paper
---------------------------

Ferman's simulation solves the SC weights with R's ``solve.QP`` (the
Goldfarb-Idnani quadratic program in ``aux.R``'s ``synth_control_est``). On a
fixed draw (:math:`J = 10`, :math:`T_0 = 20`) mlsynth's outcome-only VanillaSC --
the solver :class:`mlsynth.VanillaSC` dispatches to -- returns those weights to
:math:`7 \times 10^{-11}`. mlsynth is not approximating the paper's estimator; it
*is* the paper's estimator.

Reproduce
---------

.. code-block:: bash

   python benchmarks/run_benchmarks.py --case ferman_manyperiods

The case is self-contained: the factor-model DGP is ported from Ferman's
``main.R`` and runs mlsynth's own solver -- no external data. The pinned targets
are the published Table 1 numbers (reference bundle
``benchmarks/reference/ferman_manyperiods/``), independently cross-checked by
aggregating the authors' supplementary ``results.csv`` (40,000 rows), which
reproduces every SC cell to :math:`\le 0.001`. A clean-room R reimplementation
regenerates the table from scratch:

.. code-block:: bash

   # base R + quadprog (CRAN firewalled in CI; git clone the package):
   #   git clone --depth 1 https://github.com/cran/quadprog && R CMD INSTALL quadprog
   Rscript benchmarks/R/ferman_manyperiods.R 5000
