.. _replication-drosc:

DROSC — Basque robustness sweep vs the authors' R
=================================================

:Estimator: :doc:`../drosc` -- :class:`mlsynth.DROSC`
:Source: Koo, T. & Guo, Z. (2026). "Distributionally Robust Synthetic Control:
   Ensuring Robustness Against Highly Correlated Controls and Weight Shifts."
   `arXiv:2511.02632 <https://arxiv.org/abs/2511.02632>`_. Reference code:
   `taehyeonkoo/DRoSC <https://github.com/taehyeonkoo/DRoSC>`_ (``helpers.R``).
:Replication type: cross-validation against the authors' own R
   (``limSolve::lsei``), the deterministic worst-case point estimand.
:Status: verified -- the estimand and the donor weights match value-for-value.
:Benchmark: ``benchmarks/cases/drosc_basque.py``
   (`source <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/drosc_basque.py>`__).

Why this case exists
--------------------

DROSC's worst-case estimand is defined by an inequality-constrained
optimisation, and the authors solve it with R's ``limSolve::lsei``. This case
confirms that mlsynth's cvxpy port targets the same optimisation and returns the
same effect across the robustness-radius sweep, on the paper's own empirical
application -- the Abadie-Gardeazabal Basque Country / ETA-terrorism study
(:math:`T_0 = 15` pre-periods, :math:`N = 16` donor regions, :math:`T_1 = 28`
post-periods).

The robustness sweep
--------------------

As the radius :math:`\lambda` grows, the compatible-weight set widens and the
effect shrinks from the classical-synthetic-control neighbourhood toward zero.
mlsynth reproduces the authors' ``DRoSC`` estimand at every radius:

==========  ==============  ==============
:math:`\lambda`  mlsynth :math:`\widehat{\tau}`  R ``DRoSC``
==========  ==============  ==============
0.00        −0.742          −0.742
0.015       −0.486          −0.486
0.03        −0.256          −0.256
0.045       −0.036          −0.036
0.06        0.000           0.000
==========  ==============  ==============

The classical synthetic-control ATT on the same outcome-only fit is −0.895; the
effect is no longer distinguishable from zero once the robustness radius reaches
about 0.06. The worst-case estimand matches R to :math:`\sim 10^{-7}` at every
radius (the residual is cvxpy-vs-``lsei`` solver noise on the shared optimum).

The donor weights
-----------------

At :math:`\lambda = 0` the moment band is tightest and the weights are pinned;
mlsynth reproduces them by name to five decimals:

==========================  ==========  ==========
Donor                       mlsynth     R ``DRoSC``
==========================  ==========  ==========
Madrid                      0.388       0.388
Baleares                    0.274       0.274
Cataluna                    0.203       0.203
Principado De Asturias      0.135       0.135
==========================  ==========  ==========

The perturbation union confidence interval (``inference=True``) is stochastic and
seed-dependent -- it agrees with the R interval within Monte-Carlo error but is
not pinned value-for-value -- so the benchmark cross-validates the deterministic
estimand, which is exact.

Reproduce
---------

.. code-block:: bash

   python benchmarks/run_benchmarks.py --case drosc_basque

The mlsynth side reads ``basedata/basque_jasa.csv``. The R reference is baked
into ``benchmarks/reference/drosc_basque/`` from the authors' ``helpers.R``;
regenerate it with

.. code-block:: bash

   # CRAN firewalled; install limSolve from the GitHub mirror:
   #   git clone --depth 1 https://github.com/cran/limSolve && R CMD INSTALL limSolve
   Rscript benchmarks/R/drosc_basque.R basedata/basque_jasa.csv \
       benchmarks/reference/drosc_basque
