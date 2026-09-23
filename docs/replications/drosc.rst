.. _replication-drosc:

DROSC — Basque robustness sweep vs the authors' R
=================================================

:Estimator: :doc:`../drosc` -- :class:`mlsynth.DROSC`
:Source: Koo, T. & Guo, Z. (2026). "Distributionally Robust Synthetic Control:
   Ensuring Robustness Against Highly Correlated Controls and Weight Shifts."
   `arXiv:2511.02632 <https://arxiv.org/abs/2511.02632>`_. Reference code:
   `taehyeonkoo/DRoSC <https://github.com/taehyeonkoo/DRoSC>`_ (``helpers.R``).
:Replication type: cross-validation against the authors' own R
   (``limSolve::lsei``) run *live* via ``Rscript``, the deterministic worst-case
   point estimand.
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

===============  ==============================  ===========
:math:`\lambda`  mlsynth :math:`\widehat{\tau}`  R ``DRoSC``
===============  ==============================  ===========
0.00             −0.742                          −0.742
0.015            −0.486                          −0.486
0.03             −0.256                          −0.256
0.045            −0.036                          −0.036
0.06             0.000                           0.000
===============  ==============================  ===========

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

The estimation slack, as printed and as coded
---------------------------------------------

The band mlsynth solves is
:math:`\lVert\widehat\gamma - \widehat\Sigma\beta\rVert_\infty \le \lambda + \rho`,
and :math:`\rho` is the slack absorbing the sampling error in
:math:`\widehat\Sigma` and :math:`\widehat\gamma`. The article and the authors'
code parameterise it differently, so this records which one mlsynth follows.

The article's Section 5 gives a single constant multiplying both terms,

.. math::

   \rho = C\Big[\widehat\sigma \cdot \max_j
     \big(\tfrac{1}{T_0}\textstyle\sum_t Y_{j,t}^2\big)^{1/2} + \lambda\Big]
     \frac{\log(\max\{T_0, N\})^{1/2}}{\sqrt{T_0}},

with :math:`C` initialised small and multiplied by 1.25 until the program is
feasible. The authors' ``src/helpers.R`` uses two constants, ``nu`` and ``eta``,
both defaulting to 0.01, and inflates only the first:

.. code-block:: r

   c0 <- log(max(T0,N))^c/sqrt(T0)*(nu*sig*maxnorm + eta*lambda)
   thres <- lambda + c0
   ...
   nu <- 1.25*nu

mlsynth's :func:`mlsynth.utils.drosc_helpers.estimation.drosc_point` reproduces
the R, since the R is what the benchmark pins it against. The two
parameterisations coincide exactly at :math:`\lambda = 0`, where the
:math:`\eta\lambda` term drops out and ``nu`` plays the role of :math:`C`. They
separate for :math:`\lambda > 0`: the article's inflation widens the band through
the :math:`\lambda` term as well, while the code holds that contribution at
:math:`0.01\lambda` and inflates only the noise term, giving a narrower band at
the same number of inflation rounds.

The sweep above therefore matches the reference implementation at every radius,
and matches the article exactly at :math:`\lambda = 0`. The reported effects at
:math:`\lambda > 0` are the code's, which is the intended comparison for a
cross-validation; the difference is a property of the reference, not of the port.

Reproduce
---------

.. code-block:: bash

   python benchmarks/run_benchmarks.py --case drosc_basque

The mlsynth side reads ``basedata/basque_jasa.csv``. The reference is the
authors' own code, run live: ``benchmarks/reference/drosc_basque/reference.R``
clones ``github.com/taehyeonkoo/DRoSC`` (cached), sources its unmodified
``src/helpers.R``, and solves with ``limSolve::lsei`` each time the case runs.
The case ``BenchmarkSkipped``\ s when ``Rscript`` / ``limSolve`` / the clone is
unavailable, so a missing R toolchain never reds the suite. Provision the solver
with ``benchmarks/R/install_drosc.sh`` (``limSolve`` from the GitHub CRAN mirror,
since CRAN is firewalled), then run the reference directly with

.. code-block:: bash

   Rscript benchmarks/reference/drosc_basque/reference.R basedata/basque_jasa.csv
