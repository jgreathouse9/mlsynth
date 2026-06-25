.. _replication-scta:

SCTA — Temporal Aggregation for Synthetic Control (Sun et al. 2024)
===================================================================

:Estimator: :doc:`../scta` — :class:`mlsynth.SCTA`
:Source: Sun, L., Ben-Michael, E., & Feller, A. (2024), *"Temporal Aggregation
   for the Synthetic Control Method,"* AEA Papers and Proceedings, 114: 614-617.
:Reference implementation: the ``augsynth`` R package (the authors' own library).
:Replication type: cross-validation against ``augsynth`` on the paper's Texas
   SB8 application (Bell, Stuart & Gemmill 2023), to solver tolerance.
:Status: Verified — construction reproduced exactly; estimates agree with
   ``augsynth`` to solver tolerance (see the caveat below).

Validation strategy
-------------------

The paper revisits the Texas SB8 abortion-restriction study, building a
synthetic Texas from monthly state-level live-birth counts (2016-2022) and
asking how the estimated effect moves as the pre-period is aggregated from
months (:math:`\nu = 0`) toward years (:math:`\nu = 1`). The authors ship the
construction in ``augsynth``: append the yearly aggregates as extra
pre-periods, weight them against the months through a fixed diagonal
:math:`\mathbf{V} = \operatorname{diag}(12\nu, 1)`, demean by unit fixed
effects, and solve a simplex synthetic control.

SCTA reproduces that construction natively. mlsynth's ``dataprep`` ingests the
monthly panel; the engine aggregates the six whole calendar years into block
means, stacks them on the seventy-five disaggregated months, applies the
:math:`\nu`-weighted :math:`\mathbf{V}`, demeans, and solves the simplex at the
true optimum.

The construction is exact
-------------------------

Fed ``augsynth``'s own fitted weights, the SCTA counterfactual formula
reproduces the ``augsynth`` annualised combined ATT at :math:`\nu = 0.5` to the
digit (:math:`18{,}917.86`). The temporal-aggregation recipe -- derive the
aggregate, append as extra pre-periods, :math:`\nu`-weighted :math:`\mathbf{V}`,
fixed-effects demean, simplex -- is therefore reproduced exactly.

Cross-validation to solver tolerance
------------------------------------

Run end to end with its own solver, SCTA agrees with ``augsynth`` in direction
and magnitude across the frontier, within a few percent:

.. list-table::
   :header-rows: 1
   :widths: 34 22 22 16

   * - Fit (annualised ATT, :math:`\nu = 0.5`)
     - SCTA (mlsynth)
     - augsynth
     - Gap
   * - Plain simplex + :math:`\mathbf{V}`
     - :math:`{\approx}\,19{,}800`
     - :math:`18{,}918`
     - :math:`+4.5\%`
   * - Ridge-augmented + :math:`\mathbf{V}`
     - :math:`{\approx}\,12{,}500`
     - :math:`12{,}982`
     - :math:`-3.6\%`

Why not bit for bit
-------------------

The combined fit's base simplex is ill-conditioned: with fifty donor states
the donor matrix has condition number :math:`\operatorname{cond}(\mathbf{B}^{\top}\mathbf{B})
\approx 3\times 10^{5}`, so the :math:`\mathbf{V}`-weighted objective has a long,
shallow valley. The minimiser is unique, but ``augsynth``'s interior-point
``LowRankQP`` solver halts about five percent above the true objective
(:math:`7.78\times 10^{9}` versus the true :math:`7.42\times 10^{9}`), on a
slightly more spread-out weight vector. mlsynth reaches the true optimum. The
out-of-sample ATT amplifies that weight difference into the few-percent gaps
above. The ridge-augmented fit inherits the same base simplex, so it is not bit
for bit either.

The honest reading: the temporal-aggregation method and its construction are
reproduced exactly, and the estimates match to the inherent solver tolerance of
an ill-conditioned simplex SC. mlsynth deliberately reports the
true-optimum fit rather than cloning a specific QP solver.

Reproducing
-----------

The durable benchmark assembles the Texas panel, runs SCTA across a
:math:`\nu` grid, and checks the frontier shape and the ATT against the
``augsynth`` reference values above.
