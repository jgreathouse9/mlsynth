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

Lower in-sample risk than LowRankQP, provably
---------------------------------------------

Choosing the true optimum is not a stylistic preference: it gives a strictly
smaller in-sample balancing risk than ``augsynth``'s solver, by construction.
Both methods minimise the same objective over the same feasible set,

.. math::

   f(\gamma) = (\mathbf{a} - \mathbf{B}\gamma)^{\top}\mathbf{V}(\mathbf{a} - \mathbf{B}\gamma),
   \qquad \gamma \in \Delta = \{\gamma \ge 0,\ \textstyle\sum_i \gamma_i = 1\}.

The objective is convex (Hessian :math:`2\mathbf{B}^{\top}\mathbf{V}\mathbf{B}
\succeq 0`) and, with a full-rank :math:`\mathbf{B}` and :math:`\mathbf{V}
\succ 0`, strictly convex, so it has a unique global minimiser
:math:`\gamma^{\star}`. By the definition of a minimiser,
:math:`f(\gamma^{\star}) \le f(\gamma)` for every feasible :math:`\gamma` --
including ``augsynth``'s ``LowRankQP`` iterate, which is feasible (it lies on
the simplex). Hence

.. math::

   f(\gamma_{\text{SCTA}}) = f(\gamma^{\star}) \;\le\; f(\gamma_{\text{LowRankQP}}),

strictly whenever ``LowRankQP`` halts short of the optimum. This is exact,
holding for every dataset and every :math:`\nu`, not an asymptotic or average
statement. The one requirement -- that mlsynth attains :math:`\gamma^{\star}`
-- is certified by solving with two independent algorithms (the active-set QP
and interior-point CLARABEL) and confirming they agree on the objective to
:math:`\le 7\times 10^{-9}` relative at every :math:`\nu`.

Across the :math:`\nu` frontier on the Texas panel, the in-sample
:math:`\mathbf{V}`-weighted risk confirms it, and the suboptimality of
``LowRankQP`` grows as aggregation stretches the design:

.. list-table::
   :header-rows: 1
   :widths: 12 26 26 20

   * - :math:`\nu`
     - SCTA (true optimum)
     - augsynth LowRankQP
     - augsynth above optimum
   * - 0.0
     - :math:`6.628\times 10^{9}`
     - :math:`6.628\times 10^{9}`
     - :math:`0.0\%`
   * - 0.25
     - :math:`7.039\times 10^{9}`
     - :math:`7.076\times 10^{9}`
     - :math:`+0.5\%`
   * - 0.5
     - :math:`7.419\times 10^{9}`
     - :math:`7.780\times 10^{9}`
     - :math:`+4.9\%`
   * - 1.0
     - :math:`8.121\times 10^{9}`
     - :math:`9.813\times 10^{9}`
     - :math:`+20.8\%`
   * - 2.0
     - :math:`9.402\times 10^{9}`
     - :math:`1.163\times 10^{10}`
     - :math:`+23.7\%`
   * - 4.0
     - :math:`1.164\times 10^{10}`
     - :math:`1.807\times 10^{10}`
     - :math:`+55.3\%`

At :math:`\nu = 0` (no aggregation, well-conditioned) the two agree; as
:math:`\nu` rises the :math:`K\nu` row scaling worsens the conditioning and
``LowRankQP``'s interior iterate drifts further above the vertex optimum. The
risk here is in-sample pre-treatment balance, the quantity both solvers
target; reaching its true constrained minimum is the correct solve, and is
distinct from the out-of-sample estimation risk the paper's bias bounds
address.

Reproducing
-----------

The durable benchmark assembles the Texas panel, runs SCTA across a
:math:`\nu` grid, and checks the frontier shape and the ATT against the
``augsynth`` reference values above.
