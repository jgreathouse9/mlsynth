.. _replication-scta:

SCTA — Temporal Aggregation for Synthetic Control (Sun et al. 2024)
===================================================================

:Estimator: :doc:`../scta` — :class:`mlsynth.SCTA`
:Source: Sun, L., Ben-Michael, E., & Feller, A. (2024), *"Temporal Aggregation
   for the Synthetic Control Method,"* AEA Papers and Proceedings, 114: 614-617.
:Reference implementation: ``augsynth`` 0.2.0, the authors' own R package,
   driven by their ``time_aggregation.rmd``.
:Replication type: Path A on the authors' data and code, plus cross-validation
   against ``augsynth`` cell by cell.
:Status: Verified. The construction reproduces, and the estimates agree with
   ``augsynth`` to 0.11 percent once the two libraries' aggregation knobs are
   put on the same scale — which takes a square, described below.

Validation strategy
-------------------

The paper revisits the Texas SB8 abortion-restriction study, building a
synthetic Texas from monthly state-level live-birth counts and asking how the
estimated effect moves as the pre-period is aggregated from months toward years.
The authors ship the construction in ``augsynth``: append the yearly aggregates
as extra pre-periods, weight them against the months through a fixed diagonal
:math:`\mathbf{V}`, demean by unit fixed effects, and solve a simplex synthetic
control.

SCTA reproduces that construction natively. mlsynth's ``dataprep`` ingests the
monthly panel; the engine aggregates the six whole calendar years into block
means, stacks them on the seventy-five disaggregated months, applies the
:math:`\nu`-weighted :math:`\mathbf{V}`, demeans, and solves the simplex at the
true optimum.

The panel is the authors' ``compileddata.csv``, vendored verbatim as
``basedata/texas_sb8_births.csv``: 51 states, 84 months over 2016-2022, live
births annualised by a factor of twelve, Texas treated from April 2022. That
leaves 75 pre-treatment months — six whole years plus a three-month tail — and
nine post-treatment months. The data are CDC WONDER natality counts compiled by
Bell, Stuart and Gemmill (2023), public domain, and the authors license
redistribution.

The knob is squared
-------------------

Both libraries fit the paper's program. They disagree about what the aggregation
parameter means, and the disagreement is a square.

The paper writes the balancing objective as

.. math::

   \min_{\gamma \in \Delta}\;
   (\mathbf{a} - \mathbf{B}\gamma)^{\top}\, \mathbf{V}\, (\mathbf{a} - \mathbf{B}\gamma),
   \qquad
   \mathbf{V} = \operatorname{diag}(K\nu, \ldots, K\nu, 1, \ldots, 1).

SCTA implements it by scaling the matching rows by :math:`\sqrt{\mathbf{V}}`, so
an aggregate row's weight in the objective is :math:`K\nu`. ``augsynth`` reaches
the program by scaling the matching columns instead —
``X_c <- X_c %*% V`` in ``fit_ridgeaug_formatted`` — and then solving with the
weight matrix reset to the identity. Scaling by :math:`\mathbf{V}` and then
squaring residuals weights an aggregate row by :math:`(K\nu)^2`. So
``augsynth``'s ``year_wt`` and SCTA's ``nu`` are different parameters:

.. math::

   K\,\nu \;=\; (K \cdot \texttt{year\_wt})^{2}
   \qquad\Longrightarrow\qquad
   \nu \;=\; K \cdot \texttt{year\_wt}^{2}, \qquad K = 12.

The paper's "Yearly + Monthly" fit is ``year_wt = 1``, which is
:math:`\nu = 12`, not :math:`\nu = 1`.

This is measured, not inferred from reading the source. Handed ``augsynth``'s
own demeaned design and its own :math:`\mathbf{V}`, mlsynth's active-set QP
reproduces ``augsynth``'s weight vector to :math:`4.3 \times 10^{-7}` in
:math:`L_1` over fifty donors under the squared reading, and the objective at
``augsynth``'s weights sits at the minimum to :math:`6.7 \times 10^{-8}`
relative. Under the linear reading it does not: the weight vectors differ by
:math:`0.23` to :math:`1.40` in :math:`L_1` and the objective at ``augsynth``'s
weights sits 4.9 to 23.7 percent above the minimum.

.. list-table::
   :header-rows: 1
   :widths: 12 12 19 19 19 19

   * - ``year_wt``
     - :math:`\nu`
     - :math:`L_1`, squared
     - excess, squared
     - :math:`L_1`, linear
     - excess, linear
   * - 0
     - 0
     - :math:`5.3\times 10^{-8}`
     - :math:`-4.6\times 10^{-8}`
     - 0.0000
     - 0.0%
   * - 0.5
     - 3
     - :math:`3.9\times 10^{-7}`
     - :math:`+1.4\times 10^{-8}`
     - 0.2272
     - +4.9%
   * - 1
     - 12
     - :math:`4.3\times 10^{-7}`
     - :math:`-1.1\times 10^{-8}`
     - 1.0596
     - +20.8%
   * - 2
     - 48
     - :math:`6.0\times 10^{-8}`
     - :math:`-2.3\times 10^{-8}`
     - 1.3840
     - +23.7%
   * - 30
     - 10800
     - :math:`1.7\times 10^{-7}`
     - :math:`+6.7\times 10^{-8}`
     - 1.4033
     - +12.1%

At :math:`\nu = 0` the aggregate rows carry no weight under either reading and
the two coincide exactly, which is the control that says the rest of the table
is about :math:`\mathbf{V}` and not about anything else.

The solver is not the difference
--------------------------------

``augsynth`` 0.2.0 solves the simplex with OSQP at ``eps_abs = eps_rel = 1e-8``
(``augsynth:::synth_qp``). On an identical design and an identical objective its
weights and mlsynth's active-set QP agree to that tolerance, and the signed
objective excess in the table above lands on either side of zero — which is what
two solvers meeting at the same optimum looks like. Neither implementation is
stopping short of the other.

An earlier version of this page attributed the gap between the two libraries to
``augsynth``'s solver halting above the optimum, and argued that mlsynth
therefore attained a strictly lower in-sample balancing risk. That was wrong.
The argument assumed both libraries minimise the same objective over the same
feasible set; they minimise different objectives, and each reaches its own
minimum. The numbers that appeared to support it — an excess of 4.9 percent at
``year_wt = 0.5`` rising to 23.7 percent — are real, and are the linear column
above: they measure ``augsynth``'s weights against the wrong program.

A third thing the mapping corrects
----------------------------------

The paper's equal-weight case is ``year_wt = 1``. The authors' code says so
directly: the variable holding that fit is named ``plt_equal``, its panel in
Figure 2 is titled "Yearly + Monthly Births", and Figure 3 plots the estimate
against :math:`\texttt{year\_wt}/(\texttt{year\_wt}+1)`, on which
``year_wt = 1`` sits at :math:`0.5`.

SCTA's ``nu`` defaulted to :math:`0.5` and this documentation called that the
paper's equal-weight heuristic. It is not — :math:`0.5` is the position of the
equal-weight case on that axis, not the knob value. On mlsynth's own convention
the check is arithmetic and does not need the paper at all: the
:math:`\lfloor T_0/K \rfloor` aggregate rows carry total weight
:math:`\lfloor T_0/K \rfloor \cdot K\nu` against :math:`T_0` for the
disaggregated rows, so the halves balance at :math:`\nu = 1`. On this panel that
is 72 against 75, the slack being the three-month tail. At :math:`\nu = 0.5` the
aggregates carry 36 against 75.

The default is unchanged pending a decision, since moving it changes every
existing caller's answer; the documentation now states what the value is and
what the paper's case would be.

Cross-validation, at the mapped knob
------------------------------------

With :math:`\nu = K \cdot \texttt{year\_wt}^2`, SCTA run end to end reproduces
``augsynth``'s estimate across the grid:

.. list-table::
   :header-rows: 1
   :widths: 14 12 24 24 16

   * - ``year_wt``
     - :math:`\nu`
     - SCTA (mlsynth)
     - ``augsynth``
     - Gap
   * - 0 (monthly alone)
     - 0
     - 20877.26
     - 20895.54
     - :math:`-0.09\%`
   * - 0.5
     - 3
     - 18905.14
     - 18917.86
     - :math:`-0.07\%`
   * - 1 (yearly + monthly)
     - 12
     - 21710.99
     - 21735.25
     - :math:`-0.11\%`
   * - 2
     - 48
     - 22924.61
     - 22942.44
     - :math:`-0.08\%`
   * - 30 (yearly alone)
     - 10800
     - 24633.00
     - 24653.97
     - :math:`-0.09\%`

Estimates are the mean post-treatment effect on annualised births.

The residual: what is demeaned by what
--------------------------------------

The remaining tenth of a percent is a second convention difference, and it is
one-directional — SCTA is below ``augsynth`` at every point on the grid, which
is the signature of a systematic difference and not of noise.

``augsynth``'s ``fixedeff`` demeans each unit by ``rowMeans`` of its matching
row (``augsynth:::demean_data``). On the stacked design that row has 81 entries:
six yearly aggregates and 75 months. So the fixed effect is a weighted blend in
which the first six years count twice — once through the months they contain and
again through their own aggregate — and the three-month tail counts once. SCTA
demeans by the mean of the 75 disaggregated months alone.

For Texas the two differ by 28.7 annualised births on a level of about 379,300,
:math:`7.6 \times 10^{-5}` in relative terms. Because the weights are on the
simplex, a common shift would cancel; these are per-unit and do not. Put SCTA's
solver on ``augsynth``'s demeaning basis and the same squared objective, and the
gap closes to :math:`1.0 \times 10^{-7}` relative — so the demeaning basis
accounts for the whole of the residual.

Which basis is right is a question the paper does not settle. Demeaning by the
disaggregated pre-period mean is the unit's pre-treatment level; demeaning by
the stacked row mean is what a generic ``rowMeans`` does to a design whose rows
are not all the same kind of thing. SCTA takes the first. The size of the choice
is 0.1 percent here, and it is recorded so a reader can price it.

Path A: the paper's figures
---------------------------

The frontier of the paper's Figure 1 reproduces in shape. As weight moves onto
the yearly aggregates the monthly imbalance rises monotonically (9401 to 14403)
and the yearly imbalance falls monotonically (4891 to 3139): aggregation buys
balance on the aggregates and spends it on the months.

The three labelled points of Figure 3 come out at 20896 (monthly alone), 21735
(yearly + monthly) and 24654 (yearly alone) annualised births, all positive and
all of the same order — the paper's finding that the estimate is sensitive to
the aggregation choice, and that the sensitivity should be traced instead of
resolved by fiat.

Reproducing
-----------

.. code-block:: bash

   python benchmarks/run_benchmarks.py --case scta_texas_sb8

The case is `benchmarks/cases/scta_texas_sb8.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/scta_texas_sb8.py>`_.
It runs in two seconds and pins fifteen quantities: the design's counts against
``augsynth``'s own, both directions of the :math:`\mathbf{V}` reading, the
end-to-end agreement at the mapped knob, the demeaning residual and its size,
the three labelled estimates, and the frontier's two monotonicities. It reads
``augsynth``'s numbers from the captured bundle and runs no R.

The bundle is `benchmarks/reference/scta_texas_sb8/
<https://github.com/jgreathouse9/mlsynth/tree/main/benchmarks/reference/scta_texas_sb8>`_
— the transcribed ``reference.R``, its verbatim output, the parsed values and
full provenance. Regenerating it needs R with ``augsynth`` and ``dplyr``:

.. code-block:: bash

   python benchmarks/reference/generate.py scta_texas_sb8

A second, R-free check
----------------------

`benchmarks/cases/scta_ibex_xval.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/scta_ibex_xval.py>`_
builds the paper's Section 2 design from the equations, with no import from
``mlsynth.utils.scta_helpers``, and solves it with ``cvxpy``/CLARABEL on the
monthly ibex day-ahead price panel used by the :doc:`ibex` replication. It
agrees with SCTA on the ATT to :math:`5\times 10^{-12}` across a :math:`\nu`
grid and on the ridge-augmented ATT to :math:`2.7\times 10^{-10}`.

It covers what the Texas case cannot: the Texas case reads a captured bundle, so
it can tell you mlsynth still agrees with a fixed set of ``augsynth`` numbers,
but a change to mlsynth's own construction would move both sides of the ibex
comparison only if the change were in the shared code — and it is not, because
the ibex reference is written independently. The two together check the
construction against an independent implementation and against the authors'.
