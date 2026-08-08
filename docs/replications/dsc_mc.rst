.. _replication-dsc-mc:

DSC asymptotics Monte Carlo (Zhang, Zhang & Zhang 2026)
========================================================

:Estimator: :doc:`../dsc` -- :class:`mlsynth.DSC`
:Source: Zhang, L., Zhang, X. and Zhang, X. (2026), *"Asymptotic Properties of
   the Distributional Synthetic Controls,"* `arXiv:2405.00953v3
   <https://arxiv.org/abs/2405.00953>`_ -- the model-free Monte Carlo of
   Section 5.1 (Figures 1 and 2).
:Replication type: Path B (the paper's Monte Carlo), scenario 1 -- the paper
   alone, with no released code, no data, and no table.
:Status: partially verified -- the geometry both theorems predict reproduces in
   full, and the risk ratio matches the published figure to 0.0017 at every cell
   with :math:`M \ge 200`. The two cells at :math:`M = 50` diverge, by 0.006 and
   0.028, and the weight-error curve is steeper in the paper than in the
   reconstruction. Both distances are measured and reported by the case.
:Benchmark: ``benchmarks/cases/dsc_mc.py``
   (`source <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/dsc_mc.py>`__).

Why this case exists
--------------------

:class:`mlsynth.DSC` implements Algorithm 1 of this paper, so this is the
estimator's own source simulation.

It complements, and does not replace, the external check DSC already has.
:doc:`disco_tenure` reproduces the ``disco`` Stata Journal article's published
donor weights bit-for-bit at the reference's own settings (five weights to 5e-5,
the effects table to 5e-4), which is what establishes that mlsynth implements
this specification. What that case cannot do is say whether the specification
converges where the theory says it should, because on an empirical panel there
is no known truth to converge to.

This case supplies that. The optimum is available in closed form at every one of
its sixteen points, so the two quantities the theorems are about can be measured
against it directly. It also isolates a narrower target: the design hands the
estimator its pseudo-samples, so what is exercised is the weight solver and the
:math:`\lambda_t` aggregation, not the quantile-estimation step the empirical
cases cover.

What the paper proves, and what the simulation shows
----------------------------------------------------

Distributional synthetic control reconstructs a treated unit's whole outcome
distribution, not just its mean. Each unit-period cell carries a sample, the
sample gives an empirical quantile function, and the treated unit's
counterfactual quantile function is built as a weighted average of the donors',
with weights on the simplex (non-negative, summing to one). The distance being
minimised is the 2-Wasserstein distance, which for quantile functions is an
ordinary squared error, so choosing the weights is a constrained least-squares
problem.

The integral defining that distance is approximated by sampling :math:`M` points
of the quantile function. The paper asks what happens as :math:`M` grows, and
proves two things.

Theorem 1 (asymptotic optimality). Write :math:`\bar R_{T_1}(w)` for the
averaged post-treatment 2-Wasserstein distance at weight :math:`w`. The fitted
weight :math:`\widehat w` attains, in the limit, the smallest value any simplex
weight attains: the ratio :math:`\bar R_{T_1}(\widehat w) / \inf_{w}
\bar R_{T_1}(w)` tends to 1.

Theorem 2 (weight convergence). The fitted weight itself converges to the
minimiser :math:`w^{\mathrm{opt}}`, at a rate that slows as the donor pool
:math:`J` grows.

Section 5.1's two figures are those two statements measured on simulated data,
at :math:`J \in \{20, 50\}` and :math:`M \in \{50, 100, 200, 400\}`, over 1000
replications.

The design
----------

For every period :math:`t` and draw :math:`m`, the treated unit's pseudo-sample
is :math:`\widetilde Y_{1tm} \sim \chi^2(2)` and each donor's is
:math:`\widetilde Y_{jtm} \sim \mathcal N(\mu_j, \sigma_j^2)`, with
:math:`\mu_j \sim U(3, 10)` and :math:`\sigma_j = 3` for odd unit labels
:math:`j` and 2.5 for even ones. There are :math:`T_0 = 10` pre-treatment
periods, :math:`T_1 = 5` post-treatment periods, and the per-period weights are
aggregated with equal weights :math:`\lambda_t = 1/T_0`.

The draws are taken at ranks generated in dependent pairs: :math:`M/2` ranks
:math:`V^{(1)}_k \sim U[0, 1]`, each given a partner :math:`V^{(2)}_k =
V^{(1)}_k \pm \delta` with :math:`\delta = 0.01`, shifted toward the centre of
the unit interval so every value stays inside it. The dependence is the point of
the construction. Gunsilius (2023) requires the sampled ranks to be independent;
this paper's Assumption 4 relaxes that to a mixing condition, and the paired
draws are the illustration.

That pairing has a consequence the paper does not draw out. Shifting a rank by
0.01 barely moves a central draw and moves an extreme one a long way, so half
the sample loses tail mass: the realised second moment of a standard normal draw
is 0.928 against a nominal 1, and the donors' realised spread is 0.9633 times
:math:`\sigma_j`. The paper writes its risk in terms of the nominal
:math:`\mathcal N(\mu_j, \sigma_j^2)`, so the sampling scheme and the target are
not quite the same object. Both readings were tried, at 60 replications per
cell. The nominal one is closer to the published ratios at six of the seven
cells compared, by up to 0.009, and is what the benchmark uses; at the seventh
(:math:`J = 20`, :math:`M = 200`) the sampling-implied reading is closer, by
0.001. ``mlsynth/tests/test_dsc_simulate.py`` pins the compression so the
distinction stays visible.

Since the design is time-invariant, the averaged post-treatment risk is a single
quadratic form :math:`w' G w - 2 b' w + c`, with :math:`G = \mu \mu' +
\operatorname{diag}(\sigma^2)`, :math:`b = 2\mu` and :math:`c = 8`. The oracle
:math:`w^{\mathrm{opt}}` is therefore exact algebra, not a nested simulation,
and it is solved through the estimator's own solver: :math:`G` is positive
definite, so its Cholesky factor turns the quadratic into a least-squares
problem that :func:`~mlsynth.utils.dsc_helpers.weights.solve_simplex_weights`
accepts directly. Oracle and estimate come from one code path.

Where the target numbers come from
----------------------------------

Section 5 reports no tables. Its four panels are figures, and the numbers behind
them are not printed anywhere in the paper.

They are recoverable. The arXiv source ships the figures as vector PDFs written
by R 4.2.2, so the plotted series survive as explicit path operators in the
content stream: a ``m``/``l`` polyline through the four data points, plus the
axis gridlines and tick labels needed to calibrate it. Reading the two curves
out of each panel and mapping device coordinates onto the labelled axes gives
the sixteen plotted values. The y axis of the ratio panel resolves to about
6e-5 per device unit, which is far below the Monte-Carlo noise in the values
themselves, so digitisation error does not enter the comparison.

The recovered values, which are the ``_PAPER_RATIO`` and ``_PAPER_NORM`` tables
in the case:

.. list-table:: Figure 1 -- :math:`\bar R_{T_1}(\widehat w) / \inf_w \bar R_{T_1}(w)`
   :header-rows: 1
   :widths: 12 22 22 22 22

   * - :math:`J`
     - :math:`M = 50`
     - :math:`M = 100`
     - :math:`M = 200`
     - :math:`M = 400`
   * - 20
     - 1.0238
     - 1.0145
     - 1.0075
     - 1.0030
   * - 50
     - 1.0274
     - 1.0180
     - 1.0099
     - 1.0043

.. list-table:: Figure 2 -- :math:`\| \widehat w - w^{\mathrm{opt}} \|`
   :header-rows: 1
   :widths: 12 22 22 22 22

   * - :math:`J`
     - :math:`M = 50`
     - :math:`M = 100`
     - :math:`M = 200`
     - :math:`M = 400`
   * - 20
     - 0.1856
     - 0.1221
     - 0.0668
     - 0.0276
   * - 50
     - 0.2789
     - 0.2123
     - 0.1301
     - 0.0593

Two details had to be inferred
------------------------------

Under scenario 1 the DGP is reconstructed from prose, and two choices the paper
leaves open change the answer.

Whether every unit carries its own rank sequence. This is the one that decides
the replication, and the paper points both ways. Section 2 defines the risk as
:math:`\int_0^1 (\sum_j w_j F^{-1}_{Y_{jt}}(q) - F^{-1}_{Y_{1t}}(q))^2 dq`, a
single rank shared by every unit, which makes the units comonotonic. The moment
matrix the paper writes for that same risk has off-diagonal entries
:math:`\mu_i \mu_j`, which is independence across units, and the companion
quantile-factor design of Section 5.2 generates its pseudo-sample with no ranks
at all. Only the independent reading reproduces the published figures. Under the
shared-rank reading the estimator converges far faster than the paper shows --
a risk ratio of 1.002 against their 1.024 at :math:`J = 20`, :math:`M = 50`.
The benchmark uses the independent reading, and the population risk it is scored
against is the moment matrix implied by that same reading, so estimator and
target agree.

Whether the design is redrawn. The rank sequence is redrawn every period and
:math:`\mu_j` every replication. Neither is stated.

Result
------

The geometry both theorems predict reproduces in full. At both donor-pool sizes
the risk ratio falls monotonically toward 1 and the weight error falls
monotonically as :math:`M` grows; the :math:`J = 50` curve sits above the
:math:`J = 20` curve at every draw count, which is Theorem 2's slower
convergence for the larger pool; and the ratio's excess over 1 falls by an order
of magnitude from :math:`M = 50` to :math:`M = 400`, by 13.0x and 11.1x against
the published curves' 7.9x and 6.4x.

The risk ratio is a cell match everywhere except at the smallest draw count.

.. list-table:: Figure 1, risk ratio -- reconstruction against the published figure
   :header-rows: 1
   :widths: 10 12 20 20 20

   * - :math:`J`
     - :math:`M`
     - Paper
     - mlsynth
     - Gap
   * - 20
     - 50
     - 1.0238
     - 1.0299
     - +0.0061
   * - 20
     - 100
     - 1.0145
     - 1.0120
     - -0.0025
   * - 20
     - 200
     - 1.0075
     - 1.0063
     - -0.0012
   * - 20
     - 400
     - 1.0030
     - 1.0023
     - -0.0007
   * - 50
     - 50
     - 1.0274
     - 1.0559
     - +0.0285
   * - 50
     - 100
     - 1.0180
     - 1.0250
     - +0.0070
   * - 50
     - 200
     - 1.0099
     - 1.0116
     - +0.0017
   * - 50
     - 400
     - 1.0043
     - 1.0050
     - +0.0007

Every cell with :math:`M \ge 200` lands within 0.0017 of the published value,
and five of the eight within 0.0025. The published curve spans 1.003 to 1.027,
so those are small fractions of the effect being plotted.

The two cells at :math:`M = 50` are where the curves part, and the pattern says
where to look: the gap grows as the draw count falls toward the donor count. At
:math:`J = 50, M = 50` the donor matrix is square, its condition number is
2.0e4, and the reconstruction sits 0.0285 high. More broadly, the published
curves depend on the donor pool much less than the reconstruction does -- the
paper's :math:`J = 50` ratio at :math:`M = 50` is only 0.0036 above its
:math:`J = 20` ratio, against 0.0260 here.

Near-degeneracy makes the solver the obvious suspect, and it has been ruled out.
Handed the identical matrices, mlsynth's solver and the reference's -- CLARABEL
against ``pracma::lsqlincon`` under DiSCo's own argument construction -- return
weights agreeing to 3e-9, at that square cell as much as at the
well-conditioned ones, with the population risk at each agreeing to six
decimals. Whatever separates the two curves at :math:`M = 50` sits upstream of
the weight solve, in the design specification. The comparison is
``benchmarks/reference/dsc_mc/`` and needs the R reference.

The weight-error half is not a cell match.

.. list-table:: Figure 2, weight error -- reconstruction against the published figure
   :header-rows: 1
   :widths: 10 12 20 20 20

   * - :math:`J`
     - :math:`M`
     - Paper
     - mlsynth
     - Gap
   * - 20
     - 50
     - 0.1856
     - 0.1323
     - -0.0533
   * - 20
     - 100
     - 0.1221
     - 0.0798
     - -0.0423
   * - 20
     - 200
     - 0.0668
     - 0.0660
     - -0.0008
   * - 20
     - 400
     - 0.0276
     - 0.0448
     - +0.0172
   * - 50
     - 50
     - 0.2789
     - 0.1509
     - -0.1280
   * - 50
     - 100
     - 0.2123
     - 0.1167
     - -0.0956
   * - 50
     - 200
     - 0.1301
     - 0.0828
     - -0.0473
   * - 50
     - 400
     - 0.0593
     - 0.0552
     - -0.0041

The published curve is steeper. It starts above the reconstruction at
:math:`M = 50` and crosses below it by :math:`M = 400`, decaying at roughly
:math:`M^{-0.92}` against the reconstruction's :math:`M^{-0.48}` -- and the
latter is the rate sampling error alone produces. Decay faster than
:math:`M^{-1/2}` is what happens when the active set is being recovered: once
:math:`M` is large enough, the simplex projection pins the donors that should
carry no weight at exactly zero, and the error collapses onto the remaining
coordinates. The reconstruction shows that effect only weakly. Which unstated
detail of the design produces the stronger version is unresolved.

Those eight cells are therefore pinned as mlsynth's own deterministic output,
and the distance from the paper is reported as a number,
``max_abs_norm_gap_vs_paper`` = 0.128, instead of being hidden inside a wide
tolerance. The risk-ratio half reports the same way, in three nested windows:
``max_abs_ratio_gap_vs_paper_Mge200`` = 0.0017,
``max_abs_ratio_gap_vs_paper_J20`` = 0.0025 over that column at
:math:`M \ge 100`, and ``max_abs_ratio_gap_vs_paper`` = 0.0285 over the whole
grid.

What this validates, and what it does not
-----------------------------------------

The paper's Monte Carlo generates the pseudo-samples directly, and under the
reading that reproduces it, draws each unit independently. It therefore never
exercises the step where DSC estimates an empirical quantile function from a
sample and evaluates it on a shared grid. What it does exercise is the rest of
Algorithm 1: the simplex-constrained least-squares solver
(:func:`~mlsynth.utils.dsc_helpers.weights.solve_simplex_weights`) and the
:math:`\lambda_t` aggregation across pre-treatment periods. Those are the two
steps the theorems are about.

The case calls those helpers directly for that reason. Feeding the same draws
into :meth:`mlsynth.DSC.fit` as micro-data would sort each cell into an
empirical quantile function, and sorting is exactly the shared-rank coupling
this design does not use -- so the estimator would be measured against a
different population object and would trace a different, faster curve. The
quantile-estimation step is covered instead by :doc:`dsc` and
:doc:`disco_tenure`.

Section 5.2 is not reproduced
-----------------------------

The paper's second design is a quantile factor model,
:math:`\widetilde Y_{itm} = \lambda_{1,i,m} f_{1,t,m} + \lambda_{2,i,m}
f_{2,t,m}`, at :math:`J \in \{10, 20\}` and :math:`M \in \{100, 200, 300, 400\}`.
Its published figures are not recoverable from the stated design. They show a
risk ratio of 1.1010 and 1.1551 at :math:`M = 100` collapsing to 1.0075 and
1.0380 at :math:`M = 200`, and a weight error falling to 0.0016 and 0.0054 by
:math:`M = 400`. A faithful port of the design gives 1.011 and 1.022 at
:math:`M = 100` and weight errors of 0.048 and 0.057 at :math:`M = 400` -- the
right ordering in :math:`J` and the right direction in :math:`M`, but neither
the spike at the smallest draw count nor the collapse after it. The design is
left out of the benchmark instead of being matched with a tolerance wide enough
to accommodate a factor of ten.

Reproducing it
--------------

.. code-block:: bash

   python benchmarks/run_benchmarks.py --case dsc_mc

The case runs in about two and a half minutes at ``_R = 24`` replications per
cell against the paper's 1000, and that reduced count, not the digitisation, is
what sets the width of its Monte-Carlo bands. Every replication is seeded, so
the reported numbers are exactly reproducible at that count.

To re-derive the target values from the paper instead of trusting the
transcription in the case:

.. code-block:: bash

   curl -o 2405.00953v3.tar.gz https://arxiv.org/e-print/2405.00953v3
   mkdir -p src && tar -xzf 2405.00953v3.tar.gz -C src
   python benchmarks/reference/dsc_mc/digitize_figures.py src
