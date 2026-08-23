.. _replication-sbc:

SBC — Synthetic Business Cycle (Shi, Xi & Xie 2025)
===================================================

:Estimator: :doc:`../sbc` — :class:`mlsynth.SBC`
:Source: Shi, Zhentao, Yishen Xi & Jin Xie (2025), *"A Synthetic Business
   Cycle Approach to Counterfactual Analysis with Nonstationary Macroeconomic
   Data,"* arXiv:2505.22388.
:Replication type: Path A — the authors' German-reunification illustration —
   and cross-validation against their own released R code.
:Status: Fully verified, step by step, against the authors' code.

What SBC does
-------------

Classical synthetic control matches a treated unit to a weighted average of
donors on the raw outcome path. When that outcome is nonstationary — a
GDP-per-capita series with a strong trend — matching on the level can lock onto
a spurious comovement of trends, not a genuine common structure. The
synthetic business cycle estimator first splits every series into a slow
*trend* and a stationary *cycle* with a Hamilton filter, forecasts the treated
unit's post-treatment trend from its own history, and builds a synthetic
*cycle* from the donors. The counterfactual is the treated unit's own projected
trend plus a donor-matched cycle, so trend donors and cycle donors are allowed
to differ.

This page records how mlsynth's :class:`~mlsynth.SBC` is validated on the
authors' headline illustration — the 1990 German reunification — not against a
printed table, but against the authors' own R script run live, one step at a
time.

The reference
-------------

The authors release their code at
`github.com/jinxi-atlas/Synthetic-business-cycle-code
<https://github.com/jinxi-atlas/Synthetic-business-cycle-code>`_. Its
``SBC_Germany/Germany.R`` performs the whole procedure: a linear-projection
(Hamilton) detrending via the helper ``lsq``, a trend extrapolation via
``trend_predict``, and the synthetic-control weight solve via
``Synth::synth``. mlsynth's reference bundle
(``benchmarks/reference/sbc_germany/``) reproduces that script's computation on
the authoritative ``basedata/repgermany.dta`` (identical to the Abadie panel),
and the captured outputs of each function become golden values that the unit
tests in ``mlsynth/tests/test_sbc_reference.py`` pin mlsynth against.

Step-by-step agreement
----------------------

mlsynth's :func:`~mlsynth.utils.sbc_helpers.hamilton.fit_hamilton_filter`
reproduces ``lsq``'s AR coefficients and cyclical residuals, for both the
treated unit (detrended on the pre-treatment window) and the donors (detrended
on the full sample, since the donors are untreated), and
:func:`~mlsynth.utils.sbc_helpers.trend_forecast.forecast_treated_trend`
reproduces ``trend_predict``. The two sides differ by
:math:`1.7\times10^{-14}` of each series' scale — about :math:`10^{-11}` in
absolute terms, and the same size for the trend coefficients, both sets of
cycles and the forecast.

That number is a property of the arithmetic, not of the method: it is the
distance between R's ``lm`` QR (LINPACK's ``dqrls``) and numpy's ``lstsq``
(LAPACK's ``gelsd``) on the same design. Two least-squares kernels do not
produce bit-identical answers, so these steps agree as closely as two
implementations of them can, and exact equality is not available at any
recording precision.

How the fixture bounds that claim
---------------------------------

The captured values are decimal text, so a comparison against them can never be
tighter than the digits the text carries. The original capture printed with
``sprintf("%.8f")``, which resolves :math:`5\times10^{-9}` on values of this
size — three orders coarser than the difference above. Every deviation it
reported was therefore its own decimal grid, and the "agreement to about
:math:`10^{-8}`" this page used to claim was the format's resolution described
as a result. ``golden_steps.R`` now emits each compared quantity a second time
at ``%.17g`` under a ``hi:`` prefix, the tests compare at one unit in the last
recorded place, and the :math:`1.7\times10^{-14}` above is what that capture
measures. The comparison is set at :math:`10^{-12}` of the series scale, fifty
times the measured difference, so it fails on a defect while staying indifferent
to which BLAS is installed.

Where the two diverge — and why mlsynth is the accurate one
-----------------------------------------------------------

The only place the two implementations disagree is the synthetic-control weight
solve, and the live replication shows that the divergence is a defect in the
reference solver, not in mlsynth.

At the cycle-matching step both implementations minimise the same objective
over the simplex,

.. math::

   \widehat{w} \;=\; \arg\min_{w \ge 0,\; \mathbf{1}^\top w = 1}
   \;\bigl\lVert\, c_{1} - C\, w \,\bigr\rVert_2^2 ,

where :math:`c_1` is the treated cycle and :math:`C` the donor cycles over the
effective pre-treatment window. On the German panel this program is strictly
convex and well conditioned (the donor cycle matrix has full column rank, and
the Gram matrix's condition number is about :math:`3.8\times10^{3}`), so its
optimum is unique. mlsynth's in-house projected-gradient routine and cvxpy's
ECOS and CLARABEL all attain a cyclical sum of squares of
:math:`1266162.58`, agreeing to :math:`2.7\times10^{-6}` in the weights. OSQP
reaches the same point once its answer is projected back onto the simplex; at
its default tolerance it returns a slightly infeasible one (a weight of
:math:`-9\times10^{-7}`). SCS at its default tolerance returns a point that
violates the constraints more substantially — a weight of :math:`-4.5\times
10^{-3}` — and so reports a lower objective than the optimum; projected back
onto the simplex it is :math:`0.7\%` worse. An infeasible point's objective is
not a solution, which is why the check that matters is not a poll of solvers but
a certificate: linearising the convex objective at mlsynth's weights bounds
every feasible point from below, and mlsynth's answer sits within
:math:`1.4\times10^{-6}` (relative) of that bound. The test asserts the
certificate, so it holds without trusting any solver, mlsynth's included.

The authors' ``Synth::synth`` (the kernlab ``ipop`` interior-point solver)
instead converges to a point about :math:`2.6\%` worse, a sum of squares of
about :math:`1.299\times10^{6}`, and tightening its tolerances does not close
the gap — it simply lands on a suboptimal vertex. The consequence is a visibly
different weight split: ``ipop`` puts most of its mass on the Netherlands and
implies an average effect near :math:`-1006`, while the verified optimum (which
mlsynth attains) is Greece-dominant with an effect near :math:`-952`. The two
solutions select the same donor set; they differ only because one solver
reaches the optimum and the other does not.

The donor labels
----------------

The authors' shipped wide CSV permutes its donor column labels, and not in one
or two columns: of the sixteen donors, fifteen sit under another country's name.
Only Italy is where its label says. Column by column, against the canonical
``repgermany.dta``, the shipped ``Australia`` holds the USA's series,
``Austria`` the UK's, ``Belgium`` Austria's, ``Denmark`` Belgium's, ``France``
Denmark's, ``Greece`` France's, ``Japan`` the Netherlands', ``Netherlands``
Norway's, ``New Zealand`` Switzerland's, ``Norway`` Japan's, ``Portugal``
Greece's, ``Spain`` Portugal's, ``Switzerland`` Spain's, ``UK`` Australia's and
``US`` New Zealand's. The values themselves are the Abadie panel's, to the last
digit; only the names move. So the paper's prose
naming the cycle donors as "Italy, Japan, Portugal" refers, by the correct
labels, to Italy, the Netherlands and Greece — which is exactly the donor set
mlsynth recovers on the correctly labelled panel. Running the reference
instead of trusting the printed names, is what surfaces this.

Verification
------------

The durable check lives in ``benchmarks/cases/sbc_germany.py`` (the cycle
weights and the 1991–1994 effect), and the per-step cross-validation in
``mlsynth/tests/test_sbc_reference.py`` (twelve tests: each stage against the
authors' captured output, the capture's own precision, the feasibility of the
weights, and the optimality certificate)::

   python benchmarks/run_benchmarks.py --case sbc_germany
   python -m pytest mlsynth/tests/test_sbc_reference.py

Sec. 5.1 publishes this application as figures and prose, with no printed
weights or effect size, so the benchmark pins the paper's two checkable claims
exactly — the cyclical mass falls on Greece, the Netherlands and Italy, and the
effect is negative — and pins mlsynth's own output as values. The estimator is
deterministic here, reproducing its ATT and weights bit for bit, so each is
compared at :math:`10^{-6}` relative: six orders above the floating-point
movement a different platform introduces, and three orders below the shift a
cyclical solver truncated to 200 iterations produces.

The captured reference bundle, the golden fixture, and the provenance
(R and package versions, data checksums) are under
``benchmarks/reference/sbc_germany/``; its ``NOTICE`` records the full finding.
A separate Path-B Monte Carlo (``sbc_mc``) reproduces the paper's simulation
evidence that SBC stays competitive under cointegration, on panels drawn in R by
the authors' own ``simulation_nonnegative.R`` and estimated by mlsynth, so the
simulator sits on the reference side of that comparison.
