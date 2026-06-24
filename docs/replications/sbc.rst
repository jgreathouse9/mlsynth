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
a spurious comovement of trends rather than a genuine common structure. The
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

The Hamilton detrending and the trend forecast match the authors' functions to
machine precision. mlsynth's :func:`~mlsynth.utils.sbc_helpers.hamilton.fit_hamilton_filter`
reproduces ``lsq``'s AR coefficients and cyclical residuals, for both the
treated unit (detrended on the pre-treatment window) and the donors (detrended
on the full sample, since the donors are untreated), to about
:math:`10^{-8}`; :func:`~mlsynth.utils.sbc_helpers.trend_forecast.forecast_treated_trend`
reproduces ``trend_predict`` to the same precision. These steps are, for
practical purposes, the same computation written twice.

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
optimum is unique. Four independent solvers agree on that optimum to solver
precision — mlsynth's in-house projected-gradient routine and cvxpy's ECOS,
OSQP and SCS all attain a cyclical sum of squares of about
:math:`1.266\times10^{6}`.

The authors' ``Synth::synth`` (the kernlab ``ipop`` interior-point solver)
instead converges to a point about :math:`2.6\%` worse, a sum of squares of
about :math:`1.299\times10^{6}`, and tightening its tolerances does not close
the gap — it simply lands on a suboptimal vertex. The consequence is a visibly
different weight split: ``ipop`` puts most of its mass on the Netherlands and
implies an average effect near :math:`-1006`, while the verified optimum (which
mlsynth attains) is Greece-dominant with an effect near :math:`-952`. The two
solutions select the same donor set; they differ only because one solver
reaches the optimum and the other does not.

A note on the donor labels
--------------------------

The authors' shipped wide CSV permutes its donor column labels: its
``Japan`` and ``Portugal`` columns actually hold the Netherlands' and Greece's
series (verified against the canonical ``repgermany.dta``). So the paper's prose
naming the cycle donors as "Italy, Japan, Portugal" refers, by the correct
labels, to Italy, the Netherlands and Greece — which is exactly the donor set
mlsynth recovers on the correctly labelled panel. Running the reference, rather
than trusting the printed names, is what surfaces this.

Verification
------------

The durable check lives in ``benchmarks/cases/sbc_germany.py`` (the cycle
weights and the 1991–1994 effect), and the per-step cross-validation in
``mlsynth/tests/test_sbc_reference.py`` (eight tests pinning each stage to the
authors' captured output)::

   python benchmarks/run_benchmarks.py --case sbc_germany
   python -m pytest mlsynth/tests/test_sbc_reference.py

The captured reference bundle, the golden fixture, and the provenance
(R and package versions, data checksums) are under
``benchmarks/reference/sbc_germany/``; its ``NOTICE`` records the full finding.
A separate Path-B Monte Carlo (``sbc_mc``) reproduces the paper's simulation
evidence that SBC stays competitive under cointegration.
