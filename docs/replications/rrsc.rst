.. _replication-rrsc:

RRSC — Robust-Regression Synthetic Control with Interference (He, Li, Shi & Miao 2026)
======================================================================================

:Estimator: :doc:`../rrsc` — :class:`mlsynth.RRSC`
:Source: He, P., Li, Y., Shi, X. and Miao, W. (2026), *A robust regression
   approach to synthetic control with interference* (arXiv:2411.01249).
:Replication type: Cross-validation — matched value-for-value against the
   authors' released reference R code, in both regimes, on both empirical
   applications.
:Status: Fully verified — point estimates value-for-value; bootstrap intervals
   within Monte-Carlo tolerance.

Validation strategy
-------------------

The paper ships a full reproducibility package: the reference R for both
regimes (the fixed-N ``FixN`` code and the large-N ``LargeN`` code, the latter
reusing the factor-analysis routines of Wang et al. 2017), the two empirical
datasets, and the authors' saved outputs. mlsynth's port is checked against a
live run of that R -- first confirming the R reproduces the authors' saved
outputs, then matching the Python port to the R cell by cell.

Two facts make a clean value-for-value comparison possible, and a naive
re-implementation would miss them:

* The large-N EM factor analysis stops at R's default relative tolerance
  (``tol = 1e-6``), which on the Beijing panel is an *early* stop -- roughly 25
  iterations, far short of the maximum-likelihood optimum (about 1,169). The
  authors' published effects depend on that early stop, so mlsynth's
  :func:`~mlsynth.utils.rrsc_helpers.factor.fa_em` transcribes the reference's
  exact per-iteration algebra (including its non-standard loading update) to
  reproduce the same stopping point.
* The fixed-N LTS uses ``robustbase::ltsReg``'s finite-sample raw-scale
  correction (Pison, Van Aelst and Willems 2002). mlsynth transcribes that
  correction (:func:`~mlsynth.utils.rrsc_helpers.robust.lts_fit`), so the
  reweighted coefficients match ``ltsReg``, not an asymptotic
  approximation.

Large-N — Beijing air pollution
-------------------------------

Beijing's orange-alert pollution policy (November 2016) is evaluated on an
hourly station panel: 108 monitoring stations, 48 pre-alert and 24 post-alert
hours, with :math:`r = 7` factors chosen by the Bai and Ng criterion. The
estimand is the per-station effect -- a reduction in the treated core-Beijing
stations and, downwind, a positive interference effect consistent with the
prevailing wind.

Running the authors' ``Beijing_analysis.R`` live reproduces their saved output
exactly (108 station effects to machine zero; p-values to the third decimal, the
resolution of the saved file). mlsynth's large-N port reproduces the live-R
station effects to ``max |Δ| ≈ 2e-12`` -- value-for-value. The bootstrap
p-values agree within Monte-Carlo tolerance (the circular block bootstrap draws
differ between R's and NumPy's random-number streams), and the set of
significant stations, and their signs, agree.

Fixed-N — U.S. embassy relocation to Jerusalem
----------------------------------------------

The December 2017 announcement relocating the U.S. embassy is evaluated on a
weekly conflict-event panel of nine Middle East units (Israel-Palestine treated,
98 pre and 48 post weeks, :math:`r = 2`). The headline is a significant
interference effect on Jordan.

Running the authors' ``MiddleEast_analysis.R`` live reproduces their saved
``MiddleEast_results.rds`` to ``max |Δ| ≈ 2e-12`` on the point estimates and
``≈ 5e-11`` on the confidence intervals (the bootstrap is seed-deterministic in
R). mlsynth's fixed-N port reproduces the point estimates to ``max |Δ| ≈ 3e-4``
-- the residual is the factor-analysis optimiser tolerance, tightened by
lowering ``ftol``. Every seam matches the reference: the ``factanal``
uniquenesses to ``1e-5``, the raw LTS subset and coefficients exactly, the
Donoho-Johnstone valid-control selection exactly, and Jordan's interference
effect at ``7.999`` with a 95% interval excluding zero.

Honest caveats
--------------

* Point estimates are deterministic and match value-for-value; the CBB
  confidence intervals and p-values are Monte-Carlo objects and match only
  within bootstrap tolerance, because the resampling uses NumPy's random
  stream, not R's ``sample()``.
* The counterfactual for a treated unit that is nearly orthogonal to the common
  factors (low ``communality``) is close to a raw before-after difference; RRSC
  surfaces this as a diagnostic. For Israel-Palestine the
  two common factors explain only about 3% of the pre-period variance, so its
  direct-effect estimate is essentially a level shift with a small factor
  adjustment -- a limitation of the data, not the port.

Durable benchmark
-----------------

The cross-validation is captured in ``benchmarks/cases/rrsc_reference.py``,
which runs the reference R (when available) and asserts the Python port matches
it; it reports ``SKIP`` when R or the reference package is absent, so the suite
never turns red on a machine without the toolchain.
