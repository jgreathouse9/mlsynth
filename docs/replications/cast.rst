.. _replication-cast:

CAST — Inference under Staggered Adoption (Xia, Yan & Wainwright 2025)
======================================================================

:Estimator: :doc:`../cast` — :class:`mlsynth.CAST`
:Source: Xia, E., Yan, Y. and Wainwright, M. J. (2025), *Inference under
   Staggered Adoption: Case Study of the Affordable Care Act*
   (arXiv:2412.09482v2).
:Replication type: Path A (the paper's empirical table on the authors' data)
   plus cross-validation against their released ``CAST-panel`` package.
:Status: Verified — point estimates match the reference to machine precision
   and the paper's Table 1 reproduces cell for cell. Standard errors
   deliberately differ; see below.

Validation strategy
-------------------

The authors ship both the data and a Python package. The
`CAST_panel <https://github.com/Facta-Non-Verba/CAST_panel>`_ repository
carries the Affordable Care Act panels (uninsurance rates, infant mortality,
health expenditures, plus state populations and expansion dates), and the
estimator itself is distributed on PyPI as ``CAST-panel``. That makes the
comparison direct: run their code and ours on the same matrix and compare.

Because the estimator is a deterministic sequence of SVDs and least-squares
solves — no resampling, no random initialisation — an exact match is the
right bar, not a tolerance.

Affordable Care Act — uninsurance rates
---------------------------------------

The Medicaid expansion component of the ACA is the paper's case study. States
adopted in different years from 2014 onward (some never), giving a genuinely
staggered design over a 50-state panel spanning 2008–2022. The rank is 1,
selected by the broken-stick heuristic — mlsynth's ``rank_method="broken_stick"``
reproduces that choice.

Running the authors' package on their data reproduces Table 1 in full:

.. list-table:: Table 1 (uninsurance rates), paper versus reproduction
   :header-rows: 1
   :widths: 10 20 20 25 25

   * - Year
     - ATET (paper)
     - ATET (repro)
     - Pos/Neg/Null (paper)
     - Pop. effect (paper → repro)
   * - 2014
     - −1.7% (0.1)
     - −1.73% (0.07)
     - 1 / 23 / 2 ✓
     - −3.1M → −3.08M
   * - 2015
     - −2.3% (0.1)
     - −2.28% (0.10)
     - 1 / 28 / 0 ✓
     - −5.0M → −4.95M
   * - 2016
     - −2.4% (0.1)
     - −2.45% (0.11)
     - 2 / 28 / 1 ✓
     - −5.8M → −5.85M
   * - 2017
     - −2.7% (0.1)
     - −2.68% (0.12)
     - 1 / 28 / 3 ✓
     - −6.6M → −6.63M
   * - 2018
     - −2.8% (0.1)
     - −2.88% (0.05)
     - 1 / 30 / 1 ✓
     - −6.8M → −6.85M
   * - 2019
     - −2.6% (0.1)
     - −2.60% (0.12)
     - 0 / 29 / 5 ✓
     - −6.7M → −6.65M
   * - 2021
     - −2.9% (0.1)
     - −2.86% (0.09)
     - 1 / 36 / 0 ✓
     - −7.5M → −7.50M
   * - 2022
     - −2.2% (0.2)
     - −2.24% (0.17)
     - 0 / 33 / 6 ✓
     - −6.5M → −6.49M

The significance counts match exactly in all eight years, and the
population-weighted totals to within 0.05 million. Substantively the finding
stands: Medicaid expansion reduced uninsurance by two to three percentage
points in adopting states, covering roughly six to seven million people.

mlsynth against the reference
-----------------------------

On the same panel, mlsynth's port reproduces the reference package's
**point estimates to machine precision**: the largest absolute difference over
all treated unit-periods is :math:`1.7 \times 10^{-16}`. The four-block
routine, the staircase partition and the rank selection all agree.

A standard-error discrepancy in the reference package
-----------------------------------------------------

The standard errors do *not* agree, and we believe mlsynth's are the correct
ones.

In ``CAST-panel`` 1.0.2, the routine that estimates the noise matrix computes
its residuals on the row-sorted staircase but masks them with the *unsorted*
observation indicator::

    E_est = (self.M_sorted - M_est_full) * self.A      # self.A is unsorted

Paper equation (6a) defines the residuals blockwise on the sorted staircase, so
the mask must be sorted with the data. Three independent checks say the
unsorted mask is a bug, not a modelling choice:

* When the sort happens to be the identity permutation — a single adoption
  time with rows already in order — mlsynth and the reference agree on the
  standard errors to :math:`4.2 \times 10^{-16}`. The mask is therefore the
  only source of divergence; the variance formula is otherwise transcribed
  identically.
* On the ACA panel the sort is *not* the identity, and the shipped standard
  errors come out 16% to 65% too small (worst in 2021, a ratio of 1.65).
* In Monte Carlo on a known low-rank staggered design, coverage with the
  sorted mask is closer to nominal at every panel size tested, and the gap
  does not shrink with :math:`N` — the signature of a bug, not a
  finite-sample effect:

  .. list-table::
     :header-rows: 1
     :widths: 30 25 25

     * - Design
       - Shipped mask
       - Sorted mask
     * - :math:`N=60,\ T=40`
       - 0.865
       - 0.902
     * - :math:`N=150,\ T=90`
       - 0.902
       - 0.928
     * - :math:`N=300,\ T=180`
       - 0.918
       - 0.943

  against a nominal 0.95.

mlsynth ships the sorted-mask variance.

What this costs the published significance counts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The correction has a substantive consequence. Table 1's per-year counts of
significantly positive, negative and null states are computed at :math:`\alpha
= 0.01` from the reference package's standard errors. Pair mlsynth's point
estimates with *those* errors and all eight rows reproduce exactly --
confirming the estimator, the staircase mask and the aggregation all agree, and
that the standard error is the only divergence.

Use mlsynth's corrected errors at the same :math:`\alpha`, however, and only
two of the eight rows still match: the errors are on average 1.36 times larger,
so marginal states fall back to null. The paper's headline framing -- "for
every year between 2014 and 2022, nearly every adopting state exhibits a
statistically significant reduction" -- rests in part on the understated
errors.

The direction of the substantive conclusion is unchanged: the ATET is
unaffected (it depends only on the point estimates, which match exactly), the
sign is negative in every year, and the population totals stand. What weakens
is the claim about *how many individual states* clear the significance bar.

The golden benchmark pins both quantities: eight of eight rows under the
reference's errors (the port is exact) and two of eight under the corrected
ones (the consequence is recorded, and would move if either implementation
changed).

Honest caveats
--------------

* The published Table 1 point estimates and population totals are unaffected —
  we reproduced them exactly — but the significance counts are not: see above.
* Coverage is asymptotic and small panels undercover: on a rank-2 design,
  coverage of :math:`M^\star` was about 0.82 at :math:`N=30, T=24` rising to
  0.94 at :math:`N=240, T=160`. The ACA panel (:math:`N=50`, :math:`T=14`) is
  on the small side of that range.
* The noise model assumes independence across time. State-year outcomes are
  usually serially correlated, which would make the intervals optimistic
  independently of the issue above.
* Per-unit significance tests :math:`\tau_{i,t}`, which retains that cell's own
  noise while the interval covers only the counterfactual mean. Individual
  flags are suggestive; the per-period aggregates are the stronger evidence.

Durable benchmark
-----------------

The cross-validation is captured in ``benchmarks/cases/cast_aca.py``, which
installs ``CAST-panel`` from PyPI when available, compares point estimates
value-for-value, and pins the Table 1 aggregates. It reports ``SKIP`` when the
package or its data cannot be fetched, so the suite never turns red on a
machine without network access.
