.. _property-rsc-rank-condition:

RSC — the rank condition behind extrapolation (Amjad, Shah & Shen 2018)
========================================================================

:Estimator: :doc:`../clustersc` — :class:`mlsynth.CLUSTERSC` with
   ``method="pcr"``, ``clustering=False``
:Source: Amjad, M., Shah, D. & Shen, D. (2018), *"Robust Synthetic
   Control,"* Journal of Machine Learning Research 19(22):1-51 —
   Theorem 6 (Section 4.3) and Theorem 3 (Section 4.2.1).
:Results checked: Theorem 6, both directions; Theorem 3's bias-variance
   reading of the singular-value threshold.
:Benchmark case: `benchmarks/cases/rsc_rank_condition_mc.py
   <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/rsc_rank_condition_mc.py>`_
:Status: Reproduced. Where the ranks agree the pre-period relation
   extrapolates to 2e-15; where they do not it fails on every design tested,
   costing the estimator a 14-fold post-period error.

The assumption nobody states
----------------------------

Every synthetic control does the same two things. It finds weights that make
a combination of donors track the treated unit before the intervention, and
then it applies those weights after the intervention to say what would have
happened. The first step is fitting. The second is an extrapolation, and it
needs a reason.

Amjad, Shah and Shen give the reason, and observe that it had been missing:
the pre-period relation is assumed, in their equation (6), as
:math:`M_1^- = (M^-)^\top\beta^*` on the signal matrix :math:`M`, and then

   the question still remains: does the same relationship hold for the
   post-intervention regime and if so, under what conditions does it hold?
   [...] It is worth noting that this important aspect has been amiss in the
   literature, potentially implicitly believed or assumed starting in the
   work by Abadie and Gardeazabal (2003).

Theorem 6. Let equation (6) hold for some :math:`\beta^*`, and let
:math:`\operatorname{rank}(M^-) = \operatorname{rank}(M)`. Then
:math:`M_1^+ = (M^+)^\top\beta^*`.

*Remark.* The condition says the pre-period saw every direction the panel
ever moves in. When it did, fitting on the pre-period pins the relation
everywhere. When it did not — when some driver was dormant before the
intervention and wakes after it — the pre-period system is rank-deficient in
the donors, many weight vectors reproduce the treated unit exactly, and they
disagree precisely along the direction the post-period is about to reveal. A
fit that sees only the pre-period has no way to choose among them.

Making both sides constructible
-------------------------------

The case builds a factor panel :math:`M = \Lambda F^\top` with three
factors, and sets the treated unit's loading to a combination of the donors'
so that equation (6) holds at every date by construction. The premise is
never in question; only whether a pre-period fit recovers it.

.. list-table::
   :header-rows: 1
   :widths: 30 22 22 26

   * - Design
     - :math:`\operatorname{rank}(M^-)`
     - :math:`\operatorname{rank}(M)`
     - Theorem 6 applies
   * - all factors active
     - 3
     - 3
     - yes
   * - one factor dormant until :math:`T_0`
     - 2
     - 3
     - no

What the case measures
----------------------

First, on the noise-free signal matrix, with no estimator involved: fit
:math:`\beta` to reproduce :math:`M_1^-` exactly, then measure
:math:`\max_t |M_{1t} - (M^+)^\top\beta|` after the intervention, over eight
designs.

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * -
     - Rank preserved
     - Rank deficient
   * - Worst post-period gap
     - :math:`1.8 \times 10^{-15}`
     -
   * - Smallest post-period gap
     -
     - 0.114
   * - Mean post-period gap
     -
     - 0.493

Machine precision on one side, and failure on every single design on the
other. This half is exact linear algebra: it is the theorem, measured.

Second, what the condition costs the estimator. The same two designs with
observation noise, run through ``CLUSTERSC(method="pcr", clustering=False)``
— the Amjad-Shah-Shen RSC — scoring the post-period counterfactual against
the true signal:

.. list-table::
   :header-rows: 1
   :widths: 46 27 27

   * -
     - Rank preserved
     - Rank deficient
   * - Post-intervention RMSE
     - 0.0244
     - 0.342

A factor of 14. Nothing is wrong with the estimator in either column; it
fits the pre-period well in both. What differs is whether that fit means
anything afterwards.

Theorem 3 and the Goldilocks principle
--------------------------------------

Theorem 3 bounds the pre-intervention error for any singular-value threshold
:math:`\mu`, and the paper reads its two leading terms against each other.
Lowering :math:`\mu` enlarges the retained set :math:`S`, which shrinks the
signal left unmodelled (:math:`\lambda^*`) and grows the noise mistaken for
signal (:math:`|S|\sigma^2/T_0`). Neither extreme is right, which is what the
paper calls the Goldilocks principle.

Sweeping the retained rank on the paper's own Section 5.3 design, scoring
against the true mean:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Retained rank
     - Pre-intervention MSE
     - Post-intervention MSE
   * - 1
     - 0.1601
     - 0.2472
   * - 2
     - 0.0221
     - 0.0236
   * - 3
     - 0.0225
     - 0.0239
   * - 5
     - 0.0232
     - 0.0247
   * - 10
     - 0.0248
     - 0.0264
   * - 25
     - 0.0326
     - 0.0344

The U-shape is there in both columns, and it is lopsided: underfitting costs
a factor of 10.5, overfitting by an order of magnitude in the rank costs
1.46. Cutting the threshold too high is much the more expensive mistake, at
this noise level.

The minimum sits at rank 2, with rank 3 within 1.4% of it. The Section 5.3
signal is approximately rank 3 — a unit intercept, a shared seasonal
pattern, and a :math:`\theta`-scaled trend — and the third component is weak
beside unit-variance noise, so the two are effectively tied. The case admits
either.

What is established, and what is not
------------------------------------

Theorem 6 carries no unspecified constants, so it is checked as what it is:
an equality that holds or does not. Both directions are measured, and the
hypothesis is pinned alongside the conclusion, so the result cannot later be
read off a design that stopped satisfying it.

Theorems 3 and 7 and Corollary 4 are a different matter. Each reads
:math:`\le C_1(\cdot) + C_2(\cdot)` with :math:`C_1, C_2` described only as
"universal positive constants", so none can be checked as a numerical bound
— there is no number to compare against. What is checked instead is the
structure Theorem 3 asserts: which way the error moves as the threshold
turns, and that it turns in both directions. That is weaker than a bound and
it is what the statement supports.

One practical reading. The rank condition is about the panel, not the
method, and it is not testable from data alone — a dormant factor is
invisible until it wakes. What the pre-period can show is whether the donors
span enough directions to make the condition plausible, and what a
practitioner can ask is whether anything is known to change at the
intervention beyond the treatment itself. If something does, no synthetic
control estimator recovers it, and the failure is silent: the pre-period fit
stays excellent.

Reproducing it
--------------

.. code-block:: bash

   python benchmarks/run_benchmarks.py --case rsc_rank_condition_mc

Deterministic throughout, about three seconds.

.. code-block:: python

   import numpy as np
   from mlsynth.utils.clustersc_helpers.simulation import simulate_rank_shift_panel

   p = simulate_rank_shift_panel(dormant_factor=True, N=12, T=60, T0=40,
                                 n_factors=3, noise=0.0, seed=0)
   np.linalg.matrix_rank(p.means[:, :p.T0]), np.linalg.matrix_rank(p.means)
   # (2, 3) -- Theorem 6's hypothesis fails
