.. _property-rsc-rank-condition:

RSC — the rank condition, and the two hyperparameters (Amjad, Shah & Shen 2018)
===============================================================================

:Estimator: :doc:`../clustersc` — :class:`mlsynth.CLUSTERSC` with
   ``method="pcr"``, ``clustering=False``
:Source: Amjad, M., Shah, D. & Shen, D. (2018), *"Robust Synthetic
   Control,"* Journal of Machine Learning Research 19(22):1-51 —
   Theorem 6 (Section 4.3), Theorem 3 (Section 4.2.1), and the reading of
   Theorems 3 and 7 in "Benefits of regularization" (Section 4.3).
:Results checked: Theorem 6, both directions; Theorem 3's bias-variance
   reading of the singular-value threshold; and the paper's directional
   claim about the ridge penalty, which does not survive measurement.
:Benchmark case: `benchmarks/cases/rsc_rank_condition_mc.py
   <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/rsc_rank_condition_mc.py>`_
:Status: Reproduced. Where the ranks agree the pre-period relation
   extrapolates to 2e-15; where they do not it fails on every design tested,
   costing the estimator a 14-fold post-period error. The threshold's
   bias-variance tradeoff is measured and holds. The regularization claim
   does not: the exchange it describes — a worse pre-period fit bought in
   return for a better post-period one — appears at neither threshold
   tested.

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

What regularization does
------------------------

The threshold :math:`\mu` is one of the algorithm's two hyperparameters. The
other is the ridge penalty :math:`\eta` of equation (18),

.. math::

   \widehat\beta(\eta) = \operatorname*{argmin}_{v \in \mathbb{R}^{N-1}}
       \bigl\| Y_1^- - (\widehat M^-)^\top v \bigr\|^2
     + \eta \sum_{j=1}^{N-1} |v_j|^q,

at :math:`q = 2`. mlsynth spells the same objective as
``lambda_penalty * ||w||_p ** q``, so ``lambda_penalty=eta, p=2, q=2`` is
this estimator; ``tests/test_pcr_ridge.py`` holds that against the closed
form :math:`(\widehat M^{-\top}\widehat M^- + \eta I)^{-1}\widehat M^{-\top}
Y_1^-`.

Section 4.3 reads Theorems 3 and 7 together to say what :math:`\eta` buys.
Theorem 3's bound carries a term :math:`+\eta\|\beta^*\|^2/T_0`, and the
paper reads it as

   As seen from Theorem 3, the pre-intervention error increases linearly
   with respect to the choice of :math:`\eta`. Intuitively, this increase in
   pre-intervention error derives from the fact that regularization reduces
   the model complexity, which biases the model and handicaps its ability to
   fit the data.

Theorem 7's second term is controlled by
:math:`\|\widehat\beta(\eta) - \beta^*\|`, and the paper reads it as

   Therefore, a larger value of :math:`\eta` reduces the post-intervention
   error. […] employing ridge regression introduces extraneous bias into our
   model, yielding a higher pre-intervention error. In exchange,
   regularization reduces the post-intervention error (due to smaller
   variance).

Both halves are directional, so both can be measured. Sweeping :math:`\eta`
on the same Section 5.3 design, at the threshold that matches the signal's
rank and at one an order of magnitude too permissive, over two seeds and
three targets, scoring equation (24)'s MSE on the pre window and equation
(33)'s RMSE on the post, both against the true :math:`M`:

.. list-table::
   :header-rows: 1
   :widths: 14 21 21 22 22

   * - :math:`\eta`
     - rank 3, pre MSE
     - rank 3, post RMSE
     - rank 25, pre MSE
     - rank 25, post RMSE
   * - 0
     - 0.0235
     - 0.1568
     - 0.0341
     - 0.1902
   * - 100
     - 0.0234
     - 0.1567
     - 0.0331
     - 0.1876
   * - 300
     - 0.0234
     - 0.1569
     - 0.0315
     - 0.1835
   * - 1000
     - 0.0235
     - 0.1585
     - 0.0285
     - 0.1756
   * - 3000
     - 0.0257
     - 0.1697
     - 0.0276
     - 0.1766
   * - 10000
     - 0.0415
     - 0.2224
     - 0.0418
     - 0.2238

The exchange is at neither threshold. Where :math:`\mu` matches the signal's
rank, :math:`\eta` has nothing to offer in either window: the best
post-period cell beats :math:`\eta = 0` by 0.01 per cent, and past
:math:`\eta = 1000` both columns climb together. Where :math:`\mu` is too
permissive, :math:`\eta` helps in both windows at once — 19 per cent off the
pre-intervention error and 8 per cent off the post. The pre-period error
never pays for the post-period one, because the two move together.

The two knobs are substitutes, not opposites. Both suppress directions the
data does not support: :math:`\mu` by dropping them from the retained
subspace, :math:`\eta` by shrinking the weights placed on them. Where
:math:`\mu` has already removed the noise directions there is nothing left
for :math:`\eta` to shrink, and where it has not, :math:`\eta` does the job
:math:`\mu` skipped. Past :math:`\eta = 10^4` the two rows agree to three
decimals in both windows: the penalty dominates and the retained rank stops
mattering at all.

Where the reasoning parts from the measurement
-----------------------------------------------

Neither theorem is contradicted. Theorem 3 is an upper bound, and a bound
may rise in :math:`\eta` while the error beneath it falls. What the
measurement reaches is the prose, and it separates two quantities the
argument runs together.

The intuition offered for the :math:`\eta` term — that regularization
"handicaps its ability to fit the data" — is about the fit to the observed
:math:`Y_1^-`. That is measured too, and it is correct: the training error
rises monotonically in :math:`\eta` at both thresholds, by 1.9 per cent at
rank 3 and 2.7 per cent at rank 25 across the grid. But equation (24)'s
:math:`\mathrm{MSE}(\widehat M_1^-)` is the error against the signal
:math:`M_1^-`, not against :math:`Y_1^-`, and the two move oppositely
exactly when the retained subspace carries noise. That is the rank-25
column: the fit to :math:`Y` degrades and the error against :math:`M`
improves, at the same :math:`\eta`, on the same fits.

Theorem 7's own term behaves as the paper says. Farebrother's (1976)
existence claim, which Section 4.3 invokes, holds at both thresholds:
:math:`\|\widehat\beta(\eta) - \beta^*\|` has an interior minimum, 11 per
cent below :math:`\eta = 0` at rank 3 and 51 per cent below at rank 25. What
does not follow is the conclusion drawn from it. At both thresholds that
term is still improving at an :math:`\eta` where the post-intervention error
it bounds has already turned around and started climbing — at rank 3 it
keeps improving to :math:`\eta = 1000` while the error's minimum is at 100.
So the one part of Theorem 7's bound that depends on :math:`\eta` moves the
right way while the error moves the wrong way, which is the whole distance
between a bound and the quantity under it.

The practical reading is the ordering. :math:`\eta` is not a second
independent dial to turn after :math:`\mu`; whether it helps at all is
decided by whether :math:`\mu` was set too low. Cross-validating them
jointly, as Section 3.4.3 prescribes, is what the measurement supports.
Cross-validating :math:`\eta` alone in the expectation of trading pre-period
fit for post-period accuracy is not.

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
structure they are read as asserting: which way each error moves as a
hyperparameter turns. That is weaker than a bound, and it is what the
statements support. On the threshold the structure holds. On the
regularizer the errors move together where the paper says they move apart,
and the bounds themselves stand — the reading of them is what the
measurement reaches.

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

Deterministic throughout, about fifty seconds.

.. code-block:: python

   import numpy as np
   from mlsynth.utils.clustersc_helpers.simulation import simulate_rank_shift_panel

   p = simulate_rank_shift_panel(dormant_factor=True, N=12, T=60, T0=40,
                                 n_factors=3, noise=0.0, seed=0)
   np.linalg.matrix_rank(p.means[:, :p.T0]), np.linalg.matrix_rank(p.means)
   # (2, 3) -- Theorem 6's hypothesis fails

.. code-block:: python

   # equation (18) at q = 2, through the public API
   CLUSTERSC({..., "method": "pcr", "pcr_objective": "OLS",
              "lambda_penalty": 1000.0, "p": 2.0, "q": 2.0}).fit()
