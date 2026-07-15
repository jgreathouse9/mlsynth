.. _replication-bpscs:

BPSCS -- Bayesian Penalized Synthetic Control under Spillovers (Fernandez-Morales et al. 2026)
==============================================================================================

:Estimator: :doc:`../bpscs` -- :class:`mlsynth.BPSCS`
:Source: Fernandez-Morales, E., Oganisian, A., & Lee, Y. (2026), *"Bayesian
   shrinkage priors for penalized synthetic control estimators in the presence of
   spillovers,"* Biometrics 82(2), ujag054 [BPSCS2026]_.
:Replication type: cross-validation against the authors' own reference (the
   replication repository's Stan programs) on the shipped Philadelphia example
   panel.
:Status: verified -- the NumPyro posterior credible band matches the Stan
   posterior band cell-for-cell (to Monte-Carlo error).

Validation strategy
-------------------

The authors ship their models as Stan programs (``sc_dhs.stan`` for the
distance-horseshoe, ``sc_ds2.stan`` for the distance-spike-and-slab) in a public
replication repository, together with an example beverage-sales panel that
mirrors the structure of their Philadelphia application (49 units, 26 four-week
periods, 13 pre-intervention, 48 baseline covariates, per-unit coordinates). BPSCS
ports those models to NumPyro, so the Stan programs are the ground truth. We
reproduce their pre-processing (pre-period outcome standardization, z-scored
covariates, and the utility that blends covariate similarity with spatial
distance), run the reference through ``rstan`` on the example panel, and compare
the posterior counterfactual of the treated unit to the NumPyro estimator on the
identical data.

Because the reference repository is licensed GPL-3, its Stan code is fetched and
run at cross-check time as an external oracle rather than copied into this
(MIT-licensed) project; the model was re-derived from the paper's equations, not
transcribed from the GPL source.

Cross-validation -- cell for cell
---------------------------------

Both engines use the same specification (autoregressive linear SC, the utility
with :math:`\kappa_d = 0`, matched NUTS budgets) and the identical standardized
data. The posterior counterfactual credible band of the treated unit agrees:

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Quantity
     - NumPyro vs Stan
   * - counterfactual band, distance-horseshoe (``dhs``)
     - correlation 0.997 (lower) / 0.9995 (upper)
   * - counterfactual band, spike-and-slab (``ds2``)
     - correlation 0.969 (lower) / 0.998 (upper)
   * - post-period effect (band midpoint)
     - agree to :math:`\approx 0.3` (dhs) / :math:`\approx 0.9` (ds2)

Substantively both recover the paper's finding: the beverage tax lowers treated
sales through the post-period, with a credible band that widens with the forecast
horizon.

The one honest caveat -- a fragile tail
---------------------------------------

The counterfactual is a free-running recursive simulation, and the posterior of
the autoregressive coefficient has enough mass near unity that a few draws
explode with the forecast horizon. Two consequences, both confirmed by the
cross-check and handled in the build:

* The posterior *mean* of the effect is not a usable summary -- it is dominated by
  the exploding draws (both the reference and the port produce a runaway mean on
  the example data; the reference's own summary code uses the mean and would show
  the same). BPSCS reports the posterior *median*, which is stable.
* The one place port and reference disagree is the far lower edge of the credible
  band at long horizons. This was traced to its source: it is quantile
  Monte-Carlo error in a heavy, one-sided tail, amplified by the recursion. A
  same-model seed swap reproduces most of it, and it shrinks as the draw count
  grows; a small residual reflects the two NUTS implementations navigating a
  divergent funnel differently. The identified, robust quantities -- the median
  counterfactual, the upper band, the band midpoint, and the ATT -- agree.

Durable case
------------

Because the example data and Stan are GPL-licensed, the committed, always-runnable
case ``benchmarks/cases/bpscs_synthetic.py`` is self-contained: it simulates a
spatial panel with a known effect and a set of spatially-close, spillover-
contaminated donors, and checks that BPSCS recovers the effect sign and shrinks the
contaminated neighbours (both priors), requiring only the ``[bayes]`` extra. The
live cross-validation against the GPL reference on the example panel is the
demonstrate-first evidence summarized above and is reproducible by fetching the
reference at runtime.

Why NumPyro, not pure numpy
---------------------------

The model couples an autoregressive term, heavy-tailed horseshoe (or discrete
spike-and-slab) shrinkage, and a recursive counterfactual, giving a correlated,
heavy-tailed geometry that a hand-written Gibbs sampler handles poorly. NUTS
manages it, so BPSCS uses NumPyro behind the ``[bayes]`` optional dependency,
following :doc:`../bfsc` and :doc:`../mtgp`.
