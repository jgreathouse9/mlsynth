.. _replication-bscm:

BSCM — Bayesian Synthetic Control Methods (Kim, Lee & Gupta 2020)
=================================================================

:Estimator: :doc:`../bscm` — :class:`mlsynth.BSCM`
:Source: Kim, Sungjin, Lee, Clarence, and Gupta, Sachin (2020),
   *"Bayesian Synthetic Control Methods,"* Journal of Marketing Research
   57(5):831-852 [BSCM2020]_.
:Replication type: cross-validation against the authors' reference Stan
   implementation (``clarencejlee/bscm``) and against the forward-selected
   panel-data approach on the China anti-corruption watch panel.
:Status: verified — the horseshoe matches the reference Stan to four decimals on
   the ATT, and agrees with the forward-selection benchmark on the same data.

Why the China watch panel, and not the paper's data
---------------------------------------------------

The paper's empirical application measures the effect of a 2010 Washington soda
tax using proprietary state-level weekly Nielsen retail data, which is not
public. BSCM is designed for the "large p, small n" regime, so this replication
validates on a public panel in exactly that regime: the China anti-corruption
watch-demand series (``china_watches_long.csv``) -- one treated series
(``watches``), 87 donor series, and 35 pre-treatment months, with treatment at
2013-01 when the anti-corruption campaign began. With 87 donors and 35
pre-periods this is genuinely :math:`p > n`, the setting BSCM was built for, and
it is the original benchmark for the forward-selected panel-data approach, which
gives a second, independent method to check against.

Validation strategy
-------------------

BSCM ships a reference Stan implementation (``clarencejlee/bscm``): the program
``Horseshoe_Publish.stan`` fits the treated unit on the donor pool with an
intercept and the horseshoe prior, then produces the post-treatment
counterfactual in the ``generated quantities`` block. Sampled with rstan (4
chains, 2000 iterations, 1000 warm-up), it is the ground truth. mlsynth
reproduces it with a pure-numpy Gibbs sampler -- the horseshoe via the
Makalic-Schmidt (2016) auxiliary-variable representation of the paper's exact
hierarchy -- so there is no Stan or probabilistic-programming dependency in the
library.

Cross-validation — three methods agree
--------------------------------------

Fed the identical panel, the pure-numpy horseshoe, the reference Stan horseshoe,
and the forward-selected panel-data approach all converge:

.. list-table::
   :header-rows: 1
   :widths: 30 16 24 14 16

   * - Method
     - ATT
     - 95% interval
     - pre-RMSE
     - dominant donor
   * - Stan horseshoe (reference)
     - :math:`-0.0221`
     - :math:`[-0.066, +0.024]`
     - 0.0999
     - C60 = 0.68
   * - mlsynth BSCM horseshoe
     - :math:`-0.0221`
     - :math:`[-0.064, +0.020]`
     - 0.0988
     - C60 = 0.68
   * - FSPDA (forward selection)
     - :math:`-0.0107`
     - :math:`[-0.035, +0.013]`
     - 0.0995
     - C60 = 0.85

The pure-numpy port and the reference Stan agree to four decimals on the ATT
(:math:`-0.0221`), on the credible band, on the pre-RMSE, and donor for donor
(C60 :math:`0.68` in both; C51, C45, C57, C25 next in both). The forward-selected
approach -- a completely different, greedy sparse method, and the dataset's own
benchmark -- picks the same dominant donor and returns the same near-null ATT.
All three intervals cross zero: no statistically credible effect on this
residualised watch-demand series in this configuration.

A robustness note that cuts in the port's favour. With :math:`p = 87` donors the
horseshoe posterior is a difficult funnel-shaped geometry; the reference Stan run
reports divergent transitions and a maximum :math:`\widehat{R}` of about 1.02
even at ``adapt_delta = 0.999``. The block Gibbs with auxiliary variables has no
funnel to fall into, so it mixed cleanly and still landed on the reference
answer -- on this benchmark the pure-numpy sampler is the more robust of the two.

Overfitting — a cautionary counter-example (Basque)
---------------------------------------------------

BSCM can overfit, and the in-sample pre-fit will not warn you. On the Basque
Country panel (16 donors, 15 pre-periods, so :math:`p \approx n`) the horseshoe
drives the pre-treatment RMSE to about :math:`0.004` -- roughly twenty times
tighter than a simplex SCM. That fit is an interpolation artifact, not a
success: a held-out pre-period test (train on the early pre-period, predict the
last few pre-treatment years, where the true effect is zero) shows a
generalisation error dozens of times larger than the in-sample fit. The China
watch panel, with more than twice the pre-periods, does not suffer this: its
held-out error is close to its in-sample error. The lesson, and the reason this
replication is anchored on the watch panel rather than Basque, is that
overfitting in these models is governed by the absolute number of pre-periods.
Judge a BSCM fit by held-out or leave-one-out prediction and by the width of the
credible band, never by the in-sample pre-RMSE. The durable case is
``benchmarks/cases/bscm_china_watches.py``.
