Matrix Completion with Nuclear Norm Minimization (MCNNM)
=========================================================

.. currentmodule:: mlsynth

When to Use This Estimator
--------------------------

``MCNNM`` implements the matrix-completion estimator of Athey, Bayati,
Doudchenko, Imbens and Khosravi [MCNNM]_. Its argument for *when* it is the
right tool is unifying: the authors show (their Theorem 1) that the
unconfoundedness, synthetic-control, and difference-in-differences estimators
all minimize the same least-squares objective and differ only in the
restrictions they impose -- unconfoundedness reweights *time periods*,
synthetic control reweights *control units*, DiD imposes parallel trends.
MC-NNM replaces those hard restrictions with a nuclear-norm regularization
on a low-rank factor model, exploiting the cross-sectional *and* time-series
structure at once.

The practical payoff is robustness across regimes. Because it does not commit
to one restriction, MC-NNM performs well whether the panel is wide
(``N >> T``), tall (``T >> N``), or roughly square (``N ~ T``) -- settings
where synthetic control, DiD, or vertical regression each individually
degrade. Reach for it when you have a single treated unit *or* many, a single
adoption date *or* staggered adoption, and you would rather regularize the
latent structure than assume which comparison (units vs. periods) is the right
one. The cost is that the estimand is a low-rank *imputation* rather than an
interpretable set of donor weights.

Do not use MCNNM when
~~~~~~~~~~~~~~~~~~~~~~

* An interpretable donor-weight story is the deliverable. MC-NNM
  imputes a low-rank matrix; it does not hand you a sparse "California =
  0.4 Utah + 0.3 Montana" convex combination. If the weights *are* the
  result, use :doc:`tssc`, :doc:`scmo`, or :doc:`fdid`.
* There is no low-rank structure. The nuclear-norm regularization is
  only useful when the control matrix is approximately low rank plus noise.
  With a slowly-decaying spectrum, prefer a balancing estimator
  (:doc:`microsynth`) or a selection estimator (:doc:`fdid`).
* Missingness is informative (MNAR / self-masking) -- the probability a
  cell is observed depends on its own value, as in recommender systems.
  MC-NNM assumes a structured-but-exogenous missingness pattern; use
  :doc:`snn`, which is built for missing-not-at-random data.
* Spillovers contaminate the control block (SUTVA fails). The low-rank
  signal model treats controls as untreated; use :doc:`spsydid` or
  :doc:`spillsynth`.
* Treatment is endogenous and you have an instrument. MC-NNM imputes
  :math:`Y(0)` but does not break simultaneity; use :doc:`siv`.
* Distributional questions (quantiles, tails) -- MC-NNM targets the mean
  ATT; use :doc:`dsc`.

Notation
--------

The outcome panel is the :math:`N \times T` matrix :math:`\mathbf{Y} = (Y_{it})`
for units :math:`i = 1, \ldots, N` over periods :math:`t = 1, \ldots, T`. A
treatment-indicator matrix :math:`\mathbf{D}` marks treated cells; the
observed (untreated) cells form the index set :math:`\mathcal{O}`, and
:math:`P_{\mathcal{O}}` is the projection that zeros out the rest
(:math:`P_{\mathcal{O}}^{\perp}` its complement). The untreated potential
outcomes follow a low-rank component plus two-way fixed effects,

.. math::

   \mathbf{Y}(0) = \mathbf{L}^{*} + \boldsymbol{\Gamma}\mathbf{1}_T^{\top}
       + \mathbf{1}_N \boldsymbol{\Delta}^{\top} + \boldsymbol{\varepsilon},

with unit effects :math:`\boldsymbol{\Gamma} \in \mathbb{R}^N`, time effects
:math:`\boldsymbol{\Delta} \in \mathbb{R}^T`, and mean-zero noise. The nuclear
norm :math:`\|\mathbf{L}\|_{*} = \sum_i \sigma_i(\mathbf{L})` sums the singular
values. The intervention splits the panel into pre-period
:math:`\mathcal{T}_1 = \{1, \ldots, T_0\}` and post-period; the treatment effect
on a treated cell is :math:`\hat\tau_{it} = Y_{it} - \widehat{Y}_{it}(0)` and the
ATT is its average over treated cells.

The estimator
~~~~~~~~~~~~~

Treating the treated cells as missing, MC-NNM solves (paper Eq. 4.3)

.. math::

   (\widehat{\mathbf{L}}, \widehat{\boldsymbol{\Gamma}}, \widehat{\boldsymbol{\Delta}})
     = \operatorname*{argmin}_{\mathbf{L}, \boldsymbol{\Gamma}, \boldsymbol{\Delta}}
       \frac{1}{|\mathcal{O}|}
       \bigl\| P_{\mathcal{O}}(\mathbf{Y} - \mathbf{L}
         - \boldsymbol{\Gamma}\mathbf{1}_T^{\top}
         - \mathbf{1}_N\boldsymbol{\Delta}^{\top}) \bigr\|_F^2
       + \lambda \|\mathbf{L}\|_{*}.

Only the low-rank part :math:`\mathbf{L}` is regularized; the unit/time fixed
effects are estimated explicitly and left unregularized, which markedly
improves the imputations (paper Section 4). The counterfactual for a treated
cell is read off the completed matrix,
:math:`\widehat{Y}_{it}(0) = \widehat{L}_{it} + \widehat\Gamma_i + \widehat\Delta_t`.

Algorithm (SOFT-IMPUTE)
~~~~~~~~~~~~~~~~~~~~~~~

The problem is solved by singular-value soft-thresholding [Mazumder2010]_
(paper Eq. 4.4-4.5). With the shrink operator
:math:`\mathrm{shrink}_\lambda(\mathbf{A}) = \mathbf{S}\,\widetilde{\boldsymbol{\Sigma}}\,\mathbf{R}^{\top}`
(each singular value replaced by :math:`\max(\sigma - \lambda, 0)`), iterate

.. math::

   \mathbf{L}_{k+1} = \mathrm{shrink}_{\lambda|\mathcal{O}|/2}
       \bigl\{ P_{\mathcal{O}}(\mathbf{Y} - \widehat{\mathrm{FE}})
               + P_{\mathcal{O}}^{\perp}(\mathbf{L}_k) \bigr\},

re-fitting the fixed effects from their first-order conditions after each step
until convergence. The regularization strength :math:`\lambda` is chosen by
cross-validation over the observed cells (an ``n_lambda``-point grid,
``n_folds`` folds).

Assumptions and remarks
~~~~~~~~~~~~~~~~~~~~~~~~

*Assumption 1 (low rank + two-way effects).* The untreated outcomes are a
low-rank matrix plus unit and time fixed effects, with mean-zero idiosyncratic
noise. *Remark.* This is the single structural assumption that subsumes the
others: a rank-:math:`r` factor model nests interactive fixed effects, while the
explicit two-way effects absorb additive unit/time shifts so the nuclear-norm
penalty only has to recover the *interaction* structure.

*Assumption 2 (missingness / no anticipation).* Treatment makes a cell's
untreated outcome missing; the observed (untreated) cells are informative for
the missing ones, and there is no anticipation. *Remark.* Staggered adoption
produces a "staircase" of missing cells, which the mask-based completion fills
directly -- MC-NNM's main advantage over fixed-rank interactive-fixed-effects
methods that need a rectangular treated block.

*Assumption 3 (regularization rate).* :math:`\lambda` shrinks at the rate the
theory prescribes so the completion is consistent; in practice it is selected by
cross-validation on held-out observed cells. *Remark.* Choosing :math:`\lambda`
too small overfits noise into :math:`\mathbf{L}`; too large over-shrinks and
biases the imputed counterfactual toward the fixed-effects-only fit.

*Assumption 4 (jackknife inference).* The leave-one-control refits are
approximately exchangeable, so their dispersion estimates the ATT's sampling
variability. *Remark.* MC-NNM has no closed-form standard error; ``mlsynth``
follows the matrix-completion literature in using a jackknife (see *Inference*).

Causal use and staggered adoption
---------------------------------

``MCNNM`` marks the treated post-treatment cells as missing, imputes their
untreated outcomes, and forms :math:`\hat\tau_{it} = Y_{it} - \widehat{Y}_{it}(0)`,
aggregated to the ATT over treated cells. Adoption times are detected with
:func:`mlsynth.utils.datautils.dataprep`, and the result exposes two
staggered-aware aggregations beyond the overall ATT:

* ``cohort_att`` -- ``{adoption_time: ATT}`` for each adoption cohort.
* ``event_study`` -- ``{relative_time: average effect}``, re-centred on each
  unit's own adoption date (negative keys are pre-adoption fit checks, ~0;
  non-negative keys are the dynamic effects).

With multiple adoption times, ``display_graphs=True`` draws an event-study
plot (effect vs. time-since-adoption) rather than a single calendar trajectory,
so cohorts at different event times are not blended.

Inference
---------

Setting ``inference=True`` runs a leave-one-control jackknife for the ATT:
drop one control unit, refit at the cross-validation-selected :math:`\lambda`,
recompute the ATT, and form
:math:`\widehat{\mathrm{se}}^2 = \tfrac{q-1}{q}\sum_{q}(\hat\tau_q - \bar\tau)^2`
over the ``q`` control-deletions, with a Wald interval at level ``alpha``. This
is a standard inference for matrix-completion estimators (no analytic SE
exists); it captures donor-pool uncertainty. The interval is returned on
:class:`~mlsynth.utils.mcnnm_helpers.structures.MCNNMInference` (``se``, ``ci``).

Example
-------

Proposition 99 -- California's 1988 tobacco-control program. MC-NNM imputes
California's post-1988 counterfactual per-capita cigarette sales by matrix
completion and reports the ATT; with ``display_graphs=True`` it draws the
observed-vs-counterfactual chart.

.. code-block:: python

   import pandas as pd
   from mlsynth import MCNNM

   url = ("https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
          "refs/heads/main/basedata/smoking_data.csv")
   df = pd.read_csv(url)

   res = MCNNM({
       "df": df, "outcome": "cigsale", "treat": "Proposition 99",
       "unitid": "state", "time": "year",
       "inference": True,          # jackknife ATT SE / CI
       "display_graphs": True,     # observed vs MC-NNM counterfactual
   }).fit()

   print(f"ATT (avg 1989-2000) = {res.att:+.2f} packs/capita")
   lo, hi = res.att_ci               # standardized; jackknife when inference=True
   print(f"jackknife 95% CI    = [{lo:+.2f}, {hi:+.2f}]")
   print(f"gap by 2000         = {res.att_by_period[2000]:+.2f}")
   print(f"selected lambda     = {res.best_lambda:.2f}")

Verification
------------

.. note::

   Empirical (Proposition 99). MC-NNM imputes a near-exact California
   pre-treatment fit and an average ATT of about :math:`-20` packs per capita,
   widening to roughly :math:`-30` by 2000 -- consistent with Abadie, Diamond &
   Hainmueller [ABADIE2010]_ and with the synthetic-control / SNN estimates
   elsewhere in ``mlsynth``. The jackknife confidence interval excludes zero.

   Regime robustness. Because MC-NNM regularizes rather than restricts, the
   same estimator runs unchanged on wide (``N >> T``), tall (``T >> N``) and
   square panels and on staggered adoption, where the unconfoundedness / SC /
   DiD special cases (paper Theorem 1) individually break down.

   Cross-validation. The Prop-99 ATT is matched to ``causaltensor``'s MC-NNM
   to ~2% and pinned in ``benchmarks/cases/mcnnm_prop99.py``; see the dedicated
   page :doc:`replications/mcnnm`.

Core API
--------

.. automodule:: mlsynth.estimators.mcnnm
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.MCNNMConfig
   :members:
   :undoc-members:

Result Containers
-----------------

``MCNNM.fit()`` returns a
:class:`~mlsynth.utils.mcnnm_helpers.structures.MCNNMResults`: the ATT, the
completed counterfactual matrix and per-cell effects, ``att_by_period``,
``cohort_att`` and ``event_study`` aggregations, the low-rank factors
(``unit_factors``, ``time_factors``, ``singular_values``, ``rank``), the fixed
effects, the CV-selected ``best_lambda``, implied (non-unique) donor weights,
and -- when requested -- the
:class:`~mlsynth.utils.mcnnm_helpers.structures.MCNNMInference` jackknife.

.. automodule:: mlsynth.utils.mcnnm_helpers.structures
   :members:
   :undoc-members:
   :show-inheritance:

Helper Modules
--------------

Data preparation -- the DataFrame touchpoint: pivots to the outcome matrix and
the observed/treated masks.

.. automodule:: mlsynth.utils.mcnnm_helpers.setup
   :members:
   :undoc-members:

The SOFT-IMPUTE completion solver and the cross-validation over :math:`\lambda`.

.. automodule:: mlsynth.utils.mcnnm_helpers.completion
   :members:
   :undoc-members:

Run loop: completion, ATT and staggered aggregations, factor decomposition, and
the jackknife.

.. automodule:: mlsynth.utils.mcnnm_helpers.pipeline
   :members:
   :undoc-members:
