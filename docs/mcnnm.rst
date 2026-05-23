Matrix Completion with Nuclear Norm Minimization (MCNNM)
=========================================================

.. currentmodule:: mlsynth

Overview
--------

MC-NNM (Athey, S., Bayati, M., Doudchenko, N., Imbens, G. & Khosravi, K.
(2021). *"Matrix Completion Methods for Causal Panel Data Models,"*
Journal of the American Statistical Association 116(536):1716-1730)
estimates causal effects in panel data by treating the treated
unit/period cells as **missing entries** of the outcome matrix and
imputing them with low-rank matrix completion.

Where synthetic control reweights *control units* and unconfoundedness
reweights *time periods*, MC-NNM exploits **both** the cross-sectional
and time-series structure at once through a low-rank factor model,
regularised by the nuclear norm. The paper's central theoretical result
(Theorem 1) is that the unconfoundedness, synthetic-control, and
difference-in-differences estimators all minimise the *same* objective
and differ only in the restrictions they impose; MC-NNM replaces those
hard restrictions with regularisation, which lets it work across regimes
(``N >> T``, ``T >> N``, ``N ~ T``) where the others individually break
down.

The estimator
^^^^^^^^^^^^^

Model the untreated-outcome matrix as a low-rank component plus two-way
fixed effects, :math:`Y = L^* + \Gamma 1_T^\top + 1_N \Delta^\top +
\varepsilon`, and solve (paper eq. 4.3)

.. math::

   (\widehat L, \widehat\Gamma, \widehat\Delta)
     = \arg\min_{L, \Gamma, \Delta}
       \frac{1}{|\mathcal{O}|}
       \bigl\| P_\mathcal{O}(Y - L - \Gamma 1_T^\top
                             - 1_N \Delta^\top) \bigr\|_F^2
       + \lambda \|L\|_*,

where :math:`\mathcal{O}` is the set of observed (untreated) cells,
:math:`P_\mathcal{O}` projects onto them, and :math:`\|L\|_* =
\sum_i \sigma_i(L)` is the nuclear norm. Only the low-rank part
:math:`L` is regularised; the unit/time fixed effects are estimated
explicitly and left unregularised, which substantially improves the
imputations (paper Section 4).

Algorithm (SOFT-IMPUTE)
^^^^^^^^^^^^^^^^^^^^^^^

The problem is solved by singular-value soft-thresholding (paper
eq. 4.4-4.5). With the shrink operator
:math:`\mathrm{shrink}_\lambda(A) = S \widetilde\Sigma R^\top` (each
singular value replaced by :math:`\max(\sigma - \lambda, 0)`), iterate

.. math::

   L_{k+1} = \mathrm{shrink}_{\lambda|\mathcal{O}|/2}
     \bigl\{ P_\mathcal{O}(Y - \widehat{\mathrm{FE}})
             + P_\mathcal{O}^\perp(L_k) \bigr\},

re-fitting the fixed effects from their first-order conditions after each
step, until convergence. The regularisation strength :math:`\lambda` is
chosen by **cross-validation** over the observed cells.

Causal use in mlsynth
^^^^^^^^^^^^^^^^^^^^^^^

The :class:`MCNNM` estimator marks the treated post-treatment cells as
missing, imputes their untreated potential outcomes by MC-NNM, and forms
the treatment effect as observed minus imputed:
:math:`\widehat\tau_{it} = Y_{it} - \widehat Y_{it}(0)`, aggregated to the
ATT over the treated cells.

Staggered adoption
^^^^^^^^^^^^^^^^^^

MC-NNM handles **staggered adoption** natively -- units adopting at
different (irreversible) times produce a "staircase" of missing cells,
which the mask-based imputation fills directly (paper Section 3.1.2; this
is MC-NNM's main advantage over fixed-rank interactive-fixed-effects
methods). Adoption times are detected with
:func:`mlsynth.utils.datautils.dataprep`, and the result exposes two
staggered-aware aggregations beyond the overall ATT:

* ``cohort_att`` -- ``{adoption_time: ATT}`` for each adoption cohort.
* ``event_study`` -- ``{relative_time: average effect}``, with effects
  re-centred on each unit's own adoption date (negative keys are
  pre-adoption placebo / fit-quality checks, ~0; non-negative keys are the
  dynamic treatment effects).

With multiple adoption times, ``display_graphs=True`` draws an
**event-study** plot (effect vs time relative to adoption) rather than the
single-treatment-line trajectory, so cohorts at different event times are
not blended on a calendar axis.

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

Helper Modules
--------------

.. automodule:: mlsynth.utils.mcnnm_helpers.completion
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.mcnnm_helpers.setup
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.mcnnm_helpers.pipeline
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.mcnnm_helpers.structures
   :members:
   :undoc-members:

Example
-------

Proposition 99 -- California's 1988 tobacco-control program. MC-NNM
imputes California's post-1988 counterfactual per-capita cigarette sales
by matrix completion and reports the ATT; with ``display_graphs=True`` it
draws the observed-vs-counterfactual chart.

.. code-block:: python

   import pandas as pd

   from mlsynth import MCNNM

   file = (
       "https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
       "refs/heads/main/basedata/smoking_data.csv"
   )
   df = pd.read_csv(file)

   res = MCNNM({
       "df": df,
       "outcome": "cigsale",
       "treat": "Proposition 99",      # boolean treatment column
       "unitid": "state",
       "time": "year",
       "inference": True,              # leave-one-control jackknife
       "display_graphs": True,         # observed vs MC-NNM counterfactual
   }).fit()

   print(f"ATT (avg 1989-2000) = {res.att:+.2f} packs/capita")
   lo, hi = res.inference.ci
   print(f"jackknife 95% CI    = [{lo:+.2f}, {hi:+.2f}]")
   print(f"gap by 2000         = {res.att_by_period[2000]:+.2f}")
   print(f"selected lambda     = {res.best_lambda:.2f}")

MC-NNM returns an average ATT of about ``-20`` packs/capita, widening to
roughly ``-30`` by 2000, with a near-exact California pre-treatment fit --
consistent with Abadie, Diamond & Hainmueller (2010) and with the SNN /
synthetic-control estimates in mlsynth.

References
----------

Abadie, A., Diamond, A., & Hainmueller, J. (2010). "Synthetic Control
Methods for Comparative Case Studies." *Journal of the American
Statistical Association* 105(490):493-505.

Athey, S., Bayati, M., Doudchenko, N., Imbens, G., & Khosravi, K. (2021).
"Matrix Completion Methods for Causal Panel Data Models." *Journal of the
American Statistical Association* 116(536):1716-1730.

Mazumder, R., Hastie, T., & Tibshirani, R. (2010). "Spectral
Regularization Algorithms for Learning Large Incomplete Matrices."
*Journal of Machine Learning Research* 11:2287-2322.

Xu, Y. (2017). "Generalized Synthetic Control Method: Causal Inference
with Interactive Fixed Effects Models." *Political Analysis* 25(1):57-76.
