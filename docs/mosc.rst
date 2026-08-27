Many-outcomes Synthetic Control (MOSC)
=======================================

.. currentmodule:: mlsynth

When to Use This Estimator
--------------------------

``MOSC`` implements the many-outcomes estimator of Wang, Schein, Shou and Blei
[MOSC]_. It answers a question the rest of the library mostly leaves alone: what
if the outcome is not the sort of thing a Gaussian model describes?

Almost every synthetic-control method fits an average, and fitting an average
well is a Gaussian assumption whether or not anyone writes it down. Match a
treated unit's pre-treatment path by least squares and you have assumed the
errors are symmetric, unbounded in both directions, and equally variable however
large the outcome is. For a count that is often false. Case counts, crime
incidents by block, insurance claims, conversions by market: these are
non-negative, they are integers, they cannot go below zero, and their spread
grows with their level. A method that assumes otherwise can hand back a
counterfactual with negative cases in it.

MOSC lets you choose the likelihood instead. It fits a probabilistic factor
model to the pre-intervention panel, takes each unit's estimated latent
loadings, and adjusts for those loadings the way one would adjust for an
observed confounder. For a count outcome the factor model can be gamma-Poisson;
for a real-valued one it can be probabilistic PCA. The rest of the procedure
does not change.

The justification is what makes this legitimate, and it is the paper's real
contribution. Synthetic control is normally defended by assuming the untreated
outcomes follow a *linear* factor model. MOSC drops linearity and argues from
negative control outcomes instead -- variables known in advance to be unaffected
by the intervention. In a panel almost everything is one: every pre-intervention
observation, and every observation on an untreated unit. If some per-unit latent
variable renders a unit's outcomes conditionally independent, and enough
negative controls are observed to pin that variable down, the effect is
identified with no linearity anywhere.

Reach for MOSC when the panel is wide (many units observed at a common
intervention date), the pre-period is long, and the outcome's distribution is
the thing your other options are getting wrong.

Do not use MOSC when
~~~~~~~~~~~~~~~~~~~~~

* Donor weights are the deliverable. MOSC has none at all. The counterfactual
  is a regression prediction from the treated unit's own latent loadings,
  borrowing strength across every unit at once, so there is no "California = 0.4
  Utah + 0.3 Montana" to report. If the weights are the result, use
  :doc:`vanillasc`, :doc:`tssc` or :doc:`fdid`.
* The donor pool is small. The outcome model fits one coefficient per latent
  factor across units, so a handful of donors leaves it interpolating. MOSC
  refuses a panel with fewer than ``n_factors + 3`` units. With ten donors and a
  Gaussian outcome, :doc:`gsynth` or :doc:`mcnnm` are the better tools.
* The pre-period is short. The identification is asymptotic in the number of
  negative control outcomes, which here means pre-treatment periods. It still
  runs on a short panel and the estimate is still defined; the argument behind
  it is simply weaker than the paper's asymptotics suggest.
* Adoption is staggered. MOSC takes one treated unit and one intervention date,
  matching the authors' own design. For staggered adoption use :doc:`mcnnm` or
  :doc:`ppscm`.
* A Gaussian likelihood is already appropriate and you want a posterior. Then
  :doc:`bfsc`, :doc:`mvbbsc` or :doc:`mtgp` do the same job with machinery built
  for that case.

Notation
--------

The outcome panel is the :math:`T \times N` matrix
:math:`\mathbf{Y} = (y_{ti})` over periods
:math:`t \in \mathcal{T} \coloneqq \{1, \ldots, T\}` and units
:math:`i \in \mathcal{N} \coloneqq \{1, \ldots, N\}`. The intervention occurs
after period :math:`T_0`, so :math:`t \leq T_0` indexes the pre-intervention
period and :math:`t > T_0` the post-intervention period. Unit :math:`1` is the
treated unit and :math:`A_i \in \{0, 1\}` records treatment status.

Write :math:`\overleftarrow{\mathbf{Y}}_i` for unit :math:`i`'s pre-intervention
outcomes and :math:`\overrightarrow{\mathbf{Y}}_i` for its post-intervention
outcomes. Potential outcomes are :math:`y_{ti}(0)` and :math:`y_{ti}(1)`, and the
estimand is the average effect on the treated unit over the post-intervention
period,

.. math::

   \mathrm{ATT} = \frac{1}{T - T_0} \sum_{t > T_0}
       \bigl[ y_{t1}(1) - y_{t1}(0) \bigr].

The latent per-unit variable the method adjusts for is :math:`\mathbf{U}_i`, and
:math:`\mathbf{Z}_i \in \mathbb{R}^{K}` is the estimate of it that the factor
model returns. The two carry different symbols on purpose: a factor model is
identified only up to an invertible transformation, so :math:`\mathbf{Z}_i` is
not :math:`\mathbf{U}_i` measured with error but a different coordinate system
for the same information.

The factor model itself has per-period factors
:math:`\boldsymbol{\theta}_t \in \mathbb{R}^{K}`, and :math:`K` is its rank.

Emancipators and negative controls
----------------------------------

The paper's argument turns on one definition. A latent variable
:math:`\mathbf{U}_i` *emancipates* unit :math:`i`'s outcomes when conditioning on
it makes them independent of one another:

.. math::

   P\bigl(\overleftarrow{\mathbf{Y}}_i, \overrightarrow{\mathbf{Y}}_i
   \mid \mathbf{U}_i\bigr)
   = \prod_{t \leq T_0} P(y_{ti} \mid \mathbf{U}_i)
     \prod_{t > T_0} P(y_{ti} \mid \mathbf{U}_i).

The word does the work of "this variable is everything the unit's outcomes have
in common". Anything shared across a unit's periods is inside
:math:`\mathbf{U}_i`; what is left is independent noise.

That has a consequence the paper draws out. A confounder that shows up in both
the pre- and post-intervention periods -- a *multi-period* confounder -- induces
dependence between outcomes on both sides of the intervention. If
:math:`\mathbf{U}_i` missed one, conditioning on :math:`\mathbf{U}_i` would not
have removed that dependence, contradicting the definition. So an emancipator
captures every multi-period confounder automatically. What it cannot capture is a
confounder that touches only one period, which is why those must be observed.

Assumptions and remarks
~~~~~~~~~~~~~~~~~~~~~~~~

*Assumption 1 (SUTVA).* No interference between units, and one version of the
treatment. *Remark.* Standard, and the reason spillover-contaminated donors break the
estimate. If neighbouring units respond to the intervention, use
:doc:`spillsynth` or screen donors with :doc:`spotsynth` first.

*Assumption 2 (a determinable-in-theory emancipator).* The outcomes follow some
factor model whose latent variable emancipates them and is pinned down by the
negative control outcomes. *Remark.* This replaces the linear factor model
that normally justifies synthetic control, and it is strictly weaker in the
functional form it allows and stronger in what it asks of the data. "Pinned
down" holds exactly only as the number of pre-intervention periods goes to
infinity, which the authors say plainly. On a panel of thirty periods it holds
approximately, and how well is not something the estimate reveals.

*Assumption 3 (weak unconfoundedness).* Treatment is independent of the
potential outcomes given :math:`\mathbf{U}_i` and any observed covariates.
*Remark.* In substance this is "no unobserved single-period confounders".
Anything that moved treatment and touched only the pre-period, or only the
post-period, has to be observed, because an emancipator cannot see it.

*Assumption 4 (the estimate is consistent).* The fitted
:math:`\mathbf{Z}_i` recovers :math:`\mathbf{U}_i` up to an invertible
transformation. *Remark.* This is where a misspecified factor model does its
damage. Nothing downstream can detect it, which is why the fit is scored and
the score reported.

*Assumption 5 (overlap).* Every value of the covariates and the emancipator occurs with
positive probability under both treatment states. *Remark.* Unusually mild
here. :math:`\mathbf{Z}_i` is estimated from pre-treatment outcomes alone and
never touches the treatment assignment, so estimating it does not erode the
overlap it needs.

How the estimate is computed
----------------------------

Three stages, following the paper's Algorithm 1.

First, fit a probabilistic factor model to the pre-intervention panel. Two are
available through ``factor_model``:

.. math::

   \text{gamma-Poisson:} \quad
   \theta_{tk} \sim \mathrm{Gamma}(a, b), \quad
   Z_{ki} \sim \mathrm{Gamma}(a, b), \quad
   y_{ti} \mid \mathbf{Z}_i, \boldsymbol{\theta}_t
   \sim \mathrm{Poisson}\bigl(\boldsymbol{\theta}_t^{\top} \mathbf{Z}_i\bigr),

.. math::

   \text{PPCA:} \quad
   y_{ti} \mid \mathbf{Z}_i, \boldsymbol{\theta}_t
   \sim \mathcal{N}\bigl(\boldsymbol{\theta}_t^{\top} \mathbf{Z}_i + \mu_t,
   \sigma^2\bigr).

The gamma-Poisson posterior is drawn by a conjugate Gibbs sampler. The
augmentation that makes it conjugate splits each observed count across the
:math:`K` components with a multinomial draw, after which both factor matrices
are Gamma again. That is why MOSC needs no sampler dependency: no NumPyro, no
``[bayes]`` extra, and a fit measured in seconds. PPCA is fit by the classical
Tipping-Bishop EM.

Second, for each posterior draw :math:`s`, regress every unit's post-intervention
outcome vector on that draw's loadings and a treatment indicator,

.. math::

   \mathbb{E}\bigl[\overrightarrow{\mathbf{Y}}_i \mid A_i = a, \mathbf{Z}_i\bigr]
   = \beta_0 + \beta_A a + \boldsymbol{\beta}_Z^{\top} \mathbf{Z}_i^{(s)},

by cross-validated ridge, and read off the treated unit's fitted row with the
indicator set to zero. That prediction is the counterfactual under draw
:math:`s`. Each draw gets its own regression, not one regression on averaged
loadings, because a factor model is identified only up to relabelling and the
average of the draws need not be a configuration any draw took.

Third, the effect is the observed path minus the counterfactual,

.. math::

   \widehat{\mathrm{ATT}}^{(s)} = \frac{1}{T - T_0}
       \sum_{t > T_0} \bigl[ y_{t1} - \widehat{f}^{(s)}_{t1}(0) \bigr],

and the spread of :math:`\widehat{\mathrm{ATT}}^{(s)}` across draws is the
credible interval.

Inference and diagnostics
-------------------------

Uncertainty is the paper's Section 3.4 procedure: a nonparametric bootstrap over
units. Each replicate draws a donor pool with replacement, re-runs the algorithm
on it, and contributes one counterfactual; the interval is the pointwise
percentile range across replicates at ``ci_alpha``. This is what Theorem 4 asks
for, since the g-formula's outer expectation is over the distribution of
loadings among the treated and its sampling uncertainty comes from having
observed these units and not others. The treated unit is held in every replicate:
read literally the paper resamples all units, which can drop the one whose
counterfactual is the estimand.

``inference="posterior"`` returns instead the spread of the factor model's own
draws. That is what the paper's Figures 4 and 5 plot, and it is a band on the
counterfactual's conditional mean -- it moves only with uncertainty about
:math:`\mathbf{Z}_i` and conditions on the units that happened to be observed,
so it is systematically narrower.

The difference is not academic. The paper says it will evaluate the coverage of
its bootstrap in Section 5; that evaluation does not appear, and the word
"bootstrap" occurs three times in the paper, twice in the paragraph that
prescribes it and once in the bibliography. Running the check on the authors'
own control teams -- the twelve whose stadiums never admitted fans, where the
effect is zero by construction -- gives the following coverage of a nominal
95 percent interval over ten panels:

.. list-table::
   :header-rows: 1
   :widths: 60 20

   * - interval
     - covers zero
   * - posterior band (what the paper's figures show)
     - 4 / 10
   * - unit bootstrap, percentile (the default here)
     - 9 / 10

Two things follow for a reader. The posterior band should not be read as a
confidence interval; it is not one, and on these panels it excludes zero six
times out of ten where nothing happened. And even the bootstrap does not reach
nominal. Its one failure, Minnesota, has a point estimate that misses by 21
percent of the outcome, which is a counterfactual that is wrong and not an
interval that is narrow. No inference procedure repairs that, and it is the
reason the placebo check belongs in any applied use of this estimator: run the
method on a unit you know was untreated, and see whether it finds an effect.

Three diagnostics arrive as typed fields on
:class:`~mlsynth.utils.mosc_helpers.structures.MOSCDiagnostics`, because each is
something a reader might act on.

``heldout_log_density`` scores the factor model on cells withheld from the fit.
Higher is better, and it is a score with no size guarantee attached: it compares
two candidate models on the same panel and says nothing about whether either is
adequate in absolute terms. Use it to choose ``factor_model`` and ``n_factors``.

``residual_autocorrelation`` is the diagnostic that decides ``outcome_scale``.
Assumption 2 requires the latent factors to render a unit's outcomes
conditionally independent, so what has to be near zero is the correlation
remaining *after* conditioning on the fitted factors. A cumulative series fails
this badly and its first difference does not. If this reads far from zero, set
``outcome_scale="difference"``: the factor model is then fit to first
differences, and the counterfactual is re-integrated so it still comes back on
the outcome's own scale.

``pearson_dispersion`` is 1 under a well-specified Poisson model. It moves in
both directions -- an overdispersed panel pushes it above 1, and a smooth series
that a rank-:math:`K` model fits almost exactly pushes it below. Read a large
departure in either direction as the count assumption doing badly.

Example
-------

.. code-block:: python

   import numpy as np
   import pandas as pd
   from mlsynth import MOSC

   rng = np.random.default_rng(0)
   n_units, n_periods, pre = 24, 40, 28
   factors = rng.gamma(3.0, 3.0, size=(n_periods, 3))
   loadings = rng.gamma(3.0, 3.0, size=(3, n_units))
   rate = factors @ loadings
   rate[pre:, 0] *= 1.4                       # the treated unit's effect
   counts = rng.poisson(rate)

   panel = pd.DataFrame([
       {"county": f"c{i:02d}", "day": t, "cases": float(counts[t, i]),
        "reopened": int(i == 0 and t >= pre)}
       for i in range(n_units) for t in range(n_periods)
   ])

   result = MOSC({
       "df": panel, "outcome": "cases", "treat": "reopened",
       "unitid": "county", "time": "day",
       "factor_model": "gap", "n_factors": 3, "seed": 0,
   }).fit()

   print(result.att, result.att_ci)
   print(result.diagnostics.residual_autocorrelation)
   print(result.donor_weights)          # {} -- MOSC has none, and says so

The empty ``donor_weights`` is a statement, not an omission. Every estimator in
this library populates the same weights container, and ``{}`` records that this
one was asked and has no donor weights to give, which ``None`` would leave
ambiguous.

To see the posterior band, ask the plotter for its figure and display it
yourself:

.. code-block:: python

   from mlsynth.utils.mosc_helpers import plot_mosc_posterior

   figure = plot_mosc_posterior(result)
   figure.savefig("mosc.png", dpi=150)

Verification
------------

MOSC was assessed by a demonstrate-first spike before it was built, recorded in
`benchmarks/reference/mosc_spike/
<https://github.com/jgreathouse9/mlsynth/tree/main/benchmarks/reference/mosc_spike>`_
and in ``agents/future_integrations.md`` §21. The spike ported the authors' own
code, reproduced their semi-synthetic study across 48 cells, and compared the
result against :doc:`clustersc`'s robust synthetic control on identical panels.

The paper's substantive claim reproduces: the gamma-Poisson arm beats both
Gaussian arms on mean relative error, at 25 pre-intervention periods as well as
100. Its margin over robust synthetic control is narrower than the paper's own
figure suggests -- the Poisson arm wins 29 of 48 cells -- and the advantage
concentrates where the factor model is violated, not where it holds.

Three deviations from the paper are deliberate, each established by the spike.

The effect takes the sign of the paper's own equation, observed minus
counterfactual. The authors' code computes the reverse; on their null result the
difference is invisible, and on any real effect it returns the wrong sign.

The paper's ``p_pop`` model check is not offered. Its stated false rejection rate
is 0.05; measured on data drawn from the very model being checked, it is 0.40.
The statistic sums a discrepancy over held-out cells, so its systematic part
grows with the cell count while its spread grows with the square root, and past
roughly a hundred cells the verdict stops depending on the data.
``heldout_log_density`` is the same comparison reported as a score, which makes
no calibration claim and so cannot make a false one.

The lagged pre-intervention outcome that the authors' code adds to every
regression behind their published figure is absent here. It appears nowhere in
their equations, and the baseline it is compared against gets no equivalent
term.

.. [MOSC] Wang, Y., Schein, A., Shou, J., & Blei, D. M.
   *A Many-outcomes Perspective on the Synthetic Control Method.*

Core API
--------

.. autoclass:: MOSC
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.utils.mosc_helpers.config.MOSCConfig
   :members:
   :undoc-members:
   :show-inheritance:

Results
-------

:class:`~mlsynth.utils.mosc_helpers.structures.MOSCResults`: the ATT and its
credible interval, the posterior-mean counterfactual with a band, the draws of
the estimated confounding structure, and the diagnostics above.

.. automodule:: mlsynth.utils.mosc_helpers.structures
   :members:
   :undoc-members:
   :show-inheritance:

Helper Modules
--------------

Data preparation -- the DataFrame touchpoint. Pivots to the outcome matrix and
refuses a panel the method cannot identify from.

.. automodule:: mlsynth.utils.mosc_helpers.setup
   :members:
   :undoc-members:

The two probabilistic factor models and the score that compares them.

.. automodule:: mlsynth.utils.mosc_helpers.factor
   :members:
   :undoc-members:

Run loop: factor model, per-draw outcome regression, diagnostics, and the
re-integration that returns a differenced fit to the outcome's own scale.

.. automodule:: mlsynth.utils.mosc_helpers.pipeline
   :members:
   :undoc-members:

The posterior-band figure. Builds and returns it; displaying and saving are the
caller's.

.. automodule:: mlsynth.utils.mosc_helpers.plotter
   :members:
   :undoc-members:
