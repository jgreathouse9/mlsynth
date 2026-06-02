Spillover-Detecting Synthetic Control (SPOTSYNTH)
=================================================

.. currentmodule:: mlsynth

Overview
--------

SPOTSYNTH packages the donor-selection procedure of O'Riordan &
Gilligan-Lee (2025), *Spillover detection for donor selection in
synthetic control models* ([SPOTSYNTH]_, Journal of Causal Inference
13:20240036). It addresses a prerequisite that classical synthetic
control takes for granted: that the donor pool is **valid** -- that no
donor is itself affected by the intervention through a spillover. When
the donor pool is large, deciding which donors are valid by domain
knowledge alone is infeasible, and a single contaminated donor -- one
that moves *with* the treated unit after the intervention -- can absorb
a large weight and bias the estimated effect toward zero.

The paper's main result (Theorem 3.1) is that, under the same
assumptions that make synthetic control non-parametrically identified
(invariant causal mechanisms and proxy completeness), a valid donor's
post-intervention value is **forecastable from pre-intervention donor
data**. SPOTSYNTH turns this into a practical screen: for each candidate
donor, fit a forecast on pre-intervention data, predict the donor's
first post-intervention value, and flag the donor if the realised value
departs from the forecast. A forecast failure means the donor was hit by
a spillover (or its latent distribution shifted) -- either way it is
excluded. The surviving donors feed the authors' **Bayesian Dirichlet
simplex** synthetic control (a :math:`\mathrm{Dirichlet}(0.4)` prior on
the weights, a half-normal prior on the residual scale, pre-period
standardisation, and 95% posterior-predictive credible intervals). The
donors the screen *excludes* are not discarded: they can be reused as
proximal control variables in a two-stage (GMM) step that debiases the
weights when the kept donors are noisy proxies.

Two selection rules are exposed through :py:attr:`SPOTSYNTHConfig.selection`:

* **S1** -- keep the donors with the smallest forecast error. The analyst
  fixes how many donors to keep (e.g. "give me 30 valid donors"), which
  is convenient when a downstream method needs a set number of donors.
* **S2** -- keep the donors whose realised post-intervention value falls
  inside a posterior predictive interval (default 80%). The analyst does
  not fix the number kept; instead the interval level controls the
  false-positive rate (how often a valid donor is wrongly excluded).

Mathematical Formulation
------------------------

Setup and notation
^^^^^^^^^^^^^^^^^^

We observe a single treated unit with outcome :math:`y^t` and a pool of
:math:`N` candidate donors with outcomes :math:`x_1^t, \dots, x_N^t`,
over periods :math:`t`. The intervention indicator is
:math:`I^t = \mathbb 1\{t \ge T_0\}` -- zero before the common adoption
time :math:`T_0`, one from :math:`T_0` onward. The estimand is the
treatment effect on the treated,

.. math::

   \tau = \underbrace{\mathbb E\bigl(y^t \mid \mathrm{do}(I^t = 1)\bigr)}_{\text{observed}}
        - \underbrace{\mathbb E\bigl(y^t \mid \mathrm{do}(I^t = 0)\bigr)}_{\text{counterfactual}},
   \qquad t \ge T_0,

estimated as the post-intervention gap between the treated unit and a
synthetic control built from **valid** donors. A donor is *valid* if it
adheres to the structural causal model of the paper (Figure 1a) and
remains a proxy for the latent factors at every time point -- in
particular it must **not** be impacted by the intervention. Spillover
effects manifest as a post-intervention shift in the donor's exogenous
error :math:`P(\varepsilon_{x_i}^t)`.

SC structural causal model
^^^^^^^^^^^^^^^^^^^^^^^^^^

Following Zeitler et al. (2023), the panel is modelled as a structural
causal model with latent variables :math:`u_1, \dots, u_M`, donors
:math:`x_i` as children of the latents, and the target :math:`y` as a
child of the latents and the intervention. Two conditions are central.

*Definition 3.2 (Proxy completeness).* For any square-integrable
:math:`f`, if :math:`\mathbb E(f(x_1^t, \dots, x_N^t) \mid u_1^t, \dots,
u_M^t) = 0` then :math:`f \equiv 0`. Intuitively, the donors carry all
the "information" the latents do -- they are genuine proxies for the
latent factors.

*Definition 3.3 (Invariant causal mechanism).* The deterministic
functions mapping parents to children do not depend on the time index
:math:`t`. This generalises the time-independent factor loadings of the
classical latent-factor SC model.

The forecast theorem
^^^^^^^^^^^^^^^^^^^^

**Theorem 3.1.** *If causal mechanisms are invariant, and the donors*
:math:`x_1^{t-1}, \dots, x_N^{t-1}` *are proxies for the latents*
:math:`u_1^{t-1}, \dots, u_M^{t-1}`, *then for each donor* :math:`x_i`
*there exists a unique function* :math:`h_i` *such that for all* :math:`t`

.. math::

   \mathbb E(x_i^t)
     = \mathbb E\bigl(h_i(x_1^{t-1}, \dots, x_N^{t-1}, P(\varepsilon_{x_i}^t))\bigr).

The intuition (Figure 1b): because the donors at :math:`t-1` are proxies
for the latents at :math:`t-1`, and the latents evolve by an invariant
mechanism, we can write :math:`x_i^t` as a function of the **lagged**
donor cross-section :math:`x_1^{t-1}, \dots, x_N^{t-1}` and the
donor's own noise. So a donor's value at :math:`t` is forecastable from
the donor pool one step earlier.

This is the key contrast with standard SC identifiability, which uses
*post*-intervention donor values to predict the *target's*
counterfactual. Theorem 3.1 instead lets us forecast each *donor's* own
post-intervention value from *pre*-intervention data, and that forecast
becomes a test for validity.

From theorem to screen
^^^^^^^^^^^^^^^^^^^^^^

A donor :math:`x_i` can be forecast from its past if (a) it is valid (not
impacted by the intervention) and (b) the latent error distributions
:math:`P(\varepsilon_u)` have not shifted at time :math:`t`. Conversely,
*failing* to forecast :math:`x_i^t` from pre-intervention data implies
(a), (b), or both are violated. Assuming the latents have not shifted (or
shift only later in the post-period, which does not bias the screen --
see below), a forecast failure flags a **spillover** -- an invalid donor.

**Algorithm 1 (per candidate donor** :math:`x_i` **).**

1. Normalise the donor data and labels (zero mean, unit variance over the
   pre-intervention window). The normalisation makes the procedure
   invariant to the scale of the donors.
2. Regress :math:`x_i^{t'}` on the lagged cross-section
   :math:`x_1^{t'-1}, \dots, x_N^{t'-1}` over the pre-intervention
   transitions :math:`t' < T_0` to obtain :math:`\hat h_i`. mlsynth fits a
   factor-regularised regression (the leading donor factors of the lagged
   cross-section) so the forecast is well posed even when :math:`N`
   exceeds the number of pre-intervention periods -- the regime of large
   donor pools the method is built for.
3. Predict the first post-intervention value :math:`\hat x_i^{T_0}` from
   the **last pre-intervention** cross-section (which is clean), with a
   :math:`\phi`-level posterior predictive interval
   :math:`[\hat x_{i,-}, \hat x_{i,+}]`.
4. Forecast error (procedure S1): :math:`A_i = |x_i^{T_0} - \hat x_i^{T_0}|`.
5. PPI flag (procedure S2): :math:`B_i = 0` if
   :math:`\hat x_{i,-} < x_i^{T_0} < \hat x_{i,+}`, else :math:`B_i = 1`.

The assumed forecast model (paper equation 3) is linear and
time-invariant,

.. math::

   x_i^{t'} \sim \mathcal N\Bigl(\rho_i + \textstyle\sum_k \theta_{ik}\,
     x_k^{t'-1},\ \sigma_{x_i}\Bigr),

with coefficients :math:`\rho_i, \theta_{ik}` shared across time --
encoding that :math:`h_i` is time-independent. The two selection rules are

.. math::

   S1:\ \min_{x_i}\ \Bigl|x_i^{T_0} - \rho_i - \textstyle\sum_k \theta_{ik} x_k^{T_0 - 1}\Bigr|,
   \qquad
   S2:\ x_i^{T_0} \in \text{the } \phi\text{-PPI}.

Why the lag matters (and a leave-one-out variant)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The forecast is anchored to the **last clean pre-intervention
cross-section**. This is what lets the screen work even when *most*
donors are invalid: at :math:`T_0` the lagged predictors are
pre-intervention (spillover-free), so the forecast predicts each donor's
untreated value, and an instantaneous spillover surfaces as a large
error. This is the default, ``forecast="lag"``.

mlsynth adds a second anchor, ``forecast="loo"``, for the case of a
*single* contaminating donor whose effect builds **gradually** (so the
gap at the very first post-period is near zero). It forecasts each
donor's whole post-intervention trajectory from the *other* donors'
common factors and ranks by the mean absolute deviation -- a
contaminated noisy-proxy donor, carrying the treated unit's slow-building
effect, departs from the valid majority over the post-period. Use
``"lag"`` for the many-invalid / sharp-effect regime (and the
simulation), ``"loo"`` for the single-contaminant / gradual-effect regime
(e.g. the German reunification demonstration below).

Time averaging
^^^^^^^^^^^^^^

Two practical issues are handled by forecasting on **time-averaged**
(coarsened) data, set via :py:attr:`SPOTSYNTHConfig.time_average`
(``"lag"`` only). First, a *lag* between the intervention and the onset
of a spillover: averaging over a window still surfaces a spillover that
arrives a few periods late. Second, very noisy donors: averaging reduces
the donor noise and so reduces false negatives (Figure 3). The averaging
must not mix pre- and post-intervention periods in the same bucket;
mlsynth buckets the two windows separately. Longer windows reduce noise
but raise the risk of false positives from latent shifts within the
window.

The synthetic-control model
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once the valid donors are selected, the counterfactual is built with the
authors' synthetic-control model (paper page 12). It is **Bayesian**:

.. math::

   y^t \sim \mathcal N\Bigl(\alpha + \textstyle\sum_i \beta_i x_i^t,\ \sigma_y\Bigr),
   \quad \beta_i \ge 0,\ \textstyle\sum_i \beta_i = 1,
   \quad \beta \sim \mathrm{Dirichlet}(0.4),
   \quad \sigma_y \sim \mathcal N^+(0, 1),

with the target and donors **standardised to zero mean and unit standard
deviation over the pre-intervention window** -- which is what absorbs the
intercept :math:`\alpha` of equation (4), so no separate intercept term is
fit. The :math:`\mathrm{Dirichlet}(0.4)` prior (concentration ``< 1``)
regularises the weights toward sparse corners of the simplex, retaining the
Abadie-Diamond-Hainmueller non-negativity / sum-to-one restriction. 95%
**credible intervals** for the counterfactual and the ATT are the 2.5 / 97.5
percentiles of the posterior predictive distribution.

This posterior has no closed form (a Dirichlet prior is not conjugate to a
Gaussian likelihood under the simplex constraint). mlsynth draws from it with a
self-contained, dependency-free Metropolis-Hastings sampler -- a Dirichlet
random-walk proposal on :math:`\beta` in the native simplex space (so no
change-of-variables Jacobian is needed) and a log-random-walk on
:math:`\sigma_y`, with both proposal scales adapted to a healthy acceptance
rate during warm-up. The posterior is low-dimensional (one weight per selected
donor) and smooth, so the sampler mixes well; the donor-collinearity that makes
individual weights weakly identified leaves the ATT/counterfactual functional
sharply identified. Set :py:attr:`SPOTSYNTHConfig.inference` to ``"frequentist"``
for a fast simplex least-squares point estimate (no intervals) when running
large simulations; the donor-*selection* bias pattern is identical either way.

Assumptions
-----------

The screen rests on the SC structural causal model and three working
assumptions layered on Theorem 3.1.

**A1 (Invariant causal mechanisms; Definition 3.3).** The structural
functions are time-invariant. This is what makes the forecast function
:math:`h_i` the *same* before and after the intervention, so a
pre-intervention forecast is valid post-intervention. *Diagnostic*: a
valid donor's pre-period one-step forecast residuals should look
stationary; strong heteroskedasticity or trending residual variance
signals a non-invariant mechanism.

**A2 (Proxy completeness; Definition 3.2).** The donors are proxies for
the latents. If a *relevant latent has no donor proxy*, excluding donors
cannot close all backdoor paths and the SC is biased by omitted
variables (Section 3.4.1). *Diagnostic*: a large pre-period fit residual
for the treated unit against the donor pool is a symptom that the donors
do not span the latents.

**A3 (No contemporaneous latent shift).** The latent error distributions
:math:`P(\varepsilon_u)` do not shift *at the same time* as the
intervention. The paper is explicit that latent shifts which occur
**later** in the post-period do **not** bias the screen (the forecast
test only inspects the first post-intervention point), and that lags can
be absorbed by time averaging. A contemporaneous latent shift, however,
is indistinguishable from a spillover and produces a **false positive**
(a valid donor wrongly excluded; Figure 5).

What does *not* break the screen: spillovers that arrive late, latent
shifts that arrive late, and large donor pools (the factor-regularised
forecast handles :math:`N >` pre-period length).

Bias when the screen errs: sensitivity analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The screen can make two kinds of error, and the paper bounds the SC bias
each induces (so the analyst can gauge robustness rather than trust the
selection blindly).

* **False positive** (a *valid* donor excluded). If the excluded donors
  were proxies for a relevant latent, dropping them reintroduces omitted-
  variable bias, bounded (Section 3.4.2) by

  .. math::

     \text{FP Bias} \le N \cdot \max_{x_i}(|\beta_{x_i}|)
        \cdot \max_{z_l}\bigl(|\mathbb E(z_l^{\text{pre}}) - \mathbb E(z_l^{\text{post}})|\bigr),

  where :math:`z_l` are the excluded donors and :math:`\beta` the SC
  weights. The bound is small when the *kept* donors already span the
  latents.

* **False negative** (an *invalid* donor kept). The bias from a retained
  spillover-:math:`\tau_{x_i}` donor is bounded (Section 3.4.3) by

  .. math::

     \text{FN Bias} \le N \cdot \max_{x_i}(|\beta_{x_i}|)
        \cdot \max_{x_i}(|\tau_{x_i}|).

  The spillover :math:`\tau_{x_i}` is unknown but, following the negative-
  control literature (Miao 2024), can be treated as a sensitivity
  parameter: domain knowledge bounding :math:`\tau_{x_i}` bounds the bias,
  and one can ask how large a spillover would have to be to flip the sign
  of the estimated effect.

Using excluded donors to debias (proximal two-stage)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The donors excluded by the screen are not used to *build* the SC, but
they can still **debias** it. Because only pre-intervention data is ever
used from the excluded donors :math:`z_l`, the SC estimate is unaffected
by their post-intervention spillover dynamics. Treating the excluded
donors as proximal control variables (Shi et al. 2023), one jointly
models the target and the kept donors as functions of the excluded donors
(paper equation 5) and recovers consistent weights even when the kept
donors are imperfect (noisy) proxies of the latents.

The paper notes (page 9) that equation (5) "effectively combines [a]
two-stage process into a single model", and that this is the standard
proximal / instrumental-variables estimator: regress the kept donors
:math:`X` on the excluded donors :math:`Z` to form :math:`\hat X`, then
regress the target on :math:`\hat X`. mlsynth implements exactly this
two-stage estimator **in closed form** (no probabilistic-programming
dependency), enabled by :py:attr:`SPOTSYNTHConfig.debias`. When ``True``,
the result object carries ``att_debiased`` alongside the screened ATT. On
the paper's Figure 4 (errors-in-variables) setting, this measurably
reduces the attenuation bias that persists even under a perfect
valid-donor selection.

When to use SPOTSYNTH (and when not to)
---------------------------------------

**Reach for SPOTSYNTH when:**

* You have a **large donor pool** and cannot certify by hand that every
  donor is free of spillovers. This is the motivating case -- e.g.
  estimating the effect of a feature launch on a platform where any of
  thousands of candidate "donor" markets might have been indirectly
  exposed.
* You are worried a donor is **too good a match** -- a unit that tracks
  the treated unit suspiciously closely after the intervention. Such a
  donor grabs a large SC weight and biases the effect toward zero; the
  screen is built to catch exactly this (the semi-synthetic
  demonstrations below).
* You want a **principled, data-driven** donor screen with explicit
  sensitivity bounds on the bias from selection errors, rather than an ad
  hoc "drop the weird-looking donor" rule.

**Use** ``forecast="lag"`` **(default) when** the donor pool may contain
*many* contaminated donors and/or the spillover is roughly contemporaneous
with the intervention. **Use** ``forecast="loo"`` **when** you suspect a
*single* contaminant whose influence builds gradually over the
post-period.

**Do not use SPOTSYNTH when:**

* **A relevant latent has no valid donor proxy** (A2 fails). No amount of
  donor selection closes the backdoor path; the FP-bias bound above is
  uninformative. Switch to a factor-model-aware estimator (:doc:`fma`) or
  a design that observes the confounder.
* **The treatment effect on the target is gradual *and* the donor pool is
  mostly contaminated.** The ``"lag"`` anchor needs a sharp first-period
  signal; the ``"loo"`` anchor needs a valid majority. With neither, the
  screen has no clean reference.
* **Causal mechanisms are non-invariant** (A1 fails) -- e.g. the
  latent-to-donor map changes over the sample. The pre-period forecast
  then does not transport to the post-period.
* **You only have a tiny, hand-curated donor pool** already known to be
  valid. The screen adds variance (it may drop a good donor) without
  identification gain; a canonical SC (:doc:`tssc`, :doc:`fdid`) is the
  more honest default.
* **Interference runs treated-to-treated or is structural across many
  units** rather than a few contaminated donors. For spillover-aware
  *estimands* (rather than donor cleaning) see :doc:`spsydid` and
  :doc:`spillsynth`.

Core API
--------

.. automodule:: mlsynth.estimators.spotsynth
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.SPOTSYNTHConfig
   :members:
   :undoc-members:

Helper Modules
--------------

.. automodule:: mlsynth.utils.spotsynth_helpers.screen
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.spotsynth_helpers.setup
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.spotsynth_helpers.sc
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.spotsynth_helpers.bayes
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.spotsynth_helpers.debias
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.spotsynth_helpers.pipeline
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.spotsynth_helpers.structures
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.spotsynth_helpers.simulation
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.spotsynth_helpers.replication
   :members:
   :undoc-members:

Example
-------

A self-contained one-draw run on the paper's data-generating process: a
treated unit, a pool of 60 donors of which 80% carry a :math:`-2`
spillover, the ``S1`` screen keeping the 12 most-forecastable donors, and
the authors' Bayesian Dirichlet SC (default) returning a 95% credible
interval. The proximal (GMM) debiased ATT is requested with
``debias=True``.

.. code-block:: python

   """One draw of the SPOTSYNTH spillover-detection DGP."""

   from mlsynth import SPOTSYNTH
   from mlsynth.utils.spotsynth_helpers import simulate_spillover_panel

   # Treated unit + 60 donors; 80% invalid (spillover -2); true tau = 2.
   df, valid_mask = simulate_spillover_panel(
       n_donors=60, T0=60, n_post=15, sigma_x=0.3, seed=3,
   )

   res = SPOTSYNTH({
       "df": df, "outcome": "Y", "treat": "treated",
       "unitid": "unit", "time": "time",
       "selection": "S1", "forecast": "lag", "n_donors": 12,
       "inference": "bayes",     # Dirichlet(0.4) Bayesian simplex SC (default)
       "debias": True,           # also report the proximal/GMM debiased ATT
       "display_graphs": False,
   }).fit()

   print(f"true tau            = +2.00")
   print(f"unscreened ATT      = {res.att_unscreened:+.2f}   (all 60 donors)")
   print(f"screened   ATT      = {res.att:+.2f}   ({res.metadata['n_selected']} valid donors)")
   lo, hi = res.att_ci
   print(f"95% credible ATT    = [{lo:+.2f}, {hi:+.2f}]")
   print(f"debiased   ATT      = {res.att_debiased:+.2f}")
   print(f"donors screened out = {res.metadata['n_excluded']}")
   # forecast errors: valid donors should score lower than invalid ones
   err = res.screen.forecast_error
   print(f"mean S1 error  valid={err[valid_mask].mean():.3f}  "
         f"invalid={err[~valid_mask].mean():.3f}")

Verification (Path B): the simulation study
-------------------------------------------

This reproduces the headline finding of the paper's Figure 2 **through
the public** ``SPOTSYNTH.fit()`` **call**. On the Appendix B
data-generating process, a synthetic control built on *all* donors is
biased upward (~+1.6) by the spillover-contaminated ones; one built on
the *valid* donors is unbiased; and the ``S1`` / ``S2`` screens recover
most of that gap, degrading as the donor noise grows toward the spillover
magnitude.

.. code-block:: python

   from mlsynth.utils.spotsynth_helpers import (
       run_spotsynth_simulation, SpotSimConfig,
   )

   # A compact configuration that runs in well under a minute.
   cfg = SpotSimConfig(
       n_donors=80, T0=60, n_post=15, n_keep=12, n_reps=12,
       noise_levels=(0.1, 0.5, 1.0),
   )
   bias = run_spotsynth_simulation(cfg, seed=0)

prints a table of the bias :math:`\mathbb E[\hat\tau] - \tau` like::

   SPOTSYNTH simulation (Figure 2), 12 reps, 80 donors, 80% invalid:
     noise      All    Valid       S1       S2
       0.1    +1.61    +0.00    +0.50    +1.32
       0.5    +1.60    -0.00    +0.48    +1.28
       1.0    +1.62    -0.00    +0.87    +1.10

reproducing the qualitative finding: **All** is badly biased, **Valid**
is unbiased, **S1** removes most of the bias and is best, **S2** is
intermediate, and the screens degrade as the noise rises toward the
spillover size. (The paper's full study uses 1000 donors and 2000 reps;
``SpotSimConfig`` defaults to that scale via the ``PAPER`` preset.)

Verification (semi-synthetic real data): Figure 6
-------------------------------------------------

The paper also demonstrates the screen on two canonical SC datasets by
planting a **semi-synthetic** invalid donor -- a noisy proxy of the
treated unit, :math:`x_{\text{syn}}^t \sim \mathcal N(y^t, \sigma)`.
Being a near-copy of the target, this invalid donor receives a large SC
weight and biases the effect **toward zero**; the screen flags and
excludes it, restoring the canonical effect. Both demos run through
``SPOTSYNTH.fit()``.

California tobacco control (Figure 6b). The 39-state Abadie panel plus the
planted donor; ``S1`` keeps 30 donors with the lagged forecast.

.. code-block:: python

   import pandas as pd
   from mlsynth import SPOTSYNTH

   url = ("https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
          "main/basedata/P99data.csv")
   df = pd.read_csv(url)[["state", "year", "cigsale"]]
   df["treated"] = ((df["state"] == "California") & (df["year"] >= 1989)).astype(int)

   # Plant a noisy proxy of California into the donor pool.
   import numpy as np
   ca = df[df["state"] == "California"].sort_values("year")
   syn = ca.copy()
   syn["state"] = "Synthetic California"
   syn["cigsale"] = ca["cigsale"].to_numpy() + np.random.default_rng(0).normal(0, 0.5, len(ca))
   syn["treated"] = 0
   df_contam = pd.concat([df, syn], ignore_index=True)

   res = SPOTSYNTH({
       "df": df_contam, "outcome": "cigsale", "treat": "treated",
       "unitid": "state", "time": "year",
       "selection": "S1", "forecast": "lag", "n_donors": 30,
       "inference": "bayes",     # Dirichlet(0.4) Bayesian simplex SC (default)
       "display_graphs": False,
   }).fit()

   print(f"contaminated (all donors) ATT = {res.att_unscreened:+.2f}")
   print(f"screened ATT                  = {res.att:+.2f}")
   lo, hi = res.att_ci
   print(f"95% credible interval         = [{lo:+.2f}, {hi:+.2f}]")
   print(f"planted donor excluded?       = "
         f"{'Synthetic California' in res.screen.excluded_names}")

prints::

   contaminated (all donors) ATT = -1.43
   screened ATT                  = -20.68
   95% credible interval         = [-23.28, -18.19]
   planted donor excluded?       = True

The contaminated pool gives a near-zero effect (the synthetic donor
hijacks the SC); the screen excludes it and the Bayesian Dirichlet SC
recovers the canonical :math:`\approx -20` packs-per-capita effect, with a
credible interval that brackets it. The one-call convenience
wrappers :func:`~mlsynth.utils.spotsynth_helpers.replicate_prop99_spillover`
and :func:`~mlsynth.utils.spotsynth_helpers.replicate_germany_spillover`
reproduce both Figure 6 panels (California, and German reunification with
``forecast="loo"`` since its effect builds gradually):

.. code-block:: python

   from mlsynth.utils.spotsynth_helpers import (
       replicate_prop99_spillover, replicate_germany_spillover,
   )
   replicate_prop99_spillover()    # California tobacco control (Figure 6b)
   replicate_germany_spillover()   # German reunification    (Figure 6a)

prints::

   Prop 99 (California): oracle ATT=-19.51  contaminated=-1.43  screened=-20.68  95% CrI=(-23.28, -18.19)  synthetic-donor excluded=True
   Reunification (Germany): oracle ATT=-1297.48  contaminated=-166.86  screened=-1431.27  95% CrI=(-1562.10, -1327.83)  synthetic-donor excluded=True

In both cases the planted invalid donor is flagged and excluded, and the
Bayesian Dirichlet SC restores the large effect the contamination had
masked, with a 95% posterior-predictive credible interval (the shaded
bands of the paper's Figure 6).
