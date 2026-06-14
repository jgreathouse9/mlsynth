Spillover-Detecting Synthetic Control (SPOTSYNTH)
=================================================

.. currentmodule:: mlsynth

Overview
--------

SPOTSYNTH packages the donor-selection procedure of O'Riordan &
Gilligan-Lee (2025), *Spillover detection for donor selection in
synthetic control models* ([SPOTSYNTH]_, Journal of Causal Inference
13:20240036). It addresses a prerequisite that classical synthetic
control takes for granted: that the donor pool is valid -- that no
donor is itself affected by the intervention through a spillover. When
the donor pool is large, deciding which donors are valid by domain
knowledge alone is infeasible, and a single contaminated donor -- one
that moves *with* the treated unit after the intervention -- can absorb
a large weight and bias the estimated effect toward zero.

The paper's main result (Theorem 3.1) is that, under the same
assumptions that make synthetic control non-parametrically identified
(invariant causal mechanisms and proxy completeness), a valid donor's
post-intervention value is forecastable from pre-intervention donor
data. SPOTSYNTH turns this into a practical screen: for each candidate
donor, forecast its untreated post-intervention path from donor data the
intervention has not touched, and flag the donor if its realised path
departs from the forecast. A forecast failure means the donor was hit by
a spillover (or its latent distribution shifted) -- either way it is
excluded. (Two forecast anchors implement this -- a leave-one-out anchor,
the default, and the paper's first-post-point anchor; the
:ref:`forecast-anchor section <spotsynth-anchors>` explains when to use each.) The surviving donors feed the authors' Bayesian Dirichlet
simplex synthetic control (a :math:`\mathrm{Dirichlet}(0.4)` prior on
the weights, a half-normal prior on the residual scale, pre-period
standardisation, and 95% posterior-predictive credible intervals). The
donors the screen *excludes* are not discarded: they can be reused as
proximal control variables in a two-stage (GMM) step that debiases the
weights when the kept donors are noisy proxies.

Two selection rules are exposed through :py:attr:`SPOTSYNTHConfig.selection`:

* S1 -- keep the donors with the smallest forecast error. The analyst
  fixes how many donors to keep (e.g. "give me 30 valid donors"), which
  is convenient when a downstream method needs a set number of donors.
* S2 -- keep the donors whose realised post-intervention value falls
  inside a posterior predictive interval (default 80%). The analyst does
  not fix the number kept; instead the interval level controls the
  false-positive rate (how often a valid donor is wrongly excluded).

When to use this estimator
--------------------------

Reach for SPOTSYNTH when:

* You have a large donor pool and cannot certify by hand that every
  donor is free of spillovers. This is the motivating case -- e.g.
  estimating the effect of a feature launch on a platform where any of
  thousands of candidate "donor" markets might have been indirectly
  exposed.
* You are worried a donor is too good a match -- a unit that tracks
  the treated unit suspiciously closely after the intervention. Such a
  donor grabs a large SC weight and biases the effect toward zero; the
  screen is built to catch exactly this (the semi-synthetic
  demonstrations below).
* You want a principled, data-driven donor screen with explicit
  sensitivity bounds on the bias from selection errors, rather than an ad
  hoc "drop the weird-looking donor" rule.

Use the default ``forecast="loo"`` for applied work -- a mostly-valid
donor pool with a contaminant whose effect may arrive at any speed (it is
onset-robust and dominates this regime). Switch to ``forecast="lag"``
only when you have prior reason to believe a *large fraction* of the pool is
contaminated *and* the spillover is abrupt, the one regime where ``loo`` inverts
(see the forecast-anchor power analysis below).

Do not use SPOTSYNTH when:

* A relevant latent has no valid donor proxy (A2 fails). No amount of
  donor selection closes the backdoor path; the FP-bias bound below is
  uninformative. Switch to a factor-model-aware estimator (:doc:`fma`) or
  a design that observes the confounder.
* The treatment effect on the target is gradual *and* the donor pool is
  mostly contaminated. The ``"lag"`` anchor needs a sharp first-period
  signal; the ``"loo"`` anchor needs a valid majority. With neither, the
  screen has no clean reference.
* Causal mechanisms are non-invariant (A1 fails) -- e.g. the
  latent-to-donor map changes over the sample. The pre-period forecast
  then does not transport to the post-period.
* You only have a tiny, hand-curated donor pool already known to be
  valid. The screen adds variance (it may drop a good donor) without
  identification gain; a canonical SC (:doc:`tssc`, :doc:`fdid`) is the
  more honest default.
* Interference runs treated-to-treated or is structural across many
  units rather than a few contaminated donors. For spillover-aware
  *estimands* (rather than donor cleaning) see :doc:`spsydid` and
  :doc:`spillsynth`.

Notation
--------

Let :math:`j = 1` denote the treated unit, with all units
:math:`\mathcal{N} \coloneqq \{1, \dots, N\}` and donor pool
:math:`\mathcal{N}_0 \coloneqq \mathcal{N} \setminus \{1\}` of cardinality
:math:`N_0`. Time runs over :math:`t \in \mathcal{T} \coloneqq \{1, \dots, T\}`,
1-indexed; the intervention takes effect after the common adoption time
:math:`T_0`, splitting :math:`\mathcal{T}` into the pre-period
:math:`\mathcal{T}_1 \coloneqq \{t \in \mathcal{T} : t \le T_0\}` (of length
:math:`T_0`) and the post-period
:math:`\mathcal{T}_2 \coloneqq \{t \in \mathcal{T} : t > T_0\}`. The
intervention indicator is :math:`d_t \coloneqq \mathbf{1}\{t > T_0\}` -- zero on
:math:`\mathcal{T}_1`, one on :math:`\mathcal{T}_2`.

The treated series is :math:`\mathbf{y}_1 = (y_{11}, \dots, y_{1T})^\top \in
\mathbb{R}^{T}` with scalar outcomes :math:`y_{1t}`; each donor
:math:`j \in \mathcal{N}_0` contributes a series :math:`\mathbf{y}_j`, stacked
into the donor matrix :math:`\mathbf{Y}_0 \coloneqq [\mathbf{y}_j]_{j \in
\mathcal{N}_0} \in \mathbb{R}^{T \times N_0}` (one column per donor). The panel
is generated by latent factors :math:`u_1, \dots, u_M` for which the donors are
proxies. Donor weights are :math:`\mathbf{w} \in \mathbb{R}^{N_0}`, constrained
to the unit simplex
:math:`\Delta^{N_0} \coloneqq \{\mathbf{w} \in \mathbb{R}_{\ge 0}^{N_0} :
\|\mathbf{w}\|_1 = 1\}`; the optimiser is :math:`\mathbf{w}^\ast`. The synthetic
counterfactual is
:math:`\widehat{\mathbf{y}}_1 \coloneqq \mathbf{Y}_0\,\mathbf{w}^\ast` with
entries :math:`\widehat{y}_{1t}`, the per-period effect is
:math:`\tau_t \coloneqq y_{1t} - \widehat{y}_{1t}`, and the ATT is
:math:`\widehat{\tau} \coloneqq |\mathcal{T}_2|^{-1}
\sum_{t \in \mathcal{T}_2} \tau_t`. A donor's spillover magnitude -- the size of
the post-intervention shift its outcome suffers when it is invalid -- is written
:math:`\gamma_j`, kept distinct from the treatment effect :math:`\tau`.

Assumptions
-----------

The screen rests on the SC structural causal model and three working
assumptions layered on Theorem 3.1.

A1 (Invariant causal mechanisms; Definition 3.3). The structural
functions are time-invariant. This is what makes the forecast function
:math:`h_j` the *same* before and after the intervention, so a
pre-intervention forecast is valid post-intervention.

   *Remark.* This is what licenses transporting the pre-period forecast across
   :math:`T_0`: only if the same mechanism governs both windows does a forecast
   trained on :math:`\mathcal{T}_1` predict a valid donor's untreated value on
   :math:`\mathcal{T}_2`. A valid donor's pre-period one-step forecast residuals
   should therefore look stationary; strong heteroskedasticity or trending
   residual variance signals a non-invariant mechanism.

A2 (Proxy completeness; Definition 3.2). The donors are proxies for
the latents. If a *relevant latent has no donor proxy*, excluding donors
cannot close all backdoor paths and the SC is biased by omitted
variables (Section 3.4.1).

   *Remark.* Donor selection cleans contaminated proxies; it cannot manufacture a
   proxy for a latent the donor pool never spanned. When this fails the
   FP-bias bound below is uninformative, which is exactly the regime in which
   the page steers the reader to a factor-model-aware estimator (:doc:`fma`). A
   large pre-period fit residual for the treated unit against the donor pool is a
   symptom that the donors do not span the latents.

A3 (No contemporaneous latent shift). The latent error distributions
:math:`P(\varepsilon_u)` do not shift *at the same time* as the
intervention.

   *Remark.* The paper is explicit that latent shifts which occur later in the
   post-period do not bias the screen (the forecast test only inspects the first
   post-intervention point), and that lags can be absorbed by time averaging. A
   contemporaneous latent shift, however, is indistinguishable from a spillover
   and produces a false positive (a valid donor wrongly excluded; Figure 5).

What does *not* break the screen: spillovers that arrive late, latent
shifts that arrive late, and large donor pools (the factor-regularised
forecast handles :math:`N >` pre-period length).

Mathematical Formulation
------------------------

The estimand
^^^^^^^^^^^^

We observe the treated unit :math:`j = 1` with outcome :math:`y_{1t}` and the
donor pool :math:`\mathcal{N}_0` with outcomes :math:`y_{jt}`, over periods
:math:`t \in \mathcal{T}`. The intervention indicator is
:math:`d_t = \mathbf{1}\{t > T_0\}` -- zero on the pre-period
:math:`\mathcal{T}_1`, one on the post-period :math:`\mathcal{T}_2`. The
estimand is the treatment effect on the treated,

.. math::

   \tau_t = \underbrace{\mathbb E\bigl(y_{1t} \mid \mathrm{do}(d_t = 1)\bigr)}_{\text{observed}}
        - \underbrace{\mathbb E\bigl(y_{1t} \mid \mathrm{do}(d_t = 0)\bigr)}_{\text{counterfactual}},
   \qquad t \in \mathcal{T}_2,

estimated as the post-intervention gap between the treated unit and a
synthetic control built from valid donors. A donor is *valid* if it
adheres to the structural causal model of the paper (Figure 1a) and
remains a proxy for the latent factors at every time point -- in
particular it must not be impacted by the intervention. Spillover
effects manifest as a post-intervention shift in the donor's exogenous
error :math:`P(\varepsilon_{jt})`.

SC structural causal model
^^^^^^^^^^^^^^^^^^^^^^^^^^

Following Zeitler et al. (2023), the panel is modelled as a structural
causal model with latent variables :math:`u_1, \dots, u_M`, donors
:math:`\mathbf{y}_j` (:math:`j \in \mathcal{N}_0`) as children of the latents,
and the treated unit :math:`\mathbf{y}_1` as a child of the latents and the
intervention. Two conditions are central.

*Definition 3.2 (Proxy completeness).* For any square-integrable
:math:`f`, if :math:`\mathbb E(f(\mathbf{Y}_{0,t}) \mid u_{1t}, \dots,
u_{Mt}) = 0` then :math:`f \equiv 0`, where :math:`\mathbf{Y}_{0,t}` is the
donor cross-section :math:`(y_{jt})_{j \in \mathcal{N}_0}` at time :math:`t`.
Intuitively, the donors carry all the "information" the latents do -- they are
genuine proxies for the latent factors.

*Definition 3.3 (Invariant causal mechanism).* The deterministic
functions mapping parents to children do not depend on the time index
:math:`t`. This generalises the time-independent factor loadings of the
classical latent-factor SC model.

The forecast theorem
^^^^^^^^^^^^^^^^^^^^

Theorem 3.1. *If causal mechanisms are invariant, and the donor cross-section*
:math:`\mathbf{Y}_{0,t-1}` *is a proxy for the latents*
:math:`u_{1,t-1}, \dots, u_{M,t-1}`, *then for each donor* :math:`j`
*there exists a unique function* :math:`h_j` *such that for all* :math:`t`

.. math::

   \mathbb E(y_{jt})
     = \mathbb E\bigl(h_j(\mathbf{Y}_{0,t-1}, P(\varepsilon_{jt}))\bigr).

The intuition (Figure 1b): because the donors at :math:`t-1` are proxies
for the latents at :math:`t-1`, and the latents evolve by an invariant
mechanism, we can write :math:`y_{jt}` as a function of the lagged
donor cross-section :math:`\mathbf{Y}_{0,t-1}` and the
donor's own noise. So a donor's value at :math:`t` is forecastable from
the donor pool one step earlier.

This is the key contrast with standard SC identifiability, which uses
*post*-intervention donor values to predict the *target's*
counterfactual. Theorem 3.1 instead lets us forecast each *donor's* own
post-intervention value from *pre*-intervention data, and that forecast
becomes a test for validity.

From theorem to screen
^^^^^^^^^^^^^^^^^^^^^^

A donor :math:`j` can be forecast from its past if (a) it is valid (not
impacted by the intervention) and (b) the latent error distributions
:math:`P(\varepsilon_u)` have not shifted at time :math:`t`. Conversely,
*failing* to forecast :math:`y_{jt}` from pre-intervention data implies
(a), (b), or both are violated. Assuming the latents have not shifted (or
shift only later in the post-period, which does not bias the screen --
see below), a forecast failure flags a spillover -- an invalid donor.

Algorithm 1 (per candidate donor :math:`j` ).

1. Normalise the donor data and labels (zero mean, unit variance over the
   pre-intervention window). The normalisation makes the procedure
   invariant to the scale of the donors.
2. Regress :math:`y_{jt}` on the lagged cross-section
   :math:`y_{1,t-1}, \dots, y_{N,t-1}` over the pre-intervention
   transitions :math:`t \le T_0` to obtain :math:`\widehat{h}_j`. mlsynth fits a
   factor-regularised regression (the leading donor factors of the lagged
   cross-section) so the forecast is well posed even when :math:`N`
   exceeds the number of pre-intervention periods -- the regime of large
   donor pools the method is built for.
3. Predict the first post-intervention value :math:`\widehat{y}_{j,T_0+1}` from
   the last pre-intervention cross-section (which is clean), with a
   :math:`\phi`-level posterior predictive interval
   :math:`[\widehat{y}_{j,-}, \widehat{y}_{j,+}]`.
4. Forecast error (procedure S1):
   :math:`A_j \coloneqq |y_{j,T_0+1} - \widehat{y}_{j,T_0+1}|`.
5. PPI flag (procedure S2): :math:`B_j = 0` if
   :math:`\widehat{y}_{j,-} < y_{j,T_0+1} < \widehat{y}_{j,+}`, else :math:`B_j = 1`.

The assumed forecast model (paper equation 3) is linear and
time-invariant,

.. math::

   y_{jt} \sim \mathcal N\Bigl(\rho_j + \textstyle\sum_k \theta_{jk}\,
     y_{k,t-1},\ \sigma_{y_j}\Bigr),

with coefficients :math:`\rho_j, \theta_{jk}` shared across time --
encoding that :math:`h_j` is time-independent. The two selection rules are

.. math::

   S1:\ \min_{j}\ \Bigl|y_{j,T_0+1} - \rho_j - \textstyle\sum_k \theta_{jk} y_{k,T_0}\Bigr|,
   \qquad
   S2:\ y_{j,T_0+1} \in \text{the } \phi\text{-PPI}.

.. _spotsynth-anchors:

Choosing the forecast anchor: ``loo`` (default) vs ``lag``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The screen needs a forecast of each donor's untreated post-intervention path
to test against. mlsynth offers two anchors, and the choice between them is the
single most consequential setting on the estimator.

The paper's anchor (``lag``) is Algorithm 1 to the letter: forecast only
the *first* post-intervention point, from the last clean pre-intervention
cross-section. At :math:`T_0` the lagged predictors are spillover-free, so the
forecast predicts the untreated value and the spillover surfaces as the error.
This works even when *most* donors are contaminated (the lag is anchored to
clean pre-data, not to the donor consensus) -- but it carries a hidden
assumption: that the spillover is sharp, i.e. present at full magnitude by
:math:`T_0 + 1`. The paper's own spillover model bakes this in (the donor error
mean jumps from 0 to :math:`\gamma_j` at :math:`T_0 + 1` and stays there). If the
spillover instead builds gradually -- the realistic case for diffusion,
adoption, or accumulation -- then :math:`\gamma_{j,T_0+1} \approx 0`, the
first-post-point error is ~0, and the screen is blind by construction. (Testing
later periods does not rescue it: their lags are themselves contaminated, so a
one-step-lagged forecast then sees only the period-to-period *increment* of the
spillover, which is small for a gradual ramp.)

The default anchor (``loo``) removes that fragility. It forecasts each
donor's *whole* post-intervention trajectory from the other donors' common
factors (leave-one-out) and ranks by the mean absolute deviation. Because it
differences out the common factor and accumulates evidence over the entire
post-period, a contaminated donor's divergence is detectable regardless of how
gradually it arrives. Its one requirement is a valid majority -- the other
donors must be mostly clean, so they form a trustworthy reference. That is the
normal applied situation (a handful of suspect donors in a large pool).

Power analysis. :func:`~mlsynth.utils.spotsynth_helpers.run_forecast_power_analysis`
reproduces the comparison: detection AUC (probability an invalid donor scores
more anomalous than a valid one; 1 = perfect, 0.5 = none, ``< 0.5`` = inverted)
as the spillover onset sweeps sharp → gradual, at two contamination levels:

.. list-table:: Detection AUC by anchor, onset, and contamination
   :header-rows: 1
   :widths: 26 18 18 18

   * - regime
     - onset
     - ``lag``
     - ``loo``
   * - valid majority (30% invalid)
     - sharp
     - 0.61
     - 0.96
   * - valid majority (30% invalid)
     - gradual
     - 0.49
     - 0.92
   * - invalid majority (80%)
     - sharp
     - 0.61
     - 0.00 (inverted)
   * - invalid majority (80%)
     - gradual
     - 0.49
     - 0.02 (inverted)

reproduced by:

.. code-block:: python

   from mlsynth.utils.spotsynth_helpers import run_forecast_power_analysis
   run_forecast_power_analysis(invalid_fracs=(0.3, 0.8), ramps=(1, 6, 24))

The reading is clean: ``loo`` dominates the applied (valid-majority) regime and
is robust to onset speed; ``lag`` only has power for sharp onsets and only earns
its keep in the paper's mostly-invalid stress regime, where ``loo`` inverts
(the contaminated majority becomes the "consensus", so the screen flags the
*valid* donors). The remaining cell -- gradual onset *and* invalid majority --
is the honest limit: no forecast screen separates signal from a contaminated
consensus that creeps in slowly.

A note on CUSUM. A cumulative-sum statistic on the lagged residuals can
rescue ``lag`` on individual gradual real panels (it telescopes the increments
back into the level), but the power analysis disqualifies it as a general
default: in the shared-factor DGP it is swamped by the common innovation and
falls below chance. ``loo`` is the robust generalization, so it -- not CUSUM --
is the default.

Recommendation. Use the default ``loo`` for applied work. Switch to ``lag``
only when you have prior reason to believe a *large fraction* of the pool is
contaminated *and* the spillover is abrupt -- e.g. the paper's simulation, which
this package pins to ``lag`` for exactly that reason.

Time averaging
^^^^^^^^^^^^^^

Two practical issues are handled by forecasting on time-averaged
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
authors' synthetic-control model (paper page 12). It is Bayesian:

.. math::

   y_{1t} \sim \mathcal N\Bigl(\alpha + \textstyle\sum_j w_j\, y_{jt},\ \sigma_y\Bigr),
   \quad w_j \ge 0,\ \textstyle\sum_j w_j = 1,
   \quad \mathbf{w} \sim \mathrm{Dirichlet}(0.4),
   \quad \sigma_y \sim \mathcal N^+(0, 1),

with the target and donors standardised to zero mean and unit standard
deviation over the pre-intervention window -- which is what absorbs the
intercept :math:`\alpha` of equation (4), so no separate intercept term is
fit. The :math:`\mathrm{Dirichlet}(0.4)` prior (concentration ``< 1``)
regularises the weights toward sparse corners of the simplex, retaining the
Abadie-Diamond-Hainmueller non-negativity / sum-to-one restriction. 95%
credible intervals for the counterfactual and the ATT are the 2.5 / 97.5
percentiles of the posterior predictive distribution.

This posterior has no closed form (a Dirichlet prior is not conjugate to a
Gaussian likelihood under the simplex constraint). The authors fit it in Stan
(Hamiltonian Monte Carlo / NUTS), which is proprietary and was not shared with
us; mlsynth fits the identical model with NumPyro's NUTS -- the same HMC
family -- so it reproduces their estimation procedure as closely as an
open-source tool can. NumPyro is an optional dependency: set
:py:attr:`SPOTSYNTHConfig.inference` to ``"frequentist"`` for a fast,
dependency-free simplex least-squares point estimate (no intervals) when running
large simulations or when NumPyro is unavailable -- the donor-*selection* bias
pattern is identical either way.

.. note::

   On validating this SC model. Because the authors' Stan was unavailable, we
   implemented the model from the *published specification* (the p.12 equations
   above) rather than from their code, and validated our NUTS fit against
   exact grid quadrature of the posterior -- ground truth, not another
   sampler. On 2- and 3-donor problems the posterior means of the weights and of
   :math:`\sigma_y` match the deterministic numerical integral to
   :math:`\le 10^{-3}` (weights) and :math:`\le 10^{-4}` (:math:`\sigma_y`). That
   establishes our code correctly samples the stated model. It does *not* and
   cannot establish that the authors' Stan matches their published equations in
   every undocumented detail -- the irreducible limitation when reference code is
   withheld. The paper's headline contribution -- the donor-selection screen
   (Algorithm 1) -- is fully specified independently of the SC solver, and is what
   the Path-B (Figure 2) and Path-A (Figure 6) reproductions below exercise
   through ``.fit()``.

Bias and debiasing when the screen errs
---------------------------------------

Bias when the screen errs: sensitivity analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The screen can make two kinds of error, and the paper bounds the SC bias
each induces (so the analyst can gauge robustness rather than trust the
selection blindly).

* False positive (a *valid* donor excluded). If the excluded donors
  were proxies for a relevant latent, dropping them reintroduces omitted-
  variable bias, bounded (Section 3.4.2) by

  .. math::

     \text{FP Bias} \le N_0 \cdot \max_{j}(|w_j|)
        \cdot \max_{l}\bigl(|\mathbb E(z_l^{\text{pre}}) - \mathbb E(z_l^{\text{post}})|\bigr),

  where :math:`z_l` are the excluded donors and :math:`\mathbf{w}` the SC
  weights. The bound is small when the *kept* donors already span the
  latents.

* False negative (an *invalid* donor kept). The bias from a retained
  spillover-:math:`\gamma_j` donor is bounded (Section 3.4.3) by

  .. math::

     \text{FN Bias} \le N_0 \cdot \max_{j}(|w_j|)
        \cdot \max_{j}(|\gamma_j|).

  The spillover :math:`\gamma_j` is unknown but, following the negative-
  control literature (Miao 2024), can be treated as a sensitivity
  parameter: domain knowledge bounding :math:`\gamma_j` bounds the bias,
  and one can ask how large a spillover would have to be to flip the sign
  of the estimated effect.

Using excluded donors to debias (proximal two-stage)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The donors excluded by the screen are not used to *build* the SC, but
they can still debias it. Because only pre-intervention data is ever
used from the excluded donors :math:`z_l`, the SC estimate is unaffected
by their post-intervention spillover dynamics. Treating the excluded
donors as proximal control variables (Shi et al. 2023), one jointly
models the target and the kept donors as functions of the excluded donors
(paper equation 5) and recovers consistent weights even when the kept
donors are imperfect (noisy) proxies of the latents.

The paper notes (page 9) that equation (5) "effectively combines [a]
two-stage process into a single model", and that this is the standard
proximal / instrumental-variables estimator: regress the kept donors
:math:`\mathbf{X}` on the excluded donors :math:`\mathbf{Z}` to form
:math:`\widehat{\mathbf{X}}`, then
regress the target on :math:`\widehat{\mathbf{X}}`. mlsynth implements exactly this
two-stage estimator in closed form (no probabilistic-programming
dependency), enabled by :py:attr:`SPOTSYNTHConfig.debias`. When ``True``,
the result object carries ``att_debiased`` alongside the screened ATT. On
the paper's Figure 4 (errors-in-variables) setting, this measurably
reduces the attenuation bias that persists even under a perfect
valid-donor selection.

Durable benchmark
^^^^^^^^^^^^^^^^^

``benchmarks/cases/spotsynth_real_data.py`` reproduces three figures of the
paper end to end: Figure 6 (real-data screening on German Reunification,
California, and Basque Country -- both ``S1`` and ``S2`` exclude a planted
noisy-proxy donor and recover the canonical effect, ATT California ~ −22, Basque
~ −1.2, while the unscreened estimate collapses toward zero); Figure 2
(leave-one-out detection AUC, high under a valid majority, inverted under an
invalid majority); and Figure 4 (the proximal debias reduces errors-in-
variables bias). Run it with
``python benchmarks/run_benchmarks.py spotsynth_real_data``; see
:doc:`replications/spotsynth`.

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
``debias=True``. Because this is the *mostly-invalid* regime, the example
sets ``forecast="lag"`` (the default ``loo`` inverts when contaminated
donors are the majority -- see the forecast-anchor discussion).

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

This reproduces the headline finding of the paper's Figure 2 through
the public ``SPOTSYNTH.fit()`` call. On the Appendix B
data-generating process, a synthetic control built on *all* donors is
biased upward (~+1.6) by the spillover-contaminated ones; one built on
the *valid* donors is unbiased; and the ``S1`` / ``S2`` screens recover
most of that gap, degrading as the donor noise grows toward the spillover
magnitude.

.. note::

   This study is the paper's mostly-invalid regime (80% of donors
   contaminated, with a sharp spillover), so it pins ``forecast="lag"``. The
   package default ``loo`` provably *inverts* here (the contaminated majority
   defines the consensus) -- this is the one regime where ``lag`` is the correct
   anchor, and it is exactly why the paper uses the first-post-point screen. See
   the forecast-anchor power analysis above.

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

prints a table of the bias :math:`\mathbb E[\widehat{\tau}] - \tau` like::

   SPOTSYNTH simulation (Figure 2), 12 reps, 80 donors, 80% invalid:
     noise      All    Valid       S1       S2
       0.1    +1.61    +0.00    +0.50    +1.32
       0.5    +1.60    -0.00    +0.48    +1.28
       1.0    +1.62    -0.00    +0.87    +1.10

reproducing the qualitative finding: All is badly biased, Valid
is unbiased, S1 removes most of the bias and is best, S2 is
intermediate, and the screens degrade as the noise rises toward the
spillover size. (The paper's full study uses 1000 donors and 2000 reps;
``SpotSimConfig`` defaults to that scale via the ``PAPER`` preset.)

Verification (semi-synthetic real data): Figure 6
-------------------------------------------------

The paper also demonstrates the screen on two canonical SC datasets by
planting a semi-synthetic invalid donor -- a noisy proxy of the
treated unit, :math:`y_{\text{syn},t} \sim \mathcal N(y_{1t}, \sigma)`.
Being a near-copy of the target, this invalid donor receives a large SC
weight and biases the effect toward zero; the screen flags and
excludes it, restoring the canonical effect. Both demos run through
``SPOTSYNTH.fit()``.

California tobacco control (Figure 6b). The 39-state Abadie panel plus the
planted donor; ``S1`` keeps 30 donors with the default leave-one-out forecast
(one contaminant in a mostly-valid pool -- the regime ``loo`` is built for).

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
       "selection": "S1", "n_donors": 30,
       # forecast="loo" is the default; inference="bayes" is the default
       # (Dirichlet(0.4) Bayesian simplex SC).
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
   screened ATT                  = -22.87
   95% credible interval         = [-26.61, -18.73]
   planted donor excluded?       = True

The contaminated pool gives a near-zero effect (the synthetic donor
hijacks the SC); the screen excludes it and the Bayesian Dirichlet SC
recovers the canonical :math:`\approx -20` packs-per-capita effect, with a
credible interval that brackets it.

:func:`~mlsynth.utils.spotsynth_helpers.replicate_all_spillover` runs all
three real panels at once -- California (Figure 6b), German reunification
(Figure 6a), and, as an additional robustness check *not* in the paper, the
Basque Country (Abadie & Gardeazabal 2003, ETA terrorism 1975). All use the
default ``forecast="loo"`` (each is a single contaminant in a mostly-valid
pool) and the Bayesian Dirichlet SC, and each returns the full set of standard
outputs -- oracle / contaminated / screened ATTs, the 95% credible interval,
the pre-treatment RMSE, the SC donor weights, and the selected / excluded donor
sets:

.. code-block:: python

   from mlsynth.utils.spotsynth_helpers import replicate_all_spillover
   results = replicate_all_spillover()

   # each entry carries the standard diagnostics, e.g. for California:
   ca = results["prop99"]
   print(ca["screened_att"], ca["att_ci"], ca["pre_rmse"])
   print(ca["donor_weights"])           # {donor: weight} for the screened SC
   print(ca["synthetic_donor_excluded"])

prints::

   Prop 99 (California):
     oracle ATT=-19.514  contaminated=-1.434  screened ATT=-22.871  95% CrI=(-26.61, -18.73)
     pre-treatment RMSE=2.544  donors kept=30/39  synthetic donor excluded=True
     top SC weights: Nevada=0.28  New Hampshire=0.24  Delaware=0.06
   Reunification (Germany):
     oracle ATT=-1297.477  contaminated=-166.863  screened ATT=-1489.345  95% CrI=(-1672.92, -1314.43)
     pre-treatment RMSE=55.629  donors kept=12/17  synthetic donor excluded=True
     top SC weights: Austria=0.30  USA=0.28  Italy=0.22
   Basque Country (ETA):
     oracle ATT=-0.692  contaminated=-0.379  screened ATT=-0.795  95% CrI=(-1.03, -0.60)
     pre-treatment RMSE=0.087  donors kept=12/17  synthetic donor excluded=True
     top SC weights: Cataluna=0.34  Aragon=0.17  Madrid (Comunidad De)=0.10

   Summary (loo forecast, Bayesian Dirichlet SC):
   panel                       oracle    contam   screened  pre-RMSE  synth excl
   Prop 99 (California)       -19.514    -1.434    -22.871     2.544  True
   Reunification (Germany)  -1297.477  -166.863  -1489.345    55.629  True
   Basque Country (ETA)        -0.692    -0.379     -0.795     0.087  True

In all three the planted invalid donor is flagged and excluded, the Bayesian
Dirichlet SC restores the effect the contamination had masked (with a 95%
posterior-predictive credible interval), and the recovered donor weights match
the canonical SC literature -- Austria/USA for West Germany, Cataluna for the
Basque Country. The Basque effect builds gradually, so it is exactly the case
the ``loo`` anchor is designed for (the first-post-point ``lag`` anchor fails on
it; see the forecast-anchor discussion).
