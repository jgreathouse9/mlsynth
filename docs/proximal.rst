Proximal Inference Synthetic Control (PROXIMAL)
===============================================

.. currentmodule:: mlsynth

When to Use This Estimator
--------------------------

Proximal inference is, by design, a **different theory of identification**
from everything else in the synthetic-control family -- the Bayesian
(:doc:`bvss`), staggered-adoption (:doc:`seq_sdid`), matrix-completion
(:doc:`mcnnm`), and forward-selection (:doc:`fdid`) variants alike. All of
those identify the counterfactual by *matching*: they assume some
combination of donors reproduces the treated unit's latent trajectory, and
they treat a good pre-treatment fit as evidence that the assumption holds.
PROXIMAL begins from the opposite admission -- that a time-varying
confounder you cannot match away is present, and that a good-looking
pre-fit can still be biased -- and identifies the effect by
*instrumenting* that confounder instead of matching it. Because this is a
genuinely different identification strategy, the sections below build it up
from scratch: what a proxy is, what a surrogate is, and why this counts as
its own theory. First, the regimes where it pays off.

The synthetic control (SC) method of Abadie and co-authors [ABADIE2010]_
is justified by a latent-factor model: each unit's outcome is driven by a
common, time-varying confounder :math:`\boldsymbol{\lambda}_t` (the
"interactive fixed effect") loaded differently across units. Classical SC
regresses the treated unit's pre-treatment outcomes on the donors' and
takes the fitted weights as the synthetic control. Abadie shows this is
(approximately) unbiased only **as the number of pre-treatment periods
grows without bound**, and even then only when a good pre-treatment fit is
attainable.

That leaves two regimes where classical SC is unreliable, and where
PROXIMAL is the right tool:

1. **Short pre-period / poor pre-fit.** With few pre-treatment periods, or
   when no convex combination of donors closely tracks the treated unit,
   the bias bound does not bite and the OLS/WLS weights are
   *inconsistent* -- the donor outcomes are error-laden proxies of
   :math:`\boldsymbol{\lambda}_t`, so the regressor is correlated with the
   residual (a textbook errors-in-variables problem). The bias does **not**
   vanish as the pre-period grows.

2. **Long or structurally-broken post-period.** When the post-period is
   long or contains trend breaks, extrapolating a pre-period fit forward is
   fragile. If you also observe **surrogates** -- post-treatment series
   predictive of the treatment effect -- PROXIMAL can borrow that
   post-period information to sharpen the estimate, which classical SC
   simply discards.

The fix, due to Shi, Li, Miao, Hu and Tchetgen Tchetgen [ProxSCM]_, is to
stop using *every* control as a regressor. Instead, **split the controls**:
some become **donors** that build the synthetic control, and the rest
become **proxies** (negative controls) that are associated with the units
only through the latent factor :math:`\boldsymbol{\lambda}_t`. The proxies
serve as instruments that purge the measurement error, yielding consistent
weights and valid inference via the generalized method of moments (GMM).
Liu, Tchetgen Tchetgen and Varjão [LiuTchetgenVar]_ extend this to
**surrogates**, time-varying correlates of the causal effect observed
post-treatment.

A Different Theory of Identification
------------------------------------

Most of causal inference identifies effects by *removing* confounding.
Either you condition on enough covariates that treatment is
as-good-as-random -- the **no-unmeasured-confounding** (ignorability)
assumption -- or, in synthetic control, you find donor weights that
reproduce the treated unit so closely that whatever drove selection is
matched away. Both routes assume the confounding can be *observed and
neutralized*.

Proximal causal inference makes a different bet. It **concedes** that an
unmeasured confounder remains -- here, the latent factor
:math:`\boldsymbol{\lambda}_t` that drives both the outcomes and the timing
of treatment -- and that you will never observe it directly. Rather than
assume it away, it asks for two observable *shadows* of that confounder and
uses them to algebraically subtract the confounding from the estimate. This
is exactly the logic epidemiologists use with **negative controls** to
detect and correct hidden bias (Lipsitch et al.; Shi, Miao, Nelson and
Tchetgen Tchetgen [ShiNegControl]_, who give the double-negative-control
identification and multiply-robust estimation theory PROXIMAL descends
from). The proximal SC papers import it into panel data:
:math:`\boldsymbol{\lambda}_t` is the confounder, and the control units are
its shadows.

.. list-table::
   :header-rows: 1
   :widths: 26 37 37

   * -
     - Matching / ignorability (classical SC, DiD, the rest of mlsynth)
     - Proximal / negative control (PROXIMAL)
   * - **Core assumption**
     - The confounder is matched or conditioned away; pre-fit is good.
     - A confounder remains; we observe valid proxies for it.
   * - **What identifies the effect**
     - A donor combination that reproduces the treated trajectory.
     - Proxies that instrument the latent factor.
   * - **Good pre-fit is...**
     - necessary evidence the design is credible.
     - neither necessary nor sufficient -- bias can hide behind it.
   * - **Fails when**
     - no convex/linear match exists, or the pre-period is short.
     - no variable is a valid proxy (or proxies are irrelevant).

The practical upshot: PROXIMAL is **not** a better way to fit the donors,
nor a shrinkage/pooling trick like the Bayesian or staggered variants. It
changes the *assumption you must defend* -- from "my synthetic control
matches" to "I have valid proxies for the latent confounder."

What Counts as a Proxy?
-----------------------

A **proxy** (synonymously, a *negative control*) is a variable that is

1. **associated** with the latent confounder
   :math:`\boldsymbol{\lambda}_t`, but
2. has **no direct causal link** to the treated unit's outcome -- its only
   connection to that outcome runs *through*
   :math:`\boldsymbol{\lambda}_t`.

Condition (1) is *relevance* (a proxy unrelated to the factor is useless,
just like a weak instrument); condition (2) is *exclusion* (a proxy with
its own path to the outcome would inject new bias). Proxies come in two
roles, mirroring the negative-control pair:

* A **negative-control outcome** is *not affected by the treatment* but is
  driven by the same latent factor. In SC the **donor outcomes
  themselves** play this role -- controls are, by the no-interference
  assumption, unaffected by the treated unit's treatment -- and they build
  the synthetic control.
* A **negative-control exposure** is associated with the latent factor but
  is *not a direct cause* of the outcome. In SC the outcomes of controls
  **excluded** from the donor pool serve here: they proxy the factor but do
  not enter the synthetic control. These are the :math:`\mathbf{Z}_0` in
  the formulas below.

Where do real proxies come from?

* **Epidemiology (the origin).** To study whether the flu vaccine cuts flu
  hospitalization -- confounded by unmeasured *health-seeking behavior* --
  one uses a non-flu outcome such as injury/trauma hospitalization as a
  negative-control outcome: the vaccine cannot plausibly affect it, yet it
  shares the health-seeking confounder, so a non-zero "effect" on it
  exposes the bias.
* **Synthetic control.** Control units dropped from the donor pool because
  they ran *similar* interventions or risk *spillover* are ideal proxies:
  they track the common factor but violate no-interference if used as
  donors. (In Abadie's tobacco study, 38 of 50 states were eligible but
  only a handful received weight; the rest can be proxies.) So can
  treatment-free **contemporaneous covariates** of the donors -- a sector
  index, market trading volume, weather -- that move with
  :math:`\boldsymbol{\lambda}_t` but are not caused by the treatment.
* **Marketing / geo experiments.** In a regional campaign, a category-
  demand or search-volume index in *untreated* regions, or foot-traffic in
  markets the campaign never reached: associated with the macro demand
  factor, but with no direct line to the treated region's sales.

What Counts as a Surrogate?
---------------------------

A **surrogate** is a *post-treatment* variable driven by the same latent
factors as the **causal effect itself** -- not the confounder of the
untreated outcome. It is predictive of how big the effect is, period by
period. The defining contrast with a proxy:

* a **proxy** carries information about :math:`\boldsymbol{\lambda}_t`, the
  confounder of the *untreated* outcome, and is used in the **pre-period**
  to recover the donor weights;
* a **surrogate** carries information about :math:`\boldsymbol{\rho}_t`, the
  factors of the *treatment effect*, and is used in the **post-period** to
  sharpen or extend the estimate.

Loosely: a proxy cleans up the *denominator* (confounding); a surrogate
informs the *numerator* (the effect). Crucially, a surrogate **may itself
be affected by the treatment** -- that is fine, because it is removed from
the donor pool and used only to learn the effect's trajectory, never to
build the counterfactual.

Where do real surrogates come from?

* **Panic of 1907 (the paper's example).** The bid prices of the two
  *other* trusts that also suffered bank runs are useless as donors (the
  crisis hit them too), but their post-crisis movements track the very
  shock driving Knickerbocker's effect -- making them strong surrogates.
  Even Knickerbocker's own bid price is used this way.
* **Marketing.** After a price cut, fast downstream signals -- app opens,
  add-to-cart rate, repeat-visit rate -- respond to the same demand shock
  as revenue. They predict the revenue effect and arrive quickly, which is
  valuable when the post-launch revenue series is short or noisy.
* **Spillovers / partial treatment.** Geographies that are partially
  treated or absorb spillover should not be donors, but they carry the
  treatment-effect signal and so make good surrogates.
* **Long-run effects.** An early leading indicator of a long-horizon
  outcome (a classic "surrogate endpoint" in clinical trials) lets you
  estimate a long-run effect from a short post-treatment window.

The Methods
-----------

``PROXIMAL`` exposes **six** estimators. They are idiosyncratic -- each
makes a different identification bet and needs different inputs -- so you
**choose** the ones you want with the ``methods`` argument and the
estimator runs *exactly* those (validating that your inputs support them):

.. list-table::
   :header-rows: 1
   :widths: 12 42 22

   * - Method
     - What it uses
     - Paper
   * - **PI**
     - Donors + donor proxies; pre-period moments only.
     - Shi et al. [ProxSCM]_
   * - **PIS**
     - Adds surrogates + surrogate proxies; pre *and* post data.
     - Liu et al. [LiuTchetgenVar]_
   * - **PIPost**
     - Surrogates, **post-treatment data only**.
     - Liu et al. [LiuTchetgenVar]_
   * - **SPSC**
     - Donors only -- a **single** proxy type, with the treated unit's
       own outcome as the instrument.
     - Park & Tchetgen Tchetgen [SPSC]_
   * - **DR**
     - Donors + donor proxies; **doubly robust** -- consistent if *either*
       the outcome or the weighting model is right.
     - Qiu et al. [DRProx]_
   * - **PIPW**
     - Donors + donor proxies; a **weighting-only** estimator (treatment
       confounding bridge), no outcome model.
     - Qiu et al. [DRProx]_

.. code-block:: python

   PROXIMAL({..., "methods": ["SPSC"]})              # SPSC alone (no proxies needed)
   PROXIMAL({..., "methods": ["PI"]})                # classic proximal inference
   PROXIMAL({..., "methods": ["DR", "PIPW"]})        # doubly robust + weighting
   PROXIMAL({..., "methods": ["PI", "PIS", "PIPost", "SPSC", "DR", "PIPW"]})  # all six

``methods`` is **required** -- there is no implicit default -- so a run
only ever computes what you asked for. The config layer enforces input
consistency: ``"PI"``/``"PIS"``/``"PIPost"``/``"DR"``/``"PIPW"`` require
donor proxies (and, for the surrogate methods, surrogate units and
proxies), whereas ``"SPSC"`` needs only the donor pool. Results are
returned on a
:class:`~mlsynth.utils.proximal_helpers.structures.PROXIMALResults`, with
``results.methods`` mapping each requested method to its fit.

What Each Method Does in Practice
---------------------------------

Beyond the econometrics, the four methods answer different practical
questions. Classical SCM just asks "what weighted blend of controls tracks
my treated unit?" -- these methods each go further in a distinct way.

**PI -- de-noise the synthetic control.** *"Build a synthetic version of my
treated unit from clean controls, but correct for the fact that the
controls are noisy stand-ins for the thing that actually drives my
outcome."* A retailer launches a loyalty program in one metro; nearby
metros are controls, but their sales are noisy proxies of a shared regional
demand cycle, so a plain SC blend is biased. PI uses a *second* set of
metros -- ones kept out of the blend (say, because they ran their own
promotions) -- as instruments to purge that noise, so the counterfactual
isn't distorted by metro-specific blips.

**PIS -- borrow fast signals when the outcome is slow or broken.** *"My
post-period is long or has a structural break, and the outcome itself is
noisy -- lean on quick-moving signals that respond to the same shock as the
effect."* After a price change, monthly revenue is noisy and the clean
post-window is short, but app engagement (sessions, add-to-cart, repeat
visits) moves with the same demand shock as revenue. PIS folds those
**surrogates** in -- using both pre- and post-launch data -- to sharpen the
revenue-effect estimate.

**PIPost -- estimate the effect from post-launch data alone.** *"I don't
have a usable pre-period for the controls, but I do have surrogates after
launch."* Maybe clean control logging only began at rollout, or the
pre-period is contaminated. Because the treated outcome splits into a
donor-matched piece and a surrogate-driven effect piece, PIPost recovers
the effect from **post-treatment data only** -- at the cost of some
efficiency.

**SPSC -- the no-proxy fallback.** *"All I have is my treated series and a
pool of other series -- no curated proxy or surrogate groups."* A flagship
store's sales versus a pool of other stores, with nothing but the sales
panel. SPSC treats the other stores as noisy proxies of the flagship's own
counterfactual and uses the **flagship's own pre-period** as the
instrument, returning a de-noised synthetic flagship plus conformal bands
that stay valid even with a short post-window. It is the most practical
proximal method when no natural second proxy group exists.

**DR -- hedge against getting the model wrong.** *"I have both a synthetic
control I trust *and* a weighting model I trust -- but I'm not sure which is
right, and I don't want the answer to hinge on that."* DR combines an
outcome model (the synthetic control) with a weighting model (how the
confounding shifts at the intervention) so the ATT is consistent if
**either one** is correctly specified -- you get one shot at being right
across two tries. Useful in a vaccine roll-out study where you can build a
synthetic-control of hospitalizations *and* model how disease pressure
shifted, and want robustness to a misspecification of either.

**PIPW -- weight, don't model the outcome.** *"I'd rather not commit to a
model for the treated unit's counterfactual trajectory at all."* PIPW
estimates the effect purely by **re-weighting** the pre-period to look like
the post-period (a covariate-shift / inverse-probability-style weight built
from the proxies), with no synthetic-control trajectory. It is the natural
choice when the outcome is hard to model but the *shift* in the
confounding is easier to capture.

Notation
--------

We index units by :math:`j`, with :math:`j = 0` the sole **treated** unit
and :math:`\mathcal{N} = \{1, \ldots, N\}` the **control** units. A subset
:math:`\mathcal{D} \subseteq \mathcal{N}` is the **donor pool** used to
build the synthetic control; the remaining controls are repurposed as
**proxies**. Time runs over :math:`t \in \{1, \ldots, T\}`, split by the
intervention into a pre-treatment window
:math:`\mathcal{T}_1 = \{1, \ldots, T_0\}` and a post-treatment window
:math:`\mathcal{T}_2 = \{T_0 + 1, \ldots, T\}`; the post-period has
:math:`T - T_0` periods (Shi et al.'s :math:`T_1`). Potential outcomes are
:math:`y^0_{jt}` and :math:`y^1_{jt}`, and we observe

.. math::

   y_{0t} =
   \begin{cases}
       y^0_{0t}, & t \in \mathcal{T}_1, \\
       y^1_{0t}, & t \in \mathcal{T}_2.
   \end{cases}

Stacking the donor pool, let :math:`\mathbf{W}_t \in \mathbb{R}^{|\mathcal{D}|}`
be the donor outcomes at time :math:`t`, with weight vector
:math:`\boldsymbol{\alpha}`. Let :math:`\mathbf{Z}_{0t}` be the **donor
proxies**, :math:`\mathbf{X}_t \in \mathbb{R}^{H}` the **surrogate
outcomes** with coefficients :math:`\boldsymbol{\gamma}`, and
:math:`\mathbf{Z}_{1t}` the **surrogate proxies**. The estimand is the
average treatment effect on the treated,

.. math::

   \tau = \frac{1}{T - T_0} \sum_{t \in \mathcal{T}_2}
       \bigl(y^1_{0t} - y^0_{0t}\bigr).

.. admonition:: Notation bridge

   The source papers write the treated outcome :math:`Y_t`, donors
   :math:`W_t`, donor proxies :math:`Z_{0,t}`, surrogates :math:`X_t`,
   surrogate proxies :math:`Z_{1,t}`, the donor latent factor
   :math:`\lambda_t`, and the effect's latent factor :math:`\rho_t`. We
   keep :math:`\mathbf{W}, \mathbf{Z}_0, \mathbf{X}, \mathbf{Z}_1,
   \boldsymbol{\lambda}, \boldsymbol{\rho}` and write the treated unit as
   :math:`j = 0`.

Why Standard SC Fails Here
--------------------------

Assume the interactive fixed-effects model

.. math::

   y^0_{jt} = \boldsymbol{\mu}_j^\top \boldsymbol{\lambda}_t + \varepsilon_{jt},

where :math:`\boldsymbol{\lambda}_t` is an unobserved common factor and
:math:`\boldsymbol{\mu}_j` a unit-specific loading. A synthetic control
exists if the treated loading is a weighted average of the donor loadings,
:math:`\boldsymbol{\mu}_0 = \sum_{j \in \mathcal{D}} \alpha_j
\boldsymbol{\mu}_j`. Then in the pre-period

.. math::

   y_{0t} = \sum_{j \in \mathcal{D}} \alpha_j y_{jt}
       + \Bigl(\varepsilon_{0t} - \sum_{j \in \mathcal{D}} \alpha_j \varepsilon_{jt}\Bigr).

The donor outcomes :math:`y_{jt}` are **noisy proxies** of
:math:`\boldsymbol{\lambda}_t`: they carry the idiosyncratic errors
:math:`\varepsilon_{jt}`, which also appear in the residual. Regressing
:math:`y_{0t}` on them is therefore an errors-in-variables regression, and
the OLS/WLS weights are inconsistent **even as** :math:`T_0 \to \infty`
(Ferman and Pinto). PROXIMAL breaks this correlation with an instrument.

Mathematical Formulation
------------------------

Proximal Inference (PI)
~~~~~~~~~~~~~~~~~~~~~~~~~

Suppose we observe proxies :math:`\mathbf{Z}_{0t}` -- e.g. the outcomes of
controls *excluded* from the donor pool, or contemporaneous covariates --
that are associated with the units only through :math:`\boldsymbol{\lambda}_t`
in the pre-period. Then the pre-period residual
:math:`y_{0t} - \mathbf{W}_t^\top \boldsymbol{\alpha}` is **orthogonal** to
the proxies, giving the moment condition

.. math::

   \mathbb{E}\!\left[\mathbf{Z}_{0t}\bigl(y_{0t} - \mathbf{W}_t^\top
   \boldsymbol{\alpha}\bigr)\right] = 0, \qquad t \in \mathcal{T}_1.

Unlike the OLS normal equation
:math:`\mathbb{E}[\mathbf{W}_t(y_{0t} - \mathbf{W}_t^\top
\boldsymbol{\alpha})] = 0`, this estimating function is **mean-zero at the
truth** because :math:`\mathbf{Z}_{0t}` is uncorrelated with the
measurement error. Solving it by GMM yields a consistent
:math:`\hat{\boldsymbol{\alpha}}`, and the ATT is the mean post-period gap

.. math::

   \hat{\tau} = \frac{1}{T - T_0} \sum_{t \in \mathcal{T}_2}
       \bigl(y_{0t} - \mathbf{W}_t^\top \hat{\boldsymbol{\alpha}}\bigr).

Adding Surrogates (PIS)
~~~~~~~~~~~~~~~~~~~~~~~~~

Surrogates :math:`\mathbf{X}_t` are post-treatment series driven by the
same latent factors :math:`\boldsymbol{\rho}_t` as the treatment effect:

.. math::

   y^1_{0t} - y^0_{0t} = \boldsymbol{\rho}_t^\top \boldsymbol{\theta} + \delta_t,
   \qquad
   \mathbf{X}_t = \boldsymbol{\Phi}^\top \boldsymbol{\rho}_t + \boldsymbol{\epsilon}_{X,t}.

With surrogate proxies :math:`\mathbf{Z}_{1t}` instrumenting
:math:`\mathbf{X}_t`, the effect coefficient
:math:`\boldsymbol{\gamma}` (with :math:`\boldsymbol{\Phi}
\boldsymbol{\gamma} = \boldsymbol{\theta}`) is identified by a second,
post-period moment. The stacked conditions are

.. math::

   \mathbb{E}\!\left[\mathbf{Z}_{0t}\bigl(y_{0t} - \mathbf{W}_t^\top
   \boldsymbol{\alpha}\bigr)\right] = 0,\ t \in \mathcal{T}_1,
   \qquad
   \mathbb{E}\!\left[\mathbf{Z}_{1t}\bigl(y_{0t} - \mathbf{W}_t^\top
   \boldsymbol{\alpha} - \mathbf{X}_t^\top \boldsymbol{\gamma}\bigr)\right] = 0,\
   t \in \mathcal{T}_2,

and the ATT is :math:`\hat{\tau} = (T - T_0)^{-1} \sum_{t \in \mathcal{T}_2}
\mathbf{X}_t^\top \hat{\boldsymbol{\gamma}}`.

Post-Treatment-Only (PIPost)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because the post-period outcome carries both a latent-factor component
(matched by donors) and a surrogate-driven effect component, both
:math:`\boldsymbol{\alpha}` and :math:`\boldsymbol{\gamma}` can be
estimated from a **single post-period IV fit**, using
:math:`(\mathbf{Z}_{0t}, \mathbf{Z}_{1t})` to instrument
:math:`(\mathbf{W}_t, \mathbf{X}_t)`:

.. math::

   \mathbb{E}\!\left[
   \begin{pmatrix} \mathbf{Z}_{0t} \\ \mathbf{Z}_{1t} \end{pmatrix}
   \bigl(y_{0t} - \mathbf{W}_t^\top \boldsymbol{\alpha}
   - \mathbf{X}_t^\top \boldsymbol{\gamma}\bigr)\right] = 0,
   \qquad t \in \mathcal{T}_2.

This is the most economical method -- it needs no pre-period -- but also
the least efficient, since it discards pre-treatment information.

Inference: GMM Sandwich with HAC
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each method stacks its moment conditions into :math:`U_t(\theta)` for
parameters :math:`\theta = (\boldsymbol{\alpha}, \boldsymbol{\gamma},
\tau)` and solves the GMM problem
:math:`\hat{\theta} = \arg\min_\theta\, \bar{U}(\theta)^\top \Omega^{-1}
\bar{U}(\theta)`. Standard errors come from the sandwich variance

.. math::

   \mathrm{Cov} = \mathbf{G}^{-1} \boldsymbol{\Omega}
       \bigl(\mathbf{G}^{-1}\bigr)^\top,
   \qquad
   \mathrm{SE}(\hat{\tau}) = \sqrt{\frac{\mathrm{Cov}[-1,-1]}{T}},

where :math:`\mathbf{G}` is the Jacobian of the moment conditions and
:math:`\boldsymbol{\Omega}` is the **heteroskedasticity- and
autocorrelation-consistent (HAC)** long-run variance of the moments,

.. math::

   \boldsymbol{\Omega} = \frac{1}{T} \sum_{\ell=-J}^{J} k(\ell, J)
       \sum_{t} \mathbf{g}_t \mathbf{g}_{t+\ell}^\top,

with :math:`k(\cdot)` the Bartlett kernel and bandwidth
:math:`J = \bigl\lfloor 4 (\,(T - T_0)/100\,)^{2/9} \bigr\rfloor`. (For
PIPost the normalization uses the post-period count :math:`T - T_0` in
place of :math:`T`.) The HAC middle is what makes the intervals valid under
serially correlated errors.

Assumptions
-----------

**Assumption 1 (interactive fixed effects).** The untreated outcome obeys
:math:`y^0_{jt} = \boldsymbol{\mu}_j^\top \boldsymbol{\lambda}_t +
\varepsilon_{jt}` with :math:`\mathbb{E}[\varepsilon_{jt} \mid
\boldsymbol{\lambda}_t] = 0`, and there is no interference (the treated
unit's status does not affect controls).

*Remark.* The latent factor :math:`\boldsymbol{\lambda}_t` is the
unmeasured confounder: it both drives the outcome and is associated with
treatment timing. This is the standard SC data-generating model; PROXIMAL
does not need it to be stationary, so trending or non-stationary factors
are allowed.

**Assumption 2 (existence of a synthetic control).** There exist weights
:math:`\boldsymbol{\alpha}` with :math:`\boldsymbol{\mu}_0 = \sum_{j \in
\mathcal{D}} \alpha_j \boldsymbol{\mu}_j` (and, for surrogates,
:math:`\boldsymbol{\gamma}` with :math:`\boldsymbol{\Phi}
\boldsymbol{\gamma} = \boldsymbol{\theta}`).

*Remark.* A necessary condition is that the donor pool be at least as large
as the number of latent factors (:math:`|\mathcal{D}| \ge \dim
\boldsymbol{\lambda}_t`), and likewise that there be at least as many
surrogates as effect factors. Weights need **not** be non-negative or sum
to one -- the simplex is optional, used only for interpretability or to
avoid extrapolation.

**Assumption 3 (valid proxies).** The proxies satisfy
:math:`\mathbf{Z}_{0t} \perp\!\!\!\perp \{y_{0t}, \mathbf{W}_t\} \mid
\boldsymbol{\lambda}_t` for :math:`t \in \mathcal{T}_1` (and analogously
for :math:`\mathbf{Z}_{1t}` in the post-period).

*Remark.* Proxies must touch the units **only through the latent factor** --
they carry information about :math:`\boldsymbol{\lambda}_t` but have no
direct causal link to the treated outcome. Outcomes of controls excluded
from the donor pool (e.g. units dropped for similar interventions or
spillover risk) and treatment-free contemporaneous covariates are natural
candidates. Proxy choice is a *pre-specified, domain-knowledge* decision,
not a data-driven search.

**Assumption 4 (relevance / completeness).** The cross-moment
:math:`\mathbb{E}[\mathbf{Z}_{0t} \mathbf{W}_t^\top]` has full column rank
(and a completeness condition holds for nonparametric identification).

*Remark.* This is the instrument-relevance condition: the proxies must be
**strongly associated** with the latent factor, so that variation in
:math:`\mathbf{W}_t` is recoverable from variation in
:math:`\mathbf{Z}_{0t}`. It fails precisely when the proxies are unrelated
to :math:`\boldsymbol{\lambda}_t`, in which case they cannot purge the
measurement error.

**Assumption 5 (stationary, weakly dependent errors).** The error processes
are stationary and weakly dependent.

*Remark.* This is weaker than i.i.d. errors: it permits serial correlation,
which is why inference uses the HAC variance rather than a white-noise
formula. The *latent factors themselves* may still be non-stationary.

.. admonition:: Contaminated surrogates

   In practice "pure" surrogates are rare. Often a surrogate is an
   alternative outcome of the treated unit, or the outcome of another
   affected unit, and so is **contaminated** by the donor latent factor
   :math:`\boldsymbol{\lambda}_t` as well as the effect factor
   :math:`\boldsymbol{\rho}_t` (Appendix A.3 of [LiuTchetgenVar]_).
   ``mlsynth`` handles this by residualizing the surrogate outcomes against
   the donor proxies and donor outcomes on the pre-period (a
   confounding-bridge projection) before the surrogate stage, so the
   surrogates used downstream carry the effect signal net of
   :math:`\boldsymbol{\lambda}_t`.

Example
-------

The block below is self-contained: simulate one panel from the surrogate
data-generating process of [LiuTchetgenVar]_ -- two trending donor factors
:math:`\boldsymbol{\lambda}_t`, one effect factor :math:`\boldsymbol{\rho}_t`
with mean one (so the true ATT is :math:`\approx 1`), and **contaminated**
surrogates that load on both -- then fit ``PROXIMAL`` and read off the ATT
and standard error for all three methods.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from mlsynth import PROXIMAL

   rng = np.random.default_rng(4)
   F, T0, T, H = 2, 100, 200, 2            # donor factors, pre, total, surrogates
   post = np.arange(T) >= T0
   noise = 0.3

   lam = np.log(np.arange(1, T + 1))[:, None] + rng.normal(size=(T, F))  # trending factors
   rho = 1.0 + rng.normal(size=T)                                        # effect factor, mean 1
   Theta = np.array([[0.6, 0.4], [0.4, 0.6]])                            # surrogate contamination

   Y = lam.sum(1) + rng.normal(scale=noise, size=T)
   Y[post] += rho[post]                                                  # apply the effect
   true_att = rho[post].mean()

   W  = lam + rng.normal(scale=noise, size=(T, F))     # donor outcomes
   Z0 = lam + rng.normal(scale=noise, size=(T, F))     # donor proxies
   X  = lam @ Theta + np.outer(rho * post, np.ones(H)) + rng.normal(scale=noise, size=(T, H))
   Z1 = np.outer(rho, np.ones(H)) + lam @ Theta + rng.normal(scale=noise, size=(T, H))

   # Long panel: each donor unit carries (outcome=W, donorproxy=Z0); each surrogate
   # unit carries (donorproxy column = surrogate outcome X, surrogatevar = Z1).
   rows = []
   for t in range(T):
       rows.append({"unit": "treated", "time": t, "y": Y[t], "dp": 0.0, "sv": 0.0,
                    "treat": int(post[t])})
       for j in range(F):
           rows.append({"unit": f"donor{j}", "time": t, "y": W[t, j], "dp": Z0[t, j],
                        "sv": 0.0, "treat": 0})
       for k in range(H):
           rows.append({"unit": f"surr{k}", "time": t, "y": 0.0, "dp": X[t, k],
                        "sv": Z1[t, k], "treat": 0})
   df = pd.DataFrame(rows)

   res = PROXIMAL({
       "df": df, "outcome": "y", "treat": "treat", "unitid": "unit", "time": "time",
       "methods": ["PI", "PIS", "PIPost"],
       "donors": [f"donor{j}" for j in range(F)],
       "surrogates": [f"surr{k}" for k in range(H)],
       "vars": {"donorproxies": ["dp"], "surrogatevars": ["sv"]},
       "display_graphs": False,
   }).fit()

   print(f"true ATT = {true_att:.3f}")
   for name, fit in res.methods.items():
       print(f"{name:6s} ATT = {fit.att:+.3f}  SE = {fit.att_se:.3f}")

A representative run prints (true ATT ≈ 1.05)::

   PI     ATT = +1.001  SE = 0.138
   PIS    ATT = +1.018  SE = 0.129
   PIPost ATT = +1.080  SE = 0.120

``res`` is a
:class:`~mlsynth.utils.proximal_helpers.structures.PROXIMALResults`:
``res.pi`` / ``res.pis`` / ``res.pipost`` hold the per-method
:class:`~mlsynth.utils.proximal_helpers.structures.ProximalMethodFit`
objects, ``res.methods`` maps the names that ran, and convenience accessors
(``res.att``, ``res.att_se``, ``res.donor_weights``,
``res.att_by_method()``) forward to the headline PI fit.

Empirical Illustration: Panic of 1907
--------------------------------------

[LiuTchetgenVar]_ apply the surrogate method to the Panic of 1907, using
data from [fohlin2021]_. The crisis brought down the Knickerbocker Trust, a
major New York bank. We have log stock prices for 59 trusts, with
Knickerbocker as the treated unit. Two other trusts also suffered bank
runs and seven were tied to major firms; dropping one trust missing a
period leaves 49 potential controls. The logged **bid price** of the 49
controls serves as the donor proxy for Knickerbocker's log price -- a
sensible proxy, since the bid reflects macro forces driving the overall
price.

.. code-block:: python

    import pandas as pd
    import numpy as np
    from mlsynth import PROXIMAL

    file_path = "https://github.com/jgreathouse9/mlsynth/raw/refs/heads/main/basedata/trust.dta"
    df = pd.read_stata(file_path)
    df = df[df["ID"] != 1]  # Drop the unbalanced unit

    surrogates = df[df['introuble'] == 1]['ID'].unique().tolist()  # affected trusts
    donors = df[df['type'] == "normal"]['ID'].unique().tolist()    # pure controls

    vars = ["bid_itp", "ask_itp"]
    df[vars] = df[vars].apply(np.log)  # log, per the paper
    df['Panic'] = np.where((df['time'] > 229) & (df['ID'] == 34), 1, 0)

    treat, outcome, unitid, time = "Panic", "prc_log", "ID", "date"
    var_dict = {"donorproxies": ["bid_itp"], "surrogatevars": ["ask_itp"]}

    # Donors-only proximal inference (PI)
    res_pi = PROXIMAL({
        "df": df, "treat": treat, "time": time, "outcome": outcome, "unitid": unitid,
        "methods": ["PI"],
        "treated_color": "black", "counterfactual_color": ["blue"],
        "display_graphs": True, "vars": var_dict, "donors": donors,
    }).fit()

    # Adding surrogates (PI, PIS, PIPost)
    res_surr = PROXIMAL({
        "df": df, "treat": treat, "time": time, "outcome": outcome, "unitid": unitid,
        "methods": ["PI", "PIS", "PIPost"],
        "treated_color": "black", "counterfactual_color": ["blue", "red", "lime"],
        "display_graphs": True, "vars": var_dict, "donors": donors,
        "surrogates": surrogates,  # the affected trusts, repurposed as surrogates
    }).fit()

    print(res_surr.att_by_method())

This pulls the data straight from the repository (48 pure-control donors, 3
affected trusts as surrogates) and prints the ATT for each method::

    {'PI': -1.148, 'PIS': -1.148, 'PIPost': -1.220}

which reproduces the paper's full-window Table 3 estimates (PI -1.138,
PI-S -1.134, PI-P -1.220) to within rounding.

Using the bid price as a proxy, the synthetic control fits the
pre-intervention series well. The affected trusts -- which would be
*discarded* in a classical SC analysis because they violate the
no-interference assumption -- are instead repurposed as surrogates: they do
not enter the donor pool, but their post-intervention movements help pin
down the latent effect factors. The asking price of those trusts is their
surrogate proxy. Even using only post-intervention data (PIPost), the
estimate largely agrees with the donors-only proximal inference.

Single Proxy Synthetic Control (SPSC)
-------------------------------------

PI, PIS and PIPost all require **two** proxy types: outcome proxies (the
donors) *and* a separate group of treatment/surrogate proxies
(:math:`\mathbf{Z}_0`, :math:`\mathbf{Z}_1`) to instrument them. Park and
Tchetgen Tchetgen [SPSC]_ show this can be reduced to a **single** proxy
type -- the donor outcomes alone -- by a clever change of perspective.

Instead of viewing the donors as proxies of a latent factor, SPSC views
them as **error-prone proxies of the treated unit's own treatment-free
potential outcome** :math:`y^0_{0t}`. It posits a *synthetic-control bridge
function* :math:`h^\star` that is conditionally unbiased for that outcome,
:math:`y^0_{0t} = \mathbb{E}[h^\star(\mathbf{W}_t) \mid y^0_{0t}]`. With a
linear bridge :math:`h^\star(\mathbf{W}_t) = \mathbf{W}_t^\top
\boldsymbol{\gamma}`, this is the "reverse" measurement-error regression

.. math::

   \mathbf{W}_t^\top \boldsymbol{\gamma} = y^0_{0t} + \bar{\varepsilon}_t,
   \qquad \mathbb{E}[\bar{\varepsilon}_t \mid y^0_{0t}] = 0,

so the **treated unit's own pre-treatment outcome is a valid instrument
for the donors** -- no second proxy group is needed. The identifying
moment (Theorem 3.1 of [SPSC]_) is
:math:`\mathbb{E}[\,\phi(y_t)\,(y_t - \mathbf{W}_t^\top \boldsymbol{\gamma})\,]
= 0` over :math:`t \in \mathcal{T}_1`, where :math:`\phi(\cdot)` is a basis
of the treated outcome (the identity by default).

*Why use it.* SPSC trades the need for a curated proxy/surrogate group for
a single, always-available instrument -- the treated series itself -- which
makes it the most practical proximal method when no natural second proxy
group exists. It pairs naturally with a **conformal** prediction interval
for the per-period effect (``spsc_conformal=True``), valid even with a
short post-period.

**Estimation.** Because there are typically far fewer instruments than
donors, :math:`\boldsymbol{\gamma}` is estimated by a **ridge-regularized
GMM** (penalty selected by leave-one-out cross-validation), and the ATT is
the mean post-period gap with a GMM sandwich (HAC) standard error. Two
variants handle trends: **SPSC-NoDT** uses the raw outcome as the
instrument, while **SPSC-DT** first residualizes the treated outcome
against a cubic B-spline time trend -- essential when the series is
non-stationary (the analogue of the time-varying estimating function in
:math:`\Psi_{\text{pre}}`).

Select it with ``methods=["SPSC"]``. Unlike PI/PIS/PIPost it needs **no
proxy variables at all** -- just the treated series and the donor pool:

.. code-block:: python

    import pandas as pd
    import numpy as np
    from mlsynth import PROXIMAL

    raw = pd.read_stata("https://github.com/jgreathouse9/mlsynth/raw/refs/heads/main/basedata/trust.dta")
    raw["prc_log"] = raw["prc_log"].astype(float)

    # Park & Tchetgen Tchetgen's window: 1906-01-05 to 1908-12-30 (T0=217).
    win = raw[(raw["date"] >= "1906-01-05") & (raw["date"] <= "1908-12-31")].copy()

    # Treated unit = average log price of the two most-affected trusts.
    treated = (win[win["type"].isin(["Knickerbocker", "Trust Co of Am"])]
               .groupby(["date", "time"], as_index=False)
               .agg(prc_log=("prc_log", "mean")))
    treated["ID"] = "treated"

    # Donors = the weakly-connected "normal" trusts (drop the one unbalanced unit).
    donors_df = win[(win["type"] == "normal") & (win["ID"] != 1)][
        ["ID", "date", "time", "prc_log"]].copy()
    donors_df["ID"] = donors_df["ID"].astype(str)

    df = pd.concat([treated[["ID", "date", "time", "prc_log"]], donors_df], ignore_index=True)
    df["Panic"] = np.where((df["time"] >= 230) & (df["ID"] == "treated"), 1, 0)
    donor_ids = sorted(donors_df["ID"].unique())

    res = PROXIMAL({
        "df": df, "treat": "Panic", "time": "date", "outcome": "prc_log", "unitid": "ID",
        "methods": ["SPSC"],          # SPSC alone -- no proxies needed
        "donors": donor_ids,
        "spsc_detrend": True,         # SPSC-DT
        "display_graphs": False,
    }).fit()

    print(res.spsc.att, res.spsc.att_se, res.spsc.metadata["variant"])

This reproduces the paper's Table 3: **SPSC-DT ATT -0.815 (SE 0.067)** and,
with ``spsc_detrend=False``, **SPSC-NoDT ATT -0.812 (SE 0.085)** -- against
the paper's -0.816 / 0.066 and -0.813 / 0.084.

**Conformal intervals.** Set ``spsc_conformal=True`` (optionally
``spsc_conformal_periods=[...]`` to cover only some post-periods) to attach
pointwise prediction intervals for the per-period effect, returned on
``res.spsc.metadata["conformal"]`` as ``{"periods", "lower", "upper"}``.
Over the Panic post-period these reproduce the average interval width of
the paper's Figure 3 (≈ 0.07 for SPSC-DT). The inversion re-fits the
weights on a grid of candidate effects per period, so it is opt-in for
cost.

**Nonparametric (series) SPSC.** By default the treated unit's own outcome
enters the moment conditions linearly -- the reference's identity
``Y.basis``. Park & Tchetgen Tchetgen's supplement (S1.6) notes that a
**rich basis** of the outcome -- "polynomials, trigonometric functions,
splines, or wavelets" -- spans a larger space of the latent factor and so
identifies a bridge that need not be linear. Set ``spsc_basis_degree=p``
(:math:`p \ge 2`) to replace the instrument with the polynomial sieve
:math:`[\,y,\,y^2,\,\dots,\,y^p\,]`. This **over-identifies** the
ridge-GMM (more moments than donor weights) and is the right choice when
the synthetic-control relationship is nonlinear in the donor outcomes;
``spsc_basis_degree=1`` (the default) is bit-for-bit the linear single
proxy. The fitted variant is labelled accordingly
(``res.spsc.metadata["variant"]`` becomes e.g. ``"SPSC-DT-NP3"``), and the
detrending and conformal machinery carry the same sieve.

Doubly Robust Proximal Synthetic Control (DR & PIPW)
----------------------------------------------------

PI, PIS, PIPost and SPSC all rest on getting **one** model right -- an
outcome model (the synthetic control). Qiu, Shi, Miao, Dobriban and
Tchetgen Tchetgen [DRProx]_ add a second, complementary nuisance and
combine the two so you only need *one of them* to be correct.

There are two bridges (each augmented with an intercept):

* the **outcome bridge** :math:`h(\mathbf{W}_t) = (1, \mathbf{W}_t)^\top
  \boldsymbol{\alpha}` -- a pre-period IV fit of the treated outcome on the
  donors, instrumented by the proxies (the PI idea); and
* the **treatment confounding bridge** :math:`q(\mathbf{Z}_t) =
  \exp\{(1, \mathbf{Z}_t)^\top \boldsymbol{\beta}\}` -- a covariate-shift /
  likelihood-ratio **weight** capturing how the unmeasured confounding
  shifts at the intervention, solving
  :math:`\mathbb{E}_{\text{pre}}[q(\mathbf{Z})(1,\mathbf{W})] =
  \mathbb{E}_{\text{post}}[(1,\mathbf{W})]`.

They give three estimands:

.. math::

   \text{outcome only:}\quad & \tau = \mathbb{E}_{\text{post}}[Y - h(\mathbf{W})], \\
   \text{weighting only (PIPW):}\quad & \tau = \mathbb{E}_{\text{post}}[Y] - \mathbb{E}_{\text{pre}}[q(\mathbf{Z})\,Y], \\
   \text{doubly robust (DR):}\quad & \tau = \mathbb{E}_{\text{post}}[Y - h(\mathbf{W})] - \mathbb{E}_{\text{pre}}[q(\mathbf{Z})\{Y - h(\mathbf{W})\}].

The DR form is consistent if **either** :math:`h` **or** :math:`q` is
correctly specified -- not necessarily both. ``PIPW`` exposes the
weighting-only estimator (no outcome model at all); the outcome-only form
is the existing ``PI``.

**Estimation.** Each is a just-identified GMM (``alpha`` by IV, ``beta`` by
a small nonlinear solve, the means in closed form), so the parameters solve
the moment equations exactly and the ATT standard error is the GMM sandwich
with a Bartlett-HAC middle. ``DR`` returns the outcome-bridge synthetic
control as its counterfactual; ``PIPW``, being a pure weighting estimator,
has no imputed trajectory (its ``counterfactual`` is ``NaN``).

Both consume the same inputs as ``PI`` -- donors ``W`` and the donor
proxies ``Z`` -- so just add them to ``methods``. The block below is a
**runnable proof** of the agreement claimed in *Replication Status*: it
draws from the reference implementation's own DGP
(``DR_Proximal_SC/simulation/normal``: ``true.ATE = 2``, AR(1) confounders,
:math:`W_j = 2U_j + \text{noise}`, :math:`Z_j = 2U_j + \text{noise}`), runs
the packaged ``PROXIMAL``, and checks recovery, Wald coverage, and double
robustness:

.. code-block:: python

   import numpy as np
   import pandas as pd
   from mlsynth import PROXIMAL

   TRUE = 2.0

   def gen(T, rng, nU=2, misspecify=False):
       """Liu-Tchetgen Tchetgen-Varjao reference DGP (simulation/normal)."""
       T0 = T // 2
       U = np.empty((T, nU)); U[0] = rng.normal(size=nU)
       for t in range(1, T):
           U[t] = 0.1 * U[t - 1] + 0.9 * rng.normal(size=nU)
       sigU = U.sum(1)
       signal = sigU if not misspecify else sigU + 0.7 * sigU ** 2   # nonlinear -> breaks h-bridge
       Y = TRUE * (np.arange(1, T + 1) > T0) + 2 * signal + rng.normal(size=T)
       W = 2 * U + rng.normal(size=(T, nU))                          # donor outcomes
       Z = 2 * U + rng.normal(size=(T, nU))                          # donor proxies
       rows = []
       for t in range(T):
           rows.append({"unit": "treated", "time": t, "y": float(Y[t]), "dp": 0.0,
                        "treat": int(t >= T0)})
           for j in range(nU):
               rows.append({"unit": f"d{j}", "time": t, "y": float(W[t, j]),
                            "dp": float(Z[t, j]), "treat": 0})
       return pd.DataFrame(rows), nU

   def fit(df, nU, methods):
       return PROXIMAL({
           "df": df, "outcome": "y", "treat": "treat", "unitid": "unit", "time": "time",
           "methods": methods, "donors": [f"d{j}" for j in range(nU)],
           "vars": {"donorproxies": ["dp"]}, "display_graphs": False,
       }).fit().methods

   # (1) recovery + (2) 95% Wald coverage at T=1000
   acc = {"DR": [], "PIPW": []}; cov = {"DR": 0, "PIPW": 0}
   for r in range(200):
       m = fit(*gen(1000, np.random.default_rng(r)), ["DR", "PIPW"])
       for k in ("DR", "PIPW"):
           acc[k].append(m[k].att)
           cov[k] += abs(m[k].att - TRUE) <= 1.96 * m[k].att_se
   for k in ("DR", "PIPW"):
       print(f"{k:5s} mean ATT={np.mean(acc[k]):.3f}  coverage={cov[k]/200:.0%}")
   # DR    mean ATT=2.007  coverage=91%
   # PIPW  mean ATT=2.007  coverage=99%

   # (3) double robustness: misspecify the outcome bridge -> PI collapses, DR holds
   pi, dr = [], []
   for r in range(120):
       m = fit(*gen(1000, np.random.default_rng(1000 + r), misspecify=True), ["PI", "DR"])
       pi.append(m["PI"].att); dr.append(m["DR"].att)
   print(f"misspecified h:  PI={np.mean(pi):.2f} (collapses)  DR={np.mean(dr):.2f} (holds)")
   # misspecified h:  PI=4.30 (collapses)  DR=1.99 (holds)

.. admonition:: Over-identified / empirical use

   The paper's real analyses (Brazil, Florida, Kansas) use a **separate,
   larger set of proxy units** ``Z`` than donors ``W``, which makes the
   GMM *over-identified*. mlsynth's DR/PIPW are the *just-identified* form
   (``Z`` = the donor proxies, matched to ``W``). In the over-identified
   regime with many near-collinear control-unit instruments, the GMM
   minimizer is ill-conditioned and its value is sensitive to the
   optimizer, so those published point estimates are not bit-reproducible
   across languages. We therefore validate DR/PIPW **synthetically** (Path
   B; see *Replication Status*) rather than against the empirical tables.

Replication Status
------------------

.. note::

   **Reference-code validation (Path A).** ``mlsynth``'s PI, PIS and
   PIPost were checked **value-for-value** against the authors' reference
   implementation (``freshtaste/proximal``) on identical data-generating
   draws. Both the **ATT and the GMM/HAC standard error** match to machine
   precision for all three methods. A coverage Monte Carlo confirms the
   inference is correct: nominal-95% Wald intervals attain ≈ 93.8%
   coverage (PI), identical to the reference -- restored from a 63.8%
   undercoverage caused by an earlier Jacobian-scaling bug in the GMM
   sandwich.

   **Empirical (Path A, Panic of 1907).** Running ``mlsynth`` on the trust
   panel (see *Empirical Illustration: Panic of 1907*) reproduces the
   full-window Table 3 of [LiuTchetgenVar]_ to within rounding: PI -1.148
   vs. -1.138, PI-S -1.148 vs. -1.134, PI-P -1.220 vs. -1.220.

   **SPSC (Path A, single proxy).** SPSC is a value-for-value port of the
   authors' reference R package (``github.com/qkrcks0218/SPSC``) and
   reproduces its Panic-of-1907 Table 3: SPSC-NoDT ATT -0.812 / SE 0.085
   (paper -0.813 / 0.084) and SPSC-DT ATT -0.815 / SE 0.067 (paper -0.816 /
   0.066). The tiny ATT gap is one donor (48 vs. 49: the reference keeps a
   unit that is unbalanced in this build). The conformal prediction
   intervals of [SPSC]_ are also ported and reproduce the average interval
   width of the paper's Figure 3 (≈ 0.07 for SPSC-DT).

   **Simulation (Path B).** The robustness claim of [LiuTchetgenVar]_ Sec.
   4.1 reproduces: under a trending latent factor
   (:math:`\boldsymbol{\lambda}_t \sim N(\log t, 1)`), classical SC and
   SC-with-surrogates lose all coverage (→ 0%) while PI/PIS/PIPost remain
   near nominal with low MSE; PIS attains the lowest MSE in most cells. See
   *Example* for a one-draw illustration.

   **DR & PIPW (Path B) -- runnable proof, not a claim.** The DR/PIPW
   agreement is demonstrated by the **runnable Monte Carlo above** (the
   *Doubly Robust* section), which draws from the reference implementation's
   own DGP (``DR_Proximal_SC/simulation/normal``, ``true.ATE = 2``) and
   drives the packaged ``PROXIMAL``. At ``T = 1000`` over 200 reps both
   estimators recover the truth -- ``DR`` and ``PIPW`` mean ATT ``= 2.007``
   (sd 0.11) -- with Wald coverage of 91% (``DR``) and 99% (``PIPW``) against
   the 95% nominal. The **double-robustness** headline also reproduces:
   misspecifying the outcome bridge (``Y`` nonlinear in the confounder)
   biases the outcome-only ``PI`` estimator (mean ATT ``≈ 4.3``) while
   ``DR`` stays at ``1.99``, rescued by the correct treatment-confounding
   bridge. Copy-paste the block to re-derive these numbers. The
   over-identified empirical analyses (Brazil/Florida/Kansas) are not
   bit-reproducible cross-language (ill-conditioned GMM; see the admonition
   above), so DR/PIPW rest on this synthetic validation.

   Per the project's replication contract
   (``agents/agents_estimators.md``), PROXIMAL is considered validated on
   the strength of the machine-precision agreement with the reference code
   plus the reproduced simulation behavior.

Core API
--------

.. automodule:: mlsynth.estimators.proximal
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.PROXIMALConfig
   :members:
   :undoc-members:

Result Containers
-----------------

``PROXIMAL.fit()`` returns a
:class:`~mlsynth.utils.proximal_helpers.structures.PROXIMALResults`, whose
``pi`` / ``pis`` / ``pipost`` fields each hold a
:class:`~mlsynth.utils.proximal_helpers.structures.ProximalMethodFit`
(counterfactual, gap, ATT, GMM/HAC standard error, pre/post RMSE, donor
weights) for the methods that ran. The prepared panel is exposed as a
:class:`~mlsynth.utils.proximal_helpers.structures.PROXIMALInputs`.

.. automodule:: mlsynth.utils.proximal_helpers.structures
   :members:
   :undoc-members:
   :show-inheritance:

Helper Modules
--------------

Data preparation -- pivots the long panel, builds the donor/surrogate
outcome and proxy matrices, residualizes contaminated surrogates, and packs
everything into the typed
:class:`~mlsynth.utils.proximal_helpers.structures.PROXIMALInputs`.

.. automodule:: mlsynth.utils.proximal_helpers.setup
   :members:
   :undoc-members:

The Bartlett kernel and HAC long-run variance shared by the PI family.

.. automodule:: mlsynth.utils.proximal_helpers.inference
   :members:
   :undoc-members:

Each estimator lives in its own subpackage so new proximal methods can be
added as new subpackages. ``pi``, ``pis`` and ``pipost`` are the two-proxy
GMM family; ``spsc`` is the single-proxy ridge-GMM plus conformal
inference.

.. automodule:: mlsynth.utils.proximal_helpers.pi.estimation
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.proximal_helpers.pis.estimation
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.proximal_helpers.pipost.estimation
   :members:
   :undoc-members:

Single Proxy Synthetic Control: ridge-GMM with the treated unit's own
(optionally detrended) outcome as the instrument, plus the GMM/HAC ATT
standard error and conformal prediction intervals.

.. automodule:: mlsynth.utils.proximal_helpers.spsc.estimation
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.proximal_helpers.spsc.conformal
   :members:
   :undoc-members:

The doubly-robust family: shared confounding-bridge fits and the GMM
sandwich (``bridges``), the doubly-robust estimator (``dr``), and the
treatment-bridge weighting estimator (``pipw``).

.. automodule:: mlsynth.utils.proximal_helpers.bridges
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.proximal_helpers.dr.estimation
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.proximal_helpers.pipw.estimation
   :members:
   :undoc-members:

Drives the requested methods on a prepared panel and assembles the
per-method fits.

.. automodule:: mlsynth.utils.proximal_helpers.orchestration
   :members:
   :undoc-members:

The trajectories-and-gap overlay plot across methods.

.. automodule:: mlsynth.utils.proximal_helpers.plotter
   :members:
   :undoc-members:
