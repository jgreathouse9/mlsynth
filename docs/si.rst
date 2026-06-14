Synthetic Interventions (SI)
============================

.. currentmodule:: mlsynth

When to Use This Estimator
--------------------------

Classical synthetic control answers a single counterfactual question: *what
would the treated unit have done under the status quo?* Synthetic Interventions
(SI), due to Agarwal, Shah and Shen (2026) [SI]_, generalises this to many
interventions at once: *what would a unit have done under each of several
alternative treatments it did not receive?*

The motivating example is the canonical Proposition 99 tobacco study. In 1989
California enacted a large anti-tobacco program. Over the following decade other
states instead adopted anti-tobacco programs (Arizona, Massachusetts,
Oregon, Florida) or raised cigarette taxes (Alaska, Hawaii, Maryland,
Michigan, New Jersey, New York, Washington). SI lets you ask not only "what
would California have done under the status quo?" but also "what would
California's cigarette sales have been had it instead *raised taxes* or *run a
program*?" — by borrowing the post-treatment trajectories of the states that
actually did those things.

Reach for SI when:

* You have multiple, distinct interventions across units (policies, product
  launches, treatment arms) and want to compare a focal unit's counterfactual
  across them, not just against control.
* A low-rank factor structure is plausible. SI rests on a latent-factor
  (interactive fixed-effects) model in which each unit's latent loadings are
  *shared across time and across interventions* — the structural bridge that
  lets weights learned on pre-period control data transfer to post-period
  outcomes under a different intervention.
* You want valid inference. The bias-corrected SI-PCR estimator
  (the default here) is asymptotically normal, yielding closed-form
  confidence intervals — a feature absent from most SC point estimators.

The flip side: SI assumes no interference and no dynamic effects
(Assumption 1), and its factor model assumes each donor pool is observed under a
*single* intervention throughout the post-period. Staggered adoption is not
modelled (the paper only approximates it with a common post-window).

Notation
--------

Units are indexed by :math:`i \in \mathcal{N} \coloneqq \{1, \dots, N\}` and
time by :math:`t \in \mathcal{T} \coloneqq \{1, \dots, T\}`, with interventions
indexed by :math:`d \in \{0, 1, \dots, D\}` (and :math:`d = 0` the status quo).
Let :math:`y_{it}(d)` be the potential outcome of unit :math:`i` at time
:math:`t` under intervention :math:`d`. The intervention takes effect after
period :math:`T_0`: the pre-period
:math:`\mathcal{T}_1 \coloneqq \{t \in \mathcal{T} : t \le T_0\}` has all units
under control, and the post-period
:math:`\mathcal{T}_2 \coloneqq \{t \in \mathcal{T} : t > T_0\}` has length
:math:`T_1 \coloneqq T - T_0`. Write
:math:`\mathcal{N}(d) \subseteq \mathcal{N}` for the set of :math:`N_d` units
assigned to intervention :math:`d` after :math:`T_0` (the donor pool for
:math:`d`).

For a focal target unit :math:`i`, write
:math:`\mathbf{y}_{\text{pre}, i} \coloneqq (y_{it}(0))_{t \le T_0}
\in \mathbb{R}^{T_0}` for its pre-period (control) outcomes and
:math:`\mathbf{Y}_{\text{pre}, \mathcal{N}(d)} \in \mathbb{R}^{T_0 \times N_d}`
for the donor pool's pre-period (control) outcomes. The estimand is

.. math::

   \theta_i(d) \;\coloneqq\; \tfrac{1}{T_1}\textstyle\sum_{t \in \mathcal{T}_2} y_{it}(d),

the focal unit's average post-period outcome had it received intervention
:math:`d`.

Assumptions
-----------

SI inherits the SC identification conditions and adds one structural assumption
that does the real work — invariance of unit factors *across interventions*.
Each is stated with a plain-language remark.

Assumption 1 (SUTVA / observation pattern). Pre-treatment, every unit is
observed under control (:math:`y_{it} = y_{it}(0)` for :math:`t \in \mathcal{T}_1`);
post-treatment, each unit is observed under its assigned intervention
(:math:`y_{it} = y_{it}(d)` for :math:`t \in \mathcal{T}_2`, :math:`i \in \mathcal{N}(d)`).

*Remark.* This rules out spillovers between units and, with the static factor
model below, dynamic (carry-over) treatment effects. Estimating a
counterfactual is then exactly a tensor-completion problem: imputing the
unobserved :math:`(i, t, d)` cells of the potential-outcome tensor.

Assumption 2 (tensor factor model — the SI assumption). Each potential
outcome factorises as

.. math::

   y_{it}(d) \;=\; \langle \mathbf{u}_t(d),\, \mathbf{v}_i \rangle + \varepsilon_{it}(d),
   \qquad \mathbf{u}_t(d), \mathbf{v}_i \in \mathbb{R}^r,

where the unit factors :math:`\mathbf{v}_i` are invariant across *both* time and
interventions, and only the time-intervention factors :math:`\mathbf{u}_t(d)`
depend on :math:`d`.

*Remark.* This is the crux. In single-intervention SC the unit factors are
invariant across *time*, which lets pre-period weights predict post-period
control outcomes. SI strengthens this to invariance across *interventions*:
weights learned from pre-period control data can be applied to post-period
outcomes under a different intervention :math:`d`. Conceptually, each unit
has stable intrinsic traits (:math:`\mathbf{v}_i`) that any intervention acts
on through :math:`\mathbf{u}_t(d)`.

Assumption 3 (low rank). The signal
:math:`\mathbb{E}[\mathbf{Y}_{\text{pre}, \mathcal{N}(d)} \mid \mathcal{E}]` is
low rank (rank :math:`r_{\text{pre}} \le r`).

*Remark.* This is what makes the spectral (PCR) denoising step meaningful: the
large singular values of the noisy donor pre-matrix capture the signal, the
small ones capture noise.

Assumption 4 (span / linear span condition). The focal unit's factor lies in
the span of the donor pool's factors, so a weight vector
:math:`\mathbf{w}^{(i,d)}` exists with
:math:`\mathbf{v}_i = \sum_{j \in \mathcal{N}(d)} w^{(i,d)}_j \mathbf{v}_j`.

*Remark.* The multi-intervention analogue of "the treated unit lies in the
convex hull of the donors." A strong pre-period fit (small
:math:`\|\mathbf{y}_{\text{pre},i} - \mathbf{Y}_{\text{pre},\mathcal{N}(d)} \mathbf{w}\|`)
is the data-driven sanity check; a poor fit warns that the span condition or
low-rank structure fails.

Assumption 5 (homoskedastic noise). The idiosyncratic noise is mean-zero
with common variance :math:`\sigma^2`.

*Remark.* Used only for the variance estimate :math:`\widehat\sigma^2` (eq. 14)
behind the confidence interval; the point estimator does not need it.

Assumptions 6-8 (regularity for normality). Bounded factors / sub-Gaussian
noise / incoherence-type conditions, plus the rate constraints :math:`N_d < T_0`
and :math:`T_1 = \tilde o(\min\{r_{\text{pre}}^{-3} N_d,\, r_{\text{pre}}^{-1}
\sqrt{T_0}\})`.

*Remark.* These are what Theorem 2 needs for asymptotic normality. The practical
content: the post-window :math:`T_1` must be *small* relative to the
pre-window :math:`T_0`, and the target must have a non-vanishing pre-period
signal. The Monte Carlo below shows the CI's coverage degrade exactly when
:math:`T_1` is pushed too large.

When the assumptions bind: practical diagnostics
------------------------------------------------

Assumptions 1-8 are stated above in their structural form. Here is what each
looks like in a real dataset, and what to check in the SI fit object before
trusting an arm-level counterfactual.

(a) SUTVA / no spillovers / no carry-over (A1). SI assumes each unit's
    post-period outcome under :math:`d` is a function of :math:`d` only --
    no influence from other units' interventions, no dynamic effect from
    pre-period exposure.

    *Plausibly violated when* the interventions are geographically or
    socially adjacent (state-level tobacco programs whose advertising
    crosses borders; vertically linked markets), or when treatment has a
    persistent effect that bleeds into the post-window. *Diagnostic*:
    re-run SI dropping donors that are geographic / network neighbours of
    treated units; large changes in an arm's counterfactual flag
    interference. For genuine spillovers switch to :doc:`spillsynth` or
    :doc:`spsydid`; for dynamics switch to :doc:`tasc`/:doc:`dscar`.

(b) Factor invariance of unit loadings across interventions (A2 -- the
    SI assumption). Each unit's :math:`\mathbf{v}_i` is the same whether observed
    under control or under :math:`d`. SI's transfer step is precisely the
    statement that weights learned on pre-period control data work to
    impute post-period outcomes under :math:`d`.

    *Plausibly violated when* the intervention *changes who the donors
    are*: a tax raises the price elasticity itself for tax-state
    consumers, a marketing program builds new audience segments inside
    program states. Once :math:`\mathbf{v}_i` shifts after :math:`T_0`, the
    pre-period weights are stale. *Diagnostic*: this is the silent
    failure -- a pre-fit can look excellent while the counterfactual is
    biased, because the pre-data is all under control. The empirical
    cross-check (Section 6.2 of the paper) is to hold out a slice of the
    donor pool's *post-period under d* outcomes and verify the
    pre-period weights also reproduce those; ``SI.fit`` exposes the
    per-arm validation coverage (e.g. 26/38, 6/7, 3/5 in the Prop 99
    case study). Low validation coverage for an arm is the only honest
    flag for an A2 failure on that arm.

(c) Low-rank donor structure (A3). The donor pre-matrix is
    approximately low-rank, so the spectral truncation kept by
    Gavish-Donoho separates signal from noise.

    *Plausibly violated when* the spectrum decays slowly (no clear gap),
    or when individual donors carry idiosyncratic noise comparable to
    the signal. *Diagnostic*: print
    ``arm.singular_values`` (or recompute the SVD of the donor
    pre-matrix) and look for a sharp gap; a slow decay means the
    Gavish-Donoho cut is somewhere in the noise floor. If
    ``rank_method="donoho"`` keeps ``k`` close to ``min(T0, N_d)``, the
    low-rank story has failed and SI-PCR is essentially OLS on the donor
    columns.

(d) Span condition (A4). The focal unit's :math:`\mathbf{v}_i` lies in the
    span of the donor pool's loadings under :math:`d`.

    *Plausibly violated when* the focal unit is structurally different
    from every donor under intervention :math:`d` -- California (a
    coastal mega-state) against a donor pool that happens to be small
    interior states. *Diagnostic*: inspect the per-arm pre-period RMSE
    of :math:`\mathbf{y}_{\text{pre}, i}` against
    :math:`\mathbf{Y}_{\text{pre}, \mathcal{N}(d)} \widehat{\mathbf{w}}`; a
    visibly poor pre-fit on a particular arm means that
    arm's span condition is failing. This is the loud failure mode
    -- it shows up directly in pre-fit residuals.

(e) Homoskedastic noise (A5). Only the variance estimate
    :math:`\widehat\sigma^2` and hence the CI width depend on this; the
    point estimator is unaffected.

    *Plausibly violated when* donor variance is heavily heterogeneous
    (a quiet donor next to a noisy one). *Diagnostic*: per-donor
    pre-period residual variance; if it spans an order of magnitude,
    treat the CI as approximate. Switching to ``variance="time_iv"`` or
    ``"double"`` (the default) is more robust than the main-text
    ``"units"`` estimate.

(f) Rate condition on :math:`T_1` vs. :math:`T_0` (A6-8).
    Theorem 2 needs :math:`T_1` *small* relative to :math:`T_0` and
    factors that stay bounded. The note further below shows coverage
    collapsing from 93% to 52% when the post-window is pushed and
    factors are nonstationary.

    *Plausibly violated when* you want to track an effect over many
    post-periods (multi-year follow-up after a one-shot policy change),
    or when factors trend like a random walk (financial / business-cycle
    panels). *Diagnostic*: refit with the post-window cropped to the
    first few periods; if the counterfactual changes materially, the
    long-horizon CI was not protected by Theorem 2. Pair this with
    :doc:`sbc` (a stationary-cycle estimator) if the factor
    nonstationarity is what you suspect.

Graphical demonstration: span condition vs. factor invariance
-------------------------------------------------------------

The decisive distinction in practice is between A4 (the span condition,
which fails *loudly* -- you see it in a poor pre-fit) and A2 (the cross-
intervention factor invariance, which fails *silently* -- pre-fit looks
fine but the post-period counterfactual is wrong). The block below
generates a rank-:math:`r = 2` panel and overlays SI's counterfactual on
the true noiseless one in two regimes side-by-side, holding everything
else fixed.

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from mlsynth.utils.si_helpers.estimation import bias_corrected_fit

   N, T0, T1, r, sigma = 12, 80, 20, 2, 0.5
   T = T0 + T1
   rng = np.random.default_rng(0)

   # Shared time factors. The intervention's effect is a post-period shift in u_t.
   U_ctrl = rng.normal(0.0, 1.0, (T, r))
   U_d = U_ctrl.copy()
   U_d[T0:] += np.array([0.0, 5.0])
   V = rng.normal(0.0, 1.0, (N, r))         # donor loadings (unit factors)

   def fit_panel(v_target_pre, v_target_post, V_donor, seed):
       """One SI fit. v_target_pre / v_target_post are the focal unit's
       loading under control (pre-period) and under intervention d (post)."""
       gen = np.random.default_rng(seed)
       pre_donor   = U_ctrl[:T0] @ V_donor.T + sigma * gen.standard_normal((T0, N - 1))
       pre_target  = U_ctrl[:T0] @ v_target_pre + sigma * gen.standard_normal(T0)
       post_donor  = U_d[T0:] @ V_donor.T + sigma * gen.standard_normal((T1, N - 1))
       omega, w, _ = bias_corrected_fit(pre_donor, pre_target, rank=r)
       pre_fit     = pre_donor[:, omega] @ w
       cf          = post_donor[:, omega] @ w
       truth       = U_d[T0:] @ v_target_post   # noiseless cf under d
       return pre_target, pre_fit, cf, truth

   v_in_span = 0.5 * V[1] + 0.5 * V[2]      # focal loading inside donor span

   # Regime A: A2 and A4 both hold.
   regime_A = fit_panel(v_in_span, v_in_span, V[1:], seed=1)

   # Regime B: A2 violated -- focal unit's loading SHIFTS under d.
   # (Donors still satisfy A2; only the focal unit changes between
   # pre-control and post-d. This is the structural identifying failure.)
   v_target_under_d = v_in_span + np.array([1.5, -1.5])
   regime_B = fit_panel(v_in_span, v_target_under_d, V[1:], seed=2)

   # Regime C: A4 violated -- focal loading far outside donor span.
   v_out_span = np.array([4.0, -4.0])
   regime_C = fit_panel(v_out_span, v_out_span, V[1:], seed=3)

   fig, ax = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
   t_pre, t_post = np.arange(T0), np.arange(T0, T)
   titles = [
       "A2 & A4 hold:\nSI cf hugs truth",
       "A2 violated (silent):\npre-fit OK, cf badly biased",
       "A4 violated (loud):\npre-fit poor -- visible flag",
   ]
   for a, (pre_t, pre_f, cf, truth), title in zip(ax, [regime_A, regime_B, regime_C], titles):
       a.plot(t_pre, pre_t, "k", lw=1.0, alpha=0.6, label="focal (observed)")
       a.plot(t_pre, pre_f, "C0--", lw=1.2, label="SI pre-fit")
       a.plot(t_post, truth, "k", lw=1.8, label="true cf under d")
       a.plot(t_post, cf, "C3", lw=1.8, label="SI cf under d")
       a.axvline(T0, color="gray", ls=":")
       a.set_title(title); a.set_xlabel("time")
       pre_rmse = float(np.sqrt(((pre_t - pre_f) ** 2).mean()))
       post_err = float(np.abs(cf - truth).mean())
       a.text(0.02, 0.97,
              f"pre RMSE        = {pre_rmse:.2f}\npost |cf-truth| = {post_err:.2f}",
              transform=a.transAxes, va="top", fontsize=9, family="monospace",
              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7"))
   ax[0].set_ylabel("Y")
   ax[0].legend(loc="lower left", fontsize=8)
   plt.tight_layout(); plt.show()

Representative output (seeds fixed in the snippet)::

   Regime A (assumptions hold):  pre RMSE = 0.53   post |cf - truth| = 0.24
   Regime B (A2 violated):       pre RMSE = 0.55   post |cf - truth| = 7.46
   Regime C (A4 violated):       pre RMSE = 1.07   post |cf - truth| = 1.03

The three panels read as follows.

* Left (assumptions hold). Pre-fit is tight and the SI counterfactual
  sits on top of the truth in the post-period; the mean absolute gap is at
  the noise floor.
* Middle (A2 violated -- silent). The focal unit's loading
  :math:`\mathbf{v}_i` *changes* between the pre-period (under control) and the
  post-period (under :math:`d`). Pre-fit is just as tight as in
  Regime A (pre RMSE 0.55 vs 0.53) -- the pre-data is all under control,
  so the shift in the focal unit's loading is invisible to it. But the
  weights learned on pre-control data target the *wrong* loading for the
  post-d projection, and the SI counterfactual misses the truth by an
  order of magnitude (post error ~7.5 vs noise floor ~0.5). *Nothing in
  the pre-period fit warns of this*. The only practical defence is the
  arm-level validation-coverage check the case study uses: hold out part
  of the donor pool's post-period under :math:`d` and verify that the
  pre-period weights reproduce it. Low validation coverage for an arm =
  A2 failure on that arm.
* Right (A4 violated -- loud). The focal unit's loading is
  structurally outside the donor span. Pre-fit residuals are visibly
  large already (pre RMSE 1.07, twice the noise floor), and the
  post-period counterfactual is correspondingly wrong. This failure
  mode is detectable from pre-data alone, so it is the safer one.

The take-away: a tight pre-period fit is necessary but not sufficient
for trusting an SI counterfactual on a given arm. Always pair it with
arm-level validation coverage before reading off post-period effects.

When not to use SI
----------------------

* You only need a single counterfactual against control. SI's whole point
  is comparing a focal unit's counterfactual across *several* alternative
  interventions. If you only need the status-quo counterfactual (the
  classical SC question), :doc:`tssc`, :doc:`fdid`, *canonical SCM*, or
  :doc:`fma` are simpler and have stronger small-:math:`T_0` properties.

* Donor pool for an arm is too small for the rank. SI's bias-corrected
  inference requires :math:`N_d > k` (and ideally :math:`N_d` somewhat
  bigger than the selected rank). With three or four donors per arm and
  a Gavish-Donoho rank near that, the rank-complete subset
  :math:`\Omega` is the entire donor set and the variance is unstable.
  Either pool arms (broader policy categories), prune the rank by
  hand (``rank_method="fixed"``), or step down to a single-arm SC.

* Spillovers across treatment arms. Interventions whose effect
  propagates across units (a tobacco program's media campaign reaching
  tax-only states, a marketing intervention shared via social graph)
  break A1 at the inter-arm boundary, not just within an arm. SI cannot
  fix this; use :doc:`spillsynth` or :doc:`spsydid` and accept that
  identification is now at the aggregate (not per-arm) level.

* Dynamic / carry-over treatment effects. SI's tensor model is
  static: :math:`u_t(d)` depends on :math:`d` and :math:`t` but the
  treatment has no within-unit dynamics. Persistent effects (a tax that
  trains consumers over time) or treatment-effect dynamics on the
  treated belong in :doc:`tasc` (state-space) or :doc:`dscar`
  (autoregressive treated process).

* Staggered adoption with a wide reporting window. SI's framing
  treats each donor as observed under a *single* intervention
  throughout the post-period; the paper and ``mlsynth`` approximate
  staggered designs with a common short post-window (1999-2002 in the
  Prop 99 case). If your post-window must span many years of
  cumulative adoption -- some donors enter the intervention years
  apart -- SI's identification gradually erodes. Use the staggered
  SC variants in *FECT* or :doc:`sdid` instead.

* No low-rank factor structure. When the donor spectrum has no
  clear gap (e.g. each donor genuinely idiosyncratic), the
  Gavish-Donoho cut keeps too many components and SI-PCR essentially
  fits OLS. In that regime a covariate-balancing estimator
  (:doc:`microsynth` if you have unit-level data, a balancing-aware
  aggregate alternative otherwise) is closer to the truth than
  forcing a low-rank fit.

* Continuous or multi-valued treatment. SI partitions units into
  arms by discrete intervention label :math:`d`. Continuous dose
  (minimum wage, ad spend, drug dosage) needs the GSC framework in
  :doc:`ctsc`.

* Long post-window with nonstationary factors. As the note below
  shows, coverage collapses (~93% → ~52%) when :math:`T_1` grows and
  factors drift like a random walk. If your application requires
  many-period post-windows on trending series, either crop the
  post-window for inference and report the rest as descriptive, or
  switch to a stationary-cycle approach (:doc:`sbc`).

* You need per-period or per-unit causal estimates inside an arm.
  SI delivers the *focal unit's* counterfactual under each
  intervention, not unit-specific effects across the donor pool. For
  heterogeneous treatment effects across donors within an arm, use
  :doc:`ctsc` (which estimates unit-specific slopes).

Mathematical Formulation
------------------------

The SI Estimator (Proposition 1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Under Assumptions 1-4 there is a weight vector :math:`\mathbf{w}^{(i,d)}` such
that the estimand is recovered from donor-pool outcomes under :math:`d`, and the
weights are identified from pre-period control data alone. The SI estimator is
two steps:

.. math::

   \widehat{\mathbf{w}}^{(i,d)} \in \operatorname*{argmin}_{\mathbf{w} \in \mathcal{W}}
   \; \| \mathbf{y}_{\text{pre}, i} - \mathbf{Y}_{\text{pre}, \mathcal{N}(d)}\, \mathbf{w} \|_2^2,
   \qquad
   \widehat\theta_i(d) \coloneqq \tfrac{1}{T_1}\sum_{t \in \mathcal{T}_2}\sum_{j \in \mathcal{N}(d)} \widehat w^{(i,d)}_j\, y_{jt}(d).

The choice of constraint set :math:`\mathcal{W}` recovers the usual SC variants
(simplex, ridge, lasso, OLS). ``mlsynth`` implements the PCR variant.

SI-PCR (eq. 10)
^^^^^^^^^^^^^^^

Let the SVD of the donor pre-matrix be
:math:`\mathbf{Y}_{\text{pre}, \mathcal{N}(d)} = \sum_\ell
\widehat s_\ell \widehat{\mathbf{u}}_\ell \widehat{\mathbf{v}}_\ell^\top`. Keeping the top :math:`k` components,

.. math::

   \widehat{\mathbf{w}}^{(i,d)} \;\coloneqq\; \Big( \textstyle\sum_{\ell=1}^{k} (1/\widehat s_\ell)\,
   \widehat{\mathbf{v}}_\ell \widehat{\mathbf{u}}_\ell^\top \Big) \mathbf{y}_{\text{pre}, i}.

SI-PCR projects the donor pre-matrix onto its top-:math:`k` principal subspace
(denoising it under Assumption 3) and regresses the target onto the result. The
rank :math:`k` is chosen by the Gavish-Donoho optimal hard threshold. The
default ``rank_method="donoho"`` reproduces the authors' exact rule (the
:math:`\omega(\beta)` approximation evaluated at :math:`\beta = T_0/N_d`);
``"usvt"`` is the same threshold at the canonical ``min/max`` aspect ratio,
``"cumvar"`` keeps a spectral-energy fraction, and ``"fixed"`` takes an explicit
:math:`k`. SI-PCR reuses the HSVT primitives shared with :doc:`clustersc`.

Bias-Corrected SI-PCR and Inference (Section 4.3)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Plain SI-PCR is consistent (Corollary 1) but converges too slowly for
normality, because spreading weight across many near-collinear donors *dilutes*
the weight norm and deflates the variance term. The bias-corrected estimator
fixes this by restricting to a rank-complete donor subset :math:`\Omega
\subset \mathcal{N}(d)` with :math:`|\Omega| = k` columns of full rank, and
fitting by pseudo-inverse:

.. math::

   \widehat{\mathbf{w}}^{(i,d,\Omega)} \coloneqq (\mathbf{Y}^k_{\text{pre}, \Omega})^{+}\, \mathbf{y}_{\text{pre}, i},

where :math:`\mathbf{Y}^k` is the rank-:math:`k` approximation. This is a *second* layer
of (structured) sparsity: contributions outside :math:`\Omega` are zeroed by
explicit model selection, concentrating weight along independent directions and
stabilising the weight norm. ``mlsynth`` selects :math:`\Omega` by
column-pivoted QR on the denoised donor matrix.

The estimator is then asymptotically normal (Theorem 2):
:math:`\sqrt{T_1}\,(\widehat\theta_i^\Omega(d) - \theta_i(d)) / (\sigma
\|\mathbf{w}\|_2) \to \mathcal{N}(0, 1)`, giving the closed-form confidence interval

.. math::

   \text{CI}(\alpha) \coloneqq \widehat\theta_i^\Omega(d) \;\pm\;
   z_{\alpha/2}\, \widehat\sigma\, \frac{\| \widehat{\mathbf{w}}^{(i,d,\Omega)} \|_2}{\sqrt{T_1}},
   \qquad
   \widehat\sigma^2 \coloneqq \frac{\| (\mathbf{I} - \widehat{\mathbf{U}}_k \widehat{\mathbf{U}}_k^\top)\, \mathbf{y}_{\text{pre}, i}\|_2^2}{T_0 - k},

where :math:`\widehat{\mathbf{U}}_k` are the left singular vectors of the
rank-:math:`k` donor approximation (eq. 14) and the noise estimate is the
residual of regressing the target's pre-period onto the donor subspace.

``mlsynth`` exposes three noise-variance estimators via ``variance``: the
main-text ``"units"`` estimator (eq. 14 above), a ``"time_iv"`` estimator from
the donor post-period residual, and the degrees-of-freedom-weighted ``"double"``
combination (the default, matching the authors' code). The interval can be the
eq.-13 confidence interval (``interval="confidence"``) or the wider prediction
interval (``interval="prediction"``, half-width
:math:`z_{\alpha/2}\widehat\sigma\sqrt{1 + \|\widehat{\mathbf{w}}\|_2^2}/\sqrt{T_1}`) the case
study uses for coverage validation.

Monte Carlo: Coverage of the Confidence Interval
------------------------------------------------

The block below draws low-rank panels (a focal unit plus a donor pool sharing
:math:`r` latent factors), fits the bias-corrected estimator, and checks whether
the CI covers the *true* (noiseless) counterfactual mean
:math:`\theta_i(d)`. The focal unit receives no genuine effect, so the
counterfactual is its own noiseless post-period mean.

.. code-block:: python

   import numpy as np
   from mlsynth.utils.si_helpers.estimation import bias_corrected_fit

   rng = np.random.default_rng(0)
   N, T0, T1, r, sigma = 10, 80, 4, 3, 1.0     # paper regime: T1 small vs T0

   cov = 0
   reps = 600
   for _ in range(reps):
       T = T0 + T1
       F = rng.normal(0, 1, (T, r))            # bounded (stationary) factors
       lam = rng.normal(0, 1, (N, r))
       L = lam @ F.T
       Y = L + sigma * rng.standard_normal((N, T))
       donor_pre, target_pre = Y[1:, :T0].T, Y[0, :T0]
       omega, w, sig = bias_corrected_fit(donor_pre, target_pre, rank=r)
       theta_hat = (Y[1:, T0:].T[:, omega] @ w).mean()
       theta_true = L[0, T0:].mean()
       half = 1.96 * sig * np.linalg.norm(w) / np.sqrt(T1)
       cov += theta_hat - half <= theta_true <= theta_hat + half
   print(f"95% CI coverage: {cov / reps:.3f}")   # ~0.933

Under the paper's regime (:math:`T_0 = 80`, :math:`T_1 = 4`, bounded factors) the
empirical coverage is 0.933, close to the nominal 0.95 — the CI is valid.

.. note::

   Coverage degrades sharply when Theorem 2's conditions are violated. Repeating
   the experiment with random-walk (nonstationary) factors and a large
   post-window (:math:`T_1 = 10` vs :math:`T_0 = 30`) drops coverage to
   :math:`\approx 0.52`: weight-estimation error multiplied by an unbounded,
   growing post-period signal swamps the variance the CI accounts for. The
   practical lesson mirrors the paper — keep :math:`T_1` short relative to
   :math:`T_0`, which is also why the empirical study below fixes a short
   post-window.

Empirical Application: Proposition 99 (California)
--------------------------------------------------

We reproduce the paper's case study (Section 6) on the 50-state cigarette-sales
panel (1970-2015): California (focal) under three interventions — control
(38 status-quo states), a cigarette-tax increase (Alaska, Hawaii, Maryland,
Michigan, New Jersey, New York, Washington), and an anti-tobacco program
(Arizona, Massachusetts, Oregon, Florida). Following the paper, weights are fit
on 1970-1988 (:math:`T_0 = 19`) and the counterfactual is reported over the
common 1999-2002 window (:math:`T_1 = 4`).

.. code-block:: python

   import numpy as np
   import pandas as pd
   from mlsynth import SI

   raw = pd.read_csv(
       "https://raw.githubusercontent.com/jehangiramjad/tslib/"
       "refs/heads/master/tests/testdata/prop99.csv"
   )
   raw = raw[(raw.Year >= 1970) & (raw.Year <= 2015)]
   raw = raw[raw.SubMeasureDesc == "Cigarette Consumption (Pack Sales Per Capita)"]
   d = raw[["LocationDesc", "Year", "Data_Value"]].rename(
       columns={"LocationDesc": "state", "Year": "year", "Data_Value": "cigsale"})
   d = d[d.state != "District of Columbia"]
   d = d[(d.year <= 1988) | ((d.year >= 1999) & (d.year <= 2002))]   # fit + report

   tax = ["Alaska", "Hawaii", "Maryland", "Michigan", "New Jersey", "New York", "Washington"]
   program = ["Arizona", "Massachusetts", "Oregon", "Florida"]  # California is a program state
   treated = set(tax) | set(program) | {"California"}
   d["control"] = (~d.state.isin(treated)).astype(int)
   d["taxes"]   = d.state.isin(tax).astype(int)
   d["program"] = d.state.isin([s for s in program if s != "California"] + ["California"]).astype(int)
   d["Prop99"]  = ((d.state == "California") & (d.year >= 1999)).astype(int)

   res = SI({
       "df": d, "outcome": "cigsale", "unitid": "state", "time": "year",
       "treat": "Prop99", "inters": ["control", "taxes", "program"],
       "interval": "prediction", "display_graphs": True,
   }).fit()

   for iv, arm in res.arms.items():
       lo, hi = arm.cf_mean_ci
       print(f"{iv:>8}: k={arm.selected_rank}  cf={arm.cf_mean:.1f}  95% PI=({lo:.1f}, {hi:.1f})")

The Gavish-Donoho threshold selects k = 5 for the control donor pool and
k = 1 for both the tax and program pools — *exactly* the ranks the paper
reports (Section 6.2.1) — and the bias-corrected estimator's prediction interval
matches the published numbers:

.. list-table:: California's counterfactual per-capita sales, 1999-2002 (95% prediction interval)
   :header-rows: 1
   :widths: 22 8 18 22

   * - Intervention
     - k
     - Counterfactual
     - 95% PI
   * - Status quo (control)
     - 5
     - 75.8
     - (70.9, 80.6)
   * - Tax increase
     - 1
     - 57.5
     - (48.0, 67.1)
   * - Anti-tobacco program
     - 1
     - 59.1
     - (49.3, 68.9)

The reading mirrors the paper: California's *observed* 1999-2002 sales (~50
packs) sit below all three counterfactuals, and the control counterfactual
(75.8) is far higher than the tax (57.5) or program (59.1) ones — i.e. relative
to having done nothing, Prop 99 cut sales sharply, while relative to a tax or a
program the additional effect is modest. The tax and program counterfactuals
overlap heavily, consistent with the paper's conclusion that the two policy
levers deliver similar trajectories.

Replication against the authors' code (Path A)
----------------------------------------------

Per the project's replication contract, SI is checked against the authors'
own published code (``opre.2025.1590.cd``), not merely against the paper's
prose. Running the authors' functions and ``mlsynth``'s ``si_helpers`` on
*identical* inputs gives machine-precision agreement on every primitive, both
simulation studies, and the case-study tables:

.. list-table:: SI Path-A replication: ``mlsynth`` vs. the authors' code
   :header-rows: 1
   :widths: 40 30 30

   * - Quantity
     - Comparison
     - Result
   * - SI-PCR weights (eq. 10)
     - 300 random panels, max\|diff\|
     - ``2.2e-16``
   * - Bias-corrected weights (eq. 12)
     - 300 random panels, max\|diff\|
     - ``0``
   * - Variance estimators (units/time-iv/double)
     - 300 random panels, max\|diff\|
     - ``< 1e-15``
   * - Donoho rank selection
     - 300 random panels
     - ``0`` mismatches
   * - Consistency sim (Sec 5.1), :math:`|\widehat\theta-\theta|`
     - :math:`T_0 \in \{40,100,400\}`
     - identical to 4 d.p.
   * - Inference sim (Sec 5.2), 95% coverage
     - :math:`T_0 \in \{80,200,600\}`
     - identical (0.922 / 0.891 / 0.947)
   * - Case study, validation coverage
     - control / taxes / program
     - identical (0.684 / 0.857 / 0.600)
   * - Case study, California counterfactual + PI
     - control / taxes / program
     - identical (max\|diff\| ``0``)

The bridge is design rather than luck: ``mlsynth`` reuses the same HSVT
truncation, the authors' exact Gavish-Donoho rank rule
(``rank_method="donoho"``, :math:`\beta = T_0/N_d`), QR-pivot subset selection,
pseudo-inverse fit, and degrees-of-freedom-weighted variance
(``variance="double"``).

This side-by-side harness is now a durable benchmark, not a one-time check:
``benchmarks/cases/si_prop99.py`` runs the authors' own code -- vendored verbatim
under ``benchmarks/reference/synth_iv_OR25`` from ``opre.2025.1590.cd`` -- against
mlsynth's public :class:`~mlsynth.SI` API for all five program states under the
control and tax interventions, and confirms agreement to machine precision
(max\|diff\| ``1.4e-14``). Run it with
``python benchmarks/run_benchmarks.py si_prop99``.

The durable replication does not depend on the authors' code. Both paths are
reproduced from public data and mlsynth's own DGPs, and locked in as a test
(:mod:`mlsynth.tests.test_si_replication`):

* Path A (empirical) loads the vendored public pack-sales panel
  (``basedata/prop99_packsales.csv``) and pins the case-study numbers above —
  the :math:`k = 5/1/1` rank selection, California's counterfactuals
  (75.8 / 57.5 / 59.1), and the validation coverage (26/38, 6/7, 3/5).
* Path B (Monte Carlo) reruns the consistency and inference studies on
  mlsynth's own reimplementation of the paper's DGPs
  (:mod:`mlsynth.utils.si_helpers.simulation`), confirming SI-PCR is consistent
  only when the rank condition holds and that the bias-corrected CI's coverage
  rises toward the nominal 95% as :math:`T_0` grows.

.. note::

   The paper does not formally model staggered adoption; like the authors,
   ``mlsynth`` approximates it with a common pre-window and a fixed post-window
   (here 1999-2002). Donor states that adopted their policy after 1989 are,
   strictly, under control for part of that window — a limitation the paper
   flags in Section 6.1.

Core API
--------

.. automodule:: mlsynth.estimators.si
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.SIConfig
   :members:
   :undoc-members:

Result Containers
-----------------

``SI.fit()`` returns an
:class:`~mlsynth.utils.si_helpers.structures.SIResults`, whose ``arms`` maps each
intervention to an :class:`~mlsynth.utils.si_helpers.structures.SIArm` (donor
weights, the counterfactual, the ATT, the selected rank, and -- under bias
correction -- :math:`\widehat\sigma`, the weight norm, and the confidence
intervals), alongside the prepared
:class:`~mlsynth.utils.si_helpers.structures.SIInputs`.

.. automodule:: mlsynth.utils.si_helpers.structures
   :members:
   :undoc-members:
   :show-inheritance:

Helper Modules
--------------

.. automodule:: mlsynth.utils.si_helpers.setup
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.si_helpers.estimation
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.si_helpers.orchestration
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.si_helpers.simulation
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.si_helpers.plotter
   :members:
   :undoc-members:

References
----------

Agarwal, A., Shah, D., & Shen, D. (2026). "Synthetic Interventions: Extending
Synthetic Controls to Multiple Treatments." *Operations Research*
74(2):840-859. See [SI]_.

Abadie, A., Diamond, A., & Hainmueller, J. (2010). "Synthetic Control Methods
for Comparative Case Studies." *Journal of the American Statistical
Association* 105(490):493-505.

Agarwal, A., Shah, D., Shen, D., & Song, D. (2021). "On Robustness of
Principal Component Regression." *Journal of the American Statistical
Association* 116(536):1731-1745.

Gavish, M., & Donoho, D. L. (2014). "The Optimal Hard Threshold for Singular
Values is :math:`4/\sqrt{3}`." *IEEE Transactions on Information Theory*
60(8):5040-5053.
