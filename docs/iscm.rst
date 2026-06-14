Imperfect Synthetic Controls (ISCM)
====================================

.. currentmodule:: mlsynth

Overview
--------

ISCM (Powell, D. (2026). *"Imperfect Synthetic Controls,"* Journal of
Applied Econometrics 41(3):253-264) confronts the synthetic control
method's least defensible assumption: that a perfect synthetic
control exists. The classic SCM requires the treated unit to lie inside
the convex hull of the donors and its pre-treatment path to be matched
*exactly*. With transitory shocks -- noise with non-vanishing variance --
an exact fit is impossible even in expectation, and the convex-hull
condition may simply fail (the treated unit can be more extreme than any
weighted average of donors).

ISCM relaxes this with two ideas:

1. Synthetic controls for every unit. Rather than fitting one
   synthetic control for the treated unit, ISCM builds one for *all*
   units. The treatment effect is then identified even when the treated
   unit is *outside* the convex hull -- because it can still appear *as a
   donor* for control units, and those units' post-treatment residuals
   carry information about the effect (paper eq. 6).
2. Moment conditions robust to transitory shocks. ISCM relies on
   conditions of the form :math:`\sum_{j} w_i^j \mathbb{E}[y_{jt}] =
   \mathbb{E}[y_{it}]` that need only hold in expectation, producing
   asymptotically unbiased estimates as the pre-period grows even when no
   unit fits perfectly in sample.

It adds a data-driven fit metric :math:`a_i` that asymptotically excludes
units lacking a valid synthetic control -- removing the researcher's
eyeball "is the pre-fit good enough" decision -- and an Ibragimov-Muller
inference procedure that stays valid with a tiny donor pool.

The identifying intuition
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Suppose the treated unit (unit :math:`1`) is too extreme to be matched by
any convex combination of controls. A control unit :math:`i` whose
synthetic control *does* place weight :math:`w_i^1 > 0` on unit :math:`1`
will, after treatment, have its synthetic counterfactual contaminated by
the effect: its residual picks up :math:`-w_i^1 \tau`. Since unit
:math:`i` is itself untreated, regressing its residual on its "treatment
exposure" :math:`-w_i^1` recovers :math:`\tau`. ISCM pools this signal
across all such units.

When to use ISCM
^^^^^^^^^^^^^^^^

* The treated unit's pre-period path is not well inside the donor
  convex hull (it trends above/below all donors), so traditional SCM
  produces a visibly poor fit and a biased counterfactual.
* Outcomes are noisy (transitory shocks), so an exact pre-period
  match is implausible and would overfit.
* The donor pool is small, so permutation inference cannot reach
  conventional significance.
* You have a long pre-period (the method's guarantees are asymptotic
  in :math:`T_0`).

Notation
--------

Let the units be :math:`\mathcal{N} \coloneqq \{1, \dots, N\}`, with the
treated unit indexed :math:`1`; because ISCM builds a synthetic control for
*every* unit, a running unit index :math:`i \in \mathcal{N}` denotes the unit
whose synthetic control is being formed, and :math:`j, k \neq i` index its
donors. Time runs over :math:`t \in \mathcal{T} \coloneqq \{1, \dots, T\}`,
1-indexed; the intervention takes effect after period :math:`T_0`, splitting
:math:`\mathcal{T}` into the pre-period
:math:`\mathcal{T}_1 \coloneqq \{t \in \mathcal{T} : t \le T_0\}` (of length
:math:`T_0`) and the post-period
:math:`\mathcal{T}_2 \coloneqq \{t \in \mathcal{T} : t > T_0\}`.

The scalar outcome is :math:`y_{jt}` (unit, then time), with treatment dummy
:math:`d_{jt}` and transitory shock :math:`\epsilon_{jt}`; in Abadie's
potential-outcome notation :math:`y_{jt}^N` is the outcome without the
intervention and :math:`y_{jt}^I` under it. Unit :math:`i`'s donor weights are
:math:`\mathbf{w}_i \in \mathbb{R}^{N_0}` (entries :math:`w_i^j`), constrained
to the unit simplex
:math:`\Delta^{N_0} \coloneqq \{\mathbf{w} \in \mathbb{R}_{\ge 0}^{N_0} :
\|\mathbf{w}\|_1 = 1\}`; the optimiser is :math:`\widehat{\mathbf{w}}_i`. The
per-period, per-unit treatment effect is :math:`\tau_{it}`; the pooled ATT is
:math:`\widehat{\tau}` and the per-unit estimate :math:`\widehat{\tau}_i`. The
data-driven fit metric is :math:`a_i` and the contributing set :math:`C`.

Mathematical Formulation
------------------------

Setup (paper Section 2)
^^^^^^^^^^^^^^^^^^^^^^^^

For :math:`N` units over :math:`T` periods with a latent-factor outcome

.. math::

   y_{it} = \tau_{it}\, d_{it} + L_{it} + \epsilon_{it},
   \qquad L_{it} = \boldsymbol{\lambda}_t^\top \boldsymbol{\mu}_i,

ISCM builds, for every unit :math:`i`, a synthetic control from the
others (paper eq. 5):

.. math::

   \widehat{\mathbf{w}}_i = \operatorname*{argmin}_{\mathbf{w}}
       \sum_{t \le T_0} \Bigl( y_{it} - \sum_{j \ne i} w^j y_{jt} \Bigr)^2,
   \quad w^j \ge 0,\ \sum_{j \ne i} w^j = 1.

Fit metric (paper eq. 14)
^^^^^^^^^^^^^^^^^^^^^^^^^^

Each unit is weighted by how well its synthetic control satisfies the
SCM moment conditions in the pre-period. With residual
:math:`R_{it} = y_{it} - \sum_j \widehat{w}_i^{\,j} y_{jt}` and moment vector
:math:`M_i^k = \tfrac{1}{T_0}\sum_{t \le T_0} R_{it} y_{kt}`,

.. math::

   a_i = \frac{\min_\ell M_\ell^\top M_\ell}{M_i^\top M_i} \in (0, 1],

so the best-fitting unit gets :math:`a_i = 1` and units without a valid
synthetic control get :math:`a_i \to 0` -- they are dropped from the
estimate automatically.

Treatment effect (paper eq. 8 / 15)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With treatment exposure
:math:`E_{it} = d_{it} - \sum_j \widehat{w}_i^{\,j} d_{jt}`, the ATT is the
:math:`a_i`-weighted least-squares slope, pooled over all units and the
post-period:

.. math::

   \widehat{\tau} =
     \frac{\sum_i a_i \sum_{t > T_0} E_{it} R_{it}}
          {\sum_i a_i \sum_{t > T_0} E_{it}^2}
     = \sum_{i \in C} v_i\, \widehat{\tau}_i,
   \quad
   \widehat{\tau}_i = \frac{\sum_{t>T_0} E_{it} R_{it}}{\sum_{t>T_0} E_{it}^2},

where :math:`C` is the contributing set (units with non-zero exposure)
and :math:`v_i = a_i \sum_t E_{it}^2 / \sum_\ell a_\ell \sum_t E_{\ell t}^2`.

Inference (paper Section 5, eq. 16)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ISCM produces one estimate :math:`\widehat{\tau}_i` per contributing
unit. The Ibragimov-Muller approach forms a t-statistic from their
weighted spread and calibrates the p-value with a sign-flip (Rademacher)
randomization test on the weighted deviations
:math:`v_i(\widehat{\tau}_i - \tau_0)`. This is conservative but valid
with very few units -- though note the achievable p-value floor is about
:math:`2/2^{|C|}`, so a handful of contributing units cannot reach
conventional thresholds (exactly the small-donor-pool limitation Powell
highlights).

Scope of this implementation
----------------------------

This follows Powell's *applied* procedure: synthetic controls for all
units come from the traditional SCM (the documented starting point), the
:math:`a_i` weights are formed from the pre-period moment conditions, the
ATT is the :math:`a_i`-weighted least-squares effect, and inference is the
sign-flip test. It does not run the optional continuously-updating
GMM refinement that re-estimates the weights jointly to be orthogonal to
transitory shocks (paper Section 3.2-3.4); the SCM-initialised weights
are that procedure's starting point and deliver the headline
relaxed-convex-hull identification.

Assumptions (Powell 2026)
-------------------------

ISCM trades the canonical SCM's "perfect synthetic control" requirement
for a substantially weaker set of moment conditions on the transitory
shocks. The paper's formal assumptions (Section 4.1):

A1 (Outcomes). :math:`y_{it} = \tau_{it}\, d_{it} + L_{it} +
\epsilon_{it}` with :math:`L_{it}` a fixed (but possibly latent)
systematic component, :math:`y_{it}` continuous, and bounded products
:math:`\lVert y_{it} \epsilon_{jt} \rVert < \infty`. The latent
component nests interactive fixed effects (:math:`L_{it} =
\boldsymbol{\lambda}_t^\top \boldsymbol{\mu}_i`), additive two-way FE
(:math:`L_{it} = \theta_i + \gamma_t`), and other workhorse panel
structures.

*Remark.* The outcome is the standard latent-factor panel: a treatment term
plus a fixed systematic component plus transitory noise. Because :math:`L_{it}`
nests both interactive and additive fixed effects, ISCM does not commit to a
particular panel structure -- it only needs :math:`L_{it}` to be reproducible by
a synthetic control (A2), which is what the next assumption asserts.

A2 (Existence of Synthetic Controls). For every unit :math:`i`,
*either* (a) there exist simplex weights :math:`\mathbf{w}_i` such that
:math:`L_{it} = \sum_{k \ne i} w_i^k L_{kt}` for all :math:`t`,
*or* (b) there exists some other unit :math:`j` with
:math:`w_j^i > 0` such that :math:`L_{jt} = \sum_{k \ne j} w_j^k
L_{kt}` for all :math:`t`.

*Remark.* In words: every unit either has its own convex-hull synthetic
control, or appears as a positive-weight donor in some other unit's synthetic
control. The whole point of ISCM is that (b) suffices for the treated unit -- it
can be too extreme to admit (a) and still be identifiable through (b).

A3 (Independence of transitory shocks). (a)
:math:`\mathbb{E}[\epsilon_{it} \mid \mathbf{d}_i, L_i] = 0` (mean-independence
of the shocks from treatment and the latent component); (b)
:math:`\mathbb{E}[\epsilon_{it} \epsilon_{jt} \mid \mathbf{d}_i, L_i, \mathbf{d}_j,
L_j] = 0` for all :math:`i \ne j` (no contemporaneous cross-unit
correlation in the shocks).

*Remark.* The moment conditions that drive the ISCM estimator rely on
cross-sectional shock independence after conditioning on the latent component. A
common contemporaneous shock across units violates (b) and reintroduces a bias
term the estimator cannot remove.

A4 (Within-unit serial dependence allowed).
:math:`\epsilon_{it}` is a strongly mixing sequence in :math:`t` of
size :math:`-r/(r-1)` for some :math:`r > 1`, with
:math:`\mathbb{E}|\epsilon_{it}|^{r+\delta} < \infty` for some
:math:`\delta > 0`.

*Remark.* ISCM permits serially correlated shocks within a unit (a meaningful
relaxation vs. canonical SC's iid assumption) provided they mix at a uniform
rate. Unit roots and other persistent (non-mixing) structures are ruled out.

A5 (Regularity of the fit weights). If A2(a) holds for unit
:math:`i`, then :math:`a_i(\mathbf{w}) \xrightarrow{p} \bar a_i > 0`.

*Remark.* The data-driven fit metric does not collapse for units that actually
have a valid synthetic control. Convenient (paper footnote 10): holds
straightforwardly for any unit whose pre-period moment distance is bounded away
from zero.

Theorem 4.1 (asymptotic unbiasedness). Under A1-A5,
:math:`\widehat{\tau}_{1t} \xrightarrow{p} \tau_{1t} + V_t`
with :math:`\mathbb{E}[V_t] = 0` as :math:`T_0 \to \infty`. The
estimator is *asymptotically unbiased* but not consistent for a
single post-period -- aggregation across the post-period (eq. 15)
or across multiple treated units (Section 4.4.3) is what drives
the variance term toward zero.

When the assumptions bind: practical diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

(a) Latent component is fixed and continuous (A1).
    The systematic component :math:`L_{it}` is deterministic
    conditional on the unit; the outcome :math:`y_{it}` is continuous.

    *Plausibly violated when* outcomes are binary or low-count
    (handgun-suicide-events per month in a small state can spike
    to zero or single digits; an LPM/Tobit-style nonlinearity
    enters that A1 does not cover). *Diagnostic*: histogram the
    outcome; heaps at zero or integer values flag a non-continuous
    structure. Aggregate to coarser time bins (Powell aggregates
    monthly suicides to 12-month windows for exactly this reason)
    or move to :doc:`dsc` if the *distribution* is the object.

(b) A2(a) OR A2(b) for every unit, with at least one A2(a).
    Identification fails only if no unit has a proper
    synthetic control. The treated unit may fail A2(a) entirely
    -- that's the whole point -- as long as it appears with
    positive weight in some other unit's synthetic control.

    *Plausibly violated when* the donor pool itself sits in a
    qualitatively different regime (e.g. the eight waiting-period
    states are *all* low-suicide-rate while Wisconsin and any
    donor that would put weight on it are high-rate). *Diagnostic*:
    read ``res.fit_metric``; if every :math:`a_i \approx 0`,
    no unit in the panel has a proper synthetic control and the
    estimator has nothing to anchor on. If only the treated unit
    has :math:`a_1 \to 0` but a handful of donors have
    :math:`a_i \gtrsim 0.5` (the Wisconsin pattern: Iowa, Indiana,
    Mississippi all fit well as syntheses *and* place positive
    weight on Wisconsin), A2(b) is still doing its job.

(c) No contemporaneous cross-sectional shock correlation (A3b).
    The shocks :math:`\epsilon_{it}, \epsilon_{jt}` are uncorrelated
    across units in the *same* period given the latent component.

    *Plausibly violated when* a common national-level shock hits
    every unit in the same period -- a federal policy change, a
    macro recession, a pandemic. The ISCM moment conditions
    (paper eq. 10) are no longer zero in expectation because the
    cross-unit residual covariances enter the bias term.
    *Diagnostic*: form the panel of pre-period SCM residuals
    :math:`R_{it} = y_{it} - \sum_j \widehat{w}_i^{\,j} y_{jt}` and
    compute the cross-sectional correlation matrix; large
    off-diagonal entries flag A3b violations. Drop the common
    shock with a unit-time fixed effect *before* fitting ISCM, or
    use a time-fixed-effects pre-residualisation step (Powell
    notes ISCM does not require unit FE but does require A3b).

(d) Within-unit mixing of the shock (A4).
    Serial correlation within a unit is allowed but must decay --
    strongly mixing of size :math:`-r/(r-1)`, :math:`r > 1`. This
    rules out unit roots and other persistent (non-mixing)
    structures in the shock.

    *Plausibly violated when* the outcome contains a unit-root
    component -- raw monthly stock-price levels, cumulative
    population, undifferenced trending series. The ISCM estimator
    converges to the true effect *plus* a non-vanishing variance
    term :math:`V_t` (Theorem 4.1), and that variance fails to
    average down over the post-period when :math:`\epsilon` does
    not mix. *Diagnostic*: ADF or KPSS test on the pre-period
    residuals; if non-stationary, first-difference the outcome or
    move to a stationary-cycle estimator (:doc:`sbc`) before
    feeding into ISCM.

(e) Long pre-period (Theorem 4.1 is asymptotic in :math:`T_0`).
    Unbiasedness requires :math:`T_0 \to \infty`; in finite
    samples the bias term has a residual of order
    :math:`T_0^{-1/2}` from the empirical moment conditions.

    *Plausibly violated when* :math:`T_0` is on the order of
    :math:`N` or smaller, especially with a small donor pool.
    *Diagnostic*: the paper's application uses 161 monthly pre-
    treatment observations; if your :math:`T_0` is in the tens,
    re-estimate after lengthening the pre-period (e.g. by
    aggregating across a finer time grid) and compare. Large
    swings in the estimate flag the finite-sample bias.

(f) Inference floor under tiny contributing sets :math:`|C|`.
    The Ibragimov-Muller sign-flip test on per-unit estimates has
    p-value floor :math:`2 / 2^{|C|}`. With :math:`|C| = 3` the
    floor is 0.25; with :math:`|C| = 5` it is 0.0625; with
    :math:`|C| = 8` it is 0.0078.

    *Plausibly violated when* the donor pool is so small that
    only a few units survive the :math:`a_i` filtering. The
    Wisconsin application has :math:`|C| = 5` (Wisconsin plus
    California, Indiana, Iowa, Mississippi, Rhode Island), so the
    smallest possible p-value the test can return is
    :math:`\approx 0.0625` -- the headline 0.046 sits at the
    floor. *Diagnostic*: read
    ``res.inference.n_contributing``; if it is below 6 or so,
    interpret p-values cautiously and consider expanding the
    donor pool or aggregating multiple treated units (Powell
    Section 4.4.3) to add inference power.

When to use ISCM -- and when not to
-----------------------------------

Reach for ISCM when:

* The treated unit is visibly outside the donor convex hull
  (its pre-period trends above or below every donor, no convex
  combination can match it) and you can see the canonical SCM
  pre-fit is bad. ISCM's whole identification story is built for
  this case.
* You have a long pre-period (Powell's application uses 161
  monthly observations; Theorem 4.1 is asymptotic in :math:`T_0`).
* Transitory shocks are large -- outcomes are noisy enough
  that an exact pre-period match is implausible even when the
  hull holds. The moment-condition framework is robust to shock
  variance that breaks the canonical SCM's exact-fit assumption.
* The donor pool is small -- conventional permutation
  inference cannot reach 5% with 8 donors (the floor is roughly
  :math:`1/9`); ISCM's per-unit decomposition + Ibragimov-Muller
  sign-flip gets you to the floor :math:`2/2^{|C|}`, which is
  meaningfully tighter (0.0625 at :math:`|C|=5`) than 1/9.
* You want to remove the eyeball "is the pre-fit good enough"
  decision from your workflow. The :math:`a_i` weights
  systematise that judgement, asymptotically excluding units
  without proper synthetic controls without the researcher
  flagging them by hand.

Do not use ISCM when:

* The treated unit is well inside the hull and the canonical
  SCM pre-fit is tight. ISCM adds estimation noise from the
  per-unit decomposition and the :math:`a_i` weighting machinery
  for no identification gain. Use *canonical SCM*, :doc:`tssc`, or
  :doc:`fdid` -- they have stronger small-sample guarantees in
  the happy case.
* Contemporaneous national-level shocks are part of the DGP
  (national policy change, macro recession, pandemic spanning
  treated + donors). A3b fails; the ISCM moment conditions
  acquire a bias term you cannot remove. Either de-mean by
  time-FE before fitting or move to :doc:`spillsynth` /
  :doc:`spsydid` if the shock has spatial structure.
* The outcome is non-stationary or unit-root without
  differencing. A4's mixing condition fails; the post-period
  variance term in Theorem 4.1 does not average down.
  First-difference, or move to :doc:`sbc` (a stationary-cycle
  estimator) before feeding into ISCM.
* Binary, ordinal, or low-count outcomes. A1 requires
  continuity. With heaps at zero or integer values, aggregate to
  coarser time bins (the Wisconsin application aggregates monthly
  suicides to 12-month windows), or move to :doc:`dsc` for the
  distributional question.
* No unit anywhere in the panel has a valid synthetic control.
  A2(a) must hold for *some* unit (paper Discussion, Section
  4.2). If every :math:`a_i \to 0`, ISCM has nothing to anchor
  on. Diagnose by inspecting ``res.fit_metric``; if every value
  is near zero, the panel is structurally unsuited and you need
  a different identification strategy.
* Continuous or multi-valued treatment. ISCM encodes binary
  on/off treatment with the exposure :math:`E_{it} = d_{it} -
  \sum_j \widehat{w}_i^{\,j} d_{jt}`. Continuous dose
  (minimum wage, ad spend, drug dosage) belongs in :doc:`ctsc`.
* Staggered adoption with a long mixed-treatment pre-period.
  Section 4.4.3 sketches a multi-treated extension but assumes
  a common pre-period free of treatment. If donors adopt the
  policy at different times across a long window, drop late
  adopters or use a staggered SC variant
  (*FECT*, :doc:`sdid`).
* You need a sparse, interpretable single weight vector.
  ISCM returns per-unit weights for *all* units (one synthetic
  control per unit) and aggregates them via :math:`a_i`. If the
  policy story you need is "this single state is a convex
  combination of these four donors", report the canonical SC
  weight vector alongside ISCM, or use *canonical SCM* / :doc:`tssc`
  whose output IS a single sparse weight vector.

Empirical: Wisconsin's 48-h handgun waiting-period repeal
---------------------------------------------------------

Powell's Section 6 application: in June 2015 Wisconsin repealed its
48-h handgun-purchase waiting period. The donor pool is the eight
states that also had waiting periods during the analysis window
*and* did not repeal them (California, Hawaii, Illinois, Iowa,
Maryland, Minnesota, New Jersey, Rhode Island). The outcome is
monthly handgun-suicide deaths per 100,000 from January 2002 to
May 2019 (161 pre-treatment months, 47 post-period).

The setup is exactly the case ISCM was built for:

* Wisconsin is structurally outside the donor hull. It has
  the highest handgun-suicide rate in 73 of the 161 pre-period
  months -- no convex combination of the eight donors can match
  its level even in expectation. The canonical SCM
  (Figure 1B, blue line) drifts visibly upward in the pre-period,
  flagging the convex-hull violation; the demeaned SCM (Powell
  also runs this) corrects for level but still shows an upward
  pre-period trend, indicating the convex-hull assumption fails
  *in expectation*, not just in sample.
* The donor pool is small (:math:`N_0 = 8`), so the canonical
  permutation test's smallest achievable p-value is roughly
  :math:`1/9 \approx 0.11` -- above any conventional threshold.

ISCM produces:

* Main estimate: :math:`\widehat{\tau} = 0.105` deaths per
  100,000 (about a 30% increase relative to the pre-repeal
  Wisconsin rate), :math:`p = 0.046` via the Ibragimov-Muller
  sign-flip on the contributing units. The p-value sits at the
  inference floor :math:`2/2^{|C|}` with :math:`|C| = 5`.
* Per-state decomposition (paper Table 1, ``v_i`` weights):

  =================  ===========  =========
  State              :math:`v_i`  Estimate
  =================  ===========  =========
  Iowa               35.73%       0.038
  Indiana            34.46%       0.063
  Mississippi        19.43%       0.232
  Wisconsin          9.94%        0.232
  California         0.43%        0.336
  Rhode Island       0.02%        -1.931
  =================  ===========  =========

  Wisconsin contributes only ~10% of the total estimate -- it is
  itself a bad synthetic control for the other waiting-period
  states (its outside-hull position cuts both ways) -- while Iowa
  and Indiana, which produce *good* synthetic controls *and*
  place positive weight on Wisconsin, drive 70% of the estimate.
  This is the A2(b) identification mechanism at work: even though
  Wisconsin fails A2(a), the unbiased treatment-effect signal is
  recovered from the donors who use Wisconsin as a donor.

Because the application uses restricted-access NVSS mortality
data, this estimator is not runnable end-to-end from public
sources. The replication package
(``https://journaldata.zbw.eu/dataset/imperfect-synthetic-controls``)
documents access; the mlsynth ISCM API call is structurally
identical to the example above.

Core API
--------

.. automodule:: mlsynth.estimators.iscm
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.ISCMConfig
   :members:
   :undoc-members:

Helper Modules
--------------

.. automodule:: mlsynth.utils.iscm_helpers.setup
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.iscm_helpers.weights
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.iscm_helpers.estimate
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.iscm_helpers.inference
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.iscm_helpers.pipeline
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.iscm_helpers.structures
   :members:
   :undoc-members:

Example
-------

A one-factor panel where the treated unit has the largest factor loading
-- placing it outside the convex hull of the controls, so a traditional
SCM cannot match it. ISCM still recovers the planted effect via the
control units that use the treated unit as a donor.

.. code-block:: python

   import numpy as np
   import pandas as pd

   from mlsynth import ISCM

   # ------------------------------------------------------------------
   # 1. One-factor panel; unit 0 (treated) has the MAX loading
   # ------------------------------------------------------------------
   rng = np.random.default_rng(0)
   N, T, T0, true_alpha = 8, 60, 48, 3.0
   loadings = np.linspace(2.0, -1.5, N)        # unit 0 outside the hull
   f = np.cumsum(rng.standard_normal(T)) * 0.3 + np.linspace(0, 2, T)
   Y = np.outer(loadings, f) + rng.standard_normal((N, T)) * 0.05
   D = np.zeros((N, T))
   Y[0, T0:] += true_alpha
   D[0, T0:] = 1

   rows = [{"unit": f"u{i}", "time": t, "y": Y[i, t], "D": int(D[i, t])}
           for i in range(N) for t in range(T)]
   df = pd.DataFrame(rows)

   # ------------------------------------------------------------------
   # 2. Fit ISCM with Ibragimov-Muller inference
   # ------------------------------------------------------------------
   res = ISCM({
       "df": df, "outcome": "y", "treat": "D",
       "unitid": "unit", "time": "time",
       "inference": True,
   }).fit()

   # ------------------------------------------------------------------
   # 3. Inspect the result
   # ------------------------------------------------------------------
   print(f"ATT = {res.att:+.3f}  (true = {true_alpha})")
   print(f"treated fit metric a_0 = {res.fit_metric[0]:.3f}  "
         f"(small => outside the hull)")
   print(f"treated contribution   = {res.contribution[0]*100:.1f}%")
   print(f"p-value = {res.inference.p_value:.3f}  "
         f"(n contributing = {res.inference.n_contributing})")

References
----------

Abadie, A., Diamond, A., & Hainmueller, J. (2010). "Synthetic Control
Methods for Comparative Case Studies." *Journal of the American
Statistical Association* 105(490):493-505.

Ferman, B., & Pinto, C. (2021). "Synthetic Controls with Imperfect
Pretreatment Fit." *Quantitative Economics* 12(4):1197-1221.

Fry, J. (2024). "A Method of Moments Approach to Asymptotically Unbiased
Synthetic Controls." *Journal of Econometrics* 244:105846.

Ibragimov, R., & Muller, U. K. (2010). "T-Statistic Based Correlation
and Heterogeneity Robust Inference." *Journal of Business & Economic
Statistics* 28(4):453-468.

Powell, D. (2026). "Imperfect Synthetic Controls." *Journal of Applied
Econometrics* 41(3):253-264.
