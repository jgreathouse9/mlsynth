Partially Pooled SCM (PPSCM)
============================

.. currentmodule:: mlsynth

When to Use This Estimator
--------------------------

``PPSCM`` is a faithful port of ``augsynth::multisynth`` -- the partially
pooled synthetic control of Ben-Michael, Feller and Rothstein [PPSCM]_ for
staggered adoption. Use it when several units are treated but at *different*
times, with a pool of never-treated (or late-treated) comparison units, and you
want a single estimate of the average treatment effect on the treated (ATT) over
relative time (time-since-treatment), pooling information across cohorts.

The central idea is a pooling dial :math:`\nu`. Fitting a *separate* synthetic
control for each treated unit gives the best per-unit pre-treatment fit but high
variance; a *fully pooled* control (one synthetic match for the average treated
unit) is stable but may fit any individual unit poorly. PPSCM interpolates
between the two, choosing :math:`\nu` to balance overall and unit-level
imbalance. ``time_cohort=True`` collapses units sharing an adoption time into a
single fully-pooled cohort (one synthetic control per cohort).

The problem PPSCM solves is that the two reflexive extensions of SCM to
staggered adoption are each flawed. Separate SCM (fit a synthetic
control per treated unit, then average -- common practice) requires a good
synthetic control for *every* treated unit, which often fails, and its
strong per-unit fits can still leave the *average* poorly matched, biasing
the ATT. Pooled SCM (match the average treated unit) nails the average
fit but can fit individual units badly, biasing unit-level effects and the
average when the data-generating process drifts over time. Ben-Michael,
Feller and Rothstein bound the estimation error by *both* the average
imbalance and the per-unit imbalances, and partially pooled SCM minimises a
weighted combination of the two -- the regime where neither extreme is
trustworthy.

Reach for PPSCM when
^^^^^^^^^^^^^^^^^^^^^

* Several units are treated at different adoption times, with a pool of
  never-treated (or not-yet-treated) comparison units.
* You want an ATT over relative time (an event-study path), pooling
  information across cohorts instead of fitting each cohort alone.
* No single donor mix matches every treated unit, so separate SCM
  leaves you with unreliable per-unit fits -- the partial-pooling dial lets
  the average fit borrow strength without abandoning unit-level fit.
* You want an estimator that nests the familiar special cases (separate
  and fully pooled SCM) and a principled way to choose between them.

Do not use PPSCM when
^^^^^^^^^^^^^^^^^^^^^^

* All treated units adopt at the same time (a single cohort). The
  staggered machinery is unnecessary; use classic SC (:doc:`tssc`,
  :doc:`fdid`) or, for many treated units at one time, :doc:`sdid`.
* You are willing to assume parallel trends after weighting and want
  the DiD-flavoured double weighting / time weights. :doc:`sdid` (and, for
  efficiency under interactive fixed effects, :doc:`seq_sdid`) is the more
  natural home; PPSCM is a *synthetic-control* estimator, not a
  difference-in-differences one.
* Spillovers violate SUTVA across the donor pool -- use :doc:`spsydid`.
* The treated paths lie outside the donor convex hull / the donor pool is
  large and noisy. Partial pooling cannot manufacture a hull that does
  not contain the treated units; a factor-model (:doc:`fma`) or low-rank
  (:doc:`clustersc`, :doc:`mcnnm`) approach is better.
* The pre-period is short relative to the donor pool. Synthetic control needs
  enough pre-periods to pin the weights down; with more donors than periods to
  balance, the simplex is wide enough that near-exact balance is available for
  free and says nothing about how well the donors track the treated units. This
  is the regime where difference in differences is the justified estimator --
  it asks for many units and few periods, which is the opposite trade -- so
  reach for :doc:`sdid`, or for PPSCM's own ``method="callaway_santanna"``.
  ``design.underdetermined`` reports when a fit is in this regime.
* Distributional effects (quantiles, tails) -- use :doc:`dsc`.

Notation
--------

All units :math:`\mathcal{N} \coloneqq \{1, \ldots, N\}` are observed over
periods :math:`t \in \mathcal{T} \coloneqq \{1, \ldots, T\}`. A treated unit
(or cohort) :math:`j` adopts at period :math:`T_j`; never-treated units have
:math:`T_j = \infty` and form the donor pool
:math:`\mathcal{N}_0 \coloneqq \{j \in \mathcal{N} : T_j = \infty\}` of
cardinality :math:`N_0`. The panel is split at the last adoption time, the
canonical point :math:`T_0`, into a pre-period
:math:`\mathcal{T}_1 \coloneqq \{t \in \mathcal{T} : t \le T_0\}` of length
:math:`T_0` and a post-period
:math:`\mathcal{T}_2 \coloneqq \{t \in \mathcal{T} : t > T_0\}`. For cohort
:math:`j`, donor weights :math:`\mathbf{w}_j` live on the simplex
:math:`\Delta^{N_0} \coloneqq \{\mathbf{w} \in \mathbb{R}_{\ge 0}^{N_0} :
\|\mathbf{w}\|_1 = 1\}`; the synthetic control matches the cohort's
pre-treatment residuals. The per-period effect is :math:`\tau_t` and the
average treatment effect on the treated is :math:`\widehat{\tau}`.

Method
------

PPSCM follows ``multisynth`` in three stages.

1. Two-way fixed effects (``fixedeff=True``, the default). A time effect is
   the never-treated units' per-period mean; a unit effect is each unit's
   mean over its own pre-adoption window. Both are removed and the synthetic
   control balances
   the residuals -- the "intercept-shifted" estimator of the paper.

2. Partially pooled QP. With per-cohort pre-treatment imbalance
   :math:`\mathbf{q}_j \coloneqq \mathbf{x}_j - \mathbf{X}_{0,j}\mathbf{w}_j`
   (residuals; the pooled imbalance aligned by relative time), the weights
   solve

.. math::

   \min_{\{\mathbf{w}_j \in \Delta^{N_0}\}} \;
     \frac{\nu}{\text{norm}_{\text{pool}}\,J^2}
       \Bigl\|\textstyle\sum_j \mathbf{q}_j\Bigr\|^2
     + \frac{1-\nu}{\text{norm}_{\text{sep}}\,J}
       \sum_j \frac{\|\mathbf{q}_j\|^2}{\text{ndim}_j}
     + \lambda \sum_j \|\mathbf{w}_j\|^2 ,

where :math:`\text{norm}_{\text{pool}}` and :math:`\text{norm}_{\text{sep}}` are
the separate-fit (``nu=0``) global and individual imbalance norms. Small
:math:`\nu` approaches a separate SCM per cohort; large :math:`\nu` a fully
pooled SCM.

3. Choosing :math:`\nu`. With ``nu="auto"`` (default) PPSCM uses augsynth's
   triangle-inequality ratio
   :math:`\nu = \text{global\_l2}\cdot\sqrt{T_0}/\text{avg\_l2}`
   from the separate fit; a float in :math:`[0,1]` fixes it, and anything
   outside that interval is refused, since one of the two weights would be
   negative and the program no longer convex.

   The ratio is :math:`\lVert\bar m\rVert / \overline{\lVert m_k\rVert}` over
   the treated units' separate-fit imbalance vectors :math:`m_k`, so it is at
   most one and equals one exactly when those vectors are parallel. A single
   treated unit gives that by definition, and a small pool of similar units comes
   close. On the bound the separate term drops out, which is the right fit: with
   one treated unit there is nothing to pool against. The library computes the
   ratio in floating point and holds it inside :math:`[0,1]`, because a value a
   unit in the last place above one is not a pooling choice -- it is a negative
   weight on a squared norm, and the same panel measured in dollars and in
   thousands of dollars can land on opposite sides of the boundary.

Assumptions / Remarks.

*Assumption 1 (no anticipation, parallel residual trends).* After removing the
two-way fixed effects, the treated cohorts' residual paths would have matched a
convex combination of donor residual paths absent treatment. *Remark.* This is
the staggered-adoption analogue of the SCM identifying assumption; the fixed
effects absorb level and common-time shifts so the weights only need to match
the residual dynamics.

*Assumption 2 (overlap / donor availability).* Each cohort has eligible donors
-- never-treated units, or units treated more than ``n_leads`` periods later.
*Remark.* Late-treated units can serve as "clean" controls for earlier cohorts
until they themselves are treated, which the donor-eligibility rule enforces.

*Remark (pooling).* :math:`\nu` is a bias--variance dial, not an identification
parameter: the estimand (the wATET over the treated cohorts) is the same;
:math:`\nu` only trades per-cohort fit against stability of the pooled average.

Auxiliary covariates
--------------------

By default PPSCM matches on the pre-treatment outcome path alone. Passing
``covariates=[...]`` also balances a set of auxiliary covariates, following
the paper's Section 5.2. Each covariate is z-scored against the never-treated
controls and rescaled to the outcome scale, so covariate and outcome imbalance
share a footing; the covariate imbalance is then stacked into both the pooled
and the separate terms of the partially-pooled objective. Time-varying
covariates are aggregated to their mean over the periods before the first
adoption. Balancing covariates typically improves covariate balance at a small
cost to the pre-treatment outcome fit -- the usual bias/variance trade of
matching on more.

.. code-block:: python

   res = PPSCM({"df": df, "outcome": "y", "treat": "d",
                "unitid": "unit", "time": "period",
                "covariates": ["income_1959", "student_teacher_ratio_1959"]}).fit()

This reproduces ``augsynth::multisynth``'s covariate mode
(``y ~ d | income + ratio``); see :doc:`replications/ppscm` and the
``ppscm_paglayan_covs`` benchmark for the cell-by-cell cross-check against a
live ``augsynth 0.2.0`` run.

.. _ppscm-cs-mode:

Reaching Callaway-Sant'Anna and Sun-Abraham
-------------------------------------------

Ben-Michael, Feller and Rothstein (2022, p.369) observe that with uniform donor
weights their intercept-shifted estimator "is equivalent to recent proposals for
DiD estimators that allow for treatment effect heterogeneity with a fixed donor
set per treatment time cohort (see Callaway & Sant'Anna, 2020; Sun & Abraham,
2020)". Measured, that is not an approximation: three independent
implementations agree to 1e-14 once three conventions are aligned.

The three are separate settings, because each is independently meaningful:

``donor_weights``
   ``"scm"`` (default) solves the partially-pooled QP; ``"uniform"`` puts equal
   weight on every admissible donor, which is the comparison Callaway-Sant'Anna
   and Sun-Abraham make. It is the :math:`\lambda \to \infty` limit of the
   same program, written in closed form; the derivation is below.

``base_period``
   ``"all_pre"`` (default) is augsynth's: each unit's mean over its whole
   pre-adoption window. ``"pre_treatment"`` is the single period :math:`g-1`
   that Callaway-Sant'Anna normalise against. On its own the choice shifts each
   cohort's level without moving the event-study shape.

``donor_pool``
   ``"window"`` (default) admits any unit untreated through the cohort's whole
   estimation window, :math:`g_i > g + H`. ``"never_treated"`` and
   ``"not_yet_treated"`` are the Callaway-Sant'Anna comparison groups. The first
   two coincide exactly when every other cohort adopts inside the window.

``method="callaway_santanna"`` sets all three at once (and selects their
standard error, below), leaving any convention the caller set explicitly alone:

.. code-block:: python

   res = PPSCM({"df": df, "outcome": "y", "treat": "d",
                "unitid": "unit", "time": "period",
                "method": "callaway_santanna"}).fit()

The three estimators diverge in exactly one regime, and it is a documented
difference and not a defect: when a later cohort outlives an earlier cohort's
estimation window, augsynth admits it as a donor and Callaway-Sant'Anna do not.
On four cohorts spread over a long window the gap is about 1.2e-02. A panel with
adoptions spread widely lands there, so a difference of that size between
``donor_pool="window"`` and ``donor_pool="never_treated"`` is the conventions
disagreeing, not a bug.

Why the two estimators meet: the ridge path
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The equivalence is not a coincidence of two formulas. Callaway-Sant'Anna sits at
the end of a path the partially-pooled program already contains, and the
conventions above are the coordinates of that endpoint. Ben-Michael, Feller and
Rothstein introduce the :math:`\lambda` term (their Section 3) as

   a term that penalizes the weights towards uniformity, with hyperparameter
   :math:`\lambda`. While we penalize the sum of the squared weights, there are
   many options, for example, an entropy or elastic net penalty

so uniformity is what the penalty is for. Following it to its limit is what
produces the other estimator.

Step 1: the barycenter is what any such penalty selects. Let
:math:`\Omega : \Delta^{N_0} \to \mathbb{R}` be strictly convex and
permutation-symmetric, so :math:`\Omega(\mathbf{P}\mathbf{w}) =
\Omega(\mathbf{w})` for every permutation matrix :math:`\mathbf{P}`. Strict
convexity gives a unique minimiser :math:`\mathbf{w}^\star`; symmetry makes
:math:`\mathbf{P}\mathbf{w}^\star` a minimiser too, so
:math:`\mathbf{P}\mathbf{w}^\star = \mathbf{w}^\star` for all :math:`\mathbf{P}`,
and the only permutation-invariant point of the simplex is its barycenter:

.. math::

   \operatorname*{arg\,min}_{\mathbf{w} \in \Delta^{N_0}} \Omega(\mathbf{w})
     = \bar{\mathbf{w}} \coloneqq \tfrac{1}{N_0}\mathbf{1}.

The squared norm :math:`\sum_i w_i^2` and the negative entropy
:math:`\sum_i w_i \log w_i` are both of this form, so the alternatives BFR list
have the same endpoint. Geometrically :math:`\bar{\mathbf{w}}` is the point of
the simplex nearest the origin, which is why the ridge sends the weights there.

Step 2: the program converges to it, and forgets :math:`\nu`. Write the
partially-pooled objective as :math:`f_\nu(\mathbf{w}) + \lambda
\Omega(\mathbf{w})`, equivalently :math:`\lambda^{-1} f_\nu(\mathbf{w}) +
\Omega(\mathbf{w})`. Since :math:`\Delta^{N_0}` is compact and :math:`f_\nu`
continuous, :math:`\lambda^{-1} f_\nu \to 0` uniformly, so
:math:`\mathbf{w}_\lambda \to \bar{\mathbf{w}}` for every :math:`\nu`. The
pooling dial is inert in the limit: it weights a term that has been scaled away.

Step 3: the rate, and what :math:`\nu` does instead. The barycenter has every
coordinate :math:`1/N_0 > 0`, so it lies in the relative interior and no
non-negativity constraint is active near it. For large :math:`\lambda` the
program is therefore smooth on the affine hull :math:`\{\mathbf{1}'\mathbf{w} =
1\}`, and stationarity :math:`\nabla f_\nu(\mathbf{w}_\lambda) + 2\lambda
\mathbf{w}_\lambda + \mu\mathbf{1} = \mathbf{0}` linearised at
:math:`\bar{\mathbf{w}}` gives

.. math::

   \mathbf{w}_\lambda = \bar{\mathbf{w}}
     - \frac{1}{2\lambda}\,\mathbf{P}\,\nabla f_\nu(\bar{\mathbf{w}})
     + O(\lambda^{-2}),
   \qquad
   \mathbf{P} \coloneqq \mathbf{I} - \tfrac{1}{N_0}\mathbf{1}\mathbf{1}' ,

with :math:`\mathbf{P}` the projection onto the simplex's tangent space. So the
approach is first order in :math:`\lambda^{-1}` -- not the :math:`\lambda^{-1/2}`
a boundary solution would give -- along a fixed direction that is tangent to the
simplex. That direction is where :math:`\nu` survives: it sets the direction in
which partially pooled SCM departs from Callaway-Sant'Anna, having no say in
where the path ends.

Measured on a three-cohort panel with 41 never-treated donors, against the
Callaway-Sant'Anna estimate:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - :math:`\lambda`
     - :math:`|\widehat{\tau} - \widehat{\tau}_{\mathrm{CS}}|`
     - :math:`\max_i |w_i - 1/N_0|`
   * - 0
     - 1.0e-01
     - 2.6e-01
   * - 1e6
     - 7.7e-07
     - 1.4e-07
   * - 1e12
     - 7.7e-13
     - 1.4e-13

A decade of :math:`\lambda` buys a decade of accuracy, in the weights and in the
estimate alike, down to machine precision. The scaled departure
:math:`\lambda(\mathbf{w}_\lambda - \bar{\mathbf{w}})` settles to a fixed vector
(norms 0.410880, 0.411064, 0.411066 at :math:`\lambda = 10^4, 10^6, 10^8`;
direction cosine 1.000000 to eight decimals), and its size moves with
:math:`\nu` alone -- 0.2846, 0.6152, 0.9541 at :math:`\nu = 0, 0.5, 1` -- while
the limit does not move at all.

Step 4: the endpoint is the estimator. At :math:`\bar{\mathbf{w}}`, with
``base_period="pre_treatment"`` subtracting :math:`Y_{i,g-1}` and
``donor_pool="never_treated"`` supplying the comparison group
:math:`\mathcal{C}`, cohort :math:`g`'s horizon-:math:`k` effect is

.. math::

   \widehat{\tau}_{g,g+k}
     = \frac{1}{n_g}\sum_{i \in \mathcal{G}_g}\bigl(Y_{i,g+k} - Y_{i,g-1}\bigr)
     - \frac{1}{n_{\mathcal{C}}}\sum_{i \in \mathcal{C}}
         \bigl(Y_{i,g+k} - Y_{i,g-1}\bigr)
     = \widehat{ATT}(g, g+k),

the two-period, two-group difference in differences Callaway and Sant'Anna
identify under their Assumptions 1-4 with a never-treated comparison group. The
common time effect cancels between the two group means, so removing it changes
nothing. This is BFR's own reading of their Equation (9): with uniform weights
it "is the simple average over all two-period, two-group DiD estimates", which
they call "equivalent to recent proposals ... (see Callaway & Sant'Anna, 2020;
Sun & Abraham, 2020)".

One difference hides in that sentence. BFR average over all pre-treatment lags,
where Callaway-Sant'Anna normalise on :math:`g-1` alone -- which is exactly the
``base_period`` setting, and why the equivalence needs all three conventions and
not just uniform weights.

Aggregation closes it. PPSCM averages cohorts by size at each horizon and then
averages horizons,

.. math::

   \widehat{\theta} = \frac{1}{H}\sum_{h} \widehat{\theta}_h ,
   \qquad
   \widehat{\theta}_h = \sum_{k \in \mathcal{K}_h}
     \frac{p_k}{S_h}\,\widehat{\tau}_{g_k, g_k + h} ,
   \quad S_h = \sum_{k \in \mathcal{K}_h} p_k ,

which is Callaway-Sant'Anna's dynamic aggregation followed by an average over
event time. It coincides with their simple aggregation when every cohort is the
same size and reaches every horizon, and not otherwise -- so equal-sized cohorts
hide the distinction instead of establishing it.

The event-study window decides which cells enter that sum. ``n_leads`` defaults
to the last cohort's post window, which is the shortest, so every cohort reaches
every horizon and :math:`\mathcal{K}_h` is the full set of cohorts at each
:math:`h`. Raising it adds horizons that only the early cohorts observe: the
late cohorts report ``NaN`` there and :math:`\mathcal{K}_h` thins as :math:`h`
grows. The ceiling is the longest post window, since no cohort observes anything
past the end of the panel, and a larger request is cut to it.

That longer window is the one Callaway-Sant'Anna's simple aggregation runs over,
and the per-unit paths carry it. The mean over their finite entries weights every
unit-post-period cell equally, which is what ``did::aggte(type = "simple")``
reports, while ``res.effects.att`` stays the dynamic aggregation above:

.. code-block:: python

   import numpy as np

   res = PPSCM({"df": df, "outcome": "y", "treat": "d", "unitid": "unit",
                "time": "period", "method": "callaway_santanna",
                "n_leads": 6}).fit()

   dynamic = res.effects.att                 # aggte(type="dynamic")$overall.att
   paths = np.vstack([u.tau for u in res.per_unit.values()])
   simple = np.nanmean(paths)                # aggte(type="simple")$overall.att

Raising the window is not free with ``donor_pool="window"``, where a unit is a
donor to a cohort only if it stays untreated through that cohort's estimation
window: a longer window is a stricter pool. The Callaway-Sant'Anna comparison
groups, ``never_treated`` and ``not_yet_treated``, do not depend on it.

What this does not close is the donor pool. Uniform weights and the :math:`g-1`
baseline are choices inside the program; :math:`\mathcal{C}` is the set the
program ranges over. When a later cohort outlives an earlier cohort's estimation
window the two sets genuinely differ, and no :math:`\lambda` reconciles them --
which is the regime described above.

In practice ``donor_weights="uniform"`` is the limit written down instead of
approached: exact, with no quadratic program to solve and no :math:`\lambda` for
the caller to guess. The path matters because it explains why the two estimators
are the same object, and it is verified in
`test_ppscm_cs_ridge_limit.py <https://github.com/jgreathouse9/mlsynth/blob/main/mlsynth/tests/test_ppscm_cs_ridge_limit.py>`_,
which pins every number quoted above and checks the limit against ``diff-diff``
itself where it is installed.

Inference
---------

``PPSCM`` reports the paper's delete-one jackknife: drop each unit, refit the
full estimator (holding :math:`\nu` fixed), and form
:math:`\widehat{\text{se}}^2 = \tfrac{N-1}{N}\sum_{j \in \mathcal{N}}(\widehat{\tau}_j - \bar{\tau})^2`
for the overall ATT and each relative-time horizon, with Wald intervals.
``inference_method="bootstrap"`` swaps in augsynth's default Mammen wild
bootstrap, which reweights the single fit instead of refitting.

Analytical standard errors under the Callaway-Sant'Anna conventions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``method="callaway_santanna"`` reaches their point estimate exactly, and an
equal point estimate does not make an equal interval. The preset therefore
also selects ``inference_method="influence_function"``, which is the standard
error that goes with that estimate.

Under those conventions each cell is a two-period, two-group difference in
differences and its influence function is available in closed form:

.. math::

   \widehat{\phi}_i(g,t) =
     \frac{\Delta_i - \bar{\Delta}_{\mathcal{G}_g}}{n_g}\ \ (i \in \mathcal{G}_g),
   \qquad
   \widehat{\phi}_i(g,t) =
     -\frac{\Delta_i - \bar{\Delta}_{\mathcal{C}}}{n_{\mathcal{C}}}\ \ (i \in \mathcal{C}),

with :math:`\Delta_i = Y_{it} - Y_{i,g-1}` and
:math:`\widehat{\text{se}}(g,t) = \sqrt{\sum_i \widehat{\phi}_i(g,t)^2}`. A
standard error then costs one pass over the panel, where the jackknife costs one
refit per unit.

PPSCM averages cohorts by size at each horizon and then averages horizons,
:math:`\widehat{\theta} = \tfrac1H \sum_h \widehat{\theta}_h` with
:math:`\widehat{\theta}_h = \sum_{k \in \mathcal{K}_h} p_k \widehat{\tau}_{kh} / S_h`
and :math:`S_h = \sum_{k \in \mathcal{K}_h} p_k`. That is Callaway and
Sant'Anna's dynamic aggregation followed by an average over event time, and it
coincides with their simple aggregation when every cohort is the same size and
reaches every horizon. The cohort shares :math:`p_k` are estimated from the same
panel, so the aggregate carries their sampling error through
:math:`\partial \theta_h / \partial p_j = (\tau_{jh} - \theta_h)/S_h` -- the term
R's ``did::aggte`` calls ``wif``. Deleting it leaves a standard error that is
finite, plausible and too small, so the test suite computes the aggregate both
ways and pins the difference.

``results.inference_detail`` then carries ``group_time_att`` and
``group_time_se`` keyed by the public ``(adoption time, time)`` labels, the
per-unit ``influence`` matrix every reported standard error is a functional of,
and two bands on the event-time path. ``cband=True`` tabulates the second one:

.. math::

   c_{1-\alpha} = \text{quantile}_{1-\alpha}\ \max_h
     \Bigl| \sum_i v_i \widehat{\psi}_{h,i} \Bigr| \big/ \widehat{\text{se}}_h ,

with :math:`v_i` Mammen (1993) multipliers. One critical value covers every
horizon at once, which is the level a reader assumes when they read the path as
a path ("positive by horizon three and never back"); the pointwise band read
that way covers less. No refit is involved -- the multipliers act on the
influence functions the point estimate already produced.

The derivation assumes the conventions that produce the Callaway-Sant'Anna
estimate, so ``inference_method="influence_function"`` is available only with
``donor_weights="uniform"``, ``base_period="pre_treatment"``,
``donor_pool="never_treated"`` and ``fixedeff=True``. Solved SCM weights are
estimated too and contribute a term of their own, and a not-yet-treated pool
changes the comparison group's composition over time; outside the four the
formula is wrong and not approximate, and the configuration raises
``MlsynthConfigError`` naming the convention that broke it. Setting a convention
explicitly alongside the preset is a coherent question whose answer is the
jackknife, so the preset stands down instead of raising.

Verification: the analytical standard errors are pinned cell by cell and in
aggregate against
`benchmarks/reference/ppscm_cs/reference.py <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/reference/ppscm_cs/reference.py>`_,
a transcription of ``diff-diff`` 3.9.0 at commit ``d9cd475`` kept in-tree so the
check runs without a runtime dependency. Measured agreement is 0.0 per cell and
5.6e-17 on the aggregate.

Reading the balance diagnostics
-------------------------------

``design.global_l2`` and ``design.ind_l2`` are the pre-treatment imbalance the
fitted weights leave, and ``pct_improve_global`` / ``pct_improve_ind`` express
them against the uniform-weight baseline. A residual near zero means a good fit
only if the program could have done worse, so the design also reports the shape
of the problem that produced it:

``max_donors``
   the widest admissible donor pool across cohorts.

``balance_periods``
   the pre-periods actually balanced, capped by the cohort with the least
   history, since that cohort binds.

``underdetermined``
   whether ``max_donors - 1 > balance_periods``. Weights on the simplex carry
   ``max_donors - 1`` degrees of freedom against ``balance_periods`` equations,
   so past that point exact balance is generically attainable.

Both panels ``diff-diff`` ships illustrate the difference. On ``mpdta`` -- 500
counties over five years, the shape difference in differences is built for --
the binding cohort adopts in the second period, leaving one pre-period against
a pool of 480: ``global_l2`` comes to 3.4e-06 and 100 percent better than
uniform, which describes the geometry. On ``castle_doctrine``, 32 donors
against five periods cannot reach zero, and its 2.1e-02 is a fit.

``underdetermined`` is a statement about the program's shape and not a verdict.
The simplex is bounded, so a wide pool whose convex hull misses the treated path
still leaves a large residual, and that residual is informative. What the flag
rules out is reading a small one as evidence.

Per-unit fits alongside the pooled report
-----------------------------------------

Because partially pooled SCM fits a separate synthetic control per treated unit
(or per cohort with ``time_cohort=True``) and averages them into the ATT, the
unit-level estimates are the *components* of the pooled one -- so both are read
off a single fit. ``results.per_unit`` is a dict keyed the same as
``donor_weights_by_cohort``; each value is a ``PPSCMUnitFit`` carrying the unit's
``att``, its relative-time ``tau`` path, its ``donor_weights``, its adoption time
and member units, and its in-sample fit ``prefit_rmspe`` -- the root-mean-square
pre-treatment imbalance :math:`q_j` of that unit's synthetic control.

Each ``PPSCMUnitFit`` additionally carries a per-unit prediction interval on its
time-averaged effect -- ``ci_lower`` / ``ci_upper`` with a band-implied ``p_value``
-- populated when ``run_inference`` is on. It is built by the CFPT/SCPI
out-of-sample interval engine (:mod:`mlsynth.utils.scpi_helpers`, the same
machinery behind MSQRT's bands), applied to each unit's own pre-period residuals
and post-period gap with the synthetic-control weights held fixed. This is the
per-unit analogue of the pooled inference above: the delete-one jackknife (or
bootstrap) quantifies uncertainty *across* units for the aggregate, whereas the
per-unit SCPI band quantifies each unit's own effect. A naive permutation over the
QP-optimised pre-period residuals would over-reject -- the fit makes those residuals
small, so they are not exchangeable with the post-period gaps -- which is why the
per-unit band uses the SCPI construction, not a residual permutation.

The two levels reconcile exactly, so the unit-level and pooled reports never
disagree: the reported separate imbalance ``design.ind_l2`` equals
:math:`\sqrt{\tfrac1J\sum_j q_j^2}`, and the ``n_units``-weighted per-horizon
average of the unit ``tau`` paths reproduces ``event_study.tau`` and hence the
aggregate ``effects.att``. This makes it a one-line switch to serve either
request -- pooled error via ``design.ind_l2`` / ``global_l2`` and the aggregate
ATT, or per-unit estimates and their in-sample error via ``results.per_unit``.

A caveat for whoever reads the unit-level numbers: at a high
:math:`\nu` (heavily pooled), the per-unit synthetic controls fit poorly, so a
unit's ``att`` is only as trustworthy as its ``prefit_rmspe`` -- read the two
together, and prefer a lower :math:`\nu` (toward separate SCM) when unit-level
estimates are the deliverable.

.. code-block:: python

   res = PPSCM(config).fit()
   res.design.ind_l2                       # pooled/separate in-sample error
   res.effects.att                         # aggregate ATT
   for label, uf in res.per_unit.items():  # per-unit estimates + in-sample error
       print(label, uf.att, uf.prefit_rmspe)

The cumulative effect per unit
------------------------------

The per-unit bands above cover a unit's effect in a single period, or its average
across the post-period. A different question is what a unit gained in *total*: the
sum of its effects over the periods since adoption. Setting ``conformal_horizon``
adds a band for that total to every ``per_unit`` entry::

   res = PPSCM({..., "conformal_horizon": 8}).fit()
   for label, uf in res.per_unit.items():
       print(label, uf.cumulative_effect,
             (uf.cumulative_lower, uf.cumulative_upper), uf.cumulative_windows)

An interval for a running total is not the running total of the per-period
intervals. Adding endpoints up treats the period errors as moving in lockstep, so the
width grows with the number of periods, not with its square root; rescaling a
single period's interval by the horizon assumes the opposite. Which is right depends
on how the errors accumulate, and neither assumption measures it.

So the band measures it. An origin slides across the pre-period, and at each one every
treated unit is treated as if it had adopted there: partially-pooled SCM fits them all
in a single solve, so one pass yields each unit's summed error over the following
window at the cost of one solve per origin, not one per unit per origin. Those sums are
conformity scores for exactly the quantity being reported, and the half-width is the
:math:`\lceil (m+1)(1-\alpha) \rceil`-th order statistic of the centred scores
(:func:`mlsynth.utils.conformal.cumulative_conformal_interval`, shared with
VanillaSC's ``inference="conformal_cumulative"``). Each fit sees only data before the
window it scores, so the scores carry no in-sample optimism, and origins step by a
whole horizon, so the windows do not overlap.

Two things to note. The band is *additional*, not a mode: ``inference_method`` still
chooses the bootstrap or jackknife behind the ATT, and leaving ``conformal_horizon``
unset changes nothing. And it costs pre-period. Non-overlapping windows of length
:math:`L` are scarce, so a :math:`1-\alpha` band needs at least
:math:`\lceil 1/\alpha \rceil - 1` of them: roughly
:math:`T_0 \gtrsim L/(0.7\,\alpha)` counting the training block held back at the
start. When they run out, ``cumulative_lower``/``cumulative_upper`` are infinite and
``cumulative_windows`` says how many were available, instead of a narrow band that
does not cover.

Where the roll starts is ``conformal_min_train_frac``, a fraction of the
pre-period: the first origin sits at
:math:`\max(10,\ \text{frac} \times T_0)`, so every calibration fit has that many
periods to train on. The default 0.3 suits most panels, and two situations call
for moving it. Periods spent on the warm-up are periods not available for
calibration, so lowering it buys windows when the level is out of reach;
against that, a fit trained on fewer periods than there are donors can
interpolate its training window, and raising the fraction past the donor count
removes those origins. The two pull opposite ways, so the choice belongs to
whoever knows the panel. With :math:`T_0 = 120`, :math:`L = 7` and 60 donors,
the default starts at period 36 and yields twelve windows, four of them trained
on fewer periods than there are donors; 0.5 starts at 60 and removes all four,
leaving eight windows, which no longer supports a 90% band. Read
``cumulative_windows`` back off each unit to see what a given choice bought.

  ===========  =====  =========  ===============================================
  :math:`T_0`  frac   windows    at :math:`L = 7`
  ===========  =====  =========  ===============================================
  120          0.3    12         supports 90%, not 95%
  120          0.4    10         supports 90%
  120          0.5    8          below the 90% threshold
  ===========  =====  =========  ===============================================

The cumulative effect overall
-----------------------------

The band above is per unit. The corresponding question about the pool is what the
treated units gained in total over the first :math:`L` periods, and
``cumulative_band=True`` answers it::

   res = PPSCM({..., "cumulative_band": True}).fit()
   band = res.inference_detail.cumulative
   for L, point, lo, hi in zip(band.horizons, band.point, band.lower, band.upper):
       print(L, point, (lo, hi))

Both the jackknife and the wild bootstrap already fit the estimator many times and
get a whole per-horizon path back from each fit. Those paths are kept on
``res.inference_detail.replicate_paths``, and the band is built from them, so it
costs no refits beyond the inference that was going to run anyway.

Keeping them is what makes the band possible. Collapsing each replicate to one
standard error per horizon -- which is all a per-period band needs -- discards how
the horizons move together, and that covariance is the entire content of a
cumulative interval. A caller with only the collapsed standard errors has to
choose an assumption instead: adding period interval endpoints treats the errors
as moving in lockstep and grows the width like :math:`L`, while rescaling a
single period's interval assumes they are independent and grows it like
:math:`\sqrt{L}`. Accumulating the replicates before taking the standard error
measures which is true.

The band is *simultaneous*. A cumulative path is read as a path -- "the total is
positive by week six and stays there" is a claim about every horizon at once --
and a pointwise band read that way covers at well below its nominal level, by
more as the number of horizons grows. One shared critical value
(:func:`mlsynth.utils.supt.supt_critical_value`, the sup-t construction of
Montiel Olea and Plagborg-Moller) restores the level for the whole path.

Which ensemble produced the band is recorded on ``band.method``, because the two
are not interchangeable. The wild bootstrap reweights each unit's residual by an
independent multiplier, which does not cancel the common factors the synthetic
weights cancel in the point estimate, so its replicate variance is inflated where
factor structure is strong. The delete-one jackknife refits the weights on each
leave-one-out, so the factors re-cancel per replicate. The jackknife replicates
also carry the delete-one inflation and the bootstrap draws do not, since the
latter are already on the estimator's scale; the band applies whichever matches.

Empirical Illustration: mandatory collective bargaining
-------------------------------------------------------

The ``multisynth`` vignette studies the effect of state mandatory
collective-bargaining laws on log per-pupil education expenditure
(Paglayan 2018), a staggered design. ``basedata/Teachingaugsynth.scv`` ships the
panel; the analysis restricts to 1959--1997, drops DC and Wisconsin, and treats
a state from the year it required bargaining.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from mlsynth import PPSCM

   url = "https://raw.githubusercontent.com/jgreathouse9/mlsynth/refs/heads/main/basedata/Teachingaugsynth.scv"
   df = pd.read_csv(url)
   df = df[~df["State"].isin(["DC", "WI"])]
   df = df[(df["year"] >= 1959) & (df["year"] <= 1997)].copy()
   df["cbr"] = (df["year"] >= df["YearCBrequired"].fillna(np.inf)).astype(int)

   res = PPSCM({"df": df, "outcome": "lnppexpend", "treat": "cbr",
                "unitid": "State", "time": "year", "display_graphs": True}).fit()

   print(f"nu (auto)   : {res.design.nu_used:.4f}")
   print(f"Average ATT : {res.att:.3f}  (SE {res.inference.se:.3f})")

This prints::

   nu (auto)   : 0.2607
   Average ATT : -0.011  (SE 0.020)

reproducing the augsynth vignette (``nu = 0.2607``, Average ATT ``-0.011``).
Setting ``time_cohort=True`` collapses to adoption-time cohorts and gives
``nu = 0.3939``, Average ATT ``-0.017`` (augsynth: ``-0.018``).

Verification
------------

.. note::

   Exact replication of augsynth. On the Paglayan data PPSCM matches
   ``augsynth::multisynth`` to high precision: the auto-:math:`\nu` agrees to
   four decimals (0.2607 default, 0.3939 time-cohort), the Average ATT matches
   (:math:`-0.011` default; :math:`-0.017` vs :math:`-0.018` time-cohort), and
   the raw global/individual L2 imbalances agree (0.003 / 0.028). The full
   relative-time event study matches the vignette's per-horizon averages to
   3--4 decimals. The decisive fidelity detail is aligning the pooled
   imbalance by relative time on top of two-way fixed effects. The jackknife SE
   (0.020) is close to augsynth's default wild-bootstrap SE (0.022); they differ
   only by inference procedure. This is locked in by
   ``test_matches_augsynth_vignette`` in ``mlsynth/tests/test_ppscm.py``.

A cross-package case on real data covers both modes at once: the
cannabis-alcohol panel of Ronczewski (2026), which runs ``augsynth`` and
``did`` side by side on one sample. PPSCM's default reproduces the published
``multisynth`` ATT to 8.2e-08 and its ``callaway_santanna`` mode reproduces
the published ``did::aggte`` simple aggregate to 4.9e-17, with all six
dynamic event-study coefficients to 2.0e-16. Both need ``n_leads = 6``, which
the paper sets against a default of 2. See `benchmarks/cases/ronczewski_cannabis.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/ronczewski_cannabis.py>`_.

Core API
--------

.. automodule:: mlsynth.estimators.ppscm
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.PPSCMConfig
   :members:
   :undoc-members:

Result Containers
-----------------

``PPSCM.fit()`` returns a
:class:`~mlsynth.utils.ppscm_helpers.structures.PPSCMResults`: the
:class:`~mlsynth.utils.ppscm_helpers.structures.PPSCMDesign` (pooling level and
balance diagnostics), the relative-time
:class:`~mlsynth.utils.ppscm_helpers.structures.PPSCMEventStudy`, the overall
:class:`~mlsynth.utils.ppscm_helpers.structures.PPSCMInference`, and the
per-cohort donor weights.

.. automodule:: mlsynth.utils.ppscm_helpers.structures
   :members:
   :undoc-members:
   :show-inheritance:

Helper Modules
--------------

Staggered long-to-wide formatting (the only DataFrame touchpoint): derive
adoption times, split pre/post at the last adoption.

.. automodule:: mlsynth.utils.ppscm_helpers.setup
   :members:
   :undoc-members:

The engine: two-way fixed effects (``fit_feff``), the partially-pooled QP,
auto-:math:`\nu`, and the relative-time event study / ATT.

.. automodule:: mlsynth.utils.ppscm_helpers.engine
   :members:
   :undoc-members:

The paper's delete-one jackknife inference.

.. automodule:: mlsynth.utils.ppscm_helpers.inference
   :members:
   :undoc-members:
