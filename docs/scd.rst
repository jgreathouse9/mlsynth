Synthetic Control with Differencing (SCD)
=========================================

.. currentmodule:: mlsynth

Overview
--------

``SCD`` is a synthetic control estimator for settings where the treated and
control "units" are not single aggregated series but *groups observed through
repeated cross-sections of individuals* -- states in a labour survey, regions in
a repeated household survey, markets sampled afresh each period. It is the method
of Rincon & Song (2026), with the repeated-cross-section inference of Canen &
Song (2025).

The idea is a two-step one. First collapse the microdata to a survey-weighted
mean per group and period, so that a panel of group means emerges from the
individual records. Then run a synthetic control on those means -- but on their
*within-group differences* rather than their levels. Differencing off a
pre-period baseline removes each group's fixed level, so the donor weights are
fit on de-trended trajectories; this is the "D" in SCD, and it nests classical
synthetic control (no differencing) and a difference-in-differences style
baseline (difference off the last pre-period) as the two ends of a dial.

The payoff of keeping the microdata is inference. Classical synthetic control
has one treated series and draws its uncertainty from permuting the handful of
donors -- a coarse grid with no power below one over the donor count. SCD instead
recycles the individual observations as influence functions, giving a genuine
:math:`\sqrt{n}` (number of individuals), fixed-:math:`T` standard-error band on
the per-period effect, together with a confidence set for the counterfactuals the
data cannot rule out. The result is naturally read as an event study.

When to use this estimator
--------------------------

* The treated and control units are groups, and you observe *individuals* within
  each group and period (microdata / repeated cross-sections), optionally with
  survey weights -- not a single number per group-period.
* You want a per-period effect path with honest confidence bands, and the number
  of donor groups is too small for credible permutation inference.
* A group-level fixed effect (a persistent level gap between the treated group
  and the donors) is plausible, so differencing it out before fitting improves
  the match.

If you only have one aggregated outcome per unit and period, use the aggregate
synthetic-control estimators (:doc:`vanillasc`, :doc:`clustersc`) -- there are no
individuals to form the :math:`\sqrt{n}` band. If you care about the whole
counterfactual *distribution* rather than the group mean, use :doc:`dsc`.

Notation
--------

Index groups by :math:`j`, with :math:`j = 0` the treated group and
:math:`\mathcal{N}_0 \coloneqq \{1, \dots, K\}` the donor pool of :math:`K`
groups. Time runs over :math:`t \in \{1, \dots, T\}`, 1-indexed; treatment takes
effect at :math:`T^\star = T_0 + 1`, splitting time into the pre-period
:math:`\mathcal{T}_1 \coloneqq \{t \le T_0\}` and the post-period
:math:`\mathcal{T}_2 \coloneqq \{t \ge T^\star\}`.

Within group :math:`j` at period :math:`t` we observe individuals
:math:`i` with outcome :math:`Y_{i}` and survey weight :math:`\pi_{i}`. The
survey-weighted group mean is

.. math::

   \widehat m_{j,t} \coloneqq
   \frac{\sum_i \pi_{i}\, Y_{i}\, \mathbf{1}\{G_i = j,\ t_i = t\}}
        {\sum_i \pi_{i}\, \mathbf{1}\{G_i = j,\ t_i = t\}} ,

with :math:`\widehat n_{j,t} \coloneqq \sum_i \pi_i \mathbf{1}\{G_i=j, t_i=t\}`
the weighted cell total and :math:`n` the total number of individuals.

Differencing is set by a weight vector :math:`\boldsymbol{\lambda} \in
\mathbb{R}^{T_0}` over the pre-period. The baseline level of group :math:`j` is
:math:`\bar m_j \coloneqq \sum_{t \le T_0} \lambda_t\, \widehat m_{j,t}`, and the
differenced series is :math:`\mu_{j,t} \coloneqq \widehat m_{j,t} - \bar m_j`.
Three schemes are offered: ``did`` uses
:math:`\boldsymbol{\lambda} = (0, \dots, 0, 1)` (difference off the last
pre-period), ``uniform`` uses :math:`\lambda_t = 1/T_0` (off the pre-period
average), and ``sc`` uses :math:`\boldsymbol{\lambda} = \mathbf{0}` (no
differencing -- classical synthetic control on levels).

Donor weights are the simplex vector :math:`\mathbf{w} \in \Delta^{K-1}`
(:math:`w_j \ge 0`, :math:`\sum_j w_j = 1`). The per-period effect is
:math:`\widehat\theta_t \coloneqq \mu_{0,t} - \sum_{j=1}^K \mu_{j,t}\, w_j`, and
the ATT is :math:`\widehat\tau \coloneqq |\mathcal{T}_2|^{-1}
\sum_{t \in \mathcal{T}_2} \widehat\theta_t`. The significance level is
:math:`\alpha`, split by a Bonferroni share :math:`\kappa` for the weight
confidence set.

Identifying assumptions
-----------------------

1. Pre-treatment fit of the differenced means. There exist simplex weights
   :math:`\mathbf{w}` under which the donors reproduce the treated group's
   differenced pre-period path, :math:`\mu_{0,t} \approx \sum_j \mu_{j,t} w_j`
   for :math:`t \in \mathcal{T}_1`.

   *Remark.* As in every synthetic control, a good pre-fit is the empirical
   certificate of a credible counterfactual. Differencing changes what is being
   matched: with ``did`` or ``uniform`` the fit ignores persistent level gaps and
   targets the co-movement of trajectories.

2. No anticipation. Treatment has no effect before :math:`T^\star`, so the
   pre-period group means reflect the no-intervention path.

   *Remark.* Date :math:`T^\star` at the first plausible response, not the
   nominal policy date; a contaminated pre-period biases the extrapolation.

3. No interference and untreated donors (SUTVA at the group level). Treating
   group :math:`0` changes no donor group's outcomes, and every donor group is
   untreated throughout.

   *Remark.* With grouped microdata this also rules out individuals sorting
   between the treated and donor groups in response to treatment.

4. Sampling within cells. Individuals are sampled (with weights :math:`\pi_i`)
   within each group-period cell, and the cells are populated. The
   repeated-cross-section variance treats the periods as independent samples;
   individuals are not tracked across time.

   *Remark.* This is the assumption that earns the :math:`\sqrt{n}`
   asymptotics. It is what distinguishes SCD from an aggregate panel method: the
   uncertainty comes from sampling individuals into the group means, with
   :math:`T` fixed.

Mathematical formulation
------------------------

Write the pre-period differencing as a linear operator. For a series
:math:`x = (x_1, \dots, x_T)` and pre-period weights :math:`\boldsymbol\lambda`,
let :math:`(\mathcal D_\lambda x)_t \coloneqq x_t - \sum_{s \le T_0} \lambda_s
x_s`, so that :math:`\mu_{j,t} = (\mathcal D_\lambda \widehat m_{j,\cdot})_t`.
The weights are fit by simplex least squares on the transformed pre-period,

.. math::

   \widehat{\mathbf{w}} = \operatorname*{argmin}_{\mathbf{w} \in \Delta^{K-1}}
   \; \sum_{t \le T_0}
   \Bigl( \mu_{0,t} - \sum_{j=1}^K \mu_{j,t}\, w_j \Bigr)^2 ,

and the effect path :math:`\widehat\theta_t = \mu_{0,t} - \sum_j \mu_{j,t}
\widehat w_j` is read off over all periods. Because the pre-period residuals are
near zero by construction, the pre-period :math:`\widehat\theta_t` trace the fit
quality and the post-period :math:`\widehat\theta_t` are the dynamic effects.

Differencing as an intercept. Because :math:`\mathcal D_\lambda` subtracts a
constant in :math:`t` from each series, the fitted gap can be written on the raw
means with an additive offset,

.. math::

   \widehat\theta_t = \Bigl( \widehat m_{0,t} - \sum_j \widehat w_j
   \widehat m_{j,t} \Bigr) - \alpha, \qquad
   \alpha = \bar m_0 - \sum_j \widehat w_j\, \bar m_j ,

so SCD is a synthetic control with an intercept :math:`\alpha`: it matches the
donors to the treated group's movement, not its level. The scheme sets how the
intercept is chosen. Under ``uniform`` (:math:`\lambda_t = 1/T_0`) both sides
are demeaned by their pre-period averages, and by the Frisch-Waugh-Lovell
theorem the resulting :math:`\widehat{\mathbf w}` is exactly that of a synthetic
control with a freely estimated (least-squares) intercept -- profiling
:math:`\alpha` out of :math:`\min_{\mathbf w \in \Delta,\,\alpha} \sum_{t \le
T_0}(\widehat m_{0,t} - \alpha - \sum_j w_j \widehat m_{j,t})^2` returns the
demeaned objective. Under ``did`` (:math:`\boldsymbol\lambda = (0, \dots, 0,
1)`) the intercept is instead pinned so the counterfactual meets the treated
group exactly at :math:`t = T_0` (the difference-in-differences normalisation),
which is not the pre-period error-minimising :math:`\alpha`. Under ``sc``
(:math:`\boldsymbol\lambda = \mathbf 0`) the intercept is :math:`\alpha = 0`:
classical synthetic control on levels.

Generalized matching condition
------------------------------

Differencing is what lets SCD tolerate a persistent level gap between the
treated group and the donors, and it does so by weakening the matching
requirement. Suppose the group means follow a linear factor (interactive
fixed-effects) model

.. math::

   \widehat m_{j,t} = a_j + \mathbf F_t^\top \boldsymbol\Gamma_j + \varepsilon_{j,t},

with a group fixed effect :math:`a_j`, common factors :math:`\mathbf F_t`, group
loadings :math:`\boldsymbol\Gamma_j`, and mean-zero sampling noise
:math:`\varepsilon_{j,t}`. Applying :math:`\mathcal D_\lambda` with
:math:`\sum_{s \le T_0} \lambda_s = 1` cancels the fixed effect,

.. math::

   \mu_{j,t} = (\mathbf F_t - \bar{\mathbf F}_\lambda)^\top \boldsymbol\Gamma_j
   + \widetilde\varepsilon_{j,t}, \qquad
   \bar{\mathbf F}_\lambda = \sum_{s \le T_0} \lambda_s \mathbf F_s ,

so the pre-period matching :math:`\mu_{0,t} = \sum_j w_j \mu_{j,t}` reduces to a
condition on loadings alone. The generalized matching condition is that the
treated group's loading lies in the convex hull of the donors',

.. math::

   \boldsymbol\Gamma_0 = \sum_{j=1}^K w_j\, \boldsymbol\Gamma_j, \qquad
   \mathbf w \in \Delta^{K-1} .

Under it the no-treatment gap :math:`\mu_{0,t} - \sum_j w_j \mu_{j,t}` is mean
zero for every :math:`\boldsymbol\lambda` with :math:`\sum_s \lambda_s = 1`, so
:math:`\widehat\theta_t` is consistent for the post-period treatment effect and
the choice between ``did`` and ``uniform`` is a finite-sample efficiency choice,
not an identification one. The three schemes are the two ends and the middle of
a single dial:

* ``did`` and ``uniform`` (:math:`\sum_s \lambda_s = 1`) need only loading
  matching :math:`\boldsymbol\Gamma_0 = \sum_j w_j \boldsymbol\Gamma_j`; the
  fixed effect :math:`a_j` is differenced away and may differ freely across
  groups;
* ``sc`` (:math:`\boldsymbol\lambda = \mathbf 0`) does not remove the fixed
  effect, so classical synthetic control additionally requires level matching
  :math:`a_0 = \sum_j w_j a_j` -- the stronger demand that the donors reproduce
  the treated group's level, not only its factor structure.

Differencing therefore generalises the matching requirement: it buys
identification under the weaker loadings-only condition, at the price of leaving
the level unidentified -- which is immaterial, because the reported object is
itself a difference.

Inference
---------

Pointwise variance. Each individual enters its group mean through the influence
function

.. math::

   \psi^\star_{i,t} = \pi_{i}\, \frac{n}{\widehat n_{G_i,t}}\,
   \bigl( Y_{i} - \widehat m_{G_i,t} \bigr),

and its contribution to :math:`\widehat\theta_t = \mu_{0,t} - \sum_j \mu_{j,t}
w_j` carries the sign and weight of its group,
:math:`\psi^\theta_{i,t} = c_{G_i}\, \psi^\star_{i,t}` with :math:`c_0 = 1`
(treated) and :math:`c_j = -\widehat w_j` (donor :math:`j`). Writing
:math:`S_t \coloneqq \sum_{i:\,t_i = t} (\psi^\theta_{i,t})^2` for the
per-period sum and :math:`\delta_t = \lambda_t` for :math:`t \le T_0` (and
:math:`0` after), the repeated-cross-section pointwise variance is

.. math::

   \widehat\sigma^2_t = \frac{1}{n}\Bigl[ \sum_{s \le T_0} \delta_s^2\, S_s
   + (1 - 2\delta_t)\, S_t \Bigr], \qquad
   \widehat{\mathrm{se}}_t = \sqrt{\widehat\sigma^2_t / n} .

The first term is the sampling noise of the differencing baseline
:math:`\bar m_j`; the second is the period's own noise (at a post-period,
:math:`\delta_t = 0` and only :math:`S_t` remains). Because the group means are
:math:`\sqrt{n}`-consistent with :math:`T` fixed, this is a genuine
standard-error band, not a permutation p-value on a coarse donor grid.

Weight confidence set. Let :math:`\mathbf G \in \mathbb R^{T_0 \times K}` hold
the differenced donor means :math:`\mu_{j,t}` and :math:`\mathbf g` the treated
column, and form the pre-period moment operators

.. math::

   \widehat H = \tfrac{1}{T_0}\, \mathbf G^\top \mathbf G, \qquad
   \widehat h = \tfrac{1}{T_0}\, \mathbf G^\top \mathbf g, \qquad
   \varphi(\mathbf w) = \widehat H \mathbf w - \widehat h ,

so :math:`\varphi(\widehat{\mathbf w}) = \mathbf 0` at the interior optimum. The
weights carry a :math:`(K-1)`-dimensional asymptotic variance

.. math::

   \widehat V = \frac{1}{T_0} \sum_{t \le T_0} \widehat v_t\, B_2^\top M_t B_2,
   \qquad \widehat v_t = S_t / n ,

with :math:`\mu_t` the :math:`t`-th row of :math:`\mathbf G`, :math:`\bar\mu`
its pre-period mean,

.. math::

   M_t = \tfrac{1}{T_0}\, \mu_t \mu_t^\top
   - \lambda_t\bigl( \bar\mu\, \mu_t^\top + \mu_t\, \bar\mu^\top \bigr)
   + T_0\, \lambda_t^2\, \bar\mu\, \bar\mu^\top ,

and :math:`B_2 \in \mathbb R^{K \times (K-1)}` an orthonormal basis of the
sum-zero subspace :math:`\{\mathbf v : \mathbf 1^\top \mathbf v = 0\}` (the
non-null eigenvectors of the centring matrix :math:`I_K - \tfrac1K \mathbf 1
\mathbf 1^\top`), which absorbs the degree of freedom the simplex constraint
removes. Set :math:`P = B_2 \widehat V^{-1} B_2^\top`.

A candidate :math:`\mathbf w` is in the level-:math:`(1 - \kappa)` set when its
projected moment is small. Project :math:`\varphi(\mathbf w)` onto the cone the
active simplex constraints allow,

.. math::

   \widehat r = \operatorname*{argmin}_{r \ge 0,\; \mathbf w^\top r = 0}
   \; (\varphi - r)^\top P\, (\varphi - r) ,

form the statistic and its data-dependent degrees of freedom

.. math::

   T(\mathbf w) = n\, (\varphi - \widehat r)^\top P\, (\varphi - \widehat r),
   \qquad
   d(\mathbf w) = \bigl\lvert \{ j : w_j = 0,\; \lvert [P(\varphi - \widehat
   r)]_j \rvert < \text{tol} \} \bigr\rvert ,

and accept :math:`\mathbf w \in \mathcal C_{1-\kappa}` iff :math:`T(\mathbf w)
\le \chi^2_{1-\kappa}(K - 1 - d(\mathbf w))`. For an interior :math:`\mathbf w`
(all :math:`w_j > 0`) the constraints are slack, :math:`\widehat r = 0`, and the
test is the plain quadratic form on :math:`K - 1` degrees of freedom -- the
reason a dense grid is swept in one batched form and only the sparse candidates
need the projection.

Band. With :math:`z = \Phi^{-1}\!\bigl(1 - (\alpha - \kappa)/2\bigr)` the
period-:math:`t` band runs over the accepted set,

.. math::

   \Bigl[ \min_{\mathbf w \in \mathcal C_{1-\kappa}}
   \bigl( \mu_{0,t} - \textstyle\sum_j w_j \mu_{j,t} \bigr)
   - z\, \widehat{\mathrm{se}}_t,\;\;
   \max_{\mathbf w \in \mathcal C_{1-\kappa}}
   \bigl( \mu_{0,t} - \textstyle\sum_j w_j \mu_{j,t} \bigr)
   + z\, \widehat{\mathrm{se}}_t \Bigr] ,

so the width combines weight uncertainty (the spread of counterfactuals the data
cannot reject) with sampling noise (:math:`z\, \widehat{\mathrm{se}}_t`), and the
two shares :math:`\kappa` and :math:`\alpha - \kappa` add to :math:`\alpha` by
Bonferroni.

A note on the RC standard error. The published reference code builds the
treated/donor multiplier :math:`c_{G_i}` with an indexing expression that, in R,
silently drops the treated rows and misaligns the donor weights; mlsynth uses
the corrected length-:math:`(K+1)` lookup (matching the reference's own
weight-variance routine :math:`\widehat V`). The point estimator, the weight
variance, and the confidence set are unaffected; only the pointwise standard
error differs, and mlsynth's value is cross-validated against a corrected base-R
reference.

Example
-------

.. code-block:: python

   import pandas as pd
   from mlsynth import SCD

   df = pd.read_parquet(
       "https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
       "refs/heads/main/basedata/cps_lawa_arizona.parquet"
   )
   # treatment is applied at the group-period level (Arizona from period 55)
   df["treat"] = ((df["state_name"] == "Arizona") & (df["D"] == 1)).astype(int)

   res = SCD({
       "df":          df,
       "outcome":     "wklyearn",     # individual log weekly earnings
       "treat":       "treat",
       "unitid":      "state_name",   # the group
       "time":        "period",
       "weight_col":  "weight",       # survey weights (optional)
       "differencing": "did",         # "did", "uniform", or "sc"
       "compute_inference": True,
       "alpha":       0.10,
       "kappa":       0.05,
       "display_graphs": False,
   }).fit()

   print(res.att)                                  # post-period mean effect
   print(res.time_series.estimated_gap)            # the effect path (event study)
   det = res.inference.details
   print(det["lower"], det["upper"])               # per-period confidence band
   print(det["se"])                                # RC pointwise standard errors

Verification
------------

SCD is cross-validated, value for value, against a from-scratch base-R
reproduction of the estimator (point weights, effect path, corrected RC standard
error, weight-variance trace, and confidence-set membership decisions) on the
Arizona LAWA CPS extract, for all three differencing schemes. The upstream
package is GPL, so the method is reproduced on public-domain CPS microdata rather
than vendored. Every quantity matches to solver tolerance
(:math:`\sim 10^{-9}`). See the durable benchmark case
`benchmarks/cases/scd_cps.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/scd_cps.py>`_
and the :doc:`replications/scd` page.

Core API
--------

.. automodule:: mlsynth.estimators.scd
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.utils.scd_helpers.config.SCDConfig
   :members:
   :undoc-members:
   :show-inheritance:
