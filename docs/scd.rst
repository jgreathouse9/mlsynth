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

The weights are fit by simplex least squares on the differenced pre-period,

.. math::

   \widehat{\mathbf{w}} = \operatorname*{argmin}_{\mathbf{w} \in \Delta^{K-1}}
   \; \sum_{t \le T_0}
   \Bigl( \mu_{0,t} - \sum_{j=1}^K \mu_{j,t}\, w_j \Bigr)^2 ,

and the effect path :math:`\widehat\theta_t = \mu_{0,t} - \sum_j \mu_{j,t}
\widehat w_j` is read off over all periods. Because the pre-period residuals are
near zero by construction, the pre-period :math:`\widehat\theta_t` trace the fit
quality and the post-period :math:`\widehat\theta_t` are the dynamic effects.

Inference
---------

Each individual contributes an influence function to its group mean's sampling
error,

.. math::

   \psi^\star_{i,t} = \pi_{i}\, \frac{n}{\widehat n_{G_i,t}}\,
   \bigl( Y_{i} - \widehat m_{G_i,t} \bigr),

and the repeated-cross-section pointwise variance of :math:`\widehat\theta_t`
accumulates these over individuals, giving a standard error
:math:`\widehat{\mathrm{se}}_t = \sqrt{\widehat\sigma^2_t / n}`. Because the
group means are :math:`\sqrt{n}`-consistent with :math:`T` fixed, this is a
genuine standard-error band, not a permutation p-value on a coarse donor grid.

The donor weights are themselves uncertain, so SCD adds a weight confidence set:
the collection of simplex weights the data cannot reject at level
:math:`1 - \kappa`, via a chi-squared test on the projected pre-period moment
:math:`\widehat H \mathbf{w} - \widehat h`. The band at period :math:`t` then
runs from the smallest to the largest counterfactual over weights in that set,
widened by :math:`\pm z_{1 - (\alpha - \kappa)/2}\, \widehat{\mathrm{se}}_t` --
a Bonferroni split that gives the weight set share :math:`\kappa` and the
pointwise term the remaining :math:`\alpha - \kappa`. mlsynth sweeps the weight
set with a vectorised construction: the membership quadratic program collapses
to a small non-negative least squares on the zero-set of :math:`\mathbf{w}`, so
dense candidates are tested in one batched form and only sparse candidates need a
solve.

A note on the RC standard error. The published reference code builds the
treated/donor multiplier with an indexing expression that, in R, silently drops
the treated rows and misaligns the donor weights; mlsynth uses the corrected
length-:math:`(K+1)` lookup (matching the reference's own weight-variance
routine). The point estimator, the weight variance, and the confidence set are
unaffected; only the pointwise standard error differs, and mlsynth's value is
cross-validated against a corrected base-R reference.

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
