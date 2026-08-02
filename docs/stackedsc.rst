Stacked-in-Event-Time Synthetic Control (STACKEDSC)
===================================================

.. currentmodule:: mlsynth

When to Use This Estimator
--------------------------

Most synthetic control methods answer a question about one treated unit. Many
policies do not arrive that way. A retail chain opens stores in hundreds of
counties over a decade; a state law is adopted by different municipalities in
different years; a firm rolls a change out market by market. Each treated unit
gets its own intervention date, and the question is what happened on average,
measured from each unit's own treatment date rather than from the calendar.

STACKEDSC fits a separate synthetic control for every treated unit against a
common pool of never-treated donors, then lines the resulting effect paths up
on a shared event clock and averages them. The name comes from that stacking:
unit 1's third year after treatment sits alongside unit 2's third year, even
though those are different calendar years.

Reach for it when all of these hold: many treated units adopting at different
times, a donor pool of units that are never treated, and outcomes whose levels
differ substantially across units. That last condition is the one people
overlook, and it is why this estimator rescales before fitting. County
employment ranges from four thousand to nearly a million in the motivating
application. Matching such units on raw levels lets the largest ones dominate
the fit, and an effect of "three thousand jobs" means something entirely
different in the two places.

Notation
--------

Let :math:`\mathcal{N} \coloneqq \{1,\dots,N\}` index units. Unlike the
single-treated-unit case there is no distinguished :math:`j = 1`: write
:math:`\mathcal{N}_1` for the treated units and
:math:`\mathcal{N}_0 \coloneqq \mathcal{N}\setminus\mathcal{N}_1` for the
never-treated donor pool, with :math:`N_0 \coloneqq |\mathcal{N}_0|`.

Time runs over :math:`t \in \mathcal{T} \coloneqq \{1,\dots,T\}`. Each treated
unit :math:`j \in \mathcal{N}_1` has its own adoption time :math:`g_j`, so its
last untreated period is :math:`T_{0j} \coloneqq g_j - 1`. Event time is

.. math::

   e \coloneqq t - g_j ,

so :math:`e = -1` is the final pre-treatment period and :math:`e = 0` the first
treated one. Units sharing an adoption time form a cohort; write
:math:`\mathcal{G}` for the set of distinct adoption times.

The observed outcome is :math:`y_{jt}`, the donor matrix
:math:`\mathbf{Y}_0 \in \mathbb{R}^{T \times N_0}`. Because each treated unit is
rescaled to its own base period, define the indexed series

.. math::

   \widetilde{y}_{jt} \coloneqq 100 \cdot \frac{y_{jt}}{y_{j,T_{0j}}} ,

and likewise :math:`\widetilde{\mathbf{Y}}_0` for the donors, indexed to the
same period :math:`T_{0j}`.

Assumptions
-----------

Assumption 1 (Never-treated donors). The units in :math:`\mathcal{N}_0` are
untreated throughout :math:`\mathcal{T}`.

Remark. This is stronger than the not-yet-treated condition some staggered
estimators use, and it is deliberate: a not-yet-treated donor contributes
untreated information early and treated information later, which contaminates
long-horizon effects precisely where they are largest. The cost is a smaller
donor pool. In the motivating application the pool is 39 counties for 566
treated ones, and it is constructed rather than residual -- the donors are
places where the firm tried to open and was blocked, which is what makes them
comparable.

Assumption 2 (Pre-treatment fit). For each treated unit there exist weights on
the simplex reproducing its indexed pre-treatment path.

Remark. The usual convex-hull condition, applied after indexing. Indexing
makes it easier to satisfy: units of wildly different size can have similar
growth paths even when their levels are nowhere near each other. It is also
checkable, which is what the pre-treatment portion of the event study is for.
A fit that misses before treatment is not evidence about what happened after.

Assumption 3 (Common support in event time). Every treated unit is observed
over the reported window of event times.

Remark. Reporting a horizon only some units reach makes the average change
composition along the horizontal axis, so a trend can appear that is really
units entering and leaving. The default window is the widest one every unit
supplies; ``n_lags`` and ``n_leads`` narrow it deliberately.

Estimation
----------

For treated unit :math:`j`, donor weights solve the usual simplex-constrained
program on the indexed pre-treatment series:

.. math::

   \mathbf{w}_j^\ast \in \operatorname*{argmin}_{\mathbf{w}\in\Delta^{N_0}}
   \bigl\| \widetilde{\mathbf{y}}_j^{\,\text{pre}}
   - \widetilde{\mathbf{Y}}_0^{\,\text{pre}} \mathbf{w} \bigr\|_2^2 ,

with :math:`\Delta^{N_0} \coloneqq \{\mathbf{w} \in
\mathbb{R}_{\ge 0}^{N_0} : \|\mathbf{w}\|_1 = 1\}`. The per-unit effect at
event time :math:`e` is

.. math::

   \tau_{je} \coloneqq \widetilde{y}_{j,g_j+e}
   - \bigl(\widetilde{\mathbf{Y}}_0 \mathbf{w}_j^\ast\bigr)_{g_j+e} ,

and the reported effect is a weighted average over treated units,

.. math::

   \widehat{\tau}_e \coloneqq \sum_{j \in \mathcal{N}_1} \gamma_j \tau_{je},
   \qquad \gamma_j \ge 0, \quad \sum_{j} \gamma_j = 1 .

Three properties of this construction deserve stating plainly, because each
looks like an implementation detail and none is.

The indexing changes the constraint, not just the units. Dividing unit
:math:`j` and every donor by their own value at :math:`T_{0j}` while requiring
:math:`\|\mathbf{w}\|_1 = 1` is algebraically the same as fitting on raw levels
under the constraint

.. math::

   \sum_{i \in \mathcal{N}_0} v_i \, y_{i,T_{0j}} = y_{j,T_{0j}} ,

where :math:`v_i \coloneqq w_i \, y_{j,T_{0j}} / y_{i,T_{0j}}`. In words: the
synthetic control must reproduce the treated unit's base-period level exactly.
That is why :math:`\widehat{\tau}_{-1}` is identically zero rather than merely
small, and it is a different feasible set from the ordinary simplex on levels.
Setting ``normalize=False`` gives the level-scale estimator, which answers a
different question.

The base period belongs to the cohort. Every unit adopting at the same time
shares :math:`T_{0j}`, so the indexed donor block takes one value per adoption
time rather than one per treated unit. With six adoption years and 566 treated
units that is six donor blocks, not 566, and each cohort becomes a single
least-squares problem with many right-hand sides rather than :math:`N_g`
separate ones.

The aggregation weights are part of the specification. Equal weighting and
size weighting answer different questions, and neither is a default the data
chooses for you. Weighting by population avoids letting large percentage
swings in tiny units drive the average; equal weighting treats each adoption
as one observation. Say which you mean.

Diagnostics
-----------

The pre-treatment portion of the event study is the main diagnostic, and it is
close to free: under the indexing, :math:`\widehat{\tau}_{-1} = 0` by
construction, so the remaining pre-treatment horizons are informative about fit
rather than about level.

Two things the result reports that are easy to overlook. ``design.batched``
records whether each cohort was solved as one program; it is false when a donor
predicate binds, since restricting donors per unit gives each unit its own
design matrix. And ``per_unit`` carries every unit's own path and weights.

That last one deserves a caution. With a donor pool large relative to the
number of pre-treatment periods the individual weight vectors are not
identified: many weightings fit the pre-period equally well while implying
different post-treatment counterfactuals. The weighted mean is pinned down far
better than its parts. Read ``per_unit`` for diagnosis and spread, not as a
claim about which donors resemble a particular unit.

Bias correction
---------------

When the synthetic control does not balance the predictors exactly, the residual
imbalance can itself move the outcome. Setting ``bias_correct`` applies the
Abadie and L'Hour (2021) adjustment, subtracting the part of the gap attributable
to that imbalance:

.. math::

   \tau_{je}^{\text{bc}} \coloneqq \tau_{je}
   - \bigl(\mathbf{x}_j - \mathbf{X}_0 \mathbf{w}_j^\ast\bigr)^\top
     \boldsymbol{\beta}_e ,

with :math:`\boldsymbol{\beta}_e` the ridge-regression slope of the donor
outcome on the donor predictors. The correction is identically zero when the
predictors balance, and the ridge is what keeps a weakly explanatory predictor
set from injecting noise in place of removing bias. This is not a refinement
around the edges: in the motivating application it roughly doubles the estimate.

Example
-------

.. code-block:: python

   import pandas as pd
   from mlsynth import STACKEDSC

   # long panel: one row per (unit, period), with an adoption year per
   # treated unit and a size variable to weight the average by
   df = pd.read_parquet("basedata/allsynth_walmart.parquet")
   df["treat"] = ((df.supercenter == 1) & (df.year >= df.super_year)).astype(int)

   res = STACKEDSC({
       "df": df, "outcome": "emps_n10", "treat": "treat",
       "unitid": "cty_fips", "time": "year",
       "agg_weights": "pop90",          # gamma_j; None gives equal weights
       "n_lags": 5, "n_leads": 6,       # the balanced event window
       "display_graphs": False,
   }).fit()

   res.effects.att                       # mean post-treatment effect, percent
   res.event_study.tau                   # the path, one entry per horizon
   res.design.cohorts                    # the distinct adoption times
   res.design.batched                    # one solve per cohort?
   res.per_unit["18003"].tau             # one county's own path

Verification
------------

Reproduced against Wiltshire (2023) on the author's own panel: the
pre-treatment path is flat within :math:`\pm 0.07` percent, the effect at
entry is :math:`+0.16`, and the decline begins at :math:`e = 2` -- the shape the
paper reports. The point estimates are not claimed, because the reference
implementation's predictor-weight rule ships compiled and two defensible
readings of it move the five-year estimate by more than the estimate itself.
See :doc:`replications/stackedsc` for what that means and what would settle it,
and `benchmarks/cases/wiltshire_walmart.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/wiltshire_walmart.py>`_
for the pinned quantities.

Not to be confused with
-----------------------

:doc:`ppscm` also handles staggered adoption but partially pools across treated
units, shrinking each unit's fit toward a common one; STACKEDSC fits every unit
separately and pools only at the averaging step. :doc:`sdid` handles staggered
adoption through cohort-level time weights rather than per-unit event-time
stacking.

Core API
--------

.. autoclass:: STACKEDSC
   :members: fit

.. autoclass:: mlsynth.utils.stackedsc_helpers.config.STACKEDSCConfig
   :members:

.. autoclass:: mlsynth.utils.stackedsc_helpers.structures.STACKEDSCResults
   :members:

.. autoclass:: mlsynth.utils.stackedsc_helpers.structures.StackedUnitFit
   :members:
