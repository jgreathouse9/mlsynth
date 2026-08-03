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
measured from each unit's own treatment date, not from the calendar.

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
treated ones, and it is constructed, not residual -- the donors are
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

Three properties of this construction each look like an implementation detail
and change the estimand.

The indexing changes the constraint, not just the units. Dividing unit
:math:`j` and every donor by their own value at :math:`T_{0j}` while requiring
:math:`\|\mathbf{w}\|_1 = 1` is algebraically the same as fitting on raw levels
under the constraint

.. math::

   \sum_{i \in \mathcal{N}_0} v_i \, y_{i,T_{0j}} = y_{j,T_{0j}} ,

where :math:`v_i \coloneqq w_i \, y_{j,T_{0j}} / y_{i,T_{0j}}`. In words: the
synthetic control must reproduce the treated unit's base-period level exactly.
That is why :math:`\widehat{\tau}_{-1}` is identically zero, not merely
small, and it is a different feasible set from the ordinary simplex on levels.
Setting ``normalize=False`` gives the level-scale estimator, which answers a
different question.

The base period belongs to the cohort. Every unit adopting at the same time
shares :math:`T_{0j}`, so the indexed donor block takes one value per adoption
time, not one per treated unit. With six adoption years and 566 treated
units that is six donor blocks, not 566, and every unit in a cohort is fitted
against the same design matrix.

The aggregation weights are part of the specification. Equal weighting and
size weighting answer different questions, and neither is a default the data
chooses for you. Weighting by population avoids letting large percentage
swings in tiny units drive the average; equal weighting treats each adoption
as one observation. Say which you mean.

How a Cohort's Weights Are Solved Together
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

That the cohort shares a design is not only a modelling point. It means the
cohort's weight programs are one family, and solving them together costs a
fraction of solving them one at a time.

The simplex constraint removes the design matrix from the objective. With
:math:`\mathbf{1}^\top\mathbf{w} = 1`,

.. math::

   \mathbf{A}\mathbf{w} - \mathbf{b}_j
   \;=\; \mathbf{A}\mathbf{w} - \mathbf{b}_j\,(\mathbf{1}^\top\mathbf{w})
   \;=\; (\mathbf{A} - \mathbf{b}_j\mathbf{1}^\top)\,\mathbf{w},

so the program for unit :math:`j` is the quadratic form
:math:`\mathbf{w}^\top \mathbf{G}_j \mathbf{w}` with :math:`\mathbf{G}_j =
(\mathbf{A} - \mathbf{b}_j\mathbf{1}^\top)^\top(\mathbf{A} -
\mathbf{b}_j\mathbf{1}^\top)`. Expanding it, and writing :math:`\mathbf{c}_j
= \mathbf{A}^\top \mathbf{b}_j` and :math:`s_j = \mathbf{b}_j^\top
\mathbf{b}_j`,

.. math::

   \mathbf{G}_j
   \;=\; \mathbf{A}^\top\mathbf{A}
   \;-\; \mathbf{c}_j \mathbf{1}^\top
   \;-\; \mathbf{1} \mathbf{c}_j^\top
   \;+\; s_j \mathbf{1}\mathbf{1}^\top .

Only :math:`\mathbf{c}_j` and :math:`s_j` depend on the unit, and both come from
:math:`\mathbf{A}^\top\mathbf{B}` and the column norms of :math:`\mathbf{B}`,
each formed once. So a cohort of any size costs one Gram and one cross product,
and no product with the data occurs per unit. The active set then runs the
cohort in lockstep: each iteration is one batched linear solve over the current
supports, so the cohort costs what its hardest unit costs. On a Wiltshire cohort
of 89 counties against 39 donors that is 43ms where the one-at-a-time solve
takes 396ms. A donor predicate that binds gives each pool its own batch, down to
a batch of one per unit when no two units share a pool.

Two things have to hold before a batched answer may be reported in place of the
one-at-a-time answer, and the weights make both strict. STACKEDSC reports
:math:`\mathbf{w}_j^\ast` per unit and builds each counterfactual from it, so
an equally optimal but different point is a changed answer, not a rounding
difference.

The first is that the point actually solves the program as posed on
:math:`\mathbf{A}`. Forming :math:`\mathbf{G}` squares the design's condition
number, so a design that is merely awkward at :math:`\operatorname{cond}
(\mathbf{A}) \approx 10^{7}` gives a Gram at the edge of double precision. The
batched solver then converges on that Gram to a point that does not solve the
program the Gram came from, and nothing computed from the Gram reveals it. This
is what covariate matching runs into: predictors measured in different units
spread the spectrum, and on the covariate specification of the Wiltshire panel
62 of 76 units in a cohort fail. So each answer is certified against
:math:`\mathbf{A}` itself, from the first-order conditions
(:func:`mlsynth.utils.bilevel.minnorm.simplex_point_is_optimal`), and the units
that fail are re-solved one at a time.

The second is that the program has one solution. Where the minimiser is a face
-- every point of it optimal -- two exact solvers have equal claim to different
weights, and with 5 to 10 pre-treatment periods against 39 donors a face is not
exotic. :func:`mlsynth.utils.bilevel.minnorm.simplex_optimum_is_unique` settles
it on the solution's own support once a solution is in hand: the objective is
flat along a direction only where that direction is feasible where the solution
sits, which needs a support large relative to the design's rank. Synthetic
control solutions are sparse, so most units qualify -- on the Wiltshire panel
85.5 percent of 2264 solves do -- and the rest are re-solved.

Diagnostics
-----------

The pre-treatment portion of the event study is the main diagnostic, and it is
close to free: under the indexing, :math:`\widehat{\tau}_{-1} = 0` by
construction, so the remaining pre-treatment horizons are informative about
fit, not about level.

Two things the result reports that are easy to overlook.
``design.shared_donor_pool`` records whether every treated unit in every cohort
faced the same donor block; it is false when a donor predicate binds, since
restricting donors per unit gives each unit its own design matrix. And
``per_unit`` carries every unit's own path and weights.

That last one carries a caution, sharper than it first looks. With a
donor pool large relative to the number of pre-treatment periods the individual
weight vectors are not identified: many weightings fit the pre-period equally
well while implying different post-treatment counterfactuals. This is not a
statement about approximate solvers -- it survives solving the program exactly.
On the Wiltshire panel a cohort has seven pre-treatment periods against 39
donors, so the optimum is a face of dimension at least 32, and two exact solvers
that agree on the objective to :math:`8.7 \times 10^{-6}` still return
per-county post-treatment paths differing by up to 2.1 percentage points.

The weighted mean is pinned down far better than its parts -- across three
solvers the population-weighted aggregate moves by 0.05 percent where the
individual paths move by 2.1 -- but it is not pinned exactly either, which is
why the durable benchmark bands the aggregate instead of pinning a single
solver's answer. Read ``per_unit`` for diagnosis and spread, not as a claim
about which donors resemble a particular unit.

The weights themselves come from a primal active-set method
(:func:`mlsynth.utils.bilevel.active_set.solve_simplex_qp`), which terminates on
a Karush-Kuhn-Tucker certificate, not on an iteration budget. That matters
here: on this design a first-order method does not converge at all, leaving 20 of
39 leave-one-out columns still improving after 20,000 iterations at any
tolerance, and so returns a point that is simply suboptimal.

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
set from injecting noise in place of removing bias. In the motivating
application the correction roughly doubles the estimate.

Inference
---------

Setting ``inference="placebo"`` runs the procedure Wiltshire uses. With one
treated unit the in-space placebo test of Abadie, Diamond and Hainmueller (2010)
has an obvious comparison set: reassign treatment to each donor in turn and see
how unusual the real gap path looks among the fakes. With many treated units the
estimate is already an average, so the comparison has to be against other
averages.

Fix a treated unit :math:`j` and a donor :math:`i \in \mathcal{N}_0`. Refit the
synthetic control with :math:`i` cast as treated at :math:`j`'s adoption time
:math:`g_j`, giving a placebo gap path :math:`\tau_e^{(j,i)}`. A placebo average
draws one donor :math:`i_j` per treated unit and averages under the same weights
the estimate uses,

.. math::

   \widehat{\tau}^{\,s}_e \coloneqq \sum_{j \in \mathcal{N}_1}
   \gamma_j \, \tau_e^{(j,\,i_j^s)} .

The number of distinct averages is :math:`\prod_j |\mathcal{N}_0|`, which for
39 donors and 566 treated units is :math:`39^{566}`, so :math:`S` of them are
sampled, not enumerated (``n_placebo_samples``, default 1000).

Two statistics come off that distribution. The first ranks a ratio, not a
level, so that an average which already fits badly before treatment is not
credited for a large gap after it. With :math:`\underline{E}` the earliest
reported horizon, write

.. math::

   R_s(E) \coloneqq
   \frac{(E+1)^{-1} \sum_{e=0}^{E} (\widehat{\tau}^{\,s}_e)^2}
        {|\underline{E}|^{-1} \sum_{e=\underline{E}}^{-1}
         (\widehat{\tau}^{\,s}_e)^2} ,
   \qquad
   p_E \coloneqq \frac{1}{S+1} \sum_{s=1}^{S}
   \mathbf{1}\bigl\{ R_s(E) \ge R_0(E) \bigr\} ,

where :math:`s = 0` denotes the estimate itself. The second reads the spread of
the sampled averages as a standard error, following Algorithm 4 of Arkhangelsky
et al. (2021):

.. math::

   \widehat{V}_e \coloneqq \frac{1}{S} \sum_{s=1}^{S}
   \bigl(\widehat{\tau}^{\,s}_e - \bar{\tau}_e\bigr)^2 ,
   \qquad
   \widehat{\tau}_e \pm z_{\alpha/2} \sqrt{\widehat{V}_e} .

The ranking assumes nothing about the shape of the null distribution; the
interval assumes homoskedasticity across units and asymptotic normality, which
with many treated units follows from each average being a sum of independent
draws. Both are reported because both are informative and they can disagree.

Two mechanical caveats. Under the base-period indexing
:math:`\widehat{\tau}_{-1} = 0` for every path including the placebos, so a
reported pre-window of that single period leaves the ratio :math:`0/0`; the
RMSPE p-values are then NaN and say so. And :math:`p_E` does not count the
estimate in its own numerator, matching the reference implementation, so a
p-value of exactly zero is attainable.

One statistical caveat, which is why the two are reported side by side. A donor
is usually easier to reconstruct from the other donors than a treated unit is
from any of them, so the placebo averages tend to fit their pre-treatment window
better than the estimate fits its own. Their denominators are smaller, their
ratios :math:`R_s(E)` correspondingly larger, and :math:`p_E` is pushed up: the
ranking can be indecisive on a panel where the interval is not. Hahn and Shi
(2017) and Ferman and Pinto (2017) discuss size distortion in placebo tests more
generally. Reading whichever of the two statistics is the more favourable
defeats the purpose of computing both.

``placebo_donor_pool`` chooses whether the treated unit itself joins each of its
placebos' donor pools. The default ``"permutation"`` says yes, following the
reference implementation, on the logic that under the permutation the treated
unit is a control like any other. It has two consequences. The placebo path
becomes a property of the pair :math:`(j, i)`, not of the cohort, so the
number of solves is the number of treated units times the pool size -- for the
Walmart panel, :math:`566 \times 39`, not :math:`6 \times 39`. And under
the alternative, the treated unit's post-treatment path carries the very effect
being tested, so any weight the placebo puts on it pulls that placebo's gap away
from zero and widens the null distribution. This is the stacked-case power loss
Zhang (2019, section 3.1) describes: as the number of treated units grows,
genuinely treated units enter the null distribution more and more often, moving
it away from the null it is meant to represent. ``"donors-only"`` drops the
treated unit, which removes that channel and recovers the per-cohort saving, at
the price of departing from the reference implementation.

On the Walmart panel the two pools give the same answer. The permutation pool
runs 22,074 solves against the other's 234, and both report an ATT of
:math:`-0.367` with a standard error of :math:`0.426` and a p-value of
:math:`0.389`, agreeing on the interval to three decimals. That is not a general
result -- with a smaller donor pool a single extra column would matter more --
but on a pool of 39 the Zhang channel is real in principle and negligible in
practice, so ``"donors-only"`` is a reasonable choice at a twenty-seventh of the
cost.

The whole procedure is opt-in for the same reason the reference makes it opt-in:
its cost scales with treated units times donors, and ``allsynth`` warns that
requesting it "will greatly extend the run-time". Concretely, on the paper's
566 treated counties and 39 donors: 2.2 seconds for the point estimate alone,
2.8 with the ``"donors-only"`` distribution, 76 with the default one.

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
   res.design.shared_donor_pool          # same donor block for every unit?
   res.per_unit["18003"].tau             # one county's own path

Asking for the permutation distribution as well, on a smaller donor pool so the
example runs quickly:

.. code-block:: python

   import numpy as np
   import pandas as pd
   from mlsynth import STACKEDSC

   rng = np.random.default_rng(0)
   F = rng.normal(size=(12, 2)).cumsum(axis=0)          # two common factors
   rows = []
   for k in range(10):                                  # never-treated donors
       load, base = rng.normal(size=2), 100 + 20 * rng.random()
       rows += [{"unit": f"d{k}", "t": t, "adopt": np.nan,
                 "y": base + F[t] @ load + rng.normal() * 0.05}
                for t in range(12)]
   for k in range(5):                                   # treated, two cohorts
       g = (6, 8)[k % 2]
       load, base = rng.normal(size=2), 100 + 20 * rng.random()
       rows += [{"unit": f"x{k}", "t": t, "adopt": float(g),
                 "y": (base + F[t] @ load + rng.normal() * 0.05)
                      * (0.9 if t >= g else 1.0)}
                for t in range(12)]
   df = pd.DataFrame(rows)
   df["treat"] = ((~df.adopt.isna()) & (df.t >= df.adopt)).astype(int)

   res = STACKEDSC({
       "df": df, "outcome": "y", "treat": "treat",
       "unitid": "unit", "time": "t",
       "inference": "placebo", "n_placebo_samples": 200, "seed": 0,
       "display_graphs": False,
   }).fit()

   print(res.effects.att, res.inference.p_value)     # ATT and its p-value
   print(res.inference.ci)                           # (lower, upper) for the ATT
   print(res.placebo.rmspe_p)                        # RMSPE-ranked p by horizon
   print(res.placebo.se)                             # placebo spread by horizon
   print(res.placebo.placebo_averages.shape)         # (200, n_horizons)

On this panel the interval is decisive and the ranking is not, for the reason
given above: the donors reconstruct one another almost exactly, so every placebo
average has a near-perfect pre-treatment fit and a large :math:`R_s(E)`.

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
adoption through cohort-level time weights, not per-unit event-time
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

.. autoclass:: mlsynth.utils.stackedsc_helpers.structures.StackedPlacebo
   :members:
