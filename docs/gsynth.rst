Generalized Synthetic Control (GSYNTH)
======================================

.. currentmodule:: mlsynth

When to Use This Method
-----------------------

You have a panel, several units adopt a policy at dates that need not
coincide, and a good number of units never adopt at all. The units move
together through shared unobserved forces — a business cycle, a national
mood, a regional shock — but not in parallel, because each one responds to
those forces with its own sensitivity. Difference-in-differences assumes the
treated and untreated units share a common time effect, which is exactly what
heterogeneous responses break. Plain synthetic control builds a convex
combination of donors for one treated unit, and with nine treated units
adopting on four different dates it is not obvious what to do.

The generalized synthetic control method of Xu ([Xu2017]_) covers that
regime. It writes the untreated outcome as a small number of latent common
factors :math:`\mathbf{f}_t` weighted by unit-specific loadings
:math:`\boldsymbol{\lambda}_i`, plus additive unit and period effects and any
covariates. The units that are never treated identify the factors and the
covariate coefficients. Each treated unit is then placed in that estimated
factor space using its own pre-treatment history, and its untreated potential
outcome follows.

The result is a synthetic control that is fit by projection instead of by a
weighted average of donors: the treated unit is matched to the estimated
factors, not to particular control units. Nothing constrains the treated unit
to lie inside the convex hull of the donor pool, and units adopting at
different dates need no special handling, because each one's loadings come
from the pre-period block it happens to have.

Reach for GSYNTH when
^^^^^^^^^^^^^^^^^^^^^

* several units are treated, possibly at different dates, and a substantial
  group is never treated;
* the units co-move through latent common factors but not in parallel, so
  parallel trends is hard to defend;
* the pre-treatment histories are long enough to estimate a loading vector
  for each treated unit;
* you want the number of factors chosen by the data instead of asserted;
* you want the treated units aggregated into one ATT with a matching
  confidence interval and an effect path by time since adoption.

Do not use GSYNTH when
^^^^^^^^^^^^^^^^^^^^^^

* every unit is eventually treated. The factor space here comes from the
  never-treated units alone, and the estimator raises when there are none.
  :doc:`mcnnm` handles that design;
* treatment turns on and off. Step 2 projects onto one contiguous
  pre-treatment block per unit, which a reversing treatment does not define.
  :doc:`mcnnm` again, or :doc:`rolldid`;
* a treated unit's pre-period is shorter than the number of factors you want.
  Its loading vector is then unidentified and the fit raises instead of
  returning a number;
* the intervention plausibly moved the common factors themselves. The
  factors are estimated off units assumed to be unaffected, so a general
  equilibrium response contaminates them.

Notation
--------

Let :math:`\mathcal{T}` index the :math:`N_{tr}` treated units and
:math:`\mathcal{C}` the :math:`N_{co}` never-treated units, observed for
:math:`t = 1, \dots, T`. Unit :math:`i \in \mathcal{T}` adopts at
:math:`T_{0i} + 1`, so its pre-treatment block is :math:`t \le T_{0i}`. The
untreated potential outcome follows

.. math::

   Y_{it}(0) = \mathbf{x}_{it}^\top \boldsymbol{\beta}
               + \boldsymbol{\lambda}_i^\top \mathbf{f}_t
               + \alpha_i + \xi_t + \varepsilon_{it},

with :math:`\mathbf{f}_t` an :math:`r \times 1` vector of latent common
factors, :math:`\boldsymbol{\lambda}_i` unit :math:`i`'s loadings,
:math:`\mathbf{x}_{it}` observed time-varying covariates entering with a
common coefficient vector :math:`\boldsymbol{\beta}`, :math:`\alpha_i` and
:math:`\xi_t` additive unit and period effects, and
:math:`\varepsilon_{it}` idiosyncratic noise. Write
:math:`\mathbf{F} = (\mathbf{f}_1, \dots, \mathbf{f}_T)^\top` and let a
superscript :math:`0` restrict a quantity to a unit's pre-treatment periods.

The estimand is the average effect on the treated,

.. math::

   \widehat{ATT} = \frac{1}{|\{(i,t) : i \in \mathcal{T},\, t > T_{0i}\}|}
       \sum_{i \in \mathcal{T}} \sum_{t > T_{0i}}
       \bigl[ Y_{it}(1) - \widehat{Y}_{it}(0) \bigr],

and its path :math:`\widehat{ATT}_h` in time since adoption, with
:math:`h = 0` the first treated period.

The estimator in three steps
----------------------------

Step 1 fits the interactive fixed effects model to the never-treated units
alone, minimizing

.. math::

   \sum_{i \in \mathcal{C}}
   (\mathbf{Y}_i - \mathbf{X}_i \tilde{\boldsymbol{\beta}}
    - \tilde{\mathbf{F}} \tilde{\boldsymbol{\lambda}}_i)^\top
   (\mathbf{Y}_i - \mathbf{X}_i \tilde{\boldsymbol{\beta}}
    - \tilde{\mathbf{F}} \tilde{\boldsymbol{\lambda}}_i)

subject to Bai's normalization :math:`\mathbf{F}^\top \mathbf{F} / T =
\mathbf{I}_r` with :math:`\boldsymbol{\Lambda}_{co}^\top
\boldsymbol{\Lambda}_{co}` diagonal. Additive effects are swept out by
demeaning, and what remains alternates between a pooled regression for
:math:`\boldsymbol{\beta}` and principal components for
:math:`(\mathbf{F}, \boldsymbol{\Lambda})` until the coefficient vector stops
moving.

Step 2 recovers each treated unit's loadings from its own pre-treatment
periods, holding :math:`\widehat{\boldsymbol{\beta}}` and
:math:`\widehat{\mathbf{F}}` fixed:

.. math::

   \widehat{\boldsymbol{\lambda}}_i =
     (\widehat{\mathbf{F}}^{0\top} \widehat{\mathbf{F}}^0)^{-1}
      \widehat{\mathbf{F}}^{0\top}
      (\mathbf{Y}^0_i - \mathbf{X}^0_i \widehat{\boldsymbol{\beta}}),
   \qquad i \in \mathcal{T}.

Step 3 imputes and differences:
:math:`\widehat{Y}_{it}(0) = \mathbf{x}_{it}^\top \widehat{\boldsymbol{\beta}}
+ \widehat{\boldsymbol{\lambda}}_i^\top \widehat{\mathbf{f}}_t`.

When unit effects are in the specification a column of ones is appended to
:math:`\widehat{\mathbf{F}}` before Step 2, so a treated unit's own level is
estimated jointly with its loadings off its pre-periods. The control units'
unit effects say nothing about the level of a unit outside that group, so
this is the only place :math:`\alpha_i` for :math:`i \in \mathcal{T}` can
come from.

Which additive effects
----------------------

The ``force`` option chooses which of :math:`\alpha_i` and :math:`\xi_t`
accompany the factors, named and coded as in gsynth and fect:

.. list-table::
   :header-rows: 1
   :widths: 16 20 20 44

   * - ``force``
     - unit effects
     - period effects
     - what it means
   * - ``"none"``
     - no
     - no
     - the grand mean and the factors carry everything
   * - ``"unit"``
     - yes
     - no
     - levels differ across units, common shocks left to the factors
   * - ``"time"``
     - no
     - yes
     - a common time path, with unit levels left to the factors
   * - ``"two-way"``
     - yes
     - yes
     - the default, and the specification Xu (2017) Table 2 reports

An effect that is switched off is not estimated and not removed, so it stays
in the residual and the factors absorb what they can of it. The choice
therefore moves the estimate instead of merely relabelling it, and it moves
the estimate most where the pre-treatment fit is worst. Applied work varies
it deliberately: Lang et al. (2026) run their preregistered specification at
``force="none"`` and sweep all four in a multiverse.

Two settings differ in what they can identify, not only in what they fit. A
treated unit with :math:`T_0` pre-periods supports :math:`T_0` regressors in
Step 2, and under ``"unit"`` or ``"two-way"`` one of those is the intercept.
So the largest rank the cross-validation will consider is one lower for those
two than for ``"none"`` and ``"time"``.

``two_way`` was the first release's spelling of this option. It still
resolves — ``True`` to ``"two-way"`` and ``False`` to ``"time"`` — with a
``DeprecationWarning``. Passing both is an error.

Assumptions
-----------

1. Functional form. The untreated potential outcome follows the interactive
   fixed effects model above, with the same factors and the same
   :math:`\boldsymbol{\beta}` for treated and untreated units.

   Remark. This is what replaces parallel trends. The units need not move
   together; they need only respond to the same latent shocks, with
   loadings that may differ arbitrarily across units.

2. Strict exogeneity. The idiosyncratic error is mean-independent of
   treatment assignment, the covariates, the factors and the loadings, for
   all units and periods.

   Remark. Treatment may be assigned on the loadings — states that adopt a
   reform may be systematically more exposed to a national trend — which is
   what a plain two-way fixed effects regression cannot accommodate. What
   is ruled out is assignment on the idiosyncratic shock itself.

3. Weak serial dependence. The errors have finite variance and are weakly
   dependent over time, so the pre-period average of the noise vanishes as
   the pre-period grows.

   Remark. The treated loadings are identified from pre-period time-series
   variation, so a treated unit with a short history yields a noisy loading
   and, through it, a noisy counterfactual. The per-unit
   ``prefit_rmse`` on the result reports how well each unit's pre-period is
   tracked.

4. Regularity and a factor structure that is learnable. The factors are
   non-degenerate, the loadings have a well-behaved second moment, and the
   number of never-treated units and the pre-period lengths both grow.

   Remark. The width of the never-treated pool determines how sharply the
   factors are estimated; the length of each treated unit's pre-period
   determines how sharply its loadings are. A wide pool and short
   pre-periods gives good factors and bad loadings, and the estimator does
   not warn about that on its own.

5. No anticipation and absorbing adoption. A unit's pre-treatment outcomes
   are untreated outcomes, and once treated it stays treated.

   Remark. Both are enforced at ingestion. A panel where treatment reverses
   raises, because Step 2 needs one contiguous pre-treatment block per
   unit; anticipation is not detectable from the data and is the reader's
   to defend.

Choosing the number of factors
------------------------------

The rank is not a nuisance parameter here. The estimate is not monotone in
it, and on the paper's own application it moves the headline number by more
than a quarter over the plausible range, so how it is chosen decides the
answer.

``GSYNTH`` chooses it by the paper's Algorithm 1, a leave-one-period-out
cross-validation that scores a rank on data the treated units actually
supply. For each candidate :math:`r`: fit Step 1 once; then for each
pre-treatment period :math:`s` of each treated unit, drop that period, refit
the loadings on the rest,

.. math::

   \widehat{\boldsymbol{\lambda}}_{i,-s} =
     (\mathbf{F}^{0\top}_{-s}\mathbf{F}^0_{-s})^{-1}
      \mathbf{F}^{0\top}_{-s}
      (\mathbf{Y}^0_{i,-s} - \mathbf{X}^{0}_{i,-s}\widehat{\boldsymbol{\beta}}),

predict the held-out cell and save the error. The rank minimizing the mean
squared prediction error wins. The procedure draws no random numbers, so the
selected rank is a property of the panel and not of a seed.

Two consequences follow from that, and both are checked in the test suite.
The set of scored cells is the same at every rank, so the criterion is
comparable across :math:`r`; and repeated calls return the same rank.

Pass ``r`` to fix the count instead, and ``r_max`` to bound the search. The
estimator lowers ``r_max`` further when the shortest treated pre-period
history cannot identify that many loadings, since a unit with :math:`T_0`
pre-periods supports at most :math:`T_0` regressors and one fewer once a
period is held back.

Inference
---------

The uncertainty comes from the paper's Algorithm 2, a parametric bootstrap
that holds the estimated conditional mean fixed and resamples errors around
it. Treated and control cells draw from different pools, and that asymmetry
is the mechanism.

The factors and the coefficient vector are estimated from the control units,
so the fitted surface tracks a control unit better than it tracks a treated
counterfactual; the treated prediction error is the larger of the two. Loop 1
measures it. Pick a control unit, treat it as if it had adopted on the treated
units' dates, predict it from a resample of the remaining controls, and keep
the difference. Repeating that builds an empirical distribution of prediction
errors for a unit the model did not fit.

Loop 2 then rebuilds panels from the fitted values plus resampled errors —
treated columns drawing from Loop 1's pool, control columns from the
in-sample residuals — refits, and collects the ATT. The simulated treated
columns carry no treatment effect, so the spread of those refits is the
sampling distribution of the estimator under the estimated data-generating
process, and the point estimate is added back when the percentile intervals
are formed. The same machinery produces a band by horizon.

Set ``inference=False`` to skip it; ``n_bootstrap`` controls both loops and
``seed`` makes the result reproducible. Draws whose refit fails are discarded
and counted in ``inference_detail.n_failed``, and the routine raises when too
few survive to estimate a variance from.

Example
-------

.. code-block:: python

   import pandas as pd
   from mlsynth import GSYNTH

   # Election Day Registration and voter turnout, 1920-2012.
   # Nine states adopt EDR across four dates; thirty-eight never do.
   df = pd.read_parquet("basedata/xu_edr_turnout.parquet")

   res = GSYNTH({
       "df": df, "outcome": "turnout", "treat": "policy_edr",
       "unitid": "abb", "time": "year",
       "covariates": ["policy_mail_in", "policy_motor"],
       "inference": True, "n_bootstrap": 2000, "seed": 2139,
       "display_graphs": False,
   }).fit()

   res.att                          # 4.90 percentage points
   res.att_ci                       # bootstrap percentile interval
   res.design.r_selected            # 2, chosen by Algorithm 1
   res.design.cv.mspe               # its criterion by rank
   res.design.beta                  # covariate coefficients: 0.15, -1.05
   res.event_study.tau              # effect by period since adoption
   res.per_unit["MN"].prefit_rmse   # how well one state's pre-period is tracked

Passing ``r`` fixes the count instead, and ``design.cv`` is then ``None``
because no selection was run. Adding ``"force": "none"`` drops the additive
effects and leaves the factors to carry them, which is how a good deal of
applied work runs this estimator; ``design.force`` records the setting used.

``time_series`` carries the treated-average observed and imputed paths in
calendar time, with the gap between them; ``event_study`` carries the same
effect in time since adoption, where the staggered adopters line up. Horizon
:math:`0` is the first treated period. Reference implementations in the
gsynth and fect family number it :math:`1`, so their paths are this one
shifted by one.

GSYNTH is not :doc:`ctsc`. That is Cao, Lu and Wu's generalized synthetic
control, a differently constructed estimator that shares the name.

Verification
------------

GSYNTH reproduces Xu (2017) Table 2 columns (3) and (4) on the author's own
data, and matches a live ``fect`` 2.4.5 reference across every rank from
zero to five on both specifications. All four ``force`` settings match that
reference to 6.4e-14 over 48 fits, on the turnout panel and on a weekly
46-state panel with staggered adoption. See :doc:`replications/gsynth` for the
cell-by-cell comparison and for what the replication turned up about rank
selection, and the durable case
`benchmarks/cases/gsynth_xu_turnout.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/gsynth_xu_turnout.py>`_.

Core API
--------

.. autoclass:: GSYNTH
   :members: fit

.. autoclass:: mlsynth.config_models.GSYNTHConfig
   :members:

References
----------

.. [Xu2017] Xu, Y. (2017). "Generalized Synthetic Control Method: Causal
   Inference with Interactive Fixed Effects Models." *Political Analysis*
   25(1):57-76.

Bai, J. (2009). "Panel Data Models with Interactive Fixed Effects."
*Econometrica* 77(4):1229-1279.

Efron, B. (2012). "Bayesian Inference and the Parametric Bootstrap."
*Annals of Applied Statistics* 6(4):1971-1997.
