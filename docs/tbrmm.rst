TBRMM: Matched Markets for a Geo Experiment
===========================================

When to use it
--------------

You are planning a geo experiment and have to decide which markets to treat.
Turning the advertising off in a city is not reversible, so the choice is made
before any outcome is observed, from history alone.

TBRMM makes that choice for a :doc:`TBR <tbr>` experiment. It returns one
recommended pair of groups for each treatment-group size you are willing to
consider, so the decision you are left with is how much of the market to
disrupt, not which cities to pick.

Reach for it when the analysis will be TBR and the groups are yours to choose.
When the groups are already fixed, go straight to :doc:`TBR <tbr>`. When the
analysis will be a synthetic control, :doc:`GEOX <geox>` and
:doc:`PANGEO <pangeo>` select for that instead.

Why a design step exists at all
-------------------------------

A synthetic control solves for donor weights, so a mediocre donor pool is partly
repaired by the optimizer: a donor that does not help is given weight zero.

TBR has no weight vector. It sums the response inside each group and regresses
one sum on the other, so a geo's weight is its membership, one inside the group
and zero outside. A market whose series moves independently of the treatment
group enters the control sum at full weight, inflating the residual scale that
every interval depends on. The only way to give it weight zero is to leave it
out of the experiment.

The work a synthetic control does inside a convex program, TBR therefore moves
into the design stage, and the design stage is where the precision is bought.

Notation
--------

Index geos by :math:`i` and periods by :math:`t`. A design is a partition of the
geos into three sets: a treatment group :math:`G_{\mathrm{trt}}`, a control
group :math:`G_{\mathrm{ctl}}`, and the geos held out of the experiment
altogether. The two aggregates are unweighted sums,

.. math::

   y_t = \sum_{i \in G_{\mathrm{trt}}} y_{it},
   \qquad
   x_t = \sum_{i \in G_{\mathrm{ctl}}} y_{it},

and over the :math:`n` pretest periods TBR fits
:math:`y_t = \alpha + \beta x_t + \varepsilon_t`.

The advertiser may restrict where a geo can go. Following Au, each geo carries
an eligibility set :math:`A_i \subseteq \{\text{treatment}, \text{control},
\text{unassigned}\}`. A geo with :math:`A_i = \{\text{treatment}\}` is forced
into every treatment group, and :math:`k_0` counts those, so the smallest design
reported has :math:`\max(k_0, 1)` treated geos.

What the search maximises
-------------------------

Two objectives are available, and they are not the same. ``objective="paper"``
is Au's own, stated in his Section 3.1:

.. math::

   f = \min \big( \text{CUSUM } p,\; \text{Breusch-Godfrey } p,\; R^2 \big),

the CUSUM an OLS-based structural break test on the fitted TBR model, the
Breusch-Godfrey a test for autocorrelation in its residuals. ``objective=
"reference"`` is what ``google/matched_markets`` scores, and it is the default
here, so the benchmark below reproduces that implementation.

The difference is power. Au's Section 3.1 observes that its p-values "do not
assess the amount of statistical power that TBR provides" and then leaves power
out of :math:`f`, so a design that can detect nothing scores as well as one that
can. Measured over 400 random splits of the GeoLift markets, the reference's ten
best designs are 2.5 times better on detectable impact than Au's, and Au's ten
best score 0.63 on :math:`f` against the reference's 0.29 -- each objective wins
on its own axis. The remaining three terms are closer than their names suggest:
:math:`R^2` and the reference's correlation induce identical rankings, since a
simple regression with an intercept has :math:`R^2 = \mathrm{corr}^2`;
Breusch-Godfrey and Durbin-Watson agree on 96.2% of splits, Durbin-Watson being
the AR(1) case; and Au's :math:`R^2` term binds none of the 400 splits.
:mod:`mlsynth.utils.tbrmm_helpers.objective` carries the measurements.

The rest of this section describes the default. The objective is TBR's own
interval, read backwards.

The posterior scale of the cumulative effect over a :math:`T`-period test window
is

.. math::

   T \, s \left( v_\alpha + 2 \bar{x}_T v_{\alpha\beta}
   + v_\beta \bar{x}_T^2 + \tfrac{1}{T} \right)^{1/2},

on :math:`n - 2` degrees of freedom, where :math:`s` is the pretest residual
standard deviation and the :math:`v` terms come from the pretest design matrix.
Every quantity there is fixed by the split, so fixing a significance level and a
power target and solving for the smallest effect that would clear the threshold
gives that split's minimum detectable effect, computable with no experimental
data at all. Its inverse is what the climb maximises.

Two floors bound what any design can buy. Kerman, Wang and Vaver's Section 9.3
has the interval narrowing in the pretest length at :math:`1/\sqrt{n}` while
approaching :math:`\sigma_0 / (\bar{c}\sqrt{T})` instead of zero, and their
Section 9.4 has a longer test window touching only the :math:`1/T` term inside
the root. Neither a longer history nor a longer experiment removes the
uncertainty, so a panel can be short of what the advertiser wants to detect and
no split will fix it.

Ahead of it sit four tests of the assumptions that interval needs, and a split
is ranked on power only after passing them. The comparison is lexicographic:

.. math::

   \big( \text{corr test},\; \text{A/A test},\; \text{CUSUM test},\;
   \text{Durbin-Watson test},\; \mathrm{corr}(y, x),\;
   1/\Delta_{\min} \big).

The gates are described under Diagnostics below. All four are the reference
implementation's; Au names two of them.

Assumptions
-----------

1. The linear relation between the two group aggregates holds through the test
   window.

   Remark. This is TBR's assumption, inherited whole. TBRMM cannot make it true;
   what it can do is refuse splits whose pretest already shows the relation
   failing, which is what the stability and autocorrelation gates are for.

2. The pretest is informative about the test period.

   Remark. The design is chosen on history, so a market whose behaviour changes
   between the pretest and the experiment is outside what any pretest criterion
   can see. A longer pretest helps only if the extra periods resemble the
   experiment.

3. Treating a geo does not change the untreated geos.

   Remark. The control aggregate is the counterfactual, so advertising spillover
   from a treated city into a neighbouring control city contaminates it. TBRMM
   does not test for this; exclude a suspect neighbour by making it ineligible
   for control, or use :doc:`SPILLSYNTH <spillsynth>` where spillover is the
   object of interest.

4. The reported detectable effect is a selected maximum.

   Remark. The search takes a maximum over many candidate splits of a criterion
   estimated from one finite pretest, so the winner's detectable effect is
   optimistically biased, as in any specification search. The gates blunt this,
   being pass-or-fail instead of maximised, and the A/A test evaluates on
   periods the regression did not use, but the bias is not removed. Read the
   number as the best of many estimates.

5. The treated geos need not represent the population you want to speak about.

   Remark. Au raises this as the cost of forgoing randomisation: the treatment
   group is chosen to fit TBR, not to be representative, so the effect estimated
   is the effect on the geos the search picked. Where the experiment is meant to
   stand in for a national rollout, that gap is the design's and not the
   estimator's.

6. A local optimum need not be a runnable experiment.

   Remark. Also Au's: feasibility has to be checked before the experiment is
   commissioned, because the search returns its best local optimum whether or
   not that design is viable. His example is an advertiser requiring New York,
   Chicago and Los Angeles all in the treatment group. Nothing in the gates
   detects a constraint set that admits no workable design.

Diagnostics
-----------

Four gates, each testing something the posterior above needs. Au's Section 3.1
names a CUSUM test and a Breusch-Godfrey test; the correlation floor, the A/A
test and the Durbin-Watson band are the reference implementation's, and the
measurements above place Durbin-Watson and Breusch-Godfrey at 96.2% agreement.

The correlation gate requires :math:`\mathrm{corr}(y, x) \ge 0.8`, so the control
aggregate carries information about the treatment aggregate.

The A/A test runs TBR on the pretest against itself. It holds out the last
:math:`T` pretest periods, refits on what remains, and estimates that window.
Nothing happened there, so the honest answer is zero: the test passes when the
interval covers zero, and when it does not, it passes only if the probability of
a false positive that large stays under 0.2. This is the one gate evaluated on
periods the fit did not see.

The CUSUM test reads the cumulative pretest residuals against a boundary, which
detects a relation that drifts over the pretest instead of holding.

The Durbin-Watson test requires the pretest residuals to sit in
:math:`(1.5, 2.5)`. Autocorrelated residuals leave the posterior scale too
narrow, so the interval would be reported tighter than it is.

The search
----------

Algorithm 1 alternates two routines. Matching holds the treatment group fixed
and toggles the single control membership that most improves the objective,
stopping when no single toggle does. Augmentation holds the control group fixed
and adds the treatment-eligible geo that most improves the objective. Each time
the treatment group reaches a new size, that size's design is recorded.

Two properties to keep in view. The space has :math:`3^n` labellings, about
:math:`5 \times 10^{47}` at a hundred geos, so the result is a local optimum
with respect to one-geo moves and carries no optimality guarantee; a pair of
geos that helps only when swapped together is not reachable. And the objective
is not monotone in the treatment size, so the recommendation is the best design
across sizes and not the largest one.

What a geo contributes is not its volume. Kerman, Wang and Vaver's Section 9.6
shows that multiplying the control aggregate by a constant :math:`\kappa` leaves
the posterior scale unchanged: :math:`v_\alpha` is invariant,
:math:`v_\beta \to v_\beta / \kappa^2`,
:math:`v_{\alpha\beta} \to v_{\alpha\beta} / \kappa` and
:math:`\bar{x}_T \to \kappa \bar{x}_T`, so the three terms cancel, and
:math:`s` does not move because :math:`\beta` absorbs the rescaling. Size on the
control side is therefore free, and what a control geo buys is how its series
covaries with the treatment aggregate. On the treatment side the same section
scales :math:`s` by the factor the aggregate is scaled by. So moving a geo
across changes the objective through the fit, not through where its volume
lands.

Where the control group search starts
-------------------------------------

Matching is a climb over single toggles, so which control group it returns
depends on where it begins. Algorithm 1 begins each size at the control group
the previous size settled on, and ``control_start="carried"`` is that walk. It
is the default, and it is what the reference implementation does, so it is the
setting the benchmark reproduces.

``control_start="pool"`` begins instead from every control-eligible geo the
treatment group leaves free, discarding the previous size's answer. On the
GeoLift panel the same treatment group reaches a different control group from
the two starts in 16 of 19 cases, and the two answers can fall either side of a
rounded correlation, which is the element of the objective above the detectable
impact.

Neither start is better than the other. Over twelve simulated panels at three
treatment sizes each, ``"pool"`` wins six of the thirty six comparisons,
``"carried"`` wins seven, and the remaining twenty three tie. On the GeoLift
markets ``"pool"`` happens to win at all three sizes above one, which is a
property of that panel and not a general one.

``control_start="best"`` runs both and keeps whichever design scores higher at
each size. It costs about two and a half times the default and cannot return a
worse design than it, since a tie returns the carried walk's answer. On the
GeoLift markets it recovers the ``"pool"`` design at every size; across the
simulated panels it is strictly better at six of thirty six and equal at the
rest.

Both alternatives select different markets from the reference, so a design
chosen under either does not reproduce ``google/matched_markets``.

What is not implemented
-----------------------

Au's Section 3.1 also defines the iROAS case: separate objectives :math:`f_r`
and :math:`f_c` for a response metric and a cost metric, combined as
:math:`f = \min(f_r, f_c)` as a maximin strategy, so a design must serve both.
``TBRMMConfig`` takes a single ``outcome``, so there is no cost metric and no
maximin here. A design chosen on revenue alone can be a poor one for spend.

Example
-------

.. code-block:: python

   import pandas as pd
   from mlsynth import TBRMM
   from mlsynth.config_models import TBRMMConfig

   panel = pd.read_csv("basedata/geolift_test_data.csv").rename(
       columns={"location": "geo"})

   result = TBRMM(TBRMMConfig(
       df=panel, unitid="geo", time="date", outcome="Y",
       max_treatment_size=4, n_test=14,
   )).fit()

   for design in result.designs:
       print(design.k, design.treatment_units,
             f"detectable impact {1 / design.objective_value:,.0f}")

   result.recommended.treatment_units

Restricting where geos may go takes three 0/1 columns, constant within geo:

.. code-block:: python

   TBRMMConfig(
       df=panel, unitid="geo", time="date", outcome="Y",
       max_treatment_size=4, n_test=14,
       treatment_eligible_col="can_treat",
       control_eligible_col="can_control",
       unassigned_eligible_col="can_exclude",
   )

A geo whose ``can_treat`` is 1 and whose other two columns are 0 is forced into
every treatment group.

One further setting. ``objective="paper"`` climbs Au's :math:`f` in place of
the reference's tuple, measured against each other above.

.. code-block:: python

   TBRMMConfig(
       df=panel, unitid="geo", time="date", outcome="Y",
       max_treatment_size=4, n_test=14,
       objective="paper",
   )

Reading the design once the experiment has run
-----------------------------------------------

Pass ``post_col`` and the same call returns the effect as well as the design.
The flag already marks the periods the search is not scored on, and those are
exactly the periods there is an effect to read, so no second mode is needed:
without it you get a design, with it you get a design and what it measured.

.. code-block:: python

   panel["post"] = panel["date"] > panel["date"].sort_values().unique()[89]

   result = TBRMM(TBRMMConfig(
       df=panel, unitid="geo", time="date", outcome="Y",
       max_treatment_size=4, n_test=14, post_col="post",
   )).fit()

   result.report.effects.att                      # the pooled effect
   for m in result.recommended.effect.market_effects:
       print(m.unit, m.att, m.att_percent)        # and its market-level parts

The estimator is the augmented difference-in-differences of
Li and Van den Bulte (2022). For a treated geo :math:`i` and the control group's
average :math:`\bar{y}_{co,t}`, their equation (2.4) fits

.. math::

   y_{it} = \delta_1 + \delta_2 \bar{y}_{co,t} + e_{it},
   \qquad t = 1, \dots, T_1

on the pretest periods and projects it through the post window; the effect is
what the geo did above that projection. Forcing :math:`\delta_2 = 1` recovers
plain difference-in-differences, so the free scale is the augmentation, and it
is the same regression TBR fits. The difference is where it is applied: TBR sums
the treated geos into one series first, which buys precision and gives up the
breakdown, while this fits one regression per treated geo and pools by averaging
the per-geo effects, following the paper's Appendix C.

That pooling rule is the reason a market-level table can sit beside a headline
without the two contradicting each other: ``effect.att`` is the mean of
``effect.market_effects``, so the parts add up to the whole by construction. The
price is precision. One geo against the same control average is noisier than the
group is, and ``market_effects[i].rmse_fit`` is where that cost is visible --
compare it against the pooled fit before quoting a single market's number.

Every candidate design is measured, not only the recommendation, so
``designs[i].effect`` is populated for each treatment size. A menu chosen on
pretest detectable impact can therefore be looked at again afterwards, against
what each option would have read.

Two behaviours are deliberate. A ``post_col`` of all zeros marks no realized
periods, so ``report`` stays ``None`` and the run is design-only: an absent
window is not an effect of zero. And a geo's :math:`\hat\delta_2` is reported
per market, because its distance from one is how much work the augmentation did
for that geo.

Uncertainty on a measured effect
--------------------------------

The point estimates above are least squares. Their uncertainty is TBR's, and the
two fit together because under Kerman, Wang and Vaver's flat prior on
:math:`(\alpha, \beta, \log\sigma)` they are the same regression. The
cumulative effect over a :math:`T`-period window has a t posterior on
:math:`n - 2` degrees of freedom with their equation 6's scale,

.. math::

   T s \sqrt{v_a + 2 \bar{x}_T v_{ab} + v_b \bar{x}_T^2 + 1/T},

which is algebraically the prediction-error standard deviation of the same fit,
so one interval serves as a credible interval and a prediction interval alike.
The Bayesian reading buys the direct statement -- the posterior mass on one side
of zero -- and not a different number.

.. code-block:: python

   q = result.recommended.effect.posterior
   q.total_lower, q.total_upper      # on the cumulative effect
   q.att_lower, q.att_upper          # the same bounds, rescaled
   q.prob_direction                  # mass on the estimate's own side of zero

   result.report.inference.ci_lower  # the ATT bounds again, on the contract

   for m in result.recommended.effect.market_effects:
       print(m.unit, m.posterior.total_lower, m.posterior.total_upper)

Set the level with ``level``, which defaults to 0.9, the TBR paper's own
reporting level.

Three things about these intervals decide how they are read.

The cumulative effect is the primitive and the mean effect is a rescaling of it
by a known constant, so the bounds divide and the degrees of freedom do not move.
The constant is the post length times the treated geo count, because
``effect.att`` pools over geos as well as periods. Dividing the cumulative effect
by the post length alone gives the group's per-period effect, which is
:math:`N_{tr}` times larger. Both are defensible numbers; they are not the same
number, and the factor between them is the treated geo count.

The group's uncertainty is its own fit, not the markets' combined. Equation 6's
two terms grow at different rates -- the coefficient uncertainty is shared across
every post period and compounds as :math:`T^2`, while the test window's own noise
accumulates as :math:`T` -- so a cumulative interval cannot be assembled by adding
per-period variances. The same holds across geos: treated markets co-move, their
residuals are correlated, and only a regression on the summed series prices that
correlation. Combining the per-market scales in quadrature treats the markets as
independent and misstates the group's precision.

A market-level interval is wider than the group's, and that is the cost of the
breakdown. Reading one geo against the control average is noisier than reading
the sum, which is why TBR aggregates first. When a single market's interval
spans zero while the group's does not, both statements are correct and the
market one is the honest answer to a question about that market.

What the interval conditions on
-------------------------------

The posterior above treats the design as given. It was not given: the search
chose it, by climbing the same objective the interval is built from, on the same
pretest periods the counterfactual is fitted on. So the design that reaches the
estimation phase is the one whose pretest fit came out best among the candidates,
and :math:`\hat\sigma` is that winner's residual spread. Equation 6 is linear in
:math:`\hat\sigma`, so the interval inherits the shrinkage.

The direction is not in doubt, because a minimum over candidates is below the
average candidate by construction. The size depends on how many partitions were
searched and how alike they are. In a simulation with ten geos, sixty pretest
periods, two treated, forty-five candidate partitions and no effect present,
nominal ninety percent coverage of the cumulative effect falls from 83.6 percent
for a partition fixed in advance to 72.4 percent for the searched one, with
:math:`\hat\sigma` about twenty percent smaller. Searching a larger pool widens
the gap.

Selecting on one window and fitting on another removes it. In the same
simulation, choosing the partition on the first thirty pretest periods and
fitting the counterfactual on the last thirty returns coverage to 82.8 percent,
against 83.6 for the fixed partition, and the interval comes out wider because
fewer periods are left to fit on. TBRMM does not do this for you: ``n_test``
holds periods out inside the A/A gate during the search, but the gate is one of
the filters the search itself applies, so those periods are part of what the
design was chosen on and they are part of what it is fitted on afterwards.

Two things this does not reach. The remaining gap from ninety percent in those
figures is specification: the control average has to track the treated series up
to the noise the model assumes, and no accounting for estimated parameters
repairs it when it does not. And none of it addresses which geos the design
speaks for, which is assumption 5 above.

So read the interval as conditional on the design, and treat its width as a
lower bound on the uncertainty when the same panel chose the design and fitted
the counterfactual. Where the decision turns on the width, hold periods back
from the search by hand and fit on those.

When the residuals are serially correlated
------------------------------------------

Equation 6's second term is :math:`T\sigma^2`, which prices the test window's
errors as independent. Weekly and daily sales are not. When the pretest
residuals carry serial correlation :math:`\rho_j`, the variance of their sum
over :math:`T` periods is

.. math::

   T\sigma^2 \Big( 1 + 2\sum_{j\ge 1} (1 - j/T)\, \rho_j \Big),

so an interval built on the first expression alone is too narrow by the square
root of that bracket, and the shortfall grows with the test window.

Li and Van den Bulte's Proposition 3.4 replaces both of the variance's terms
with Newey-West truncated sums at a bandwidth :math:`\ell`, consistent by the
argument of Newey and West (1987). Set ``variance="hac"`` to use it, and
``hac_bandwidth`` to override the paper's default of
:math:`\lceil T_{\text{pre}}^{1/4}\rceil`.

.. code-block:: python

   result = TBRMM(TBRMMConfig(
       df=panel, unitid="geo", time="date", outcome="Y",
       max_treatment_size=4, n_test=14, post_col="post",
       variance="hac",
   )).fit()

   q = result.recommended.effect.posterior
   q.variance, q.bandwidth          # 'hac', and the truncation lag used

The point estimates do not move. ``variance`` chooses how uncertainty is priced
and nothing else, so ``att``, ``total_effect`` and every market's effect are
identical under either setting.

What it buys, and what it costs. On panels built with a known AR(1) disturbance,
60 pretest and 8 post periods, nominal ninety percent coverage of the cumulative
effect runs:

.. list-table::
   :header-rows: 1

   * - residual :math:`\rho`
     - ``variance="iid"``
     - ``variance="hac"``
   * - 0.0
     - 0.907
     - 0.868
   * - 0.3
     - 0.795
     - 0.830
   * - 0.6
     - 0.632
     - 0.792
   * - 0.8
     - 0.512
     - 0.708

Three things follow. The correction earns its place from about
:math:`\rho = 0.3` upward and the margin widens with the dependence. It costs
about four points when there is no dependence to price, because a long-run
variance estimated from sixty periods is noisier than a single
:math:`\hat\sigma^2`. And it does not restore nominal coverage at strong
dependence -- 0.708 at :math:`\rho = 0.8` is better than 0.512 and is not 0.90 --
so on a panel whose Durbin-Watson and Breusch-Godfrey gates are near their
limits, treat the interval as improved and still optimistic.

The correction is not a widening. It prices whatever dependence is in the
residuals, so negatively correlated residuals give a narrower interval than
equation 6 does, and a design whose residuals are close to independent is barely
affected. A bandwidth of zero leaves only the diagonal, which is the
heteroskedasticity-robust sandwich and not equation 6: the two agree only when
the residual variance is constant across the pretest.

``effect.delta1`` and ``effect.delta2`` carry the group's own fit on the summed
treated series, so a caller comparing designs does not have to refit it.
:math:`\hat\delta_2` is the sum of the implied donor weights, and its distance
from one is how far the design sits from a convex average of the controls.




Verification
------------

Cross-validated against ``google/matched_markets``, the reference implementation
of Au (2018), on the GeoLift panel this repository ships: the same treatment
group and the same control group at every treatment size, the same four gates,
and the inverse detectable impact agreeing to :math:`9.3 \times 10^{-15}`. Of the
39 candidates at the first augmentation step, none is scored differently and both
engines take the same one, so the two agree on the path and not only the answer.

The case is
`benchmarks/cases/tbrmm.py <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/tbrmm.py>`_,
which pins 33 metrics and needs no external checkout. The comparison that
produced it is
`benchmarks/studies/tbrmm_match <https://github.com/jgreathouse9/mlsynth/tree/main/benchmarks/studies/tbrmm_match>`_.

References
----------

Au, T. C. (2018). A Time-Based Regression Matched Markets Approach for Designing
Geo Experiments. Technical report, Google LLC.

Kerman, J., Wang, P. and Vaver, J. (2017). Estimating Ad Effectiveness using Geo
Experiments in a Time-Based Regression Framework. Google.

Li, K. T. and Van den Bulte, C. (2022). Augmented Difference-in-Differences.
*Marketing Science* 42(4):746-767.

Core API
--------

.. autoclass:: mlsynth.TBRMM
   :members:

.. autoclass:: mlsynth.config_models.TBRMMConfig
   :members:
