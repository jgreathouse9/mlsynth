TBR: Time-Based Regression for Geo Experiments
==============================================

When to use it
--------------

You ran a geo experiment: some geographic markets got the advertising change,
others did not, and you want to know what the change did to sales. The obvious
approach is to compare the two groups across markets, which is what geo-based
regression does, and it needs many markets to work, because each market is one
observation. In a small country, a subregion, or a matched-market test with a
single test city and a single control city, there are not enough.

TBR turns the problem on its side. It adds up the response in the treated
markets to get one series through time, adds up the untreated markets to get
another, and learns how the two moved together before the change. That learned
relation predicts what the treated group would have done had nothing happened.
Observations are now time periods, not markets, so a handful of geos is enough.

If the experiment has not run yet, the groups are still yours to choose, and
that choice is where the precision comes from. TBR will make it: give it the
largest treatment group you are willing to disrupt and it searches the splits,
returning one recommended pair of groups per treatment-group size, so the
decision you are left with is how much of the market to disrupt, not which
cities to pick. That search is Au (2018), and it is a mode of this estimator,
not a separate one.

Reach for TBR when the unit count is small and the assignment is an
experimental one, whether the groups are already fixed or still to be chosen.
Reach for :doc:`vanillasc` or :doc:`sdid` instead when you are picking donor
weights from a large pool of untreated units and no experiment was run.
:doc:`cmbsts` produces a similarly shaped answer, a cumulative effect with a
credible interval, from a full state-space model; TBR is one regression on two
series, which is why it survives a panel with five geos in it. When the analysis
will be a synthetic control, :doc:`GEOX <geox>` and :doc:`PANGEO <pangeo>`
select the markets for that instead.

The two modes
-------------

One configuration, one class, two ways to arrive at the two groups.

The named mode takes the split as given. ``treat`` names which units were
treated and when, or ``treatment_col`` and ``control_col`` name the groups
directly; TBR fits the pretest relation and reports the cumulative effect.

The searched mode takes ``max_treatment_size`` and finds the split. Nothing in
the panel is treated, the objective is computed from pretest history alone, and
the result is a menu of designs. Adding ``post_col`` marks periods the search is
not scored on, which are exactly the periods there is an effect to read, so the
same call returns the design and what it measured.

The two are exclusive, and the configuration says so: naming a split and asking
for one to be found at the same time raises. Both return the same type, a
:class:`~mlsynth.config_models.DesignResult` whose ``report`` carries the
estimate and whose ``designs`` carry the menu, which is the shape
:doc:`lexscm` and :doc:`marex` already use for design-then-realise. A named
split has nothing to choose between, so its ``designs`` is empty.

Notation
--------

Index geos by :math:`i` and periods by :math:`t`. The experiment splits the
geos into a treatment group, a control group, and geos assigned to neither. Let

.. math::

   y_t = \sum_{i \in \text{treatment}} m_{i,t},
   \qquad
   x_t = \sum_{i \in \text{control}} m_{i,t}

be the group totals of the response metric :math:`m`. Periods before the
intervention form the pretest window, of length :math:`n`; periods from the
intervention onward form the test window.

Over the pretest window TBR fits

.. math::

   y_t = \alpha + \beta x_t + \epsilon_t,
   \qquad \epsilon_t \sim N(0, \sigma^2),

and for each test period takes the counterfactual to be
:math:`y^{*}_t = \alpha + \beta x_t + \epsilon^{*}_t`. The per-period effect is
:math:`\phi_t = y_t - y^{*}_t` and the estimand is its running total,

.. math::

   \Delta(T) = \sum_{t=1}^{T} \phi_t .

In the searched mode the two groups are not given but are a partition to be
chosen: a treatment group :math:`G_{\mathrm{trt}}`, a control group
:math:`G_{\mathrm{ctl}}`, and the geos held out of the experiment altogether,
with :math:`y_t` and :math:`x_t` the unweighted sums over the first two.

The advertiser may restrict where a geo can go. Following Au, each geo carries
an eligibility set :math:`A_i \subseteq \{\text{treatment}, \text{control},
\text{unassigned}\}`. A geo with :math:`A_i = \{\text{treatment}\}` is forced
into every treatment group, and :math:`k_0` counts those, so the smallest design
reported has :math:`\max(k_0, 1)` treated geos.

Assumptions
-----------

1. The two group totals are linearly related, and that relation is stable
   through the test window.

   Remark. This is the assumption the method rests on, and only half of it is
   checkable. The pretest half is visible in the residuals; the test half is
   not, because the treated group's untreated path stops existing the moment the
   intervention starts. A design that hands TBR two groups whose relation drifts
   will still produce an interval, and the interval will be wrong. Au (2018)
   exists to choose groups for which the relation is plausible, and that is a
   separate step from estimating with them.

2. The pretest residuals are independent and identically distributed normal.

   Remark. Serial correlation in the residuals is the common failure, and it
   makes the interval too narrow because the effective sample size is smaller
   than the period count. A Durbin-Watson or Breusch-Godfrey test on the pretest
   residuals detects it, and both are part of how Au scores a candidate design.
   The fit reports all three parts of this assumption on ``assumptions`` and
   warns when one fires; see Diagnostics.

3. The control geos are unaffected by the intervention.

   Remark. If advertising in the treated markets moves demand in the control
   markets, :math:`x_t` carries part of the effect and the counterfactual
   absorbs it, which shrinks the estimate toward zero. Geographic separation is
   the usual defence. Geos assigned to neither group are the mechanism for
   removing a contaminated market without deleting it from the panel.

4. Aggregation is over a fixed set of geos.

   Remark. A geo that enters or leaves mid-experiment changes the meaning of the
   totals. An absent geo-period is summed as a zero, which is correct when the
   cell is absent because nothing happened and wrong when it is absent because
   the data are missing. The estimator reports how many cells it filled so the
   distinction is visible, and ``assumptions`` adds the two readings of that
   count that matter: whether the fills fall evenly across the treatment
   boundary, since a fill is a zero and an imbalance there is confounded with
   the effect, and whether any geo spans only part of the panel.

The searched mode adds four, all of them about the search and none about the
regression.

5. The pretest is informative about the test period.

   Remark. The design is chosen on history, so a market whose behaviour changes
   between the pretest and the experiment is outside what any pretest criterion
   can see. A longer pretest helps only if the extra periods resemble the
   experiment.

6. The reported detectable effect is a selected maximum.

   Remark. The search takes a maximum over many candidate splits of a criterion
   estimated from one finite pretest, so the winner's detectable effect is
   optimistically biased, as in any specification search. The gates blunt this,
   being pass-or-fail instead of maximised, and the A/A test evaluates on
   periods the regression did not use, but the bias is not removed. Read the
   number as the best of many estimates. What it does to the interval is
   measured below.

7. The treated geos need not represent the population you want to speak about.

   Remark. Au raises this as the cost of forgoing randomisation: the treatment
   group is chosen to fit TBR, not to be representative, so the effect estimated
   is the effect on the geos the search picked. Where the experiment is meant to
   stand in for a national rollout, that gap is the design's and not the
   estimator's.

8. A local optimum need not be a runnable experiment.

   Remark. Also Au's: feasibility has to be checked before the experiment is
   commissioned, because the search returns its best local optimum whether or
   not that design is viable. His example is an advertiser requiring New York,
   Chicago and Los Angeles all in the treatment group. Nothing in the gates
   detects a constraint set that admits no workable design.

Inference
---------

Under a prior uniform on :math:`(\alpha, \beta, \log \sigma)`, the posterior of
:math:`\Delta(T)` is a shifted and scaled t-distribution on :math:`n - 2`
degrees of freedom, with median

.. math::

   \Delta(T) = T \left( \bar{y}_T - \alpha - \bar{x}_T \beta \right)

and scale

.. math::

   T s \left( v_\alpha + 2 \bar{x}_T v_{\alpha\beta}
              + v_\beta \bar{x}_T^2 + 1/T \right)^{1/2},

where :math:`\bar{y}_T` and :math:`\bar{x}_T` are test-window averages,
:math:`s` is the residual standard deviation, and :math:`v_\alpha`,
:math:`v_\beta`, :math:`v_{\alpha\beta}` are entries of the unscaled
:math:`(X'X)^{-1}`. No resampling is involved.

The first three terms inside the root carry uncertainty about the fitted
relation and the :math:`1/T` carries the unobserved errors of the test window
itself, so parameter uncertainty grows as :math:`T^2` while observation noise
grows only as :math:`T`. A longer experiment buys precision about the
per-period effect and spends it on the total.

The scale is reported and not the standard deviation, which does not exist for
four or fewer pretest periods.

Setting :math:`T = 1` gives the effect in a single period, :math:`\phi_t`, whose
scale collapses to

.. math::

   s \left( v_\alpha + 2 x_t v_{\alpha\beta} + v_\beta x_t^2 + 1 \right)^{1/2},

so the per-period posterior is the same formula at one horizon. The estimator
reports it on ``pointwise`` over the whole panel, and the test-window estimates
sum to the final cumulative one. The pretest half is the fitted model's own
residuals: on a well-specified fit they centre on zero and their intervals
cover it at about the nominal rate, which makes the panel a fit diagnostic as
well as a result. A per-period effect drawn without its interval cannot be
read, since one indistinguishable from zero looks like one that is not, so
``plot_tbr`` draws the band.

Return on ad spend
------------------

Supplying a spend column adds a second estimand, the incremental return on ad
spend,

.. math::

   \mathrm{iROAS}(T) = \Delta_{\text{resp}}(T) / \Delta_{\text{cost}}(T),

the cumulative effect on the response over the cumulative effect on cost. The
ratio of two t-distributed quantities has no closed form, so it is simulated
from the two posteriors and summarised by its median.

One case avoids the simulation. When the campaign is new, there is no spend
anywhere in the pretest window and none in the control group, so the cost
counterfactual is zero with certainty and the denominator is a known constant.
The ratio is then a rescaled t, and the estimator reports which route it took
on ``iroas.fixed_cost``.

Cooldown
--------

An intervention's effect does not always stop when the intervention does.
``cooldown_col`` flags the periods after the campaign ends, as a 0/1 indicator
that is a property of the period: constant across geos within a period, and
once it turns 1 it stays 1. The cumulative posterior covers the whole
post-treatment window either way; the flag says where inside it the campaign
stopped, so ``effect_at_intervention_end`` and ``effect_at_cooldown_end`` can be
read against each other. Effects that have stopped accumulating by the end of
the campaign mean the cooldown added uncertainty and nothing else.

Why the groups are a design problem
-----------------------------------

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

What the search maximises
-------------------------

Three objectives are available. The two that come from the literature are not
the same, and the third, ``objective="relative"``, is described at the end of
this section. ``objective="paper"``
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
:mod:`mlsynth.utils.tbr_helpers.design.objective` carries the measurements.

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

Ranking by relative MDE
~~~~~~~~~~~~~~~~~~~~~~~

The minimum detectable effect :math:`\Delta_{\min}` is in the outcome's own
units: the smallest cumulative lift, in sales, that the test would detect. That
is the right yardstick for comparing two control groups for the same treated
markets, and the wrong one for comparing two sets of treated markets. A small
market has small absolute noise, so its :math:`\Delta_{\min}` is small, but a
lift of that size can be a third of what the market sells; a large market has a
larger :math:`\Delta_{\min}` that is a far smaller share of its sales. A
campaign's lift is usually judged as a percentage, so the comparison across
treated groups that matches the question is the relative MDE

.. math::

   \frac{\Delta_{\min}}{T \, \bar{y}},

where :math:`\bar{y}` is the treated aggregate's mean per period over the
scoring window and :math:`T` the test length, so :math:`T \bar{y}` is what the
treated group would be expected to sell over the test. A relative MDE of 0.05
means the test can detect a lift of 5% of the treated group's sales.

``objective="relative"`` ranks designs by it. Its key is the single number
:math:`\bar{y} / \Delta_{\min}`, larger is better, with no gates and no
correlation term ahead of it; each design's ``detail`` reports
``required_impact``, ``level`` (:math:`\bar{y}`), ``relative_mde``, ``corr``
and ``r2``, and only the first two enter the key. For a fixed treated group
:math:`\bar{y}` does not change, so control groups are compared exactly as the
reference's power term compares them. What changes is the comparison across
treated groups, during the search and in the recommended design, which is the
size whose relative MDE is smallest. Each size's relative MDE is a share of its
own group's sales, so that comparison is like for like.

A larger treated group puts more sales in the denominator, so the relative MDE
usually falls as the group grows and the recommended design is usually the
largest size allowed. On the GeoLift panel with ``max_treatment_size=4`` and
``n_test=14`` it falls at every size, from 2.72% at one treated geo to 2.04%,
1.65% and 1.38% at four, and the recommendation is the four-geo design (the
reference objective recommends one geo, at 3.04%). Under ``"relative"``,
``max_treatment_size`` is therefore the setting that decides the test's size
and what it costs to run; the designs at every smaller size remain in
``result.designs`` to choose between.

Two measurements, from scratch studies over a pool of 300 random treated
groups per window, each with the control group of least residual sum of
squares, on the GeoLift panel, a public panel of monthly beer shipments by
state, and simulated panels, motivate it:

* The correlation gate never bound. Every winning design under every ranking
  had a group correlation of at least 0.97, so the reference's 0.8 cap changed
  no pick.
* Ranking by absolute fit favours treated groups with low sales, because the
  residual sum of squares is in sales units. Ranking by relative MDE picked
  groups with 5 times (beer) and about 10 times (simulated) the sales of the
  absolute-fit winner, with intervals 1.6 to 3.2 times narrower as a share of
  sales, and the same calibrated coverage at a nominal 80%: 77% against 76% on
  beer, and 85% and 81% against 83% and 86% on the two simulated designs. On
  GeoLift the two rankings picked similar groups, the relative one's 1.2 times
  the sales of the other's.

The evidence is not one-sided. On GeoLift the relative ranking's calibrated
intervals covered 67% of the time against 80% for the absolute ranking, over 84
overlapping windows. A McNemar test puts that at :math:`p = 0.04` if the
windows are counted as independent, which overlapping windows are not, so the
true uncertainty is larger than that figure. The shortfall is measured on this
panel, and ``"reference"`` remains the default.

The estimand does not change. The estimate is still the cumulative effect on
the treated group the search chose, the average effect of the treatment on
those treated markets; choosing a larger group changes which markets the answer
is about, not what kind of answer it is. It is not an effect on the population
of markets, and the relative MDE is a design yardstick, not a rescaling of the
estimate.

A relative MDE needs a positive level. The fit refuses, and names, every geo
that may be treated whose mean on the scoring window is zero or negative, which
guarantees every candidate group's level is positive; a geo that cannot be
treated is not checked. The objective works with ``search="local"`` and with
``inference="calibrated"``, whose placebo control groups are matched under the
same key.

A split whose treated aggregate is an exact linear function of its control
aggregate over the scoring window leaves no residual, so it has no residual
scale, no required impact and no interval. Every objective scores such a split
as infeasible, below any split with a residual, so the search passes over it
(`#1000 <https://github.com/jgreathouse9/mlsynth/issues/1000>`_). A fit is
exact when its residual is within :math:`10^{-10}` of the series it is computed
from, which covers the rounding of any linear relation. A named split that fits
exactly is still analysed: its estimate is exact and its interval has zero
width, and the fit warns, naming the geos.

Keeping the treated group close to the population
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ranking by relative MDE favours the treatment groups that are easiest to
forecast. Those are often markets whose sales move least with the market-wide
trend: a holiday season, a price change across the category, a swing in the
economy. The estimate is the effect on the markets that were treated, whichever
they are. An advertiser who reads it as the effect the campaign would have had
in every market is reading a stand-in, and when the lift is larger in markets
that respond more to the market-wide trend, the easiest group to forecast is a
stand-in that comes out too small.

``population_tolerance`` bounds how unlike the population the treated group may
be. Its yardstick, the anchor distance, compares the shape of the treated
group's sales path with the shape of the whole panel's. Let

.. math::

   p_t = \sum_{i} m_{i,t}

be the total over every geo in the panel, treated, control or held out, and
:math:`\bar{y}` and :math:`\bar{p}` the means of :math:`y_t` and :math:`p_t`
over the :math:`n` periods the search scores (under ``inference="calibrated"``,
the pre-period minus its blank windows). Dividing each series by its own mean
removes its size, so a group with a tenth of the population's sales and the same
seasonal swings sits at distance zero. The anchor distance is the root mean
squared gap between the two rescaled paths,

.. math::

   d(G_{\mathrm{trt}}) = \Big( \frac{1}{n} \sum_{t=1}^{n}
   \big( y_t / \bar{y} - p_t / \bar{p} \big)^2 \Big)^{1/2}.

It is MAREX's fit of the treated side to the population mean, with every weight
zero or one. Each design reports it as ``population_distance``, whether the
tolerance is on or off, so the menu says how representative each group's path
is.

With ``population_tolerance=C`` a treatment group of :math:`k` geos is
admissible only when

.. math::

   d(G_{\mathrm{trt}}) \le C \, d_{\min}(k),

where :math:`d_{\min}(k)` is the smallest anchor distance among the groups of
:math:`k` geos that the other rules admit (forced and forbidden geos, conflicts,
quotas, budget) and that leave a control group. The group that attains it is the
witness. :math:`C = 1` admits the witness alone, up to exact ties, and a larger
:math:`C` admits every group within that multiple of it. The bound is relative
to what the panel can achieve at each size, not a fixed share of the path: on a
noisy daily panel no group comes within 1% or 1.5% of the population's path, and
on a smooth monthly one many do, so a fixed tolerance is too tight on the first
and too loose on the second.

Finding :math:`d_{\min}(k)` is a search of its own. When the groups of one size
number at most two million, every one is scored, which on the 40 GeoLift geos
takes about 0.02 seconds at four treated geos. Above that, a descent that swaps
one geo at a time runs from a greedy start and 64 random ones and keeps the
closest group it reaches. That can only overstate :math:`d_{\min}(k)`, so the
bound it sets is looser than the exact one, never tighter. On a simulated panel
of 100 geos, the 3.9 million groups of four take 1.4 seconds to enumerate; the
descent takes under 0.1 seconds and lands 1.3% above the exact minimum.
``result.population_tolerance`` records, per size, :math:`d_{\min}(k)`, the
threshold, whether it was ``"enumerated"`` or ``"searched"`` and the witness.

The bound holds at every stage of the search. The greedy adds a geo only when
the group it makes is within the bound at its new size, and when no geo
qualifies it continues from that size's witness, so every size the rules allow
still returns a design (``greedy_from_witness`` marks those sizes). The local
search's moves and restarts stay within it, and calibrated inference draws its
placebo groups from the groups within it, so the placebos carry the same
selection as the real design.

The tolerance admits few groups, often under 1 in 1,000 of the candidates, so
the restarts and the placebos take their treatment groups from it directly.
The quadratic form that finds :math:`d_{\min}(k)` also screens candidates
against the bound, and every group the screen passes is checked against the
definition. Candidates are drawn uniformly in batches of 32,768 and the first
ones within the bound are kept. Up to two million candidates, when 200 draws
per group asked for do not find them all, every candidate is scored instead
and the groups are chosen from the full list, whose size is then exact. Above
two million the draws continue, up to ten million. Either way every group is
a uniform draw from those the tolerance and the business rules admit.

Before this, the restarts and the placebos drew uniformly over every candidate
and kept what the rules admitted, 50 tries a restart and 4,800 draws for 24
placebos. On 200 simulated panels of 80 markets at a tolerance of 1.5, where a
median of 1 in 1,494 groups of four was admitted, the calibration received a
median of 3 placebos of the 24 it asked for, and none in 35 panels. Under no
effect its 80% band covered zero 63% of the time with no placebos and 68% to
77% with one to three, against 83% to 90% with eight or more. On three of those
panels, 9 or 10 of 10 local restarts at four geos were abandoned. Both now
receive every group they ask for whenever that many are admitted. On a
simulated panel of 80 geos at a tolerance of 1.2, listing the 634 admitted
groups of four among 1.6 million takes about 0.6 seconds; at 100 geos the
sampler finds 24 groups admitted at a rate of 1 in 100,000 in about 0.6
seconds.

What the bound buys was measured in scratch studies with 80% calibrated
intervals and a 5% lift injected into the test window:

* Simulated panels of 20 markets, 4 treated, with all 4,845 groups enumerated
  and 100 panels per scenario. When the lift varied with each market's
  sensitivity to the market-wide trend, which the pre-period shows, ranking by
  relative MDE alone picked low-sensitivity groups: a bias of -1.03 points (of
  5) against the population lift, which its intervals covered 55% of the time.
  With the tolerance at 1.5 the bias was -0.18 points, the intervals covered
  82% of both the population lift and the group's own effect, and the RMSE
  fell from 1.56 to 1.28. When the lift was the same everywhere the tolerance
  cost about 10% in RMSE (1.07 against 0.97) and in interval width (2.56
  against 2.32 points).
* GeoLift, 21 overlapping test windows with 4 treated geos. The tolerance at
  1.5 covered 90% against 86% for relative MDE alone, with an RMSE of 1.48
  against 1.61. The pure anchor, the single closest group, picked one group in
  17 of the 21 windows whose counterfactual was badly forecast: it covered its
  own effect 43% of the time with a bias of +3.2 points. Fixed tolerances of 1%
  and 1.5% of the path admitted no group on this noisy daily panel and fell back
  to the pure anchor.
* Monthly beer shipments by state, a public panel, over 62 windows. The
  tolerance at 1.5 never bound: relative MDE's choice was already within 1.06
  times the closest group, so the picks were the same. A fixed 2% tolerance was
  the best rule on beer and the worst on GeoLift, which is why the bound is
  relative to the closest achievable distance.

The real markets were nearly homogeneous in their sensitivity to the trend, so
on them the tolerance mostly guards against a failure mode, and those results
rest on few distinct designs (3 to 10 per rule): they are directional.

The estimand does not change. The estimate is still the cumulative effect on the
treated group the search chose. The tolerance makes that group a closer
stand-in for the population only when effects vary with something the
pre-period shows, such as sensitivity to the market-wide trend. When effects
vary for reasons the pre-period does not show, such as a competitor's campaign
in some markets or a local event during the test, no choice of group made from
the pre-period corrects for it.

The recommended setting is ``population_tolerance=1.5`` with
``objective="relative"``. :math:`C = 1` is the pure anchor, which ignores how
well the group can be forecast, the failure measured on GeoLift above; 1.5
leaves the ranking by relative MDE a choice among groups near the closest one.
The tolerance works with every objective and with ``search="local"``. On the
GeoLift window the relative MDE section uses, with ``max_treatment_size=4`` and
``n_test=14``:

.. list-table::
   :header-rows: 1
   :widths: 6 28 10 10 28 10 10

   * - k
     - Treated, no tolerance
     - Rel. MDE
     - :math:`d`
     - Treated, tolerance 1.5
     - Rel. MDE
     - :math:`d` / :math:`d_{\min}`
   * - 1
     - denver
     - 2.72%
     - 0.070
     - denver
     - 2.72%
     - 0.070 / 0.070
   * - 2
     - denver, cleveland
     - 2.04%
     - 0.090
     - denver, austin
     - 1.77%
     - 0.038 / 0.036
   * - 3
     - denver, cleveland, austin
     - 1.65%
     - 0.053
     - denver, austin, minneapolis
     - 1.45%
     - 0.036 / 0.028
   * - 4
     - denver, cleveland, austin, minneapolis
     - 1.38%
     - 0.030
     - denver, austin, minneapolis, dallas
     - 1.52%
     - 0.030 / 0.023

At two treated geos the unbounded walk adds cleveland, whose path sits 2.5 times
the closest pair's distance from the population, and the bounded walk adds
austin instead. The greedy is a heuristic, so the changed path also reaches a
better relative MDE at two and three geos, and the recommendation moves from
four geos to three. All four sizes were enumerated, and no size needed the
witness.

The distance needs positive levels. With the tolerance on, the fit refuses, and
names, every treatment-eligible geo whose mean on the scoring window is zero or
negative, and refuses a population whose mean is not positive, naming the geos
with non-positive means. A named split has no search to bound, so the option is
refused there.

Diagnostics
-----------

Assumption 1's pretest half and assumptions 2 and 4 are checkable from the
panel the fit was handed, and ``report.assumptions`` carries seven checks.
Assumption 3 is not covered.

.. list-table::
   :header-rows: 1
   :widths: 26 30 44

   * - Check
     - Statistic
     - What firing means
   * - ``backdating``
     - Mean squared standardised error on a held-out tail of the pretest,
       against an :math:`F` reference, with the difference-in-differences
       error on the same window beside it
     - The fitted relation does not predict the pretest it was not shown, so
       the counterfactual it extrapolates may be biased.
   * - ``stationary_residual``
     - Engle-Granger on the treated and control aggregates
     - The gap between the two series is not stationary, so parallel
       pre-trends does not reduce to something the pretest can establish.
   * - ``serial_correlation``
     - Breusch-Godfrey, Durbin-Watson reported beside it
     - The interval is too narrow. Refit with ``variance="hac"``.
   * - ``normality``
     - Shapiro-Wilk
     - The t posterior rests on normal errors, and at these pretest lengths
       that is not an asymptotic argument.
   * - ``homoskedasticity``
     - Breusch-Pagan on the control aggregate
     - Equation 6's single :math:`s` misprices the interval. The point
       estimate is unaffected.
   * - ``balanced_panel``
     - Filled cells, and a Fisher test of their rate either side of the
       boundary
     - Fills are zeros, so an imbalance across the boundary is confounded
       with the effect.
   * - ``stable_membership``
     - Geos spanning only part of the panel
     - The totals change meaning from one period to the next.

The first two bear on the estimate and the next three on the interval around
it, and the warning is grouped that way: at most one warning per consequence,
naming the checks that fired and what they cost. Five of the seven are
hypothesis tests at the 5% level, so between them they fire on about a quarter
of sound panels, and one warning per check would report that quarter as seven
separate alarms.

Why the pretest can say anything about identification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first two checks need an argument the other five do not. Li and Van den
Bulte (2022) separate the identifying assumption into a part that holds in the
pretest, which is testable, and a part that continues into the test window,
which is not, because the treated group's untreated path stops existing after
the treatment lands. Nothing computed from the pretest reaches the second part.

What makes the first part informative is the factor structure. Under the
one-factor model of Li (2024) web appendix A, where each unit's untreated
outcome is :math:`y^0_{jt} = a_j + b_j f_t + u_{jt}`, the correlation between
the treated series and the control average does not depend on :math:`t`. The
assumption then reduces to parallel pre-trends, which the observed data can
test. Definition 1 there states it as :math:`y^0_{\mathrm{tr},t} -
\bar{y}_{U,t} = \alpha_U + v_{U,t}` with :math:`v_{U,t}` stationary, which is
what ``stationary_residual`` tests, through Engle-Granger and not by putting an
augmented Dickey-Fuller on the fitted residuals, whose critical values do not
hold for residuals from an estimated relation.

``backdating`` is Li (2024) section 3.2's exercise on the aggregates. It splits
the pretest, fits on the front, predicts the tail, and divides each held-out
error by its own prediction scale, equation 6 at a horizon of one, so a period
the fit was always going to find hard does not count as evidence against the
model. Under a model that holds, the mean square of those standardised errors
sits near one. TBR nests difference-in-differences at :math:`\beta = 1`, so the
same exercise with the slope forced to one comes free, and its ratio to TBR's
error says whether the slope adjustment paid for itself out of sample. Near one
on a large statistic, neither traces the treated series, and the question is
not which of the two to use.

Neither check looks inside the test window, and neither is a licence for the
estimate.

Size and power
~~~~~~~~~~~~~~

Measured over 200 simulated sound panels per cell, the share firing:

.. list-table::
   :header-rows: 1
   :widths: 34 16 16 16 16

   * - Check
     - 20 periods
     - 30 periods
     - 40 periods
     - 60 periods
   * - ``backdating``
     - 0.060
     - 0.055
     - 0.050
     - 0.060
   * - ``stationary_residual``
     - no verdict
     - 0.105
     - 0.100
     - 0.065
   * - ``serial_correlation``
     - 0.055
     - 0.095
     - 0.075
     - 0.060
   * - ``normality``
     - 0.055
     - 0.025
     - 0.060
     - 0.045
   * - ``homoskedasticity``
     - 0.050
     - 0.045
     - 0.045
     - 0.050
   * - at least one
     - 0.185
     - 0.285
     - 0.285
     - 0.240

The last row is :math:`1 - 0.95^5` and not a defect, and it is why ``flagged``
names the checks instead of reducing to a single verdict. Engle-Granger is
oversized at 30 and 40 periods, which is a small-sample property of its
critical values. Grouped into warnings, 70 to 82% of sound panels raise none at
all and none raises more than two.

Power is uneven, and a check that does not fire is weak evidence only where it
has power to begin with. Against panels that do violate:

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - Violation
     - 20 periods
     - 40 periods
     - 80 periods
   * - AR(1) errors, :math:`\rho = 0.7`
     - 0.580
     - 0.960
     - 1.000
   * - :math:`t_2` errors
     - 0.335
     - 0.455
     - 0.775
   * - spread varying with the control aggregate
     - 0.145
     - 0.290
     - 0.525

So serial correlation is caught reliably from about 40 pretest periods and the
other two are not.

The identification checks are measured against the violation Li (2024)
describes operationally for assumption 2.1, a treated series trending away
from every control with a true effect of zero. At 20, 30 and 40 pretest
periods, ``backdating`` catches 0.930, 0.925 and 0.915 of them, and
``stationary_residual`` 0.000, 0.960 and 0.980. Grouped, the identification
warning fires on 98 to 100% of such panels against 6 to 17% of sound ones.

Where a window is too short for a test to mean anything, or the pretest fits
exactly and leaves no residual to test, the check reports no verdict, which is
distinct from a pass and is left out of ``flagged``. Engle-Granger needs 30
pretest periods, below which it establishes stationarity on no sound panel at
all and a verdict would flag every one. Backdating needs a fit of at least 8
periods and a held-out window of at least 3: one or two periods give an
:math:`F` that is honest about its own size and almost powerless.


Design gates
~~~~~~~~~~~~

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

A local search after the greedy
-------------------------------

Algorithm 1 stops at the first design no single toggle of a control geo
improves. Two moves never happen in it: swapping one control geo for another,
which two toggles reach only by passing through a worse design, and replacing a
treated geo once it has been added. ``search="local"`` adds them. From each
size's greedy design, and from ``local_restarts`` random designs of the same
size, it repeatedly scores every design one move away -- a control geo added,
dropped or swapped, a treated geo swapped for another -- and takes the best,
stopping when none improves the objective. The result at each size is the best
of those descents, so it is never worse than the greedy's design.

Scoring a whole neighbourhood at every step is affordable because every term of
the reference objective is a quadratic form in the group memberships. With
:math:`\mathbf{u}_T` and :math:`\mathbf{u}_C` the indicator vectors of the two
groups and :math:`\mathbf{G}` the Gram matrix of the geo series centred over
the scoring window,

.. math::

   S_{yy} = \mathbf{u}_T^\top \mathbf{G}\, \mathbf{u}_T, \qquad
   S_{xy} = \mathbf{u}_T^\top \mathbf{G}\, \mathbf{u}_C, \qquad
   S_{xx} = \mathbf{u}_C^\top \mathbf{G}\, \mathbf{u}_C,

and the correlation, the slope, the residual scale and the required impact are
functions of these three numbers. The Durbin-Watson gate uses the same forms over
the first-differenced series, the A/A test the same forms over its fit window,
and the Brownian-bridge gate the cumulative residual path. The forms are cached
once per window, so a neighbourhood of a few hundred designs is scored in one
array computation. A design whose forms are within rounding of zero is scored
by the direct fit, and the move the search takes is always rescored by the
direct fit before it is taken, so the search ranks designs by exactly the
objective the greedy uses.

Eligibility binds every move: a geo eligible for treatment alone stays treated,
a geo whose only other role is control stays in the control group, and a geo
leaving the treatment group goes to whichever role it allows.

On the GeoLift markets (the benchmark's window, ``n_test=14``), every design
passes the four gates, and the inverse of the smallest detectable impact rises
over the greedy at every size:

=====  ==================  ==================  ==================
k      greedy              10 restarts         50 restarts
=====  ==================  ==================  ==================
1      0.000999            0.001045 (+5%)      0.001283 (+28%)
2      0.000450            0.000849 (+89%)     0.000849 (+89%)
3      0.000433            0.000463 (+7%)      0.000628 (+45%)
4      0.000369            0.000455 (+24%)     0.000480 (+30%)
=====  ==================  ==================  ==================

The greedy takes 0.2 seconds; the local search takes 2.8 seconds with 10
restarts and 11.7 with 50. The rounded group correlation is 0.99 for every
greedy design and reaches 1.00 at three and four treated geos. On a ten-geo
subset of the same markets, where every design can be scored, the local search
with 50 restarts reaches the best design at all four sizes, and with 10
restarts at one of them; the greedy reaches it at none, and at two to four
treated geos returns a design that fails the Durbin-Watson gate. More restarts
buy more of the design space, and nothing short of scoring every design
certifies the best one: the search is a heuristic.

The local search needs ``objective="reference"`` or ``objective="relative"``,
since the paper's p-values are not quadratic forms in the memberships. Under
``"relative"`` the screen also needs the treated level, the indicator vector
contracted with each geo's mean over the window, and every step is confirmed by
the direct fit under the relative key. The restarts at size :math:`k` draw
from ``local_seed + k``. Each design records ``search`` and, under
``search="local"``, the greedy's key at its size as ``greedy_key``.

Business constraints on the design
----------------------------------

The eligibility columns decide, geo by geo, which of the three groups a geo may
join. Advertisers also state rules about the groups as a whole: two markets that
share a media buy cannot both run the campaign, every sales region needs a
treated market, the test cannot cost more than the budget. The searched mode
takes these rules in the vocabulary SYNDES, MAREX and LEXSCM use, field for
field, so a constraint written for one design estimator reads the same in the
others. Write :math:`\mathcal{T}` for the treatment group and :math:`\mathcal{C}`
for the control group.

* ``to_be_treated`` -- geos forced into every treatment group
  (:math:`i \in \mathcal{T}`). A forced geo is eligible for treatment alone,
  which is how the search already reads a geo that ``treatment_eligible_col``
  alone admits, so forced geos count toward :math:`k_0` and the smallest size
  reported. ``not_to_be_treated`` -- geos kept out of every treatment group (a
  regulated market, one already in another test); they lose the treatment role
  and keep the others, so they stay control candidates.
* ``size_col`` with ``min_size`` / ``max_size`` -- a size band on treated geos.
  A geo outside it loses the treatment role and stays a control candidate.
* ``cluster_col`` and/or ``adjacency`` with ``spillover_threshold`` -- a conflict
  graph. Two geos that share a cluster value, or whose adjacency entry exceeds
  the threshold, are never both treated, because a campaign in one reaches the
  other and the two are then not separate treated units.
* ``stratum_col`` with ``min_per_stratum`` / ``max_per_stratum`` -- coverage
  quotas, :math:`\min \le |\mathcal{T} \cap s| \le \max` for each stratum
  :math:`s`. The minimum binds only in a stratum with a treatment-eligible geo,
  as in SYNDES.
* ``market_cost_col`` with ``max_total_cost`` -- a budget,
  :math:`\sum_{i \in \mathcal{T}} c_i \le B`, where :math:`c_i` is the geo's
  cost of being in the test (the media held back, the fees to run it). This is
  a different column from ``cost_col``, the per-period spend series the iROAS
  estimate divides by.
* ``exclude_bordering_donors`` -- a treated geo's conflict neighbours stay out
  of the control group. A neighbour partly reached by the campaign would carry
  some of the effect into the counterfactual and shrink the estimate. Requires
  ``cluster_col`` or ``adjacency``.
* ``min_controls`` / ``max_controls`` -- a band on :math:`|\mathcal{C}|`.

Every rule binds every step of the search. Augmentation considers only the geos
whose addition leaves the treatment group completable: no conflict inside it, no
stratum over its maximum, the budget not exceeded, and every stratum minimum
still reachable through geos that can still join, in the room
``max_treatment_size`` leaves and within the budget left. Among those it keeps
the geos that let the group be completed at the smallest size, so with two
strata asking for one treated geo each, the second geo comes from the stratum
the first did not cover and size two is reported. A treated geo's banned
neighbours leave the control group in the same step, and matching keeps the
control group inside its band, first bringing a starting group into the band by
the best single adds or drops. A size is reported only when its treatment group
satisfies every rule, so the menu can skip small sizes (three strata with a
minimum of one have no design below three treated geos) and can stop short of
``max_treatment_size`` (a cluster structure can cap how many geos are treatable
together). The local search filters its moves the same way, and a random
restart draws until it finds a design the rules admit, giving up after 50 tries
and counting the restart in the design's ``abandoned_restarts``. Under
``population_tolerance`` the restart's treatment group comes from the
tolerance instead (see that section), and a restart is abandoned only when no
admitted group is found.

The constraints are checked at three points. The configuration refuses
contradictions in the fields themselves: a geo both forced and forbidden, more
forced geos than ``max_treatment_size``, a forced geo outside the size band, a
minimum above its maximum, a stratum whose geos left eligible cannot meet its
minimum, ``exclude_bordering_donors`` with no conflict source, and either half
of the budget without the other. Every constraint field is refused for a named
split, which has nothing to search. Before the climb, an audit refuses a
problem no treatment size admits and names the binding rule: forced geos that
conflict, forced geos that cost more than the budget, quotas that need more
treated geos than ``max_treatment_size`` or more money than the budget, a geo
that has to be a control bordering a forced geo, a control band no pool can
meet. Each audit check is a condition every admissible design meets, so a
refusal is never wrong. The audit cannot catch every infeasible problem: a
conflict graph and a quota together can close every completion in a way no
count detects, and the search raises
:class:`~mlsynth.exceptions.MlsynthDataError` when it records no size. The
search is a heuristic under rules as without them, so it can also raise on a
problem some design satisfies, when the greedy walk has committed to geos that
leave no admissible completion.

The rules in force come back on ``result.design_constraints``: the forced and
forbidden geos, the geos the size band bars, the conflicting pairs by label, the
quota of each stratum, each geo's market cost, the budget, the control band,
and ``rules``, the group-level rules that bound the search. With none of these
fields set the search is the one above, step for step, and
``design_constraints`` is ``None``.

.. code-block:: python

   import pandas as pd
   from mlsynth import TBR
   from mlsynth.config_models import TBRConfig

   panel = pd.read_csv("basedata/geolift_test_data.csv").rename(
       columns={"location": "geo"})
   geos = sorted(panel.geo.unique())
   volume = panel.groupby("geo").Y.mean()

   # Per-geo attributes, constant within geo: a media market shared by four
   # geos, a sales tier, and each geo's cost of being in the test.
   panel["dma"] = panel.geo.map({g: f"dma{i // 4}" for i, g in enumerate(geos)})
   panel["tier"] = panel.geo.map({g: f"tier{i % 3}" for i, g in enumerate(geos)})
   panel["test_cost"] = panel.geo.map(100 * volume / volume.sum())

   result = TBR(TBRConfig(
       df=panel, unitid="geo", time="date", outcome="Y",
       max_treatment_size=4, n_test=14,
       to_be_treated=["chicago"],             # leadership committed to it
       not_to_be_treated=["new york"],        # already in another test
       cluster_col="dma",                     # one treated geo per market
       exclude_bordering_donors=True,         # ...and no control beside it
       stratum_col="tier", max_per_stratum=2,
       market_cost_col="test_cost", max_total_cost=10.0,
       min_controls=10,
   )).fit()

   for design in result.designs:
       print(design.k, design.treatment_units, len(design.control_units))
   result.design_constraints.rules

Every design treats Chicago, none treats New York or two geos from one market,
none holds a treated geo's market neighbour as a control, and each keeps at
least ten controls. The tests are in
``mlsynth/tests/test_tbr_design_constraints.py`` and
``mlsynth/tests/test_tbr_design_rules.py``; the property tests check every
returned design against the rules over random panels and random rule sets, and
check against enumeration that the augmentation's filter never discards a
treatment group some admissible design contains.

What the search does not do
---------------------------

Au's Section 3.1 also defines the iROAS case: separate objectives :math:`f_r`
and :math:`f_c` for a response metric and a cost metric, combined as
:math:`f = \min(f_r, f_c)` as a maximin strategy, so a design must serve both.
The search scores one ``outcome`` and the objective has no second term, so
there is no maximin here. ``cost_col`` buys the iROAS estimate described above;
it does not enter what the climb maximises. A design chosen on revenue alone can
be a poor one for spend.

Reading the design once the experiment has run
----------------------------------------------

Pass ``post_col`` and the same call returns the effect as well as the design.
The flag already marks the periods the search is not scored on, and those are
exactly the periods there is an effect to read, so one call does both:
without ``post_col`` you get a design, with it you get a design and what it
measured.

.. code-block:: python

   panel["post"] = panel["date"] > panel["date"].sort_values().unique()[89]

   result = TBR(TBRConfig(
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
fewer periods are left to fit on. The search does not do this for you:
``n_test`` holds periods out inside the A/A gate during the search, but the gate
is one of the filters the search itself applies, so those periods are part of
what the design was chosen on and they are part of what it is fitted on
afterwards.

Two things this does not reach. The remaining gap from ninety percent in those
figures is specification: the control average has to track the treated series up
to the noise the model assumes, and no accounting for estimated parameters
repairs it when it does not. And none of it addresses which geos the design
speaks for, which is assumption 5 above.

So read the interval as conditional on the design, and treat its width as a
lower bound on the uncertainty when the same panel chose the design and fitted
the counterfactual. Where the decision turns on the width, hold periods back
from the search by hand and fit on those, or set ``inference="calibrated"``,
which holds them back from the search and calibrates the interval on them
(:ref:`tbr-calibrated`).

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

   result = TBR(TBRConfig(
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

.. _tbr-calibrated:

Calibrated inference
--------------------

The posterior interval above is exact for the model it assumes: independent
errors around a regression whose design was fixed in advance. Neither half
holds on most geo panels. Weekly and daily sales are serially correlated, so
errors accumulate over the test window faster than the independent model
allows, and a searched design is the one whose pretest fit came out best, so
its residual spread understates what the experiment will see. Together they
leave the posterior covering 25 to 50 percent at a nominal 80 percent on
autocorrelated panels.

``inference="calibrated"`` measures how wrong the regression actually is and
widens the interval to match. The point estimate does not change: the
counterfactual is the same regression of the treatment-group sum on the
control-group sum, fitted on the whole pre-period, so ``report.cumulative.estimate``
is identical under either setting. Only the lower and upper bounds move.

.. code-block:: python

   result = TBR(TBRConfig(
       df=panel, unitid="geo", time="date", outcome="Y",
       max_treatment_size=4, n_test=14, post_col="post",
       inference="calibrated",
   )).fit()

   c = result.calibration
   result.report.cumulative.lower[-1], result.report.cumulative.upper[-1]
   c.posterior_lower[-1], c.posterior_upper[-1]     # the posterior band, kept
   c.k                                              # the miscalibration factor
   result.recommended.calibrated_required_impact     # the calibrated MDE

The default stays ``inference="posterior"``, which is the interval the
replication against ``google/matched_markets`` pins, bit for bit.

The honest split
~~~~~~~~~~~~~~~~

The calibration needs windows on which the regression's error can be measured
without the measurement being part of what chose the design. Those are the last
:math:`B` pre-periods, ``blank_periods``, three test lengths by default. In
searched mode the design search (the greedy and the local search, under any
business constraints) runs on the pre-period minus those :math:`B` periods, so
no design choice saw them; the fit that reports on the chosen design still uses
the whole pre-period. ``n_periods_scored`` on the result is the shorter window.

A named split is the caller's choice and nothing can be held back from it. The
calibration runs the same way, ``calibration.honest_split`` is ``False`` and
the fit warns: the blank windows are honest only if the groups were chosen
without them.

:math:`B` is refused below two test lengths, so the longest horizon still has
blank origins to pool over, and below the length that leaves three periods to
fit from.

How the band is built
~~~~~~~~~~~~~~~~~~~~~

Index the pre-period :math:`0, \dots, E-1`, with :math:`E` the first test
period, and let :math:`D_0 = E - B`. For a horizon :math:`h` and an origin
:math:`s`, the regression fitted by least squares on :math:`[0, s)` predicts the
sum of :math:`y` over :math:`[s, s+h)`. With
:math:`a = (h, \sum_{t=s}^{s+h-1} x_t)'` the prediction and its scale are

.. math::

   c_s = a'\hat\beta, \qquad
   \mathrm{sc}_s = \sqrt{S\,\big(h + a'(X'X)^{-1}a\big)}, \qquad
   S = \frac{\mathrm{RSS}}{s - 2},

on :math:`s - 2` degrees of freedom. At :math:`s = E` this is the posterior
above, term for term, so the calibration starts from the interval TBR already
reports.

1. Blank errors. For the real design and every placebo design, and every origin
   :math:`s \in [D_0, E - h]`, the standardised error is
   :math:`z_s = (\sum_{t=s}^{s+h-1} y_t - c_s)/\mathrm{sc}_s`. Under the model
   these are t-distributed with scale one; on a real panel their spread is the
   miscalibration.
2. Distance trend. The errors change with the distance :math:`d = s - D_0`
   from the period the design was chosen on, and the experiment sits furthest
   away, at :math:`d = B`. Two forces pull in opposite directions. A gap that
   drifts makes the errors grow with the distance. Selection makes the errors
   nearest the design window too large: the regression at those origins is
   fitted almost entirely on periods the design was chosen to fit, so its scale
   is too small, and the share of such periods in the fit falls as the origin
   moves away. The pooled squared errors are regressed on the distance,
   :math:`z^2 = \alpha + \beta d`, the slope is clamped at zero, and each error
   is carried to the experiment's distance:
   :math:`\tilde z = z \sqrt{(\alpha + \beta B) / \max(\alpha + \beta d, 10^{-6})}`.
   The clamp lets the trend widen the band and never narrow it. On the panels
   the calibration was chosen on (`#1043
   <https://github.com/jgreathouse9/mlsynth/issues/1043>`_) the trend falls on
   the independent and autocorrelated panels and rises on the drifting, hotel
   and beer panels. A slope left free to fall takes coverage at 80 percent
   nominal down to 55 to 77 percent on those panels, against 75 to 85 percent
   with the clamp: the experiment's error stops falling where the line keeps
   going, and the real design, chosen harder than the placebos, has a larger
   error at the experiment than they do.
3. Dirichlet-process predictive (Florens and Rolin 2024, eq. 5.13; Blackwell
   and MacQueen 1973). A Dirichlet process with base measure the posterior and
   concentration :math:`n_0` (``calibration_prior_weight``, default 2), updated
   with the blank errors, has the predictive

   .. math::

      F(v) = w_0\, F_{t}\!\left(\frac{v - c}{\mathrm{sc}}\right)
             + (1 - w_0)\, \frac{1}{m} \sum_{i=1}^{m}
               \mathbf{1}\{c + \mathrm{sc}\,\tilde z_i \le v\},
      \qquad w_0 = \frac{n_0}{n_0 + n_\text{eff}},

   for the counterfactual sum, with :math:`c` and :math:`\mathrm{sc}` the real
   design's prediction and scale at :math:`s = E`. Overlapping windows are not
   independent, so the empirical part counts as
   :math:`n_\text{eff} = (\text{designs}) \times \max(1, \lfloor B/h \rfloor)`
   windows, not as :math:`m` points.
4. The band. The counterfactual interval is
   :math:`[F^{-1}((1-\ell)/2),\, F^{-1}((1+\ell)/2)]` at level :math:`\ell`,
   found by bisection to ``QUANTILE_TOL`` (:math:`10^{-9}`) posterior scales,
   and the effect interval is the observed sum minus its endpoints. It is
   computed at every horizon, so ``report.cumulative`` is a band along the
   path.

As :math:`n_0` grows the band returns to the posterior; at :math:`n_0 = 0` it is
the blank errors alone.

Placebo designs carry the selection effect into the blank errors. Each is
another treatment group of the real group's size, drawn uniformly with
``calibration_seed`` from the groups the eligibility and the business
constraints admit, holding every forced geo and excluding the real group, with
its control group chosen by the search's own matching climb on the design
window :math:`[0, D_0)`. ``n_placebos`` (default 24, at least 8) are drawn; when
fewer groups are admissible all are used, the count is recorded on
``calibration.n_placebos``, and below eight the fit warns. Up to 20,000
candidate groups every one is checked, and ``calibration.n_admissible`` records
how many are admissible. Above that the groups are drawn uniformly, 200 draws
per placebo, unless ``population_tolerance`` is set: the tolerance then lists
or samples the groups within it and the business constraints filter them.

The calibrated MDE
~~~~~~~~~~~~~~~~~~

The same blank errors say how much the reference's required impact understates
the smallest detectable effect. With :math:`k_H` the root mean square of
:math:`\tilde z` at the planned test length, every design's
``calibrated_required_impact`` is :math:`k_H` times its ``required_impact``, and
the power table carries the column. :math:`k_H` is one number per panel, so the
ranking of designs, and anything the search proves about it, does not change.
``calibration.k`` holds the factor at every horizon.

What it measured
~~~~~~~~~~~~~~~~

Coverage at 80 percent nominal of the 14-period cumulative effect, from the
calibration study behind issue #1043. Each simulated panel has a known truth;
the two real panels are public.

.. list-table::
   :header-rows: 1

   * - panel
     - ``posterior``
     - ``calibrated``
   * - iid errors
     - 78%
     - 83%
   * - AR(1), :math:`\rho = 0.5`
     - 50%
     - 85%
   * - AR(1), :math:`\rho = 0.8`
     - 25%
     - 81%
   * - drifting gap
     - 30%
     - 76%
   * - AR(1) :math:`\rho = 0.8` and drift
     - 29%
     - 80%
   * - weekly hotel prices
     - 25%
     - 75%
   * - monthly beer shipments
     - 38%
     - 81%

The test suite reproduces the direction on forty seeded AR(1) panels with
:math:`\rho = 0.8` and a named split: 35 percent for the posterior, 90 percent
calibrated, with eight placebo designs and a 24-period blank window.

The calibration buys nominal coverage with width. On the simulated panels the
calibrated band runs 1.1 to 1.5 times the width of the interval from the true
model, and on
the GeoLift test panel with a three-market design the 80 percent interval for
the 15-day effect of +3,097 widens from :math:`[+2{,}120, +4{,}073]` to about
:math:`[-70, +5{,}160]`.

Its limits. A gap that drifts is the hardest case: the regression has no term
for it, the distance trend extrapolates a straight line, and coverage on the
drift-only panel stops at 76 percent. The weekly hotel panel runs five points
short for the same reason. The blank errors also describe the recent past, so a
change in regime between the blank window and the experiment is outside what
they can see. And a named split calibrated on its own pre-period is honest only
to the extent that the groups were chosen without the blank window.
``inference="calibrated"`` and ``variance="hac"`` are refused together: both
answer serial correlation, and the calibration standardises by the
independent-error scale. The iROAS interval stays the posterior's.

Scoring a named split before the experiment runs
------------------------------------------------

The named mode does not need a treated panel. Give it ``treatment_col`` and
``control_col`` with ``post_col`` and no ``treat`` at all, and it runs on a panel
in which nothing happened. This is what scoring a hypothetical split by hand
requires, and what an A/A test requires: take the control geos of a finished
experiment, split them in two, and TBR should report no effect. Au's Example 3
measures exactly that, and it is how a design's error rate is checked before an
experiment is commissioned. :doc:`lexscm` and :doc:`syndes` separate the two
phases the same way.

The arithmetic is identical once the window and the groups are named, so the
number this returns for a hand-built split is the number the searched mode would
return for the same split.

Example
-------

A finished experiment, groups fixed by the assignment:

.. code-block:: python

   import pandas as pd
   from mlsynth import TBR
   from mlsynth.config_models import TBRConfig

   # long panel: one row per geo per day
   res = TBR(TBRConfig(
       df=panel,
       unitid="geo", time="date", outcome="sales", treat="D",
       control_col="is_control",      # which untreated geos are controls
       cost_col="cost",               # optional: adds iROAS
       level=0.9,
       display_graphs=False,
   )).fit()

   est = res.report
   est.cumulative.estimate[-1]        # cumulative effect at the end
   est.cumulative.lower[-1], est.cumulative.upper[-1]
   est.iroas.estimate                 # return on ad spend
   est.tbr_fit.beta                   # the fitted pretest relation
   est.filled_cells                   # absent geo-days summed as zero

The three-panel figure of the paper's Section 3.3, the two series, the per-period
difference, and the cumulative band, comes from the estimator's own plotter,
which returns the figure for the caller to display or save:

.. code-block:: python

   from mlsynth.utils.tbr_helpers.plotter import plot_tbr
   fig = plot_tbr(res)

An experiment still to be designed. ``max_treatment_size`` is the only
difference, and there is no ``treat`` column because nothing has been treated:

.. code-block:: python

   panel = pd.read_csv("basedata/geolift_test_data.csv").rename(
       columns={"location": "geo"})

   result = TBR(TBRConfig(
       df=panel, unitid="geo", time="date", outcome="Y",
       max_treatment_size=4, n_test=14,
   )).fit()

   for design in result.designs:
       print(design.k, design.treatment_units,
             f"detectable impact {1 / design.objective_value:,.0f}")

   result.recommended.treatment_units

Restricting where geos may go takes three 0/1 columns, constant within geo:

.. code-block:: python

   TBRConfig(
       df=panel, unitid="geo", time="date", outcome="Y",
       max_treatment_size=4, n_test=14,
       treatment_eligible_col="can_treat",
       control_eligible_col="can_control",
       unassigned_eligible_col="can_exclude",
   )

A geo whose ``can_treat`` is 1 and whose other two columns are 0 is forced into
every treatment group.

``objective="paper"`` climbs Au's :math:`f` in place of the reference's tuple,
measured against each other above.

.. code-block:: python

   TBRConfig(
       df=panel, unitid="geo", time="date", outcome="Y",
       max_treatment_size=4, n_test=14,
       objective="paper",
   )

``objective="relative"`` ranks designs by relative MDE, and each design reports
it:

.. code-block:: python

   result = TBR(TBRConfig(
       df=panel, unitid="geo", time="date", outcome="Y",
       max_treatment_size=4, n_test=14,
       objective="relative",
   )).fit()

   for design in result.designs:
       print(design.k, design.treatment_units,
             f"relative MDE {design.detail['relative_mde']:.1%}")

``population_tolerance`` keeps each treated group's path within a multiple of
the closest achievable distance from the population's, and the result records
the bound per size:

.. code-block:: python

   result = TBR(TBRConfig(
       df=panel, unitid="geo", time="date", outcome="Y",
       max_treatment_size=4, n_test=14,
       objective="relative", population_tolerance=1.5,
   )).fit()

   for design in result.designs:
       print(design.k, design.treatment_units,
             f"distance {design.population_distance:.3f}")
   for anchor in result.population_tolerance.anchors:
       print(anchor.k, anchor.d_min, anchor.threshold, anchor.method,
             anchor.witness_units)

Verification
------------

Both modes are cross-validated against Google's reference implementation,
`google/matched_markets <https://github.com/google/matched_markets>`_.

The estimate is checked on the reference's own panel and against the paper's
Section 5.2 simulation. The study is `benchmarks/studies/tbr_geo
<https://github.com/jgreathouse9/mlsynth/tree/main/benchmarks/studies/tbr_geo>`_,
and the agreement is pinned durably by `benchmarks/cases/tbr.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/tbr.py>`_,
which runs the estimator against reference values on two panels: the GeoLift
markets this repository ships, with the groups named by hand on an untreated
panel, and a generated panel matched to the reference's shape for the cost,
cooldown and absent-cell paths a real untreated panel cannot reach.

Every reported quantity agrees with the reference: the pretest coefficients and
residual variance to the digit, the cumulative response effect to 6.4e-10, the
cumulative cost effect to 7.3e-12, and the iROAS point estimate to 7.1e-15 with
its interval to 2.4e-12.

Agreement is not calibration, and the two are checked separately. On the paper's
own simulation design the study's port attains 0.8996 and 0.5024 coverage of the
90% and 50% posterior intervals over 36 cells at 2000 replications each. The
estimator itself is measured by `benchmarks/cases/tbr_montecarlo.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/tbr_montecarlo.py>`_,
which runs the same grid through ``TBR.fit`` under two geo-assignment schemes:
the free permutation the study draws, and the volume-matched pairing the
authors' own R package uses. Both attain nominal coverage, and per cell the
paper's own criterion is used -- the posterior of the rate under a neutral
prior, :math:`\mathrm{Beta}(1/3 + y,\, 1/3 + n - y)`, has to contain the
nominal rate.

Both of those measure calibration where the method's own condition holds. What
the condition buys, and what its absence costs, is measured separately by
`benchmarks/cases/tbr_factor_dimension.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/tbr_factor_dimension.py>`_.

Assumption 1 is stated under a one-factor model, and one factor is the
condition and not a simplification. Aggregating gives each group the mean
loading of its members, and the fitted relation
:math:`\bar{y}_{\mathrm{tr},t} = \alpha + \beta \bar{y}_{\mathrm{co},t}` holds
at every period exactly when those two mean loading vectors are proportional.
At one factor they are scalars, so proportionality is automatic. Past one factor
two independently drawn mean vectors are not proportional, and a single
regressor cannot absorb the difference, so a gap remains with no treatment
anywhere.

The case measures that on panels carrying no treatment, where the cumulative
effect is zero and coverage is the share of intervals containing zero. With the
noise switched off, the best affine fit leaves a relative gap of order
:math:`10^{-16}` at one factor and at least :math:`10^{-3}` past it, twelve
orders apart, and the sine of the angle between the two mean loading vectors is
exactly zero at one factor. At a nominal 0.90 the one-factor cell attains 0.92;
the two-, three- and five-factor cells come back at 0.55, 0.51 and 0.48, with
the interval widening as the factor count rises without the coverage following.

Averaging more geos does not restore it. Treated groups of 2, 5 and 25 of 50
geos all fail past one factor, so this is a property of the regression having
one control series and not of a small treated group. What it means in practice
is that the pretest checks on ``report.assumptions`` are the ones to read before
the interval: the backdating check and the Engle-Granger check are what detect
a treated series the control aggregate cannot trace.

The study also records what Section 5.2's squared-bias-over-MSE figure measures.
For any unbiased estimator that statistic has expectation :math:`1/n`, and it
tracks that floor across a sixteenfold range of replication counts, reaching
0.0391% over the full grid at the paper's own 2000 replications against the
published 0.04%. The estimator is consistent with being unbiased; the figure
reports the replication count, so the case asserts the iROAS median at 2.0 and
carries the ratio as a diagnostic.

The search is checked on the GeoLift panel this repository ships: the same
treatment group and the same control group at every treatment size, the same
four gates, and the inverse detectable impact agreeing to
:math:`9.3 \times 10^{-15}`. Of the 39 candidates at the first augmentation
step, none is scored differently and both engines take the same one, so the two
agree on the path and not only the answer. The case is `benchmarks/cases/tbrmm.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/tbrmm.py>`_,
which pins 33 metrics and needs no external checkout. The comparison that
produced it is `benchmarks/studies/tbrmm_match
<https://github.com/jgreathouse9/mlsynth/tree/main/benchmarks/studies/tbrmm_match>`_.

Core API
--------

.. autoclass:: mlsynth.TBR
   :members:

.. autoclass:: mlsynth.config_models.TBRConfig
   :members:
   :undoc-members:

.. autoclass:: mlsynth.utils.tbr_helpers.design.structures.TBRResults
   :members:
   :undoc-members:

.. autoclass:: mlsynth.utils.tbr_helpers.structures.TBREstimate
   :members:
   :undoc-members:

.. autoclass:: mlsynth.utils.tbr_helpers.structures.TBRFit
   :members:
   :undoc-members:

.. autoclass:: mlsynth.utils.tbr_helpers.structures.CumulativeEffect
   :members:
   :undoc-members:

.. autoclass:: mlsynth.utils.tbr_helpers.structures.IROASResult
   :members:
   :undoc-members:

.. autoclass:: mlsynth.utils.tbr_helpers.design.structures.TBRMMDesign
   :members:
   :undoc-members:

.. autoclass:: mlsynth.utils.tbr_helpers.design.structures.TBRMMEffect
   :members:
   :undoc-members:

.. autoclass:: mlsynth.utils.tbr_helpers.design.structures.TBRMMMarketEffect
   :members:
   :undoc-members:

.. autoclass:: mlsynth.utils.tbr_helpers.design.structures.TBRMMPosterior
   :members:
   :undoc-members:

References
----------

.. [Kerman2017] Kerman, J., Wang, P., & Vaver, J. (2017). *Estimating Ad
   Effectiveness using Geo Experiments in a Time-Based Regression Framework.*
   Google.

.. [Au2018] Au, T. C. (2018). *A Time-Based Regression Matched Markets Approach
   for Designing Geo Experiments.* Google LLC.

.. [BlackwellMacQueen1973] Blackwell, D., & MacQueen, J. B. (1973).
   *Ferguson Distributions via Polya Urn Schemes.* The Annals of Statistics
   1(2):353-355.

.. [FlorensRolin2024] Florens, J.-P., & Rolin, J.-M. (2024). *Bayes, Bootstrap,
   and Moments.* In J.-P. Florens & M. Mouchart (Eds.), *Nonparametric
   Bayesian Inference: Contributions by Jean-Marie Rolin.* Springer.

.. [LiVandenBulte2022] Li, K. T., & Van den Bulte, C. (2022). *Augmented
   Difference-in-Differences.* Marketing Science 42(4):746-767.

.. [NeweyWest1987] Newey, W. K., & West, K. D. (1987). *A Simple, Positive
   Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance
   Matrix.* Econometrica 55(3):703-708.

.. [VaverKoehler2011] Vaver, J., & Koehler, J. (2011). *Measuring Ad
   Effectiveness Using Geo Experiments.* Google.
