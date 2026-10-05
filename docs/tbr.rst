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
its interval to 2.4e-12. On the paper's own simulation design, the 90% and 50%
posterior intervals attain 0.8999 and 0.4981 coverage over 36 cells at 2000
replications each.

The study also records what Section 5.2's squared-bias-over-MSE figure measures.
For any unbiased estimator that statistic has expectation :math:`1/n`, and it
tracks that floor across a sixteenfold range of replication counts, landing on
the published 0.04% at the paper's own 2000. The estimator is consistent with
being unbiased; the figure reports the replication count.

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

.. [LiVandenBulte2022] Li, K. T., & Van den Bulte, C. (2022). *Augmented
   Difference-in-Differences.* Marketing Science 42(4):746-767.

.. [NeweyWest1987] Newey, W. K., & West, K. D. (1987). *A Simple, Positive
   Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance
   Matrix.* Econometrica 55(3):703-708.

.. [VaverKoehler2011] Vaver, J., & Koehler, J. (2011). *Measuring Ad
   Effectiveness Using Geo Experiments.* Google.
