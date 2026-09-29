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

Not a similarity score and not the pretest :math:`R^2`. The objective is TBR's
own interval, read backwards.

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

Ahead of it sit four tests of the assumptions that interval needs, and a split
is ranked on power only after passing them. The comparison is lexicographic:

.. math::

   \big( \text{corr test},\; \text{A/A test},\; \text{CUSUM test},\;
   \text{Durbin-Watson test},\; \mathrm{corr}(y, x),\;
   1/\Delta_{\min} \big).

The gates are described under Diagnostics below.

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

Diagnostics
-----------

Four gates, each testing something the posterior above needs.

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
is not monotone in the treatment size, because a geo added to the treatment
group brings its volume to :math:`y_t` and takes it out of the pool available to
:math:`x_t`. The recommendation is therefore the best design across sizes and
not the largest one.

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

Au, T. C. (2018). Robust Design and Analysis of Geo Experiments with Matched
Markets. Google.

Kerman, J., Wang, P. and Vaver, J. (2017). Estimating Ad Effectiveness using Geo
Experiments in a Time-Based Regression Framework. Google.

Core API
--------

.. autoclass:: mlsynth.TBRMM
   :members:

.. autoclass:: mlsynth.config_models.TBRMMConfig
   :members:
