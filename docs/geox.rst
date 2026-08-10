Geo Experiment Design (GEOX)
============================

.. currentmodule:: mlsynth

When to Use This Estimator
--------------------------

GEOX decides which geographic markets to treat in a marketing geo
experiment, before the experiment runs. You have daily or weekly sales
for a few dozen cities, you can turn a campaign on in some of them, and
you must pick which ones. GEOX answers that by rehearsing the
experiment on history you already have: it slides a pretend treatment
window backwards through the past, injects a lift of known size, and
counts how often the estimator would have caught it. The region with the
smallest lift it can reliably detect is the one to test in.

Use it when:

- you can assign treatment at the geo level, and geos are the unit of
  analysis;
- you have pre-period history for every candidate market, with no gaps;
- the number of markets is in the tens, as with DMAs or metros;
- you want the design chosen by the same estimator that will analyse the
  result, so the power calculation is not about a different model than
  the one you will report.

Use something else when the design question is different in kind.
:doc:`pangeo` pairs geos into balanced supergeos and randomises within
pairs, which suits a rollout where every market must be used. GEOX
picks a small test region and leaves the rest as donors. :doc:`spcd`
scores a design from holdout residuals when the estimator is a fixed
linear contrast.

What Distinguishes It
---------------------

The market-selection loop follows GeoLift, Meta's geo-experiment
package: anchor a candidate region at each market, add that market's
most correlated neighbours, sweep effect sizes over several backward
backtests, and rank by a composite score. That loop is the estimator.
The thing that scores a candidate inside it is a choice.

``engine`` makes the choice. Everything else in the design --
nomination, the backtest windows, effect injection, power, the minimum
detectable effect, the composite rank, the constraint layer, the plots
-- is shared code and does not know which engine ran.

``engine="augsynth"`` is the ridge-augmented synthetic control of
Ben-Michael, Feller and Rothstein (2021), which is what GeoLift itself
scores with. Pick it to reproduce GeoLift, or when the analysis that
will be reported is a synthetic control fit.

``engine="sdid"``, the default, is synthetic difference-in-differences
(Arkhangelsky, Athey, Hirshberg, Imbens and Wager, 2021), which weights
donor markets and pre-periods at the same time. Two consequences
follow.

The first is that no single pre-period has to be matched. Synthetic
control asks the donors to reproduce the treated path everywhere before
treatment; synthetic DiD asks only that a weighted average of
pre-periods line up, and lets the weights decide which periods carry the
comparison. On a 105-day panel of 40 US metros, the fitted design puts
weight on 7 of 91 pre-days and none on the rest. Days that resemble the
post-period window count; days that do not are set aside.

The second is that a level difference between the treated region and its
donors is absorbed. Synthetic DiD differences it out, so the donor pool
does not have to contain the treated region's scale, only its shape.
A two-city test region in a pool of small markets remains estimable.

The two engines report different imbalance measures, because each
reports its own estimator's. ``pre_rmspe`` on the SDID path is the
root-mean-square pre-period gap; ``scaled_l2`` on the augsynth path is
augsynth's ratio of the fitted imbalance to the imbalance uniform donor
weights would leave. Compare either across candidates within one design;
neither is comparable across engines.

The name. This page was ``sdidgeo`` while synthetic DiD was the only
thing that could score a design, and the estimator, its config, its
result class and its plotter were ``SDIDGEO``, ``SDIDGEOConfig``,
``SDIDGEOResults`` and ``plot_sdidgeo_design``. Those spellings are gone,
not deprecated. Code written against them raises ``ImportError`` at the
import line; substitute ``GEOX`` for ``SDIDGEO`` throughout, which is the
whole of the change.

Notation
--------

Let :math:`i = 1, \dots, N` index markets and :math:`t = 1, \dots, T`
index periods, with outcome :math:`Y_{it}`. A candidate test region
:math:`\mathcal{T}` is a set of :math:`N_{\mathrm{tr}}` markets; the
donors are the rest, :math:`\mathcal{C}`. Write
:math:`y_t = N_{\mathrm{tr}}^{-1}\sum_{i \in \mathcal{T}} Y_{it}` for the
treated average and :math:`\mathbf{Y}_{0}` for the
:math:`T \times |\mathcal{C}|` donor matrix.

A pseudo-experiment is a pre/post split of the observed history. For a
duration :math:`D` and a backtest :math:`s = 1, \dots, S`, the pretend
treatment runs over periods

.. math::

   [\,T_0,\; T_0 + D - 1\,], \qquad T_0 = T - D - s + 1 ,

so :math:`s = 1` ends flush with the last observed period and each
increment slides the window one period earlier. Everything before
:math:`T_0` is the pre-period.

What the engine supplies is a counterfactual path for :math:`y_t` fitted
on the pre-period alone, and :math:`\hat\tau` is the average gap against
it over the window. The default engine is synthetic DiD, which chooses
unit weights :math:`\omega` over donors and time weights
:math:`\lambda` over pre-periods, and estimates

.. math::

   \hat\tau \;=\;
   \Big( \bar y_{\mathrm{post}} - \lambda' y_{\mathrm{pre}} \Big)
   \;-\;
   \Big( \omega' \bar{\mathbf{Y}}_{0,\mathrm{post}}
         - \lambda' \mathbf{Y}_{0,\mathrm{pre}}\, \omega \Big).

Both weight vectors are non-negative and sum to one. Rearranged, the
counterfactual path is
:math:`\mathbf{Y}_{0}\omega + \big(\lambda' y_{\mathrm{pre}} - \lambda'
\mathbf{Y}_{0,\mathrm{pre}}\omega\big)`.

The augsynth engine has no :math:`\lambda`. It weights donors alone, and
every pre-period counts equally: simplex synthetic-control weights first,
then a ridge augmentation that corrects the pre-period imbalance those
weights leave, which is what takes :math:`\omega` off the simplex. With
``fixed_effects`` (its default, augsynth's ``fixedeff``) every unit is
demeaned by its own pre-period mean before fitting and the level comes
back as an intercept, so its counterfactual path is
:math:`\alpha + \mathbf{Y}_{0}\omega`. Everything downstream of the path
is the same for both engines.

An effect of size :math:`e` is injected multiplicatively,
:math:`y_t \mapsto (1 + e)\, y_t` on the window, matching GeoLift's
convention that an effect size reads as a percentage lift on the treated
markets' own volume.

Assumptions
-----------

1. Balanced panel, no gaps.

   Every market is observed in every period. Ingestion goes through
   ``geoex_dataprep``, which raises when the panel is ragged.

   *Remark.* A market that starts reporting halfway through cannot serve
   as a donor for a backtest that begins before it exists, and silently
   dropping it would change the donor pool between backtests.

2. The history is untreated.

   The simulation reuses observed periods as pretend post-periods, so
   those periods must carry no real treatment effect. If a campaign ran
   in month three, backtests overlapping it inherit its effect and
   report power that the design will not reproduce.

   *Remark.* Where a genuine post-period exists, name it with
   ``post_col`` and it is held out of the simulation.

3. No interference between markets.

   Treating one market does not move another's outcome. Neighbouring
   metros that share media markets or commuters violate this, and the
   donor pool then contains partially treated units.

   *Remark.* Where the interference is known, declare it: ``cluster_col``
   or ``adjacency`` makes the affected pairs conflict, which keeps them
   out of the same test region and out of each other's donor pool. Where
   it is only suspected, ``not_to_be_treated`` bars a market from
   treatment while leaving it a donor. Neither detects interference the
   panel does not declare.

4. The pre-period relationship persists into the test.

   Weights fit on history are used for the future. A market that
   tracked the test region for a year and then diverges breaks the
   design regardless of how the design was chosen.

   *Remark.* ``scaled_l2`` and ``pre_rmspe`` in the shortlist report how
   well the fit held historically, which is the available evidence for
   this assumption and not proof of it.

5. The reference distribution represents the null.

   Detection compares the estimate to a distribution built from the panel
   itself, and which one depends on ``inference``. Placebo reassignment
   presumes the donors are exchangeable enough that a donor-based null
   describes what a no-effect test region would look like. Conformal
   presumes the pre-period residuals are exchangeable with the
   post-period ones, so a permutation of the pre-period residuals
   describes what no effect would have produced.

   *Remark.* Each has its own failure mode. With few donors the placebo
   draws overlap heavily and the standard error is optimistic, so the
   design needs a donor pool comfortably larger than the test region.
   Conformal is the one that breaks under a trending or
   seasonally-structured residual, since permuting then compares periods
   that were never comparable; ``conformal_type="block"`` permutes in
   contiguous blocks, which preserves short-run dependence and not a
   trend.

Inference and Diagnostics
-------------------------

``inference`` chooses the null a detection is taken against, and it
varies separately from ``engine``. Left unset, each engine takes its
own default: placebo for ``sdid``, conformal for ``augsynth``, which is
GeoLift's choice.

Placebo reassignment is Arkhangelsky et al.'s Algorithm 4: reassign
:math:`N_{\mathrm{tr}}` donors as pretend-treated, drop them from the
pool, refit, and take the standard deviation of the resulting estimates.
Detection is the two-sided normal test
:math:`2\big(1 - \Phi(|\hat\tau| / \hat\sigma)\big)` against ``alpha``.
The procedure needs only a donor-built counterfactual, so it runs on
either engine. It is the only one available on ``sdid``, because the
other two variance procedures those authors give -- jackknife and
bootstrap -- are undefined for the single treated series a candidate
region collapses to.

Conformal inference permutes the pre-period residuals and asks where the
observed post-period residual falls among the permuted ones. Its
argument needs those residuals to be exchangeable across the matching
window, which is what synthetic DiD's time weights exist to deny --
:math:`\lambda` says pre-periods are not interchangeable. So conformal
is available on ``augsynth`` and refused on ``sdid``, with the reason
stated in the error. ``ns`` sets the number of permutations and
``conformal_type`` the scheme. ``"block"`` is the default and takes the
panel's cyclic shifts, which preserve serial dependence; ``"iid"`` permutes
residuals freely and assumes an exchangeability a trending or seasonal
panel denies. ``"iid"`` is also the scheme whose p-value can reach exactly
zero, since a free permutation can be beaten by the observed statistic
every time; ``finite_sample_p=True`` reports
``(1 + #{stat >= observed}) / (1 + ns)`` instead, which cannot. That
correction is off by default so the GeoLift reproduction keeps augsynth's
convention; turn it on for inference you intend to report.

Holding one of the two fixed and varying the other separates the two
sources of a difference between designs. Scoring one panel with both
engines under placebo isolates the objective; scoring it with augsynth
under both nulls isolates the inference.

Power at an effect size is the detection rate across backtests. The
minimum detectable effect is the smallest magnitude whose power exceeds
``power_threshold``, taking the smaller of the detectable positive and
negative effects. Candidates are then ranked on a composite of three
dense ranks: :math:`|\mathrm{MDE}|`, the power at the MDE, and how far
the recovered lift sits from the injected one.

Under placebo inference the shortlist also carries ``mde_exact_up`` and
``mde_exact_down``, the effect sizes at which the design starts
detecting in each direction. These solve in closed form, so they do not
inherit the effect grid's step: a design that detects 0.1014 reports
that instead of the 0.150 the grid rounds it to. They also read as a
validity check. The rejection region is :math:`e < \mathrm{down}` or
:math:`e > \mathrm{up}`, and nothing forces
:math:`\mathrm{down} < 0 < \mathrm{up}`; a design whose two crossings
have the same sign already rejects at zero injected effect, which means
it is firing on its own drift. Under conformal both columns are
``nan``, because a conformal p-value re-permutes against the injected
series and is not analytic in the effect size.

Two properties keep the sweep cheap, and both are consequences of where
the fit gets its information. Every weight program reads the pre-period
only -- SDID's unit weights read the treated pre-period, its time weights
and ridge read donors and counts, augsynth's ridge reads the pre-period
matching matrices -- so none of them sees the treated post block.
Injecting an effect there cannot move the weights, which makes

.. math:: \hat\tau(e) = \hat\tau(0) + e \cdot \bar y_{\mathrm{post}}

exact. The placebo draws reassign control markets, so
:math:`\hat\sigma` does not depend on :math:`e` either. One fit and one
placebo run therefore cover the whole grid of effect sizes.

Conformal gives up the second property and keeps the first. Its p-value
tests the injected series against permutations of the pre-period
residuals, so it moves with :math:`e` and is recomputed at every grid
point. That, and not the fit, is where the augsynth path's cost sits.

Example
-------

.. code-block:: python

   import numpy as np
   import pandas as pd
   from mlsynth import GEOX
   from mlsynth.config_models import GEOXConfig

   df = pd.read_csv("basedata/geolift_test_data.csv")
   df["date"] = pd.to_datetime(df["date"])

   design = GEOX(GEOXConfig(
       df=df, unitid="location", time="date", outcome="Y",
       treatment_size=2,
       durations=[14],
       effect_sizes=[round(x, 2) for x in np.arange(-0.30, 0.35, 0.05)],
       n_backtests=5,
       n_draws=100,
       seed=0,
       n_jobs=-1,
   )).fit()

   print(design.selected_units)
   print(design.metadata["winner_mde_optimistic"])   # 0.15
   print(design.metadata["winner_mde_planning"])     # 0.10
   print(design.power.head())

On the 40-market, 105-day panel this selects ``atlanta`` and
``nashville`` with a minimum detectable effect of 0.15, out of 31
candidate regions:

.. code-block:: text

                    candidate  duration   mde  power  scaled_l2  rank
          atlanta + nashville        14  0.15    1.0      0.324   1.0
   jacksonville + minneapolis        14  0.20    1.0      0.529   2.0
          milwaukee + orlando        14  0.15    1.0      0.263   2.0
           cleveland + denver        14 -0.15    1.0      0.223   4.0
       detroit + new orleans        14 -0.15    1.0      0.530   5.0

Read that as: a 15% lift in Atlanta and Nashville together would be
detected at the 10% level in essentially every backtest tried.
Anything smaller would not be, so an experiment expecting a 5% lift needs
a longer test, a bigger region, or a different metric.

The two MDEs in ``metadata`` are the same quantity measured on different
backtests. ``winner_mde_optimistic`` is what the scan produced, and it is
the smallest MDE in a field of candidates -- the region most likely to be
picked is the one whose estimate happened to come out low, so selection
biases it downward. ``winner_mde_planning`` re-scores the winning region
on backtests deeper in history that took no part in choosing it, which is
why it is the one to plan against.

That correction is a tendency across panels, not an inequality on any
one. Here it comes out lower, 0.10 against 0.15, because the held-out
windows are different windows and carry their own noise. The bias it
corrects, and how it shrinks as ``n_backtests`` grows, are measured
directly in :doc:`replications/geox`. ``n_validation_backtests`` sets how
many held-out backtests the re-scoring gets; zero turns it off and leaves
``winner_mde_planning`` as ``None``.

``design.design_weights`` carries both weight vectors, ``donor_weights``
over markets and ``time_weights`` over pre-period dates. The time weights
are sparse: on this panel 7 of 91 pre-days carry any weight.

``design.report`` is ``None``. It is the slot for the realized effect and
stays empty until the experiment has run and post-treatment outcomes
exist.

Scanning several region sizes
-----------------------------

``treatment_size`` takes a list, so one run can score two-market regions
against five-market ones (GeoLift's ``N = c(2, 3, 4, 5)``). Candidates are
nominated once per size, pooled, and ranked together, so the shortlist
answers how large a test region has to be and which markets it should
contain at the same time. A ``treatment_size`` column carries each
candidate's size, and ``metadata["treatment_sizes"]`` the sizes scanned.

Each candidate is fit with its own treated count. Under ``engine="sdid"``
that count enters the ridge as
:math:`(N_{\mathrm{tr}} T_{\mathrm{post}})^{1/4}`, so a five-market
region is regularised more strongly than a two-market one on the same
panel.

Scanning sizes 2 through 5 on the test panel gives 123 candidates, and
the best of each size:

.. code-block:: text

    size  candidate                                                          mde  scaled_l2
       5  columbus + jacksonville + milwaukee + minneapolis + new orleans   0.10      0.370
       4  columbus + jacksonville + milwaukee + minneapolis                 0.15      0.290
       2  atlanta + nashville                                               0.15      0.324
       3  atlanta + chicago + nashville                                     0.15      0.325

Bigger regions detect smaller lifts, which is the usual trade: five
markets carry more volume than two, so the same proportional effect is
easier to see. Set against that, a larger test region costs more to run
and holds out more of the country from the control pool. The scan prices
that choice instead of assuming it.

Design constraints
------------------

``to_be_treated`` and ``not_to_be_treated`` name individual markets. The
constraint fields express rules instead, and each one narrows where the
search may look.

Interference. Two markets interfere when treating both contaminates the
comparison, either because they share a media market or because one
spills into the other. ``cluster_col`` names a per-market column (a DMA,
a state) and makes markets sharing a value conflict. ``adjacency`` takes
a square DataFrame of pairwise spillover strengths, and any off-diagonal
entry above ``spillover_threshold`` is a conflict. Supplying both takes
the union.

A conflict binds twice. No candidate region may hold two conflicting
markets, so the treated set is an independent set of the conflict graph.
And a treated market's conflicting partners are dropped from its own
donor pool, since a market contaminated by the treatment cannot serve as
its own control. The second half is the exclusion restriction, and it
applies to the backtests that score the candidate as well as to the
deployed fit, so the reported MDE reflects the pool the experiment will
actually have.

Coverage. ``stratum_col`` names a grouping the test region has to
represent, with ``min_per_stratum`` requiring at least that many treated
markets in every stratum holding an eligible market, and
``max_per_stratum`` capping any one stratum. Use this when the region has
to span regions or store formats instead of concentrating wherever the
correlations happen to be highest.

Size band. ``size_col`` with ``min_size`` and ``max_size`` bounds which
markets may be treated, both ends inclusive. The floor is a power or
operational minimum. The ceiling encodes synthesizability: a market far
larger than every donor cannot sit inside their convex hull, and the
scaled :math:`L^2` imbalance grows accordingly. Markets outside the band
stay available as donors, since the band restricts treatment eligibility
alone.

.. code-block:: python

    config = GEOXConfig(
        df=df, unitid="location", time="date", outcome="Y",
        treatment_size=[2, 3],
        durations=[14], effect_sizes=[0.05, 0.10, 0.15, 0.20],
        cluster_col="dma",             # no two treated markets in one DMA
        stratum_col="region", min_per_stratum=1,   # every region represented
        size_col="volume", min_size=5_000,         # skip markets too small to power
    )

When no combination of markets satisfies the constraints, the failure
names which constraint bound the search, in have-versus-need form, so a
design that cannot be run says why:

.. code-block:: text

    MlsynthConfigError: GEOX design is infeasible -- the binding constraint(s):
      - spillover/cluster: the largest conflict-free treated set is 2 <
        treatment_size=3. Relax the cluster/adjacency constraint, widen the
        candidate pool, or reduce treatment_size.

Every constraint is off by default, and with none configured the search
runs exactly as it does above.

Reading out the experiment
--------------------------

``fit()`` chooses a region before the experiment runs, so ``report`` is
``None`` on the returned design. Once outcomes exist, name the
post-treatment periods with ``post_col`` and the same call fills it with
the realized effect: the ATT over the post window, the observed,
counterfactual and gap paths across the whole panel, both weight
vectors, and the pre-period fit diagnostics.

The readout uses the configured engine and the configured null, the same
two the design scored itself with. A minimum detectable effect computed
one way and a readout computed another would leave the reported power
describing a test nobody ran, and the promise that the design is chosen
by the estimator that will analyse the result is the reason to use GEOX
at all. For the same reason the readout inherits the design's donor
pool: where a spillover constraint barred a treated market's
conflict-neighbours, they stay barred here.

``report.weights.summary_stats`` carries whatever the engine has to say
about its own fit alongside the shared keys: SDID's ridge and bias
correction, augsynth's intercept, its penalty and the augmentation by
name. Under ``augment=None`` there is no penalty and ``lambda_`` is
``None``, which is the answer and not a missing value.

.. code-block:: python

    design = GEOX(GEOXConfig(
        df=df, unitid="location", time="date", outcome="Y",
        post_col="post",              # 1 on the periods the campaign ran
        treatment_size=2, durations=[14],
        effect_sizes=[0.05, 0.10, 0.15],
    )).fit()

    design.report.effects.att          # realized effect
    design.report.inference.p_value    # tested against the same null

The design itself is unaffected by ``post_col``: ingestion truncates to
the pre-period before any candidate is scored, so a design fit on a
pre-only panel and one fit on the full panel choose the same region. Only
the readout sees the post periods.

``how`` sets the scale the readout is written in. The fit always runs on
the per-market mean, which keeps the target at donor scale, so ``how``
changes reported units and neither the region chosen nor the p-value:
``"mean"`` gives the per-market effect, ``"sum"`` the summed incremental
across the treated markets, which is GeoLift's convention and the one to
use when the number is going next to a spend figure. Cost from ``cpic``
is computed off the summed incremental either way.

Plots
-----

``plot_geox_design`` draws the two views GeoLift shows for a market
selection: the power curve, with the detection threshold and the minimum
detectable effect marked and the other candidates drawn faintly behind, and
the fit the design rests on.

.. code-block:: python

   from mlsynth import plot_mde_ranking, plot_geox_design

   plot_geox_design(design, power_threshold=0.8)
   plot_mde_ranking(design, top=12)

The power curve is U-shaped by construction: power falls to zero at no
injected effect and rises to one as the lift grows in either direction. Where
it crosses the threshold is the minimum detectable effect.
``plot_mde_ranking`` gives the market-selection view instead, ranking the
shortlisted regions by the smallest lift each can detect.

Verification
------------

The default pairing -- GeoLift's market-selection loop scored with
synthetic DiD -- exists in no published implementation, so GEOX has no
replication path in the sense the other estimators do: nothing to
reproduce, no simulation table to match, no reference to agree with. Its
validation is assembled from three pieces instead, and they are not
equally strong.

Each engine is cross-validated. mlsynth's synthetic DiD matches the
authors' ``synthdid`` R package on Proposition 99 and Stata ``sdid`` on
the EU ETS panel, and ``tests/test_geox.py`` asserts that the ATT GEOX
scores a backtest with matches ``SDID(...).fit()`` on the same panel and
window. The augsynth engine is the shared ``BilevelSCM``, which carries
its own validation, reached here through the adapter the seam tests pin.
The two structural properties the effect sweep relies on are checked
against brute-force recomputation in ``tests/test_geox.py``, so the
shortcut is proved and not assumed.

The harness does not perturb the engine. Force one market as the treated
region and hand GEOX the real post-period, and its readout equals
``SDID(...).fit()`` on Proposition 99 exactly -- the ATT and all 38 donor
weights -- over six treated units and seven design-knob settings
(`benchmarks/cases/geox_sdid_equivalence.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/geox_sdid_equivalence.py>`_).
That is what makes the engine's validation the design's: mlsynth's SDID
sits 1.6e-3 packs from the authors' ``synthdid`` R, and GEOX sits zero
from mlsynth's SDID. See :doc:`replications/geox_sdid_equivalence`.

The harness is cross-validated. With ``engine="augsynth"`` selected,
GEOX reproduces the market selection GeoLift itself publishes: on the
walkthrough panel, all five of its top-ranked designs come back with the
same rank, minimum detectable effect, CPIC investment and
``abs_lift_in_zero``, fourteen quantities value for value
(`benchmarks/cases/geox_augsynth_geolift.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/geox_augsynth_geolift.py>`_).
This reaches further than the engine: nomination, the backtest windows,
effect injection, the power sweep, the MDE rule and the composite rank
are shared code, so a divergence anywhere in that stack would move a
rank or an investment. It is also what licenses reading the two engines
against each other, since the harness around them is the same.

The composition is self-validated. Whether an MDE from SDID-scored
backtests means what it claims has no external referent, so
`benchmarks/cases/geox_mc.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/geox_mc.py>`_
imposes ground truth on a constructed factor DGP and measures size at
the null, out-of-sample power at the reported MDE, and the gap between
the selected winner's MDE and a region fixed in advance. See
:doc:`replications/geox`. This is the weakest of the three, because
the DGP and the claim come from the same place.

Core API
--------

.. automodule:: mlsynth.estimators.geox
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: mlsynth.config_models.GEOXConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mlsynth.utils.geox_helpers.engine
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.geox_helpers.simulate
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.geox_helpers.batch
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.geox_helpers.aggregate
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.geox_helpers.orchestration
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.geox_helpers.structures
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.geox_helpers.plotter
   :members:
   :undoc-members:
