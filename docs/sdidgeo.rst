Synthetic DiD Geo Experiment Design (SDIDGEO)
=============================================

.. currentmodule:: mlsynth

When to Use This Estimator
--------------------------

SDIDGEO decides which geographic markets to treat in a marketing geo
experiment, before the experiment runs. You have daily or weekly sales
for a few dozen cities, you can turn a campaign on in some of them, and
you must pick which ones. SDIDGEO answers that by rehearsing the
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
pairs, which suits a rollout where every market must be used. SDIDGEO
picks a small test region and leaves the rest as donors. :doc:`spcd`
scores a design from holdout residuals when the estimator is a fixed
linear contrast.

What Distinguishes It
---------------------

The market-selection loop follows GeoLift, Meta's geo-experiment
package: anchor a candidate region at each market, add that market's
most correlated neighbours, sweep effect sizes over several backward
backtests, and rank by a composite score. What changes is the estimator
doing the scoring.

GeoLift scores with augmented synthetic control, which matches the
treated market's path period by period. SDIDGEO scores with synthetic
difference-in-differences (Arkhangelsky, Athey, Hirshberg, Imbens and
Wager, 2021), which weights donor markets and pre-periods at the same
time. Two consequences follow.

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

Synthetic DiD chooses unit weights :math:`\omega` over donors and time
weights :math:`\lambda` over pre-periods, and estimates

.. math::

   \hat\tau \;=\;
   \Big( \bar y_{\mathrm{post}} - \lambda' y_{\mathrm{pre}} \Big)
   \;-\;
   \Big( \omega' \bar{\mathbf{Y}}_{0,\mathrm{post}}
         - \lambda' \mathbf{Y}_{0,\mathrm{pre}}\, \omega \Big).

Both weight vectors are non-negative and sum to one. Rearranged, the
counterfactual path is
:math:`\mathbf{Y}_{0}\omega + \big(\lambda' y_{\mathrm{pre}} - \lambda'
\mathbf{Y}_{0,\mathrm{pre}}\omega\big)`, and :math:`\hat\tau` is the
average gap against it over the window.

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

   *Remark.* SDIDGEO does not model spillover. Exclude the markets you
   suspect with ``not_to_be_treated``, which keeps them out of candidate
   regions while leaving them as donors.

4. The pre-period relationship persists into the test.

   Weights fit on history are used for the future. A market that
   tracked the test region for a year and then diverges breaks the
   design regardless of how the design was chosen.

   *Remark.* ``scaled_l2`` and ``pre_rmspe`` in the shortlist report how
   well the fit held historically, which is the available evidence for
   this assumption and not proof of it.

5. The placebo distribution represents the null.

   Detection compares the estimate to a standard error built by
   reassigning donor markets as pretend-treated. This presumes the donors
   are exchangeable enough that a donor-based null describes what a
   no-effect test region would look like.

   *Remark.* With few donors the placebo draws overlap heavily and the
   standard error is optimistic. The design needs a donor pool comfortably
   larger than the test region.

Inference and Diagnostics
-------------------------

Arkhangelsky et al. give three variance procedures. Jackknife and
bootstrap are undefined for a single treated series, which is what a
candidate region collapses to, so SDIDGEO uses the placebo procedure
(their Algorithm 4): reassign :math:`N_{\mathrm{tr}}` donors as
pretend-treated, drop them from the pool, refit, and take the standard
deviation of the resulting estimates. Detection is the two-sided normal
test :math:`2\big(1 - \Phi(|\hat\tau| / \hat\sigma)\big)` against
``alpha``.

This is where SDIDGEO parts company with GeoLift, which uses a conformal
permutation test. The conformal argument needs the residuals to be
exchangeable across the matching window, and synthetic DiD's time
weights exist because pre-periods are not interchangeable.

Power at an effect size is the detection rate across backtests. The
minimum detectable effect is the smallest magnitude whose power exceeds
``power_threshold``, taking the smaller of the detectable positive and
negative effects. Candidates are then ranked on a composite of three
dense ranks: :math:`|\mathrm{MDE}|`, the power at the MDE, and how far
the recovered lift sits from the injected one.

Two properties keep the sweep cheap, and both are consequences of where
the weights get their information. The unit-weight program reads the
treated pre-period, the time-weight program reads only donors, and the
ridge reads only donors and counts, so no program sees the treated post
block. Injecting an effect there cannot move :math:`\omega` or
:math:`\lambda`, which makes

.. math:: \hat\tau(e) = \hat\tau(0) + e \cdot \bar y_{\mathrm{post}}

exact. The placebo draws reassign control markets, so
:math:`\hat\sigma` does not depend on :math:`e` either. One fit and one
placebo run therefore cover the whole grid of effect sizes.

Example
-------

.. code-block:: python

   import numpy as np
   import pandas as pd
   from mlsynth import SDIDGEO
   from mlsynth.config_models import SDIDGEOConfig

   df = pd.read_csv("basedata/geolift_test_data.csv")
   df["date"] = pd.to_datetime(df["date"])

   design = SDIDGEO(SDIDGEOConfig(
       df=df, unitid="location", time="date", outcome="Y",
       treatment_size=[2, 3, 4, 5],
       durations=[14],
       effect_sizes=[round(x, 2) for x in np.arange(-0.30, 0.35, 0.05)],
       n_backtests=5,
       n_draws=100,
       seed=0,
       n_jobs=-1,
   )).fit()

   print(design.selected_units)
   print(design.metadata["winner_mde"])
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

Each candidate is fit with its own treated count, which enters the SDID
ridge as :math:`(N_{\mathrm{tr}} T_{\mathrm{post}})^{1/4}`, so a
five-market region is regularised more strongly than a two-market one on
the same panel.

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

    config = SDIDGEOConfig(
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

    MlsynthConfigError: SDIDGEO design is infeasible -- the binding constraint(s):
      - spillover/cluster: the largest conflict-free treated set is 2 <
        treatment_size=3. Relax the cluster/adjacency constraint, widen the
        candidate pool, or reduce treatment_size.

Every constraint is off by default, and with none configured the search
runs exactly as it does above.

Plots
-----

``plot_sdidgeo_design`` draws the two views GeoLift shows for a market
selection: the power curve, with the detection threshold and the minimum
detectable effect marked and the other candidates drawn faintly behind, and
the fit the design rests on.

.. code-block:: python

   from mlsynth import plot_mde_ranking, plot_sdidgeo_design

   plot_sdidgeo_design(design, power_threshold=0.8)
   plot_mde_ranking(design, top=12)

The power curve is U-shaped by construction: power falls to zero at no
injected effect and rises to one as the lift grows in either direction. Where
it crosses the threshold is the minimum detectable effect.
``plot_mde_ranking`` gives the market-selection view instead, ranking the
shortlisted regions by the smallest lift each can detect.

Verification
------------

The engine is cross-validated against mlsynth's own :doc:`sdid`
implementation: ``tests/test_sdidgeo.py`` asserts that the ATT SDIDGEO
scores a backtest with matches ``SDID(...).fit()`` on the same panel and
window. The two structural properties the effect sweep relies on are
checked against brute-force recomputation in the same file, so the
shortcut is proved and not assumed.

A durable benchmark case is not yet attached. Ranking a design against an
external reference needs known ground truth, since no published
implementation pairs synthetic DiD with GeoLift's market-selection loop;
the natural route is a Path B simulation that injects a known lift and
scores recovery.

Core API
--------

.. automodule:: mlsynth.estimators.sdidgeo
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: mlsynth.config_models.SDIDGEOConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mlsynth.utils.sdidgeo_helpers.engine
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.sdidgeo_helpers.simulate
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.sdidgeo_helpers.batch
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.sdidgeo_helpers.aggregate
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.sdidgeo_helpers.orchestration
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.sdidgeo_helpers.structures
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.sdidgeo_helpers.plotter
   :members:
   :undoc-members:
