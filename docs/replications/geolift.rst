.. _replication-geolift:

GEOLIFT — Meta's GeoLift walkthrough (augsynth cross-validation)
================================================================

:Estimator: :doc:`../geolift` — :class:`mlsynth.GEOLIFT`
:Source: Meta's **GeoLift** package (``facebookincubator/GeoLift``), the
   ``GeoLift_Walkthrough`` vignette, which runs Ben-Michael, Feller & Rothstein's
   Augmented SCM ([BMFR2021]_) via **augsynth** (``ebenmichael/augsynth``) with
   Chernozhukov–Wüthrich–Zhu conformal inference ([CWZ2021]_).
:Replication type: **Cross-validation** — match an authoritative reference
   implementation (GeoLift/augsynth) value-for-value on the package's own
   published example.
:Status: **Done** — fully verified; the realized effect report reproduces both
   of GeoLift's walkthrough summaries (the unaugmented base model and the
   ridge-augmented "best" model) — ATT, percent lift, incremental, conformal
   p-value, L2 imbalance, scaled L2, percent improvement, bias removed, and the
   donor weights.
:Durable check: ``benchmarks/cases/geolift.py`` (``geolift_walkthrough``, vs the
   published vignette), ``benchmarks/cases/geolift_marketselection.py``
   (``geolift_marketselection``, vs the BestMarkets ranking), and
   ``benchmarks/cases/geolift_augsynth_ref.py`` (``geolift_augsynth_ref``, vs
   **live** augsynth via Rscript); plus ``mlsynth/tests/test_geolift_walkthrough.py``.

Why this is the replication target
----------------------------------

The earlier port had **no** value-for-value anchor (only an end-to-end null on
the no-effect panel), because GeoLift's *market-selection* routine has no
published table. But GeoLift's **realized effect report** — the ATT and conformal
p-value it prints once a test has run — *is* published, in the
``GeoLift_Walkthrough``: it is the augsynth Augmented SCM with
``fixed_effects=TRUE``, the package's default. That gives a hard cross-validation
target for the part of ``GEOLIFT`` that does the causal inference
(:func:`~mlsynth.utils.geolift_helpers.marketselect.realize.realize_design`).

The walkthrough treats ``chicago`` + ``portland`` over the last 15 of 105 days
(``GeoLift_Test``: 40 markets, the other 38 as donors) and prints **two**
summaries — the unaugmented base model (``GeoLift(...)``) and the ridge-augmented
"best" model (``GeoLift(..., model = "best")``):

=============================  ==============  ===================
Quantity                       GeoLift base    GeoLift augmented
=============================  ==============  ===================
Average ATT (per unit/period)  ``155.556``     ``156.805``
Percent Lift                   ``5.4%``        ``5.5%``
Incremental Y (summed)         ``4667``        ``4704``
Conformal p-value              ``0.01``        ``0.01``
L2 imbalance                   ``909.489``     ``903.525``
Scaled L2                      ``0.1636``      ``0.1626``
Percent improvement (naive)    ``83.64%``      ``83.74%``
Avg estimated bias removed     —               ``-1.249``
=============================  ==============  ===================

.. note::

   The two columns are two **models**, not two augsynth versions. The base
   ``GeoLift()`` call is the unaugmented (simplex) fit — ``mlsynth`` reproduces it
   with ``augment=None`` — and ``model = "best"`` selects the ridge-augmented fit,
   which ``mlsynth`` reproduces with ``augment="ridge"`` (its default). Both match
   the printed summaries to the published digits, including the L2 imbalance,
   scaled L2, and percent-improvement diagnostics (the average bias removed is just
   the base-minus-augmented ATT gap, ``155.556 - 156.805 = -1.249``). The live
   augsynth cross-check below independently confirms the ridge fit to floating
   point.

The walkthrough's public call (``GeoLift`` names the locations and the post
window — it is an *analysis* of a given test region, not a market search):

.. code-block:: r

   GeoLift_Test <- GeoLift(Y_id = "Y", data = GeoTestData_Test,
                           locations = c("chicago", "portland"),
                           treatment_start_time = 91, treatment_end_time = 105)
   summary(GeoLift_Test)   # base:  ATT 155.556, Lift 5.4%, Incremental 4667, p 0.01

   GeoTestBest <- GeoLift(Y_id = "Y", data = GeoTestData_Test,
                          locations = c("chicago", "portland"),
                          treatment_start_time = 91, treatment_end_time = 105,
                          model = "best")
   summary(GeoTestBest)    # ridge: ATT 156.805, Lift 5.5%, Incremental 4704, p 0.01

``mlsynth`` reaches the same numbers through its **public estimator** —
``GEOLIFT(...).fit()`` with ``fixed_effects=True`` (the default). The estimator is
a market-selection *design*, so the two markets are pinned with ``to_be_treated``
+ ``treatment_size`` (the only candidate of that size) and the post window is
marked by ``post_col``; ``res.report`` is the realized effect report — the
analogue of ``summary(GeoLift_Test)``:

.. code-block:: python

   import pandas as pd
   from mlsynth import GEOLIFT

   df = pd.read_csv("basedata/geolift_test_data.csv")          # GeoLift_Test
   dates = sorted(df["date"].unique())
   df["post"] = df["date"].isin(set(dates[90:])).astype(int)   # days 91-105

   res = GEOLIFT({
       "df": df, "outcome": "Y", "unitid": "location", "time": "date",
       "treatment_size": 2, "to_be_treated": ["chicago", "portland"],
       "durations": [15], "effect_sizes": [0.0, 0.10], "post_col": "post",
       "how": "mean", "fixed_effects": True, "display_graphs": False,
       "augment": "ridge",   # the "best" model; use augment=None for the base model
   }).fit()

   res.selected_units            # ['chicago', 'portland']
   res.report.effects.att        # 156.805  (ridge "best"; augment=None gives 155.556)
   res.report.inference.p_value  # 0.011    (GeoLift 0.01)
   # how="sum" reports the summed incremental: ATT 313.6/period, p identical.

Pinned end-to-end through the public API in ``benchmarks/cases/geolift.py``
(``geolift_walkthrough``) — both models and every printed quantity (ATT, lift,
incremental, conformal p, L2 imbalance, scaled L2, percent improvement, bias
removed, and the 13 donor weights) — and ``mlsynth/tests/
test_geolift_walkthrough.py``.

Live cross-check vs augsynth
----------------------------

To pin the augmented fit against the gold-standard reference rather than a doc
string, the durable cross-check fits **augsynth itself** and compares.
``benchmarks/R/augsynth_geolift.R`` runs

.. code-block:: r

   augsynth(Y ~ trt, unit = location, time = t, data = panel,
            progfunc = "ridge", scm = TRUE, fixedeff = TRUE)   # GeoLift's fit

on the same chicago+portland panel (the two test geos averaged into one treated
series, exactly as GeoLift aggregates them), and ``benchmarks/cases/
geolift_augsynth_ref.py`` (``geolift_augsynth_ref``) checks ``mlsynth`` against
it. The agreement is essentially floating-point:

========================  ====================  ===================
Quantity                  augsynth (live)       ``mlsynth``
========================  ====================  ===================
Ridge penalty ``λ``        ``1.673102e9``        rel-diff ``1.6e-11``
Post-period ATT            ``156.8054``          ``156.8052``
Donor weights (max ``Δ``)  13 non-zero           ``4.3e-7``
========================  ====================  ===================

Install the reference once with ``benchmarks/R/install_augsynth.sh`` (augsynth
only — GeoLift's fit *is* augsynth, so the heavy ``MarketMatching`` → ``Boom``
chain is not needed). The install is **commit-pinned** — augsynth ``0.2.0 @
7a90ea4`` and every source-compiled dependency frozen to a SHA (``S7``,
``LiblineaR``, ``osqp``) as of 2026-06-12 — so the cross-check runs the *same*
reference code every time, rather than a moving ``master`` tip whose results
could shift release to release. The case skips itself
when ``Rscript`` / ``augsynth`` is absent, so it is a no-op in CI and runs only
where the reference is installed.
This is what licenses the strong claim above: ``mlsynth``'s ridge ASCM, its CV
λ-selection (the 1-SE rule), and its fixed-effect conformal refit are not merely
*close* to augsynth — they are the **same computation**, to ~7–11 significant
figures.

Market selection (the BestMarkets ranking)
------------------------------------------

The walkthrough's other half is the *search* for a test region, run on the
90-period pre-test panel:

.. code-block:: r

   GeoLiftMarketSelection(data = GeoTestData_PreTest, treatment_periods = c(10, 15),
       N = c(2, 3, 4, 5), effect_size = seq(0, 0.2, 0.05), include_markets = "chicago",
       exclude_markets = "honolulu", cpic = 7.50, budget = 1e5, fixed_effects = TRUE,
       side_of_test = "two_sided")

GeoLift prints a ranked ``BestMarkets`` table; its top five designs are reproduced
by ``mlsynth`` value-for-value — rank, CPIC investment (exact to the cent), MDE,
and ``abs_lift_in_zero``:

===========================================  ===  ====  =============  ====
Design (chicago forced in)                   dur  rank  investment     |MDE
===========================================  ===  ====  =============  ====
chicago, portland                            15   1     ``$64,563.75``  0.10
chicago, cincinnati, houston, portland       15   1     ``$74,118.38``  0.05
chicago, portland                            10   3     ``$43,646.25``  0.10
chicago, cincinnati, houston, portland       10   3     ``$99,027.75``  0.10
chicago, houston, portland                   10   5     ``$75,389.25``  0.10
===========================================  ===  ====  =============  ====

``mlsynth``'s ``GEOLIFT`` design takes one ``treatment_size``, so the case runs it
for ``N = 2, 3, 4, 5`` and pools the per-design MDE rows, then applies GeoLift's
composite rank (the mean of three ``dense_rank``s over |MDE|, power, and
``abs_lift_in_zero``, ties = ``min``) across the pool — exactly how
``GeoLiftMarketSelection`` ranks its single results table.

.. note::

   Matching this top-five required fixing ``include_markets`` handling to GeoLift's
   *generate-then-filter* semantics (``pre_test_power.R``): candidates are
   generated ignoring the forced markets, then kept only if they already contain
   them, so a forced market is never welded onto an anchor it is uncorrelated
   with. The earlier *remove-and-reattach* approach manufactured low-correlation
   candidates (e.g. ``{chicago, las vegas}``) that GeoLift never forms, which then
   polluted the ranking (the composite ranks on MDE/power, not fit). Only the
   stable top-five are pinned; the marginal rank-six design differs by augsynth's
   own version-to-version scoring at the tail.

Pinned in ``benchmarks/cases/geolift_marketselection.py``
(``geolift_marketselection``); the per-design investment is also pinned
independently by ``geolift_cpic``.

.. note::

   The walkthrough's *power curve* (``GeoLiftPower`` over an effect-size grid) is
   published only as a plot, with no numeric table, so it has no value-for-value
   target. The quantities that build it — each design's per-effect-size power and
   MDE — are the same ones validated by ``geolift_walkthrough`` and
   ``geolift_marketselection``; a live numeric cross-check would require the full
   GeoLift R package (the ``MarketMatching`` → ``Boom`` chain), which is heavier
   than the ``augsynth``-only reference used here.

What it took to match — the four ingredients
--------------------------------------------

Reaching parity required reproducing augsynth's pipeline **from scratch** and
verifying each component against the published number. Four ingredients, each
necessary; drop any one and the ATT or the p-value diverges.

1. **Unit fixed effects** (augsynth ``fixed_effects=TRUE``, GeoLift's default).
   ``demean_data`` subtracts **each unit's own pre-period mean** from all of its
   periods, fits the SCM on the residuals (matching *shapes*, not *levels*), and
   restores the level with an intercept. This is what stops the donor pool from
   absorbing a treated-unit **level shift**: a convex/ridge combination of
   level-matched donors *can* chase a post-period jump, but once every unit is
   demeaned it cannot. Without it the realized ATT is wrong (≈209 vs 311) **and**
   the conformal refit absorbs the effect (p ≈ 0.56).

2. **Fit the mean of the treated units** (augsynth ``colMeans``), not their sum.
   This is *not* scale-invariant for the conformal: a sum of :math:`k` markets
   sits at :math:`k\times` donor scale, outside the convex hull, so the simplex
   base fits it badly and the residual path changes — sum gives p ≈ 0.68 where
   mean gives p ≈ 0.01. ``GEOLIFT`` fits the per-unit mean and rescales the
   *reported* paths by :math:`k` when ``how="sum"`` (the p-value, a ratio of
   norms, is invariant to that global reporting scale).

3. **The faithful conformal refit** ([CWZ2021]_, augsynth ``conformal``). For the
   joint null the Augmented SCM is **refit on all periods** (augsynth's
   ``cbind(X, y)``); under fixed effects the refit demeans by the **full-path**
   mean (``rowMeans`` of the augmented matching matrix). The post-block statistic
   :math:`(\sum |u_t|^q / \sqrt{n})^{1/q}` is compared to permutations of the
   residual path. The all-period refit is what makes the pre/post residuals
   **exchangeable** — and hence the test calibrated.

4. **augsynth's ridge ASCM** itself: a simplex base + a **period-space** ridge
   correction
   :math:`w = w_\text{scm} + (X_1 - X_c^\top w_\text{scm})^\top (X_c X_c^\top +
   \lambda I)^{-1} X_c`, with :math:`\lambda` selected by leave-one-period-out CV
   under the **1-SE rule** (augsynth's default ``min_1se = TRUE``). ``mlsynth``'s
   :func:`~mlsynth.utils.bilevel.ridge_augment.ridge_augment_weights` reproduces
   these weights to ``corr = 1.0000`` on matched inputs.

Two traps we walked into (and out of)
-------------------------------------

These are the cross-codebase-consistency lessons worth carrying to the next port.

* **A calibrated test can look "anti-powered."** Before isolating the fixed
  effect, the symptom was "our conformal p (0.57) is far from GeoLift's (0.01),
  so the conformal must be broken/anti-powered." It is not. A 40-market
  **placebo study** on the no-effect panel showed the all-period refit is
  *well-calibrated* (rejection rate ≈ 0.10 at :math:`\alpha = 0.10`), and the
  tempting "fix" — fitting once on the pre-period and permuting the gap path —
  is the one that is **broken** (≈ 50 % false-positive rate, because pre
  residuals are in-sample and post residuals are out-of-sample, so they are
  *not* exchangeable). The low p was never the test; it was the **fit** (missing
  fixed effects), which estimated a smaller, level-absorbed effect that a
  *correct* test then correctly judged insignificant. **Diagnose the estimand
  before blaming the inference.**

* **Match defaults before mechanisms.** Two of the four ingredients
  (``fixed_effects=TRUE``, ``min_1se=TRUE``) are just augsynth/GeoLift *defaults*
  we had not mirrored; one (mean vs sum) is an aggregation default. Only the
  fourth is "mechanism." When two codebases disagree, **enumerate the reference's
  defaults first** — most divergences are an unmatched default, not a wrong
  formula. Reproducing the reference end-to-end *from scratch in a scratch
  script* (here, ~40 lines of NumPy that hit ATT 312 / p 0.011) localizes which
  default matters far faster than reading either codebase.

.. [BMFR2021] Ben-Michael, E., Feller, A., & Rothstein, J. (2021). The Augmented
   Synthetic Control Method. *Journal of the American Statistical Association*.

.. [CWZ2021] Chernozhukov, V., Wüthrich, K., & Zhu, Y. (2021). An Exact and Robust
   Conformal Inference Method for Counterfactual and Synthetic Controls.
   *Journal of the American Statistical Association*.
