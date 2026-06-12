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
:Status: **Done** — fully verified; the realized effect report reproduces
   GeoLift's walkthrough ATT, percent lift, incremental, and conformal p-value.
:Durable check: ``benchmarks/cases/geolift.py`` (``geolift_walkthrough``, vs the
   published vignette) and ``benchmarks/cases/geolift_augsynth_ref.py``
   (``geolift_augsynth_ref``, vs **live** augsynth via Rscript); plus
   ``mlsynth/tests/test_geolift_walkthrough.py``.

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
(``GeoLift_Test``: 40 markets, the other 38 as donors) and reports:

============================  =================
Quantity                      GeoLift
============================  =================
Average ATT (per unit/period)  ``155.556``
Percent Lift                   ``5.4%``
Incremental Y (summed)         ``4667``
Conformal p-value              ``0.01``
============================  =================

.. note::

   The vignette's printed ``155.556`` / ``4667`` is from an **older augsynth
   release**. Run against augsynth *today* the same fit returns
   ``ATT = 156.81`` (``λ = 1.673102e9``, 13 donors); ``mlsynth`` reproduces that
   **live** augsynth output to floating point — see *Live cross-check vs augsynth*
   below — so the ~0.8 % gap to the printed number is augsynth's own
   version-to-version drift, not an mlsynth discrepancy.

The walkthrough's public call (``GeoLift`` names the locations and the post
window — it is an *analysis* of a given test region, not a market search):

.. code-block:: r

   GeoLift_Test <- GeoLift(Y_id = "Y", data = GeoTestData_Test,
                           locations = c("chicago", "portland"),
                           treatment_start_time = 91, treatment_end_time = 105)
   summary(GeoLift_Test)   # ATT 155.556, Lift 5.4%, Incremental 4667, p 0.01

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
   }).fit()

   res.selected_units            # ['chicago', 'portland']
   res.report.effects.att        # 156.8  (GeoLift per-unit ATT 155.6)
   res.report.inference.p_value  # 0.011  (GeoLift 0.01)
   # how="sum" reports the summed incremental: ATT 313.6/period, p identical.

Pinned end-to-end through the public API in ``benchmarks/cases/geolift.py``
(``geolift_walkthrough``) and ``mlsynth/tests/test_geolift_walkthrough.py``.

Live cross-check vs augsynth
----------------------------

Because the printed vignette number has drifted with augsynth's version, the
durable cross-check fits **augsynth itself** and compares — the gold-standard
reference rather than a doc string. ``benchmarks/R/augsynth_geolift.R`` runs

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
chain is not needed). The case skips itself when ``Rscript`` / ``augsynth`` is
absent, so it is a no-op in CI and runs only where the reference is installed.
This is what licenses the strong claim above: ``mlsynth``'s ridge ASCM, its CV
λ-selection (the 1-SE rule), and its fixed-effect conformal refit are not merely
*close* to augsynth — they are the **same computation**, to ~7–11 significant
figures.

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
