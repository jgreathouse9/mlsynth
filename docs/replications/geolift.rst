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
:Durable check: ``benchmarks/cases/geolift.py`` (``geolift_walkthrough``) and
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

``mlsynth`` reproduces this with ``fixed_effects=True`` (the default):

.. code-block:: python

   import pandas as pd
   from mlsynth.utils.datautils import geoex_dataprep
   from mlsynth.utils.geolift_helpers.marketselect.realize import realize_design

   df = pd.read_csv("basedata/geolift_test_data.csv")          # GeoLift_Test
   Ywide = geoex_dataprep(df, "location", "date", "Y")["Ywide"]
   rep = realize_design(Ywide, frozenset({"chicago", "portland"}), pre_periods=90,
                        how="mean", augment="ridge", fixed_effects=True,
                        ns=2000, seed=0, conformal_type="iid")
   rep.effects.att          # 156.8  (GeoLift per-unit ATT 155.6)
   rep.inference.p_value     # 0.011  (GeoLift 0.01)
   # how="sum" reports the summed incremental: ATT 313.6/period, p identical.

Pinned in ``mlsynth/tests/test_geolift_walkthrough.py``.

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
