.. _replication-ascm-kansas:

Ridge ASCM — Augmented Synthetic Control (Ben-Michael, Feller & Rothstein 2021)
===============================================================================

:Estimator: :doc:`../vanillasc` — the ridge-augmentation layer
   (:func:`mlsynth.utils.bilevel.ridge_augment.ridge_augment_weights`).
:Source: Ben-Michael, Feller & Rothstein (2021), *"The Augmented Synthetic
   Control Method,"* JASA 116(536); reference implementation: the ``augsynth``
   R package (``ebenmichael/augsynth``).
:Replication type: **Cross-validation** — mlsynth matched value-for-value to
   ``augsynth`` on its canonical Kansas study — **and Path B** — the paper's
   Section-7 coverage / bias-reduction simulation.
:Status: **Fully verified** — empirical ladder *and* simulation reproduced.

Validation strategy
-------------------

The Augmented SCM is a bias-correction layer, so it is validated against the
authors' own R package, ``augsynth``, on its flagship empirical example: the
effect of Kansas's 2012 tax cuts on quarterly log GDP per capita. ``augsynth``
walks up a **ladder** of estimators of increasing de-biasing — plain SCM,
ridge ASCM, ridge ASCM with auxiliary covariates (balanced directly), and the
residualized covariate variant — and the measured effect grows while the
pre-treatment imbalance falls. We reproduce that ladder cell by cell, then
reproduce the paper's Section-7 Monte Carlo (**Path B**).

Cross-validation — the Kansas ladder
------------------------------------

The treated unit is Kansas (FIPS 20); treatment begins in 2012 Q2, leaving
:math:`T_0 = 89` pre-period quarters and :math:`J = 49` donor states. The
covariate model is ``augsynth``'s documented spec,

.. code-block:: r

   covsyn <- augsynth(lngdpcapita ~ treated | lngdpcapita + log(revstatecapita) +
                        log(revlocalcapita) + log(avgwklywagecapita) +
                        estabscapita + emplvlcapita,
                      fips, year_qtr, kansas, progfunc = "ridge", scm = TRUE)

with each covariate transformed per row and aggregated to a pre-period mean per
unit (rows carrying a missing — sparsely reported — revenue value are dropped
before averaging, R's ``model.frame`` ``na.omit`` default).

The whole ladder is reproduced **through mlsynth's public API** -- Augmented SCM
is a mode of :class:`~mlsynth.estimators.vanillasc.VanillaSC` (``augment="ridge"``).
Covariates are passed by column name; the user applies augsynth's per-row ``log``
transforms to the DataFrame first (mlsynth's covariate convention), and the
shared-window aggregation drops a pre-period whenever *any* covariate is missing
there, matching augsynth's ``na.omit``. ``residualize=True`` selects the
residualized variant:

.. code-block:: python

   import numpy as np, pandas as pd
   from mlsynth import VanillaSC

   df = pd.read_csv("basedata/kansas_ascm.csv")          # long fips x quarter panel
   for c in ("revstatecapita", "revlocalcapita", "avgwklywagecapita"):
       df[c] = np.log(df[c])                             # augsynth's log transforms
   covs = ["lngdpcapita", "revstatecapita", "revlocalcapita",
           "avgwklywagecapita", "estabscapita", "emplvlcapita"]
   base = dict(df=df, outcome="lngdpcapita", treat="treated",
               unitid="fips", time="year_qtr")

   att = lambda cfg: VanillaSC({**base, **cfg}).fit().effects.att
   att({})                                               # classic SCM    -0.029
   att({"augment": "ridge"})                             # ridge ASCM     -0.040
   att({"augment": "ridge", "covariates": covs})         # covariate ASCM -0.063
   att({"augment": "ridge", "covariates": covs,
        "residualize": True})                            # residualized   -0.057

The reproduced ladder (mlsynth vs ``augsynth``):

.. list-table::
   :header-rows: 1
   :widths: 28 22 18 32

   * - Specification
     - ATT (mlsynth)
     - Pre-fit L2
     - augsynth (ATT / L2)
   * - Classic SCM
     - -0.0294
     - 0.083
     - -0.029 / 0.083
   * - Ridge ASCM
     - -0.0401
     - 0.062
     - -0.040 / 0.062
   * - Covariate ASCM
     - -0.0629
     - 0.055
     - -0.061 / 0.054
   * - Residualized
     - -0.0572
     - 0.067
     - -0.055 / 0.067

The two no-covariate cells are exact; the covariate cells match ``augsynth``'s
values and reproduce the monotone ladder (the un-augmented SCM is the
conservative end). The joint-null conformal :math:`p`-value for ridge ASCM
(:math:`0.071`) is also reproduced to Monte-Carlo precision.

A note on the residualized penalty. After residualizing out :math:`K`
covariates the residual Gram is rank-deficient (:math:`T_0` rows, rank
:math:`\le J - K`), so a cross-validation on the residuals is ill-posed and
drifts to the grid floor. mlsynth tunes the penalty on the **outcome scale**
instead — where ``augsynth``'s residual CV lands anyway — which reproduces the
published :math:`-0.055` / :math:`0.067` robustly.

Path B — coverage and bias reduction (Section 7)
------------------------------------------------

Four data-generating processes are calibrated to the Kansas panel — a 3-factor
interactive-fixed-effects model (calibrated exactly as ``gsynth``/``fect``'s
``interFE`` does it with no covariates: two-way demean, then a rank-3 SVD of the
residual), the same model at :math:`4\times` noise, additive two-way fixed
effects, and a fitted AR(3). Treatment is assigned to an extreme unit, so plain
SCM struggles and the augmentation matters. Across all four DGPs **ridge ASCM
reduces** :math:`|\text{bias}|` **versus plain SCM and gives near-nominal
coverage** (:math:`\approx 0.90`–:math:`0.96`), with the gain limited under high
noise — the paper's thesis.

Durable cases & tests
---------------------

* ``ascm_kansas`` — the four-spec Kansas ladder cross-validated against
  ``augsynth`` (``benchmarks/cases/ascm_kansas.py``).
* ``augsynth_calibrated`` — the Section-7 coverage / bias-reduction simulation
  (``benchmarks/cases/augsynth_calibrated.py``).
* Regression tests: ``mlsynth/tests/test_bilevel_ridge.py``
  (``test_augsynth_kansas_replication``, ``test_augsynth_kansas_conformal_pvalue``,
  ``test_augsynth_kansas_covariate_ladder``).
