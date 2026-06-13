Rolling-Transformation DiD (ROLLDID)
====================================

.. currentmodule:: mlsynth

When to use
-----------

``ROLLDID`` is for difference-in-differences on panel data when the number of
**treated** units, **control** units, or both is *small* — the regime where the
usual cluster-robust / large-:math:`N` asymptotics are unreliable. It is a
clean-room (MIT) implementation of the rolling-transformation method of Lee &
Wooldridge (2026) [LW2026]_: collapse the panel to a single cross-sectional
observation per unit by a pre-treatment transformation, then read the ATT off a
cross-sectional regression on a treatment indicator. Because that regression is
cross-sectional with one observation per unit, **exact** finite-sample inference
is available — even with a single treated unit — and it composes naturally with
randomization inference. It is a regression/doubly-robust complement to the
synthetic-control family (:doc:`sdid`, :doc:`fdid`): same small-donor regime, a
different identification lever (parallel trends after removing unit means or
trends, rather than a convex donor combination).

Notation
--------

There are :math:`N` units over :math:`T` periods; treatment first turns on at
:math:`S`. Unit :math:`i` has outcome :math:`Y_{it}`, a cohort
:math:`g_i \in \{S, \dots, T\}` (its first treated period) or is **never
treated**. The treatment indicator is :math:`W_{it} = D_i \cdot \mathbf 1\{t \ge
g_i\}`, absorbing once on.

Mathematical formulation
------------------------

Rolling transformation
~~~~~~~~~~~~~~~~~~~~~~~~

Each unit's outcome is residualized against its **own pre-treatment** path, then
averaged over the post window. With ``rolling="demean"`` (Procedure 2.1) the
pre-period mean is removed,

.. math::

   \dot{Y}_{it} = Y_{it} - \overline{Y}_{i,\mathrm{pre}(g)},
   \qquad
   \overline{Y}_{i,\mathrm{pre}(g)} = \frac{1}{g-1}\sum_{t=1}^{g-1} Y_{it},
   \qquad t \ge g,

and with ``rolling="detrend"`` (Procedure 3.1) a pre-period linear trend, fit on
:math:`t = 1, \dots, g-1`, is projected into the post period and removed,

.. math::

   \ddot{Y}_{it} = Y_{it} - \bigl(\hat A_i + \hat B_i\, t\bigr),
   \qquad t \ge g .

The unit's single cross-sectional regressand is the post-average of its
transformed outcome, :math:`\Delta \overline{Y}_i = (T-g+1)^{-1}\sum_{t\ge g}
\dot{Y}_{it}` (or :math:`\ddot{Y}`).

Estimation
~~~~~~~~~~

The ATT is the coefficient on the treatment indicator in the **cross-sectional**
regression — equivalently a difference in means,

.. math::

   \widehat{\tau} = \operatorname*{arg\,min}_{\alpha,\tau}
   \sum_{i=1}^{N}\bigl(\Delta \overline{Y}_i - \alpha - \tau D_i\bigr)^2
   \;=\; \overline{\Delta Y}_{\text{treated}} - \overline{\Delta Y}_{\text{control}} .

**Common timing** (one cohort) additionally yields a per-period ATT
:math:`\widehat{\tau}_t` by regressing the period-:math:`t` transformed value on
:math:`D_i` (the event study); the aggregate is their mean. **Staggered**
adoption uses the same machinery with never-treated units as the comparison: the
collapsed regressand is a treated unit's own-cohort post-average and a
never-treated unit's cohort-share-weighted average over all cohort definitions
(:math:`\widehat\omega_g = N_g / N_{\mathrm{treat}}`), so a single cross-sectional
regression returns the cohort-share-weighted aggregate :math:`\widehat\tau_\omega`.

Assumptions
-----------

1. **No anticipation.** Potential outcomes equal the never-treated state before
   treatment, :math:`Y_{it}(g) = Y_{it}(\infty)` for :math:`t < g`.

   *Remark.* The pre-period average/trend is estimated only on :math:`t < g`, so
   anticipation in the last pre-periods would contaminate the baseline; the
   ``detrend`` option and shorter pre-windows are the sensitivity levers.

2. **Parallel trends.** The transformed control regressand is mean-independent of
   treatment, :math:`E[\Delta\overline{Y}_i(0)\mid D_i] = \alpha`. Demeaning
   allows an arbitrary common trend; detrending additionally allows
   **unit-specific linear trends** (a strictly weaker requirement).

   *Remark.* This is the method's edge over synthetic control when pre-trends are
   heterogeneous but approximately linear — exactly the Prop 99 picture, where
   detrending tracks California pre-1989 far better than a donor average.

3. **Classical linear model (for exact inference).** Conditional on :math:`D_i`,
   :math:`U_i \mid D_i \sim \mathcal N(0, \sigma^2)`.

   *Remark.* Because :math:`\Delta\overline{Y}_i` averages over time, a CLT across
   :math:`t` makes normality a good approximation even when :math:`N` is tiny.
   This is what licenses exact inference with a single treated unit.

Inference
---------

* ``inference="exact"`` — under the CLM the :math:`t`-statistic is exactly
  :math:`\mathcal T_{N-2}`, giving exact tests and confidence intervals even with
  :math:`N_1 = 1` (the single-treated case is the studentized-residual / outlier
  test of Donald & Lang).
* ``inference="hc3"`` — heteroskedasticity-robust (MacKinnon–White HC3); use only
  with a handful of treated **and** control units. With a single treated unit its
  leverage is 1 and HC3 is undefined; ROLLDID raises rather than returning a
  degenerate SE.
* ``inference="ri"`` — randomization inference: an exact-style permutation
  :math:`p`-value that does not require normality.

Example: California Prop 99
---------------------------

Reproduces the paper's Table 3 (common timing, single treated unit) on the
bundled Abadie smoking panel. The outcome is **log** per-capita cigarette sales.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from mlsynth import ROLLDID

   url = ("https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
          "refs/heads/main/basedata/smoking_data.csv")
   df = pd.read_csv(url)                       # 39 states x 1970-2000
   df["logcig"] = np.log(df["cigsale"])
   df["treat"] = df["Proposition 99"].astype(int)   # California from 1989

   res = ROLLDID({
       "df": df, "outcome": "logcig", "treat": "treat",
       "unitid": "state", "time": "year",
       "rolling": "detrend", "inference": "exact", "display_graphs": False,
   }).fit()

   print(res.effects.att, res.inference.standard_error, res.inference.p_value)
   #  -0.227   0.094   0.021   (paper Table 3)
   print(res.per_period[["time", "att"]].tail(3))     # event study; tau_2000 ~ -0.403

For the staggered case, pass a panel whose treatment indicator turns on at
different times per unit (e.g. ``basedata/castle.csv``); ``res.effects.att`` is
then the cohort-share-weighted aggregate and ``res.per_cohort`` the breakdown.

Verification
------------

Path A, both empirical applications reproduced to the reported precision and
cross-validated (during development) against the AGPL ``lwdid`` package as a
black-box oracle — clean-room, sharing no code. California Prop 99 (Table 3):
demean ATT ``-0.422`` (se ``0.121``), detrend ``-0.227`` (se ``0.094``), detrend
exact :math:`p` ``0.021``. Castle laws (§7.2, staggered): demean aggregate
``0.092`` (se ``0.057``), detrend ``0.067`` (HC3 se ``0.055``). See
:doc:`replications/rolldid`; durable case ``benchmarks/cases/rolldid_lw.py``.

.. [LW2026] Lee, S. J., & Wooldridge, J. M. (2026). Simple Approaches to
   Inference with Difference-in-Differences Estimators with Small
   Cross-Sectional Sample Sizes.

Core API
--------

.. autoclass:: mlsynth.ROLLDID
   :members: fit

.. autoclass:: mlsynth.utils.rolldid_helpers.config.ROLLDIDConfig
   :members:

.. autoclass:: mlsynth.utils.rolldid_helpers.structures.ROLLDIDResults
   :members:
