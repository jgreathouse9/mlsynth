Rolling-Transformation DiD (ROLLDID)
====================================

.. currentmodule:: mlsynth

When to Use This Estimator
--------------------------

``ROLLDID`` implements the rolling-transformation difference-in-differences
estimator of Lee & Wooldridge [LW2026]_ — a clean-room (MIT) build from the
paper's equations. It is for panel DiD when the number of treated units, the
number of control units, or both is small: the regime where the usual
cluster-robust / large-:math:`N` asymptotics are unreliable and a single
mis-measured cluster can drive the answer. Its defining feature is that it
collapses the panel into one cross-sectional observation per unit by a
pre-treatment transformation, and then reads the treatment effect off an
ordinary cross-sectional regression. Two consequences follow.

First, because that regression is cross-sectional with independent observations,
inference does not require clustering, weak time-series dependence, or a long
panel: under the classical linear model it is exact in finite samples — valid
even with a *single* treated unit (:math:`N_1 = 1`) — and it composes naturally
with randomization inference.

Second, in its detrending form it allows unit-specific linear trends, a
strict relaxation of parallel trends. This is what lets it track a treated unit
whose pre-period drifts away from the donor average — exactly the California
Proposition 99 picture — without a convex donor combination.

``ROLLDID`` is therefore the regression complement to the synthetic-control
family in mlsynth (:doc:`sdid`, :doc:`fdid`, :doc:`vanillasc`): the same
small-donor regime, a different identification lever (parallel trends after
removing unit means or trends, rather than a weighted donor combination). On
short, donor-starved staggered panels — where SC-style per-cohort weight
optimisation becomes unstable — it stays well behaved, because it estimates no
weights at all.

Reach for ROLLDID when
~~~~~~~~~~~~~~~~~~~~~~~

* you have few treated and/or few control units and want inference you can
  defend in finite samples (down to one treated unit);
* pre-trends are heterogeneous but approximately linear, so detrending buys
  you a weaker identifying assumption than parallel trends;
* adoption is common-timing or staggered, and you want a per-period event
  study (common timing) or a cohort-share-weighted aggregate (staggered) with
  honest standard errors;
* you want a transparent, weight-free alternative to SC / SDID to report
  alongside them.

Do not use ROLLDID when
~~~~~~~~~~~~~~~~~~~~~~~~

* No never-treated units exist (staggered case). The aggregate uses the
  never-treated group as the comparison; with everyone eventually treated use a
  not-yet-treated design or :doc:`sdid`.
* The treated unit's pre-trend is non-linear / complex. Linear detrending
  cannot remove it; a synthetic control (:doc:`fdid`, :doc:`vanillasc`) or a
  factor model (:doc:`mcnnm`) may fit better — plot the pre-period to decide.
* You want the broader DiD ecosystem — Callaway–Sant'Anna, Sun–Abraham,
  Wooldridge ETWFE, honest-DiD sensitivity, large-:math:`N` staggered
  estimators. mlsynth ships ``ROLLDID`` for the small-:math:`N` exact regime
  where it complements synthetic control; it does not aim to cover DiD
  comprehensively. For the full toolkit see the sibling package
  `diff-diff <https://github.com/igerber/diff-diff>`_.

Notation
--------

Let :math:`\mathcal{N} \coloneqq \{1, \dots, N\}` index the units and
:math:`t \in \mathcal{T} \coloneqq \{1, \dots, T\}` the periods (1-indexed). Unit
:math:`j` has observed outcome :math:`y_{jt}`, with potential outcomes
:math:`y_{jt}^{N}` absent the intervention and :math:`y_{jt}^{I}` under it; the
treatment dummy is :math:`d_{jt}`, so
:math:`y_{jt} = y_{jt}^{N} + (y_{jt}^{I} - y_{jt}^{N})\,d_{jt}`. Treatment is
absorbing: once on it stays on. The unit-level "ever-treated" indicator is

.. math::

   d_j \coloneqq \max_{t\in\mathcal{T}} d_{jt} = \mathbf{1}\{\text{unit } j
   \text{ is ever treated}\},

partitioning :math:`\mathcal{N}` into the eventually-treated
:math:`\mathcal{N}_1 \coloneqq \{j : d_j = 1\}` (size :math:`N_1`) and the
never-treated :math:`\mathcal{N}_0 \coloneqq \{j : d_j = 0\}` (size
:math:`N_0`), with :math:`N = N_0 + N_1 \ge 3`.

*Common timing.* All treated units adopt after period :math:`T_0`, splitting
time into the pre-period :math:`\mathcal{T}_1 \coloneqq \{t \le T_0\}` and the
post-period :math:`\mathcal{T}_2 \coloneqq \{t > T_0\}`.

*Staggered adoption.* Treated unit :math:`j` has a cohort
:math:`g_j \coloneqq \min\{t : d_{jt} = 1\}`; its pre-period is
:math:`\{t < g_j\}`. Cohorts are collected in
:math:`\mathcal{G} \coloneqq \{g_j : j \in \mathcal{N}_1\}` with sizes
:math:`N_g \coloneqq |\{j : g_j = g\}|`.

Transformed series carry a tilde: :math:`\widetilde{y}_{jt}` is the residualised
post-period outcome and :math:`\widetilde{y}_j` its collapse to a single scalar
per unit. The per-period effect is :math:`\tau_t` and the ATT is
:math:`\widehat{\tau}` (the ``gap`` and ``att`` of the result object).

Assumptions
-----------

Assumption 1 (no anticipation). Treated potential outcomes equal the
never-treated ones before adoption:
:math:`\mathbb{E}[y_{jt}^{I} - y_{jt}^{N} \mid d_j = 1] = 0` for all
:math:`t < g_j` (sufficient: :math:`y_{jt}^{I} = y_{jt}^{N}` for :math:`t < g_j`).

*Remark.* The pre-period mean / trend is estimated only on :math:`t < g_j`, so
anticipation in the final pre-periods would contaminate the baseline. Detrending
and shorter pre-windows are the sensitivity levers (the package exposes the
pre-window directly).

Assumption 2 (parallel trends after the rolling transformation). With the
within-unit transformed regressand :math:`\widetilde{y}_j(0)` formed from the
*untreated* potential outcomes (defined in the next section), there is a constant
:math:`\alpha` with

.. math::

   \mathbb{E}\!\left[\widetilde{y}_j(0) \mid d_j\right] = \alpha .

Two cases: (a) under demeaning, :math:`\widetilde{y}_j(0)` differences
out a unit-specific *level*, so :math:`\alpha` absorbs an arbitrary common
trend; (b) under detrending, it differences out a unit-specific *linear
trend*, so :math:`\alpha` permits unit-specific linear trends.

*Remark.* Case (b) is strictly weaker than standard parallel trends and is the
estimator's edge over SC/SDID when pre-trends are heterogeneous but linear:
treatment may be correlated with a unit's *level and slope* as long as it is
mean-independent of the post-minus-pre *deviation*.

Assumption 3 (classical linear model, for exact inference). In the
cross-sectional regression error :math:`u_j` (next section),

.. math::

   u_j \mid d_j \sim \mathcal{N}(0, \sigma^2)
   \quad\text{i.i.d. across } j .

*Remark.* Normality is plausible *despite small* :math:`N` because
:math:`\widetilde{y}_j` averages :math:`y_{jt}` over :math:`\mathcal{T}_2`: if the
series is weakly dependent over time, a central limit theorem across time
makes the average approximately normal. The leverage for inference comes from
:math:`T`, not :math:`N`. HC3 relaxes homoskedasticity (Assumption 3 then only
needs mean-independence) at the cost of requiring a handful of treated *and*
control units; randomization inference drops normality entirely.

Mathematical Formulation
------------------------

The rolling transformation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each unit's outcome is residualised against its own pre-treatment path and
then averaged over the post window. For cohort :math:`g` (in common timing,
:math:`g = T_0 + 1` for every treated unit), ``rolling="demean"`` (the paper's
Procedure 2.1) removes the pre-period mean,

.. math::

   \widetilde{y}_{jt} \coloneqq y_{jt} - \bar{y}_j^{\,\mathrm{pre}(g)},
   \qquad
   \bar{y}_j^{\,\mathrm{pre}(g)} \coloneqq \frac{1}{g-1}\sum_{t < g} y_{jt},
   \qquad t \ge g,

while ``rolling="detrend"`` (Procedure 3.1) fits a unit-specific line on the
pre-period, :math:`(\widehat{a}_j, \widehat{b}_j) =
\operatorname*{argmin}_{a,b}\sum_{t<g}(y_{jt} - a - b\,t)^2`, and removes its
projection into the post-period,

.. math::

   \widetilde{y}_{jt} \coloneqq y_{jt} - \bigl(\widehat{a}_j + \widehat{b}_j\,t\bigr),
   \qquad t \ge g .

The unit's single cross-sectional regressand is the post-average of its
transformed outcome,

.. math::

   \widetilde{y}_j \coloneqq \frac{1}{|\mathcal{T}_2|}\sum_{t\in\mathcal{T}_2}
   \widetilde{y}_{jt}
   \;=\; \bar{y}_j^{\,\mathrm{post}} - \bar{y}_j^{\,\mathrm{pre}}
   \quad\text{(demeaning)} .

The collapse equivalence
~~~~~~~~~~~~~~~~~~~~~~~~

The key algebraic fact (Lee & Wooldridge): the panel DiD coefficient equals the
slope of the cross-sectional regression of :math:`\widetilde{y}_j` on the
ever-treated indicator,

.. math::

   \widetilde{y}_j = \alpha + \tau\,d_j + u_j,
   \qquad j \in \mathcal{N},

so that, in closed form,

.. math::

   \widehat{\tau}
   = \frac{1}{N_1}\sum_{j\in\mathcal{N}_1}\widetilde{y}_j
   - \frac{1}{N_0}\sum_{j\in\mathcal{N}_0}\widetilde{y}_j
   = \overline{\widetilde{y}}_{\,\text{treated}}
   - \overline{\widetilde{y}}_{\,\text{control}} .

The transformation does two things at once: differencing post-minus-pre within
each unit cancels any time-constant heterogeneity (a unit fixed effect, even a
unit root), and collapsing time to a scalar pushes the serial correlation in
:math:`\{y_{jt}\}` *inside* :math:`\widetilde{y}_j`. Across units the
:math:`\widetilde{y}_j` are independent, so the object on which we do inference is
an ordinary cross-section — no clustering, no large-:math:`T` requirement, and
strong time-series dependence is *absorbed* rather than modelled.

Per-period effects (common timing)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rather than collapse, regress the transformed value at each post period on
:math:`d_j` to obtain the event study,

.. math::

   \widetilde{y}_{jt} = \alpha_t + \tau_t\,d_j + u_{jt},
   \qquad t \in \mathcal{T}_2 ,

with :math:`\tau_t` the coefficient on :math:`d_j`. Because OLS is linear and
:math:`\widetilde{y}_j = |\mathcal{T}_2|^{-1}\sum_{t}\widetilde{y}_{jt}`, the
aggregate is exactly the mean of the per-period effects,
:math:`\widehat{\tau} = |\mathcal{T}_2|^{-1}\sum_{t\in\mathcal{T}_2}\tau_t`.

Staggered aggregation
~~~~~~~~~~~~~~~~~~~~~

With several cohorts, transform each unit relative to every cohort date and
form, per cohort, :math:`\widetilde{y}_j^{(g)} \coloneqq (T-g+1)^{-1}\sum_{t\ge g}
\widetilde{y}_{jt}`. Using never-treated units as the comparison and cohort
shares :math:`\widehat{\omega}_g \coloneqq N_g / N_1`, the collapsed regressand is

.. math::

   \widetilde{y}_j \coloneqq
   \begin{cases}
     \widetilde{y}_j^{(g_j)}, & j \in \mathcal{N}_1 \ \text{(its own cohort)},\\[2mm]
     \displaystyle\sum_{g\in\mathcal{G}} \widehat{\omega}_g\,\widetilde{y}_j^{(g)},
       & j \in \mathcal{N}_0 \ \text{(never treated)} .
   \end{cases}

Running the single cross-sectional regression :math:`\widetilde{y}_j = \alpha +
\tau_\omega d_j + u_j` then returns the cohort-share-weighted aggregate
:math:`\widehat{\tau}_\omega` whose standard error automatically accounts for the
covariance across cohort effects (it is one regression, not a stitched-together
sum). Per-cohort effects :math:`\widehat{\tau}_g` are the analogous cohort-:math:`g`
vs never-treated contrasts. Using only never-treated comparisons sidesteps the
"forbidden comparison" / negative-weighting pathologies of two-way fixed effects
under staggered adoption.

Inference
---------

All three modes operate on the cross-sectional regression
:math:`\widetilde{y}_j = \alpha + \tau d_j + u_j` (or its per-period /
per-cohort variants), with residual degrees of freedom :math:`N - 2`.

* ``inference="exact"`` — under Assumption 3 the studentised statistic is
  exactly :math:`t`-distributed,

  .. math::

     \frac{\widehat{\tau} - \tau}{\operatorname{se}(\widehat{\tau})}
     \;\sim\; t_{N-2},

  giving exact two-sided tests and exact-coverage intervals
  :math:`\widehat{\tau} \pm t_{1-\alpha/2,\,N-2}\,\operatorname{se}(\widehat{\tau})`,
  with any :math:`N \ge 3`, including :math:`N_1 = 1`. (With a single
  treated unit this is the studentised-residual / outlier statistic of Donald &
  Lang.)
* ``inference="hc3"`` — heteroskedasticity-robust (MacKinnon–White HC3). Use only
  with a handful of treated and control units: with a single treated unit its
  leverage is :math:`1` and HC3 is undefined, so ``ROLLDID`` raises rather
  than returning a degenerate standard error.
* ``inference="ri"`` — randomization inference: re-assign the :math:`N_1` treated
  labels across units ``ri_reps`` times and report the permutation
  :math:`p`-value :math:`\Pr(|\widehat{\tau}^{\,\text{perm}}| \ge
  |\widehat{\tau}|)`, requiring no normality.

The event study (common timing) and the per-cohort table (staggered) carry the
same effect-size-level uncertainty: each :math:`\tau_t` (resp.
:math:`\widehat{\tau}_g`) ships with its own ``se``, ``p_value`` and
:math:`(1-\alpha)` confidence band, which is what ``plot_rolldid`` renders around
the effect path.

Why it improves on SC / SDID
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The synthetic-control family needs the treated unit inside the convex hull of the
donors and, for inference, large :math:`N_0, T_0, T_1` (SDID additionally assumes
:math:`I(0)` weak dependence and normality). ROLLDID needs none of those
dimensions to grow: the collapse buys exact finite-sample inference from large
:math:`T` alone, and detrending *weakens* parallel trends rather than requiring a
donor combination to exist. The trade-off, which the paper is explicit about, is
efficiency: the cross-sectional estimator can have larger variance than SDID
when SDID's assumptions hold — but SDID's packaged intervals can under-cover,
whereas ROLLDID's are exact. It is best read as a complement: a transparent,
weight-free tool that is the more trustworthy of the two precisely when units or
periods are scarce, or when the per-cohort weight optimisation of staggered SDID
becomes ill-posed.

Example: California Proposition 99
----------------------------------

Reproduces the paper's Table 3 (common timing, single treated unit) on the
bundled Abadie–Diamond–Hainmueller smoking panel. The outcome is log
per-capita cigarette sales; California is treated from 1989 against 38
never-treated states.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from mlsynth import ROLLDID

   BASE = "https://raw.githubusercontent.com/jgreathouse9/mlsynth/main/basedata/"
   df = pd.read_csv(BASE + "smoking_data.csv")          # 39 states x 1970-2000
   df["logcig"] = np.log(df["cigsale"])
   df["treat"] = df["Proposition 99"].astype(int)       # W_jt: California from 1989

   res = ROLLDID({
       "df": df, "outcome": "logcig", "treat": "treat",
       "unitid": "state", "time": "year",
       "rolling": "detrend", "inference": "exact", "display_graphs": False,
   }).fit()

   print(res.effects.att, res.inference.standard_error, res.inference.p_value)
   #  -0.227   0.094   0.021     (paper Table 3)
   print(res.per_period[["time", "att", "ci_lower", "ci_upper"]].tail(3))
   #  tau_2000 = -0.403, 95% CI [-0.712, -0.094]

Staggered: castle-doctrine laws
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The bundled ``castle.csv`` panel (50 states, 2000–2010; 21 staggered adopters,
29 never-treated) gives the cohort-share-weighted aggregate via the same call,
with the treatment indicator turning on at each state's adoption year. Then
``res.effects.att`` is :math:`\widehat{\tau}_\omega` (``0.092`` demeaning /
``0.067`` detrending, matching §7.2) and ``res.per_cohort`` is the per-cohort
breakdown with its own confidence intervals.

Verification
------------

Path A, both empirical applications reproduced to the reported precision and
cross-validated (during development) against the AGPL ``lwdid`` package used only
as a black-box oracle — clean-room, sharing no code. California Prop 99
(Table 3): demeaning ATT :math:`-0.422` (se :math:`0.121`), detrending
:math:`-0.227` (se :math:`0.094`), detrend exact :math:`p = 0.021`,
:math:`\tau_{2000} = -0.667` / :math:`-0.403`. Castle laws (§7.2, staggered):
demeaning aggregate :math:`0.092` (se :math:`0.057`), detrending :math:`0.067`
(HC3 se :math:`0.055`). See :doc:`replications/rolldid`; the durable case is
``benchmarks/cases/rolldid_lw.py`` and unit-level reproduction is pinned in
``mlsynth/tests/test_rolldid.py``.

.. [LW2026] Lee, S. J., & Wooldridge, J. M. (2026). Simple Approaches to
   Inference with Difference-in-Differences Estimators with Small
   Cross-Sectional Sample Sizes.

Core API
--------

.. autoclass:: mlsynth.ROLLDID
   :members: fit

Configuration
-------------

.. autoclass:: mlsynth.utils.rolldid_helpers.config.ROLLDIDConfig
   :members:

Result Containers
-----------------

``ROLLDID.fit()`` returns a
:class:`~mlsynth.utils.rolldid_helpers.structures.ROLLDIDResults`, a standardized
:class:`~mlsynth.config_models.BaseEstimatorResults`: the aggregated ATT is
``res.effects.att``; ``res.inference`` carries the standard error, :math:`p`-value,
confidence interval and ``method``; ``res.time_series`` is the event-study path
(common timing). The rolling-DiD specifics sit alongside — ``res.transformation``
(``demean`` / ``detrend``), ``res.inference_type``, ``res.design`` (``common`` /
``staggered``), ``res.n_treated`` / ``res.n_control``, and the per-period
(``res.per_period``) or per-cohort (``res.per_cohort``) effect tables, each with
its own ``se`` / ``p_value`` / ``ci_lower`` / ``ci_upper``.

.. autoclass:: mlsynth.utils.rolldid_helpers.structures.ROLLDIDResults
   :members:

Helper Modules
--------------

Panel ingestion: resolves the long panel into per-unit series, the treatment
cohorts, and the never-treated set; checks the absorbing-treatment and
never-treated conditions.

.. automodule:: mlsynth.utils.rolldid_helpers.setup
   :members:

The rolling transformations, the cross-sectional estimator (aggregate,
per-period, per-cohort), and the exact / HC3 / randomization inference.

.. automodule:: mlsynth.utils.rolldid_helpers.pipeline
   :members:

Event-study / effect plot.

.. automodule:: mlsynth.utils.rolldid_helpers.plotter
   :members:
