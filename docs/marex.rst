Synthetic Controls for Experimental Design (MAREX)
==================================================

.. currentmodule:: mlsynth

When to Use This Estimator
--------------------------

The estimators elsewhere in ``mlsynth`` are *retrospective*: a treatment has
already happened and you reweight donors to reconstruct the treated unit's
counterfactual. **MAREX**, due to Abadie and Zhao (2026) [ABADIE2024]_, is
*prospective* — it **designs** an experiment. Before any treatment is assigned,
and using only pre-experimental data, it chooses **which aggregate units to
treat** and **which to hold out as controls**, so that the experiment you are
about to run yields a credible estimate.

The motivating setting is a firm (say a ride-sharing company) that wants to test
a new policy but can only deploy it in a *few* whole markets. A within-market
A/B test is contaminated by interference (treated and control drivers compete);
randomizing whole markets to treatment is unbiased *ex ante* but, with a handful
of large units, routinely produces treated and control groups with very
different baselines, so any single realisation is badly off. MAREX instead picks
the treated and control markets so their pre-experiment predictors match the
population — a non-randomized design that, the paper shows, substantially
reduces estimation bias relative to randomization.

Reach for MAREX when:

* **Units are large aggregates** (markets, regions, stores) and only one or a
  few can be treated.
* **You control the assignment** and want to choose it well, rather than
  estimate after the fact.
* **Interference or equity rules out within-unit randomization**, forcing
  whole-unit treatment.

Notation
--------

There are :math:`J` units and :math:`T` periods, with :math:`T_0` pre-experiment
periods; the experiment runs over :math:`t = T_0 + 1, \dots, T`. Each unit has a
pre-intervention predictor vector :math:`X_j` (pre-period outcomes and optional
covariates); :math:`\bar X = \sum_j f_j X_j` is the population predictor mean
for known weights :math:`f_j` (e.g. market shares, or :math:`1/J`). The
experimenter chooses **treated weights** :math:`w` and **control weights**
:math:`v`, both on the simplex, and *disjoint*:

.. math::

   \sum_j w_j = 1,\quad \sum_j v_j = 1,\quad w_j, v_j \ge 0,\quad w_j v_j = 0
   \;\;\forall j.

Units with :math:`w_j > 0` are **treated**; among the rest, units with
:math:`v_j > 0` form the **synthetic control**. Writing :math:`Y_{jt}` for the
observed outcome (treated units realise :math:`Y^I_{jt}` post-treatment,
everyone else :math:`Y^N_{jt}`), the design estimator of the average effect is

.. math::

   \hat\tau_t(w, v) = \sum_j w_j Y_{jt} - \sum_j v_j Y_{jt},
   \qquad t > T_0.

Assumptions
-----------

**Assumption 1 (linear factor model).** Potential outcomes follow

.. math::

   Y^N_{jt} = \delta_t + \theta_t' Z_j + \lambda_t' \mu_j + \varepsilon_{jt},
   \qquad
   Y^I_{jt} = \upsilon_t + \gamma_t' Z_j + \eta_t' \mu_j + \xi_{jt},

with observed covariates :math:`Z_j`, unobserved factors :math:`\mu_j`, and
mean-zero idiosyncratic noise.

*Remark.* This is the interactive-fixed-effects model of the SC literature
(Abadie-Diamond-Hainmueller 2010), extended with a *separate* factor structure
for the treated potential outcome — necessary because a design must choose a
treatment group, not just a comparison group.

**Assumption 2 (regularity).** The factor loadings are non-degenerate
(:math:`F \le T_E`, smallest eigenvalue bounded below) and the noise is
i.i.d. sub-Gaussian with common variance, independent across the two potential
outcomes; dependence *across units* is allowed.

**Assumption 3 / 4 (fit quality).** A weight vector reproducing the population
predictor means exists exactly (Assumption 3), or approximately within a
tolerance :math:`d` (Assumption 4). This is the design-time analogue of "the
treated unit lies in the convex hull of the donors."

*Remark.* Under these conditions Abadie & Zhao bound the bias of
:math:`\hat\tau_t(w, v)` and develop the permutation test below; the better the
pre-experiment match, the smaller the bias.

Mathematical Formulation
------------------------

The Design Optimization
^^^^^^^^^^^^^^^^^^^^^^^

MAREX chooses :math:`w, v` (and a binary selection mask :math:`z`, with
:math:`w_j \le z_j`, :math:`v_j \le 1 - z_j`, so a unit is treated *or* a
control, never both) to match the population predictor mean. The ``base`` design
minimises

.. math::

   \min_{w, v, z}\;
   \Bigl\| \bar X - \sum_j w_j X_j \Bigr\|_2^2
   + \Bigl\| \bar X - \sum_j v_j X_j \Bigr\|_2^2
   \quad \text{s.t. the simplex / disjointness / cardinality constraints,}

with the number of treated units pinned by ``m_eq`` (exactly) or bounded by
``m_min``/``m_max``. This is a **mixed-integer quadratic program** (the binary
``z``); ``mlsynth`` solves it with SCIP by default, or — via ``relaxed=True`` —
relaxes ``z`` to :math:`[0, 1]`, solves the QP, and discretises post hoc.

``mlsynth`` exposes four objective variants through ``design`` (clear names
that map to the paper's formulations):

* ``"standard"`` — match each predictor mean with both synthetic units
  (formulation 5);
* ``"weakly_targeted"`` — match the treated synthetic to the mean and softly tie
  the control synthetic to it (weight ``beta``);
* ``"penalized"`` — ``standard`` plus a distance penalty that down-weights
  units far from the population mean (``lambda1`` / ``lambda2``);
* ``"unit_penalized"`` — ``standard`` plus unit-level penalties
  (``lambda1_unit`` / ``lambda2_unit``).

Covariates
^^^^^^^^^^

By default the design matches on pre-period outcomes. Passing ``covariates``
(time-invariant column names) appends them to the predictor vector,
:math:`X_j = [Y^E_j ; Z_j]`, exactly as in the paper — the synthetic treated and
control are then balanced on both pre-period outcomes *and* covariates (with an
optional ``covariate_weight`` scale). When the pre-period is long, the outcomes
already encode the covariates' contribution, so covariates matter most when few
pre-periods are available.

Clustering, Costs, and Budgets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Passing a ``cluster`` column solves the design *within* each cluster (one or a
few treated units per cluster), which better approximates the population
predictor distribution and limits interpolation bias (paper OA.1). Per-unit
``costs`` and a ``budget`` (scalar or per-cluster) add a knapsack constraint
:math:`\sum_j c_j w_j \le B`, so the chosen treatment group respects a spend cap.

Inference
^^^^^^^^^

When ``inference=True`` with ``blank_periods > 0``, the last few pre-experiment
periods are held out as **blanks**: there the synthetic treated minus synthetic
control is pure noise, so its distribution calibrates inference for the
post-period effect. MAREX reports a permutation p-value for the global null of
no effect, per-period p-values, and a split-conformal confidence band
(Chernozhukov-Wuthrich-Zhu 2021), all on
:class:`~mlsynth.utils.marex_helpers.structures.MAREXInference`.

Monte Carlo: Recovering the Treatment Effect
--------------------------------------------

The block below replicates the qualitative finding of the paper's simulation
study (Section 5) using ``mlsynth``'s own reimplementation of the linear-factor
DGP. A sample is drawn, the design is fit on the pre-period, the treated units
realise :math:`Y^I` in the experiment, and the estimate is compared to the true
average effect.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from mlsynth import MAREX
   from mlsynth.utils.marex_helpers.simulation import generate_marex_sample

   rng = np.random.default_rng(0)

   def design_mae(sample, **card):
       J, T = sample.Y_N.shape
       T0 = sample.T0
       df = pd.DataFrame(
           [{"unit": f"u{j}", "time": t, "y": float(sample.Y_N[j, t])}
            for j in range(J) for t in range(T)]
       )
       res = MAREX({"df": df, "outcome": "y", "unitid": "unit",
                    "time": "time", "T0": T0, **card}).fit()
       w = res.globres.treated_weights_agg
       v = res.globres.control_weights_agg
       treated = np.where(w > 1e-8)[0]
       Y_obs = sample.Y_N.copy()
       Y_obs[treated, T0:] = sample.Y_I[treated, T0:]   # experiment realises Y^I
       tau_hat = w @ Y_obs[:, T0:] - v @ Y_obs[:, T0:]
       return np.mean(np.abs(tau_hat - sample.tau[T0:]))

   maes, scales = [], []
   for _ in range(5):
       s = generate_marex_sample(J=12, T=30, T0=25, rng=rng)
       maes.append(design_mae(s, m_min=1, m_max=11))     # Unconstrained
       scales.append(np.mean(np.abs(s.tau[s.T0:])))
   print(f"MAE {np.mean(maes):.2f}  vs  effect scale {np.mean(scales):.2f}")

The synthetic-control design recovers the average treatment effect with mean
absolute error far below the effect's own scale (≈ 4.4 vs. ≈ 14, i.e. under a
third), the central message of the paper's Table 2. Over the paper's full 1000
simulations the error also *decreases* as more units are allowed into the
treated group (the Unconstrained design is best), with the largest gains moving
from one to two or three treated units.

.. note::

   This is a **Path-B replication**: it reproduces the simulation study's
   conclusions from public DGPs and ``mlsynth`` code, with no dependency on the
   authors' replication package. It is locked in as
   :mod:`mlsynth.tests.test_marex_replication`.

Empirical Application: Walmart (Placebo Experiment)
---------------------------------------------------

We replicate the paper's empirical illustration (Section 4) on the Walmart
store-sales panel (``basedata/walmart_weekly_sales.csv``): weekly sales for
**45 stores over 143 weeks** (Feb 2010 – Oct 2012). Following the paper, we
design a **placebo** experiment with a fictitious intervention at week 129:
:math:`T_0 = 128` pre-experiment weeks, of which the first :math:`T_E = 100` are
the fitting period and the last 28 are blank, leaving 15 experimental weeks. The
design uses the constrained formulation with :math:`m = 2` treated stores,
uniform weights, and predictors normalised to unit variance (``standardize``).

.. code-block:: python

   import pandas as pd
   from mlsynth import MAREX

   df = pd.read_csv(
       "https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
       "refs/heads/main/basedata/walmart_weekly_sales.csv"
   )

   res = MAREX({
       "df": df, "outcome": "sales", "unitid": "store", "time": "week",
       "T0": 128, "blank_periods": 28, "T_post": 15,   # TE=100, 28 blank, 15 post
       "m_eq": 2,                  # constrained design, two treated stores
       "design": "standard",
       "standardize": True,        # unit-variance predictors (paper's normalisation)
       "inference": True,
       "display_graph": True,
   }).fit()

   print("treated stores:", res.treated_units)              # [1, 15]
   print("placebo p-value:", round(res.globres.inference.global_p_value, 3))

Because the intervention is a placebo (no real effect), a correct design should
produce synthetic treated and control units that track closely and an estimated
effect near zero. ``mlsynth`` reproduces exactly that — and the paper's headline
number:

.. list-table:: Walmart placebo design (m = 2)
   :header-rows: 1
   :widths: 38 30 30

   * - Quantity
     - ``mlsynth``
     - Paper (Section 4)
   * - Pre-fit RMSE / mean sales
     - **2.2%**
     - small (close tracking)
   * - Experimental ATT / mean sales
     - **-1.0%**
     - near zero
   * - Placebo permutation p-value
     - **0.937**
     - **0.933**
   * - Confidence band covers zero
     - yes (all post weeks)
     - yes

The synthetic treated and control units track to within ~2% of mean sales over
the fitting *and* blank periods, the estimated placebo effect is ~1% of sales,
and the permutation test fails to reject the null of no effect
(:math:`p = 0.937`, matching the paper's :math:`0.933`) — exactly the
"no spurious effect" result a good design should deliver on a placebo.

.. note::

   This uses the **exact** MIQP (``relaxed=False``, the default) with
   ``standardize=True``; the unit-variance normalisation is essential here
   because Walmart stores differ enormously in sales level, and without it the
   level differences dominate the match. The solve takes roughly a minute with
   the open-source SCIP solver (the paper used commercial Gurobi).

Correspondence with the Authors' Code
-------------------------------------

The authors' R replication code (``Random_Data_Generator.R``,
``Synthetic_Experiments.R``, ``Different_optimization_methods.R``) maps directly
onto ``mlsynth``'s implementation, which was checked against it:

.. list-table:: Authors' R ↔ ``mlsynth`` MAREX
   :header-rows: 1
   :widths: 42 42

   * - Authors' R
     - ``mlsynth``
   * - DGP (``Random_Data_Generator.R``)
     - :func:`~mlsynth.utils.marex_helpers.simulation.generate_marex_sample`
   * - Formulation (5), Gurobi non-convex QCQP
     - ``design="standard"`` (MIQP with binary ``z``; same optimum)
   * - Penalization formulation
     - ``design="penalized"`` (identical :math:`\lambda` distance penalty)
   * - Cardinality formulation
     - ``m_eq`` / ``m_min`` / ``m_max``
   * - Predictors :math:`X = [Y^E ; Z]`
     - ``covariates=[...]`` (matched on pre-outcomes + covariates)
   * - "treated = smaller set" swap
     - applied in :func:`~mlsynth.utils.marex_helpers.orchestration.solve_marex`
   * - Exact permutation test (sum statistic)
     - permutation inference (mlsynth defaults to a mean statistic / sampled
       permutations)

Driving ``mlsynth``'s ``solve_design`` on the authors' exact DGP and predictor
matrix recovers the average treatment effect to within its scale, and the effect
estimate degrades gracefully as the noise SD rises from 1 to 5 to 10 (the
figures 2-7 settings) — matching the paper's qualitative findings.

.. note::

   Two faithfulness details from their R code: ``rnorm(N, 0, noise.variance)``
   passes the value as a *standard deviation*, so the figures' "variance"
   1/5/10 are SDs; and the R code uses random population weights :math:`f_j`
   whereas the 2026 paper (and ``mlsynth``) use :math:`f_j = 1/J`. Also note that
   the *unconstrained* ``standard`` design (formulation 5) is **degenerate** —
   many disjoint splits match :math:`\bar X` equally well, so the realised
   design (and hence a single ATE estimate) is solver-dependent; the
   cardinality-constrained design is the stable, recommended choice.

Example
-------

.. code-block:: python

   import pandas as pd
   from mlsynth import MAREX

   # long panel: one row per (market, period)
   res = MAREX({
       "df": df, "outcome": "revenue", "unitid": "market", "time": "week",
       "T0": 40,            # 40 pre-experiment weeks
       "m_eq": 2,           # treat exactly two markets
       "design": "standard",
       "inference": True, "blank_periods": 4, "T_post": 4,
       "display_graph": True,
   }).fit()

   print("treated markets:", res.treated_units)
   print("global p-value:", res.globres.inference.global_p_value)
   for label, c in res.clusters.items():
       print(label, c.unit_weight_map["Treated"])

Core API
--------

.. automodule:: mlsynth.estimators.scexp
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.MAREXConfig
   :members:
   :undoc-members:

Result Containers
-----------------

``MAREX.fit()`` returns a
:class:`~mlsynth.utils.marex_helpers.structures.MAREXResults`: a dict of
per-cluster :class:`~mlsynth.utils.marex_helpers.structures.MAREXClusterDesign`
objects, the aggregate
:class:`~mlsynth.utils.marex_helpers.structures.MAREXGlobalDesign`, the
:class:`~mlsynth.utils.marex_helpers.structures.MAREXStudy` hyperparameters, and
(optionally) :class:`~mlsynth.utils.marex_helpers.structures.MAREXInference`.

.. automodule:: mlsynth.utils.marex_helpers.structures
   :members:
   :undoc-members:
   :show-inheritance:

Helper Modules
--------------

.. automodule:: mlsynth.utils.marex_helpers.setup
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.marex_helpers.formulation
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.marex_helpers.optimization
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.marex_helpers.inference
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.marex_helpers.orchestration
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.marex_helpers.simulation
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.marex_helpers.plotter
   :members:
   :undoc-members:

References
----------

Abadie, A., & Zhao, J. (2026). "Synthetic Controls for Experimental Design."
See [ABADIE2024]_.

Abadie, A., Diamond, A., & Hainmueller, J. (2010). "Synthetic Control Methods
for Comparative Case Studies." *Journal of the American Statistical
Association* 105(490):493-505.

Chernozhukov, V., Wuthrich, K., & Zhu, Y. (2021). "An Exact and Robust
Conformal Inference Method for Counterfactual and Synthetic Controls."
*Journal of the American Statistical Association* 116(536):1849-1864.
