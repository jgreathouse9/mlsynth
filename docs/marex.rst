Synthetic Controls for Experimental Design (MAREX)
==================================================

.. currentmodule:: mlsynth

When to Use This Estimator
--------------------------

The estimators elsewhere in ``mlsynth`` are *retrospective*: a treatment has
already happened and you reweight donors to reconstruct the treated unit's
counterfactual. MAREX, due to Abadie and Zhao (2026) [ABADIE2024]_, is
*prospective* — it designs an experiment. Before any treatment is assigned,
and using only pre-experimental data, it chooses which aggregate units to
treat and which to hold out as controls, so that the experiment you are
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

* Units are large aggregates (markets, regions, stores) and only one or a
  few can be treated.
* You control the assignment and want to choose it well, rather than
  estimate after the fact.
* Interference or equity rules out within-unit randomization, forcing
  whole-unit treatment.

Notation
--------

There are :math:`N` units :math:`\mathcal{N} \coloneqq \{1, \dots, N\}` and
:math:`T` periods :math:`t \in \mathcal{T} \coloneqq \{1, \dots, T\}`,
1-indexed; the experiment takes effect after period :math:`T_0`, splitting
:math:`\mathcal{T}` into the pre-experiment window
:math:`\mathcal{T}_1 \coloneqq \{t \in \mathcal{T} : t \le T_0\}` (of length
:math:`T_0`) and the experimental window
:math:`\mathcal{T}_2 \coloneqq \{t \in \mathcal{T} : t > T_0\}`. Because MAREX
designs an experiment rather than reweighting around one already-treated unit,
units are indexed generically by :math:`i, j \in \mathcal{N}` with no forced
treated unit. Each unit has a pre-intervention predictor vector
:math:`\mathbf{x}_j` (pre-period outcomes and optional covariates);
:math:`\bar{\mathbf{x}} = \sum_j f_j \mathbf{x}_j` is the population predictor
mean for known weights :math:`f_j` (e.g. market shares, or :math:`1/N`). The
experimenter chooses treated weights :math:`\mathbf{w}` and control weights
:math:`\mathbf{v}`, both on the simplex, and *disjoint*:

.. math::

   \sum_j w_j = 1,\quad \sum_j v_j = 1,\quad w_j, v_j \ge 0,\quad w_j v_j = 0
   \;\;\forall j.

Units with :math:`w_j > 0` are treated; among the rest, units with
:math:`v_j > 0` form the synthetic control. Writing :math:`y_{jt}` for the
observed outcome (treated units realise :math:`y_{jt}^I` over
:math:`\mathcal{T}_2`, everyone else :math:`y_{jt}^N`), the design estimator of
the average effect is

.. math::

   \widehat{\tau}_t(\mathbf{w}, \mathbf{v}) = \sum_j w_j y_{jt} - \sum_j v_j y_{jt},
   \qquad t \in \mathcal{T}_2.

Assumptions
-----------

Assumption 1 (linear factor model). Potential outcomes follow

.. math::

   y_{jt}^N = \delta_t + \boldsymbol{\theta}_t^\top \mathbf{z}_j
   + \boldsymbol{\lambda}_t^\top \boldsymbol{\mu}_j + \varepsilon_{jt},
   \qquad
   y_{jt}^I = \upsilon_t + \boldsymbol{\gamma}_t^\top \mathbf{z}_j
   + \boldsymbol{\eta}_t^\top \boldsymbol{\mu}_j + \xi_{jt},

with observed covariates :math:`\mathbf{z}_j`, unobserved factors
:math:`\boldsymbol{\mu}_j`, and mean-zero idiosyncratic noise.

*Remark.* This is the interactive-fixed-effects model of the SC literature
(Abadie-Diamond-Hainmueller 2010), extended with a *separate* factor structure
for the treated potential outcome — necessary because a design must choose a
treatment group, not just a comparison group.

Assumption 2 (regularity). The factor loadings are non-degenerate
(:math:`F \le T_E`, smallest eigenvalue bounded below) and the noise is
i.i.d. sub-Gaussian with common variance, independent across the two potential
outcomes; dependence *across units* is allowed.

*Remark.* These conditions keep the factor structure recoverable from the
pre-experiment window and the noise well-behaved, so the population predictor
mean :math:`\bar{\mathbf{x}}` is a meaningful matching target rather than an
artifact of a degenerate loading matrix.

Assumption 3 / 4 (fit quality). A weight vector reproducing the population
predictor means exists exactly (Assumption 3), or approximately within a
tolerance :math:`d` (Assumption 4). This is the design-time analogue of "the
treated unit lies in the convex hull of the donors."

*Remark.* Under these conditions Abadie & Zhao bound the bias of
:math:`\widehat{\tau}_t(\mathbf{w}, \mathbf{v})` and develop the permutation test
below; the better the pre-experiment match, the smaller the bias.

Mathematical Formulation
------------------------

The Design Optimization
^^^^^^^^^^^^^^^^^^^^^^^

MAREX chooses :math:`\mathbf{w}, \mathbf{v}` (and a binary selection mask
:math:`\mathbf{z}`, with :math:`w_j \le z_j`, :math:`v_j \le 1 - z_j`, so a unit
is treated *or* a control, never both) to match the population predictor mean.
The ``base`` design minimises

.. math::

   \min_{\mathbf{w}, \mathbf{v}, \mathbf{z}}\;
   \Bigl\| \bar{\mathbf{x}} - \sum_j w_j \mathbf{x}_j \Bigr\|_2^2
   + \Bigl\| \bar{\mathbf{x}} - \sum_j v_j \mathbf{x}_j \Bigr\|_2^2
   \quad \text{s.t. the simplex / disjointness / cardinality constraints,}

with the number of treated units pinned by ``m_eq`` (exactly) or bounded by
``m_min``/``m_max``. This is a mixed-integer quadratic program (the binary
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
:math:`\mathbf{x}_j = [\mathbf{y}^E_j ; \mathbf{z}_j]`, exactly as in the paper — the synthetic treated and
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

Geographic design restrictions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MAREX already carries most geographic structure natively. Its ``cluster`` field
*is* the paper's "region as a distinct experimental design" (p.13: weather
patterns dependent across cities in the same region motivate treating each region
separately); ``m_min`` / ``m_max`` per cluster is a stratum quota ("at least /
at most this many treated per region"); ``costs`` / ``budget`` is the cost bound;
and because each cluster's control synthetic :math:`v_{\cdot,k}` is built only
from cluster-``k`` members, donors are automatically same-region. The same
restriction vocabulary SYNDES and GEOLIFT expose then adds the rest, as
constraints on the MIP:

* ``to_be_treated`` / ``not_to_be_treated`` -- force a market in (one leadership
  already committed to) or out (a regulated market, or one already under another
  test); a forbidden market stays a donor.
* ``adjacency`` + ``spillover_threshold`` -- no two treated markets may share a
  border (:math:`\sum_k z_{ik} + \sum_k z_{jk} \le 1`), for when one test
  market's campaign would bleed into a neighbouring one and bias the lift.
* ``size_col`` + ``min_size`` / ``max_size`` -- a treated-unit size band (the
  floor is a power minimum, the ceiling is synthesizability); out-of-band markets
  stay donors.
* ``exclude_bordering_donors`` -- drop a treated market's bordering neighbours
  from its (within-cluster) control pool, so a market partly treated by spillover
  never sits in its counterfactual. Requires ``adjacency``.

These are supported in the exact MIQP only (not ``relaxed=True``, whose post-hoc
rounding cannot guarantee them). An over-constrained design -- e.g. forbidding
every market of a region, leaving its required treated synthetic with no member
-- raises a translated :class:`~mlsynth.exceptions.MlsynthEstimationError` naming
the restrictions. The example below runs on the bundled real US DMA contiguity
map and Census regions, with a region-grouped linear factor sales model (after
Liao, Shi & Zheng [RelaxSC]_); the geography is real, the sales reproducible.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from mlsynth import MAREX

   base = ("https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
           "refs/heads/main/basedata/markets")
   adj = pd.read_csv(f"{base}/dma_adjacency.csv", index_col=0)
   meta = pd.read_csv(f"{base}/dma_metadata.csv")
   CDC = {**{s: "Midwest" for s in ["OH", "IN", "MI", "KY"]},
          **{s: "South" for s in ["WV", "VA", "TN", "GA", "FL", "SC"]}}
   meta = meta[meta["state"].isin(CDC)].copy()
   meta["region"] = meta["state"].map(CDC)
   names = [n for n in meta["dma_name"] if n in adj.index]
   meta = meta[meta["dma_name"].isin(names)].reset_index(drop=True)
   adj = adj.loc[names, names]

   rng = np.random.default_rng(7)
   reg = meta["region"].to_numpy(); n, r, T, T0 = len(names), 3, 30, 24
   Lam = (np.array([{g: rng.normal(size=r) for g in sorted(set(reg))}[g] for g in reg])
          + 0.15 * rng.normal(size=(n, r)))
   F = np.cumsum(rng.normal(size=(T, r)), axis=0)
   pop = np.round(rng.lognormal(12.5, 0.7, n)).astype(int)
   Y = 100 + rng.normal(0, 5, n) + F @ Lam.T + rng.normal(0, 1, (T, n))
   df = pd.DataFrame([{"dma": names[j], "week": t, "sales": float(Y[t, j]),
                       "region": reg[j], "population": int(pop[j])}
                      for j in range(n) for t in range(T)])

   B = dict(df=df, outcome="sales", unitid="dma", time="week",
            cluster="region", T0=T0)                    # cluster = Census region

   MAREX({**B, "m_max": 2}).fit()                       # native quota: <=2 per region
   MAREX({**B, "m_max": 2, "to_be_treated": ["Atlanta, GA"],
          "not_to_be_treated": ["Miami-Fort Lauderdale, FL"]}).fit()
   MAREX({**B, "m_max": 2, "adjacency": adj,
          "spillover_threshold": 0.5}).fit()            # no two treated border
   MAREX({**B, "m_max": 2, "size_col": "population",
          "min_size": 200_000, "max_size": 3_000_000}).fit()
   MAREX({**B, "m_max": 1, "adjacency": adj, "spillover_threshold": 0.5,
          "exclude_bordering_donors": True}).fit()      # non-bordering donors

Across groups: cluster vs. quotas vs. restrictions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Three mechanisms touch on grouping/geography; they are distinct, not
interchangeable, and ``cluster`` is *not* made obsolete by the restrictions:

* A *separate experiment per group* -- each group's own representativeness
  target and donor pool: ``cluster``. This is the design's objective, not a
  constraint: each cluster ``k`` reconstructs its own predictor mean
  :math:`\overline{X}_k`. It is the SYNDES analog of that estimator's ``arm``
  (:doc:`syndes`), but baked into the objective rather than run as separate
  solves -- which is why, in MAREX, the restrictions compose *with* ``cluster``
  (they apply within each cluster) and the per-cluster cardinality ``m_min`` /
  ``m_max`` *is* the stratum quota.
* *One* design with geographic / forcing limits within those clusters: the
  restriction suite above (force in/out, border conflict, size band, donor
  exclusion).

So a stratum quota is not an alternative to ``cluster``; in MAREX it is a
property *of* the clustering. Drop ``cluster`` (a single global cluster) and you
recover one design against the whole-population mean -- a different estimand, not
a constrained version of the clustered one.

Inference
^^^^^^^^^

When ``inference=True`` with ``blank_periods > 0``, the last few pre-experiment
periods are held out as blanks: there the synthetic treated minus synthetic
control is pure noise, so its distribution calibrates inference for the
post-period effect. MAREX reports a permutation p-value for the global null of
no effect, per-period p-values, and a split-conformal confidence band
(Chernozhukov-Wuthrich-Zhu 2021), all on
:class:`~mlsynth.utils.marex_helpers.structures.MAREXInference`.

Standardized Post-Fit and Power Analysis
----------------------------------------

Every call to :meth:`MAREX.fit` attaches a
:class:`~mlsynth.utils.post_fit.SyntheticControlPostFit` to ``res.post_fit``.
This is the single, estimator-agnostic surface for the diagnostic numbers a
consumer of the design typically needs: effects, fit RMSEs, conformal /
permutation inference, covariate balance (when covariates were used), and
power analysis. It is computed by
:func:`~mlsynth.utils.post_fit.compute_post_fit` from MAREX's own
``synthetic_treated`` / ``synthetic_control`` trajectories and weight vectors,
so by construction it agrees with what the underlying optimization produced.

.. code-block:: python

   pf = res.post_fit                          # SyntheticControlPostFit
   pf.ate, pf.ate_percent, pf.total_effect    # treatment-effect scalars
   pf.rmse_fit, pf.rmse_blank, pf.rmse_post   # fit quality, per phase
   pf.p_value, pf.ci_lower, pf.ci_upper       # inference (when computed)
   pf.covariate_smd                           # treated-vs-control SMD dict
   pf.covariate_smd_treated_vs_pop            # treated-vs-population
   pf.covariate_smd_control_vs_pop            # control-vs-population
   pf.power                                   # PowerAnalysis (see below)

Three Standardized Mean Differences
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When ``covariates=[...]`` is set, the post-fit reports the three covariate
balance diagnostics that match the structure of Abadie & Zhao's objective.
Each is a per-covariate signed dict (``covariate_smd_*``) plus two summary
scalars (max absolute SMD, sum of squared SMDs). With :math:`\bar{\mathbf{x}}`
the population covariate aggregate,
:math:`\mathbf{x}_w \coloneqq \sum_j w_j \mathbf{x}_j`,
:math:`\mathbf{x}_v \coloneqq \sum_j v_j \mathbf{x}_j`, and :math:`s_m` the
cross-unit standard deviation of covariate :math:`m`, each comparison is the
unit-free vector

.. math::

   \mathrm{SMD}_m^{(a,b)} = \frac{x_a[m] - x_b[m]}{s_m}.

The three pairs ``(a, b)`` reported are:

* ``covariate_smd``                — :math:`(\mathbf{x}_w, \mathbf{x}_v)`: synthetic treated vs
  synthetic control. The internal-validity check ("is the experiment
  apples-to-apples?").
* ``covariate_smd_treated_vs_pop`` — :math:`(\mathbf{x}_w, \bar{\mathbf{x}})`: synthetic treated vs
  population aggregate. Tracks the first term of MAREX's objective,
  :math:`\|\bar{\mathbf{x}} - \sum_j w_j \mathbf{x}_j\|^2`. Tells you whether the *chosen
  treated group* represents the population.
* ``covariate_smd_control_vs_pop`` — :math:`(\mathbf{x}_v, \bar{\mathbf{x}})`: synthetic control vs
  population aggregate. Tracks the second term of the objective. Tells you
  whether the *control set* represents the population.

A rule-of-thumb threshold of :math:`|\mathrm{SMD}| < 0.1` is conventionally
"well balanced"; below :math:`0.25` is acceptable; above is a red flag.

Power Analysis and Minimum Detectable Effect
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Power analysis answers the pre-experiment planning question: *given the
design I've chosen, how large a treatment effect can I detect with high
probability?* This is the dual of inference: inference asks "is the observed
effect distinguishable from noise?", power asks "what effect sizes would be?"

The paper develops permutation inference for MAREX but does not provide a
matching MDE. ``mlsynth`` fills this gap with an analytical, AR(1)-inflated
Gaussian MDE computed from the same residual series the permutation test
draws on. Set ``inference=True`` and the result auto-populates
``res.post_fit.power`` — ``blank_periods`` defaults to
``max(1, floor(0.3 * T0))`` so you do not need to pick a scalar yourself
(matching the LEXSCM / SYNDES / PANGEO convention).

Where the noise standard deviation comes from
""""""""""""""""""""""""""""""""""""""""""""""

Under the linear factor model of Assumption 1, the per-period contrast
:math:`g_t \coloneqq \sum_j w_j y_{jt} - \sum_j v_j y_{jt}` has expectation zero
under the no-effect null. Its sample SD on the blank window
:math:`\mathcal{B}` (the held-out tail of the pre-period) is the natural
estimator of the noise scale:

.. math::

   \widehat{\sigma}_{\text{placebo}} = \sqrt{\frac{1}{|\mathcal{B}| - 1}
       \sum_{t \in \mathcal{B}} \bigl(g_t - \bar g\bigr)^2}.

When no blank window is carved out (``inference=False``) the pre-period gap
serves as the placebo proxy. The blank-window estimator is preferred because
it uses periods that played no role in fitting the weights — it is honest in
exactly the same sense Chernozhukov-Wuthrich-Zhu's conformal residuals are.

Serial correlation matters
""""""""""""""""""""""""""

Synthetic-control gap residuals are virtually always serially correlated:
the donor weighting absorbs the level but the persistent components of the
factor structure (business cycles, seasonality, slow trends) leak through.
Ignoring this systematically under-states the SE at long horizons. We model
it as an AR(1) process with lag-1 autocorrelation

.. math::

   \widehat{\rho} = \frac{\sum_t g_t g_{t-1}}{\sum_t g_t^2},

clipped to :math:`(-0.99, 0.99)` for numerical safety. The variance of the
mean of :math:`T` consecutive AR(1) periods, expressed as a multiple of
:math:`\sigma^2`, is the *variance inflation factor*

.. math::

   \mathrm{VIF}(T, \rho) =
   \frac{1}{T}\!\left(1 + 2 \sum_{k=1}^{T-1}\!\Bigl(1 - \frac{k}{T}\Bigr)\rho^k\right),

which collapses to the textbook :math:`1/T` when :math:`\rho = 0` and grows
substantially for :math:`\rho > 0.3`. The same formula is used by PANGEO's
power module.

The MDE formula
"""""""""""""""

Combining: the standard error of the mean of :math:`T` post-period contrasts
under :math:`H_0` is :math:`\mathrm{SE}(T) = \widehat{\sigma}_{\text{placebo}} \,
\sqrt{\mathrm{VIF}(T, \widehat{\rho})}`. For a two-sided test at level :math:`\alpha`
with target power :math:`1 - \beta`, the minimum detectable effect is

.. math::

   \mathrm{MDE}(T) = \bigl(z_{1-\alpha/2} + z_{1-\beta}\bigr) \cdot
   \widehat{\sigma}_{\text{placebo}} \cdot \sqrt{\mathrm{VIF}(T, \widehat{\rho})}.

The corresponding power to detect a *given* true effect :math:`\tau` at
horizon :math:`T` is

.. math::

   \pi(\tau, T) = \Phi\!\Bigl(\frac{|\tau|}{\mathrm{SE}(T)} - z_{1-\alpha/2}\Bigr)
                 + \Phi\!\Bigl(-\frac{|\tau|}{\mathrm{SE}(T)} - z_{1-\alpha/2}\Bigr),

which is reported as ``power_at_observed`` for each horizon point using the
realised :math:`\widehat{\tau}`.

What the surface looks like
"""""""""""""""""""""""""""

.. code-block:: python

   p = res.post_fit.power                  # PowerAnalysis dataclass

   p.headline.mde_absolute                 # MDE at the realised T_post
   p.headline.mde_pct                      # ... as % of post-period baseline
   p.headline.se                           # implied SE of mean(g_t) over T_post
   p.headline.power_at_observed            # power to detect res.post_fit.ate

   p.curve                                 # tuple of MDEPoint, one per horizon
   for pt in p.curve:
       print(pt.post_periods, pt.mde_absolute, pt.mde_pct, pt.power_at_observed)

   p.sigma_placebo                         # σ̂ used (from blank or pre window)
   p.serial_correlation                    # ρ̂ AR(1) of the placebo gaps
   p.baseline                              # mean(synthetic_control) on post window
   p.alpha, p.power_target                 # 0.05 / 0.80 by default
   p.method                                # "analytical_ar1"

The default horizon grid covers :math:`T \in \{1, 2, 4, 6, 8, 12\}` plus the
realised ``n_post``, so the table also doubles as a *"how long do I need to
run?"* answer — pick the smallest :math:`T` whose MDE drops below your
target effect size.

Practical reading
"""""""""""""""""

A typical MAREX run with ``T_post = 6``, ``blank_periods = 4`` and modest
serial correlation (:math:`\widehat{\rho} \approx 0.5`) on a Walmart-style sales
panel produces an MDE on the order of 0.05–0.15% of mean sales, well
below the 1–3% effect sizes typical marketing interventions aim for; this
is the quantitative substance of *"good designs are well-powered"*.
Conversely, an MDE much above the expected effect is a signal the design
needs more units (lower ``m_eq``/``m_max`` are typically *worse* for power)
or more post-periods (extend the experiment).

Opting out
""""""""""

The power computation is wrapped in a ``try/except`` in
:func:`~mlsynth.utils.marex_helpers.orchestration.solve_marex` — a power
analysis failure (e.g. degenerate residual variance) never breaks the fit,
``res.post_fit.power`` is just left as ``None``. To compute power on a
non-default horizon grid or significance level, call the free function
directly:

.. code-block:: python

   from mlsynth.utils.post_fit import compute_power_analysis

   alt = compute_power_analysis(
       res.post_fit, alpha=0.10, power_target=0.90,
       post_grid=[2, 4, 8, 16, 32, 52],     # weekly horizons out to a year
   )

Monte Carlo: Recovering the Treatment Effect
--------------------------------------------

The block below replicates the qualitative finding of the paper's simulation
study (Section 5) using ``mlsynth``'s own reimplementation of the linear-factor
DGP. A sample is drawn, the design is fit on the pre-period, the treated units
realise :math:`y^I` in the experiment, and the estimate is compared to the true
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

   This is a Path-B replication: it reproduces the simulation study's
   conclusions from public DGPs and ``mlsynth`` code, with no dependency on the
   authors' replication package. It is locked in as
   :mod:`mlsynth.tests.test_marex_replication`.

Empirical Application: Walmart (Placebo Experiment)
---------------------------------------------------

We replicate the paper's empirical illustration (Section 4) on the Walmart
store-sales panel (``basedata/walmart_weekly_sales.csv``): weekly sales for
45 stores over 143 weeks (Feb 2010 – Oct 2012). Following the paper, we
design a placebo experiment with a fictitious intervention at week 129:
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
     - 2.2%
     - small (close tracking)
   * - Experimental ATT / mean sales
     - -1.0%
     - near zero
   * - Placebo permutation p-value
     - 0.937
     - 0.933
   * - Confidence band covers zero
     - yes (all post weeks)
     - yes

The synthetic treated and control units track to within ~2% of mean sales over
the fitting *and* blank periods, the estimated placebo effect is ~1% of sales,
and the permutation test fails to reject the null of no effect
(:math:`p = 0.937`, matching the paper's :math:`0.933`) — exactly the
"no spurious effect" result a good design should deliver on a placebo.

.. note::

   This uses the exact MIQP (``relaxed=False``, the default) with
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
   * - Predictors :math:`\mathbf{x} = [\mathbf{y}^E ; \mathbf{z}]`
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
   the *unconstrained* ``standard`` design (formulation 5) is degenerate —
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
       # Equivalently: pass a 0/1 column marking the experiment window
       # "post_col": "in_experiment",
       "m_eq": 2,           # treat exactly two markets
       "design": "standard",
       "inference": True,   # blank_periods defaults to floor(0.3 * T0) = 12
       "display_graph": True,
   }).fit()

   print("treated markets:", res.treated_units)
   print("global p-value:", res.globres.inference.global_p_value)
   for label, c in res.clusters.items():
       print(label, c.unit_weight_map["Treated"])

Verification
------------

Validated against Abadie & Zhao's Section 4 Walmart application (their reference
code is `jinglongzhao2/SCDesign <https://github.com/jinglongzhao2/SCDesign>`_):
on a 10-store subset of ``walmart_weekly_sales.csv`` MAREX's exact MIQP designs a
placebo experiment that tracks closely pre-period (pre-fit RMSE ~2.7% of mean
sales, matching LEXSCM) and yields a placebo effect indistinguishable from zero
(~1% of mean, CI covering zero) -- the paper's "no spurious effect" result. This
is an independent commit-stamped check (MAREX's own optimizer) complementing the
LEXSCM Walmart benchmark. See :doc:`replications/marex`; run it with
``python benchmarks/run_benchmarks.py marex_walmart``.

.. note::

   The benchmark uses the exact MIQP (free SCIP), not the relaxed
   continuous-``z`` mode: the relaxation shares the design objective but drops the
   integrality that defines the selection, so its top-``m`` rounding is degenerate
   and non-deterministic for small treated counts. The authors' full 45-store
   MIQP uses Gurobi, so the validator is Path A on a subset rather than a live R
   cross-validation.

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

.. note::

   ``MAREX.fit()`` returns a :class:`~mlsynth.config_models.DesignResult` (the
   experimental-design family, *not* an ``EffectResult``): MAREX designs an
   experiment, so it exposes the standardized design surface --
   ``res.report`` (the realized effect as an
   :class:`~mlsynth.config_models.EffectResult`, the single source for ATT / CI
   / pre-fit; ``res.report.att`` / ``res.report.counterfactual`` / ...),
   ``res.selected_units`` / ``res.assignment`` (treated vs control),
   ``res.design_weights``, and ``res.power``. The MAREX-specific design detail
   stays on ``res.clusters`` / ``res.study`` / ``res.globres`` / ``res.post_fit``
   (the same ``SyntheticControlPostFit`` that ``res.report`` is built from).

.. automodule:: mlsynth.utils.marex_helpers.structures
   :members:
   :undoc-members:
   :show-inheritance:

In addition, ``MAREX.fit()`` attaches a
:class:`~mlsynth.utils.post_fit.SyntheticControlPostFit` as
``results.post_fit``: the standardized diagnostics container shared across the
MAREX family (LEXSCM, MAREX, SYNDES, PANGEO). It carries the ATE / total /
lift / per-period / cumulative effect summaries, the inference triple
(:math:`p`, CI), the pre-/blank-/post-period RMSEs, the three
standardized-mean-difference blocks (treated-vs-control, treated-vs-population,
control-vs-population), and — when a valid noise window exists — a
:class:`~mlsynth.utils.post_fit.PowerAnalysis` block with the headline MDE and
the MDE-versus-horizon curve.

.. autoclass:: mlsynth.utils.post_fit.SyntheticControlPostFit
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: mlsynth.utils.post_fit.PowerAnalysis
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: mlsynth.utils.post_fit.MDEPoint
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

The shared post-fit module — :func:`~mlsynth.utils.post_fit.compute_smd`,
:func:`~mlsynth.utils.post_fit.compute_post_fit`, and
:func:`~mlsynth.utils.post_fit.compute_power_analysis` — lives outside the
``marex_helpers`` package so the other MAREX-family estimators (LEXSCM,
SYNDES, PANGEO) can call into the same one-source-of-truth diagnostics:

.. automodule:: mlsynth.utils.post_fit
   :members:
   :undoc-members:
   :show-inheritance:

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
