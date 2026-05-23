Parallel-Trends Supergeo Design (PANGEO)
=========================================

.. currentmodule:: mlsynth

Overview
--------

PANGEO is a prospective **experimental-design** estimator for geographic
(geo) experiments, in the lineage of **Supergeo Design** (Chen,
Doudchenko, Jiang, Stein & Ying 2023) and its scalable successor
*Optimized Supergeo Design* (Shaw 2025). It keeps the Supergeo idea --
group geos into composite "supergeos", form balanced **pairs**, randomise
treatment within each pair, and **trim no geo** -- including the
set-partitioning mixed-integer program that selects the design.

What changes is the **matching objective**. Supergeo matches supergeos on
a *scalar* aggregate (the summed response); OSD balances a handful of
*scalar covariate summaries*. Both collapse the time dimension. Two
candidate groups can have identical totals yet completely different
*shapes* over the pre-period -- which a downstream
difference-in-differences or synthetic-control analysis (that differences
trajectories) treats as not interchangeable at all.

PANGEO matches on the full pre-treatment **trajectory** instead. It
chooses the partition whose treatment and control halves are as
**parallel as possible** over the pre-period, scored by the DiD
pre-period residual sum of squares -- the *level-removed gap variance*:

.. math::

   \text{score}(A, B) = \sum_{t}\big[(\bar Y_{A,t}-\bar Y_{B,t})-\delta\big]^2,
   \qquad \delta = \overline{(\bar Y_A - \bar Y_B)} ,

for a supergeo pair split into halves :math:`A` (treatment) and
:math:`B` (control) with mean trajectories :math:`\bar Y_A, \bar Y_B`.
This is exactly the pre-period residual of a DiD fit (cf.
:func:`mlsynth.utils.selector_helpers._did_from_mean`). Minimising it
makes the halves run parallel; the **level shift** :math:`\delta` is
absorbed, so two supergeos can differ in level yet match perfectly on
*shape* -- precisely what parallel-trends DiD needs, and what scalar
matching discards.

Multi-arm support
^^^^^^^^^^^^^^^^^

A single categorical column names each geo's eligible treatment arm
(e.g. values ``A`` / ``B`` / ``C``). Arms occupy non-overlapping geos, and
PANGEO designs **each arm independently** -- only same-arm geos are
combined into supergeos. The output is a **design** (supergeo pairs +
treatment/control assignment + achieved parallelism), not a treatment
effect; PANGEO is run *before* the experiment, on historical sales.

Algorithm
---------

For each arm:

1. **Enumerate** admissible supergeo pairs over the arm's geos -- subsets
   that split into two halves each of size :math:`\le Q` -- scoring each
   by its best-split level-removed gap variance.
2. **Solve** the set-partitioning MIP (cvxpy + HiGHS): choose pairs that
   cover every geo exactly once, minimising total non-parallelism

   .. math::

      \min_{x\in\{0,1\}}\ \sum_G \text{score}(G)\,x_G
      \quad\text{s.t.}\quad M^\top x = \mathbf 1,\ \mathbf 1^\top x \ge \kappa .

3. Within each chosen pair, treatment/control is the score-minimising
   split (randomised in the field).

Setting ``max_supergeo_size = 1`` (``Q = 1``) recovers the classic
matched-pairs design; ``Q > 1`` allows composite supergeos when no single
geo matches another well -- without trimming.

Automatic Q
^^^^^^^^^^^

``Q`` is a granularity knob with an *interior* optimum: too small and no
parallel matches exist (singletons are too noisy), too large and you get
fewer, coarser pairs. Crucially the program-level MDE is **not** monotone
in ``Q`` and is **not** tracked by parallelism :math:`R^2` (which is
scale-free and keeps rising with ``Q``) -- only by the absolute residual
variance that drives power. So if ``max_supergeo_size`` is left unset,
PANGEO selects it automatically: it designs every feasible ``Q`` in
``1 .. min(ceil(smallest_arm / 2), 6)`` and returns the one with the
smallest mean program MDE. The full sweep -- each ``Q``'s program-pair
count, mean program MDE, and the ``2/2**pairs`` randomization-inference
p-value floor -- is stored in ``results.metadata["q_sweep"]`` (and the
choice in ``results.metadata["q_selected"]``), so the decision is auditable
and a user who prizes design-based inference can override with an explicit
``Q``.

Choice of objective
^^^^^^^^^^^^^^^^^^^

Because each pair's score is precomputed, the outer problem is a *linear*
MILP regardless of how the per-pair cost is defined. PANGEO exposes three
costs via ``objective`` (all leave the solver a MILP):

* ``"ss_res"`` (default) -- the absolute DiD residual sum of squares
  :math:`\sum_t (g_t-\bar g)^2`. Scale-dependent: high-amplitude pairs
  weigh more, so the design prioritises getting large markets parallel.
* ``"r2"`` -- ``1 - R^2 = ss_res / ss_tot``. Scale-free: every pair counts
  equally (this is FDID's R^2 criterion, but optimised *exactly* by the
  MILP rather than greedily).
* ``"weighted"`` -- a recency-weighted residual SS
  :math:`\sum_t w_t (g_t-\bar g_w)^2` with the level removed at the
  *weighted* mean, and weights ``w_t = recency_decay**(T0-1-t)`` so recent
  pre-period parallelism (closest to the upcoming experiment) matters
  most.

The reported ``gap_variance`` and ``parallelism_r2`` per pair are always
the unweighted DiD quantities, so designs from different objectives are
comparable on a common yardstick.

Balancing baseline covariates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Parallelism is **level-blind**: the DiD level shift :math:`\delta` absorbs
any constant gap between two supergeos, so a time-invariant characteristic
(population, income) that merely shifts a market's *level* is differenced
out and never enters the trajectory score. That is exactly right for
parallel-trends DiD, but it means raw parallelism says nothing about
whether treatment and control are balanced on such baseline
characteristics -- the gap OSD's scalar covariate matching was built to
close.

PANGEO closes it the same way OSD does -- with a standardized
**mean-difference** penalty -- but adds it to the per-pair score so the
outer problem stays a linear MILP. Pass ``covariates=[...]`` (baseline
columns; each unit's value is its mean over the panel) and the cost of a
split gains

.. math::

   \sum_m w_m \Big(\frac{\bar c_{A,m} - \bar c_{B,m}}{s_m}\Big)^2 ,

the weighted squared standardized mean difference (SMD) between the
treatment and control halves' covariate means, with :math:`s_m` the
cross-unit standard deviation (``standardize_covariates=True``, default)
and :math:`w_m` a per-covariate weight from ``covariate_weights`` (default
``1`` each). Larger weights buy tighter covariate balance at the cost of
some pre-period parallelism; the achieved per-pair SMDs are reported in
``SupergeoPair.covariate_smd`` so the tradeoff is visible. With no
``covariates`` the design is unchanged.

.. code-block:: python

   df = make_seasonal_sales_panel(units_per_arm=6, arms=("A", "B", "C"),
                                  T=104, seed=0, covariates=True)

   res = PANGEO({
       "df": df, "outcome": "sales", "arm": "arm",
       "unitid": "unit", "time": "time", "max_supergeo_size": 3,
       "covariates": ["population", "income"],
       "covariate_weights": {"population": 5.0, "income": 5.0},
   }).fit()

   for arm, design in res.arm_designs.items():
       for p in design.pairs:
           print(p.treatment, p.control, p.parallelism_r2, p.covariate_smd)

Power and minimum detectable effect
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A design is only useful if it can *detect* the effects a program cares
about, so :meth:`mlsynth.PANGEO.fit` also returns a power / MDE analysis
(``results.power``, a :class:`mlsynth.utils.pangeo_helpers.power.PangeoPower`).
The key observation is that **power is the design objective**: for a
supergeo pair the no-effect gap :math:`g_t = \bar Y^T_t - \bar Y^C_t` sits
on its parallel-trends line :math:`\delta_p`, and its per-period residual
variance is exactly the score the MILP minimised,

.. math::

   \sigma_p^2 = \text{ss\_res}_p / (T_0 - 1) .

The ``X``-period difference-in-differences ATT
:math:`\hat\tau_p = \bar g_{\text{post}} - \delta_p` then has variance
:math:`\sigma_p^2\,[f(X,\rho) + f(T_0,\rho)]`, where
:math:`f(n,\rho)` is the variance-inflation factor of the mean of ``n``
serially-correlated periods. **Serial correlation is decisive and is the
trap a naive i.i.d. power calculation falls into**: weekly sales are highly
autocorrelated, so ``X`` post weeks are worth far fewer than ``X``
independent observations, and adding post periods yields sharply
diminishing returns. :math:`\rho` is estimated as the pooled lag-1 (AR(1))
autocorrelation of the chosen pairs' pre-period gap residuals.

The minimum detectable effect at power :math:`1-\beta` and two-sided level
:math:`\alpha` is :math:`\text{MDE} = (z_{1-\alpha/2}+z_{1-\beta})\,
\operatorname{SE}(\hat\tau)`, reported both in outcome units and as a
percentage of the treated-group baseline level.

**Program level is the headline.** Small arms are individually
under-powered -- with only a handful of supergeo pairs, a pure
within-pair *randomisation* test has a hard p-value floor of
:math:`2/2^{P}` (one pair can never reject; you need :math:`\ge 6` pairs
to even reach :math:`p<0.05`). PANGEO therefore reports a model-based MDE
whose precision comes from the long pre-period, and **pools across arms**:
the program ATT is the treated-size-weighted average of the pair ATTs, so
its effective sample size is the *total* number of pairs. Pooling routinely
detects effects several points smaller than any one arm could, which is the
number a program owner should take to leadership. Per-arm curves are stored
too (``results.power.arms``), and pairs are treated as independent across
the program (cross-pair common shocks within an arm are ignored -- a mild
optimism).

By default the MDE is computed at **80% power** for post-period horizons
**2..12**; ``power_target``, ``power_alpha`` and ``power_post_periods``
configure this, and ``compute_power=False`` skips it.

.. code-block:: python

   res = PANGEO({
       "df": df, "outcome": "sales", "arm": "arm",
       "unitid": "unit", "time": "time", "max_supergeo_size": 3,
   }).fit()

   pw = res.power
   print(f"serial correlation rho = {pw.serial_correlation:.2f}")
   print(pw.summary())                       # MDE % by horizon: program + arms
   # "With this design we can detect a {x}% lift after 8 weeks at 80% power."
   print(pw.program.mde_pct_by_horizon()[8])
   # Or invert: power to detect a specific effect.
   print(pw.power_for_effect(effect_pct=5.0, post_periods=8))

Estimating the realized ATT (Augmented DiD)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PANGEO is a *design* method, but once the experiment has run you can score
the **same design** against the realized outcomes by passing a ``post_col``
(a 0/1 indicator of post-treatment periods, as in LEXSCM). The design is
built on the pre rows alone -- so it is byte-for-byte identical to the
design-only result -- and ``results.effects`` carries the realized ATT at
the **arm** and **program** levels (the program ATT is the headline). An
optional ``weight_col`` (e.g. population) makes both the design and the ATT
population-weighted.

Theory
""""""

The realized ATT uses the **Augmented Difference-in-Differences** estimator
and inference of Li & Van den Bulte (2022). For a treated supergeo
aggregate :math:`y^{T}_t` and the control supergeo aggregate :math:`y^{C}_t`
at a level (arm or program), the counterfactual is the pre-period regression
projection

.. math::

   y^{T}_t = \delta_1 + \delta_2\, y^{C}_t + \gamma\, t + e_t ,
   \qquad t = 1,\dots,T_1 ,

fit by OLS. This *augments* plain DiD in two ways: the control scale
:math:`\delta_2` is free (rather than forced to 1) and a linear time trend
:math:`\gamma t` is included. With :math:`x_t=(1,y^{C}_t,t)'` and
:math:`\hat\delta` the OLS fit, the per-period effect is
:math:`\hat u_t = y^{T}_t - x_t'\hat\delta` and
:math:`\hat\Delta = T_2^{-1}\sum_{t>T_1}\hat u_t`.

Li & Van den Bulte show (Prop. 3.1-3.3) that
:math:`\sqrt{T_2}(\hat\Delta-\Delta)\to N(0,\Sigma_1+\Sigma_2)`, giving the
**prediction-variance** standard error (their Web Appendix C.13)

.. math::

   \widehat{\operatorname{Var}}(\hat\Delta)
   = \hat\omega^2\Big[\,\bar x_{\text{post}}'
       \big(\textstyle\sum_{t=1}^{T_1} x_t x_t'\big)^{-1}
       \bar x_{\text{post}} + \tfrac{1}{T_2}\Big] ,

where :math:`\bar x_{\text{post}}` is the post-period mean of :math:`x_t`,
the first term is the coefficient-estimation variance :math:`\Sigma_1` and
:math:`1/T_2` is the post-averaging variance :math:`\Sigma_2`.
:math:`\hat\omega^2` is the residual variance estimated over the long
**pre**-period -- a Newey-West/Bartlett long-run variance with lag
:math:`\lfloor T_1^{1/4}\rfloor` to admit serial correlation (lag 0 is the
i.i.d. case :math:`\hat e'\hat e/(T_1-k)`). The CI is
:math:`\hat\Delta \pm z_{1-\alpha/2}\,\mathrm{SE}` and the p-value is the
two-sided normal test of :math:`\Delta=0`.

Why this estimator fits the supergeo gap. Li & Van den Bulte's assumptions
(C2/C3) explicitly admit **trend and unit-root (integrated) common
factors** -- the regimes that break naive, conformal and permutation
inference. The augmentation :math:`\delta_2` makes treated-on-control a
*cointegrating* regression that scales out a shared integrated factor; the
trend term absorbs deterministic drift; and the prediction-variance term
:math:`\bar x_{\text{post}}'(\sum x_t x_t')^{-1}\bar x_{\text{post}}`
**inflates automatically when the post-period control drifts outside its
pre-period range**, pricing the extrapolation uncertainty. The validity
condition is simply that the *residual* :math:`e_t` be (weakly dependent)
stationary -- which the augmentation and trend deliver. The design's
parallelism is retained; it minimises the residual variance, which directly
tightens this SE (and thus the MDE in the power analysis, which is left
untouched and continues to use the design's pre-period gap variance).

Smoke tests / validity envelope
"""""""""""""""""""""""""""""""

A Monte Carlo over the bundled simulator (which can place the unobserved
common factor on an i.i.d., AR(1) or random-walk process via ``factor`` and
toggle ``season_amp``/``trend_sd``) confirms the diagnosis that *validity
hinges on residual stationarity*, not on the interval recipe. With
``factor="iid", season_amp=0, trend_sd=0`` -- a stationary gap matching the
paper's factor-model DGP -- the program ATT is unbiased and the 95%
interval attains **~0.93 coverage** with **~0.07 type-I** error. Adding a
linear trend and seasonality (which the augmentation + trend regressor
partly absorb) gives ~0.87 coverage; the adversarial random-walk-plus-
seasonality gap, where two integrated factors and amplitude-heterogeneous
seasonality exceed what a single :math:`\delta_2` can cointegrate, drops to
~0.6 -- the honest assumption boundary. The point estimate is unbiased
throughout. These experiments live in ``mlsynth/tests/test_pangeo.py``
(``TestADIDInference``).

.. code-block:: python

   df = make_seasonal_sales_panel(units_per_arm=6, arms=("A", "B", "C"),
                                  T=104, seed=0, n_post=8)
   res = PANGEO({
       "df": df, "outcome": "sales", "arm": "arm",
       "unitid": "unit", "time": "time", "post_col": "post_col",
       "max_supergeo_size": 3,
       "att_augment": True, "att_trend": True,   # Augmented DiD (defaults)
   }).fit()

   print(res.effects.summary())           # program + per-arm ATT, SE, CI, p
   pe = res.effects.program
   print(f"program ATT = {pe.att_pct:.1f}% "
         f"[{pe.ci_lower_pct:.1f}, {pe.ci_upper_pct:.1f}], "
         f"p={pe.p_value:.3f}, scale delta_2={pe.scale:.2f}")

Core API
--------

.. automodule:: mlsynth.estimators.pangeo
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.PANGEOConfig
   :members:
   :undoc-members:

Helper Modules
--------------

.. automodule:: mlsynth.utils.pangeo_helpers.parallelism
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.pangeo_helpers.mip
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.pangeo_helpers.setup
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.pangeo_helpers.pipeline
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.pangeo_helpers.power
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.pangeo_helpers.effects
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.pangeo_helpers.simulation
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.pangeo_helpers.structures
   :members:
   :undoc-members:

Example
-------

A seasonal, multi-arm sales panel (the bundled simulator), designed into
parallel supergeo pairs. With ``display_graphs=True`` PANGEO plots each
arm's treatment vs control aggregate pre-period trajectories.

.. code-block:: python

   from mlsynth import PANGEO
   from mlsynth.utils.pangeo_helpers import make_seasonal_sales_panel

   # 3 arms (non-overlapping geos), 6 geos each, 156 weeks of history.
   df = make_seasonal_sales_panel(units_per_arm=6, arms=("A", "B", "C"),
                                  T=156, seed=0)

   res = PANGEO({
       "df": df,
       "outcome": "sales",
       "arm": "arm",                # single categorical arm column
       "unitid": "unit",
       "time": "time",
       "max_supergeo_size": 3,      # Q
   }).fit()

   for arm, design in res.arm_designs.items():
       print(f"Arm {arm}: {len(design.pairs)} pair(s), "
             f"parallel-trends R^2 = {design.mean_parallelism_r2:.3f}")
       for p in design.pairs:
           print(f"   T={p.treatment}  C={p.control}  R^2={p.parallelism_r2:.3f}")

   # res.assignment maps every geo -> 'treatment' / 'control'.

On the simulated data this returns designs with parallel-trends R^2 around
0.90-0.98 -- roughly 10-35x more parallel than a random treatment/control
split of the same geos.

References
----------

Chen, A., Doudchenko, N., Jiang, S., Stein, C., & Ying, B. (2023).
"Supergeo Design: Generalized Matching for Geographic Experiments."
arXiv:2301.12044.

Shaw, C. (2025). "Optimized Supergeo Design: A Scalable Framework for
Geographic Marketing Experiments." arXiv:2506.20499.

Li, K. T. (2023). "Frontiers: A Simple Forward Difference-in-Differences
Method." *Marketing Science* 43(2):267-279.

Li, K. T., & Van den Bulte, C. (2022). "Augmented Difference-in-Differences."
*Marketing Science* 42(4):746-767.

Abadie, A., Diamond, A., & Hainmueller, J. (2010). "Synthetic Control
Methods for Comparative Case Studies." *Journal of the American
Statistical Association* 105(490):493-505.
