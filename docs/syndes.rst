Synthetic Design (SYNDES)
=========================

.. currentmodule:: mlsynth

When to Use This Estimator
--------------------------

Most synthetic-control work takes the treated unit as *given* and asks only how
to weight the donors. ``SYNDES`` answers the prior question Doudchenko et al.
[SYNDES]_ pose: when you are about to run an experiment and have
pre-treatment outcome data, *which units should you treat*? Treating units at
random -- or by hand -- leaves accuracy on the table, because the variance of
the resulting treatment-effect estimate depends on which units are treated and
how the rest are weighted into a synthetic comparison.

The authors argue this is exactly the regime of market-level experiments:
treatment can only be applied to coarse units (media markets, regions, whole
products), each unit is expensive to treat (so ``K`` is small), and
interference or equilibrium effects rule out a more granular randomization. In
that setting the experimenter both *chooses* the treated set and *estimates* the
effect, and SYNDES does both at once -- it minimizes the mean squared error of
the average-treatment-effect-on-the-treated estimator directly over the joint
choice of treatment assignment and synthetic weights. Use it when:

* you control assignment and have a panel of pre-treatment outcomes;
* you want a small, well-chosen treated set rather than a random one;
* you are willing to solve a mixed-integer program for a provably optimal design
  (or to bound the achievable :ref:`power <syndes-inference>` of one).

Notation
--------

We observe an outcome :math:`y_{it}` for all units
:math:`\mathcal{N} \coloneqq \{1, \ldots, N\}` over pre-treatment periods
:math:`t \in \mathcal{T}_1 \coloneqq \{1, \ldots, T_0\}` of length
:math:`T_0 = T`. After period :math:`T_0` the experimenter assigns a binary
treatment :math:`D_i \in \{0, 1\}` to be applied over the post-treatment
periods :math:`\mathcal{T}_2 \coloneqq \{t : t > T_0\}` (of length
:math:`S - T_0`), with exactly ``K`` treated units
(:math:`\sum_i D_i = K`). The assignment vector
:math:`\mathbf{D} \coloneqq (D_1, \ldots, D_N)^\top` is itself a decision variable.
Each unit has potential outcomes :math:`(y_{it}^N, y_{it}^I)` and observed
outcome :math:`y_{it} = y_{it}(D_i)`. Synthetic weights :math:`\mathbf{w}` live
on the simplex (non-negative, summing to one on the relevant side). The estimand
is the weighted average treatment effect on the treated (wATET)
:math:`\tau \coloneqq \sum_{i:D_i=1} w_i \tau_i`, where :math:`\tau_i` is unit
:math:`i`'s additive effect.

.. note::

   Notation bridge. The single-treated-unit synthetic-control canon (treated
   :math:`j=1`, donor pool :math:`\mathcal{N}_0 \coloneqq \mathcal{N}
   \setminus \{1\}`) takes the treated unit as given, so it does not fit a
   *design* problem in which ``K`` treated units are themselves chosen. We
   therefore keep the design index :math:`i` for units, with the assignment
   vector :math:`\mathbf{D}` a decision variable, and write :math:`T_0` for the
   pre-treatment length (the page's :math:`T`) and :math:`\mathcal{T}_1` /
   :math:`\mathcal{T}_2` for the pre/post period sets.

The design problem
~~~~~~~~~~~~~~~~~~

Under the outcome model :math:`y_{it}^N = \mu_{it} + \varepsilon_{it}` with
mean-zero, homoskedastic noise (:math:`\operatorname{Var}\varepsilon_{it} =
\sigma^2`) and additive effects :math:`y_{it} = y_{it}^N + D_i \tau_i`, the
conditional MSE of the per-unit synthetic-control estimator
:math:`\widehat{\tau}_i \coloneqq y_{i,T_0+1} - \sum_{j:D_j=0} w^i_j y_{j,T_0+1}` is

.. math::

   \mathbb{E}\bigl[(\widehat{\tau}_i - \tau_i)^2 \mid \mathbf{D}, \mathbf{w}\bigr]
   = \Bigl(\mu_{i,T_0+1} - \textstyle\sum_{j:D_j=0} w^i_j \mu_{j,T_0+1}\Bigr)^2
     + \sigma^2\Bigl(1 + \textstyle\sum_{j:D_j=0} (w^i_j)^2\Bigr).

The first term is a bias from imperfect pre-treatment matching; the second a
variance that grows with the weight concentration. SYNDES minimizes the
empirical, pre-period analogue of this MSE jointly over
:math:`(\mathbf{D}, \mathbf{w})` -- the :math:`\sigma^2 \sum w^2` term becomes
the ridge penalty :math:`\lambda` below.
Because the choice of treated set makes the estimand itself stochastic, the
target is the wATET *for the units SYNDES selects*, not a fixed population ATE.

The three MIP formulations
--------------------------

The joint optimization over assignment and weights is a bilevel /
mixed-integer program and is NP-hard. Doudchenko et al. give three forms,
all exposed through ``mode`` and all solved as MIPs (auxiliary variables
linearize the weight-assignment products). They differ in *how the treated and
control sides are weighted*.

Per-unit (``mode="per_unit"``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A separate synthetic control for every treated unit:

.. math::

   \operatorname*{argmin}_{\mathbf{D}, \{w^i_j\}} \;
     \frac{1}{K T_0} \sum_{i} \sum_{t \in \mathcal{T}_1} D_i
       \Bigl(y_{it} - \textstyle\sum_j w^i_j (1 - D_j) y_{jt}\Bigr)^2
     + \frac{\lambda}{K} \sum_i \sum_j D_i (w^i_j)^2,

subject to :math:`w^i_j \ge 0`, :math:`\sum_i D_i = K`, and
:math:`\sum_j w^i_j(1 - D_j) = 1` for each treated :math:`i`. Each treated unit
draws its own simplex of control weights.

Two-way global (``mode="two_way_global"``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A single weight vector applied to both sides of one global contrast:

.. math::

   \operatorname*{argmin}_{\mathbf{D}, \{w_i\}} \;
     \frac{1}{T_0} \sum_{t \in \mathcal{T}_1} \Bigl(\textstyle\sum_i w_i D_i y_{it}
       - \sum_i w_i (1 - D_i) y_{it}\Bigr)^2 + \lambda \sum_i w_i^2,

subject to :math:`w_i \ge 0`, :math:`\sum_i D_i = K`,
:math:`\sum_i w_i D_i = 1` and :math:`\sum_i w_i (1 - D_i) = 1`. ``mlsynth``
linearizes :math:`q_i \coloneqq w_i D_i` and enforces the two normalizations with
:math:`\sum_i q_i = 1`, :math:`\sum_i w_i = 2`, so the per-period contrast is
:math:`(2\mathbf{q} - \mathbf{w})^\top \mathbf{y}_t`.

One-way global (``mode="one_way_global"``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The two-way program with the treated weights pinned equal (a simple average,
:math:`w_i = 1/K` on the treated), while the control side stays a free
synthetic control:

.. math::

   \operatorname*{argmin}_{\mathbf{D}, \mathbf{c}} \;
     \frac{1}{T_0}\sum_{t \in \mathcal{T}_1} \Bigl(\tfrac{1}{K}\textstyle\sum_i D_i y_{it}
       - \sum_i c_i y_{it}\Bigr)^2 + \lambda\Bigl(\tfrac{1}{K} + \textstyle\sum_i c_i^2\Bigr),

subject to :math:`c_i \ge 0`, :math:`\sum_i c_i = 1`, :math:`c_i \le 1 - D_i`
(treated units carry no control weight) and :math:`\sum_i D_i = K`.

.. warning::

   One-way global is not difference-in-means. Only the *treated* side is
   fixed at :math:`1/K`; the control side :math:`\mathbf{c}` is a free synthetic
   control to be optimized. Pinning *both* sides (treated :math:`1/K`, control
   :math:`1/(N-K)`) would be the randomized difference-in-means baseline, a
   different (and weaker) design.

Assumptions / Remarks.

*Assumption 1 (additive effects, homoskedastic noise).* Outcomes follow
:math:`y_{it}^N = \mu_{it} + \varepsilon_{it}` with
:math:`\mathbb{E}\varepsilon_{it}=0`, :math:`\operatorname{Var}\varepsilon_{it}
= \sigma^2`, and treatment adds :math:`\tau_i`. *Remark.* This is what makes the
MSE above decompose into the matching-bias and weight-variance terms the MIP
minimizes; :math:`\sigma^2` is unknown and is supplied through ``lam`` (default:
the pre-period sample variance).

*Assumption 2 (admissible weights).* Weights are non-negative and normalized on
their side (a convex combination), so the synthetic comparison does not
extrapolate. *Remark.* The simplex is what gives the design an interpretable
"synthetic unit" reading and bounds the variance term.

*Assumption 3 (homogeneous vs. heterogeneous effects).* When effects are
homogeneous (:math:`\tau_i \equiv \tau`) any weighted average recovers
:math:`\tau`, so the global modes can choose weights freely to minimize MSE.
When effects are heterogeneous, different weightings target different estimands;
per_unit (or the fixed-treated-weight one_way_global) keeps the estimand
well-defined. *Remark.* This is the authors' guidance for choosing a mode -- it
is about which wATET you are willing to target, not just fit.

*Assumption 4 (sharp null for inference).* The permutation test targets the
sharp null :math:`\tau_i = 0` for all treated :math:`i`. *Remark.* Correct test
size holds under the exchangeability the moving-block permutation imposes; the
authors note this requires "rather strong assumptions" in finite samples.

.. _syndes-inference:

Inference and minimum detectable effect
---------------------------------------

For any mode the fitted design yields a unit-level contrast vector
:math:`\mathbf{c}` such that the ATT estimate at period :math:`t` is
:math:`\mathbf{y}_t^\top \mathbf{c}` (treated weights minus control weights; for
``per_unit`` the :math:`K` per-unit estimators are averaged). ``SYNDES`` tests
the sharp null with the moving-block permutation
test of Chernozhukov, Wuethrich and Zhu (2021): the post-period mean contrast is
compared to the distribution obtained by cyclically shifting the stacked panel.

For design-time power, :func:`~mlsynth.power_analysis` returns a per-horizon
minimum detectable effect (MDE). Because the moving-block test averages a
contrast over correlated periods, the relevant null standard error is the
Newey-West (Bartlett HAC) long-run std of the per-period contrast, not the
i.i.d. :math:`\sigma_{\text{perm}}/\sqrt{n_{\text{post}}}`:

.. math::

   \mathrm{MDE}(n_{\text{post}}) = (z_{1-\alpha/2} + z_{1-\beta})\,
       \frac{\widehat{\sigma}_{\mathrm{LR}}}{\sqrt{n_{\text{post}}}},

reported as ``long_run_sigma``. It reduces to the textbook formula when the
contrast series is serially uncorrelated.

Standardized Post-Fit and Power Analysis
----------------------------------------

Every call to :meth:`SYNDES.fit` attaches a
:class:`~mlsynth.utils.post_fit.SyntheticControlPostFit` to ``res.post_fit``
— the same diagnostic surface used by LEXSCM, MAREX, and PANGEO. This is
the one-stop container downstream consumers (dashboards, paper-style reports,
comparison tables) read from, regardless of which member of the family
produced the design:

.. code-block:: python

   pf = res.post_fit                          # SyntheticControlPostFit
   pf.ate, pf.ate_percent, pf.total_effect    # treatment-effect scalars
   pf.rmse_fit, pf.rmse_post                  # pre / post fit quality
   pf.p_value, pf.ci_lower, pf.ci_upper       # permutation inference
   pf.power                                   # PowerAnalysis (see below)

The synthetic treated / control trajectories used to populate ``post_fit`` are
the per-unit weighted aggregates ``Y[:, j] @ treated_weights`` and
``Y[:, j] @ control_weights`` over the full timeline. SYNDES has no
pre-period blank window (its inference is a moving-block permutation on the
post-period rather than a placebo test on a held-out pre-tail), so
``pf.n_blank = 0`` and the power-analysis module falls back to the
pre-period gap as its placebo proxy. Mathematically the MDE surface is the
same Gaussian + AR(1) construction used across the family:

.. math::

   \mathrm{MDE}(T) = \bigl(z_{1-\alpha/2} + z_{1-\beta}\bigr) \cdot
       \widehat{\sigma}_{\text{placebo}} \cdot \sqrt{\mathrm{VIF}(T, \widehat{\rho})},

with :math:`\widehat{\sigma}_{\text{placebo}}` the per-period contrast SD on the
pre-period (the SYNDES paper's "pre-period imbalance"), :math:`\widehat{\rho}` the
lag-1 autocorrelation of that contrast clipped to :math:`(-0.99, 0.99)`, and
:math:`\mathrm{VIF}(T, \rho) = \tfrac{1}{T}\bigl(1 + 2\sum_{k=1}^{T-1}
(1-k/T)\rho^k\bigr)` the AR(1) variance-inflation factor (textbook
:math:`1/T` when :math:`\rho = 0`). See :doc:`marex` for the full derivation;
the same module powers all three estimators.

.. code-block:: python

   p = res.post_fit.power                      # PowerAnalysis
   p.headline.mde_absolute                     # MDE at the realised T_post
   p.headline.mde_pct                          # ... as % of post-period baseline
   p.headline.power_at_observed                # power to detect res.post_fit.ate
   p.curve                                     # tuple of MDEPoint per horizon

Power-analysis failures (e.g. degenerate pre-period contrast) never break a
fit; ``res.post_fit.power`` is simply left as ``None`` in that case. To
compute on a non-default horizon grid or significance level call
:func:`~mlsynth.utils.post_fit.compute_power_analysis` directly.

post_col vs T0
~~~~~~~~~~~~~~

``SYNDES`` accepts either a scalar ``T0`` (count of pre-treatment periods) or
``post_col`` (a 0/1 column marking the post-treatment window). Both express
the same pre/post split — passing ``post_col`` is just the more ergonomic
form when the panel already carries an experiment-window flag. If both are
supplied and disagree, ``post_col`` wins and a ``UserWarning`` is emitted so
the override is visible.

Choosing among the modes
------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - Mode
     - Weighting
     - Use when
   * - ``per_unit``
     - one synthetic control per treated unit
     - effects are heterogeneous; you want unit-level estimates and the tightest per-unit fit
   * - ``two_way_global``
     - one weight vector, both sides free
     - effects are homogeneous; you want the lowest-MSE single contrast
   * - ``one_way_global``
     - treated fixed at ``1/K``, control free
     - heterogeneous effects but a simple, fixed treated average is the target estimand



Solver runtime and the 5%-gap default
-------------------------------------

The SYNDES MIP is structurally hard. The ``two_way_global`` formulation
contains a bilinear product :math:`q_i \coloneqq w_i D_i` between the
continuous weight :math:`w_i` and the binary assignment :math:`D_i`, encoded
via the standard McCormick linearisation (``q_i \le D_i``,
``q_i \le w_i``, ``q_i \ge w_i - (1 - D_i)``). McCormick is the
tightest *linear* relaxation of a bilinear term, but it is still loose
at the root LP — so the SCIP optimality gap closes slowly on long
panels even when the primal incumbent is essentially optimal. For
example on the Walmart weekly-sales panel
(:math:`N = 45,\ T_0 = 128`, :math:`K = 3`) SCIP finds the optimal
treated set within a minute, then spends an additional 30+ minutes
*proving* optimality by climbing the dual bound. The treated set
itself does not change during this proof phase.

This matters because in practice our SCM bias bounds do *not* require
optimality of the solver. `Abadie and Zhao (2026) <https://economics.mit.edu/sites/default/files/2026-02/Synthetic%20Controls%20for%20Experimental%20Design%20Feb%202026.pdf>`_ (2026, eq. 10 discussion,
p. 10 and 13), writing about their formulation, state explicitly:

   *"we do not strictly require optimality of* :math:`\{w^*, v^*\}`,
   *provided* :math:`\{w^*, v^*\}` *is feasible and*
   :math:`\bar{X} - \sum_j w^*_j X_j \approx 0` *and*
   :math:`X_j - \sum_i v^*_{ij} X_i \approx 0` *for all j such that*
   :math:`w^*_j > 0`."

Their Theorems 1 and 2 are written in terms of the residual fit, not
the QP optimality gap, so a 5%-suboptimal solution that achieves
approximate balance inherits the same econometric guarantees as a
proven-optimal one. SYNDES is not the same problem as the ones AZ are concerned with, but the conclusion still holds.

mlsynth therefore exposes two SCIP-knob fields on :class:`SYNDESConfig`
and defaults them to the production-friendly setting:

* ``gap_limit`` (default ``0.05``, i.e. 5%) -- handed to SCIP as
  ``scip_params={"limits/gap": value}``. The MIP terminates as soon
  as the primal-dual gap is within this fraction of the incumbent.
* ``time_limit`` (default ``60.0`` seconds) -- wall-clock cap on the
  solve, passed through as ``scip_params={"limits/time": value}``.

With these defaults Walmart-scale designs return in under a minute
with a known :math:`\le 5\%` gap to the (provable) optimum. Tighten
either knob -- or set it to ``None`` -- for research-grade
optimality:

.. code-block:: python

   # Default: 5% gap, 60s wall-clock — production-suitable.
   SYNDES({
       "df": df, "outcome": "y", "unitid": "unit", "time": "time",
       "K": 3, "mode": "two_way_global", "post_col": "post",
   }).fit()

   # Loosen the gap to return in seconds when you just need a
   # plausible design for prototyping.
   SYNDES({...,
           "gap_limit": 0.25, "time_limit": 5.0,
   }).fit()

   # Disable both limits for an asymptotic-optimality run. Be
   # prepared for hours-long solves on long panels.
   SYNDES({...,
           "gap_limit": None, "time_limit": None,
   }).fit()

The MIP status codes ``user_limit`` and ``user_limit_inaccurate``
(SCIP's "stopped early with a valid incumbent") are accepted as
successful returns alongside the standard ``optimal`` /
``optimal_inaccurate`` codes — again, because the theory only needs
the incumbent's feasibility, not the proof of optimality.

.. note::

   If you have a commercial solver (Gurobi, CPLEX, MOSEK) installed,
   pass ``solver="GUROBI"`` and the MIP closes the gap orders of
   magnitude faster than SCIP — these solvers handle MIQP / MIQCP
   relaxations natively. The default of SCIP is chosen because it
   ships with mlsynth (via ``pyscipopt``) with no license required.

Multiple Treatment Arms
-----------------------

When a single experiment runs several treatment arms (e.g. different
creatives, offers, or price points, each rolled out to its own set of
markets), pass an ``arm`` column. ``SYNDES`` then solves the design problem
independently within each arm's units and returns a
:class:`~mlsynth.utils.syndes_helpers.structures.SYNDESMultiArmResults` —
a dict of per-arm results keyed by arm label. Every option (``mode``, ``K``,
``lam``, inference) applies within each arm, and ``K`` is interpreted
per arm (so it must be smaller than the smallest arm's unit count).

.. code-block:: python

   res = SYNDES({
       "df": df, "outcome": "sales", "unitid": "DMA", "time": "week",
       "arm": "treat",                 # categorical arm label per unit
       "K": 3, "mode": "two_way_global", "post_col": "post",
       "run_inference": True,
   }).fit()

   res.arm_designs["A"]                 # full SYNDESResults for arm A
   res.atet_by_arm()                    # {arm: ATET}
   res.selected_unit_labels_by_arm()    # {arm: treated units}

The arm column must be constant within each unit over time. ``arm`` is not
compatible with the global ``costs``/``budget`` constraint (the cost vector is
defined over all units, not per arm). When ``arm`` is ``None`` (default), a
single :class:`SYNDESResults` is returned, exactly as before.

Example
-------

``SYNDES`` takes a long balanced panel and a pre/post split (``post_col`` or
``T0``). The example below is self-contained -- it generates a small panel and
runs end to end (``pyscipopt`` ships with ``mlsynth``, so the ``SCIP`` solver is
available on install). The same call shape serves all three designs.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from mlsynth import SYNDES, power_analysis

   # A small balanced panel: 8 units, 20 periods (last 6 are post-treatment).
   rng = np.random.default_rng(0)
   n_units, n_periods, n_post = 8, 20, 6
   factors = rng.normal(size=(n_periods, 2))
   loadings = rng.uniform(0.3, 1.0, size=(n_units, 2))
   level = rng.uniform(8.0, 12.0, size=n_units)          # positive unit baselines
   Y = level + factors @ loadings.T + rng.normal(scale=0.3, size=(n_periods, n_units))
   df = pd.DataFrame(
       [{"unit": j, "time": t, "Y": float(Y[t, j]),
         "post": int(t >= n_periods - n_post)}
        for j in range(n_units) for t in range(n_periods)]
   )

   res = SYNDES({
       "df": df, "outcome": "Y", "unitid": "unit", "time": "time",
       "K": 3, "mode": "two_way_global", "post_col": "post",
       "run_inference": True, "alpha": 0.05, "solver": "SCIP",
   }).fit()

   print(res.design.selected_unit_labels)   # which units to treat
   print(res.design.control_weights)        # synthetic-control weights
   print(res.design.pre_fit_rmse)           # pre-period balance of the design
   print(res.inference.atet, res.inference.p_value)

   mde = power_analysis(res, n_post_periods=[4, 8, 12], power=0.80)
   print(mde.to_dataframe())                # minimum detectable effect by horizon

A budget constraint (``costs`` + ``budget``) adds
:math:`\sum_i \mathrm{cost}_i D_i \le B` to the MIP; ``mode="two_way_global"``
also accepts ``K=None`` to let the program choose the number of treated units.

Solution pool (``top_K``): a menu, not one answer
-------------------------------------------------

The MIP returns the single MSE-optimal design, but that is optimal for *fit
alone* -- it discards every other feasible design, some of which may be cheaper,
more detectable, or operationally preferable at a negligible fit cost. Setting
``top_K > 1`` returns a ranked pool of the best ``top_K`` distinct designs,
obtained by *no-good cuts*: after each solve the chosen treated set
:math:`S` is forbidden (:math:`\sum_{i \in S} D_i \le |S|-1`) and the MIP is
re-solved for the next-best design. The pool is attached as ``results.pool`` --
a list of dicts ranked by MSE, each with its ``markets``, ``objective`` (MSE),
``pre_fit_rmse``, ``mde_pct`` (the same permutation-null MDE
:func:`~mlsynth.power_analysis` uses), and ``cost`` (when ``costs`` is given).
Because the objective only ranks fit, the value is precisely the re-scoring on
the dimensions it ignored: a manager can trade a small fit increase for lower
cost or higher power. ``top_K=1`` (default) is unchanged -- only the optimum is
returned and ``results.pool`` is ``None``.

.. code-block:: python

   res = SYNDES({"df": df, "outcome": "y", "unitid": "unit", "time": "time",
                 "K": 3, "mode": "two_way_global", "top_K": 5}).fit()
   for d in res.pool:                        # ranked menu, best fit first
       print(d["markets"], round(d["objective"], 1), round(d["mde_pct"], 2), d["cost"])

Verification
------------

.. note::

   Simulation (all three designs). Following the paper's Section 5, each
   replication draws a fresh noisy panel (stationary AR(1) factors + unit
   levels), re-solves the design MIP on the pre-period, estimates the ATT on
   the post-period and runs the moving-block permutation test. Setup:
   :math:`N=10` units, :math:`T_{\text{pre}}=18`, :math:`T_{\text{post}}=6`,
   :math:`K=3`, :math:`\sigma=0.25`, 40 replications; the effect is injected at
   :math:`\tau` equal to the mean analytic MDE (0.165). Rejection at the 5%
   level:

   .. list-table::
      :header-rows: 1
      :widths: 26 12 12 12 10 10

      * - design
        - MDE
        - bias
        - RMSE
        - size
        - power
      * - ``per_unit``
        - 0.157
        - 0.020
        - 0.098
        - 0.12
        - 0.50
      * - ``two_way_global``
        - 0.166
        - 0.013
        - 0.095
        - 0.12
        - 0.50
      * - ``one_way_global``
        - 0.171
        - -0.004
        - 0.115
        - 0.23
        - 0.45
      * - random DiM (baseline)
        - --
        - 0.096
        - 0.982
        - 0.15
        - 0.25

   The paper's headline result reproduces: all three SYNDES designs are
   approximately unbiased and cut estimator RMSE roughly ten-fold versus
   a randomized difference-in-means design (``~0.10`` vs. ``0.98``). The
   moving-block permutation test is mildly over-sized / under-powered at this
   short pre-period -- the design-optimized contrast tightens the pre-period
   permutation null, and the analytic MDE is a normal-theory benchmark -- a
   finite-sample inference caveat (the authors note correct sizes hold "under
   rather strong assumptions") that shrinks as the pre-period grows. The
   simulation script ships alongside the estimator's tests.

A second, data-faithful replication reproduces the paper's own Monte Carlo
(Section 5, Table 1) on the exact BLS unemployment panel the authors use
(``basedata/urate_cps.csv``, footnote 4): each simulation samples a 10×10 panel,
the design selects :math:`K \in \{3, 7\}` treated units on the pre-period, a
homogeneous ``0.05`` effect is added to the treated post-periods, and the ATET
RMSE (×1000) is compared to Table 1. mlsynth's three design modes land at
``8.7 / 9.2 / 9.2`` (``K=3``) and ``7.7 / 9.0 / 9.0`` (``K=7``) against the
paper's ``8.5 / 8.4 / 8.5`` and ``8.3 / 8.4 / 8.5``, and all beat the randomized
difference-in-means baseline — the paper's headline. See
:doc:`replications/syndes`; run it with
``python benchmarks/run_benchmarks.py syndes_bls``.

Core API
--------

.. automodule:: mlsynth.estimators.syndes
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.SYNDESConfig
   :members:
   :undoc-members:

Result Containers
-----------------

``SYNDES.fit()`` returns a
:class:`~mlsynth.utils.syndes_helpers.structures.SYNDESResults`, bundling the
optimized :class:`~mlsynth.utils.syndes_helpers.structures.SYNDESDesign`
(assignment, treated/control/contrast weights, the pre-period
``contrast_series`` and ``pre_fit_rmse``, objective value), the prepared
:class:`~mlsynth.utils.syndes_helpers.structures.SYNDESInputs`, and optional
:class:`~mlsynth.utils.syndes_helpers.structures.SYNDESInference`. The
``mode="two_way_global_annealed"`` path instead returns a
:class:`~mlsynth.utils.syndes_helpers.relaxed_structures.RelaxedSolverResults`.

.. automodule:: mlsynth.utils.syndes_helpers.structures
   :members:
   :undoc-members:
   :show-inheritance:

In addition, ``SYNDES.fit()`` attaches a
:class:`~mlsynth.utils.post_fit.SyntheticControlPostFit` as
``results.post_fit`` — the standardized diagnostics container shared with
the rest of the MAREX family (LEXSCM / MAREX / PANGEO). It carries the ATE
/ total effect / percentage lift / per-period gap, pre- and post-period
RMSEs, the inference triple (p-value, CI), and a
:class:`~mlsynth.utils.post_fit.PowerAnalysis` block with the headline MDE
and the MDE-versus-horizon curve.

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

Data preparation -- the only DataFrame touchpoint: pivots to wide pre/post
matrices and builds the unit/time ``IndexSet``\es.

.. automodule:: mlsynth.utils.syndes_helpers.setup
   :members:
   :undoc-members:

The CVXPY objective/constraint builders for the three MIP formulations.

.. automodule:: mlsynth.utils.syndes_helpers.formulation
   :members:
   :undoc-members:

The solver wrapper: builds the MIP, applies optional budget constraints, solves,
and extracts the assignment, weights, and pre-period prediction.

.. automodule:: mlsynth.utils.syndes_helpers.optimization
   :members:
   :undoc-members:

The moving-block permutation test (shared contrast dispatch across modes).

.. automodule:: mlsynth.utils.syndes_helpers.inference
   :members:
   :undoc-members:

The minimum-detectable-effect power analysis (Newey-West long-run SE).

.. automodule:: mlsynth.utils.syndes_helpers.power
   :members:
   :undoc-members:

Standardized post-fit (shared across the MAREX family) — the
:func:`~mlsynth.utils.post_fit.compute_post_fit` /
:func:`~mlsynth.utils.post_fit.compute_power_analysis` /
:func:`~mlsynth.utils.post_fit.compute_smd` helpers that populate
``res.post_fit`` live outside this package so LEXSCM, MAREX, and PANGEO
all consume the same diagnostics machinery:

.. automodule:: mlsynth.utils.post_fit
   :members:
   :undoc-members:
   :show-inheritance:
