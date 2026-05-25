Synthetic Design (SYNDES)
=========================

.. currentmodule:: mlsynth

When to Use This Estimator
--------------------------

Most synthetic-control work takes the treated unit as *given* and asks only how
to weight the donors. ``SYNDES`` answers the prior question Doudchenko et al.
[SYNDES]_ pose: when you are about to **run an experiment** and have
pre-treatment outcome data, *which units should you treat*? Treating units at
random -- or by hand -- leaves accuracy on the table, because the variance of
the resulting treatment-effect estimate depends on which units are treated and
how the rest are weighted into a synthetic comparison.

The authors argue this is exactly the regime of **market-level experiments**:
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

We observe an outcome :math:`Y_{it}` for units :math:`i = 1, \ldots, N` over
pre-treatment periods :math:`t = 1, \ldots, T`. At :math:`t = T` the
experimenter assigns a binary treatment :math:`D_i \in \{0, 1\}` to be applied
over the :math:`S - T` post-treatment periods, with **exactly** ``K`` treated
units (:math:`\sum_i D_i = K`). Each unit has potential outcomes
:math:`(Y_{it}(0), Y_{it}(1))` and observed outcome
:math:`Y_{it} = Y_{it}(D_i)`. Synthetic weights :math:`w` live on the simplex
(non-negative, summing to one on the relevant side). The estimand is the
**weighted average treatment effect on the treated** (wATET)
:math:`\tau = \sum_{i:D_i=1} w_i \tau_i`, where :math:`\tau_i` is unit
:math:`i`'s additive effect.

.. note::

   Notation bridge. The single-treated-unit synthetic-control canon (treated
   :math:`j=0`, donors :math:`1,\ldots,N`) does not fit a *design* problem with
   ``K`` chosen treated units, so we follow the paper's convention: units are
   indexed :math:`i`, the assignment vector :math:`D` is itself a decision
   variable, and :math:`T` denotes the pre-treatment length.

The design problem
~~~~~~~~~~~~~~~~~~

Under the outcome model :math:`Y_{it}(0) = \mu_{it} + \varepsilon_{it}` with
mean-zero, homoskedastic noise (:math:`\operatorname{Var}\varepsilon_{it} =
\sigma^2`) and additive effects :math:`Y_{it} = Y_{it}(0) + D_i \tau_i`, the
conditional MSE of the per-unit synthetic-control estimator
:math:`\hat\tau_i = Y_{i,T+1} - \sum_{j:D_j=0} w^i_j Y_{j,T+1}` is

.. math::

   \mathbb{E}\bigl[(\hat\tau_i - \tau_i)^2 \mid D, w\bigr]
   = \Bigl(\mu_{i,T+1} - \textstyle\sum_{j:D_j=0} w^i_j \mu_{j,T+1}\Bigr)^2
     + \sigma^2\Bigl(1 + \textstyle\sum_{j:D_j=0} (w^i_j)^2\Bigr).

The first term is a **bias** from imperfect pre-treatment matching; the second a
**variance** that grows with the weight concentration. SYNDES minimizes the
empirical, pre-period analogue of this MSE jointly over :math:`(D, w)` -- the
:math:`\sigma^2 \sum w^2` term becomes the ridge penalty :math:`\lambda` below.
Because the choice of treated set makes the estimand itself stochastic, the
target is the wATET *for the units SYNDES selects*, not a fixed population ATE.

The three MIP formulations
--------------------------

The joint optimization over assignment and weights is a **bilevel /
mixed-integer program** and is NP-hard. Doudchenko et al. give three forms,
all exposed through ``mode`` and all solved as MIPs (auxiliary variables
linearize the weight-assignment products). They differ in *how the treated and
control sides are weighted*.

Per-unit (``mode="per_unit"``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A **separate** synthetic control for every treated unit:

.. math::

   \min_{D, \{w^i_j\}} \;
     \frac{1}{KT} \sum_{i} \sum_{t} D_i
       \Bigl(Y_{it} - \textstyle\sum_j w^i_j (1 - D_j) Y_{jt}\Bigr)^2
     + \frac{\lambda}{K} \sum_i \sum_j D_i (w^i_j)^2,

subject to :math:`w^i_j \ge 0`, :math:`\sum_i D_i = K`, and
:math:`\sum_j w^i_j(1 - D_j) = 1` for each treated :math:`i`. Each treated unit
draws its own simplex of control weights.

Two-way global (``mode="two_way_global"``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A **single** weight vector applied to both sides of one global contrast:

.. math::

   \min_{D, \{w_i\}} \;
     \frac{1}{T} \sum_t \Bigl(\textstyle\sum_i w_i D_i Y_{it}
       - \sum_i w_i (1 - D_i) Y_{it}\Bigr)^2 + \lambda \sum_i w_i^2,

subject to :math:`w_i \ge 0`, :math:`\sum_i D_i = K`,
:math:`\sum_i w_i D_i = 1` and :math:`\sum_i w_i (1 - D_i) = 1`. ``mlsynth``
linearizes :math:`q_i = w_i D_i` and enforces the two normalizations with
:math:`\sum_i q_i = 1`, :math:`\sum_i w_i = 2`, so the per-period contrast is
:math:`(2q - w)^\top Y_t`.

One-way global (``mode="one_way_global"``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The two-way program with the **treated weights pinned equal** (a simple average,
:math:`w_i = 1/K` on the treated), while the **control side stays a free
synthetic control**:

.. math::

   \min_{D, c} \;
     \frac{1}{T}\sum_t \Bigl(\tfrac{1}{K}\textstyle\sum_i D_i Y_{it}
       - \sum_i c_i Y_{it}\Bigr)^2 + \lambda\Bigl(\tfrac{1}{K} + \textstyle\sum_i c_i^2\Bigr),

subject to :math:`c_i \ge 0`, :math:`\sum_i c_i = 1`, :math:`c_i \le 1 - D_i`
(treated units carry no control weight) and :math:`\sum_i D_i = K`.

.. warning::

   One-way global is **not** difference-in-means. Only the *treated* side is
   fixed at :math:`1/K`; the control side ``c`` is a free synthetic control to
   be optimized. Pinning *both* sides (treated :math:`1/K`, control
   :math:`1/(N-K)`) would be the randomized difference-in-means baseline, a
   different (and weaker) design.

**Assumptions / Remarks.**

*Assumption 1 (additive effects, homoskedastic noise).* Outcomes follow
:math:`Y_{it}(0) = \mu_{it} + \varepsilon_{it}` with
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
:math:`\tau`, so the **global** modes can choose weights freely to minimize MSE.
When effects are heterogeneous, different weightings target different estimands;
**per_unit** (or the fixed-treated-weight **one_way_global**) keeps the estimand
well-defined. *Remark.* This is the authors' guidance for choosing a mode -- it
is about which wATET you are willing to target, not just fit.

*Assumption 4 (sharp null for inference).* The permutation test targets the
sharp null :math:`\tau_i = 0` for all treated :math:`i`. *Remark.* Correct test
size holds under the exchangeability the moving-block permutation imposes; the
authors note this requires "rather strong assumptions" in finite samples.

.. _syndes-inference:

Inference and minimum detectable effect
---------------------------------------

For any mode the fitted design yields a unit-level **contrast vector** ``c`` such
that the ATT estimate at period :math:`t` is :math:`Y_t^\top c` (treated weights
minus control weights; for ``per_unit`` the :math:`K` per-unit estimators are
averaged). ``SYNDES`` tests the sharp null with the moving-block permutation
test of Chernozhukov, Wuethrich and Zhu (2021): the post-period mean contrast is
compared to the distribution obtained by cyclically shifting the stacked panel.

For design-time **power**, :func:`~mlsynth.power_analysis` returns a per-horizon
minimum detectable effect (MDE). Because the moving-block test averages a
contrast over correlated periods, the relevant null standard error is the
**Newey-West (Bartlett HAC) long-run** std of the per-period contrast, not the
i.i.d. :math:`\sigma_{\text{perm}}/\sqrt{n_{\text{post}}}`:

.. math::

   \mathrm{MDE}(n_{\text{post}}) = (z_{1-\alpha/2} + z_{1-\beta})\,
       \frac{\hat\sigma_{\mathrm{LR}}}{\sqrt{n_{\text{post}}}},

reported as ``long_run_sigma``. It reduces to the textbook formula when the
contrast series is serially uncorrelated.

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

Example
-------

``SYNDES`` takes a long balanced panel and a pre/post split (``post_col`` or
``T0``). The same call shape serves all three designs:

.. code-block:: python

   from mlsynth import SYNDES, power_analysis

   res = SYNDES({
       "df": df, "outcome": "Y", "unitid": "unit", "time": "time",
       "K": 3, "mode": "two_way_global", "post_col": "post",
       "run_inference": True, "alpha": 0.05, "solver": "SCIP",
   }).fit()

   res.design.selected_unit_labels   # which units to treat
   res.design.control_weights        # synthetic-control weights
   res.design.pre_fit_rmse           # pre-period balance of the design
   res.inference.atet, res.inference.p_value

   mde = power_analysis(res, n_post_periods=[4, 8, 12], power=0.80)
   mde.to_dataframe()                # minimum detectable effect by horizon

A budget constraint (``costs`` + ``budget``) adds
:math:`\sum_i \text{cost}_i D_i \le B` to the MIP; ``mode="two_way_global"``
also accepts ``K=None`` to let the program choose the number of treated units.

Verification
------------

.. note::

   **Simulation (all three designs).** Following the paper's Section 5, each
   replication draws a fresh noisy panel (stationary AR(1) factors + unit
   levels), **re-solves** the design MIP on the pre-period, estimates the ATT on
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
   approximately **unbiased** and cut estimator **RMSE roughly ten-fold** versus
   a randomized difference-in-means design (``~0.10`` vs. ``0.98``). The
   moving-block permutation test is mildly **over-sized / under-powered** at this
   short pre-period -- the design-optimized contrast tightens the pre-period
   permutation null, and the analytic MDE is a normal-theory benchmark -- a
   finite-sample inference caveat (the authors note correct sizes hold "under
   rather strong assumptions") that shrinks as the pre-period grows. The
   simulation script ships alongside the estimator's tests.

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

Helper Modules
--------------

Data preparation -- the only DataFrame touchpoint: pivots to wide pre/post
matrices and builds the unit/time ``IndexSet``es.

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
