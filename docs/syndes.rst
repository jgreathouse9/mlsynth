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

The test and the power curve are both functionals of a single scalar series:
the per-period contrast the design induces. Any fitted design (any mode) reduces
to a unit-level contrast vector :math:`\mathbf{c} \in \mathbb{R}^N`, and the
estimated treatment effect at period :math:`t` is its projection onto the
cross-section,

.. math::

   g_t \;\coloneqq\; \mathbf{y}_t^\top \mathbf{c}, \qquad
   \widehat{\tau} \;=\; \frac{1}{|\mathcal{T}_2|}\sum_{t \in \mathcal{T}_2} g_t .

For the two global modes :math:`\mathbf{c} = 2\mathbf{q} - \mathbf{w}` (per-unit
treated weight minus control weight); for ``per_unit`` the :math:`K` per-unit
synthetic-control estimators are averaged, giving
:math:`c_j = (D_j - \sum_i q_{ij})/K`. Because the design balances the treated
and synthetic-control sides over the pre-period, under the sharp null
:math:`\tau_i \equiv 0` the series has mean zero, :math:`\mathbb{E}[g_t]=0`, and
its pre-period fluctuations are a direct readout of the estimator's noise. Every
quantity below is computed from :math:`\{g_t\}`.

The moving-block permutation test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``SYNDES`` tests :math:`H_0: \tau_i \equiv 0` with the moving-block permutation
test of Chernozhukov, Wuethrich and Zhu (2021). The observed statistic is the
post-period mean :math:`\widehat{\tau}`; the reference distribution is built by
cyclically shifting the stacked contrast series and recomputing the
length-:math:`|\mathcal{T}_2|` block mean at every offset. The two-sided
:math:`p`-value is the share of shifted block means at least as large in
magnitude as :math:`\widehat{\tau}`, and the test is exact under the
exchangeability the cyclic shift imposes.

Where the per-period standard error comes from
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The test compares a length-:math:`h` block mean of :math:`g_t` to its null
distribution, so the standard error that governs detectability at horizon
:math:`h \coloneqq n_{\text{post}}` is the standard deviation of that block
mean. It is estimated on the pre-period contrast
:math:`\{g_t\}_{t\in\mathcal{T}_1}` -- the only periods untouched by the
post-period effect. The i.i.d. per-period scale is the sample SD

.. math::

   \widehat{\sigma}_{\text{perm}}
     = \sqrt{\frac{1}{T_0 - 1}\sum_{t\in\mathcal{T}_1}\bigl(g_t - \bar g\bigr)^2},

reported as ``sigma_perm``. Were the :math:`g_t` serially independent, the mean
of :math:`h` of them would have standard error
:math:`\widehat{\sigma}_{\text{perm}}/\sqrt{h}` -- the textbook scaling.

Serial correlation matters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Contrast residuals are almost never independent: the weighting absorbs the level
but persistent factors (seasonality, business cycles, slow trends) leak through
-- exactly the correlation the moving-block test is exposed to. The variance of
a length-:math:`h` mean is then :math:`\sigma_{\mathrm{LR}}^2/h`, not
:math:`\sigma_{\text{perm}}^2/h`, where :math:`\sigma_{\mathrm{LR}}^2` is the
long-run variance. :func:`~mlsynth.power_analysis` estimates it with a
Newey-West (Bartlett-kernel) HAC estimator on the pre-period contrast,

.. math::

   \widehat{\sigma}_{\mathrm{LR}}^2
     = \widehat{\gamma}_0
       + 2\sum_{k=1}^{L}\Bigl(1 - \frac{k}{L+1}\Bigr)\widehat{\gamma}_k,
   \qquad
   \widehat{\gamma}_k
     = \frac{1}{T_0}\sum_{t=k+1}^{T_0}(g_t-\bar g)(g_{t-k}-\bar g),

with automatic bandwidth :math:`L = \lfloor 4\,(T_0/100)^{2/9}\rfloor`. The
Bartlett weights :math:`1 - k/(L+1)` keep the estimate non-negative, and
:math:`\widehat{\sigma}_{\mathrm{LR}}` (reported as ``long_run_sigma``)
collapses to :math:`\widehat{\sigma}_{\text{perm}}` when the contrast is white
noise. When a short horizon leaves too few overlapping blocks the estimator
falls back to the i.i.d. scaling.

The per-horizon MDE
~~~~~~~~~~~~~~~~~~~~

Collecting terms, the null standard error of the ATT estimator at horizon
:math:`h` is :math:`\mathrm{SE}(h) = \widehat{\sigma}_{\mathrm{LR}}/\sqrt{h}`,
and for a two-sided level-:math:`\alpha` test with target power
:math:`1-\beta` the minimum detectable effect -- the smallest true ATT the
design would reject :math:`H_0` for with probability :math:`1-\beta` -- is

.. math::

   \mathrm{MDE}(h) = \bigl(z_{1-\alpha/2} + z_{1-\beta}\bigr)\,
       \frac{\widehat{\sigma}_{\mathrm{LR}}}{\sqrt{h}},
   \qquad
   \mathrm{MDE}_\%(h) = 100 \times \frac{\mathrm{MDE}(h)}{\text{baseline}},

with ``baseline`` the mean pre-period outcome over the SYNDES-selected treated
units by default (alternatives: ``"overall"``, ``"control"``, or a user scalar).
This is exactly the number tabulated for horizons :math:`h = 1,\dots,12` in
``res.power_curve`` and returned by :func:`~mlsynth.power_analysis`. The power to
detect a given true effect :math:`\tau` at horizon :math:`h` uses the same SE,

.. math::

   \pi(\tau, h) = \Phi\!\Bigl(\frac{|\tau|}{\mathrm{SE}(h)} - z_{1-\alpha/2}\Bigr)
                 + \Phi\!\Bigl(-\frac{|\tau|}{\mathrm{SE}(h)} - z_{1-\alpha/2}\Bigr).

Two estimators of the same standard error
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``mlsynth`` exposes two SYNDES power surfaces. They share the contrast series
:math:`\{g_t\}` and the multiplier :math:`z_{1-\alpha/2}+z_{1-\beta}`, and
differ only in how they turn the pre-period contrast into :math:`\mathrm{SE}(h)`:

* ``res.power_curve`` and :func:`~mlsynth.power_analysis` -- the nonparametric
  Newey-West HAC above, :math:`\mathrm{SE}(h)=\widehat{\sigma}_{\mathrm{LR}}/\sqrt{h}`,
  tabulated per horizon :math:`h=1,\dots,12`.
* ``res.post_fit.power`` -- the family-wide parametric surface, which models the
  contrast as an AR(1) and uses
  :math:`\mathrm{SE}(h)=\widehat{\sigma}_{\text{placebo}}\sqrt{\mathrm{VIF}(h,\widehat{\rho})}`
  (next section).

Both collapse to the textbook
:math:`(z_{1-\alpha/2}+z_{1-\beta})\,\widehat{\sigma}/\sqrt{h}` when the contrast
is serially uncorrelated; they differ only in how the serial-correlation
correction is extrapolated (a nonparametric lag sum versus a one-parameter AR(1)
fit).

Standardized Post-Fit and Power Analysis
----------------------------------------

Every call to :meth:`SYNDES.fit` attaches a
:class:`~mlsynth.utils.post_fit.SyntheticControlPostFit` to ``res.post_fit``
— the same diagnostic surface used by LEXSCM, MAREX, and PANGEO. This is
the one-stop container downstream consumers (dashboards, paper-style reports,
comparison tables) read from, regardless of which member of the family
produced the design:

.. code-block:: python

   import numpy as np
   import pandas as pd
   from mlsynth import SYNDES

   rng = np.random.default_rng(0)
   n_units, n_periods, n_post = 8, 20, 6
   factors = rng.normal(size=(n_periods, 2))
   loadings = rng.uniform(0.3, 1.0, size=(n_units, 2))
   level = rng.uniform(8.0, 12.0, size=n_units)
   Y = level + factors @ loadings.T + rng.normal(scale=0.3, size=(n_periods, n_units))
   df = pd.DataFrame(
       [{"unit": j, "time": t, "Y": float(Y[t, j]),
         "post": int(t >= n_periods - n_post)}
        for j in range(n_units) for t in range(n_periods)]
   )
   res = SYNDES({"df": df, "outcome": "Y", "unitid": "unit", "time": "time",
                 "K": 3, "mode": "two_way_global", "post_col": "post",
                 "run_inference": True}).fit()

   pf = res.post_fit                            # SyntheticControlPostFit
   print(pf.ate, pf.ate_percent, pf.total_effect)   # treatment-effect scalars
   print(pf.rmse_fit, pf.rmse_post)                 # pre / post fit quality
   print(pf.p_value, pf.ci_lower, pf.ci_upper)      # permutation inference
   print(pf.power)                                  # PowerAnalysis (see below)

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

   import numpy as np
   import pandas as pd
   from mlsynth import SYNDES

   rng = np.random.default_rng(0)
   n_units, n_periods, n_post = 8, 20, 6
   factors = rng.normal(size=(n_periods, 2))
   loadings = rng.uniform(0.3, 1.0, size=(n_units, 2))
   level = rng.uniform(8.0, 12.0, size=n_units)
   Y = level + factors @ loadings.T + rng.normal(scale=0.3, size=(n_periods, n_units))
   df = pd.DataFrame(
       [{"unit": j, "time": t, "Y": float(Y[t, j]),
         "post": int(t >= n_periods - n_post)}
        for j in range(n_units) for t in range(n_periods)]
   )
   res = SYNDES({"df": df, "outcome": "Y", "unitid": "unit", "time": "time",
                 "K": 3, "mode": "two_way_global", "post_col": "post",
                 "run_inference": True}).fit()

   p = res.post_fit.power                        # PowerAnalysis
   print(p.headline.mde_absolute)                # MDE at the realised T_post
   print(p.headline.mde_pct)                     # ... as % of post-period baseline
   print(p.headline.power_at_observed)           # power to detect res.post_fit.ate
   print(p.curve)                                # tuple of MDEPoint per horizon

A second, per-horizon table comes back from every fit by default as
``res.power_curve`` -- a :class:`~mlsynth.SYNDESPower` over post-period horizons
``1..12``, so you can read the minimum-detectable-effect curve without a
separate :func:`~mlsynth.power_analysis` call:

.. code-block:: python

   import pandas as pd
   from mlsynth import SYNDES

   rng = __import__("numpy").random.default_rng(0)
   n_units, n_periods, n_post = 8, 20, 6
   Y = (rng.uniform(8.0, 12.0, n_units)
        + rng.normal(size=(n_periods, 2)) @ rng.uniform(0.3, 1.0, (n_units, 2)).T
        + rng.normal(scale=0.3, size=(n_periods, n_units)))
   df = pd.DataFrame(
       [{"unit": j, "time": t, "Y": float(Y[t, j]),
         "post": int(t >= n_periods - n_post)}
        for j in range(n_units) for t in range(n_periods)]
   )
   res = SYNDES({"df": df, "outcome": "Y", "unitid": "unit", "time": "time",
                 "K": 3, "mode": "two_way_global", "post_col": "post"}).fit()

   print(res.power_curve.to_dataframe())         # n_post 1..12, mde_absolute, mde_percent

Power-analysis failures (e.g. degenerate pre-period contrast) never break a
fit; ``res.post_fit.power`` and ``res.power_curve`` are simply left as ``None``
in that case. For a custom horizon grid, significance level, or baseline, call
:func:`~mlsynth.power_analysis` (per-horizon table) or
:func:`~mlsynth.utils.post_fit.compute_power_analysis` (the headline AR(1)
surface) directly.

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

   import numpy as np
   import pandas as pd
   from mlsynth import SYNDES

   # A small balanced panel: 8 units, 20 periods (last 6 are post-treatment).
   rng = np.random.default_rng(0)
   n_units, n_periods, n_post = 8, 20, 6
   factors = rng.normal(size=(n_periods, 2))
   loadings = rng.uniform(0.3, 1.0, size=(n_units, 2))
   level = rng.uniform(8.0, 12.0, size=n_units)
   Y = level + factors @ loadings.T + rng.normal(scale=0.3, size=(n_periods, n_units))
   df = pd.DataFrame(
       [{"unit": j, "time": t, "Y": float(Y[t, j]),
         "post": int(t >= n_periods - n_post)}
        for j in range(n_units) for t in range(n_periods)]
   )

   base = {"df": df, "outcome": "Y", "unitid": "unit", "time": "time",
           "K": 3, "mode": "two_way_global", "post_col": "post"}

   # Default: 5% gap, 60s wall-clock — production-suitable.
   SYNDES(base).fit()

   # Loosen the gap to return in seconds when you just need a
   # plausible design for prototyping.
   SYNDES({**base, "gap_limit": 0.25, "time_limit": 5.0}).fit()

   # Disable both limits for an asymptotic-optimality run. Be
   # prepared for hours-long solves on long panels.
   SYNDES({**base, "gap_limit": None, "time_limit": None}).fit()

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

   import numpy as np
   import pandas as pd
   from mlsynth import SYNDES

   # Two arms (A, B), each with 6 markets over 20 periods (last 6 post).
   rng = np.random.default_rng(0)
   n_per_arm, n_periods, n_post = 6, 20, 6
   rows = []
   for arm in ("A", "B"):
       factors = rng.normal(size=(n_periods, 2))
       for j in range(n_per_arm):
           loading = rng.uniform(0.3, 1.0, size=2)
           level = rng.uniform(8.0, 12.0)
           series = level + factors @ loading + rng.normal(scale=0.3, size=n_periods)
           for t in range(n_periods):
               rows.append({"DMA": f"{arm}{j}", "week": t, "sales": float(series[t]),
                            "treat": arm, "post": int(t >= n_periods - n_post)})
   df = pd.DataFrame(rows)

   res = SYNDES({
       "df": df, "outcome": "sales", "unitid": "DMA", "time": "week",
       "arm": "treat",                 # categorical arm label per unit
       "K": 3, "mode": "two_way_global", "post_col": "post",
       "run_inference": True,
   }).fit()

   print(res.arm_designs["A"])              # full SYNDESResults for arm A
   print(res.atet_by_arm())                 # {arm: ATET}
   print(res.selected_unit_labels_by_arm()) # {arm: treated units}

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
a list of dicts ranked by MSE. Each entry is *actionable*, not merely rankable:
it carries everything needed to deploy that design, not just the rank-1 winner
kept on ``results.design``. Its keys are:

* ``markets`` -- labels of the treated units (the design's arms).
* ``control_group`` -- labels of the donor units carrying nonzero control
  weight (the synthetic-control pool backing the treated arms).
* ``objective`` -- the MIP objective (fit) the design was ranked by.
* ``pre_fit_rmse`` -- root-mean-square pre-period contrast.
* ``mde_pct`` -- minimum detectable effect at the realised post horizon, as a
  percent of the treated baseline. This is the entry's ``power_curve`` value at
  that horizon (the Newey-West HAC MDE), so the headline number and the curve
  always agree.
* ``cost`` -- summed cost of the treated units (``None`` when no ``costs`` given).
* ``design`` -- the full :class:`~mlsynth.utils.syndes_helpers.structures.SYNDESDesign`
  for the entry, with its treated, control, and contrast weights.
* ``power_curve`` -- the entry's own :class:`~mlsynth.SYNDESPower` over horizons
  ``1..12`` (``None`` if the computation is degenerate), so every candidate is
  comparable on power, not just the rank-1 winner on ``results.power_curve``.

Because the objective only ranks fit, the value is precisely the re-scoring on
the dimensions it ignored: a manager can trade a small fit increase for lower
cost or higher power, then read the chosen entry's ``control_group`` and
``design`` weights to deploy it. ``top_K=1`` (default) is unchanged -- only the
optimum is returned and ``results.pool`` is ``None``.

The example below imports a subset of the GeoLift pre-test panel (the same
40-market data the :doc:`geolift` page uses) and returns a five-design menu:

.. code-block:: python

   import pandas as pd
   from mlsynth import SYNDES

   df = pd.read_csv(                                      # GeoLift_PreTest, 40 mkts x 90d
       "https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
       "refs/heads/main/basedata/geolift_market_data.csv"
   )
   markets = sorted(df["location"].unique())[:8]         # 8-market subset keeps the MIP small
   df = df[df["location"].isin(markets)].copy()
   cut = sorted(df["date"].unique())[-14]                # last 14 days = post window
   df["post"] = (df["date"] >= cut).astype(int)

   # top_K=5 runs five MIPs (one per no-good cut), so expect ~a minute on SCIP.
   res = SYNDES({
       "df": df, "outcome": "Y", "unitid": "location", "time": "date",
       "K": 3, "mode": "two_way_global", "post_col": "post", "top_K": 5,
       "gap_limit": 0.2, "time_limit": 10.0,
   }).fit()

   print(sorted(res.design.selected_unit_labels.tolist()))   # the MSE-optimal design
   for d in res.pool:                                         # ranked menu, best fit first
       print("treated:", sorted(d["markets"]),
             "| control:", sorted(d["control_group"]),        # donors backing the SC
             "| obj:", round(d["objective"], 1),
             "| mde%:", round(d["mde_pct"], 3))
   # any entry is deployable: read its control group and weights, not just rank-1
   chosen = res.pool[1]
   print(chosen["control_group"], chosen["design"].control_weights)

The lesson is the whole point of the menu: the rank-1 design minimises MSE, but
fit is not power. On this subset the best-fitting design is not the most
detectable — a design ranked further down the pool, with an essentially
identical objective, can carry a materially smaller minimum detectable effect
(or a lower ``cost`` once ``costs`` is supplied). The menu lets you choose the
design that agrees with what you will actually deploy, instead of the one number
the optimiser happens to minimise.

Out-of-sample selection (``holdout_frac``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The MIP ranks designs by the *in-sample* pre-period contrast, so the winning
treated set is the one that balances best over the periods it was fit on. When
the treated unit moves with the donor pool only transiently, that tight balance
need not persist, and the in-sample optimum can overfit. Setting ``holdout_frac``
turns selection into a train/validate procedure, in the spirit of the holdout
split LEXSCM and MAREX use: the ``top_K`` candidate pool is learned on the
leading :math:`1 - \texttt{holdout\_frac}` of the pre-period, and the winner is
the candidate whose *held-out* contrast error on the trailing
``holdout_frac`` is smallest. For example, ``holdout_frac=0.3`` is a 70/30
split; the out-of-sample error of a design with contrast weights
:math:`\mathbf{c}` is :math:`\sqrt{\operatorname{mean}((\mathbf{Y}^{\text{val}}
\mathbf{c})^2)}`, the validation-tail analogue of ``pre_fit_rmse``.

The returned ``results.pool`` is then ranked by this out-of-sample error (rank-1
is the holdout winner kept on ``results.design``), and each entry carries an
``oos_rmse`` key alongside the in-sample ``pre_fit_rmse``. Holdout selection
needs a candidate pool to choose among, so ``top_K >= 2`` is required, and it
applies to the MIP modes only (not the annealed relaxation). Power and inference
are computed exactly as in the in-sample path. ``holdout_frac=None`` (the
default) leaves selection on the in-sample optimum, unchanged.

.. code-block:: python

   res = SYNDES({
       "df": df, "outcome": "Y", "unitid": "location", "time": "date",
       "K": 3, "mode": "two_way_global", "post_col": "post",
       "top_K": 5, "holdout_frac": 0.3,            # 70/30 train/validate
       "gap_limit": 0.2, "time_limit": 10.0,
   }).fit()

   for d in res.pool:                              # ranked by out-of-sample error
       print("treated:", sorted(d["markets"]),
             "| oos:", round(d["oos_rmse"], 3),
             "| in-sample:", round(d["pre_fit_rmse"], 3))

In the cross-method comparison, :func:`~mlsynth.compare_methods` defaults to this
holdout selection for SYNDES (``syndes_holdout_frac=0.3``), adds an ``oos_rmse``
column to the comparison table, and orders the SYNDES rows by it; pass
``syndes_holdout_frac=None`` to compare in-sample designs instead.

Information-criterion selection (``selection="ic"``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The holdout split spends part of the pre-period on validation, which is noisy
exactly when the pre-period is short -- the regime SYNDES is built for. An
information criterion avoids the split: it scores every candidate on the *whole*
pre-window and penalises the model-selection flexibility directly. Pouliot, Xie
& Liu (2024) show that in short-:math:`T_0` synthetic-control settings such a
criterion outperforms cross-validation / holdout. Setting ``selection="ic"``
ranks the ``top_K`` pool (solved on the full pre-period) by

.. math::

   \mathrm{IC}(d) = \mathrm{SSR}^{\text{pre}}(d)
                  + 2\,\hat{\sigma}^2\,\mathrm{df}(d),

mirroring Pouliot et al.'s :math:`\mathbb{E}\lVert \mathbf{Y} - \hat{\mathbf{Y}}
\rVert^2 + 2\sigma^2\,\mathrm{df}`. Here :math:`\mathrm{SSR}^{\text{pre}}` is the
in-sample contrast sum of squares; :math:`\mathrm{df} = \lvert A\rvert - 1` with
:math:`A` the active control donors (their closed form for the unpenalised SCM,
in which searching over *which* donors to use is free); and
:math:`\hat{\sigma}^2` is a Mallows-:math:`C_p`-style noise estimate -- the
best-fitting candidate's per-period contrast variance. The candidate with the
smallest IC wins, so a design that buys a tighter fit by activating more donors
is penalised for it.

The returned ``results.pool`` is ranked by IC (rank-1 is the winner on
``results.design``), and each entry carries an ``ic`` value and its ``df``.
Like holdout, IC selection needs ``top_K >= 2`` and a MIP mode; it does not use
``holdout_frac``. The ``selection`` field unifies the three rules -- ``None``
(default) infers ``"holdout"`` when ``holdout_frac`` is set and ``"in_sample"``
otherwise, so existing configs are unchanged.

.. code-block:: python

   res = SYNDES({
       "df": df, "outcome": "Y", "unitid": "location", "time": "date",
       "K": 3, "mode": "two_way_global", "post_col": "post",
       "top_K": 5, "selection": "ic",              # IC over the whole pre-window
       "gap_limit": 0.2, "time_limit": 10.0,
   }).fit()

   for d in res.pool:                              # ranked by information criterion
       print("treated:", sorted(d["markets"]),
             "| ic:", round(d["ic"], 1), "| df:", d["df"])

Pass ``syndes_options={"selection": "ic"}`` to :func:`~mlsynth.compare_methods`
to use IC selection there (it overrides the default holdout).

Ranking the menu by power
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each menu entry carries its own per-horizon ``power_curve`` (a
:class:`~mlsynth.SYNDESPower` over horizons ``1..12``), computed with the same
Newey-West machinery as the previous section, so the whole pool is comparable on
power -- not just the rank-1 winner on ``res.power_curve``. Reading an entry's
MDE at the horizon you plan to run is a single array lookup. The example below
takes a 20-market GeoLift subset and plots every menu design's MDE at horizon
:math:`h = 3`, sorted most-detectable first -- turning the menu into a power
ranking the MSE objective never sees:

.. code-block:: python

   import matplotlib.pyplot as plt
   import pandas as pd
   from mlsynth import SYNDES

   df = pd.read_csv(                                      # GeoLift_PreTest panel
       "https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
       "refs/heads/main/basedata/geolift_market_data.csv"
   )
   markets = sorted(df["location"].unique())[:20]        # 20-market subset
   df = df[df["location"].isin(markets)].copy()
   cut = sorted(df["date"].unique())[-14]                # last 14 days = post window
   df["post"] = (df["date"] >= cut).astype(int)

   # top_K=6 runs six MIPs (one per no-good cut) -- expect ~a minute on SCIP.
   res = SYNDES({
       "df": df, "outcome": "Y", "unitid": "location", "time": "date",
       "K": 3, "mode": "two_way_global", "post_col": "post", "top_K": 6,
       "gap_limit": 0.2, "time_limit": 10.0,
   }).fit()

   H = 3                                                  # horizon to plan for
   scored = [("+".join(map(str, sorted(d["markets"]))),
              float(d["power_curve"].mde_percent[H - 1]))   # 1-indexed horizon
             for d in res.pool]
   scored.sort(key=lambda r: r[1])                        # smallest MDE -> left

   labels = [s[0] for s in scored]
   mde = [s[1] for s in scored]
   fig, ax = plt.subplots(figsize=(10, 4.5))
   ax.bar(range(len(mde)), mde, color="#4C72B0")
   ax.set_xticks(range(len(mde)))
   ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
   ax.set_ylabel(f"MDE at horizon {H} (% of treated baseline)")
   ax.set_xlabel("SYNDES menu design (treated set) -- most detectable first")
   ax.set_title(f"SYNDES top_K menu: minimum detectable effect at horizon {H}")
   fig.tight_layout()
   fig.savefig("syndes_menu_mde_h3.png", dpi=120)

Reading the bars left to right gives the designs in order of detectability at the
horizon you will actually run: the leftmost is the most powerful experiment in
the pool, which -- per the lesson above -- need not be the rank-1 (best-fitting)
design.

Pareto recommendation
---------------------

Ranking by power alone has the opposite blind spot to ranking by fit: the most
detectable design may balance the pre-period poorly. The honest object is the
trade-off between the two, so whenever a pool is produced ``SYNDES`` attaches a
recommendation to ``res.recommendation`` (a
:class:`~mlsynth.SYNDESRecommendation`), mirroring the LEXSCM recommender. It has
two parts.

First, a Pareto frontier on fit versus power, where fit is the pre-period RMSE
between the treated group and the weighted average of the controls (downwards)
and power is ``mde_pct`` at the realised horizon (downwards): the designs for
which neither can be improved without worsening the other. Dominated designs --
including, very often, the rank-1 best-fitting design, which buys its fit at the
cost of power -- are set aside. The frontier is always exposed in
``res.recommendation.pareto_ids`` for transparency, with cost as a tie-break
rather than a third axis.

Second, a single recommended design picked by a GeoLift-style composite score:
each design is dense-ranked on fit and on power (best metric ranks first, exactly
as GeoLift's market-selection score aggregates its component ranks), and the two
ranks are combined with the configurable ``power_weight`` / ``fit_weight``
(normalised to sum to one; default ``0.51`` / ``0.49``, a slight preference for
power). The smallest combined score wins, with cost then pre-period RMSE breaking
ties. Selection never raises: with no design whose MDE is finite it falls back to
the best-fitting design and reports ``status="POWER_NOT_ESTABLISHED"``; with no
pool, ``status="EMPTY"``.

.. code-block:: python

   import pandas as pd
   from mlsynth import SYNDES

   df = pd.read_csv(                                      # GeoLift_PreTest panel
       "https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
       "refs/heads/main/basedata/geolift_market_data.csv"
   )
   markets = sorted(df["location"].unique())[:20]        # 20-market subset
   df = df[df["location"].isin(markets)].copy()
   cut = sorted(df["date"].unique())[-14]
   df["post"] = (df["date"] >= cut).astype(int)

   # top_K=6 runs six MIPs -- expect ~a minute on SCIP. power_weight/fit_weight
   # default to 0.51/0.49; raise power_weight to prefer detectability harder.
   res = SYNDES({
       "df": df, "outcome": "Y", "unitid": "location", "time": "date",
       "K": 3, "mode": "two_way_global", "post_col": "post", "top_K": 6,
       "gap_limit": 0.2, "time_limit": 10.0,
   }).fit()

   rec = res.recommendation
   print(rec.status, rec.weights)                        # 'OK' {'power':0.51,'fit':0.49}
   print(rec.winner.design_id, rec.winner.markets)       # the recommended design
   print(rec.winner.control_group)                       # its synthetic-control donors
   print(rec.pareto_ids)                                 # fit-power Pareto frontier
   print(pd.DataFrame(rec.table))                        # every design, scored + flagged

Designs are numbered by fit: ``D1`` is always the best-fitting design (smallest
pre-period RMSE), ``D2`` the next, and so on. The recommended design need not be
``D1`` -- when power outweighs fit it is a higher-numbered design with a smaller
MDE, which is exactly the trade-off the recommendation surfaces.

When ``display_graph=True`` and a pool exists, two figures are drawn: Panel A,
the normal SYNDES design plot (synthetic treated vs synthetic control), and
Panel B, this fit-versus-power Pareto frontier with the recommended design
starred (x-axis: the pre-period RMSE). Panel B can also be drawn on its own with
:func:`~mlsynth.utils.syndes_helpers.plotter.plot_syndes_pareto`. Both render in
the in-house mlsynth plot style.

Comparing designs across methods (GEOLIFT vs SYNDES)
----------------------------------------------------

SYNDES and :doc:`GEOLIFT <geolift>` use different optimisers but share a grammar:
each emits a design that reduces to a unit-level contrast
:math:`\mathbf{c} = \mathbf{w}_{\text{treated}} - \mathbf{w}_{\text{control}}`
(both sides summing to one) over the same panel. From :math:`\mathbf{c}` two
comparable numbers follow -- the pre-period fit RMSE
:math:`\sqrt{\operatorname{mean}((\mathbf{Y}_{\text{pre}}\mathbf{c})^2)}` and,
by injecting a known effect at a fixed horizon, a minimum detectable effect. So
the two methods' designs can be placed on one fit-versus-power plane and their
Pareto frontiers overlaid.

The one rule that keeps the comparison honest is that the MDE must be computed by
a single shared harness for both methods -- same horizon, same effect grid, same
moving-block permutation null, same baseline -- so the frontier gap reflects the
designs, not two different power methodologies.
:mod:`mlsynth.utils.design_compare` does exactly this. It rests on the fact that
adding an effect :math:`\tau` to the treated units shifts the contrast mean
:math:`\mathbf{y}_t^\top\mathbf{c}` by exactly :math:`\tau` (the treated weights
sum to one), so the length-:math:`h` block-mean null shifted by :math:`\tau` is
the alternative, and the power at :math:`\tau` is the share of that shifted null
clearing the two-sided critical value.

The example fixes the horizon at five post-periods and overlays both frontiers on
the full native GeoLift ``GeoLift_Test`` panel (all 40 markets):

:func:`~mlsynth.compare_methods` is the one-call wrapper: feed it the common
options (panel, treated-set size, horizon, post window) plus any per-method
overrides, and it fits both estimators and scores them on the shared plane,
returning the comparison in dataframe (``.table``) and plot (``.plot()``) form.

.. code-block:: python

   import pandas as pd
   from mlsynth import compare_methods

   df = pd.read_csv(                                       # full native GeoLift_Test panel
       "https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
       "refs/heads/main/basedata/geolift_test_data.csv"
   )                                                       # all 40 markets

   cmp = compare_methods(
       df, outcome="Y", unitid="location", time="date",
       treated_size=3, horizon=5, n_post=5, top_K=6,       # only five post-periods
       syndes_options={"gap_limit": 0.05, "time_limit": 20.0},  # ~2 min for SYNDES
   )
   print(cmp.table[["method", "label", "fit_rmse", "mde_pct", "pareto"]])
   cmp.plot()                                              # overlaid frontiers

``cmp.table`` has one row per design scored on the shared plane (``fit_rmse``,
``mde_pct`` at horizon five) with a per-method ``pareto`` flag, and ``cmp.plot()``
overlays the two frontiers so you can read directly which method's designs
dominate, and where. ``cmp.syndes`` and ``cmp.geolift`` keep the underlying fits
for further inspection. To do it by hand instead -- fit each method yourself and
pass the designs through :func:`~mlsynth.from_syndes` / :func:`~mlsynth.from_geolift`
into :func:`~mlsynth.compare_pareto` -- see those functions.

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

Pareto recommendation -- the composite-score selector that builds
``res.recommendation`` from the solution pool:

.. automodule:: mlsynth.utils.syndes_helpers.select
   :members:
   :undoc-members:

Cross-method comparison -- score GEOLIFT and SYNDES designs on one shared
fit-vs-power plane:

.. automodule:: mlsynth.utils.design_compare
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
