Orthogonalized Synthetic Control
================================

.. currentmodule:: mlsynth

When to Use This Estimator
--------------------------

The synthetic control gives you a counterfactual; getting an honest standard
error for the resulting ATT is the hard part. The control weights are a
nuisance parameter that is high-dimensional, pinned to the boundary of the
simplex, and -- in a factor model with more donors than the factor rank --
only partially identified even when the ATT itself is point-identified. Each
of those three features breaks the usual asymptotic-normality argument, which
is why placebo and subsampling inference are the norm and why honest
confidence intervals for the ATT are scarce.

Use ORTHSC, due to Fry [ORTHSC]_, when you want a confidence interval and a
t-test for the ATT that stay valid in spite of all three complications. The
method estimates the control weights with a regularization penalty, then
estimates the ATT from moment conditions that are Neyman-orthogonal to those
weights, so the ATT estimate is asymptotically normal -- and insensitive to
which weight vector in the identified set was chosen. Its variance is estimated
with a fixed-smoothing orthonormal-series long-run variance, and the test is
referred to a :math:`t` distribution whose degrees of freedom are the smoothing
parameter, so the test controls size without requiring a consistent variance.

The weights are identified by an instrumental-variables device: the outcomes of
untreated units that are excluded from the control pool serve as instruments.
In a factor model these excluded units load on the same common factors as the
treated unit but are independent of its idiosyncratic shocks, which is exactly
the exclusion restriction the moment conditions need.

Do not use this estimator when
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* You have no untreated units to spare as instruments. ORTHSC sets aside a
  subset of the never-treated units as instruments rather than controls; with a
  tiny donor pool that price is too high -- use placebo inference
  (:doc:`vanillasc`) or a conformal interval instead.
* Either the pre-period or the post-period is very short. The size control is
  a joint-asymptotic result in :math:`T_0` and :math:`T_2`; when one of them is
  tiny the test can over-reject (see the Monte Carlo below at :math:`T_2 = 4`).
* You only need a point estimate. The orthogonalization machinery exists to
  make the *inference* valid; if you do not need a CI, a plain synthetic control
  (:doc:`vanillasc`) or another IV-SC (:doc:`proximal`, :doc:`siv`) is simpler.
* The sharp null in every post period -- not the average -- is the target.
  Conformal inference (:doc:`vanillasc` with ``inference="conformal"``) tests
  that; ORTHSC tests a hypothesis about the ATT.

Notation
--------

Let :math:`j = 1` denote the treated unit, with outcome series
:math:`\mathbf{y}_1`. The remaining units split into two disjoint roles: a
control pool :math:`\mathcal{N}_0` (the synthetic-control donors, with donor
matrix :math:`\mathbf{Y}_0 \in \mathbb{R}^{T \times N_0}`) and an instrument set
:math:`\mathcal{Z}` of untreated units excluded from the controls, stacked at
time :math:`t` into the vector :math:`\mathbf{z}_t \in \mathbb{R}^{Q}` (a
constant is appended as an extra instrument when ``include_constant=True``).
Time is :math:`t \in \mathcal{T} \coloneqq \{1, \dots, T\}`, intervention after
:math:`T_0`, with pre-period :math:`\mathcal{T}_1` (length :math:`T_0`) and
post-period :math:`\mathcal{T}_2` (length :math:`T_2`).

The control weights :math:`\mathbf{w} \in \Delta^{N_0}` are the nuisance
parameter; the ATT :math:`\tau` is the parameter of interest, estimated as
:math:`\widehat{\tau}`. The orthogonalization weights :math:`\boldsymbol{\eta}`
combine the moment conditions.

Assumptions
-----------

Assumption 1 (linear factor model). For all units and periods the
no-intervention outcome follows
:math:`y_{jt}^N = \boldsymbol{\lambda}_j^\top \mathbf{f}_t + u_{jt}`, with
:math:`r` common factors :math:`\mathbf{f}_t`, loadings
:math:`\boldsymbol{\lambda}_j`, and idiosyncratic error :math:`u_{jt}`.

*Remark.* This is the standard interactive-effects DGP (Abadie, Diamond &
Hainmueller). The control pool and the instruments load on the same
:math:`\mathbf{f}_t`; that shared structure is what lets the instruments
identify the weights and what makes the synthetic control track the treated
unit absent the intervention.

Assumption 2 (instrument exclusion). The excluded units' idiosyncratic shocks
are uncorrelated with the treated unit's, so
:math:`\mathbb{E}[\mathbf{z}_t (y_{1t}^N - \mathbf{y}_{0,t}^\top \mathbf{w}_0)] =
\mathbf{0}` at the population weights :math:`\mathbf{w}_0` over the pre-period.

*Remark.* This is the moment condition that point-identifies the ATT. The
instruments need only be orthogonal to the treated unit's idiosyncratic error
after the factor structure is matched -- not to the factors themselves -- so
they may be strongly correlated with :math:`\mathbf{f}_t`.

Assumption 3 (regularized weights converge into the identified set). The
penalized weight estimate :math:`\widehat{\mathbf{w}}` converges (in the
:math:`\ell_1` sense, at a rate tied to the donor dimension) to an element of
the identified set :math:`\mathcal{D}_0`.

*Remark.* The penalty does not need to point-identify :math:`\mathbf{w}` --
which is generally impossible here -- only to drive the estimate to a unique,
well-behaved element of the set. The orthogonalization then makes
:math:`\widehat{\tau}` first-order insensitive to which element that is.

Assumption 4 (fixed-smoothing regime). Both :math:`T_0` and :math:`T_2` grow,
and the number of moment conditions and the parameter of interest are
fixed-dimensional.

*Remark.* Under fixed smoothing the orthonormal-series variance does not
converge to a constant; instead the studentized statistic has a :math:`t`
limit with degrees of freedom equal to the smoothing parameter, which is what
delivers size control without a consistent variance estimate.

The estimator
-------------

Stack the moment conditions: the :math:`Q` pre-period instrument moments and
the single post-period ATT moment,

.. math::

   \mathbf{g}(\tau, \mathbf{w}) \coloneqq
   \begin{pmatrix}
     T_0^{-1} \sum_{t \in \mathcal{T}_1} \mathbf{z}_t\,(y_{1t} - \mathbf{y}_{0,t}^\top \mathbf{w}) \\[2pt]
     T_2^{-1} \sum_{t \in \mathcal{T}_2} (y_{1t} - \tau - \mathbf{y}_{0,t}^\top \mathbf{w})
   \end{pmatrix}.

Step 1: regularized control weights. Among the simplex weights whose pre-period
instrument moments sit within a data-driven slack :math:`\lambda`, take the
minimum-norm one,

.. math::

   \widehat{\mathbf{w}} \in \operatorname*{argmin}_{\mathbf{w} \in \Delta^{N_0}}
     \|\mathbf{w}\|_2^2
     \quad \text{s.t.} \quad
     \bigl\| T_0^{-1}\textstyle\sum_{t \in \mathcal{T}_1}
        \mathbf{z}_t (y_{1t} - \mathbf{y}_{0,t}^\top \mathbf{w}) \bigr\|_\infty
        \le \lambda ,

with :math:`\lambda` itself the smallest achievable slack (an LP), inflated by
a :math:`\log` factor in the sample size and dimensions.

Step 2: Neyman orthogonalization. Choose moment weights
:math:`\boldsymbol{\eta}` (normalized so the post-moment entry is one) that make
the combined moment insensitive to the control weights to first order -- that
is, its derivative with respect to :math:`\mathbf{w}` vanishes,
:math:`\partial_{\mathbf{w}}\,\boldsymbol{\eta}^\top \mathbf{g} = \mathbf{0}`.
Intuitively, a small error in the estimated weights then does not move the
equation we solve for the ATT. The ATT is read off the orthogonalized moment,

.. math::

   \widehat{\tau} = \boldsymbol{\eta}^\top
     \mathbf{g}_0(\widehat{\mathbf{w}}), \qquad
   \mathbf{g}_0(\mathbf{w}) \coloneqq \mathbf{g}(0, \mathbf{w}),

so that perturbing :math:`\widehat{\mathbf{w}}` within the identified set leaves
:math:`\widehat{\tau}` unchanged to first order.

Step 3: fixed-smoothing inference. Form the pre- and post-period moment
residual paths, estimate the long-run variance :math:`\widehat{V}` with an
orthonormal-series (fixed-:math:`b`) estimator using :math:`K` basis terms, and
test :math:`H_0\!:\tau = \tau_0` with

.. math::

   t = \frac{\sqrt{n}\,(\widehat{\tau} - \tau_0)}{\sqrt{\widehat{V}}}
   \ \sim\ t_{K}, \qquad n = \min\{T_0, T_2\},

where :math:`K` is the CPE-optimal smoothing parameter of Sun (2013). The
confidence interval inverts the same :math:`t_K` test.

Example
-------

A runnable synthetic panel with a known effect (factor DGP; the treated unit is
a convex mix of the controls, the instruments share the factors):

.. code-block:: python

   import numpy as np, pandas as pd
   from mlsynth import ORTHSC

   rng = np.random.default_rng(0)
   T0, T1, J, Q, R, tau = 30, 16, 8, 5, 2, -0.3
   T = T0 + T1
   F = rng.normal(size=(T, R))
   YJ = (F @ rng.uniform(0.5, 1.5, (R, J))).T + 0.3 * rng.normal(size=(J, T))
   w = rng.dirichlet(np.ones(J))
   treated = w @ YJ + 0.3 * rng.normal(size=T)
   treated[T0:] += tau                                   # additive post effect
   Z = (F @ rng.uniform(0.5, 1.5, (R, Q))).T + 0.3 * rng.normal(size=(Q, T))

   cols = {"treated": treated}
   cols.update({f"c{j}": YJ[j] for j in range(J)})
   cols.update({f"z{q}": Z[q] for q in range(Q)})
   wide = pd.DataFrame(cols, index=pd.Index(np.arange(T), name="year"))
   long = wide.reset_index().melt(id_vars="year", var_name="unit", value_name="Y")
   long["treat"] = ((long.unit == "treated") & (long.year >= T0)).astype(int)

   res = ORTHSC({
       "df": long, "outcome": "Y", "treat": "treat", "unitid": "unit",
       "time": "year",
       "controls": [f"c{j}" for j in range(J)],
       "instruments": [f"z{q}" for q in range(Q)],
       "display_graphs": False,
   }).fit()

   print(f"ATT = {res.att:+.3f}  (true {tau})")
   print(f"95% CI = [{res.inference.ci_lower:+.3f}, {res.inference.ci_upper:+.3f}]")
   print(f"p = {res.inference.p_value:.3g},  smoothing K = "
         f"{res.method_details.parameters_used['smoothing_K']}")

Verification
------------

ORTHSC is validated against the paper's empirical result (Path A) and its
simulation study (Path B). Both are pinned as durable benchmark cases --
``orthsc_carbontax`` and ``orthsc_size_power`` -- and the empirical case also
matches a live run of the author's R reference to the digit.

Path A: Sweden's carbon tax (Andersson 2019)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fry applies ORTHSC to Andersson's carbon-tax panel: the control pool is
Andersson's 14 OECD donors and the instruments are the 7 countries Andersson
excluded for having their own carbon or fuel taxes -- exactly the "excluded
units as instruments" device. The estimate is an average reduction of
:math:`0.29` metric tons of transport CO\ :sub:`2` per capita, with a t-test
p-value of :math:`0.00018` -- significant where placebo, conformal, and
cross-fitting inference are not.

.. code-block:: python

   import pandas as pd
   from mlsynth import ORTHSC

   url = ("https://raw.githubusercontent.com/jgreathouse9/mlsynth/main/"
          "basedata/carbontax_fullsample_data.dta.txt")
   df = pd.read_stata(url).rename(columns={"CO2_transport_capita": "Y"})
   df["treat"] = ((df.country == "Sweden") & (df.year >= 1990)).astype(int)

   controls = ["Australia", "Belgium", "Canada", "Denmark", "France", "Greece",
               "Iceland", "Japan", "New Zealand", "Poland", "Portugal", "Spain",
               "Switzerland", "United States"]
   instruments = ["Finland", "Germany", "Ireland", "Italy", "Netherlands",
                  "Norway", "United Kingdom"]

   res = ORTHSC({
       "df": df, "outcome": "Y", "treat": "treat", "unitid": "country",
       "time": "year", "controls": controls, "instruments": instruments,
       "display_graphs": False,
   }).fit()

   print(f"ATT = {res.att:.5f}")                               # -0.29013
   print(f"p   = {res.inference.p_value:.6f}")                 #  0.000183
   print(f"K   = {res.method_details.parameters_used['smoothing_K']}")  # 4
   print(res.inference.ci_lower, res.inference.ci_upper)       # -0.476, -0.105

mlsynth's NumPy/cvxpy port reproduces the live R reference (and the paper) to
the digit:

.. list-table::
   :header-rows: 1
   :widths: 30 24 24

   * - Quantity
     - mlsynth
     - R / paper
   * - ATT :math:`\widehat{\tau}`
     - :math:`-0.29013`
     - :math:`-0.29013`
   * - p-value
     - :math:`0.000183`
     - :math:`0.000183`
   * - smoothing :math:`K`
     - :math:`4`
     - :math:`4`
   * - 95% CI
     - :math:`[-0.476,\,-0.105]`
     - :math:`[-0.476,\,-0.105]`

The control weights themselves differ slightly from the reference (a different,
equally valid element of the identified set), yet the ATT and p-value match --
a direct demonstration of the orthogonalization: :math:`\widehat{\tau}` does
not depend on which weight solver pinned the nuisance.

Path B: size and power (Fry Tables 1-2)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The paper's headline simulation finding is that the orthogonalized t-test
controls size while keeping high power, where naive IV-SC, cross-fitting, and
ArCo over-reject. On a clean linear-factor DGP -- the treated unit a convex mix
of controls plus idiosyncratic noise, instruments sharing the factors but
independent of the treated unit's shocks -- ORTHSC reproduces that behaviour.
At the :math:`5\%` level over 200 replications:

.. list-table::
   :header-rows: 1
   :widths: 14 14 22 22

   * - :math:`T_0`
     - :math:`T_2`
     - size (effect = 0)
     - power (effect = :math:`-0.25`)
   * - 30
     - 16
     - 0.070
     - 0.655
   * - 30
     - 32
     - 0.035
     - 0.880
   * - 60
     - 32
     - 0.060
     - 0.960

Size sits at or below the nominal :math:`0.05` (up to Monte Carlo noise) and
power rises with the number of post periods -- the pattern of Fry's Tables 1-2.
The benchmark below runs the full grid (it is intentionally a few thousand
small convex solves, so it takes ~30s):

.. code-block:: python

   import warnings
   import numpy as np
   from mlsynth.utils.orthsc_helpers.pipeline import orthogonalized_sce

   NSIM, ALPHA, J, Q, R, NOISE = 200, 0.05, 8, 5, 2, 0.3

   def draw(T0, T1, tau, rng):
       T = T0 + T1
       F = rng.normal(size=(T, R))
       YJ = (F @ rng.uniform(0.5, 1.5, (R, J))).T + NOISE * rng.normal(size=(J, T))
       w = rng.dirichlet(np.ones(J))
       y = w @ YJ + NOISE * rng.normal(size=T); y[T0:] += tau
       Z = (F @ rng.uniform(0.5, 1.5, (R, Q))).T + NOISE * rng.normal(size=(Q, T))
       return y[:T0], YJ[:, :T0], Z[:, :T0], y[T0:], YJ[:, T0:]

   def reject_rate(T0, T1, tau, seed):
       rng = np.random.default_rng(seed); rej = 0
       with warnings.catch_warnings():
           warnings.simplefilter("ignore")
           for _ in range(NSIM):
               r = orthogonalized_sce(*draw(T0, T1, tau, rng), alpha=ALPHA)
               rej += int(r["pvalue"] < ALPHA)
       return rej / NSIM

   for T0, T1 in [(30, 16), (30, 32), (60, 32)]:
       size = reject_rate(T0, T1, 0.0, seed=10 * T1)
       power = reject_rate(T0, T1, -0.25, seed=10 * T1 + 1)
       print(f"T0={T0} T2={T1}: size={size:.3f}  power={power:.3f}")

See :doc:`replications/orthsc` for the live-R cross-check and the demonstrate-
first port story.

References
----------

.. [ORTHSC] Fry, J. (2026). Orthogonalized Synthetic Controls. arXiv:2510.22828.
   https://arxiv.org/abs/2510.22828

Core API
--------

.. automodule:: mlsynth.estimators.orthsc
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.utils.orthsc_helpers.config.OrthSCConfig
   :members:
   :undoc-members:

Helper Modules
--------------

Data preparation -- ``dataprep`` then the treated / control / instrument
three-way split.

.. automodule:: mlsynth.utils.orthsc_helpers.setup
   :members:
   :undoc-members:

Regularized nuisance estimation: the simplex control weights and the
normalized orthogonalization weights.

.. automodule:: mlsynth.utils.orthsc_helpers.regularized
   :members:
   :undoc-members:

The orthogonalized ATT and its pre/post moment-residual paths.

.. automodule:: mlsynth.utils.orthsc_helpers.orthogonal
   :members:
   :undoc-members:

Fixed-smoothing orthonormal-series variance, the Sun (2013) smoothing
parameter, and the t-test / confidence interval.

.. automodule:: mlsynth.utils.orthsc_helpers.serieshac
   :members:
   :undoc-members:

End-to-end orchestration (``orthogonalized_sce`` for the array API,
``run_orthsc`` for the estimator).

.. automodule:: mlsynth.utils.orthsc_helpers.pipeline
   :members:
   :undoc-members:
