Distributionally Robust Synthetic Control (DROSC)
=================================================

Overview
--------

Classical synthetic control estimates a treatment effect by matching the treated
unit's pre-treatment outcomes with a weighted average of control units, and then
assumes that the same weighting carries over to the post-treatment period. Two
things break that assumption in practice. When the control units are highly
correlated, many different weight vectors fit the pre-period almost equally well,
so the chosen weights -- and the effect read off them -- are unstable. And when
the relationship between the treated unit and the controls shifts across the
intervention, the pre-period weighting no longer identifies the counterfactual at
all.

DROSC, the method of Koo & Guo (2026), replaces the single fitted weight vector
with a worst-case over an entire *set* of weights: every weight vector compatible
with the pre-treatment moments up to a radius :math:`\lambda`. At
:math:`\lambda = 0` the set is tightest and DROSC agrees with classical synthetic
control; as :math:`\lambda` grows the set widens and the estimate becomes a
deliberately conservative proxy for the effect that the data can no longer pin
down. Sweeping :math:`\lambda` is an honest sensitivity display: it shows how much
robustness the conclusion can absorb before the effect is no longer
distinguishable from zero.

When to use this estimator
--------------------------

Reach for DROSC when the donor pool is highly collinear -- many similar regions,
stores, or products, so the classical weights are non-unique and fragile -- or
when you suspect the treated-control relationship shifts across the intervention
and want an effect and interval that stay valid when the standard identification
does not. The robustness radius is the dial: report the effect as a function of
:math:`\lambda`, not a single number.

If your donors are well separated and the standard assumptions are credible, the
plain estimators (:doc:`vanillasc`) are more efficient -- DROSC's conservatism is
wasted. If the fragility is specifically pre-fit non-uniqueness and you only need
a stable tie-break, not a robust interval, :doc:`lexscm` resolves the
non-uniqueness lexicographically instead. DROSC targets a *different, robust*
estimand; do not read its number as an estimate of the same quantity classical
synthetic control targets when the two disagree.

Notation
--------

The treated unit's outcome is :math:`y_t`; the :math:`N` donors' outcomes are the
rows of :math:`\mathbf{X}_t`. Time splits at :math:`T^\star = T_0 + 1` into the
pre-period :math:`t \le T_0` and the post-period. Write the pre-treatment donor
second-moment matrix and the treated-donor cross-moment as

.. math::

   \widehat{\boldsymbol{\Sigma}} = \frac{1}{T_0}\sum_{t \le T_0}
     \mathbf{X}_t \mathbf{X}_t^\top, \qquad
   \widehat{\boldsymbol{\gamma}} = \frac{1}{T_0}\sum_{t \le T_0} y_t \mathbf{X}_t,

and the post-treatment means :math:`\bar y_1 = \operatorname{mean}(y_t)` and
:math:`\bar{\mathbf{x}}_1 = \operatorname{mean}(\mathbf{X}_t)` over
:math:`t > T_0`. The compatible-weight set at radius :math:`\lambda` is the
simplex intersected with a moment band,

.. math::

   \mathcal{B}(\lambda) = \Big\{ \boldsymbol{\beta} \ge 0,\ \textstyle\sum_j
     \beta_j = 1 \ :\ \big\lVert \widehat{\boldsymbol{\Sigma}}\boldsymbol{\beta}
     - \widehat{\boldsymbol{\gamma}} \big\rVert_\infty \le \lambda + c_{T_0}
     \Big\},

where :math:`c_{T_0} \propto \log(\max(T_0, N))^{1/2}/\sqrt{T_0}` is a
data-driven slack that vanishes as :math:`T_0` grows. DROSC reports the effect at
the weight in :math:`\mathcal{B}(\lambda)` that best matches the post-treatment
mean,

.. math::

   \widehat{\boldsymbol{\beta}}(\lambda) = \operatorname*{arg\,min}_{
     \boldsymbol{\beta} \in \mathcal{B}(\lambda)}
     \big( \bar{\mathbf{x}}_1^\top \boldsymbol{\beta} - \bar y_1 \big)^2,
   \qquad
   \widehat{\tau}(\lambda) = \bar y_1 - \bar{\mathbf{x}}_1^\top
     \widehat{\boldsymbol{\beta}}(\lambda).

The objective is rank one in :math:`\boldsymbol{\beta}`, so
:math:`\widehat{\tau}(\lambda)` is identified even when
:math:`\widehat{\boldsymbol{\beta}}(\lambda)` is not -- which is exactly the
non-uniqueness DROSC is built to tolerate.

Assumptions
-----------

1. Single treated unit, complete panel. Ingestion is the standard
   :func:`mlsynth.utils.datautils.dataprep` contract: one treated unit with a
   pre/post split and a fully observed outcome matrix.

   Remark. There is no covariate cube or spatial input; DROSC uses only the
   outcome panel. A ``dependent`` flag switches the moment covariances to a
   Newey-West HAC estimator when the outcomes are autocorrelated.

2. The compatible set contains the truth. At the chosen radius,
   :math:`\mathcal{B}(\lambda)` includes a weighting that reproduces the treated
   counterfactual. Under the classical identification this holds at
   :math:`\lambda = 0`; when it fails, :math:`\widehat{\tau}(\lambda)` is a
   conservative proxy, not the classical effect.

   Remark. This is why the radius is reported as a sweep. The value of
   :math:`\lambda` at which :math:`\widehat{\tau}` first reaches zero is the
   amount of robustness the finding can absorb.

3. Simplex weights. Donor weights are non-negative and sum to one, so the
   counterfactual is an interpolation of the donors (no extrapolation).

   Remark. The moment band is layered on top of the simplex, so DROSC never
   leaves the convex hull; it only restricts which hull points are admissible.

Inference
---------

Set ``inference=True`` for the perturbation-based confidence interval. Because
:math:`\widehat{\tau}(\lambda)` is the value of an inequality-constrained
optimisation, its limiting law is non-normal, so a normal interval is invalid.
DROSC instead perturbs the pre-treatment moments and the post-treatment means
from their (HAC-)sampling distributions, re-solves the moment-band problem for
each of ``n_perturbations`` draws, forms a normal interval around each draw's
effect, and returns the *union* of those intervals -- a confidence set that may
be a single interval or a disjoint union. The enveloping hull is exposed as
``res.inference.ci_lower`` / ``ci_upper`` (and ``res.att_ci``); the full list of
pieces is in ``res.inference.details["ci_intervals"]``.

Two practical notes. The interval is a genuine confidence *set*, so read it as
"the effects the data cannot reject", not as a point estimate plus symmetric
error. And it is computationally heavy and mildly seed-sensitive: the moment band
starts far tighter than the perturbation scale, so an internal slack is inflated
until enough draws are feasible (dozens of rounds of
:math:`\texttt{n\_perturbations}` solves), and the endpoints move with the seed.
Fix ``seed`` for reproducibility and keep the default off unless you need the
interval.

Example
-------

.. code-block:: python

   import pandas as pd
   from mlsynth import DROSC

   basque = pd.read_csv("basedata/basque_jasa.csv")
   basque = basque[basque.regionname != "Spain (Espana)"].copy()
   treat_year = sorted(basque.year.unique())[15]          # T0 = 15
   basque["treat"] = ((basque.regionname == "Basque Country (Pais Vasco)")
                      & (basque.year >= treat_year)).astype(int)

   base = dict(df=basque, outcome="gdpcap", treat="treat",
               unitid="regionname", time="year")

   # a robustness sweep: how fast does the effect shrink toward zero?
   for lam in (0.0, 0.03, 0.06):
       res = DROSC({**base, "robustness_lambda": lam}).fit()
       print(lam, round(res.effects.att, 3))     # -0.742, -0.256, 0.000

   # perturbation union confidence interval at a fixed radius (slower)
   res = DROSC({**base, "robustness_lambda": 0.0, "inference": True,
                "seed": 1}).fit()
   print(res.att_ci)                             # enveloping hull, contains 0

Verification
------------

DROSC is cross-validated against the authors' own R implementation (``helpers.R``,
``limSolve::lsei``) -- sourced from their repository and run live via ``Rscript``
-- on the Basque study: the worst-case estimand
:math:`\widehat{\tau}(\lambda)` and the :math:`\lambda = 0` donor weights match
value-for-value (to :math:`\sim 10^{-7}`) across the robustness sweep. See the
durable case ``benchmarks/cases/drosc_basque.py`` and the replication page
:doc:`replications/drosc`.

Core API
--------

.. autoclass:: mlsynth.DROSC
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: mlsynth.config_models.DROSCConfig
   :members:
   :undoc-members:
   :show-inheritance:
