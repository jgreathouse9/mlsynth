Immunized Doubly-Robust Synthetic Control (BEAST)
=================================================

.. currentmodule:: mlsynth

When to Use This Estimator
--------------------------

A classic synthetic control chooses donor weights to make a weighted average
of the control units track the treated unit's pre-treatment outcomes, then
reads the treatment effect off the post-treatment gap. That works well when a
few donors clearly resemble the treated unit, but it says nothing about how to
use covariates when there are many of them, and it gives no easy standard error
for the resulting effect.

Use BEAST, due to Bléhaut, D'Haultfœuille, L'Hour and Tsybakov [BEAST]_, when
you have a moderately rich set of unit-level covariates -- economic predictors,
lagged outcomes -- of which only a handful truly matter, and you want a
treatment effect that comes with an honest, analytic confidence interval. BEAST
picks donor weights by covariate balancing rather than outcome fitting: it finds
exponential-tilting weights that make the covariates of the weighted donor pool
match the treated unit, with an ℓ₁ penalty that selects the informative
covariates and discards the rest. It then corrects the effect with an
immunizing outcome regression, which makes the estimator doubly robust -- valid
if either the balancing or the outcome model is right -- and asymptotically
normal, so its standard error is a closed-form expression rather than a
placebo distribution.

The name is the authors': BEAST is the immunized doubly-robust estimator built
from the Calibration-Lasso balancing weights and the immunizing WLS-Lasso
outcome model.

Do not use this estimator when
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Your covariate set is over-saturated -- as many (or nearly as many)
  covariates as control units. The exponential-tilting calibration then
  degenerates: the balancing weights no longer sum to one, so they are not a
  valid synthetic control. BEAST detects this and raises rather than returning a
  meaningless answer (see `The balance-validity guard`_). Reduce to a sparse,
  informative covariate set, or use a high-dimensional-donor estimator built for
  that regime (:doc:`sparse_sc`, :doc:`clustersc`).
* You have no covariates worth balancing on. BEAST *is* the covariate machinery;
  with nothing to balance, a plain synthetic control (:doc:`vanillasc`) is the
  right tool.
* The post-period is a single period, or you need period-by-period sharp-null
  inference. BEAST's interval is for the average post-treatment effect under an
  approximately-independent-errors argument; conformal inference
  (:doc:`vanillasc` with ``inference="conformal"``) targets the sharp null.

Notation
--------

Let :math:`d_j \in \{0, 1\}` indicate whether unit :math:`j` is the treated one,
with :math:`n_1 = \sum_j d_j` treated units (here :math:`n_1 = 1`) and
:math:`\pi = n_1 / n` the treated share. Each unit carries a covariate vector
:math:`\mathbf{x}_j \in \mathbb{R}^{p}` whose first entry is a constant; these
are pre-treatment covariate means and, optionally, a few lagged outcomes, all
normalized. Stacked over units these form :math:`\mathbf{X} \in
\mathbb{R}^{n \times p}`. For a single post period :math:`t` write
:math:`y_{jt}` for unit :math:`j`'s outcome.

BEAST has two nuisance parameters and one target. The balancing coefficients
:math:`\boldsymbol{\beta} \in \mathbb{R}^{p}` define the donor weights; the
outcome-model coefficients :math:`\boldsymbol{\mu} \in \mathbb{R}^{p}` immunize
the effect; the target is the average treatment effect on the treated,
:math:`\theta`, estimated as :math:`\widehat\theta`.

Assumptions
-----------

Assumption 1 (sparsity). The treated-vs-control covariate contrast is driven by
a small number :math:`s \ll p` of the covariates -- :math:`\boldsymbol{\beta}`
and :math:`\boldsymbol{\mu}` are sparse.

*Remark.* This is what lets the method admit many covariates: the ℓ₁ penalty
selects the few that carry the signal, so :math:`p` may be large relative to the
sample as long as the *active* set stays small. It is also the assumption the
balance guard protects -- when it fails and every covariate is active, the
calibration has no sparse solution to find and the weights stop summing to one.

Assumption 2 (overlap / valid balancing). There exist tilting weights that
balance the covariates between the treated unit and the donor pool; equivalently
the exponential-tilting calibration has a finite solution whose weights sum to
one.

*Remark.* The intercept (the constant column of :math:`\mathbf{X}`) is the
balance condition for the total mass: balancing it forces
:math:`\sum_j W_j = 1`, which is exactly what makes the tilting weights a
synthetic control. When covariates are so rich that no tilting balances them,
this fails and the design is not a synthetic control at all.

Assumption 3 (double robustness). Either the balancing model or the outcome
model is correctly specified (not necessarily both).

*Remark.* This is the payoff of immunization. The effect is built from a moment
that is Neyman-orthogonal to both nuisances, so a first-order error in
:math:`\widehat{\boldsymbol\beta}` or :math:`\widehat{\boldsymbol\mu}` does not
bias :math:`\widehat\theta`; only if *both* are wrong does the estimate suffer.
The same orthogonality is what makes :math:`\widehat\theta` asymptotically normal
with the analytic variance below.

The estimator
-------------

Step 1: Calibration-Lasso balancing weights. The donor weights are an
exponential tilt of the covariates,

.. math::

   W_j = \frac{(1 - d_j)\,\exp(\mathbf{x}_j^\top \boldsymbol\beta)}{n_1},

with :math:`\boldsymbol\beta` solving the ℓ₁-penalized calibration

.. math::

   \widehat{\boldsymbol\beta} = \operatorname*{arg\,min}_{\boldsymbol\beta}
     \ \frac{1}{n}\sum_j \bigl[(1 - d_j)\exp(\mathbf{x}_j^\top\boldsymbol\beta)
        - d_j\,\mathbf{x}_j^\top\boldsymbol\beta \bigr]
     \ +\ \lambda \, \lVert \boldsymbol\Psi \boldsymbol\beta \rVert_1 ,

the intercept left unpenalized. The first-order conditions are the covariate
balancing equations :math:`\frac{1}{n}\sum_j (d_j - (1-d_j)
\exp(\mathbf{x}_j^\top\boldsymbol\beta))\,\mathbf{x}_j = \mathbf{0}`; the
unpenalized intercept condition is :math:`\sum_j W_j = 1`. The penalty level
:math:`\lambda = c\,\Phi^{-1}(1 - g/2p)/\sqrt{n}` is the Belloni--Chernozhukov--
Hansen choice with data-driven loadings :math:`\boldsymbol\Psi` iterated to
convergence (mlsynth solves it by proximal-gradient descent, the authors by
OWL-QN; the two agree to five decimals).

Step 2: immunizing outcome model. For each post period, regress the outcome on
the covariates over the weighted control pool by a weighted-least-squares Lasso,

.. math::

   \widehat{\boldsymbol\mu} = \operatorname*{arg\,min}_{\boldsymbol\mu}
     \ \frac{1}{n}\sum_j W_j\,(y_{jt} - \mathbf{x}_j^\top\boldsymbol\mu)^2
     \ +\ \lambda' \lVert \boldsymbol\mu \rVert_1 .

Step 3: immunized ATT and its standard error. The effect is the sample analogue
of the orthogonal moment,

.. math::

   \widehat\theta = \frac{1}{\pi}\,\frac{1}{n}\sum_j
     \bigl(d_j - (1 - d_j)\exp(\mathbf{x}_j^\top\widehat{\boldsymbol\beta})\bigr)
     \bigl(y_{jt} - \mathbf{x}_j^\top\widehat{\boldsymbol\mu}\bigr),

with influence function :math:`\psi_j = (d_j - (1-d_j)
\exp(\mathbf{x}_j^\top\widehat{\boldsymbol\beta}))(y_{jt} -
\mathbf{x}_j^\top\widehat{\boldsymbol\mu}) - d_j\widehat\theta` and analytic
standard error :math:`\widehat\sigma = \sqrt{\tfrac{1}{n}\sum_j \psi_j^2}\,/\,
(\pi\sqrt{n})`. Setting :math:`\boldsymbol\mu = \mathbf{0}`
(``immunity=False``) gives the non-immunized plug-in, which is consistent only
if the balancing model is right.

The post-treatment ATT reported by :attr:`~mlsynth.BEAST.fit` averages
:math:`\widehat\theta` over the post periods; its interval combines the
per-period standard errors treating them as independent, which is optimistic
under serially correlated post residuals.

The balance-validity guard
---------------------------

Assumption 2 is checkable from the fit: a valid tilting synthetic control has
:math:`\sum_j W_j = 1`. BEAST computes :math:`\lvert \sum_j W_j - 1 \rvert` and,
if it exceeds ``balance_tol``, raises :class:`~mlsynth.exceptions.MlsynthEstimationError`
rather than returning weights that are not a synthetic control. This is the
concrete signature of the over-saturated regime: when the covariate set is as
rich as the donor pool, the exponential tilt has no sparse balancing solution
and the mass condition breaks. The guard turns that failure into an explicit
error instead of a silently wrong estimate.

Example
-------

California's Proposition 99 (Abadie, Diamond & Hainmueller 2010): California
raises its cigarette tax in 1989; the donors are the other US states, the
covariates are four economic predictors plus a few lagged sales.

.. code-block:: python

   import pandas as pd
   from mlsynth import BEAST

   url = ("https://raw.githubusercontent.com/jgreathouse9/mlsynth/main/"
          "basedata/augmented_cali_long.csv")
   df = pd.read_csv(url)
   df["treated"] = ((df.state == "California") & (df.year >= 1989)).astype(int)

   res = BEAST({
       "df": df, "outcome": "cigsale", "treat": "treated",
       "unitid": "state", "time": "year",
       "covariates": ["loginc", "p_cig", "pct15-24", "pc_beer"],
       "outcome_lags": [1975, 1980, 1988],
       "display_graphs": False,
   }).fit()

   print(f"ATT = {res.att:.2f}")                              # -22.42
   print(res.inference.ci_lower, res.inference.ci_upper)      # ~[-23.2, -21.6]
   det = res.inference.details
   print("sum of weights =", round(det["sum_weights"], 4))    # 1.0  (valid SC)
   print("covariates selected =", det["n_selected"])          # sparse
   print("effect by 2000 =", round(float(det["tau"][-1]), 2)) # -31.51

The average post-1989 reduction is about 22 packs per capita, growing to roughly
32 by 2000 -- the classic Proposition 99 result, here with an analytic interval
that excludes zero. The balancing weights sum to one (a valid synthetic control)
and the ℓ₁ calibration selects a sparse covariate set.

Verification
------------

BEAST is cross-validated against the authors' own R code (Path: cross-
validation). On the Proposition 99 panel, mlsynth's Calibration-Lasso /
immunizing-regression / immunized-ATT port reproduces a live run of
``jeremylhour/alternative-synthetic-control-sparsity``
(``CalibrationLasso.R`` / ``OrthogonalityReg.R`` / ``LassoFISTA.R`` /
``ImmunizedATT.R``) on the identical design matrix: the balancing coefficients
match to five decimals, and the immunized ATT reproduces the R (mean
:math:`-22.44`, matched to ``0.02`` packs; :math:`-31.51` by 2000, to
``0.006``), with the largest per-year gap on the path about ``0.15`` packs and
the standard errors matched to ``~0.05``. The durable check is
``benchmarks/cases/beast_prop99.py``, run
against the pinned R reference under
`benchmarks/reference/beast_prop99/
<https://github.com/jgreathouse9/mlsynth/tree/main/benchmarks/reference/beast_prop99>`_::

   python benchmarks/run_benchmarks.py --case beast_prop99

See :doc:`replications/beast` for the demonstrate-first port story, the
cell-by-cell comparison, and the finding that maps out BEAST's operating
envelope (why the over-rich augmented specification breaks the balancing).

References
----------

.. [BEAST] Bléhaut, M., D'Haultfœuille, X., L'Hour, J., & Tsybakov, A. B.
   (2021). An alternative to synthetic control for models with many covariates
   under sparsity. arXiv:2005.12225. https://arxiv.org/abs/2005.12225

Core API
--------

.. automodule:: mlsynth.estimators.beast
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.utils.beast_helpers.config.BEASTConfig
   :members:
   :undoc-members:

Helper Modules
--------------

Data preparation -- ``dataprep`` then the treated / donor covariate design
matrix (covariate pre-means plus optional lagged outcomes, normalized).

.. automodule:: mlsynth.utils.beast_helpers.setup
   :members:
   :undoc-members:

Core numerics: the Calibration-Lasso balancing weights, the immunizing
WLS-Lasso outcome model, the immunized ATT with its analytic standard error,
and the balance-validity check.

.. automodule:: mlsynth.utils.beast_helpers.estimator
   :members:
   :undoc-members:

End-to-end orchestration: fit the immunized ATT path, apply the balance guard,
and assemble the standardized results.

.. automodule:: mlsynth.utils.beast_helpers.pipeline
   :members:
   :undoc-members:
