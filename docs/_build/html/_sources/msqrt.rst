Multivariate Square-root Lasso Synthetic Control (MSQRT)
========================================================

.. currentmodule:: mlsynth

When to Use This Estimator
--------------------------

``MSQRT`` implements the high-dimensional synthetic-control estimator of Shen,
Song and Abadie [MSQRT]_. It is built for *disaggregated* panels: many
fine-grained units (stores, ZIP codes, geos, products) where the number of
candidate donors :math:`N_0` is comparable to or larger than the number of
pre-treatment periods :math:`T_0`, and where several units are treated at the
same time (a block design). Two ideas distinguish it from running an
ordinary synthetic control unit-by-unit.

First, it pools the treated units into one matrix regression
:math:`\mathbf{Y}_1 = \mathbf{Y}_0\boldsymbol{\Theta} + \mathbf{E}` and
estimates the entire donor-weight matrix
:math:`\boldsymbol{\Theta}` jointly, borrowing strength across treated units
rather than solving one independent, noisy fit per treated unit. Second, the
loss is the nuclear norm of the residual matrix, a "square-root" (pivotal)
objective whose optimal penalty level does not depend on the unknown noise
variance -- so a single cross-validated :math:`\lambda` regularises the whole
problem, while the element-wise :math:`\ell_1` term performs donor selection in
the regime where the classical quadratic program is non-unique and plain Lasso
over-selects.

Reach for ``MSQRT`` when you have a block of treated units adopting
together, a wide donor pool (:math:`N_0 \gtrsim T_0`), and you want a single
interpretable, sparse set of donor weights shared across the treated block,
with an ATT read as the mean post-period gap. It is the natural tool for
retrospective "matched-market" lift studies and other disaggregated multi-unit
roll-outs.

Do not use MSQRT when
~~~~~~~~~~~~~~~~~~~~~~

* Adoption is staggered -- treated units switch on at different times.
  MSQRT assumes a block design and will refuse staggered panels. Use
  :doc:`sdid`, :doc:`seq_sdid`, :doc:`ppscm`, or :doc:`mcnnm` (which handles
  staggered missingness natively).
* There is a single treated unit. The pooling that makes MSQRT efficient
  has nothing to borrow across. Start at :doc:`fdid` or :doc:`tssc`, or the
  high-dimensional single-unit tools (:doc:`fscm`, :doc:`sparse_sc`).
* The donor pool is small (:math:`N_0 \ll T_0`). The classical quadratic
  program is well-posed; use :doc:`tssc`, :doc:`scmo`, or :doc:`fdid` and keep
  the simplex interpretation.
* The outcome panel has missing cells / gaps. MSQRT needs a balanced
  block panel; for informative or structured missingness use :doc:`mcnnm`
  (MAR) or :doc:`snn` (MNAR).
* Spillovers contaminate the donor block (SUTVA fails) -- use
  :doc:`spsydid` or :doc:`spillsynth`.
* Treatment is endogenous and you have an instrument -- use :doc:`siv`.
* You are designing an experiment (choosing whom to treat) rather than
  estimating a retrospective effect -- use :doc:`marex`, :doc:`syndes`, or
  :doc:`lexscm`.

Notation
--------

There are :math:`N` units :math:`\mathcal{N} \coloneqq \{1, \dots, N\}` observed
over periods :math:`t \in \mathcal{T} \coloneqq \{1, \dots, T\}`, 1-indexed,
with all treated units adopting after period :math:`T_0`. The pre-period is
:math:`\mathcal{T}_1 \coloneqq \{t \in \mathcal{T} : t \le T_0\}` (of length
:math:`T_0`) and the post-period is
:math:`\mathcal{T}_2 \coloneqq \{t \in \mathcal{T} : t > T_0\}`. The treated
block is the units :math:`j = 1, \dots, J` and the donor pool is the
never-treated units :math:`\mathcal{N}_0 \coloneqq \mathcal{N} \setminus
\{1, \dots, J\}` of cardinality :math:`N_0`.

Stack the treated units' pre-period outcomes into the time-major matrix
:math:`\mathbf{Y}_1 \in \mathbb{R}^{T_0 \times J}` (column :math:`j` is treated
series :math:`\mathbf{y}_j`) and the never-treated donors into the donor matrix
:math:`\mathbf{Y}_0 \coloneqq [\mathbf{y}_j]_{j \in \mathcal{N}_0} \in
\mathbb{R}^{T_0 \times N_0}` (one column per donor) over
:math:`\mathcal{T}_1`. The donor-weight matrix
:math:`\boldsymbol{\Theta} \in \mathbb{R}^{N_0 \times J}` has one column per
treated unit; column :math:`j` is that unit's synthetic-control weights. The
nuclear norm :math:`\lVert \mathbf{A} \rVert_{*} \coloneqq \sum_i
\sigma_i(\mathbf{A})` sums the singular values of :math:`\mathbf{A}`. The
treatment effect on treated unit :math:`j` at post-period time :math:`t` is
:math:`\tau_{jt} \coloneqq y_{jt} - (\mathbf{Y}_0\widehat{\boldsymbol{\Theta}})_{tj}`,
and the ATT :math:`\widehat{\tau}` is its average over the treated block and
:math:`\mathcal{T}_2`.

The estimator
~~~~~~~~~~~~~~

MSQRT estimates the weight matrix by Multivariate Square-root Lasso (paper
Eq. 5):

.. math::

   \widehat{\boldsymbol{\Theta}} \;=\; \operatorname*{argmin}_{\boldsymbol{\Theta}}\;
     \frac{1}{\sqrt{T_0}}\,\bigl\lVert \mathbf{Y}_1 - \mathbf{Y}_0\boldsymbol{\Theta} \bigr\rVert_{*}
     \;+\; \lambda \sum_{i,j} \lvert \Theta_{ij} \rvert .

The first term is a *pivotal* (square-root-type) loss: the nuclear norm of the
residual matrix, scaled by :math:`1/\sqrt{T_0}`. Its key property is that the
penalty level :math:`\lambda` that delivers the oracle rate does not depend
on the unknown noise scale, so the same :math:`\lambda` regularises every column
of :math:`\boldsymbol{\Theta}` simultaneously. The second term is the
element-wise :math:`\ell_1` penalty driving sparse donor selection. The
synthetic counterfactual for the treated block is
:math:`\mathbf{Y}_0\widehat{\boldsymbol{\Theta}}`, extended into the post-period
with the post-period donor matrix :math:`\mathbf{Y}_{0,\mathcal{T}_2}`.

Internal solver (two-split ADMM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The objective is convex but couples a nuclear norm and an :math:`\ell_1`
penalty, so ``mlsynth`` solves it with a purpose-built two-split ADMM rather
than a generic conic solver (the paper itself emphasises computational
efficiency). Introduce two consensus blocks -- :math:`\mathbf{R} =
\mathbf{Y}_0\boldsymbol{\Theta}` carrying the nuclear-norm term and
:math:`\mathbf{Z} = \boldsymbol{\Theta}` carrying the :math:`\ell_1` term. With
scaled dual variables :math:`\mathbf{U}, \mathbf{W}` and penalty parameter
:math:`\rho`, each iteration is three closed-form updates (Boyd et al. 2011,
§3):

.. math::

   \boldsymbol{\Theta} &\;\leftarrow\; (\mathbf{Y}_0^\top\mathbf{Y}_0 + \mathbf{I})^{-1}
       \bigl(\mathbf{Y}_0^\top(\mathbf{R} - \mathbf{U}) + (\mathbf{Z} - \mathbf{W})\bigr), \\
   \mathbf{R} &\;\leftarrow\; \mathbf{Y}_1 - \mathcal{S}^{*}_{(1/\sqrt{T_0})/\rho}
       \bigl(\mathbf{Y}_1 - \mathbf{Y}_0\boldsymbol{\Theta} - \mathbf{U}\bigr), \\
   \mathbf{Z} &\;\leftarrow\; \mathcal{S}_{\lambda/\rho}
       \bigl(\boldsymbol{\Theta} + \mathbf{W}\bigr),

followed by the dual ascents :math:`\mathbf{U} \leftarrow \mathbf{U} +
(\mathbf{Y}_0\boldsymbol{\Theta} - \mathbf{R})` and :math:`\mathbf{W} \leftarrow
\mathbf{W} + (\boldsymbol{\Theta} - \mathbf{Z})`. Here
:math:`\mathcal{S}^{*}_\tau` is singular-value soft-thresholding -- the
nuclear-norm proximal map, which SVDs its argument and shrinks each singular
value toward zero by :math:`\tau` -- and :math:`\mathcal{S}_\tau` is elementwise
soft-thresholding, the :math:`\ell_1` proximal map. The returned weight matrix is
the exactly-sparse :math:`\mathbf{Z}` block.

Three design choices make this fast. The :math:`\boldsymbol{\Theta}`-step is an
*exact* least-squares solve whose system matrix
:math:`\mathbf{Y}_0^\top\mathbf{Y}_0 + \mathbf{I}` does not depend on
:math:`\rho`, so it is Cholesky-factored **once** and reused at every iteration;
the coupling is **over-relaxed**; and :math:`\rho` is **balanced adaptively** --
doubled or halved whenever the primal and dual residuals diverge by more than a
factor of ten. Iteration stops on the standard ADMM primal/dual residual
criterion (Boyd et al. 2011, §3.3): the primal residual
:math:`(\mathbf{Y}_0\boldsymbol{\Theta} - \mathbf{R},\,
\boldsymbol{\Theta} - \mathbf{Z})` and the dual residual
:math:`\rho\,\mathbf{A}^\top(\mathbf{z}^{k} - \mathbf{z}^{k-1})` both fall below
an absolute/relative tolerance. A high-dimensional fit therefore converges in a
few hundred lightweight iterations -- orders of magnitude faster than the conic
reformulation, which it matches to numerical precision. A ``cvxpy`` backend
(:func:`~mlsynth.utils.msqrt_helpers.optimization._fit_msqrt_cvxpy`) is retained
as a validation oracle, selectable with ``solver="cvxpy"``.

Tuning the penalty
~~~~~~~~~~~~~~~~~~~

The penalty :math:`\lambda` is chosen by rolling-origin (expanding-window)
cross-validation on the pre-period. From an initial training window the model is
fit; the next ``cv_val_window`` periods are forecast and scored by mean squared
error; the window expands by ``cv_step`` and the fold repeats; the :math:`\lambda`
minimising the mean validation MSE across folds is selected. The schedule adapts
to :math:`T_0` by default (initial window :math:`\approx 0.6\,T_0`, validation
block :math:`\approx T_0/5`) and is overridable (``cv_initial_train``,
``cv_val_window``, ``cv_step``, ``cv_folds``). Two properties keep it cheap and
honest: the search is **pathwise** -- candidate penalties are visited in
descending order and each ADMM solve is warm-started from the previous
(larger-:math:`\lambda`, sparser) solution, which sharply cuts the iteration
count without changing the selected penalty -- and the expanding window never
lets a future period inform an earlier fit. A fixed ``lambda_`` skips
cross-validation entirely.

Assumptions and remarks
~~~~~~~~~~~~~~~~~~~~~~~~

1. Block design. All treated units adopt at the same period, and the panel is
   balanced.

   *Remark.* This is what lets the treated units share one
   :math:`\boldsymbol{\Theta}` problem; ``mlsynth`` enforces it at ingestion and
   points staggered panels to :doc:`sdid`/:doc:`mcnnm`.

2. Approximate factor structure / sparsity. Each treated unit is well
   approximated by a sparse combination of donors driven by a common
   low-dimensional factor structure.

   *Remark.* The nuclear-norm loss exploits the shared factor structure across
   treated units; the :math:`\ell_1` penalty keeps the per-unit weight vectors
   sparse and interpretable when :math:`N_0 \gtrsim T_0`.

3. Pivotal tuning. The noise is homoskedastic enough that the square-root loss's
   variance-free penalty calibration applies.

   *Remark.* This is the payoff of the nuclear-norm formulation -- one
   :math:`\lambda` for the whole matrix, no per-unit variance estimation.

Causal use
----------

``MSQRT`` forms
:math:`\tau_{jt} = y_{jt} - (\mathbf{Y}_0\widehat{\boldsymbol{\Theta}})_{tj}`
over the post-treatment block and aggregates to the overall ATT
:math:`\widehat{\tau}` (mean post-period gap), reporting it both in levels
(``att``) and as a percentage of the synthetic
counterfactual (``att_percent``). It also exposes ``att_t`` (mean treated gap at
each post period) and ``unit_att`` (per-treated-unit post-period mean gap), so
heterogeneity across the treated block is visible.

Inference
---------

The Shen, Song & Abadie paper establishes finite-sample *estimation-error
bounds* for :math:`\widehat{\boldsymbol{\Theta}}` rather than a
confidence-interval procedure for the treatment effect. For uncertainty
quantification ``mlsynth`` instead uses the non-asymptotic prediction intervals
of Cattaneo, Feng, Palomba and Titiunik [SCPI]_ (the ``scpi`` framework). Those
intervals decompose the prediction error :math:`\widehat{\tau} - \tau` into an
in-sample error (from estimating the SC weights) and an out-of-sample error (the
irreducible post-treatment sampling noise), and bound each separately:

.. math::

   I(\tau) = \bigl[\, \widehat{\tau} - \overline{M}_{\text{in}}
       - \overline{M}_{\text{out}},\;\;
       \widehat{\tau} - \underline{M}_{\text{in}}
       - \underline{M}_{\text{out}} \,\bigr].

The in-sample bound is derived from the optimality condition of a
quadratic-loss SC program. MSQRT instead minimises a nuclear-norm
square-root-Lasso objective with no constraints on
:math:`\boldsymbol{\Theta}`, so that derivation does not strictly apply.
``mlsynth`` therefore models, for MSQRT, only the out-of-sample error -- the
rigorous component -- via the sub-Gaussian concentration bound

.. math::

   \overline{M}_{\text{out}},\,\underline{M}_{\text{out}}
     = \widehat{\mathbb{E}}[u\mid\mathcal{H}]
       \pm \sqrt{2\,\widehat{\sigma}^2 \log(2/\alpha_{\text{out}})},

with the conditional mean and variance proxy :math:`\widehat{\sigma}^2` estimated
from each treated unit's pre-treatment residuals. The full miscoverage budget
:math:`\alpha` is spent on the out-of-sample band
(:math:`\alpha_{\text{in}} = 0`). The resulting intervals reflect
post-treatment sampling uncertainty but not weight-estimation uncertainty;
they are correspondingly narrower, and this is recorded on the result
(``in_sample_included = False``).

Setting ``inference=True`` returns a
:class:`~mlsynth.utils.scpi_helpers.structures.SCPIResults` with the four
``scpi`` predictands, each with its own band:

* ``taua`` -- T\ ime-a\ veraged u\ nit-a\ veraged: the overall ATT.
* ``tsua`` -- T\ ime-s\ pecific u\ nit-a\ veraged: ``{period: band}``,
  the average effect across treated units at each post-period.
* ``taus`` -- T\ ime-a\ veraged u\ nit-s\ pecific: ``{unit: band}``,
  each treated unit's effect averaged over the post-window.
* ``tsus`` -- T\ ime-s\ pecific u\ nit-s\ pecific:
  ``{(unit, period): band}``, the per-cell effect.

plus ``simultaneous`` -- TSUS bands widened (Bonferroni over the post-periods)
for joint coverage across all post-periods of a unit, which supports
statements such as "the largest per-period effect is significant". The
time-averaged predictands accept a ``time_dependence`` setting (``"iid"``,
default, shrinks the band by :math:`\sqrt{L}`; ``"general"`` makes no
serial-dependence assumption). The reusable
:mod:`mlsynth.utils.scpi_helpers` module also implements the in-sample
simulation bound for use by quadratic-loss SC estimators.

Example
-------

A high-dimensional, multiple-treated panel in the Shen-Song-Abadie regime: five
treated units adopt together at period 100, against forty never-treated donors
following an AR(1) process, with each treated unit a sparse convex combination
of the donors plus noise and a constant treatment effect of ``+2``. ``MSQRT``
pools the treated units, selects :math:`\lambda` by cross-validation, and
recovers the ATT with a CFPT/scpi prediction band.

.. code-block:: python

   from mlsynth import MSQRT
   from mlsynth.utils.msqrt_helpers.simulation import simulate_msqrt_panel

   df = simulate_msqrt_panel(
       n_treated=5, n_control=40, T0=100, n_post=10, att=2.0, seed=0,
   )

   res = MSQRT({
       "df": df, "outcome": "Y", "treat": "treated",
       "unitid": "unit", "time": "time",
       "n_lambda": 15,            # log-spaced CV grid
       "inference": True,         # CFPT/scpi prediction intervals
       "display_graphs": True,    # treated mean vs synthetic mean
   }).fit()

   print(f"ATT (true 2.0)   = {res.att:+.3f}")
   lo, hi = res.att_ci                          # overall ATT (TAUA) band
   print(f"90% scpi band    = [{lo:+.3f}, {hi:+.3f}]")
   # per-period (unit-averaged) effects with their bands (raw SCPI object)
   for period, band in res.inference_intervals.tsua.items():
       print(f"  k={period}: {band.point:+.3f}  [{band.lower:+.3f}, {band.upper:+.3f}]")
   print(f"selected lambda  = {res.best_lambda:.3f}")
   print(f"avg active donors per treated = "
         f"{res.weights.summary_stats['avg_active_donors_per_treated']:.1f}")

This prints an ATT of roughly ``+2.05`` with a 90% band that brackets the true
``+2`` and a pre-treatment RMSE below the noise floor.

Verification
------------

.. note::

   Monte-Carlo replication. :mod:`mlsynth.utils.msqrt_helpers.replication`
   reproduces the Shen, Song & Abadie [MSQRT]_ simulation study (their
   Section 6) through the public :meth:`mlsynth.MSQRT.fit` API:
   :math:`T_0 = 100` pre-periods, :math:`N_0 = 400` donors, a treated-block grid
   :math:`J \in \{50, \ldots, 400\}`, and the paper's two data-generating
   settings. The headline findings reproduce -- the ATT estimator is
   essentially unbiased (bias centred at zero) and the RMSE for the imputed
   :math:`\mathbf{Y}(0)` stays flat in :math:`J` at roughly ``0.71``-``0.73``,
   matching the paper's Table 1. The ``PAPER`` preset runs the full
   500-replication study; the ``DEMO`` preset is a faster, reduced-count
   version.

   Solver. The two-split ADMM matches the cvxpy conic solution of eq. (5) to
   numerical precision while being orders of magnitude faster -- the property
   that makes the high-dimensional, many-treated regime tractable.

   Block-design guard. Feeding a staggered panel raises
   :class:`~mlsynth.exceptions.MlsynthDataError`, redirecting to the
   staggered-adoption estimators -- MSQRT's assumptions are enforced, not
   assumed.

Core API
--------

.. automodule:: mlsynth.estimators.msqrt
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.MSQRTConfig
   :members:
   :undoc-members:

Result Containers
-----------------

``MSQRT.fit()`` returns a
:class:`~mlsynth.utils.msqrt_helpers.structures.MSQRTResults`: the ATT (levels
and percent), the donor-weight matrix :math:`\boldsymbol{\Theta}`, per-treated-unit and
aggregate donor weights (:class:`~mlsynth.config_models.WeightsResults`), the
synthetic counterfactual and gap matrices, ``att_t`` and ``unit_att``, the
cross-unit observed/synthetic means the plotter draws, the CV-selected
``best_lambda``, per-unit sparsity, the pre-period RMSE, and -- when requested
-- the :class:`~mlsynth.utils.scpi_helpers.structures.SCPIResults` prediction
intervals.

.. automodule:: mlsynth.utils.msqrt_helpers.structures
   :members:
   :undoc-members:
   :show-inheritance:

Helper Modules
--------------

Data preparation -- the DataFrame touchpoint: pivots the long panel into the
stacked :math:`\mathbf{Y}_1 = \mathbf{Y}_0\boldsymbol{\Theta} + \mathbf{E}` block
matrices and enforces the block design.

.. automodule:: mlsynth.utils.msqrt_helpers.setup
   :members:
   :undoc-members:

The two-split ADMM solve (singular-value + element-wise soft-thresholding with
a cached exact :math:`\boldsymbol{\Theta}` step, adaptive penalty and over-relaxation) and
the warm-started pathwise rolling-origin cross-validation over :math:`\lambda`.
A ``cvxpy`` reference solve is retained for validation.

.. automodule:: mlsynth.utils.msqrt_helpers.optimization
   :members:
   :undoc-members:

Run loop: CV / fit, ATT and per-period/per-unit effects, donor-weight
assembly, and the optional scpi prediction intervals.

.. automodule:: mlsynth.utils.msqrt_helpers.pipeline
   :members:
   :undoc-members:

The data-generating process: AR(1) donors and a sparse convex donor-weight
matrix (the Shen-Song-Abadie regime), wrapped into a long panel with an
injected treatment effect for examples and tests.

.. automodule:: mlsynth.utils.msqrt_helpers.simulation
   :members:
   :undoc-members:

Replication of the paper's Monte-Carlo study (Section 6) through the public
``MSQRT.fit`` API -- the two data-generating settings, the ATT-bias and RMSE
metrics, and the ``PAPER`` / ``DEMO`` presets.

.. automodule:: mlsynth.utils.msqrt_helpers.replication
   :members:
   :undoc-members:

Uncertainty quantification is provided by the shared CFPT/scpi module
(Cattaneo, Feng, Palomba & Titiunik 2025), which any synthetic-control
estimator can reuse.

.. automodule:: mlsynth.utils.scpi_helpers.intervals
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.scpi_helpers.moments
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.scpi_helpers.structures
   :members:
   :undoc-members:
   :show-inheritance:
