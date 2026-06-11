Synthetic Historical Control (SHC)
==================================

.. currentmodule:: mlsynth

Overview
--------

The Synthetic Historical Control (SHC) method estimates the time-varying
intervention effect on a **single treated unit using only its own time
series** — no cross-sectional control units are required. It is the answer
to the setting where *every* unit is treated (a nationwide policy, a global
shock such as COVID-19) so the synthetic control method has no donor pool
to draw on.

SHC is due to Chen, Yang & Yang (2024). It builds on a semi-parametric
time-series regression in which a smooth latent trend :math:`\ell_t` is a
time-varying confounder. Rather than extrapolate an *assumed* parametric
trend (as the interrupted-time-series, Prophet, and CausalImpact
approaches do), SHC carries the synthetic-control idea over to a single
series: it replaces the treated *unit* with a treated *block* and the
cross-sectional donors with overlapping *historical blocks* from the same
series, then matches the treated block's pre-intervention segment with a
simplex combination of its historical counterparts. The matching quality
is therefore *detectable* from the pre-period fit, unlike parametric trend
extrapolation whose misspecification is invisible after the intervention.

When to use this estimator
--------------------------

Reach for SHC when **there is one treated unit and no credible untreated
controls**, but a reasonably long pre-intervention series with **recurring
local structure** (cycles, seasonal-like swings — not strict periodicity):

* **Nationwide / global interventions.** A country-level minimum-wage hike,
  a national pension reform, or the macroeconomic impact of a pandemic,
  where no other unit is plausibly untreated. The paper's applications are
  Brexit's effect on UK GDP growth and COVID-19's effect on US GDP growth.
* **Cross-sectional controls exist but fail the SC matching condition.**
  Even when donors are available, SHC can match the treated pre-period
  better than SC if the donors track the treated series poorly (the paper's
  Brexit case: SHC pre-period MSE 0.029 vs SC 0.256).

If you have a clean panel with valid untreated donors, the cross-sectional
estimators (:class:`mlsynth.CLUSTERSC`, :class:`mlsynth.PDA`,
:class:`mlsynth.SBC`) are the appropriate tools.

Mathematical formulation
------------------------

Setup
^^^^^

For a single treated unit with outcome :math:`\{y_t\}` and intervention
indicator :math:`d_t` (0 for :math:`t \le T_o`, 1 afterwards), the
semi-parametric model (Eq. 2; the implementation uses the simplified
:math:`x_t`-free form) is

.. math::

   y_t = \ell_t + \delta_t d_t + \varepsilon_t,

where :math:`\ell_t = \ell(t)` is a non-stochastic, smooth latent trend,
:math:`\delta_t` is the time-varying intervention effect, and
:math:`\varepsilon_t` is a zero-mean error. Because both :math:`\ell_t` and
:math:`\delta_t` are unobserved post-intervention, naive pre/post or
semi-parametric (Robinson 1988) methods cannot separate the two; SHC
identifies :math:`\delta_t` by reconstructing the post-intervention
:math:`\ell_t`.

Treated and historical blocks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fix a pre-intervention block length :math:`m` and a post horizon
:math:`n`. The **treated block** spans :math:`[T_o - (m-1),\, T_o + n]`,
with pre-segment :math:`\boldsymbol\ell_{pre}` and post-segment
:math:`\boldsymbol\ell_{post}`. The pre-period is sliced into
:math:`N = T_o - n - (m-1)` overlapping **historical blocks**, each with
the same pre/post split (Eq. 7). The SHC weights solve a simplex-matching
problem on the latent pre-segments,

.. math::

   \widehat{\boldsymbol w}
     = \arg\min_{\boldsymbol w \in \mathbb{W}}
       \bigl\| \widehat{\boldsymbol\ell}_{pre}
       - \widehat{\boldsymbol L}_{pre} \boldsymbol w \bigr\|^2,
   \qquad
   \mathbb{W} = \{\boldsymbol w \ge 0 : \mathbf{1}^\top \boldsymbol w = 1\},

and the post-intervention counterfactual is the same combination applied
to the historical forward segments,
:math:`\widehat\ell_t(\widehat{\boldsymbol w}) = \sum_i \widehat w_i \,
\widehat\ell_{t(i)}`.

Identifying assumptions
^^^^^^^^^^^^^^^^^^^^^^^^

*Assumption 1 (regularity).* :math:`\{\varepsilon_t\}` is i.i.d. with zero
mean and finite variance. *Remark.* This is what identifies the
(semi-parametric) nuisance components in the pre-period.

*Assumption 2(a) (smoothness).* :math:`\ell(\cdot)` has a bounded
:math:`(H+1)`-th derivative, with :math:`m - 1 \ge H \ge 3`. *Remark.* The
degree of smoothness :math:`H` controls the bias bound
:math:`b_\epsilon(H, k) = 2\epsilon |k|^{H+1}/(H+1)!` (Proposition 1): the
estimator is *approximately unbiased*, with bias vanishing as the latent
component gets smoother or the post horizon :math:`k` shrinks. This is why
SHC favors a **small post horizon** and why larger-horizon estimates
should be read cautiously.

*Assumption 2(b) (matching).* The treated pre-segment is reproducible as a
convex combination of its historical counterparts,
:math:`\boldsymbol\ell_{pre} = \boldsymbol\ell_{pre}(\boldsymbol w_o)` for
some :math:`\boldsymbol w_o \in \mathbb{W}`. *Remark.* This is the
distributional analogue of the SC matching condition, transplanted from
cross-sectional donors to historical blocks. It is **checkable** from the
pre-period fit. It also precludes a pure growth trend (which cannot be
reproduced by its own history), so differencing/detrending the series
first is recommended.

Algorithm (implementation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The two-stage estimator (Section 2.3) is orchestrated by
:func:`mlsynth.utils.shc_helpers.orchestration.solve_shc`:

1. **Latent trend.** Estimate :math:`\widehat\ell_t` over the pre-period by
   local-linear kernel regression, with the bandwidth chosen by
   leave-one-out cross-validation (``bandwidth_grid``).
2. **Blocks.** Build the treated block and the :math:`N` historical blocks.
3. **Matching.** Weight *all* :math:`N` historical blocks by the
   simplex-constrained matching QP (Eq. 23), solving the nearest-PD
   approximation :math:`\widehat{\boldsymbol L}_{\mathrm{pre}}^{\top}
   \widehat{\boldsymbol L}_{\mathrm{pre}} + \varsigma C_2 C_2^{\top}`; the
   simplex constraint itself zeroes out the irrelevant blocks.
4. **Augmentation (optional).** ``use_augmented=True`` adds an
   ASHC ridge refinement on top of the simplex weights.
5. **Counterfactual.** Apply the weights to the historical forward
   segments to obtain :math:`\widehat\ell_t(\widehat{\boldsymbol w})` over
   the post horizon; the gap :math:`y_t - \widehat\ell_t(\widehat{\boldsymbol w})`
   estimates :math:`\delta_t`.

Inference
^^^^^^^^^

``SHC`` reports the conformal permutation test of Chen, Yang & Yang (2024,
footnote 21) — their application of Chernozhukov, Wüthrich & Zhu (2021) —
for the sharp null :math:`H_0: \delta_t = 0` over the post period. The test
statistic is

.. math::

   S = n^{-1/2} \sum_{t=T_o+1}^{T_o+n} \bigl| \hat\varepsilon_t^0 \bigr|,
   \qquad \hat\varepsilon_t^0 = y_t - \widehat\ell_t,

and the null distribution is built by sampling :math:`n` residuals **with
replacement** from the :math:`T_o` pre-intervention residuals, 1,000 times.
``results.inference`` exposes ``p_value``, ``test_statistic``, the 1/5/10%
``critical_values`` and ``reject`` decisions, the resampled
``null_distribution``, and Andrews-Genton conformal bands for the plot.

.. note::

   The test is designed for the empirical setting where a genuine effect is
   present (in the paper's Brexit application it rejects at the 1% level:
   :math:`S = 2.492 > 2.190`). Because the reference residuals are the
   in-sample kernel-smoother residuals, which are mildly under-dispersed
   relative to the true noise, the test can over-reject under an exact
   null; the paper does not run it in the (effect-free) simulation.

Core API
--------

.. autoclass:: mlsynth.SHC
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Configuration
-------------

.. autoclass:: mlsynth.config_models.SHCConfig
   :members:
   :undoc-members:

Helper Modules
--------------

.. automodule:: mlsynth.utils.shc_helpers.setup
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.shc_helpers.orchestration
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.shc_helpers.inference
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.shc_helpers.structures
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.shc_helpers.plotter
   :members:
   :undoc-members:

Example
-------

A self-contained one-draw example using the paper's own data-generating
process (a smooth, recurring latent trend plus noise, with no intervention
effect, so the counterfactual should track the latent component). Paste it
into a fresh interpreter:

.. code-block:: python

   import numpy as np
   from mlsynth import SHC
   from mlsynth.utils.shc_helpers import simulate_shc_panel

   # One panel from the Chen-Yang-Yang (2024) DGP (Section 3.1):
   # T_o = m(4h+1) = 90 pre-periods, n = 8 post-periods, delta_t = 0.
   df, info = simulate_shc_panel(
       m=10, h=2, n=8, P=10, sigma=0.1, w_f=(1, 0), regular=True, seed=0,
   )

   res = SHC({
       "df": df, "outcome": "y", "treat": "treated",
       "unitid": "unit", "time": "time", "m": 10, "display_graphs": False,
   }).fit()

   # SHC reconstructs the latent confounder over the post window.
   cf_post = res.counterfactual[10:]
   mse_post = np.mean((cf_post - info["latent_post"]) ** 2)
   print(f"ATT                = {res.att:+.4f}   (true effect = 0)")
   print(f"pre-period R^2     = {res.fit_diagnostics['r_squared_pre']:.3f}")
   print(f"MSE_post vs latent = {mse_post:.5f}")
   print(f"historical blocks  = {res.inputs.N}, "
         f"selected = {len(res.weights_by_block)}")
   print(f"conformal p-value  = {res.inference.p_value:.3f}")

Monte Carlo Validation
======================

The estimator is validated against the paper's simulation design
(Section 3.1), re-implemented in
:mod:`mlsynth.utils.shc_helpers.simulation`. The latent confounder
:math:`\ell_t` is a globally :math:`C^1` curve of alternating cosine
"local trends" and cubic-Hermite connectors; the treated block's shape is
a convex combination of :math:`h` historical shapes (Assumption 2(b)). The
construction reproduces the paper's exact dimensions: with :math:`h = 4`,
:math:`T_o = m(4h+1)` (425 for :math:`m=25`, 850 for :math:`m=50`) and
:math:`N = T_o - n - (m-1)` historical blocks (376 and 776).

With :math:`\delta_t = 0`, the exercise measures how well SHC recovers
:math:`\ell_t`, via the mean squared matching error (``MSE_pre``, Eq. 31)
and the mean squared prediction error against the *true* latent
(``MSE_post(k)``, Eq. 38).

.. code-block:: python

   from mlsynth.utils.shc_helpers import monte_carlo_shc

   out = monte_carlo_shc(
       n_reps=8, m=25, h=4, n=25, P=10, sigma=0.1,
       w_f=(1, 0, 0, 0), regular=True, k_grid=(1, 5, 10, 15, 25),
   )
   print(out["mse_pre"], out["mse_post"])

Representative output (Regular-:math:`\ell`, :math:`\sigma = 0.1`,
:math:`m = 25`)::

   MSE_pre      = 0.0011
   MSE_post(1)  = 0.0010
   MSE_post(5)  = 0.0017
   MSE_post(10) = 0.0016
   MSE_post(15) = 0.0017
   MSE_post(25) = 0.0016

Both measures are near zero, and ``MSE_post(k)`` rises from :math:`k = 1`
before plateauing — consistent with the paper's finding that the bias
bound (Proposition 1) grows with the horizon but stays mild for a smooth,
regularly recurring latent component at low noise.

.. automodule:: mlsynth.utils.shc_helpers.simulation
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.shc_helpers.monte_carlo
   :members:
   :undoc-members:

Developer API
=============

Internal optimization and tuning routines for SHC and ASHC. When
``use_augmented`` is true, the simplex SHC weights from the matching QP
are passed to the ASHC ridge refinement for bias correction.

.. autofunction:: mlsynth.utils.inferutils.shc_conformal_test
.. autofunction:: mlsynth.utils.estutils._solve_SHC_QP
.. autofunction:: mlsynth.utils.estutils.tune_lambda_ashc

References
----------

Chen, Yi-Ting, Jui-Chung Yang, and Tzu-Ting Yang (2024). "Synthetic
Historical Control for Policy Evaluation." SSRN 4995085.

Chernozhukov, V., Wüthrich, K., & Zhu, Y. (2021). "An Exact and Robust
Conformal Inference Method for Counterfactual and Synthetic Controls."
*Journal of the American Statistical Association* 116(536):1849-1864.

Hamilton, J. D. (2018). "Why You Should Never Use the Hodrick-Prescott
Filter." *Review of Economics and Statistics* 100(5):831-843.

Robinson, P. M. (1988). "Root-N-Consistent Semiparametric Regression."
*Econometrica* 56(4):931-954.
