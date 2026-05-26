Distributional Synthetic Control (DSC)
======================================

.. currentmodule:: mlsynth

Overview
--------

DSC generalizes the synthetic control method from *aggregate values* to
*entire outcome distributions*. Where classical synthetic control builds a
donor combination that matches the treated unit's pre-period **mean** and
returns a single ATT, DSC builds a donor combination that matches the
treated unit's pre-period **quantile function** and returns the full
counterfactual distribution at every post-period -- hence the quantile
treatment effect (QTE) at any quantile :math:`q \in (0, 1)`, and any
functional of the distribution (Lorenz curves, Gini coefficients,
interquartile ranges, stochastic-dominance comparisons).

The method is due to Gunsilius (2023); mlsynth's implementation is
validated against the author's reference ``DiSCo`` R package, and the
large-sample theory is from Zhang, Zhang & Zhang (2026). The core idea is
the **2-Wasserstein barycenter**: the natural
generalization of a weighted average from points on the real line to
probability distributions. Averaging quantile functions (not densities)
keeps the synthetic unit *geometrically faithful* -- it reproduces the
support, the moments, and the quantiles of the target -- whereas a naive
linear mixture of densities is multimodal and matches none of them.

When to use this estimator
--------------------------

Reach for DSC when **the distribution is the object of interest, and you
have micro-data**. Three motivating regimes:

* **Inequality and the whole distribution, not the mean.** A minimum-wage
  increase, a cash-transfer program, or a tax reform may leave the mean of
  family income almost unchanged while compressing the lower tail or
  fattening the middle. Gunsilius (2023, Section 6.2) studies exactly this:
  the distribution of equalized family income (as multiples of the federal
  poverty line) across U.S. states, treating Michigan as the unit of
  interest. A mean-only synthetic control would miss the distributional
  story entirely.

* **Heterogeneous treatment effects across the outcome distribution.** In
  marketing, a promotion might lift spending among already-heavy buyers
  while doing nothing for light buyers; the QTE curve :math:`q \mapsto
  \widehat\alpha_{1t, q}` reveals *where* in the spending distribution the
  effect lands. DSC returns this directly, without pre-specifying which
  quantile to test.

* **Repeated cross-sections with many individuals per cell.** Whenever each
  ``(unit, time)`` cell is itself a sample -- households in a state-year,
  customers in a store-week, patients in a hospital-month -- DSC uses *all*
  of that within-cell information rather than collapsing it to a mean.

If you only have one aggregate number per ``(unit, time)`` cell, DSC
reduces to classical synthetic control (Gunsilius 2023, Section 3.1) and
the dedicated aggregate estimators (e.g. :class:`mlsynth.CLUSTERSC`,
:class:`mlsynth.FMA`) are the appropriate tools.

Notation
--------

We follow the repository's canonical conventions. There are :math:`J + 1`
units; unit :math:`1` is treated and units :math:`j = 2, \dots, J + 1` are
donors. The panel runs over periods :math:`t \in \{1, \dots, T\}` with
treatment beginning at :math:`T_0 + 1`; the pre-period is
:math:`\mathcal{T}_0 = \{1, \dots, T_0\}` and the post-period is
:math:`\mathcal{T}_1 = \{T_0 + 1, \dots, T\}`. Each cell :math:`(j, t)`
carries :math:`n_{jt}` individual observations
:math:`\{Y_{l, jt}\}_{l = 1}^{n_{jt}}`. :math:`F_{Y_{jt}}` is the cell's
outcome distribution and :math:`F^{-1}_{Y_{jt}}` its quantile function;
:math:`\mathcal{H} = \{w \in \mathbb{R}^J_{\ge 0} : \mathbf{1}^\top w = 1\}`
is the unit simplex. Weights :math:`\widehat w_t` are fit per pre-period
and aggregated to :math:`\widehat w`. :math:`W_2` denotes the 2-Wasserstein
distance, and :math:`F^{-1}_{Y_{1t, N}}` the *counterfactual* (no-treatment)
quantile function of the treated unit.

Mathematical Formulation
------------------------

Setup
^^^^^

The empirical quantile estimator for a cell is the order-statistic rule

.. math::

   \widehat F^{-1}_{Y_{jt, n_j}}(q) = Y_{t, n_j(k)},
   \quad \frac{k - 1}{n_j} < q \le \frac{k}{n_j},

where :math:`Y_{t, n_j(k)}` is the :math:`k`-th order statistic of cell
:math:`(j, t)`. The DSC counterfactual quantile function for the treated
unit at a post-period :math:`t > T_0` is the **2-Wasserstein barycenter**
of the donor quantile functions,

.. math::

   \widehat F^{-1}_{Y_{1t, N}}(q) =
       \sum_{j = 2}^{J + 1} \widehat w_j\, \widehat F^{-1}_{Y_{jt, n_j}}(q),
   \qquad \widehat w \in \mathcal{H},

and the quantile treatment effect is
:math:`\widehat \alpha_{1t, q} = \widehat F^{-1}_{Y_{1t, I}}(q)
- \widehat F^{-1}_{Y_{1t, N}}(q)`, the gap between the *observed* treated
quantile function and its counterfactual. Averaging quantile functions
rather than densities is what makes the synthetic unit geometrically
faithful: a weighted average of quantile functions is itself a valid
quantile function, so the barycenter has the same kind of support and shape
as the target (Gunsilius 2023, Figure 1; Agueh & Carlier 2011).

Identifying assumptions
^^^^^^^^^^^^^^^^^^^^^^^^

The structural content sits in the causal model :math:`h(t, \cdot)` mapping
latent variables to observed distributions. DSC recovers the *correct*
counterfactual under a distributional analogue of the classical
factor-model restriction.

*Assumption 1 (scaled-isometry causal model -- identification).* The
data-generating map :math:`h(t, \cdot)` on the 2-Wasserstein space is a
**scaled isometry** for every :math:`t`: it preserves relative distances
between distributions up to a common scale,
:math:`d(U_1, U_j) = \tau\, d\bigl(h(t, U_1), h(t, U_j)\bigr)`. Then
(Gunsilius 2023, Theorem 1) the DSC quantile function
:math:`\widehat F^{-1}_{Y_{1t, N}}` coincides with the treated unit's
quantile function had it not been treated. For the panel model it suffices
that the maps :math:`U_{jt} \mapsto U_{jt'}` preserve the optimal weights
:math:`\widehat\lambda^\star` across periods. *Remark.* This is the
distributional counterpart of the linear factor model behind classical
synthetic control: on the real line, scaled isometries in Euclidean space
are exactly affine maps, so the classical method's affine factor structure
is the special case. Working in Wasserstein space buys generality -- by the
Kloeckner (2010) characterization, isometries there can even deform the
support -- so identification can hold with as little as one pre- and one
post-period, analogous to changes-in-changes (Athey & Imbens 2006), without
a rank-invariance assumption.

*Assumption 2 (finite second moments and independence -- consistency).*
All cell distributions have finite second moments,
:math:`\mathbb{E}[Y_{jt}^2] < \infty`, and are independent across units for
each :math:`t`. *Remark.* This is all that is needed for the estimated
weights to be consistent (Gunsilius 2023, Proposition 1): the plug-in
empirical-quantile weights :math:`\widehat{\widetilde\lambda}^\star_{tn}`
converge to the population optimal weights as the within-cell sample sizes
grow. It is deliberately weak -- no smoothness or continuity is required
just to estimate the weights.

*Assumption 3 (absolute continuity -- uniform inference).* For uniform
results on the whole quantile process, each :math:`F_{Y_{jt}}` is
absolutely continuous with a density bounded away from zero on its support.
*Remark.* This is the standard regularity condition (Csörgő & Horváth 1993)
that delivers the Gaussian large-sample distribution of the counterfactual
quantile process (Proposition 2) and hence bootstrap confidence bands. It
can be relaxed at the cost of a non-Gaussian limit (discrete case,
Proposition 5), which is why the inference below leans on Monte-Carlo /
permutation routines rather than closed-form critical values.

Algorithm
^^^^^^^^^

mlsynth implements Gunsilius's four-step recipe (formalized as Algorithm 1
in Zhang et al. 2026).

**Step 1 -- Empirical quantile functions.** For each cell :math:`(j, t)`,
compute :math:`\widehat F^{-1}_{Y_{jt, n_j}}` via the order-statistic
estimator above.

**Step 2 -- Per-pre-period weights.** Draw an :math:`M`-point quantile grid
:math:`\{V_m\}_{m=1}^M \subset (0, 1)` (Halton / Sobol low-discrepancy by
default, or uniform i.i.d.) and form the pseudo-sample matrices
:math:`\widetilde Y_{jt, m} = \widehat F^{-1}_{Y_{jt, n_j}}(V_m)`. The
squared 2-Wasserstein loss
:math:`W_2^2(\cdot) = \int_0^1 \lvert \sum_j w_j \widehat F^{-1}_{Y_{jt}}(q)
- \widehat F^{-1}_{Y_{1t}}(q) \rvert^2 dq` is approximated by the empirical
risk :math:`L_t(w) = M^{-1} \sum_m \lvert \widetilde Y_{1t, m} - \sum_j w_j
\widetilde Y_{jt, m}\rvert^2`, and the per-pre-period weights solve the
simplex-constrained quadratic program

.. math::

   \widehat w_t = \arg\min_{w \in \mathcal{H}} L_t(w),
   \qquad t \in \mathcal{T}_0.

mlsynth solves this with **accelerated projected gradient descent** (FISTA,
Beck & Teboulle 2009) and the exact simplex projection of Duchi et al.
(2008). This is the dependency-free analogue of the reference R package's
``pracma::lsqlincon`` constrained least-squares call, and -- unlike a
generic SLSQP solver -- it stays robust as the donor pool grows into the
hundreds, the regime the method is designed for (Section 6.1). By the
Koksma-Hlawka inequality the QMC approximation error is
:math:`O(\log M / M)` for Halton / Sobol vs. :math:`O(M^{-1/2})` for i.i.d.
draws.

**Step 3 -- Aggregate over the pre-period.** The final weight is a convex
combination :math:`\widehat w = \sum_{t \in \mathcal{T}_0} \lambda_t
\widehat w_t`, with :math:`\lambda_t \ge 0` and :math:`\sum_t \lambda_t = 1`.
mlsynth offers ``"uniform"`` (default; :math:`\lambda_t = 1/T_0`) and
``"recency"`` (geometric decay) rules, and accepts caller-supplied weights
so Arkhangelsky et al. (2021) SDiD-style :math:`\lambda_t` can be plugged in.

**Step 4 -- Post-period QTE.** For each :math:`t > T_0`, evaluate the
counterfactual quantile function at the QTE grid and difference it against
the observed treated quantile function to obtain
:math:`\widehat\alpha_{1t, q}`.

Inference
^^^^^^^^^

**Placebo permutation test (Gunsilius 2023, Algorithm 1).** The
distributional analogue of the Abadie-Diamond-Hainmueller (2010) placebo
test, and of the reference package's ``DiSCo_per``. Each donor is treated
as a pseudo-treated unit in turn: DSC is refit on the remaining units (the
real treated unit re-enters the donor pool), and the per-period squared
2-Wasserstein distance between that unit's observed and barycenter quantile
functions is recorded,

.. math::

   d_{\iota t} = \int_0^1 \bigl| F^{-1}_{Y_{\iota t, N}}(q)
                  - F^{-1}_{Y_{\iota t}}(q) \bigr|^2 dq .

If the model fits the placebos pre-treatment and there is a genuine
post-treatment effect, the real treated unit's distance sits in the extreme
upper tail of the :math:`J + 1` distances. The permutation p-value at
post-period :math:`t` is :math:`p_t = r(d_{1t}) / (J + 1)`, the rank of the
treated unit's distance (rank 1 = largest). Enable it with
``compute_inference=True``; the :class:`DSCInference` object exposes the
full distance paths (treated and every placebo) and the per-post-period
p-values. Because it refits the weights :math:`J` times it costs roughly
:math:`J\times` the point estimate.

**Goodness-of-fit and large-sample bands.** Working with whole
distributions also licenses a Wasserstein goodness-of-fit test of
:math:`H_0: F_{Y_{1t, N}} = F_{Y_{1t}}` and tests of first-/second-order
stochastic dominance (Gunsilius 2023, Propositions 3-4). Under Assumption 3
the counterfactual quantile process is asymptotically Gaussian
(Proposition 2), so confidence bands follow from the standard bootstrap; in
the discrete case the limit is non-Gaussian (Proposition 5) and a
Monte-Carlo plug-in is used instead. The pre-period fit
:math:`\xi_t` is surfaced directly via
:py:attr:`DSCResults.pre_period_wasserstein` so users can check fit quality
before trusting any post-period contrast.

Core API
--------

.. automodule:: mlsynth.estimators.dsc
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.DSCConfig
   :members:
   :undoc-members:

Helper Modules
--------------

.. automodule:: mlsynth.utils.dsc_helpers.setup
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.dsc_helpers.quantiles
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.dsc_helpers.weights
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.dsc_helpers.aggregation
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.dsc_helpers.pipeline
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.dsc_helpers.inference
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.dsc_helpers.structures
   :members:
   :undoc-members:

Example
-------

A self-contained one-draw Monte Carlo. The treated unit shares the donors'
pre-period DGP; in the post-period it receives a location shift of
:math:`+1.5` (a constant effect across quantiles). We expect a roughly flat
QTE near :math:`1.5`, and -- because the shift is real -- the placebo
permutation test should rank the treated unit most extreme.

.. code-block:: python

   """One draw of a DSC location-shift simulation with placebo inference."""

   import numpy as np
   import pandas as pd

   from mlsynth import DSC

   rng = np.random.default_rng(0)
   J, T_pre, T_post, n_per_cell = 12, 8, 4, 200
   T = T_pre + T_post
   delta_post = 1.5

   unit_loc = rng.standard_normal(J + 1) * 0.5
   time_shift = np.linspace(0.0, 1.0, T)
   rows = []
   for j in range(J + 1):
       for t in range(T):
           loc = unit_loc[j] + time_shift[t]
           if j == 0 and t >= T_pre:
               loc += delta_post
           for y in rng.normal(loc=loc, scale=1.0, size=n_per_cell):
               rows.append({"unit": j, "time": t, "y": float(y),
                            "D": int(j == 0 and t >= T_pre)})
   df = pd.DataFrame(rows)

   res = DSC({
       "df": df, "outcome": "y", "treat": "D", "unitid": "unit", "time": "time",
       "M": 400, "grid_method": "halton", "lambda_method": "uniform",
       "n_qte_points": 49,
       "compute_inference": True,        # placebo permutation test
       "inference_grid_points": 100,
   }).fit()

   print(f"true location shift = {delta_post:+.3f}")
   print(f"DSC mean-of-QTE     = {res.att:+.3f}")
   print(f"placebo p-values    = "
         f"{{t: round(p, 3) for t, p in res.inference.p_values.items()}}")
   print(f"pre-period W2 fit   = {res.pre_period_wasserstein.round(4).tolist()}")

Reproducing the paper's simulation (Gunsilius 2023, Figure 4)
-------------------------------------------------------------

The headline simulation in Gunsilius (2023, Section 6.1) shows the DSC
barycenter replicating a Gaussian-mixture target *better as the donor pool
grows*: rough with 4 controls, good with 30, near-perfect with 500. The
controls are mixtures of 3 Gaussians and the target a mixture of 4, with
means drawn uniformly on :math:`[-10, 10]` and variances on
:math:`[0.5, 6]`. The snippet below reproduces that finding using the same
quantile machinery the estimator calls internally, reporting the squared
2-Wasserstein distance between barycenter and target as :math:`J` grows.

.. code-block:: python

   import numpy as np
   from mlsynth.utils.dsc_helpers import (
       empirical_quantile, sample_quantile_grid, solve_simplex_weights,
   )

   def mixture(rng, n_comp, n):
       means = rng.uniform(-10.0, 10.0, n_comp)
       var = rng.uniform(0.5, 6.0, n_comp)
       comp = rng.integers(0, n_comp, n)
       return rng.normal(means[comp], np.sqrt(var[comp]))

   def w2_to_target(J, rng, n=1000, M=1000, n_eval=1000):
       target = mixture(rng, 4, n)
       controls = [mixture(rng, 3, n) for _ in range(J)]
       V = sample_quantile_grid(M=M, method="uniform",
                                random_state=int(rng.integers(1_000_000_000)))
       donor_mat = np.column_stack([empirical_quantile(c, V) for c in controls])
       w = solve_simplex_weights(donor_mat, empirical_quantile(target, V))
       q = np.linspace(0.005, 0.995, n_eval)
       bc = np.column_stack([empirical_quantile(c, q) for c in controls]) @ w
       w2 = float(np.mean((bc - empirical_quantile(target, q)) ** 2))
       return w2, float(np.mean(w > 1e-4))   # distance, fraction of active weights

   rng = np.random.default_rng(12345)
   for J in (4, 30, 100, 500):
       reps = 20 if J <= 100 else 6
       out = np.array([w2_to_target(J, rng) for _ in range(reps)])
       print(f"J={J:>4}:  mean W2^2 = {out[:, 0].mean():7.3f}   "
             f"frac weights > 1e-4 = {out[:, 1].mean():.3f}")

   # J=   4:  mean W2^2 ~  4.2     frac weights > 1e-4 ~ 0.58
   # J=  30:  mean W2^2 ~  0.6     frac weights > 1e-4 ~ 0.17
   # J= 100:  mean W2^2 ~  0.8     frac weights > 1e-4 ~ 0.05
   # J= 500:  mean W2^2 ~  0.4     frac weights > 1e-4 ~ 0.02

The distance collapses once the donor pool is rich enough for the target to
sit inside (or near) the convex hull of the controls, and the weight vector
is "essentially sparse" -- only a small fraction of donors carry
non-negligible weight -- exactly the two phenomena Gunsilius reports. (The
residual distance at large :math:`J` is empirical-quantile noise from the
finite within-cell sample of 1000 draws, not bias.)

References
----------

Abadie, A., Diamond, A., & Hainmueller, J. (2010). "Synthetic Control
Methods for Comparative Case Studies." *Journal of the American Statistical
Association* 105(490):493-505.

Agueh, M., & Carlier, G. (2011). "Barycenters in the Wasserstein Space."
*SIAM Journal on Mathematical Analysis* 43(2):904-924.

Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S.
(2021). "Synthetic Difference-in-Differences." *American Economic Review*
111(12):4088-4118.

Athey, S., & Imbens, G. W. (2006). "Identification and Inference in
Nonlinear Difference-in-Differences Models." *Econometrica* 74(2):431-497.

Beck, A., & Teboulle, M. (2009). "A Fast Iterative Shrinkage-Thresholding
Algorithm for Linear Inverse Problems." *SIAM Journal on Imaging Sciences*
2(1):183-202.

Dube, A. (2019). "Minimum Wages and the Distribution of Family Incomes."
*American Economic Journal: Applied Economics* 11(4):268-304.

Duchi, J., Shalev-Shwartz, S., Singer, Y., & Chandra, T. (2008). "Efficient
Projections onto the l1-Ball for Learning in High Dimensions." *ICML*.

Gunsilius, F. F. (2023). "Distributional Synthetic Controls."
*Econometrica* 91(3):1105-1117. Reference implementation: the ``DiSCo`` R
package.

Kloeckner, B. (2010). "A Geometric Study of Wasserstein Spaces: Euclidean
Spaces." *Annali della Scuola Normale Superiore di Pisa* 9(2):297-323.

Zhang, L., Zhang, X., & Zhang, X. (2026). "Asymptotic Properties of the
Distributional Synthetic Controls." arXiv:2405.00953.
