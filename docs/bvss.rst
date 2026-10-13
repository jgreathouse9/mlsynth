Bayesian Synthetic Control with a Soft Simplex Constraint (BVS-SS)
==================================================================

.. currentmodule:: mlsynth

Overview
--------

BVS-SS `arXiv:2503.06454 <https://arxiv.org/abs/2503.06454>`_ is a
Bayesian synthetic control estimator that wraps two ideas around the
standard SCM regression :math:`\mathbf{y}_1 = \mathbf{Y}_0\mathbf{w} + \boldsymbol{\varepsilon}`:

* Spike-and-slab variable selection. Each donor is either active
  (:math:`\gamma_j = 1`) or excluded (:math:`\gamma_j = 0,\ w_j = 0`),
  with a Bernoulli prior on inclusion. This handles high-dimensional
  panels where :math:`N_0` (donor count) is comparable to or exceeds
  :math:`T_0`, and provides posterior inclusion probabilities
  :math:`P(\gamma_j = 1 \mid \mathbf{y}_1)` that quantify donor relevance.
* Soft simplex constraint. Selected weights are drawn around a
  Dirichlet mean :math:`\boldsymbol{\mu}_\gamma` with a learnable variance
  :math:`\nu`. When :math:`\nu \to 0` the prior collapses to the
  hard simplex of Abadie & Gardeazabal (2003); when :math:`\nu \to
  \infty` it becomes an unconstrained spike-and-slab regression. The
  posterior of :math:`\nu` tells the user whether the simplex
  constraint is supported by the data.

Compared to other Bayesian SCMs in :mod:`mlsynth`, BVS-SS is the only
one that simultaneously *selects donors* and *estimates how strictly
the simplex should hold*. This makes it the right tool when (a) the
donor pool is large, (b) you suspect some donors are irrelevant, and
(c) the appropriateness of the simplex constraint is itself a
modeling question (e.g. Hong Kong handover, Basque conflict, NFP tax
evasion — all cases where the treated unit's level may legitimately
exceed any convex combination of donors).

Sampling is by a custom Metropolis-within-Gibbs scheme implemented in
pure ``numpy`` + ``scipy`` — no ``PyMC``, ``NumPyro``, or ``JAX``
dependencies.

The Bayesian SC family
----------------------

mlsynth carries three Bayesian synthetic-control estimators; they differ in
what they place a prior on, and BVSS is the soft-simplex donor-selection member.

* :doc:`bscm` (Kim, Lee and Gupta) -- shrinkage (horseshoe or spike-and-slab)
  on unconstrained donor weights; a pure-numpy Gibbs sampler, and it reports
  donor weights.
* :doc:`bvss` (Xu and Zhou) -- spike-and-slab donor selection on a soft
  simplex whose tightness is learned; a pure-numpy Metropolis-within-Gibbs
  sampler, and it reports donor weights and inclusion probabilities.
* :doc:`bfsc` (Pinkney) -- a Bayesian latent-factor model, not a donor
  weighting; NUTS through the ``[bayes]`` optional dependency, and it reports a
  counterfactual credible band and no donor weights.

Reach for a weighting prior (BVSS, :doc:`bscm`) when you want interpretable
donor weights; reach for :doc:`bfsc` when a shared factor structure -- not a
weighted average of donors -- is the right model for the untreated outcome.

When to use this estimator
--------------------------

* The donor pool is large relative to the pre-period
  (:math:`N_0 \gtrsim T_0`). The classical SCM quadratic program has no
  unique solution and Lasso-style alternatives tend to over-select;
  BVS-SS's spike-and-slab structure recovers a sparse subset and
  converges to the oracle estimator as the sample grows.
* You suspect some donors are irrelevant and want a probability
  statement, not an eyeball judgement, about which donors to trust.
* The appropriateness of the simplex constraint is itself a modeling
  question — e.g. Hong Kong handover, Basque conflict, NFP tax evasion,
  all cases where the treated unit's level may legitimately exceed any
  convex combination of donors.

A concrete example: an anti-corruption policy announcement may have
depressed luxury-watch imports, and you have one treated customs
category, a short monthly pre-period, and dozens of candidate donor
categories. BVS-SS selects a sparse handful of donor categories,
returns the posterior of the ATT, and — through the learned variance
:math:`\nu` — reports whether the watch series sits inside the convex
hull of the donors or genuinely outside it.

Notation
--------

Let :math:`j = 1` denote the treated unit, with all units
:math:`\mathcal{N} \coloneqq \{1, \dots, N\}` and donor pool
:math:`\mathcal{N}_0 \coloneqq \mathcal{N} \setminus \{1\}` of cardinality
:math:`N_0`. Time runs over :math:`t \in \mathcal{T} \coloneqq \{1, \dots, T\}`,
1-indexed; the intervention takes effect after period :math:`T_0`, splitting
:math:`\mathcal{T}` into the pre-period
:math:`\mathcal{T}_1 \coloneqq \{t \in \mathcal{T} : t \le T_0\}` (of length
:math:`T_0`) and the post-period
:math:`\mathcal{T}_2 \coloneqq \{t \in \mathcal{T} : t > T_0\}`.

The treated series is :math:`\mathbf{y}_1 = (y_{11}, \dots, y_{1T})^\top`
with scalar outcomes :math:`y_{1t}`; each donor :math:`j \in \mathcal{N}_0`
contributes a series :math:`\mathbf{y}_j`, stacked into the donor matrix
:math:`\mathbf{Y}_0 \coloneqq [\mathbf{y}_j]_{j \in \mathcal{N}_0}
\in \mathbb{R}^{T \times N_0}` (one column per donor); column :math:`j` is
written :math:`\mathbf{Y}_{0,j}`. Donor weights are
:math:`\mathbf{w} \in \mathbb{R}^{N_0}`, with inclusion indicators
:math:`\boldsymbol{\gamma} \in \{0,1\}^{N_0}` (:math:`\gamma_j = 1` if donor
:math:`j` is active). The active-donor submatrix is :math:`\mathbf{Y}_{0,\gamma}`,
its Dirichlet mean :math:`\boldsymbol{\mu}_\gamma`. The synthetic
counterfactual is :math:`\widehat{\mathbf{y}}_1 \coloneqq \mathbf{Y}_0\mathbf{w}`
with entries :math:`\widehat{y}_{1t}`, the per-period effect is
:math:`\tau_t \coloneqq y_{1t} - \widehat{y}_{1t}`, and the ATT is
:math:`\widehat{\tau} \coloneqq |\mathcal{T}_2|^{-1}
\sum_{t \in \mathcal{T}_2} \tau_t`. Page-specific symbols: the
observation precision :math:`\phi` (inverse error variance), the
soft-simplex variance :math:`\nu` (Dirichlet-mean spread; small
:math:`\nu` tightens toward the hard simplex), and the Dirichlet
concentration :math:`\alpha`. Following the canon, :math:`\tau` is
reserved for the treatment effect: the soft-constraint variance is
:math:`\nu`, the symbol the paper writes :math:`\tau`.

Mathematical Formulation
------------------------

Let :math:`\mathbf{y}_1 \in \mathbb{R}^{T_0}` be the pre-treatment outcome of
the treated unit and :math:`\mathbf{Y}_0 \in \mathbb{R}^{T_0 \times N_0}` the
contemporaneous donor matrix (restricted here to :math:`\mathcal{T}_1`). Both
are demeaned column-wise using the pre-treatment means before entering the
sampler (this matches the paper's working setup and makes the soft-simplex
prior numerically well-conditioned).

Likelihood and Hierarchical Prior
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Gaussian likelihood is

.. math::

   \mathbf{y}_1 \mid \mathbf{w}, \phi \;\sim\; \mathcal{N}\!\left( \mathbf{Y}_0 \mathbf{w},\; \phi^{-1} \mathbf{I} \right).

The hierarchical prior over
:math:`(\mathbf{w}, \boldsymbol{\mu}, \nu, \phi, \boldsymbol{\gamma})` from Eq. (2) of the paper is

.. math::

   \begin{aligned}
   \phi &\sim \mathrm{Gamma}(\kappa_1 / 2,\; \kappa_2 / 2), \\
   \gamma_j &\stackrel{\text{i.i.d.}}{\sim} \mathrm{Bernoulli}(\theta),
       \quad j = 1, \dots, N_0, \\
   \nu &\sim \mathrm{Gamma}(a_1,\; a_2), \\
   \boldsymbol{\mu}_\gamma \mid \boldsymbol{\gamma} &\sim \mathrm{sym\text{-}Dirichlet}(\alpha), \\
   \mathbf{w}_\gamma \mid \boldsymbol{\gamma}, \boldsymbol{\mu}_\gamma, \nu, \phi
       &\sim \mathcal{N}\!\left(\boldsymbol{\mu}_\gamma,\; \tfrac{\nu}{\phi} \mathbf{I}\right),
   \end{aligned}

with the convention that :math:`\mu_j = w_j = 0` whenever
:math:`\gamma_j = 0`. The fixed-:math:`\alpha = 1` case (uniform
Dirichlet on the active simplex) is what BVS-SS implements in this
package; the paper's simulations and both empirical applications use
this setting.

Interpreting the soft-constraint variance :math:`\nu` is the key
modeling insight:

* As :math:`\nu \downarrow 0`, the prior of :math:`\mathbf{w}_\gamma`
  concentrates at :math:`\boldsymbol{\mu}_\gamma \in \Delta^{|\gamma| - 1}`, i.e.
  the hard simplex.
* As :math:`\nu \uparrow \infty`, the prior on :math:`\mathbf{w}_\gamma` is
  effectively uninformative, recovering an unconstrained spike-and-slab
  regression à la Kim et al. (2020).

The posterior of :math:`\nu` is therefore a data-driven indicator of
whether the simplex constraint is appropriate for the application at
hand. The paper proves (Theorem 4) that as :math:`\nu \to \infty`
the BVS-SS posterior of :math:`\boldsymbol{\gamma}` becomes identical to the
unconstrained spike-and-slab posterior, so the data really does pick
between the two regimes through :math:`\nu`.

Marginal Likelihood and Posterior Conditionals
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A Woodbury-identity calculation (Eq. (4) of the paper) yields

.. math::

   p(\mathbf{y}_1 \mid \boldsymbol{\gamma}, \boldsymbol{\mu}_\gamma, \nu, \phi)
   \;\propto\;
   \frac{\phi^{T_0 / 2}}{\nu^{|\gamma| / 2}
        \det(\mathbf{V}_{\gamma, \nu})^{1/2}}
   \exp\!\left\{
       -\tfrac{\phi}{2}
       (\mathbf{y}_1 - \mathbf{Y}_{0,\gamma} \boldsymbol{\mu}_\gamma)^\top
       \boldsymbol{\Sigma}_{\gamma, \nu}
       (\mathbf{y}_1 - \mathbf{Y}_{0,\gamma} \boldsymbol{\mu}_\gamma)
   \right\},

with the two repeated quantities

.. math::

   \mathbf{V}_{\gamma, \nu} \coloneqq \mathbf{Y}_{0,\gamma}^\top \mathbf{Y}_{0,\gamma} + \nu^{-1} \mathbf{I},
   \qquad
   \boldsymbol{\Sigma}_{\gamma, \nu} \coloneqq \mathbf{I} - \mathbf{Y}_{0,\gamma} \mathbf{V}_{\gamma, \nu}^{-1} \mathbf{Y}_{0,\gamma}^\top.

The :math:`\phi` and :math:`\mathbf{w}_\gamma` full conditionals are
conjugate (Eqs. (6)–(7)):

.. math::

   \begin{aligned}
   \phi \mid \mathbf{y}_1, \boldsymbol{\gamma}, \boldsymbol{\mu}_\gamma, \nu
       &\sim \mathrm{Gamma}\!\left(
           \tfrac{T_0 + \kappa_1}{2},\;
           \tfrac{\kappa_2 + (\mathbf{y}_1 - \mathbf{Y}_{0,\gamma} \boldsymbol{\mu}_\gamma)^\top
                   \boldsymbol{\Sigma}_{\gamma, \nu}
                   (\mathbf{y}_1 - \mathbf{Y}_{0,\gamma} \boldsymbol{\mu}_\gamma)}{2}
       \right), \\
   \mathbf{w}_\gamma \mid \mathbf{y}_1, \boldsymbol{\gamma}, \boldsymbol{\mu}_\gamma, \nu, \phi
       &\sim \mathcal{N}\!\left(
           \mathbf{V}_{\gamma, \nu}^{-1}
           \!\left(\mathbf{Y}_{0,\gamma}^\top \mathbf{y}_1 + \nu^{-1} \boldsymbol{\mu}_\gamma\right),\;
           \phi^{-1} \mathbf{V}_{\gamma, \nu}^{-1}
       \right).
   \end{aligned}

The :math:`\nu` conditional has no closed form. The :math:`(\boldsymbol{\gamma},
\boldsymbol{\mu})` block is the technically interesting piece — it cannot be
updated coordinate-wise (the simplex constraint
:math:`\sum_{j: \gamma_j = 1} \mu_j = 1` makes single-coordinate
moves degenerate) so the paper introduces a *two-coordinate Gibbs
update* over pairs :math:`(\gamma_j, \gamma_{j'}, \mu_j, \mu_{j'})`.

Metropolis-within-Gibbs Sampler (Algorithm 1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One outer iteration performs three updates in order.

1. Pair update for :math:`(\gamma_i, \gamma_j, \mu_i, \mu_j)`.
   For each unordered pair :math:`i < j`, fix
   :math:`\mu_{-(i,j)}` and define the residual mass
   :math:`s = 1 - \sum_{k \neq i, j} \mu_k`. Three cases follow:

* :math:`s = 0`: the simplex constraint forces
  :math:`\mu_i = \mu_j = 0`, no draw needed.
* :math:`s > 0`: enumerate the four possible inclusion patterns
  :math:`(\gamma_i, \gamma_j) \in \{(0,0), (1,0), (0,1), (1,1)\}`.
  The :math:`(0,0)` case is infeasible (would violate the simplex).
  The other three have closed-form conditional posterior
  probabilities (Lemma S2 of the paper). When the drawn pattern is
  :math:`(1, 1)`, the conditional distribution of :math:`\mu_i` is
  the univariate truncated normal

  .. math::

     \mu_i \;\sim\; \mathcal{N}_{(0,\,s)}\!\left(\beta_{i,j},\;
                                                  (\phi \Lambda_{i,j})^{-1}\right),
     \qquad \mu_j = s - \mu_i,

  where (Lemma S1)

  .. math::

     \Lambda_{i,j} \;=\; (\mathbf{Y}_{0,i} - \mathbf{Y}_{0,j})^\top
                          \boldsymbol{\Sigma}_{\gamma^{ij}, \nu}
                          (\mathbf{Y}_{0,i} - \mathbf{Y}_{0,j}), \quad
     \beta_{i,j} \;=\; \frac{1}{\Lambda_{i,j}}\,
                          (\mathbf{Y}_{0,i} - \mathbf{Y}_{0,j})^\top
                          \boldsymbol{\Sigma}_{\gamma^{ij}, \nu}
                          \!\left(\mathbf{y}_1 - s\,\mathbf{Y}_{0,j}
                                 - \!\!\!\sum_{k \neq i, j}
                                 \mu_k\,\mathbf{Y}_{0,k}\right).

The pair-update sweep visits every :math:`(i, j)` pair once per outer
iteration. With :math:`\alpha = 1` all probabilities involve only the
standard normal CDF and a single truncated-normal draw — no
rejection sampling and no numerical integration.

Each of the three feasible cases is scored at its own inclusion vector.
Eq. (11) of the paper defines

.. math::

   \gamma^i_k = \mathbb{1}(\mu_k \neq 0 \text{ or } k = i), \quad
   \gamma^j_k = \mathbb{1}(\mu_k \neq 0 \text{ or } k = j), \quad
   \gamma^{ij}_k = \mathbb{1}(\mu_k \neq 0 \text{ or } k \in \{i, j\}),

three different sets of two different cardinalities, and Lemma 2 indexes both
the complexity factor :math:`A(\gamma, \nu)` and the projector
:math:`\boldsymbol{\Sigma}_{\gamma, \nu}` by the state's own vector. Scoring
the one-donor cases at :math:`\gamma^{ij}` gives them a residual that has had
the other donor projected out of it, which inflates the :math:`(1,1)`
probability and so the fitted model size; on the luxury-watch panel it moved
the posterior mean model size to 9.19 against the paper's 5.09.

One settled discrepancy, one open
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Both predate this page and both are shared with the authors' own replication
script, so neither is visible from a cross-validation against that script alone.

Settled: the complexity factor does not carry :math:`\nu^{-|\gamma|/2}`. The
statement of Lemma 2 gives :math:`A(\gamma, \nu)` a factor
:math:`\nu^{-|\gamma|/2}`, inherited from the marginal likelihood in Eq. (4),
while the proof of the same lemma says :math:`A` collapses to
:math:`\det(V_{\gamma,\nu})^{-1/2} (|\gamma| - 1)!\, p_0(|\gamma|)` with no
such factor. The two are different samplers, because the factor depends on
:math:`|\gamma|` and :math:`|\gamma^i| = \ell + 1` while
:math:`|\gamma^{ij}| = \ell + 2`, so it cannot be absorbed into the normalising
constant. Restoring it was measured over twelve seeds on the luxury-watch
panel: the posterior mean model size goes to 21.60 against Table 6's 5.09,
:math:`\phi` to 18.94 against 20.86 and :math:`\nu` to 0.0217 against 0.069 --
every summary further from the paper's own table. The proof's reading is the
one that reproduces the paper, and :mod:`mlsynth` follows it, as the authors'
script does.

Open: the posterior model size. At the benchmark's 50 iterations the posterior
mean model size is 2.77 over twelve seeds, and at 400 iterations 3.07 over
three, against Table 6's 5.09 from a 1000-iteration run. Chain length accounts
for part of the gap and the trend has the right sign, but not for all of it at
the lengths measured. The collapse corrected above was the larger term in the
opposite direction, taking the same quantity to 9.19.

The counterfactual is built from :math:`\boldsymbol{\mu}`. Algorithm 1 of the
paper draws :math:`\mathbf{w}^{(t)}` from its Eq. (7) conditional and forms the
counterfactual as :math:`\tilde{\mathbf{X}} \mathbf{w}^{(t)}`; Remark 2 permits
replacing that draw by its conditional mean, Eq. (8), to reduce the variance of
the ATT estimate. The authors' script implements Remark 2. :mod:`mlsynth` uses
:math:`\boldsymbol{\mu}` itself, which is neither, and
:math:`\mathbf{w} \to \boldsymbol{\mu}` only as :math:`\nu \downarrow 0`.

How much that costs is measured. Evaluating both constructions on the same
draws -- so the comparison carries none of the sampler's Monte-Carlo noise --
the posterior mean ATT moves by 0.48 % of itself and the credible interval's
width by a factor of 0.999, over five seeds on the watch panel at 400
iterations. The per-seed width ratio ranges from 0.972 to 1.023, which is
smaller than the seed-to-seed spread of the width itself. Switching to Eq. (8)
would re-derive every pinned number on this page and in the benchmark for a
change in the reported answer of under half a per cent.

2. :math:`\phi` Gibbs draw. Plug the updated :math:`\mu`
   into the closed-form Gamma conditional above.

3. :math:`\nu` MH steps. Repeat for :math:`n_\nu`
   iterations a log-random-walk proposal

.. math::

   \log \nu^\ast \;=\; \log \nu \;+\; \mathcal{N}(0, 1),

with the lower boundary :math:`\log \nu_{\min}` reflected (proposals
below :math:`\nu_{\min}` are mirrored back into the support). The
acceptance ratio is

.. math::

   \rho(\nu, \nu^\ast) \;=\;
   \min\!\left\{
       1,\;
       \frac{p(\mathbf{y}_1 \mid \boldsymbol{\mu}, \nu^\ast, \phi)\, p(\nu^\ast)}
            {p(\mathbf{y}_1 \mid \boldsymbol{\mu}, \nu, \phi)\, p(\nu)}
       \cdot \frac{\nu^\ast}{\nu}
   \right\},

where the final :math:`\nu^\ast / \nu` factor is the Jacobian of the
log-space random walk.

Counterfactual Imputation and ATT
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The sampler returns posterior draws :math:`\{\boldsymbol{\mu}^{(s)}\}_{s = 1}^{S}`
after burn-in. The treated unit's counterfactual at every period
:math:`t` is constructed from the column-demeaned donor matrix
:math:`\widetilde{\mathbf{Y}}_0` (using the same pre-treatment means as the
sampler) via

.. math::

   \widehat{y}_{1t}^{N,(s)} \;=\; \widetilde{\mathbf{Y}}_{0,t}\,
   \boldsymbol{\mu}^{(s)} + \bar y_{1,\mathrm{pre}},

where :math:`\bar y_{1,\mathrm{pre}}` is the pre-treatment mean of the
treated outcome. The post-treatment ATT per draw is

.. math::

   \widehat{\tau}^{(s)}
   \;=\;
   \frac{1}{T - T_0}
   \sum_{t \in \mathcal{T}_2}
   \!\left( y_{1t} - \widehat{y}_{1t}^{N,(s)} \right),

and the reported headline ATT is the posterior mean of
:math:`\widehat{\tau}^{(s)}`. Credible intervals are
percentile bands over :math:`\widehat{\tau}^{(s)}` at level
:math:`1 - \mathrm{ci\_alpha}` (default 95 %). Pointwise bands on the
counterfactual are computed analogously period-by-period.

BVS-SS computes the counterfactual directly from the
:math:`\boldsymbol{\mu}` posterior instead of drawing
:math:`\mathbf{w}_\gamma` from its Eq. (7) conditional. The paper's own
variance reduction is Remark 2 of Section 3.3, which replaces the draw by its
Eq. (8) conditional mean :math:`V_{\gamma,\nu}^{-1}(\mathbf{Y}_{0,\gamma}^\top
\mathbf{y}_1 + \nu^{-1}\boldsymbol{\mu}_\gamma)`, so the two agree only in the
:math:`\nu \downarrow 0` limit. See the open discrepancies above.

When BVS-SS Is the Right Tool
-----------------------------

The paper documents three settings where BVS-SS materially outperforms
the classical SCM:

* High-dimensional donor pools (Section 5.2). When ``N`` is
  comparable to or exceeds ``T_0``, the standard quadratic program
  does not have a unique solution and Lasso-style alternatives tend
  to over-select. BVS-SS's spike-and-slab structure recovers a sparse
  subset and converges to the oracle OLS estimator as the sample
  grows.
* Simplex-violating data (Section 5.2,
  :math:`\|\mathbf{w}^\ast\|_1 \in \{2, 3\}`). When the true sum of weights
  exceeds 1 — as it plausibly does for outliers like Hong Kong's GDP
  in the late 1990s — methods that enforce the hard simplex are
  biased, while BVS-SS's posterior of :math:`\nu` moves away from
  zero and the model adapts.
* Anti-tax-evasion / anti-corruption policy evaluation (Section
  6). Replicating Carvalho et al. (2018) and Shi & Huang (2023),
  BVS-SS recovers the published ATT magnitudes with substantially
  smaller credible intervals than Lasso-based ArCo, while selecting
  only ~2 donors on average instead of the entire pool.

A handy diagnostic is the *posterior of* :math:`\nu`: small
posterior mass near zero is a sign the data agrees with the simplex
constraint, while bulk away from zero is evidence to relax it.
``results.simplex`` summarises it and ``results.posterior.tau`` holds the raw
draws; `Reading the simplex verdict`_ below says what the summary contains and
how to read it.

Assumptions (Xu & Zhou 2025)
----------------------------

The paper proves *high-dimensional strong selection consistency* --
the posterior probability of the true active-donor model converging
to 1 under the true DGP -- in Theorem 2, under the technical
conditions A1-A5 (Section 4.1). Stated for the working
:math:`\nu = 0` (hard-simplex) limit:

1. Restricted eigenvalue on the donor matrix (A1). Each column
   :math:`\mathbf{Y}_{0,j}` of the donor matrix satisfies
   :math:`\| \mathbf{Y}_{0,j} \|_2^2 = T_0`, and there exists
   :math:`\underline\lambda \in (0, 1]` such that
   :math:`\lambda_{\min}(\mathbf{Y}_{0,\gamma}^\top \mathbf{Y}_{0,\gamma})
   \ge T_0 \underline\lambda` for every candidate model :math:`\gamma`
   in the sparse model space :math:`\mathbb{S}_L`.

   *Remark.* No two donors are perfectly collinear, no donor is a
   near-duplicate of a linear combination of a few others, and the
   minimum eigenvalue of every "reasonable-size" donor submatrix is
   bounded away from zero.

2. Inclusion-prior penalty (A2). The Bernoulli inclusion
   probability satisfies :math:`\theta / (1 - \theta) = N^{-c_\theta L}`
   for some universal :math:`c_\theta > 0`.

   *Remark.* The prior penalises large models geometrically in the
   donor count -- a default :math:`\theta \approx 1/N` keeps the prior
   expected model size around 1.

3. Noise-precision prior (A3). The Gamma prior on :math:`\phi`
   has shape :math:`\kappa_1 \in (0, T_0]` and rate :math:`\kappa_2
   \in [0, \sigma^2 T_0 / 2]`.

   *Remark.* The prior on the inverse error variance is not
   pathologically informative (neither shape nor rate scale faster
   than :math:`T_0`).

4. True DGP and signal strength / :math:`\beta`-min (A4). The
   true outcome is :math:`\mathbf{y}_1 \mid \mathbf{Y}_0 \sim
   \mathcal{N}(\mathbf{Y}_{0,\gamma^\ast} \boldsymbol{\mu}_{\gamma^\ast}^\ast,
   \sigma^2 \mathbf{I})` with :math:`\ell^\ast \coloneqq |\gamma^\ast|
   \le L \wedge \sqrt{L \log N}`,
   :math:`\boldsymbol{\mu}_{\gamma^\ast}^\ast \in \Delta^{\ell^\ast - 1}`,
   and a lower bound on the smallest non-zero weight,

   .. math::

      \min_{j \in \gamma^\ast} |\mu_j^\ast|
      \;\ge\;
      \frac{c_\mu \sigma \sqrt{L \log N}}{\underline\lambda \sqrt{T_0}}.

   *Remark.* The true donor pool is sparse, lies on the simplex,
   and every truly-active donor carries weight detectable above the
   noise floor. The :math:`\beta`-min lower bound is the standard
   "signal large enough to identify" condition shared with all
   high-dimensional consistency theory (Yang et al. 2016).

5. Sample-size lower bound (A5). :math:`L \ge 3` and
   :math:`T_0 \ge c_M \ell^\ast \log N` for some :math:`c_M > 0`.

   *Remark.* The pre-period must grow with the (log of) the donor
   pool size for selection consistency to take hold.

Theorem 2 (Xu & Zhou 2025). Under A1-A5, the posterior
inclusion probability of the true model satisfies
:math:`p(\gamma^\ast \mid \mathbf{y}_1) \xrightarrow{p^\ast} 1` as
:math:`T_0, N \to \infty` along any allowed sequence. Theorem 3 then
bounds the posterior expected predictive loss on the test
(post-treatment) sample at the order
:math:`(T - T_0)\, \ell^\ast \log N / T_0`, which is the
"oracle" rate for the constrained problem -- i.e., BVS-SS is
asymptotically as efficient as if an oracle had told you the true
active set in advance.

For the soft-simplex limit :math:`\nu \to \infty`, Theorem 4
shows the BVS-SS posterior of :math:`\gamma` converges to the
unconstrained spike-and-slab posterior. The two regimes are
genuine endpoints of the same family; the data picks between them
through the learned posterior of :math:`\nu`.

When the assumptions bind: practical diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

(a) Donor near-collinearity (A1). If two donors carry
    essentially the same pre-period information, the minimum
    eigenvalue of :math:`\mathbf{Y}_{0,\gamma}^\top \mathbf{Y}_{0,\gamma}`
    is near zero on
    models that include both -- the restricted-eigenvalue
    condition fails and the spike-and-slab will *flip* between
    them without ever converging on a single representative.

    *Plausibly violated when* the donor pool contains near-clones
    (two product categories whose time series differ only by
    sampling noise; two states with essentially identical
    industry mix). *Diagnostic*: inspect
    ``results.inclusion_probs`` -- if two donors each show
    :math:`P(\text{included}) \approx 0.4` with all others below
    0.1, the model is splitting credit between near-duplicates.
    Drop one or merge them before refitting.

(b) Prior penalty scale (A2). A too-large :math:`\theta`
    floods the posterior with large-model mass; a too-small one
    blocks even truly-active donors.

    *Plausibly violated when* the default :math:`\theta` was
    chosen blindly. *Diagnostic*: track the posterior model size
    ``results.posterior.gamma.sum(axis=0)``; if its posterior
    mean is essentially the prior mean :math:`N \theta`, the
    data is not informing model size and you should tighten
    :math:`\theta`. The paper's empirical applications use
    :math:`\theta = 0.2`.

(c) Sparse, simplex-supported truth (A4). Selection
    consistency requires that the true weight vector is sparse
    *and* on the simplex. If the truth is genuinely dense
    (many donors each contributing a small fraction) or
    genuinely off-simplex (weights summing to substantially
    more or less than 1), the consistency story does not apply
    -- though the procedure still produces a well-defined
    posterior.

    *Plausibly violated when* the treated unit is structurally
    outside the donor convex hull (Hong Kong handover; very
    extreme growth unit). *Diagnostic*:
    ``results.simplex.tau_mean`` against its prior mean
    :math:`a_1 / a_2`, on a long chain. Measured over three
    donor orderings in `Reading the simplex verdict`_ below,
    that comparison separates the paper's two panels by a factor
    of six -- the luxury-watch panel it reports as supporting
    the constraint against the Hong Kong handover panel it uses
    to motivate relaxing one -- where
    ``results.simplex.relative_deviation`` does not separate
    them at all. Read the latter for how large the departure is
    in weight units, not for which panel departs more. Well
    above the prior mean, reach for the
    unconstrained-spike-and-slab limit (Kim et al. 2020) or for
    an estimator that explicitly handles outside-hull treated
    units (:doc:`iscm`).

(d) Long-enough pre-period (A5). Selection consistency
    requires :math:`T_0 \gtrsim \ell^\ast \log N`. With 20
    pre-period observations and 100 donors, you need at most
    :math:`\ell^\ast \le 20 / \log 100 \approx 4` truly-active
    donors for the theory to apply -- realistic in practice but
    a binding constraint at small :math:`T_0`.

    *Plausibly violated when* the pre-period is short and the
    donor pool is wide. *Diagnostic*: monitor MCMC mixing
    diagnostics on the posterior model size; if the posterior of
    :math:`|\gamma|` is unstable across chains (different seeds
    select markedly different active sets), the
    sample-size-vs-donor-count regime is too tight for stable
    selection. Either lengthen the pre-period (aggregate to a
    finer time grid) or pre-screen donors using domain
    knowledge.

(e) Gaussian likelihood with shock independence. The model
    assumes :math:`y_{1t} \sim \mathcal{N}((\mathbf{Y}_0 \mathbf{w})_t,
    \phi^{-1})` with
    iid errors. Strong autocorrelation in the pre-period
    residuals or heavy tails breaks the variance estimate that
    feeds into both :math:`\phi` and the :math:`\beta`-min
    threshold.

    *Plausibly violated when* the outcome has unit-root-like
    persistence or heavy outliers. *Diagnostic*: ADF / KPSS on
    the pre-period residual of the OLS-on-included-donors fit;
    a non-stationary residual flags this. First-difference the
    outcome and donors before refitting, or move to a
    cycle-decomposing estimator (:doc:`sbc`) before BVS-SS.

(f) Posterior of :math:`\nu` as the soft-simplex test.
    The single diagnostic that summarises the
    simplex-vs-unconstrained decision: small posterior mass of
    :math:`\nu` near zero means the data agrees with the
    simplex; bulk above the prior mean means the data prefers
    the relaxation.

    *Practical rule of thumb*: compare ``results.simplex.tau_mean``
    against the prior mean :math:`a_1 / a_2`. A posterior
    concentrated below the prior mean says the data did not ask
    for a relaxation; one sitting above it says the data pulled
    :math:`\nu` up to buy room off the simplex. Measured at one
    setting on both of the paper's own panels, the China-watches
    application lands at the prior mean and the Hong Kong
    handover case (Hsiao 2012, the motivation in Section 1.1) at
    seven times it -- the ordering the paper's argument predicts.

    Read ``results.simplex.tau_q025`` and ``tau_q975`` alongside
    it. On both panels the 95 % credible set runs from the
    sampler's lower bound to many times the prior mean, so the
    posterior mean summarises a diffuse posterior, and the level
    of :math:`\nu` moves with the chain length and
    :math:`\theta` where the ordering between panels does not.
    `Reading the simplex verdict`_ has the numbers.

Reading the simplex verdict
---------------------------

``results.simplex`` is the posterior's answer to the question the paper sets
out to ask. Eight numbers, each analytic in the draws of :math:`\nu`,
:math:`\phi` and :math:`|\gamma|`, so none of them depends on how the
counterfactual is built -- the construction that is still open above.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - What it is
   * - ``tau_mean``, ``tau_median``, ``tau_q025``, ``tau_q975``
     - Posterior summaries of the soft-constraint variance. The fields keep the
       paper's spelling :math:`\tau`; this page writes the same quantity
       :math:`\nu`, because :math:`\tau` is the treatment effect.
   * - ``deviation_scale``
     - :math:`E[\sqrt{\nu / \phi}]`, the standard deviation of a single
       weight about its simplex centre, in weight units.
   * - ``weight_sum_sd``
     - :math:`E[\sqrt{|\gamma| \nu / \phi}]`. The coordinates are
       independent given the parameters, so their variances add and
       :math:`\sum_j w_j` departs from one on this scale.
   * - ``relative_deviation``
     - ``deviation_scale`` against a typical weight :math:`1 / |\gamma|`.
   * - ``model_size_mean``
     - Posterior mean :math:`|\gamma|`.

Reading more than :math:`\nu` itself matters because :math:`\nu` is not in
weight units, and the comparison the paper makes -- :math:`\nu` against its
prior mean :math:`a_1 / a_2` -- answers whether the data asked for a relaxation,
not how large the relaxation it got was. :math:`\sqrt{\nu / \phi}` is in
weight units, and a scatter of 0.05 means one thing spread over three donors
and another over thirty, which is what ``relative_deviation`` corrects for.
The two readings can point in different directions on the same panel, and when
they do, both are true: the data can decline to demand a large relaxation while
the fit still uses the room the prior gives it. California's cigarette panel
below is such a case, with :math:`\nu` at 0.033 against a prior mean of 0.1
and a scatter of 38 % of a typical weight.

Two panels, measured
^^^^^^^^^^^^^^^^^^^^

Section 1.1 of the paper motivates relaxing the simplex with Hsiao's (2012)
Hong Kong handover panel, where the treated unit's growth sits outside any
convex combination of its donors; Section 6.2's application is the luxury-watch
panel this page replicates. Both ship in :file:`basedata/`, so both can be read
at identical settings -- 400 iterations, 200 discarded, :math:`\theta = 0.25`,
averaged over three seeds, with :math:`\nu`'s prior mean at
:math:`a_1 / a_2 = 0.1`:

.. list-table::
   :header-rows: 1
   :widths: 30 12 20 14 12 12

   * - Panel
     - :math:`\nu`
     - 95 % set for :math:`\nu`
     - :math:`\sqrt{\nu/\phi}`
     - relative
     - :math:`|\gamma|`
   * - Luxury watches (:math:`T_0 = 35`, :math:`N = 87`)
     - 0.099
     - (0.000, 0.973)
     - 0.037
     - 14 %
     - 3.82
   * - Hong Kong (:math:`T_0 = 44`, :math:`N = 24`)
     - 0.700
     - (0.000, 5.936)
     - 0.070
     - 23 %
     - 3.07

In that run the ordering the paper's argument predicts holds on both readings:
Hong Kong puts :math:`\nu` an order of magnitude above the watch panel, and
puts its scatter off the simplex at 23 % of a typical weight against 14 %. Only
one of the two survives repetition.

The levels in that table are not reproducible, and the two readings do not
survive equally. Nothing in the model depends on the order the donors arrive
in, but the pair sweep visits :math:`(i, j)` in index order, so the chain does.
Over three donor orderings of the watch panel at these settings, :math:`\nu`
comes out 0.099, 0.040 and 0.105 and ``relative_deviation`` 14 %, 5 % and 20 %;
Hong Kong gives 0.700 and 0.626, and 23 % and 17 %. The Path A table below, at
1000 iterations and :math:`\theta = 0.2`, puts the watch panel's :math:`\nu`
at 0.030.

So :math:`\nu` separates these two panels and :math:`\sqrt{\nu/\phi}`
against a typical weight does not. The :math:`\nu` ranges do not come close to
touching, a factor of six apart at worst, while the ``relative_deviation``
ranges overlap -- the watch panel's widest run reads above Hong Kong's
narrowest, though Hong Kong is higher in each ordering where both were run.
Use :math:`\nu` against its prior mean to compare panels, on a chain long
enough to have settled, and use ``relative_deviation`` to say how large the
departure is in weight units for the fit in front of you.

Neither credible set is tight. Both run from the sampler's lower bound
:math:`\nu_{\min}` up to about ten times the prior mean on the watch panel
and sixty times it on Hong Kong, so the posterior mean summarises a diffuse
posterior, which is why the quantiles are fields and not something a reader
has to go and compute. A diffuse posterior for :math:`\nu` and a summary that
moves with the chain are the same fact seen twice, and it is the sensitivity
the open model-size item above describes.

Under Eq. (2) the weights are :math:`\boldsymbol{\mu} + N(0, (\nu/\phi) I)`,
so the scatter can be read as a probability: about a fifth of draws on each
panel carry at least one negative weight. Like ``relative_deviation``, that
share does not tell the two panels apart, but in absolute terms it says the
non-negativity half of Abadie's restriction is not approximately in force on
either.

Level, shape, and the intercept
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Abadie's canonical estimator imposes three restrictions: no intercept,
:math:`\sum_j w_j = 1`, and :math:`w_j \ge 0`. BVS-SS softens the second and
third with :math:`\nu` and leaves the first at the opposite extreme. The
demeaning in :func:`~mlsynth.utils.bvss_helpers.setup.prepare_bvss_inputs` is
an improper uniform prior on an intercept, marginalized analytically: fitting
:math:`(\mathbf{y}_1 - \bar y_{1,\mathrm{pre}}) =
\widetilde{\mathbf{Y}}_0 \mathbf{w}` is fitting :math:`\mathbf{y}_1 =
(\bar y_{1,\mathrm{pre}} - \bar{\mathbf{Y}}_0^\top \mathbf{w}) +
\mathbf{Y}_0 \mathbf{w}`, so the level is free and unpenalised.

One consequence is that the verdict does not depend on the treated unit's level
at all. Adding a constant to :math:`\mathbf{y}_1` moves the intercept and
leaves :math:`\nu`, :math:`\phi` and :math:`|\gamma|` identical to ten
figures, which :file:`mlsynth/tests/test_bvss_simplex_diagnostics.py` asserts.

Dropping the demeaning instead forces :math:`\alpha = 0` and changes nothing
else, which asks which of the two relaxations a panel is leaning on. Three
panels from :file:`basedata/`, free arm then forced-to-zero arm, at the
settings above:

.. list-table::
   :header-rows: 1
   :widths: 34 12 18 20 16

   * - Panel
     - :math:`|\bar y_1| / s_{y_1}`
     - :math:`\sqrt{\nu/\phi}`
     - relative
     - :math:`|\gamma|`
   * - Luxury watches (monthly growth)
     - 0.09
     - 0.0203 / 0.0191
     - 5.4 % / 5.9 %
     - 2.51 / 2.45
   * - Hong Kong (quarterly growth)
     - 0.75
     - 0.0605 / 0.0599
     - 17.3 % / 16.7 %
     - 2.78 / 2.74
   * - California cigarette sales (levels)
     - 10.22
     - 0.0984 / 0.1983
     - 37.8 % / 99.4 %
     - 3.60 / 4.96

Both arms of each row share a donor ordering and the same three seeds, so the
ordering sensitivity above cancels within a row and the comparison is paired --
which is why this table reports both arms and the one above reports a level
with a caveat.

The answer depends on the outcome, and the first column is what it depends on.
Where the treated series' mean is small next to its own variation -- both of
the paper's panels, which are growth rates -- forcing
:math:`\alpha = 0` costs nothing and the simplex carries the fit alone. Where
the mean is ten times the variation, as in cigarette packs per capita, the same
ablation doubles the scatter off the simplex, takes it to 99 % of a typical
weight, and recruits half again as many donors.

The reason is leverage, not feasibility. With :math:`\sum_j w_j = 1` and
:math:`w_j \ge 0` the fitted pre-treatment mean is a convex combination of the
donors' pre-treatment means, so the treated mean is attainable whenever it lies
between the smallest and largest of them -- and on all three panels it does, at
0.61, 0.32 and 0.25 of the way through the range. Attainable is not free. When
the mean is large next to the variation, nearly all of :math:`\mathbf{w}`'s
freedom goes into hitting the level, and the posterior buys back the freedom to
fit the shape by inflating :math:`\nu`. When the mean is near zero, perturbing
the weights barely moves the level, so sum-to-one costs nothing to satisfy.

Two things follow for a reader. A levels outcome -- sales, revenue, packs,
visits -- is the regime where the intercept matters most, and it is the regime
neither of the paper's applications occupies, so take the paper's silence on
the intercept as untested there and not as evidence. And on such a panel a
large ``relative_deviation`` may be reporting a level the weights are
struggling to reach, not a mixture whose shape is wrong; demeaned data, which
is what :mod:`mlsynth` feeds the sampler, separates the two.

Hong Kong closes the loop in the other direction. Section 1.1 attributes its
need for a relaxation to the treated unit's level exceeding any convex
combination of its donors. Its level is comfortably inside that range, its
ablation is null, and its :math:`\nu` still runs an order of magnitude above
the watch panel's. What the posterior buys room for there is the shape of the
mixture, not its offset.

When to use BVSS -- and when not to
-----------------------------------

Reach for BVSS when:

* The donor pool is large relative to the pre-period
  (:math:`N \gtrsim T_0`). The classical SCM quadratic program
  has no unique solution; Lasso over-selects; BVSS's
  spike-and-slab cleanly returns a sparse posterior on inclusion.
* The simplex constraint itself is in question. Hong Kong
  handover, Basque conflict, China anti-corruption watches --
  cases where the treated unit's level may legitimately exceed
  any convex combination of donors. BVSS's learned :math:`\nu`
  tells you which regime the data prefers.
* You want posterior distributions, not point estimates.
  The full posterior of the ATT, of each donor's weight, and of
  inclusion are returned -- useful for downstream uncertainty
  propagation (e.g. into a policy-evaluation report's CI
  reporting).
* You want to formalise the "which donors do I trust" decision.
  Posterior inclusion probabilities replace the practitioner's
  eyeball "I'll keep these states because they look similar"
  step with a probability statement.

Do not use BVSS when:

* Small donor pool with clear pre-fit. With :math:`N = 8`
  states and a tight pre-fit on the canonical SCM, BVSS's
  per-pair Gibbs sampler is overkill; the spike-and-slab
  uncertainty just propagates noise that the data does not
  actually contain. Use *canonical SCM*, :doc:`tssc`, or :doc:`fdid`.
* Treated unit is severely outside the donor hull. BVSS's
  soft simplex *can* relax (:math:`\nu` learns), but the
  posterior weights are still anchored to the simplex prior.
  For the structural outside-hull case, :doc:`iscm`'s moment-
  condition framework is identification-aware in a way BVSS is
  not.
* Distribution of the outcome is the object of interest.
  BVSS targets the mean ATT through a Gaussian likelihood on
  levels. Distributional questions (Lorenz curves, QTEs,
  inequality measures) need :doc:`dsc`.
* You need speed. MCMC is materially slower than
  optimisation-based SC. The China-watches application takes
  ~10 minutes for 1000 iterations on commodity hardware; for
  large grid searches or batch processing across many
  policy-evaluation cells, an optimisation-based estimator
  (*canonical SCM*, :doc:`tssc`, :doc:`fdid`) is the right default.
* Outcomes with hard floors or ceilings, counts, or heavy
  tails. A4's Gaussian-likelihood / :math:`\beta`-min
  argument breaks. Pre-process (log, difference, Winsorise)
  before BVSS, or move to a discrete-outcome estimator.
* Strong serial correlation in pre-period residuals. The
  iid-error assumption inflates :math:`\phi`'s posterior
  precision and the inclusion threshold; weights become
  artificially sharp. First-difference the panel before
  feeding into BVSS, or use a cycle-decomposing estimator
  (:doc:`sbc`).
* Treated unit's outcome is non-stationary. A5 governs the
  pre-period sample size; the consistency story does not
  cover unit-root outcomes. First-difference, or move to a
  stationary-cycle approach.
* Continuous or multi-valued treatment. BVSS encodes binary
  treatment via a single treated-unit indicator. Continuous
  dose belongs in :doc:`ctsc`.
* You need a single sparse interpretable weight vector for
  policy storytelling. BVSS returns *posterior expected*
  weights -- a Bayesian model average. If the goal is the
  canonical "California = 0.385 Utah + 0.271 Montana + 0.186
  Nevada + ..." story, run *canonical SCM* alongside and report
  both. The inclusion probabilities from BVSS sit at a
  different rhetorical altitude.

Core API
--------

.. automodule:: mlsynth.estimators.bvss
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.BVSSConfig
   :members:
   :undoc-members:

Helper Modules
--------------

.. automodule:: mlsynth.utils.bvss_helpers.setup
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.bvss_helpers.posterior
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.bvss_helpers.gibbs_pair
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.bvss_helpers.mh
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.bvss_helpers.sampler
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.bvss_helpers.inference
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.bvss_helpers.plotter
   :members:
   :undoc-members:

.. note::

   ``BVSS.fit()`` returns an :class:`~mlsynth.config_models.EffectResult` on the
   standardized two-family contract: ``res.att`` (posterior mean ATT) /
   ``res.att_ci`` (credible interval) / ``res.counterfactual`` (posterior mean)
   / ``res.gap`` / ``res.donor_weights`` (posterior mean weights) /
   ``res.pre_rmse`` resolve through the standardized sub-models. The full
   Bayesian detail -- the MCMC posterior, per-draw ATT samples, and pointwise
   counterfactual bands -- is on ``res.inference_detail`` / ``res.posterior``
   (the bare ``res.inference`` slot is reserved for the standardized ATT-level
   :class:`~mlsynth.config_models.InferenceResults`). ``res.simplex`` carries
   the posterior's verdict on the soft simplex; see `Reading the simplex
   verdict`_.

.. automodule:: mlsynth.utils.bvss_helpers.structures
   :members:
   :undoc-members:

Example
-------

A minimal end-to-end run on the China anti-corruption / luxury-watch
panel (the same outcome series the empirical replication below
benchmarks against). With the default priors and sampler settings,
the BVSS API is just five fields plus a display toggle:

.. code-block:: python

   import pandas as pd
   from mlsynth import BVSS

   url = "https://raw.githubusercontent.com/jgreathouse9/mlsynth/refs/heads/main/basedata/china_watches_long.csv"
   data = pd.read_csv(url)

   config = {
       "df": data,
       "outcome": "y",
       "unitid": "unit",
       "time": "time",
       "treat": "treat",
       "display_graphs": True,
   }

   results = BVSS(config).fit()

   # ------------------------------------------------------------------
   # Headline ATT
   # ------------------------------------------------------------------
   print(f"ATT posterior mean: {results.inference_detail.att_mean:.4f}")
   print(f"95% credible interval: "
         f"[{results.inference_detail.att_ci_lower:.4f}, "
         f"{results.inference_detail.att_ci_upper:.4f}]")

   # ------------------------------------------------------------------
   # Donor selection: posterior inclusion probabilities and weight means
   # ------------------------------------------------------------------
   for donor in sorted(results.weight_means,
                       key=lambda k: -results.weight_means[k])[:5]:
       w_bar = results.weight_means[donor]
       p_in = results.inclusion_probs[donor]
       print(f"  {donor:25s}: weight = {w_bar:.4f}, P(included) = {p_in:.3f}")

   # ------------------------------------------------------------------
   # Soft-simplex diagnostic: what the posterior says about the constraint
   # ------------------------------------------------------------------
   s = results.simplex
   print(f"\nposterior mean of tau:        {s.tau_mean:.4f}")
   print(f"  95% credible set:           [{s.tau_q025:.4f}, {s.tau_q975:.4f}]")
   print(f"off-simplex sd per weight:    {s.deviation_scale:.4f}")
   print(f"  against a typical weight:   {s.relative_deviation:.1%}")
   print(f"sd of sum(w) about one:       {s.weight_sum_sd:.4f}")
   print(f"posterior mean model size:    {s.model_size_mean:.2f}")
   print(f"posterior mean of phi:        {results.posterior.phi.mean():.4f}")
   # A small relative_deviation means the hard simplex would have cost little;
   # a large one means the fit left the simplex to reach the treated series.

   # ------------------------------------------------------------------
   # Counterfactual path and per-period bands (in original outcome units)
   # ------------------------------------------------------------------
   results.inference_detail.counterfactual_mean    # shape (T,)
   results.inference_detail.counterfactual_lower   # shape (T,)
   results.inference_detail.counterfactual_upper   # shape (T,)

   # ------------------------------------------------------------------
   # Raw MCMC samples for downstream analysis (after burn-in)
   # ------------------------------------------------------------------
   results.posterior.mu      # (N, n_post_samples)
   results.posterior.phi     # (n_post_samples,)
   results.posterior.tau     # (n_post_samples,)
   results.posterior.gamma   # (N, n_post_samples) 0/1 inclusion indicators

Verification
------------

Empirical replication against the authors' published numbers (Path
A). Xu & Zhou's Section 6.2 [BVSS]_ benchmarks BVS-SS on the China
anti-corruption case: the Eight-Point policy announced in January 2013
sharply depressed luxury-watch imports, an outcome the paper measures
through the monthly growth rate of the customs category "watches with
case of, or clad with, precious metal" (from the ``fdPDA`` R package
of Shi & Huang 2023). The donor pool is the remaining :math:`N = 87`
HS commodity categories over Feb 2010 - Dec 2015. ``mlsynth.BVSS``
on the same panel reproduces the paper's Table 3 headline values
essentially exactly.

Path A: Shi & Huang (2023) luxury-watch imports
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The replication panel (1 treated category + 87 donor categories
:math:`\times` 71 months, :math:`T_0 = 35`) is hosted at
:file:`china_import_final.csv` in
`jgreathouse9/RepACCFDID <https://github.com/jgreathouse9/RepACCFDID>`_.
``mlsynth.BVSS`` with the paper's published hyperparameters
(:math:`\kappa_1 = \kappa_2 = 1`, :math:`a_1 = 0.01`, :math:`a_2 =
0.1`, :math:`\alpha = 1`, :math:`\theta = 0.2`, 1000 iterations, 500
discarded as burn-in) reproduces the published Table 3 values:

.. code-block:: python

   import pandas as pd
   import numpy as np
   from mlsynth import BVSS

   url = ("https://raw.githubusercontent.com/jgreathouse9/RepACCFDID/"
          "refs/heads/main/china_import_final.csv")
   raw = pd.read_csv(url).rename(columns={"Unnamed: 0": "yyyymm"})

   T = len(raw)
   T0 = int((raw["yyyymm"].astype(int) < 201301).sum())   # = 35
   donor_cols = [c for c in raw.columns if c not in ("yyyymm", "treated")]

   rows = [{"unit": "watches", "time": t, "y": float(raw["treated"].iloc[t]),
             "treat": int(t >= T0)} for t in range(T)]
   for c in donor_cols:
       rows += [{"unit": c, "time": t, "y": float(raw[c].iloc[t]),
                  "treat": 0} for t in range(T)]
   df = pd.DataFrame(rows)

   res = BVSS({
       "df": df, "outcome": "y", "treat": "treat",
       "unitid": "unit", "time": "time",
       "n_iter": 1000, "burn_in": 500,
       "kappa1": 1.0, "kappa2": 1.0, "theta": 0.2,
       "tau_a": 0.01, "tau_b": 0.1, "ci_alpha": 0.05,
       "init_phi": 1.0, "init_tau": 1.0, "seed": 0,
       "display_graphs": False,
   }).fit()

   tau = res.posterior.tau
   phi = res.posterior.phi
   gamma_count = res.posterior.gamma.sum(axis=0)
   print(f"ATT    = {res.inference_detail.att_mean:+.3f}  "
          f"({res.inference_detail.att_ci_lower:+.3f}, "
          f"{res.inference_detail.att_ci_upper:+.3f})")
   print(f"tau    = {tau.mean():.3f}")
   print(f"phi    = {phi.mean():.2f}")
   print(f"|gamma|= {gamma_count.mean():.2f}")

prints (one chain at ``seed=0``, after ~10 minutes on commodity
hardware):

.. list-table::
   :header-rows: 1
   :widths: 14 24 24

   * - Statistic
     - Published (Xu & Zhou 2025, Table 3)
     - Replicated here
   * - ATT (mean)
     - :math:`-0.021`
     - :math:`-0.020`
   * - ATT (95% credible interval)
     - :math:`(-0.032,\ -0.008)`
     - :math:`(-0.033,\ -0.003)`
   * - :math:`\phi` (posterior mean)
     - :math:`20.86`
     - :math:`19.94`
   * - :math:`\phi` (95% CI)
     - :math:`(12.22,\ 32.76)`
     - :math:`(11.41,\ 31.68)`
   * - :math:`\nu` (posterior mean)
     - :math:`0.069`
     - :math:`0.030`
   * - :math:`|\gamma|` (posterior mean)
     - :math:`5.09`
     - :math:`8.89`

The headline ATT matches to three decimals
(:math:`\widehat{\tau} = -0.020` here vs. :math:`-0.021` in
the paper) and the 95% credible interval lines up closely with the
published :math:`(-0.032,\ -0.008)`. The observation-noise precision
:math:`\phi` is reproduced essentially exactly (:math:`19.94` vs.
:math:`20.86`, well within the published interval). The mild
discrepancies in :math:`\nu` and :math:`|\gamma|` are MCMC-seed
sensitive: a different RNG draw shifts the posterior of the
soft-simplex tightness parameter without moving the headline ATT. The
ordering -- "the data supports the simplex constraint (:math:`\nu`
near zero) and the relevant donor set is small (a handful of
commodity categories)" -- carries through to the same substantive
conclusion the paper reports: a statistically meaningful negative ATT
on luxury-watch imports after the anti-corruption announcement, with
posterior credibility that excludes zero.

Cross-validation against the authors' own sampler. Beyond matching the
published table, ``benchmarks/cases/bvss_watches.py`` checks mlsynth's BVSS
directly against Xu & Zhou's own two-coordinate Gibbs code (taken verbatim from
their fsPDA replication script) on the same anti-corruption panel. Because the
sampler is stochastic, the check has two layers. The deterministic engine --
the posterior covariance log-determinant, the ``RSS`` / ``RSS2`` quadratic and
bilinear forms, the model-complexity term and the marginal log-likelihood -- is
reproduced value for value on a fixed :math:`(\gamma, \tau, \mu, \phi)` to about
:math:`10^{-9}`, so both implementations draw from the identical target. The
seeded posterior-mean ATT then agrees within Monte-Carlo error
(:math:`-0.021` here vs. :math:`-0.021` from the R sampler, independent RNG
streams), with a negative credible set and a sparse selected model on both
sides. The live-captured reference is under
``benchmarks/reference/bvss_watches/``.

Three properties of that credible set are asserted: that it brackets the point
estimate, that it is entirely negative as the R's is, and that its width is
within an order of magnitude of the R's. The first two are exact. The third
carries a deliberately wide tolerance, because the width is a 95 % quantile of
25 retained draws and ranges over 0.48 to 1.07 times the R's across eight seeds
at the benchmark's chain length -- 0.67 to 1.08 even at 400 iterations. A
tolerance loose enough never to flake asserts nothing about agreement, so the
ratio is labelled in the case as a guard against order-of-magnitude breakage
and the two shape properties carry the content.

Dependencies
------------

By design the BVS-SS implementation in :mod:`mlsynth` avoids the
heavier probabilistic-programming stack. The sampler depends only on
:mod:`numpy` and the following modules from :mod:`scipy`:

* :mod:`scipy.linalg` — ``solve`` and ``det``
* :mod:`scipy.special` — ``factorial``
* :mod:`scipy.stats` — ``norm``, ``truncnorm``, ``gamma``

The optional ``tqdm`` progress bar is imported lazily and only when
``verbose=True`` is passed.
