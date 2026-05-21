Bayesian Synthetic Control with a Soft Simplex Constraint (BVS-SS)
==================================================================

.. currentmodule:: mlsynth

Overview
--------

BVS-SS `arXiv:2503.06454 <https://arxiv.org/abs/2503.06454>`_ is a
Bayesian synthetic control estimator that wraps two ideas around the
standard SCM regression :math:`Y = Xw + \varepsilon`:

* **Spike-and-slab variable selection.** Each donor is either active
  (:math:`\gamma_i = 1`) or excluded (:math:`\gamma_i = 0,\ w_i = 0`),
  with a Bernoulli prior on inclusion. This handles high-dimensional
  panels where ``N`` (donor count) is comparable to or exceeds ``T``,
  and provides posterior inclusion probabilities :math:`P(\gamma_i = 1
  \mid y)` that quantify donor relevance.
* **Soft simplex constraint.** Selected weights are drawn around a
  Dirichlet mean :math:`\mu_\gamma` with a learnable variance
  :math:`\tau`. When :math:`\tau \to 0` the prior collapses to the
  hard simplex of Abadie & Gardeazabal (2003); when :math:`\tau \to
  \infty` it becomes an unconstrained spike-and-slab regression. The
  posterior of :math:`\tau` tells the user whether the simplex
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

Mathematical Formulation
------------------------

Let :math:`Y \in \mathbb{R}^{T_0}` be the pre-treatment outcome of the
treated unit and :math:`X \in \mathbb{R}^{T_0 \times N}` the
contemporaneous donor matrix. Both are demeaned column-wise using the
pre-treatment means before entering the sampler (this matches the
paper's working setup and makes the soft-simplex prior numerically
well-conditioned).

Likelihood and Hierarchical Prior
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Gaussian likelihood is

.. math::

   Y \mid w, \phi \;\sim\; \mathcal{N}\!\left( X w,\; \phi^{-1} I \right).

The hierarchical prior over
:math:`(w, \mu, \tau, \phi, \gamma)` from Eq. (2) of the paper is

.. math::

   \begin{aligned}
   \phi &\sim \mathrm{Gamma}(\kappa_1 / 2,\; \kappa_2 / 2), \\
   \gamma_i &\stackrel{\text{i.i.d.}}{\sim} \mathrm{Bernoulli}(\theta),
       \quad i = 1, \dots, N, \\
   \tau &\sim \mathrm{Gamma}(a_1,\; a_2), \\
   \mu_\gamma \mid \gamma &\sim \mathrm{sym\text{-}Dirichlet}(\alpha), \\
   w_\gamma \mid \gamma, \mu_\gamma, \tau, \phi
       &\sim \mathcal{N}\!\left(\mu_\gamma,\; \tfrac{\tau}{\phi} I\right),
   \end{aligned}

with the convention that :math:`\mu_i = w_i = 0` whenever
:math:`\gamma_i = 0`. The fixed-:math:`\alpha = 1` case (uniform
Dirichlet on the active simplex) is what BVS-SS implements in this
package; the paper's simulations and both empirical applications use
this setting.

Interpreting the soft-constraint variance :math:`\tau` is the key
modeling insight:

* As :math:`\tau \downarrow 0`, the prior of :math:`w_\gamma`
  concentrates at :math:`\mu_\gamma \in \Delta^{|\gamma| - 1}`, i.e.
  the hard simplex.
* As :math:`\tau \uparrow \infty`, the prior on :math:`w_\gamma` is
  effectively uninformative, recovering an unconstrained spike-and-slab
  regression à la Kim et al. (2020).

The posterior of :math:`\tau` is therefore a data-driven indicator of
whether the simplex constraint is appropriate for the application at
hand. The paper proves (Theorem 4) that as :math:`\tau \to \infty`
the BVS-SS posterior of :math:`\gamma` becomes identical to the
unconstrained spike-and-slab posterior, so the data really does pick
between the two regimes through :math:`\tau`.

Marginal Likelihood and Posterior Conditionals
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A Woodbury-identity calculation (Eq. (4) of the paper) yields

.. math::

   p(y \mid \gamma, \mu_\gamma, \tau, \phi)
   \;\propto\;
   \frac{\phi^{T_0 / 2}}{\tau^{|\gamma| / 2}
        \det(V_{\gamma, \tau})^{1/2}}
   \exp\!\left\{
       -\tfrac{\phi}{2}
       (y - X_\gamma \mu_\gamma)^\top
       \Sigma_{\gamma, \tau}
       (y - X_\gamma \mu_\gamma)
   \right\},

with the two repeated quantities

.. math::

   V_{\gamma, \tau} = X_\gamma^\top X_\gamma + \tau^{-1} I,
   \qquad
   \Sigma_{\gamma, \tau} = I - X_\gamma V_{\gamma, \tau}^{-1} X_\gamma^\top.

The :math:`\phi` and :math:`w_\gamma` full conditionals are
conjugate (Eqs. (6)–(7)):

.. math::

   \begin{aligned}
   \phi \mid y, \gamma, \mu_\gamma, \tau
       &\sim \mathrm{Gamma}\!\left(
           \tfrac{T_0 + \kappa_1}{2},\;
           \tfrac{\kappa_2 + (y - X_\gamma \mu_\gamma)^\top
                   \Sigma_{\gamma, \tau}
                   (y - X_\gamma \mu_\gamma)}{2}
       \right), \\
   w_\gamma \mid y, \gamma, \mu_\gamma, \tau, \phi
       &\sim \mathcal{N}\!\left(
           V_{\gamma, \tau}^{-1}
           \!\left(X_\gamma^\top y + \tau^{-1} \mu_\gamma\right),\;
           \phi^{-1} V_{\gamma, \tau}^{-1}
       \right).
   \end{aligned}

The :math:`\tau` conditional has no closed form. The :math:`(\gamma,
\mu)` block is the technically interesting piece — it cannot be
updated coordinate-wise (the simplex constraint
:math:`\sum_{i: \gamma_i = 1} \mu_i = 1` makes single-coordinate
moves degenerate) so the paper introduces a *two-coordinate Gibbs
update* over pairs :math:`(\gamma_i, \gamma_j, \mu_i, \mu_j)`.

Metropolis-within-Gibbs Sampler (Algorithm 1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One outer iteration performs three updates in order.

**1. Pair update for** :math:`(\gamma_i, \gamma_j, \mu_i, \mu_j)`.
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

     \Lambda_{i,j} \;=\; (X_i - X_j)^\top \Sigma_{\gamma^{ij}, \tau}
                          (X_i - X_j), \quad
     \beta_{i,j} \;=\; \frac{1}{\Lambda_{i,j}}\,
                          (X_i - X_j)^\top \Sigma_{\gamma^{ij}, \tau}
                          \!\left(y - s X_j - \!\!\!\sum_{k \neq i, j}
                                 \mu_k X_k\right).

The pair-update sweep visits every :math:`(i, j)` pair once per outer
iteration. With :math:`\alpha = 1` all probabilities involve only the
standard normal CDF and a single truncated-normal draw — no
rejection sampling and no numerical integration.

**2.** :math:`\phi` **Gibbs draw.** Plug the updated :math:`\mu`
into the closed-form Gamma conditional above.

**3.** :math:`\tau` **MH steps.** Repeat for :math:`n_\tau`
iterations a log-random-walk proposal

.. math::

   \log \tau^* \;=\; \log \tau \;+\; \mathcal{N}(0, 1),

with the lower boundary :math:`\log \tau_{\min}` reflected (proposals
below :math:`\tau_{\min}` are mirrored back into the support). The
acceptance ratio is

.. math::

   \rho(\tau, \tau^*) \;=\;
   \min\!\left\{
       1,\;
       \frac{p(y \mid \mu, \tau^*, \phi)\, p(\tau^*)}
            {p(y \mid \mu, \tau, \phi)\, p(\tau)}
       \cdot \frac{\tau^*}{\tau}
   \right\},

where the final :math:`\tau^* / \tau` factor is the Jacobian of the
log-space random walk.

Counterfactual Imputation and ATT
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The sampler returns posterior draws :math:`\{\mu^{(t)}\}_{t = 1}^{S}`
after burn-in. The treated unit's counterfactual at every period
:math:`t` is constructed from the column-demeaned donor matrix
:math:`\tilde{X}` (using the same pre-treatment means as the
sampler) via

.. math::

   \hat Y_t^{(0), (t)} \;=\; \tilde{X}_t \mu^{(t)} + \bar y_{\text{pre}},

where :math:`\bar y_{\text{pre}}` is the pre-treatment mean of the
treated outcome. The post-treatment ATT per draw is

.. math::

   \widehat{\mathrm{ATT}}^{(t)}
   \;=\;
   \frac{1}{T - T_0}
   \sum_{t = T_0 + 1}^{T}
   \!\left( Y_t - \hat Y_t^{(0), (t)} \right),

and the reported headline ATT is the posterior mean of
:math:`\widehat{\mathrm{ATT}}^{(t)}`. Credible intervals are
percentile bands over :math:`\widehat{\mathrm{ATT}}^{(t)}` at level
:math:`1 - \mathrm{ci\_alpha}` (default 95 %). Pointwise bands on the
counterfactual are computed analogously period-by-period.

Note that BVS-SS computes the counterfactual directly from the
:math:`\mu` posterior rather than drawing :math:`w_\gamma` from its
Eq. (7) conditional. The paper's empirical Section 6 uses this
lower-variance estimator; the implementation in :mod:`mlsynth` matches
that choice.

When BVS-SS Is the Right Tool
-----------------------------

The paper documents three settings where BVS-SS materially outperforms
the classical SCM:

* **High-dimensional donor pools** (Section 5.2). When ``N`` is
  comparable to or exceeds ``T_0``, the standard quadratic program
  does not have a unique solution and Lasso-style alternatives tend
  to over-select. BVS-SS's spike-and-slab structure recovers a sparse
  subset and converges to the oracle OLS estimator as the sample
  grows.
* **Simplex-violating data** (Section 5.2,
  :math:`\|w^*\|_1 \in \{2, 3\}`). When the true sum of weights
  exceeds 1 — as it plausibly does for outliers like Hong Kong's GDP
  in the late 1990s — methods that enforce the hard simplex are
  biased, while BVS-SS's posterior of :math:`\tau` moves away from
  zero and the model adapts.
* **Anti-tax-evasion / anti-corruption policy evaluation** (Section
  6). Replicating Carvalho et al. (2018) and Shi & Huang (2023),
  BVS-SS recovers the published ATT magnitudes with substantially
  smaller credible intervals than Lasso-based ArCo, while selecting
  only ~2 donors on average instead of the entire pool.

A handy diagnostic is the *posterior of* :math:`\tau`: small
posterior mass near zero is a sign the data agrees with the simplex
constraint, while bulk away from zero is evidence to relax it. BVS-SS
exposes ``results.posterior.tau`` for direct inspection.

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

.. automodule:: mlsynth.utils.bvss_helpers.structures
   :members:
   :undoc-members:

Example
-------

.. code-block:: python

   import pandas as pd
   from mlsynth import BVSS

   url = "https://raw.githubusercontent.com/jgreathouse9/mlsynth/refs/heads/main/basedata/basque_data.csv"
   data = pd.read_csv(url)

   config = {
       "df": data,
       "outcome": "gdpcap",
       "unitid": "regionname",
       "time": "year",
       "treat": "Terrorism",
       # Prior hyperparameters
       "kappa1": 1.0, "kappa2": 1.0,        # phi ~ Gamma(kappa1/2, kappa2/2)
       "tau_a": 0.01, "tau_b": 0.1,         # tau ~ Gamma(tau_a, tau_b)
       "theta": 0.25,                        # Bernoulli inclusion prior
       # Sampler
       "n_iter": 2000, "burn_in": 1000,
       "n_tau": 11,                          # MH steps for tau per outer iter
       "tau_min": 1e-6,
       "ci_alpha": 0.05,                     # 95% credible intervals
       "seed": 42,
       "display_graphs": True,
   }

   results = BVSS(config).fit()

   # ------------------------------------------------------------------
   # Headline ATT
   # ------------------------------------------------------------------
   print(f"ATT posterior mean: {results.inference.att_mean:.4f}")
   print(f"95% credible interval: "
         f"[{results.inference.att_ci_lower:.4f}, "
         f"{results.inference.att_ci_upper:.4f}]")

   # ------------------------------------------------------------------
   # Donor selection: posterior inclusion probabilities and weight means
   # ------------------------------------------------------------------
   for donor in sorted(results.weight_means,
                       key=lambda k: -results.weight_means[k])[:5]:
       w_bar = results.weight_means[donor]
       p_in = results.inclusion_probs[donor]
       print(f"  {donor:25s}: weight = {w_bar:.4f}, P(included) = {p_in:.3f}")

   # ------------------------------------------------------------------
   # Soft-simplex diagnostic: posterior of tau
   # ------------------------------------------------------------------
   print(f"\nPosterior mean of tau:   {results.posterior.tau.mean():.4f}")
   print(f"Posterior mean of phi:   {results.posterior.phi.mean():.4f}")
   # Small tau values => simplex is well-supported by the data.
   # Large tau values => the data prefers a relaxation.

   # ------------------------------------------------------------------
   # Counterfactual path and per-period bands (in original outcome units)
   # ------------------------------------------------------------------
   results.inference.counterfactual_mean    # shape (T,)
   results.inference.counterfactual_lower   # shape (T,)
   results.inference.counterfactual_upper   # shape (T,)

   # ------------------------------------------------------------------
   # Raw MCMC samples for downstream analysis (after burn-in)
   # ------------------------------------------------------------------
   results.posterior.mu      # (N, n_post_samples)
   results.posterior.phi     # (n_post_samples,)
   results.posterior.tau     # (n_post_samples,)
   results.posterior.gamma   # (N, n_post_samples) 0/1 inclusion indicators

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
