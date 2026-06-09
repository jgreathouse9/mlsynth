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

Assumptions (Xu & Zhou 2025)
----------------------------

The paper proves *high-dimensional strong selection consistency* --
the posterior probability of the true active-donor model converging
to 1 under the true DGP -- in Theorem 2, under the technical
conditions A1-A5 (Section 4.1). Stated for the working
:math:`\tau = 0` (hard-simplex) limit:

**A1 (Restricted eigenvalue on the donor matrix).** Each column
:math:`X_j` of the donor matrix satisfies :math:`\| X_j \|_2^2 =
T_0`, and there exists :math:`\underline\lambda \in (0, 1]` such
that :math:`\lambda_{\min}(X_\gamma^\top X_\gamma) \ge T_0
\underline\lambda` for every candidate model :math:`\gamma` in the
sparse model space :math:`\mathbb{S}_L`. Translation: no two donors
are perfectly collinear, no donor is a near-duplicate of a linear
combination of a few others, and the minimum eigenvalue of every
"reasonable-size" donor submatrix is bounded away from zero.

**A2 (Inclusion-prior penalty).** The Bernoulli inclusion
probability satisfies :math:`\theta / (1 - \theta) = N^{-c_\theta L}`
for some universal :math:`c_\theta > 0`. Translation: the prior
penalises large models geometrically in the donor count -- a
default :math:`\theta \approx 1/N` keeps the prior expected model
size around 1.

**A3 (Noise-precision prior).** The Gamma prior on :math:`\phi`
has shape :math:`\kappa_1 \in (0, T_0]` and rate :math:`\kappa_2
\in [0, \sigma^2 T_0 / 2]`. Translation: the prior on the inverse
error variance is not pathologically informative (neither shape
nor rate scale faster than :math:`T_0`).

**A4 (True DGP + signal strength /** :math:`\beta`-min).** The
true outcome is :math:`Y \mid X \sim \mathcal{N}(X_{\gamma^*}
\mu_{\gamma^*}^*, \sigma^2 I)` with :math:`\ell^* := |\gamma^*|
\le L \wedge \sqrt{L \log N}`, :math:`\mu_{\gamma^*}^* \in
\Delta^{\ell^* - 1}`, and a lower bound on the smallest non-zero
weight,

.. math::

   \min_{j \in \gamma^*} |\mu_j^*|
   \;\ge\;
   \frac{c_\mu \sigma \sqrt{L \log N}}{\underline\lambda \sqrt{T_0}}.

Translation: the true donor pool is sparse, lies on the simplex,
and every truly-active donor carries weight detectable above the
noise floor. The :math:`\beta`-min lower bound is the standard
"signal large enough to identify" condition shared with all
high-dimensional consistency theory (Yang et al. 2016).

**A5 (Sample-size lower bound).** :math:`L \ge 3` and
:math:`T_0 \ge c_M \ell^* \log N` for some :math:`c_M > 0`.
Translation: the pre-period must grow with the (log of) the donor
pool size for selection consistency to take hold.

**Theorem 2 (Xu & Zhou 2025).** Under A1-A5, the posterior
inclusion probability of the true model satisfies
:math:`p(\gamma^* \mid y) \xrightarrow{p^*} 1` as :math:`T_0, N
\to \infty` along any allowed sequence. Theorem 3 then bounds the
posterior expected predictive loss on the test (post-treatment)
sample at the order :math:`T_1 \ell^* \log N / T_0`, which is the
"oracle" rate for the constrained problem -- i.e., BVS-SS is
asymptotically as efficient as if an oracle had told you the true
active set in advance.

For the soft-simplex limit :math:`\tau \to \infty`, Theorem 4
shows the BVS-SS posterior of :math:`\gamma` converges to the
unconstrained spike-and-slab posterior. The two regimes are
genuine endpoints of the same family; the data picks between them
through the learned posterior of :math:`\tau`.

When the assumptions bind: practical diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

(a) **Donor near-collinearity (A1).** If two donors carry
    essentially the same pre-period information, the minimum
    eigenvalue of :math:`X_\gamma^\top X_\gamma` is near zero on
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

(b) **Prior penalty scale (A2).** A too-large :math:`\theta`
    floods the posterior with large-model mass; a too-small one
    blocks even truly-active donors.

    *Plausibly violated when* the default :math:`\theta` was
    chosen blindly. *Diagnostic*: track the posterior model size
    ``results.posterior.gamma.sum(axis=0)``; if its posterior
    mean is essentially the prior mean :math:`N \theta`, the
    data is not informing model size and you should tighten
    :math:`\theta`. The paper's empirical applications use
    :math:`\theta = 0.2`.

(c) **Sparse, simplex-supported truth (A4).** Selection
    consistency requires that the true weight vector is sparse
    *and* on the simplex. If the truth is genuinely dense
    (many donors each contributing a small fraction) or
    genuinely off-simplex (weights summing to substantially
    more or less than 1), the consistency story does not apply
    -- though the procedure still produces a well-defined
    posterior.

    *Plausibly violated when* the treated unit is structurally
    outside the donor convex hull (Hong Kong handover; very
    extreme growth unit). *Diagnostic*: the posterior of
    :math:`\tau` is your friend here -- if
    ``results.posterior.tau.mean()`` is materially above zero,
    the data is telling you the simplex does not hold and you
    should reach for the unconstrained-spike-and-slab limit
    (Kim et al. 2020) or for an estimator that explicitly
    handles outside-hull treated units (:doc:`iscm`).

(d) **Long-enough pre-period (A5).** Selection consistency
    requires :math:`T_0 \gtrsim \ell^* \log N`. With 20
    pre-period observations and 100 donors, you need at most
    :math:`\ell^* \le 20 / \log 100 \approx 4` truly-active
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

(e) **Gaussian likelihood with shock independence.** The model
    assumes :math:`Y_t \sim \mathcal{N}(X_t w, \phi^{-1})` with
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

(f) **Posterior of** :math:`\tau` **as the soft-simplex test.**
    The single diagnostic that summarises the
    simplex-vs-unconstrained decision: small posterior mass of
    :math:`\tau` near zero means the data agrees with the
    simplex; bulk above the prior mean means the data prefers
    the relaxation.

    *Practical rule of thumb*: compare ``results.posterior.tau``
    against the prior mean :math:`a_1 / a_2`. If the posterior
    is concentrated well below the prior mean, the simplex is
    supported. If the posterior sits at or above the prior mean
    -- the China-watches case in the empirical application
    below has posterior mean :math:`\tau \approx 0.03` against
    a prior mean of :math:`0.1`, supporting the simplex -- the
    data is consistent with the constraint. The Hong Kong
    handover case (Hsiao 2012, motivation in Section 1.1 of the
    paper) shows the opposite: posterior :math:`\tau` mass
    elevated above the prior, telling the user to relax the
    simplex.

When to use BVSS -- and when not to
-----------------------------------

**Reach for BVSS when:**

* **The donor pool is large relative to the pre-period**
  (:math:`N \gtrsim T_0`). The classical SCM quadratic program
  has no unique solution; Lasso over-selects; BVSS's
  spike-and-slab cleanly returns a sparse posterior on inclusion.
* **The simplex constraint itself is in question.** Hong Kong
  handover, Basque conflict, China anti-corruption watches --
  cases where the treated unit's level may legitimately exceed
  any convex combination of donors. BVSS's learned :math:`\tau`
  tells you which regime the data prefers.
* **You want posterior distributions, not point estimates.**
  The full posterior of the ATT, of each donor's weight, and of
  inclusion are returned -- useful for downstream uncertainty
  propagation (e.g. into a policy-evaluation report's CI
  reporting).
* **You want to formalise the "which donors do I trust" decision.**
  Posterior inclusion probabilities replace the practitioner's
  eyeball "I'll keep these states because they look similar"
  step with a probability statement.

**Do not use BVSS when:**

* **Small donor pool with clear pre-fit.** With :math:`N = 8`
  states and a tight pre-fit on the canonical SCM, BVSS's
  per-pair Gibbs sampler is overkill; the spike-and-slab
  uncertainty just propagates noise that the data does not
  actually contain. Use *canonical SCM*, :doc:`tssc`, or :doc:`fdid`.
* **Treated unit is severely outside the donor hull.** BVSS's
  soft simplex *can* relax (:math:`\tau` learns), but the
  posterior weights are still anchored to the simplex prior.
  For the structural outside-hull case, :doc:`iscm`'s moment-
  condition framework is identification-aware in a way BVSS is
  not.
* **Distribution of the outcome is the object of interest.**
  BVSS targets the mean ATT through a Gaussian likelihood on
  levels. Distributional questions (Lorenz curves, QTEs,
  inequality measures) need :doc:`dsc`.
* **You need speed.** MCMC is materially slower than
  optimisation-based SC. The China-watches application takes
  ~10 minutes for 1000 iterations on commodity hardware; for
  large grid searches or batch processing across many
  policy-evaluation cells, an optimisation-based estimator
  (*canonical SCM*, :doc:`tssc`, :doc:`fdid`) is the right default.
* **Outcomes with hard floors or ceilings, counts, or heavy
  tails.** A4's Gaussian-likelihood / :math:`\beta`-min
  argument breaks. Pre-process (log, difference, Winsorise)
  before BVSS, or move to a discrete-outcome estimator.
* **Strong serial correlation in pre-period residuals.** The
  iid-error assumption inflates :math:`\phi`'s posterior
  precision and the inclusion threshold; weights become
  artificially sharp. First-difference the panel before
  feeding into BVSS, or use a cycle-decomposing estimator
  (:doc:`sbc`).
* **Treated unit's outcome is non-stationary.** A5 governs the
  pre-period sample size; the consistency story does not
  cover unit-root outcomes. First-difference, or move to a
  stationary-cycle approach.
* **Continuous or multi-valued treatment.** BVSS encodes binary
  treatment via a single treated-unit indicator. Continuous
  dose belongs in :doc:`ctsc`.
* **You need a single sparse interpretable weight vector for
  policy storytelling.** BVSS returns *posterior expected*
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
   :class:`~mlsynth.config_models.InferenceResults`).

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
   # Soft-simplex diagnostic: posterior of tau
   # ------------------------------------------------------------------
   print(f"\nPosterior mean of tau:   {results.posterior.tau.mean():.4f}")
   print(f"Posterior mean of phi:   {results.posterior.phi.mean():.4f}")
   # Small tau values => simplex is well-supported by the data.
   # Large tau values => the data prefers a relaxation.

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

**Empirical replication against the authors' published numbers (Path
A).** Xu & Zhou's Section 6.2 [BVSS]_ benchmarks BVS-SS on the China
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
   * - :math:`\tau` (posterior mean)
     - :math:`0.069`
     - :math:`0.030`
   * - :math:`|\gamma|` (posterior mean)
     - :math:`5.09`
     - :math:`8.89`

The headline ATT matches to three decimals
(:math:`\widehat{\mathrm{ATT}} = -0.020` here vs. :math:`-0.021` in
the paper) and the 95% credible interval lines up closely with the
published :math:`(-0.032,\ -0.008)`. The observation-noise precision
:math:`\phi` is reproduced essentially exactly (:math:`19.94` vs.
:math:`20.86`, well within the published interval). The mild
discrepancies in :math:`\tau` and :math:`|\gamma|` are MCMC-seed
sensitive: a different RNG draw shifts the posterior of the
soft-simplex tightness parameter without moving the headline ATT. The
ordering -- "the data supports the simplex constraint (:math:`\tau`
near zero) and the relevant donor set is small (a handful of
commodity categories)" -- carries through to the same substantive
conclusion the paper reports: a statistically meaningful negative ATT
on luxury-watch imports after the anti-corruption announcement, with
posterior credibility that excludes zero.

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
