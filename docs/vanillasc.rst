Vanilla Synthetic Control (VanillaSC)
=====================================

.. currentmodule:: mlsynth

Overview
--------

``VanillaSC`` is the *standard* synthetic control method (Abadie &
Gardeazabal 2003; Abadie, Diamond & Hainmueller 2010, 2015), built on
mlsynth's self-contained bilevel engine. It estimates the effect on a
single treated unit by constructing a weighted average of donor units --
the *synthetic control* -- that tracks the treated unit's pre-treatment
path, and reads the effect as the post-treatment gap between the treated
unit and its synthetic counterpart.

What distinguishes this implementation is how it treats the two regimes
of the SCM optimisation honestly:

* **No covariates** -> the donor weights :math:`W` solve the convex
  simplex least-squares fit on the pre-treatment outcomes. This is a
  single, well-posed convex program -- deterministic and reproducible
  (unique up to donor collinearity).
* **Covariates** -> the predictor weights :math:`V` and donor weights
  :math:`W` are chosen jointly through a **bilevel** program. This is
  non-convex, and the predictor weights are generically *non-identified*.
  ``VanillaSC`` solves it with a reliable backend and reports a diagnostic
  (:math:`\text{v\_agreement}`) so that fragility is visible rather than
  silent.

Mathematical formulation
------------------------

For a treated unit with pre-treatment outcomes :math:`y_1 \in
\mathbb{R}^{T_0}` and donors :math:`Y_0 \in \mathbb{R}^{T_0 \times J}`:

**Outcome-only (no covariates).**

.. math::

   \widehat W = \arg\min_{W} \; \lVert y_1 - Y_0 W \rVert^2
   \quad \text{s.t.} \quad W \ge 0,\ \mathbf{1}'W = 1.

**Covariate matching (bilevel).** With predictor matrices :math:`X_1 \in
\mathbb{R}^{P}` (treated) and :math:`X_0 \in \mathbb{R}^{P \times J}`
(donors), each predictor averaged over its window and scaled to unit
variance, the lower level solves, for given diagonal predictor weights
:math:`V`,

.. math::

   W^\*(V) = \arg\min_{W \in \Delta} \; (X_1 - X_0 W)' V (X_1 - X_0 W),

and the upper level chooses :math:`V` to minimise the pre-treatment
*outcome* fit,

.. math::

   \min_{V} \; \lVert y_1 - Y_0\, W^\*(V) \rVert^2 .

The donor weights :math:`W` and the counterfactual are pinned by this
program; the predictor weights :math:`V` are generically not (a whole
polytope of :math:`V` reproduces the same :math:`W`).

Backends
--------

The covariate path exposes three reliable solvers via ``backend=``:

``"outcome-only"``
    No predictor weights; the convex simplex fit above. The well-posed
    default (also selected by ``backend="auto"`` when no covariates are
    given).

``"mscmt"``
    Becker & Kloessner (2018): a global differential-evolution search over
    :math:`\log_{10} V` with the simplex inner solve. The default when
    covariates are supplied. Set ``canonical_v="min.loss.w"`` (or
    ``"max.order"``) to report a canonical, reproducible :math:`V` via the
    MSCMT ``determine_v`` step.

``"malo"``
    Malo et al. (2024): a staged corner search. Fast and exact when the
    optimum is a predictor corner -- but note that when a *lagged outcome*
    is among the predictors, the loss-minimising corner puts all weight on
    that lag, collapsing the inner match to pure outcome-fitting (it
    drifts toward the outcome floor).

``"penalized"``
    Abadie & L'Hour (2021): a pairwise-penalized estimator with
    leave-one-out :math:`\lambda` selection, giving a **unique, sparse**
    :math:`W`. Works with or without covariates.

The identification diagnostic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When covariates are used, ``res.weights.summary_stats["v_agreement"]``
reports the maximum absolute difference between the two MSCMT canonical
predictor-weight vectors (``min.loss.w`` and ``max.order``). It is small
when :math:`V` is well identified and large (up to 1) when the predictor
weights -- and the donor weights they imply -- are fragile. A large value
is a warning that the covariate-matched solution should not be
over-interpreted.

Inference
---------

Two inference modes are available via ``inference=``:

``"placebo"`` (default, ``inference=True``)
    Abadie's in-space placebo test: the synthetic control is refit treating
    each donor as pseudo-treated, and the treated unit's post/pre RMSPE ratio
    is ranked against the placebo distribution to give a p-value. Simple and
    assumption-light, but the smallest achievable p-value is about
    :math:`1/(J+1)`.

``"scpi"`` -- prediction intervals (Cattaneo, Feng & Titiunik 2021)
    Treats :math:`\tau_T` as a *predictand* (a random variable) and builds
    **prediction intervals**, decomposing the prediction error as

    .. math::

       \widehat\tau_T - \tau_T = e_T - \mathbf{p}_T'(\widehat\beta - \beta_0),

    an out-of-sample shock :math:`e_T` plus an in-sample weight-estimation
    error. The counterfactual prediction band is assembled period-by-period as
    :math:`[\,Y_{\text{fit}} + w_L + e_L,\; Y_{\text{fit}} + w_U + e_U\,]`,
    and the treatment-effect interval is
    :math:`[\,Y_{\text{obs}} - \text{cf}_U,\; Y_{\text{obs}} - \text{cf}_L\,]`.

    * **In-sample** (:math:`w_L`/:math:`w_U`): a simulation-based bound. With
      :math:`Q = Z'Z/T_0` (donor pre-outcomes), :math:`\widehat\Sigma = Z'
      \mathrm{diag}(\omega)\,Z / T_0^2` where :math:`\omega_t =
      \tfrac{T_0}{T_0-\mathrm{df}}(u_t - E[u_t])^2` (HC1), and pre-period
      residuals :math:`u = A - B\widehat w`, draw :math:`G^\star \sim
      N(0,\widehat\Sigma)`. For each draw and predictor :math:`\mathbf{p}_T`,
      solve over the *localised* simplex set

      .. math::

         \min/\max\ \mathbf{p}_T'x \quad\text{s.t.}\quad
         (x-\widehat w)'Q(x-\widehat w) - 2G^{\star\prime}(x-\widehat w) \le 0,\;
         \textstyle\sum x = 1,\; x \ge \ell,

      with :math:`\ell_j = \widehat w_j` if :math:`\widehat w_j < \rho` else
      :math:`0`. The regularisation parameter :math:`\rho` is data-driven and
      capped at :math:`\rho_{\max} = 0.2`; :math:`Q` is reduced via a
      thresholded eigen-square-root so collinear (near-null) donor directions
      are left unconstrained. :math:`w_L`/:math:`w_U` are the
      :math:`\alpha_1/2` / :math:`1-\alpha_1/2` quantiles of
      :math:`\mathbf{p}_T'(\widehat w - x)` across draws.
    * **Out-of-sample** (:math:`e_L`/:math:`e_U`): a location-scale model,
      :math:`e_T = E[e] + \sqrt{\mathrm{Var}[e]}\,\varepsilon`. The conditional
      mean and a log-variance scale (capped by the residual IQR, Gaussian
      :math:`\varepsilon`) are estimated by regressing :math:`u` on the
      active-donor design; ``"ls"`` and ``"empirical"`` use standardized /
      raw residual quantiles.

    ``VanillaSC`` returns the average-effect (ATT) interval in
    ``res.inference.ci_lower``/``ci_upper`` and the full per-period sequence
    (point effects, prediction intervals, counterfactual bands, and the
    in-/out-of-sample components) in ``res.inference.details``. This
    implements the canonical simplex / outcome-only case; for covariate
    backends it uses the same outcome design and is approximate.

    .. note::

       This is a self-contained, **MIT-licensed** re-derivation of the
       Cattaneo-Feng-Titiunik algorithm -- it does **not** import the GPL
       reference package ``scpi``. It is validated to reproduce ``scpi``'s
       ``CI_all_gaussian`` on the Proposition 99 panel to within Monte-Carlo
       error (see ``test_scpi_matches_reference_package``, which is skipped
       unless ``scpi_pkg`` happens to be installed).

``"lto"`` -- leave-two-out refined placebo (Lei & Sudijono 2025)
    A design-based randomization test that fixes the two structural weaknesses
    of the ordinary placebo test -- its **coarse** :math:`\{1/N, 2/N, \dots\}`
    grid and its **zero size when** :math:`\alpha < 1/N`. It replaces the "one
    turn each" permutation with a *tournament over triples* and reports both a
    naive p-value (``res.inference.p_value``) and a powered one
    (``details["p_powered"]``), together with the Type-I bound and tournament
    tallies. It shares the placebo test's assumptions but is far more powerful
    in small donor pools. See *The leave-two-out refined placebo test* and the
    two theory subsections below for the full treatment.

How the SCPI machinery works (one fit)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``scpi_intervals(y, Y0, pre, W, ...)`` takes the fitted donor weights
:math:`\widehat w` (from *any* backend), the donor outcome matrix, and the
number of pre-treatment periods, and runs the following steps. Let
:math:`A = y_{1:T_0}` be the treated pre-outcomes, :math:`B = Y_{0,\,1:T_0}`
the donor pre-outcomes, :math:`P` the donor post-outcomes, and
:math:`u = A - B\widehat w` the pre-period residuals.

1. **Degrees of freedom.** For the simplex, :math:`\mathrm{df} =
   (\#\{\widehat w_j \neq 0\}) - 1`, giving the HC1 correction
   :math:`\mathrm{vc} = T_0/(T_0-\mathrm{df})`.

2. **Regularisation parameter** :math:`\rho`. The data-driven ``type-1`` value
   :math:`\rho = \tfrac{\sigma_u}{\min_j \mathrm{sd}(B_j)}
   \sqrt{\log(J)\, d_0 \log T_0}/\sqrt{T_0}`, capped at
   :math:`\rho_{\max}=0.2` (with a fallback bump if it comes out below
   :math:`0.001`). :math:`\rho` defines the "active" donor set
   :math:`\{\,j : \widehat w_j > \rho\,\}`.

3. **Conditional mean & variance.** Regress :math:`u` on the active-donor
   design :math:`[\,B_{\cdot,\text{active}},\,\mathbf{1}\,]` to get
   :math:`E[u]` (the ``u_missp`` step), then
   :math:`\omega_t = \mathrm{vc}\,(u_t - E[u_t])^2`. Form
   :math:`Q = B'B/T_0` and :math:`\widehat\Sigma = B'\mathrm{diag}(\omega)B/T_0^2`,
   and its matrix square root :math:`\Sigma^{1/2}`.

4. **Localised feasible set.** Lower bounds
   :math:`\ell_j = \widehat w_j` if :math:`\widehat w_j < \rho` else :math:`0`
   (near-binding donors are pinned at their tiny weight; active donors may move
   down to zero). :math:`Q` is reduced by a thresholded eigen-square-root so the
   near-null (collinear) directions are left unconstrained.

5. **In-sample simulation.** For each of ``scpi_sims`` draws
   :math:`G^\star = \Sigma^{1/2}\,z`, :math:`z\sim N(0,I)`, and each post
   predictor :math:`\mathbf{p}_T`, solve the small conic program in
   :math:`x` (donor weights) twice -- minimise and maximise
   :math:`\mathbf{p}_T'x` subject to
   :math:`(x-\widehat w)'Q(x-\widehat w) - 2G^{\star\prime}(x-\widehat w)\le 0`,
   :math:`\sum x = 1`, :math:`x\ge\ell`. Record :math:`\mathbf{p}_T'(\widehat w
   - x)` for each branch; :math:`w_L`/:math:`w_U` are the
   :math:`\alpha_1/2` / :math:`1-\alpha_1/2` quantiles across draws.

6. **Out-of-sample band.** From the location-scale model on :math:`u` get
   :math:`e_L`/:math:`e_U` per post period (Section above).

7. **Assemble.** Counterfactual band
   :math:`[\,Y_{\text{fit}} + w_L + e_L,\; Y_{\text{fit}} + w_U + e_U\,]`,
   effect interval :math:`[\,Y_{\text{obs}} - \text{cf}_U,\; Y_{\text{obs}} -
   \text{cf}_L\,]`, and an ATT interval from an appended post-period-average
   predictor row. An extra averaged row is carried through steps 5-6 so the ATT
   interval uses the same simulation, not a naive average of the per-period
   bounds.

The result is an ``InferenceResults`` with ``ci_lower``/``ci_upper`` (the ATT
interval), ``confidence_level`` :math:`= 1-2\alpha`, and a ``details`` dict
holding the per-period ``periods``, ``tau``, ``pi_lower``/``pi_upper``,
``counterfactual_lower``/``upper``, the ``in_sample_*`` (:math:`w_L,w_U`) and
``out_of_sample_*`` (:math:`e_L,e_U`) components, ``sims`` and ``e_method``.

Composing SCPI with the backends
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``backend`` (how :math:`W` is estimated) and ``inference`` (how uncertainty is
quantified) are **orthogonal** -- any of the four backends pairs with any of
the three inference modes:

.. code-block:: python

   VanillaSC({..., "backend": "mscmt", "inference": "scpi"}).fit()
   VanillaSC({..., "backend": "malo",  "inference": "scpi"}).fit()

The pipeline fits the weights with the chosen backend and hands the resulting
``res.W`` to ``scpi_intervals``. Two things to keep in mind:

* The in-sample simulation rebuilds :math:`Q` and :math:`\widehat\Sigma` from
  the donor **pre-outcomes** :math:`B`, treating :math:`\widehat w` as simplex
  weights. With **outcome-only** this is the exact Cattaneo-Feng-Titiunik
  interval (the case validated against ``scpi``). With **mscmt**/**malo** the
  weights were also shaped by the covariate predictors, so SCPI uses the
  outcome design as a stand-in -- it is **approximate** for covariate backends.
  The point effects, the ATT, and the out-of-sample band are unaffected; only
  the in-sample :math:`w_L`/:math:`w_U` term carries the approximation.
* Read the SCPI interval **alongside** :math:`\text{v\_agreement}`. When the
  predictor weights are non-identified (``v_agreement`` near 1, e.g. Prop 99
  with lagged outcomes) the *point* counterfactual is still pinned, but the
  covariate-matched solution is fragile; the placebo test, which is exact for
  any backend, is the conservative cross-check.

The leave-two-out (LTO) refined placebo test
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**What it is.** The ordinary placebo test (above) gives each of the :math:`N`
units exactly one turn as the pseudo-treated unit and ranks the real treated
unit's fit statistic against those :math:`N` values. That is its weakness: the
p-value can only land on the grid :math:`\{1/N, 2/N, \dots, 1\}`, and at a
conventional level like :math:`\alpha = 0.05` with a small donor pool the test
is either coarse or -- when :math:`\alpha < 1/N` -- literally unable to reject
(its size is zero). The Lei-Sudijono (2025) **leave-two-out** test keeps the
same design-based logic but replaces the "one turn each" permutation with a
**tournament over triples**. Think of every triple :math:`\{i, j, I\}` (two
controls and the treated unit) as a match: leave all three out of the donor
pool, build a synthetic control for each of them from the remaining
:math:`N-3` units, score each by its post/pre RMSPE ratio, and the unit with
the largest ratio "wins" the match. The treated unit *should* win often if the
treatment had a real effect (a large post-period gap relative to a tight
pre-period fit). The p-value is the fraction of matches the treated unit does
**not** win,

.. math::

   p_{\mathrm{naive\text{-}LTO}}
     = \frac{1}{(N-1)(N-2)} \sum_{i \neq j}
       \mathbf{1}\bigl\{R_{i,j,I;I} \le \max(R_{i,j,I;i}, R_{i,j,I;j})\bigr\},

where :math:`R_{i,j,I;k} = \lvert S_{\text{ratio-RMSPE}}(Y_k, \widehat Y_k)\rvert`
is the score of unit :math:`k` when the pool excludes :math:`\{i, j, I\}`.
Because there are :math:`\binom{N-1}{2}` matches rather than :math:`N`, the
p-value lives on an :math:`O(N^2)`-fine grid -- the granularity problem
disappears.

**Two p-values.** ``res.inference.p_value`` is the *naive* LTO p-value above.
``res.inference.details["p_powered"]`` is the *powered* variant
:math:`p_{\mathrm{naive\text{-}LTO}} - c(N, \alpha) + \delta`, which shifts the
naive value down by the largest amount the discrete Type-I bound allows
(``powered_offset_c``), strictly increasing power. The powered value is a
**decision rule tied to one** :math:`\alpha` -- reject when it is
:math:`\le \alpha` (``reject_at_alpha``) -- not a general-purpose p-value, so do
not compare it across levels or report it as "the" p-value.

LTO: design-based assumptions and econometric theory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The LTO test is **design-based**, not outcome-model-based: the potential
outcomes are treated as fixed, and all randomness comes from *which unit got
treated*. Its validity rests on two assumptions.

* **Uniform assignment.** The treated index :math:`I` is uniformly distributed
  over :math:`\{1, \dots, N\}` -- a priori, any unit was equally likely to be
  the treated one. Under the null this makes the :math:`N` units *exchangeable*,
  which is exactly what licenses the tournament. This holds **by construction**
  in (cluster-)randomized experiments. In observational work it is a modelling
  choice: it is most defensible when the treated unit is comparable to the
  donors (often after covariate adjustment), and in quasi-experimental settings
  -- e.g. natural disasters, where which locality is hit is plausibly close to
  random over a small comparable region.
* **Sharp null.** The hypothesis tested is Fisher's sharp null
  :math:`H_0 : Y_{i,t}(1) = Y_{i,t}(0)` for all :math:`t > T_0` (no effect for
  *any* unit in any post period), or a known-:math:`\tau` additive version
  :math:`Y_{i,t}(1) = Y_{i,t}(0) + \tau_{i,t}`. Sharpness is what lets the test
  impute every unit's counterfactual under the null and so run the tournament.

Under these, the test has a **finite-sample Type-I error guarantee** (no
large-:math:`N`, no long-:math:`T`, no asymptotics):

.. math::

   \mathbb{P}_{H_0}\!\left(p_{\mathrm{naive\text{-}LTO}} \le \alpha\right)
     \le \frac{\lfloor N f(N, \alpha)\rfloor}{N},

reported as ``type_i_bound``. This bound is **never worse** than the
approximate-placebo bound :math:`(\lfloor N\alpha\rfloor + 1)/N`, and for the
levels and sizes typical of SCM applications (:math:`\alpha \in \{0.01, 0.02\}`
for :math:`6 < N < 200`; :math:`\alpha = 0.05` for most :math:`N`) it is
*identical* to it -- so switching to LTO costs nothing in worst-case Type-I
error. Crucially, the placebo bound is *tight* whereas the LTO bound generally
is **not**: in practice the LTO test's actual Type-I error is often strictly
below :math:`\alpha`, i.e. it can be **unconditionally valid** even when
:math:`\alpha < 1/N`.

Two further theoretical properties matter in practice:

* **Consistency where the placebo test fails (Theorem 6.1).** When
  :math:`\alpha < 1/N`, the LTO test is *uniformly consistent* -- its power goes
  to 1 as the effect size grows. The approximate placebo test is **not**: in
  this regime it can have essentially zero power no matter how large the true
  effect (zero if :math:`N` is even, :math:`\le 1/N` if odd). This is the single
  strongest reason to prefer LTO in small donor pools.
* **Confidence regions.** Inverting the additive-:math:`\tau` test
  (:math:`\{\theta : p_{\mathrm{naive\text{-}LTO}}(\theta) > \alpha\}`) yields a
  region for the post-period effect path with guaranteed coverage
  :math:`\ge 1 - \lfloor N f(N,\alpha)\rfloor / N`. (mlsynth currently reports
  the p-values; the inversion is a straightforward extension.)

Methodologically, the LTO test is a *new* kind of randomization inference: it
generalises the Jackknife+ of Barber et al. (2021) (which leaves **one** point
out and so still has :math:`1/N` granularity) and is distinct from classical
permutation/rank inference. It also -- unlike most asymptotic SCM inference --
does **not** simplify the synthetic-control construction: the full
weight/predictor machinery (any ``VanillaSC`` backend) is re-run inside every
match, so the test reflects the estimator you actually use.

When the LTO assumptions are violated
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The sharp null is testable and usually uncontroversial; the **uniform
assignment** assumption is where care is needed.

* **Selection on outcomes / non-comparable treated unit.** If the treated unit
  was chosen *because* of its (anticipated) trajectory, or is structurally
  unlike every donor, exchangeability fails and the Type-I guarantee no longer
  holds. The usual remedy is to restore comparability through the
  specification -- match on covariates, restrict the donor pool to genuinely
  similar units -- before trusting any placebo-type p-value.
* **Known non-uniform assignment.** When the treatment probabilities
  :math:`\pi_k` are known or estimable (e.g. seismic risk for an
  earthquake study), Lei-Sudijono give a *weighted* LTO p-value
  :math:`p_{\text{w-LTO}}(\pi)` that reweights each match by
  :math:`\pi_j\pi_k / ((1-\pi_I)^2 - \sum_{l\neq I}\pi_l^2)` and reduces to the
  naive value when :math:`\pi_i \equiv 1/N`.
* **Sensitivity analysis (the** :math:`\Gamma` **).** Rather than commit to
  uniformity, one can ask *how far* from it the design could be before the
  conclusion flips. Following Rosenbaum, constrain
  :math:`\pi_i \in [\tfrac{1}{\Gamma N}, \tfrac{\Gamma}{N}]` and find the
  smallest :math:`\Gamma \ge 1` at which the worst-case weighted p-value crosses
  :math:`\alpha`. In the paper, Prop 99 tolerates :math:`\Gamma \approx 1.4`
  (robust) while German reunification flips at only :math:`\Gamma \approx 1.1`
  (fragile). The weighted p-value and :math:`\Gamma` search require solving a
  non-convex (NP-hard) quadratic program and are **not yet implemented** in
  ``VanillaSC``; the uniform-assignment naive/powered p-values are.

Choosing among placebo, LTO, and SCPI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Prefer **LTO over the ordinary placebo** whenever the donor pool is small --
  especially in the :math:`\alpha < 1/N` regime (e.g. :math:`N \le 20` at
  :math:`\alpha = 0.05`), where the placebo test cannot reject and LTO can. The
  ``powered`` variant almost Pareto-improves the placebo: same worst-case
  Type-I error, more power. Both share the *same assumptions*, so LTO is close
  to a free upgrade.
* Keep the **ordinary placebo** when you want the most familiar, widely-reported
  statistic, when :math:`N` is large enough that granularity is a non-issue, or
  as a cheap (:math:`O(N)` vs :math:`O(N^2)`) cross-check.
* Reach for **SCPI** when the question is *how big* the effect is (a prediction
  interval / confidence statement on the magnitude), not just *whether* there is
  one. SCPI rests on different (model-based, conditional) foundations than the
  design-based placebo/LTO tests, so the two are complementary: LTO answers
  "is the treated unit special?" by randomization, SCPI quantifies the effect's
  uncertainty.

When to use it
--------------

* You want the **standard synthetic control** done reliably, with the
  solver choice and identification fragility surfaced.
* **Outcome-only** matching when you have a long, informative pre-period
  -- this is the well-posed, reproducible case.
* **Covariate** matching with ``mscmt`` when the donor pool is rich
  enough that the problem is well-conditioned (see the replications
  below). When :math:`\text{v\_agreement}` comes back near 1, prefer
  outcome-only or ``penalized``.

Empirical replications
----------------------

The three canonical studies, each trained on its **full pre-treatment
period**. All run from the datasets shipped under ``basedata/``. These are
locked as regression tests in
``mlsynth/tests/test_vanillasc_replications.py``.

California / Proposition 99 (ADH 2010)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Treatment in 1989; pre-period 1970-1988. Covariates averaged over
1980-1988 (beer 1984-1988) plus three lagged cigarette-sales predictors
(1975, 1980, 1988). With ``mscmt`` this reproduces ADH Table 2 almost
exactly -- Utah 0.335, Nevada 0.236, Montana 0.202, Colorado 0.160,
Connecticut 0.068 (ADH: 0.334 / 0.234 / 0.199 / 0.164 / 0.069) -- and an
ATT of about :math:`-19` packs.

.. code-block:: python

   import pandas as pd
   from mlsynth import VanillaSC

   d = pd.read_csv("basedata/augmented_cali_long.csv")
   for yr, col in [(1975, "cig_1975"), (1980, "cig_1980"), (1988, "cig_1988")]:
       d[col] = d.state.map(d[d.year == yr].set_index("state").cigsale)
   d["treated"] = ((d.state == "California") & (d.year >= 1989)).astype(int)

   res = VanillaSC({
       "df": d, "outcome": "cigsale", "treat": "treated",
       "unitid": "state", "time": "year",
       "backend": "mscmt", "canonical_v": "min.loss.w", "seed": 1,
       "covariates": ["p_cig", "pct15-24", "loginc", "pc_beer",
                      "cig_1975", "cig_1980", "cig_1988"],
       "covariate_windows": {"p_cig": (1980, 1988), "pct15-24": (1980, 1988),
                             "loginc": (1980, 1988), "pc_beer": (1984, 1988)},
       "display_graphs": False,
   }).fit()
   print(res.effects.att)                 # ~ -19
   print(res.weights.donor_weights)       # Utah/Nevada/Montana/Colorado/Connecticut

(In ``augmented_cali_long.csv`` the columns are labelled such that
``p_cig`` is log GDP per capita and ``loginc`` is the retail price -- the
predictor means reproduce ADH's Table 1 "Real California" column.)

German reunification (ADH 2015)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Treatment (reunification) in 1990; pre-period 1960-1990. GDP, trade,
inflation and industry share averaged over 1981-1990; investment rate and
schooling over 1980-1985. With ``mscmt`` the synthetic West Germany is
Austria-dominant with the USA, Switzerland, Japan and the Netherlands --
the ADH 2015 set -- and a negative ATT (reunification lowered per-capita
GDP relative to the synthetic).

.. code-block:: python

   import pandas as pd
   from mlsynth import VanillaSC

   d = pd.read_stata("basedata/repgermany.dta")
   d["treated"] = ((d.country == "West Germany") & (d.year >= 1990)).astype(int)

   res = VanillaSC({
       "df": d, "outcome": "gdp", "treat": "treated",
       "unitid": "country", "time": "year",
       "backend": "mscmt", "seed": 1,
       "covariates": ["gdp", "trade", "infrate", "industry", "invest80", "schooling"],
       "covariate_windows": {"gdp": (1981, 1990), "trade": (1981, 1990),
                             "infrate": (1981, 1990), "industry": (1981, 1990),
                             "invest80": (1980, 1980), "schooling": (1980, 1985)},
       "display_graphs": False,
   }).fit()
   print(res.weights.donor_weights)       # Austria/USA/Switzerland/Japan/Netherlands

Basque terrorism (Abadie-Gardeazabal 2003)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The treatment indicator (terrorism) first turns on in **1975**, so the
model trains on the full **1955-1974** pre-period. On this long pre-period
the problem is well-conditioned and the synthetic Basque is **Cataluna
:math:`\approx 0.8`, Madrid :math:`\approx 0.2`** -- the published
Abadie-Gardeazabal result -- with an ATT of about :math:`-0.68` (the
roughly 10% per-capita GDP gap). Outcome-only already recovers this;
``mscmt`` with the special-predictor covariates confirms it.

.. note::

   This is instructive: on the *short* 1960-1969 window used by some later
   papers the Basque donor weights are fragile (they drift to
   Baleares/Madrid), but on the full 1955-1974 pre-period the long outcome
   path pins :math:`W` to the Cataluna/Madrid solution. The training
   window matters; ``VanillaSC`` uses the full pre-period defined by the
   treatment indicator.

.. code-block:: python

   import pandas as pd
   from mlsynth import VanillaSC

   b = pd.read_csv("basedata/basque_data.csv")
   b = b[b.regionno != 1]                                  # drop Spain
   b["treated"] = ((b.regionno == 17) & (b.year >= 1975)).astype(int)

   res = VanillaSC({
       "df": b, "outcome": "gdpcap", "treat": "treated",
       "unitid": "regionno", "time": "year",
       "backend": "outcome-only", "display_graphs": False,
   }).fit()
   print(res.effects.att)                 # ~ -0.68
   print(res.weights.donor_weights)       # region 10 (Cataluna) ~0.8, 14 (Madrid) ~0.2

Core API
--------

.. automodule:: mlsynth.estimators.vanillasc
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.utils.vanillasc_helpers.config.VanillaSCConfig
   :members:
   :undoc-members:

Engine
------

.. automodule:: mlsynth.utils.vanillasc_helpers.engine
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.vanillasc_helpers.pipeline
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.vanillasc_helpers.scpi
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.vanillasc_helpers.lto
   :members:
   :undoc-members:

SCPI prediction intervals
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To request Cattaneo-Feng-Titiunik prediction intervals instead of the placebo
test, set ``inference="scpi"``. On Prop 99 (outcome-only) this yields an ATT
around :math:`-19` with a 90% prediction interval that excludes zero, and
per-period intervals that widen as the post-period extends.

.. code-block:: python

   import pandas as pd
   from mlsynth import VanillaSC

   d = pd.read_csv("basedata/augmented_cali_long.csv")
   d["treated"] = ((d.state == "California") & (d.year >= 1989)).astype(int)

   res = VanillaSC({
       "df": d[["state", "year", "cigsale", "treated"]],
       "outcome": "cigsale", "treat": "treated", "unitid": "state", "time": "year",
       "backend": "outcome-only", "inference": "scpi", "alpha": 0.05,
       "scpi_sims": 200, "display_graphs": False,
   }).fit()

   print(res.inference.ci_lower, res.inference.ci_upper)   # ATT prediction interval
   det = res.inference.details                              # per-period sequence
   for yr, lo, up in zip(det["periods"], det["pi_lower"], det["pi_upper"]):
       print(yr, round(lo, 1), round(up, 1))

SCPI with the covariate backends (MSCMT and Malo)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The same ``inference="scpi"`` switch composes with the covariate-matching
backends. Running each of the three canonical studies under both ``mscmt`` and
``malo`` (``alpha=0.05`` -> 90% intervals, ``scpi_sims=200``, ``seed=1``) gives
the table below. The ATT prediction interval **excludes zero in every case**,
and the two backends agree to within Monte-Carlo / weight-choice differences --
a useful robustness cross-check. Note the ``v_agreement`` column: for Prop 99
and Germany under ``mscmt`` the predictor weights are non-identified
(:math:`\approx 1`), so those intervals should be read with the caveat above.

.. list-table::
   :header-rows: 1
   :widths: 22 12 22 12 16

   * - Study (backend)
     - ATT
     - ATT 90% PI
     - v_agreement
     - top donors
   * - California (mscmt)
     - :math:`-18.98`
     - :math:`[-27.31,\,-5.28]`
     - :math:`\approx 1` (fragile)
     - Utah .34, Nevada .24, Montana .20
   * - California (malo)
     - :math:`-19.60`
     - :math:`[-31.32,\,-3.27]`
     - n/a
     - Utah .38, Montana .25, Nevada .21
   * - Germany (mscmt)
     - :math:`-1396`
     - :math:`[-2368,\,-949]`
     - :math:`\approx 1` (fragile)
     - Austria .40, Switz .16, USA .15
   * - Germany (malo)
     - :math:`-1306`
     - :math:`[-2025,\,-521]`
     - n/a
     - USA .35, Austria .33, Switz .11
   * - Basque (mscmt)
     - :math:`-0.70`
     - :math:`[-1.13,\,-0.32]`
     - :math:`0.63`
     - Cataluna .84, Madrid .16
   * - Basque (malo)
     - :math:`-0.63`
     - :math:`[-1.14,\,-0.18]`
     - :math:`\approx 0` (clean)
     - Cataluna .47, Madrid .33

The Basque case is the cleanest: with the special-predictor covariates,
``malo`` returns a well-identified :math:`V` (``v_agreement`` :math:`\approx 0`)
and ``mscmt`` recovers the published Cataluna/Madrid split, both with tight
intervals that exclude zero. The early German post-years (1990-1992) are *not*
significant under either backend -- the interval includes zero -- and only turn
decisively negative later, exactly as the reunification narrative implies.

.. code-block:: python

   import pandas as pd
   from mlsynth import VanillaSC

   # --- California / Prop 99 (ADH 2010) ---
   d = pd.read_csv("basedata/augmented_cali_long.csv")
   for yr, col in [(1975, "cig_1975"), (1980, "cig_1980"), (1988, "cig_1988")]:
       d[col] = d.state.map(d[d.year == yr].set_index("state").cigsale)
   d["treated"] = ((d.state == "California") & (d.year >= 1989)).astype(int)
   cov = ["p_cig", "pct15-24", "loginc", "pc_beer", "cig_1975", "cig_1980", "cig_1988"]
   win = {"p_cig": (1980, 1988), "pct15-24": (1980, 1988),
          "loginc": (1980, 1988), "pc_beer": (1984, 1988)}
   common = dict(df=d, outcome="cigsale", treat="treated", unitid="state", time="year",
                 covariates=cov, covariate_windows=win, inference="scpi",
                 alpha=0.05, scpi_sims=200, seed=1, display_graphs=False)

   mscmt = VanillaSC({**common, "backend": "mscmt", "canonical_v": "min.loss.w"}).fit()
   malo  = VanillaSC({**common, "backend": "malo"}).fit()
   for name, r in [("mscmt", mscmt), ("malo", malo)]:
       i = r.inference
       print(name, round(r.effects.att, 2), (round(i.ci_lower, 2), round(i.ci_upper, 2)),
             "v_agreement=", r.weights.summary_stats.get("v_agreement"))

   # --- German reunification (ADH 2015): outcome "gdp", same pattern ---
   # --- Basque (AG 2003): outcome "gdpcap", special-predictor covariates ---
   # (swap df/outcome/covariates; everything else is identical.)

The per-period sequence is always in ``res.inference.details``; switching
backend changes :math:`\widehat w` (and hence the centre and width of the band)
but not the inference code path.

Leave-two-out refined placebo test
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set ``inference="lto"`` for the Lei-Sudijono (2025) refined placebo test. It is
a drop-in replacement for the ordinary placebo with a much finer p-value grid
and valid rejections when :math:`\alpha < 1/N`.

.. code-block:: python

   import pandas as pd
   from mlsynth import VanillaSC

   d = pd.read_csv("basedata/augmented_cali_long.csv")
   d["treated"] = ((d.state == "California") & (d.year >= 1989)).astype(int)

   res = VanillaSC({
       "df": d[["state", "year", "cigsale", "treated"]],
       "outcome": "cigsale", "treat": "treated", "unitid": "state", "time": "year",
       "backend": "outcome-only", "inference": "lto", "alpha": 0.05,
       "display_graphs": False,
   }).fit()

   det = res.inference.details
   print(res.inference.p_value)        # naive LTO p-value (703 pairs for N = 39)
   print(det["p_powered"], det["powered_offset_c"])   # powered p-value at alpha
   print(det["type_i_bound"], det["reject_at_alpha"])

Empirical relations across the three studies
""""""""""""""""""""""""""""""""""""""""""""

Lei-Sudijono's Table 1 (their covariate-matched Synth specification,
:math:`\alpha = 0.05`) lays out how the methods relate on the canonical
datasets:

.. list-table::
   :header-rows: 1
   :widths: 30 14 14 14

   * - quantity
     - Prop 99
     - Basque
     - German
   * - :math:`N`
     - 39
     - 17
     - 17
   * - :math:`p_{\text{app-placebo}}`
     - 0.00
     - 0.35
     - 0.00
   * - :math:`p_{\text{exact-placebo}}`
     - 0.026
     - 0.41
     - 0.059
   * - :math:`p_{\mathrm{naive\text{-}LTO}}`
     - 0.024
     - 0.67
     - 0.042
   * - :math:`p_{\mathrm{powered\text{-}LTO}}(\alpha)`
     - 0.022
     - 0.66
     - 0.03
   * - :math:`\Gamma_{\mathrm{LTO}}`
     - 1.4
     - NA
     - 1.1

Three relations are worth internalising:

* **LTO can change the conclusion (German).** The exact placebo p-value of
  0.059 does *not* reject at 0.05, but the naive LTO (0.042) and powered LTO
  (0.03) both do. With only 16 donors the placebo grid is too coarse to resolve
  a borderline effect; LTO's finer grid does. The small :math:`\Gamma = 1.1`,
  though, warns that this significance is fragile to mild departures from
  uniform assignment.
* **LTO is not mechanically smaller (Basque).** Here LTO (0.67) is *larger*
  than the placebo (0.41); nothing is significant by any method. The refinement
  changes granularity, not direction -- it does not manufacture significance.
  (Abadie-Gardeazabal's original Basque analysis dropped poorly-fitting
  regions; LTO makes no such adjustment, which partly explains the larger
  value.)
* **LTO ≈ placebo when both already reject (Prop 99).** The two p-values
  (0.024 vs 0.026) nearly coincide; the powered version (0.022) buys a little
  extra margin, and :math:`\Gamma = 1.4` says the conclusion survives moderate
  confounding.

A note on specification and validation. The LTO p-value, like the ordinary
placebo, is only as good as the synthetic-control fit it is built on. In
mlsynth the ``f(N, \alpha)`` and powered-offset ``c(N, \alpha)`` functions
reproduce the paper's reported values exactly (``c(39, 0.05) = 0.002``,
``c(17, 0.05) = 0.0125``; see ``test_lto_helpers_match_paper``), and the
covariate-matched ordinary placebo reproduces California's exact-placebo
p-value (rank 1 of 39, :math:`p = 0.0256` vs the paper's 0.026). The *p-value
itself* tracks the chosen specification: with the paper's covariate-matched
Synth, California is the dominant unit and :math:`p_{\mathrm{naive\text{-}LTO}}
\approx 0.024`, whereas the ``outcome-only`` fit -- where California is only
rank 3 of 39 -- gives :math:`\approx 0.10`. Both are internally consistent with
their respective ordinary placebo p-values; the covariate spec is what
concentrates the effect on the treated unit, so choose the specification before
reading the test. Because the cost is :math:`O(J^2)` engine fits, run the
covariate-matched (``mscmt``) version on the smaller studies or cap pairs with
``lto_max_pairs``; for the 38-donor Prop 99 panel the ``outcome-only`` LTO runs
in well under two minutes.

References
----------

Abadie, A., & Gardeazabal, J. (2003). "The Economic Costs of Conflict: A
Case Study of the Basque Country." *American Economic Review* 93(1):113-132.

Abadie, A., Diamond, A., & Hainmueller, J. (2010). "Synthetic Control
Methods for Comparative Case Studies." *Journal of the American
Statistical Association* 105(490):493-505.

Abadie, A., Diamond, A., & Hainmueller, J. (2015). "Comparative Politics
and the Synthetic Control Method." *American Journal of Political Science*
59(2):495-510.

Abadie, A., & L'Hour, J. (2021). "A Penalized Synthetic Control Estimator
for Disaggregated Data." *Journal of the American Statistical Association*
116(536):1817-1834.

Becker, M., & Kloessner, S. (2018). "Fast and Reliable Computation of
Generalized Synthetic Controls." *Econometrics and Statistics* 5:1-19.

Lei, L., & Sudijono, T. (2025). "Inference for Synthetic Controls via
Refined Placebo Tests." *arXiv:2401.07152*.

Malo, P., Eskelinen, J., Zhou, X., & Kuosmanen, T. (2024). "Computing
Synthetic Controls Using Bilevel Optimization." *Computational Economics*
64:1113-1136.
