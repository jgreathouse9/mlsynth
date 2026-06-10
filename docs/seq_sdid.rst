Sequential Synthetic Difference-in-Differences (Sequential SDiD)
================================================================

.. currentmodule:: mlsynth

Overview
--------

Sequential Synthetic Difference-in-Differences (Sequential SDiD,
`arXiv:2404.00164v2 <https://arxiv.org/abs/2404.00164v2>`_, Arkhangelsky &
Samkov, 2025) is an event-study estimator for staggered-adoption designs
that remains robust when the parallel-trends assumption fails. It adapts
the canonical SDiD of Arkhangelsky et al. (2021) by operating on
*cohort-level aggregates* and *sequentially* imputing treated cells with
their estimated counterfactuals so that bias from early cohorts does not
cascade into later ones.

The estimator is asymptotically equivalent to an infeasible oracle OLS
regression that knows the unobserved interactive fixed effects (Proposition
3.1 of the paper), giving it the first formal efficiency guarantees for an
SC-type method. Five structural differences distinguish Sequential SDiD
from the canonical SDID estimator already in :mod:`mlsynth`:

* It works on **aggregated cohort outcomes** :math:`Y_{a, t} =
  n_a^{-1} \sum_{i:\,A_i = a} Y_{i, t}` rather than unit-level data, with
  cohort shares :math:`\pi_a = n_a / n` carrying the unit-count information.
* Weights satisfy only the simplex sum constraint :math:`\sum \omega = 1`
  — non-negativity is dropped.
* The unit-weight penalty is the population-share-scaled
  :math:`\eta^{2} \sum_j \omega_j^2 / \pi_j`; the time-weight penalty is
  :math:`\eta^{2} \sum_l \lambda_l^2`.
* Donors for cohort :math:`a` are restricted to **later-adopting cohorts**
  :math:`j > a` (including the never-treated cohort), not the universe of
  controls.
* Cohort-by-horizon effects are estimated in a **sequential cascade**:
  each :math:`\hat\tau_{a, k}^{\,SSDiD}` is computed, then the treated cell
  :math:`Y_{a, a + k}` is overwritten with :math:`Y_{a, a + k} - \hat
  \tau_{a, k}^{\,SSDiD}` so subsequent ``(a', k')`` steps see an imputed
  panel free of treatment contamination.

When to Use This Estimator
--------------------------

Event studies with **staggered adoption** -- units switching on at
different dates -- are the workhorse design of applied micro. The modern
estimators built for them (Callaway & Sant'Anna 2020, de Chaisemartin &
d'Haultfœuille 2020, Sun & Abraham 2020, Borusyak et al. 2024) fixed the
*heterogeneous-effects* bug in two-way fixed effects, but they still rest
on **parallel trends**: absent treatment, treated and comparison cohorts
would have moved together. When unobserved **interactive fixed effects**
(latent unit factors loading on latent time factors) drive selection into
treatment, parallel trends fails and those estimators are biased --
exactly the regime Sequential SDiD (Arkhangelsky & Samkov 2025) targets.

Its bet is to *model* the confounder. Within a linear interactive-fixed-
effects model, Sequential SDiD is proven **asymptotically equivalent to
the infeasible oracle OLS** that knows the latent factors (Prop. 3.1).
That equivalence buys three things at once: robustness to parallel-trends
violations of the IFE type, **asymptotic normality with standard
inference**, and the first formal **efficiency** guarantee for an SC-type
estimator. The sequential imputation -- estimate the earliest cohort,
subtract its effect, reuse the cleaned panel for later cohorts -- gives a
*cohesive* analysis of the whole event study, in contrast to per-cohort SC
methods (:doc:`ppscm`, Cattaneo et al. 2021) that treat each adoption
cohort as a separate problem.

Reach for Sequential SDiD when
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* You have a **staggered-adoption event study** and you suspect
  **parallel trends fails** because of unobserved confounders that look
  like interactive fixed effects (selection on latent trends, not just
  levels).
* Adoption **cohorts are reasonably large.** The method averages outcomes
  within each cohort and leans on a law of large numbers to kill
  idiosyncratic noise; this is the estimator's core requirement and its
  main limitation.
* You want **valid, standard inference** (asymptotic normality) and a
  defensible **efficiency** claim, plus a dynamic **event-study path** of
  cohort-by-horizon effects.
* You are in the **fixed-:math:`T`, large-:math:`N`** regime typical of
  county/firm/individual panels aggregated into a handful of adoption
  cohorts.

Do not use Sequential SDiD when
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Cohorts are small** (few units per adoption date, or single-unit
  cohorts). The within-cohort averaging has nothing to average and the
  oracle equivalence does not engage. Use :doc:`ppscm`, whose partial
  pooling is designed for noisy per-unit cohort fits, or canonical
  :doc:`sdid` per cohort.
* **There is a single treated unit or one common adoption time.** The
  sequential cascade has nothing to cascade; use canonical :doc:`sdid`
  (many units, one time) or classic SC (:doc:`tssc`, :doc:`fdid`,
  :doc:`scmo`) for a single unit.
* **Parallel trends is credible** and you want the simplest transparent
  thing. Standard staggered-DiD (Callaway-Sant'Anna and relatives) is
  fine; on the Bailey-Goodman-Bacon application where PT holds, Sequential
  SDiD merely reproduces standard DiD.
* **SUTVA is violated by spillovers** onto the comparison cohorts -- use
  :doc:`spsydid` or :doc:`spillsynth`.
* **Distributional** questions (quantiles, tails) -- use :doc:`dsc`.

Mathematical Formulation
------------------------

Setup
^^^^^

Let :math:`Y_{i, t}` denote the outcome of unit :math:`i` at period
:math:`t \in \{1, \dots, T\}`, and :math:`A_i` the (possibly infinite)
adoption period of unit :math:`i`. Cohorts are indexed by adoption period
:math:`a`; the never-treated cohort uses :math:`A_i = +\infty`. For each
cohort the implementation computes

.. math::

   Y_{a, t}
   \;=\;
   \frac{1}{n_a} \sum_{i:\, A_i = a} Y_{i, t},
   \qquad
   \pi_a = \frac{n_a}{n}.

Under Assumption 2.2 of the paper, the aggregate outcomes inherit the
interactive-fixed-effects structure

.. math::

   Y_{a, t}
   \;=\;
   \alpha_a + \beta_t + \theta_a^\top \psi_t
   + \mathbb{1}\{a \le t\}\,\tau_{a, t - a} + \epsilon_{a, t},

where :math:`\theta_a^\top \psi_t` captures unobserved confounders that
break parallel trends. Aggregation drives :math:`\epsilon_{a, t}` to zero
under the paper's "large cohort" asymptotics, leaving the IFE structure
identifiable.

Algorithm 1
^^^^^^^^^^^

For each horizon :math:`k = 0, 1, \dots, K` (outer loop) and each treated
cohort :math:`a = a_{\min}, \dots, a_{\max}` (inner loop), Sequential SDiD
runs three steps.

**Step 1: Solve two regularized QPs.** Both QPs are equality-constrained
convex quadratic programs (no non-negativity). The unit-weight QP is

.. math::

   \hat\omega^{(a, k)}, \hat\omega_0
   \;=\;
   \arg\min_{\sum_{j > a} \omega_j = 1}
   \quad
   \sum_{l < a + k}
     \left(
       \omega_0
       + \sum_{j > a} \omega_j Y_{j, l}
       - Y_{a, l}
     \right)^{\!2}
   \;+\;
   \eta^2 \sum_{j > a} \frac{\omega_j^2}{\pi_j},

and the time-weight QP is

.. math::

   \hat\lambda^{(a, k)}, \hat\lambda_0
   \;=\;
   \arg\min_{\sum_{l < a + k} \lambda_l = 1}
   \quad
   \sum_{j > a}
     \left(
       \lambda_0
       + \sum_{l < a + k} \lambda_l Y_{j, l}
       - Y_{j, a + k}
     \right)^{\!2}
   \;+\;
   \eta^2 \sum_{l < a + k} \lambda_l^{\,2}.

Both are solved in closed form via their KKT linear systems in
:mod:`mlsynth.utils.seq_sdid_helpers.weights`.

**Step 2: Weighted double-difference.**

.. math::

   \hat\tau_{a, k}^{\,SSDiD}
   \;=\;
   \left(
     Y_{a, a + k}
     -
     \sum_{j > a} \hat\omega_j^{(a, k)} \, Y_{j, a + k}
   \right)
   \;-\;
   \sum_{l < a + k}
     \hat\lambda_l^{(a, k)}
     \left(
       Y_{a, l}
       -
       \sum_{j > a} \hat\omega_j^{(a, k)} \, Y_{j, l}
     \right).

This is the same SDID-style contrast as the canonical estimator, just
evaluated on cohort-level aggregates.

**Step 3: Sequential imputation.**

.. math::

   Y_{a, a + k} \;:=\; Y_{a, a + k} \;-\; \hat\tau_{a, k}^{\,SSDiD}.

The treated cell is replaced with its estimated counterfactual *in place*
in the panel matrix. When the outer loop advances to a longer horizon or
the inner loop advances to a later cohort, subsequent QPs use this
imputed panel — which is the mechanism that prevents bias from cascading
through the estimator.

Pooled Event Study
^^^^^^^^^^^^^^^^^^

Cohort-specific effects are aggregated into a single event-study trajectory
via Equation 2.5 of the paper:

.. math::

   \hat\tau_k^{\,SSDiD}(\mu)
   \;=\;
   \sum_{a \in [a_{\min}, a_{\max}]} \mu_a \hat\tau_{a, k}^{\,SSDiD},
   \qquad
   \mu_a = \frac{\pi_a}{\sum_{a' \in [a_{\min}, a_{\max}]} \pi_{a'}}.

The default ``mu`` is proportional to cohort shares (i.e. larger cohorts
get more weight), recovering the unit-uniform interpretation common in the
DiD literature. The result lives on :py:attr:`SeqSDIDEventStudy.tau`.

The :math:`\eta \to \infty` Limit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Remark 2.2 of the paper notes that as :math:`\eta \to \infty`, the unit-
weight QP's penalty :math:`\sum \omega_j^2 / \pi_j` forces :math:`\omega_j
\propto \pi_j` (each unit in the donor pool gets equal weight), and the
time-weight QP's penalty :math:`\sum \lambda_l^{\,2}` forces
:math:`\lambda_l = 1 / (a + k - 1)` (uniform). The resulting estimator is
a sequential DiD imputation estimator closely related to Borusyak,
Jaravel, and Spiess (2024). The :mod:`mlsynth` implementation exposes this
mode via :attr:`SequentialSDIDConfig.mode`:

* ``mode = "ssdid"``: paper's main estimator with finite ``eta`` (default).
* ``mode = "sdid_imputation"``: forces the :math:`\eta \to \infty` limit
  internally and returns the sequential-DiD-style result.

Inference (Section 2.3)
^^^^^^^^^^^^^^^^^^^^^^^

Inference uses the Bayesian bootstrap of Rubin (1981) and Chamberlain &
Imbens (2003). At each bootstrap iteration:

1. Draw independent weights :math:`\xi_i \sim \mathrm{Exp}(1)` for every
   underlying unit (not cohort).
2. Reconstruct cohort-level outcomes as weighted means:

   .. math::

      Y_{a, t}(\xi)
      \;=\;
      \frac{\sum_{i:\, A_i = a} Y_{i, t} \xi_i}{\sum_{i:\, A_i = a} \xi_i}.

3. Re-run Algorithm 1 on the perturbed panel.
4. Record the pooled event-study vector.

Standard errors are sample standard deviations of the bootstrap replicates,
and confidence intervals are Wald-type at :py:attr:`SequentialSDIDConfig.alpha`.
The full replicate matrix is retained on
:py:attr:`SeqSDIDEventStudy.bootstrap_draws` in case quantile-based
intervals are preferred downstream.

Limitations
^^^^^^^^^^^

The paper's formal guarantees require *large adoption cohorts* — cohort
sizes that grow with the sample so the aggregation kills the idiosyncratic
noise. The algorithm still runs on single-treated-unit panels (e.g., the
Proposition 99 dataset), but with only one treated cohort and one
never-treated cohort the time-weight QP becomes effectively
underdetermined; the practical recommendation is to use canonical
:class:`SDID` for those panels and reserve Sequential SDiD for genuine
staggered designs with multiple sizable cohorts.

A second requirement is *donor balance*. Each treated cohort :math:`a` is
balanced only against the cohorts adopting after it (:math:`j > a`) plus any
never-treated cohort, so it needs at least **two** such donor cohorts to
balance even a rank-one interactive fixed effect — matching a one-dimensional
loading and the intercept spans :math:`(1, \lambda)`. With a single donor the
sum-to-one constraint forces :math:`\omega = [1]` and the cohort effect
collapses to an unbalanced DiD, biased whenever the loadings differ; that bias
also cascades backward through the sequential imputation. On a *noiseless*
rank-one factor the estimator recovers the effect **exactly** for every cohort
that is balanced, and the entire residual bias is attributable to the
donor-starved late cohorts. Because the latest cohorts are the most exposed
(the default ``a_max`` is the last adopting cohort), :meth:`SequentialSDID.fit`
emits a :class:`UserWarning` naming any donor-starved cohort and the largest
``a_max`` that keeps every estimated cohort balanced — report, don't silently
relax.

Verification
------------

**Path B** — the paper's Section-5.2.2 calibrated-panel Monte Carlo (Table 1)
is reconstructed from its description (the authors' CPS log-wage panel is not
public). Under an interactive-fixed-effects violation of parallel trends with
adoption correlated to the leading loading, standard DiD's 95% CI coverage
collapses (:math:`\approx 0.45`; paper :math:`\approx 0.70`) while Sequential
SDiD stays near nominal (:math:`0.945`) with roughly five-times-smaller bias
and lower RMSE — the paper's "DiD severely biased, Sequential SDiD reliable"
result. A noiseless rank-one IFE corollary pins exact machine-precision
recovery for every donor-balanced cohort. The durable check is
``benchmarks/cases/seq_sdid_mc.py``; see the dedicated replication page,
:doc:`replications/seq_sdid`, for the design, code, and the full table.

Core API
--------

.. automodule:: mlsynth.estimators.seq_sdid
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.SequentialSDIDConfig
   :members:
   :undoc-members:

Helper Modules
--------------

.. automodule:: mlsynth.utils.seq_sdid_helpers.setup
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.seq_sdid_helpers.weights
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.seq_sdid_helpers.algorithm
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.seq_sdid_helpers.inference
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.seq_sdid_helpers.plotter
   :members:
   :undoc-members:

.. note::

   ``SequentialSDID.fit()`` returns an
   :class:`~mlsynth.config_models.EffectResult` on the standardized two-family
   contract. Sequential SDiD is an event-study estimator, so its standardized
   ``time_series`` is laid out over *event-time horizons* (``res.gap`` is the
   pooled horizon effect ``tau_hat_k``) and ``res.att`` is the simple average
   of those pooled effects. The per-(cohort, horizon) effects, the pooled event
   study, and the bootstrap config stay on ``res.cohort_effects`` /
   ``res.event_study`` / ``res.inference_detail``.

.. automodule:: mlsynth.utils.seq_sdid_helpers.structures
   :members:
   :undoc-members:

Example
-------

.. code-block:: python

   import pandas as pd
   from mlsynth import SequentialSDID

   df = pd.read_csv("staggered_panel.csv")  # state-level panel with treat in {0, 1}

   results = SequentialSDID({
       "df":           df,
       "outcome":      "log_wage",
       "treat":        "treated",
       "unitid":       "state",
       "time":         "year",
       "eta":          1.0,
       "K":            10,
       "n_bootstrap":  500,
       "alpha":        0.05,
       "display_graphs": True,
   }).fit()

   # Pooled event-study trajectory (Eq. 2.5 of the paper).
   for k, tau, se in zip(results.event_study.horizons,
                         results.event_study.tau,
                         results.event_study.se):
       print(f"k = {k:>2}  tau = {tau:+.3f}  se = {se:.3f}")

   # Cohort-by-horizon decomposition.
   for (a, k), effect in results.cohort_effects.items():
       print(f"cohort a = {a}  horizon k = {k}  tau = {effect.tau:+.3f}")

   # Bayesian-bootstrap replicate matrix (B x (K + 1)) — for quantile CIs
   # or downstream diagnostics.
   results.event_study.bootstrap_draws.shape

References
----------

Arkhangelsky, D., & Samkov, A. (2025). "Sequential Synthetic Difference
in Differences." `arXiv:2404.00164v2 <https://arxiv.org/abs/2404.00164v2>`_.

Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S.
(2021). "Synthetic Difference-in-Differences." *American Economic Review*
111(12): 4088-4118.

Borusyak, K., Jaravel, X., & Spiess, J. (2024). "Revisiting Event-Study
Designs: Robust and Efficient Estimation." *Review of Economic Studies.*
