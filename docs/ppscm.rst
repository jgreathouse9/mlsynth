Partially Pooled SCM (PPSCM)
============================

.. currentmodule:: mlsynth

Overview
--------

Partially Pooled SCM (PPSCM) extends classical synthetic control to
the staggered-adoption setting following Ben-Michael, Feller &
Rothstein (2022, *JRSS-B* 84(2):351-381). The estimator targets panels
where each treated unit adopts at its own time :math:`T_j`, and the
goal is to estimate the average treatment effect on the treated (ATT)
across both treated units and post-treatment horizons.

The key innovation is a single weight matrix :math:`\Gamma` of shape
``(N, J)`` (donors by treated units, each column on the simplex) that
trades off two pre-treatment imbalance measures:

* :math:`q_{\text{sep}}(\Gamma)^2` — the average per-treated-unit fit
  error (what you'd get from running SCM separately for each treated
  unit);
* :math:`q_{\text{pool}}(\Gamma)^2` — the fit error for the *average*
  treated unit (what you'd get from averaging the treated units first
  and running one SCM on the average).

PPSCM minimizes a convex combination, normalized by the
``\nu = 0`` (separate-SCM) baseline:

.. math::

   \min_{\Gamma \in \Delta_{\text{scm}}^J}
       \;\nu\, \tilde q_{\text{pool}}(\Gamma)^2
       + (1 - \nu)\, \tilde q_{\text{sep}}(\Gamma)^2
       + \lambda\, \|\Gamma\|_F^2.

* :math:`\nu = 0` recovers separate SCM (one synthetic control per
  treated unit, then averaged).
* :math:`\nu = 1` recovers fully pooled SCM (average the treated units
  first, then one SCM).
* Intermediate :math:`\nu` moves smoothly along a convex frontier of
  the two imbalances.

This module implements the *outcome-only* version of the paper
(Sections 3-4). The auxiliary-covariate extension of Section 5.2 is
intentionally not included.

Mathematical Formulation
------------------------

Setup
^^^^^

Let :math:`Y_{i, t}` denote unit :math:`i`'s outcome at period
:math:`t`, for ``J`` treated units and ``N`` never-treated controls.
Treated unit :math:`j` adopts at period :math:`T_j` (1-based), so its
pre-treatment outcomes live at periods :math:`T_j - L, \dots, T_j - 1`
for a common lag count :math:`L \le \min_j (T_j - 1)`. Stacking these
pre-windows gives the ``(L, J)`` matrix ``Y_treated_pre`` and the
``(L, N, J)`` tensor ``Y_donors_pre``. Post-treatment outcomes at
event-time horizons :math:`k = 0, 1, \dots, K` with
:math:`K = \min_j (T - T_j)` form ``Y_treated_post`` of shape
``(K+1, J)``.

Imbalance Measures (Eq. 5)
^^^^^^^^^^^^^^^^^^^^^^^^^^

For a weight matrix :math:`\Gamma`, define the per-treated residual

.. math::

   r_j(\ell) = Y_{j, T_j - \ell} - \sum_i \Gamma_{i, j} Y_{i, T_j - \ell},
   \qquad \ell = 1, \dots, L.

Then

.. math::

   q_{\text{sep}}(\Gamma)^2
   = \frac{1}{J} \sum_j \frac{1}{L} \|r_j\|^2,
   \qquad
   q_{\text{pool}}(\Gamma)^2
   = \frac{1}{L} \left\| \frac{1}{J} \sum_j r_j \right\|^2.

Note that :math:`q_{\text{pool}} \le q_{\text{sep}}` always (the
average of squared norms dominates the norm of the average), and the
gap is typically large when treatment timing varies.

Normalization (Section 4)
^^^^^^^^^^^^^^^^^^^^^^^^^

The paper normalizes both imbalances by their values at the
:math:`\nu = 0` (separate-SCM) solution :math:`\hat\Gamma_{\text{sep}}`:

.. math::

   \tilde q_{\text{sep}}(\Gamma)
   = \frac{q_{\text{sep}}(\Gamma)}{q_{\text{sep}}(\hat\Gamma_{\text{sep}})},
   \qquad
   \tilde q_{\text{pool}}(\Gamma)
   = \frac{q_{\text{pool}}(\Gamma)}{q_{\text{pool}}(\hat\Gamma_{\text{sep}})}.

This puts both terms on the same scale regardless of the panel and
makes :math:`\nu` interpretable across applications. :mod:`mlsynth`
solves the separate-SCM problem first to fix these baselines, then
runs the partially-pooled QP.

Auto-:math:`\nu` Selection
^^^^^^^^^^^^^^^^^^^^^^^^^^

When the user does not supply :math:`\nu`, :mod:`mlsynth` selects it
via the "balance frontier knee" heuristic: sweep :math:`\nu` over a
grid (default size 21) on :math:`[0, 1]`, and pick the value that
minimizes

.. math::

   \tilde q_{\text{sep}}(\Gamma_\nu) + \tilde q_{\text{pool}}(\Gamma_\nu).

At :math:`\nu = 0` both normalized imbalances equal 1 (sum = 2). At
intermediate :math:`\nu` the pooled term shrinks below 1 while the
separate term grows above 1; the minimum-sum point is the practical
equal-marginal-improvement compromise the paper recommends when the
error-bound parameters of Theorems 1 and 2 are unknown. The full
frontier :math:`\{\nu \mapsto (q_{\text{sep}}, q_{\text{pool}})\}` is
retained on :py:attr:`PPSCMDesign.frontier` for downstream inspection.

Intercept Shift (Section 4)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The paper recommends as a default subtracting each unit's
pre-treatment mean before fitting -- equivalent to applying PPSCM to
a weighted DiD on demeaned series. This is exposed via
:py:attr:`PPSCMConfig.demean`; the default in :mod:`mlsynth` is
``False`` (paper-faithful to Eq. 3) but the demeaned variant is one
flag away.

Treatment Effect (Eq. 5)
^^^^^^^^^^^^^^^^^^^^^^^^

With :math:`\hat\Gamma` in hand, the per-horizon ATT is

.. math::

   \widehat{\text{ATT}}_k
   = \frac{1}{J} \sum_j
       \left( Y_{j, T_j + k} - \sum_i \hat\Gamma_{i, j} Y_{i, T_j + k} \right),
   \qquad k = 0, \dots, K.

The overall ATT averages over horizons; both are exposed on
:py:attr:`PPSCMResults.inference` (overall) and
:py:attr:`PPSCMResults.event_study` (per-horizon).

Jackknife Inference (Section 5)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For each treated unit :math:`j`, drop it from the panel and refit
PPSCM on the remaining :math:`J - 1` treated units. The leave-one-out
ATT estimates yield a jackknife standard error and Wald confidence
band on :py:attr:`PPSCMInference.se` /
:py:attr:`PPSCMInference.ci`. When :math:`J < 2` the jackknife is
undefined and :mod:`mlsynth` returns ``NaN`` SE/CI.

Core API
--------

.. automodule:: mlsynth.estimators.ppscm
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.PPSCMConfig
   :members:
   :undoc-members:

Helper Modules
--------------

.. automodule:: mlsynth.utils.ppscm_helpers.setup
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.ppscm_helpers.imbalance
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.ppscm_helpers.optimization
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.ppscm_helpers.inference
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.ppscm_helpers.plotter
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.ppscm_helpers.structures
   :members:
   :undoc-members:

Example
-------

.. code-block:: python

   import pandas as pd
   from mlsynth import PPSCM

   df = pd.read_csv("staggered_panel.csv")  # treat in {0, 1}, J >= 2 treated

   results = PPSCM({
       "df":             df,
       "outcome":        "log_spending",
       "treat":          "bargaining_law",
       "unitid":         "state",
       "time":           "year",
       "nu":             "auto",      # default: knee of the balance frontier
       "demean":         False,       # paper recommends True for staggered DiD-style
       "lam":            1e-6,        # ridge for uniqueness; tiny by default
       "run_inference":  True,        # leave-one-treated-unit-out jackknife
       "display_graphs": True,
   }).fit()

   print(results.design.nu_used)
   print(results.design.q_sep, results.design.q_pool)
   print(results.inference.att, results.inference.ci)

   # Per-horizon ATT trajectory.
   for k, tau, se in zip(results.event_study.horizons,
                         results.event_study.tau,
                         results.event_study.se):
       print(f"k = {k:>2}  ATT_k = {tau:+.3f}  se = {se:.3f}")

   # Per-treated weight matrix (column j sums to 1).
   results.design.Gamma                       # (N, J)
   results.donor_weights["state_X"]           # {donor_name: gamma_ij}

   # Balance frontier from the auto-nu sweep.
   for nu, (q_sep, q_pool) in sorted(results.design.frontier.items()):
       print(nu, q_sep, q_pool)

References
----------

Ben-Michael, E., Feller, A., & Rothstein, J. (2022). "Synthetic
Controls with Staggered Adoption." *Journal of the Royal Statistical
Society: Series B* 84(2):351-381.
