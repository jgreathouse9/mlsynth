Spillover-Aware Synthetic Control (SPILLSYNTH)
================================================

.. currentmodule:: mlsynth

Overview
--------

SPILLSYNTH is mlsynth's dispatcher for **synthetic-control estimators
that explicitly model spillover (interference) onto control units**.
Classical SCM (and most of its variants) assume SUTVA: that the donor
units are not themselves affected by the treatment. When that fails --
neighboring states, supply-chain partners, geographic spillovers,
informational contagion -- the synthetic counterfactual is contaminated
by the spillover, and the resulting ATT is biased.

Currently ships ``method="cd"`` (Cao & Dowd 2023), with additional
methods to follow under the same dispatcher.

When to use SPILLSYNTH
^^^^^^^^^^^^^^^^^^^^^^^

* You have **geographic or institutional reasons** to suspect specific
  control units are exposed to spillover (e.g. neighboring states for
  a tax change, neighboring firms for a procurement rule, partner
  countries for a sanctions regime).
* You can **enumerate** the potentially-affected units before fitting.
  The estimator does not discover them; it estimates the size of each
  declared unit's spillover effect jointly with the treatment effect.
* You have a **moderate pre-period** (paper sims use :math:`T_0 \geq 15`)
  so the leave-one-out SCM fits are well-estimated.

Method: ``method='cd'`` -- Cao & Dowd (2023)
--------------------------------------------

The Cao-Dowd estimator works in two steps. First, fit a Ferman-Pinto
(2021) demeaned simplex SCM **for every unit** in the panel, treating
each in turn as the focal unit and using the others as donors. Call
the resulting :math:`N \times N` weight matrix :math:`\widehat B`
(with :math:`\widehat B_{ii} = 0`) and the length-:math:`N` intercept
vector :math:`\widehat a`.

Second, encode the spillover structure in an :math:`N \times (1+p)`
matrix :math:`A` whose first column is a basis vector for the treated
unit and whose remaining :math:`p` columns are basis vectors for each
**declared potentially-affected** control unit (Example 3 of the
paper). The treatment and spillover effects at post-period :math:`t` are
recovered via the closed form (paper eq. 5)

.. math::

   \widehat \gamma_t = \left(A' \widehat M A\right)^{-1} A' (I - \widehat
   B)' \left[ (I - \widehat B) Y_t - \widehat a \right],
   \qquad \widehat M = (I - \widehat B)' (I - \widehat B).

The full effect vector is :math:`\widehat \alpha_t = A \widehat \gamma_t`.
Its first entry is the **spillover-adjusted ATT on the treated unit**;
its remaining :math:`p` entries are the **per-affected-unit spillover
effects**; the remaining :math:`N - 1 - p` entries are identically zero
(reflecting the user's assertion that those units are unaffected).

Why the demeaned simplex
^^^^^^^^^^^^^^^^^^^^^^^^

Ferman & Pinto (2021) showed that ordinary SCM is biased when the
pre-treatment fit is imperfect. Demeaning (subtracting each unit's
pre-period mean before fitting the simplex weights, then recovering
the intercept analytically) gives an asymptotically unbiased estimator
even under imperfect pre-treatment fit, which is the regime Cao-Dowd
assume throughout. SPILLSYNTH ships this demeaned variant as the
underlying SCM in :func:`fit_leave_one_out_sc`.

Why the joint inversion
^^^^^^^^^^^^^^^^^^^^^^^

Vanilla SCM applied to the treated unit alone ignores the residual
information in the control units' own SCM fits. Cao-Dowd show that
once :math:`A` encodes the spillover structure, the joint system uses
ALL units' residual information to back out :math:`(\alpha_1,
\gamma_2, \dots, \gamma_{1+p})` simultaneously, and this aggregation
substantially reduces the bias in :math:`\widehat \alpha_1` compared
with the vanilla SCM that throws away the cross-unit residuals.

Identification: Assumption 1(d) (the paper's invertibility condition)
requires :math:`A' M A` to be non-singular. The Discussion in Section
3.4.1 of the paper shows this holds whenever not all controls are
exposed to spillover (i.e. at least one row of :math:`A` is zero) and
the spillover structure is not pathologically aligned with the SCM
weight pattern. The fit container exposes ``cond(A'MA)`` as a
diagnostic.

Scope of this implementation
----------------------------

Currently shipped:

* Per-unit demeaned simplex SCM (eq. 1-2 of the paper).
* The closed-form treatment-and-spillover estimator with Example-3
  spillover structure (eq. 5 of the paper).
* A vanilla-SCM counterfactual for side-by-side comparison.

Not yet shipped (planned):

* Andrews (2003) end-of-sample P-test inference (paper Section 4).
* Distance-decay (Example 2) and homogeneous-set (Example 1)
  spillover-structure helpers.
* The GMM-efficient weighting variant from Section 7.1.

Core API
--------

.. automodule:: mlsynth.estimators.spillsynth
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.SPILLSYNTHConfig
   :members:
   :undoc-members:

Helper Modules
--------------

.. automodule:: mlsynth.utils.spillsynth_helpers.setup
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.spillsynth_helpers.cd.scm_core
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.spillsynth_helpers.cd.estimation
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.spillsynth_helpers.cd.pipeline
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.spillsynth_helpers.structures
   :members:
   :undoc-members:

Example
-------

A one-factor panel with a true treatment effect of :math:`-3` on the
treated unit and a true spillover of :math:`+1.5` on one declared
neighbor. Vanilla SCM gets biased toward zero because the
spillover-affected donor inflates the synthetic counterfactual;
SPILLSYNTH with the correct ``affected_units`` recovers the planted
effect.

.. code-block:: python

   import numpy as np
   import pandas as pd

   from mlsynth import SPILLSYNTH

   # ------------------------------------------------------------------
   # 1. One-factor panel; unit 0 treated, unit 1 receives spillover
   # ------------------------------------------------------------------
   rng = np.random.default_rng(0)
   N, T, T0 = 8, 40, 30
   true_treatment = -3.0
   true_spillover = 1.5

   loadings = rng.uniform(0.5, 1.5, size=N)
   intercept = rng.uniform(-1.0, 1.0, size=N)
   f = np.cumsum(rng.standard_normal(T)) * 0.4 + 0.05 * np.arange(T)
   Y = intercept[:, None] + np.outer(loadings, f) + 0.10 * rng.standard_normal((N, T))
   Y[0, T0:] += true_treatment
   Y[1, T0:] += true_spillover     # spillover injected on unit 1

   D = np.zeros((N, T))
   D[0, T0:] = 1
   rows = [{"unit": f"u{i}", "year": t, "y": Y[i, t], "treat": int(D[i, t])}
           for i in range(N) for t in range(T)]
   df = pd.DataFrame(rows)

   # ------------------------------------------------------------------
   # 2. Fit SPILLSYNTH with u1 declared as the affected donor
   # ------------------------------------------------------------------
   res = SPILLSYNTH({
       "df": df, "outcome": "y", "treat": "treat",
       "unitid": "unit", "time": "year",
       "method": "cd",
       "affected_units": ["u1"],
       "display_graphs": False,
   }).fit()

   print(f"ATT_SP   = {res.att:+.3f}  (true treatment = {true_treatment})")
   print(f"ATT_SCM  = {res.att_scm:+.3f}  (vanilla SCM, no spillover correction)")
   print(f"spillover on u1 (avg) = {res.spillover_effects['u1'].mean():+.3f}  "
         f"(true = {true_spillover})")
   print(f"cond(A'MA) = {res.cd.cond_AMA:.2e}")

Verification
------------

.. note::

   **Algorithmic equivalence with Cao-Dowd's R reference (verified to
   four decimals).** The Cao-Dowd authors published their estimator as
   the R package ``scmSpillover``. The package ships a bundled
   :file:`cigs.rda` dataset that covers 39 states from 1970-2000.

   Running both implementations on the same augmented 51-unit panel
   (the 50 states in :file:`prop99_packsales.csv` plus Washington DC
   sourced from the CDC Tax Burden on Tobacco compilation) with the
   13 spillover-affected states from Cao-Dowd's Section 5 footnote 5
   (AK, AZ, DC, FL, HI, MA, MD, MI, NJ, NV, NY, OR, WA) yields:

   .. code-block::

      year         R port           Python (this estimator)
      1989         +0.0827          +0.0828
      1990         +3.7144          +3.7136
      1991         -3.7584          -3.7575
      1992         -3.4271          -3.4269
      1993         -7.6146          -7.6147
      1994        -10.9137         -10.9143
      1995        -12.8346         -12.8346
      1996        -13.0843         -13.0838
      1997        -14.9136         -14.9143
      1998        -16.0812         -16.0808
      1999        -18.9588         -18.9591
      2000        -15.4901         -15.4904

      Avg ATT 1989-2000:  SP_R = -9.4399   SP_Py = -9.4400
      Avg ATT 1989-1992:  SP_R = -0.8471   SP_Py = -0.8470

   The ATT difference between SCM (:math:`-10.81`) and SP
   (:math:`-9.44`) on this panel reproduces Cao-Dowd's headline
   empirical finding: spillover to neighboring states (most notably
   Nevada, Oregon, and DC) inflates the vanilla SCM's estimated
   treatment effect, particularly in the first four post-treatment
   years where the spillover-adjusted ATT is close to zero
   (:math:`-0.85`) while vanilla SCM produces :math:`-8.18`.

References
----------

Abadie, A., Diamond, A., & Hainmueller, J. (2010). "Synthetic Control
Methods for Comparative Case Studies." *Journal of the American
Statistical Association* 105(490):493-505.

Andrews, D. W. K. (2003). "End-of-Sample Instability Tests."
*Econometrica* 71(6):1661-1694.

Cao, J., & Dowd, C. (2023). "Estimation and Inference for Synthetic
Control Methods with Spillover Effects." Working paper.

Ferman, B., & Pinto, C. (2021). "Synthetic Controls with Imperfect
Pretreatment Fit." *Quantitative Economics* 12(4):1197-1221.
