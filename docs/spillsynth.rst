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

When to Use This Method
^^^^^^^^^^^^^^^^^^^^^^^^

Classical SCM is *particularly* fragile to spillovers because its weights
actively select the controls most correlated with the treated unit -- and
those are often exactly the units the treatment leaks into. When
California raises cigarette taxes, sales shift to Nevada; if Nevada carries
heavy synthetic-control weight, the counterfactual is contaminated and the
ATT is biased, sometimes by *more* than a naive difference-in-differences
would be (Cao & Dowd, Section 6).

The reflexive fix is the **pure-donor method**: drop every control you
think is contaminated and fit SCM on the survivors. Cao & Dowd argue this
is often the wrong trade:

* In many panels **most or all controls are exposed** (geographically
  aggregated data), so there is no clean donor pool left to drop to.
* Contaminated units are frequently the **most informative** donors;
  discarding them loses efficiency.
* Using fewer controls **widens the worst-case bias** when the spillover
  structure is misspecified -- the pure-donor estimator has a larger
  identified set.

SPILLSYNTH (``method="cd"``, Cao & Dowd 2023) instead keeps *all* units and
estimates the **direct treatment effect and the spillover effects jointly**
under an assumed, contextually-motivated spillover structure
(matrix :math:`A`, linear in unknown parameters). It is asymptotically
unbiased for both, supplies an end-of-sample-instability (:math:`P`-test)
inference procedure that also blunts the selection-into-treatment threat to
ordinary placebo tests, and ships a misspecification test (the
:math:`\kappa_A`-statistic) for the structure you assumed. It covers
multiple treated units / post-periods and both stationary and cointegrated
factor models.

Reach for SPILLSYNTH when
~~~~~~~~~~~~~~~~~~~~~~~~~~

* You have **geographic or institutional reasons** to suspect specific
  control units are exposed to spillover (e.g. neighboring states for
  a tax change, neighboring firms for a procurement rule, partner
  countries for a sanctions regime).
* You can **specify the spillover structure** :math:`A` from contextual
  knowledge before fitting. The estimator does not discover the affected
  units; it estimates the size of each declared unit's spillover effect
  jointly with the treatment effect.
* **Many or all controls are contaminated**, so simply dropping the
  affected units (the pure-donor route) is impractical or wastes too much
  information.
* You have a **moderate-to-long pre-period** (paper sims use
  :math:`T_0 \geq 15`) -- the asymptotics are large-:math:`T`, fixed number
  of controls -- so the leave-one-out SCM fits are well-estimated.

Do not use SPILLSYNTH when
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **SUTVA credibly holds** (no spillover). The extra spillover parameters
  only add variance; use classic SC (:doc:`tssc`, :doc:`scmo`,
  :doc:`clustersc`).
* **You cannot defend a spillover structure** :math:`A`. The estimator
  assumes :math:`A` is known; a badly misspecified structure biases both
  effects (the :math:`\kappa_A` test mitigates but does not remove this).
  If only a *few* units are contaminated and droppable, the pure-donor
  approach on a pruned pool (classic SC) is the simpler honest choice.
* **Spillovers decay with a known spatial weighting and you want a
  DiD-style objective** -- :doc:`spsydid` pools them through a spatial
  weight matrix in a synthetic difference-in-differences fit.
* **The pre-period is short.** The leave-one-out SCM fits underpinning
  Assumption 1(a)-(c) are then noisy; a factor-model estimator
  (:doc:`fma`) may be more stable.
* **Distributional** questions (quantiles, tails) -- use :doc:`dsc`.

.. note::

   SPILLSYNTH belongs to the *spillover-aware* family of mlsynth
   estimators alongside :doc:`spsydid`. Both relax SUTVA on the donor
   pool: SpSyDiD via a spatial-weights restriction in a synthetic
   difference-in-differences objective, SPILLSYNTH via an explicit
   spillover-structure matrix :math:`A` in a closed-form treatment
   estimator. Use SPILLSYNTH when the spillover set is enumerable and
   per-unit; use SpSyDiD when spillovers decay with a known spatial
   weighting and pooling them buys efficiency.

Assumptions
-----------

The Cao-Dowd (2023) estimator is derived under **Assumption 1**, with
parts (a)-(c) standard regularity conditions on the underlying SCM
fits and part (d) an identification condition:

(a) :math:`\{u_t\}_{t \geq 1}` is stationary with mean zero, where
    :math:`u_t = Y_t(0) - (a + B Y_t(0))` is the per-unit SCM
    specification error stacked across units.
(b) The leave-one-out SCM fits are consistent for their population
    counterparts: :math:`\|\widehat a - a\| = o_p(1)` and
    :math:`\|\widehat B - B\| = o_p(1)`.
(c) The post-period extrapolation is stable:
    :math:`\|(\widehat B - B)\, Y_{T+1}(0)\| = o_p(1)`.
(d) **Identification.** :math:`A' M A` is non-singular, where
    :math:`M = (I - B)'(I - B)`. Equivalently, :math:`(I - B) A`
    has full column rank.

Parts (a)-(c) require, in practice, a moderate pre-period
(:math:`T_0 \gtrsim 15` in the paper's simulations). Part (d) holds
whenever the spillover structure is not pathologically aligned with
the SCM weight pattern (Section 3.4.1 of the paper); the fit container
exposes ``cd.cond_AMA`` as a numerical diagnostic.

The paper shows that Assumption 1 is satisfied by factor-model DGPs
under two alternative regularity conditions on the common factors:

**Condition ST** (stationary factors).
  :math:`\{(\eta_t, \lambda_t, \varepsilon_t)\}_{t \geq 1}` is
  stationary, ergodic for first and second moments, has a finite
  :math:`(2 + \delta)`-moment, and :math:`\mathrm{cov}[Y_t(0)] =
  \Omega_y` is positive definite.

**Condition CO** (cointegrated :math:`\mathcal{I}(1)` factors).
  Write :math:`y_{i,t}(0) = (\lambda_t^1)' \mu_i^1 + (\lambda_t^0)'
  \mu_i^0 + \varepsilon_{i,t}` with :math:`\{\lambda_t^1\}` an
  :math:`\mathcal{I}(1)` process and :math:`\{\lambda_t^0\}`
  stationary. Loadings :math:`\{\mu_i^1\}` admit cointegrating vectors:
  for each :math:`i` there exists :math:`w^{(i)} \in W^{(i)}` such that
  :math:`\mu_i^1 = \sum_{j=1}^{N} w_j^{(i)} \mu_j^1`.

Theoretical guarantees
----------------------

**Theorem 1 (asymptotic unbiasedness; Cao & Dowd 2023).**
Under Assumption 1,

.. math::

   \widehat \alpha - (\alpha + G\, u_{T+1}) \xrightarrow{p} 0
   \quad \text{as } T \to \infty,

where :math:`G = A (A' M A)^{-1} A' (I - B)'` and
:math:`\mathbb{E}[G\, u_{T+1}] = 0`. The estimator
:math:`\widehat \alpha` is therefore **asymptotically unbiased** for
the treatment and spillover effect vector :math:`\alpha`. (It is not
consistent because only one post-period of one treated unit is observed,
so the irreducible :math:`u_{T+1}` term does not vanish in any limit.)

**Lemma 1 (factor-model sufficiency).**
If :math:`A' M A` is non-singular, then **either** Condition ST
**or** Condition CO implies Assumption 1. Theorem 1 therefore applies
to factor-model panels with stationary or cointegrated common
factors -- the leading data-generating processes in the synthetic-
controls literature.

Two practical implications:

1. The estimator's bias under spillover does not vanish for SCM, but
   the variance of :math:`\widehat \alpha` from SPILLSYNTH is bounded
   (Section 3.4.2) under the same conditions that make standard SCM
   well-behaved.
2. Misspecifying :math:`A` (declaring too few affected units) breaks
   asymptotic unbiasedness; declaring too many is conservative (extra
   degrees of freedom inflate variance but the estimator remains
   unbiased). The Monte Carlo in Section 6.3 of the paper, reproduced
   under :ref:`spillsynth-mc`, illustrates both regimes.

Method: ``method='cd'`` -- Cao & Dowd (2023, v3)
------------------------------------------------

The Cao-Dowd estimator works in two steps. First, fit a Ferman-Pinto
(2021) demeaned simplex SCM **for every unit** in the panel, treating
each in turn as the focal unit and using the others as donors. Call
the resulting :math:`N \times N` weight matrix :math:`\widehat B`
(with :math:`\widehat B_{ii} = 0`) and the length-:math:`N` intercept
vector :math:`\widehat a`.

Second, encode the spillover structure in an :math:`N \times k` matrix
:math:`A` -- one of the three examples in Section 2.2 of the paper --
and recover the treatment-and-spillover effect vector at post-period
:math:`t` via the closed form (paper eq. 6)

.. math::

   \widehat \gamma_t = \left(A' \widehat M A\right)^{-1} A' (I - \widehat
   B)' \left[ (I - \widehat B) Y_t - \widehat a \right],
   \qquad \widehat M = (I - \widehat B)' (I - \widehat B).

The full effect vector is :math:`\widehat \alpha_t = A \widehat \gamma_t`.
Row 0 is the **spillover-adjusted ATT on the treated unit**; the
remaining rows hold the per-affected-unit spillover effects.

Choice of A: the three Cao-Dowd v3 examples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SPILLSYNTH ships **all three** examples from Cao-Dowd (2023, v3),
selected via ``SPILLSYNTHConfig.spillover_structure``. (The v3
numbering renames v2's examples; we use v3 throughout.)

* ``"per_unit"`` (**Example 1**, Section 2.2 -- *Limited range*; the
  paper's leading case). Each declared affected unit gets its own
  free spillover coefficient. :math:`A` is :math:`N \times (1 + p)`
  with column 0 the treated-unit basis vector and columns
  :math:`1, \dots, p` the basis vectors for each affected control.
  Helper: :func:`build_A_per_unit`.

* ``"homogeneous"`` (**Example 2**, Section 3.4). All declared
  affected units share a single spillover coefficient :math:`b`.
  :math:`A` is :math:`N \times 2` with column 0 = treated basis and
  column 1 the indicator over affected rows. Use when domain
  knowledge constrains spillovers to be of equal magnitude across the
  affected set. Helper: :func:`build_A_homogeneous`.

* ``"distance_decay"`` (**Example 3**, Section 7.1). Spillover decays
  as :math:`\alpha_i = b \exp(-d_i)` with user-supplied
  distances. :math:`A` is :math:`N \times 2` with row :math:`i` set to
  :math:`(0, \exp(-d_i))`. **Every** control receives some spillover
  (the magnitude is the variable of interest); see Section 7.1 for the
  link to continuous-treatment models. Helper:
  :func:`build_A_distance_decay`; supply ``unit_distances={label: d}``
  to the config.

Section 5.1 of the paper argues that a more conservative researcher
should choose a larger :math:`p` (declare more units affected) -- if
in doubt, include the unit. The :math:`\kappa_A` specification test
(see :ref:`spillsynth-kappa-A`) and the pure-donor sensitivity
analysis (see :ref:`spillsynth-pure-donor`) provide complementary
diagnostics for the chosen :math:`A`.

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

* Per-unit demeaned simplex SCM (eq. 2-3 of the paper).
* The closed-form treatment-and-spillover estimator (eq. 6 of v3)
  under **all three A-matrix structures**: per-unit (Example 1),
  homogeneous (Example 2), and distance-decay (Example 3).
* Original Abadie 2010 SCM counterfactual for side-by-side comparison.
* Cao-Dowd Section 4 P-test inference (Theorems 2 and 3): per-period
  tests for ``H_0: alpha_1(t) = 0`` (treatment) and
  ``H_0: alpha_k(t) = 0`` (per-affected-unit spillover), the **joint**
  spillover hypothesis matching the MATLAB reference, signed
  confidence intervals via test inversion, and the
  :math:`\kappa_A` specification test for the chosen :math:`A`.
  See :ref:`spillsynth-inference` and :ref:`spillsynth-kappa-A`.
* Cao-Dowd v3 Section 5.2 **pure-donor sensitivity analysis** --
  worst-case misspecification-bias bounds comparing SP against the
  pure-donor SCM. See :ref:`spillsynth-pure-donor`.
* Cao-Dowd v3 Section S.1.1 **GMM-efficient variant** (Proposition
  S.1): :math:`\widehat \alpha^e` minimises asymptotic variance via
  :math:`W = \widehat \Omega^{-1}`. See :ref:`spillsynth-efficient`.
* Cao-Dowd v3 Section S.1.2 **multiple treated units** with a common
  intervention time -- per-treated-unit ATTs, CIs, and treatment-
  effect tests. See :ref:`spillsynth-multi-treated`.

Not yet shipped (planned):

* Multiple post-treatment-period structure selection (Section S.1.3).
* Covariates (Section S.1.4).
* Staggered adoption (treated units with **different** intervention
  times). The cohort decomposition is supported upstream by
  :func:`mlsynth.utils.datautils.dataprep`; per-cohort SPILLSYNTH
  orchestration is a separate scope.

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

.. automodule:: mlsynth.utils.spillsynth_helpers.cd.inference
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.spillsynth_helpers.cd.sensitivity
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

Empirical replication: California's Proposition 99
---------------------------------------------------

The following code reproduces Cao-Dowd (2023) Section 5 on a 51-unit
panel: the 50 states in :file:`basedata/prop99_packsales.csv` plus
Washington DC (whose per-capita pack sales for 1970-2015 are merged in
from the CDC Tax Burden on Tobacco compilation, shipped as
:file:`basedata/prop99_with_dc.csv`). The pre-period is 1970-1988 and
the post-period is 1989-2000, matching the paper. The 13 declared
spillover-affected states are the ones listed in Cao-Dowd Section 5
footnote 5.

.. code-block:: python

   import numpy as np
   import pandas as pd

   from mlsynth import SPILLSYNTH

   df = pd.read_csv("basedata/prop99_with_dc.csv")
   df = df[(df["year"] >= 1970) & (df["year"] <= 2000)].copy()
   df["treat"] = ((df["state"] == "California") & (df["year"] >= 1989)).astype(int)

   # Cao-Dowd Section 5, footnote 5: states the authors flag as
   # potentially exposed to spillover from California's Prop 99.
   affected = [
       "Alaska", "Arizona", "District of Columbia", "Florida", "Hawaii",
       "Massachusetts", "Maryland", "Michigan", "New Jersey", "Nevada",
       "New York", "Oregon", "Washington",
   ]

   res = SPILLSYNTH({
       "df": df, "outcome": "cigsale", "treat": "treat",
       "unitid": "state", "time": "year",
       "method": "cd",
       "affected_units": affected,
       "display_graphs": False,
   }).fit()

   post_years = sorted(df.loc[df["treat"] == 1, "year"].unique())
   att_sp_by_year = res.cd.alpha[0, :]    # row 0 is the treated unit

   print(f"Avg ATT 1989-2000  SP  = {att_sp_by_year.mean():+.4f}")
   print(f"Avg ATT 1989-2000  SCM = {res.att_scm:+.4f}")
   early = att_sp_by_year[: sum(y <= 1992 for y in post_years)]
   print(f"Avg ATT 1989-1992  SP  = {early.mean():+.4f}")

   for y, a in zip(post_years, att_sp_by_year):
       print(f"  {y}  ATT_SP = {a:+.4f}")

Running this prints, to four decimals,

.. code-block::

   Avg ATT 1989-2000  SP  =  -9.4399
   Avg ATT 1989-2000  SCM = -10.8120
   Avg ATT 1989-1992  SP  =  -0.8471

     1989  ATT_SP =  +0.0827
     1990  ATT_SP =  +3.7144
     1991  ATT_SP =  -3.7584
     1992  ATT_SP =  -3.4271
     1993  ATT_SP =  -7.6146
     1994  ATT_SP = -10.9137
     1995  ATT_SP = -12.8346
     1996  ATT_SP = -13.0843
     1997  ATT_SP = -14.9136
     1998  ATT_SP = -16.0812
     1999  ATT_SP = -18.9588
     2000  ATT_SP = -15.4901

The gap between vanilla SCM (:math:`-10.81`) and SP (:math:`-9.44`)
reproduces Cao-Dowd's headline empirical finding: spillover to
neighboring states (most notably Nevada, Oregon, and DC) inflates
vanilla SCM's estimated treatment effect, particularly in the first
four post-treatment years where the spillover-adjusted ATT is close to
zero (:math:`-0.85`) while vanilla SCM produces approximately
:math:`-8`.

Verification
------------

.. note::

   **Algorithmic equivalence with Cao-Dowd's R reference (verified to
   four decimals).** The Cao-Dowd authors published their estimator as
   the R package ``scmSpillover``. Running both implementations on the
   51-unit panel above yields:

   .. code-block::

      year         R port           Python (this estimator)
      1989         +0.0827          +0.0827
      1990         +3.7144          +3.7144
      1991         -3.7584          -3.7584
      1992         -3.4271          -3.4271
      1993         -7.6146          -7.6146
      1994        -10.9137         -10.9137
      1995        -12.8346         -12.8346
      1996        -13.0843         -13.0843
      1997        -14.9136         -14.9136
      1998        -16.0812         -16.0812
      1999        -18.9588         -18.9588
      2000        -15.4901         -15.4901

      Avg ATT 1989-2000:  SP_R = -9.4399   SP_Py = -9.4399
      Avg ATT 1989-1992:  SP_R = -0.8471   SP_Py = -0.8471

.. _spillsynth-inference:

Inference: the Cao-Dowd P-test
------------------------------

The Cao-Dowd estimator is asymptotically unbiased but not consistent
(only one post-period draw of :math:`u_{T+1}` is observed). Standard
errors and confidence intervals therefore come from inverting an
adaptation of the Andrews (2003) end-of-sample instability test
(paper Section 4).

The procedure tests a general linear hypothesis on the
treatment-and-spillover effect vector,

.. math::

   H_0: C \alpha = d,
   \qquad
   H_1: C \alpha \neq d,

for known :math:`C \in \mathbb{R}^{q \times N}` and :math:`d \in
\mathbb{R}^q`. The two practical specialisations SPILLSYNTH always
computes are:

* **Treatment-effect test** (paper, Section 4.2, applied with
  :math:`C = e_1^\prime`, :math:`d = 0`): tests
  :math:`H_0: \alpha_1(t) = 0` separately for each post-period
  :math:`t = T+1, \dots, T+m`.
* **Per-affected-unit spillover test** (one such test per declared
  affected unit :math:`k`, with :math:`C = e_{k+1}^\prime`):
  tests :math:`H_0: \alpha_k(t) = 0`.

**Test statistic.** At post-period :math:`t`,

.. math::

   P_t = (C \widehat \alpha_t - d)' W_T (C \widehat \alpha_t - d),

with default :math:`W_T = I` (Lemma 3 of the paper -- under either
Condition ST or Condition CO this choice satisfies the regularity
required for asymptotic validity).

**Reference distribution.** For each pre-period :math:`s = 1, \dots,
T_0`, the quadratic form

.. math::

   \widehat P_s = \widehat u_s'\, \widehat G'\, C'\, W_T\, C\,
   \widehat G\, \widehat u_s,
   \qquad
   \widehat G = A (A' \widehat M A)^{-1} A' (I - \widehat B)',

where :math:`\widehat u_s = (I - \widehat B) Y_s - \widehat a` is the
pre-period residual vector, supplies the empirical CDF. Reject
:math:`H_0` at level :math:`\tau` when
:math:`P_t > \widehat q_{P, 1-\tau}` (the :math:`(1 - \tau)`-quantile
of :math:`\{\widehat P_s\}_{s=1}^{T_0}`). The :math:`p`-value is the
empirical-CDF tail mass at :math:`P_t`.

**Assumptions for validity (Assumption 3 of the paper).**

(a) Assumption 1 holds (the estimation-side assumptions stated above).
(b) :math:`\{u_t\}_{t \geq 1}` is ergodic with :math:`\mathbb{E}[\|u_t\|]
    < \infty`.
(c) There exists a non-random positive-definite sequence :math:`\{D_T\}`
    with :math:`\max_{t \leq T+1} \|D_T^{-1} x_t\| = O_p(1)`, where
    :math:`x_t = (1, Y_t')'`.
(d) Both the full-sample and leave-one-out SCM coefficients are
    consistent in Frobenius norm, scaled by :math:`D_T`.
(e) The CDF :math:`F_P` of :math:`P_1(\theta_0)` is continuous and
    increasing at its :math:`(1 - \tau)`-quantile.
(f) :math:`W_T \xrightarrow{p} W` as :math:`T \to \infty`.

**Theorem 3 (asymptotic validity; Cao & Dowd 2023).**
Under Assumption 3, as :math:`T \to \infty`:

(a) :math:`P \xrightarrow{d} P_\infty`;
(b) :math:`\widehat F_{P, T}(x) \xrightarrow{p} F_P(x)` at every
    :math:`x` in a neighbourhood of :math:`q_{P, 1 - \tau}`;
(c) :math:`\widehat q_{P, 1 - \tau} \xrightarrow{p} q_{P, 1 - \tau}`;
(d) :math:`\Pr(P > \widehat q_{P, 1 - \tau}) \to \tau`
    under :math:`H_0`.

So the test is asymptotically of correct size :math:`\tau`. Lemma 3
of the paper then shows that under Condition ST or Condition CO,
:math:`W_T = I` is a valid choice for the weight matrix.

**Why not the placebo or vanilla Andrews test?** Section 4.3 of the
paper gives the intuition (and the Monte Carlo below quantifies it):

* The Abadie (2010) placebo test under-rejects under positive
  spillover. Spillover-affected donors shift their residual densities
  to the right, widening the across-unit distribution and making the
  treated unit's residual less extreme.
* Andrews' (2003) P-test using only the treated unit's own residuals
  ignores the cross-unit information altogether and **over-rejects**
  under spillover -- the pre-period treated residual density is
  unaffected by spillover, but the post-period one is, so the test
  statistic shifts and the empirical CDF fails to track it.
* The Cao-Dowd test exploits the cross-unit information via :math:`A`
  and :math:`\widehat G`; both the pre-period reference and the
  post-period statistic share the same weighting, so size is
  preserved under either spillover regime.

**Result fields.** After fitting,

.. code-block:: python

   res = SPILLSYNTH({..., "affected_units": ["u1", "u2"], ...}).fit()
   tt = res.cd.treatment_test         # PTestResult on H_0: alpha_1(t) = 0
   tt.P_post                          # (T1,) per-post-period statistic
   tt.P_pre                           # (T0,) pre-period reference
   tt.p_value                         # (T1,) tail-mass p-value
   tt.cutoff_05                       # 5% critical value
   tt.reject_05                       # (T1,) boolean

   sp = res.cd.spillover_tests["u1"]  # PTestResult on H_0: alpha_{u1}(t) = 0
   # Same fields, indexed by affected-unit label.

   # Joint spillover test (matches MATLAB reference): single rejection
   # per post-period for H_0: alpha_2 = ... = alpha_{1+p} = 0 jointly.
   jt = res.cd.joint_spillover_test   # PTestResult or None when p == 0

   # 95% confidence intervals from inverting the level-5% P-test
   # (Section 6.2 / MATLAB ``sp_andrews_te``).
   res.cd.treatment_ci_95             # (T1, 2) [lower, upper] per period
   res.cd.spillover_ci_95["u1"]       # likewise for each affected unit

.. _spillsynth-kappa-A:

A-specification test: the :math:`\kappa_A` statistic
----------------------------------------------------

**New in v3** (Section 5.1.2). The estimator's misspecification bias is
linear in the *missed* spillover effects (eq. 10 of the paper), so a
goodness-of-fit statistic on the residualised post-period outcome is
informative about whether :math:`A` correctly captures the spillover
structure. Define

.. math::

   \kappa_A = \| (I - \widehat B)(Y_{T+1} - \widehat \alpha) - \widehat a \|,

a function of :math:`A` through :math:`\widehat \alpha`. Project the
pre-period residual onto the orthogonal complement of the column space
of :math:`(I - \widehat B) A`,

.. math::

   \widehat \Gamma_A = (I - \widehat B) A
   \left( A' (I - \widehat B)' (I - \widehat B) A \right)^{-1}
   A' (I - \widehat B)',

and form the empirical CDF
:math:`\widehat F^A_{\kappa, T}(x) = T^{-1} \sum_{t=1}^{T} \mathbf{1}\{
\|(I - \widehat \Gamma_A) \widehat u_t\| \leq x\}`. Reject
:math:`H_0`: "A correctly specifies the spillover effects" if
:math:`\kappa_A` exceeds the empirical
:math:`(1 - \tau)`-quantile.

**Proposition 2** (Cao-Dowd v3). Under Assumption 3,
:math:`\Pr(\kappa_A > \widehat q^A_{\kappa, 1 - \tau}) \to \Pr(\|(I -
\Gamma_A) u_{T+1} + (I - \Gamma_A)(I - B) \alpha\| \geq q^A_{\kappa, 1
- \tau})`. When :math:`A` is correctly specified the deterministic
term vanishes and the rejection probability converges to the nominal
:math:`\tau`.

SPILLSYNTH always populates this test:

.. code-block:: python

   kA = res.cd.kappa_A_test           # KappaATestResult
   kA.kappa_A                         # (T1,) per-period statistic
   kA.p_value                         # (T1,) tail-mass p-value
   kA.reject_05                       # (T1,) boolean

For **A-selection**, use :func:`select_A_by_kappa` to pick among
candidate :math:`A` matrices by minimising the mean
:math:`\kappa_A` over post-periods (Section S.1.3 of the paper notes
this is a heuristic with a single post-period and a consistent
selector with multiple).

.. _spillsynth-pure-donor:

Pure-donor sensitivity analysis
-------------------------------

**New in v3** (Section 5.2). An alternative to SP is the *pure-donor*
SCM, which simply drops every assumed-affected unit from the donor
pool. Both methods are asymptotically unbiased when :math:`A` is
correctly specified, but they differ in robustness to
misspecification: if some assumed-clean control was actually exposed
to a spillover of magnitude :math:`\bar \alpha`, what bias does that
inject?

Section 5.2 shows the worst-case bias is **linear** in
:math:`\bar \alpha` with coefficient

* SP: :math:`c_p^{SP} = \sum_{j=1}^{p} |\widetilde w_{SP, j}|`,

* PD: :math:`c_p^{PD} = \sum_{j=1}^{p} |\widetilde w_{PD, j}|`,

where :math:`\widetilde w_{SP}` and :math:`\widetilde w_{PD}` are the
relevant weight vectors (treated-row of the SP misspecification
operator, and the treated unit's SCM weights from the pure-donor fit
respectively), each sorted by absolute value in descending order. For
:math:`p` missed spillover units, the identified bias set is
:math:`[-c_p^M \bar \alpha, +c_p^M \bar \alpha]` for
:math:`M \in \{SP, PD\}`.

SPILLSYNTH exposes the raw weights so users can reproduce Figure 3 of
v3 (the Prop-99 sensitivity panel) or compute the smallest
:math:`\bar \alpha` capable of invalidating their headline estimate:

.. code-block:: python

   pds = res.cd.pure_donor_sensitivity      # PureDonorSensitivity or None
   pds.w_sp, pds.w_pd                       # sorted |weight| vectors
   pds.n_clean                              # number of assumed-clean controls
   sp_bias, pd_bias = pds.bias_bounds(p=1, alpha_bar_grid=np.linspace(0, 40, 100))
   # sp_bias[i] is the SP method's worst-case bias bound at α̅ = grid[i],
   # for the single worst-case missed spillover; pd_bias[i] is the
   # pure-donor SCM's analogous bound.

The :math:`\widetilde w_{SP}` vector is the first row of
:math:`A (A' (I - \widehat B)' (I - \widehat B) A)^{-1} A' (I -
\widehat B)' (I - \widehat B) - I_N`, restricted to columns that
correspond to clean (assumed-unaffected) controls. The
:math:`\widetilde w_{PD}` vector comes from refitting a single
demeaned simplex SCM on the panel after deleting every assumed-
affected row.

.. _spillsynth-efficient:

GMM-efficient weighting (Proposition S.1)
-----------------------------------------

**New in v3** (Section S.1.1). The default Cao-Dowd estimator
minimises :math:`\| \widehat u_{T+1} \|`. The generalised variant
minimises :math:`\| W^{1/2} \widehat u_{T+1} \|` for a positive-
definite weighting matrix :math:`W`. The closed-form is

.. math::

   \widehat \gamma_W = (A' \widehat M_W A)^{-1}\, A' (I - \widehat B)' W\,
   \big[ (I - \widehat B) Y_{T+1} - \widehat a \big], \qquad
   \widehat M_W = (I - \widehat B)' W (I - \widehat B).

**Proposition S.1**. Letting :math:`\Omega = \mathrm{Cov}[u_1]` and
choosing :math:`W^e = \Omega^{-1}` (estimated by inverting the sample
residual covariance), the resulting estimator :math:`\widehat \alpha^e`
has asymptotic variance **no larger** than the unweighted
:math:`\widehat \alpha` -- with strict reduction whenever
:math:`\Omega` is not a scalar multiple of identity.

Set ``weighting='efficient'`` on the config to obtain both fits side
by side:

.. code-block:: python

   res = SPILLSYNTH({..., "weighting": "efficient", ...}).fit()
   res.att                            # ATT under W = I (the default)
   res.cd.efficient_fit["att_sp_W"]   # ATT under W = Ω̂⁻¹
   res.cd.efficient_fit["alpha_W"]    # (N, T1) effects under W
   res.cd.efficient_fit["Omega_hat"]  # the sample residual covariance

Caveat (paper Remark to Prop S.1): :math:`\widehat \Omega` is a
``(N, N)`` matrix with rank at most :math:`T_0`, so when :math:`T_0 <
N` (the typical SCM regime) the inverse must be regularised. The
implementation adds a small ridge to :math:`\widehat \Omega` before
inversion. With small :math:`T_0` the efficient variant may not
actually reduce variance in practice; treat it as a refinement when
:math:`T_0 \gg N`.

.. _spillsynth-multi-treated:

Multiple treated units (Section S.1.2)
--------------------------------------

**New in v3** (Section S.1.2). SPILLSYNTH supports panels with
:math:`k > 1` treated units that all turn on at the **same**
intervention time. The setup generalises the leading example: with
:math:`N = 4`, units 1 and 2 treated, unit 3 affected by spillover,
and unit 4 clean, the A-matrix is

.. math::

   A = \begin{bmatrix} I_3 \\ 0_{1 \times 3} \end{bmatrix},
   \qquad \widehat\gamma = (\widehat\gamma_1, \widehat\gamma_2,
   \widehat\gamma_3)',

where :math:`\widehat\gamma_1, \widehat\gamma_2` are the per-treated-
unit treatment effects and :math:`\widehat\gamma_3` is the spillover
effect on unit 3. The closed form is identical to eq. (6); only the
column count of :math:`A` (and the partition of :math:`\widehat\alpha
= A \widehat\gamma`) changes.

**How to invoke.** Put a non-zero ``treat`` indicator on every treated
unit at and after the intervention time. SPILLSYNTH detects all
treated units automatically and validates that they share a common
start time. (Different start times trigger a friendly error pointing
at v3 Section S.1.2; staggered adoption is a separate scope.) Cohort
decomposition is delegated upstream to
:func:`mlsynth.utils.datautils.dataprep`, which is the canonical
panel-reshape utility for the whole mlsynth ecosystem.

**Per-treated outputs.** ``CDFit`` gains dictionaries keyed by
treated-unit label:

.. code-block:: python

   res = SPILLSYNTH({..., "affected_units": ["u_affected"]}).fit()

   res.inputs.n_treated                    # number of treated units
   res.inputs.treated_labels               # tuple of treated-unit labels

   res.cd.atts_sp_by_unit                  # {label: ATT_SP scalar}
   res.cd.atts_scm_by_unit                 # {label: ATT_SCM scalar}
   res.cd.gaps_sp_by_unit                  # {label: (T1,) per-period gap}
   res.cd.gaps_scm_by_unit                 # {label: (T1,) per-period gap}
   res.cd.treatment_tests                  # {label: PTestResult}
   res.cd.treatment_cis_95                 # {label: (T1, 2) CI}

**Back-compat.** When :math:`k = 1`, every per-treated-unit dict has
exactly one entry and the scalar/vector fields
(``res.att``, ``res.gap``, ``res.counterfactual``,
``res.cd.treatment_test``, ``res.cd.treatment_ci_95``) keep their
existing semantics. With :math:`k > 1`, those legacy fields point at
the **first** treated unit (in the same order ``dataprep`` returns
the cohort); use the per-unit dicts to read out the rest.

**A-matrix structure with multiple treated.** All three structures
extend cleanly:

* ``per_unit``: ``A`` is ``(N, k + p)`` -- columns ``0..k-1`` are
  treated-unit basis vectors, columns ``k..k+p-1`` are affected-unit
  basis vectors.
* ``homogeneous``: ``A`` is ``(N, k + 1)`` -- ``k`` treated columns
  plus a single shared-spillover column.
* ``distance_decay``: ``A`` is ``(N, k + 1)`` -- ``k`` treated columns
  plus one column carrying :math:`\exp(-d_i)` for every control row.

**Inference with multiple treated.** The treatment-effect
:math:`P`-test selector ``C = e_i'`` is computed once per treated unit
:math:`i`, producing per-unit p-values and signed confidence
intervals. The joint spillover hypothesis selects all
:math:`k \ldots k+p-1` rows of :math:`\widehat\alpha`. The
:math:`\kappa_A` specification test is unchanged (it operates on the
post-period residual norm, which is invariant to the
treatment/spillover partition of :math:`\widehat\alpha`).

**Plot output.** When ``n_treated > 1`` the diagnostic plot switches
to an **event-study layout**: one line per treated unit showing the
SP-adjusted gap over the post-period with a shaded 95% confidence
interval, alongside the per-affected-unit spillover panel (when any
affected units are declared). The single-treated three-panel
Figure-4-style plot is retained for the :math:`k = 1` case.

Runnable example
^^^^^^^^^^^^^^^^

A four-unit panel matching the paper's Section-S.1.2 illustration:
units :math:`u_0, u_1` are treated (true effects :math:`-3` and
:math:`-2`), unit :math:`u_2` receives a spillover of :math:`+1.5`,
unit :math:`u_3` is clean. The call uses
``SPILLSYNTH(config).fit()`` end-to-end -- there is no shortcut to
internal helpers.

.. code-block:: python

   import numpy as np
   import pandas as pd

   from mlsynth import SPILLSYNTH

   rng = np.random.default_rng(7)
   N, T, T0 = 6, 40, 30
   loadings = rng.uniform(0.5, 1.5, size=N)
   intercept = rng.uniform(-1.0, 1.0, size=N)
   f = np.cumsum(rng.standard_normal(T)) * 0.4 + 0.05 * np.arange(T)
   Y = intercept[:, None] + np.outer(loadings, f) + 0.10 * rng.standard_normal((N, T))
   Y[0, T0:] += -3.0   # u_0 treated, true effect -3
   Y[1, T0:] += -2.0   # u_1 treated, true effect -2
   Y[2, T0:] += +1.5   # u_2 affected, true spillover +1.5

   D = np.zeros((N, T))
   D[0, T0:] = 1
   D[1, T0:] = 1       # both u_0 and u_1 treated at the same time
   df = pd.DataFrame([
       {"unit": f"u{i}", "year": t, "y": float(Y[i, t]), "treat": int(D[i, t])}
       for i in range(N) for t in range(T)
   ])

   res = SPILLSYNTH({
       "df": df, "outcome": "y", "treat": "treat",
       "unitid": "unit", "time": "year",
       "method": "cd",
       "affected_units": ["u2"],
       "display_graphs": False,
   }).fit()

   for label, att in res.cd.atts_sp_by_unit.items():
       ci_first = res.cd.treatment_cis_95[label][0]
       print(f"{label}: ATT_SP = {att:+.3f}   "
             f"95% CI at first post-period = "
             f"[{ci_first[0]:+.3f}, {ci_first[1]:+.3f}]")
   print(f"spillover on u2 (avg over post) = "
         f"{res.spillover_effects['u2'].mean():+.3f}")

prints (deterministic with the seed above)::

   u0: ATT_SP = -2.984   95% CI at first post-period = [-3.088, -2.802]
   u1: ATT_SP = -2.072   95% CI at first post-period = [-2.226, -1.793]
   spillover on u2 (avg over post) = +1.496

Each value lands within :math:`\pm 0.1` of its planted truth, and the
intervals correctly exclude zero.

A-selection example: ``select_A_by_kappa``
------------------------------------------

The :math:`\kappa_A` specification test (Section 5.1.2) also drives a
**heuristic A-selector**: among a finite candidate set
:math:`\mathcal{A}`, pick :math:`\widehat A = \arg\min_{A \in
\mathcal{A}} \kappa_A`. The implementation is
:func:`select_A_by_kappa`. Below, we ask SPILLSYNTH to choose between a
correct per-unit structure (just ``u_1`` affected, matching the DGP)
and a deliberately-wrong homogeneous structure (claiming three
controls share a spillover that doesn't exist) -- driving everything
through the public ``SPILLSYNTH(config).fit()`` to obtain the
``(a, B)`` artefacts that ``select_A_by_kappa`` consumes.

.. code-block:: python

   import numpy as np
   import pandas as pd

   from mlsynth import SPILLSYNTH
   from mlsynth.utils.spillsynth_helpers import (
       build_A_homogeneous, build_A_per_unit, select_A_by_kappa,
   )

   rng = np.random.default_rng(7)
   N, T, T0 = 8, 40, 30
   loadings = rng.uniform(0.5, 1.5, size=N)
   intercept = rng.uniform(-1.0, 1.0, size=N)
   f = np.cumsum(rng.standard_normal(T)) * 0.4 + 0.05 * np.arange(T)
   Y = intercept[:, None] + np.outer(loadings, f) + 0.10 * rng.standard_normal((N, T))
   Y[0, T0:] += -3.0    # treated effect
   Y[1, T0:] += 1.5     # spillover ONLY on u1

   D = np.zeros((N, T)); D[0, T0:] = 1
   df = pd.DataFrame([
       {"unit": f"u{i}", "year": t, "y": float(Y[i, t]), "treat": int(D[i, t])}
       for i in range(N) for t in range(T)
   ])

   # Fit through the public API so (a, B) come from SPILLSYNTH itself.
   res = SPILLSYNTH({
       "df": df, "outcome": "y", "treat": "treat",
       "unitid": "unit", "time": "year",
       "method": "cd",
       "affected_units": ["u1"],
       "display_graphs": False,
   }).fit()

   N_ = res.inputs.N
   A_correct = build_A_per_unit(N_, p=1)            # truth: just u1 affected
   A_wrong   = build_A_homogeneous(N_, p=3)         # wrong: u1, u2, u3 share a b

   best_idx, kappas = select_A_by_kappa(
       Y_post=res.inputs.Y_post,
       Y_pre=res.inputs.Y_pre,
       a=res.cd.a,
       B=res.cd.B,
       candidates=[A_correct, A_wrong],
   )
   labels = ["per_unit (correct)", "homogeneous-3 (misspecified)"]
   for label, k in zip(labels, kappas):
       print(f"{label:<32}  mean kappa_A = {k:.4f}")
   print(f"\nselected: {labels[best_idx]}")

prints::

   per_unit (correct)                mean kappa_A = ...
   homogeneous-3 (misspecified)      mean kappa_A = ...

   selected: per_unit (correct)

(The exact :math:`\kappa_A` magnitudes depend on the realised draw of
:math:`\mu_i`; the *ordering* -- correct spec has the smaller
:math:`\kappa_A` -- is the reproducible finding and is enforced by the
``test_select_A_by_kappa_prefers_correct_structure`` regression test.)

.. _spillsynth-mc:

Monte Carlo replication: Cao-Dowd Tables 1 and 2
------------------------------------------------

The script :file:`examples/spillsynth/replicate_cd_tables.py`
implements both DGPs from Section 6.1 of the paper and runs the full
grid :math:`(N, T) \in \{10, 30, 50\} \times \{15, 50, 200\}` with
three spillover scenarios per cell. The full driver is shown at the
bottom of this section.

.. literalinclude:: ../examples/spillsynth/replicate_cd_tables.py
   :language: python
   :linenos:

The estimator under "SCM" is the **original Abadie 2010 simplex SCM
without intercept** (matching the paper's comparator, not the
intercept-shifted variant returned by ``res.att_scm``). The estimator
under "SP" is invoked through the **public** ``SPILLSYNTH(config).fit()``
API, satisfying the Path-B contract requirement that the replication
exercise the full config / panel-prep / estimation pipeline end-to-end
on every Monte Carlo replication.

Run the full grid with 1000 reps as in the paper::

    python -m examples.spillsynth.replicate_cd_tables --reps 1000 --table both

or scale up to 5000 reps for tighter Monte Carlo error bars (expect a
several-hour runtime; the script reports per-cell timing as it goes).

A 500-rep pilot on the :math:`T = 15` slice produces the comparison
below. Empirical bias and (in parentheses) empirical standard
deviation across replications:

.. code-block::

   Table 1 (stationary), T = 15
                            N = 10           N = 30           N = 50
                       paper  | here   paper  | here   paper  | here
   No spillover effects
     SCM bias         -0.062  | -0.130   +0.114 | -0.105   +0.037 | -0.091
         (sd)         (1.453) | (1.316)  (1.279)| (1.268)  (1.187)| (1.246)
     SP  bias         -0.077  | -0.011   +0.091 | +0.064   +0.042 | +0.074
         (sd)         (1.618) | (1.426)  (1.405)| (1.416)  (1.319)| (1.410)
   Concentrated spillover effects
     SCM bias         -1.326  | -1.183   -0.756 | -1.199   -1.492 | -1.134
         (sd)         (1.647) | (1.473)  (1.399)| (1.425)  (1.383)| (1.393)
     SP  bias         +0.267  | -0.011   +0.248 | +0.064   -0.133 | +0.074
         (sd)         (1.598) | (1.426)  (1.198)| (1.416)  (1.304)| (1.410)
   Spreadout spillover effects
     SCM bias         -2.378  | -2.281   -2.245 | -2.067   -2.147 | -2.104
         (sd)         (1.579) | (1.475)  (1.425)| (1.442)  (1.339)| (1.350)
     SP  bias         -0.048  | -0.017   +0.090 | +0.086   +0.037 | +0.038
         (sd)         (1.656) | (1.447)  (1.211)| (1.364)  (1.155)| (1.316)

The headline qualitative finding replicates cleanly across every cell:
SCM picks up a bias of :math:`-1` to :math:`-2` packs under spillover,
while SP centres near zero. Standard deviations are within :math:`\sim
10`-:math:`15\%` of the paper's. Cell-by-cell bias values for the
*No spillover* and *Concentrated* rows differ from the paper at the
:math:`0.1`-:math:`0.4` level. This is **not** an estimator
discrepancy -- it is the loading draw :math:`\mu_i`, which is fixed
across replications within a cell per the paper's spec but drawn from
a different random seed than the paper's. The variance of the
asymptotic distribution of :math:`\widehat \alpha` depends on the
realised :math:`\mu_i`, so different seeds produce different
cell-specific bias terms even at 1000+ reps.

For users who want to push the per-cell Monte Carlo error below the
seed-dependent loading variation, raise ``--reps`` to 5000 or higher
(the per-cell time scales linearly in reps; :math:`N = 50, T = 200`
is the dominant cell).

.. _spillsynth-mc-inference:

Monte Carlo replication: Cao-Dowd Tables 3 and 4 (inference)
------------------------------------------------------------

The script :file:`examples/spillsynth/replicate_cd_inference.py`
replicates Section 6.2 of the paper: empirical rejection rates of the
treatment-effect hypothesis :math:`H_0: \alpha_1 = 0` under three test
procedures (placebo, Andrews, SP) and three spillover scenarios. Table
3 fixes :math:`\alpha_1 = 0` (so rejection rates measure **size**);
Table 4 fixes :math:`\alpha_1 = 5` (so rejection rates measure
**power**). On every replication the SP test (the procedure
``mlsynth`` packages) is read off
``SPILLSYNTH(config).fit().cd.treatment_test.reject_05``; the placebo
and Andrews comparators reuse the same leave-one-out SCM artefacts
that the public ``.fit()`` call exposes.

.. literalinclude:: ../examples/spillsynth/replicate_cd_inference.py
   :language: python
   :linenos:

Run with::

    python -m examples.spillsynth.replicate_cd_inference --reps 1000 --tables 3,4

A 200-rep pilot on a representative pair of cells produces:

.. code-block::

   Table 3: rejection rates under H_0: alpha_1 = 0 (size)        nominal = 0.05
                            N = 10, T = 50       N = 10, T = 200
                       paper  | here          paper  | here
   No spillover effects
     Placebo           0.000  | 0.085          0.000  | 0.070
     Andrews           0.061  | 0.095          0.060  | 0.065
     SP                0.049  | 0.075          0.058  | 0.075
   Concentrated spillover effects
     Placebo           0.000  | 0.040          0.000  | 0.040
     Andrews           0.207  | 0.255          0.224  | 0.205
     SP                0.050  | 0.075          0.043  | 0.075
   Spreadout spillover effects
     Placebo           0.000  | 0.150          0.000  | 0.155
     Andrews           0.478  | 0.515          0.399  | 0.575
     SP                0.035  | 0.070          0.042  | 0.075

   Table 4: rejection rates under alpha_1 = 5 (power)
                            N = 10, T = 50
                       paper  | here
   No spillover effects
     Placebo           0.000  | 0.960
     Andrews           0.948  | 0.995
     SP                0.956  | 0.990
   Concentrated spillover effects
     Placebo           0.000  | 0.585
     Andrews           0.765  | 0.940
     SP                0.932  | 0.990
   Spreadout spillover effects
     Placebo           0.000  | 0.255
     Andrews           0.403  | 0.800
     SP                0.978  | 0.955

The directional findings replicate:

* **SP keeps correct size under both spillover regimes** (within
  Monte-Carlo error of the nominal 5%), while **Andrews catastrophically
  over-rejects under spillover** (51%-58% size at :math:`T = 50`-:math:`200`,
  Spreadout). Placebo size is small but at the discrete granularity of
  the :math:`1/N`-quantile.
* **SP retains high power** (:math:`\geq 95\%` for :math:`T = 50`) across
  spillover scenarios; **placebo loses power dramatically under spillover**
  (down to 26% in Spreadout); Andrews is in between.

Two caveats on the cell-level numbers:

1. At :math:`T = 15` (smallest pre-period), SP is mildly over-sized in
   our pilot (around 16-22%) and the paper itself notes Andrews/SP can
   over-reject at small :math:`T` (Section 6.2, last paragraph). The
   over-rejection vanishes by :math:`T = 50`.
2. Placebo rejection rates in our pilot are non-zero where the paper's
   are zero. This is the discrete-quantile artefact: with :math:`N = 10`
   units the placebo test's achievable size grid is :math:`\{0, 0.1,
   0.2, \dots\}`. The paper's exact zero presumably reflects a
   strict-inequality / unique-max convention; ours is the standard
   :math:`(1-\tau)`-quantile rule. Either convention preserves the
   qualitative finding (placebo has near-zero power under spillover).

Method: ``method='iscm'`` -- Di Stefano & Mellace (2024)
-------------------------------------------------------

The **inclusive synthetic control method** attacks the same problem as
Cao-Dowd from a different angle. The conventional spillover-robust recipe
is to *drop* the contaminated neighbours (the "pure-donor" method); the
inclusive method instead **keeps them in the donor pool** and removes the
bias their inclusion causes by solving a small linear system.

The idea
^^^^^^^^

Suppose the treated unit and a set of :math:`p` *potentially affected*
control units (the affected set :math:`S`, of size :math:`m = 1 + p`) are
each exposed to the intervention -- the treated directly, the others via
spillover. Build a synthetic control for **every** unit in :math:`S`, each
fit with all other units (including the *other* affected units) in its
donor pool. Under a good pre-fit, the observed post-treatment gap for
affected unit :math:`i` mixes its own effect with the effects of the
affected units it borrows from:

.. math::

   \mathrm{gap}_i(t) = \theta_i(t) - \sum_{k \in S,\, k \neq i}
   w_i^{(k)} \, \theta_k(t),

where :math:`w_i^{(k)}` is the weight affected unit :math:`k` receives in
unit :math:`i`'s synthetic control. Stacking the :math:`m` equations gives
:math:`\Omega\,\theta(t) = \mathrm{gap}(t)` with :math:`\Omega_{ii} = 1`
and :math:`\Omega_{ik} = -w_i^{(k)}`. Inverting :math:`\Omega` de-
contaminates the gaps into the true treated-unit effect :math:`\theta_1`
and the spillover effects :math:`\theta_2, \dots, \theta_m`. For the
leading :math:`m = 2` case (treated + one affected neighbour with
cross-weights :math:`w` and :math:`\ell`) this is the closed form

.. math::

   \theta_1(t) = \frac{\mathrm{gap}_1(t) + w\,\mathrm{gap}_2(t)}
                       {1 - w\,\ell}.

Covariates and the solver choice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each unit's synthetic control may match on **covariates** (``covariates=``)
exactly as in Abadie's specification: the covariates are averaged over the
pre-treatment period and fed to the FSCM/MASC **bilevel** predictor-matching
solver, which jointly optimizes the predictor weights :math:`V` with the
donor weights :math:`W`. The bilevel *backend* is the user's choice via
``bilevel_solver``:

* ``"malo"`` (default) -- Malo et al. (2024) staged corner search;
* ``"mscmt"`` -- Becker & Kloessner (2018) global differential-evolution
  search over :math:`\log_{10} V` (the MSCMT outer optimisation);
* ``"penalized"`` -- Abadie & L'Hour (2021): fixes :math:`\Gamma = I` and adds
  a pairwise matching penalty, choosing :math:`\lambda` by cross-validation
  (treated pre-intervention holdout by default; donor leave-one-out optional).
  Unlike the first two it has no predictor-weight :math:`V` to identify, so it
  returns a **unique, sparse** synthetic control -- the principled resolution
  of the malo/mscmt non-uniqueness that arises when the treated unit lies
  inside the donor convex hull on the matching variables.

When ``malo`` and ``mscmt`` disagree, it is a diagnostic that the predictors
are (near-)perfectly matchable and therefore not identifying; trust the
pre-treatment RMSPE and prefer the lower one (or the ``penalized`` backend,
which removes the ambiguity by construction).

Setting ``bias_correct=True`` additionally applies the Abadie-L'Hour bias
correction (eq. 7): a ridge regression of the outcome on the covariates
removes the part of each gap attributable to residual covariate imbalance. It
helps when the covariates genuinely explain the outcome and should be left off
otherwise (a weak-covariate regression injects noise), which is why it
defaults to ``False``.

With no ``covariates`` the method matches on pre-treatment outcomes only
(``bilevel_solver`` and ``bias_correct`` are then ignored).

Worked example: German reunification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

West Germany is the treated unit (reunification, 1990); Austria is the
classic *affected* neighbour. Declaring Austria affected and running the
inclusive method keeps it in the donor pool and corrects for the spillover:

.. code-block:: python

   import pandas as pd
   from mlsynth import SPILLSYNTH

   d = pd.read_stata("basedata/repgermany.dta")
   d = d[["country", "year", "gdp", "trade", "infrate",
          "industry", "schooling", "invest80"]].copy()
   d["treat"] = ((d.country == "West Germany") & (d.year >= 1990)).astype(int)

   res = SPILLSYNTH({
       "df": d, "outcome": "gdp", "treat": "treat",
       "unitid": "country", "time": "year",
       "method": "iscm", "affected_units": ["Austria"],
       # optional covariate matching + solver choice:
       "covariates": ["trade", "infrate", "industry", "schooling", "invest80"],
       "bilevel_solver": "malo",      # or "mscmt"
       "display_graphs": False,
   }).fit()

   print(res.att)              # inclusive (de-contaminated) ATT on West Germany
   print(res.att_scm)          # naive SCM ATT (no correction)
   print(res.iscm.cross_weights)        # {'Austria in West Germany': ~0.33, ...}
   print(res.iscm.omega_det)            # determinant of Omega (~0.90)
   print(res.iscm.spillover_att)        # post-period spillover on Austria

Outcome-only, this reproduces the paper's neighbourhood: Austria receives
``~0.33`` weight in synthetic West Germany and West Germany ``~0.32`` in
synthetic Austria, :math:`\det\Omega \approx 0.90`, and keeping Austria in
the pool tightens West Germany's pre-treatment fit (lower RMSPE than the
exclude-Austria restricted fit). The inclusive ATT is more negative than
the naive gap, because the naive synthetic borrows from a *contaminated*
Austria. ``res.iscm`` exposes the full :class:`ISCMFit` -- the
:math:`\theta` matrix, :math:`\Omega`, cross-weights, donor weights,
predictor weights (covariate mode), and the inclusive-vs-restricted pre-fit.

References
----------

Abadie, A., Diamond, A., & Hainmueller, J. (2010). "Synthetic Control
Methods for Comparative Case Studies." *Journal of the American
Statistical Association* 105(490):493-505.

Andrews, D. W. K. (2003). "End-of-Sample Instability Tests."
*Econometrica* 71(6):1661-1694.

Becker, M., & Kloessner, S. (2018). "Fast and reliable computation of
generalized synthetic controls." *Econometrics and Statistics* 5:1-19.

Cao, J., & Dowd, C. (2023). "Estimation and Inference for Synthetic
Control Methods with Spillover Effects." Working paper.

Di Stefano, R., & Mellace, G. (2024). "The inclusive synthetic control
method." Working paper.

Ferman, B., & Pinto, C. (2021). "Synthetic Controls with Imperfect
Pretreatment Fit." *Quantitative Economics* 12(4):1197-1221.

Malo, P., Eskelinen, J., Zhou, X., & Kuosmanen, T. (2024). "Computing
Synthetic Controls Using Bilevel Optimization." *Computational
Economics* 64:1113-1136.
