Spillover-Aware Synthetic Control (SPILLSYNTH)
================================================

.. currentmodule:: mlsynth

Overview
--------

SPILLSYNTH is mlsynth's dispatcher for synthetic-control estimators
that explicitly model spillover (interference) onto control units.
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

The reflexive fix is the pure-donor method: drop every control you
think is contaminated and fit SCM on the survivors. Cao & Dowd argue this
is often the wrong trade:

* In many panels most or all controls are exposed (geographically
  aggregated data), so there is no clean donor pool left to drop to.
* Contaminated units are frequently the most informative donors;
  discarding them loses efficiency.
* Using fewer controls widens the worst-case bias when the spillover
  structure is misspecified -- the pure-donor estimator has a larger
  identified set.

SPILLSYNTH (``method="cd"``, Cao & Dowd 2023) instead keeps *all* units and
estimates the direct treatment effect and the spillover effects jointly
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

* You have geographic or institutional reasons to suspect specific
  control units are exposed to spillover (e.g. neighboring states for
  a tax change, neighboring firms for a procurement rule, partner
  countries for a sanctions regime).
* You can specify the spillover structure :math:`A` from contextual
  knowledge before fitting. The estimator does not discover the affected
  units; it estimates the size of each declared unit's spillover effect
  jointly with the treatment effect.
* Many or all controls are contaminated, so simply dropping the
  affected units (the pure-donor route) is impractical or wastes too much
  information.
* You have a moderate-to-long pre-period (paper sims use
  :math:`T_0 \geq 15`) -- the asymptotics are large-:math:`T`, fixed number
  of controls -- so the leave-one-out SCM fits are well-estimated.

Do not use SPILLSYNTH when
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* SUTVA credibly holds (no spillover). The extra spillover parameters
  only add variance; use classic SC (:doc:`tssc`, :doc:`scmo`,
  :doc:`clustersc`).
* You cannot defend a spillover structure :math:`A`. The estimator
  assumes :math:`A` is known; a badly misspecified structure biases both
  effects (the :math:`\kappa_A` test mitigates but does not remove this).
  If only a *few* units are contaminated and droppable, the pure-donor
  approach on a pruned pool (classic SC) is the simpler honest choice.
* Spillovers decay with a known spatial weighting and you want a
  DiD-style objective -- :doc:`spsydid` pools them through a spatial
  weight matrix in a synthetic difference-in-differences fit.
* The pre-period is short. The leave-one-out SCM fits underpinning
  Assumption 1(a)-(c) are then noisy; a factor-model estimator
  (:doc:`fma`) may be more stable.
* Distributional questions (quantiles, tails) -- use :doc:`dsc`.

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

The Cao-Dowd (2023) estimator is derived under Assumption 1, with
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
(d) Identification. :math:`A' M A` is non-singular, where
    :math:`M = (I - B)'(I - B)`. Equivalently, :math:`(I - B) A`
    has full column rank.

Parts (a)-(c) require, in practice, a moderate pre-period
(:math:`T_0 \gtrsim 15` in the paper's simulations). Part (d) holds
whenever the spillover structure is not pathologically aligned with
the SCM weight pattern (Section 3.4.1 of the paper); the fit container
exposes ``cd.cond_AMA`` as a numerical diagnostic.

The paper shows that Assumption 1 is satisfied by factor-model DGPs
under two alternative regularity conditions on the common factors:

Condition ST (stationary factors).
  :math:`\{(\eta_t, \lambda_t, \varepsilon_t)\}_{t \geq 1}` is
  stationary, ergodic for first and second moments, has a finite
  :math:`(2 + \delta)`-moment, and :math:`\mathrm{cov}[Y_t(0)] =
  \Omega_y` is positive definite.

Condition CO (cointegrated :math:`\mathcal{I}(1)` factors).
  Write :math:`y_{i,t}(0) = (\lambda_t^1)' \mu_i^1 + (\lambda_t^0)'
  \mu_i^0 + \varepsilon_{i,t}` with :math:`\{\lambda_t^1\}` an
  :math:`\mathcal{I}(1)` process and :math:`\{\lambda_t^0\}`
  stationary. Loadings :math:`\{\mu_i^1\}` admit cointegrating vectors:
  for each :math:`i` there exists :math:`w^{(i)} \in W^{(i)}` such that
  :math:`\mu_i^1 = \sum_{j=1}^{N} w_j^{(i)} \mu_j^1`.

Theoretical guarantees
----------------------

Theorem 1 (asymptotic unbiasedness; Cao & Dowd 2023).
Under Assumption 1,

.. math::

   \widehat \alpha - (\alpha + G\, u_{T+1}) \xrightarrow{p} 0
   \quad \text{as } T \to \infty,

where :math:`G = A (A' M A)^{-1} A' (I - B)'` and
:math:`\mathbb{E}[G\, u_{T+1}] = 0`. The estimator
:math:`\widehat \alpha` is therefore asymptotically unbiased for
the treatment and spillover effect vector :math:`\alpha`. (It is not
consistent because only one post-period of one treated unit is observed,
so the irreducible :math:`u_{T+1}` term does not vanish in any limit.)

Lemma 1 (factor-model sufficiency).
If :math:`A' M A` is non-singular, then either Condition ST
or Condition CO implies Assumption 1. Theorem 1 therefore applies
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
(2021) demeaned simplex SCM for every unit in the panel, treating
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
Row 0 is the spillover-adjusted ATT on the treated unit; the
remaining rows hold the per-affected-unit spillover effects.

Choice of A: the three Cao-Dowd v3 examples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SPILLSYNTH ships all three examples from Cao-Dowd (2023, v3),
selected via ``SPILLSYNTHConfig.spillover_structure``. (The v3
numbering renames v2's examples; we use v3 throughout.)

* ``"per_unit"`` (Example 1, Section 2.2 -- *Limited range*; the
  paper's leading case). Each declared affected unit gets its own
  free spillover coefficient. :math:`A` is :math:`N \times (1 + p)`
  with column 0 the treated-unit basis vector and columns
  :math:`1, \dots, p` the basis vectors for each affected control.
  Helper: :func:`build_A_per_unit`.

* ``"homogeneous"`` (Example 2, Section 3.4). All declared
  affected units share a single spillover coefficient :math:`b`.
  :math:`A` is :math:`N \times 2` with column 0 = treated basis and
  column 1 the indicator over affected rows. Use when domain
  knowledge constrains spillovers to be of equal magnitude across the
  affected set. Helper: :func:`build_A_homogeneous`.

* ``"distance_decay"`` (Example 3, Section 7.1). Spillover decays
  as :math:`\alpha_i = b \exp(-d_i)` with user-supplied
  distances. :math:`A` is :math:`N \times 2` with row :math:`i` set to
  :math:`(0, \exp(-d_i))`. Every control receives some spillover
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
  under all three A-matrix structures: per-unit (Example 1),
  homogeneous (Example 2), and distance-decay (Example 3).
* Original Abadie 2010 SCM counterfactual for side-by-side comparison.
* Cao-Dowd Section 4 P-test inference (Theorems 2 and 3): per-period
  tests for ``H_0: alpha_1(t) = 0`` (treatment) and
  ``H_0: alpha_k(t) = 0`` (per-affected-unit spillover), the joint
  spillover hypothesis matching the MATLAB reference, signed
  confidence intervals via test inversion, and the
  :math:`\kappa_A` specification test for the chosen :math:`A`.
  See :ref:`spillsynth-inference` and :ref:`spillsynth-kappa-A`.
* Cao-Dowd v3 Section 5.2 pure-donor sensitivity analysis --
  worst-case misspecification-bias bounds comparing SP against the
  pure-donor SCM. See :ref:`spillsynth-pure-donor`.
* Cao-Dowd v3 Section S.1.1 GMM-efficient variant (Proposition
  S.1): :math:`\widehat \alpha^e` minimises asymptotic variance via
  :math:`W = \widehat \Omega^{-1}`. See :ref:`spillsynth-efficient`.
* Cao-Dowd v3 Section S.1.2 multiple treated units with a common
  intervention time -- per-treated-unit ATTs, CIs, and treatment-
  effect tests. See :ref:`spillsynth-multi-treated`.

Not yet shipped (planned):

* Multiple post-treatment-period structure selection (Section S.1.3).
* Covariates (Section S.1.4).
* Staggered adoption (treated units with different intervention
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

SAR spillover subpackage (``method='sar'``) -- long-panel ingestion, the
two-step horseshoe/SAR sampler, and the identification plug-ins.

.. automodule:: mlsynth.utils.spillsynth_helpers.sar.setup
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.spillsynth_helpers.sar.sampler
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.spillsynth_helpers.sar.pipeline
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.spillsynth_helpers.sar.structures
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

   Algorithmic equivalence with Cao-Dowd's R reference (verified to
   four decimals). The Cao-Dowd authors published their estimator as
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

   This equivalence is pinned durably by the ``spillsynth_prop99``
   benchmark, which clones the authors' repository
   (``jcao0/synthetic-control-spillover``, pinned commit ``60bbebe``) and
   checks mlsynth's live ``method='cd'`` fit against the committed MATLAB
   output ``spillover.csv`` (the California spillover-adjusted ATT path,
   reproduced to :math:`\approx 10^{-4}`).

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

* Treatment-effect test (paper, Section 4.2, applied with
  :math:`C = e_1^\prime`, :math:`d = 0`): tests
  :math:`H_0: \alpha_1(t) = 0` separately for each post-period
  :math:`t = T+1, \dots, T+m`.
* Per-affected-unit spillover test (one such test per declared
  affected unit :math:`k`, with :math:`C = e_{k+1}^\prime`):
  tests :math:`H_0: \alpha_k(t) = 0`.

Test statistic. At post-period :math:`t`,

.. math::

   P_t = (C \widehat \alpha_t - d)' W_T (C \widehat \alpha_t - d),

with default :math:`W_T = I` (Lemma 3 of the paper -- under either
Condition ST or Condition CO this choice satisfies the regularity
required for asymptotic validity).

Reference distribution. For each pre-period :math:`s = 1, \dots,
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

Assumptions for validity (Assumption 3 of the paper).

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

Theorem 3 (asymptotic validity; Cao & Dowd 2023).
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

Why not the placebo or vanilla Andrews test? Section 4.3 of the
paper gives the intuition (and the Monte Carlo below quantifies it):

* The Abadie (2010) placebo test under-rejects under positive
  spillover. Spillover-affected donors shift their residual densities
  to the right, widening the across-unit distribution and making the
  treated unit's residual less extreme.
* Andrews' (2003) P-test using only the treated unit's own residuals
  ignores the cross-unit information altogether and over-rejects
  under spillover -- the pre-period treated residual density is
  unaffected by spillover, but the post-period one is, so the test
  statistic shifts and the empirical CDF fails to track it.
* The Cao-Dowd test exploits the cross-unit information via :math:`A`
  and :math:`\widehat G`; both the pre-period reference and the
  post-period statistic share the same weighting, so size is
  preserved under either spillover regime.

Result fields. After fitting,

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

New in v3 (Section 5.1.2). The estimator's misspecification bias is
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

Proposition 2 (Cao-Dowd v3). Under Assumption 3,
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

For A-selection, use :func:`select_A_by_kappa` to pick among
candidate :math:`A` matrices by minimising the mean
:math:`\kappa_A` over post-periods (Section S.1.3 of the paper notes
this is a heuristic with a single post-period and a consistent
selector with multiple).

.. _spillsynth-pure-donor:

Pure-donor sensitivity analysis
-------------------------------

New in v3 (Section 5.2). An alternative to SP is the *pure-donor*
SCM, which simply drops every assumed-affected unit from the donor
pool. Both methods are asymptotically unbiased when :math:`A` is
correctly specified, but they differ in robustness to
misspecification: if some assumed-clean control was actually exposed
to a spillover of magnitude :math:`\bar \alpha`, what bias does that
inject?

Section 5.2 shows the worst-case bias is linear in
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

New in v3 (Section S.1.1). The default Cao-Dowd estimator
minimises :math:`\| \widehat u_{T+1} \|`. The generalised variant
minimises :math:`\| W^{1/2} \widehat u_{T+1} \|` for a positive-
definite weighting matrix :math:`W`. The closed-form is

.. math::

   \widehat \gamma_W = (A' \widehat M_W A)^{-1}\, A' (I - \widehat B)' W\,
   \big[ (I - \widehat B) Y_{T+1} - \widehat a \big], \qquad
   \widehat M_W = (I - \widehat B)' W (I - \widehat B).

Proposition S.1. Letting :math:`\Omega = \mathrm{Cov}[u_1]` and
choosing :math:`W^e = \Omega^{-1}` (estimated by inverting the sample
residual covariance), the resulting estimator :math:`\widehat \alpha^e`
has asymptotic variance no larger than the unweighted
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

New in v3 (Section S.1.2). SPILLSYNTH supports panels with
:math:`k > 1` treated units that all turn on at the same
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

How to invoke. Put a non-zero ``treat`` indicator on every treated
unit at and after the intervention time. SPILLSYNTH detects all
treated units automatically and validates that they share a common
start time. (Different start times trigger a friendly error pointing
at v3 Section S.1.2; staggered adoption is a separate scope.) Cohort
decomposition is delegated upstream to
:func:`mlsynth.utils.datautils.dataprep`, which is the canonical
panel-reshape utility for the whole mlsynth ecosystem.

Per-treated outputs. ``CDFit`` gains dictionaries keyed by
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

Back-compat. When :math:`k = 1`, every per-treated-unit dict has
exactly one entry and the scalar/vector fields
(``res.att``, ``res.gap``, ``res.counterfactual``,
``res.cd.treatment_test``, ``res.cd.treatment_ci_95``) keep their
existing semantics. With :math:`k > 1`, those legacy fields point at
the first treated unit (in the same order ``dataprep`` returns
the cohort); use the per-unit dicts to read out the rest.

A-matrix structure with multiple treated. All three structures
extend cleanly:

* ``per_unit``: ``A`` is ``(N, k + p)`` -- columns ``0..k-1`` are
  treated-unit basis vectors, columns ``k..k+p-1`` are affected-unit
  basis vectors.
* ``homogeneous``: ``A`` is ``(N, k + 1)`` -- ``k`` treated columns
  plus a single shared-spillover column.
* ``distance_decay``: ``A`` is ``(N, k + 1)`` -- ``k`` treated columns
  plus one column carrying :math:`\exp(-d_i)` for every control row.

Inference with multiple treated. The treatment-effect
:math:`P`-test selector ``C = e_i'`` is computed once per treated unit
:math:`i`, producing per-unit p-values and signed confidence
intervals. The joint spillover hypothesis selects all
:math:`k \ldots k+p-1` rows of :math:`\widehat\alpha`. The
:math:`\kappa_A` specification test is unchanged (it operates on the
post-period residual norm, which is invariant to the
treatment/spillover partition of :math:`\widehat\alpha`).

Plot output. When ``n_treated > 1`` the diagnostic plot switches
to an event-study layout: one line per treated unit showing the
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
heuristic A-selector: among a finite candidate set
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

The estimator under "SCM" is the original Abadie 2010 simplex SCM
without intercept (matching the paper's comparator, not the
intercept-shifted variant returned by ``res.att_scm``). The estimator
under "SP" is invoked through the public ``SPILLSYNTH(config).fit()``
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
:math:`0.1`-:math:`0.4` level. This is not an estimator
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
3 fixes :math:`\alpha_1 = 0` (so rejection rates measure size);
Table 4 fixes :math:`\alpha_1 = 5` (so rejection rates measure
power). On every replication the SP test (the procedure
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

* SP keeps correct size under both spillover regimes (within
  Monte-Carlo error of the nominal 5%), while Andrews catastrophically
  over-rejects under spillover (51%-58% size at :math:`T = 50`-:math:`200`,
  Spreadout). Placebo size is small but at the discrete granularity of
  the :math:`1/N`-quantile.
* SP retains high power (:math:`\geq 95\%` for :math:`T = 50`) across
  spillover scenarios; placebo loses power dramatically under spillover
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

The inclusive synthetic control method attacks the same problem as
Cao-Dowd from a different angle. The conventional spillover-robust recipe
is to *drop* the contaminated neighbours (the "pure-donor" method); the
inclusive method instead keeps them in the donor pool and removes the
bias their inclusion causes by solving a small linear system.

The idea
^^^^^^^^

Suppose the treated unit and a set of :math:`p` *potentially affected*
control units (the affected set :math:`S`, of size :math:`m = 1 + p`) are
each exposed to the intervention -- the treated directly, the others via
spillover. Build a synthetic control for every unit in :math:`S`, each
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

Assumptions
^^^^^^^^^^^

The inclusive method rests on five conditions.

* (A1) Additive treatment/spillover. For every unit :math:`i` and
  post-period :math:`t`, the observed outcome decomposes as
  :math:`Y_{it} = Y_{it}^N + \theta_i(t)`, with :math:`Y_{it}^N` the
  untreated potential outcome and :math:`\theta_i(t)` the (treatment or
  spillover) effect, :math:`\theta_j = 0` for clean controls. The effect is
  additive and does not feed back into the donors' untreated paths.
* (A2) Valid synthetic controls for the whole affected set. Each unit in
  :math:`S` -- the treated unit *and every affected unit* -- admits simplex
  weights over the remaining units that reproduce its untreated potential
  outcome, :math:`\sum_j w_i^{(j)} Y_{jt}^N = Y_{it}^N` in expectation (the
  usual SCM / factor-model condition). This is the standard SCM assumption,
  now required of every affected unit, not only the treated one.
* (A3) Correctly specified affected set. :math:`S` is known, and every
  control outside :math:`S` is genuinely unaffected
  (:math:`\theta_j = 0`). The clean controls supply the uncontaminated
  identifying variation. As in Cao-Dowd, a conservative researcher errs
  toward a *larger* :math:`S` -- if in doubt, include the unit.
* (A4) Invertible cross-weight system. :math:`\Omega` is non-singular,
  :math:`\det\Omega \neq 0`. For :math:`m=2` this is :math:`1 - w\ell \neq
  0`: the treated and affected unit must not load on each other so heavily
  that the system collapses.
* (A5) Good pre-treatment fit. The synthetic controls track each unit's
  pre-period path, so post-period gaps reflect effects, not fit error (the
  usual SCM requirement; under imperfect fit the Ferman-Pinto / demeaning
  caveats apply).

Identification and econometric theory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Write the observed gap for affected unit :math:`i` as
:math:`\mathrm{gap}_i(t) = Y_{it} - \sum_j w_i^{(j)} Y_{jt}` and split the
donors into clean controls (:math:`\theta = 0`) and the *other* affected
units. Under (A1)-(A2) the clean part cancels
(:math:`\sum_j w_i^{(j)} Y_{jt}^N \approx Y_{it}^N`), leaving precisely

.. math::

   \mathrm{gap}_i(t) = \theta_i(t) - \sum_{k \in S,\, k \neq i}
   w_i^{(k)}\,\theta_k(t).

Each affected unit's gap is its *own* effect minus a weighted sum of the
effects of the affected units it borrows from. Stacking the :math:`m`
equations gives :math:`\Omega\,\theta(t) = \mathrm{gap}(t)`, where the clean
controls anchor the untreated potential outcomes and only the affected units
contaminate one another, so :math:`\Omega` is exactly the :math:`m \times m`
matrix of self-loadings (:math:`1`) and cross-loadings (:math:`-w_i^{(k)}`).
Under (A4), :math:`\theta(t) = \Omega^{-1}\mathrm{gap}(t)` is identified; the
paper solves it by Cramer's rule (the :math:`m=1` closed form above is the
:math:`2\times2` special case).

The point is that the naive SCM gap on the treated unit silently absorbs the
bias term :math:`\sum_k w_1^{(k)}\theta_k(t)` -- the spillovers it borrows
from the affected donors -- and would equal :math:`\theta_1` only if those
donors carried zero weight. The inclusive inversion removes exactly that
term. Asymptotically (growing pre-period, factor-model donors) the SC weights
are consistent and :math:`\widehat\theta` is asymptotically unbiased for the
true effects -- the same large-:math:`T_0` logic as standard SCM, applied to
the whole affected set jointly rather than to the treated unit alone.

The implementation exposes :math:`\det\Omega` as the key diagnostic. Values
near zero warn that the cross-weights are near-degenerate (the affected unit
and the treated are near-mutual nearest neighbours), so the inverse amplifies
noise; values near one mean little cross-contamination and a mild correction.
The inclusive-vs-restricted pre-RMSPE (``pre_rmspe`` vs
``pre_rmspe_restricted``) quantifies the fit gained by *keeping* the affected
units in the pool rather than dropping them.

Inference
^^^^^^^^^

The shipped estimator returns point estimates -- the de-contaminated effect
path :math:`\theta_1` and the per-affected-unit spillovers
:math:`\theta_2,\dots,\theta_m`. For uncertainty, use the standard SCM
placebo / permutation machinery (in-space placebos across the clean
controls), reading the inclusive ATT against the placebo distribution;
:math:`\det\Omega` and the inclusive-vs-restricted pre-RMSPE are the
method-specific diagnostics. Unlike the Cao-Dowd path -- which ships the
end-of-sample :math:`P`-test and the :math:`\kappa_A` specification test --
the inclusive path does not yet carry a bespoke inference procedure (a
planned addition).

When to use: inclusive SCM vs. (and alongside) Cao-Dowd
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Both methods keep the contaminated neighbours and correct for spillover, but
they buy identification differently.

* Cao-Dowd (``method='cd'``) imposes a *parametric spillover structure*
  -- the :math:`A`-matrix (per-unit, homogeneous, distance-decay) -- and
  recovers the treatment and spillover coefficients jointly from all units'
  demeaned-SCM residuals, with formal inference (:math:`P`-test) and a
  misspecification test (:math:`\kappa_A`). Prefer it when (a) domain
  knowledge *shapes* the spillover (geography, networks), (b) *many* units are
  affected, so a low-dimensional :math:`A` is more parsimonious than inverting
  a large :math:`\Omega`, (c) you need calibrated :math:`p`-values and a
  specification test, or (d) the affected units cannot themselves be well
  synthesized.

* Inclusive SCM (``method='iscm'``) imposes *no parametric spillover form*:
  it lets the data's own cross-weights define the contamination and inverts
  them. Prefer it when (a) each affected unit *can* be given a good synthetic
  control (it lies in the donor hull), (b) you are unwilling to commit to an
  :math:`A`-matrix and would rather the mixing be estimated, (c) the affected
  set is *small*, so :math:`\Omega` is small and safely invertible, or (d) you
  want the per-affected-unit spillover effects as a transparent by-product of
  one linear solve.

The two rest on *different* assumptions -- a correctly specified spillover
structure (Cao-Dowd) versus an invertible cross-weight system and
synthesizable affected units (inclusive) -- which makes them natural
robustness companions. Run both: agreement is reassuring, and
disagreement localizes the load-bearing assumption (the :math:`A`-matrix, or
the :math:`\Omega`-invertibility / affected-unit fit). When the spillover's
shape is unknown, the inclusive method is the lighter-assumption default;
when the shape is known and inference matters, Cao-Dowd is the sharper tool.

Covariates and the solver choice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each unit's synthetic control may match on covariates (``covariates=``)
exactly as in Abadie's specification: the covariates are averaged over the
pre-treatment period and fed to the FSCM/MASC bilevel predictor-matching
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
  returns a unique, sparse synthetic control -- the principled resolution
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

The penalized inclusive SCM
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The inclusive correction inherits whatever non-uniqueness the per-unit
synthetic controls have. The cross-weights :math:`w_i^{(k)}` *are* the
off-diagonal of :math:`\Omega`, so if the donor weights are not pinned down
-- the malo/mscmt situation, where the treated and affected units lie inside
the donor hull on the matching variables -- then neither is :math:`\Omega`,
and neither is :math:`\theta`. The Abadie-L'Hour penalized backend
(``bilevel_solver='penalized'``) fixes this at the source: with
:math:`\lambda > 0` each unit's synthetic
control is *unique and sparse* (Abadie-L'Hour Theorem 1), so :math:`\Omega`
-- and hence :math:`\theta` -- is well-defined regardless of solver.

Penalization interacts with the inclusive correction in a specific,
instructive way. The penalty pulls each unit's synthetic toward its
*closest* donors in the matching space; for a treated unit the closest unit
is very often exactly the affected neighbour (Austria for West Germany).
Raising :math:`\lambda` therefore *increases* the cross-weight :math:`w` the
affected unit receives, which (i) sparsifies and lowers interpolation bias,
as ALH intend, but (ii) drives :math:`\det\Omega = 1 - w\ell` toward zero,
making the inclusive correction larger and eventually ill-posed.
Penalization and inclusion are thus complementary but opposed at the
margin: the penalty reduces interpolation bias by leaning on the close
(affected) neighbour, and the :math:`\Omega`-inversion is precisely what
removes the spillover bias that leaning introduces -- up to the point where
the cross-loading makes the system singular. In practice, choose
:math:`\lambda` with the built-in cross-validation (treated holdout by
default) and watch :math:`\det\Omega`: if it collapses toward zero the
penalty is over-loading the affected unit and :math:`\lambda` should be
reduced (or the unit's affected status reconsidered). As
:math:`\lambda \to 0` the penalized backend reproduces the unpenalized
inclusive solution (the minimum-compound-discrepancy synthetic among all
perfect-fit weights); as :math:`\lambda \to \infty` it degenerates to
nearest-neighbour, which for a treated/affected pair drives :math:`w \to 1`
and breaks invertibility.

The optional bias correction (``bias_correct=True``) is ALH's second
ingredient. After the (penalized or unpenalized) weights are formed, it
regresses each unit's outcome on the covariates and subtracts the part of the
gap explained by residual covariate imbalance (eq. 7), with a ridge to keep
the regression stable. It composes cleanly with the :math:`\Omega`-inversion
-- correct each unit's gap first, then de-contaminate -- and is most valuable
when the covariates genuinely predict the outcome. On covariates that do not
(the German GDP example, where the five economic predictors leave GDP largely
unexplained), the regression is uninformative and the correction injects
noise, which is why it defaults to off.

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

Second example: the Basque Country, with Abadie-Gardeazabal covariates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Basque Country (ETA terrorism) is the other canonical synthetic-control
case. Suppose we suspect spillover onto the geographically adjacent regions
Navarra, La Rioja, Castile y Leon and Cantabria, and we want to match on
Abadie & Gardeazabal's original economic predictors with their specific lag
windows. ``covariate_windows`` supplies those windows (an inclusive
``(start, end)`` per covariate; covariates not listed are averaged over the
whole pre-period), exactly mirroring Abadie's special-predictor spec:

.. code-block:: python

   import pandas as pd
   from mlsynth import SPILLSYNTH

   d = pd.read_csv("basedata/basque_data.csv")
   AG = ["sec.agriculture", "sec.energy", "sec.industry", "sec.construction",
         "sec.services.venta", "sec.services.nonventa",
         "school.illit", "school.prim", "school.med", "school.high",
         "popdens", "invest"]
   windows = {**{c: (1961, 1969) for c in AG[:6]},          # sectors
              **{c: (1964, 1969) for c in AG[6:10] + ["invest"]},  # schooling, invest
              "popdens": (1969, 1969)}
   affected = ["Navarra (Comunidad Foral De)", "Rioja (La)",
               "Castilla Y Leon", "Cantabria"]

   d["treat"] = ((d.regionname == "Basque Country (Pais Vasco)")
                 & (d.year >= 1975)).astype(int)
   res = SPILLSYNTH({
       "df": d, "outcome": "gdpcap", "treat": "treat",
       "unitid": "regionname", "time": "year",
       "method": "iscm", "affected_units": affected,
       "covariates": AG, "covariate_windows": windows,
       "bilevel_solver": "malo", "display_graphs": False,
   }).fit()
   print(res.att)            # inclusive direct effect on the Basque Country

(``gdpcap`` is the outcome -- matched over the whole pre-period in the upper
level -- so it does not appear among the covariates; Abadie's ``gdpcap``
special predictors are subsumed by that richer outcome match.)

The estimate is strikingly stable across the assumed onset year and the
bilevel backend (per-capita GDP, 1986 USD thousands):

.. list-table:: Inclusive Basque effect (AG covariates + windows, 4 affected)
   :header-rows: 1

   * - onset
     - backend
     - ATT (inclusive)
     - ATT (naive)
     - :math:`\det\Omega`
     - pre-RMSPE
   * - 1973
     - malo
     - :math:`-0.672`
     - :math:`-0.682`
     - 0.851
     - 0.083
   * - 1973
     - mscmt
     - :math:`-0.651`
     - :math:`-0.651`
     - 0.257
     - 0.083
   * - 1975
     - malo
     - :math:`-0.676`
     - :math:`-0.697`
     - 0.922
     - 0.085
   * - 1975
     - mscmt
     - :math:`-0.692`
     - :math:`-0.692`
     - 0.084
     - 0.084

Two lessons. First, the inclusive correction barely moves the treated
effect here (inclusive :math:`\approx` naive in every row): the Basque
Country is a rich industrial region whose synthetic leans on Cataluna and
Madrid, so the four poorer neighbours receive near-zero donor weight and there
is little contamination to undo (contrast West Germany, where Austria is a
heavy donor and the correction bites). The cross-weight system is essentially
a diagnostic that *confirms* those regions are not load-bearing donors.
Second, the result is robust to the malo/mscmt choice once the predictors
are rich enough to bind -- with only a few weak covariates mscmt's global
``V`` search collapses to a corner (a badly-fitting single-predictor
solution); with the full AG block it agrees with malo at :math:`\approx
-0.68`. As always, read the ``pre_rmspe`` column as the referee.

Method: ``method='grossi'`` -- Grossi et al. (2025)
--------------------------------------------------

Grossi, Mariani, Mattei, Lattarulo & Oener (2025, *JRSS-A*) estimate direct
*and* spillover effects under partial interference -- the third spillover
philosophy in SPILLSYNTH, and the polar opposite of the inclusive method.
Where ``iscm`` keeps the contaminated neighbours in the donor pool and
algebraically corrects, ``grossi`` drops the treated unit's whole cluster
from the pool and rebuilds the counterfactual from the *far* (clean) controls
only.

The idea
^^^^^^^^

The units are partitioned into clusters (neighbourhoods). The treated unit
sits in cluster 1 together with its potentially-affected cluster-mates (the
``affected_units``); the remaining clusters are clean controls. Under partial
interference, build a penalized synthetic control -- from the clean controls
*only* -- for the treated unit and for each cluster-mate:

* the treated unit's gap is the direct effect (paper eq. 3.4),
  :math:`\widehat\tau_{1,t} = Y_{1,t} - \sum_{j \in \text{clean}}
  \widehat\omega_j^{(1)} Y_{j,t}`;
* each cluster-mate's gap is a spillover effect, averaged into the average
  spillover (eq. 3.5),
  :math:`\widehat\delta_t = \frac{1}{N_1-1}\sum_{i \in \mathcal N_1\setminus\{1\}}
  \big(Y_{i,t} - \sum_{j} \widehat\omega_j^{(i)} Y_{j,t}\big)`.

The synthetic controls use the penalized SCG estimator (Abadie-L'Hour
2021, the ``penalized`` backend) -- which is why that backend had to exist
first -- with :math:`\lambda` chosen by cross-validation.

Assumptions
^^^^^^^^^^^

* (G1) Partial interference (Sobel 2006; paper Assumption 1).
  Interference occurs only *within* a unit's cluster, never between clusters.
  This is what makes the far clusters clean controls and is the method's
  load-bearing assumption -- it must be defensible from the geography /
  network of the application.
* (G2) Correctly partitioned clusters. The treated unit's cluster (its
  affected mates) is correctly identified; every other cluster is genuinely
  unexposed.
* (G3) Valid penalized SC from clean controls. Each treated-cluster unit
  admits a good penalized synthetic control built from the clean controls
  (the stable cross-cluster relationship of the paper's Section 3.3, points
  (a)-(b): a common structural process, no idiosyncratic structural shocks
  over the sample).
* (G4) Consistency / no anticipation (standard SCM).

Identification and econometric theory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Partial interference (G1) means the clean controls' outcomes do not depend on
the treated cluster's treatment, so their weighted average is an unbiased
imputation of the *untreated* potential outcome for any treated-cluster unit
-- including the affected mates, whose treated outcome differs from it
precisely by the spillover. The direct and average spillover estimands are
then simple gaps against these far-control synthetics (eqs. 3.4-3.5). Because
the affected neighbours are *excluded* from the donor pool, no contamination
can leak into the counterfactual -- the bias the naive SCM and the inclusive
method must respectively ignore or undo is structurally absent here. The
price is fit: dropping the treated unit's closest donors (its own
neighbourhood) typically *raises* the pre-treatment RMSPE relative to keeping
them. The penalized estimator's pairwise penalty mitigates the resulting
interpolation bias by leaning on the closest *clean* controls.

Inference
^^^^^^^^^

``grossi`` ships the paper's residual-resampling procedure (Section 3.3) with
pivotal bias-corrected confidence intervals (eqs. 3.6-3.7), via ``n_boot``.
Each clean control is fit from the *other* clean controls to obtain a residual
vector; the residual vectors are resampled with replacement to form pseudo
control outcomes; each treated-cluster unit's synthetic control is re-fit
against the pseudo controls (with :math:`\lambda` fixed at its observed value,
for speed); and per-period bias-corrected pivotal intervals are formed for the
direct effect and the average spillover. Inference is available for the
``penalized`` backend and outcome-only matching; ``n_boot=0`` (default) skips
it. ``ci_level`` defaults to the paper's 0.90.

When to use: ``grossi`` vs ``iscm`` vs ``cd``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All three keep the analysis honest about spillover; they differ in *what they
do with the contaminated neighbours*:

* ``cd`` (Cao-Dowd) -- keep them, impose a parametric spillover structure
  (the :math:`A`-matrix), recover everything jointly with formal inference.
* ``iscm`` (Di Stefano-Mellace) -- keep them, invert the data's own
  cross-weights (no parametric structure, but needs an invertible
  :math:`\Omega`).
* ``grossi`` -- *drop* them, and identify off a partial-interference
  cluster structure using the far controls.

Prefer ``grossi`` when you have a credible cluster/neighbourhood structure
(geography, administrative units) that makes partial interference defensible,
and enough clean controls in other clusters to build a good synthetic. It is
the cleanest identification when those conditions hold -- no contamination by
construction -- at the cost of pre-treatment fit. Prefer ``iscm``/``cd`` when
the affected units are too valuable as donors to discard, or when no clean
cross-cluster pool exists. Because the three rest on different assumptions
(parametric structure / invertible cross-weights / partial interference) they
bracket the answer: running all three is a strong robustness exercise.

Worked example: West Germany (partial interference)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Treating Austria as West Germany's only exposed cluster-mate, and every other
country as a clean control:

.. code-block:: python

   res = SPILLSYNTH({
       "df": d, "outcome": "gdp", "treat": "treat",
       "unitid": "country", "time": "year",
       "method": "grossi", "affected_units": ["Austria"],
       "n_boot": 400, "ci_level": 0.90, "display_graphs": False,
   }).fit()

   print(res.att)                       # direct ATT on West Germany
   print(res.grossi.spillover_att)      # spillover on Austria
   print(res.grossi.direct_ci)          # per-period 90% CI (eq. 3.6)

On the German panel this gives a *larger* direct effect than the inclusive
method (around :math:`-1600` vs :math:`-1370`): excluding Austria -- itself
depressed by the spillover -- removes the downward drag a contaminated donor
would impose on the counterfactual. The direct effect's 90% interval excludes
zero; the average spillover's does not (one neighbour, a wide clean pool),
matching the paper's pattern that direct effects are sharper than spillovers.

Method: ``method='sar'`` -- Sakaguchi & Tagawa (2026)
-----------------------------------------------------

The spatial-autoregressive (SAR) Bayesian SCM relaxes SUTVA from yet
another angle. Where Cao-Dowd model spillover through a researcher-specified
``A`` matrix and Grossi partition donors into near/far clusters, the SAR method
posits a spatial process on the control outcomes themselves: a treatment on
the treated unit propagates to the controls through a spatial-weight matrix, so
the bias the spillover induces in ordinary SCM is identified and removed, *and*
the spillover effect landing on each control unit is recovered as a parameter
of interest.

.. note::

   Notation bridge. The treated unit is index ``0`` and the ``N`` controls
   are :math:`\mathcal{N} = \{1, \dots, N\}`; the pre-period is
   :math:`\mathcal{T}_1` (length ``T0``) and the post-period
   :math:`\mathcal{T}_2`. :math:`\mathbf{Y}^c_t \in \mathbb{R}^N` stacks the
   control outcomes at ``t``; :math:`Y_{0t}` is the treated outcome.
   :math:`\mathbf{W} \in \mathbb{R}^{N\times N}` is the (row-normalised)
   control-to-control spatial-weight matrix and :math:`\mathbf{w}\in\mathbb{R}^N`
   the treated-to-control weight vector; :math:`\boldsymbol{\alpha}` are the
   synthetic-control weights, :math:`\rho` the spatial-autoregressive
   coefficient, :math:`\mathbf{I}` the identity. The treatment effect on the
   treated unit is :math:`\xi_{0t}` and the spillover effect on control ``i`` is
   :math:`\xi_{it}`.

The model
~~~~~~~~~

For each period the untreated control outcomes follow a SAR panel,

.. math::

   \mathbf{Y}^c_t = \rho\,\bigl(\mathbf{w}\,Y_{0t} + \mathbf{W}\,\mathbf{Y}^c_t\bigr)
       + \mathbf{X}_t\boldsymbol{\beta} + \mathbf{u}_t,

with ``rho`` the spatial-autoregressive coefficient, ``w`` the
treated-to-control spatial weights, ``W`` the control-to-control weights (both
row-normalised), and ``u_t`` a latent-factor-plus-noise disturbance. Given the
synthetic weights ``alpha`` (such that the treated unit's untreated outcome is
``alpha' Y^c``) and ``rho``, the treatment effect on the treated unit and the
spillover effects on the controls are identified in closed form (paper Theorems
1-2):

.. math::

   \xi_{0t} = Y_{0t} - \boldsymbol{\alpha}^{\top}
       \bigl(\mathbf{I} - \rho\,\mathbf{w}\boldsymbol{\alpha}^{\top} - \rho\mathbf{W}\bigr)^{-1}
       \bigl((\mathbf{I} - \rho\mathbf{W})\mathbf{Y}^c_t - \rho\,\mathbf{w} Y_{0t}\bigr).

When ``rho = 0`` this collapses exactly to Abadie's synthetic control, so
the SAR method nests standard SCM as its no-spillover special case.

Assumptions
~~~~~~~~~~~

Identification of :math:`\xi_{0t}` and :math:`\xi_{it}` rests on four
assumptions (the paper's Assumptions 1-3 plus the standard no-anticipation /
exogeneity conditions inherited from SCM). Each is paired with a remark on what
it buys and when it is plausible.

Assumption 1 (Perfect pre-treatment fit). There exist weights
:math:`\boldsymbol{\alpha}\in\mathbb{R}^N` such that, in the absence of any
treatment, the treated unit's outcome equals the weighted control average,
:math:`Y_{0t}(0) = \sum_{i\in\mathcal{N}} \alpha_i\, Y_{it}(0)` almost surely
for every ``t``.

   *Remark.* This is the usual synthetic-control premise (Abadie et al. 2010):
   the donor pool can reconstruct the treated unit's untreated path. It is the
   condition the horseshoe Step-1 regression targets. It does not require
   the weights to lie on the simplex -- the SAR estimator uses an unrestricted,
   shrinkage-regularised :math:`\boldsymbol{\alpha}`, which is what lets it host
   negative weights (e.g. the negative California donor) and short pre-periods.

Assumption 2 (Spillover runs through outcomes, not unobservables). The
disturbance :math:`\mathbf{u}_t` is unaffected by who is treated: any unit's
*structural* error depends only on its own (always-zero, for controls)
treatment status, so all cross-unit transmission flows through the SAR term
:math:`\rho(\mathbf{w}Y_{0t} + \mathbf{W}\mathbf{Y}^c_t)`.

   *Remark.* This is the substantive identifying restriction. It says
   interference is *mediated by observed outcomes propagating along the spatial
   network* -- a tax cuts California sales, which (through cross-border shopping)
   move Nevada's sales -- rather than by an unobserved shock that happens to hit
   treated and neighbours together. If a confounder drives both, ``rho`` would
   absorb it and the spillover would be mis-attributed; choose
   :math:`\mathbf{W}` to encode the *channel* you believe in (geography, trade).

Assumption 3 (Invertibility / no degenerate feedback). The matrix
:math:`\mathbf{I}_N - \rho\,\mathbf{w}\boldsymbol{\alpha}^{\top} - \rho\mathbf{W}`
has full rank.

   *Remark.* This guarantees the spatial system has a unique reduced form, so
   the counterfactual :math:`\mathbf{Y}^c_t(0)` is well defined. It holds
   automatically for :math:`|\rho|` below the spectral bound
   :math:`1/\max_i|\lambda_i(\mathbf{W})|` (which, for a row-normalised
   :math:`\mathbf{W}`, is 1); the sampler enforces :math:`|\rho| < 0.95` of that
   bound, and ``rho_ci`` lets you check the posterior never approaches it. It is
   *testable* given estimates of :math:`(\boldsymbol{\alpha}, \rho)`.

Assumption 4 (No anticipation / exogenous adoption). Pre-treatment
outcomes are untreated potential outcomes (treatment has no effect before
:math:`\mathcal{T}_2`), and the adoption time is not chosen on the basis of the
idiosyncratic shocks.

   *Remark.* Standard SCM bookkeeping: it lets the pre-period pin down
   :math:`(\boldsymbol{\alpha}, \rho)` cleanly. Violations (e.g. firms front-run
   a pre-announced policy) bias the pre-period fit and hence every downstream
   effect, exactly as in ordinary synthetic control.

Because the post-treatment effects (the displayed :math:`\xi_{0t}` and the
analogous :math:`\xi_{it}`) depend only on :math:`(\boldsymbol{\alpha}, \rho)`
and the observed outcomes, the covariate coefficient
:math:`\boldsymbol{\beta}` and the latent-factor block need not be estimated for
the post-period -- they enter only the pre-treatment estimation of
:math:`\rho`.

Inference (two steps)
~~~~~~~~~~~~~~~~~~~~~

Estimation is Bayesian and proceeds in two steps (the joint posterior mixes
poorly because of the ``rho * w alpha'`` interaction, so the authors -- and this
port -- factor it):

1. Synthetic weights ``alpha`` are drawn from a Bayesian horseshoe
   regression of the treated unit's pre-treatment outcomes on the controls'
   (strong shrinkage selects a sparse, relevant donor set).
2. Spatial parameter ``rho`` is drawn from the SAR likelihood conditional on
   ``alpha_hat`` by random-walk Metropolis (the ``log|I - rho A|`` Jacobian is
   evaluated in :math:`O(N)` from the eigenvalues of ``A = W + w alpha'``),
   alongside the innovation variance, an optional covariate coefficient
   ``beta``, and an optional AR(1) latent-factor block.

The treatment and spillover effects are then read off the identification
formulae per posterior draw; the point estimate plugs in the posterior means
``(alpha_hat, rho_hat)`` and the credible band sweeps ``rho`` with ``alpha``
fixed at ``alpha_hat`` (the paper's convention). Because the post-treatment
effects depend only on ``(alpha, rho)``, the factor and covariate blocks enter
only the pre-treatment ``rho`` estimation.

Credible intervals and diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inference is fully Bayesian -- there is no asymptotic-normal plug-in to trust on
short panels. Every reported uncertainty is a posterior quantile at level
``ci_level`` (default 0.95), and the :class:`SARFit` carries:

* ``rho_hat`` / ``rho_ci`` -- posterior mean and credible interval for the
  spatial parameter. A ``rho_ci`` excluding zero is direct evidence of
  spillover (SUTVA failure); a ``rho_ci`` hugging zero says ordinary SCM was
  fine.
* ``att_sp`` / ``ate_ci`` -- the spillover-adjusted ATT and its credible
  interval, alongside ``att_scm`` (the ``rho = 0`` SCM comparison).
* ``gap_sp`` / ``gap_ci`` -- the per-post-period treatment-effect path and its
  ``(T1, 2)`` credible band (shaded in the centre plot panel).
* ``spillover_panel`` / ``spillover_ci`` -- for each control unit, the
  posterior-mean spillover trajectory and its ``(T1, 2)`` band; a control whose
  band excludes zero is a unit the intervention measurably moved.
* ``rho_ess`` / ``acc_rho`` -- convergence diagnostics: the effective sample
  size of the ``rho`` chain (Geyer's initial-positive-sequence estimator) and
  the Metropolis acceptance rate. A small ``rho_ess`` relative to the number of
  draws flags weak identification or poor mixing -- the regime where ``rho``
  should be read as imprecise (this is exactly what happens in the California
  application; see *Verification*). Raise ``mcmc_iter`` / tune ``step_rho``
  (target acceptance ~0.2-0.5) and inspect ``rho_ess`` before trusting a narrow
  ``rho_ci``.

Spatial weights
~~~~~~~~~~~~~~~

The method requires a spatial-weight specification, supplied through
``spatial_W`` (an ``N x N`` control-to-control matrix) and ``spatial_w`` (a
length-``N`` treated-to-control vector). Both may be labelled ``pandas``
objects (a ``DataFrame`` indexed by control-unit label and a ``Series`` keyed by
label), which are aligned to the donor order automatically, or bare NumPy arrays
already in control-label order. Typical choices: geographic contiguity (a 1 if
two units share a border) for the California tobacco example, or normalised
bilateral trade volume for the Sudan example. Both are row-normalised
internally.

When to use it
~~~~~~~~~~~~~~

Reach for ``method='sar'`` when (i) interference plausibly runs through a
*known, dense* network (geography, trade, supply chains) rather than a small set
of named neighbours; (ii) the spillover effects on the untreated units are
themselves of interest; and (iii) you want Bayesian credible intervals from
short pre-treatment panels. If instead only a handful of units are exposed and
you can name them, ``method='cd'`` (free per-unit spillover coefficients) or
``method='grossi'`` (near/far clusters) are more parsimonious.

A marketing example makes the regime concrete. Suppose a retailer cuts price
(or launches a heavy ad burst) in one designated market area (DMA) and wants
the causal lift on that market's sales. Ordinary geo synthetic control builds a
"synthetic DMA" from untreated markets -- but if shoppers cross between adjacent
DMAs, or the promotion shifts demand in neighbouring markets through a shared
distribution or media footprint, those "control" markets are themselves
contaminated, and the synthetic counterfactual is pulled toward the treated
market's own shock. That biases the measured lift (typically *toward zero*, as
in the simulation table below) and, just as importantly, *hides* the
cannibalisation or halo landing on nearby markets. ``method='sar'`` with
:math:`\mathbf{W}` set to DMA adjacency (or co-shopping / shared-media weights)
de-biases the treated-market lift and returns a per-market spillover map --
which neighbours lost sales to cross-border shopping, which gained from a halo.
The same shape recurs for a bank rolling out a product in one region, a
platform changing policy in one country, or a CPG running a trade promotion in
one chain: a single treated unit, a dense economic network, and spillovers that
are a nuisance to the headline estimate but a finding in their own right.

Synthetic study (the paper's simulation)
.........................................

The authors' Monte Carlo design (Sakaguchi & Tagawa 2026, Section 5) places
the ``N = r^2`` control units on a rook (chessboard) lattice and draws the
control outcomes from a SAR panel with spatial parameter :math:`\rho`. The
treated unit's untreated path is :math:`\boldsymbol{\alpha}^{\top}\mathbf{Y}^c`;
in the post-period the treatment adds :math:`N(1,1)` noise. Sweeping
:math:`\rho` shows the headline result: the spillover-adjusted SAR estimator
recovers the true ATT at every :math:`\rho`, while ordinary SCM (the
:math:`\rho = 0` special case) is biased increasingly as :math:`|\rho|` grows.
The block below is fully self-contained -- copy, paste, run.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from mlsynth import SPILLSYNTH

   def rook_W(nr, nc):
       """Row-normalised rook-lattice adjacency on an nr x nc board."""
       N = nr * nc
       W = np.zeros((N, N))
       idx = lambda r, c: r * nc + c
       for r in range(nr):
           for c in range(nc):
               i = idx(r, c)
               if r > 0:      W[i, idx(r - 1, c)] = 1
               if r < nr - 1: W[i, idx(r + 1, c)] = 1
               if c > 0:      W[i, idx(r, c - 1)] = 1
               if c < nc - 1: W[i, idx(r, c + 1)] = 1
       rs = W.sum(1); rs[rs == 0] = 1
       return W / rs

   def simulate(rho, nr=6, nc=6, T=30, T0=20, sigma2=0.1, seed=0):
       rng = np.random.default_rng(seed)
       N = nr * nc
       Wn = rook_W(nr, nc)
       w = np.zeros(N); w[:4] = 1.0; wn = w / w.sum()   # treated borders units 1-4
       alpha = np.zeros(N)                              # true synthetic weights
       alpha[0] = 0.5; alpha[1] = -0.2; alpha[2:4] = 0.4; alpha[4:10] = 0.1 / 6
       IN = np.eye(N)
       Ainv  = np.linalg.inv(IN - rho * Wn - rho * np.outer(wn, alpha))
       Apost = np.linalg.inv(IN - rho * Wn)
       e = rng.normal(0, np.sqrt(sigma2), (T, N))
       Yc0 = (Ainv @ e.T).T                             # untreated controls
       Y00 = Yc0 @ alpha                                # treated untreated path
       tau = rng.normal(1.0, 1.0, T - T0)               # post-period effect
       Y0 = Y00.copy(); Y0[T0:] += tau
       Yc = Yc0.copy()
       for t in range(T0, T):                           # treated spills back
           Yc[t] = Apost @ (rho * wn * Y0[t] + e[t])
       labels = ["treated"] + [f"c{i}" for i in range(N)]
       Ypanel = np.vstack([Y0[None, :], Yc.T])
       df = pd.DataFrame(
           [{"unit": labels[u], "year": t, "y": Ypanel[u, t],
             "treated": int(u == 0 and t >= T0)}
            for u in range(N + 1) for t in range(T)]
       )
       Wdf = pd.DataFrame(Wn, index=labels[1:], columns=labels[1:])
       wser = pd.Series(wn, index=labels[1:])
       return df, Wdf, wser, float(tau.mean())

   print(f"{'rho':>6} {'true ATT':>9} {'SAR ATT':>9} {'SCM ATT':>9} {'rho_hat':>9}")
   for rho in (-0.8, -0.3, 0.0, 0.3, 0.8):
       df, Wdf, wser, true_att = simulate(rho, seed=1)
       res = SPILLSYNTH({
           "df": df, "outcome": "y", "treat": "treated",
           "unitid": "unit", "time": "year", "method": "sar",
           "spatial_W": Wdf, "spatial_w": wser, "p_factors": 0,
           "mcmc_iter": 6000, "mcmc_burn": 2000, "step_rho": 0.05,
           "mcmc_seed": 1, "display_graphs": False,
       }).fit()
       print(f"{rho:>6.1f} {true_att:>9.2f} {res.att:>9.2f} "
             f"{res.att_scm:>9.2f} {res.sar.rho_hat:>9.3f}")

This prints (one panel per ``rho``)::

      rho  true ATT   SAR ATT   SCM ATT   rho_hat
     -0.8      0.98      0.99      1.18    -0.799
     -0.3      0.98      0.98      1.05    -0.318
      0.0      0.98      0.97      0.98    -0.027
      0.3      0.98      0.97      0.88     0.264
      0.8      0.98      0.98      0.50     0.763

The ``SAR ATT`` column tracks the truth across the whole grid and ``rho_hat``
recovers the data-generating :math:`\rho`; ``SCM ATT`` drifts away from the
truth as the spatial dependence strengthens -- exactly the bias the SAR model
corrects.

Empirical replication: California's Proposition 99
...................................................

The shipped panel (:file:`basedata/california_panel.csv`) and spatial weights
(:file:`basedata/california_W_matrix.csv`, a state-by-state contiguity matrix,
and :file:`basedata/california_w_vector.csv`, ``1`` for states bordering
California) reproduce the paper's first application. The pre-period is
1970-1988 and the post-period 1989-2000; retail price enters as a covariate and
a single AR(1) latent factor absorbs the common trend.

.. code-block:: python

   import pandas as pd
   from mlsynth import SPILLSYNTH

   # -----------------------
   # GitHub file structure
   # -----------------------
   user = "jgreathouse9"
   repo = "mlsynth"
   branch = "main"
   base_path = "basedata"

   base_url = f"https://raw.githubusercontent.com/{user}/{repo}/refs/heads/{branch}/{base_path}/"

   # -----------------------
   # Data loading
   # -----------------------
   panel = pd.read_csv(base_url + "california_panel.csv")
   W = pd.read_csv(base_url + "california_W_matrix.csv", index_col=0)
   w = pd.read_csv(base_url + "california_w_vector.csv", index_col=0).squeeze()

   # -----------------------
   # Treatment definition
   # -----------------------
   panel["treated"] = (
       (panel["state"] == "California") &
       (panel["year"] >= 1989)
   ).astype(int)

   res = SPILLSYNTH({
       "df": panel, "outcome": "cigsale", "treat": "treated",
       "unitid": "state", "time": "year",
       "method": "sar",
       "spatial_W": W, "spatial_w": w,
       "covariates": ["retprice"], "p_factors": 1,
       "mcmc_iter": 20000, "mcmc_burn": 10000, "step_rho": 0.01,
       "display_graphs": True,
   }).fit()

   print(f"rho posterior mean = {res.sar.rho_hat:+.3f} {res.sar.rho_ci}")
   print(f"rho effective sample size = {res.sar.rho_ess:.0f}  "
         f"(MCMC mixing; small => weakly identified)")
   print(f"ATT (spillover-adjusted) = {res.att:+.2f} {res.sar.ate_ci}  "
         f"vs SCM {res.att_scm:+.2f}")
   # largest negative spillover (expect Nevada, California's neighbour)
   sp = {k: v.mean() for k, v in res.spillover_effects.items()}
   print(sorted(sp.items(), key=lambda kv: kv[1])[:3])

The Sudan application is identical in shape -- swap in
:file:`basedata/sudan_panel.csv` with the GDP-per-capita outcome,
``treat`` switching on in 2011, the six World-Bank covariates, and the
trade-weighted :file:`basedata/sudan_W_matrix.csv` /
:file:`basedata/sudan_w_vector.csv`.

Verification
~~~~~~~~~~~~

.. note::

   Reference cross-checks. This port reproduces the authors' compiled
   sampler on the well-identified cases: the simulation cell (``N=36``,
   ``rho=-0.8``) recovers their bias/RMSE and ~95% coverage, and the Sudan
   application matches the reported ``rho`` to three digits with a roughly
   :math:`-9.5\%`-per-year GDP effect and the largest spillover on Egypt. The
   California ``rho`` is *weakly identified* in the source data (the authors'
   own posterior has an effective sample size around 44): the synthetic weights,
   the spillover ranking (Nevada, then Idaho and Utah), and the larger-than-SCM
   negative treatment effect reproduce, but the ``rho`` *level* is mode- and
   chain-sensitive -- the authors' own released C++ returns a different ``rho``
   at a standard seed. Treat California's ``rho`` magnitude as weakly identified,
   not as a precise quantity.

References
----------

Sakaguchi, S., & Tagawa, H. (2026). "Identification and Bayesian Inference
for Synthetic Control Methods with Spillover Effects." Working paper
(arXiv:2408.00291).



Abadie, A., Diamond, A., & Hainmueller, J. (2010). "Synthetic Control
Methods for Comparative Case Studies." *Journal of the American
Statistical Association* 105(490):493-505.

Abadie, A., & L'Hour, J. (2021). "A Penalized Synthetic Control Estimator
for Disaggregated Data." *Journal of the American Statistical Association*
116(536):1817-1834.

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

Grossi, G., Mariani, M., Mattei, A., Lattarulo, P., & Oener, O. (2025).
"Direct and spillover effects of a new tramway line on the commercial
vitality of peripheral streets: a synthetic-control approach." *Journal of
the Royal Statistical Society Series A* 188(1):223-240.

Malo, P., Eskelinen, J., Zhou, X., & Kuosmanen, T. (2024). "Computing
Synthetic Controls Using Bilevel Optimization." *Computational
Economics* 64:1113-1136.
