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
under "SP" is the Cao-Dowd procedure invoked via mlsynth's internal
``fit_leave_one_out_sc`` + ``sp_estimate`` helpers, bypassing the
DataFrame-construction overhead of the public ``SPILLSYNTH`` API for
speed in the inner loop.

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
