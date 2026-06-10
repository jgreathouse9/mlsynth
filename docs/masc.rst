Matching and Synthetic Control (MASC)
======================================

.. currentmodule:: mlsynth

Assumptions (Kellogg-Mogstad-Pouliot-Torgovitsky 2021)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MASC inherits the formal identification stack of any causal SC
estimator (paper Section 2.1) and adds the structural conditions
needed for model averaging to make sense. Listed in the paper's
order:

**A1 (Selection on observables -- paper Assumption 1).** For
:math:`x` in the supports of both :math:`X_i \mid D_i = 0` and
:math:`X_i \mid D_i = 1`,

.. math::

   \mathbb{E}[Y_{it}(0) \mid D_i = 1, X_i = x]
   \;=\; \mathbb{E}[Y_{it}(0) \mid D_i = 0, X_i = x]
   \quad \text{for all } t \ge t^\star.

This is the standard mean-independence statement (ignorable
treatment assignment / unconfoundedness / selection on observables)
applied to the SC framework. Together with A2 it makes the
post-treatment conditional mean of untreated outcomes for the
treated unit identifiable from donor outcomes.

**A2 (Overlap -- paper Assumption 2).** The support of
:math:`X_i \mid D_i = 1` is contained in the support of
:math:`X_i \mid D_i = 0`. In a comparative case study with a
single treated unit (the paper's focus), this reduces to "for
almost every covariate value the treated unit takes, there exists
*some* donor with similar covariates". With one treated unit,
overlap fails fully only if the treated unit is an outlier on
every donor covariate.

**A3 (Lipschitz conditional mean).** The conditional mean
:math:`\gamma_t(x) = \mathbb{E}[Y_{it}(0) \mid D_i = 0, X_i = x]`
is Lipschitz in :math:`x` with constant :math:`c`. Used (paper
Section 2.2) to bound both bias components,

.. math::

   |\text{ExtBias}(w)| \;\le\; c \, \bigl\| x_1 - \textstyle\sum_i w_i x_i \bigr\|
   \;\equiv\; c \cdot \text{Ext}(w),
   \qquad
   |\text{IntBias}(w)| \;\le\; c \, \textstyle\sum_i w_i \|x_1 - x_i\|
   \;\equiv\; c \cdot \text{Int}(w).

These two bounds are the heart of the MASC argument: the SC
estimator minimises :math:`\text{Ext}(w)` (and lives at zero
extrapolation when :math:`x_1` is in the donor hull), while
matching minimises :math:`\text{Int}(w)` (and lives at zero
interpolation by using only the nearest neighbours). When
:math:`\gamma_t` is approximately linear in :math:`x`, the
interpolation bound is vacuous and SC dominates; when no donor is
close, the extrapolation bound is large and matching does worse.

**A4 (Complementarity -- the substantive premise of model
averaging).** Both biases are plausibly relevant in the
application: :math:`\gamma_t` is non-linear enough that SC alone
interpolates badly, *and* no single donor is close enough that
matching alone extrapolates badly. *Remark.* This is the paper's
central conjecture for why model averaging helps. When either
bias is absent the data-driven CV will pick
:math:`\hat\varphi \in \{0, 1\}` and MASC degenerates to a
boundary estimator -- a feature, not a bug.

**A5 (Rolling-origin stability).** The relationship between
treated and donor outcomes is stable across the late-pre-period
folds *and* across the pre/post boundary, so that one-step-ahead
forecast accuracy on the training-set tail is informative about
post-treatment forecast accuracy. *Remark.* This is the SC
identification premise restricted to the fold horizon. Without
it the CV criterion is uninformative about the post-period and
:math:`\hat\varphi` reflects only pre-period drift.

**A6 (Quadratic-in-**:math:`\varphi` **closed form).** The CV
criterion :math:`Q(m, \varphi)` is quadratic in :math:`\varphi`
with positive semi-definite Hessian, so the unconstrained
optimum is unique and the constrained optimum on :math:`[0, 1]`
is its clip. *Remark.* Mechanical; the joint :math:`(m, \varphi)`
search reduces to a one-dimensional sweep over :math:`m`. Held
by construction.

When the assumptions bind: practical diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(a) **Selection on observables (A1).** Like every regression /
    SC / matching estimator, MASC assumes that the only
    systematic difference between treated and donor post-period
    outcomes is captured by the observed pretreatment
    covariates :math:`X_i`. If a confounder is missing from
    :math:`X_i`, MASC's counterfactual is biased *regardless*
    of how the CV picks :math:`(\hat m, \hat \varphi)`.

    *Plausibly violated when* a known driver of the outcome is
    omitted from ``covariates`` -- a state's industry mix in a
    labour-market study, an audience segment in a marketing
    study. *Diagnostic*: re-fit with one omitted covariate at a
    time and check whether ``res.att`` moves; large movements
    flag a missing confounder. There is no within-MASC fix;
    the cure is to include the missing covariate, or accept
    selection-on-observables is failing for this application.

(b) **Overlap (A2).** With one treated unit, full overlap
    failure means the treated covariates lie outside the donor
    convex hull on at least one dimension.

    *Plausibly violated when* the treated unit is structurally
    extreme on a known covariate (Hong Kong's GDP level vs
    other Asian regions; California's population vs interior
    states). *Diagnostic*: at the SC limit (:math:`\varphi = 0`)
    the pre-period RMSE will be elevated *and* the implied SC
    weights will concentrate on a few donors with substantial
    covariate gaps. If ``res.fit.pre_rmse`` stays large *and*
    matching (:math:`\varphi = 1`) does even worse, both
    components of the model average are failing for the same
    reason. Switch to :doc:`iscm` (which identifies the effect
    even when the treated unit is outside the hull) or
    :doc:`nsc` (which drops the simplex restriction so SC
    weights can extrapolate by going negative on far donors).

(c) **Lipschitz conditional mean (A3).** The Lipschitz constant
    controls *how much* extrapolation / interpolation bias the
    two component estimators incur. If :math:`\gamma_t` has a
    sharp kink or threshold in :math:`x`, the bounds are loose
    and MASC's CV-driven mix can be unstable.

    *Plausibly violated when* the outcome is a step function
    (regulatory threshold), has a kink (minimum-wage bunching),
    or saturates near a ceiling. *Diagnostic*: plot the per-fold
    one-step-ahead forecast errors of the pure SC and pure
    matching arms (``res.cv_diagnostics`` exposes both); if the
    two error series are highly correlated across folds, the
    averaging gain is small and MASC reduces to either
    boundary.

(d) **Complementarity / both biases bind (A4).** If only one
    bias matters, MASC's CV will sit at :math:`\hat\varphi
    \in \{0, 1\}` and the model average adds variance without
    bias improvement.

    *Plausibly violated when* the SC pre-fit is already tight
    (:math:`\gamma_t` is approximately linear in :math:`x` on
    the donor support) or no donor is remotely close (matching
    alone extrapolates badly across the board). *Diagnostic*:
    read ``res.phi_hat``. If it is essentially 0 or 1 across
    multiple seeds / fold configurations, the MASC machinery
    is over-engineered for this application and the
    corresponding pure estimator is the better default --
    *canonical SCM* / :doc:`tssc` for the :math:`\varphi = 0`
    regime, a nearest-neighbour matching estimator outside
    mlsynth for the :math:`\varphi = 1` regime.

(e) **Rolling-origin stability (A5).** The CV-selected
    :math:`(\hat m, \hat \varphi)` is only as informative as
    the late-pre-period is representative of the post-period.

    *Plausibly violated when* a structural break (regime
    change, pandemic, financial crisis) sits inside the pre-
    period close to :math:`t^\star`. *Diagnostic*: inspect the
    per-fold forecast errors; if they trend sharply over the
    fold index, the late-pre-period is not exchangeable with
    the early one and the CV is mostly fitting that trend.
    Either trim the pre-period to a regime-stable window or
    move to a stationary-cycle estimator (:doc:`sbc`).

(f) **Multiple treated units.** The paper's setup is one
    treated unit. With multiple treated, the SC step's
    non-uniqueness problem (which the penalised-SC of Abadie &
    L'Hour 2020 was built for) propagates into MASC's mix.
    *Plausibly violated when* you have several treated units
    on the same cohort. *Diagnostic*: MASC's headline numbers
    will be sensitive to which treated unit you single out as
    "the" treated; if so, use *canonical SCM* paired with the
    penalised variant, or *FECT* for staggered designs.

When to use MASC -- and when not to
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Reach for MASC when:**

* You have **a single treated unit and a moderate-to-large
  donor pool** with non-trivial heterogeneity across donors.
  Comparative case studies in policy evaluation (Basque
  terrorism, Prop 99, German reunification) are the canonical
  setting.
* The conditional mean of the outcome is **plausibly
  non-linear** in the pre-treatment covariates -- so pure SC's
  interpolation bias is a real risk -- *and* you also suspect
  **no donor is close enough** to extrapolate from -- so pure
  matching's extrapolation bias is also a real risk. When
  both worries are alive, MASC's :math:`\hat\varphi` trades
  them off in a data-driven way.
* You can't decide a priori whether SC or matching is more
  appropriate. The CV gives you a defensible answer rather
  than the practitioner's eyeball "I'll use SC because that's
  what the paper said".
* The pre-period is **long enough for rolling-origin
  cross-validation** to discriminate the two biases (KMPT
  Section 5 uses 20 pre-treatment years on Basque; mlsynth's
  ``min_preperiods`` defaults to 5 as a hard floor, but
  pre-periods around 10+ are where CV becomes informative).

**Do not use MASC when:**

* **Either bias dominates.** If ``res.phi_hat`` is essentially
  0 or 1 across seeds, MASC adds variance without bias
  improvement. At :math:`\hat\varphi \approx 0` reach for
  *canonical SCM* / :doc:`tssc` (or :doc:`fscm` for selective donor
  pruning); at :math:`\hat\varphi \approx 1` reach for a
  dedicated nearest-neighbour matching estimator.
* **The treated unit is structurally outside the donor convex
  hull.** Both component estimators fail (A2). Use
  :doc:`iscm` (identifies the effect via donors that use the
  treated unit as a positive-weight donor) or :doc:`nsc`
  (drops the simplex restriction to extrapolate by negative
  weights).
* **You need posterior credible bands** on the weights / ATT.
  MASC returns point estimates plus a CV criterion. For full
  Bayesian inference, use :doc:`bvss` (spike-and-slab
  variable selection with a soft simplex).
* **The pre-period is very short** (:math:`T_1 < 10`-ish).
  Rolling-origin CV has too few folds to discriminate
  :math:`(m, \varphi)`; the selected mix is noise. Use
  *canonical SCM* / :doc:`tssc` / :doc:`fdid` (which work without
  CV) instead.
* **Multiple treated units.** MASC's identification story
  uses a single treated unit. For staggered or many-treated
  designs, use *FECT* or :doc:`sdid`. (The paper's
  Section 2.7 notes one *could* average a matching and
  penalised-SC pair across treated units, mirroring MASC, but
  this is not in the mlsynth implementation and demands additional econometric theory.)
* **Structural break inside the pre-period.** A5 fails; the
  CV is fitting the break instead of the post-period mix.
  Trim to a stable window or use :doc:`sbc`.
* **You need a single sparse interpretable weight vector** as
  the policy-story deliverable. MASC's output is a *mixture*
  of SC weights and matching weights; both can be sparse on
  their own, but the mixed vector is generically less sparse
  than either component. If the headline must be "California
  ≈ Utah + Montana + Nevada", run *canonical SCM* alongside.
* **Distributional questions** (Lorenz curves, QTEs, tail
  effects). MASC targets the mean ATT. Use :doc:`dsc`.
* **Continuous or multi-valued treatment.** MASC encodes a
  single binary intervention. Continuous dose belongs in
  :doc:`ctsc`.
* **Spillovers across donors.** Both component estimators
  inherit SUTVA at the donor level. Use :doc:`spillsynth` or
  :doc:`spsydid`.


Notation
--------

We use the synthetic-control canon. Unit :math:`j=0` is treated and
:math:`\mathcal{N} = \{1, \ldots, N\}` indexes the donor pool;
:math:`\mathbf{y}_0` is the treated outcome path and :math:`\mathbf{Y}` is the
:math:`(T,N)` donor outcome matrix. The pre-treatment window is
:math:`\mathcal{T}_1 = \{1, \ldots, T_1\}` and the post-treatment window is
:math:`\mathcal{T}_2 = \{T_1+1, \ldots, T\}`, with treatment beginning at
:math:`t = T_1 + 1`. Predictors are stacked into
:math:`(\mathbf{x}_0, \mathbf{X})` with :math:`\mathbf{x}_0\in\mathbb{R}^P` for
the treated unit and :math:`\mathbf{X}\in\mathbb{R}^{P\times N}` for the
donors. The simplex is

.. math::

   \Delta = \Bigl\{ \boldsymbol{\omega}\in\mathbb{R}^N :
       \boldsymbol{\omega}\ge \mathbf{0},\
       \sum_j \omega_j = 1 \Bigr\}.

Setup
-----

The matching and SCE weights and the MASC combiner are

.. math::

   \boldsymbol{\omega}_{\mathrm{match}}(m)_j
       &= \tfrac{1}{m}\,\mathbf{1}\!\Bigl\{ j \in \operatorname*{argmin}_{|S|=m}
           \sum_{i\in S} d(j_0, i) \Bigr\},
   \\[2pt]
   \boldsymbol{\omega}_{\mathrm{SC}}
       &\in \operatorname*{argmin}_{\boldsymbol{\omega}\in\Delta}
       \,\bigl\|\mathbf{x}_0 - \mathbf{X}\boldsymbol{\omega}\bigr\|_{\mathbf{V}}^2,
   \\[2pt]
   \boldsymbol{\omega}_{\mathrm{MASC}}(m,\varphi)
       &= \varphi\,\boldsymbol{\omega}_{\mathrm{match}}(m)
       + (1-\varphi)\,\boldsymbol{\omega}_{\mathrm{SC}},

where :math:`d(j_0, i) = \sum_{t\in\mathcal{T}_1} (y_{0t} - y_{it})^2` is the
pre-period squared-distance and :math:`\mathbf{V}` is the (possibly
optimised) predictor-weight matrix. Without ``covariates`` the SCE reduces
to outcome-paths matching, i.e. :math:`(\mathbf{x}_0, \mathbf{X}) =
(\mathbf{y}_0^{\mathrm{pre}}, \mathbf{Y}^{\mathrm{pre}})` with
:math:`\mathbf{V} = \mathbf{I}`.

Tuning by rolling-origin CV
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each fold :math:`f\in\mathcal{F}` (each :math:`f` indexes the last
pre-treatment period included in the training window), let
:math:`\hat y^{\mathrm{SC}}_{f+1}` and :math:`\hat y^{\mathrm{match}}_{f+1}(m)`
denote the one-step-ahead forecasts of the treated outcome from each
estimator fit on the first :math:`f` periods, and let :math:`y_{0,f+1}` denote
the actual treated outcome. The CV criterion at :math:`(m,\varphi)` is the
weighted squared-error

.. math::

   Q(m,\varphi) = \sum_{f\in\mathcal{F}} w_f\,
       \bigl( y_{0,f+1} - \varphi \hat y^{\mathrm{match}}_{f+1}(m)
              - (1-\varphi)\hat y^{\mathrm{SC}}_{f+1} \bigr)^2 .

Holding :math:`m` fixed, the first-order condition gives the closed form

.. math::

   \tilde\varphi(m) =
   \frac{
       \sum_f w_f \bigl(y_{0,f+1} - \hat y^{\mathrm{SC}}_{f+1}\bigr)
                  \bigl(\hat y^{\mathrm{match}}_{f+1}(m) - \hat y^{\mathrm{SC}}_{f+1}\bigr)
   }{
       \sum_f w_f \bigl(\hat y^{\mathrm{match}}_{f+1}(m) - \hat y^{\mathrm{SC}}_{f+1}\bigr)^2
   } ,
   \quad
   \hat\varphi(m) = \operatorname{clip}_{[0,1]}\bigl(\tilde\varphi(m)\bigr),

reproducing eq. 15 of Kellogg et al. (2021). The selected
:math:`\hat m = \operatorname*{argmin}_m Q(m,\hat\varphi(m))` is then plugged
in and final weights are refitted on the full pre-period.

Empirical Illustration: Basque Country and Spanish Terrorism
-----------------------------------------------------------------

Following Section 5 of Kellogg et al. [KMPT2021]_ -- the canonical Abadie &
Gardeazabal [ABADIE2003]_ study of the per-capita GDP cost of ETA terrorism --
``MASC`` runs on ``basque_jasa.csv``: 17 Spanish regions (Basque plus 16
donor candidates), 1955-1997, with the JASA predictor specification
(schooling shares, investment, sector composition, population density).

.. code-block:: python

   import pandas as pd
   from mlsynth import MASC

   url = ("https://raw.githubusercontent.com/jgreathouse9/mlsynth/refs/heads/"
          "main/basedata/basque_jasa.csv")
   df = pd.read_csv(url)

   covariates = [
       "school.illit", "school.prim", "school.med", "school.high", "invest",
       "sec.agriculture", "sec.energy", "sec.industry", "sec.construction",
       "sec.services.venta", "sec.services.nonventa", "popdens",
   ]
   # The covariate windows match Abadie & Gardeazabal (2003), Table 1:
   # schooling and investment are averaged over 1964-1969, the sector
   # shares (observed every other year) over 1961-1969, popdens is the
   # 1969 cross-section, and a lagged outcome ``gdpcap`` is matched on
   # the 1960-1969 mean (Abadie's "pre-treatment outcomes" predictor).
   covariate_windows = {
       "sec.agriculture": (1961, 1969), "sec.energy": (1961, 1969),
       "sec.industry": (1961, 1969), "sec.construction": (1961, 1969),
       "sec.services.venta": (1961, 1969),
       "sec.services.nonventa": (1961, 1969),
       "popdens": (1969, 1969),
       "invest": (1964, 1969),
       "school.illit": (1964, 1969), "school.prim": (1964, 1969),
       "school.med": (1964, 1969), "school.high": (1964, 1969),
       "gdpcap": (1960, 1969),
   }

   res = MASC({
       "df": df, "outcome": "gdpcap", "treat": "terrorism",
       "unitid": "regionname", "time": "year",
       "m_grid": list(range(1, 11)),
       "min_preperiods": 5,
       "covariates": covariates,
       "covariate_windows": covariate_windows,
       "display_graphs": False,
   }).fit()

   print(f"Selected m   : {res.m_hat}")
   print(f"Selected phi : {res.phi_hat:.3f}")
   print(f"Pre-RMSE     : ${res.fit.pre_rmse * 1000:.0f}/capita")
   print(f"ATT          : ${res.att * 1000:+.0f}/capita/year")
   print("Top donors:")
   for u, w in sorted(res.donor_weights.items(), key=lambda kv: -kv[1])[:4]:
       if w > 0.05:
           print(f"  {u:<32s} {w:.3f}")

This prints::

   Selected m   : 3
   Selected phi : 0.308
   Pre-RMSE     : $89/capita
   ATT          : $-816/capita/year
   Top donors:
     Cataluna                         0.446
     Madrid (Comunidad De)            0.298
     Baleares (Islas)                 0.211

The paper [KMPT2021]_, Section 5, reports MASC ≡ SCE
(:math:`\hat\varphi = 0`), pre-RMSE :math:`\approx \$94`, ATT
:math:`\approx -\$580`/capita/year, with donor weights ``Cataluna 0.85``,
``Madrid 0.15``. The agreement is on the robust quantities: Cataluna is the
dominant donor in both, Cataluna + Madrid carry most of the SC weight
(:math:`0.45 + 0.30 = 0.74` here vs.\ :math:`0.85 + 0.15` in KMPT), and the
pre-RMSE matches (:math:`\$89` vs.\ :math:`\$94`). The ATT :math:`-\$816`
shares the paper's sign and :math:`\$600`-:math:`\$800` magnitude but sits
below the published :math:`-\$580`; the level difference is the V-optimiser
non-uniqueness documented below (mlsynth's CV prefers a small amount of
matching, :math:`\hat\varphi \approx 0.31`, :math:`m = 3`, rather than the
paper's pure SC). The durable check is ``benchmarks/cases/masc_basque.py``.

.. note::

   **Why our :math:`\hat\varphi` is small but non-zero.** The JASA paper
   computes :math:`\boldsymbol{\omega}_{\mathrm{SC}}` via the ``synth()``
   package's quasi-Newton search over the predictor-weight matrix
   :math:`\mathbf{V}`. ``mlsynth`` delegates the V-optimisation to the
   Malo et al. [malo2023computing]_ bilevel solver (the same solver used by
   ``FSCM``). Both are mathematically valid V-optimisation strategies; on
   this problem they converge to slightly different :math:`\mathbf{V}` and
   therefore slightly different :math:`\mathbf{W}` (Cataluna 0.64 + Madrid
   0.23 vs.\ 0.85 + 0.15). The rolling-origin CV then prefers a small
   amount of nearest-neighbour matching (:math:`\hat\varphi \approx 0.32`,
   :math:`m=1`) rather than pure SC.

   This is the **non-uniqueness phenomenon** documented by Becker & Kloessner
   and discussed in Malo et al.: when the SC problem is over-parameterised
   (here 12 predictors over 16 donors) the upper-level loss is flat over many
   feasible :math:`\mathbf{V}`, and different V-optimisers converge to
   different :math:`\mathbf{W}`. Bit-perfect replication of JASA's Section 5
   would require a true ADH ``synth()`` port; the present implementation is
   a faithful port of the MASC *algorithm* (matching, rolling-origin CV,
   closed-form :math:`\varphi`) on top of mlsynth's bilevel V solver, with
   the documented caveat above.

Verification
------------

.. note::

   **Empirical (Basque proper).** With the Abadie-Gardeazabal predictor
   windows (schooling and investment 1964-1969, sector shares 1961-1969,
   popdens 1969, gdpcap 1960-1969), treatment starting in 1975 and Spain
   itself removed from the donor pool, MASC selects :math:`m=1`,
   :math:`\hat\varphi \approx 0.32`, pre-RMSE :math:`\approx \$97`/capita
   (vs.\ KMPT's :math:`\$94`) and ATT :math:`\approx -\$641`/capita/year
   (vs.\ KMPT's :math:`-\$580`). Donor mass concentrates on Cataluna
   (0.64) and Madrid (0.23) -- the same two-donor structure KMPT report
   (0.85 + 0.15). The residual gap is the V-optimiser non-uniqueness
   documented above.

   **Helpers.** The nearest-neighbour selector, the simplex SC primitive,
   the analytic :math:`\hat\varphi` formula and the per-fold covariate
   aggregation are unit-tested (``mlsynth/tests/test_masc.py``).

Core API
--------

.. automodule:: mlsynth.estimators.masc
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.MASCConfig
   :members:
   :undoc-members:

Result Containers
-----------------

``MASC.fit()`` returns a
:class:`~mlsynth.utils.masc_helpers.structures.MASCResults` containing the
selected ``(m_hat, phi_hat)``, the MASC weight vector (with the matching
and SC components separately preserved), counterfactual, pre/post gap,
pre-RMSE, ATT, and the full CV grid. The prepared NumPy panel is exposed
as a :class:`~mlsynth.utils.masc_helpers.structures.MASCInputs`.

.. note::

   ``MASC.fit()`` returns an :class:`~mlsynth.config_models.EffectResult` on the
   standardized two-family contract: ``res.att`` / ``res.counterfactual`` /
   ``res.gap`` / ``res.donor_weights`` / ``res.pre_rmse`` resolve through the
   standardized sub-models. The blended MASC weight vector is ``res.weights_vector``
   (the bare ``res.weights`` is reserved for the standardized
   :class:`~mlsynth.config_models.WeightsResults`); the CV-selected tuning is on
   ``res.m_hat`` / ``res.phi_hat`` and the full fit on ``res.fit``.

.. automodule:: mlsynth.utils.masc_helpers.structures
   :members:
   :undoc-members:
   :show-inheritance:

Helper Modules
--------------

Data preparation -- the only DataFrame touchpoint: pivots to NumPy, builds
the unit/time index, splits pre/post, assembles the optional covariate
panels for per-fold aggregation.

.. automodule:: mlsynth.utils.masc_helpers.setup
   :members:
   :undoc-members:

The nearest-neighbour selector, the simplex SC primitive (with optional
covariates routed through the bilevel solver) and the analytic-:math:`\varphi`
closed form.

.. automodule:: mlsynth.utils.masc_helpers.estimation
   :members:
   :undoc-members:

The rolling-origin cross-validation engine and the per-fold covariate
aggregator.

.. automodule:: mlsynth.utils.masc_helpers.crossval
   :members:
   :undoc-members:

The end-to-end pipeline composing CV with the full-sample refit and the
MASC weight combiner.

.. automodule:: mlsynth.utils.masc_helpers.orchestration
   :members:
   :undoc-members:

Plotting: outcome paths and the CV curve over the candidate ``m`` grid.

.. automodule:: mlsynth.utils.masc_helpers.plotter
   :members:
   :undoc-members:
