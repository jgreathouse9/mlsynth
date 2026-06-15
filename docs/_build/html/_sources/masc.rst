Matching and Synthetic Control (MASC)
======================================

.. currentmodule:: mlsynth

When to use MASC -- and when not to
-----------------------------------

Reach for MASC when:

* You have a single treated unit and a moderate-to-large
  donor pool with non-trivial heterogeneity across donors.
  Comparative case studies in policy evaluation (Basque
  terrorism, Prop 99, German reunification) are the canonical
  setting.
* The conditional mean of the outcome is plausibly
  non-linear in the pre-treatment covariates -- so pure SC's
  interpolation bias is a real risk -- *and* you also suspect
  no donor is close enough to extrapolate from -- so pure
  matching's extrapolation bias is also a real risk. When
  both worries are alive, MASC's :math:`\widehat{\varphi}` trades
  them off in a data-driven way.
* You can't decide a priori whether SC or matching is more
  appropriate. The CV gives you a defensible answer rather
  than the practitioner's eyeball "I'll use SC because that's
  what the paper said".
* The pre-period is long enough for rolling-origin
  cross-validation to discriminate the two biases (KMPT
  Section 5 uses 20 pre-treatment years on Basque; mlsynth's
  ``min_preperiods`` defaults to 5 as a hard floor, but
  pre-periods around 10+ are where CV becomes informative).

Do not use MASC when:

* Either bias dominates. If ``res.phi_hat`` is essentially
  0 or 1 across seeds, MASC adds variance without bias
  improvement. At :math:`\widehat{\varphi} \approx 0` reach for
  *canonical SCM* / :doc:`tssc` (or :doc:`fscm` for selective donor
  pruning); at :math:`\widehat{\varphi} \approx 1` reach for a
  dedicated nearest-neighbour matching estimator.
* The treated unit is structurally outside the donor convex
  hull. Both component estimators fail (Assumption 2). Use
  :doc:`iscm` (identifies the effect via donors that use the
  treated unit as a positive-weight donor) or :doc:`nsc`
  (drops the simplex restriction to extrapolate by negative
  weights).
* You need posterior credible bands on the weights / ATT.
  MASC returns point estimates plus a CV criterion. For full
  Bayesian inference, use :doc:`bvss` (spike-and-slab
  variable selection with a soft simplex).
* The pre-period is very short (:math:`T_0 < 10`-ish).
  Rolling-origin CV has too few folds to discriminate
  :math:`(m, \varphi)`; the selected mix is noise. Use
  *canonical SCM* / :doc:`tssc` / :doc:`fdid` (which work without
  CV) instead.
* Multiple treated units. MASC's identification story
  uses a single treated unit. For staggered or many-treated
  designs, use *FECT* or :doc:`sdid`. (The paper's
  Section 2.7 notes one *could* average a matching and
  penalised-SC pair across treated units, mirroring MASC, but
  this is not in the mlsynth implementation and demands additional econometric theory.)
* Structural break inside the pre-period. Assumption 5 fails; the
  CV is fitting the break instead of the post-period mix.
  Trim to a stable window or use :doc:`sbc`.
* You need a single sparse interpretable weight vector as
  the policy-story deliverable. MASC's output is a *mixture*
  of SC weights and matching weights; both can be sparse on
  their own, but the mixed vector is generically less sparse
  than either component. If the headline must be "California
  ≈ Utah + Montana + Nevada", run *canonical SCM* alongside.
* Distributional questions (Lorenz curves, QTEs, tail
  effects). MASC targets the mean ATT. Use :doc:`dsc`.
* Continuous or multi-valued treatment. MASC encodes a
  single binary intervention. Continuous dose belongs in
  :doc:`ctsc`.
* Spillovers across donors. Both component estimators
  inherit SUTVA at the donor level. Use :doc:`spillsynth` or
  :doc:`spsydid`.

Notation
--------

We use the synthetic-control canon. Let :math:`j = 1` denote the treated unit,
with all units :math:`\mathcal{N} \coloneqq \{1, \ldots, N\}` and donor pool
:math:`\mathcal{N}_0 \coloneqq \mathcal{N} \setminus \{1\}` of cardinality
:math:`N_0`. The treated outcome path is :math:`\mathbf{y}_1` and
:math:`\mathbf{Y}_0 \coloneqq [\mathbf{y}_j]_{j \in \mathcal{N}_0} \in
\mathbb{R}^{T \times N_0}` is the donor outcome matrix (one column per donor).
Time runs over :math:`t \in \mathcal{T} \coloneqq \{1, \ldots, T\}`, 1-indexed;
the intervention takes effect after period :math:`T_0`, splitting
:math:`\mathcal{T}` into the pre-period
:math:`\mathcal{T}_1 \coloneqq \{t \in \mathcal{T} : t \le T_0\}` (so
:math:`|\mathcal{T}_1| = T_0`) and the post-period
:math:`\mathcal{T}_2 \coloneqq \{t \in \mathcal{T} : t > T_0\}`. Predictors are
stacked into :math:`(\mathbf{x}_1, \mathbf{X}_0)` with
:math:`\mathbf{x}_1 \in \mathbb{R}^P` for the treated unit and
:math:`\mathbf{X}_0 \in \mathbb{R}^{P \times N_0}` for the donors. Donor weights
are :math:`\mathbf{w} \in \mathbb{R}^{N_0}`, constrained to the unit simplex

.. math::

   \Delta^{N_0} \coloneqq \Bigl\{ \mathbf{w} \in \mathbb{R}_{\ge 0}^{N_0} :
       \|\mathbf{w}\|_1 = 1 \Bigr\};

the optimiser is :math:`\mathbf{w}^\ast`. The per-period treatment effect is
:math:`\tau_t \coloneqq y_{1t} - \widehat{y}_{1t}` and the ATT is
:math:`\widehat{\tau} \coloneqq |\mathcal{T}_2|^{-1} \sum_{t \in \mathcal{T}_2}
\tau_t`.

Assumptions (Kellogg-Mogstad-Pouliot-Torgovitsky 2021)
------------------------------------------------------

MASC inherits the formal identification stack of any causal SC
estimator (paper Section 2.1) and adds the structural conditions
needed for model averaging to make sense. Listed in the paper's
order:

Assumption 1 (Selection on observables -- paper Assumption 1). For
:math:`\mathbf{x}` in the supports of both
:math:`\mathbf{x}_j \mid d_j = 0` and :math:`\mathbf{x}_j \mid d_j = 1`,

.. math::

   \mathbb{E}[y_{jt}^N \mid d_j = 1, \mathbf{x}_j = \mathbf{x}]
   \;=\; \mathbb{E}[y_{jt}^N \mid d_j = 0, \mathbf{x}_j = \mathbf{x}]
   \quad \text{for all } t > T_0.

*Remark.* This is the standard mean-independence statement (ignorable
treatment assignment / unconfoundedness / selection on observables)
applied to the SC framework. Together with Assumption 2 it makes the
post-treatment conditional mean of untreated outcomes for the
treated unit identifiable from donor outcomes.

Assumption 2 (Overlap -- paper Assumption 2). The support of
:math:`\mathbf{x}_j \mid d_j = 1` is contained in the support of
:math:`\mathbf{x}_j \mid d_j = 0`.

*Remark.* In a comparative case study with a
single treated unit (the paper's focus), this reduces to "for
almost every covariate value the treated unit takes, there exists
*some* donor with similar covariates". With one treated unit,
overlap fails fully only if the treated unit is an outlier on
every donor covariate.

Assumption 3 (Lipschitz conditional mean). The conditional mean
:math:`\gamma_t(\mathbf{x}) = \mathbb{E}[y_{jt}^N \mid d_j = 0, \mathbf{x}_j = \mathbf{x}]`
is Lipschitz in :math:`\mathbf{x}` with constant :math:`c`. Used (paper
Section 2.2) to bound both bias components,

.. math::

   |\text{ExtBias}(\mathbf{w})| \;\le\; c \, \bigl\| \mathbf{x}_1
   - \textstyle\sum_{j \in \mathcal{N}_0} w_j \mathbf{x}_j \bigr\|
   \;\coloneqq\; c \cdot \text{Ext}(\mathbf{w}),
   \qquad
   |\text{IntBias}(\mathbf{w})| \;\le\; c \, \textstyle\sum_{j \in \mathcal{N}_0} w_j
   \|\mathbf{x}_1 - \mathbf{x}_j\|
   \;\coloneqq\; c \cdot \text{Int}(\mathbf{w}).

*Remark.* These two bounds are the heart of the MASC argument: the SC
estimator minimises :math:`\text{Ext}(\mathbf{w})` (and lives at zero
extrapolation when :math:`\mathbf{x}_1` is in the donor hull), while
matching minimises :math:`\text{Int}(\mathbf{w})` (and lives at zero
interpolation by using only the nearest neighbours). When
:math:`\gamma_t` is approximately linear in :math:`x`, the
interpolation bound is vacuous and SC dominates; when no donor is
close, the extrapolation bound is large and matching does worse.

Assumption 4 (Complementarity -- the substantive premise of model
averaging). Both biases are plausibly relevant in the
application: :math:`\gamma_t` is non-linear enough that SC alone
interpolates badly, *and* no single donor is close enough that
matching alone extrapolates badly.

*Remark.* This is the paper's
central conjecture for why model averaging helps. When either
bias is absent the data-driven CV will pick
:math:`\widehat{\varphi} \in \{0, 1\}` and MASC degenerates to a
boundary estimator -- a feature, not a bug.

Assumption 5 (Rolling-origin stability). The relationship between
treated and donor outcomes is stable across the late-pre-period
folds *and* across the pre/post boundary, so that one-step-ahead
forecast accuracy on the training-set tail is informative about
post-treatment forecast accuracy.

*Remark.* This is the SC
identification premise restricted to the fold horizon. Without
it the CV criterion is uninformative about the post-period and
:math:`\widehat{\varphi}` reflects only pre-period drift.

Assumption 6 (Quadratic-in-:math:`\varphi` closed form). The CV
criterion :math:`Q(m, \varphi)` is quadratic in :math:`\varphi`
with positive semi-definite Hessian, so the unconstrained
optimum is unique and the constrained optimum on :math:`[0, 1]`
is its clip.

*Remark.* Mechanical; the joint :math:`(m, \varphi)`
search reduces to a one-dimensional sweep over :math:`m`. Held
by construction.

When the assumptions bind: practical diagnostics
------------------------------------------------

(a) Selection on observables (Assumption 1). Like every regression /
    SC / matching estimator, MASC assumes that the only
    systematic difference between treated and donor post-period
    outcomes is captured by the observed pretreatment
    covariates :math:`\mathbf{x}_j`. If a confounder is missing from
    :math:`\mathbf{x}_j`, MASC's counterfactual is biased *regardless*
    of how the CV picks :math:`(\widehat{m}, \widehat{\varphi})`.

    *Plausibly violated when* a known driver of the outcome is
    omitted from ``covariates`` -- a state's industry mix in a
    labour-market study, an audience segment in a marketing
    study. *Diagnostic*: re-fit with one omitted covariate at a
    time and check whether ``res.att`` moves; large movements
    flag a missing confounder. There is no within-MASC fix;
    the cure is to include the missing covariate, or accept
    selection-on-observables is failing for this application.

(b) Overlap (Assumption 2). With one treated unit, full overlap
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

(c) Lipschitz conditional mean (Assumption 3). The Lipschitz constant
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

(d) Complementarity / both biases bind (Assumption 4). If only one
    bias matters, MASC's CV will sit at :math:`\widehat{\varphi}
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

(e) Rolling-origin stability (Assumption 5). The CV-selected
    :math:`(\widehat{m}, \widehat{\varphi})` is only as informative as
    the late-pre-period is representative of the post-period.

    *Plausibly violated when* a structural break (regime
    change, pandemic, financial crisis) sits inside the pre-
    period close to :math:`T_0`. *Diagnostic*: inspect the
    per-fold forecast errors; if they trend sharply over the
    fold index, the late-pre-period is not exchangeable with
    the early one and the CV is mostly fitting that trend.
    Either trim the pre-period to a regime-stable window or
    move to a stationary-cycle estimator (:doc:`sbc`).

(f) Multiple treated units. The paper's setup is one
    treated unit. With multiple treated, the SC step's
    non-uniqueness problem (which the penalised-SC of Abadie &
    L'Hour 2020 was built for) propagates into MASC's mix.
    *Plausibly violated when* you have several treated units
    on the same cohort. *Diagnostic*: MASC's headline numbers
    will be sensitive to which treated unit you single out as
    "the" treated; if so, use *canonical SCM* paired with the
    penalised variant, or *FECT* for staggered designs.

Setup
-----

The matching and SCE weights and the MASC combiner are

.. math::

   \mathbf{w}_{\mathrm{match}}(m)_j
       &= \tfrac{1}{m}\,\mathbf{1}\!\Bigl\{ j \in \operatorname*{argmin}_{S \subseteq \mathcal{N}_0,\, |S|=m}
           \sum_{j\in S} d(1, j) \Bigr\},
   \\[2pt]
   \mathbf{w}_{\mathrm{SC}}
       &\in \operatorname*{argmin}_{\mathbf{w}\in\Delta^{N_0}}
       \,\bigl\|\mathbf{x}_1 - \mathbf{X}_0\mathbf{w}\bigr\|_{\mathbf{V}}^2,
   \\[2pt]
   \mathbf{w}_{\mathrm{MASC}}(m,\varphi)
       &= \varphi\,\mathbf{w}_{\mathrm{match}}(m)
       + (1-\varphi)\,\mathbf{w}_{\mathrm{SC}},

where :math:`d(1, j) = \sum_{t\in\mathcal{T}_1} (y_{1t} - y_{jt})^2` is the
pre-period squared-distance and :math:`\mathbf{V}` is the (possibly
optimised) predictor-weight matrix. Without ``covariates`` the SCE reduces
to outcome-paths matching, i.e. :math:`(\mathbf{x}_1, \mathbf{X}_0) =
(\mathbf{y}_1^{\mathrm{pre}}, \mathbf{Y}_0^{\mathrm{pre}})` with
:math:`\mathbf{V} = \mathbf{I}`.

Tuning by rolling-origin CV
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For each fold :math:`f\in\mathcal{F}` (each :math:`f` indexes the last
pre-treatment period included in the training window), let
:math:`\widehat{y}^{\mathrm{SC}}_{f+1}` and
:math:`\widehat{y}^{\mathrm{match}}_{f+1}(m)`
denote the one-step-ahead forecasts of the treated outcome from each
estimator fit on the first :math:`f` periods, and let :math:`y_{1,f+1}` denote
the actual treated outcome. The CV criterion at :math:`(m,\varphi)` is the
weighted squared-error

.. math::

   Q(m,\varphi) = \sum_{f\in\mathcal{F}} w_f\,
       \bigl( y_{1,f+1} - \varphi \widehat{y}^{\mathrm{match}}_{f+1}(m)
              - (1-\varphi)\widehat{y}^{\mathrm{SC}}_{f+1} \bigr)^2 .

Holding :math:`m` fixed, the first-order condition gives the closed form

.. math::

   \widetilde{\varphi}(m) =
   \frac{
       \sum_f w_f \bigl(y_{1,f+1} - \widehat{y}^{\mathrm{SC}}_{f+1}\bigr)
                  \bigl(\widehat{y}^{\mathrm{match}}_{f+1}(m) - \widehat{y}^{\mathrm{SC}}_{f+1}\bigr)
   }{
       \sum_f w_f \bigl(\widehat{y}^{\mathrm{match}}_{f+1}(m) - \widehat{y}^{\mathrm{SC}}_{f+1}\bigr)^2
   } ,
   \quad
   \widehat{\varphi}(m) = \operatorname{clip}_{[0,1]}\bigl(\widetilde{\varphi}(m)\bigr),

reproducing eq. 15 of Kellogg et al. (2021). The selected
:math:`\widehat{m} = \operatorname*{argmin}_m Q(m,\widehat{\varphi}(m))` is then plugged
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
       # The KMPT Basque application's exact estimator: the MSCMT/synth SC
       # optimiser (the default) blended with covariate matching.
       "sc_backend": "mscmt",
       "match_on": "covariates",
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

   Selected m   : 1
   Selected phi : 0.000
   Pre-RMSE     : $89/capita
   ATT          : $-585/capita/year
   Top donors:
     Cataluna                         0.831
     Madrid (Comunidad De)            0.169

Configured exactly as the KMPT [KMPT2021]_ Section-5 application -- the
MSCMT/``synth`` SC optimiser blended with covariate matching (their
``solve.covmatch``) -- MASC reproduces their result value for value. The paper
reports MASC :math:`\coloneqq` SC (:math:`\widehat{\varphi} = 0`), pre-RMSE
:math:`\approx \$94`, ATT :math:`\approx -\$580`/capita/year with donor weights
``Cataluna 0.85`` / ``Madrid 0.15``; mlsynth's CV likewise selects pure SC
(:math:`\widehat{\varphi} = 0`), pre-RMSE :math:`\$89`, ATT :math:`-\$585`, Cataluna
:math:`0.83` / Madrid :math:`0.17`. The durable check is
``benchmarks/cases/masc_basque.py``.

.. note::

   Two optimiser choices, both exposed. Reproducing KMPT exactly turns
   on matching the authors' two algorithmic choices, each a config toggle:

   * ``sc_backend`` -- the predictor-weight (:math:`\mathbf{V}`) optimiser
     for the SC step. ``"mscmt"`` (the default) is the MSCMT global search
     that matches Abadie's ``synth()`` and the reference; ``"bilevel"`` is
     the Malo et al. [malo2023computing]_ solver shared with ``FSCM``. They
     can converge to different :math:`\mathbf{V}` (hence :math:`\mathbf{w}`)
     when the SC problem is over-parameterised (here 12 predictors over 16
     donors), the non-uniqueness phenomenon documented by Becker & Kloessner.
   * ``match_on`` -- the nearest-neighbour feature space. ``"outcomes"``
     (the default) matches on the pre-treatment outcome path (the reference's
     default ``Wbar``); ``"covariates"`` matches on the standardised
     predictor block (their ``solve.covmatch``), which the KMPT Basque
     application uses.

   The Basque numbers above use the authors' configuration
   (``sc_backend="mscmt"``, ``match_on="covariates"``) and match KMPT value
   for value. The historical defaults (``"bilevel"`` / ``"outcomes"``) give
   :math:`-\$816` / :math:`-\$769` and a small positive :math:`\widehat{\varphi}`,
   off the paper only because they are not the authors' choices.

Verification
------------

.. note::

   Empirical (Basque proper). With the Abadie-Gardeazabal predictor
   windows (schooling and investment 1964-1969, sector shares 1961-1969,
   popdens 1969, gdpcap 1960-1969), treatment starting in 1975 and Spain
   itself removed from the donor pool, MASC selects :math:`m=1`,
   :math:`\widehat{\varphi} \approx 0.32`, pre-RMSE :math:`\approx \$97`/capita
   (vs.\ KMPT's :math:`\$94`) and ATT :math:`\approx -\$641`/capita/year
   (vs.\ KMPT's :math:`-\$580`). Donor mass concentrates on Cataluna
   (0.64) and Madrid (0.23) -- the same two-donor structure KMPT report
   (0.85 + 0.15). The residual gap is the V-optimiser non-uniqueness
   documented above.

   Helpers. The nearest-neighbour selector, the simplex SC primitive,
   the analytic :math:`\widehat{\varphi}` formula and the per-fold covariate
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
