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

* No covariates -> the donor weights :math:`\mathbf{w}` solve the convex
  simplex least-squares fit on the pre-treatment outcomes. This is a
  single, well-posed convex program -- deterministic and reproducible
  (unique up to donor collinearity).
* Covariates -> the predictor weights :math:`\mathbf{V}` and donor weights
  :math:`\mathbf{w}` are chosen jointly through a bilevel program. This is
  non-convex, and the predictor weights are generically *non-identified*.
  ``VanillaSC`` solves it with a reliable backend and reports a diagnostic
  (:math:`\text{v\_agreement}`) so that fragility is visible rather than
  silent.

When to use this estimator
--------------------------

* You want the standard synthetic control done reliably, with the
  solver choice and identification fragility surfaced.
* Outcome-only matching when you have a long, informative pre-period
  -- this is the well-posed, reproducible case.
* Covariate matching with ``mscmt`` when the donor pool is rich
  enough that the problem is well-conditioned (see the replications
  below). When :math:`\text{v\_agreement}` comes back near 1, prefer
  outcome-only or ``penalized``.

A concrete example: a state passes a tobacco-control law and you want its
effect on cigarette sales. There is one treated state, a long pre-law
history, and dozens of untreated states to draw on. ``VanillaSC`` builds a
synthetic California as a weighted blend of those donor states that matches
the pre-law sales path, then reads the policy effect as the post-law gap
between the real and synthetic series.

Notation
--------

Let :math:`j = 1` denote the treated unit, with all units
:math:`\mathcal{N} \coloneqq \{1, \dots, N\}` and donor pool
:math:`\mathcal{N}_0 \coloneqq \mathcal{N} \setminus \{1\}` of cardinality
:math:`N_0`. Time runs over :math:`t \in \mathcal{T} \coloneqq \{1, \dots, T\}`,
1-indexed; the intervention takes effect after period :math:`T_0`, splitting
:math:`\mathcal{T}` into the pre-period
:math:`\mathcal{T}_1 \coloneqq \{t \in \mathcal{T} : t \le T_0\}` (of length
:math:`T_0`) and the post-period
:math:`\mathcal{T}_2 \coloneqq \{t \in \mathcal{T} : t > T_0\}`.

The treated series is :math:`\mathbf{y}_1 = (y_{11}, \dots, y_{1T})^\top \in
\mathbb{R}^{T}` with scalar outcomes :math:`y_{1t}`; each donor
:math:`j \in \mathcal{N}_0` contributes a series :math:`\mathbf{y}_j`, stacked
into the donor matrix :math:`\mathbf{Y}_0 \coloneqq [\mathbf{y}_j]_{j \in
\mathcal{N}_0} \in \mathbb{R}^{T \times N_0}` (one column per donor). Donor
weights are :math:`\mathbf{w} \in \mathbb{R}^{N_0}`, constrained to the unit
simplex
:math:`\Delta^{N_0} \coloneqq \{\mathbf{w} \in \mathbb{R}_{\ge 0}^{N_0} :
\|\mathbf{w}\|_1 = 1\}` in the canonical SCM; the optimiser is
:math:`\mathbf{w}^\ast` and the fitted vector :math:`\widehat{\mathbf{w}}`. The
synthetic counterfactual is :math:`\widehat{\mathbf{y}}_1 \coloneqq
\mathbf{Y}_0\,\widehat{\mathbf{w}}` with entries :math:`\widehat{y}_{1t}`, the
per-period effect is :math:`\tau_t \coloneqq y_{1t} - \widehat{y}_{1t}`, and the
ATT is :math:`\widehat{\tau} \coloneqq |\mathcal{T}_2|^{-1}
\sum_{t \in \mathcal{T}_2} \tau_t`. The significance level is :math:`\alpha`.

Identifying assumptions
------------------------

1. Pre-treatment fit / convex-hull support. There exist weights
   :math:`\mathbf{w} \in \Delta^{N_0}` under which the treated pre-period path is
   reproduced by the donors, :math:`y_{1t} \approx (\mathbf{Y}_0\mathbf{w})_t`
   for :math:`t \in \mathcal{T}_1` -- equivalently, the treated unit lies inside
   (or near) the convex hull of the donors' lagged outcomes (Abadie, Diamond &
   Hainmueller 2010).

   *Remark.* This is the workhorse identifying condition: when the simplex fit
   balances the pre-period exactly the synthetic control is (near) unbiased, and
   a good pre-fit is the empirical certificate one inspects. When no convex
   combination fits -- few donors, an outlying treated unit, a long
   high-dimensional pre-period -- residual imbalance becomes bias, which is
   exactly the regime the ridge augmentation below targets.

2. No anticipation. Treatment has no effect before :math:`T_0`:
   :math:`y_{1t} = y_{1t}^N` for all :math:`t \in \mathcal{T}_1`, so the
   pre-period outcomes reflect the no-intervention path.

   *Remark.* If the treated unit reacts in advance of the formal intervention
   date, the pre-period fit is contaminated by the effect itself and the gap
   understates it. The remedy is to date :math:`T_0` at the first plausible
   response, not the nominal policy date.

3. No interference and untreated donors (SUTVA). The treatment of unit
   :math:`1` does not change any donor's outcome, and every
   :math:`j \in \mathcal{N}_0` is untreated over :math:`\mathcal{T}`, so
   :math:`\mathbf{Y}_0` carries only no-intervention outcomes.

   *Remark.* Donors contaminated by the same shock (spillovers, a co-incident
   policy) bias the counterfactual. Drop or quarantine such units from the donor
   pool before fitting.

4. Outcome-model stability. The no-intervention outcomes follow a stable
   data-generating process across :math:`\mathcal{T}` -- for instance a linear
   factor structure :math:`y_{jt}^N = \boldsymbol{\lambda}_j^\top \mathbf{f}_t +
   \varepsilon_{jt}` -- so that weights matching the treated unit's factor
   loadings on :math:`\mathcal{T}_1` continue to reproduce its
   no-intervention path on :math:`\mathcal{T}_2` (Abadie, Diamond &
   Hainmueller 2010).

   *Remark.* This is what licenses extrapolating the pre-period fit forward: the
   synthetic control tracks the treated unit after :math:`T_0` only if the same
   latent structure governs both windows. A regime change unrelated to treatment
   (a structural break in :math:`\mathbf{f}_t`) breaks the counterfactual even
   with a perfect pre-fit.

Mathematical formulation
------------------------

For a treated unit with pre-treatment outcomes
:math:`\mathbf{y}_{1,\mathcal{T}_1} \in \mathbb{R}^{T_0}` and donors
:math:`\mathbf{Y}_{0,\mathcal{T}_1} \in \mathbb{R}^{T_0 \times N_0}`:

Outcome-only (no covariates).

.. math::

   \widehat{\mathbf{w}} = \operatorname*{argmin}_{\mathbf{w}} \;
   \bigl\| \mathbf{y}_{1,\mathcal{T}_1} - \mathbf{Y}_{0,\mathcal{T}_1}\mathbf{w}
   \bigr\|_2^2
   \quad \text{s.t.} \quad \mathbf{w} \ge 0,\ \mathbf{1}^\top\mathbf{w} = 1.

Covariate matching (bilevel). With predictor matrices :math:`\mathbf{X}_1 \in
\mathbb{R}^{P}` (treated) and :math:`\mathbf{X}_0 \in \mathbb{R}^{P \times N_0}`
(donors), each predictor averaged over its window and scaled to unit
variance, the lower level solves, for given diagonal predictor weights
:math:`\mathbf{V}`,

.. math::

   \mathbf{w}^\ast(\mathbf{V}) = \operatorname*{argmin}_{\mathbf{w} \in \Delta^{N_0}}
   \; (\mathbf{X}_1 - \mathbf{X}_0 \mathbf{w})^\top \mathbf{V}
   (\mathbf{X}_1 - \mathbf{X}_0 \mathbf{w}),

and the upper level chooses :math:`\mathbf{V}` to minimise the pre-treatment
*outcome* fit,

.. math::

   \min_{\mathbf{V}} \; \bigl\| \mathbf{y}_{1,\mathcal{T}_1} -
   \mathbf{Y}_{0,\mathcal{T}_1}\, \mathbf{w}^\ast(\mathbf{V}) \bigr\|_2^2 .

The donor weights :math:`\mathbf{w}` and the counterfactual are pinned by this
program; the predictor weights :math:`\mathbf{V}` are generically not (a whole
polytope of :math:`\mathbf{V}` reproduces the same :math:`\mathbf{w}`).

Backends
--------

The covariate path exposes three reliable solvers via ``backend=``:

``"outcome-only"``
    No predictor weights; the convex simplex fit above. The well-posed
    default (also selected by ``backend="auto"`` when no covariates are
    given).

``"mscmt"``
    Becker & Kloessner (2018): a global differential-evolution search over
    :math:`\log_{10} \mathbf{V}` with the simplex inner solve. The default when
    covariates are supplied. Set ``canonical_v="min.loss.w"`` (or
    ``"max.order"``) to report a canonical, reproducible :math:`\mathbf{V}` via the
    MSCMT ``determine_v`` step.

``"malo"``
    Malo et al. (2024): a staged corner search. Fast and exact when the
    optimum is a predictor corner -- but note that when a *lagged outcome*
    is among the predictors, the loss-minimising corner puts all weight on
    that lag, collapsing the inner match to pure outcome-fitting (it
    drifts toward the outcome floor).

``"penalized"``
    Abadie & L'Hour (2021): a pairwise-penalized estimator with
    leave-one-out :math:`\lambda` selection, giving a unique, sparse
    :math:`\mathbf{w}`. Works with or without covariates. The solver is
    cross-validated against the authors' own ``wsoll1`` (durable case
    ``pensynth_prop99``); see :doc:`replications/pensynth`. Set
    ``penalized_lambda`` to supply :math:`\lambda` directly and skip the
    cross-validation: a large value approaches nearest-neighbour matching, an
    infinitesimal value gives the lexicographic (fit-preserving) tie-break that
    resolves non-uniqueness without distorting the fit, and ``0`` recovers the
    unpenalized synthetic control. ``penalized_cv`` selects the CV criterion
    only when ``penalized_lambda`` is ``None``.

.. note::

   Both covariate backends are pinned on a paper specification by the
   ``vanillasc_carbontax`` benchmark: under Andersson (2019)'s own
   synthetic-control spec for the Swedish carbon tax -- GDP per capita, motor
   vehicles, gasoline consumption and urban population averaged over 1980-1989,
   plus lagged CO2 for 1970/1980/1989 -- ``malo`` and ``mscmt`` both reproduce
   his reported average ATT of :math:`-0.29` metric tons per capita and the
   :math:`-0.35` gap in 2005, with a pre-treatment RMSE of :math:`\approx
   0.034`. The lagged-outcome predictors anchor the pre-period fit and bring the
   two ``V`` searches into agreement.

.. note::

   The outcome-only backend is cross-validated against a published paper's own
   replication code by the ``ibex_dap`` benchmark: on Haro Ruiz, Schult and
   Wunder (2024)'s study of the Iberian exception mechanism, ``VanillaSC`` with
   ``backend="outcome-only"`` reproduces the authors' ``lsei`` synthetic control
   for the day-ahead electricity price value-for-value -- the Spain and Portugal
   donor weights match to solver tolerance and the per-period gap agrees
   pointwise. See :doc:`replications/ibex`.

The identification diagnostic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When covariates are used, ``res.weights.summary_stats["v_agreement"]``
reports the maximum absolute difference between the two MSCMT canonical
predictor-weight vectors (``min.loss.w`` and ``max.order``). It is small
when :math:`\mathbf{V}` is well identified and large (up to 1) when the predictor
weights -- and the donor weights they imply -- are fragile. A large value
is a warning that the covariate-matched solution should not be
over-interpreted.

Inference
---------

Five inference modes are available via ``inference=``:

``"placebo"`` (default, ``inference=True``)
    Abadie's in-space placebo test: the synthetic control is refit treating
    each donor as pseudo-treated, and the treated unit's post/pre RMSPE ratio
    is ranked against the placebo distribution to give a p-value. Simple and
    assumption-light, but the smallest achievable p-value is about
    :math:`1/(N_0+1)`. Its pretreatment-horizon companion is the Truncated
    History check (:doc:`truncated_history`), which re-fits this estimator on
    truncated pre-treatment windows to see whether the effect is robust to how
    far back the fit reaches.

``"conformal"`` -- prediction intervals (Chernozhukov, Wüthrich & Zhu 2021)
    The ``augsynth`` default for Augmented SCM, and a *distribution-free* test
    by inversion. For a sharp null :math:`H_0:\ \tau_t = \tau_0` the post-period
    treated outcome is adjusted by :math:`\tau_0`, the weights are refit on the
    adjusted data, and the post-period residual is checked for whether it
    conforms with the pre-treatment residuals -- its rank among them is the
    :math:`p`-value,

    .. math::

       p(\tau_0) = \frac{1 + \#\{\,t \le T_0 : |\widehat{u}_t| \ge
       |\widehat{u}_{\mathrm{post}}(\tau_0)|\,\}}{T_0 + 1}.

    Inverting the test (the :math:`\tau_0` not rejected at level :math:`\alpha`)
    gives a per-period prediction interval for the random counterfactual
    :math:`y_{1t}^N`; the same machinery returns one joint-null :math:`p`-value
    for the whole effect path. It is *exactly valid* when the residuals are
    exchangeable, and finite-sample bounded otherwise -- the ridge penalty
    controls the SCM-vs-ASCM weight difference, so validity holds as
    :math:`T_0 \to \infty`. Unlike the placebo test it needs no donor pool, and
    unlike asymptotic intervals no normality; in the Kansas calibrations its
    intervals attain near-nominal coverage where plain SCM under-covers from
    poor-fit bias. The bands are returned in
    ``res.inference.details["counterfactual_lower" / "counterfactual_upper"]``
    (shaded on the plot) alongside the joint ``["joint_p_value"]`` --
    :func:`mlsynth.utils.bilevel.ridge_inference.conformal_intervals`.

``"scpi"`` -- prediction intervals (Cattaneo, Feng & Titiunik 2021)
    Treats :math:`\tau_T` as a *predictand* (a random variable) and builds
    prediction intervals, decomposing the prediction error as

    .. math::

       \widehat{\tau}_T - \tau_T = e_T - \mathbf{p}_T^\top(\widehat{\boldsymbol{\beta}}
       - \boldsymbol{\beta}_0),

    an out-of-sample shock :math:`e_T` plus an in-sample weight-estimation
    error. The counterfactual prediction band is assembled period-by-period as
    :math:`[\,Y_{\text{fit}} + w_L + e_L,\; Y_{\text{fit}} + w_U + e_U\,]`,
    and the treatment-effect interval is
    :math:`[\,Y_{\text{obs}} - \text{cf}_U,\; Y_{\text{obs}} - \text{cf}_L\,]`.

    * In-sample (:math:`w_L`/:math:`w_U`): a simulation-based bound. With
      :math:`\mathbf{Q} = \mathbf{B}^\top\mathbf{B}/T_0` (donor pre-outcomes),
      :math:`\widehat{\boldsymbol{\Sigma}} = \mathbf{B}^\top
      \mathrm{diag}(\boldsymbol{\omega})\,\mathbf{B} / T_0^2` where :math:`\omega_t =
      \tfrac{T_0}{T_0-\mathrm{df}}(u_t - \mathbb{E}[u_t])^2` (HC1), and pre-period
      residuals :math:`\mathbf{u} = \mathbf{A} - \mathbf{B}\widehat{\mathbf{w}}`,
      draw :math:`\mathbf{G}^\ast \sim
      N(\mathbf{0},\widehat{\boldsymbol{\Sigma}})`. For each draw and predictor
      :math:`\mathbf{p}_T`, solve over the *localised* simplex set

      .. math::

         \min/\max\ \mathbf{p}_T^\top\mathbf{x} \quad\text{s.t.}\quad
         (\mathbf{x}-\widehat{\mathbf{w}})^\top\mathbf{Q}(\mathbf{x}-\widehat{\mathbf{w}})
         - 2\mathbf{G}^{\ast\top}(\mathbf{x}-\widehat{\mathbf{w}}) \le 0,\;
         \textstyle\sum \mathbf{x} = 1,\; \mathbf{x} \ge \boldsymbol{\ell},

      with :math:`\ell_j = \widehat{w}_j` if :math:`\widehat{w}_j < \rho` else
      :math:`0`. The regularisation parameter :math:`\rho` is data-driven and
      capped at :math:`\rho_{\max} = 0.2`; :math:`\mathbf{Q}` is reduced via a
      thresholded eigen-square-root so collinear (near-null) donor directions
      are left unconstrained. :math:`w_L`/:math:`w_U` are the
      :math:`\alpha_1/2` / :math:`1-\alpha_1/2` quantiles of
      :math:`\mathbf{p}_T^\top(\widehat{\mathbf{w}} - \mathbf{x})` across draws.
    * Out-of-sample (:math:`e_L`/:math:`e_U`): a location-scale model,
      :math:`e_T = \mathbb{E}[e] + \sqrt{\mathrm{Var}[e]}\,\varepsilon`. The conditional
      mean and a log-variance scale (capped by the residual IQR, Gaussian
      :math:`\varepsilon`) are estimated by regressing :math:`\mathbf{u}` on the
      active-donor design; ``"ls"`` and ``"empirical"`` use standardized /
      raw residual quantiles.

    ``VanillaSC`` returns the average-effect (ATT) interval in
    ``res.inference.ci_lower``/``ci_upper`` and the full per-period sequence
    (point effects, prediction intervals, counterfactual bands, and the
    in-/out-of-sample components) in ``res.inference.details``. This
    implements the canonical simplex / outcome-only case; for covariate
    backends it uses the same outcome design and is approximate.

    When the outcome and donors are cointegrated -- their levels share a
    common stochastic trend, as with the GDP-per-capita series in the German
    reunification study -- set ``scpi_cointegrated=True`` (scpi's
    ``cointegrated_data``). The in-sample :math:`\mathbb{E}[u]` and
    out-of-sample :math:`\mathbb{E}[e]` models are then fit on first
    differences of the donor design (dropping the first pre-period the
    differencing consumes), with the pre-to-post bridge
    :math:`\Delta \mathbf{p}_1 = \mathbf{p}_1 - \mathbf{b}_{T_0}`. The point
    counterfactual is unchanged; only the prediction bands move (tighter in
    the near-post years, where the differenced predictors carry less
    extrapolation risk). The default is the levels model.

    Alongside the pointwise band, the engine returns a *simultaneous*
    (joint-coverage) band that holds over the whole post-treatment horizon at
    once (scpi's ``simultaneousPredGet``): a uniform in-sample bound -- the
    :math:`\alpha_1/2` / :math:`1-\alpha_1/2` quantile, across post periods, of
    the per-period extreme deviation over draws -- plus the out-of-sample
    component inflated by :math:`\sqrt{\log(T_1+1)}`. It is never tighter than
    the pointwise band. Both appear in ``res.inference.details`` (keys
    ``pi_lower``/``pi_upper`` and ``pi_lower_simultaneous``/
    ``pi_upper_simultaneous``, and the matching ``counterfactual_*`` keys). The
    ``plot_bands`` config field selects which band(s) the observed-vs-synthetic
    plot shades -- ``"pointwise"`` (default), ``"simultaneous"``, or ``"both"``
    (the wider simultaneous band drawn underneath the pointwise one).

    The underlying engine :func:`~mlsynth.utils.vanillasc_helpers.scpi.scpi_intervals`
    also supports scpi's full Table-2 weight-constraint family via
    ``w_constr`` -- ``ols`` / ``simplex`` (the default here) / ``lasso`` /
    ``ridge`` / ``L1-L2`` -- which changes the in-sample compatible set and the
    effective degrees of freedom (:math:`\mathrm{df} = J` for ols, ``#nonzero``
    for lasso, ``#nonzero`` :math:`-1` for simplex, and
    :math:`\sum_k d_k^2/(d_k^2+\lambda)` over the pre-period donor singular
    values :math:`d_k` for ridge / L1-L2). ``VanillaSC`` itself fits a simplex
    control, so it uses the simplex constraint; the ridge constraint is scpi's
    Table-3 inference setting for Amjad, Kim, Shah & Shen (2018) Robust
    Synthetic Control, which :doc:`CLUSTERSC <clustersc>` routes its PCR / RSC
    fit through (``compute_scpi_pi=True, scpi_constraint="ridge"``).

    .. note::

       This is a self-contained, MIT-licensed re-derivation of the
       Cattaneo-Feng-Titiunik algorithm -- it does not import the GPL
       reference package ``scpi``. It is validated to reproduce ``scpi``'s
       ``CI_all_gaussian`` on the Proposition 99 panel to within Monte-Carlo
       error (see ``test_scpi_matches_reference_package``, which is skipped
       unless ``scpi_pkg`` happens to be installed).

       Under staggered adoption (several treated units adopting at possibly
       different times), the same ``inference="scpi"`` produces the
       Cattaneo-Feng-Palomba-Titiunik (2025) cross-unit causal-predictand
       intervals via a dedicated clean-room engine. These reproduce ``scpi``'s
       event-time band to solver tolerance; ``scpi_compat`` selects the
       statistically correct (default) vs. ``scpi``-matching in-sample scaling.
       See :doc:`replications/vanillasc_staggered`.

       The staggered engine also carries the full weight-constraint family:
       ``staggered_spec={"features": [...], "w_constr": "ridge"}`` fits every
       treated unit's synthetic control (and its prediction-interval compatible
       set) under ``ols`` / ``simplex`` (default) / ``lasso`` / ``ridge`` /
       ``L1-L2``, matching ``scpi``'s ``scdataMulti`` weights value-for-value and
       its deterministic inference budgets (``Q_star``, ``lb``, ``df``)
       cell-for-cell on the two-treated Germany panel. This is the only setting
       -- multiple treated units, covariate-adjusted, non-convex weights -- that
       combines all three of ``scpi``'s dimensions at once.

       For the single-treated-unit case, the ``scpi_germany_pi`` benchmark
       cross-checks both bands against a live ``scpi_pkg`` run on German
       reunification: the levels (``scpi_cointegrated=False``) and cointegrated
       (``=True``) ``CI_all_gaussian`` bands each reproduce to Monte-Carlo error
       (width mean difference :math:`\approx 0.04` cointegrated, :math:`\approx
       0.10` levels, at 2000 draws), and the simplex weights match to four
       decimals. The ``scpi_ridge_germany`` benchmark separately cross-checks the
       ridge-constraint machinery (budget :math:`Q`, penalty :math:`\lambda`,
       degrees of freedom) against ``scpi_pkg``'s ``scest`` / ``df_EST`` to
       :math:`10^{-6}`, routed through CLUSTERSC's ``.fit()``.

``"lto"`` -- leave-two-out refined placebo (Lei & Sudijono 2025)
    A design-based randomization test that fixes the two structural weaknesses
    of the ordinary placebo test -- its coarse :math:`\{1/N, 2/N, \dots\}`
    grid and its zero size when :math:`\alpha < 1/N`. It replaces the "one
    turn each" permutation with a *tournament over triples* and reports both a
    naive p-value (``res.inference.p_value``) and a powered one
    (``details["p_powered"]``), together with the Type-I bound and tournament
    tallies. It shares the placebo test's assumptions but is far more powerful
    in small donor pools. See *The leave-two-out refined placebo test* and the
    two theory subsections below for the full treatment.

``"ttest"`` -- debiased SC t-test for the ATT (Chernozhukov, Wüthrich & Zhu 2025)
    A :math:`K`-fold cross-fitting debiasing with a self-normalized statistic
    that is asymptotically :math:`t_{K-1}`, giving the ATT in the familiar
    one-number form :math:`\widehat{\tau} \pm t_{K-1}(1-\alpha/2)\,\mathrm{se}`
    -- robust to misspecification and valid with stationary or non-stationary
    data, with no long-run-variance estimation. The pre-period is split into
    ``ttest_K`` blocks; each block's weights are refit (with the configured
    backend) on its complement, and the held-out block gap removes the SC bias.
    The debiased ATT, ``se``, ``tstat`` and the two-sided ``p_value`` land in
    ``res.inference.details`` with the interval in
    ``res.inference.ci_lower``/``ci_upper``. Set ``ttest_K="auto"`` to choose
    :math:`K` from the SC-residual persistence and the RAE formula (their
    Section 3.2); :math:`K = 3` is the small-:math:`T_0` benchmark. Because it
    only needs :math:`\ell_2`-consistent weights it composes with every backend.
    Supplying ``oracle_weights`` (a ``{donor: weight}`` map) bypasses the weight
    solve and uses those weights in every fold -- the paper's oracle benchmark,
    and a way to plug in externally computed weights.
    Reference: :func:`mlsynth.utils.inferutils.debiased_sc_ttest`; reproduces the
    paper's Table 5 carbon-tax estimate (durable case
    ``benchmarks/cases/cwz_ttest.py``).

``"eiv"`` -- error-in-variables prediction intervals (Hirshberg 2021)
    Normal/:math:`t` prediction intervals from the error-in-variables view of
    synthetic control (Hirshberg 2021, arXiv 2104.08931). The donor pre-outcomes
    you regress on are themselves noisy -- a low-rank signal plus idiosyncratic
    noise, :math:`\mathbf{X} = \mathbf{A} + \boldsymbol{\varepsilon}` -- so SC is
    a least-squares fit with error *in the variables*. Hirshberg's Corollary 3
    gives conditions (a low-rank panel, mild Tikhonov regularisation, and
    :math:`T_0 \to \infty`) under which :math:`\widehat{\tau} = y_e -
    \mathbf{x}_e'\widehat{\boldsymbol{\theta}}` is asymptotically normal with an
    estimable variance :math:`\sigma_\tau = \sigma_e\, p_{\text{eff}}^{-1/2}`,
    where :math:`p_{\text{eff}} = 1/\|\widehat{\boldsymbol{\theta}}\|^2` is the
    weights' participation ratio -- and crucially *without* assuming
    time-stationarity, unit-exchangeability, or the absence of weak factors, the
    invariants the placebo test and earlier theory rely on.

    The estimator (Hirshberg eq. 1) is Tikhonov-regularised least squares; for the
    factor-model case its penalty reduces to ridge with scale :math:`\eta`, and
    :math:`\eta = 1` -- which the paper shows suffices in the ideal case -- is
    ordinary SC, so this mode reads the fitted simplex weights and forms the
    interval. mlsynth estimates :math:`\sigma_\tau` as the (degrees-of-freedom
    corrected) standard deviation of the pre-treatment residuals
    :math:`u_t = y_t - \mathbf{x}_t'\widehat{\boldsymbol{\theta}}`, which are
    distributed like the post-period error :math:`\nu_e -
    \boldsymbol{\varepsilon}_e'\widehat{\boldsymbol{\theta}}`; this consistently
    estimates the *full* per-period error scale
    :math:`\sqrt{\sigma_\nu^2 + \sigma_e^2\|\widehat{\boldsymbol{\theta}}\|^2}`,
    whereas Hirshberg's donor-only :math:`\sigma_e\|\widehat{\boldsymbol{\theta}}\|`
    drops the treated unit's own shock :math:`\nu_e` and under-covers a single
    treated unit with :math:`O(1)` noise (it is valid only in his scaling where
    the treated noise is :math:`p^{-1/2}`-negligible). Intervals use a
    :math:`t(T_0 - \mathrm{df})` reference. The per-period effect intervals,
    counterfactual band, ``sigma_tau`` and ``p_eff`` land in
    ``res.inference.details``, with the ATT interval in
    ``res.inference.ci_lower``/``ci_upper``. Reference:
    :func:`mlsynth.utils.vanillasc_helpers.eiv.eiv_intervals`; the coverage
    behaviour (near-nominal, converging as :math:`T_0` grows) is pinned by the
    Path-B ``eiv_coverage_mc`` benchmark.

The debiased t-test: assumptions and econometric theory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Setup and estimand. The treated unit :math:`j = 1` is untreated through
:math:`T_0` and treated over :math:`\mathcal{T}_2`; the target is the ATT
:math:`\tau \coloneqq |\mathcal{T}_2|^{-1}\sum_{t\in\mathcal{T}_2}(y_{1t}^I -
y_{1t}^N)`. SC predicts the missing :math:`y_{1t}^N` from the donor outcomes
:math:`\mathbf{x}_t \coloneqq (y_{jt})_{j\in\mathcal{N}_0}`; write the linear
prediction model

.. math::

   y_{1t}^N = \mathbf{x}_t^\top \mathbf{w}^\ast + u_t, \qquad
   \mathbf{w}^\ast \coloneqq \operatorname*{argmin}_{\mathbf{w}\in\Delta^{N_0}}
   \mathbb{E}\bigl(y_{1t}^N - \mathbf{x}_t^\top\mathbf{w}\bigr)^2 .

This is a predictive, not a structural, model: the pseudo-true weights
:math:`\mathbf{w}^\ast` need not be "true" SC weights, so the test tolerates
misspecification (for instance a linear factor model under which SC is biased).

Why cross-fitting debiases. The naive SC ATT is biased because the weights are
estimated in high dimension relative to :math:`T_0`; under stationarity that
bias is the same in :math:`\mathcal{T}_1` and :math:`\mathcal{T}_2`. The
:math:`K`-fold cross-fit estimates it from a held-out pre-period block and
subtracts it. Splitting :math:`\{1,\dots,T_0\}` into :math:`K` blocks of length
:math:`r = \min(\lfloor T_0/K\rfloor, T_1)` and refitting
:math:`\widehat{\mathbf{w}}_{(k)}` on each block's complement keeps the weights
approximately independent of the held-out block :math:`H_k`, so

.. math::

   \widehat{\tau}_k = |\mathcal{T}_2|^{-1}\!\!\sum_{t\in\mathcal{T}_2}
   \bigl(y_{1t} - \mathbf{x}_t^\top\widehat{\mathbf{w}}_{(k)}\bigr)
   - |H_k|^{-1}\!\!\sum_{t\in H_k}
   \bigl(y_{1t} - \mathbf{x}_t^\top\widehat{\mathbf{w}}_{(k)}\bigr),
   \qquad \widehat{\tau} = K^{-1}\!\sum_{k=1}^{K} \widehat{\tau}_k ,

where the second (held-out) term estimates the pre-period bias that, under
stationarity, also contaminates the first (post-period) term.

The identifying assumptions are the following (stationary case; Section 4 of the
paper relaxes the first to nonstationary data).

1. Covariance-stationary data. :math:`\{(y_{1t}^N, \mathbf{x}_t)\}` is
   covariance-stationary.

   *Remark.* This is what makes the held-out pre-period gap a valid estimate of
   the post-period bias -- the load-bearing restriction. It is plausibly
   violated by a structural break shortly after :math:`T_0`; the placebo test
   (``inference="placebo"``) can be used to probe it.

2. :math:`\ell_2`-consistent weights.
   :math:`\max_{k} \|\widehat{\mathbf{w}}_{(k)} - \mathbf{w}^\ast\|_2 = o_p(1)`.

   *Remark.* Mild and generic: it holds for SC even when the number of donors
   :math:`N_0` grows with :math:`T_0` (no sparsity needed) and for many
   penalized-regression estimators -- the reason the test rides any backend
   (outcome-only / mscmt / malo / penalized) and, more broadly, any
   :math:`\ell_2`-consistent weighting.

3. Weak dependence. :math:`\{(\mathbf{x}_t, u_t)\}` is :math:`\beta`-mixing
   with sufficient moments and covariance eigenvalues bounded away from zero.

   *Remark.* Satisfied by ARMA, GARCH and many stochastic-volatility processes;
   it rules out unit-root / near-unit-root prediction errors in the stationary
   theory (the nonstationary results in Section 4 handle those separately).

Limiting distribution. Under Assumptions 1-3, as :math:`T_0, T_1 \to \infty`,
the component estimators :math:`(\widehat{\tau}_1,\dots,\widehat{\tau}_K)` are
jointly asymptotically normal but share a common term :math:`\xi_0` (the
post-treatment average), so they are not independent. Estimating the long-run
variance :math:`\sigma^2` is unreliable in small samples, so the test instead
self-normalizes:

.. math::

   \mathbb{T}_K = \frac{\sqrt{K}\,(\widehat{\tau} - \tau)}
   {\widehat{\sigma}_{\widehat{\tau}}}, \qquad
   \widehat{\sigma}_{\widehat{\tau}} = \sqrt{1 + \tfrac{Kr}{T_1}}\,
   \Bigl(\tfrac{1}{K-1}\textstyle\sum_{k}(\widehat{\tau}_k - \widehat{\tau})^2
   \Bigr)^{1/2}.

The shared :math:`\xi_0` cancels between numerator and denominator -- this is
exactly what the :math:`\sqrt{1 + Kr/T_1}` rescale corrects for -- giving an
asymptotically pivotal :math:`\mathbb{T}_K \xrightarrow{d} t_{K-1}`, a
Student-:math:`t` law with :math:`K-1` degrees of freedom. No LRV estimate, no
subsampling, no permutation distribution is needed; self-normalization also
delivers higher-order refinements that explain the strong small-sample
performance. Inverting the statistic gives
:math:`\widehat{\tau} \pm t_{K-1}(1-\alpha/2)\,
\widehat{\sigma}_{\widehat{\tau}}/\sqrt{K}` with asymptotic coverage
:math:`1-\alpha` (a confidence interval when :math:`\tau` is fixed, a prediction
interval when it is random).

Efficiency and nonstationarity. The debiased SC estimator's asymptotic variance
is no larger than difference-in-differences', whether or not SC is correctly
specified, and the t-test stays valid when DID's common-trends assumption
fails. With nonstationary data it is valid when all units share a common
nonstationarity, or under bounded heterogeneity in the deviations (then SC must
be correctly specified); its robustness improves as :math:`T_0` grows. Reach for
it with one treated unit and enough post-periods (the asymptotics send
:math:`T_1 \to \infty`); when :math:`T_1` is very small or a break hits just
after :math:`T_0`, prefer the conformal or SCPI modes, which are valid for fixed
:math:`T_1`.

Choosing :math:`K`. ``ttest_K`` trades interval length against coverage
accuracy: larger :math:`K` shortens the interval -- its relative asymptotic
efficiency (the RAE of eq. 14, reproduced in
:func:`mlsynth.utils.inferutils.rae`) rises from about 64% at :math:`K=3` toward
92% at :math:`K=10` for :math:`c_0 = T_0/T_1 = 30/16` -- but degrades coverage
when :math:`T_0` is small or the prediction errors are persistent. ``ttest_K=3``
is the robust small-:math:`T_0` benchmark; ``ttest_K="auto"`` gauges persistence
by an AR(1) fit to the SC residuals and selects :math:`K` per Section 3.2 (it
bumps to :math:`K=4` when persistence is low and climbs further only when
:math:`T_0` is large enough to keep each block well-sized).

How the SCPI machinery works (one fit)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``scpi_intervals(y, Y0, pre, W, ...)`` takes the fitted donor weights
:math:`\widehat{\mathbf{w}}` (from *any* backend), the donor outcome matrix, and
the number of pre-treatment periods, and runs the following steps. Let
:math:`\mathbf{A} = \mathbf{y}_{1,\mathcal{T}_1}` be the treated pre-outcomes,
:math:`\mathbf{B} = \mathbf{Y}_{0,\mathcal{T}_1}`
the donor pre-outcomes, :math:`\mathbf{P}` the donor post-outcomes, and
:math:`\mathbf{u} = \mathbf{A} - \mathbf{B}\widehat{\mathbf{w}}` the pre-period
residuals.

1. Degrees of freedom. For the simplex, :math:`\mathrm{df} =
   (\#\{\widehat{w}_j \neq 0\}) - 1`, giving the HC1 correction
   :math:`\mathrm{vc} = T_0/(T_0-\mathrm{df})`.

2. Regularisation parameter :math:`\rho`. The data-driven ``type-1`` value
   :math:`\rho = \tfrac{\sigma_u}{\min_j \mathrm{sd}(\mathbf{B}_j)}
   \sqrt{\log(N_0)\, d_0 \log T_0}/\sqrt{T_0}`, capped at
   :math:`\rho_{\max}=0.2` (with a fallback bump if it comes out below
   :math:`0.001`). :math:`\rho` defines the "active" donor set
   :math:`\{\,j : \widehat{w}_j > \rho\,\}`.

3. Conditional mean & variance. Regress :math:`\mathbf{u}` on the active-donor
   design :math:`[\,\mathbf{B}_{\cdot,\text{active}},\,\mathbf{1}\,]` to get
   :math:`\mathbb{E}[\mathbf{u}]` (the ``u_missp`` step), then
   :math:`\omega_t = \mathrm{vc}\,(u_t - \mathbb{E}[u_t])^2`. Form
   :math:`\mathbf{Q} = \mathbf{B}^\top\mathbf{B}/T_0` and
   :math:`\widehat{\boldsymbol{\Sigma}} =
   \mathbf{B}^\top\mathrm{diag}(\boldsymbol{\omega})\mathbf{B}/T_0^2`,
   and its matrix square root :math:`\boldsymbol{\Sigma}^{1/2}`.

4. Localised feasible set. Lower bounds
   :math:`\ell_j = \widehat{w}_j` if :math:`\widehat{w}_j < \rho` else :math:`0`
   (near-binding donors are pinned at their tiny weight; active donors may move
   down to zero). :math:`\mathbf{Q}` is reduced by a thresholded
   eigen-square-root so the near-null (collinear) directions are left
   unconstrained.

5. In-sample simulation. For each of ``scpi_sims`` draws
   :math:`\mathbf{G}^\ast = \boldsymbol{\Sigma}^{1/2}\,\mathbf{z}`,
   :math:`\mathbf{z}\sim N(\mathbf{0},\mathbf{I})`, and each post
   predictor :math:`\mathbf{p}_T`, solve the small conic program in
   :math:`\mathbf{x}` (donor weights) twice -- minimise and maximise
   :math:`\mathbf{p}_T^\top\mathbf{x}` subject to
   :math:`(\mathbf{x}-\widehat{\mathbf{w}})^\top\mathbf{Q}(\mathbf{x}-\widehat{\mathbf{w}})
   - 2\mathbf{G}^{\ast\top}(\mathbf{x}-\widehat{\mathbf{w}})\le 0`,
   :math:`\sum \mathbf{x} = 1`, :math:`\mathbf{x}\ge\boldsymbol{\ell}`. Record
   :math:`\mathbf{p}_T^\top(\widehat{\mathbf{w}}
   - \mathbf{x})` for each branch; :math:`w_L`/:math:`w_U` are the
   :math:`\alpha_1/2` / :math:`1-\alpha_1/2` quantiles across draws.

6. Out-of-sample band. From the location-scale model on :math:`\mathbf{u}` get
   :math:`e_L`/:math:`e_U` per post period (Section above).

7. Assemble. Counterfactual band
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

Cointegration (``scpi_cointegrated=True``). When the treated and donor series
are cointegrated, their levels share a common stochastic trend and are
individually :math:`I(1)`; regressing residuals on the level donor design then
extrapolates a non-stationary predictor into the post period. The cointegrated
model instead fits the uncertainty on first differences: in step 3 the
conditional-mean design becomes :math:`\Delta\mathbf{B}_t = \mathbf{B}_t -
\mathbf{B}_{t-1}` (and in step 6 the out-of-sample design and predictand are
differenced, with the pre-to-post bridge :math:`\Delta\mathbf{p}_1 =
\mathbf{p}_1 - \mathbf{b}_{T_0}`). The first pre-period, which differencing
consumes, is dropped from these designs; the level design :math:`\mathbf{B}`
still drives :math:`\mathbf{Q}`/:math:`\boldsymbol{\Sigma}` and the QP in steps
3-5, and :math:`Y_{\text{fit}} = \mathbf{P}\widehat{\mathbf{w}}` is unchanged.
Only the bands move. This mirrors ``scpi``'s ``cointegrated_data`` exactly (the
``scpi_germany_pi`` benchmark reproduces both bands to Monte-Carlo error).

.. code-block:: python

   import pandas as pd
   from mlsynth import VanillaSC

   df = pd.read_csv("basedata/scpi_germany.csv")[["country", "year", "gdp"]].dropna()
   df["treated"] = ((df.country == "West Germany") & (df.year >= 1991)).astype(int)

   res = VanillaSC({
       "df": df, "outcome": "gdp", "treat": "treated",
       "unitid": "country", "time": "year",
       "inference": "scpi", "scpi_cointegrated": True,   # GDP levels are I(1)
       "scpi_sims": 2000, "seed": 8894, "display_graphs": False,
   }).fit()

   det = res.inference.details          # per-period bands
   print(res.inference.ci_lower, res.inference.ci_upper)   # ATT prediction interval
   # det["counterfactual_lower"] / ["counterfactual_upper"] is scpi's CI_all_gaussian

Composing SCPI with the backends
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``backend`` (how :math:`\mathbf{w}` is estimated) and ``inference`` (how
uncertainty is quantified) are orthogonal -- any of the four backends pairs with
any of
the three inference modes:

.. code-block:: python

   VanillaSC({..., "backend": "mscmt", "inference": "scpi"}).fit()
   VanillaSC({..., "backend": "malo",  "inference": "scpi"}).fit()

The pipeline fits the weights with the chosen backend and hands the resulting
``res.W`` to ``scpi_intervals``. Two things to keep in mind:

* The in-sample simulation rebuilds :math:`\mathbf{Q}` and
  :math:`\widehat{\boldsymbol{\Sigma}}` from
  the donor pre-outcomes :math:`\mathbf{B}`, treating :math:`\widehat{\mathbf{w}}`
  as simplex weights. With outcome-only this is the exact Cattaneo-Feng-Titiunik
  interval (the case validated against ``scpi``). With mscmt/malo the
  weights were also shaped by the covariate predictors, so SCPI uses the
  outcome design as a stand-in -- it is approximate for covariate backends.
  The point effects, the ATT, and the out-of-sample band are unaffected; only
  the in-sample :math:`w_L`/:math:`w_U` term carries the approximation.
* Read the SCPI interval alongside :math:`\text{v\_agreement}`. When the
  predictor weights are non-identified (``v_agreement`` near 1, e.g. Prop 99
  with lagged outcomes) the *point* counterfactual is still pinned, but the
  covariate-matched solution is fragile; the placebo test, which is exact for
  any backend, is the conservative cross-check.

The leave-two-out (LTO) refined placebo test
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

What it is. The ordinary placebo test (above) gives each of the :math:`N`
units exactly one turn as the pseudo-treated unit and ranks the real treated
unit's fit statistic against those :math:`N` values. That is its weakness: the
p-value can only land on the grid :math:`\{1/N, 2/N, \dots, 1\}`, and at a
conventional level like :math:`\alpha = 0.05` with a small donor pool the test
is either coarse or -- when :math:`\alpha < 1/N` -- literally unable to reject
(its size is zero). The Lei-Sudijono (2025) leave-two-out test keeps the
same design-based logic but replaces the "one turn each" permutation with a
tournament over triples. Think of every triple :math:`\{i, j, I\}` (two
controls and the treated unit) as a match: leave all three out of the donor
pool, build a synthetic control for each of them from the remaining
:math:`N-3` units, score each by its post/pre RMSPE ratio, and the unit with
the largest ratio "wins" the match. The treated unit *should* win often if the
treatment had a real effect (a large post-period gap relative to a tight
pre-period fit). The p-value is the fraction of matches the treated unit does
not win,

.. math::

   p_{\mathrm{naive\text{-}LTO}}
     = \frac{1}{(N-1)(N-2)} \sum_{i \neq j}
       \mathbf{1}\bigl\{R_{i,j,I;I} \le \max(R_{i,j,I;i}, R_{i,j,I;j})\bigr\},

where :math:`R_{i,j,I;k} = \lvert S_{\text{ratio-RMSPE}}(\mathbf{y}_k, \widehat{\mathbf{y}}_k)\rvert`
is the score of unit :math:`k` when the pool excludes :math:`\{i, j, I\}`.
Because there are :math:`\binom{N-1}{2}` matches rather than :math:`N`, the
p-value lives on an :math:`O(N^2)`-fine grid -- the granularity problem
disappears.

Two p-values. ``res.inference.p_value`` is the *naive* LTO p-value above.
``res.inference.details["p_powered"]`` is the *powered* variant
:math:`p_{\mathrm{naive\text{-}LTO}} - c(N, \alpha) + \delta`, which shifts the
naive value down by the largest amount the discrete Type-I bound allows
(``powered_offset_c``), strictly increasing power. The powered value is a
decision rule tied to one :math:`\alpha` -- reject when it is
:math:`\le \alpha` (``reject_at_alpha``) -- not a general-purpose p-value, so do
not compare it across levels or report it as "the" p-value.

Verification (authors' own code). ``benchmarks/cases/lto_refined_placebo.py``
cross-validates mlsynth's LTO against the authors' own replication code
(`tsudijon/LeaveTwoOutSCI <https://github.com/tsudijon/LeaveTwoOutSCI>`_,
MIT-licensed) on all three of the paper's empirical applications -- California
Proposition 99, West German reunification, and the Basque Country. On a shared
outcome-only synthetic control (so the check isolates the pair-loop inference
machinery from the SC solver, which is validated separately), mlsynth reproduces
the authors' naive LTO p-value to the digit on every panel: :math:`0.10384`
(Prop 99), :math:`0.00833` (West Germany) and :math:`0.69167` (Basque), with the
loss and pair counts identical and the treated unit's post/pre RMSPE ratio
matching to solver tolerance.

LTO: design-based assumptions and econometric theory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The LTO test is design-based, not outcome-model-based: the potential
outcomes are treated as fixed, and all randomness comes from *which unit got
treated*. Its validity rests on two assumptions.

* Uniform assignment. The treated index :math:`I` is uniformly distributed
  over :math:`\{1, \dots, N\}` -- a priori, any unit was equally likely to be
  the treated one. Under the null this makes the :math:`N` units *exchangeable*,
  which is exactly what licenses the tournament. This holds by construction
  in (cluster-)randomized experiments. In observational work it is a modelling
  choice: it is most defensible when the treated unit is comparable to the
  donors (often after covariate adjustment), and in quasi-experimental settings
  -- e.g. natural disasters, where which locality is hit is plausibly close to
  random over a small comparable region.
* Sharp null. The hypothesis tested is Fisher's sharp null
  :math:`H_0 : y_{it}^I = y_{it}^N` for all :math:`t > T_0` (no effect for
  *any* unit in any post period), or a known-:math:`\tau` additive version
  :math:`y_{it}^I = y_{it}^N + \tau_{it}`. Sharpness is what lets the test
  impute every unit's counterfactual under the null and so run the tournament.

Under these, the test has a finite-sample Type-I error guarantee (no
large-:math:`N`, no long-:math:`T`, no asymptotics):

.. math::

   \mathbb{P}_{H_0}\!\left(p_{\mathrm{naive\text{-}LTO}} \le \alpha\right)
     \le \frac{\lfloor N f(N, \alpha)\rfloor}{N},

reported as ``type_i_bound``. This bound is never worse than the
approximate-placebo bound :math:`(\lfloor N\alpha\rfloor + 1)/N`, and for the
levels and sizes typical of SCM applications (:math:`\alpha \in \{0.01, 0.02\}`
for :math:`6 < N < 200`; :math:`\alpha = 0.05` for most :math:`N`) it is
*identical* to it -- so switching to LTO costs nothing in worst-case Type-I
error. Crucially, the placebo bound is *tight* whereas the LTO bound generally
is not: in practice the LTO test's actual Type-I error is often strictly
below :math:`\alpha`, i.e. it can be unconditionally valid even when
:math:`\alpha < 1/N`.

Two further theoretical properties matter in practice:

* Consistency where the placebo test fails (Theorem 6.1). When
  :math:`\alpha < 1/N`, the LTO test is *uniformly consistent* -- its power goes
  to 1 as the effect size grows. The approximate placebo test is not: in
  this regime it can have essentially zero power no matter how large the true
  effect (zero if :math:`N` is even, :math:`\le 1/N` if odd). This is the single
  strongest reason to prefer LTO in small donor pools.
* Confidence regions. Inverting the additive-:math:`\tau` test
  (:math:`\{\theta : p_{\mathrm{naive\text{-}LTO}}(\theta) > \alpha\}`) yields a
  region for the post-period effect path with guaranteed coverage
  :math:`\ge 1 - \lfloor N f(N,\alpha)\rfloor / N`. (mlsynth currently reports
  the p-values; the inversion is a straightforward extension.)

Methodologically, the LTO test is a *new* kind of randomization inference: it
generalises the Jackknife+ of Barber et al. (2021) (which leaves one point
out and so still has :math:`1/N` granularity) and is distinct from classical
permutation/rank inference. It also -- unlike most asymptotic SCM inference --
does not simplify the synthetic-control construction: the full
weight/predictor machinery (any ``VanillaSC`` backend) is re-run inside every
match, so the test reflects the estimator you actually use.

When the LTO assumptions are violated
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The sharp null is testable and usually uncontroversial; the uniform
assignment assumption is where care is needed.

* Selection on outcomes / non-comparable treated unit. If the treated unit
  was chosen *because* of its (anticipated) trajectory, or is structurally
  unlike every donor, exchangeability fails and the Type-I guarantee no longer
  holds. The usual remedy is to restore comparability through the
  specification -- match on covariates, restrict the donor pool to genuinely
  similar units -- before trusting any placebo-type p-value.
* Known non-uniform assignment. When the treatment probabilities
  :math:`\pi_k` are known or estimable (e.g. seismic risk for an
  earthquake study), Lei-Sudijono give a *weighted* LTO p-value
  :math:`p_{\text{w-LTO}}(\pi)` that reweights each match by
  :math:`\pi_j\pi_k / ((1-\pi_I)^2 - \sum_{l\neq I}\pi_l^2)` and reduces to the
  naive value when :math:`\pi_i \equiv 1/N`.
* Sensitivity analysis (the :math:`\Gamma` ). Rather than commit to
  uniformity, one can ask *how far* from it the design could be before the
  conclusion flips. Following Rosenbaum, constrain
  :math:`\pi_i \in [\tfrac{1}{\Gamma N}, \tfrac{\Gamma}{N}]` and find the
  smallest :math:`\Gamma \ge 1` at which the worst-case weighted p-value crosses
  :math:`\alpha`. In the paper, Prop 99 tolerates :math:`\Gamma \approx 1.4`
  (robust) while German reunification flips at only :math:`\Gamma \approx 1.1`
  (fragile). The weighted p-value and :math:`\Gamma` search require solving a
  non-convex (NP-hard) quadratic program and are not yet implemented in
  ``VanillaSC``; the uniform-assignment naive/powered p-values are.

Choosing among placebo, LTO, and SCPI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Prefer LTO over the ordinary placebo whenever the donor pool is small --
  especially in the :math:`\alpha < 1/N` regime (e.g. :math:`N \le 20` at
  :math:`\alpha = 0.05`), where the placebo test cannot reject and LTO can. The
  ``powered`` variant almost Pareto-improves the placebo: same worst-case
  Type-I error, more power. Both share the *same assumptions*, so LTO is close
  to a free upgrade.
* Keep the ordinary placebo when you want the most familiar, widely-reported
  statistic, when :math:`N` is large enough that granularity is a non-issue, or
  as a cheap (:math:`O(N)` vs :math:`O(N^2)`) cross-check.
* Reach for SCPI when the question is *how big* the effect is (a prediction
  interval / confidence statement on the magnitude), not just *whether* there is
  one. SCPI rests on different (model-based, conditional) foundations than the
  design-based placebo/LTO tests, so the two are complementary: LTO answers
  "is the treated unit special?" by randomization, SCPI quantifies the effect's
  uncertainty.

Empirical replications
----------------------

VanillaSC reproduces the three canonical synthetic-control studies on their
original datasets -- California / Proposition 99 (ADH 2010; synthetic
Utah/Nevada/Montana/Colorado/Connecticut, ATT :math:`\approx -19` packs),
German reunification (ADH 2015; Austria-dominant donor pool, negative ATT) and
the Basque Country (Abadie-Gardeazabal 2003; Cataluna :math:`\approx 0.8` +
Madrid :math:`\approx 0.2`, ATT :math:`\approx -0.68`). See the dedicated
replication page, :doc:`replications/vanillasc`, for the full datasets, code and
donor-weight tables. These are locked as regression tests in
``mlsynth/tests/test_vanillasc_replications.py``.

Staggered adoption: multiple treated units
-------------------------------------------

Everything above concerns a single treated unit. ``VanillaSC`` also handles the
case where several units adopt treatment, at possibly different times -- the
*staggered adoption* design of Cattaneo, Feng, Palomba and Titiunik (2025). The
switch is automatic: when ``dataprep`` finds more than one treated unit it groups
them into adoption *cohorts* (one per distinct treatment-start time) instead of
raising, and ``fit()`` returns the aggregated design described here. No new
configuration is required; ``inference="scpi"`` extends to the cross-unit
intervals.

Let there be :math:`\iota` treated units, indexed :math:`i = 1, \dots, \iota`,
with unit :math:`i` adopting at :math:`T_0^{(i)}` and observed for
:math:`T_1^{(i)}` post-treatment periods. The donor pool is the set of units that
are *never* treated over the whole sample. Each treated unit is fit on its own
pre-treatment window with its own simplex synthetic control -- exactly the
outcome-only program above, run :math:`\iota` times,

.. math::

   \widehat{\mathbf{w}}^{(i)} = \operatorname*{argmin}_{\mathbf{w}\in\Delta^{N_0}}
   \bigl\| \mathbf{y}^{(i)}_{1,\mathcal{T}_1^{(i)}}
   - \mathbf{Y}_{0,\mathcal{T}_1^{(i)}}\mathbf{w} \bigr\|_2^2 ,

so the per-unit counterfactual is
:math:`\widehat{y}^{(i)}_{1t} = (\mathbf{Y}_0\widehat{\mathbf{w}}^{(i)})_t` and
the per-unit, per-period effect is the gap
:math:`\widehat{\tau}_{i,t} = y^{(i)}_{1t} - \widehat{y}^{(i)}_{1t}`. Indexing by
*event time* :math:`\ell \ge 1` (the :math:`\ell`-th post-treatment period for a
unit, so calendar time :math:`T_0^{(i)} + \ell`) lines the units up on a common
clock, :math:`\widehat{\tau}_{i,\ell}`.

The four identifying assumptions carry over unit by unit (each treated unit must
admit a convex-hull pre-fit, no anticipation, a stable outcome model). Staggered
adoption adds two requirements. First, the donor pool must stay clean for every
cohort: the never-treated units are untreated throughout, and the treated units
are excluded from each other's donor pools, so no treated unit's effect leaks
into another's counterfactual (no cross-unit interference). Second, the
event-time average is only comparable across units where all are observed, which
is why the event study below is *balanced* -- truncated at
:math:`\ell \le \min_i T_1^{(i)}`.

How the staggered ATT is computed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The per-unit gaps aggregate into the causal predictands of Cattaneo, Feng,
Palomba and Titiunik (2025). Writing :math:`\mathcal{N}_{\text{tr}}` for the
treated units and :math:`\mathcal{P}_i` for unit :math:`i`'s post periods:

* TSUS -- the unit-by-period effect :math:`\widehat{\tau}_{i,t}` itself (no
  aggregation).
* TAUS -- the per-unit time average,
  :math:`\widehat{\tau}^{\text{TAUS}}_i = \tfrac{1}{|\mathcal{P}_i|}
  \sum_{t\in\mathcal{P}_i}\widehat{\tau}_{i,t}`. This is each treated unit's own
  ATT and is reported per unit.
* TSUA -- the event-time average treatment effect on the treated, averaging
  across the units present at event time :math:`\ell`,

  .. math::

     \widehat{\tau}^{\text{TSUA}}_\ell
     = \frac{1}{\iota}\sum_{i=1}^{\iota}\widehat{\tau}_{i,\ell},
     \qquad \ell = 1,\dots,\min_i T_1^{(i)} ,

  the staggered analogue of an event study: it is the ATT by time-since-adoption,
  not by calendar period.
* TAUA -- the overall ATT, the cell-weighted average of every post-treatment
  unit/period effect,
  :math:`\widehat{\tau}^{\text{TAUA}}
  = \bigl(\sum_i|\mathcal{P}_i|\bigr)^{-1}
  \sum_i\sum_{t\in\mathcal{P}_i}\widehat{\tau}_{i,t}`.

The overall number reported in ``res.effects.att`` is :math:`\widehat{\tau}^{
\text{TAUA}}` (the post-period-weighted mean of the per-unit ATTs); the
event-study series :math:`\{\widehat{\tau}^{\text{TSUA}}_\ell\}` is in
``res.additional_outputs["event_study"]``.

Inference for the aggregated predictands
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With ``inference="scpi"`` each treated unit carries the single-unit SCPI band of
the previous section, and the cross-unit (TSUA) intervals are built by a
dedicated, self-contained engine
(:mod:`mlsynth.utils.vanillasc_helpers.staggered_engine`) that reimplements the
multiple-treated-unit construction from the published methodology. It does not
import the GPL ``scpi`` package and is validated against it to solver tolerance.

The prediction error of an aggregated predictand decomposes, as in the scalar
case, into an in-sample term (the weight-estimation error) and an out-of-sample
term (the irreducible forecast error), and the band is
:math:`[\,\text{point} - \overline{M}_{\text{in}} - \overline{M}_{\text{out}},\;
\text{point} - M_{\text{in}} - M_{\text{out}}\,]`.

* In-sample. The :math:`\iota` per-unit conic programs (step 5 of the scalar
  machinery) are simulated under a *joint* Gaussian draw whose covariance is
  block-diagonal across units -- the units' weight-estimation errors are taken as
  independent, which is what licenses averaging them. For each draw the worst-case
  prediction errors are combined with the predictand's loadings (for TSUA, weight
  :math:`1/\iota` on each unit at event time :math:`\ell`) and the
  :math:`\alpha_1/2` / :math:`1-\alpha_1/2` quantiles are read off the aggregated
  draws.
* Out-of-sample. The per-unit location-scale models are aggregated the same way
  -- averaged across the units at each event time -- to give the out-of-sample
  bounds for the averaged forecast.

A scaling subtlety is worth stating because it controls the width of the
event-time band. The average of :math:`\iota` independent per-unit in-sample
errors has an interval that scales as :math:`1/\iota`. The ``scpi`` package, by
contrast, scales its published time-aggregated in-sample interval as
:math:`1/\iota^{2}` (it divides the predictand matrix by :math:`\iota` once when
forming the average and again when simulating the draws). ``VanillaSC`` defaults
to the statistically correct :math:`1/\iota` and exposes a ``scpi_compat`` flag
to reproduce ``scpi``'s published numbers exactly:

.. code-block:: python

   VanillaSC({..., "inference": "scpi", "scpi_compat": False}).fit()  # default, 1/iota
   VanillaSC({..., "inference": "scpi", "scpi_compat": True}).fit()   # matches scpi, 1/iota^2

With :math:`\iota = 2` the default in-sample band is exactly twice ``scpi``'s at
every event time; the point estimates and the out-of-sample term are identical
either way. The discrepancy, and the machine-precision check that isolates it,
are documented on the replication page :doc:`replications/vanillasc_staggered`.

What ``fit()`` returns
^^^^^^^^^^^^^^^^^^^^^^^

A staggered fit populates:

* ``res.effects.att`` -- the overall ATT (TAUA), with ``additional_effects``
  carrying ``per_unit_att`` and the cohort count;
* ``res.sub_method_results`` -- a per-unit record (keyed by treated-unit name)
  with that unit's weights, observed and counterfactual paths, gap, ATT and,
  under ``inference="scpi"``, its per-period and average-effect bands;
* ``res.additional_outputs["event_study"]`` -- the balanced TSUA series
  :math:`\{\widehat{\tau}^{\text{TSUA}}_\ell\}` keyed by event time;
* ``res.additional_outputs["event_study_intervals"]`` -- under
  ``inference="scpi"`` with more than one treated unit, the TSUA prediction
  intervals per event time: the ``effect_ci`` on
  :math:`\widehat{\tau}^{\text{TSUA}}_\ell`, the ``synthetic_ci`` on the averaged
  counterfactual (directly comparable to ``scpi``'s ``CI_all_gaussian``), and the
  in-sample-only band.

.. code-block:: python

   import pandas as pd
   from mlsynth import VanillaSC

   df = pd.read_csv("basedata/scpi_germany.csv")
   df["status"] = 0
   df.loc[(df.country == "West Germany") & (df.year >= 1991), "status"] = 1
   df.loc[(df.country == "Italy") & (df.year >= 1992), "status"] = 1

   res = VanillaSC({"df": df, "outcome": "gdp", "treat": "status",
                    "unitid": "country", "time": "year",
                    "inference": "scpi", "display_graphs": False}).fit()
   res.effects.att                                   # overall ATT (TAUA)
   res.additional_outputs["event_study"]             # TSUA by event time
   res.additional_outputs["event_study_intervals"]   # TSUA prediction intervals

The construction is cross-validated against ``scpi`` on this exact panel: the
event-time bands reproduce ``scpi``'s to solver tolerance (durable benchmarks
``scpi_staggered`` and ``scpi_staggered_pi``). See
:doc:`replications/vanillasc_staggered`.

Covariate (multi-feature) matching
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Everything above matches on the outcome alone. To match on several series at
once -- scpi's *features* framework, with covariate adjustment and cointegration
-- attach a ``staggered_spec`` to the config. The spec is shared across treated
units and never names them; the treated units come from the treatment indicator,
as everywhere else in mlsynth.

.. code-block:: python

   res = VanillaSC({"df": df, "outcome": "gdp", "treat": "status",
                    "unitid": "country", "time": "year",
                    "inference": "scpi", "display_graphs": False,
                    "staggered_spec": {
                        "features": ["gdp", "trade"],     # match GDP and trade
                        "cov_adj": ["constant", "trend"], # per-feature adjustment
                        "constant": True,                 # global constant
                        "cointegrated": True}}).fit()     # difference the data

With a ``staggered_spec`` each treated unit is fit on the stacked features with
the covariate-adjustment terms partialled out (and the data differenced when
``cointegrated`` is set), and ``fit()`` returns the same three predictands -- the
per-unit ATTs, the event study and the overall ATT -- now with covariate
matching. The spec mirrors scpi's ``scdataMulti`` arguments (``features``,
``cov_adj``, ``constant``, ``cointegrated_data``) but applies them uniformly:
mlsynth detects who is treated, so there is no per-unit naming. This reproduces
scpi's multiple-treated illustration; on the Germany panel the per-unit average
effects match scpi's ``scest`` (West Germany :math:`-1.75`, Italy
:math:`-0.89`) and the event-time prediction intervals match its
``CI_all_gaussian`` (durable benchmark ``scpi_staggered_covariate``).

.. note::

   scpi's multi-feature design produces duplicate donor-column names that recent
   ``scikit-learn`` releases reject, so the upstream package cannot compute these
   covariate prediction intervals in a current environment; mlsynth's clean-room
   engine does, and was validated against scpi with a one-line column-name
   coercion.

Ridge augmentation (Augmented SCM)
----------------------------------

The ridge-augmented synthetic control of Ben-Michael, Feller & Rothstein
(2021) -- ``progfunc="Ridge"`` in the ``augsynth`` R package -- is not a
separate estimator but a bias-correction layer on top of the simplex SCM.
Given simplex weights :math:`\mathbf{w}` and the centered pre-treatment outcomes
:math:`\mathbf{A} = \mathbf{X}_1 - \bar{\mathbf{X}}`,
:math:`\mathbf{B} = \mathbf{X}_0 - \bar{\mathbf{X}}`, it adds a ridge correction
that closes the residual pre-treatment imbalance the simplex cannot,

.. math::

   \mathbf{w}_{\text{aug}} = \mathbf{w} + (\mathbf{A} - \mathbf{B}^\top\mathbf{w})^\top
       \left(\mathbf{B}\mathbf{B}^\top + \lambda \mathbf{I}\right)^{-1} \mathbf{B},

at the cost of leaving the simplex (the augmented weights may go negative and
need not sum to one). Because any base :math:`\mathbf{w}` can be augmented, the
capability lives in the bilevel engine
(:func:`mlsynth.utils.bilevel.ridge_augment.ridge_augment_weights`) and rides
along wherever the solver goes. The penalty :math:`\lambda` is chosen by
leave-one-period-out cross-validation (augsynth's 1-SE rule); inference is by
the conformal permutation test of Chernozhukov, Wüthrich & Zhu (2021)
(:func:`mlsynth.utils.bilevel.ridge_inference.conformal_pvalue`).

When to prefer augmentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Plain SCM is justified only when the pre-treatment fit is excellent: when
the treated unit lies inside the convex hull of the donors' lagged outcomes the
simplex weights balance :math:`\mathbf{X}_1` exactly and the estimator is (near)
unbiased (Abadie, Diamond & Hainmueller 2010). Outside the hull -- few donors, a
long or high-dimensional pre-period, an outlying treated unit -- no convex
combination fits, and the residual imbalance
:math:`\mathbf{X}_1 - \mathbf{X}_0^\top\widehat{\mathbf{w}}^{\mathrm{scm}}`
turns into bias. SCM has no way to correct it.

Augmented SCM is the middle ground. With :math:`\widehat{\mathbf{m}}` the ridge
outcome model, the ASCM estimate is the SCM estimate plus an estimate of that
bias,

.. math::

   \widehat{y}_{1T}^{\,\mathrm{aug},N} = \underbrace{\textstyle\sum_{j \in \mathcal{N}_0}
   \widehat{w}_j^{\,\mathrm{scm}} y_{jT}}_{\text{SCM}}
   + \Big( \widehat{m}_T - \textstyle\sum_{j \in \mathcal{N}_0}\widehat{w}_j^{\,\mathrm{scm}}
   \widehat{m}_{jT} \Big),

exactly analogous to bias correction for inexact matching (Abadie & Imbens 2011)
and connected to doubly-robust estimation (Robins, Rotnitzky & Zhao 1994). Ridge
ASCM is itself a *penalized* SCM whose penalty is on deviations from the
simplex weights: it starts at the SCM solution and extrapolates beyond the
hull (admitting negative weights) only as far as needed. :math:`\lambda` sets the
amount -- :math:`\lambda \to \infty` recovers plain SCM (no extrapolation),
:math:`\lambda \to 0` drives the pre-fit to zero (full extrapolation). That is a
bias-variance dial: augmentation removes imbalance bias at the cost of a larger
weight norm / extrapolation variance, and :math:`\lambda` (leave-one-period-out
CV, one-standard-error rule) negotiates it.

The authors' practical rule -- and ours -- is to decide from the estimated bias
itself: it is the imbalance term above, in the units of the estimand, and the
first quantity ASCM computes. If it is large relative to the effect you expect,
augment; if pre-fit is already excellent the correction is negligible and ASCM
and SCM coincide. Two diagnostics accompany it: the pre-treatment RMSE
(:math:`\lVert \mathbf{X}_1 - \mathbf{X}_0^\top\widehat{\mathbf{w}}\rVert/\sqrt{T_0}`,
the imbalance that remains) and the extrapolation distance
(:math:`\lVert\widehat{\mathbf{w}}^{\mathrm{aug}} - \widehat{\mathbf{w}}^{\mathrm{scm}}\rVert
/\sqrt{N_0}`, how far the weights left the simplex). Across the paper's calibrated
DGPs, ASCM has both lower bias *and* lower RMSE than SCM -- gains largest under
misspecification and poor fit, modest when SCM already fits well.

Auxiliary covariates enter in either of augsynth's two ways: parallel
(``residualize=False``, the default) standardizes the covariates to the
outcome scale and stacks them as extra matching rows; residualized
(``residualize=True``) regresses the covariates out of the outcomes, matches on
the residuals, and restores covariate balance with an add-back on the weights.

Example
^^^^^^^

Augmented SCM is a *mode* of :class:`~mlsynth.estimators.vanillasc.VanillaSC` --
set ``augment="ridge"`` (and ``inference="conformal"`` for the CWZ prediction
intervals). Covariates are passed by column name; following mlsynth's
convention, apply any transforms (e.g. ``log``) to the DataFrame yourself first.
``residualize=True`` switches parallel inclusion for the residualized variant.
The four-cell augsynth Kansas ladder, reproduced through the public API:

.. code-block:: python

   import numpy as np, pandas as pd
   from mlsynth import VanillaSC

   df = pd.read_csv("basedata/kansas_ascm.csv")          # long fips x quarter panel
   for c in ("revstatecapita", "revlocalcapita", "avgwklywagecapita"):
       df[c] = np.log(df[c])                             # log the covariates up front
   covs = ["lngdpcapita", "revstatecapita", "revlocalcapita",
           "avgwklywagecapita", "estabscapita", "emplvlcapita"]
   base = dict(df=df, outcome="lngdpcapita", treat="treated",
               unitid="fips", time="year_qtr")

   VanillaSC({**base}).fit().effects.att                              # -0.029  classic SCM
   VanillaSC({**base, "augment": "ridge"}).fit().effects.att          # -0.040  ridge ASCM
   VanillaSC({**base, "augment": "ridge",
              "covariates": covs}).fit().effects.att                  # -0.063  covariate ASCM
   VanillaSC({**base, "augment": "ridge", "covariates": covs,
              "residualize": True}).fit().effects.att                 # -0.057  residualized

   # conformal prediction intervals (and a plotted band with display_graphs=True):
   res = VanillaSC({**base, "augment": "ridge", "inference": "conformal"}).fit()
   res.inference.ci_lower, res.inference.ci_upper        # ATT prediction interval
   res.inference.details["joint_p_value"]                # conformal joint-null p-value

.. _vanillasc-ascm-verification:

Verification
^^^^^^^^^^^^

The augmentation is validated against ``augsynth`` on its flagship Kansas
tax-cut study (quarterly log GDP per capita): the de-biasing ladder --
classic SCM (ATT :math:`-0.029`), ridge ASCM (:math:`-0.040`), covariate ASCM
(:math:`-0.061`) and the residualized variant (:math:`-0.055`) -- is reproduced
value-for-value, with pre-fit :math:`L_2` imbalance falling monotonically from
:math:`0.083` to :math:`0.054`. The paper's Section-7 thesis (near-nominal
coverage and bias reduction across calibrated DGPs) is reproduced as a Path-B
simulation. The full ladder is reproduced through the public API -- pinned
in ``mlsynth/tests/test_vanillasc_ascm.py::test_augsynth_kansas_ladder_public_api``
-- not just at the engine level. See the dedicated page
:doc:`replications/ascm_kansas`; durable cases ``ascm_kansas`` (cross-validation
vs augsynth) and ``augsynth_calibrated`` (Path B), locked in
``mlsynth/tests/test_bilevel_ridge.py``.

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
the table below. The ATT prediction interval excludes zero in every case,
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
``malo`` returns a well-identified :math:`\mathbf{V}` (``v_agreement`` :math:`\approx 0`)
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
backend changes :math:`\widehat{\mathbf{w}}` (and hence the centre and width of the band)
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

On the canonical datasets the LTO test reproduces Lei-Sudijono's (2025) Table 1:
it can change the conclusion where the placebo grid is too coarse (German: exact
placebo 0.059 does not reject, LTO 0.042 does), is not mechanically smaller
(Basque: LTO 0.67 > placebo 0.41), and nearly coincides with the placebo when
both reject (Prop 99: 0.024 vs 0.026). The helper constants match the paper
exactly (``c(39, 0.05) = 0.002``, ``c(17, 0.05) = 0.0125``). See
:doc:`replications/vanillasc` for the full Table-1 relations and discussion.
Because the cost is :math:`O(N_0^2)` engine fits, run the covariate-matched
(``mscmt``) version on the smaller studies or cap pairs with ``lto_max_pairs``;
the 38-donor Prop 99 ``outcome-only`` LTO runs in under two minutes.

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
