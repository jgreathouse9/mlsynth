EASC: Exposure-Augmented Synthetic Control
==========================================

.. currentmodule:: mlsynth

When to Use This Estimator
--------------------------

Synthetic control trusts pre-treatment fit. If a weighted average of untreated
units tracks the treated unit closely for many periods, the argument goes, the
weights must also reproduce the treated unit's exposure to whatever drives the
outcome, so the weighted average stays a valid counterfactual afterwards.

Spruk (2026) identifies a case where that argument fails in a predictable way.
Suppose an observed common shock -- a commodity price, an exchange rate,
demand in a partner market -- affects every unit, but with different
intensity. Before treatment the shock moves with the general trend: oil prices
rose through the 2000s, and so did incomes in most emerging economies. A
synthetic unit can then fit the treated unit's pre-treatment path almost
perfectly while being far less exposed to the shock, because higher exposure
and faster underlying growth look the same before treatment. After treatment
the shock breaks away from the trend, the two paths diverge, and the estimated
treatment effect absorbs the difference. Nothing in the pre-treatment fit warns
of this. The paper calls such a shock a *silent factor*.

EASC does three things about it. It diagnoses the problem without looking at
the treated unit's post-treatment outcomes: it measures how far each
post-treatment period lies outside the range of the pre-treatment data, how
much of that comes from the observed shock, and how unequal the synthetic
unit's exposure is to the treated unit's. It corrects the exposure imbalance
with a regression across donors of outcome changes on an exposure index
measured before treatment. And it reports intervals that allow for the bias
pre-treatment fit cannot rule out, with a breakdown value saying how much of
that bias a conclusion survives.

Use it when one unit is treated, an observed common shock moved with the trend
before treatment and departed from it afterwards, and the treated unit's
exposure to that shock can be measured and differs from the donors'. A
retailer evaluating a store redesign in a region whose sales depend on tourism
faces it when the exchange rate moves after the launch; a state evaluating a
tax change faces it when the price of the commodity it produces collapses. The
paper's application is Iran's 2012 sanctions, which coincided with the 2014
oil-price collapse in a country more dependent on oil than any of its natural
comparators.

The diagnostics cost nothing and apply to any synthetic control whose treated
unit is unusually exposed to a measurable shock. The correction is a narrower
tool. When no silent factor is present it adds noise without
removing bias (the paper's simulations put its error at 1.5 to 1.9 times
synthetic control's), so apply it after the diagnostic has flagged the problem.

Notation
--------

Let :math:`j = 1` denote the treated unit and :math:`\mathcal{N}_0` the donor
pool of :math:`N_0` units. Periods are :math:`t \in \mathcal{T} \coloneqq
\{1, \dots, T\}`; treatment takes effect after period :math:`T_0`, with
pre-period :math:`\mathcal{T}_1` and post-period :math:`\mathcal{T}_2`. Weights
:math:`\mathbf{w} \in \Delta^{N_0}` produce the gap
:math:`\tau_t(\mathbf{w}) \coloneqq y_{1t} - \sum_{j\in\mathcal{N}_0} w_j y_{jt}`.

The observed common shock is :math:`x_t` (in the application, the log price of
Brent crude). Each unit has an exposure index :math:`s_j`, measured before
treatment (the 1996-2011 mean of oil rents as a share of GDP), and the
exposure imbalance of weights :math:`\mathbf{w}` is

.. math::

   \operatorname{EI}(\mathbf{w}) \coloneqq s_1 - \sum_{j\in\mathcal{N}_0} w_j s_j .

Notation bridge: Spruk writes the latent factors as :math:`\lambda_t` with
loadings :math:`\mu_j`; following the library's canon they are
:math:`\mathbf{f}_t` and :math:`\boldsymbol{\lambda}_j` here.

Model and Assumptions
---------------------

Assumption 1 (factor model with an observed common shock).
  (a) :math:`y_{jt}^N = \delta_t + \boldsymbol{\lambda}_j^\top \mathbf{f}_t + \theta_j x_t + u_{jt}`,
      where :math:`\mathbf{f}_t` are latent factors whose first element is 1
      (so :math:`\boldsymbol{\lambda}_j` contains a unit fixed effect) and
      :math:`\theta_j` is unit :math:`j`'s exposure to the shock;
  (b) :math:`u_{jt}` has mean zero given the factors, the loadings and treatment
      assignment.

Remark. The shock could be treated as one more latent factor. Singling it out
pays off because it is measured: its behaviour after treatment can be
inspected, and each unit's exposure to it can often be measured too.

Assumption 2 (exposure index).
  :math:`\theta_j = \gamma s_j` for a known, pre-determined index :math:`s_j`
  and an unknown :math:`\gamma`.

Remark. The index must be measured before treatment and must not respond to
it. Oil rents relative to GDP measure exposure to oil prices; trade shares
measure exposure to partner demand; sectoral employment shares measure
exposure to industry shocks. The linearity matters: with a concave exposure and
the treated unit outside the donors' range, the correction overshoots.

Assumption 3 (spanning controls).
  Across donors, the composite loadings on the non-constant latent factors are
  linear in the donors' own pre-treatment growth over a few sub-periods,
  :math:`c_j`, once exposure is held fixed, and donors' loading deviations and
  noise are mean-independent of the other donors' :math:`(s_k, c_k)`.

Remark. Donors' growth before treatment reflects their latent loadings, so
conditioning on it lets the regression separate exposure from underlying
growth. The assumption holds exactly when the shock is exactly in the span of
the latent factors before treatment, and approximately when it is close to it,
which is the silent-factor case.

Assumption 4 (for inference).
  The noise in a phase-average estimate is normal with standard deviation
  :math:`\sigma`, and the bias is at most :math:`B` in absolute value.

Remark. The bias bound is indexed by a sensitivity parameter :math:`M`, below.
Since the systematic part of the pre-treatment misfit is not identified, no
single :math:`M` delivers an unconditional 95 percent interval; the breakdown
value is the summary statistic.

The Diagnostics
---------------

Stack :math:`\mathbf{g}_t \coloneqq (\mathbf{f}_t^\top, x_t)^\top` and let
:math:`\mathbf{S}_0` be its pre-treatment second-moment matrix. The leverage of
period :math:`t`,

.. math::

   \kappa_t \coloneqq \bigl(\mathbf{g}_t^\top \mathbf{S}_0^{+} \mathbf{g}_t\bigr)^{1/2},

measures how far period :math:`t` lies from the pre-treatment data. Over the
pre-period :math:`\kappa_t^2` averages the number of factors; far larger values
mean the factors have left their pre-treatment range. Proposition 1 of the
paper bounds the post-treatment bias of any weighting estimator in period
:math:`t` by :math:`\kappa_t` times the systematic part of the pre-treatment
misfit, so leverage converts "the post-period does not look like the
pre-period" into a number.

Regress the shock on the latent factors over the pre-period and call the
residual :math:`r_t`. The partial leverage
:math:`\ell_t \coloneqq r_t / (T_0^{-1}\sum_{t\in\mathcal{T}_1} r_t^2)^{1/2}`
is that residual measured in units of its pre-period size, and
:math:`\kappa_t^2 = (\kappa_t^{f})^2 + \ell_t^2` exactly, where
:math:`\kappa_t^{f}` is the leverage of the latent factors alone
(Corollary 1). A pre-treatment :math:`R^2` of the shock on the factors near one
together with post-treatment :math:`|\ell_t|` well above 2 is the signature of
a silent factor. The latent factors are estimated from the donors, who are
untreated throughout, after removing each period's cross-sectional projection
on the exposure index, so none of these quantities uses the treated unit's
post-treatment outcomes.

The exposure geometry completes the picture. Simplex weights can balance
exposure exactly only when :math:`s_1` lies inside the donors' exposure range;
otherwise every simplex-weighted estimator -- synthetic control, penalized
synthetic control, synthetic difference-in-differences -- carries an imbalance
of at least :math:`d_{\min}`, the distance from :math:`s_1` to that range
(Proposition 3). The exposure-fit frontier :math:`R(d)`, the smallest
pre-treatment RMSPE attainable with imbalance at most :math:`d`, is
non-increasing and convex, and shows what balance would cost in fit.

The Estimators
--------------

For each period, regress the donors' outcome changes since a base period
:math:`b \le T_0` (the last pre-treatment period by default) across donors on
the exposure index and the growth controls,

.. math::

   y_{jt} - y_{jb} \;\text{ on }\; (1,\ s_j,\ \mathbf{c}_j), \qquad j \in \mathcal{N}_0,

and call the coefficient on :math:`s_j` :math:`\widehat{\phi}_t`. The
exposure-augmented estimate is

.. math::

   \tau_t^{\text{EA}}(\mathbf{w}) \coloneqq y_{1t} - \sum_{j\in\mathcal{N}_0} w_j y_{jt}
   - \widehat{\phi}_t \operatorname{EI}(\mathbf{w}).

Under Assumptions 1-3 the regression coefficient identifies the part of the
shock that has left the factors' span (Proposition 4), so the correction
targets the component that weighting cannot handle. The correction needs donors
whose exposure brackets the treated unit's, but those donors need not receive
weight; outside that range the regression extrapolates, and EASC warns.

``EASC`` returns four fits on the same panel, and ``method`` chooses which one
heads the standardized results:

* ``"sc"`` -- synthetic control, always returned as the baseline;
* ``"easc"`` (default) -- the correction at the synthetic-control weights, the
  paper's EASC;
* ``"ebsc_min"`` -- the best-fitting simplex weights with the smallest
  attainable imbalance, uncorrected;
* ``"easc_tuned"`` -- the correction at the frontier point that minimises the
  paper's heuristic mean-squared-error criterion (EASC*).

The synthetic-control weights come from mlsynth's active-set solver. The
frontier adds the constraint :math:`|\operatorname{EI}(\mathbf{w})| \le d`,
which that solver does not take, and is solved with cvxpy and CLARABEL; with
more donors than pre-periods its weights are not unique, and an interior-point
solver returns the member the paper's code does.

Inference and Diagnostics
-------------------------

Inference concerns averages of the effect over post-treatment phases, set with
``phases``; the whole post-period is always reported as ``"post"``. The bias of
a phase average is bounded by

.. math::

   B(M) = M\,\kappa\,\operatorname{RMSPE}(\widehat{\mathbf{w}}), \qquad M \ge 0,

where :math:`M` is the share of the pre-treatment RMSPE treated as systematic
and aligned with the post-treatment factor direction, and :math:`\kappa` is the
phase's leverage (for EASC, the latent leverage combined with the base-period
partial leverage). The interval
:math:`\widehat{\tau} \pm \sigma\,\mathrm{cv}_\alpha(B/\sigma)`, with
:math:`\mathrm{cv}_\alpha(\rho)` the :math:`1-\alpha` quantile of :math:`|Z+\rho|`,
covers at :math:`1-\alpha` whenever the bias is within :math:`B` (Armstrong and
Kolesár). It is shorter than adding the bias bound to a conventional interval.
The breakdown value :math:`M^\ast` is the largest :math:`M` at which the
interval still excludes zero: what share of the pre-treatment misfit would have
to be systematic to overturn the conclusion.

The noise scale :math:`\sigma` comes from in-space placebos: each donor in turn
is treated as the treated unit, its phase average is divided by its own
pre-treatment RMSPE, and the standard deviation of those ratios is rescaled to
the treated unit's RMSPE; a robust version uses the median absolute deviation.
For EASC the correction's sampling error, from a delete-one-donor jackknife, is
added in quadrature. The paper's text and its code differ on that last term;
``correction_se="period_mean"`` (the default) follows the code and the paper's
tables, ``correction_se="phase"`` the text, and the second is never larger.
The conventional placebo p-value, whose smallest attainable value is
:math:`1/(N_0 + 1)`, is reported alongside.

The standardized ``res.inference`` carries the EASC interval for the whole
post-period at ``headline_m`` (0 by default); ``res.bias_aware`` holds every
estimator, phase and value of :math:`M`. ``res.diagnostics`` holds
:math:`\kappa_t`, :math:`\kappa_t^{f}`, :math:`\ell_t`, the pre-treatment
:math:`R^2`, :math:`d_{\min}`, :math:`\widehat{\phi}_t` with its standard error,
the regression's hat values (a donor near 1 drives the correction) and each
phase's leverage; ``res.frontier`` holds :math:`R(d)`.

Example
-------

The paper's application, on its exposure-feasible pool of 21 donors.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from mlsynth import EASC

   url = ("https://raw.githubusercontent.com/jgreathouse9/mlsynth/refs/heads/main/"
          "basedata/iran_sanctions_oil.csv")
   df = pd.read_csv(url)
   donors = ["KAZ", "NGA", "ECU", "MEX", "COL", "JOR", "ROU", "BGR", "VNM", "ZAF",
             "PHL", "THA", "KEN", "BRA", "CHL", "PER", "CRI", "DOM", "PAN", "AZE", "AGO"]
   df = df[df.iso.isin(["IRN", *donors])].copy()
   # The Dominican Republic and Panama report no oil rents; the paper sets them to zero.
   df.loc[df.iso.isin(["DOM", "PAN"]), "oilrent"] = 0.0
   df["log_brent"] = np.log(df["brent"])

   res = EASC({
       "df": df, "outcome": "lgdppc", "treat": "sanctions", "unitid": "iso", "time": "year",
       "exposure": "oilrent", "exposure_window": (1996, 2011), "shock": "log_brent",
       "phases": {"shock": (2012, 2015), "relief": (2016, 2017), "siege": (2018, 2024)},
       "display_graphs": False,
   }).fit()

   dg = res.diagnostics
   print(f"R2 {dg.shock_r2:.3f}, max post |ell| {dg.max_partial_leverage_post:.2f}, "
         f"EI(SC) {res.fits['sc'].exposure_imbalance:.2f}, d_min {dg.d_min:.2f}")
   years = np.asarray(res.inputs.time_labels)
   for name in ("sc", "easc", "ebsc_min"):
       gap = res.fits[name].gap
       print(name, [round(float(gap[(years >= a) & (years <= b)].mean()), 3)
                    for a, b in [(2012, 2015), (2016, 2017), (2018, 2024)]])
   for est in ("SC", "EASC"):
       row = res.bias_aware.row(est, "shock")
       print(est, "2012-15 interval at M=0:", np.round(row.intervals[0.0], 2),
             "breakdown M*:", round(row.breakdown, 2))

Oil is a silent factor here: its pre-treatment :math:`R^2` on the latent factors
is 0.915 and its post-treatment partial leverage reaches 8.22. Synthetic Iran
fits the pre-treatment path with an RMSPE of 0.014 and is 15.5 points of GDP
less dependent on oil. Synthetic control finds losses of 0.127, 0.087 and 0.089
log points in 2012-15, 2016-17 and 2018-24. EASC finds -0.122, -0.011 and
+0.109, and EBSC-min (feasible here because Azerbaijan and Angola place Iran
inside the donors' exposure range) -0.109, -0.022 and +0.054: once oil exposure
is accounted for, the persistent loss disappears. The 2012-15 loss survives
both estimators, with breakdown values of 0.27 (SC) and 0.21 (EASC); every
later phase has a breakdown value of zero. The defensible reading, and the
paper's, is a short-run loss of about 12 percent and a long-run effect these
comparators cannot identify.

Verification
------------

Every quantity EASC computes -- weights, gaps, leverage, partial leverage,
:math:`\widehat{\phi}_t` and its jackknife, the frontier, the placebo noise
scales, intervals and breakdown values -- was cross-validated against the
author's replication code, run live, to between :math:`10^{-17}` and
:math:`5\times10^{-6}`, and 270 of 270 printed cells of the paper's Tables 4,
5 and 7 reproduce. The paper's Monte Carlo was reproduced on its own seeds,
with 496 of 496 printed cells for SC, EBSC-min, EASC and EASC* matching. The
study is
`benchmarks/studies/easc_replicate <https://github.com/jgreathouse9/mlsynth/tree/main/benchmarks/studies/easc_replicate>`_,
and ``mlsynth/tests/test_easc.py`` pins Tables 4 and 5 against the estimator.

Core API
--------

.. autoclass:: mlsynth.estimators.easc.EASC
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
~~~~~~~~~~~~~

.. autoclass:: mlsynth.utils.easc_helpers.config.EASCConfig
   :members:
   :undoc-members:
   :show-inheritance:

Results and Inputs
~~~~~~~~~~~~~~~~~~

.. automodule:: mlsynth.utils.easc_helpers.structures
   :members:
   :undoc-members:
   :show-inheritance:

Engine
~~~~~~

.. automodule:: mlsynth.utils.easc_helpers.setup
   :members:

.. automodule:: mlsynth.utils.easc_helpers.pipeline
   :members:

.. automodule:: mlsynth.utils.easc_helpers.diagnostics
   :members:

.. automodule:: mlsynth.utils.easc_helpers.correction
   :members:

.. automodule:: mlsynth.utils.easc_helpers.frontier
   :members:

.. automodule:: mlsynth.utils.easc_helpers.inference
   :members:
