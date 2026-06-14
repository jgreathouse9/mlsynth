Parallel-Trends Supergeo Design (PANGEO)
=========================================

.. currentmodule:: mlsynth

When to Use This Estimator
--------------------------

PANGEO is a tool for designing a geo experiment — deciding, *before*
you run it, which geographic markets to treat and which to hold out, so
that the after-the-fact comparison is as clean as possible. Use it when:

- you can assign treatment at the geo level (turn an ad campaign,
  price, or feature on in some markets and not others);
- you have a panel of pre-period history (weekly/monthly sales,
  conversions, GMV…) for every candidate geo;
- the number of geos is modest (tens, not thousands), as is typical
  with DMAs/regions; and
- you want the eventual treatment-effect estimate to be precise —
  i.e. you want treated and control markets that already move together.

A worked geo-experiment. A brand wants to measure the incremental
sales from a new TV/CTV campaign. Ads are bought at the DMA level, so the
experimental units are ~50–210 large, heterogeneous markets — not
exchangeable shoppers. The plan: run the campaign in a *treatment* set of
DMAs, withhold it in a *control* set, and read the post-launch sales gap
as the lift. The whole experiment lives or dies on the split: if the
treated DMAs were already trending differently from the controls, the
post-launch gap mixes the campaign effect with that pre-existing
divergence. PANGEO reads the DMAs' pre-period sales panel and chooses the
treatment/control split (bundling DMAs into balanced *supergeos*) so the
two sides' sales trajectories run parallel beforehand — turning the
post-period gap into a clean read on the campaign.

PANGEO is two stages:

#. Design (pre-period data only). A set-partitioning mixed-integer
   program groups each arm's geos into composite *supergeos*, forms
   balanced pairs with no geo trimmed, and selects the partition that
   maximises pre-period parallelism. A power analysis reports the
   minimum detectable effect (MDE) implied by the chosen supergeo size
   :math:`Q`.

#. Evaluation (after the experiment). The *same* design is scored
   against the realised outcomes with the Augmented
   Difference-in-Differences estimator of Li & Van den Bulte (2022),
   giving the ATT, percent ATT, and CIs at the arm and program levels.

The two stages share one quantity: both the design objective and the
standard error of the realised effect are governed by the variance of the
supergeo *gap* residual. Minimising non-parallelism simultaneously
minimises the MDE and tightens the CI — optimising parallelism is
optimising inferential precision.

This is a principled deviation from Google's Supergeo Design. Chen et
al. (2023) and OSD (Shaw 2025) match supergeos on a scalar summary —
the summed baseline response, or a few covariate totals — which collapses
the time dimension. PANGEO matches on the full pre-period trajectory.
That difference is not cosmetic: the downstream analysis is a
difference-in-differences, which *differences trajectories over time*, so
two markets with identical totals but different seasonal shapes are not
interchangeable for it even though scalar matching scores them as a
perfect match. In trending, seasonal data — which is essentially all
geo-marketing data — matching on shape rather than on a single number is
what makes the post-period comparison valid. (The simulation at the end
of this page quantifies the gap: when geos share a baseline mean but
differ in shape, PANGEO recovers the effect ~30× more precisely than a
scalar match.)

When *not* to use it. If the assignment is already fixed (an
observational study, or a campaign that already ran in specific markets),
there is no design to choose — use an estimation-stage method
(:doc:`tssc`, :doc:`sbc`, :doc:`fdid`). If you have *hundreds* of geos the
exact MIP may be intractable (see *When PANGEO Fails or Stalls* below) —
the scalable OSD relaxation is the better fit there. And if the outcome is
plausibly stationary with no trend or seasonality, scalar matching is
already adequate and simpler.

What is a supergeo?
^^^^^^^^^^^^^^^^^^^

Geo experiments differ from ordinary A/B tests in one decisive way: the
experimental units are a *small* number of *large, heterogeneous* aggregates
--- markets, regions, DMAs --- rather than many exchangeable individuals.
Randomising treatment across a handful of dissimilar markets routinely
produces treatment and control groups with very different baseline
characteristics, and the resulting post-randomisation bias does not average
away over the single assignment a practitioner actually runs (Abadie & Zhao
2026). Classic matched-pair designs help, but with heterogeneous geos there
may be *no* good one-to-one match for a given market.

A supergeo resolves this by relaxing the unit of matching. Rather than
insisting that single geos match, geos are pooled into composite aggregates:
a supergeo is simply a bundle of geos treated as one unit, with outcome equal
to their (population-weighted) mean. Composite units can be made comparable
even when their constituents are not --- a small, noisy market combined with
a complementary one can, in aggregate, track another composite closely. The
design then pairs *supergeos*, randomises treatment within each pair, and ---
unlike trimming-based approaches --- assigns every geo to some supergeo,
so the experiment spans the entire market with nothing discarded (Chen,
Doudchenko, Jiang, Stein & Ying 2023).

PANGEO keeps this structure but changes *what* the supergeos are matched on.
Supergeo Design and OSD match on a scalar summary (the summed response, or a
few covariate totals), which collapses the time dimension; PANGEO matches on
the full pre-treatment trajectory, choosing pairs whose aggregate paths
run as parallel as possible. The reason is that the downstream
difference-in-differences analysis differences trajectories: two markets with
identical totals but different seasonal shapes are not interchangeable for
it, even though scalar matching treats them as equivalent.

Assumptions, and When PANGEO Fails or Stalls
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PANGEO can make a good experiment *likely*, but it cannot manufacture one
that the data do not support. Three assumptions underpin it, and each
points at a way the method can fail or stall.

1. Parallel trends (the crux). The design maximises parallelism in the
pre-period; the validity of the Stage-2 effect estimate rests on that
parallelism persisting into the post-period absent treatment — i.e.
the treatment and control supergeos *would have continued to move
together* had the campaign never launched. This is exactly the
difference-in-differences parallel-trends assumption, and it is the
assumption PANGEO is organised around. The crucial honesty: PANGEO
*optimises* pre-period parallelism (making the assumption as plausible as
the data allow) but it cannot guarantee the assumption holds
out-of-sample. If a shock hits the treated markets, a competitor reacts
only there, or the pre-period co-movement was coincidental, the
post-period gap diverges on its own and the ATT is biased — the same
Achilles' heel as any DiD. *Diagnostics that flag the risk:* the achieved
parallelism :math:`R^2` (low values mean no balanced design exists — see
below), the reported MDE, and a placebo / blank-window check on the
held-out pre-period.

2. A linear factor structure for the no-treatment outcomes
(Eq. :eq:`factor`). The gap decomposition that makes "match on trajectory"
equivalent to "balance the factor loadings" relies on this model. It is
the standard synthetic-control / interactive-fixed-effects assumption and
is mild for sales-like panels, but a wildly non-factor outcome (e.g. one
driven by an idiosyncratic, unit-specific regime change) is not balanceable
by any partition.

3. A modest, designable geo pool. Each arm needs enough geos to form
at least one supergeo pair, and the geos must be heterogeneous-but-
matchable.

The concrete failure / stall modes:

- Parallel trends breaks post-launch — the dominant risk, above. No
  design fixes it; only the diagnostics warn of it.
- No matchable structure. If the geos are so heterogeneous that *no*
  partition achieves high parallelism, PANGEO still returns the best
  feasible design, but the parallelism :math:`R^2` stays low and the MDE
  blows up — a signal that a geo experiment here is underpowered and the
  read will be noisy regardless of split.
- The MIP stalls. Set partitioning is NP-hard. With many geos and a
  large supergeo size :math:`Q`, the exact mixed-integer program can be
  slow or intractable. Mitigations: cap :math:`Q` (smaller supergeos),
  use the automatic :math:`Q` selection, raise ``min_pairs``, or — for
  hundreds of geos — fall back to the scalable OSD relaxation. The
  examples on this page use a handful of geos, so the solve is instant.
- Too few geos / arms. A pool that cannot form a balanced pair (e.g.
  two wildly different markets) has no good design to find.

In short: PANGEO improves the *plausibility* of parallel trends by
construction and *quantifies the residual risk* (parallelism
:math:`R^2`, MDE), but it inherits DiD's identifying assumption rather
than removing it. Treat a low parallelism :math:`R^2` or a large MDE as
the design telling you the experiment is fragile.

Setup and notation
------------------

Let :math:`Y_{it}` denote the outcome of geo :math:`i\in\{1,\dots,N\}` in
period :math:`t\in\{1,\dots,T\}`. The first :math:`T_0` periods are the
pre-treatment (design) window and the remaining
:math:`T_{\mathrm{post}} = T - T_0` periods are the experimental window.
A single categorical column assigns each geo to an arm; arms occupy
disjoint geo pools :math:`\mathcal N_a` and are designed independently, so
the exposition below fixes one arm and drops the arm subscript.

Throughout we maintain the linear factor model used by both the synthetic-
control and DiD literatures (Abadie, Diamond & Hainmueller 2010; Li &
Van den Bulte 2022) for the no-treatment potential outcome,

.. math::
   :label: factor

   Y_{it}^{N} = \delta_t + \theta_t^{\top} Z_i + \lambda_t^{\top}\mu_i
                + \varepsilon_{it},

where :math:`\delta_t` is a common time effect, :math:`Z_i` are observed
covariates with time-varying loadings :math:`\theta_t`, :math:`\mu_i` are
unobserved factor loadings with factors :math:`\lambda_t`, and
:math:`\varepsilon_{it}` is mean-zero idiosyncratic noise.

A supergeo is a set :math:`S` of same-arm geos with aggregate
trajectory

.. math::

   \bar Y_{S,t} = \frac{\sum_{i\in S}\omega_i\,Y_{it}}{\sum_{i\in S}\omega_i},

where :math:`\omega_i>0` are aggregation weights (the ``weight_col``
population, or :math:`\omega_i\equiv 1`). A pair
:math:`p=(A_p,B_p)` consists of two disjoint supergeos with
:math:`|A_p|,|B_p|\le Q`; :math:`A_p` is the treatment half and
:math:`B_p` the control half. Its gap is

.. math::
   :label: gap

   g_{p,t} = \bar Y_{A_p,t} - \bar Y_{B_p,t}.

Under :eq:`factor` the common time effect cancels and

.. math::
   :label: gapdecomp

   g_{p,t} = \lambda_t^{\top}\big(\bar\mu_{A_p}-\bar\mu_{B_p}\big)
           + \theta_t^{\top}\big(\bar Z_{A_p}-\bar Z_{B_p}\big)
           + \big(\bar\varepsilon_{A_p,t}-\bar\varepsilon_{B_p,t}\big),

with :math:`\bar\mu_S, \bar Z_S` the weighted means over :math:`S`. The
pair exhibits parallel trends precisely when the loadings are balanced,
:math:`\bar\mu_{A_p}=\bar\mu_{B_p}` (and :math:`\bar Z_{A_p}=\bar Z_{B_p}`);
the gap is then constant in expectation and a difference-in-differences
comparison within the pair is unbiased.

Stage 1 --- the supergeo design
-------------------------------

The parallelism objective
^^^^^^^^^^^^^^^^^^^^^^^^^^

The pre-treatment window is split into an estimation window
:math:`\mathcal E` (the first :math:`\lfloor \kappa T_0\rfloor` periods,
:math:`\kappa=` ``frac_E``, default :math:`0.7`) and a held-out blank
window :math:`\mathcal B=\{1,\dots,T_0\}\setminus\mathcal E`. A pair is
scored by the variance of its *level-removed* gap over the estimation
window,

.. math::
   :label: score

   c(p) = \sum_{t\in\mathcal E}\big(g_{p,t}-\bar g_p\big)^2,
   \qquad
   \bar g_p = \frac{1}{|\mathcal E|}\sum_{t\in\mathcal E} g_{p,t},

which is exactly the pre-period residual sum of squares of a
difference-in-differences fit (cf.
:func:`mlsynth.utils.selector_helpers._did_from_mean`). Taking expectations
under :eq:`gapdecomp` with balanced covariates,

.. math::

   \mathbb E\,c(p)
     = \big(\bar\mu_{A_p}-\bar\mu_{B_p}\big)^{\top}
       \Big[\textstyle\sum_{t\in\mathcal E}(\lambda_t-\bar\lambda)
            (\lambda_t-\bar\lambda)^{\top}\Big]
       \big(\bar\mu_{A_p}-\bar\mu_{B_p}\big)
     \;+\; \mathbb E\!\sum_{t\in\mathcal E}
       \big(\bar\varepsilon_{A_p,t}-\bar\varepsilon_{B_p,t}-\overline{\cdot}\big)^2 .

The first term is a positive-definite quadratic form in the loading
imbalance, so minimising :eq:`score` drives
:math:`\bar\mu_{A_p}\to\bar\mu_{B_p}` --- it balances the *unobserved*
factor loadings, which is what parallel-trends DiD requires. The
time-constant component of the loading difference is absorbed by the level
shift :math:`\bar g_p` and never penalised: two supergeos may differ
arbitrarily in level yet match perfectly in *shape*. Scalar sum-matching,
by contrast, collapses the time dimension and is blind to shape.

The set-partitioning program
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let :math:`\mathcal F` be the family of admissible pairs: every subset
of the arm's geos of size :math:`2,\dots,2Q` that can be split into two
halves each of size :math:`\le Q`, each subset scored at its best such split
by :eq:`score`. Let :math:`M\in\{0,1\}^{N\times|\mathcal F|}` be the geo-by-
pair incidence matrix (:math:`M_{iG}=1` iff geo :math:`i\in G`) and
:math:`c_G` the score of pair :math:`G`. The design solves the
set-partitioning program

.. math::
   :label: mip

   \min_{x\in\{0,1\}^{|\mathcal F|}} \sum_{G\in\mathcal F} c_G\,x_G
   \quad\text{s.t.}\quad
   M x = \mathbf 1 \ \ (\text{exact cover}),\qquad
   \mathbf 1^{\top} x \ge \kappa_{\min}\ \ (\text{minimum pairs}),

solved with ``cvxpy`` and the HiGHS mixed-integer backend. The exact-cover
constraint :math:`Mx=\mathbf 1` assigns every geo to exactly one chosen
pair (no geo is trimmed). Because each :math:`c_G` is *precomputed* offline,
the objective is linear in :math:`x` --- the program is a mixed-integer
*linear* program regardless of the (possibly nonlinear) per-pair cost,
which is what keeps it tractable. Within each chosen pair the treatment and
control halves are the score-minimising split; which half is actually
treated is randomised in the field.

Per-pair objectives
^^^^^^^^^^^^^^^^^^^^

The ``objective`` argument selects the per-pair cost :math:`c_G`; all three
choices leave :eq:`mip` a linear program. Writing :math:`g_t` for the gap
of a candidate split and :math:`\bar g` for its estimation-window mean,

* ``"ss_res"`` (default) --- the absolute residual sum of squares
  :math:`\sum_t (g_t-\bar g)^2`. Scale-dependent, so high-amplitude pairs
  weigh more and the design prioritises making large markets parallel.
* ``"r2"`` --- the scale-free criterion
  :math:`1-R^2 = \sum_t(g_t-\bar g)^2 / \sum_t(\bar Y_{A,t}-\overline{\bar Y_A})^2`,
  so every pair counts equally (FDID's :math:`R^2` criterion, optimised
  *exactly* by the program rather than greedily).
* ``"weighted"`` --- a recency-weighted residual SS
  :math:`\sum_t w_t (g_t-\bar g_w)^2`, the level removed at the weighted mean
  :math:`\bar g_w`, with weights :math:`w_t=\rho_{\mathrm{dec}}^{\,T_0-1-t}`
  (``recency_decay``), up-weighting the recent pre-period closest to the
  experiment.

The per-pair ``gap_variance`` and ``parallelism_r2`` reported on the result
are always the unweighted quantities of :eq:`score`, so designs from
different objectives are comparable on a common yardstick.

Supergeo size :math:`Q` and automatic selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Setting ``max_supergeo_size`` :math:`=Q=1` recovers the classic
matched-pairs design; :math:`Q>1` permits composite supergeos when no single
geo matches another well, without trimming. :math:`Q` is a granularity knob
with an interior optimum: too small and no parallel matches exist
(singleton geos are too noisy); too large and the arm yields few, coarse
pairs. The program-level MDE is *not* monotone in :math:`Q` and is *not*
tracked by the parallelism :math:`R^2` (which is scale-free and rises with
:math:`Q`); only the absolute residual variance that drives power matters.

Consequently, if ``max_supergeo_size`` is left unset, PANGEO selects
:math:`Q` automatically: it solves :eq:`mip` for every feasible
:math:`Q\in\{1,\dots,\min(\lceil N/2\rceil, 6)\}` and returns the design
with the smallest mean program MDE. The full sweep --- each :math:`Q`'s
program-pair count, mean program MDE, and the :math:`2/2^{P}`
randomisation-inference p-value floor for :math:`P` pairs --- is recorded in
``results.metadata["q_sweep"]`` and the choice in
``results.metadata["q_selected"]``, so the decision is auditable and may be
overridden with an explicit :math:`Q`.

Balancing baseline covariates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Parallelism is level-blind: by :eq:`score` the level shift
:math:`\bar g_p` absorbs any time-constant gap, so a baseline characteristic
(population, income) that merely shifts a market's level is differenced out
and never enters the trajectory score. This is correct for parallel-trends
DiD but says nothing about balance on such characteristics --- the role of
OSD's scalar covariate matching. PANGEO restores it with a standardised
mean-difference penalty appended to :eq:`score`,

.. math::
   :label: covpen

   c(p) \;\longmapsto\; c(p)
     \;+\; \sum_{m} \omega^{\mathrm{cov}}_m
       \Big(\frac{\bar c_{A_p,m}-\bar c_{B_p,m}}{s_m}\Big)^2,

the weighted squared standardised mean difference (SMD) between the halves'
covariate means, where :math:`s_m` is the cross-geo standard deviation
(``standardize_covariates``, default ``True``) and :math:`\omega^{\mathrm{cov}}_m`
a per-covariate weight (``covariate_weights``, default :math:`1`). Because
:eq:`covpen` is precomputed it preserves linearity in :eq:`mip`. Larger
weights buy tighter covariate balance at the cost of some parallelism; the
achieved per-pair SMDs are reported in ``SupergeoPair.covariate_smd``. Pass
``covariates=[...]`` (baseline columns, each reduced to its per-geo mean) to
enable; with no covariates the design is unchanged. This is also the
Abadie & Zhao (2026, Thm. 1) prescription --- moving structure from the
unobserved :math:`\mu_i` into the observed :math:`Z_i` lowers the
estimator's bias --- and the Stage-2 device for restoring inferential
validity (below).

.. code-block:: python

   df = make_seasonal_sales_panel(units_per_arm=6, arms=("A", "B", "C"),
                                  T=104, seed=0, covariates=True)

   res = PANGEO({
       "df": df, "outcome": "sales", "arm": "arm",
       "unitid": "unit", "time": "time", "max_supergeo_size": 3,
       "covariates": ["population", "income"],
       "covariate_weights": {"population": 5.0, "income": 5.0},
   }).fit()

   for arm, design in res.arm_designs.items():
       for p in design.pairs:
           print(p.treatment, p.control, p.parallelism_r2, p.covariate_smd)

Power and the minimum detectable effect
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Because power and the design objective are governed by the same supergeo
gap residual, :meth:`mlsynth.PANGEO.fit` returns a power analysis
(``results.power``). For pair :math:`p` the per-period noise is estimated
honestly on the held-out blank window :math:`\mathcal B` (out of sample
with respect to the optimisation) as the residual of the *same counterfactual
model used at evaluation* (:eq:`adid`) --- fit on the estimation window
:math:`\mathcal E`, evaluated on :math:`\mathcal B`:

.. math::

   \hat\sigma_p^2
     = \frac{1}{|\mathcal B|-1}\sum_{t\in\mathcal B} \hat e_{p,t}^2 .

Using the evaluation model here (the augmented-DiD residual by default,
or the plain level-removed gap when ``att_augment=False``) rather than a
fixed recipe keeps the projected MDE and the realised standard error
(:eq:`adidvar`) coherent. The :math:`X`-period effect for the pair then has
variance :math:`\hat\sigma_p^2\,[f(X,\rho)+f(T_0,\rho)]`, where

.. math::

   f(n,\rho) = \frac{1}{n}\Big(1 + 2\sum_{k=1}^{n-1}(1-\tfrac{k}{n})\rho^{k}\Big)

is the variance-inflation factor of the mean of :math:`n` AR(1)-correlated
periods and :math:`\rho` is the pooled lag-1 autocorrelation of the blank
residuals. Serial correlation is decisive: weekly sales are highly
autocorrelated, so :math:`X` post weeks are worth far fewer than :math:`X`
independent observations and adding post periods yields sharply diminishing
returns --- the trap a naive i.i.d. power calculation falls into.

The program-level effect is the treated-size-weighted average of the pair
effects, with weights
:math:`w_p = (\sum_{i\in A_p}\omega_i)/\sum_{q}\sum_{i\in A_q}\omega_i`, and
(treating pairs as independent across the program)

.. math::

   \widehat{\operatorname{Var}}(\hat\tau_{\mathrm{prog}})
     = \sum_p w_p^2\,\hat\sigma_p^2\,\big[f(X,\rho)+f(T_0,\rho)\big],
   \qquad
   \mathrm{MDE}(X) = \big(z_{1-\alpha/2}+z_{1-\beta}\big)\,
       \sqrt{\widehat{\operatorname{Var}}(\hat\tau_{\mathrm{prog}})}.

The program level is the headline: small arms are individually
under-powered (with :math:`P` pairs a pure within-pair randomisation test
has a hard p-value floor of :math:`2/2^{P}`, so one needs :math:`P\ge 6` to
reach :math:`p<0.05`), whereas pooling across arms gives the program an
effective sample size equal to the total pair count and routinely detects
effects several points smaller than any one arm. Per-arm curves are stored
in ``results.power.arms``. The MDE is reported in outcome units and as a
percent of the treated baseline, by default at :math:`1-\beta=0.80` power
for horizons :math:`X=2,\dots,12`; ``power_target``, ``power_alpha`` and
``power_post_periods`` configure this and ``compute_power=False`` skips it.

.. code-block:: python

   res = PANGEO({
       "df": df, "outcome": "sales", "arm": "arm",
       "unitid": "unit", "time": "time", "max_supergeo_size": 3,
   }).fit()

   pw = res.power
   print(f"serial correlation rho = {pw.serial_correlation:.2f}")
   print(pw.summary())                       # MDE % by horizon: program + arms
   print(pw.program.mde_pct_by_horizon()[8]) # detectable % lift after 8 weeks
   print(pw.power_for_effect(effect_pct=5.0, post_periods=8))  # invert: power

Stage 2 --- evaluation by Augmented DiD
---------------------------------------

The estimator
^^^^^^^^^^^^^

Once the experiment has run, pass a ``post_col`` (a :math:`0/1` indicator of
post-treatment periods, as in LEXSCM). The design is rebuilt on the pre rows
alone --- so it is identical to the design-only result --- and
``results.effects`` carries the realised ATT at the arm and program
levels using the Augmented Difference-in-Differences estimator of Li &
Van den Bulte (2022).

Fix a level (an arm, or the program) and write :math:`y^{T}_t` for its
treated supergeo aggregate and :math:`y^{C}_t` for its control supergeo
aggregate, both treated-size-weighted across the level's pairs. The
counterfactual is the pre-period least-squares projection

.. math::
   :label: adid

   y^{T}_t = \delta_1 + \delta_2\,y^{C}_t + \gamma\,t + e_t,
   \qquad t = 1,\dots,T_0 .

This *augments* plain DiD in two ways: the control scale :math:`\delta_2` is
estimated rather than fixed at :math:`1`, and a linear time trend
:math:`\gamma t` is included (``att_augment`` and ``att_trend``, both default
``True``). With regressor :math:`x_t=(1,\,y^{C}_t,\,t)^{\top}` and OLS
estimate :math:`\hat\delta`, the per-period effect and the ATT are

.. math::

   \hat u_t = y^{T}_t - x_t^{\top}\hat\delta,
   \qquad
   \hat\Delta = \frac{1}{T_{\mathrm{post}}}
       \sum_{t=T_0+1}^{T} \hat u_t .

The percent ATT is taken relative to the post-period counterfactual
(cf. :func:`mlsynth.utils.resultutils.effects.calculate`), not the
pre-treatment baseline:

.. math::

   \hat\Delta_{\%} = 100\times\frac{\hat\Delta}{\bar y^{0}_{\mathrm{post}}},
   \qquad
   \bar y^{0}_{\mathrm{post}}
     = \frac{1}{T_{\mathrm{post}}}\sum_{t=T_0+1}^{T} x_t^{\top}\hat\delta
     = \frac{1}{T_{\mathrm{post}}}\sum_{t=T_0+1}^{T}\big(y^{T}_t-\hat u_t\big).

Inference
^^^^^^^^^

Li & Van den Bulte (2022, Prop. 3.1--3.3) show
:math:`\sqrt{T_{\mathrm{post}}}\,(\hat\Delta-\Delta)\xrightarrow{d}
N(0,\Sigma_1+\Sigma_2)`, where :math:`\Sigma_1` is the variance from
estimating :math:`\delta` and :math:`\Sigma_2` from averaging the
post-period errors. Their Web Appendix C.13 gives the
prediction-variance estimator

.. math::
   :label: adidvar

   \widehat{\operatorname{Var}}(\hat\Delta)
     = \hat\omega^2\Big[\,
         \bar x_{\mathrm{post}}^{\top}
         \Big(\textstyle\sum_{t=1}^{T_0} x_t x_t^{\top}\Big)^{-1}
         \bar x_{\mathrm{post}}
         \;+\; \frac{1}{T_{\mathrm{post}}}\Big],

with :math:`\bar x_{\mathrm{post}}=T_{\mathrm{post}}^{-1}\sum_{t>T_0}x_t`.
The first bracketed term is :math:`\Sigma_1` (it inflates automatically when
the post-period control drifts outside its pre-period range, pricing the
extrapolation uncertainty) and the second is :math:`\Sigma_2`. The residual
variance :math:`\hat\omega^2` is estimated over the long pre-period as a
Newey--West/Bartlett long-run variance with truncation lag
:math:`\lfloor T_0^{1/4}\rfloor` (Li & Van den Bulte's
:math:`O(T^{1/4})` rule); lag :math:`0` is the i.i.d. case
:math:`\hat\omega^2=\hat e^{\top}\hat e/(T_0-k)` for :math:`k` regressors.
The confidence interval is
:math:`\hat\Delta \pm z_{1-\alpha/2}\sqrt{\widehat{\operatorname{Var}}(\hat\Delta)}`
and the p-value is the two-sided normal test of :math:`\Delta=0`.

Why this estimator suits the supergeo gap
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Li & Van den Bulte's regularity conditions (Assumptions C2--C3) explicitly
admit trend and unit-root (integrated) common factors :math:`\lambda_t`
--- the regimes under which naive i.i.d. standard errors collapse. The
mechanism is the augmentation: regressing the treated aggregate on a
*scaled* control is a cointegrating regression, and a single
:math:`\delta_2` cancels a shared integrated factor in :eq:`gapdecomp`,
while :math:`\gamma t` absorbs deterministic drift. The validity condition
reduces to a single requirement --- that the regression residual
:math:`e_t` be (weakly dependent) stationary --- which the augmentation and
trend deliver. The design's parallelism is retained throughout; it minimises
the residual variance, which by :eq:`adidvar` directly tightens the standard
error --- and, because the power analysis reads the *same* held-out residual,
the planning MDE as well.

Plain DiD as an option. Setting ``att_augment=False`` (and optionally
``att_trend=False``) recovers Li & Van den Bulte's ordinary
difference-in-differences --- ``y^{T}_t - y^{C}_t = \delta_1 [+ \gamma t] +
e_t`` with the control coefficient fixed at one --- and the power analysis
follows suit, so the two stages stay coherent. A head-to-head Monte-Carlo
comparison (R\ :sup:`2` design + plain DiD versus the augmented defaults)
found augmented DiD both more precise (lower realised MDE) and
better-covering across the stationary, trend-plus-seasonal and
integrated-factor regimes, because plain DiD has no mechanism to absorb a
control-scale mismatch or a trend and leaves that structure in its residual.
The augmented estimator is therefore the default; plain DiD remains available
for settings where its textbook simplicity is preferred.

Validity envelope (smoke tests)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A Monte-Carlo study over the bundled simulator confirms that validity hinges
on residual stationarity, not on the interval recipe. The simulator can
place the unobserved factor on an i.i.d., AR(1) or random-walk process
(``factor``) and toggle the seasonal amplitude (``season_amp``) and per-geo
trend (``trend_sd``):

.. list-table::
   :header-rows: 1
   :widths: 50 25 25

   * - Gap structure
     - Program coverage (nominal 0.95)
     - Type-I (nominal 0.05)
   * - stationary i.i.d. factor (paper DGP)
     - 0.93
     - 0.07
   * - + linear trend + seasonality
     - 0.87
     - 0.13
   * - + integrated factor (random walk)
     - 0.60
     - 0.40

The point estimate is unbiased in every regime. On a stationary gap ---
matching Li & Van den Bulte's factor-model design --- the prediction-
variance interval is at its nominal rate; the augmentation and trend
regressor recover most of the coverage lost to a deterministic trend and
seasonality; and the adversarial random-walk-plus-seasonality gap, where two
integrated factors and amplitude-heterogeneous seasonality exceed what a
single :math:`\delta_2` can cointegrate, marks the honest assumption
boundary. In practice the fitted :math:`\hat\delta_2` (reported as
``AttEstimate.scale``) and the residual diagnose whether the assumption
holds; if a single scale cannot flatten the gap, add covariate or seasonal
regressors before trusting the interval.

Because the power analysis now uses the evaluation model's held-out
residual, the planning MDE is calibrated to the realised standard error:
on the stationary gap the projected MDE matches the realised value to within
roughly 7% (ratio :math:`\approx 0.93`), and on integrated gaps it is
conservative (over-states the MDE), the safe direction. These experiments
live in ``mlsynth/tests/test_pangeo.py`` (``TestADIDInference``).

.. code-block:: python

   df = make_seasonal_sales_panel(units_per_arm=6, arms=("A", "B", "C"),
                                  T=104, seed=0, n_post=8)
   res = PANGEO({
       "df": df, "outcome": "sales", "arm": "arm",
       "unitid": "unit", "time": "time", "post_col": "post_col",
       "max_supergeo_size": 3,
       "att_augment": True, "att_trend": True,   # Augmented DiD (defaults)
   }).fit()

   print(res.effects.summary())           # program + per-arm ATT, SE, CI, p
   pe = res.effects.program
   print(f"program ATT = {pe.att_pct:.1f}% "
         f"[{pe.ci_lower_pct:.1f}, {pe.ci_upper_pct:.1f}], "
         f"p={pe.p_value:.3f}, scale delta_2={pe.scale:.2f}")

Core API
--------

.. automodule:: mlsynth.estimators.pangeo
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.PANGEOConfig
   :members:
   :undoc-members:

Helper Modules
--------------

.. automodule:: mlsynth.utils.pangeo_helpers.parallelism
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.pangeo_helpers.mip
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.pangeo_helpers.setup
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.pangeo_helpers.pipeline
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.pangeo_helpers.power
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.pangeo_helpers.effects
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.pangeo_helpers.simulation
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.pangeo_helpers.structures
   :members:
   :undoc-members:

Example
-------

A seasonal, multi-arm sales panel (the bundled simulator), designed into
parallel supergeo pairs. With ``display_graphs=True`` PANGEO plots, per arm,
the observed treated supergeo aggregate against the augmented-DiD
counterfactual prediction (:func:`mlsynth.utils.pangeo_helpers.effects.adid_counterfactual`):
the in-sample pre-period fit when designing, and -- once ``post_col`` data
are supplied -- the counterfactual extended past the treatment date, so the
post-window gap is the estimated effect.

.. code-block:: python

   from mlsynth import PANGEO
   from mlsynth.utils.pangeo_helpers import make_seasonal_sales_panel

   # 3 arms (non-overlapping geos), 6 geos each, 156 weeks of history.
   df = make_seasonal_sales_panel(units_per_arm=6, arms=("A", "B", "C"),
                                  T=156, seed=0)

   res = PANGEO({
       "df": df,
       "outcome": "sales",
       "arm": "arm",                # single categorical arm column
       "unitid": "unit",
       "time": "time",
       "max_supergeo_size": 3,      # Q
   }).fit()

   for arm, design in res.arm_designs.items():
       print(f"Arm {arm}: {len(design.pairs)} pair(s), "
             f"parallel-trends R^2 = {design.mean_parallelism_r2:.3f}")
       for p in design.pairs:
           print(f"   T={p.treatment}  C={p.control}  R^2={p.parallelism_r2:.3f}")

   # res.assignment maps every geo -> 'treatment' / 'control'.

On the simulated data this returns designs with parallel-trends :math:`R^2`
around 0.90--0.98 --- roughly 10--35x more parallel than a random
treatment/control split of the same geos.

Simulation: Trajectory Matching vs. a Scalar Supergeo
-----------------------------------------------------

The supergeo design of Chen et al. (2023) matches (super)geos on a
scalar summary of baseline response (its variance term is
:math:`\sum_k (Z_{G_{k,+}} - Z_{G_{k,-}})^2`, a sum over scalar baseline
differences). PANGEO carries this into the panel setting: it matches
on the full pre-treatment *trajectory* (level-removed parallelism), so it
can separate geos that look identical on a scalar yet move differently
over time. The self-contained Monte Carlo below makes the gap concrete —
adapting the paper's RMSE comparison (supergeo vs. matched pairs) to a
panel where every geo has the same pre-period mean but a distinct
trajectory shape, a setting in which scalar matching is by construction
blind. Six geos keep the MIP instantaneous.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from mlsynth import PANGEO

   def make_panel(rng, T_pre=20, S=6, level=100.0, noise=0.6):
       """Six geos = three parallel pairs (up-trend, down-trend, cycle);
       each shape is demeaned over the pre-period so all pre-means match."""
       T = T_pre + S
       t = np.arange(T)
       up = (t - t.mean()) / t.std()
       cyc = np.sin(2 * np.pi * t / 5.0)
       shapes = [5 * up, 5 * up, -5 * up, -5 * up, 5 * cyc, 5 * cyc]
       cols = []
       for s in shapes:
           s = s - s[:T_pre].mean()              # equal pre-means => scalar-blind
           cols.append(level + s + rng.normal(0, noise, T))
       return np.column_stack(cols), T_pre

   def did(Y, T_pre, treated, control, tau):
       Yo = Y.copy()
       Yo[T_pre:, treated] += tau                # inject the true effect
       t_eff = Yo[T_pre:, treated].mean() - Yo[:T_pre, treated].mean()
       c_eff = Yo[T_pre:, control].mean() - Yo[:T_pre, control].mean()
       return t_eff - c_eff

   def pangeo_design(Y, T_pre):
       df = pd.DataFrame(
           {"geo": f"g{g}", "t": int(t), "y": Y[t, g], "arm": "A"}
           for g in range(6) for t in range(T_pre)
       )
       a = PANGEO({"df": df, "outcome": "y", "unitid": "geo", "time": "t",
                   "arm": "arm", "max_supergeo_size": 1,
                   "compute_power": False, "display_graphs": False}).fit().assignment
       T = [int(g[1:]) for g, v in a.items() if v == "treatment"]
       C = [int(g[1:]) for g, v in a.items() if v == "control"]
       return T, C

   def scalar_match(Y, T_pre, rng):              # Google-style: pair on scalar mean
       order = np.argsort(Y[:T_pre].mean(0))
       T, C = [], []
       for k in range(0, 6, 2):
           a, b = order[k], order[k + 1]
           if rng.random() < 0.5:
               T.append(a); C.append(b)
           else:
               T.append(b); C.append(a)
       return T, C

   tau, R = 4.0, 60
   rng = np.random.default_rng(1)
   errs = {"PANGEO (trajectory)": [], "scalar matched-pairs": []}
   for _ in range(R):
       Y, T_pre = make_panel(rng)
       Tp, Cp = pangeo_design(Y, T_pre)
       errs["PANGEO (trajectory)"].append(did(Y, T_pre, Tp, Cp, tau) - tau)
       Ts, Cs = scalar_match(Y, T_pre, rng)
       errs["scalar matched-pairs"].append(did(Y, T_pre, Ts, Cs, tau) - tau)

   for name, e in errs.items():
       e = np.array(e)
       print(f"{name:22s} RMSE = {np.sqrt((e ** 2).mean()):.2f}")

Because all geos share a pre-period mean, the scalar match pairs them
essentially at random, and the difference-in-differences estimate
inherits the up/down/cycle shape mismatch (RMSE ≈ 6 against a true effect
of 4). PANGEO reads the trajectory shape, recovers the three parallel
pairs, and estimates the effect about 30× more precisely (RMSE ≈ 0.2).
That is the supergeo idea carried into the panel world: match on *how the
series move*, not on a single number.

References
----------

Chen, A., Doudchenko, N., Jiang, S., Stein, C., & Ying, B. (2023).
"Supergeo Design: Generalized Matching for Geographic Experiments."
arXiv:2301.12044.

Shaw, C. (2025). "Optimized Supergeo Design: A Scalable Framework for
Geographic Marketing Experiments." arXiv:2506.20499.

Li, K. T. (2023). "Frontiers: A Simple Forward Difference-in-Differences
Method." *Marketing Science* 43(2):267-279.

Li, K. T., & Van den Bulte, C. (2022). "Augmented Difference-in-Differences."
*Marketing Science* 42(4):746-767.

Abadie, A., & Zhao, J. (2026). "Synthetic Controls for Experimental Design."
*Working paper.*

Abadie, A., Diamond, A., & Hainmueller, J. (2010). "Synthetic Control
Methods for Comparative Case Studies." *Journal of the American
Statistical Association* 105(490):493-505.
