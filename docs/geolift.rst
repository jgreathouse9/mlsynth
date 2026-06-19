:orphan:

GeoLift Market Selection (GEOLIFT)
==================================

.. currentmodule:: mlsynth

Overview
--------

Most estimators in ``mlsynth`` are *retrospective*: a treatment has happened and
we want its effect. ``GEOLIFT`` is *prospective* — a tool for synthetic
experimental design in geo-experiments. Before any ad spend, it answers:

   *Which* markets should be treated, for *how long*, so that a real lift would
   be detectable?

It is a faithful port of Meta's `GeoLift
<https://github.com/facebookincubator/GeoLift>`_ market-selection routine onto
the ``mlsynth`` Augmented-SCM machinery (Ben-Michael, Feller & Rothstein, 2021
[BMFR2021]_), with conformal inference (Chernozhukov, Wüthrich & Zhu, 2021
[CWZ2021]_) and the standardized design/effect result contract. Reach for it
when test markets must be chosen up front, you want a minimum detectable
effect (MDE) and power per candidate region plus the deployable synthetic
control, you may need to force markets in or out, and — once the experiment
runs — you want to realize the chosen design into an effect report.

Mathematical Formulation
------------------------

Setup and notation
~~~~~~~~~~~~~~~~~~~

There are :math:`N` markets :math:`\mathcal{N} \coloneqq \{1, \dots, N\}` and
:math:`T` periods :math:`t \in \mathcal{T} \coloneqq \{1, \dots, T\}`. The design
uses only the pre-treatment window
:math:`\mathcal{T}_1 \coloneqq \{t \in \mathcal{T} : t \le T_0\}` of length
:math:`T_0`; if post-treatment data exist they are sliced off (see
*Pre/post split*). The outcome of market :math:`j` at time :math:`t` is
:math:`y_{jt}`, with market series
:math:`\mathbf{y}_j = (y_{j1}, \dots, y_{jT})^\top \in \mathbb{R}^{T}`.

A candidate test region is a set :math:`\mathcal{S} \subseteq \mathcal{N}`
of :math:`k \coloneqq |\mathcal{S}|` markets (the ``treatment_size``). It plays
the role of the canonical treated unit through its aggregate series

.. math::

   \mathbf{y}^{\mathcal{S}}, \qquad
   y^{\mathcal{S}}_t \coloneqq \operatorname{agg}_{j \in \mathcal{S}} y_{jt},
   \qquad \operatorname{agg} \in \Bigl\{\textstyle\sum,\ \operatorname{mean}\Bigr\}.

The donor pool is every other market,
:math:`\mathcal{N}_0(\mathcal{S}) \coloneqq \mathcal{N} \setminus \mathcal{S}`
with :math:`N_0 \coloneqq N - k`, giving the donor matrix
:math:`\mathbf{Y}_0^{\mathcal{S}} \coloneqq [\mathbf{y}_j]_{j \in \mathcal{N}_0(\mathcal{S})}
\in \mathbb{R}^{T \times N_0}`. The ``sum`` aggregate is GeoLift's default (the
right object for total spend/lift); the ``mean`` keeps :math:`\mathbf{y}^{\mathcal{S}}`
at donor scale — inside the donor convex hull — for a better-posed fit.

Stage 1 — Candidate nomination
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Enumerating all :math:`\binom{N}{k}` regions is intractable, so GeoLift nominates
a tractable shortlist by correlation similarity. On the pre-period, form the
Pearson correlation matrix :math:`\mathbf{P} = [\rho_{ij}] \in \mathbb{R}^{N \times N}`,

.. math::

   \rho_{ij} \coloneqq
   \frac{\sum_{t \in \mathcal{T}_1}(y_{it} - \bar{y}_i)(y_{jt} - \bar{y}_j)}
        {\sqrt{\sum_{t \in \mathcal{T}_1}(y_{it} - \bar{y}_i)^2}\;
         \sqrt{\sum_{t \in \mathcal{T}_1}(y_{jt} - \bar{y}_j)^2}},
   \qquad \bar{y}_i \coloneqq T_0^{-1}\!\!\sum_{t \in \mathcal{T}_1} y_{it}.

For each anchor :math:`i`, the :math:`k-1` nearest neighbours are the
solution of a per-anchor selection problem — pick the :math:`k-1` other markets
of greatest correlation to the anchor:

.. math::

   \mathcal{S}_i \coloneqq \{i\} \cup
   \operatorname*{arg\,max}_{\substack{\mathcal{T} \subseteq \mathcal{N}
   \setminus \{i\} \\ |\mathcal{T}| = k-1}}
   \sum_{j \in \mathcal{T}} \rho_{ij},

which, because the objective is additively separable in :math:`j`, is solved
exactly by ranking: with :math:`\pi_i` ordering the other markets by descending
correlation
(:math:`\rho_{i,\pi_i(1)} \ge \dots \ge \rho_{i,\pi_i(N-1)}`),
:math:`\mathcal{S}_i = \{i\} \cup \{\pi_i(1), \dots, \pi_i(k-1)\}`. This is the
``mlsynth`` substitute for the intractable global
:math:`\operatorname{arg\,min}_{\mathcal{S}} L(\mathcal{S})` (Stage 2's imbalance
:math:`L`) over all :math:`\binom{N}{k}` regions — :math:`N` anchored top-:math:`k`
problems instead, each closed-form. The shortlist is
:math:`\{\mathcal{S}_i\}_{i \in \mathcal{N}}` deduplicated —
:math:`N` candidates instead of :math:`\binom{N}{k}`. The stochastic
("paired-jitter") variant replaces ranks :math:`1, \dots, k-1` by one draw from
each adjacent pair :math:`\{1,2\}, \{3,4\}, \dots`, exploring near-rank neighbours
(``run_stochastic``; ``stochastic_mode="global"`` is faithful to GeoLift,
``"per_anchor"`` draws independently per anchor).

Forcing constraints. Given a forced-in set
:math:`\mathcal{S}_{\mathrm{in}}` (``to_be_treated``) and a forbidden set
:math:`\mathcal{S}_{\mathrm{out}}` (``not_to_be_treated``,
:math:`\mathcal{S}_{\mathrm{in}} \cap \mathcal{S}_{\mathrm{out}} = \varnothing`),
nominees are drawn from the free pool
:math:`\mathcal{F} \coloneqq \mathcal{N} \setminus (\mathcal{S}_{\mathrm{in}} \cup
\mathcal{S}_{\mathrm{out}})` at size :math:`k - |\mathcal{S}_{\mathrm{in}}|` and
unioned with the forced-in set, so every candidate satisfies
:math:`\mathcal{S}_{\mathrm{in}} \subseteq \mathcal{S}` and
:math:`\mathcal{S} \cap \mathcal{S}_{\mathrm{out}} = \varnothing`.

Stage 1b — Design constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Beyond hard forcing lists, the rule-based constraints restrict the admissible
regions and donor pools (the prose walkthrough with runnable examples is in
*Design constraints (geography, coverage, size)* below). Each is one of two kinds,
following the LEXSCM constraint algebra (:doc:`lexscm`): a treatment
criterion filtering admissible :math:`\mathcal{S}`, or a control criterion
restricting the donor pool. None enters the inner weight program of Stage 2.

Interference graph. Encode market interference by a symmetric
:math:`\mathbf{A} = [A_{ij}] \in \{0,1\}^{N \times N}` with zero diagonal, where
:math:`A_{ij} = 1` iff markets :math:`i, j` interfere. It is built from a cluster
labelling :math:`c : \mathcal{N} \to \mathcal{C}` (``cluster_col``,
:math:`A_{ij} = 1` iff :math:`c(i) = c(j)`) and/or a spillover matrix
:math:`\mathbf{W}` (``adjacency``, :math:`A_{ij} = 1` iff
:math:`W_{ij} > \theta`, the ``spillover_threshold``), combined entrywise by
logical OR. The conflict-neighbours of a region are

.. math::

   \mathcal{A}(\mathcal{S}) \coloneqq
   \bigl\{\, k \in \mathcal{N} : A_{jk} = 1 \ \text{for some}\ j \in \mathcal{S}
   \,\bigr\} \setminus \mathcal{S}.

*Treatment criterion — no interference.* The treated region must be an
independent set of :math:`\mathbf{A}`:
:math:`A_{ij} = 0 \ \forall\, i, j \in \mathcal{S}` (at most one market per
cluster). *Control criterion — spillover exclusion.* The donor pool drops the
conflict-neighbours, refining the Stage-2 pool to

.. math::

   \mathcal{N}_0(\mathcal{S}) \;=\;
   \mathcal{N} \setminus \bigl(\mathcal{S} \cup \mathcal{A}(\mathcal{S})\bigr),

so a treated market's interferers can be neither co-treated nor used as its own
donors.

Coverage quotas. With a stratum labelling :math:`g : \mathcal{N} \to
\mathcal{G}` (``stratum_col``) and the strata that contain an eligible market
:math:`\mathcal{G}_{\mathrm{elig}}`, the region must satisfy, for the bounds
:math:`q_{\min}` (``min_per_stratum``) and :math:`q_{\max}` (``max_per_stratum``),

.. math::

   q_{\min} \le \bigl|\{\, j \in \mathcal{S} : g(j) = s \,\}\bigr|
   \quad \forall\, s \in \mathcal{G}_{\mathrm{elig}},
   \qquad
   \bigl|\{\, j \in \mathcal{S} : g(j) = s \,\}\bigr| \le q_{\max}
   \quad \forall\, s \in \mathcal{G}.

Size band. With market sizes :math:`z_j` (``size_col``) and bounds
:math:`[\underline{z}, \overline{z}]` (``min_size`` / ``max_size``), only in-band
markets are treatment-eligible,
:math:`\mathcal{N}_{\mathrm{size}} \coloneqq \{\, j : \underline{z} \le z_j \le
\overline{z} \,\}`, so :math:`\mathcal{S} \subseteq \mathcal{N}_{\mathrm{size}}`;
out-of-band markets remain available as donors. The ceiling
:math:`\overline{z}` is the a-priori analogue of the synthesizability the scaled
L2 imbalance :math:`\kappa(\mathcal{S})` (below) measures post-hoc.

All three treatment criteria act on the admissible supports — they sit exactly
where the cardinality constraint :math:`|\mathcal{S}| = k` already lives and only
*shrink* the candidate set. If no region satisfies them (e.g.
:math:`k` exceeds :math:`|\mathcal{C}|` or :math:`|\mathcal{G}_{\mathrm{elig}}|`,
or :math:`|\mathcal{N}_{\mathrm{size}}| < k`) the design reports infeasibility
(:class:`~mlsynth.exceptions.MlsynthConfigError`) rather than returning a
degenerate region. With none of the constraints supplied,
:math:`\mathbf{A} = \mathbf{0}` and every region is admissible — the unconstrained
nomination.

Stage 2 — The synthetic control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a candidate :math:`\mathcal{S}`, the counterfactual is a weighted donor
combination. The default is the Augmented SCM [BMFR2021]_. Each period is
first centred by the donor mean
:math:`\mu_t \coloneqq N_0^{-1}\sum_{j \in \mathcal{N}_0}\! y_{jt}` (the augsynth
intercept), giving
:math:`\widetilde{\mathbf{y}}^{\mathcal{S}} = \mathbf{y}^{\mathcal{S}} - \boldsymbol{\mu}`
and :math:`\widetilde{\mathbf{Y}}_0 = \mathbf{Y}_0 - \boldsymbol{\mu}\mathbf{1}^\top`.
A base simplex SCM is solved on the pre-period,

.. math::

   \mathbf{w}^{\mathrm{scm}} \in
   \operatorname*{argmin}_{\mathbf{w} \in \Delta^{N_0}}
   \bigl\| \widetilde{\mathbf{y}}^{\mathcal{S}}_{\mathcal{T}_1}
         - \widetilde{\mathbf{Y}}_{0,\mathcal{T}_1}\mathbf{w} \bigr\|_2^2,
   \quad
   \Delta^{N_0} \coloneqq \{\mathbf{w} \in \mathbb{R}_{\ge 0}^{N_0} :
   \|\mathbf{w}\|_1 = 1\},

then ridge-augmented. The augmented weights are themselves the solution of an
optimization problem — a ridge-penalized balance objective *anchored* at the
simplex fit, dropping the simplex constraint so the correction can debias
(augsynth's ASCM, hence the possible small negative weights):

.. math::

   \mathbf{w}^\ast \;=\;
   \operatorname*{arg\,min}_{\mathbf{w} \in \mathbb{R}^{N_0}}
   \;\bigl\| \widetilde{\mathbf{y}}^{\mathcal{S}}_{\mathcal{T}_1}
         - \widetilde{\mathbf{Y}}_{0,\mathcal{T}_1}\mathbf{w} \bigr\|_2^2
   \;+\; \lambda\,\bigl\| \mathbf{w} - \mathbf{w}^{\mathrm{scm}} \bigr\|_2^2 ,

whose stationarity condition has the closed form (push-through identity)

.. math::

   \mathbf{w}^\ast = \mathbf{w}^{\mathrm{scm}} +
   \widetilde{\mathbf{Y}}_{0,\mathcal{T}_1}^{\top}
   \bigl(\widetilde{\mathbf{Y}}_{0,\mathcal{T}_1}\widetilde{\mathbf{Y}}_{0,\mathcal{T}_1}^{\top}
   + \lambda \mathbf{I}\bigr)^{+}
   \bigl(\widetilde{\mathbf{y}}^{\mathcal{S}}_{\mathcal{T}_1}
   - \widetilde{\mathbf{Y}}_{0,\mathcal{T}_1}\mathbf{w}^{\mathrm{scm}}\bigr),

with the penalty :math:`\lambda` chosen by leave-one-period-out cross-validation
(:math:`\lambda \to 0` recovers the plain simplex SCM; :math:`\lambda \to \infty`
pulls back to :math:`\mathbf{w}^{\mathrm{scm}}`).

How the cross-validation is solved (a faster route than the native method).
The reference implementation (augsynth) selects :math:`\lambda` by, for every
fold and every penalty on the grid, inverting
:math:`\widetilde{\mathbf{Y}}_{0}\widetilde{\mathbf{Y}}_{0}^{\top} + \lambda\mathbf{I}`
and cold-refitting the base simplex weights. ``mlsynth`` computes the identical
quantity by a cheaper algebraic route. Within a fold the matrix
:math:`\widetilde{\mathbf{Y}}_{0}\widetilde{\mathbf{Y}}_{0}^{\top}` and the
anchor :math:`\mathbf{w}^{\mathrm{scm}}` do not depend on :math:`\lambda`, so a
single matrix factorization serves the whole grid: one symmetric
eigendecomposition
:math:`\widetilde{\mathbf{Y}}_{0}\widetilde{\mathbf{Y}}_{0}^{\top}=\mathbf{V}\operatorname{diag}(\mathbf{d})\mathbf{V}^{\top}`
gives
:math:`(\widetilde{\mathbf{Y}}_{0}\widetilde{\mathbf{Y}}_{0}^{\top}+\lambda\mathbf{I})^{-1}=\mathbf{V}\operatorname{diag}\!\bigl(1/(\mathbf{d}+\lambda)\bigr)\mathbf{V}^{\top}`,
replacing one matrix inversion per penalty with one decomposition per fold. When
there are more pre-periods than donors (:math:`J < m`, the usual geo case) the
factorization is taken in *dual* form — an economy singular value decomposition
of the :math:`m \times J` donor matrix (:math:`J` components, cost
:math:`O(mJ^2)`) rather than the :math:`m \times m` eigendecomposition (cost
:math:`O(m^3)`) — which is the cheaper of the two whenever periods outnumber
donors. The single-penalty correction is taken with a linear solve rather than
an explicit inverse; and the leave-one-out folds (tiny perturbations of one
another) are
warm-started from the previous fold's base weights. Because the base simplex
objective is strictly convex under full column rank, the cross-validation curve,
the selected :math:`\lambda`, and the fitted weights are unchanged to numerical
tolerance — only the runtime differs. On a representative design this cuts the
fit roughly five-fold relative to the per-penalty-inversion native method, which
matters because the cross-validation is re-run across every candidate, duration,
and lookback placement in the market-selection search.

The counterfactual and gap follow the canon,

.. math::

   \widehat{y}^{\mathcal{S}}_t \coloneqq \bigl(\mathbf{Y}_0\mathbf{w}^\ast\bigr)_t,
   \qquad
   \tau_t \coloneqq y^{\mathcal{S}}_t - \widehat{y}^{\mathcal{S}}_t,
   \qquad
   \widehat{\tau} \coloneqq |\mathcal{T}_2|^{-1}\!\!\sum_{t \in \mathcal{T}_2}\! \tau_t .

The ``augment=None`` variant is the plain simplex SCM with an explicit intercept
:math:`\alpha = \operatorname{mean}_{\mathcal{T}_1}(\mathbf{y}^{\mathcal{S}}
- \mathbf{Y}_0\mathbf{w}^\ast)`, predicting
:math:`\widehat{y}^{\mathcal{S}}_t = \alpha + (\mathbf{Y}_0\mathbf{w}^\ast)_t`.

Pre-fit quality is the scaled L2 imbalance — the fitted pre-period imbalance
relative to the imbalance of uniform donor weights
:math:`\mathbf{w}^{\mathrm{unif}} \coloneqq N_0^{-1}\mathbf{1}`,

.. math::

   \kappa(\mathcal{S}) \coloneqq
   \frac{\bigl\|\mathbf{Y}_{0,\mathcal{T}_1}\mathbf{w}^\ast
            - \mathbf{y}^{\mathcal{S}}_{\mathcal{T}_1}\bigr\|_2}
        {\bigl\|\mathbf{Y}_{0,\mathcal{T}_1}\mathbf{w}^{\mathrm{unif}}
            - \mathbf{y}^{\mathcal{S}}_{\mathcal{T}_1}\bigr\|_2}
   \;\in\; [0, \infty),

so :math:`\kappa = 0` is a perfect fit and :math:`\kappa = 1` is no better than
the donor average. It is unitless (hence comparable across regions of different
magnitudes) and is *reported* per candidate.

Stage 3 — Power simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Power is estimated by placebo-in-time experiments carved from the end of the
pre-period. For a treatment duration :math:`\ell` (a ``durations`` entry) and
a lookback placement :math:`s \in \{1, \dots, L\}` (with
:math:`L =` ``lookback_window``), the pseudo-treatment window is the :math:`\ell`
periods ending :math:`s - 1` before :math:`T_0`,

.. math::

   \mathcal{T}^{(s,\ell)}_{2} \coloneqq
   \{\,T_0 - \ell - s + 2,\ \dots,\ T_0 - s + 1\,\},
   \qquad
   \mathcal{T}^{(s,\ell)}_{1} \coloneqq \{1, \dots, T_0 - \ell - s + 1\},

faithful to GeoLift's ``max_time - tp - sim + 2`` with :math:`T_0` the pre-period
end. The SCM is fit on :math:`\mathcal{T}^{(s,\ell)}_{1}`. For an effect size
:math:`\delta` (an ``effect_sizes`` entry) a known multiplicative lift is injected
on the pseudo-post block,
:math:`y^{\mathcal{S},(\delta)}_t = (1+\delta)\,y^{\mathcal{S}}_t` for
:math:`t \in \mathcal{T}^{(s,\ell)}_2`, which shifts the gap by
:math:`\delta\,y^{\mathcal{S}}_t` there.

Detection uses the conformal test [CWZ2021]_. With the post-block statistic

.. math::

   S_q(\boldsymbol{\tau}) \coloneqq
   \Bigl( |\mathcal{T}_2|^{-1/2}
   \textstyle\sum_{t \in \mathcal{T}_2} |\tau_t|^{q} \Bigr)^{1/q}
   \qquad (q = 1 \text{ by default}),

the joint-null p-value compares the observed statistic to :math:`n_s` i.i.d.
permutations :math:`\Pi` of the residual path,

.. math::

   p \coloneqq \frac{1}{n_s}\sum_{\Pi}
   \mathbf{1}\!\bigl\{\, S_q(\boldsymbol{\tau}_{\mathcal{T}_2})
   \le S_q\bigl((\Pi\boldsymbol{\tau})_{\mathcal{T}_2}\bigr) \,\bigr\},

and an effect is *detected* when :math:`p < \alpha`. The permutation set
:math:`\Pi` follows ``conformal_type``: ``"iid"`` (the augsynth/GeoLift default —
:math:`n_s` independent draws) or ``"block"`` (the :math:`T` moving-block cyclic
shifts :math:`\Pi_k(t) = ((t + k) \bmod T)`, which preserve serial dependence
and are deterministic, ignoring :math:`n_s`). The :math:`n_s` reference
statistics are evaluated in one broadcast over the whole permutation block
rather than one Python call per permutation; the permutations and their random
draws are unchanged, so the p-value is bit-identical to the per-permutation
loop (at the default :math:`q=1`) — only faster, which matters because this
reference is rebuilt for every candidate, duration, lookback, and effect size
in the search. Power is the detection rate across the :math:`L` lookback
placements,

.. math::

   \beta(\mathcal{S}, \ell, \delta) \coloneqq
   \frac{1}{L}\sum_{s=1}^{L}
   \mathbf{1}\!\bigl\{\, p^{(s)}(\mathcal{S}, \ell, \delta) < \alpha \,\bigr\}.

.. admonition:: Fit-once, sweep-:math:`\delta` (an exact optimization)

   The injection touches only the post block, so the pre-period the
   cross-validation sees is identical across effect sizes; the CV-selected
   :math:`\lambda` is therefore the same for every :math:`\delta`. ``mlsynth``
   cross-validates once per :math:`(\mathcal{S}, \ell, s)` and reuses
   :math:`\lambda` across :math:`\delta` (augsynth's own behaviour). This is
   *provably identical* to GeoLift's per-:math:`\delta` refit — pinned by
   ``test_simulate_lookback_cv_once_equals_per_es_refit`` — at
   :math:`1/|\{\delta\}|` the cross-validation cost.

Stage 4 — MDE and the composite rank
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The minimum detectable effect for a region/duration is the smallest-magnitude
effect whose power clears the threshold :math:`\beta_0` (``power_threshold``,
default :math:`0.8`),

.. math::

   \delta^\ast(\mathcal{S}, \ell) \coloneqq
   \operatorname*{arg\,min}_{\delta \,:\, \beta(\mathcal{S},\ell,\delta) \ge \beta_0}
   |\delta|,

with GeoLift's signed positive/negative tie rule. Writing
:math:`\widehat{\delta}` for the recovered lift at :math:`\delta^\ast` and the
recovery error :math:`\eta(\mathcal{S},\ell) \coloneqq |\widehat{\delta} -
\delta^\ast|`, the composite rank is the mean of three dense ranks
(:math:`\operatorname{dr}` over the surviving candidates), faithful to GeoLift,

.. math::

   r(\mathcal{S}, \ell) =
   \operatorname{rank}\!\left(\tfrac13\Bigl[
   \operatorname{dr}\!\bigl(|\delta^\ast|\bigr) +
   \operatorname{dr}\!\bigl(\beta\bigr) +
   \operatorname{dr}\!\bigl(\eta\bigr) \Bigr]\right),
   \qquad \text{(lower is better).}

.. note::

   Two GeoLift-fidelity quirks, replicated as-is: :math:`\operatorname{dr}(\beta)`
   is ascending (an MDE whose power sits *just* above :math:`\beta_0` is a
   tighter estimate of the threshold, so it ranks better), and the scaled L2
   imbalance :math:`\kappa` is not a ranking term — only :math:`\delta^\ast`,
   :math:`\beta`, and :math:`\eta` enter. Both are documented and one line to
   change.

Identifying assumptions
~~~~~~~~~~~~~~~~~~~~~~~~~

1. Pre-period synthesizability. The aggregate :math:`\mathbf{y}^{\mathcal{S}}`
   lies in (or near) the span/convex hull of the donor pool over
   :math:`\mathcal{T}_1`. Quantified by :math:`\kappa(\mathcal{S})`: a low value
   certifies that the synthetic tracks the region, the prerequisite for a
   credible counterfactual.

   *Remark.* With ``how="sum"`` the target is :math:`k\times` donor scale and can
   sit outside the convex hull, inflating :math:`\kappa` toward 1; ``how="mean"``
   restores synthesizability. The choice is the user's.

2. Exchangeability under the null. The conformal test treats the residual
   path as exchangeable under :math:`H_0`: no effect, which the all-period refit
   underlying :math:`p` is designed to deliver [CWZ2021]_.

3. Stationarity of the placebo windows. Power from the lookback placements
   transports to the real experiment only if the pre-period dynamics resemble the
   experiment window — the usual SC stability assumption.

Budget planning (CPIC)
~~~~~~~~~~~~~~~~~~~~~~~~

Setting ``cpic`` (cost per incremental conversion) turns each candidate's MDE
into a spend, faithful to ``GeoLiftMarketSelection``:

.. math::

   \mathrm{investment} \;=\; \mathrm{cpic} \times \delta \times
   \sum_{i \in \mathcal{S}} \sum_{t \in \mathcal{W}} Y_{it},

i.e. cost-per-incremental :math:`\times` effect size :math:`\times` the
summed treated volume over the (lookback) window — the baseline outcome, on
the total scale, independent of the mean-of-units fit. The shortlist carries an
``investment`` column; supplying ``budget`` drops candidates whose detectable
investment exceeds it (GeoLift's ``abs(budget) > abs(Investment)`` gate). The
realized report adds the post-test ``cost`` :math:`= \mathrm{cpic} \times`
incremental outcome. The investment is a deterministic data transform, so it
matches ``GeoLiftMarketSelection`` to the cent (durable case
``geolift_cpic``); ROI (a value/margin per conversion) is a planned extension
beyond GeoLift's cost-only ``cpic``.

Inference and the realized design
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The design phase reports :math:`\delta^\ast` and :math:`\beta` per candidate.
When a ``post_col`` leaves a post window, :meth:`GEOLIFT.fit` realizes the
winning design *under the hood* — applying the winner's pre-period weights
:math:`\mathbf{w}^\ast` to the full panel and running conformal inference:
the per-period effect :math:`\tau_t` for :math:`t \in \mathcal{T}_2`, prediction
intervals by test inversion (a grid of nulls :math:`\tau_0`, the interval being
the non-rejected range at level :math:`\alpha`), and the joint-null p-value
:math:`p` — exposed on ``result.report`` (the ``DesignResult`` resolving to its
``EffectResult``). A design over a *no-effect* post window returns :math:`p`
non-significant and intervals covering zero.

Pipeline and Options
--------------------

The estimator is a thin front door over the helper pipeline; each stage above is
a tested leaf in ``mlsynth/utils/geolift_helpers/``. The public surface is a
single :meth:`GEOLIFT.fit` — realization and plotting are handled inside it,
driven by the data and config (a ``post_col`` triggers realization;
``display_graphs`` triggers the plot).

* ``treatment_size`` :math:`k`, ``durations`` :math:`\{\ell\}`, ``effect_sizes``
  :math:`\{\delta\}`, ``lookback_window`` :math:`L`.
* ``to_be_treated`` / ``not_to_be_treated`` —
  :math:`\mathcal{S}_{\mathrm{in}}` / :math:`\mathcal{S}_{\mathrm{out}}`.
* ``cluster_col`` / ``adjacency`` / ``spillover_threshold``, ``stratum_col`` /
  ``min_per_stratum`` / ``max_per_stratum``, ``size_col`` / ``min_size`` /
  ``max_size`` — rule-based design constraints (see *Design constraints* below).
* ``post_col`` — a 0/1 column marking post-treatment periods; the design slices to
  :math:`\mathcal{T}_1`, so it is identical whether you pass the full
  post-treatment panel or a pre-only one (the "rerun after treatment"
  invariance). Different post lengths simply change :math:`T_0`.
* ``how`` (:math:`\operatorname{sum}` / :math:`\operatorname{mean}`),
  ``augment`` (``"ridge"`` / ``None``), ``alpha`` :math:`\alpha`,
  ``power_threshold`` :math:`\beta_0`, ``ns`` :math:`n_s`,
  ``run_stochastic`` / ``stochastic_mode``.
* ``conformal_type`` — the conformal permutation scheme, ``"iid"`` (default,
  matching GeoLift) or ``"block"`` (moving-block cyclic shifts for
  serially-dependent residuals; GeoLift's ``conformal_type = "block"`` option).
* ``n_jobs`` — parallel workers for the candidate search (default ``-1``, all
  cores; set a positive integer to cap the count, or ``1`` to run serially).
  The search is embarrassingly parallel: each candidate's power simulation and
  deployable design fit are independent and use the fixed ``seed``, so any
  worker count spreads the candidates across workers (via joblib) and returns
  the *identical* shortlist, only faster.

Scanning several ``durations`` yields an MDE *per duration*
(":math:`\ell = 7` detects 10%, but :math:`\ell = 14` is needed for 5%"):

.. code-block:: python

   res = GEOLIFT({..., "durations": [7, 14, 21]}).fit()
   res.power[["candidate", "duration", "mde", "power"]]   # one row per (S, l)

Plotting
~~~~~~~~

With ``display_graphs`` (default ``True``), :meth:`GEOLIFT.fit` plots the
recommended design in the mlsynth house style
(:func:`mlsynth.utils.plotting.mlsynth_style`): the design phase shows
:math:`\mathbf{y}^{\mathcal{S}}` vs :math:`\widehat{\mathbf{y}}^{\mathcal{S}}`
over :math:`\mathcal{T}_1`; the post phase (when the design was realized)
adds the conformal band and the per-period gap :math:`\tau_t` over
:math:`\mathcal{T}_2`, with the intervention line at :math:`T_0`. The standalone
helper :func:`mlsynth.utils.geolift_helpers.marketselect.plotter.plot_design`
re-draws from a result on demand.

Example: GeoLift's 40-Market Panel
----------------------------------

The package ships GeoLift's example panel
(``basedata/geolift_market_data.csv``): :math:`N = 40` markets over
:math:`T = 90` days. We design a :math:`k = 3` test region, then realize it over a
10-day no-effect post window (so the realized effect should be null).

.. code-block:: python

   import pandas as pd
   from mlsynth import GEOLIFT

   df = pd.read_csv(
       "https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
       "refs/heads/main/basedata/geolift_market_data.csv"
   )
   df["post"] = df["date"].isin(sorted(df["date"].unique())[-10:]).astype(int)

   geo = GEOLIFT({
       "df": df, "outcome": "Y", "unitid": "location", "time": "date",
       "post_col": "post",                 # design on the pre-80, post-10 reserved
       "treatment_size": 3, "durations": [14], "effect_sizes": [0.0, 0.1, 0.2],
       "lookback_window": 3, "how": "mean", "augment": "ridge", "ns": 100,
   })
   res = geo.fit()              # designs, auto-realizes (post window) and plots
   print(res.selected_units, res.search.winner.mde, res.search.winner.power)
   print("joint conformal p:", round(res.report.inference.p_value, 3))   # ~0.66 -> null

Without a ``post_col`` the same call returns a design-only result
(``res.report is None``); set ``display_graphs=False`` to suppress the plot.

The synthetic tracks the test markets, :math:`\widehat{\tau} \approx 0`, and the
joint conformal p-value is far from significant — the correct null over a placebo
post period.

Every option at once — the full market-selection call
-----------------------------------------------------

GeoLift's ``GeoLiftMarketSelection`` exposes the whole design surface in one
call:

.. code-block:: r

   MarketSelections <- GeoLiftMarketSelection(data = GeoTestData_PreTest,
       treatment_periods = c(10, 15),
       N = c(2, 3, 4, 5),
       Y_id = "Y", location_id = "location", time_id = "time",
       effect_size = seq(0, 0.2, 0.05),
       lookback_window = 1,
       include_markets = c("chicago"),
       exclude_markets = c("honolulu"),
       cpic = 7.50, budget = 100000,
       alpha = 0.1, Correlations = TRUE,
       fixed_effects = TRUE, side_of_test = "two_sided")

Every argument maps onto a ``GEOLIFT`` config field:

.. code-block:: python

   import pandas as pd
   from mlsynth import GEOLIFT

   df = pd.read_csv(                                      # GeoLift_PreTest, 40 mkts x 90d
       "https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
       "refs/heads/main/basedata/geolift_market_data.csv"
   )

   cfg = {
       "df": df, "outcome": "Y", "unitid": "location", "time": "date",
       "durations": [10, 15],                       # treatment_periods
       "effect_sizes": [0.0, 0.05, 0.10, 0.15, 0.20],
       "lookback_window": 1,
       "to_be_treated": ["chicago"],                # include_markets
       "not_to_be_treated": ["honolulu"],           # exclude_markets
       "cpic": 7.50, "budget": 100000.0,            # budget planning
       "alpha": 0.1,
       "fixed_effects": True,                       # GeoLift default
       "how": "mean", "augment": "ridge",
       "conformal_type": "iid",                     # two-sided |stat| permutation
       "display_graphs": False,
   }
   # N = c(2,3,4,5): GEOLIFT takes one treatment_size; scan by looping it.
   shortlist = pd.concat(
       GEOLIFT({**cfg, "treatment_size": k}).fit().search.shortlist for k in (2, 3, 4, 5)
   ).sort_values("rank")
   shortlist[["candidate", "duration", "mde", "power", "investment", "scaled_l2", "rank"]]

A few argument notes: ``Correlations=TRUE`` is GeoLift's correlation ranking,
which mlsynth always uses to nominate candidates (Stage 1); ``side_of_test =
"two_sided"`` is the default conformal statistic (the symmetric :math:`|x|^q`
norm); ``cpic`` + ``budget`` drop candidates whose detectable investment busts
the budget (see *Budget planning* above).

The recommended designs (GeoLift's ``BestMarkets`` table; mlsynth's
``shortlist`` carries the same candidates and reproduces ``Investment`` to the
cent, see :doc:`replications/geolift`):

================================================  ===  ==========  =======  =====  =====
Test markets                                      Dur  Investment  AvgATT   L2     Rank
================================================  ===  ==========  =======  =====  =====
chicago, cincinnati, houston, portland            15   74,118.38   159.36   0.197  1
chicago, portland                                 15   64,563.75   290.01   0.174  1
chicago, cincinnati, houston, portland            10   99,027.75   316.62   0.197  3
chicago, portland                                 10   43,646.25   300.94   0.168  3
chicago, houston, portland                        10   75,389.25   350.31   0.231  5
chicago, cincinnati, houston, nashville, san d.   15   95,755.50   146.80   0.270  6
atlanta, chicago                                  15   81,348.75   336.78   0.446  7
atlanta, chicago, cleveland, las vegas            15   86,661.75   220.82   0.532  7
================================================  ===  ==========  =======  =====  =====

(28 designs total; the maximum surviving investment is $99,321.75 < the $100k
budget, so the budget gate held.) Lower ``Rank`` is better; the smallest, lowest-
imbalance region that clears power wins — here ``chicago, portland``.


Reading the results — plots, tables, and weights
------------------------------------------------

Everything the design and the realized report produce lives on the result
object. The complete, runnable example below loads the data, realizes a design
with a budget (``cpic``), and then draws every view — the power / MDE / budget
table, observed vs synthetic, the effect with its conformal band, the donor
weights, the realized cost, and the built-in plot:

.. code-block:: python

   import pandas as pd
   import matplotlib.pyplot as plt
   from mlsynth import GEOLIFT
   from mlsynth.utils.geolift_helpers.marketselect.plotter import plot_design

   url = ("https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
          "refs/heads/main/basedata/geolift_test_data.csv")
   df = pd.read_csv(url)                                   # GeoLift_Test: 40 mkts x 105 days
   dates = sorted(df["date"].unique())
   df["post"] = df["date"].isin(dates[90:]).astype(int)   # last 15 days = treatment window

   res = GEOLIFT({
       "df": df, "outcome": "Y", "unitid": "location", "time": "date",
       "treatment_size": 2, "to_be_treated": ["chicago", "portland"],
       "durations": [15], "effect_sizes": [0.0, 0.10], "post_col": "post",
       "cpic": 7.50, "how": "mean", "fixed_effects": True, "display_graphs": False,
   }).fit()

   # headline + the power / MDE / budget shortlist (one row per candidate x duration)
   print(res.selected_units, res.report.effects.att, res.report.inference.p_value)
   print(res.power[["candidate", "duration", "mde", "power", "investment"]])
   print(res.report.weights.summary_stats["cost"])        # realized spend = cpic x incremental

   # observed vs synthetic
   ts = res.report.time_series
   plt.plot(ts.time_periods, ts.observed_outcome, "k", label="observed")
   plt.plot(ts.time_periods, ts.counterfactual_outcome, "r--", label="synthetic")
   plt.axvline(ts.intervention_time, color="grey", ls=":"); plt.legend(); plt.show()

   # the effect (gap) with its conformal prediction band
   d = res.report.inference.details
   plt.plot(ts.time_periods, ts.estimated_gap, "k")
   plt.fill_between(d["periods"], d["lower"], d["upper"], alpha=0.3)
   plt.axhline(0, color="grey", ls=":"); plt.show()

   # donor weights, biggest contributors
   w = res.report.weights.donor_weights
   top = dict(sorted(w.items(), key=lambda kv: -abs(kv[1]))[:10])
   plt.bar(top.keys(), top.values()); plt.xticks(rotation=45, ha="right"); plt.show()

   # the built-in plot (design + realized phases: conformal band + gap)
   plot_design(res, report=res.report, show=True)

For the full power-vs-effect-size curve of a region (GeoLift's
``GeoLiftPower`` plot) — power rising through the threshold marks the MDE — run
the scoring helpers directly:

.. code-block:: python

   import pandas as pd
   import matplotlib.pyplot as plt
   from mlsynth.utils.datautils import geoex_dataprep
   from mlsynth.utils.geolift_helpers.marketselect.helpers.batch import run_simulations
   from mlsynth.utils.geolift_helpers.marketselect.helpers.aggregate import compute_power

   url = ("https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
          "refs/heads/main/basedata/geolift_market_data.csv")
   Ywide = geoex_dataprep(pd.read_csv(url), "location", "date", "Y")["Ywide"]
   cube = run_simulations(
       Ywide, [frozenset({"chicago", "portland"})], durations=[15],
       lookback_window=1, effect_sizes=[0.0, 0.05, 0.10, 0.15, 0.20],
       fixed_effects=True, ns=500)
   pw = compute_power(cube, alpha=0.10)
   plt.plot(pw["effect_size"], pw["power"], marker="o")
   plt.axhline(0.8, ls="--", color="grey")                # power threshold -> MDE crossing
   plt.xlabel("injected lift"); plt.ylabel("power"); plt.show()


Design constraints (geography, coverage, size)
----------------------------------------------

Upstream GeoLift restricts the treated side only through hard market lists
(``to_be_treated`` / ``not_to_be_treated``). ``mlsynth`` adds a rule-based
constraint layer so the candidate nomination and donor pool can encode geography
and other design considerations. Every restriction is one of two kinds (the
LEXSCM vocabulary, see :doc:`lexscm`): a treatment criterion filtering which
candidate test regions are admissible, or a control criterion restricting a
candidate's donor pool. Neither touches the inner weight solve — they only change
*where the design is allowed to look*. With none of these fields set, the design
is identical to the unconstrained run.

* ``cluster_col`` — a per-market cluster label (DMA / state). Markets in the same
  cluster interfere: at most one may be treated per candidate (treatment
  criterion, the *independent-set* rule) and a treated market's same-cluster
  geos are dropped from its donor pool (control criterion, the *spillover
  exclusion*). An ``adjacency`` matrix of pairwise spillover strengths, thresholded
  by ``spillover_threshold`` and combined with ``cluster_col`` by logical OR, is
  the continuous alternative.
* ``stratum_col`` + ``min_per_stratum`` / ``max_per_stratum`` — coverage
  quotas: require at least ``min`` treated markets in every stratum that has an
  eligible market ("test in every region"), and/or at most ``max`` per stratum (a
  quota). A treatment criterion.
* ``size_col`` + ``min_size`` / ``max_size`` — a treated-unit size band: only
  in-band markets are eligible for *treatment* (they remain available as donors).
  The floor is a power / operational minimum; the ceiling encodes
  synthesizability — a market far larger than the donors cannot sit inside their
  convex hull, which is exactly what the scaled-L2 imbalance :math:`\kappa`
  measures, so ``max_size`` is an a-priori version of :math:`\kappa \to 1`.

When the constraints admit no candidate region (e.g. ``treatment_size`` exceeds
the number of clusters or strata to cover, or the size band leaves too few
markets), :meth:`GEOLIFT.fit` raises :class:`~mlsynth.exceptions.MlsynthConfigError`
rather than returning a degenerate design.

.. admonition:: Real geography — the bundled DMA contiguity map

   ``adjacency`` is where you "literally take geography into account": pass a
   real bordering matrix and treated markets are forced apart while each
   one's neighbours are barred from its donor pool. The package ships the
   Nielsen US market-area map at ``basedata/markets/dma_adjacency.csv`` — a
   ``206 × 206`` symmetric 0/1 contiguity matrix indexed by DMA name (with
   ``dma_metadata.csv`` carrying state, division, and lat/long) — so a panel
   keyed by DMA can use true borders directly:
   ``GEOLIFT({..., "adjacency": pd.read_csv(".../dma_adjacency.csv", index_col=0)})``.
   The same artifact powers LEXSCM's spillover designs. A worked end-to-end check
   on real borders lives in ``mlsynth/tests/test_geolift_dma_borders.py``. Note
   the tension this exposes: correlation nomination favours *similar* (often
   *bordering*) markets, so the independent-set filter does real work — it can
   prune many nominees, and a very dense border graph can leave none (a reported
   infeasibility, not a silent degenerate design).

Gallery: every constraint on real geography. The block below is a single
self-contained gallery, mirroring the SYNDES one (:doc:`syndes`): it pulls the
real DMA contiguity map and metadata, simulates a grouped linear factor model of
seasonal weekly sales — latent group structure in the loadings (groups = census
divisions), in the spirit of Liao, Shi & Zheng [RelaxSC]_ — over a real Southeast /
Mid-South footprint, and then shows each geographic constraint GEOLIFT supports
as a compact call. The geography is real; the sales outcome is reproducible and
exists only so the snippets run end to end (a few seconds each). Every block
below assumes this setup has run.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from mlsynth import GEOLIFT

   base = ("https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
           "refs/heads/main/basedata/markets")
   A_full = pd.read_csv(f"{base}/dma_adjacency.csv", index_col=0)   # 206x206 borders
   meta = pd.read_csv(f"{base}/dma_metadata.csv")                   # dma_name, state, ...

   # Restrict to a real Southeast / Mid-South footprint; groups = census divisions.
   DIV = {**{s: "S. Atlantic" for s in ["GA", "FL", "SC", "NC", "VA", "WV"]},
          **{s: "E.S. Central" for s in ["TN", "AL", "KY", "MS"]}}
   meta = meta[meta["state"].isin(DIV)].copy()
   meta["division"] = meta["state"].map(DIV)
   names = [n for n in meta["dma_name"] if n in A_full.index]
   meta = meta[meta["dma_name"].isin(names)].reset_index(drop=True)
   adj = A_full.loc[names, names]                                   # real contiguity

   # Grouped linear factor model of seasonal weekly sales (one loading per division).
   rng = np.random.default_rng(0)
   div = meta.set_index("dma_name").loc[names, "division"].to_numpy()
   state = meta.set_index("dma_name").loc[names, "state"].to_numpy()
   n, r, T = len(names), 3, 60
   group_loading = {g: rng.normal(size=r) for g in sorted(set(div))}
   Lambda = np.array([group_loading[g] for g in div]) + 0.1 * rng.normal(size=(n, r))
   F = np.cumsum(rng.normal(size=(T, r)), axis=0)                   # common factors
   season = 12 * np.sin(2 * np.pi * np.arange(T) / 52)             # annual seasonality
   size = np.round(rng.lognormal(12.5, 0.7, n)).astype(int)        # market size
   sales = (1000 + rng.normal(0, 40, n) + F @ Lambda.T
            + season[:, None] + rng.normal(0, 5, (T, n)))
   df = pd.DataFrame([
       {"dma": names[j], "week": t, "sales": float(sales[t, j]),
        "division": div[j], "state": state[j], "size": int(size[j])}
       for j in range(n) for t in range(T)])

   BASE = dict(df=df, outcome="sales", unitid="dma", time="week",
               durations=[8], effect_sizes=[0.0, 0.10], lookback_window=1,
               how="mean", ns=40, seed=0, display_graphs=False)

Plain cardinality -- nominate the 3-market test region with the best detectable
effect; and force one market in, another out (the latter stays a donor) -- when
the business has already committed to running the test in a given market, or
must keep one out (a regulated market, a market already running another test):

.. code-block:: python

   GEOLIFT({**BASE, "treatment_size": 3}).fit()
   GEOLIFT({**BASE, "treatment_size": 3, "to_be_treated": ["Atlanta, GA"],
            "not_to_be_treated": ["Miami-Fort Lauderdale, FL"]}).fit()

Spillover non-interference. No two test markets in the same state (a
``cluster_col``), or no two sharing a real border (the contiguity matrix) --
reach for this when treating two nearby markets would let one's campaign bleed
into the other (overlapping DMAs/media, cross-border shopping), which biases the
lift. Both do double duty: the treated markets are forced apart, and -- the
control criterion -- each treated market's same-cluster / bordering neighbours
are dropped from its donor pool, so a spilling-over market never sits in the
synthetic control (a contaminated donor would drag the counterfactual toward the
treated trend):

.. code-block:: python

   GEOLIFT({**BASE, "treatment_size": 3, "cluster_col": "state"}).fit()

   res = GEOLIFT({**BASE, "treatment_size": 3, "adjacency": adj,
                  "spillover_threshold": 0.5}).fit()
   treated = res.selected_units
   assert all(adj.loc[a, b] == 0 for a in treated for b in treated if a != b)
   cd = res.search.candidates[0]                    # donors avoid the treated markets' borders
   nbr = {b for a in cd.candidate for b in names if adj.loc[a, b] == 1}
   assert all(str(d) not in nbr for d in cd.weights.donor_weights)

Coverage quotas. Require at least one treated market in every region ("test
everywhere"), or cap the count per region -- use ``min_per_stratum`` when
stakeholders need a read in every region, not just the ones the optimizer finds
easiest to fit, and ``max_per_stratum`` to keep the test from concentrating in a
single area:

.. code-block:: python

   GEOLIFT({**BASE, "treatment_size": 2, "stratum_col": "division",
            "min_per_stratum": 1}).fit()                # >= 1 test market per division
   GEOLIFT({**BASE, "treatment_size": 2, "stratum_col": "division",
            "max_per_stratum": 1}).fit()                # <= 1 test market per division

Size band -- only mid-sized markets are eligible for treatment (the rest stay
donors); the floor is a power minimum (a too-small market cannot detect the
effect) and the ceiling encodes synthesizability (a market far larger than the
donors cannot sit inside their convex hull):

.. code-block:: python

   lo, hi = int(np.quantile(size, 0.2)), int(np.quantile(size, 0.8))
   GEOLIFT({**BASE, "treatment_size": 3, "size_col": "size",
            "min_size": lo, "max_size": hi}).fit()

Budget planning -- with a cost-per-incremental-conversion and a budget, drop any
candidate whose detectable investment busts the cap:

.. code-block:: python

   GEOLIFT({**BASE, "treatment_size": 3, "cpic": 7.50, "budget": 100_000.0}).fit()

Everything at once, returning the ranked shortlist. The constraints filter the
nominated candidates and reshape each one's donor pool, so every design in the
shortlist already honours the geography:

.. code-block:: python

   res = GEOLIFT({**BASE, "treatment_size": 3, "adjacency": adj,
                  "spillover_threshold": 0.5, "stratum_col": "division",
                  "max_per_stratum": 2, "size_col": "size",
                  "min_size": int(np.quantile(size, 0.1)),
                  "max_size": int(np.quantile(size, 0.95))}).fit()
   print(res.selected_units)
   print(res.search.shortlist[["candidate", "duration", "mde", "power"]].head())

When the constraints admit no candidate region (``treatment_size`` exceeds the
clusters/strata to cover, a border graph too dense to seat that many
non-adjacent markets, or a size band that leaves too few), :meth:`GEOLIFT.fit`
raises :class:`~mlsynth.exceptions.MlsynthConfigError` rather than returning a
degenerate design.

Multi-cell designs
------------------

A multi-cell experiment runs several treatments at once — different channels,
budgets, or creatives — each on its own group of geos ("cells" :math:`A, B,
\dots`), all measured against a shared control pool over the same window
(GeoLift's ``GeoLiftMultiCell``). The dedicated estimator is
:class:`mlsynth.MULTICELLGEOLIFT`; its data model is a unit-level
cell-membership column (``"A"`` / ``"B"`` / … for treated geos; blank or a
``control_label`` for controls) plus a ``post_col`` window:

.. code-block:: python

   import pandas as pd
   from mlsynth import MULTICELLGEOLIFT

   url = ("https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
          "refs/heads/main/basedata/geolift_test_data.csv")
   df = pd.read_csv(url)                                   # GeoLift_Test: 40 mkts x 105 days
   dates = sorted(df["date"].unique())
   df["post"] = df["date"].isin(dates[90:]).astype(int)   # last 15 days = treatment window

   #   cell:  "A" -> social-media markets, "B" -> paid-search markets, "" -> control
   cell = {"chicago": "A", "portland": "A", "atlanta": "B", "boston": "B"}
   df["cell"] = df["location"].map(cell).fillna("")        # blank = shared control pool

   res = MULTICELLGEOLIFT({
       "df": df, "outcome": "Y", "unitid": "location", "time": "date",
       "cell_column_name": "cell", "post_col": "post", "fixed_effects": True,
   }).fit()

   res.cells["A"].effects.att          # cell A's per-unit ATT (a full EffectResult)
   res.cells["A"].inference.p_value    # cell A's conformal p
   res.comparison                      # pairwise [{cell_a, cell_b, att_diff, winner}, ...]
   res.winner                          # cell that wins every comparison, or None

Each cell is measured with the same fixed-effect ASCM + conformal inference as a
single cell, excluding the other cells' markets from its donor pool (they are
treated, hence contaminated). The cross-cell ``winner`` uses GeoLift's
non-overlapping-CI rule; with one cell it is *identical* to single-cell GEOLIFT.
See :doc:`multicellgeolift` for the full treatment, the per-cell plots
(``plot_multicell``), and the augsynth cross-validation.

Verification
------------

The realized effect report is cross-validated against GeoLift/augsynth
value-for-value on the package's own ``GeoLift_Walkthrough`` example: with
``fixed_effects=True`` (the default), ``GEOLIFT`` reproduces the walkthrough's
per-unit ATT (155.6), percent lift (5.4%), summed incremental (4667), and
conformal p-value (0.01). See :doc:`replications/geolift` for the four
ingredients required to match (unit fixed effects, mean-of-units fit target, the
all-period conformal refit, and augsynth's period-space ridge ASCM) and the
calibration/placebo evidence behind them. The market-*selection* stages (no
published table) remain a faithful port validated end-to-end on GeoLift's own
data, with each documented divergence (the CV-once optimization *proven exact*,
the corrected per-anchor RNG) available as an opt-in, tested swap.

.. [BMFR2021] Ben-Michael, E., Feller, A., & Rothstein, J. (2021). The Augmented
   Synthetic Control Method. *Journal of the American Statistical Association*.

.. [CWZ2021] Chernozhukov, V., Wüthrich, K., & Zhu, Y. (2021). An Exact and Robust
   Conformal Inference Method for Counterfactual and Synthetic Controls.
   *Journal of the American Statistical Association*.

Comparing GeoLift and SYNDES designs on one plane
-------------------------------------------------

When the same panel could be analysed either with a GeoLift-style design or with
SYNDES (:doc:`syndes`), :func:`mlsynth.compare_methods` scores every candidate
design from both methods on one shared fit-versus-power plane, so the choice
between them is apples-to-apples rather than two different power methodologies.
Each design reduces to a unit-level contrast over the panel; from that contrast
follow two comparable numbers -- the pre-period fit RMSE and a simulated minimum
detectable effect (MDE) at a common horizon -- computed by the same harness for
both methods.

.. code-block:: python

   import pandas as pd
   from mlsynth import compare_methods

   url = ("https://raw.githubusercontent.com/jgreathouse9/mlsynth/main/"
          "basedata/geolift_market_data.csv")          # 40 markets x 90 periods
   df = pd.read_csv(url)

   cmp = compare_methods(
       df, outcome="Y", unitid="location", time="date",
       treated_size=2, n_post=15,
       syndes_options={"time_limit": 5.0},              # cap the SYNDES MIP
   )
   print(cmp.table)        # columns: method, label, treated, fit_rmse, mde_pct, pareto
   cmp.plot()              # overlaid per-method Pareto frontiers

``cmp.table`` is one row per candidate design (GeoLift's anchored neighbourhoods
and SYNDES's pool), with ``fit_rmse`` (lower is a tighter pre-period fit),
``mde_pct`` (lower is more powerful), and a ``pareto`` flag marking the designs
on the joint frontier. ``cmp.plot()`` overlays the two methods' frontiers; a
design from one method that sits below and to the left of the other's frontier
dominates it on both fit and power. The SYNDES side of the comparison is
documented in full on the :doc:`syndes` page.

Core API
--------

.. autoclass:: mlsynth.GEOLIFT
   :members: fit

.. autoclass:: mlsynth.utils.geolift_helpers.config.GeoLiftConfig
   :members:
