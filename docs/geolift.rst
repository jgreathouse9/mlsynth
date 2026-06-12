:orphan:

GeoLift Market Selection (GEOLIFT)
==================================

.. currentmodule:: mlsynth

Overview
--------

Most estimators in ``mlsynth`` are *retrospective*: a treatment has happened and
we want its effect. ``GEOLIFT`` is *prospective* — a tool for **synthetic
experimental design** in geo-experiments. Before any ad spend, it answers:

   *Which* markets should be treated, for *how long*, so that a real lift would
   be **detectable**?

It is a faithful port of Meta's `GeoLift
<https://github.com/facebookincubator/GeoLift>`_ market-selection routine onto
the ``mlsynth`` Augmented-SCM machinery (Ben-Michael, Feller & Rothstein, 2021
[BMFR2021]_), with conformal inference (Chernozhukov, Wüthrich & Zhu, 2021
[CWZ2021]_) and the standardized design/effect result contract. Reach for it
when test markets must be chosen up front, you want a **minimum detectable
effect (MDE)** and power per candidate region plus the deployable synthetic
control, you may need to **force** markets in or out, and — once the experiment
runs — you want to **realize** the chosen design into an effect report.

Mathematical Formulation
------------------------

Setup and notation
~~~~~~~~~~~~~~~~~~~

There are :math:`N` markets :math:`\mathcal{N} \coloneqq \{1, \dots, N\}` and
:math:`T` periods :math:`t \in \mathcal{T} \coloneqq \{1, \dots, T\}`. The design
uses only the **pre-treatment** window
:math:`\mathcal{T}_1 \coloneqq \{t \in \mathcal{T} : t \le T_0\}` of length
:math:`T_0`; if post-treatment data exist they are sliced off (see
*Pre/post split*). The outcome of market :math:`j` at time :math:`t` is
:math:`y_{jt}`, with market series
:math:`\mathbf{y}_j = (y_{j1}, \dots, y_{jT})^\top \in \mathbb{R}^{T}`.

A **candidate test region** is a set :math:`\mathcal{S} \subseteq \mathcal{N}`
of :math:`k \coloneqq |\mathcal{S}|` markets (the ``treatment_size``). It plays
the role of the canonical treated unit through its **aggregate series**

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
a tractable shortlist by **correlation similarity**. On the pre-period, form the
Pearson correlation matrix :math:`\mathbf{P} = [\rho_{ij}] \in \mathbb{R}^{N \times N}`,

.. math::

   \rho_{ij} \coloneqq
   \frac{\sum_{t \in \mathcal{T}_1}(y_{it} - \bar{y}_i)(y_{jt} - \bar{y}_j)}
        {\sqrt{\sum_{t \in \mathcal{T}_1}(y_{it} - \bar{y}_i)^2}\;
         \sqrt{\sum_{t \in \mathcal{T}_1}(y_{jt} - \bar{y}_j)^2}},
   \qquad \bar{y}_i \coloneqq T_0^{-1}\!\!\sum_{t \in \mathcal{T}_1} y_{it}.

For each **anchor** :math:`i`, let :math:`\pi_i` order the other markets by
descending correlation,
:math:`\rho_{i,\pi_i(1)} \ge \rho_{i,\pi_i(2)} \ge \dots \ge \rho_{i,\pi_i(N-1)}`.
The **deterministic** nominee anchored at :math:`i` is that market plus its
:math:`k-1` nearest neighbours,

.. math::

   \mathcal{S}_i \coloneqq \{i\} \cup \bigl\{\pi_i(1), \dots, \pi_i(k-1)\bigr\},

and the shortlist is :math:`\{\mathcal{S}_i\}_{i \in \mathcal{N}}` deduplicated —
:math:`N` candidates instead of :math:`\binom{N}{k}`. The **stochastic**
("paired-jitter") variant replaces ranks :math:`1, \dots, k-1` by one draw from
each adjacent pair :math:`\{1,2\}, \{3,4\}, \dots`, exploring near-rank neighbours
(``run_stochastic``; ``stochastic_mode="global"`` is faithful to GeoLift,
``"per_anchor"`` draws independently per anchor).

**Forcing constraints.** Given a forced-in set
:math:`\mathcal{S}_{\mathrm{in}}` (``to_be_treated``) and a forbidden set
:math:`\mathcal{S}_{\mathrm{out}}` (``not_to_be_treated``,
:math:`\mathcal{S}_{\mathrm{in}} \cap \mathcal{S}_{\mathrm{out}} = \varnothing`),
nominees are drawn from the **free pool**
:math:`\mathcal{F} \coloneqq \mathcal{N} \setminus (\mathcal{S}_{\mathrm{in}} \cup
\mathcal{S}_{\mathrm{out}})` at size :math:`k - |\mathcal{S}_{\mathrm{in}}|` and
unioned with the forced-in set, so every candidate satisfies
:math:`\mathcal{S}_{\mathrm{in}} \subseteq \mathcal{S}` and
:math:`\mathcal{S} \cap \mathcal{S}_{\mathrm{out}} = \varnothing`.

Stage 2 — The synthetic control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a candidate :math:`\mathcal{S}`, the counterfactual is a weighted donor
combination. The default is the **Augmented SCM** [BMFR2021]_. Each period is
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

then ridge-augmented to close the residual pre-period imbalance,

.. math::

   \mathbf{w}^\ast = \mathbf{w}^{\mathrm{scm}} +
   \widetilde{\mathbf{Y}}_{0,\mathcal{T}_1}^{\top}
   \bigl(\widetilde{\mathbf{Y}}_{0,\mathcal{T}_1}\widetilde{\mathbf{Y}}_{0,\mathcal{T}_1}^{\top}
   + \lambda \mathbf{I}\bigr)^{+}
   \bigl(\widetilde{\mathbf{y}}^{\mathcal{S}}_{\mathcal{T}_1}
   - \widetilde{\mathbf{Y}}_{0,\mathcal{T}_1}\mathbf{w}^{\mathrm{scm}}\bigr),

with the penalty :math:`\lambda` chosen by leave-one-period-out cross-validation.
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

Pre-fit quality is the **scaled L2 imbalance** — the fitted pre-period imbalance
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

Power is estimated by **placebo-in-time** experiments carved from the end of the
pre-period. For a treatment **duration** :math:`\ell` (a ``durations`` entry) and
a **lookback placement** :math:`s \in \{1, \dots, L\}` (with
:math:`L =` ``lookback_window``), the pseudo-treatment window is the :math:`\ell`
periods ending :math:`s - 1` before :math:`T_0`,

.. math::

   \mathcal{T}^{(s,\ell)}_{2} \coloneqq
   \{\,T_0 - \ell - s + 2,\ \dots,\ T_0 - s + 1\,\},
   \qquad
   \mathcal{T}^{(s,\ell)}_{1} \coloneqq \{1, \dots, T_0 - \ell - s + 1\},

faithful to GeoLift's ``max_time - tp - sim + 2`` with :math:`T_0` the pre-period
end. The SCM is fit on :math:`\mathcal{T}^{(s,\ell)}_{1}`. For an **effect size**
:math:`\delta` (an ``effect_sizes`` entry) a known multiplicative lift is injected
on the pseudo-post block,
:math:`y^{\mathcal{S},(\delta)}_t = (1+\delta)\,y^{\mathcal{S}}_t` for
:math:`t \in \mathcal{T}^{(s,\ell)}_2`, which shifts the gap by
:math:`\delta\,y^{\mathcal{S}}_t` there.

Detection uses the **conformal** test [CWZ2021]_. With the post-block statistic

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
and are deterministic, ignoring :math:`n_s`). **Power** is the detection rate
across the :math:`L` lookback placements,

.. math::

   \beta(\mathcal{S}, \ell, \delta) \coloneqq
   \frac{1}{L}\sum_{s=1}^{L}
   \mathbf{1}\!\bigl\{\, p^{(s)}(\mathcal{S}, \ell, \delta) < \alpha \,\bigr\}.

.. admonition:: Fit-once, sweep-:math:`\delta` (an exact optimization)

   The injection touches only the post block, so the pre-period the
   cross-validation sees is identical across effect sizes; the CV-selected
   :math:`\lambda` is therefore the same for every :math:`\delta`. ``mlsynth``
   cross-validates **once** per :math:`(\mathcal{S}, \ell, s)` and reuses
   :math:`\lambda` across :math:`\delta` (augsynth's own behaviour). This is
   *provably identical* to GeoLift's per-:math:`\delta` refit — pinned by
   ``test_simulate_lookback_cv_once_equals_per_es_refit`` — at
   :math:`1/|\{\delta\}|` the cross-validation cost.

Stage 4 — MDE and the composite rank
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **minimum detectable effect** for a region/duration is the smallest-magnitude
effect whose power clears the threshold :math:`\beta_0` (``power_threshold``,
default :math:`0.8`),

.. math::

   \delta^\ast(\mathcal{S}, \ell) \coloneqq
   \operatorname*{arg\,min}_{\delta \,:\, \beta(\mathcal{S},\ell,\delta) \ge \beta_0}
   |\delta|,

with GeoLift's signed positive/negative tie rule. Writing
:math:`\widehat{\delta}` for the recovered lift at :math:`\delta^\ast` and the
**recovery error** :math:`\eta(\mathcal{S},\ell) \coloneqq |\widehat{\delta} -
\delta^\ast|`, the **composite rank** is the mean of three dense ranks
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
   is **ascending** (an MDE whose power sits *just* above :math:`\beta_0` is a
   tighter estimate of the threshold, so it ranks better), and the scaled L2
   imbalance :math:`\kappa` is **not** a ranking term — only :math:`\delta^\ast`,
   :math:`\beta`, and :math:`\eta` enter. Both are documented and one line to
   change.

Identifying assumptions
~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Pre-period synthesizability.** The aggregate :math:`\mathbf{y}^{\mathcal{S}}`
   lies in (or near) the span/convex hull of the donor pool over
   :math:`\mathcal{T}_1`. Quantified by :math:`\kappa(\mathcal{S})`: a low value
   certifies that the synthetic tracks the region, the prerequisite for a
   credible counterfactual.

   *Remark.* With ``how="sum"`` the target is :math:`k\times` donor scale and can
   sit outside the convex hull, inflating :math:`\kappa` toward 1; ``how="mean"``
   restores synthesizability. The choice is the user's.

2. **Exchangeability under the null.** The conformal test treats the residual
   path as exchangeable under :math:`H_0`: no effect, which the all-period refit
   underlying :math:`p` is designed to deliver [CWZ2021]_.

3. **Stationarity of the placebo windows.** Power from the lookback placements
   transports to the real experiment only if the pre-period dynamics resemble the
   experiment window — the usual SC stability assumption.

Inference and the realized design
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The design phase reports :math:`\delta^\ast` and :math:`\beta` per candidate.
When a ``post_col`` leaves a post window, :meth:`GEOLIFT.fit` **realizes** the
winning design *under the hood* — applying the winner's pre-period weights
:math:`\mathbf{w}^\ast` to the **full** panel and running conformal inference:
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
* ``post_col`` — a 0/1 column marking post-treatment periods; the design slices to
  :math:`\mathcal{T}_1`, so it is **identical** whether you pass the full
  post-treatment panel or a pre-only one (the "rerun after treatment"
  invariance). Different post lengths simply change :math:`T_0`.
* ``how`` (:math:`\operatorname{sum}` / :math:`\operatorname{mean}`),
  ``augment`` (``"ridge"`` / ``None``), ``alpha`` :math:`\alpha`,
  ``power_threshold`` :math:`\beta_0`, ``ns`` :math:`n_s`,
  ``run_stochastic`` / ``stochastic_mode``.
* ``conformal_type`` — the conformal permutation scheme, ``"iid"`` (default,
  matching GeoLift) or ``"block"`` (moving-block cyclic shifts for
  serially-dependent residuals; GeoLift's ``conformal_type = "block"`` option).

Scanning several ``durations`` yields an MDE *per duration*
(":math:`\ell = 7` detects 10%, but :math:`\ell = 14` is needed for 5%"):

.. code-block:: python

   res = GEOLIFT({..., "durations": [7, 14, 21]}).fit()
   res.power[["candidate", "duration", "mde", "power"]]   # one row per (S, l)

Plotting
~~~~~~~~

With ``display_graphs`` (default ``True``), :meth:`GEOLIFT.fit` plots the
recommended design in the mlsynth house style
(:func:`mlsynth.utils.plotting.mlsynth_style`): the **design phase** shows
:math:`\mathbf{y}^{\mathcal{S}}` vs :math:`\widehat{\mathbf{y}}^{\mathcal{S}}`
over :math:`\mathcal{T}_1`; the **post phase** (when the design was realized)
adds the conformal band and the per-period gap :math:`\tau_t` over
:math:`\mathcal{T}_2`, with the intervention line at :math:`T_0`. The standalone
helper :func:`mlsynth.utils.geolift_helpers.marketselect.plotter.plot_design`
re-draws from a result on demand.

Example: GeoLift's 40-Market Panel
----------------------------------

The package ships GeoLift's example panel
(``basedata/geolift_market_data.csv``): :math:`N = 40` markets over
:math:`T = 90` days. We design a :math:`k = 3` test region, then realize it over a
10-day **no-effect** post window (so the realized effect should be null).

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
   print("joint conformal p:", round(res.report.inference.p_value, 3))   # ~0.83 -> null

Without a ``post_col`` the same call returns a design-only result
(``res.report is None``); set ``display_graphs=False`` to suppress the plot.

The synthetic tracks the test markets, :math:`\widehat{\tau} \approx 0`, and the
joint conformal p-value is far from significant — the correct null over a placebo
post period.

Verification
------------

GeoLift's market-selection routine has no published empirical benchmark or Monte
Carlo table, so there is no Path-A/Path-B replication. ``GEOLIFT`` is a **faithful
port** of the methodology (Stages 1–4), validated end-to-end on GeoLift's own
example data, with each documented divergence (mean aggregation, the CV-once
optimization *proven exact*, the corrected per-anchor RNG) available as an opt-in,
tested swap.

.. [BMFR2021] Ben-Michael, E., Feller, A., & Rothstein, J. (2021). The Augmented
   Synthetic Control Method. *Journal of the American Statistical Association*.

.. [CWZ2021] Chernozhukov, V., Wüthrich, K., & Zhu, Y. (2021). An Exact and Robust
   Conformal Inference Method for Counterfactual and Synthetic Controls.
   *Journal of the American Statistical Association*.

Core API
--------

.. autoclass:: mlsynth.GEOLIFT
   :members: fit

.. autoclass:: mlsynth.utils.geolift_helpers.config.GeoLiftConfig
   :members:
