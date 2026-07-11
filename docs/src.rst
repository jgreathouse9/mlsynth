Synthetic Regressing Control (SRC)
==================================

.. currentmodule:: mlsynth

When to use SRC -- and when not to
----------------------------------

Synthetic Regressing Control (Zhu 2023) is for the case where the ordinary
synthetic control fits the treated unit's pre-treatment path poorly. The
canonical synthetic control restricts the donor weights to the unit simplex
(non-negative, summing to one), which keeps the counterfactual inside the
convex hull of the donors but cannot track a treated unit that sits near or
outside that hull -- the interpolation-bias / imperfect-pre-fit problem. SRC
relaxes that restriction in a disciplined way and is a good choice when:

* You have a single treated unit and a donor pool with real heterogeneity, and
  a simplex synthetic control leaves a visible pre-treatment gap.
* You are willing to let the counterfactual extrapolate a little beyond the
  donor hull, but want the amount of extrapolation regularised rather than
  unbounded (as an unconstrained regression would give).
* You want a deterministic, reproducible estimate. By default SRC's weights are
  pinned by the risk criterion, not by an Abadie predictor-weight (``V``) search,
  so there is no optimiser-seed dependence. (The paper's covariate + ``V``-search
  variant is available as a seeded opt-in; see below.)

Do not use SRC when:

* A simplex synthetic control already fits well and you want an interpretable
  convex weighting. Use canonical SCM (:doc:`vanillasc`) or :doc:`tssc`; SRC's
  combined coefficients can be negative and do not sum to one.
* You want an explicit bias--variance trade-off between matching and synthetic
  control chosen by cross-validation. That is :doc:`masc`.
* You want denoising of a low-rank donor matrix before weighting. Use
  :doc:`clustersc` (robust PCA / PCR) or :doc:`snn`.
* You need posterior uncertainty on the weights. Use :doc:`bvss`.
* There are multiple treated units, spillovers, or a continuous dose -- SRC
  encodes a single binary intervention on one unit.

Notation
--------

We use the synthetic-control canon. Let :math:`\mathcal{N} \coloneqq
\{1, \ldots, N\}` index the units, with :math:`j = 1` the treated unit and
:math:`\mathcal{N}_0 \coloneqq \mathcal{N} \setminus \{1\} = \{2, \ldots, N\}`
the donor pool of :math:`N_0 \coloneqq N - 1` controls. Time is
:math:`t \in \mathcal{T} \coloneqq \{1, \ldots, T\}`; the intervention takes
effect after :math:`T_0`, so the pre-period is :math:`\mathcal{T}_1 \coloneqq
\{t \in \mathcal{T} : t \le T_0\}` with :math:`|\mathcal{T}_1| = T_0` and the
post-period is :math:`\mathcal{T}_2 \coloneqq \{t \in \mathcal{T} : t > T_0\}`
with length :math:`T_2 \coloneqq |\mathcal{T}_2|`.

Notation bridge. Zhu (2023) indexes the treated unit as one of :math:`j = 1,
\ldots, J + 1` with :math:`J` controls, and writes the pre-period as
:math:`\mathcal{T}_0`; his :math:`J` is our :math:`N_0`, and his
:math:`\mathcal{T}_0` is our :math:`\mathcal{T}_1`. He names the method
Synthetic Regressing Control (SRC); mlsynth ships it as SRC.

Panel objects. Write the treated unit's pre-period outcome path as
:math:`\mathbf{y}_1 \coloneqq (y_{1t})_{t \in \mathcal{T}_1} \in
\mathbb{R}^{T_0}` and stack the donors' pre-period paths columnwise into the
donor matrix :math:`\mathbf{Y}_0 \coloneqq [\mathbf{y}_j]_{j \in \mathcal{N}_0}
\in \mathbb{R}^{T_0 \times N_0}`, one column
:math:`\mathbf{y}_j = (y_{jt})_{t \in \mathcal{T}_1} \in \mathbb{R}^{T_0}` per
donor. These :math:`n \coloneqq T_0` matching rows are the pre-period outcomes in
the default specification; the covariate variant (below) stacks standardized
covariate rows on top, giving an :math:`n \times N_0` matching matrix with
:math:`n > T_0`, and every formula below carries over with :math:`T_0` replaced
by that row count :math:`n`. Centering (demeaning) uses the projector
:math:`\mathbf{Q} \coloneqq \mathbf{I}_{T_0} - T_0^{-1}
\mathbf{1}\mathbf{1}^{\top}`, and :math:`\bar{y}_j \coloneqq T_0^{-1}
\sum_{t \in \mathcal{T}_1} y_{jt}` is donor :math:`j`'s pre-period mean, so
:math:`\mathbf{Q}\mathbf{y}_j = \mathbf{y}_j - \bar{y}_j \mathbf{1}` is the
centered column.

Unit regression. SRC first rescales each donor to the treated unit by a
*separate* demeaned univariate least-squares fit -- Zhu's "unit regression":

.. math::

   \widehat{\theta}_j
     \coloneqq \frac{(\mathbf{y}_j - \bar{y}_j \mathbf{1})^{\top}
                     (\mathbf{y}_1 - \bar{y}_1 \mathbf{1})}
                    {\lVert \mathbf{y}_j - \bar{y}_j \mathbf{1} \rVert_2^2},
   \qquad j \in \mathcal{N}_0.

The regressed (matched) control for donor :math:`j` is the centered, rescaled
column :math:`\widehat{\theta}_j (\mathbf{y}_j - \bar{y}_j \mathbf{1})`; because
:math:`\widehat{\theta}_j` is unconstrained it can already extrapolate. Collect
these columnwise into the regressed-control matrix

.. math::

   \widetilde{\mathbf{Y}}_0
     \coloneqq \bigl[\, \widehat{\theta}_j (\mathbf{y}_j - \bar{y}_j \mathbf{1})
                 \,\bigr]_{j \in \mathcal{N}_0}
     = (\mathbf{Q}\mathbf{Y}_0)\operatorname{diag}(\widehat{\boldsymbol{\theta}})
     \in \mathbb{R}^{T_0 \times N_0},

whose column :math:`j` is donor :math:`j`'s matched control (the object the
previous version of this page wrote :math:`\mathbf{E}`).

Synthesis. The matched controls are combined by box weights :math:`\mathbf{w}
\in \mathcal{C}_{\mathrm{box}} \coloneqq [0, 1]^{N_0} = \{\mathbf{w} \in
\mathbb{R}^{N_0} : 0 \le w_j \le 1\}` -- note there is *no* sum-to-one
constraint, which is what separates SRC from a simplex synthetic control (Zhu
writes this set :math:`\mathcal{H}_J`). With the level
:math:`\widehat{b}_0 \coloneqq \bar{y}_1 - \sum_{j \in \mathcal{N}_0}
\widehat{w}_j \widehat{\theta}_j \bar{y}_j` that restores the mean removed by
centering, the SRC counterfactual is

.. math::

   \widehat{y}_{1t}
     = \bar{y}_1 + \sum_{j \in \mathcal{N}_0}
        w_j\, \widehat{\theta}_j\, (y_{jt} - \bar{y}_j)
     = \widehat{b}_0 + \sum_{j \in \mathcal{N}_0}
        w_j \widehat{\theta}_j\, y_{jt},
   \qquad t \in \mathcal{T},

which estimates the no-intervention outcome :math:`y_{1t}^N`. The effective
coefficient on donor :math:`j` is :math:`\widehat{\theta}_j w_j`, which may be
negative. The per-period effect is the scalar :math:`\tau_t \coloneqq y_{1t} -
\widehat{y}_{1t}`, and the ATT is :math:`\widehat{\tau} \coloneqq T_2^{-1}
\sum_{t \in \mathcal{T}_2} \tau_t`.

Weights. The box weights minimise a Mallows :math:`C_p` criterion -- an unbiased
estimator of the synthetic estimator's squared risk -- written in the canon's
loss-plus-penalty form,

.. math::

   \widehat{\mathbf{w}}
     \in \operatorname*{argmin}_{\mathbf{w} \in \mathcal{C}_{\mathrm{box}}}
       \; \underbrace{\bigl\lVert \mathbf{Q}\mathbf{y}_1
                      - \widetilde{\mathbf{Y}}_0 \mathbf{w} \bigr\rVert_2^2
                     }_{\mathcal{L}(\mathbf{w})}
       + \underbrace{2\,\widehat{\sigma}^2 \lVert \mathbf{w} \rVert_1
                    }_{\mathcal{P}(\mathbf{w})},

where :math:`\lVert \mathbf{w} \rVert_1 = \sum_{j \in \mathcal{N}_0} w_j`
because :math:`\mathbf{w} \ge \mathbf{0}` on the box. The noise level
:math:`\widehat{\sigma}^2` is the plug-in residual variance of the full
(multivariate) demeaned regression of the treated path on the donor matrix,

.. math::

   \widehat{\sigma}^2
     \coloneqq \frac{1}{\max(T_0 - N_0,\, 1)}
       \bigl\lVert \mathbf{Q}\mathbf{y}_1
         - (\mathbf{Q}\mathbf{Y}_0)\,\widehat{\boldsymbol{\beta}}
       \bigr\rVert_2^2,
   \qquad
   \widehat{\boldsymbol{\beta}} \coloneqq (\mathbf{Q}\mathbf{Y}_0)^{+}\,
     \mathbf{Q}\mathbf{y}_1,

with :math:`(\cdot)^{+}` the Moore--Penrose pseudoinverse and the degrees of
freedom floored at one when :math:`N_0 \ge T_0` (a minimum-norm fit; a small
``ridge`` then keeps the synthesis QP strictly convex). This
:math:`\widehat{\sigma}^2` estimates the working-model noise level
:math:`\sigma^2` and enters the penalty as the per-donor complexity charge
:math:`2\widehat{\sigma}^2`, which shrinks weight toward zero unless a matched
control earns its place. That charge -- not a sum-to-one constraint -- is what
identifies :math:`\widehat{\mathbf{w}}`.

   A note on the paper's display formula. Zhu's eq. 19 writes
   :math:`\widehat{\sigma}^2 = \lVert \mathbf{Q}\mathbf{y}_1 -
   \mathbf{Q}\mathbf{Y}_0[\operatorname{diag}(\mathbf{Y}_0^{\top}
   \mathbf{Q}\mathbf{Y}_0)]^{-1}\mathbf{Y}_0^{\top}\mathbf{Q}\mathbf{y}_1
   \rVert_2^2`, which reduces to the residual of the *equal-weight* sum of all
   :math:`N_0` univariate fits, :math:`\lVert \mathbf{Q}\mathbf{y}_1 -
   \widetilde{\mathbf{Y}}_0\mathbf{1} \rVert_2^2`. Because that sum stacks
   :math:`N_0` controls that each already track the treated unit, it overshoots
   by roughly a factor of :math:`N_0`, so the quantity scales with
   :math:`N_0^2`, not with the noise level: in simulations with a known
   :math:`\sigma^2` it over-estimates by two orders of magnitude, and the
   resulting penalty is large enough to drive *every* weight to zero (a
   degenerate flat-at-the-pre-mean counterfactual) in both the paper's
   donors-fewer-than-periods regime and out of it. The author's reference
   ``Code_SMC.R`` does not use eq. 19; it uses the full-regression residual
   variance given above, which recovers the true :math:`\sigma^2` to within a
   few percent. mlsynth follows the reference implementation (cross-validated to
   :math:`< 2\text{e-}13`; see Verification), and reads the printed eq. 19 as an
   uncorrected typo rather than the estimator the paper's own non-degenerate
   Basque results were produced with.

Assumptions
-----------

#. Single treated unit, balanced panel. One unit is treated after
   :math:`T_0`; every unit is observed at every period.

   Remark. The setup boundary (:mod:`mlsynth.utils.src_helpers.setup`) enforces
   both, translating a violation to :class:`~mlsynth.exceptions.MlsynthDataError`.

#. At least two pre-treatment periods. The univariate matches and the risk
   criterion need pre-period variation; the plug-in :math:`\widehat{\sigma}^2`
   uses the full-model residual, so more pre-periods than donors gives it
   degrees of freedom.

   Remark. When donors are not fewer than matching rows
   (:math:`N_0 \ge T_0`, or :math:`N_0 \ge n` with stacked covariates) the
   residual degrees of freedom :math:`\max(T_0 - N_0, 1)` are floored at one and
   :math:`\widehat{\boldsymbol{\beta}}` is the minimum-norm (pseudoinverse) fit;
   a small ``ridge`` keeps the synthesis QP strictly convex. For a
   high-dimensional donor pool, screen first (``screen="sirs"``, below) or use
   :doc:`clustersc`.

#. Regularised extrapolation is acceptable. The combined coefficients can leave
   the simplex, so the counterfactual can extrapolate; the :math:`C_p` penalty
   bounds how much.

   Remark. If extrapolation is unacceptable for the application (you need a
   convex, interpretable weighting), SRC is the wrong tool -- use a simplex SCM.

Screening a wide donor pool
---------------------------

When the pool is not small relative to the pre-period (:math:`N_0 \ge T_0`, and
the paper recommends screening already at :math:`N_0 \ge 4T_0/5`), the unit
regression is ill-posed: the centered pre-period system :math:`\mathbf{Q}
\mathbf{y}_1 \approx \mathbf{Q}\mathbf{Y}_0\boldsymbol{\beta}` has more donor
coefficients than periods, so the full-model residual collapses,
:math:`\widehat{\sigma}^2 \to 0`, and the :math:`C_p` penalty switches off --
the box fit then interpolates the pre-period with no complexity charge. The
paper's Algorithm 2 handles this by screening the pool *before* Algorithm 1.
Set ``screen="sirs"``:

.. code-block:: python

   import pandas as pd
   from mlsynth import SRC

   df = pd.read_csv("basedata/basque_data.csv")
   df["treat"] = ((df["regionname"] == "Basque Country (Pais Vasco)")
                  & (df["year"] >= 1975)).astype(int)
   res = SRC({"df": df, "outcome": "gdpcap", "treat": "treat",
              "unitid": "regionname", "time": "year",
              "screen": "sirs", "display_graphs": False}).fit()

Donors are ranked by the sure-independent-ranking-and-screening (SIRS) marginal
utility of Zhu, Fan, Li & Zhu (2011) on the pre-period outcomes -- a model-free
measure of dependence between each donor and the treated path -- and the top
:math:`k \coloneqq \min\bigl(\lfloor T_0/\log(T_0/2)\rfloor,\, T_0 - 1\bigr)` are
kept (override with ``n_screen``). The :math:`T_0 - 1` cap guarantees the
screened pool is smaller than the pre-period, so the fit is well posed and
:math:`\widehat{\sigma}^2 > 0`. On the 1975-treated Basque panel
(:math:`T_0 = 20`, :math:`N_0 = 16`) screening keeps :math:`k = 8` donors and
lifts :math:`\widehat{\sigma}^2` by roughly two orders of magnitude, so the
penalty is active rather than switched off.

A shape-based alternative: ``screen="fpca"``. SIRS ranks donors by marginal
dependence; ``screen="fpca"`` instead selects an economically coherent peer
group by *clustering* the pre-period trajectories and keeping the donors in the
treated unit's cluster. It reuses :doc:`clustersc`'s own donor-clustering
routines -- functional PCA (cubic-B-spline smoothing then a PCA truncated at
95% variance) followed by silhouette-chosen k-means (Bayani 2021,
:func:`~mlsynth.utils.clustersc_helpers.rpca.fpca.compute_fpca_features` and
:func:`~mlsynth.utils.clustersc_helpers.rpca.clustering.assign_clusters`) -- so
the two estimators screen donors by the same principle. The cluster size is
data-driven (no ``n_screen``), and the selection is deterministic. On the
1975-treated Basque panel it keeps the developed / service regions
(Baleares, Cataluña, Madrid) in the Basque Country's cluster, where SIRS instead
ranks in Aragón; the two rules give genuinely different pools, and hence
different counterfactuals -- a reminder that donor selection is a modelling
choice, not an objective one.

Two cautions. Screening is a well-posedness and reproducibility device, not a
route to the paper's Basque weight cells: even a screened pool that contains
Cantabria still lands on the same Rioja/Madrid decomposition, because the
remaining freedom lives on the non-identified :math:`\mathbf{V}` manifold (see
below), not in the pool. And the SIRS statistic is scale-free by construction
(donor paths are standardised); the paper's printed Eq. 22 carries the inner
term as :math:`Y_{jt}`, which we read as the cited-form :math:`Y_{j\ell}`.

Inference and diagnostics
-------------------------

SRC returns a point estimate. The primary diagnostic is the pre-treatment fit
(``res.fit_diagnostics.rmse_pre``): SRC's reason for being is a tighter
pre-fit than a simplex SCM, so compare the two. The combined coefficients
``res.weights_vector`` (which may be negative) and the box weights
``res.box_weights`` are both exposed; a donor with a hard zero box weight is
genuinely dropped. For placebo inference, run the estimator on each control
region in turn and compare the treated gap to the placebo distribution (the
fit is deterministic, so placebos are exact).

Example
-------

.. code-block:: python

   import pandas as pd
   from mlsynth import SRC

   df = pd.read_csv("basedata/basque_data.csv")
   df["treat"] = ((df["regionname"] == "Basque Country (Pais Vasco)")
                  & (df["year"] >= 1970)).astype(int)

   res = SRC({
       "df": df, "outcome": "gdpcap", "treat": "treat",
       "unitid": "regionname", "time": "year",
       "display_graphs": False,
   }).fit()

   print(f"pre-period RMSE : {res.fit_diagnostics.rmse_pre:.3f}")
   print(f"mean post ATT   : {res.att:+.3f}")
   for u, c in sorted(res.donor_weights.items(), key=lambda kv: -abs(kv[1]))[:3]:
       print(f"  {u:<28s} {c:+.3f}")

This prints::

   pre-period RMSE : 0.048
   mean post ATT   : -0.858
     Murcia (Region de)           +0.625
     Madrid (Comunidad De)        +0.370
     Castilla Y Leon              +0.242

The Basque Country tracks its SRC counterfactual to a pre-period RMSE of about
0.048 (thousand-1986-USD per capita), then diverges steadily -- the familiar
Abadie-Gardeazabal shape of the economic cost of ETA terrorism -- recovered
here by the box-weighted matched controls rather than a simplex.

Note that this deterministic Algorithm 1 is *not* the specification behind the
paper's Basque table: it selects Murcia / Madrid / Castilla y León and a mean
ATT of about :math:`-0.86`, whereas the paper (Table 5 / Figure 1) reports
Rioja / Madrid / Cantabria and an ATT reaching :math:`\approx -1.9`. The
difference is the predictor-weight layer, covered next.

Reproducing the paper's covariate specification
-----------------------------------------------

The paper's Basque result uses Algorithm 3: covariate matching with an Abadie
predictor-weight optimisation over the matching rows. Stack the standardized
covariate rows on top of the outcome rows to form the :math:`n \times N_0`
matching matrix, and let :math:`\mathbf{V} \coloneqq \operatorname{diag}(v_1,
\ldots, v_n)` with :math:`v_i \ge 0` be the diagonal predictor weights on those
rows. Every quantity above is then computed on the :math:`\mathbf{V}^{1/2}`-
scaled rows -- the treated matching vector becomes :math:`\mathbf{V}^{1/2}
\mathbf{y}_1` and the donor matrix :math:`\mathbf{V}^{1/2}\mathbf{Y}_0` before
centering -- so :math:`\mathbf{V}` sets each matching row's relative importance
in the unit regression and the :math:`C_p` synthesis. Choosing :math:`\mathbf{V}`
is an explicit, seeded opt-in in mlsynth -- ``covariates`` with per-covariate
``covariate_windows``, the ``fit_window`` for the outcome rows, and
``v_search="de"``:

.. code-block:: python

   res = SRC({
       "df": df, "outcome": "gdpcap", "treat": "treat",
       "unitid": "regionname", "time": "year",
       "covariates": ["school.illit", "school.prim", "school.med", "school.high",
                      "invest", "sec.agriculture", "sec.energy", "sec.industry",
                      "sec.construction", "sec.services.venta",
                      "sec.services.nonventa", "popdens"],
       "covariate_windows": {"invest": (1964, 1969), "sec.industry": (1961, 1969)},  # ...
       "fit_window": (1960, 1969),          # Abadie's time.optimize.ssr
       "v_search": "de", "v_seed": 0,       # seeded global V search
       "display_graphs": False,
   }).fit()

This recovers the paper's donor structure -- Rioja dominant, then Madrid -- and
the Figure 1 ATT magnitude (:math:`\approx -1.4` mean, :math:`-2.4` by 1997).

A caveat stated plainly. The :math:`\mathbf{V}` optimum is *not identified*: a
manifold of :math:`\mathbf{V}` fits the pre-period equally well while disagreeing
out of sample, so the exact split among the top donors is seed-dependent (a
global search reproduces the paper's *average* placebo MSPE but not its
per-region cells). The search is seeded so a given call is reproducible, and it
is opt-in so the identified Algorithm 1 stays the default. Read the ``v_search``
weights as "the paper's specification, one draw from its non-identified
:math:`\mathbf{V}` manifold", not as identified quantities. When you want a
single reproducible weighting, use the default.

Verification
------------

The SRC weight computation is cross-validated against the author's reference R
implementation (``Code_SMC.R``): on the identical Basque matching matrix, the
per-donor :math:`\widehat{\theta}_j`, the box weights, the combined
coefficients, the intercept :math:`\widehat{b}_0` (``bias``) and
:math:`\widehat{\sigma}^2` all match the reference
``SMCV`` (whose synthesis QP is solved by ``solve.QP``) to ``< 2e-13``. The
active-set box QP in :mod:`mlsynth.utils.src_helpers.solver` reproduces
``solve.QP`` to machine precision while pinning the box bounds exactly. With the
paper's covariate specification and ``v_search="de"`` the estimator additionally
reproduces the Table 5 / Figure 1 donor structure and ATT magnitude (subject to
the :math:`\mathbf{V}` non-identification noted above). See the replication page
:doc:`replications/src` and the durable case ``benchmarks/cases/src_basque.py``.
The solver, weight computation, the covariate / ``V``-search paths, setup and
result contract are unit-tested (``mlsynth/tests/test_src.py``, full coverage).

Core API
--------

.. automodule:: mlsynth.estimators.src
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.SRCConfig
   :members:
   :undoc-members:

Result Containers
-----------------

``SRC.fit()`` returns a
:class:`~mlsynth.utils.src_helpers.structures.SRCResults` -- an
``EffectResult`` whose standardized sub-models carry the ATT, counterfactual,
gap and pre-RMSE, with the SRC-specific ``theta`` rescalings, box weights and
:math:`\widehat{\sigma}^2` on ``res.fit``. The prepared NumPy panel is exposed
as a :class:`~mlsynth.utils.src_helpers.structures.SRCInputs`.
