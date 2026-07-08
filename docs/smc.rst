Synthetic Matching Control (SMC)
================================

.. currentmodule:: mlsynth

When to use SMC -- and when not to
----------------------------------

Synthetic Matching Control (Zhu 2023) is for the case where the ordinary
synthetic control fits the treated unit's pre-treatment path poorly. The
canonical synthetic control restricts the donor weights to the unit simplex
(non-negative, summing to one), which keeps the counterfactual inside the
convex hull of the donors but cannot track a treated unit that sits near or
outside that hull -- the interpolation-bias / imperfect-pre-fit problem. SMC
relaxes that restriction in a disciplined way and is a good choice when:

* You have a single treated unit and a donor pool with real heterogeneity, and
  a simplex synthetic control leaves a visible pre-treatment gap.
* You are willing to let the counterfactual extrapolate a little beyond the
  donor hull, but want the amount of extrapolation regularised rather than
  unbounded (as an unconstrained regression would give).
* You want a deterministic, reproducible estimate. By default SMC's weights are
  pinned by the risk criterion, not by an Abadie predictor-weight (``V``) search,
  so there is no optimiser-seed dependence. (The paper's covariate + ``V``-search
  variant is available as a seeded opt-in; see below.)

Do not use SMC when:

* A simplex synthetic control already fits well and you want an interpretable
  convex weighting. Use canonical SCM (:doc:`vanillasc`) or :doc:`tssc`; SMC's
  combined coefficients can be negative and do not sum to one.
* You want an explicit bias--variance trade-off between matching and synthetic
  control chosen by cross-validation. That is :doc:`masc`.
* You want denoising of a low-rank donor matrix before weighting. Use
  :doc:`clustersc` (robust PCA / PCR) or :doc:`snn`.
* You need posterior uncertainty on the weights. Use :doc:`bvss`.
* There are multiple treated units, spillovers, or a continuous dose -- SMC
  encodes a single binary intervention on one unit.

Notation
--------

We use the synthetic-control canon. Let :math:`j = 1` denote the treated unit
and :math:`\mathcal{N}_0 \coloneqq \{2, \ldots, J + 1\}` the donor pool. The
intervention takes effect after period :math:`T_0`. Write
:math:`\mathbf{y}_1 \in \mathbb{R}^{T_0}` for the treated unit's pre-treatment
outcome path and :math:`\mathbf{y}_j \in \mathbb{R}^{T_0}` for donor
:math:`j`'s. Optional covariates add matching rows stacked on top of the
pre-treatment outcomes; the stacked matrix has :math:`n` rows.

Unit matching. For each donor separately, a univariate ordinary least squares
regression rescales that donor to the treated unit,

.. math::

   \widehat{\theta}_j
     = \frac{\mathbf{y}_j^{\top} \mathbf{y}_1}{\mathbf{y}_j^{\top} \mathbf{y}_j},

giving a matched control :math:`\widehat{\theta}_j \mathbf{y}_j`. Because
:math:`\widehat{\theta}_j` is unconstrained, the matched control can already
extrapolate.

Synthesis. The matched controls are combined with weights :math:`\mathbf{w}` on
the box :math:`\mathcal{H} \coloneqq \{ w_j \in [0, 1] \}` -- note there is no
sum-to-one constraint. The SMC counterfactual is

.. math::

   \widehat{Y}_{1t}(0) = \sum_{j=2}^{J+1} w_j\, \widehat{\theta}_j\, Y_{jt},

so the effective coefficient on donor :math:`j` is
:math:`\widehat{\theta}_j w_j`, which may be negative.

Weights. The box weights minimise a Mallows / :math:`C_p` unbiased-risk
criterion (the estimator of the synthetic estimator's squared risk),

.. math::

   \mathcal{C}(\mathbf{w})
     = \bigl\lVert \mathbf{y}_1 - \mathbf{E}\, \mathbf{w} \bigr\rVert^2
       + 2 \widehat{\sigma}^2 \sum_{j=2}^{J+1} w_j,

where column :math:`j` of :math:`\mathbf{E}` is the matched control
:math:`\widehat{\theta}_j \mathbf{y}_j` and :math:`\widehat{\sigma}^2` is the
full-model residual variance. The penalty :math:`2 \widehat{\sigma}^2
\sum_j w_j` is what identifies :math:`\mathbf{w}`: it is a per-donor complexity
charge that shrinks weight toward zero unless a matched control earns its place.

Assumptions
-----------

#. Single treated unit, balanced panel. One unit is treated after
   :math:`T_0`; every unit is observed at every period.

   Remark. The setup boundary (:mod:`mlsynth.utils.smc_helpers.setup`) enforces
   both, translating a violation to :class:`~mlsynth.exceptions.MlsynthDataError`.

#. At least two pre-treatment periods. The univariate matches and the risk
   criterion need pre-period variation; the plug-in :math:`\widehat{\sigma}^2`
   uses the full-model residual, so more pre-periods than donors gives it
   degrees of freedom.

   Remark. When donors are not fewer than matching rows (:math:`J \ge n`) the
   residual degrees of freedom are floored at one and a minimum-norm fit is
   used; a small ``ridge`` keeps the synthesis QP well posed. For a
   high-dimensional donor pool, screen first (see the paper's SIRS extension)
   or use :doc:`clustersc`.

#. Regularised extrapolation is acceptable. The combined coefficients can leave
   the simplex, so the counterfactual can extrapolate; the :math:`C_p` penalty
   bounds how much.

   Remark. If extrapolation is unacceptable for the application (you need a
   convex, interpretable weighting), SMC is the wrong tool -- use a simplex SCM.

Inference and diagnostics
-------------------------

SMC returns a point estimate. The primary diagnostic is the pre-treatment fit
(``res.fit_diagnostics.rmse_pre``): SMC's reason for being is a tighter
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
   from mlsynth import SMC

   df = pd.read_csv("basedata/basque_data.csv")
   df["treat"] = ((df["regionname"] == "Basque Country (Pais Vasco)")
                  & (df["year"] >= 1970)).astype(int)

   res = SMC({
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

The Basque Country tracks its SMC counterfactual to a pre-period RMSE of about
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
predictor-weight (:math:`\mathbf{V}`) optimisation over the matching rows. That
is an explicit, seeded opt-in in mlsynth -- ``covariates`` with per-covariate
``covariate_windows``, the ``fit_window`` for the outcome rows, and
``v_search="de"``:

.. code-block:: python

   res = SMC({
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

The SMC weight computation is cross-validated against the author's reference R
implementation (``Code_SMC.R``): on the identical Basque matching matrix, the
per-donor :math:`\widehat{\theta}_j`, the box weights, the combined
coefficients, ``bias`` and :math:`\widehat{\sigma}^2` all match the reference
``SMCV`` (whose synthesis QP is solved by ``solve.QP``) to ``< 2e-13``. The
active-set box QP in :mod:`mlsynth.utils.smc_helpers.solver` reproduces
``solve.QP`` to machine precision while pinning the box bounds exactly. With the
paper's covariate specification and ``v_search="de"`` the estimator additionally
reproduces the Table 5 / Figure 1 donor structure and ATT magnitude (subject to
the :math:`\mathbf{V}` non-identification noted above). See the replication page
:doc:`replications/smc` and the durable case ``benchmarks/cases/smc_basque.py``.
The solver, weight computation, the covariate / ``V``-search paths, setup and
result contract are unit-tested (``mlsynth/tests/test_smc.py``, full coverage).

Core API
--------

.. automodule:: mlsynth.estimators.smc
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.SMCConfig
   :members:
   :undoc-members:

Result Containers
-----------------

``SMC.fit()`` returns a
:class:`~mlsynth.utils.smc_helpers.structures.SMCResults` -- an
``EffectResult`` whose standardized sub-models carry the ATT, counterfactual,
gap and pre-RMSE, with the SMC-specific ``theta`` rescalings, box weights and
:math:`\widehat{\sigma}^2` on ``res.fit``. The prepared NumPy panel is exposed
as a :class:`~mlsynth.utils.smc_helpers.structures.SMCInputs`.
