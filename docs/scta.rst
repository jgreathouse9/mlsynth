Synthetic Control with Temporal Aggregation (SCTA)
==================================================

.. currentmodule:: mlsynth

When to Use This Estimator
--------------------------

The synthetic control (SC) method of Abadie and co-authors [ABADIE2010]_
builds a counterfactual for one treated unit as a weighted average of donor
units that reproduces the treated unit's pre-treatment outcome path. When the
outcome is measured at high frequency -- monthly births, daily sales, weekly
visits -- a practitioner faces a choice that changes the answer: match
the donors on the raw high-frequency series, or first average it into coarser
intervals (say, yearly) and match on those?

Neither extreme is safe. Matching on the disaggregated series gives many
pre-periods to balance, but each one is a noisy realisation of the latent
factors, so the weights can overfit that noise -- a bias Abadie and
Vives-i-Bastida explicitly caution against. Matching on the aggregated series
averages the noise away (variance falls by a factor of the block length), but
it also throws away within-interval signal: if there is little long-run
variation left to learn the donor loadings from, aggregation can inflate the
bias instead of shrinking it.

SCTA [SunBenMichaelFellerTA]_ refuses the binary. It finds one set of donor
weights that jointly balances both the disaggregated high-frequency outcomes
and their temporal aggregates, trading the two off through a single knob
:math:`\nu`. Aggregation reduces noise in the balancing objective only to the
extent that long-run signal for the factors survives; the joint fit lets the
data keep whichever view balances better while never fully abandoning the
other.

Use SCTA when you have one treated unit, a high-frequency outcome, and a
genuine question about whether to aggregate -- typically because the
disaggregated fit looks like it is chasing seasonal or idiosyncratic noise, or
because the aggregated fit alone discards too much. It is the temporal sibling
of :doc:`scmo`: where SCMO adds an outcome dimension to pin the weights down,
SCTA adds an aggregation-level dimension.

Notation and the Stacked Design
-------------------------------

For each unit :math:`i` and each high-frequency period we observe an outcome,
grouped into :math:`T_0` pre-treatment intervals of :math:`K` observations
each (for example, :math:`K = 12` months per year). Write :math:`\dot{Y}_{itk}`
for the outcome of unit :math:`i` in interval :math:`t`, sub-period :math:`k`,
demeaned by that unit's pre-treatment average
:math:`\bar{Y}_{i\cdot\cdot}` -- the intercept shift of Doudchenko and Imbens.
The treated unit is :math:`i = 1`; the remaining units are donors. The
estimator is the demeaned weighting form

.. math::

   \hat{Y}_{1Tk}(0) \;=\; \bar{Y}_{1\cdot\cdot}
   \;+\; \sum_{W_i = 0} \gamma_i\, \dot{Y}_{iTk},

so only the weights :math:`\gamma` (on the simplex by default) must be chosen.

SCTA chooses them by stacking two balance targets into one matching design.
The disaggregated target is every pre-period value :math:`\dot{Y}_{itk}`; the
aggregated target is the block mean
:math:`\bar{\dot{Y}}_{it} = \tfrac{1}{K}\sum_{k} \dot{Y}_{itk}`. For each unit
the matching vector is the concatenation

.. math::

   \big[\, \underbrace{\bar{\dot{Y}}_{i1}, \ldots, \bar{\dot{Y}}_{i,T_0/K}}_{\text{aggregated blocks}}
   \;\big|\;
   \underbrace{\dot{Y}_{i11}, \ldots, \dot{Y}_{i T_0 K}}_{\text{disaggregated periods}} \,\big].

A fixed diagonal weight matrix :math:`\mathbf{V} = \operatorname{diag}(K\nu,
\ldots, K\nu, 1, \ldots, 1)` puts weight :math:`K\nu` on each aggregated row
and :math:`1` on each disaggregated row, and the weights solve the
:math:`\mathbf{V}`-weighted simplex least-squares problem

.. math::

   \min_{\gamma \ge 0,\, \sum_i \gamma_i = 1}
   \; (\mathbf{a} - \mathbf{B}\gamma)^{\top}\, \mathbf{V}\, (\mathbf{a} - \mathbf{B}\gamma),

where :math:`\mathbf{a}` and :math:`\mathbf{B}` are the stacked treated vector
and donor matrix. At :math:`\nu = 0` the aggregated rows drop out and SCTA is
the conventional disaggregated SC; larger :math:`\nu` shifts balance onto the
aggregates. mlsynth solves this at the true optimum with its active-set QP, so
the weights do not depend on solver idiosyncrasies.

Assumptions
-----------

1. Single treated unit, block-structured pre-period. Exactly one unit is
   treated, and the pre-period contains at least one whole block of :math:`K`
   high-frequency observations.

   Remark. The leading :math:`\lfloor T_0 / K \rfloor` complete blocks are
   aggregated; every disaggregated period is retained, so a ragged tail
   (:math:`T_0` not a multiple of :math:`K`) is simply kept disaggregated. The
   Texas application below has six whole years plus three spare months and is
   handled without special-casing.

2. Common latent factors. Outcomes under control follow a linear factor model
   with two-way fixed effects; donors and the treated unit share the time
   factors, and the factor variance-covariance is non-degenerate on the
   disaggregated series (:math:`\bar\xi^{\,\mathrm{dis}} > 0`), the usual
   no-weak-identification condition.

   Remark. Aggregation tightens the bias bound only when long-run signal
   survives -- formally when :math:`\sqrt{K}\,\bar\xi^{\,\mathrm{agg}} >
   \bar\xi^{\,\mathrm{dis}}`. If the aggregated factor variance is tiny
   (little long-run variation), aggregating can inflate the bias. The frontier
   diagnostic below is how you check which regime you are in.

3. Intercept-shifted weights. The counterfactual allows a level difference
   between the treated unit and its synthetic control (the demeaning), so only
   the shape of the donor combination is matched.

   Remark. This is intrinsic to the estimator form, not a tuning choice;
   ``demean=True`` is the default and matches the paper. Setting it off
   reverts to a raw simplex combination and is offered only for diagnostics.

Choosing nu and the Imbalance Frontier
--------------------------------------

The optimal :math:`\nu` depends on unknown factor-model quantities and is
infeasible to compute. The paper asks you to assess sensitivity instead of
trusting a single number, and reports three reference cases: aggregates only,
months only, and equal weight between them.

Equal weight is :math:`\nu = 1`. The :math:`\lfloor T_0/K \rfloor` aggregate
rows carry total weight :math:`\lfloor T_0/K \rfloor \cdot K\nu`, which is
:math:`T_0 \nu` when :math:`K` divides :math:`T_0`, against :math:`T_0` for the
disaggregated rows — so the two halves of the objective balance at
:math:`\nu = 1`. On the Texas panel that is 72 against 75, the slack being the
three-month tail that belongs to no whole block.

``nu`` defaults to :math:`0.5`, which puts the aggregates at just under half the
weight of the months. Earlier versions of this page called that the paper's
equal-weight heuristic. It is not: :math:`0.5` is where the equal-weight case
falls on the horizontal axis of the paper's Figure 3, which plots
:math:`\texttt{year\_wt} / (\texttt{year\_wt} + 1)`, and the knob value there is
:math:`\texttt{year\_wt} = 1`. Set ``nu=1.0`` for the paper's equal-weight fit.

Passing a ``frontier`` grid traces the imbalance frontier: for each
:math:`\nu` it reports the disaggregated and aggregated pre-treatment RMSE, the
two axes of Figure 1 in the paper. A good :math:`\nu` is one where both
imbalances are small; a frontier that collapses to one axis tells you
aggregation is buying (or costing) you signal.

Inference and Diagnostics
-------------------------

Each fit carries a conformal prediction interval and p-value for the average
post-treatment effect (the CWZ moving-block construction of
Chernozhukov-Wuethrich-Zhu, shared with :doc:`scmo`), controlled by
``conformal_alpha``. The fit diagnostics expose the disaggregated
pre-treatment RMSE; the frontier exposes both the disaggregated and aggregated
RMSE across :math:`\nu`. Donor weights are returned for inspection of the
comparison group.

Example
-------

.. code-block:: python

   import numpy as np
   import pandas as pd
   from mlsynth import SCTA

   # A monthly panel: 8 units, 16 months, treatment on u0 from month 12.
   rng = np.random.default_rng(0)
   f = np.linspace(0, 3, 16) + 0.4 * np.sin(np.linspace(0, 6, 16))
   rows = []
   for u in range(8):
       base = rng.normal(100, 8) + rng.normal(1, 0.3) * f
       for t in range(16):
           y = base[t] + (8.0 if (u == 0 and t >= 12) else 0.0)
           rows.append((f"u{u}", t, float(y), int(u == 0 and t >= 12)))
   df = pd.DataFrame(rows, columns=["unit", "time", "y", "treat"])

   config = {
       "df": df, "outcome": "y", "treat": "treat",
       "unitid": "unit", "time": "time",
       "block_length": 4,            # K = 4 high-frequency periods per block
       "nu": 1.0,                    # equal weight on aggregated and disaggregated
       "frontier": [0.0, 0.5, 1.0, 2.0],
   }
   results = SCTA(config).fit()

   print(results.effects.att)                 # average post-treatment effect
   print(results.inference.p_value)           # conformal p-value
   for point in results.frontier:             # the imbalance frontier
       print(point["nu"], point["rmse_dis"], point["rmse_agg"])

Ridge augmentation. Set ``augment="ridge"`` to add the bilevel
ridge-augmented correction (the Augmented SCM of Ben-Michael, Feller and
Rothstein) on top of the joint simplex fit, closing residual pre-treatment
imbalance at the cost of leaving the simplex. ``ridge_lambda`` fixes the
penalty; left unset it is chosen by cross-validation.

Verification
------------

SCTA reproduces the temporal-aggregation construction of the paper's Texas
SB8 study [BellStuartGemmill]_ and is cross-validated against ``augsynth`` 0.2.0,
the authors' own R package, on the authors' own panel — both of which ship with
the library (`benchmarks/cases/scta_texas_sb8.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/scta_texas_sb8.py>`_).

The two libraries spell the aggregation knob differently, and the difference is
a square. SCTA scales the matching rows by :math:`\sqrt{\mathbf{V}}`, so an
aggregate row's weight in the objective is :math:`K\nu`, as the paper writes it.
``augsynth`` scales the matching columns by :math:`\mathbf{V}` and solves with
the weight matrix set to the identity, which weights an aggregate row by
:math:`(K\nu)^2`. The knobs therefore map as
:math:`\nu = K \cdot \texttt{year\_wt}^2` — the paper's "Yearly + Monthly" fit
at ``year_wt = 1`` is :math:`\nu = 12`, not :math:`\nu = 1`. At the mapped
knob the two agree to 0.11 percent across the grid, and the residual is a second
convention difference in what the unit fixed effect averages over.

A second check needs no R: `benchmarks/cases/scta_ibex_xval.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/scta_ibex_xval.py>`_
rebuilds the paper's Section 2 design from the equations, solves it with
``cvxpy``/CLARABEL, and matches SCTA's ATT to :math:`5\times 10^{-12}` across the
:math:`\nu` grid and its ridge-augmented ATT to :math:`2.7\times 10^{-10}`. See
the dedicated page :doc:`replications/scta`, and :doc:`validation` for the
side-by-side tables.

Core API
--------

.. autoclass:: mlsynth.estimators.scta.SCTA
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
~~~~~~~~~~~~~

.. autoclass:: mlsynth.utils.scta_helpers.config.SCTAConfig
   :members:
   :undoc-members:
   :show-inheritance:

Results and Inputs
~~~~~~~~~~~~~~~~~~

.. automodule:: mlsynth.utils.scta_helpers.structures
   :members:
   :undoc-members:
   :show-inheritance:

Engine
~~~~~~

.. automodule:: mlsynth.utils.scta_helpers.setup
   :members:

.. automodule:: mlsynth.utils.scta_helpers.pipeline
   :members:
