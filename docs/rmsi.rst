Robust Matrix estimation with Side Information (RMSI)
=====================================================

.. currentmodule:: mlsynth

When to Use This Estimator
--------------------------

``RMSI`` implements the side-information matrix estimator of Agarwal, Choi and
Yuan [RMSI]_. It is a causal-panel matrix-completion method -- like
:doc:`mcnnm`, it imputes the treated units' missing counterfactual cells -- but
it exploits covariates on both margins of the panel: unit-level (row)
characteristics :math:`X` and time-level (column) characteristics :math:`Z`.

Its robustness comes from decomposing the target matrix into four complementary
pieces and estimating each separately:

.. math::

   M = \underbrace{G_1(X) Q_1(Z)^\top}_{M_1:\ \text{both margins}}
     + \underbrace{G_2(X) V_1^\top}_{M_2:\ \text{row-driven}}
     + \underbrace{W_1 Q_2(Z)^\top}_{M_3:\ \text{column-driven}}
     + \underbrace{W_2 V_2^\top}_{M_4:\ \text{residual low-rank}} .

Unlike inductive matrix completion (which forces an exact low-rank linear
covariate-interaction term and no genuine noise component), this decomposition
accommodates nonlinear covariate effects, a part explained by only one
margin, and a part explained by neither -- and it degrades gracefully when
the covariates are uninformative (it then reduces to a de-meaned low-rank
completion). Reach for ``RMSI`` when you have a block-adoption causal panel,
informative unit and/or time covariates, and you expect those covariates to
carry signal the raw outcome matrix alone would estimate noisily.

Do not use RMSI when
~~~~~~~~~~~~~~~~~~~~

* Adoption is staggered (treated units switch on at different times). RMSI
  assumes a block design; use :doc:`mcnnm`, :doc:`ssc`, :doc:`sdid`, or
  :doc:`ppscm`.
* You have no covariates and no reason to expect one margin's structure to
  help. With no side information RMSI reduces to a low-rank completion, so
  :doc:`mcnnm` (purpose-built, with inference) is the more direct choice.
* An interpretable donor-weight story is the deliverable -- RMSI imputes a
  matrix, not a sparse convex combination; use :doc:`tssc`/:doc:`scmo`.
* Spillovers contaminate the controls (SUTVA fails) -- use :doc:`spsydid`
  or :doc:`spillsynth`.

The estimator
~~~~~~~~~~~~~

*Algorithm 1 (fully observed).* With polynomial sieve bases of the covariates
and projectors :math:`P_X, P_Z` onto them, the component explained by both
margins is :math:`\widehat M_1 = P_X Y P_Z`, and the other three are
singular-value soft-thresholds of the projected residuals (the penalised least
squares :math:`\arg\min_A \lVert B - A\rVert_F^2 + \nu\lVert A\rVert_*` has the
closed form :math:`\operatorname{svt}(B, \nu/2)`):

.. math::

   \widehat M_2 = \operatorname{svt}\!\big(P_X Y (I - P_Z),\ \nu_2/2\big),\quad
   \widehat M_3 = \operatorname{svt}\!\big((I - P_X) Y P_Z,\ \nu_3/2\big),\quad
   \widehat M_4 = \operatorname{svt}\!\big((I - P_X) Y (I - P_Z),\ \nu_4/2\big),

with :math:`\nu_2 = C_2\sqrt{T}`, :math:`\nu_3 = C_3\sqrt{N}`,
:math:`\nu_4 = C_4(\sqrt{N}+\sqrt{T})`. The estimate is
:math:`\widehat M = \widehat M_1 + \widehat M_2 + \widehat M_3 + \widehat M_4`
-- only projections and SVDs, no iterative solver.

*Algorithm 3 (block-missing causal).* The treated post-treatment cells form a
missing block. RMSI applies Algorithm 1 to the fully observed tall submatrix
(all units, pre-treatment periods) and wide submatrix (control units, all
periods), then recombines their singular subspaces,
:math:`\widehat M = \widehat U_{\text{tall}}\, \widehat H\, \widehat D_{\text{wide}} \widehat V_{\text{wide}}^\top`,
where :math:`\widehat H` rotates the wide left-singular vectors onto the tall
ones over the control rows. The ATT is the observed minus the imputed outcome
over the treated cells. ``RMSI`` targets the block (common adoption time)
setting.

Side information
~~~~~~~~~~~~~~~~

Pass the covariate columns through the config: ``unit_covariates`` are columns
(approximately) constant within a unit -- averaged per unit to form the row
feature matrix :math:`X` -- and ``time_covariates`` are columns constant within
a period -- averaged per period to form :math:`Z`. Either may be empty.

Example
-------

A block panel of forty units (eight treated at period 20) whose untreated
outcomes are driven by nonlinear functions of two unit covariates and two time
covariates plus a residual low-rank part, with a constant effect of ``+5``:

.. code-block:: python

   from mlsynth import RMSI
   from mlsynth.utils.rmsi_helpers.simulation import simulate_rmsi_panel

   df = simulate_rmsi_panel(
       n_units=40, n_treated=8, T0=20, n_post=11, att=5.0, seed=0,
   )

   res = RMSI({
       "df": df, "outcome": "Y", "treat": "treated",
       "unitid": "unit", "time": "time",
       "unit_covariates": ["x0", "x1"],   # row side information
       "time_covariates": ["z0", "z1"],   # column side information
       "sieve_order": 2,
       "display_graphs": True,            # observed vs. imputed counterfactual
   }).fit()

   print(f"ATT (true 5.0) = {res.att:+.3f}   [rank {res.rank}]")

Replication
-----------

Both of the paper's empirical pieces are reproduced through the public API.

Path A -- Proposition 99 (Section 5.2). Using the Abadie et al. (2010)
tobacco panel shipped at
`basedata/P99data.csv <https://raw.githubusercontent.com/jgreathouse9/mlsynth/main/basedata/P99data.csv>`_,
``RMSI`` treats California from 1989 and uses the state-level Abadie predictors
as unit covariates and the year-average retail price as the time covariate:

.. code-block:: python

   from mlsynth.utils.rmsi_helpers.replication import replicate_prop99

   # downloads basedata/P99data.csv by default; pass a path/DataFrame to override
   res = replicate_prop99(rank=3)
   # -> California Proposition 99 ATT ~ -21 packs/capita (1989 ~ -7, 2000 ~ -32)

The estimated ATT of about :math:`-21` packs per capita is consistent with
Abadie, Diamond & Hainmueller [ABADIE2010]_ and with the :doc:`mcnnm` / :doc:`snn`
estimates elsewhere in ``mlsynth``. The equivalent explicit call:

.. code-block:: python

   import pandas as pd
   from mlsynth import RMSI

   url = ("https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
          "main/basedata/P99data.csv")
   d = pd.read_csv(url)
   for c in ["lnincome", "beer", "age15to24", "retprice"]:
       d[c] = d.groupby("state")[c].transform(lambda s: s.fillna(s.mean()))
       d[c] = d[c].fillna(d[c].mean())
   d["treated"] = ((d["state"] == "California") & (d["year"] >= 1989)).astype(int)

   res = RMSI({"df": d, "outcome": "cigsale", "treat": "treated",
               "unitid": "state", "time": "year",
               "unit_covariates": ["lnincome", "beer", "age15to24", "retprice"],
               "time_covariates": ["retprice"], "rank": 3,
               "display_graphs": False}).fit()
   print(res.att)

Path B -- synthetic Monte Carlo (Section 5.1). The paper's four-component
DGP under the block-missing (MNAR) pattern; RMSI imputes the missing block at a
lower average mean-squared error than the no-side-information baseline:

.. code-block:: python

   from mlsynth.utils.rmsi_helpers.replication import (
       run_rmsi_simulation, RMSISimConfig, PAPER,
   )

   # quick, reduced-count preset (use PAPER for the full N=T=400, 100-rep study)
   out = run_rmsi_simulation(RMSISimConfig(N=120, T=120, N0=60, T0=60,
                                           J=5, n_reps=10))
   # -> AMSE side-info < AMSE no-side-info (side information lowers the error)

Verification
------------

.. note::

   Path B (synthetic). On the paper's four-component MNAR DGP
   (:func:`~mlsynth.utils.rmsi_helpers.replication.simulate_rmsi_dgp`), RMSI's
   block imputation achieves a lower missing-block AMSE than the
   no-side-information baseline -- reproducing the paper's central finding that
   leveraging side information improves imputation accuracy.

   Path A (Proposition 99). ``replicate_prop99`` recovers a California
   Proposition 99 ATT of about :math:`-21` packs per capita (widening toward
   :math:`-32` by 2000), matching the Abadie-Diamond-Hainmueller [ABADIE2010]_
   baseline and the other ``mlsynth`` estimators on the same panel.

Core API
--------

.. automodule:: mlsynth.estimators.rmsi
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.RMSIConfig
   :members:
   :undoc-members:

Result Containers
-----------------

``RMSI.fit()`` returns a
:class:`~mlsynth.utils.rmsi_helpers.structures.RMSIResults`: the ATT, the imputed
counterfactual matrix ``counterfactual`` and per-cell ``effects``,
``att_by_period``, the cross-treated-unit observed/imputed means the plotter
draws, the factor ``rank`` used, and metadata.

.. automodule:: mlsynth.utils.rmsi_helpers.structures
   :members:
   :undoc-members:
   :show-inheritance:

Helper Modules
--------------

The numerical core: sieve bases, projections, singular-value soft-thresholding,
Algorithm 1 (four-component) and Algorithm 3 (block-missing recombination).

.. automodule:: mlsynth.utils.rmsi_helpers.core
   :members:
   :undoc-members:

Panel + side-information ingestion (block design enforced).

.. automodule:: mlsynth.utils.rmsi_helpers.setup
   :members:
   :undoc-members:

Run loop: Algorithm 3, the ATT and per-period effects, and the imputed series.

.. automodule:: mlsynth.utils.rmsi_helpers.pipeline
   :members:
   :undoc-members:

Block causal DGP with side information for examples and tests.

.. automodule:: mlsynth.utils.rmsi_helpers.simulation
   :members:
   :undoc-members:

Path-A (Proposition 99) and Path-B (synthetic Monte-Carlo) replications.

.. automodule:: mlsynth.utils.rmsi_helpers.replication
   :members:
   :undoc-members:
