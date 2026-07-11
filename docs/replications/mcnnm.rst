.. _replication-mcnnm:

MCNNM — Matrix Completion with Nuclear-Norm Minimization (Athey et al. 2021)
============================================================================

:Estimator: :doc:`../mcnnm` — :class:`mlsynth.MCNNM`
:Source: Athey, S., Bayati, M., Doudchenko, N., Imbens, G., & Khosravi, K.
   (2021), *"Matrix Completion Methods for Causal Panel Data Models,"* Journal
   of the American Statistical Association 116(536):1716-1730.
:Replication type: **Path A** — Proposition 99 empirical, with a
   **cross-validation** against both the authors' own ``MCPanel`` R package and
   the independent ``causaltensor`` reference implementation.
:Status: **Fully verified** — estimand reproduced and matched against an
   independent MC-NNM implementation.

Validation strategy
-------------------

MC-NNM imputes the treated post-period cells of the outcome matrix as missing
entries via low-rank completion with two-way fixed effects. Athey et al. report
a Proposition 99 effect of roughly **-20** packs per capita. mlsynth recovers
that estimand, and we cross-validate against
``causaltensor.MC_NNM_with_cross_validation`` on the same matrix.

Why this is an estimand-level cross-check
-----------------------------------------

Both implementations solve the same SOFT-IMPUTE objective (Athey et al. 2021,
eq. 4.3), but they differ in the two-way fixed-effect sub-solver:

* **mlsynth** fits the unit/time effects on the **observed cells only** — the
  Athey et al. observed-set :math:`\mathcal{O}` convention.
* **causaltensor** fits them on the full :math:`O - M` matrix.

With each library choosing its own regulariser by cross-validation, the two
completed matrices are therefore close but not bit-identical. We accordingly
cross-validate the **ATT** — the public estimand surfaced by ``res.att`` — not
the raw low-rank fit.

Path A — Proposition 99
-----------------------

.. code-block:: python

   import pandas as pd
   from mlsynth import MCNNM

   df = pd.read_csv("basedata/smoking_data.csv")
   df["treat"] = df["Proposition 99"].astype(int)
   res = MCNNM({"df": df[["state", "year", "cigsale", "treat"]],
                "outcome": "cigsale", "treat": "treat",
                "unitid": "state", "time": "year",
                "display_graphs": False}).fit()
   res.att        # -19.83

Cross-validation against ``causaltensor``
-----------------------------------------

.. code-block:: python

   import numpy as np, causaltensor as ct

   wide = df.pivot(index="state", columns="year", values="cigsale").sort_index()
   states, years = wide.index.tolist(), wide.columns.tolist()
   O = wide.values.astype(float)
   ti, sc = states.index("California"), years.index(1989)
   Omega = np.ones_like(O); Omega[ti, sc:] = 0   # 1 = observed, 0 = missing
   _, _, _, tau = ct.MC_NNM_with_cross_validation(O, Omega)
   tau            # -20.27

.. list-table::
   :header-rows: 1
   :widths: 34 22 22 22

   * - Quantity
     - mlsynth
     - causaltensor
     - JASA headline
   * - ATT (packs/capita)
     - -19.83
     - -20.27
     - :math:`\approx` -20

The two CV-selected ATTs agree to :math:`|\Delta| = 0.44` packs — about 2% of
the estimand — with the residual attributable to the documented FE-solver
difference plus the two independent cross-validation grids.

Durable check
-------------

The benchmark lives in ``benchmarks/cases/mcnnm_prop99.py`` and runs in the
default suite (skipping gracefully if ``causaltensor`` is absent)::

   pip install causaltensor
   python benchmarks/run_benchmarks.py --case mcnnm_prop99

It asserts the ATT lands near the published -20 (tol 1.5) and matches
``causaltensor`` to within 1.0 pack.

Cross-validation against the authors' ``MCPanel`` R
---------------------------------------------------

We additionally cross-validate against the method's own authors' code -- the
``susanathey/MCPanel`` R package -- run live on the identical outcome matrix.
This exercise also pins down *where* two faithful MC-NNM implementations can and
cannot agree, which is instructive for the estimator generally.

At a *matched* singular-value threshold the two engines are effectively
identical: they reconstruct the observed cells to RMSE :math:`\approx 3\times
10^{-3}` with the same singular spectrum. The entire disagreement lives in the
imputed treated block -- the counterfactual -- because that is the extrapolated
quantity, and nuclear-norm completion of the held-out cells is threshold
sensitive. On top of that, each library selects its own regulariser: mlsynth by
K-fold partition on a grand-mean-demeaned spectrum grid, MCPanel by Bernoulli
80/20 folds on a :math:`2\sigma_{\max}(P_\Omega M)/|\Omega|`-scaled grid with an
explicit :math:`\lambda = 0` rung. The two therefore land on different
penalties.

Under each side's own default cross-validation, ``mcnnm_cv`` returns
:math:`-19.98` and mlsynth :math:`-19.83` -- the ATT agrees to :math:`0.15`
packs and the California post-treatment counterfactual path to RMSE
:math:`0.47` (under one pack). This is the honest end-to-end agreement for an
estimator whose target is an extrapolated block; the tight cell match is a
property of the shared-threshold engine, not the CV-selected fit. The reference
is pinned under ``benchmarks/reference/mcnnm_prop99_mcpanel/`` (R 4.3.3,
``MCPanel`` commit 6b2706f, ``set.seed(1)``, data checksum) and the durable case
is ``benchmarks/cases/mcnnm_prop99_mcpanel.py``::

   python benchmarks/run_benchmarks.py --case mcnnm_prop99_mcpanel

References
----------

Athey, S., Bayati, M., Doudchenko, N., Imbens, G., & Khosravi, K. (2021).
"Matrix Completion Methods for Causal Panel Data Models." *Journal of the
American Statistical Association* 116(536):1716-1730.
