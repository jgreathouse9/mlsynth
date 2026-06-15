.. _replication-mcnnm:

MCNNM — Matrix Completion with Nuclear-Norm Minimization (Athey et al. 2021)
============================================================================

:Estimator: :doc:`../mcnnm` — :class:`mlsynth.MCNNM`
:Source: Athey, S., Bayati, M., Doudchenko, N., Imbens, G., & Khosravi, K.
   (2021), *"Matrix Completion Methods for Causal Panel Data Models,"* Journal
   of the American Statistical Association 116(536):1716-1730.
:Replication type: **Path A** — Proposition 99 empirical, with a
   **cross-validation** against the ``causaltensor`` reference implementation.
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

References
----------

Athey, S., Bayati, M., Doudchenko, N., Imbens, G., & Khosravi, K. (2021).
"Matrix Completion Methods for Causal Panel Data Models." *Journal of the
American Statistical Association* 116(536):1716-1730.
