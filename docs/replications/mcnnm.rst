.. _replication-mcnnm:

MCNNM — Matrix Completion with Nuclear-Norm Minimization (Athey et al. 2021)
============================================================================

:Estimator: :doc:`../mcnnm` — :class:`mlsynth.MCNNM`
:Source: Athey, S., Bayati, M., Doudchenko, N., Imbens, G., & Khosravi, K.
   (2021), *"Matrix Completion Methods for Causal Panel Data Models,"* Journal
   of the American Statistical Association 116(536):1716-1730.
:Replication type: **Path A** — Proposition 99 empirical, with a
   **cross-validation** against the authors' own ``MCPanel`` R package.
:Status: **Fully verified** — estimand reproduced and matched against the
   authors' MC-NNM implementation.

Validation strategy
-------------------

MC-NNM imputes the treated post-period cells of the outcome matrix as missing
entries via low-rank completion with two-way fixed effects. Athey et al. report
a Proposition 99 effect of roughly **-20** packs per capita. mlsynth recovers
that estimand, and we cross-validate against a live run of the authors' own
``MCPanel`` package (``mcnnm_cv``) on the same matrix.

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

Cross-validation against the authors' ``MCPanel`` R
---------------------------------------------------

We cross-validate against the method's own authors' code -- the
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
property of the shared-threshold engine, not the CV-selected fit.

.. list-table::
   :header-rows: 1
   :widths: 40 28 28

   * - Quantity
     - mlsynth
     - MCPanel R / JASA
   * - ATT (packs/capita)
     - -19.83
     - -19.98 / :math:`\approx` -20
   * - CA counterfactual path (RMSE)
     - \—
     - 0.47

Durable check
-------------

The reference is pinned under ``benchmarks/reference/mcnnm_prop99/`` (R 4.3.3,
``MCPanel`` commit 6b2706f, ``set.seed(1)``, data checksum), captured by its
``reference.R``. The benchmark reads the frozen values (no R at test time)::

   python benchmarks/run_benchmarks.py --case mcnnm_prop99

It asserts the ATT matches ``MCPanel`` to within :math:`0.4` (observed
:math:`0.15`) and the California counterfactual path to RMSE :math:`0.9`
(observed :math:`0.47`).

References
----------

Athey, S., Bayati, M., Doudchenko, N., Imbens, G., & Khosravi, K. (2021).
"Matrix Completion Methods for Causal Panel Data Models." *Journal of the
American Statistical Association* 116(536):1716-1730.
