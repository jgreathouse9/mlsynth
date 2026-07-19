.. _replication-cscipca:

CSC-IPCA — Counterfactual and Synthetic Control with Instrumented PCA (Wang 2024)
=================================================================================

:Estimator: :doc:`../cscipca` — :class:`mlsynth.CSCIPCA`
:Source: Wang, C. (2024), *"Counterfactual and Synthetic Control Method:
   Causal Inference with Instrumented Principal Component Analysis"*
   [Wang2024]_.
:Reference implementation: ``CongWang141/JMP``, ``src/csc_ipca.py``.
:Replication type: Path A — the paper's Brexit / UK foreign-direct-investment
   application, reproduced on the author's data; supported by a cross-validation
   against the reference core and the Path-B Monte Carlo.
:Status: Verified — the reported per-year ATT path is matched to the second
   decimal, and the dataprep-fed port matches the reference counterfactual to
   machine precision.

Path A — Brexit and UK foreign direct investment
------------------------------------------------

Wang's empirical study estimates the causal effect of the 2016 Brexit
referendum on UK foreign-direct-investment (FDI) net inflows, using the UK as
the treated unit (from 2017) against a pool of OECD controls, with nine
macroeconomic covariates instrumenting the factor loadings and :math:`K = 2`
factors. The panel ships as ``basedata/fdi_oecd_brexit.csv`` (30 countries,
1995-2022), processed from the author's WDI export exactly as the
``test7_empirical_study`` notebook does — restrict to OECD members, log GDP,
GDP per capita and population, drop countries with any missing FDI, and drop
"extreme" countries whose FDI-to-GDP ratio exceeds :math:`\pm 25\%` in any year.

.. code-block:: python

   import pandas as pd
   from mlsynth import CSCIPCA

   df = pd.read_csv("basedata/fdi_oecd_brexit.csv")
   covs = ["log_gdp", "log_gdp_percap", "import_to_gdp", "export_to_gdp",
           "inflation_gdp_deflator", "gross_capital_forma_gdp", "unemployment",
           "employment_15", "log_population"]

   res = CSCIPCA({"df": df, "outcome": "fdi", "treat": "treated",
                  "unitid": "country", "time": "year", "covariates": covs,
                  "n_factors": 2, "display_graphs": False}).fit()

The reproduced per-year ATT for the first three post-treatment years:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Year
     - CSC-IPCA (mlsynth)
     - Wang (2024)
   * - 2017
     - :math:`-7.76`
     - :math:`-7.8`
   * - 2018
     - :math:`-12.90`
     - :math:`-12.9`
   * - 2019
     - :math:`-18.34`
     - :math:`-18.3`

A cell-by-cell match to the paper's headline numbers: FDI net inflows to the UK
fall short of their instrumented-factor counterfactual by roughly 8, 13 and 18
percentage points of GDP in the three years after the referendum, and the
90% conformal band over these years is entirely negative. The confidence band
widens after 2019 (the paper flags the same instability, driven by the
volatility of the FDI series). The durable case is
``benchmarks/cases/cscipca_brexit.py``.

Cross-validation — reference core through dataprep
--------------------------------------------------

To confirm the ingestion itself — in particular the time-varying covariate cube
that ``dataprep``'s per-unit-mean covariate path cannot supply — the same
simulated panel is fed through mlsynth's :func:`dataprep`-based path and through
the authors' own pandas-pivot path, running the identical alternating-least-
squares core. On a seeded draw of the paper's eq-13 DGP (:math:`K = 3`,
:math:`L = 10`, five treated units, :math:`T_0 = 20`) the two agree on the
rotation-invariant quantities to machine precision:

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - Quantity
     - max :math:`|\Delta|` (mlsynth vs reference)
   * - counterfactual :math:`\widehat Y(0)`
     - :math:`9.9 \times 10^{-13}`
   * - ATT path
     - :math:`2.4 \times 10^{-13}`

The two paths order the panel units differently, but the covariate cube is
pivoted in the same order the outcomes are fed and the ATT is invariant to the
factor rotation, so the comparison holds exactly.

Path B — the observed-covariate-share finding
---------------------------------------------

The paper's Table 1 shows the bias falling as the proportion :math:`\alpha` of
observed covariates rises, dropping to near-zero once all covariates are seen.
Reproduced with a single treated unit (mlsynth's result contract), driving the
public ``CSCIPCA.fit()`` at :math:`L = 9`, :math:`K = 3`, :math:`T_0 = 40` over
150 draws per grid point:

.. list-table::
   :header-rows: 1
   :widths: 34 33 33

   * - Observed-covariate share :math:`\alpha`
     - CSC-IPCA bias
     - naive simplex-SC bias
   * - 1/3
     - :math:`\approx +1.3`
     - :math:`\approx +7.7`
   * - 2/3
     - :math:`\approx +0.8`
     - :math:`\approx +7.7`
   * - 1
     - :math:`\approx 0.0`
     - :math:`\approx +7.7`

The bias is monotone in :math:`\alpha` and near-zero when every covariate is
observed, and CSC-IPCA is far less biased than the synthetic control at every
:math:`\alpha` — the treated unit's covariate drift places it outside the donor
hull, so the simplex fit extrapolates and inherits a large positive bias that
the instrumented factor model avoids. The durable case is
``benchmarks/cases/cscipca_mc.py``.

Seam notes
----------

Four details were pinned while porting, and each is held by a unit test:

* Covariate cube, not covariate means. ``dataprep(covariates=...)`` aggregates
  each covariate to a per-unit pre-period mean (the Abadie predictor block),
  which discards the time variation the instrumented loadings
  :math:`\boldsymbol{\lambda}_{it} = \mathbf{x}_{it}^\top \boldsymbol{\Gamma}`
  depend on. The port rides dataprep for the outcome contract and pivots each
  covariate column into the full :math:`(N, T, L)` cube separately.
* Convergence on the fit, not the parameters. The bilinear objective is
  invariant to the :math:`\boldsymbol{\Gamma} \to \boldsymbol{\Gamma} R`,
  :math:`\mathbf{f} \to R^{-1}\mathbf{f}` rotation, so the raw parameters can
  drift within that subspace forever while the counterfactual has already
  converged. ALS convergence is measured on the fitted values, which reports a
  meaningful ``converged`` flag (77 iterations on the FDI panel) and leaves the
  matched numbers unchanged.
* Identification of the treated mapping. With a single treated unit the
  :math:`LK` entries of :math:`\boldsymbol{\Gamma}` are solved from the treated
  unit's :math:`T_0` pre-periods, so the setup requires :math:`T_0 \ge LK` and
  raises a clear error otherwise. The paper's five-treated grid hides this; a
  single treated unit makes it binding.
* At least as many covariates as factors. The projected loadings cannot span
  :math:`K` factors from fewer than :math:`K` covariates, so :math:`L < K`
  yields a singular factor solve; the config rejects it up front.
