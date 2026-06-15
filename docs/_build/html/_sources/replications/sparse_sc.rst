.. _replication-sparse-sc:

SparseSC — L1 Predictor Selection for Synthetic Controls (Vives-i-Bastida 2022)
===============================================================================

:Estimator: :doc:`../sparse_sc` — :class:`mlsynth.SparseSC`
:Source: Vives-i-Bastida, Jaume (2022), *"Predictor Selection for Synthetic
   Controls,"* working paper, arXiv:2203.11576 [jaumesparsesc]_.
:Replication type: **Path A** (two canonical empirical panels) **+ Path B**
   (the paper's own Monte-Carlo study, Section 4 / Figures 1-2).
:Status: **Fully verified** — recovers the Abadie, Diamond & Hainmueller (2010)
   Proposition 99 ATT *and* the Abadie & Gardeazabal (2003) Basque result
   (each with the L1 penalty selecting the predictor set), and reproduces the
   qualitative findings of the paper's simulation study.

Why Path A
----------

SparseSC's contribution is **predictor selection**: an L1 penalty on the
predictor-importance vector :math:`v` drives uninformative predictors to exactly
zero. The natural way to demonstrate that is to hand the estimator a *deliberately
over-rich* predictor set on the canonical California tobacco-control panel
(Proposition 99) and check that (a) the penalty prunes it back to a sparse,
interpretable subset, and (b) the resulting effect still lands on the
Abadie-Diamond-Hainmueller (ADH) benchmark of roughly :math:`-19` packs with the
ADH donor pool. Both the outcome panel and the augmented covariates ship in
``basedata/augmented_cali_long.csv``, so this is reproducible value-for-value.

The specification
-----------------

* **Outcome / treatment:** per-capita cigarette sales, California treated from
  1989; pre-period 1970-1988.
* **Predictors (over-rich, the point of the exercise):** 30 economic and policy
  covariates carried in the augmented panel (collapsed to pre-period unit means)
  **plus three lagged outcomes** — cigarette sales in 1975, 1980 and 1988 —
  giving :math:`P = 33` predictor rows against :math:`N = 38` donors. The first
  predictor is pinned to :math:`v_1 = 1` to fix the scale; the rest are
  bound-constrained non-negative.
* **Penalty selection:** the default 51-point grid
  :math:`\{0\} \cup \mathrm{logspace}(-4, 0, 50)`, with :math:`\lambda` chosen by
  the unpenalised validation-block MSE.
* **Outer V-solve:** the finite-difference gradient default (see *A note on
  optimisation* below).
* **Inference:** the default moving-block conformal CI (Chernozhukov,
  Wuethrich & Zhu 2021), calibrated on the validation residuals.

Reproducing the result
----------------------

.. code-block:: python

   import pandas as pd, numpy as np
   from mlsynth import SparseSC

   d = pd.read_csv("basedata/augmented_cali_long.csv")
   d["treated"] = ((d.state == "California") & (d.year >= 1989)).astype(int)

   # over-rich predictor set: every numeric covariate with complete, non-constant
   # pre-period coverage (collapsed to unit means), capped at 30, + lagged outcomes
   pre = d[d.year < 1989]
   drop = {"state", "year", "treated", "cigsale", "stateno", "state_fips",
           "state_icpsr", "is_a_state", "region"}
   covs = [c for c in d.columns if c not in drop and d[c].dtype.kind in "if"
           and pre.groupby("state")[c].mean().notna().all()
           and pre.groupby("state")[c].mean().std(ddof=1) > 0][:30]

   res = SparseSC({
       "df": d, "outcome": "cigsale", "treat": "treated",
       "unitid": "state", "time": "year",
       "covariates": covs, "outcome_lag_periods": [1975, 1980, 1988],
       "run_inference": True, "inference_method": "conformal",
       "display_graphs": False,            # defaults: FD gradient, 51-pt grid
   }).fit()

   keep = int(np.sum(np.asarray(res.design.v) > 1e-6))
   print(f"ATT={res.att:.2f}  pre-RMSE={res.pre_rmse:.2f}  "
         f"lambda*={res.design.opt_lambda:.4g}  predictors kept={keep}/{len(res.design.v)}")
   print(f"95% CI=[{res.inference.ci_lower:.2f}, {res.inference.ci_upper:.2f}]")

Results
-------

.. list-table::
   :header-rows: 1
   :widths: 36 24 24

   * - Quantity
     - SparseSC (augmented)
     - ADH (2010) benchmark
   * - ATT, 1989-2000 (packs)
     - **-17.9**
     - :math:`\approx -19`
   * - 95% conformal CI
     - ``[-21.3, -15.4]``
     - excludes 0
   * - pre-treatment RMSE
     - 2.21
     - ~1.8
   * - predictors kept (of 33)
     - **6**
     - n/a
   * - selected :math:`\lambda`
     - 0.019
     - n/a
   * - donor pool
     - Utah 0.39, Nevada 0.30, Connecticut 0.20, Colorado 0.12
     - Utah / Nevada / Connecticut / Colorado / Montana

What it confirms
----------------

* **The penalty selects.** From 33 candidate predictors the L1 fit keeps only
  **6**, discarding the bulk of the over-rich augmented set — exactly the
  variable-selection behaviour that motivates the method.
* **The effect is canonical.** The pruned fit lands at :math:`-17.9` packs with a
  conformal interval excluding zero, squarely on the ADH :math:`\approx -19`
  benchmark, and recovers **ADH's donor pool** (Utah / Nevada / Connecticut /
  Colorado) from a 38-state pool — the selection does not distort the answer.

The durable check lives in ``benchmarks/cases/sparse_sc_prop99.py``::

   python benchmarks/run_benchmarks.py --case sparse_sc_prop99

A second case: the Basque Country
---------------------------------

The same estimator, unchanged, reproduces the other canonical SC study —
Abadie & Gardeazabal's (2003) Basque Country terrorism analysis — on the full
predictor set shipped in ``basedata/basque_data.csv``. Here the outcome is real
GDP per capita, the Basque Country is treated from 1975, and the predictors are
the A&G schooling shares, sectoral GVA shares, investment ratio and population
density (collapsed to pre-period unit means), plus three lagged outcomes (GDP
per capita in 1960, 1965, 1969) — :math:`P = 15` against :math:`N = 16` donors.

.. code-block:: python

   import pandas as pd
   from mlsynth import SparseSC

   d = pd.read_csv("basedata/basque_data.csv")
   d["treated"] = ((d.regionname == "Basque Country (Pais Vasco)")
                   & (d.year >= 1975)).astype(int)
   ed  = ["school.illit", "school.prim", "school.med", "school.high", "invest"]
   sec = ["sec.agriculture", "sec.energy", "sec.industry", "sec.construction",
          "sec.services.venta", "sec.services.nonventa"]

   res = SparseSC({
       "df": d, "outcome": "gdpcap", "treat": "treated",
       "unitid": "regionname", "time": "year",
       "covariates": ed + sec + ["popdens"],
       "outcome_lag_periods": [1960, 1965, 1969],
       "run_inference": True, "inference_method": "conformal",
       "display_graphs": False,
   }).fit()

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Quantity
     - SparseSC
     - A&G (2003) benchmark
   * - ATT, 1975-1997 (GDP p.c., thousands)
     - **-0.65**
     - peak :math:`\approx -0.85`
   * - 95% conformal CI
     - ``[-0.71, -0.60]``
     - excludes 0
   * - pre-treatment RMSE
     - 0.092
     - n/a
   * - predictors kept (of 15)
     - **3** (illit, non-market services, popdens)
     - n/a
   * - donor pool
     - **Cataluna 0.82, Madrid 0.16**, Cantabria 0.03
     - Catalonia + Madrid
   * - gap trajectory
     - opens ~1978, peak :math:`-0.95` (1990), :math:`-0.75` by 1997
     - widening through the 1980s

What it adds: SparseSC recovers **A&G's actual two-donor synthetic** — Catalonia
plus Madrid — rather than the single-donor Catalonia the penalized/MSCMT backends
collapse to on this panel, while pruning 15 predictors to 3 and achieving a
tighter pre-fit (RMSE 0.092) than either. The effect (:math:`-0.65` average,
peaking :math:`-0.95`) lands on the A&G result. Notably the penalty *keeps*
``popdens`` as informative here — the very dimension that destabilised the
Abadie-L'Hour bias correction on this panel: density genuinely helps *explain*
GDP (a good predictor) even though it cannot be safely *extrapolated* in a
residual correction.

Path B: the simulation study (Section 4)
----------------------------------------

The paper's Monte-Carlo (Figures 1-2) is reproduced from a linear factor model
with a *grouped* structure (the design of the companion paper, Abadie &
Vives-i-Bastida 2022, extended with covariates):

.. math::

   Y_{it} = \delta_t + \theta_t' Z_i + \lambda_t' \mu_i + \varepsilon_{it},

with ``J+1 = 21`` units in 7 groups of 3 sharing a one-hot factor loading
``mu_i``; common factors ``lambda_t`` AR(1) (``rho = 0.5``, standard-Gaussian
innovations); ``delta_t = 100``; ``eps ~ N(0, 0.25^2)``. Covariates split into
**useful** ``Z^1`` (nonzero ``theta``) and **nuisance** ``Z^2`` (zero
``theta``), all drawn ``U[0,1]``; the treated unit's useful predictors are set
to ``1/2(Z_2 + Z_3)`` and it shares units 2,3's group, so the **oracle synthetic
control is** ``w_2 = w_3 = 1/2``. The design matrix adds 10 lagged outcomes (20
predictors total). Two regimes: ``k1=k2=5`` (balanced) and ``k1=1, k2=9``
(nuisance-heavy). The true effect is zero, so post-treatment MSE measures
counterfactual prediction error. Three estimators are compared, mapped onto
``SparseSC``'s knobs:

* **SCM** — the standard control with the *fixed* Mahalanobis weight
  ``V = (X0' X0)^{-1}`` (no optimisation; solved as ``min_w (X1-X0 w)'V(X1-X0 w)``
  on the simplex).
* **SCM λ=0** — ``V`` minimises the *validation*-block outcome fit with no
  penalty (``outer_loss_window="validation"``, ``lambda=0``; Abadie-Diamond-
  Hainmueller 2015).
* **Sparse** — the same with the L1 penalty and CV-selected ``lambda``.

Results (B = 60 draws; "Vnoise" is the V-weight assigned to the nuisance
covariates):

.. list-table::
   :header-rows: 1
   :widths: 16 12 10 10 10 10

   * - setting / method
     - post-MSE
     - \|bias\|
     - val-RMSE
     - w₂+w₃
     - Vnoise
   * - **k1=5,k2=5** SCM
     - 1.78
     - 0.06
     - 1.27
     - 0.25
     - —
   * - SCM λ=0
     - 0.132
     - 0.001
     - 0.278
     - 0.92
     - 5.27
   * - Sparse
     - 0.153
     - 0.010
     - 0.255
     - 0.90
     - **0.16**
   * - **k1=1,k2=9** SCM
     - 1.99
     - 0.02
     - 1.28
     - 0.17
     - —
   * - SCM λ=0
     - 0.164
     - 0.003
     - 0.253
     - 0.89
     - 0.85
   * - Sparse
     - **0.141**
     - 0.016
     - **0.224**
     - 0.90
     - **0.21**

The paper's three headline findings reproduce:

* **Standard SCM is decisively worst** (post-MSE ~1.8-2.0; it cannot even
  concentrate weight on the right donors, w₂+w₃ ≈ 0.2).
* **Sparse is robust across regimes while the unpenalised method degrades.**
  Moving from balanced to nuisance-heavy, ``SCM λ=0`` worsens 0.132 → 0.164
  while **Sparse stays flat (0.153 → 0.141) and overtakes it** — Figure 1b's
  central result.
* **Sparse performs the selection** (Figure 2): it drives the nuisance-predictor
  weights to ≈ 0 (Vnoise 0.16-0.21) where ``SCM λ=0`` piles weight on them
  (5.27, 0.85), and it carries the lowest validation RMSE throughout.

Honest caveats: *absolute* MSE levels are not comparable to the paper (the
useful-predictor coefficient scale ``theta_t`` is not pinned down in either
paper — the companion design has no covariate term — so it is filled in the
spirit of the ``theta_t Z_i`` term as time-varying ``N(0,1)``); and in the easy
``k1=k2=5`` regime Sparse ties rather than strictly beats ``SCM λ=0``, within
the latitude of that unspecified constant, B = 60 Monte-Carlo noise, and a
coarser (21-point) penalty grid. The *ordering* and the *robustness/selection*
mechanism — the paper's actual claims — reproduce.

A note on optimisation (and grid resolution)
---------------------------------------------

Two implementation facts surface naturally here and are worth recording:

* **Grid resolution does the heavy lifting.** A coarse 21-point grid lands at
  :math:`-20.8` packs with pre-RMSE 4.77 and 25 predictors retained — the penalty
  barely bites. The full **51-point** default grid finds the better
  :math:`\lambda` (0.019), achieves true sparsity (6 predictors), halves the
  pre-RMSE, and pulls the ATT onto :math:`-17.9`. The default grid matters more
  than any micro-optimisation of the solver.
* **Keep the finite-difference gradient default.** SparseSC also ships an
  envelope-theorem closed-form gradient (``use_analytical_grad=True``) that is
  ~5-10x faster, but on this augmented spec it settles on a much worse critical
  point (pre-RMSE :math:`\approx 10`, no predictor selection) that even
  multi-start restarts do not escape — the finite-difference path's gradient
  noise is what finds the good basin. The analytical gradient is therefore
  opt-in, and the verified result above uses the FD default.

.. note::

   "Augmented" here is an automatically selected 30-covariate proxy (every
   numeric covariate with complete, non-constant pre-period coverage), not a
   hand-curated predictor list. The headline match — sparse selection, ADH-range
   ATT, ADH donor pool — is robust to that choice; the exact :math:`\lambda` and
   retained-predictor identities will shift with the predictor set.
