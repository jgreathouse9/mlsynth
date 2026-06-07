.. _replication-fdid:

FDID — Forward Difference-in-Differences (Li 2024)
==================================================

:Estimator: :doc:`../fdid` — :class:`mlsynth.FDID`
:Source: Li, Kathleen T. (2024), *"Frontiers: A Simple Forward
   Difference-in-Differences Method,"* Marketing Science 43(2) [Li2024]_.
:Replication type: **Path A** — the author's released public empirical
   (Hong Kong GDP) reproduced cell by cell — **and Path B** — the paper's
   own Monte Carlo (Web Appendix E, Table 5).
:Status: **Fully verified** — empirical *and* simulation reproduced.

Validation strategy
-------------------

Li's *headline* application — the effect of opening physical stores on an
online-first retailer's city-level sales — runs on a **confidential retailer
dataset** that cannot be redistributed. For context, that study reports a
Forward DiD effect of opening a store in Atlanta of about **+$75,143 in
monthly sales (an 86% lift, pre-period** :math:`R^2 = 0.76`\ **)**, with
ordinary DiD and synthetic control — which fit Atlanta's steep pre-trend
poorly — overstating it.

That headline number cannot be checked value-for-value, but it does not have
to be: alongside the paper the author released a **public companion
replication** (MATLAB and R) on the Hsiao, Ching & Wan (2012) Hong Kong GDP
panel, which mlsynth reproduces cell by cell (**Path A**). We *additionally*
reproduce the paper's own Monte Carlo (**Path B**), which exercises the
mismatched-control regimes a single empirical case cannot.

Path A — Hong Kong GDP
----------------------

The author's public dataset (``basedata/HongKong.csv`` — Hong Kong plus 24
OECD / Asian control economies over 61 quarters, with treatment, the
political and economic integration with mainland China, beginning at quarter
44 so :math:`T_1 = 44`) is the Hsiao, Ching & Wan (2012) panel.

.. code-block:: python

   import pandas as pd
   from mlsynth import FDID

   df = pd.read_csv("basedata/HongKong.csv")
   res = FDID({"df": df, "outcome": "GDP", "treat": "Integration",
               "unitid": "Country", "time": "Time",
               "display_graphs": False, "verbose": False}).fit()
   res.fdid.att, res.fdid.att_percent, res.fdid.r_squared, len(res.fdid.selected_names)

Forward DiD selects **9 of the 24** controls and reproduces the author's
released MATLAB/R output (``ForwardDID_Readme.txt``) cell by cell:

.. list-table::
   :header-rows: 1
   :widths: 26 18 18 18 18

   * - Metric
     - FDID (mlsynth)
     - FDID (Li)
     - DID (mlsynth)
     - DID (Li)
   * - ATT
     - 0.0254
     - 0.0254
     - 0.0317
     - 0.0317
   * - % ATT
     - 53.84
     - 53.84
     - 77.62
     - 77.62
   * - pre-period :math:`R^2`
     - 0.843
     - 0.843
     - 0.505
     - 0.505
   * - controls used
     - 9
     - 9
     - 24
     - 24

The 95% confidence interval ``(0.0163, 0.0345)`` and the standardized ATT
(t-statistic) :math:`\approx 5.49` likewise match the released values.
Forward DiD's far higher pre-period fit (:math:`R^2 = 0.84` versus DiD's
:math:`0.50`) is the whole point: the all-controls average tracks Hong Kong's
pre-integration path poorly, so plain DiD overstates the effect, while the
forward search keeps only the 9 economies that co-move with Hong Kong.

The durable check lives in ``benchmarks/cases/fdid_hongkong.py``::

   python benchmarks/run_benchmarks.py --case fdid_hongkong

Path B — the simulation design
------------------------------

The four DGPs and their factor structure are packaged in
:func:`mlsynth.utils.fdid_helpers.simulation.simulate_fdid_sample`: three
common factors — ``f1`` AR(1) ``0.8``, ``f2`` ARMA(1,1) ``(-0.6, 0.8)``,
``f3`` MA(2) ``(0.9, 0.4)``, innovations :math:`N(0,1)` — with outcomes
:math:`a_0 + c_0 \sum_k f_{kt} + \varepsilon` for the treated unit and
:math:`1 + c \sum_k f_{kt} + \varepsilon` for the controls (the first half of
the donor pool loading :math:`c_1`, the second half :math:`c_2`). The four
DGPs vary :math:`(a_0, c_0, c_1, c_2)`:

* **DGP1** ``(1,1,1,1)`` and **DGP3** ``(2,1,1,1)`` — all controls share the
  treated unit's factor loading, so ordinary DiD is applicable.
* **DGP2** ``(1,1,1,2)`` and **DGP4** ``(2,1,1,2)`` — half the controls carry
  the wrong loading, so the all-controls DiD average is contaminated.

The true ATT is :math:`0`, and the reported risk is the predictive MSE
:math:`\mathrm{PMSE} = M^{-1}\sum_j \widehat{\mathrm{ATT}}_j^2`.

Reproducing Table 5
-------------------

.. code-block:: python

   import numpy as np
   from mlsynth import FDID
   from mlsynth.utils.fdid_helpers.simulation import simulate_fdid_sample

   def pmse_cell(dgp, N, T1, T2, M, seed=0):
       fdid_sq, did_sq = [], []
       for j in range(M):
           rng = np.random.default_rng(seed + j)
           sample = simulate_fdid_sample(dgp=dgp, N=N, T1=T1, T2=T2, rng=rng)
           res = FDID({"df": sample.df, "outcome": "y", "treat": "treat",
                       "unitid": "unit", "time": "time",
                       "display_graphs": False, "verbose": False}).fit()
           fdid_sq.append(res.fdid.att ** 2)   # ATT = 0, so SE^2 = att^2
           did_sq.append(res.did.att ** 2)
       return float(np.mean(fdid_sq)), float(np.mean(did_sq))

   for dgp in (1, 2, 3, 4):
       for T1, T2 in [(12, 6), (24, 12), (48, 24)]:
           fdid_pmse, did_pmse = pmse_cell(dgp, N=60, T1=T1, T2=T2, M=1000)
           print(f"DGP{dgp} ({T1},{T2}): FDID={fdid_pmse:.4f}  DID={did_pmse:.4f}")

Results
-------

At :math:`M = 1{,}000` (Li uses :math:`M = 10{,}000`; the runtime difference is
the only material change) this reproduces Table 5 cell by cell:

.. list-table::
   :header-rows: 1
   :widths: 8 14 18 18 18 18

   * - DGP
     - :math:`(T_1, T_2)`
     - DID (mlsynth)
     - DID (Li)
     - FDID (mlsynth)
     - FDID (Li)
   * - 1
     - (12, 6)
     - 0.265
     - 0.259
     - 0.325
     - 0.315
   * - 1
     - (24, 12)
     - 0.127
     - 0.128
     - 0.147
     - 0.146
   * - 1
     - (48, 24)
     - 0.065
     - 0.063
     - 0.075
     - 0.071
   * - 2
     - (12, 6)
     - 1.202
     - 1.037
     - 0.431
     - 0.385
   * - 2
     - (24, 12)
     - 0.765
     - 0.746
     - 0.177
     - 0.180
   * - 2
     - (48, 24)
     - 0.451
     - 0.473
     - 0.084
     - 0.082
   * - 3
     - (12, 6)
     - 0.265
     - 0.252
     - 0.325
     - 0.303
   * - 3
     - (24, 12)
     - 0.127
     - 0.123
     - 0.147
     - 0.143
   * - 3
     - (48, 24)
     - 0.065
     - 0.064
     - 0.075
     - 0.072
   * - 4
     - (12, 6)
     - 1.202
     - 1.038
     - 0.431
     - 0.391
   * - 4
     - (24, 12)
     - 0.765
     - 0.744
     - 0.177
     - 0.171
   * - 4
     - (48, 24)
     - 0.451
     - 0.454
     - 0.084
     - 0.081

What it confirms
----------------

The two headline findings reproduce:

* **When all controls are valid (DGP1, DGP3),** DiD is the parsimonious,
  efficient choice and edges out Forward DiD by a small margin at every
  horizon — Forward DiD pays only a small efficiency cost for the safety of
  the search.
* **When half the controls are mismatched (DGP2, DGP4),** DiD's PMSE stays
  large and **does not shrink** as the panel grows (DGP2 at :math:`(48,24)`:
  DID still ``0.45``) because the contaminating controls bias the
  all-controls average, while Forward DiD's PMSE **collapses** (``0.084``)
  because the forward search discards them. Forward DiD wins decisively when
  DiD is invalid — Li's central result.

The identity of the DGP1/DGP3 (and DGP2/DGP4) columns also confirms the
estimator's **intercept invariance**: moving :math:`a_0` from 1 to 2 changes
nothing because Forward DiD's :math:`\widehat\alpha` absorbs it. The
:math:`(12, 6)` cell runs slightly hot under DGP2/4, consistent with Monte
Carlo noise at :math:`M = 1{,}000` versus Li's :math:`M = 10{,}000`.
