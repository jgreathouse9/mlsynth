.. _replication-sdid:

SDID — Synthetic Difference-in-Differences (Arkhangelsky et al. 2021)
=====================================================================

:Estimator: :doc:`../sdid` — :class:`mlsynth.SDID`
:Source: Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., &
   Wager, S. (2021), *"Synthetic Difference-in-Differences,"* American
   Economic Review 111(12):4088-4118.
:Replication type: **Path A** — the paper's Proposition 99 empirical, with a
   **cross-validation** against the authors' own ``synthdid`` R package.
:Status: **Fully verified** — empirical headline reproduced and matched to the
   authors' reference implementation.

Validation strategy
-------------------

Arkhangelsky et al.'s headline application is California's Proposition 99
tobacco-control program, estimated on the canonical Abadie-Diamond-Hainmueller
smoking panel (39 states, 1970-2000; California treated from 1989). The paper
reports an SDID ATT of about **-15.6** packs per capita, matched by the
authors' R ``synthdid`` package (-15.604). mlsynth reproduces that number to
three significant figures, and we cross-validate the implementation against a
live run of ``synthdid`` on the same matrix.

Path A — Proposition 99
-----------------------

The panel ships as ``basedata/smoking_data.csv`` with a ready-made
``Proposition 99`` indicator flagging the treated unit/period cells.

.. code-block:: python

   import pandas as pd
   from mlsynth import SDID

   df = pd.read_csv("basedata/smoking_data.csv")
   df["treat"] = df["Proposition 99"].astype(int)
   res = SDID({"df": df[["state", "year", "cigsale", "treat"]],
               "outcome": "cigsale", "treat": "treat",
               "unitid": "state", "time": "year",
               "display_graphs": False}).fit()
   res.att                  # -15.605

mlsynth returns :math:`\widehat{\mathrm{ATT}} = -15.605`, matching the
AER headline (-15.6) and the ``synthdid`` value (-15.604).

Cross-validation against the authors' ``synthdid`` R
----------------------------------------------------

We cross-validate against the method's own authors' code -- the
``synth-inference/synthdid`` R package -- run live on the identical outcome
matrix. On Proposition 99 ``synthdid_estimate`` returns :math:`-15.6038`, and
mlsynth reproduces it to :math:`1.6 \times 10^{-3}` packs. The residual is the
unit-weight ridge (:math:`\zeta`) optimiser, not a methodological difference --
the SDID weight QPs and the final regression are identical. The same package's
DiD (:math:`-27.349`) and pure-SC (:math:`-19.620`) estimates on the same matrix
are recorded for context (mlsynth's SDID targets the SDID column).

.. list-table::
   :header-rows: 1
   :widths: 40 28 28

   * - Quantity
     - mlsynth
     - synthdid R / AER
   * - SDID ATT
     - -15.605
     - -15.604 / -15.6
   * - DiD ATT (context)
     - —
     - -27.349
   * - SC ATT (context)
     - —
     - -19.620

Durable check
-------------

The reference is pinned under ``benchmarks/reference/sdid_prop99/`` (R 4.3.3,
``synthdid`` commit 70c1ce3, data checksum), captured by its ``reference.R``.
The benchmark reads the frozen values (no R needed at test time)::

   python benchmarks/run_benchmarks.py --case sdid_prop99

It asserts the ATT lands on the published -15.604 (tol 0.05) and matches the
authors' ``synthdid`` to within :math:`0.02` (observed :math:`1.6 \times
10^{-3}`).

Synthetic triple difference — Virginia's HPV mandate (Path A)
-------------------------------------------------------------

The SC-DDD mode (``subgroup`` / ``target_subgroup``; Zhuang 2024) is
cross-validated against the Stata ``sdid`` output of Feldman & Semprini (2026),
who evaluate Virginia's 2008 school-entry HPV vaccine mandate on cervical
cancer incidence. Virginia is the treated state; ages 20-24 (the first
mandate-exposed cohort by 2016) are the target subgroup; older age bands are the
within-state controls. mlsynth demeans the age-adjusted incidence by the
non-target ages within each treatment-group-by-year cell, then runs SDID on the
20-24 subgroup with Virginia treated from 2016.

The data ship as ``basedata/hpv_cervical_ddd.csv`` (39 states x 17 years,
2003-2019, public NPCR/SEER via the authors' repository
``jsemprini/Virginia_HPVmandate_causal``).

.. list-table::
   :header-rows: 1
   :widths: 46 27 27

   * - Estimator
     - mlsynth
     - Stata ``sdid``
   * - SC-DDD (transformed outcome)
     - +1.559
     - +1.559
   * - naive SC-DD (untransformed 20-24)
     - +0.252
     - +0.252

The SC-DDD point estimate is a cell-for-cell match: mlsynth's SDID engine is
already validated against ``synthdid`` R above, so feeding it the Zhuang-demeaned
outcome reproduces the Stata result exactly. The naive SC-DD on the untransformed
20-24 outcome lands on the paper's near-null 0.252, so the triple-difference
demeaning is what flips the estimate positive and significant. The placebo 95%
interval excludes zero (mlsynth 0.35-2.76 vs the paper's 0.42-2.70; the small
difference is the placebo resampling, which is seed- and implementation-
dependent). The durable case is ``benchmarks/cases/sdid_ddd_hpv.py``::

   python benchmarks/run_benchmarks.py --case sdid_ddd_hpv

References
----------

Feldman, C., & Semprini, J. (2026). "Causal inference, cancer registry data,
and a single state policy change: Evaluating Virginia's HPV vaccine mandate."
*Journal of Cancer Policy* 49:100777.

Zhuang, C. C. (2024). "A Way to Synthetic Triple Difference."
arXiv:2409.12353.


Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S.
(2021). "Synthetic Difference-in-Differences." *American Economic Review*
111(12):4088-4118.
