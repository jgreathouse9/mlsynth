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
   res.inference.att        # -15.605

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

References
----------

Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S.
(2021). "Synthetic Difference-in-Differences." *American Economic Review*
111(12):4088-4118.
