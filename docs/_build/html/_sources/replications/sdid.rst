.. _replication-sdid:

SDID — Synthetic Difference-in-Differences (Arkhangelsky et al. 2021)
=====================================================================

:Estimator: :doc:`../sdid` — :class:`mlsynth.SDID`
:Source: Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., &
   Wager, S. (2021), *"Synthetic Difference-in-Differences,"* American
   Economic Review 111(12):4088-4118.
:Replication type: **Path A** — the paper's Proposition 99 empirical, with a
   **cross-validation** against the ``causaltensor`` reference implementation.
:Status: **Fully verified** — empirical headline reproduced and matched
   cell-for-cell to an independent implementation.

Validation strategy
-------------------

Arkhangelsky et al.'s headline application is California's Proposition 99
tobacco-control program, estimated on the canonical Abadie-Diamond-Hainmueller
smoking panel (39 states, 1970-2000; California treated from 1989). The paper
reports an SDID ATT of about **-15.6** packs per capita, matched by the
authors' R ``synthdid`` package (-15.604). mlsynth reproduces that number to
three significant figures, *and* we cross-validate the implementation against
``causaltensor.SDID`` — a fully independent Python port of the
:math:`\widehat\tau^{\text{sdid}}` estimator — on the same matrix.

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

Cross-validation against ``causaltensor``
-----------------------------------------

The same outcome matrix :math:`O` (39 × 31) and treatment mask :math:`Z`
are handed to ``causaltensor.SDID``:

.. code-block:: python

   import numpy as np, causaltensor as ct

   wide = df.pivot(index="state", columns="year", values="cigsale").sort_index()
   states, years = wide.index.tolist(), wide.columns.tolist()
   O = wide.values.astype(float)
   ti, sc = states.index("California"), years.index(1989)
   Z = np.zeros_like(O); Z[ti, sc:] = 1
   ct.SDID(O, Z, treat_units=[ti], starting_time=sc)   # -15.602

The two implementations agree to :math:`|\Delta| = 3.1 \times 10^{-3}` packs.
The residual is the unit-weight ridge (:math:`\zeta`) optimiser, not a
methodological difference — the SDID weight QPs and the final regression are
identical.

.. list-table::
   :header-rows: 1
   :widths: 34 22 22 22

   * - Quantity
     - mlsynth
     - causaltensor
     - AER / synthdid
   * - Overall ATT
     - -15.605
     - -15.602
     - -15.6 / -15.604

Durable check
-------------

The benchmark lives in ``benchmarks/cases/sdid_prop99.py`` and runs in the
default suite (skipping gracefully if ``causaltensor`` is absent)::

   pip install causaltensor
   python benchmarks/run_benchmarks.py --case sdid_prop99

It asserts the ATT lands on the published -15.604 (tol 0.05) and matches
``causaltensor`` to within :math:`5 \times 10^{-3}`.

References
----------

Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S.
(2021). "Synthetic Difference-in-Differences." *American Economic Review*
111(12):4088-4118.
