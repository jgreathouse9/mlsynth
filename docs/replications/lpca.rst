.. _replication-lpca:

LPCA — Local Principal Component Analysis (Feng 2024)
======================================================

:Estimator: :doc:`../lpca` — :class:`mlsynth.LPCA`
:Source: Feng, Y. (2024), *"Optimal Estimation of Large-Dimensional Nonlinear
   Factor Models,"* working paper, arXiv:2311.07243.
:Replication type: **Path A** — the Kansas tax-cut application — and
   **Path B** — the Section 5 Monte Carlo.
:Status: **Fully verified** — both paths reproduce, and a defect in the
   paper's first version was found and confirmed against the author's own
   revision.

Validation strategy
-------------------

Feng releases replication code (``yingjieum/Replication_NonlinearFactorModel_2023``),
and the panel behind the empirical application is the augsynth ``kansas``
dataset, already in this repository as ``basedata/kansas_taxcut.csv``. Both
targets are therefore checkable end to end: the empirical estimate in Section
6.1, and the simulation table in Section 5.

Path A — the 2012 Kansas tax cuts
---------------------------------

Feng analyses quarterly log GDP per capita for the 50 states from 1990Q1 to
2016Q1, first-differenced into growth rates, with Kansas treated from 2012Q2.
The tuning is fixed by the author's script: :math:`K = \operatorname{round}(n^{2/3}) = 14`,
the first 40 differenced quarters for matching, at most three local components.

.. code-block:: python

   import pandas as pd
   from mlsynth import LPCA

   df = pd.read_csv("basedata/kansas_taxcut.csv").sort_values(["state", "year_qtr"])
   df["growth"] = df.groupby("state")["lngdpcapita"].diff() * 100.0
   df = df.dropna(subset=["growth"])

   res = LPCA({"df": df, "outcome": "growth", "treat": "treated",
               "unitid": "state", "time": "year_qtr",
               "match_periods": 40, "display_graphs": False}).fit()
   res.att        # -0.5306

.. list-table::
   :header-rows: 1
   :widths: 60 20 20

   * - Quantity
     - mlsynth
     - Paper
   * - Observed minus counterfactual, mean post-treatment growth
     - :math:`-0.5306` pp
     - :math:`-0.53` pp
   * - Post-treatment quarters with observed below the LPCA path
     - 9 of 16
     - 9 of 16

Pinned in ``mlsynth/tests/test_lpca.py``
(``TestEstimator::test_reproduces_fengs_kansas_application``).

Path B — Table 1
----------------

Three data-generating processes at :math:`n = p = 1000`, half the columns for
matching, the neighbourhood grid 49/99/149, against a global-PCA baseline whose
factor count comes from an eigenvalue ratio on doubly demeaned data. Run at 500
replications against the paper's 2000; across the 48 cells the median
disagreement is 0.83 Monte Carlo standard errors, 43 fall within 2 and 47 within
3. The maximum-absolute-error row, mlsynth against the paper:

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * -
     - K=49
     - K=99
     - K=149
     - GPCA
   * - Model 1
     - 0.683 / 0.680
     - 0.963 / 0.937
     - 2.208 / 2.203
     - 1.141 / 1.148
   * - Model 2
     - 0.602 / 0.599
     - 0.633 / 0.636
     - 0.703 / 0.706
     - 0.866 / 0.870
   * - Model 3
     - 0.475 / 0.475
     - 0.461 / 0.461
     - 0.461 / 0.461
     - 0.469 / 0.470

Every qualitative claim in Section 5 holds: local PCA beats global PCA on the
two nonlinear designs at the smaller neighbourhoods, ties it on the binary
design, and the advantage widens with the severity of the nonlinearity. Model 1
at :math:`K = 149` blows up to 2.208 against a baseline of 1.141, reproducing
the paper's warning that too large a neighbourhood destroys the local
approximation where the surface bends hardest.

Two details of the reference decide whether the table reproduces at all. The
neighbourhood grid is 49/99/149 and not 50/100/150, because the script computes
:math:`n^{2/3}`, which is 99.99999999999999 in double precision, and floors the
products. And the neighbour rule is a threshold, so exact ties widen the
neighbourhood -- Model 3 is binary and ties constantly, inflating a nominal 49
to between 51 and 73. That is a plausible explanation for Model 3's row being
flat across :math:`K`, and it is why the estimator reports the realised
neighbourhood next to the requested one.

One cell is unresolved. Model 1's :math:`q_\alpha = .9` prediction error comes
in at 0.066 against the published 0.076 at :math:`K = 49`, which is 3.9 standard
errors; the :math:`K = 99` cell shares its draws, so it is one effect and not
two. Model 1's surface is symmetric in the latent variable and the published
:math:`.1` and :math:`.9` rows reflect that symmetry, while this port's do not.
No claim turns on the cell, and local PCA still beats the baseline there.

A defect in the paper's first version
-------------------------------------

The replication also settled something about the paper. arXiv version 1
(November 2023) compares LPCA against a simplex synthetic control and reports
that SC predicts growth 0.19 points *below* observed Kansas, against LPCA's 0.53
*above* -- opposite signs, which is the contrast Section 6.1 is built on, and the
basis for calling the SC answer implausible.

That number came from a one-token omission in the November 2023 application
script: the SC line lacked the ``+ col.mean`` that the LPCA line carried, so the
synthetic-control path was compared against an observed series it had never been
re-centred onto. Reproducing the omission gives :math:`+0.1948`, matching the
published value; correcting it gives :math:`-0.3340`.

The author's current version confirms the correction. Feng (2024), dated 31 July
2024, reports SC with "an average growth rate 0.33 percentage points higher than
that of the observed Kansas". Corrected, both estimators agree the tax cut cost
growth and differ by 0.20 points of magnitude. The revision also drops the v1
claim that SC's pre-treatment fit was poor, which is consistent with the
measurement: on the window where both arms predict, LPCA's pre-treatment RMSE is
0.866 pp against the synthetic control's 0.624 pp.

Cite the July 2024 version. The :doc:`../lpca` page quotes the corrected
numbers.

Durable artifacts
-----------------

* ``benchmarks/reference/lpca_kansas/`` — the oracle, both drivers, the
  generated results and the full write-up, including the tuning sensitivity and
  the provenance trail.
* ``mlsynth/tests/test_lpca.py`` — the Kansas estimate, the re-centring
  invariant that the v1 defect violated, and the rank-rule properties.
