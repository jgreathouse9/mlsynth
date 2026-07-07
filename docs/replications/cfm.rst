.. _replication-cfm:

CFM — Causal Inference Using Factor Models (Bai & Wang 2026)
============================================================

:Estimator: :doc:`../cfm` — :class:`mlsynth.CFM`
:Source: Bai, J., & Wang, P. (2026), *"Causal Inference Using Factor
   Models"* [BaiWang2026]_.
:Replication type: Path A — the paper's two empirical applications
   (California Prop 99 and German reunification) reproduced on the authors'
   data.
:Status: Verified — factor counts, the structural-break diagnostics, and
   the intercept-shift tests reproduce the paper's reported numbers.

Validation strategy
-------------------

Bai and Wang re-analyze the two canonical synthetic-control panels — Abadie,
Diamond and Hainmueller's California tobacco-control (Proposition 99) and
German reunification data — with the causal factor model. The paper does not
release code, so the target is the set of numbers it reports: the number of
factors chosen by the Ahn-Horenstein criteria, the Chow structural-break
statistic at the intervention date, the Quandt likelihood-ratio break date,
and the intercept-shift t-statistics. These are sensitive to the extracted
factor, so matching them confirms the factor extraction and the treated
regressions, not just an endpoint.

Both datasets ship with mlsynth: ``basedata/smoking_data.csv`` (California
plus 38 control states, 1970-2000) and ``basedata/german_reunification.csv``
(West Germany plus 16 control countries, 1960-2003).

Path A — California Prop 99
---------------------------

.. code-block:: python

   import pandas as pd
   from mlsynth import CFM
   from mlsynth.utils.cfm_helpers.setup import prepare_cfm_inputs
   from mlsynth.utils.cfm_helpers.factors import extract_cfm_factors
   from mlsynth.utils.cfm_helpers.pipeline import chow_break_statistic

   df = pd.read_csv("basedata/smoking_data.csv")
   df["treat"] = ((df.state == "California") & (df.year >= 1989)).astype(int)

   res = CFM({"df": df, "outcome": "cigsale", "treat": "treat",
              "unitid": "state", "time": "year", "n_factors": 1,
              "display_graphs": False}).fit()
   res.att, res.metadata["kappa_t"], res.metadata["chow_fstat"]

The intervention starts in 1989, so :math:`T_0 = 1988` with a 1970-1988
pre-period and a 1989-2000 post-period. The reproduced quantities:

.. list-table::
   :header-rows: 1
   :widths: 44 28 28

   * - Quantity
     - CFM (mlsynth)
     - Bai & Wang
   * - ER factor count
     - 1
     - 1
   * - GR factor count
     - 1
     - 1
   * - Chow F, break at 1989 (1 factor)
     - 16.84
     - 16.84
   * - QLR sup-F break date (15% trim)
     - 1984
     - 1984
   * - intercept-shift :math:`t(\kappa)`, 1 factor
     - 1.38
     - 1.38
   * - intercept-shift :math:`t(\kappa)`, 2 factors
     - 0.10
     - 0.10

The one-factor systematic effect path has a post-period mean of about
:math:`-20.7` packs and tracks the synthetic-control estimate
(:math:`\approx -19.5`) at correlation :math:`0.80`, consistent with the
paper's Figure 5. The small intercept-shift t-statistics indicate little
evidence of a post-treatment level break, so the effect operates through the
loading change.

Path A — German reunification
-----------------------------

.. code-block:: python

   df = pd.read_csv("basedata/german_reunification.csv")
   # paper convention: 1990 marked; treated periods 1991-2003 (T0 = 1990)
   df["treat"] = ((df.country == "West Germany") & (df.year >= 1991)).astype(int)

   res = CFM({"df": df, "outcome": "gdp", "treat": "treat",
              "unitid": "country", "time": "year", "n_factors": 1,
              "display_graphs": False}).fit()

.. list-table::
   :header-rows: 1
   :widths: 44 28 28

   * - Quantity
     - CFM (mlsynth)
     - Bai & Wang
   * - ER / GR factor count
     - 1 / 1
     - 1 / 1
   * - Chow F, break at 1991 (1 factor)
     - 634.5
     - 634.5
   * - QLR sup-F break date (15% trim)
     - 1993
     - 1993
   * - intercept-shift :math:`t(\kappa)`, 1 factor
     - 11.77
     - 11.81

The one-factor path tracks synthetic control at correlation :math:`0.98`.
Unlike California, the intercept-shift test is strongly significant, matching
the paper's finding of a post-treatment level shift for West Germany (either
a direct intercept change or a constant shift in the factor process, which
the specification cannot separate).

Seam notes
----------

Two details were pinned while reproducing these numbers, and both are held by
unit tests:

* The intercept-shift t-statistic reproduces only under the paper's
  block-additive heteroskedasticity-robust construction —
  :math:`\mathrm{Var}(\widehat\kappa) = \mathrm{Var}(\widehat a_1(0)) +
  \mathrm{Var}(\widehat a_1(1))` from separate pre- and post-regressions,
  each with an HC1 sandwich (appendix A.14-A.19). A naive pooled-OLS
  t-statistic is badly off on the heteroskedastic German panel (about 21
  against the reported 11.8); the robust block form recovers 11.77.
* The German ``treat`` flag follows the paper's convention: 1990 is the
  marked reunification year but the treated periods run 1991-2003, so
  :math:`T_0 = 1990` and the Chow break is tested at 1991 (matching the
  reported F of 634.5).

Not reproduced here
^^^^^^^^^^^^^^^^^^^

* The per-period confidence bands' factor-estimation component :math:`V^f`
  (appendix A.20) is implemented but not separately checked against a paper
  number; its acceptance target is the Monte Carlo coverage of Section 6,
  left as a future simulation benchmark.
* The potential-factors regime (Proposition 2, large treated cross section)
  is out of scope for this estimator.

The durable check lives in ``benchmarks/cases/cfm.py``.
