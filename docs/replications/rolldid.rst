ROLLDID — Lee & Wooldridge rolling-transformation DiD
=====================================================

.. currentmodule:: mlsynth

Path A (empirical), both applications of Lee & Wooldridge (2026), *Simple
Approaches to Inference with Difference-in-Differences Estimators with Small
Cross-Sectional Sample Sizes*. ``ROLLDID`` is a **clean-room MIT**
implementation written from the paper's equations; the AGPL ``lwdid`` package
was used during development **only as a black-box oracle** (its point estimates
agreed to ~4 decimals) and shares no code with mlsynth.

California Prop 99 (common timing, Table 3)
-------------------------------------------

Data: ``basedata/smoking_data.csv`` — the Abadie, Diamond & Hainmueller (2010)
panel, 39 states × 1970–2000, California treated from 1989; outcome = log
per-capita cigarette sales, 38 never-treated controls.

================================  =========================  ===========================
Quantity                          ROLLDID                    Paper (Table 3)
================================  =========================  ===========================
Demeaning (Proc 2.1) avg ATT      −0.422 (se 0.121)          −0.422 (0.121)
Detrending (Proc 3.1) avg ATT     −0.227 (se 0.094)          −0.227 (0.094)
Detrend exact :math:`p`           0.021                      0.021
:math:`\tau_{1989}` (demean)      −0.168                     −0.168
:math:`\tau_{2000}` (demean)      −0.667                     −0.667
:math:`\tau_{2000}` (detrend)     −0.403                     −0.403
================================  =========================  ===========================

Castle laws (staggered rollout, §7.2)
-------------------------------------

Data: ``basedata/castle.csv`` — Cunningham (2021), 50 states × 2000–2010, 21
states adopting "castle" laws across 2005–2009 and 29 never-treated; outcome =
log homicides per 100k. The cohort-share-weighted aggregate (eq. 7.18–7.19) uses
never-treated states as the comparison.

================================  =========================  ===========================
Quantity                          ROLLDID                    Paper (§7.2)
================================  =========================  ===========================
Demeaning aggregate ATT           0.092 (OLS se 0.057)       0.092 (0.057), t ≈ 1.61
Demeaning HC3 :math:`t`           1.50                       1.50
Detrending aggregate ATT          0.067 (HC3 se 0.055)       0.067 (0.055), t ≈ 1.21
================================  =========================  ===========================

Reproduce
---------

.. code-block:: bash

   python benchmarks/run_benchmarks.py rolldid_lw

The durable case is ``benchmarks/cases/rolldid_lw.py``; unit-level reproduction
is pinned in ``mlsynth/tests/test_rolldid.py``. Both run on the in-repo data, so
no network or external reference is required.
