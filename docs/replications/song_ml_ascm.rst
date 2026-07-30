.. _replication-song-ml-ascm:

Ridge ASCM — clean winter heating in China (Song et al. 2023)
==============================================================

:Estimator: :doc:`../vanillasc` — the ridge-augmentation layer
   (``augment="ridge"``), with ``inference="jackknife_plus"`` for the interval
   columns.
:Source: Song, Liu, Cheng, Cole, Dai, Elliott & Shi (2023), *"Attribution of Air
   Quality Benefits to Clean Winter Heating Policies in China,"* Environmental
   Science & Technology 57:17707–17717,
   `10.1021/acs.est.2c06800 <https://doi.org/10.1021/acs.est.2c06800>`_
   (open access).
:Replication type: Path A — the authors' published results on their own data —
   and cross-validation against a live ``augsynth`` 0.2.0 run on the same cells.
:Status: Verified, with a documented and quantified disagreement between the
   authors' published artifact and the package it names.

Why two reference bases
-----------------------

This page carries two comparisons rather than one, because the obvious single
comparison cannot answer the question anyone actually has.

The authors' method is a two-stage pipeline they call ML-ASCM: a random-forest
weather normalization, then Ridge Augmented SCM through the ``augsynth`` R
package. Comparing mlsynth to their published ``main_result.csv`` measures the
sum of two things — whether mlsynth implements ridge ASCM correctly, and whether
the artifact they published was produced by the version of ``augsynth`` that is
pinned here. Those are different questions with different right answers, and a
single number cannot separate them.

So the case measures three quantities:

.. list-table::
   :header-rows: 1
   :widths: 34 22 44

   * - Comparison
     - Agreement
     - What it answers
   * - mlsynth vs live ``augsynth`` 0.2.0
     - tight
     - Does mlsynth implement this specification?
   * - live ``augsynth`` vs the published CSV
     - loose
     - Was the artifact produced by this version of it?
   * - mlsynth vs the published CSV
     - loose
     - Path A — the sum of the two above

The third is bounded by the first two, which is checkable rather than asserted,
and each fails independently. That matters: with only the published basis, a real
regression in mlsynth could hide inside a tolerance widened to accommodate the
known drift.

What is reproduced
------------------

The design is transcribed from the authors' ``main_result.R`` rather than
inferred from the paper's prose. For each of 8 heating-year windows (1 May to
30 April) and each of 8 treatment groups, treatment is marked from 23 October of
the starting year, the fit runs against a fixed pool of 37 southern control
cities, and the whole thing repeats for each of 16 pollutant series. That is
:math:`8 \times 8 \times 16 = 1024` ``augsynth`` fits, which is exactly what
``main_result.csv`` contains.

The first stage is not reimplemented and does not need to be: the authors ship
the deweathered series, vendored here as
``basedata/song_ml_ascm_china.parquet``, so the synthetic-control half is
reproducible on its own.

Two details about the units change the estimand and are easy to miss. The
treated unit in every cell is a pre-aggregated regional average — ``"2+26
cities"`` and ``"Northern"`` are themselves rows in the panel, not collections
resolved at fit time. And the sign convention is theirs: a positive ATT means
winter heating raised the pollutant, because the counterfactual is the
no-heating trajectory.

Where the published values drift
--------------------------------

Across all 1024 cells, 70 percent reproduce the published value to
:math:`10^{-5}`. The breakdown by heating year is sharper than that average
suggests:

.. list-table::
   :header-rows: 1
   :widths: 14 20 20 20 26

   * - Year
     - max :math:`|\Delta|`
     - mean :math:`|\Delta|`
     - within :math:`10^{-5}`
     -
   * - 2014
     - 0.330
     - 0.0113
     - 79.7%
     -
   * - 2015
     - 0.187
     - 0.0084
     - 77.3%
     -
   * - 2016
     - 1.450
     - 0.2010
     - 0.0%
     - every cell disagrees
   * - 2017
     - 0.172
     - 0.0099
     - 78.9%
     -
   * - 2018
     - 0.136
     - 0.0057
     - 82.8%
     -
   * - 2019
     - 0.276
     - 0.0071
     - 77.3%
     -
   * - 2020
     - 0.280
     - 0.0087
     - 76.6%
     -
   * - 2021
     - 0.319
     - 0.0101
     - 83.6%
     -

There are two patterns here, not one. A broad tail of roughly 20 percent appears
in every year, and 2016 fails wholesale — none of its 128 cells reproduces to
:math:`10^{-5}`, and its worst reaches 1.45 on an ATT of about 25, some 6
percent. 2016 accounts for about 40 percent of the non-reproducing cells.

The pre-treatment imbalance is the quantity that localises this. It agrees to
:math:`6.2 \times 10^{-4}` on every cell, 2016 included. The same optimum value
is reached even where the reported effect differs, which places the disagreement
in the ridge penalty rather than in the fit.

The cause is reference-version drift. The authors ran whatever ``augsynth`` was
current in 2022–2023, and its ridge cross-validation has changed since — see
:doc:`ascm_ridge_cv` for two conventions in that code path that move the selected
penalty. Against a live ``augsynth`` 0.2.0 run at the pinned commit, mlsynth
agrees on these same cells, 2016 included.

What mlsynth agrees with, and the one cell it does not
------------------------------------------------------

Against the live ``augsynth`` 0.2.0 run on the same 30 stratified cells:

.. list-table::
   :header-rows: 1
   :widths: 34 22 44

   * - Quantity
     - Worst over 29 cells
     -
   * - average ATT
     - :math:`6.5 \times 10^{-6}`
     - median :math:`4.3 \times 10^{-8}`
   * - pre-fit :math:`L_2` imbalance
     - :math:`9.3 \times 10^{-7}`
     - in both directions
   * - selected ridge penalty
     - :math:`5.8 \times 10^{-10}`
     - all 30 cells, degenerate one included

The ATT and imbalance rows are a solver floor rather than a modelling difference.
``augsynth`` solves the simplex program with ``quadprog`` and mlsynth with an
active-set method; the residual disagreement runs in both directions, which is
what a floor looks like and what a systematic difference would not.

The penalty row is the informative one. It is selected before the quadratic
program is solved, so it has no solver floor to hide behind, and it agrees on
every cell.

One cell of the 30 disagrees beyond that floor: ``Southern control`` at 2015
PM2.5, where mlsynth gives 0.0792 and ``augsynth`` 0.0033. It is not a defect,
and the reason is structural. ``Southern control`` is an aggregate of the donor
pool being used as its own treated unit, so it lies inside the donors' convex
hull by construction and the simplex optimum is not unique. Its ``scaled_l2`` is
:math:`6.3 \times 10^{-6}` against :math:`6.0 \times 10^{-3}` for the next
smallest of the 30 — a factor of 964, so calling it degenerate is reading the
data rather than choosing a threshold.

On the objective actually being minimised, mlsynth is the closer of the two:
its pre-treatment imbalance is :math:`2.1 \times 10^{-15}` where ``augsynth``
stops at :math:`4.9 \times 10^{-5}`. Both effectively fit perfectly and then
extrapolate from different members of the same optimal set. The case pins each
side's own imbalance rather than the difference between them, so the cell is
reported rather than quietly dropped.

Two corrections worth recording
-------------------------------

Both are about this explanation, and the second was made while fixing the first.

An earlier version of the case attributed the whole disagreement with the
published values to non-unique simplex optima. That was fitted to two cells, and
running all 1024 refuted it as a general account: the worst-disagreeing cells are
well conditioned, with gold ``Scaled_L2`` between 0.25 and 0.60 — the worst of
all, ``2+26 cities`` / 2016 / ``PM2.5``, sits at 0.333. Non-uniqueness would have
produced a geometric pattern; the data shows a temporal one.

The next version then declared non-uniqueness refuted outright, which overshot in
the other direction. The live comparison above found precisely one cell where it
is the right explanation. Non-uniqueness explains that cell and does not explain
the 2016 drift; both statements hold and neither generalises to the other.

The first mistake is the failure mode ``agents_tests.md`` step 0 exists to
prevent: confirm the reference implements the same version of the specification
before comparing bit for bit. Skipping it produced a plausible story about solver
geometry when the answer was that the artifact predates the package. The second
is the failure mode of over-correcting — discarding an explanation entirely
because it was over-applied.

What the cross-validation fix bought
------------------------------------

The cells that do reproduce, reproduce to about :math:`10^{-6}`, and that is
recent. Before two defects in the ridge penalty's cross-validation were fixed —
a fold off-by-one and a population-versus-sample standard error, both in
:doc:`ascm_ridge_cv` — the 2015 PM2.5 cell disagreed by 4.3 percent,
:math:`+18.744` against the published :math:`+17.969`. It now agrees to
:math:`10^{-7}`.

These panels are exactly the shape that exposed those defects: 25 pre-treatment
periods against 37 donors, with the final pre-period sitting on the seasonal ramp
into the heating season, so the fold ``augsynth`` never holds out is the one
carrying nine times the average error. That is why this panel joined
:doc:`ascm_ridge_cv` as a second test panel — the Kansas panel cannot detect
those defects at all.

Durable cases & tests
---------------------

* ``song_ml_ascm`` — a stratified 30-cell subset, cheap enough for the routine
  sweep (``benchmarks/cases/song_ml_ascm.py``). The strata sweep each dimension
  independently: all 8 groups at one pollutant-year, all 16 pollutants for one
  group-year, all 8 years for one group-pollutant, so a defect confined to one
  group, pollutant or year is still caught.
* ``benchmarks/reference/song_ml_ascm/run_full_sweep.py`` — all 1024 cells, not
  part of ``run_benchmarks``. This is what produced the table above.
* ``benchmarks/reference/song_ml_ascm/reference.R`` — regenerates the live
  ``augsynth`` gold; install the reference first with
  ``bash benchmarks/R/install_augsynth.sh``. The captured gold is committed, so
  the case and the tests run without an R toolchain.
* The ridge penalty's cross-validation against this panel is pinned separately in
  ``mlsynth/tests/test_ascm_ridge_cv.py``.
