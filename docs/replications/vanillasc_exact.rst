.. _replication-vanillasc-exact:

VanillaSC exact backend — Abadie and Shang (2026)
=================================================

:Estimator: :doc:`../vanillasc` — :class:`mlsynth.VanillaSC` with ``backend="exact"``
:Source: Alberto Abadie and Yucheng Shang (2026), "Synthetic Control: An Exact
   Geometric Formulation," arXiv:2610.26576, and their replication package
   `lychemon/synth-AS-replication <https://github.com/lychemon/synth-AS-replication>`__
   (v1.0.0).
:Replication type: Path A — the paper's two applications on the authors' own
   data — checked number by number against the authors' replication run.
:Status: Verified. Every donor weight, loss, mean gap, face count and
   tightly solved predictor weight the run writes is reproduced; the analytic
   centres agree to the authors' solver tolerance, and on the authors' own
   programs mlsynth's centres score at least as well as theirs.

Validation strategy
-------------------

The authors publish ``code/scripts/run_all.py``, which regenerates every
data-based table and figure of the paper and ends with ``verify_results.py``,
a report that lists the numbers the paper prints next to the values just
computed. The reference here is that script, run unchanged at commit
``791a3865fe3e9fd0b82a8121c0b338791b7ffa39`` (v1.0.0) with
``lychemon/synthx_package`` at ``a10a28c5`` beside it (``verify_results.py``
also compares the replication's solvers with the Stata bridge of that
package). ``benchmarks/reference/vanillasc_exact/reference.py`` fetches both
repositories at those commits, runs ``run_all.py`` in a scratch copy, requires
its report to read "every reported number reproduced", and records the 633
numbers its ``output/`` holds, together with the authors' predictor-weight
functions called at full precision and their positive definite matrices entry
by entry. The report itself is kept verbatim in ``reference.out``.

``VanillaSC`` reads the same two panels. ``basedata/abadie_shang_smoking.csv``
and ``basedata/abadie_shang_germany.csv`` are the replication's
``data/smoking_data.csv`` and ``data/german_reunification.csv`` copied byte for
byte (CC BY 4.0); the bundle's provenance records their checksums, and the
capture refuses to run if the commit's files differ from them. The older
Proposition 99 and Germany covariate files in ``basedata/`` store several
columns as 32-bit floats, which moves the predictors by up to
:math:`5 \times 10^{-7}`.

The comparison is the durable case
`benchmarks/cases/vanillasc_exact.py <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/vanillasc_exact.py>`__,
asserted metric by metric in ``mlsynth/tests/test_vanillasc_exact_replication.py``.

The specification
-----------------

The predictors follow the authors' ``data_preparation.py``. Each is averaged
over its window, missing values omitted, and divided by its standard deviation
across all units (ddof=1), which is what ``VanillaSC`` does with
``covariate_windows``.

.. list-table:: Proposition 99: California treated from 1989, validation 1970-1988
   :header-rows: 1
   :widths: 22 18 60

   * - Covariate
     - Window
     - The authors' predictor
   * - ``lnincome``
     - 1980-1988
     - Log per-capita state income
   * - ``age15to24``
     - 1980-1988
     - Share of population aged 15-24
   * - ``retprice``
     - 1980-1988
     - Retail cigarette price
   * - ``beer``
     - 1984-1988
     - Per-capita beer consumption
   * - ``cigsale_1988``, ``cigsale_1980``, ``cigsale_1975``
     - one year each
     - Cigarette sales per capita in 1988, 1980 and 1975

.. list-table:: West Germany: treated from 1991, validation 1981-1990 (``fit_window``)
   :header-rows: 1
   :widths: 22 18 60

   * - Covariate
     - Window
     - The authors' predictor
   * - ``gdp``, ``trade``, ``infrate``, ``industry``
     - 1971-1980
     - Per-capita GDP, trade openness, inflation, industry share
   * - ``schooling``
     - 1970, 1975
     - Schooling (reported every five years)
   * - ``invest70``
     - 1980
     - Investment rate (1970s)

The leakage-free California sample of the paper's Section 6 drops 1975, 1980
and 1988 from validation, the years whose outcomes are predictors:
``fit_exclude=[1975, 1980, 1988]``. The California call, as the case builds it:

.. code-block:: python

   import pandas as pd
   from mlsynth import VanillaSC

   url = ("https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
          "refs/heads/main/basedata/abadie_shang_smoking.csv")
   df = pd.read_csv(url)
   df["year"] = df["year"].astype(int)
   df["treated"] = ((df.state == "California") & (df.year >= 1989)).astype(int)
   for year in (1988, 1980, 1975):
       df[f"cigsale_{year}"] = df["cigsale"]         # read in its own year below

   windows = {"lnincome": (1980, 1988), "age15to24": (1980, 1988),
              "retprice": (1980, 1988), "beer": (1984, 1988),
              "cigsale_1988": (1988, 1988), "cigsale_1980": (1980, 1980),
              "cigsale_1975": (1975, 1975)}
   res = VanillaSC({
       "df": df, "outcome": "cigsale", "treat": "treated", "unitid": "state",
       "time": "year", "backend": "exact", "covariates": list(windows),
       "covariate_windows": windows, "exact_ranges": True,
       "exact_face_counts": True, "inference": False, "display_graphs": False,
   }).fit()

Results
-------

Donor weights, validation losses and mean gaps (Tables 1, 5, D.1 and D.2). The
mean post-treatment gap is ``VanillaSC``'s ATT (1989-2000 and 1991-2003).

.. list-table::
   :header-rows: 1
   :widths: 30 18 18 34

   * - Quantity
     - mlsynth
     - ``run_all.py``
     - Largest gap
   * - California, diagonal: loss
     - 58.456614003799
     - 58.456614003829
     - weights :math:`8 \times 10^{-13}`
   * - California, diagonal: mean gap
     - -18.981600
     - -18.981600
     - :math:`8 \times 10^{-13}`
   * - California, positive definite: loss
     - 54.703457873090
     - 54.703457873090
     - weights :math:`3 \times 10^{-11}`
   * - California, positive definite: mean gap
     - -19.779984
     - -19.779984
     - :math:`2 \times 10^{-10}`
   * - California, leakage-free: loss
     - 55.766850861594
     - 55.766850861611
     - weights :math:`4 \times 10^{-13}`
   * - West Germany, diagonal: loss
     - 45803.567601519
     - 45803.567601519
     - weights :math:`4 \times 10^{-11}`
   * - West Germany, diagonal: mean gap
     - -1391.2137
     - -1391.2137
     - :math:`8 \times 10^{-8}`
   * - West Germany, positive definite: loss
     - 20286.569521049
     - 20286.569521049
     - weights :math:`9 \times 10^{-10}`
   * - West Germany, positive definite: mean gap
     - -1582.1299
     - -1582.1299
     - :math:`6 \times 10^{-7}`

The weights themselves: California's diagonal synthetic is Utah 0.3351, Nevada
0.2356, Montana 0.2019, Colorado 0.1595 and Connecticut 0.0679, and its
positive definite one Utah 0.3194, Montana 0.3015, Nevada 0.2484, Connecticut
0.1025 and Colorado 0.0282. West Germany's diagonal synthetic is Austria 0.5074,
Switzerland 0.1659, Japan 0.1464, the USA 0.1351 and Australia 0.0452, and its
positive definite one Switzerland 0.3005, Japan 0.2058, Austria 0.1906, Norway
0.1608, the Netherlands 0.1330 and Belgium 0.0092. The positive definite search
sees the same 132 and 39 visible facets as the authors'.

Predictor weights (Tables 2, 4 and 6). ``run_all.py`` writes them to four
decimals; the bundle also holds the authors' functions run at full precision.

* The positive definite log-det maximiser and the identified ranges, which the
  authors solve at tolerances of :math:`10^{-12}`, agree to
  :math:`1.2 \times 10^{-8}` (log-det) and :math:`1.0 \times 10^{-8}` (ranges)
  at full precision, and to rounding at four decimals. Table 4's largest
  income share over the rationalisers, 0.038, and its smallest eigenvalue,
  :math:`3.7 \times 10^{-3}`, come out as printed.
* California's diagonal log-det maximiser puts one third on each of the three
  predictors the rationalisers can weight (the youth share, the retail price and
  1975 sales) and zero on the other four, with all 33 inactive donors' slacks
  vanishing identically: the paper's Table 2. West Germany's diagonal analytic
  centre is Table 6's first column.
* The analytic centres and the diagonal log-det maximiser agree to
  :math:`2.3 \times 10^{-5}` at full precision; see below.

Faces of the donor hull (Table 3). With ``exact_face_counts=True`` the counts
match the authors' for both applications, dimension by dimension: California's
hull has 1,854 facets, 132 of them visible from California, and 14 visible
facets, 49 ridges and 69 four-dimensional faces hold attainable points.

Where the numbers differ, and why
---------------------------------

The analytic centres. The authors compute the analytic centres, and the
diagonal log-det maximiser, with Clarabel at its default tolerances, while the
positive definite log-det maximiser gets tolerances of :math:`10^{-12}`. Their
centres carry solver error of order :math:`10^{-5}`: the relative norm of the
centre's projected gradient, which is zero at the centre, is
:math:`2.5 \times 10^{-6}` for theirs on California and
:math:`1.5 \times 10^{-4}` on West Germany, against
:math:`7 \times 10^{-9}` and :math:`1 \times 10^{-11}` for mlsynth's, and the
centre objective :math:`\log\det\mathbf{V} + \sum_j \log s_j(\mathbf{V})` is
higher at mlsynth's point by :math:`1.3 \times 10^{-7}` and
:math:`9.5 \times 10^{-8}`. One coordinate rounds differently at four decimals:
West Germany's positive definite centre puts 0.176358 on GDP, which is 0.1764,
where ``run_all.py`` writes 0.1763 (its value is 0.176346). The paper prints
0.176, as both round to. The case asserts, for each of the four programs, that
mlsynth's point scores at least as well as the authors' on the authors'
objective.

The facet of a positive definite optimum. California's positive definite
optimum puts weight on five donors, and the face they span lies in four visible
facets, each with the same fit. ``run_all.py`` reports the facet
{Colorado, Connecticut, Idaho, Montana, Nevada, New Mexico, Utah}: its search
pops the four in order of a bound and keeps a later one only when its fit is
strictly lower, so which one it keeps is decided by its quadratic solver's
rounding. mlsynth reports all four (``res.exact.facets``); the authors' is
among them. West Germany's optimum has six donors, one facet, and the same
facet on both sides.

Table 4's beer share. The paper prints 0.145 for the log-det maximiser's
weight on beer, which is 0.144465 rounded to four decimals and then to three.
``verify_results.py`` checks the four decimals, as the test does.

Runtime
-------

Single machine, the authors' run and mlsynth's fits back to back. The authors'
times are from their own run log; mlsynth's are warm ``VanillaSC.fit()``
calls, data preparation and both sets of predictor weights included.

.. list-table::
   :header-rows: 1
   :widths: 46 27 27

   * - Task
     - ``run_all.py``
     - mlsynth
   * - California, diagonal donor weights
     - 1.6 s (search)
     - 0.21 s (whole fit)
   * - West Germany, diagonal donor weights
     - 0.9 s (search)
     - 0.11 s (whole fit)
   * - Both applications, positive definite
     - 1.7 s (script)
     - 0.14 s and 0.07 s
   * - Table 3, California
     - 395 s (exhaustive sweep)
     - 8.3 s, with the ranges
   * - Table 3, West Germany
     - 15 s
     - 1.6 s, with the ranges
   * - In-space placebo, California
     - not computed
     - 19 s diagonal, 16 s positive definite

Four of California's donors lie inside the hull of the others, so the placebo
runs the interior search four times. In the diagonal class those four take
1.6 s and the other 34 refits 14.7 s; in the positive definite class the four
interior refits take 11.0 s, the semidefinite attainability tests of the cells,
and the other 34 take 4.8 s.

Each face fit is now certified optimal (the KKT conditions' backward error,
checked on the face's own data) in place of a cap on its condition number, which
lets the backend fit panels such as the Basque one. Measured back to back on one
machine, the certificate costs about a quarter on the diagonal class: California's
fit went from 0.22 s to 0.28 s and its in-space placebo from 23.6 s to 30.0 s. The
positive definite class is unchanged (0.16 s; placebo 22.0 s and 23.6 s).

Reproducing
-----------

.. code-block:: bash

   python benchmarks/reference/generate.py vanillasc_exact   # the authors' run, about 7 minutes
   python benchmarks/run_benchmarks.py --case vanillasc_exact
   python -m pytest mlsynth/tests/test_vanillasc_exact_replication.py -q

The unit tests of the backend read the authors' answers on 27 exterior and 28
interior designs, captured from their code at the same commits by
``benchmarks/reference/vanillasc_exact/capture_designs.py``.

What it confirms
----------------

``VanillaSC(backend="exact")`` solves the paper's programs to the authors'
answers on their own data, in both predictor-weight classes and in the
leakage-free sample, and returns the predictor weights and diagnostics the
paper reports. The two applications both put the treated unit outside the
donor hull; the interior case is checked against the authors' ``synthx``
package on Proposition 99's in-space placebos and on seeded designs, in
``mlsynth/tests/test_exact_interior.py`` and ``test_exact_interior_pd.py``.
