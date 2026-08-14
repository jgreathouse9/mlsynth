.. _replication-cwz-ttest:

A t-test for synthetic controls (Chernozhukov, Wüthrich & Zhu)
==============================================================

:Estimator: :doc:`../vanillasc` — :class:`mlsynth.VanillaSC`, ``inference="ttest"``
:Source: Chernozhukov, V., Wüthrich, K., & Zhu, Y., *"A t-test for synthetic
   controls"* (arXiv:1812.10820), Tables 1, 3 and 5.
:Replication type: Path A on the authors' data and Path B on their simulation
   design, both cross-validated against their own R.
:Status: verified — the empirical estimate, the Table 3 Monte Carlo and the
   Table 1 efficiency formula all reproduced against live runs.
:Durable cases: `cwz_ttest <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/cwz_ttest.py>`__,
   `cwz_ttest_mc <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/cwz_ttest_mc.py>`__,
   `cwz_rae <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/cwz_rae.py>`__,
   `cwz_mc <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/cwz_mc.py>`__

What the method does
--------------------

A synthetic control gives an estimate but not a standard error, and the usual
routes to one are awkward here: there is a single treated unit, the
post-treatment window is short, and the weights are estimated from the same
pre-period the errors are judged against. The t-test replaces all of that with
cross-fitting.

Split the pre-period into :math:`K` blocks. For each block, refit the weights on
its complement and form the difference between the mean post-treatment gap and
the mean gap on the held-out block. That gives :math:`K` estimates of the same
effect, each computed from weights that never saw its own held-out block, so the
bias the fit induces is removed by the subtraction. Their mean is the debiased
ATT and their sample standard deviation supplies the standard error, giving a
statistic that is asymptotically :math:`t_{K-1}` — a familiar
:math:`\widehat{\tau} \pm t_{K-1}(1-\alpha/2)\,\mathrm{se}` with no long-run
variance to estimate and no requirement that the synthetic control be correctly
specified.

The price is that :math:`K` is a choice, and it is a real trade-off: more folds
shorten the interval and shorter blocks make each fold's estimate noisier. Table
1 quantifies the first half of that trade, and mlsynth's ``ttest_K="auto"``
implements a rule on top of it.

Path A — the Swedish carbon tax
-------------------------------

Table 5(a) applies the test to Andersson's (2019) study of Sweden's 1990 carbon
tax on transport CO\ :sub:`2` per capita: 15 countries, 1960–2005,
:math:`T_0 = 30`, :math:`T_1 = 16`, :math:`K = 3`, outcome-only weights. The
paper reports an ATT of −0.27 with a 90% interval of [−0.41, −0.14].

.. code-block:: python

   import pandas as pd
   from mlsynth import VanillaSC

   ct = pd.read_stata("basedata/carbontax_data.dta")
   ct["treated"] = ((ct.country == "Sweden") & (ct.year >= 1990)).astype(int)
   res = VanillaSC({
       "df": ct, "outcome": "CO2_transport_capita", "treat": "treated",
       "unitid": "country", "time": "year", "backend": "outcome-only",
       "inference": "ttest", "ttest_K": 3, "alpha": 0.1, "display_graphs": False,
   }).fit()

   res.inference.details["att_debiased"]                    # -0.273903
   res.inference.ci_lower, res.inference.ci_upper           # -0.406425, -0.141380

Those are the authors' ``scinference`` outputs, captured live in
``benchmarks/reference/cwz_ttest/``, not the paper's rounded cells. The same
case also pins the debiased ATT on the Basque Country (−0.657511) and California
Proposition 99 (−17.991121).

Path B — Table 3, live
----------------------

Table 3 calibrates a simulation to the same carbon tax panel: a four-factor
model fitted to the detrended controls, an AR(1) per control unit, and an AR(1)
for the SC prediction errors. Nine DGPs vary whether a simplex synthetic control
can span the treated unit and how the panel departs from stationarity. The
treatment effect is zero, so what is measured is the coverage of a nominal 90%
interval.

``cwz_ttest_mc`` runs the authors' own ``calibration_dgps.R`` and
``common_functions.R``, and separates two claims that a single end-to-end rate
cannot tell apart. Ten seed-matched panels per DGP are dumped with the ATT and
standard error R computed on each, and mlsynth is handed those exact panels: the
debiased t-test is a deterministic function of a panel, so those must agree
exactly, and they do to 1e−11. Then mlsynth draws its own panels from its own
port of the design and reproduces the reference's coverage, length and bias
across all nine DGPs.

The geometry the paper reports comes through. Coverage is near nominal where the
theory covers the design (DGP 1: 0.92 against 0.88; DGP 2: 0.92 against 0.91),
mildly short under a common trend with one deviating donor (DGP 6: 0.82 against
0.82), and short by a wide margin under heterogeneous trends (DGP 8: 0.56
against 0.63), which lies outside the theory and is the paper's stress case.
The recovered :math:`\rho_u = 0.3125` matches the paper's reported ~0.31.

``cwz_mc`` remains alongside it, tracking the paper's printed cells from a
Python reimplementation of the design. The two fail differently: one would catch
a drift away from what the paper reports, the other a drift away from what the
authors' code computes.

Path B — Table 1, the efficiency formula
----------------------------------------

Table 1 is the relative asymptotic efficiency of the :math:`K`-fold interval:
the ratio of its limiting expected length as :math:`K \to \infty` to its length
at finite :math:`K`, a closed form in :math:`K`, the level and
:math:`c_0 = T_0/T_1`. ``cwz_rae`` runs the authors' ``RAE.R`` and matches
:func:`mlsynth.utils.inferutils.rae` to 1e−9 across :math:`K = 2, \dots, 10` at
their :math:`c_0 = 30/16`:

.. list-table::
   :header-rows: 1
   :widths: 12 12 12 12 12 12 12 12 12

   * - :math:`K=2`
     - 3
     - 4
     - 5
     - 6
     - 7
     - 8
     - 9
     - 10
   * - 0.327
     - 0.636
     - 0.759
     - 0.821
     - 0.858
     - 0.882
     - 0.900
     - 0.913
     - 0.923

Two folds cost most of the efficiency, three recovers two thirds of it, and the
returns flatten past six. That shape is why the paper's guidance settles on a
small :math:`K` and why ``select_K`` treats three as the floor. The rule was
exercised at every level before this except against the table it comes from.

What the reproduction found
---------------------------

``scinference`` solves the synthetic control through ``limSolve::lsei``, whose
``lsei_type`` argument selects between two solvers, and the authors' simulation
scripts pass ``type = 1``. On the real carbon tax panel that is fine — Path A
above cross-validates it to 4e−7. On the simulated panels it is not: type 1
warns "inequalities contradictory" on the fold refits and returns a solution
that ignores the non-negativity constraint. Over the ten dumped DGP 8 draws it
does so on five of thirty fold refits, with one weight reaching −32.4, and those
are exactly the draws where the R and Python answers separate. The consequence
is visible in the aggregate: under type 1 the misspecified DGPs report a mean
interval length of 103 and a mean bias of −8.5 on an outcome of order one, where
type 2 gives 0.66 and 0.006.

Weights off the simplex are not a synthetic control, so what type 1 returns
there is an answer to a different problem. The reference run uses ``type = 2``,
which solves the stated program on every draw, and records the off-simplex count
in its output so the choice is a measurement and not a claim. The Path A case is
unaffected and keeps ``type = 1``.

Reproducing it
--------------

.. code-block:: bash

   bash benchmarks/R/install_scinference.sh
   python benchmarks/reference/generate.py cwz_ttest
   python benchmarks/reference/generate.py cwz_ttest_mc
   python benchmarks/reference/generate.py cwz_rae
   python benchmarks/run_benchmarks.py --case cwz_ttest
   python benchmarks/run_benchmarks.py --case cwz_ttest_mc
   python benchmarks/run_benchmarks.py --case cwz_rae
