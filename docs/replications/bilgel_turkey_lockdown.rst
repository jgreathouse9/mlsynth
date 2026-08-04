.. _replication-bilgel-turkey-lockdown:

Partially pooled ASCM — Covid-19 lockdowns in Turkey (Bilgel 2022)
==================================================================

:Estimator: :doc:`../ppscm` — partially pooled synthetic control at
   ``nu = 0.5``, with ``inference_method="bootstrap"``.
:Source: Bilgel, F. (2022), *"Effects of Covid-19 lockdowns on social distancing
   in Turkey,"* Econometrics Journal 25(3):781–805,
   `10.1093/ectj/utac016 <https://doi.org/10.1093/ectj/utac016>`_.
:Replication type: Path A — the paper's published estimates on the author's own
   data, from his deposited replication package — and cross-validation against a
   pinned ``augsynth`` 0.2.0 run of his own call.
:Status: Verified against both. All six ATTs reproduce to 1.5e-4 against live
   ``augsynth`` and to within half the paper's last printed digit against the
   table, and live ``augsynth`` itself reproduces the table.

What the paper estimates
------------------------

Turkey imposed weekend and holiday lockdowns across 31 provinces in April and
May 2020. The question is how much of the observed collapse in movement the
lockdowns caused, as against what people would have done anyway. The paper
answers it by building a synthetic control for the treated provinces out of the
untreated ones, on Google Community Mobility Reports, and reading the gap over
the 17 post-lockdown days.

Six mobility series are estimated separately: retail and recreation, grocery and
pharmacy, parks, transit stations, workplaces, and residential. The first five
measure time spent away from home and fall under lockdown; residential measures
time spent at home and rises, which is why its sign is opposite.

Why this estimator
------------------

The paper's column 1 is ``multisynth`` from the ``augsynth`` R package — the
partially pooled estimator of Ben-Michael, Feller and Rothstein — at
``nu = 0.5``. Pooling matters here because there are 31 treated provinces, not
one. At ``nu = 0`` each treated province gets its own synthetic control, which
fits each one well and averages noisily; at large ``nu`` a single synthetic
control serves all of them, which is stable but fits none exactly. Half is the
author's choice between the two.

:doc:`../ppscm` is a port of that estimator, so this replication needed no new
code: the case reads the author's frames and fits them.

Two reference bases
-------------------

This page carries two comparisons. Comparing mlsynth to the printed table
measures two things at once — whether mlsynth implements this specification, and
whether the printed table was produced by it — and one number cannot separate
them. So the case also runs the author's own call through a commit-pinned
``augsynth`` 0.2.0 and compares against that.

The separation resolves cleanly here. Live ``augsynth`` reproduces every printed
figure to within rounding, so the table came from this call at this version.
Unlike the :doc:`Song et al. <song_ml_ascm>` case there is no drift between the
published artifact and the pinned package, which makes the live comparison the
binding one.

Results
-------

Every published estimate, against live ``augsynth`` and against what PPSCM
returns:

.. list-table::
   :header-rows: 1
   :widths: 24 14 20 20 11 11

   * - Mobility series
     - Paper
     - Live ``augsynth``
     - mlsynth
     - Paper SE
     - live SE
   * - Retail and recreation
     - −25.08
     - −25.0818493
     - −25.0818010
     - 5.49
     - 5.4937
   * - Grocery and pharmacy
     - −53.10
     - −53.1003895
     - −53.1004403
     - 10.43
     - 10.4301
   * - Parks
     - −33.45
     - −33.4503451
     - −33.4503250
     - 7.69
     - 7.6928
   * - Transit stations
     - −16.76
     - −16.7622384
     - −16.7622501
     - 3.90
     - 3.9005
   * - Workplaces
     - −27.61
     - −27.6115815
     - −27.6114318
     - 6.37
     - 6.3666
   * - Residential
     - 12.02
     - 12.0190048
     - 12.0189919
     - 2.04
     - 2.0410

Against live ``augsynth`` the largest distance is 1.5e-4, and the post-period
trajectories — 17 event times per outcome, 102 points — agree to 4.1e-4. Both
sit above the 1e-6 this project reaches on ridge ASCM. The gap is the quadratic
program: PPSCM and ``augsynth`` reach the partially pooled optimum through
different solvers at different stopping tolerances, which moves the last few
digits and nothing a reported estimate would notice.

The paper prints two decimals, so a published figure pins the true one only to
within 0.005. Every distance above is under that, on estimates ranging from
−53.10 to +12.02. This is the tightest agreement the published precision can
express — a closer claim would need the author's unrounded output, which the
package does not carry.

Standard errors
---------------

The standard errors agree to between 0.8 and 3.2 percent, and they are held to a
looser tolerance than the estimates for a reason that is not tolerance-shopping.
Column 1's errors come from a wild bootstrap. Reproducing them to the digit would
mean reproducing R's random number stream, not merely the estimator, so the
question the case can answer is whether the inference lands in the same place.
It does. A broken inference routine would miss by far more than three percent,
which is what the 10 percent tolerance is set to catch.

Two structural checks
---------------------

Two rows guard the design itself, and both come from the paper.

``n_treated_residential`` is 24 where every other series has 31. That is the
paper's own footnote a: residential mobility is reported for fewer provinces
than the rest. Reading the panel wrongly would collapse the difference.

``n_post_periods`` is 17, which is the post-lockdown window Table 3 states. It is
recomputed from the data — last period minus first treated period plus one — so
a misread adoption date would move it.

Reading the replication package
-------------------------------

Two features of the deposited package change what a reader has to do with it.

The ``.rdata`` files are cumulative workspace saves. The retail file holds one
frame, the grocery file two, the residential file all six. Loading one and taking
the first object returns a different outcome's panel, and the fit will succeed
and be wrong. Each frame has to be selected by name. The vendored
``basedata/bilgel_turkey_lockdown.parquet`` does that once and tags each panel
with its ``outcome``.

Sixteen provinces lift lockdown before the window closes, so the raw treatment
is not absorbing. The author handles this upstream, and Table 3 records the
action as "post-lockdown concatenation": the ``_t0c1`` frames reach the estimator
with a single adoption day and no reversal. The estimator therefore never sees
the on-and-off treatment that Table 3's attribute row marks ASCM as unable to
accommodate — the paper's own interactive-fixed-effects and matrix-completion
columns exist to cover that case.

Running the reference
---------------------

The reference is committed under
``benchmarks/reference/bilgel_turkey_lockdown/`` as
``gold_live_augsynth.csv`` and ``gold_live_trajectory.csv``, so the case runs in
CI without R. The script that produced them, ``reference.R``, is committed
alongside and reads the same vendored parquet the Python side reads, through
``nanoparquet``. One file, two sides: an earlier comparison in this project ran
33 donors on the R side and 37 on the Python side because each read its own copy
of the inputs, and reading one file makes that impossible.

To install the pinned reference and re-run it live::

   bash benchmarks/R/install_augsynth.sh
   MLSYNTH_BENCH_LIVE_R=1 python benchmarks/run_benchmarks.py \
       --case bilgel_turkey_lockdown

That regenerates the gold into a temporary directory and checks it against the
committed copy. A regenerated run currently reproduces it at 0.0, and the case
raises if that moves by more than 1e-8. It raises instead of reporting a row
because a row that exists only when R does could not be pinned in ``EXPECTED``,
and an unpinned row is one nobody checks.

Case
----

``benchmarks/cases/bilgel_turkey_lockdown.py``. Every row is a distance from a
reference, so a regression moves it and cannot be absorbed by re-fitting.
