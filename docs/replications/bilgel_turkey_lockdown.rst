.. _replication-bilgel-turkey-lockdown:

Partially pooled ASCM — Covid-19 lockdowns in Turkey (Bilgel 2022)
==================================================================

:Estimator: :doc:`../ppscm` — partially pooled synthetic control at
   ``nu = 0.5``, with ``inference_method="bootstrap"``.
:Source: Bilgel, F. (2022), *"Effects of Covid-19 lockdowns on social distancing
   in Turkey,"* Econometrics Journal 25(3):781–805,
   `10.1093/ectj/utac016 <https://doi.org/10.1093/ectj/utac016>`_.
:Replication type: Path A — the paper's published estimates on the author's own
   data, from his deposited replication package.
:Status: Verified. All six reported ATTs reproduced to within half the paper's
   last printed digit.

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

Results
-------

Every published estimate, against what PPSCM returns:

.. list-table::
   :header-rows: 1
   :widths: 26 18 18 14 12 12

   * - Mobility series
     - Paper (Table 3)
     - mlsynth
     - Distance
     - Paper SE
     - mlsynth SE
   * - Retail and recreation
     - −25.08
     - −25.0818
     - 0.0018
     - 5.49
     - 5.31
   * - Grocery and pharmacy
     - −53.10
     - −53.1004
     - 0.0004
     - 10.43
     - 10.19
   * - Parks
     - −33.45
     - −33.4503
     - 0.0003
     - 7.69
     - 7.75
   * - Transit stations
     - −16.76
     - −16.7623
     - 0.0023
     - 3.90
     - 3.78
   * - Workplaces
     - −27.61
     - −27.6114
     - 0.0014
     - 6.37
     - 6.27
   * - Residential
     - 12.02
     - 12.0190
     - 0.0010
     - 2.04
     - 2.06

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

Case
----

``benchmarks/cases/bilgel_turkey_lockdown.py``. Every row is a distance from a
published figure, so a regression moves it and cannot be absorbed by re-fitting.
