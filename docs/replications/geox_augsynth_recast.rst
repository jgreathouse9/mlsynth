.. _replication-geox-augsynth-recast:

GEOX — the augsynth engine against an independent simulation study
==================================================================

:Estimator: :doc:`../geox` — :class:`mlsynth.GEOX`, ``engine="augsynth"``
:Reference: `getrecast/geolift-simulation-study
   <https://github.com/getrecast/geolift-simulation-study>`_, whose GeoLift arm
   is augmented SC (ridge) with block conformal inference at
   :math:`\alpha = 0.05` — GeoLift 2.7.5 (commit ``4d2afd4``) with augsynth
   ``65c5a6f``.
:Replication type: cross-validation. Their data-generating process, their tool
   configuration, and their published numbers.
:Status: verified — bias agrees on all four scenarios within
   :math:`|z| \le 1.1`, the false-positive rate within 0.021, coverage within
   0.031, and the interval width to 2% on three of the four.

Why this case and not another Monte Carlo
-----------------------------------------

The design's other calibration case, ``geox_mc``, imposes ground truth on a process defined inside this repository
and asks whether GEOX is internally consistent. That cannot detect an error
shared between the estimator and the data it is tested on. This case takes the
process, the tool settings and the target numbers from outside: the study runs
1000 iterations per cell over four stress scenarios and publishes each tool's
bias, coverage and false-positive rate, and its GeoLift arm is configured as the
estimator GEOX runs under ``engine="augsynth"``.

The study evaluates estimation, not market selection — a fixed treated geo, a
known lift, how well each tool recovers it. So the comparison runs the engine's
readout with the treated geo fixed. GEOX's selection loop is cross-validated
separately, against R ``GeoLiftMarketSelection`` in
:doc:`../geox`'s ``geox_augsynth_geolift``.

What it establishes
-------------------

.. list-table:: Bias against a truth of zero, percentage points
   :header-rows: 1
   :widths: 34 22 22 22

   * - Scenario
     - mlsynth
     - Recast GeoLift
     - :math:`z`
   * - A1 textbook
     - +0.48
     - +0.22
     - +0.4
   * - A2 outlier (5x)
     - +3.51
     - +3.22
     - +0.4
   * - A3 small pool
     - +0.73
     - +1.03
     - −0.4
   * - A4 short pre-period
     - +0.57
     - −0.39
     - +1.1

.. list-table:: False-positive rate at :math:`\alpha = 0.05`
   :header-rows: 1
   :widths: 34 22 22

   * - Scenario
     - mlsynth
     - Recast GeoLift
   * - A1
     - 0.067
     - 0.046
   * - A2
     - 0.042
     - 0.042
   * - A3
     - 0.033
     - 0.049
   * - A4
     - 0.050
     - 0.033

The signature is A2. Inflating the treated geo fivefold puts it outside the
donors' convex hull, the ridge augmentation extrapolates to reach it, and the
bias jumps an order of magnitude above every other cell — +3.22 points in their
table, +3.51 here. Reproducing that pattern, and not merely a similar average,
is what indicates the engine and the ported process both match. The case pins it
as ``a2_bias_excess_pp``: A2's bias must stand clear of the largest of the other
three, observed at 2.78 points.

Power is low everywhere — 0.05 to 0.18 against the study's 7.5% lift, against
their GeoLift's 0.043 to 0.107. The design is under-powered at this effect and
panel length in both implementations. That is the finding reproduced, not a
defect in either.

Coverage, and why width is the sharper test
-------------------------------------------

Coverage is the study's headline metric. It is a rate, and two procedures can
both cover at 0.93 with intervals of very different length, so the case pins the
average interval width against theirs as well. Agreement on width says the two
tests reject in the same places, which coverage on its own does not.

.. list-table:: Coverage of the true ATT, and interval width in outcome levels
   :header-rows: 1
   :widths: 22 15 15 15 15 18

   * - Scenario
     - mlsynth null
     - Recast null
     - mlsynth 7.5%
     - Recast 7.5%
     - width (ours / theirs)
   * - A1 textbook
     - 0.933
     - 0.954
     - 0.892
     - 0.922
     - 2062 / 2020
   * - A2 outlier (5x)
     - 0.950
     - 0.958
     - 0.917
     - 0.936
     - 10182 / 9955
   * - A3 small pool
     - 0.967
     - 0.951
     - 0.925
     - 0.925
     - 1968 / 1926
   * - A4 short pre-period
     - 0.950
     - 0.967
     - 0.925
     - 0.951
     - 6339 / 5575

The interval is the set of constant effects the conformal test does not reject,
so its coverage in the null cell is the complement of the false-positive rate by
construction — the truth is covered exactly when the test does not reject it.
The effect cell is a real test of something else: the injected lift is
multiplicative, so the true effect path is not constant, and the constant-effect
null the interval inverts is misspecified there. Coverage holds anyway, in both
implementations.

A4 is the one width that does not agree, at 1.14 times theirs. Thirty pre-days
against fifteen post ones means a third of the block reference set overlaps the
post window, the p-value cannot fall as low as it does elsewhere, and on some
panels no candidate is rejected in a direction at all. Those intervals are
unbounded and are excluded from the average, leaving the wider tail of the ones
that remain.

Empty confidence sets are pinned too, at 2.5% to 6.7% of panels. An empty set is
a correct reading of a constant-effect null against a multiplicative lift, but it
is also what a broken search would produce, so it is bounded and not left
unwatched.

The case runs 120 replications per cell against the study's 1000, so each pinned
value carries a Monte-Carlo standard error near 0.7 points on a bias and 0.02 on
a rate. It is seeded throughout, so a re-run reproduces every value exactly.

The case
--------

`benchmarks/cases/geox_augsynth_recast.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/geox_augsynth_recast.py>`_,
about 90 seconds. The process is ported in
:func:`mlsynth.utils.geox_helpers.simulate.simulate_recast_panel` from their
``src/R/generate_panels.R``, not inlined, so a later case can draw from it.
