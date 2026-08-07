SDIDGEO: does the minimum detectable effect mean what it says?
==============================================================

:doc:`../sdidgeo` reports, for each candidate test region, a minimum
detectable effect: the smallest lift the design claims to detect with
probability ``power_threshold`` at level ``alpha``. That number is
estimated by rehearsing pseudo-experiments on history. Whether it
transfers to an experiment that has not happened yet is a separate
question, and it cannot be answered on the panel the design was fit on,
because the backtests reuse those very periods.

Path B, on a constructed data-generating process. The case is
`benchmarks/cases/sdidgeo_mc.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/sdidgeo_mc.py>`_.

Design
------

A 30-market, 60-period panel is drawn from the interactive-fixed-effects
factor model standard in this literature (Abadie, Diamond and
Hainmueller 2010; Kaul, Klossner, Pfeifer and Schieler 2022):

.. math::

   Y_{it} = \mu_i + \delta_t + \gamma_t f_i + \varepsilon_{it},

with market levels drawn over a 5x range so the panel resembles a real
set of metros. Under the null every market, treated or not, comes from
this same process, so no treatment effect exists by construction.

The panel is split. SDIDGEO sees the first 48 periods and nothing else.
Experiments are then run on the held-out final 12 periods, with a lift
of known size injected and the analysis SDIDGEO promises will be used --
SDID plus the placebo standard error -- applied to each. Detection rates
over 200 replications are the reported quantities.

What it establishes
-------------------

.. list-table::
   :header-rows: 1
   :widths: 34 16 50

   * - Quantity
     - Value
     - Reading
   * - Size at the null
     - 0.135
     - Nominal 0.10. Mildly over-rejecting; see below.
   * - Power at the reported MDE
     - 0.770
     - Against a claimed 0.80. The central result.
   * - Power at half the MDE
     - 0.390
     - Clearly short, so the MDE is a minimum.
   * - Power at twice the MDE
     - 0.990
     - Monotone in the size of the lift.
   * - MDE, region of 2
     - 0.02
     - 
   * - MDE, region of 4
     - 0.01
     - A larger region detects a smaller lift.

At a backtest count of 16 the design's claim holds out of sample: it
promises 0.80 and delivers 0.77, on periods it never saw.

The winner's curse on the selected region
----------------------------------------

The design reports the smallest MDE in a field of candidates, and the
smallest of many noisy estimates is optimistic. Sweeping the backtests
depth on this DGP, alongside a region fixed before any MDE was computed:

.. list-table::
   :header-rows: 1
   :widths: 18 20 20 20 22

   * - ``n_backtests``
     - Winner MDE
     - Winner power
     - Fixed-region MDE
     - Fixed-region power
   * - 4
     - 0.010
     - 0.450
     - 0.020
     - 0.775
   * - 8
     - 0.020
     - 0.767
     - 0.020
     - 0.775
   * - 16
     - 0.020
     - 0.792
     - 0.020
     - 0.775

The claim is against 0.80 throughout. A region chosen in advance is
calibrated at every depth, including four backtests. Only the selected
winner is optimistic, and the gap closes as the scan deepens.

That identifies the mechanism. It is selection across the 24 candidates,
not the backtest count: fewer backtests make each candidate's MDE
noisier, which makes the minimum over candidates more optimistic, but the
noise alone does nothing to a region nobody selected. Placement depth is
the amplifier and selection is the cause.

Two things follow for a reader. Set ``n_backtests`` to eight or more,
which shrinks the per-candidate noise the selection feeds on. And read the
winning region's MDE as the optimistic end of a range: the honest number
for a region you have already committed to is what the fixed-region column
shows, and a design chosen from a wide field will beat its own MDE less
often than the threshold promises.

The case pins both arms, so the gap between them cannot widen unnoticed.

Size sits above nominal
-----------------------

The realized size of 0.135 exceeds the nominal 0.10, and the benchmark
pins the realized value. The placebo draws reassign two of 28 donors per
draw, so they overlap heavily and the resulting standard error is
slightly optimistic. This is recorded and not tuned away: it is a
property of the placebo procedure on a thin donor pool, and it is the
same caution the estimator's fifth assumption states. A design with a
donor pool comfortably larger than its test region will not see it as
sharply. Pinning 0.135 with a band of 0.055 means a regression that
pushed size to 0.20 fails here.
