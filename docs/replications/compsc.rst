.. _replication-compsc:

COMPSC — Compositional Synthetic Controls (Boussim 2026)
========================================================

:Estimator: :doc:`../compsc` — :class:`mlsynth.COMPSC`
:Source: Boussim, Onil (2026), *"Compositional Synthetic Controls,"*
   arXiv:2607.16991 [Boussim2026]_.
:Replication type: Path A — the paper's empirical application (Pennsylvania's
   Alternative Energy Portfolio Standard) reproduced cell by cell from public
   EIA data.
:Status: Fully verified — Table 1, Table 2 and section 6.4 all reproduced at
   the paper's printed precision.

Validation strategy
-------------------

The paper states that a replication package is available, but none was released
with the preprint and none could be located, so cross-validation against the
author's code is not available. Nor is there a Monte Carlo section, which rules
out Path B. What the paper does supply is an empirical application resting
entirely on public data — annual state-level net generation from the U.S. Energy
Information Administration — together with three tables precise enough to be
checked value for value. That makes Path A both possible and, here, decisive:
all nine donor weights, both fit statistics, every cell of the effects table,
and the placebo test are reproduced from the raw EIA file.

One step had to be recovered rather than read off. Section 6.1 describes the
three categories as "natural gas; coal and oil; and renewables (conventional
hydro, wind, solar, geothermal, and other)", which does not by itself determine
where EIA's ``Other Gases``, ``Other``, ``Wood and Wood Derived Fuels``,
``Other Biomass`` and ``Pumped Storage`` series belong. Those five assignments
were identified by searching the assignment space against the paper's own
Table 1 balance row (Pennsylvania's pre-treatment mean shares of 3.6 / 93.2 /
3.2 percent), which is uniquely matched by putting ``Other Gases`` with natural
gas and the remaining four with renewables. Nuclear is excluded and the three
categories renormalised. The recovered recipe is recorded in
``benchmarks/cases/compsc_pennsylvania.py`` so it is not lost.

Path A — Pennsylvania's Alternative Energy Portfolio Standard
-------------------------------------------------------------

Act 213 of 2004 set Pennsylvania's alternative energy targets; the paper takes
:math:`T_0 = 2003`, giving fourteen pre-treatment and twenty post-treatment
years over 1990–2023. The donor pool follows section 6.1: states with a
structural zero in any category are dropped, as are Ohio and West Virginia
(which adopted comparable standards during the sample) and Vermont (whose gas
and fossil shares are each below 0.1 percent throughout), leaving 42 donors.

The paper's estimator uses the additive log-ratio map with renewables as the
baseline category, which in mlsynth is ``geometry="alr"``:

.. code-block:: python

   import pandas as pd
   from mlsynth import COMPSC

   df = pd.read_csv("basedata/pa_aeps_generation.csv")
   res = COMPSC({"df": df, "outcomes": ["gas", "fossil", "renewables"],
                 "treat": "treat", "unitid": "state", "time": "year",
                 "geometry": "alr", "baseline": "renewables", "target": "gas",
                 "display_graphs": False}).fit()

Table 1 — donor weights and pre-treatment balance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Nine donors receive weight, led by coal-intensive states. Every weight agrees
with the paper at the three decimals it reports:

.. list-table::
   :header-rows: 1
   :widths: 30 20 20

   * - Donor
     - mlsynth
     - Boussim (2026)
   * - Illinois
     - 0.233
     - 0.233
   * - Wyoming
     - 0.208
     - 0.208
   * - Iowa
     - 0.168
     - 0.168
   * - Nebraska
     - 0.149
     - 0.149
   * - Massachusetts
     - 0.102
     - 0.102
   * - Connecticut
     - 0.048
     - 0.048
   * - New Jersey
     - 0.043
     - 0.043
   * - Indiana
     - 0.032
     - 0.032
   * - Montana
     - 0.016
     - 0.016
   * - Aitchison RMSPE (pre)
     - 0.187
     - 0.187
   * - Share RMSPE (pre, pp)
     - 1.34
     - 1.34

The synthetic control reproduces Pennsylvania's pre-treatment mix of 3.6
percent gas, 93.2 percent coal and oil, and 3.2 percent renewables to within
0.04 percentage points on every category.

One immaterial numerical difference: mlsynth's active-set quadratic program
leaves Arkansas at 0.000924, where the paper's ``quadprog::solve.QP`` drives it
to exactly zero. No reported statistic changes, and the benchmark tolerates
weights below :math:`10^{-3}` that the paper reports as absent.

Table 2 — compositional treatment effects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All eleven reported years reproduce across all six columns — the three share
effects, the two log-ratio effects, and the Aitchison displacement — with a
maximum deviation of 0.05, i.e. agreement at the precision the paper prints.
Selected rows:

.. list-table::
   :header-rows: 1
   :widths: 12 16 16 16 20 20

   * - Year
     - ATT gas (pp)
     - ATT fossil (pp)
     - ATT renew. (pp)
     - LRTE gas\|fossil
     - :math:`\delta_A`
   * - 2004
     - +4.6
     - −4.8
     - +0.2
     - +0.98
     - 0.76
   * - 2012
     - +25.3
     - −13.7
     - −11.6
     - +1.41
     - 1.73
   * - 2020
     - +52.1
     - −15.8
     - −36.3
     - +1.79
     - 2.19
   * - 2022
     - +60.4
     - −24.5
     - −35.9
     - +2.37
     - 2.53
   * - 2023
     - +59.1
     - −18.9
     - −40.2
     - +2.35
     - 2.43

Each row is mlsynth's output and matches the paper's corresponding row to the
displayed digits. The share effects sum to zero to machine precision, as the
geometry guarantees.

Section 6.4 — placebo inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The permutation test reproduces exactly: Pennsylvania's post-to-pre Aitchison
RMSPE ratio is 9.18 against 36 retained units, and the three placebos exceeding
it are Michigan (10.46), Louisiana (10.22) and Florida (10.07), giving the
paper's exact p-value of :math:`4/36 = 0.111`.

That p-value does not clear conventional thresholds, and the paper is candid
about why: the nearest placebos are states that underwent their own gas-driven
transitions in the 2010s. Readers should weigh the compositional shift as
described rather than as established at the 5 percent level.

A correction carried by the implementation
------------------------------------------

The replication above runs the paper's geometry so that its numbers can be
checked. It is not what mlsynth does by default, because the paper's Remark 3
contains an error.

Remark 3 asserts that the additive log-ratio map of equation (9) is an isometry
between the simplex under the Aitchison metric and Euclidean space, and hence
that minimising the stacked squared error of equations (17)–(18) minimises
pre-treatment Aitchison distance. The additive log-ratio map is not an isometry;
the centered log-ratio map is. The consequence is not academic: the fitted
weights depend on which category is nominated as the baseline, which is an
arbitrary labelling choice. On this panel, refitting with each of the three
categories as baseline moves the weight vector by up to 1.16 in :math:`L_1`.

``geometry="clr"`` is therefore the default, and it does minimise what the paper
says it wants to minimise. On the Pennsylvania panel the correction is
substantial rather than cosmetic:

.. list-table::
   :header-rows: 1
   :widths: 34 22 22

   * - Quantity
     - alr (paper)
     - clr (default)
   * - Leading donors
     - IL .233, WY .208, IA .168
     - IN .207, IA .204, IL .127
   * - Aitchison RMSPE (pre)
     - 0.187
     - 0.177
   * - ATT gas, 2022 (pp)
     - +60.4
     - +51.1
   * - Placebo p-value
     - 0.111
     - 0.167

The corrected fit is the better one on the paper's own criterion, and it moves
the headline effect down by about fifteen percent. Every result carries
``baseline_sensitivity`` so this divergence is visible at the point of use.
Users reproducing the published article should set
``geometry="alr"`` with ``baseline="renewables"``; users doing new work should
leave the default alone.

Data
----

``basedata/pa_aeps_generation.csv`` — annual net generation in megawatt hours by
state and category, 1990–2023, for the 43 units (Pennsylvania plus 42 donors)
surviving the section 6.1 screens. Built from the EIA state historical table
``annual_generation_state.xls`` (Total Electric Power Industry). Raw megawatt
hours are stored rather than shares, so the file is auditable directly against
the EIA source; COMPSC closes each row to the simplex on ingestion.

Reproducing
-----------

.. code-block:: bash

   python benchmarks/run_benchmarks.py --case compsc_pennsylvania

The case asserts the maximum deviation from every published number and also
reports the clr comparison above, so a regression in either the estimator or the
geometry default is caught.
