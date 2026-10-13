.. _replication-rfpda-watches:

rfPDA — Liu, Long and Luo (2025) luxury watches
================================================

:Estimator: :doc:`../pda` — :class:`mlsynth.PDA` (``method="rf"``)
:Source: Liu, Guannan, Long, Wei and Luo, Xuehong (2025), *"A Random
   Forest-Based Panel Data Approach for Program Evaluation,"* Journal of
   Applied Econometrics 40(5), 591-607, Section 5.2 and Figure A8.
   Replication package: the article's Journal Data Archive deposit,
   ``Application_2``.
:Replication type: Path A in the reference implementation, distributional
   cross-validation in mlsynth.
:Status: The reference reproduces its own published numbers exactly. The port
   is held to the distribution the reference draws from, and to the long-run
   variance, which is deterministic.
:Durable case: ``benchmarks/cases/rfpda_watches.py``.

What the paper reports
----------------------

Section 5.2 revisits Shi and Huang's (2023) study of China's anti-corruption
campaign. The treated series is the monthly import growth rate of watches with
cases of, or clad with, precious metal; the donor pool is 87 other commodities,
from fabric to minerals. The panel runs February 2010 to December 2016 and the
campaign begins in January 2013, so :math:`T_1 = 35` and :math:`T_2 = 36`.

rfPDA selects seven commodities -- wool and fine animal hair; horsehair yarn and
woven fabric; preparations of cereals, flour, starch or milk; pastrycooks'
products; plastics and articles thereof; ores, slag and ash; pharmaceutical
products; raw hides and skins other than furskins, and leather; and knitted or
crocheted fabrics -- and reports an average effect of :math:`-2.66\%` with an
approximate :math:`R^2` of 0.78 at a p value of 0.063, in line with Shi and
Huang's :math:`-3.09\%`.

The reference reproduces itself
-------------------------------

Running the deposited ``RF.R`` on the deposited ``china_import.rda`` at the
deposited seed returns every one of those numbers:

.. list-table::
   :header-rows: 1
   :widths: 34 22 22

   * - Quantity
     - Reference run
     - Published
   * - ATE
     - -0.0266
     - -2.66%
   * - :math:`R^2`
     - 0.7771
     - ~0.78
   * - Controls selected
     - 7
     - 7
   * - p value
     - 0.0634
     - 0.063

The companion application in the same deposit, the effect of the Brexit
referendum on United Kingdom GDP growth, does not reproduce. Its
``data_WB.csv`` is a later World Bank vintage than the one the paper was run
on: after dropping economies with a gap between 2000 and 2019 it leaves 167
usable controls against the 193 the paper describes, and the selected set and
the estimate move accordingly. The code is not at fault, and the Brexit numbers
are therefore not a target this case can hold anything to.

Why the port is not held to the point estimate
----------------------------------------------

The selected set is a function of the random pre-period split and of the forest
grown on it. ``randomForest`` and scikit-learn build trees differently and draw
from different RNG streams, so a port that returned the same seven commodities
would be doing so by coincidence.

That is the smaller half of the reason. The larger half is that the point
estimate is itself a draw. Re-running the reference over twenty seeds on its own
data gives:

.. list-table::
   :header-rows: 1
   :widths: 30 22 22 26

   * - Quantity
     - Mean
     - Standard deviation
     - Range
   * - ATE
     - -0.0199
     - 0.0128
     - [-0.0629, -0.0033]
   * - Controls selected
     - 13.8
     -
     - [2, 87]

The mean pairwise overlap between the selected sets is 0.230, and the published
:math:`-0.0266` sits inside a range nearly twenty times as wide as its own
distance from zero. On the Brexit panel the same sweep selects, across twenty
seeds, every one of the 167 available controls at least once, at a mean overlap
of 0.156. Comparing one draw against one draw across two RNGs measures nothing,
so the case compares the distributions and asks that the published estimate and
the reference's own mean both fall inside the range the port draws.

Under the released configuration -- the random pre-period split, out-of-bag
importance, no cap -- mlsynth draws:

.. list-table::
   :header-rows: 1
   :widths: 24 18 18 26 16

   * -
     - Mean
     - Standard deviation
     - Range
     - Overlap
   * - Reference, 20 seeds
     - -0.0199
     - 0.0128
     - [-0.0629, -0.0033]
     - 0.230
   * - mlsynth, 10 seeds
     - -0.0271
     - 0.0142
     - [-0.0601, -0.0157]
     - 0.220

Two forests that build their trees differently draw differently, and these do:
the means sit 0.007 apart on an estimate of 0.02. What matches is the shape --
the same sign throughout, a spread of the same order, and the same low selection
overlap -- and the containment the case requires holds in both directions, with
the published -0.0266 and the reference's own mean both inside the port's range.
Under the paper's own configuration, a temporal three-block split with
permutation importance and the cap, the port's mean is -0.0185.

What is pinned exactly
----------------------

One component is deterministic on both sides: the West (1997) long-run variance
of Equation (9) and (10), which the test statistic is standardised by. On a
fixed error series :math:`e_t = \sin t + 0.3\cos 3t` split at
:math:`T_1 = 40`, the reference's ``HAC_function`` returns 0.436241897207 before
and 0.850073133287 after, for :math:`z = -0.096661351544`;
:func:`mlsynth.utils.pda_helpers.rf.west_lrvar` returns 0.437027454211 and
0.851024367725 for :math:`z = -0.096596161105`. The two agree to 0.18% and 0.11%
on the components and to 0.07% on the statistic. The residual is the MA(1) fit:
R's ``arima`` and ``statsmodels``' ``ARIMA`` maximise the same likelihood from
different starting values.

The cap
-------

The released search runs the prefix length from 2 to :math:`n - 1` with nothing
tying it to :math:`T_1`, while Assumption 3 of the paper requires
:math:`|\hat U| / T_1 \to 0`. Where it over-selects, the OLS fit interpolates
the pre-treatment window, its residual falls to rounding, and the long-run
variance it is standardised by collapses with it. On this panel that happens in
4 of 20 reference seeds, returning test statistics as large as :math:`-17.8` at
a p value of zero; on the Brexit panel it happens in 7 of 20, with one statistic
at :math:`-966336`. The case requires the released configuration to be seen
doing this and the default cap at :math:`T_0 - 2` to be seen preventing it. In
the port both hold: the released configuration selects past :math:`T_0` and
degenerates, and under the cap the largest selected set over ten seeds is 33,
which is :math:`T_0 - 2` exactly, with no degenerate fit.

Reproducing
-----------

.. code-block:: bash

   python benchmarks/run_benchmarks.py --case rfpda_watches

The released configuration is reachable from the estimator:

.. code-block:: python

   from mlsynth import PDA

   fit = PDA(dict(df=panel, outcome="y", treat="treat", unitid="unit",
                  time="time", method="rf",
                  rf_split="random", rf_train_fraction=0.7,
                  rf_importance="oob", rf_k_max=87,
                  rf_n_estimators=1000, rf_seed=372236)).fit()

and the default -- a temporal three-block split, permutation importance and the
:math:`T_0 - 2` cap -- is what ``method="rf"`` gives without those arguments.
Pass ``rf_n_seeds`` to get the spread reported beside the estimate.
