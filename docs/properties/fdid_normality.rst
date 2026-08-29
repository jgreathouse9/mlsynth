.. _property-fdid-normality:

FDID — asymptotic normality of the ATT (Li 2023)
================================================

:Estimator: :doc:`../fdid` — :class:`mlsynth.FDID`
:Source: Li, Kathleen T. (2023), *"Frontiers: A Simple Forward
   Difference-in-Differences Method,"* Marketing Science 43(2) [Li2024]_ —
   Proposition 2.1 and Online Appendix B.
:Results checked: Proposition 2.1 (the studentised ATT is asymptotically
   standard normal), and what Assumption 4(ii) is buying.
:Benchmark case: `benchmarks/cases/fdid_normality_mc.py
   <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/fdid_normality_mc.py>`_
:Status: Reproduced. Where Assumption 4 holds the statistic's dispersion
   falls to 1.00 and the interval's coverage climbs to 0.952. Where
   Assumption 4(ii) fails, the statistic settles at
   :math:`\sqrt{1 + T_2/T_1}` — measured 1.224 against a predicted 1.2247 —
   and mlsynth's standard error, which carries that factor, converges
   anyway.

What this page checks
---------------------

Forward DiD reports a standard error, a confidence interval and a p-value
alongside its ATT. Proposition 2.1 is what licenses them: it says the
estimate, divided by an estimate of its own scale, behaves like a draw from
a standard normal once the sample is large enough. A replication of the
paper's application cannot check this — one panel gives one interval, and
whether that interval covers at its stated rate is a statement about
repeated samples.

The companion page :doc:`fdid_selection` checks the results about which
donors get selected. This one checks what happens after the selection, and
the two interact: the scale estimate is computed on the very subset chosen
to make it small, so a naive reading would expect the interval to be too
narrow. Measuring says by how much, and for how long.

The statistic
-------------

Write :math:`\widehat{ATT}` for the estimate,
:math:`\hat v_t = y_{tr,t} - \bar y_{\widehat{\mathcal{N}}_{co},t}
- \widehat\alpha` for the pre-treatment residuals of the selected subset,
and

.. math::

   \widehat\sigma^2
     = \frac{1}{T_1}\sum_{t=1}^{T_1} \hat v_t^2

for their mean square. The proposition studentises by :math:`\widehat\sigma`
and scales by the post-period length.

Proposition 2.1. Under Assumption 2.1 and Assumptions 2 to 4 (reproduced on
:doc:`fdid_selection`), as :math:`T_1, T_2 \to \infty`,

.. math::

   \left|\Pr\!\left(
     \frac{\sqrt{T_2}\,(\widehat{ATT} - ATT)}{\widehat\sigma} \le a
   \right) - \Phi(a)\right| \to 0
   \qquad \text{for all } a \in \mathbb{R},

with :math:`\Phi` the standard normal distribution function.

*Remark.* The scaling is asymmetric on purpose: the numerator averages over
the :math:`T_2` post-periods while the scale is estimated on the :math:`T_1`
pre-periods. Assumption 4 governs the exchange rate between them, requiring
:math:`T_2 \log N / T_1 \to 0` — the pre-period has to out-grow the
post-period. The next section measures what that requirement is doing.

Two statistics, differing by one finite-sample factor
-----------------------------------------------------

The proposition divides by :math:`\widehat\sigma` alone. mlsynth's reported
standard error carries one term more,

.. math::

   \widehat{se} = \frac{\widehat\sigma}{\sqrt{T_2}}\sqrt{1 + \frac{T_2}{T_1}},

so that :math:`\widehat{ATT}/\widehat{se}` is the proposition's statistic
divided by :math:`\sqrt{1 + T_2/T_1}`. That extra factor prices in the error
from estimating the level shift :math:`\widehat\alpha` on :math:`T_1`
pre-periods, which the ATT inherits. The two statistics agree in the limit
exactly because Assumption 4(ii) drives :math:`T_2/T_1 \to 0`. The case
measures both, so the factor's contribution is visible instead of assumed.

What the case measures
----------------------

Both regimes use DGP 2 at :math:`N = 20`, whose donor pool is half
mismatched, and the designs' true ATT of zero.

Assumption 4 holds — :math:`T_2 = 10` fixed, :math:`T_1` growing, 1000
draws per cell:

.. list-table::
   :header-rows: 1
   :widths: 16 28 28 28

   * - :math:`T_1`
     - Dispersion of the statistic
     - Coverage of the 95% interval
     - Distance to :math:`\Phi`
   * - 50
     - 1.329
     - 0.889
     - 0.075
   * - 100
     - 1.130
     - 0.931
     -
   * - 200
     - 1.112
     - 0.938
     -
   * - 400
     - 1.019
     - 0.947
     -
   * - 800
     - 0.998
     - 0.952
     - 0.035

Both columns move monotonically to where the proposition says they should
end up. The excess dispersion at small :math:`T_1` is the post-selection
effect anticipated above: :math:`\widehat\sigma` is estimated on the subset
picked to minimise it, so the scale comes out too small and the statistic
too spread out, and at :math:`T_1 = 50` that costs six points of coverage.
It is gone by :math:`T_1 = 800`.

Assumption 4(ii) fails — :math:`T_2 = T_1/2`, so :math:`T_2\log N/T_1` never
falls, 500 draws per cell:

.. list-table::
   :header-rows: 1
   :widths: 14 16 35 35

   * - :math:`T_1`
     - :math:`T_2`
     - Dispersion, the proposition's statistic
     - Dispersion, through mlsynth's standard error
   * - 100
     - 50
     - 1.401
     - 1.144
   * - 400
     - 200
     - 1.237
     - 1.010
   * - 1600
     - 800
     - 1.224
     - 1.000

The proposition's statistic does not converge on a standard normal here,
and the value it converges on is not arbitrary: :math:`\sqrt{1 + T_2/T_1} =
\sqrt{1.5} = 1.2247`, against a measured 1.224. The failed assumption costs
exactly the level-shift estimation term and nothing else. Since mlsynth's
standard error divides by that same factor, its statistic converges to 1.000
across the identical grid, with a distance to :math:`\Phi` of 0.040 and
coverage of 0.936.

This is not a counterexample to Proposition 2.1, which asserts nothing when
its hypothesis fails. It identifies what the hypothesis is for, and shows
the library's interval standing without it.

Reading the distances
---------------------

A Kolmogorov-Smirnov distance computed from :math:`M` draws sits near
:math:`0.86/\sqrt{M}` even when the statistic is exactly normal — 0.027 at
:math:`M = 1000`, 0.038 at :math:`M = 500`. The 0.035 at :math:`T_1 = 800`
and the 0.040 in the violation regime are at that floor, so they bound
whatever gap remains without measuring it. The dispersion and coverage
columns carry the convergence; the distances confine it.

What is established, and what is not
------------------------------------

The proposition is a limit statement, so no finite grid confirms it. What
the case pins is that the two quantities it implies — a dispersion of one
and coverage at nominal — are reached monotonically on the grid measured,
and that the interval mlsynth reports covers at its stated rate by
:math:`T_1 = 800` in this design.

Two limits transfer to practice. Coverage is materially below nominal at
short pre-periods: 0.889 at :math:`T_1 = 50`, against a stated 0.95. The
cause is structural, not a defect — any procedure that selects a subset by
minimising a residual variance and then estimates scale from that same
residual variance will understate it — but it means Forward DiD's interval
should be read as optimistic when the pre-period is short relative to the
donor pool. Second, how short is short depends on the design, so the
:math:`T_1 = 800` figure here is where the effect vanishes for a 20-donor
pool at this noise level, not a general threshold.

Reproducing it
--------------

.. code-block:: bash

   python benchmarks/run_benchmarks.py --case fdid_normality_mc

The case is seeded end to end and takes about three minutes. Tolerances are
roughly three Monte-Carlo standard errors: a dispersion estimated on
:math:`M` draws has standard error near :math:`\mathrm{sd}/\sqrt{2M}`, and a
coverage rate near 0.95 has :math:`\sqrt{0.95 \times 0.05/M}`.

.. code-block:: python

   import numpy as np
   from mlsynth import FDID

   res = FDID({...}).fit().fdid
   z_paper = np.sqrt(T2) * res.att / res.pre_rmse   # Proposition 2.1
   z_mlsynth = res.att / res.att_se                 # with the sqrt(1 + T2/T1) factor
