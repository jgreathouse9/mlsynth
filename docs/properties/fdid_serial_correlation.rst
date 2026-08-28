.. _property-fdid-serial-correlation:

FDID — the standard error under serial correlation (Li 2023)
=============================================================

:Estimator: :doc:`../fdid` — :class:`mlsynth.FDID`
:Source: Li, Kathleen T. (2023), *"Frontiers: A Simple Forward
   Difference-in-Differences Method,"* Marketing Science 43(2) [Li2024]_ —
   Proposition 2.1 and Online Appendix A.
:Results checked: where Proposition 2.1's variance formula stops applying,
   and by how much.
:Benchmark case: `benchmarks/cases/fdid_serial_correlation_mc.py
   <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/fdid_serial_correlation_mc.py>`_
:Status: Measured, and repaired. Coverage of the nominal 95% interval falls
   from 0.942 to 0.533 as the residual's autocorrelation goes from 0 to 0.9,
   and a closed-form long-run-variance prediction accounts for essentially
   the whole 2.79-fold spread in the statistic. ``inference="hac"`` prices
   those autocovariances in and holds coverage at 0.92 to 0.95 across the
   same range, at a cost of 2.8% of interval width where there is nothing to
   correct.

What this page checks
---------------------

The companion pages check that results hold where their assumptions hold.
This one checks the other side: what a specific assumption is protecting,
and what happens without it. The estimator is not at fault anywhere below —
it stays consistent throughout. What fails is the interval reported
alongside it.

Marginal variance against long-run variance
-------------------------------------------

Forward DiD's whole sampling error is a difference of two block means of
the parallel-trends residual (see :doc:`fdid_normality` for the
decomposition):

.. math::

   \widehat{\text{ATT}} - \text{ATT}
     = \bar v_{\text{post}} - \bar v_{\text{pre}}.

Li's standard error prices that with :math:`\sigma^2 = \mathbb{E}[v_t^2]`,
the residual's variance at a single date. But the variance of an average
over a window is not the variance at a date; it is the long-run variance,
which sums the autocovariances:

.. math::

   \operatorname{Var}(\bar v_T)
     = \frac{1}{T}\Bigl[\gamma_0
         + 2\sum_{k=1}^{T-1}\bigl(1 - \tfrac{k}{T}\bigr)\gamma_k\Bigr],
   \qquad \gamma_k = \operatorname{Cov}(v_t, v_{t+k}).

The two agree exactly when every :math:`\gamma_k` with :math:`k \ge 1` is
zero. Online Appendix A imposes precisely that, with iid
:math:`\epsilon_{it}` (Assumption 2(ii)) and iid :math:`f_t` (Assumption
3(i)). So under the appendix's own conditions the formula is correct, and
nothing here contradicts Proposition 2.1.

Assumption 2.1 in the main text is looser. It asks only that :math:`v_t` be
"a weakly dependent process with zero mean and finite variance", and the
appendix adds that the iid assumptions "can be easily relaxed to weakly
dependent processes". Under that relaxation the estimator is still
consistent — the block means still converge — but the standard error is
not, because nothing in :math:`\Omega_1 + \Omega_2` estimates an
autocovariance.

Why the paper's own designs cannot show this
--------------------------------------------

The four Web Appendix E designs do have serially correlated factors, so a
badly chosen donor subset would give a serially correlated residual. But
the residual is

.. math::

   v_{Ut} = \varepsilon_{tr,t} - \bar\varepsilon_{Ut}
            + (c_0 - \bar c_U)\,\mathbf{1}'f_t,

and at the subset the forward search actually selects,
:math:`\bar c_U = c_0`. The factor term vanishes and the residual is iid.
The design protects the standard error exactly where it would be tested —
which is Assumption 3(ii)'s :math:`\bar\lambda_U = 0` doing its work — so
neither the paper's Monte Carlo nor mlsynth's Path B case built on it can
detect the gap.

The case therefore uses a variant: DGP 2 with the treated unit's
idiosyncratic shock made AR(1) at unit marginal variance, so that only its
dependence changes with :math:`\rho` and never its size. That serial
correlation survives the selection.

What the case measures
----------------------

:math:`N = 20`, :math:`T_1 = 400` so the post-selection effect the
:doc:`normality page <fdid_normality>` measures is long gone,
:math:`T_2 = 10`, true ATT zero, 600 draws per cell.

.. list-table::
   :header-rows: 1
   :widths: 12 24 22 20 22

   * - :math:`\rho`
     - Coverage of the 95% CI
     - Dispersion of the statistic
     - Predicted
     - Measured / predicted
   * - 0.0
     - 0.942
     - 1.016
     - 1.000
     - 1.016
   * - 0.3
     - 0.877
     - 1.317
     - 1.293
     - 1.019
   * - 0.5
     - 0.787
     - 1.608
     - 1.570
     - 1.024
   * - 0.7
     - 0.678
     - 2.047
     - 1.975
     - 1.036
   * - 0.9
     - 0.533
     - 2.790
     - 2.637
     - 1.058

A nominal 95% interval covering 53%.

The prediction column is
:func:`~mlsynth.utils.fdid_helpers.population.long_run_inflation`, which
prices the long-run variance of the two block means in closed form and
divides by what Li's formula predicts. It carries almost the whole effect:
the final column moves only from 1.016 to 1.058 while the dispersion itself
moves by a factor of 2.79. So the gap is identified, not merely reported.

What the prediction leaves out is visible in that same drift. It prices the
long-run variance and stops there; it does not price the downward bias in
:math:`\widehat\sigma^2` that comes from demeaning an autocorrelated series
on :math:`T_1` periods, which grows with :math:`\rho` — as the column does.
The 1.016 at :math:`\rho = 0`, where there is no serial correlation at all,
is the residual post-selection effect at this :math:`T_1`, and matches what
the normality page reports independently.

A standard error that prices the dependence in
----------------------------------------------

The residual :math:`\hat v_t` is observable on the pre-period, so the
diagnosis is checkable on a real panel and the repair uses the same series.
``inference="hac"`` estimates the autocovariances there — the only stretch
long enough to estimate them, and the stretch :math:`\widehat\sigma^2`
already uses — and puts them through the exact variance of both block means:

.. math::

   \mathrm{SE}_{\text{HAC}}^2
     = \sum_{T \in \{T_1,\, T_2\}} \frac{1}{T}
       \Bigl[\widehat\gamma_0
         + 2\sum_{k=1}^{\min(L,\,T-1)}
             \bigl(1 - \tfrac{k}{T}\bigr)\widehat\gamma_k\Bigr].

The weight :math:`1 - k/T` is not a kernel choice. It is the exact
coefficient lag :math:`k` carries in the variance of a length-:math:`T`
mean, so at :math:`L = T - 1` the bracket is that variance and not an
approximation to it. Truncation at :math:`L` is the only approximation, and
each block's sum is floored at :math:`\widehat\gamma_0`, since truncating
an alternating sequence can drive it below the iid value or negative.

Measured on the same 600 draws per cell, on the same selected subsets and
the same point estimates:

.. list-table::
   :header-rows: 1
   :widths: 12 24 24 20 20

   * - :math:`\rho`
     - Coverage, analytic
     - Coverage, HAC
     - Dispersion, HAC
     - HAC / analytic width
   * - 0.0
     - 0.942
     - 0.947
     - 0.987
     - 1.028
   * - 0.3
     - 0.877
     - 0.935
     - 1.049
     - 1.268
   * - 0.5
     - 0.787
     - 0.938
     - 1.061
     - 1.532
   * - 0.7
     - 0.678
     - 0.933
     - 1.077
     - 1.919
   * - 0.9
     - 0.533
     - 0.920
     - 1.116
     - 2.517

The studentised statistic goes back to unit dispersion — 0.99 to 1.12
against 1.02 to 2.79 — so the interval is tracking the sampling variability
again and not merely widened until it happens to cover. At
:math:`\rho = 0`, where there is nothing to correct, the interval is 2.8%
wider and coverage is unchanged.

The residual 0.92 at :math:`\rho = 0.9` is the truncation. The default lag
is :math:`L = \min(T_2 - 1,\, T_1/10) = 9` here, and an AR(1) at 0.9 still
has :math:`\gamma_{10} = 0.35`, so the tail past lag 9 is genuinely
missing. Both terms in that default bind, and dropping either loses more
than the truncation does: :math:`T_2 - 1` alone gives 0.53 at
:math:`T_1 = T_2 = 100`, where nine hundred autocovariances are estimated
from a hundred observations, and a :math:`T_1`-only Newey-West rule gives
0.81 at :math:`T_1 = 400, T_2 = 40`, where it truncates at 5 in a post block
that carries lags out to 39.

Two things this does not say. It is not a counterexample to Proposition
2.1, which assumes the serial correlation away and is correct under its own
conditions, and the analytic standard error stays the default for that
reason. And it is not a defect in Forward DiD's point estimate, which is
consistent across the whole table; a practitioner reading the ATT is fine,
and a practitioner reading the analytic interval on an autocorrelated
residual is not.

Reproducing it
--------------

.. code-block:: bash

   python benchmarks/run_benchmarks.py --case fdid_serial_correlation_mc

Seeded end to end, about ninety seconds.

.. code-block:: python

   from mlsynth.utils.fdid_helpers.population import long_run_inflation

   long_run_inflation(rho=0.5, n=10, T1=400, T2=10)   # 1.5696

.. code-block:: python

   from mlsynth import FDID

   common = dict(df=df, outcome="y", treat="treat", unitid="unit",
                 time="time", display_graphs=False)

   FDID(common).fit().att_se                                    # Li (2023)
   FDID({**common, "inference": "hac"}).fit().att_se            # robust
   FDID({**common, "inference": "hac", "lrvar_lag": 4}).fit()   # fixed lag
