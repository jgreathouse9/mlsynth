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
:Status: Measured. Coverage of the nominal 95% interval falls from 0.942 to
   0.533 as the residual's autocorrelation goes from 0 to 0.9, and a
   closed-form long-run-variance prediction accounts for essentially the
   whole 2.79-fold spread in the statistic.

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

What this means in practice
---------------------------

The residual :math:`\hat v_t` is observable on the pre-period, so this is
checkable on a real panel and not a leap of faith. Autocorrelation there
means the reported interval is too narrow by roughly the factor above, and
a long-run variance estimate belongs in place of :math:`\widehat\sigma^2` —
or a scheme that preserves serial dependence, which is what
``conformal_type="block"`` provides for the estimators that offer it.

Two things this does not say. It is not a counterexample to Proposition
2.1, which assumes the serial correlation away and is correct under its own
conditions. And it is not a defect in Forward DiD's point estimate, which
is consistent across the whole table; a practitioner reading the ATT is
fine, and a practitioner reading the interval is not.

Reproducing it
--------------

.. code-block:: bash

   python benchmarks/run_benchmarks.py --case fdid_serial_correlation_mc

Seeded end to end, about forty seconds.

.. code-block:: python

   from mlsynth.utils.fdid_helpers.population import long_run_inflation

   long_run_inflation(rho=0.5, n=10, T1=400, T2=10)   # 1.5696
