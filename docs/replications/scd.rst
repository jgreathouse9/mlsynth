.. _replication-scd:

SCD -- Synthetic Control with Differencing (Arizona LAWA, CPS microdata)
========================================================================

:Estimator: :doc:`../scd` -- :class:`mlsynth.SCD`
:Source: Rincon & Song (2026), "Synthetic Control with Differencing"
   (arXiv:2510.26106), with the repeated-cross-section inference of Canen & Song
   (2025); the authors' reference package is ``ratzanyelrincon/scd`` (GPL).
:Replication type: Cross-validation -- a from-scratch base-R reproduction of the
   estimator (point weights, effect path, corrected RC variance, and the
   confidence-set membership test) on public-domain CPS microdata. The GPL
   package is reproduced, not vendored.
:Status: Done -- mlsynth reproduces the base-R reference value for value on the
   Arizona LAWA CPS extract, for all three differencing schemes: the simplex
   weights, the effect path, the ATT, the repeated-cross-section standard error,
   the weight-variance trace, and the confidence-set decisions all agree to
   solver tolerance (:math:`\sim 10^{-9}`).
:Durable check: ``benchmarks/cases/scd_cps.py`` (``scd_cps``), pinned against the
   captured base-R reference under ``benchmarks/reference/scd_cps/``; plus
   ``mlsynth/tests/test_scd.py``.

The target
----------

SCD is built for grouped microdata / repeated cross-sections: the units are
groups (here, US states), each observed through a fresh sample of individuals per
period. The authors' headline application is the Legal Arizona Workers Act
(LAWA), an employer-sanctions/E-Verify law; the treated group is Arizona and the
outcome is individual log weekly earnings from the Current Population Survey. The
in-repo extract ``basedata/cps_lawa_arizona.parquet`` is a 5% CPS sample (public-domain
microdata): 47 states, 84 monthly periods, Arizona treated at period 55, so
:math:`T_0 = 54`, :math:`K = 46`, and roughly 59,000 individual records.

Demonstrate-first port
----------------------

Before wiring anything into mlsynth, the estimator was ported to NumPy and
validated cell by cell against a base-R reproduction of the authors' code path
(survey-weighted group means via weighted collapse, differencing off the last
pre-period, and an SLSQP simplex solve):

* the donor weights match to :math:`\sim 10^{-8}` (Ohio 0.2235, Missouri 0.1693,
  Connecticut 0.1457, Arkansas 0.1373, Wyoming 0.1194, ...), summing to one;
* the effect path :math:`\widehat\theta_t` matches to :math:`\sim 10^{-8}`, with
  a post-period mean ATT of 0.116 log points;
* the repeated-cross-section pointwise variance and the weight variance
  :math:`\widehat V` match to machine precision;
* the confidence-set object :math:`B_2 \widehat V^{-1} B_2^\top` -- which is
  invariant to the arbitrary basis of the centring matrix -- matches to
  :math:`\sim 10^{-9}`, and the membership decisions agree with the authors' OSQP
  implementation on interior and boundary test points.

The three differencing schemes (``did``, ``uniform``, ``sc``) are all reproduced
value for value.

The honest findings
-------------------

Two things surfaced in the replication and are worth recording.

First, a bug in the reference RC standard error. The upstream
``gen_hat_sigma_squared_RC`` forms its treated/donor multiplier as
``ifelse(G_idx == 1, 1, -w[G_idx - 1])``; the ``no`` branch indexes ``w`` at 0
for every treated row, and R silently drops zero indices, so the vector is too
short and ``ifelse`` recycles it -- misaligning the donor weights. The sibling
weight-variance routine ``gen_hat_V_RC`` uses the correct length-:math:`(K+1)`
lookup, so only the RC pointwise standard error is affected. mlsynth implements
the corrected lookup, and the benchmark cross-validates it against a corrected
base-R reference (the uncorrected code gives 7339 for the first post-period
:math:`\widehat\sigma^2`; the correct value, which mlsynth reproduces, is 6950).

Second, an optimisation. The reference sweeps the weight confidence set with one
OSQP solve per grid point (about 23 ms each, roughly two minutes for a few
thousand candidates). The membership program
:math:`\min_r (\phi - r)^\top P (\phi - r)` s.t. :math:`\mathbf{w}^\top r = 0,\
r \ge 0` forces :math:`r_j = 0` wherever :math:`w_j > 0`, so :math:`r` lives only
on the zero-set of :math:`\mathbf{w}` and the program collapses to a small
non-negative least squares; dense (interior) candidates then need no solve at all
and are tested in one batched quadratic form. mlsynth's sweep returns the
identical confidence set roughly 290 times faster.

A caveat on the data. The in-repo extract is a 5% CPS subsample, so the weights
are only weakly identified and the confidence set is wide -- the honest band is
dominated by weight uncertainty. On the paper's full sample the band is much
tighter. The cross-validation is exact regardless, because both sides run on the
same extract.

Reproducing
-----------

.. code-block:: bash

   python benchmarks/run_benchmarks.py --only scd_cps

runs the case against the captured base-R reference values. Regenerate the
reference with ``python benchmarks/reference/generate.py scd_cps`` (needs R with
``nloptr``, ``osqp``, ``Matrix``, and ``nanoparquet`` to read the extract).
