CSCM -- Vision Zero (Bonander 2021)
===================================

.. currentmodule:: mlsynth

This page documents the cross-validation of :doc:`../cscm` against the authors'
own R implementation.

What is reproduced
------------------

Bonander [CSCM]_ evaluates Sweden's 1997 Vision Zero road-safety policy with the
flexible count synthetic control: the treated unit is Sweden, the outcome is the
road-death rate, and nine European countries serve as donors (the United
Kingdom, Norway and the Netherlands are excluded for adopting similar policies,
and others for data issues). The panel is ``basedata/viszero.csv``; treatment is
indexed from 1996 (pre-period 1970-1995, twenty post-period years).

Reference
---------

The ground truth is a live captured run of the authors' R package
(``CSCM_helper_functions.R`` from OSF ``osf.io/uvt5p``). Two packages were
unavailable in the replication sandbox and were replaced by exact equivalents:
``osqp`` (the simplex quadratic program) by ``quadprog::solve.QP``, which
returns the identical optimum, and ``Synth::dataprep`` by a hand-built matrix
constructor. Every substantive numerical routine -- the ``glmnet`` Poisson-ridge
importance matrix and the ``optimx`` penalised solve -- is the authors' own.

What matched
------------

The port was validated cell by cell. Given identical inputs, the classic simplex
warm-start reproduces the R weights to machine precision, and the penalised
relaxation reproduces them to about :math:`10^{-11}`; the Poisson-ridge
importance matrix matches ``glmnet`` to correlation about 0.98. On this panel
``glmnet``'s importance matrix collapses to nearly uniform, so the fast,
deterministic ``v_method="uniform"`` is the faithful reproduction.

End to end, mlsynth reproduces the reference:

.. list-table::
   :header-rows: 1

   * - Quantity
     - R (glmnet)
     - mlsynth (uniform V)
   * - classic SCM weights
     - 100% Finland
     - 100% Finland
   * - sum of relaxed weights
     - 0.486
     - 0.500
   * - full-sample rate ratio
     - 1.037
     - 1.051
   * - cross-fitted rate ratio (K=2)
     - 1.065, 95% CI [0.644, 1.760]
     - 1.056, 95% CI [0.671, 1.663]

The classic warm-start concentrates entirely on Finland; the relaxation drops
the adding-up constraint, so the weights sum below one (they extrapolate below
the simplex) while remaining non-negative. The headline cross-fitted rate ratio
agrees with the reference to about one percent. The residual differences trace
to the Poisson-ridge importance matrix (``glmnet`` versus a uniform default) and
to the penalty-path grid point selected by cross-validation; both are small.

Honest reading
--------------

The substantive finding is a rate ratio near one with a wide interval that spans
it: no effect of Vision Zero on Sweden's road-death rate is detectable by this
method. The interval is wide because ``K=2`` leaves a single degree of freedom;
this is a property of the cross-fitted inference, not of the reproduction.

Durable benchmark: ``benchmarks/cases/cscm_viszero.py``.
