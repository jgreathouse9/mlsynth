CPDA -- Covariate-Adjusted Panel Data Approach (Hsiao and Zhou 2019)
====================================================================

.. currentmodule:: mlsynth

Validation strategy
-------------------

CPDA is validated on Path B, the paper's own simulation design, together with
a closed-form cross-check of the step that has one. Path A, the paper's
empirical result, is deliberately not the pin, and the reason is a finding
about the paper's description and not about this implementation.

Why the published empirical value is not pinned
-----------------------------------------------

Table 9 of Hsiao and Zhou reports a CPDA mean absolute effect of 9.56 on the
California cigarette panel: 38 control states, 19 pre-treatment years,
1989 to 2000 as the treatment window.

Equations 11 to 15 fix every step of the construction except one. The paper
says the control subset "can be chosen using a model selection criterion as in
Hsiao, Ching, and Wan (2012), or the least absolute shrinkage and selection
operator (LASSO) method (Tibshirani, 1996), as suggested by Li and Bell
(2017)", and does not say which, nor how the penalty is set, nor whether the
design is standardized first.

The estimate moves with that choice. Six readings of the sentence, measured on
the paper's own panel:

.. list-table::
   :header-rows: 1
   :widths: 30 12 18 14 12

   * - selector
     - kept
     - nested LOO error
     - mean abs effect
     - ratio
   * - LASSO, cross-validated penalty
     - 9
     - 2.167
     - 4.93
     - 0.52
   * - LARS path, AICc, standardized
     - 6
     - 2.227
     - 3.79
     - 0.40
   * - LARS path, AICc
     - 5
     - 2.559
     - 5.04
     - 0.53
   * - LARS path, BIC, standardized
     - 15
     - 2.653
     - 13.78
     - 1.44
   * - LASSO, cross-validated, standardized
     - 14
     - 2.932
     - 10.92
     - 1.14
   * - LARS path, BIC
     - 15
     - 3.849
     - 14.04
     - 1.47

A factor of 3.7, with the published 9.56 inside the range. The middle column
decides the matter. It is leave-one-pre-period-out error with the selection
repeated inside every fold, so no held-out period informs its own prediction.
The lowest such error belongs to the reading that gives 4.93, and the two
readings landing nearest 9.56 score worst on it. Nothing computable from the
pre-period picks the published value out, so reaching it would mean choosing
the selector by how close its answer comes to the post-period, which is the
one thing a counterfactual may not do.

CPDA therefore ships ``lasso_cv`` as its default because that reading has the
lowest honest out-of-sample pre-period error, and the estimator exposes
``selector`` and ``sensitivity`` so a caller can see the spread instead of
receiving a point estimate that hides it.

One wrong turn is recorded because it is the trap. An earlier version of the
leave-one-out check selected once on the full pre-period and then refit inside
each fold. That leaks the selection into every fold, and it ranked the readings
in the opposite order, making the standardized LASSO look both best on error
and closest to the published value. The selection is the thing the fold is
meant to test, and holding it fixed tests nothing.

What is pinned
--------------

``benchmarks/cases/cpda.py`` measures, on the paper's Equations 2 and 3 design
with a planted effect of -5 and ten seeds:

.. list-table::
   :header-rows: 1
   :widths: 46 20 18

   * - quantity
     - measured
     - tolerance
   * - ATT mean absolute error
     - 0.109
     - 0.15
   * - ATT worst absolute error
     - 0.288
     - 0.20
   * - donor-only error over CPDA's, worst case
     - 7.36
     - 4.0
   * - ``beta_cce`` against Equation 16 computed directly
     - 0.0
     - 1e-10
   * - ``beta_bai`` improvement on pooled OLS
     - 47.5
     - 47.0
   * - selectors reported under ``sensitivity=True``
     - 4
     - exact
   * - the spread brackets the point estimate
     - yes
     - exact

The third row is what says the covariate step earns its place. The same panels
fit by a donor-only LASSO have an error 7.4 times CPDA's at worst and 37 times
at the median, because the treated unit's first covariate swings independently
of every control's and no control outcome can span it.

The fourth row is a closed form against a closed form, so the only gap is
floating point. Pesaran's Equation 16 is written out directly in the case and
compared cell for cell with ``beta_cce``.

Provenance
----------

The measurements above come from
`benchmarks/studies/hsiao_zhou_counterfactuals
<https://github.com/jgreathouse9/mlsynth/tree/main/benchmarks/studies/hsiao_zhou_counterfactuals>`_,
which replicates the paper's Section 7 in full and carries the selector sweep,
the slope comparison against the ``xtife`` R package, and the residual
disagreements in the paper's other columns.

See also :doc:`../cpda` for the method and its assumptions.
