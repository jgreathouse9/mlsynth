SCUL: California (Proposition 99) vs hollina/scul
=================================================

.. currentmodule:: mlsynth

This page documents the verification of mlsynth's :doc:`../scul` estimator
against the authors' reference R package
(`hollina/scul <https://github.com/hollina/scul>`_, Hollingsworth & Wing 2022).

Cross-validation -- California (Proposition 99)
-----------------------------------------------

On the cigarette panel shipped with the reference package (California plus 38
donor states, 28 years, treatment from the 19th period), SCUL builds the
synthetic control as a rolling-origin cross-validated lasso of California's
pre-treatment cigarette sales on a 76-column multi-type donor pool -- every
donor state's per-capita sales and its retail price. mlsynth, fed the identical
panel, reproduces the authors' ``SCUL()`` value-for-value:

.. list-table::
   :header-rows: 1
   :widths: 34 22 22 18

   * - Quantity
     - mlsynth
     - ``hollina/scul``
     - abs. diff
   * - cross-validation penalty (median)
     - :math:`0.02121478`
     - :math:`0.02121478`
     - :math:`0`
   * - ATT (post-1988 mean, packs)
     - :math:`-13.171`
     - :math:`-13.306`
     - :math:`0.135`
   * - Cohen's D (pre-fit)
     - :math:`0.0137`
     - :math:`0.0160`
     - :math:`0.0023`

The rolling-origin cross-validation penalty matches ``glmnet`` to ten digits --
the selection procedure ports exactly. The small ATT difference is not a method
difference: the lasso solution is unique for continuously distributed donors
[TIBSHIRANI2013]_, and ``glmnet``'s default convergence threshold slightly
under-converges this correlated, high-dimensional (donors > pre-periods)
problem. mlsynth solves the same penalty *exactly* (the Langlois & Darbon
[LangloisDarbon2025]_ differential-inclusion homotopy, no convergence tolerance)
and lands on the unique optimum; when ``glmnet`` is run to a tight threshold, the
two agree on the donor support and on the weights to within :math:`10^{-5}`.

The case runs the R package live and asserts the penalty matches bit-for-bit and
the ATT and synthetic series agree to solver tolerance. Durable case
``scul_prop99`` (skips gracefully when ``Rscript`` or ``glmnet`` is
unavailable); the committed side-by-side table is
``benchmarks/reference/scul_prop99/comparison.csv``.
