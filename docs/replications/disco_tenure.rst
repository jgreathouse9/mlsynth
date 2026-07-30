.. _replication-disco-tenure:

Distributional SC — the ``disco`` Stata Journal published results
==================================================================

:Estimator: :class:`mlsynth.DSC` — Distributional Synthetic Controls.
:Source: Gunsilius & Van Dijcke, *"disco: Distributional Synthetic Controls,"*
   Stata Journal (forthcoming); replication package
   `Davidvandijcke/disco_stata_journal <https://github.com/Davidvandijcke/disco_stata_journal>`_
   (MIT). Method: Gunsilius (2023), Econometrica 91:1105–1117.
:Replication type: Cross-validation against the authors' Stata implementation,
   on their published numbers.
:Status: Verified — both published quantities reproduced to their printed
   precision.

Why this page exists
--------------------

:doc:`dsc` validates DSC against the ``DiSCo`` R vignette, and can only do so
qualitatively: that vignette is built with ``eval=FALSE``, so it publishes code
and no numbers. Its single quantitative claim is that a placebo permutation test
fails to reject. That is one bit of information, and a materially wrong
implementation could satisfy it.

The R package cannot supply more, and the reason is worth stating because it is
not obvious. Its weight solver draws the quadrature points at random —
``DiSCo_weights_reg`` calls ``runif(M)`` — so the fitted weights are a Monte
Carlo estimate. On the Dube panel it disagrees with itself across seeds by up to
0.119, while sitting 0.044 from mlsynth. No tolerance against a single run of it
would mean anything.

The Stata implementation is deterministic. ``disco_prob_grid`` builds an evenly
spaced grid and the seed is consumed only by the bootstrap, so its published
numbers are a real target.

What is reproduced
------------------

The article's tenure example, verbatim:

.. code-block:: stata

   disco y_col id_col time_col, idtarget(2) t0(3) agg("quantileDiff") ///
       seed(12143) g(10) m(100) ci boots(300)

Two independent quantities from the same fitted object. The donor weights test
the quadrature grid, the quantile convention and the simplex solve; the effects
table tests the counterfactual construction and the reporting grid.

.. list-table:: Top-5 donor weights
   :header-rows: 1
   :widths: 34 22 22 22

   * - Firm
     - Published
     - mlsynth
     - :math:`|\Delta|`
   * - amazon
     - 0.2203
     - 0.22033
     - 2.9e-05
   * - autodesk
     - 0.1271
     - 0.12711
     - 6.3e-06
   * - cisco
     - 0.1066
     - 0.10655
     - 4.9e-05
   * - dell technologies
     - 0.0991
     - 0.09909
     - 1.3e-05
   * - slalom consulting
     - 0.0962
     - 0.09618
     - 2.5e-05

.. list-table:: Quantile effects, period 3
   :header-rows: 1
   :widths: 30 24 24 22

   * - Quantile range
     - Published
     - mlsynth
     - :math:`|\Delta|`
   * - 0.00–0.25
     - −6.659
     - −6.659
     - < 5e-04
   * - 0.25–0.50
     - −21.443
     - −21.443
     - < 5e-04
   * - 0.50–0.75
     - −50.509
     - −50.509
     - < 5e-04
   * - 0.75–1.00
     - −54.751
     - −54.751
     - < 5e-04

Both match to the precision Stata prints — four decimals for weights, three for
effects. The standard errors and confidence intervals are bootstrap output under
``seed(12143)`` and are deliberately not pinned: reproducing them would mean
reproducing Stata's RNG stream, which is not a property of the estimator.

What it took to match
---------------------

Three alignments, in descending order of effect. Each was found by reading the
reference source, and each is separately justified there.

.. list-table::
   :header-rows: 1
   :widths: 34 40 26

   * - Choice
     - Reference
     - Cost of the alternative
   * - quadrature grid
     - ``linspace(0, 1, M)``, endpoints included
     - 0.0869
   * - simplex solve
     - exact QP
     - 0.0047
   * - empirical quantile
     - type 7
     - 0.0026

The grid endpoints dominate, and evaluating them means fitting the sample
minimum and maximum. mlsynth previously refused ``q = 0`` and ``q = 1``
outright.

The solver point is subtler and was hiding behind its own justification. mlsynth
used projected gradient, documented as returning "the identical optimum (the
objective is convex with a unique minimum value over the simplex)". A unique
minimum *value* does not imply a unique *argmin*, and the argmin is the reported
quantity: FISTA matches an exact QP to 0.00 percent on the objective while its
weights differ enough to miss the published table.

On the quantile type, the papers and the implementations disagree with each
other. Gunsilius (2023) §3.1 and the Stata Journal paper's equation (1) both
define the generalized inverse, which is type 1. Both shipped implementations
use type 7. mlsynth follows the implementations, because agreement with running
code is checkable and agreement with prose neither author implemented is not.

A caveat on the published values
--------------------------------

The example runs at ``m(100)``, and at that resolution the answer is not
converged. 100 quadrature points approximate quantile functions over cells
holding up to 71,438 observations:

.. list-table::
   :header-rows: 1
   :widths: 30 30

   * - :math:`M`
     - amazon weight
   * - 100
     - 0.2203 (published)
   * - 1,000
     - 0.1860
   * - 5,000
     - 0.1827
   * - 20,000
     - 0.1821

Both grid rules converge to about 0.182; the R package's random rule reaches
0.1826 by :math:`M = 10{,}000`. So the apparent disagreement between the two
official implementations is largely the two being compared at different
effective resolutions, not a difference of method.

The benchmark therefore pins both readings. The published values verify that
mlsynth implements this specification, bit-for-bit at the reference's own
settings; the converged values record what the estimator actually estimates.
Pinning only the first would quietly assert that 0.2203 is the right answer.

Durable cases & tests
---------------------

* ``disco_tenure`` — 12 rows covering both published quantities and the
  converged readings (``benchmarks/cases/disco_tenure.py``).
* ``mlsynth/tests/test_dsc_stata_exact.py`` — the published weights through the
  public API, plus the three spec choices pinned individually so a regression
  names which one broke.
* ``mlsynth/tests/test_dsc_quantile_convention.py`` — the type-1 versus type-7
  distinction, against a live R run.
* Investigation and the hypotheses that were wrong: issue #304.
