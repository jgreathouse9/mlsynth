.. _properties:

Theoretical properties
======================

The :doc:`replications` page asks whether mlsynth reproduces the numbers a
paper printed. This page asks a different question: whether an estimator's
*algorithm* behaves the way its own proofs say it does.

The distinction matters because the two catch different mistakes. A
replication pins one estimate on one panel, or one cell of one simulation
table. It is silent about any claim the paper makes concerning the
estimator's behaviour as a sample grows — that a selection converges on the
right subset, that an interval attains its nominal coverage, that a bound
holds uniformly, that one estimator dominates a class of alternatives. Bugs
in exactly those places can leave a replicated number intact.

What a property case can and cannot do
--------------------------------------

Most results in this literature are asymptotic, and a Monte Carlo cannot
verify an asymptotic statement. What it can do is falsify one, and record
that the quantity a theorem says should move did move, in the claimed
direction, at a rate consistent with the claimed one. Pages here are
written to that standard: they say what held on the grid measured, and they
say where the convergence is slow or the design fails to separate.

Two further limits are stated on every page. An :math:`O_p` bound is
one-sided, so a measured constant carries no information — only the
flatness of a normalised deviation does. And a theorem's conditions are
part of what gets checked: where a design does not satisfy a hypothesis,
the page says so instead of reporting the number as though it did.

Which results qualify
---------------------

A result is checkable here when the objects in its statement are ones the
code produces. Forward DiD's Lemma B.1 bounds
:math:`\widehat\alpha_U` and :math:`\widehat\sigma^2_U` uniformly over
control subsets, and those are numbers the selection loop computes for
every candidate it considers — so it qualifies, despite being a lemma.
A restricted-eigenvalue constant defined as an infimum over a cone, or a
concentration inequality imported from elsewhere, does not: no simulation
materialises them, and a test of one would be a test of the linear-algebra
library.

The cases behind these pages live in the durable benchmark suite alongside
the replication cases; see :doc:`benchmarks` for how to run them.

Catalogue
---------

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - Estimator
     - Results checked
     - Headline
   * - :doc:`FDID <properties/fdid_selection>`
     - Li (2023) Propositions 2.2 / D.1, Lemma B.1
     - The forward selection recovers the population-optimal donor subset
       with probability rising from 0.00 to 0.77 as :math:`T_1` goes from
       25 to 1600; the uniform deviations stay inside a
       :math:`\sqrt{\log N / T_1}` band across a 32-fold range of
       :math:`T_1`.
   * - :doc:`FDID <properties/fdid_normality>`
     - Li (2023) Proposition 2.1
     - The studentised ATT's dispersion falls to 1.00 and its interval's
       coverage climbs to 0.952 where Assumption 4 holds. Where 4(ii)
       fails, the statistic settles at :math:`\sqrt{1 + T_2/T_1}` — 1.224
       measured against 1.2247 predicted — which identifies the term the
       assumption removes; mlsynth's standard error carries that factor and
       converges anyway.
   * - :doc:`FDID <properties/fdid_serial_correlation>`
     - Li (2023) Proposition 2.1, where it stops applying
     - The standard error prices the residual's marginal variance where a
       block mean needs its long-run variance. Under an AR(1) residual the
       nominal 95% interval covers 0.533 at :math:`\rho = 0.9`, and a
       closed-form long-run-variance prediction accounts for essentially
       the whole 2.79-fold spread. The point estimate is unaffected
       throughout. ``inference="hac"`` prices those autocovariances in and
       holds coverage at 0.92 to 0.95 over the same range, costing 2.8% of
       interval width where there is nothing to correct.
   * - :doc:`RSC <properties/rsc_rank_condition>`
     - Amjad, Shah & Shen (2018) Theorems 6, 3 and 7
     - The rank condition that lets a pre-period donor relation extrapolate
       holds to 2e-15 when :math:`\operatorname{rank}(M^-) =
       \operatorname{rank}(M)` and fails on every design when it does not,
       costing the estimator a 14-fold post-period error. The
       singular-value threshold traces the paper's Goldilocks U-shape, with
       underfitting the far more expensive side. The ridge penalty does not
       trace the exchange Section 4.3 describes: it buys nothing where the
       threshold is right and improves both windows at once where it is too
       permissive.

.. toctree::
   :hidden:
   :caption: Property pages

   properties/fdid_selection
   properties/fdid_normality
   properties/fdid_serial_correlation
   properties/rsc_rank_condition
