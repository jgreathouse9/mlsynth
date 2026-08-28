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

.. toctree::
   :hidden:
   :caption: Property pages

   properties/fdid_selection
