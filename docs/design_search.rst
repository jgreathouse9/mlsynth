.. _design-search:

How the design searches work
============================

:doc:`syndes`, :doc:`marex` and :doc:`lexscm` each decide which markets to
treat. Behind every one of them is a search over sets of markets: find the set
whose treated markets, weighted, track the rest of the market best before the
experiment starts. This page explains how that search finds the best set and how
it proves that no other set is better, for a reader who has not met integer
programming. Each estimator's page gives its own objective and the details of
its search; this page gives the ideas they share and the words they use.

One word needs settling first. In marketing, *lift* is the incremental effect of
a campaign: the extra sales the treated markets show over their synthetic
control. The estimators report it as the treatment effect, and that is the only
sense in which these pages use the word for a result. The optimization
literature uses the same word for something unrelated, a step of the search
described under `Where a bound comes from`_, and SYNDES's search statistics
count those steps as ``n_lifts``. When this page means that step, it says
*relaxation*.

The problem: choosing a set
---------------------------

Take :math:`N` candidate markets and a budget of :math:`m` treated markets. A
design is a set :math:`\mathcal{S}` of :math:`m` markets. For each design the
estimator computes a loss :math:`f(\mathcal{S})`: how far the best weighted
combination of the treated markets falls from its target over the
pre-treatment periods (and, for MAREX and SYNDES, the same for the control
markets). The weights live on the simplex :math:`\Delta`, so they are
non-negative and sum to one. The design the estimator returns is the set with
the smallest loss,

.. math::

   \mathcal{S}^\ast \in \operatorname*{argmin}_{|\mathcal{S}| = m} f(\mathcal{S}).

Computing :math:`f(\mathcal{S})` for one set is called *pricing* the set. It is
a small quadratic program in the weights, which ``mlsynth`` solves exactly, and
it takes a microsecond or two. The difficulty is the number of sets: there are
:math:`\binom{N}{m}` of them. Eight treated among forty markets is 77 million
sets, ten among forty is 848 million, and six among eighty is 300 million.

The restrictions an estimator accepts (forced or forbidden markets, conflict
pairs, a cost budget, quotas per stratum) only remove sets from the list. They
change which sets are searched, not how.

Enumeration: price every set
----------------------------

The plainest exact method prices every set and keeps the best. Nothing is
skipped, so the answer is certain. Each estimator speeds it up with a
*screen*: a lower bound on each set's loss, in closed form, that costs far less
than the quadratic program. For MAREX and LEXSCM it comes from dropping the
requirement that the weights be non-negative. A set whose screened bound is
already worse than the best set found so far is never solved, and most sets are
settled that way.

Enumeration is the right method whenever the sets are few enough, and every
search here uses it below a limit: ``candidate_limit`` (twenty million) for
SYNDES and MAREX, ``enumerate_max`` (three million) for LEXSCM. Above the limit
the time grows with :math:`\binom{N}{m}`, which grows very fast in both
:math:`N` and :math:`m`.

Branch and bound: discard sets in bulk
--------------------------------------

Above the limit, SYNDES and MAREX switch to branch and bound, an exact method
that prices only a small fraction of the sets and still proves its answer. It
rests on five ideas.

The incumbent
   The best design found so far, with its loss. A fast heuristic (a swap
   search, which repeatedly replaces one treated market by another while that
   lowers the loss) supplies a good one at the start, and the search replaces
   it whenever it prices a better set.

A node
   A family of designs described by three lists: markets forced into the
   treated set, markets forced out of it, and free markets. The node holds every
   design of size :math:`m` that contains the forced-in markets, avoids the
   forced-out ones, and fills the rest from the free ones. The first node, the
   root, forces nothing and holds every design.

A bound
   A number that no design in the node can go under. If the bound is above the
   incumbent's loss, no design in the node can be the best, and the node is
   discarded with all its designs, unpriced. This step is called *pruning*.

Branching
   A node that cannot be discarded is split on one free market into two nodes:
   one that forces the market in and one that forces it out. Every design of the
   parent is in exactly one child, and each child holds fewer designs and has a
   bound at least as high as its parent's.

Pricing
   A node with few enough designs is priced design by design, as enumeration
   would, and a better design found there becomes the incumbent.

The search ends when every node has been pruned or priced. Every design is then
either priced or covered by a bound above the incumbent, so the incumbent is the
best design: the search has proved it, not only found it.

An illustration. Suppose the incumbent's loss is 0.020 and a node holds three
million designs whose bound is 0.025. Every one of those designs has a loss of at
least 0.025, so all three million are ruled out at the cost of one bound, and
none of them is ever priced. The stronger the bounds, the more of the tree is
discarded this way and the less is priced.

A search with a work limit can stop before it ends. It then returns its
incumbent and the smallest bound among the nodes it left open. The optimum lies
between the two, and their relative distance is the *gap* the result reports.
A gap of zero means the search proved its design optimal.

Where a bound comes from
------------------------

A bound comes from a *relaxation*: an easier problem whose solutions include
every design in the node. Because its minimum is taken over more than the
node's designs, it can only be lower than the best of them, which makes it a
valid bound.

The obvious relaxation does not work here. Letting each market be partly
treated, a fraction between zero and one, admits the design that spreads the
weight evenly over every market. Measured from the target, the markets'
predictors average out, so that spread reproduces the target exactly and scores
zero. A bound of zero rules out nothing.

The relaxations the searches use keep the information that this one loses: how
the weights multiply each other. The loss is a quadratic in the weights,
:math:`\mathbf{w}^\top \mathbf{G}\, \mathbf{w}` with :math:`\mathbf{G}` the
inner products of the markets' predictors, so it depends on the products
:math:`w_i w_j`. The relaxation replaces the weight vector by a matrix
:math:`\mathbf{X}` that stands for all those products,
:math:`\mathbf{X} \approx \mathbf{w}\mathbf{w}^\top`. In terms of
:math:`\mathbf{X}` the loss is linear, :math:`\langle \mathbf{G}, \mathbf{X}
\rangle`, and every actual design's :math:`\mathbf{w}\mathbf{w}^\top` has
properties the relaxation keeps as constraints:

- it is positive semidefinite (a covariance-like matrix);
- its entries are non-negative, because the weights are;
- its entries sum to one, because the weights do;
- only :math:`m` markets carry weight, which the relaxation keeps through
  indicators :math:`z_j` and small constraints linking each market's weight to
  its indicator.

Every design satisfies all of them, so the relaxation's minimum is a valid
bound, and it is far from zero. On GeoLift's forty markets with four treated,
the relaxation of this family that :doc:`marex` describes reaches 99 percent of
the optimum at the root of the tree, and dropping only the non-negativity of
the entries takes it to 29.5 percent. The relaxation is a semidefinite program,
a convex problem that an interior-point solver (Clarabel) solves reliably.

Two practical points follow from its form.

- It is the expensive step. :math:`\mathbf{X}` has :math:`N^2` entries, so a
  relaxation takes a few seconds at forty markets and grows quickly with the
  number of free markets. Pricing a set takes microseconds. The searches count
  relaxations because they are where the time goes.
- Replacing a vector by the matrix of its products is what the optimization
  literature calls *lifting* the problem into a larger space, and a relaxation
  of this kind is a *lifted* relaxation. That is the origin of ``n_lifts`` in
  SYNDES's statistics and of "lift" in the code. It is not a treatment effect,
  and nothing about it depends on the experiment's outcome.

SYNDES's two-way design has a different structure (each market's sign, treated
or control, enters as :math:`\pm 1`), and its search uses two relaxations of
that structure: a cheap one that replaces the signs by the sphere they lie on,
solved in under a millisecond, and a semidefinite one of the same kind as
above, tried only when the cheap one does not settle the node. The
:doc:`syndes` page gives both.

The certificate: a bound that can be trusted
--------------------------------------------

A solver computes in floating point, and a floating-point bound can sit slightly
above the true value. A bound that is too high could discard the node holding
the optimum. So the searches do not use the solver's number directly. Every
relaxation comes with *multipliers*, one for each constraint, and any choice of
multipliers defines a bound that is valid by construction. The search reads the
solver's multipliers and recomputes the bound they give in exact rational
arithmetic from the data, with the one eigenvalue it needs bounded exactly as
well (the method of Jansson, Chaykin and Keil, 2007). A node is discarded only on
that exact bound.

The floating-point arithmetic therefore decides where the search spends its
time, never what it discards. With ``certify=True`` the result carries a
certificate: a lower bound, the incumbent's loss, and the gap between them,
with the guarantee that the optimum lies in between.

Pricing or bounding: the cost rule
----------------------------------

A relaxation costs seconds and a priced set costs microseconds, so a node with a
few hundred thousand designs is cheaper to price than to bound. The searches
compare the two before every node: they bound a node only when pricing its
designs would cost more than the relaxation (SYNDES also weighs how often bounds
have ruled out nodes of that size so far). That rule makes enumeration and
branch and bound one method: a small problem is priced at the root and never
bounded, and a large one is bounded near the root and priced in its leaves.

The costs are fixed numbers measured once, not the clock, so the same data
makes the same decisions on any machine, and the work limit stops a search at
the same node everywhere.

What each estimator reports
---------------------------

SYNDES (``backend="exact"``, the default)
   ``raw_results["search"]`` is ``"price_all"`` (enumeration) or
   ``"branch_and_bound"``.
   ``n_nodes`` counts the nodes bounded, ``n_lifts`` the semidefinite
   relaxations solved among them, and ``raw_results["lower_bound"]`` is the
   search's own bound, the optimum itself when the search closes. See
   :doc:`syndes`, "The two-way design above the candidate limit".

MAREX (``backend="exact"``)
   ``res.search.counts`` gives every candidate set exactly one fate (cleared by
   a bound, fitted, and so on). Above the candidate limit, ``bnb_bounds`` counts
   the relaxations solved, ``bnb_priced`` the sets priced one at a time, and
   ``bnb_open`` the nodes the work limit left open, zero when the design is
   proved optimal. See :doc:`marex`, "What the search reports" and "Large
   clusters: a certified branch and bound".

LEXSCM
   ``stats["search"]["method"]`` says whether the treated sets were enumerated
   or searched locally, and ``stats["termination"]["status"]`` is ``OPTIMAL``
   when they were enumerated. Above ``enumerate_max`` LEXSCM runs a multi-start
   swap search, which returns good designs without a proof that they are the
   best, and reports ``FEASIBLE``. See :doc:`lexscm`.

Glossary
--------

Bound
   A number no design in a node can go under, from a relaxation.

Branching
   Splitting a node on one free market into a child that treats it and one that
   does not.

Certificate
   A lower bound recomputed in exact arithmetic, reported with the incumbent
   and the gap between them.

Design
   A set of treated markets; with its weights, what the estimator returns.

Enumeration
   Pricing every design.

Gap
   The relative distance between the incumbent's loss and the smallest bound
   left open. Zero means the design is proved optimal.

Incumbent
   The best design found so far.

Lift (in the search)
   A relaxation solve, so called because it replaces the weights by the matrix
   of their products, a larger space. Unrelated to lift as a treatment effect.
   Counted as ``n_lifts`` in SYNDES's statistics.

Lift (in the results)
   The incremental effect of the treatment: the gap between the treated markets
   and their synthetic control after treatment.

Node
   A family of designs fixed by markets forced in, markets forced out, and free
   markets.

Pricing
   Computing one design's loss exactly, by its small quadratic program.

Pruning
   Discarding a node because its bound is above the incumbent's loss.

Relaxation
   An easier problem whose solutions include every design in a node, so that
   its minimum bounds them all.

Screen
   A cheap closed-form lower bound on one design's loss, which lets enumeration
   skip most quadratic programs.
