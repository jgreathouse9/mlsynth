.. _identification:

When a perfect pre-treatment fit means nothing
==============================================

A synthetic control that tracks the treated unit exactly before the
intervention looks like the best possible outcome. Sometimes it carries less
information than it appears to: the fit criterion may be satisfied by a whole
family of donor weightings that disagree about the counterfactual, in which case
the number you are about to report was settled by your solver rather than by
your panel.

This page explains when that happens, how to check for it in a few lines, and
what to do about it. It is written around a published example, so the failure is
concrete rather than hypothetical.

The problem
-----------

The classical synthetic control picks donor weights by solving

.. math::

   \min_{w}\; \big\| y_1 - Y_0 w \big\|^2
   \quad\text{subject to}\quad
   w_j \ge 0,\; \sum_j w_j = 1,

where :math:`y_1` is the treated unit's pre-treatment path over :math:`T_0`
periods and :math:`Y_0` holds the :math:`J` donors' paths. This is a convex
quadratic program, so the optimal *value* is unique. The optimal *weights* need
not be.

Count the constraints. Matching the pre-treatment path exactly imposes
:math:`T_0` equations, and the weights must sum to one, so :math:`T_0 + 1`
equations in :math:`J` unknowns. When :math:`J` comfortably exceeds
:math:`T_0 + 1` and the treated unit lies inside the donors' convex hull, the
set of weight vectors achieving a perfect fit is not a point but a polytope of
dimension :math:`J - T_0 - 1`. Every point in it reproduces the pre-treatment
path exactly and *none* of them is preferred by the objective.

Those weight vectors do not agree after treatment. They generally imply
different counterfactuals, and therefore different estimated effects. The
optimiser returns one of them, determined by its starting point and its internal
path — not by the data.

This is not an edge case in the wide-panel regime. It is the generic outcome
whenever a large donor pool is matched on a short pre-period, and a perfect
pre-treatment fit is its signature rather than its refutation.

A caveat worth stating immediately, because it changes what to conclude: none of
this says a large donor pool is bad. See `Adding donors is not the problem`_
below. What it says is that the objective stops choosing for you, so something
else must — and you should know what.

Diagnosing it
-------------

The check is direct. If the pre-treatment fit is (near) exact, ask what range of
post-treatment effects is consistent with an exact fit. That is a linear program:
maximise and minimise the synthetic post-treatment outcome over the same
feasible set,

.. math::

   \max_{w} \;/\; \min_{w} \;\; c' w
   \quad\text{subject to}\quad
   Y_0^{\text{pre}} w = y_1^{\text{pre}},\;
   \textstyle\sum_j w_j = 1,\; w \ge 0,

where :math:`c` holds each donor's mean outcome over the post-treatment
periods. The spread between the two solutions is the range of effects the fit criterion
cannot distinguish. If the program is infeasible, the treated unit lies outside
the donors' hull, which is a different problem (see below).

.. code-block:: python

   import numpy as np
   from scipy.optimize import linprog

   def identified_interval(y1_pre, Y0_pre, y1_post, Y0_post):
       """Range of post-treatment effects consistent with an exact pre-fit.

       y1_pre : (T0,)      treated pre-treatment path
       Y0_pre : (T0, J)    donor pre-treatment paths
       y1_post, Y0_post    the same over the post-treatment periods

       Returns (low, high), or None if exact interpolation is infeasible
       (the treated unit is outside the donors' convex hull).
       """
       J = Y0_pre.shape[1]
       A_eq = np.vstack([Y0_pre, np.ones(J)])
       b_eq = np.append(y1_pre, 1.0)
       c = Y0_post.mean(axis=0)                 # synthetic post-treatment mean
       lo = linprog(c,  A_eq=A_eq, b_eq=b_eq, bounds=(0, 1), method="highs")
       hi = linprog(-c, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1), method="highs")
       if not (lo.success and hi.success):
           return None
       observed = y1_post.mean()
       return observed + hi.fun, observed - lo.fun

Run it whenever ``pre_rmse`` is near zero and the donor pool is large relative
to the pre-period. A narrow interval means the fit pins the counterfactual down;
a wide one means it does not, and no amount of solver tuning will change that --
only a stated rule for choosing among the tied solutions will.

.. _Adding donors is not the problem:

Adding donors is not the problem
--------------------------------

It would be easy to read the above as an argument for small donor pools. It is
not, and the evidence points the other way.

Spiess, Imbens and Venugopal (2023) study exactly this regime and find that
synthetic control exhibits *single* descent: average out-of-time imputation error
falls monotonically as control units are added, with no deterioration at or past
the point where the pre-treatment fit becomes perfect. Their Proposition 4 gives
the mechanism -- because the weights are convex, a synthetic control built from
:math:`J` donors is itself a convex combination of the fits built from
:math:`J - 1` of them, so a larger pool behaves like a model average over smaller
ones. The result holds with or without interpolation, and they note it extends to
penalized synthetic control at a fixed penalty.

Their conclusion is explicit that the conventional worry is misplaced -- but so
is its condition, and the condition is the whole point of this page: the result
holds *as long as ties are broken by a suitable regularisation procedure*, the
minimum-norm solution in their case. Non-uniqueness is not denied; it is assumed
away by fixing a rule.

So the two readings fit together. The polytope is real, and which point you land
on is not determined by fit. What follows is not "use fewer donors" but "decide
the rule, and report it" -- and given a rule, more donors can help rather than
hurt.

How much does the rule matter? On the panel below, every one of these tracks the
pre-treatment path to within about thirteen cents a month, on contributions
averaging some ``$66,000`` a month:

.. list-table::
   :header-rows: 1
   :widths: 40 18 20

   * - Tie-breaking rule
     - Donors used
     - Effect (four-month)
   * - minimum norm (Spiess et al.)
     - 47
     - ``$113,700``
   * - Abadie-L'Hour penalty, :math:`\lambda = 10^{-6}`
     - 10
     - ``$124,600``
   * - mlsynth's interior-point default
     - 98
     - ``$126,800``
   * - the original study's general-purpose NLP solver
     - 96
     - ``$130,500``

A spread of about ``$17,000`` across four defensible-looking routes, none of
which is meaningfully distinguishable on pre-treatment fit. Note what the last row is not: an
argument that its answer is wrong. It is that nothing in that analysis names the
rule, so the number came from an optimiser's path rather than from a modelling
choice anyone made deliberately.

Two things this comparison does *not* show. Spiess et al. measure average
imputation risk over randomly chosen donor subsets, which is a different question
from the spread of estimates at a fixed donor set -- their result can hold while
this interval stays wide. And mlsynth's own outcome-only default is an
interior-point solution: deterministic and reproducible, but a solver artifact
rather than a rule anyone chose. If the identified interval is wide, choose the
rule explicitly rather than inheriting that one.

A worked example
----------------

Van Parys (2026) studies whether the two Senate holdouts on the Build Back
Better Act received more campaign contributions than they otherwise would have.
The design is a synthetic control on monthly PAC contributions, matching on all
nine pre-treatment months with no covariates — the Doudchenko-Imbens constrained
regression, which is ``VanillaSC(backend="outcome-only")`` in mlsynth. The donor
pool is the rest of the Senate, so :math:`J = 98` against :math:`T_0 = 9`.

For Kyrsten Sinema the pre-treatment fit is essentially perfect, and the
interpolation polytope has dimension :math:`98 - 10 = 88`. Applying the check
above, weight vectors that fit the pre-treatment path to machine precision imply
four-month effects anywhere from about ``$36,600`` to ``$180,800``. The published
figure, near ``$130,000``, sits inside that range — but so does an estimate a
fifth its size.

The author anticipated the concern and ran the donor-trimming robustness check
of Abadie and Vives-i-Bastida, which is the right instinct: restricting the pool
shrinks the polytope. Re-running the diagnostic across his trim levels shows how
sharply that bites.

.. list-table::
   :header-rows: 1
   :widths: 24 40 20

   * - Donor pool
     - Identified interval (four-month total)
     - Width
   * - top 10 / 30 / 50
     - exact interpolation infeasible — an ordinary identified fit
     - --
   * - top 70
     - ``$128,600`` -- ``$135,300``
     - ``$6,700``
   * - top 90
     - ``$59,800`` -- ``$180,800``
     - ``$121,000``
   * - all 98 (headline)
     - ``$36,600`` -- ``$180,800``
     - ``$144,200``

The trimmed specifications are identified and cluster tightly, and they agree
with the headline. The substantive conclusion survives — but it survives on the
robustness check rather than on the main specification, which by itself
identifies the sign and not the magnitude. That distinction is worth making
explicitly in any write-up, and it is exactly what the diagnostic surfaces.

What to do about it
-------------------

Penalise. Abadie and L'Hour (2021) add a pairwise matching penalty to the same
program,

.. math::

   \min_{w}\; \big\| y_1 - Y_0 w \big\|^2
   \;+\; \lambda \sum_j w_j \big\| y_1 - Y_{0,j} \big\|^2 ,

and their Theorem 1 says that for *any* :math:`\lambda > 0` the solution is
unique, with at most :math:`T_0 + 1` non-zero weights. Among the weight vectors
that fit equally well, it selects the one built from donors individually close
to the treated unit — the least interpolation bias. In mlsynth that is
``backend="penalized"``, and it needs no covariates: the pre-treatment outcome
lags are the matching variables.

On the panel above, an essentially infinitesimal penalty is enough to restore
uniqueness at no meaningful cost in fit:

.. list-table::
   :header-rows: 1
   :widths: 30 14 18 22

   * - Specification
     - Donors
     - Pre-treatment RMSE
     - Effect (four-month)
   * - outcome-only
     - 98
     - 0.00
     - ``$126,800``
   * - penalized, :math:`\lambda = 10^{-6}`
     - 10
     - 0.13
     - ``$124,600``
   * - penalized, :math:`\lambda = 10^{-2}`
     - 9
     - 984
     - ``$122,700``

Ten donors at :math:`\lambda = 10^{-6}` is exactly the :math:`T_0 + 1` bound,
and the estimate is stable across four orders of magnitude of :math:`\lambda`.

Three other routes are worth knowing:

Trim the donor pool, as above, following Abadie and Vives-i-Bastida — the
simplest fix and the one to reach for when a principled similarity ranking is
available.

Shrink instead of interpolating. The high-dimensional estimators exist for this
regime: see :doc:`clustersc`, :doc:`sparse_sc`, :doc:`rescm`, :doc:`fscm` and
:doc:`bvss`, and Q1.6 of :doc:`choose`.

Do not reach for :doc:`masc` here. It trades extrapolation bias against
interpolation bias and is the right tool when the hull condition *fails*. That
is the opposite problem: too few fits rather than too many.

The other failure
-----------------

The diagnostic returns ``None`` when exact interpolation is infeasible, which
means the treated unit is outside the donors' convex hull. Nothing is
under-determined then — the fit is a genuine constrained least-squares problem
with a well-defined solution — but the pre-treatment fit will be poor and the
convex-hull condition that motivates synthetic control does not hold.

In the same study, the second treated senator is in exactly that position:
exact interpolation is infeasible at every donor-pool size, and his
pre-treatment RMSE is roughly ``$12,600`` on an outcome of that order. A poor
pre-treatment fit is easier to notice than a suspiciously perfect one, which is
part of why the perfect-fit case deserves a deliberate check.

References
----------

Abadie, A., and J. L'Hour (2021). "A Penalized Synthetic Control Estimator for
Disaggregated Data." Journal of the American Statistical Association 116(536),
1817-1834.

Abadie, A., and J. Vives-i-Bastida (2022). "Synthetic Controls in Action."
arXiv:2203.06279.

Spiess, J., G. Imbens, and A. Venugopal (2023). "Double and Single Descent in
Causal Inference with an Application to High-Dimensional Synthetic Control."
Advances in Neural Information Processing Systems 36.

Doudchenko, N., and G. W. Imbens (2016). "Balancing, Regression,
Difference-in-Differences and Synthetic Control Methods: A Synthesis." NBER
Working Paper 22791.

Van Parys, A. (2026). "Vote Buying and Negative Agenda Control: A Problem for
the Study of Money in Politics." American Journal of Political Science.
