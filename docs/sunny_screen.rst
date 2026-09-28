.. _sunny_screen:

Which donors can ever matter
============================

A synthetic control is a weighted average of donors, with weights that are
non-negative and sum to one. Before fitting one, there is a question you can
answer about the donor pool alone: is any donor redundant, in the sense that some
blend of the others already does everything it could do? A donor like that can
never carry weight, so it can be deleted before any estimation begins, and the
answer does not change.

Becker and Klossner (2018) call the donors that survive this test sunny and the
ones it deletes shady. mlsynth implements the test in
:mod:`mlsynth.utils.solvers.sunny`, and ``VanillaSC`` runs it automatically when
``backend="mscmt"``.

This page says what the test decides, what it does not decide, and the one thing
an applied reader most needs to know about it: on the panel shapes synthetic
control is usually run on, it deletes nothing, and that is the expected answer,
not a failure.

Notation
--------

Write :math:`A` for the treated unit's matching vector, with one entry per
matching condition -- a pre-treatment period, or a predictor averaged over a
window. Write :math:`B` for the donor matrix, one column per donor. Let
:math:`m` be the number of matching conditions and :math:`J` the number of
donors.

Centre every donor on the treated unit,

.. math::

   x_j = B_{\cdot j} - A ,

so the treated unit sits at the origin and each donor is its deviation from it.
A synthetic control is a convex combination of the :math:`x_j`, so the set of
achievable fits is the convex hull

.. math::

   H = \operatorname{conv}(x_1, \ldots, x_J) ,

and the best synthetic control is the point of :math:`H` nearest the origin.

The centring is the definition, not a convention: asking whether donors can
reproduce the treated unit is asking where the origin sits relative to
:math:`H`. Subtracting the same vector from every unit leaves each :math:`x_j`
unchanged, so the test is invariant to it; subtracting a different amount from
each unit, as unit demeaning does, gives a different design and can give a
different answer.

The test
--------

Walk outward from the origin along the ray toward donor :math:`j`. The ray
passes through the points :math:`\alpha x_j` as :math:`\alpha` grows from zero,
and reaches the donor at :math:`\alpha = 1`. Define

.. math::

   \alpha^*(j) = \min \{ \alpha \ge 0 : \alpha x_j \in H \} .

Since :math:`x_j` generates :math:`H`, the value :math:`\alpha = 1` is always
feasible, so :math:`\alpha^*(j) \le 1` and the test is one-sided by
construction. Donor :math:`j` is sunny when :math:`\alpha^*(j) = 1` and shady
when :math:`\alpha^*(j) < 1`.

A picture makes the name. Put a lamp at the treated unit and treat :math:`H` as
opaque. Light travels outward and stops at the first surface it meets. If the
ray reaches :math:`x_j` before entering the solid, the donor is the first thing
the light touches in that direction and it is lit. If the ray enters the solid
first, the donor sits behind that surface, in shadow.

The economics of the shadow case is the part to hold on to. If the ray enters
:math:`H` at some :math:`\alpha^* < 1`, the entry point is itself a blend of
donors. So there is a weighted average of the others lying directly between the
treated unit and donor :math:`j`, and closer to the treated unit than
:math:`j` is. That blend does whatever :math:`j` does, in the same direction,
only nearer the target. Donor :math:`j` is redundant.

A one-variable example
~~~~~~~~~~~~~~~~~~~~~~

Match on a single variable, pre-treatment GDP per capita. The treated unit sits
at 10, donor A at 12, donor B at 20. Centred, :math:`x_A = 2` and
:math:`x_B = 10`, so :math:`H` is the segment :math:`[2, 10]` on the line.

Standing at the origin and looking along the line, the segment begins at 2. The
light stops there, so A is lit and B is in shadow: A stands between the observer
and B. The estimator agrees. Both donors overshoot the target in the same
direction, convex weights cannot go below the nearest donor, and the best
achievable fit puts all the weight on A. B takes zero weight, and it does so for
any choice of predictor weighting, since in one dimension a positive weight
cannot reorder :math:`2 < 10`.

In higher dimensions the picture is the same with a polygon in place of a
segment: the faces turned toward the lamp are lit, the ones around the back are
hidden, and anything strictly inside is dark.

What the test guarantees
------------------------

Two results give the screen its use. Both are pinned in
``mlsynth/tests/test_sunny_donors.py``.

Proposition 1
~~~~~~~~~~~~~

No donor is sunny exactly when the origin lies in :math:`H`, which is exactly
when the donors can reproduce the treated unit's matching vector exactly.

So an all-shady verdict does not mean the donor pool is worthless. It means the
opposite: a perfect pre-treatment fit exists. Every weight vector attaining it
ties on the matching objective, so a search over predictor weights is choosing
among exact ties on numerical accident, and Becker and Klossner's Eq (10) states
the intended rule instead -- among the perfect-fit weights, take the one with
the best outcome fit. ``VanillaSC`` with ``backend="mscmt"`` takes that branch
when it arises.

Proposition 2
~~~~~~~~~~~~~

Provided at least one donor is sunny, a shady donor takes zero weight at every
optimum, for every predictor weighting :math:`V`.

The quantifier over :math:`V` is what makes the screen usable before estimation.
A test that held only at the weighting the search eventually selects would
require finishing the search to know what to delete. Holding for every
:math:`V` means the deletion is safe to make first.

The proviso is not slack. A donor whose centred column is the origin reproduces
the treated unit on its own; it is shady with :math:`\alpha^* = 0`, and deleting
it would discard an exact fit. Proposition 1 is what rules that out, which is
why :func:`~mlsynth.utils.solvers.sunny.sunny_support` checks for a sunny donor
before it prunes anything.

What sunny does not mean
------------------------

The guarantee runs one way. Shady implies zero weight; sunny does not imply
positive weight. Sunny means the screen failed to rule the donor out, which is
weaker than a claim that some weighting would use it.

Abadie and Gardeazabal's Basque study makes the distinction concrete. Baleares
receives zero weight in the fitted synthetic Basque Country, and looks like the
kind of donor a screen ought to delete. It is emphatically sunny. No combination
of the other fifteen regions reproduces its path: leaving it out and fitting it
from the rest leaves a residual of 0.97 in relative terms. Baleares is the
richest region in the pool, an extreme point, and it takes zero weight because
the objective has no use for its direction at the selected weighting -- not
because anything else can stand in for it.

Redundant and unwanted are different properties, and only the first is what the
screen detects. The second is a statement about one weighting, and it is
available cheaply from the fitted solution: at an optimum with residual
:math:`u^* = B w^* - A`, donor :math:`j` holds zero weight at every optimum of
that particular program exactly when
:math:`\langle x_j, u^* \rangle > \lVert u^* \rVert^2`. That is a diagnostic to
report at a fitted :math:`V`, never a filter to prune with: of 245 donors it
excluded at one weighting, 111 carried weight at another.

When the screen finds anything
------------------------------

This is the practical question, and the answer is narrow.

Shadiness needs more donors than matching conditions, and by a comfortable
margin. Holding :math:`J = 30` and varying :math:`m` over 40 Gaussian designs
per row:

.. list-table::
   :header-rows: 1
   :widths: 10 12 18 22 22

   * - :math:`m`
     - :math:`J/m`
     - exact fit
     - prunes something
     - median donors pruned
   * - 2
     - 15.0
     - 31
     - 9
     - 27
   * - 3
     - 10.0
     - 23
     - 17
     - 25
   * - 5
     - 6.0
     - 8
     - 32
     - 14
   * - 6
     - 5.0
     - 3
     - 37
     - 10
   * - 8
     - 3.8
     - 0
     - 37
     - 2
   * - 10
     - 3.0
     - 0
     - 12
     - 1
   * - 14 and above
     - 2.1 and below
     - 0
     - 0
     - --

Shadiness is squeezed from both ends. Above :math:`J/m` of about 10 the origin
falls inside the hull often enough that the screen is answering Proposition 1
and reporting an exact fit. Below about 3 there is nothing to prune. Between
them, at :math:`J/m` roughly 4 to 10, shady donors are abundant.

Applied panels sit below that band. On the Basque design every donor is sunny:
16 of 16 on the outcome-only specification with 20 pre-periods, and 16 of 16 in
the 13-predictor space where the backend runs the screen, matching the R package
MSCMT's own console line. Shady donors appear on that panel only when the
matching window is cut to 5 pre-periods (12 of 16 sunny), 3 (5 of 16) or 2
(2 of 16) -- designs nobody would estimate on. Across the four simulation
designs this library benchmarks, only Xu's produces shady donors at all, at 0 to
2.4 percent of the pool.

So on a conventional application the screen deletes nothing, and the value it
delivers is Proposition 1's case detection. That is the expected outcome, not a
shortfall.

The case where the answer is free
---------------------------------

When the centred columns are linearly independent the screen cannot find
anything, and a rank computation says so without solving anything.

If :math:`\operatorname{rank}(X_t) = J`, then :math:`\alpha x_j = X_t \lambda`
with :math:`\sum \lambda = 1` rearranges to
:math:`X_t (\lambda - \alpha e_j) = 0`, forcing :math:`\lambda = \alpha e_j` and
so :math:`\alpha = 1`. Every donor is sunny by algebra.
:func:`~mlsynth.utils.solvers.sunny.sunny_screen_is_vacuous` reports this, and
the screen short-circuits on it.

The distinction it draws matters to a reader of the output. An all-sunny verdict
on a design where the gate fires says nothing about the donor pool -- it is a
fact about the design's shape. An all-sunny verdict where the gate does not fire
says the pool really is irreducible. The counts alone cannot tell those apart,
which is why the flag is reported alongside them. On the four benchmarked
simulation designs, 35 of 47 cells are settled by the gate before any linear
program runs.

Example
-------

.. code:: python

    import numpy as np
    import pandas as pd
    from mlsynth.utils.solvers.sunny import (
        sunny_alphas, sunny_donors, sunny_screen_is_vacuous,
    )

    url = ("https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
           "main/basedata/basque_mscmt.csv")
    df = pd.read_csv(url)
    treated, aggregate = "Basque Country (Pais Vasco)", "Spain (Espana)"

    panel = df.pivot(index="year", columns="regionname", values="gdpcap")
    donors = [c for c in panel.columns if c not in (treated, aggregate)]

    def design(years):
        A = panel.loc[years, treated].to_numpy(float)
        B = np.column_stack([panel.loc[years, d].to_numpy(float) for d in donors])
        return B, A

    # The design Abadie and Gardeazabal estimate on: 20 pre-treatment periods
    # against 16 donors. The columns are independent, so the gate settles it.
    B, A = design([y for y in panel.index if y < 1975])
    print(sunny_screen_is_vacuous(B, A))            # True
    print(int(sunny_donors(B, A).sum()), "of", len(donors))   # 16 of 16

    # Cut the matching window to three periods and the pool becomes reducible.
    B, A = design([1972, 1973, 1974])
    print(sunny_screen_is_vacuous(B, A))            # False
    print(int(sunny_donors(B, A).sum()), "of", len(donors))   # 5 of 16

    # alpha* is the underlying quantity; 1 is sunny, below 1 is shady.
    print(np.round(sunny_alphas(B, A), 3))

Cost
----

The definition costs one linear program per donor. Two shortcuts apply, both
exact.

The rank gate above answers the whole question with one decomposition when it
fires. And when the design has been reduced to its column space -- which
preserves every inner product and leaves the origin in place, so the
classification is identical -- a small enough row count lets the sunny set be
read off a single convex hull instead. The threshold is measured: at six reduced
rows the hull route wins by 21 times or better at every donor count up to 500,
while at eight it runs about three times slower than the programs once there are
100 donors or more, because a hull of :math:`n` points in dimension :math:`d`
carries up to about :math:`n^{\lfloor d/2 \rfloor}` facets.

Verification
------------

The screen's own tests are ``mlsynth/tests/test_sunny_donors.py``,
``test_sunny_column_space.py`` and ``test_sunny_hull_route.py``. The predicate
was checked against the variational form of :math:`\alpha^*` to 6e-13 over 884
donors, and against the exposed-face characterisation over 84 designs with no
donor classified differently. The MSCMT backend that consumes it is
cross-validated against the R package in
`mscmt_basque <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/mscmt_basque.py>`__.

References
----------

Becker, M. and Klossner, S. (2018). Fast and reliable computation of generalized
synthetic controls. *Econometrics and Statistics*, 5, 1-19.

Abadie, A. and Gardeazabal, J. (2003). The economic costs of conflict: a case
study of the Basque Country. *American Economic Review*, 93(1), 113-132.

Core API
--------

.. automodule:: mlsynth.utils.solvers.sunny
   :members: sunny_alphas, sunny_donors, sunny_support, certified_sunny,
             sunny_screen_is_vacuous
   :noindex:
