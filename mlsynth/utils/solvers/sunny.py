"""Becker and Klossner's sunny/shady donor screen for simplex programs.

Write ``x_j = B[:, j] - A`` for the centred donor columns and
``H = conv(x_1, ..., x_J)``. The screen asks one question per donor:

    alpha*(j) = min { alpha >= 0 : alpha x_j in H }

Donor ``j`` is sunny when ``alpha*(j) = 1`` and shady when ``alpha*(j) < 1``, i.e.
when some shorter multiple of its own column already lies in the hull. Since
``x_j`` is one of the generators of ``H``, ``alpha = 1`` is always feasible, so
``alpha* <= 1`` and the test is one-sided by construction.

Two facts give the screen its use, and both are pinned in
``mlsynth/tests/test_sunny_donors.py``:

Proposition 1. No donor is sunny exactly when ``0 in H``, which is exactly when
``min ||A - B w||`` over the simplex is zero. So a screen that returns all-shady
has found an exact fit, not a set of columns to drop.

Proposition 2. If no exact fit exists, every optimum puts zero weight on every
shady donor. At an optimum ``u* = B w* - A`` the simplex stationarity conditions
give ``<x_k, u*> = min_i <x_i, u*> = ||u*||^2`` for each ``k`` in the support, and
the feasible move ``w* - t e_j + t lambda`` built from ``alpha x_j = sum_i lambda_i
x_i`` sends ``u*`` to ``u* - t (1 - alpha) x_j``. Together these force
``alpha ||u*||^2 >= ||u*||^2``, so ``alpha < 1`` implies ``u* = 0``. The conclusion
holds at every optimum, so the columns may be dropped even when the argmin is a
face.

The hypothesis on Proposition 2 is not slack that can be ignored. A donor with
``x_j = 0`` reproduces the treated path on its own and is shady with
``alpha*(j) = 0``; dropping it would discard an exact fit. Proposition 1 is what
rules this out: when at least one donor is sunny there is no exact fit, and only
then is pruning licensed. ``sunny_support`` encodes that branch.

``alpha*`` costs one linear program per donor. ``certified_sunny`` decides a subset
of them from a single Gram matrix instead. For any direction ``c``,
``alpha x_j = sum_i lambda_i x_i`` gives ``alpha c'x_j = sum_i lambda_i c'x_i >=
min_i c'x_i``, so whenever ``c'x_j > 0``

    alpha*(j) >= (min_i c'x_i) / (c'x_j),

and a bound reaching 1 certifies donor ``j`` sunny. Sweeping ``c`` over the donor
columns themselves reads every such bound off ``G = Xt' Xt``: row ``i`` supplies
``min_k G[i, k] / G[i, j]`` for each ``j`` with ``G[i, j] > 0``. The direction that
certifies a donor is usually not its own column, so the ``c = x_j`` term alone
leaves most of the power unused -- on 40x8 Gaussian designs the own-column test
certifies nothing at all while the full sweep certifies 40 percent of donors.

Where the screen pays, the certificate helps least. Over 200 Gaussian designs, 40
per regime, it certifies 40 percent of donors at 40x8, 29 percent at 12x12, 10
percent at 8x40 and 2 percent at 5x74 -- and the first two regimes have every donor
sunny, so there is nothing there to prune. At 5x74, where only 10 percent of donors
are sunny and pruning pays, skipping the certified donors cuts 2 percent off the
runtime. The certificate is sound and near free, not a replacement for the linear
program.

One family of designs admits no shady donor at all, and the screen detects it for
the price of one rank. If ``Xt`` has trivial kernel then ``alpha x_j = Xt lambda``
rearranges to ``Xt (lambda - alpha e_j) = 0``, forcing ``lambda = alpha e_j`` and so
``sum lambda = alpha = 1``: every donor is sunny by algebra. Linear independence of
the centred columns is equivalent to their being affinely independent with ``0``
outside their affine hull, so this is exactly the case Proposition 1 leaves no room
in. ``sunny_screen_is_vacuous`` reports it and the gate skips all ``J`` linear
programs.

The case is the ordinary one for outcome-only panels with a long pre-period. On
Abadie and Gardeazabal's Basque design with treatment in 1975 and ``gdpcap`` alone,
20 pre-periods against 16 donors give ``rank(Xt) = 16``, and every donor comes back
at ``alpha* = 1`` exactly. Shortening the window does not help much: 16 of 16 are
still sunny at 10 and at 7 pre-periods, and in the 13-predictor space where MSCMT
runs the screen the columns are affinely dependent yet all 16 remain sunny. Shady
donors appear only at 5 pre-periods (12 of 16 sunny), 3 (5 of 16) and 2 (2 of 16).
The concept bites when ``J`` greatly exceeds ``m``, not merely when it exceeds it.

``alpha*`` is the radial function of ``H`` read from outside. Hug and Weil
(Lectures on Convex Geometry, Exercise 2.3.3) define ``rho(K, x) = max{lam >= 0 :
lam x in K}`` for ``0`` in the interior of ``K``, the far intersection of the ray
with the body, and give ``rho = h(K-polar, .)^-1``. Here ``0`` lies outside ``H``
and the near intersection is wanted, so the same ray construction runs on the other
side of the origin. Writing ``H`` through its support function, ``alpha x_j in H``
holds exactly when ``alpha <c, x_j> <= h(H, c)`` for every ``c``; splitting on the
sign of ``<c, x_j>`` and using ``-h(H, -c) = min_i <c, x_i>`` gives

    alpha*(j) = sup over c with <c, x_j> > 0 of  min_i <c, x_i> / <c, x_j>,

the supremum being attained because normalising ``<c, x_j> = 1`` makes it a linear
program, feasible for ``x_j`` nonzero and bounded by ``t <= 1``. Checked against
Eq (9) to 6e-13 over 884 donors on designs with ``0`` outside ``H``; where ``0`` is
inside, Eq (9) clamps ``alpha >= 0`` while the formula runs negative, and the two
still agree on the sunny predicate.

Three things follow. The certificate above is this supremum evaluated at the ``J``
directions ``c = x_i``, a finite sample of a supremum over the sphere, which is why
it is weak and why no cheaper sound certificate is available: the exact predicate is
the supremum, and taking a supremum over directions is a linear program. Second,
sunny is equivalent to the existence of a single ``c`` with ``<c, x_j> > 0`` and
``<c, x_j> = min_i <c, x_i>`` -- a hyperplane supporting ``H`` at ``x_j`` that
separates the origin -- so ``bilevel/mscmt.py::_sunny_mask``, which solves for that
``c``, and Eq (9), which solves for ``alpha``, are the two sides of one linear
program. Third, the transposed shape buys no speed: timed across five designs the
support-side program runs at 0.84 to 0.98 times Eq (9), so the formulation explains
the code here without replacing it.

In the vocabulary of convex geometry the predicate has a short form. Balestro,
Martini and Teixeira give ``h_P(u) = max_j <x_j, u>`` for a polytope generated by
the ``x_j`` (Lemma 2.3.4), ``x`` in ``K`` exactly when ``<x, u> <= h_K(u)`` for every
``u`` (Corollary 2.3.1), and ``F(K, u) = {x in K : <x, u> = h_K(u)}`` as the face
exposed by ``u`` (Exercise 2.36); Hug and Weil carry the same two identities as
Theorem 2.7(b) and 2.7(g). Together with the supremum formula above,

    x_j is sunny  <=>  x_j lies on an exposed face F(H, u) with h_H(u) < 0,

since ``h_H(u) < 0`` puts the origin strictly outside that face's supporting
halfspace. Tested as a linear program against Eq (9) over 84 designs, no donor
classified differently. This is also why sunny does not mean vertex: a sunny donor
may sit in the relative interior of a higher-dimensional exposed face, as
``x_1 = (1, 0)`` does among ``(1, 2)`` and ``(1, -2)`` in the tests here.

The quantity itself is not named in the standard references, and three candidate
classical objects each miss it. The radial function ``rho(K, x) = max{lam >= 0 :
lam x in K}`` (Hug and Weil Exercise 2.3.3; Balestro et al. Section 2.3) requires the
origin in the interior of ``K`` and takes the far intersection of the ray. The
Minkowski gauge ``p_C(x) = inf{lam >= 0 : x in lam C}`` (Correa, Hantoute and Lopez
(2.25)) needs no interior origin but scales the set instead of the point and takes an
infimum: measured on a donor with ``alpha* = 0.530`` its reciprocal reads 1.000, so it
does not see shadiness. The support cone ``cl{lam (z - x) : z in K, lam > 0}``
(Balestro et al., Section 2.4) has its apex at a point of the body, not outside it.
Illumination appears only as Hadwiger and Boltyanski's covering problem for parallel
light, and is absent from Hug and Weil; the phrase "central illumination", which names
this configuration, occurs twice in Balestro et al. without a definition.

One consequence for the proof. The equivalence cannot be had by separating ``H`` from
the half-open segment ``{alpha x_j : 0 <= alpha < 1}``: that segment is not compact, so
Hug and Weil's Remark 1.18 gives no strong separation and Theorem 1.17 gives only
proper separation, which leaves the supporting value at zero and the argument stuck.
Attainment in the linear program above is what closes it.

Reference: Becker and Klossner (2018), Fast and reliable computation of generalized
synthetic controls, Econometrics and Statistics 5, 1-19, Section 3.1 and Definition
1; ``isSunny`` in MSCMT's ``R/Helpers.r`` and the donor loop in ``R/multiOpt.r``.
Convex geometry as cited above: Hug and Weil, Lectures on Convex Geometry (GTM 286);
Balestro, Martini and Teixeira, Convexity from the Geometric Point of View; Correa,
Hantoute and Lopez, Fundamentals of Convex Analysis and Optimization.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import linprog

__all__ = ["sunny_alphas", "certified_sunny", "sunny_donors", "sunny_support",
           "sunny_screen_is_vacuous"]

#: A donor counts as sunny when ``alpha*`` reaches 1 to within this slack. The
#: linear program returns ``alpha*`` to roughly solver precision, so the default
#: sits several orders above that and well below the gap to any shady donor met in
#: practice.
DEFAULT_TOL = 1e-7


def _centred(B: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Validate the design and return the centred donor columns ``B - A``."""
    B = np.asarray(B, dtype=float)
    A = np.asarray(A, dtype=float)
    if B.ndim != 2:
        raise ValueError(f"B must be a 2-D (m, J) matrix; got shape {B.shape}.")
    if B.shape[1] == 0:
        raise ValueError("B has no columns; the screen needs at least one donor.")
    if A.ndim != 1:
        raise ValueError(f"A must be a 1-D vector of length m; got shape {A.shape}.")
    if A.shape[0] != B.shape[0]:
        raise ValueError(
            f"A has {A.shape[0]} entries but B has {B.shape[0]} rows; they must match."
        )
    if not (np.isfinite(B).all() and np.isfinite(A).all()):
        raise ValueError("B and A must be finite; got a NaN or an infinity.")
    return B - A[:, None]


def _program(Xt: np.ndarray):
    """Build the shared part of the donor LP.

    The variables are ``[alpha, lambda_1, ..., lambda_J]``; the rows say
    ``alpha x_j = Xt lambda`` and ``sum lambda = 1``. Only the first column depends
    on ``j``, so the rest is built once and reused across donors.
    """
    m, J = Xt.shape
    A_eq = np.zeros((m + 1, J + 1))
    A_eq[:m, 1:] = -Xt
    A_eq[m, 1:] = 1.0
    b_eq = np.zeros(m + 1)
    b_eq[m] = 1.0
    c = np.zeros(J + 1)
    c[0] = 1.0
    return A_eq, b_eq, c


def _alpha(Xt: np.ndarray, j: int, A_eq: np.ndarray, b_eq: np.ndarray,
           c: np.ndarray) -> float:
    """Solve for ``alpha*(j)``.

    ``alpha = 1, lambda = e_j`` is always feasible and ``alpha >= 0`` bounds the
    objective, so a failure here is numerical. The fallback is 1, which reports the
    donor as sunny: that keeps the column, where a wrong "shady" would drop one
    that should carry weight.
    """
    A_eq[:-1, 0] = Xt[:, j]
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0.0, None), method="highs")
    if not res.success:
        warnings.warn(
            f"The sunny/shady linear program for donor {j} did not solve "
            f"({res.message.strip()}); reporting the donor as sunny, which keeps it.",
            RuntimeWarning,
            stacklevel=3,
        )
        return 1.0
    return float(res.fun)


def sunny_screen_is_vacuous(B: np.ndarray, A: np.ndarray) -> bool:
    """Report whether no donor on this design can be shady.

    True when the centred columns are linearly independent, which forces
    ``alpha*(j) = 1`` for every donor. A caller that sees True learns that the screen
    has nothing to say here and can skip it: there is no pruning to be had, whatever
    the treated path looks like.
    """
    return _full_column_rank(_centred(B, A))


def _full_column_rank(Xt: np.ndarray) -> bool:
    """Decide ``rank(Xt) == J``, conservatively.

    A near-dependent design is reported as dependent, which costs the linear programs
    and cannot give a wrong answer; claiming independence that does not hold would
    report a shady donor as sunny.
    """
    m, J = Xt.shape
    if m < J:
        return False
    sv = np.linalg.svd(Xt, compute_uv=False)
    return bool(sv[-1] > 1e-10 * sv[0])


def sunny_alphas(B: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Return ``alpha*(j)`` for every donor, one linear program each.

    This is the reference: it consults no certificate, so it is what the cheap
    tests are measured against.
    """
    Xt = _centred(B, A)
    if _full_column_rank(Xt):
        return np.ones(Xt.shape[1])
    A_eq, b_eq, c = _program(Xt)
    return np.array([_alpha(Xt, j, A_eq, b_eq, c) for j in range(Xt.shape[1])])


def certified_sunny(B: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Flag the donors that a single Gram matrix proves sunny.

    Row ``i`` of ``G = Xt' Xt`` carries the bound from direction ``c = x_i``, namely
    ``min_k G[i, k] / G[i, j]`` for every ``j`` whose inner product with ``x_i`` is
    positive; donor ``j`` is certified when the best bound over all rows reaches 1.
    The flags are a subset of the sunny donors -- a donor left unflagged has not been
    shown to be shady and still needs its linear program.
    """
    Xt = _centred(B, A)
    G = Xt.T @ Xt
    scale = float(np.diag(G).max())
    if not scale > 0.0:
        return np.zeros(Xt.shape[1], dtype=bool)
    usable = G > 1e-12 * scale
    bounds = np.where(
        usable, G.min(axis=1)[:, None] / np.where(usable, G, 1.0), -np.inf
    )
    return bounds.max(axis=0) >= 1.0


def sunny_donors(B: np.ndarray, A: np.ndarray, *, tol: float = DEFAULT_TOL,
                 certify: bool = True) -> np.ndarray:
    """Classify every donor as sunny (``True``) or shady (``False``).

    With ``certify`` the Gram test above settles what it can and a linear program is
    solved only for the rest; the answer is the same either way, since the
    certificate proves sunniness and never asserts shadiness.
    """
    Xt = _centred(B, A)
    if _full_column_rank(Xt):
        return np.ones(Xt.shape[1], dtype=bool)
    flags = (certified_sunny(B, A) if certify
             else np.zeros(Xt.shape[1], dtype=bool))
    undecided = np.flatnonzero(~flags)
    if undecided.size:
        A_eq, b_eq, c = _program(Xt)
        for j in undecided:
            flags[j] = _alpha(Xt, j, A_eq, b_eq, c) >= 1.0 - tol
    return flags


def sunny_support(B: np.ndarray, A: np.ndarray, *, tol: float = DEFAULT_TOL,
                  certify: bool = True) -> np.ndarray:
    """Return the donor indices a simplex solve can be restricted to.

    When some donor is sunny these are the sunny ones: by Proposition 2 the shady
    columns take zero weight at every optimum, so the restricted program has the
    same value and the same reachable fits. When no donor is sunny an exact fit
    exists, Proposition 2 does not apply, and every index is returned -- the shady
    columns are the ones that reach the fit.
    """
    flags = sunny_donors(B, A, tol=tol, certify=certify)
    if not flags.any():
        return np.arange(flags.size)
    return np.flatnonzero(flags)
