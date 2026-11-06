"""Exact lower bounds for every stage of the certified search (#965).

Hull QP. f(w) = ||y - X w||^2 is convex, so f(w) >= f(wbar) + grad f(wbar)'(w - wbar)
for every w and any wbar (Boyd and Vandenberghe 2004, Sect. 3.1.3, eq. (3.2)), and
the minimum of the right side over the hull is a lower bound whatever solver
produced wbar. The linear minimum over {0 <= w <= hi, m lo <= sum w <= m hi} is
a continuous knapsack, solved by filling the cheapest coordinates first
(Dantzig 1957).

Shor SDP. The same certificate as the joint lift below (Jansson, Chaykin and Keil 2007,
Thm 3.2), with tr Z = 1 + tr W <= 1 + hi sum(w) <= 1 + m hi^2 from diag(W) <= hi w
and sum(w) <= m hi (derived here).

Slope cap, derived here. A design whose RSS is below y'y has b = y'x / x'x > 0,
and by Cauchy-Schwarz b <= ||y|| / ||x||; x'x = u'G u >= lambda_min(G) ||u||^2
>= lambda_min(G) for a nonzero binary u (the Rayleigh quotient bound,
Horn and Johnson 2013, Sect. 4.2), so b <= ||y|| / sqrt(lambda_min(G)), with lambda_min(G) bounded below
exactly by lambda_min_lower.

Boyd, S. and Vandenberghe, L. (2004). Convex Optimization. Cambridge.
Dantzig, G. B. (1957). Discrete-variable extremum problems. Oper. Res. 5, 266-288.
Horn, R. A. and Johnson, C. R. (2013). Matrix Analysis, 2nd ed. Cambridge.
Jansson, C., Chaykin, D. and Keil, C. (2007). SIAM J. Numer. Anal. 46, 180-200.

Joint lift. Jansson, Chaykin and Keil (2007), Thm 3.2, with one semidefinite block and its
Lemma 3.1 step <D, X> >= lambda_min(D) tr(X) for X >= 0 used with a trace bound
in place of the eigenvalue bound: any multipliers, inequality ones clipped at
zero, give the affine Lagrangian L(Z) = <S, Z> + k with obj(Z) >= L(Z) on the
lift's feasible set (weak duality), so obj >= k + min(0, lambda_min(S)) tr_max.

The trace bound is derived here. On the lift's feasible set
tr Z = 1 + sum(U_jj) + sum(W_jj) = 1 + m + tr W, and the diagonal of the
products (w - lo u)(hi u - w) >= 0 and (hi u - w) u >= 0 gives
W_jj <= (lo + hi) w_j - lo hi u_j and w_j <= hi u_j, so tr W <= m hi^2 for lo >= 0.

S and k are summed in rationals from the double data and the double multipliers,
and lambda_min(S) is bounded below exactly by the library's Cholesky-residual
bound (mlsynth.utils.marex_helpers.branch.lambda_min_lower, by Weyl's and
Gershgorin's theorems as its docstring states). The data are the doubles y and X
as given. Fixed geos enter as equality multipliers on u_j. The no-good cut
sum_{C} u - sum_{not C} u <= |C| - 1 excludes the incumbent design C and no other
(Balas and Jeroslow 1972), so a node whose certified bound clears the incumbent
holds no design below it.

Jansson, C., Chaykin, D. and Keil, C. (2007). Rigorous error bounds for the
    optimal value in semidefinite programming. SIAM J. Numer. Anal. 46, 180-200.
Balas, E. and Jeroslow, R. (1972). Canonical cuts on the unit hypercube. SIAM J.
    Appl. Math. 23, 61-69.
"""
import functools
import math
from fractions import Fraction as Q
from types import SimpleNamespace

import numpy as np

from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.marex_helpers.branch import _from_fractions, lambda_min_lower

from .oracle import FAMS, TERMS, _coef, unpack


def shor_trace(m, hi):
    """tr Z <= 1 + m hi^2 on the Shor lift's feasible set (diag W <= hi w, sum w <= m hi)."""
    return 1 + m * hi * hi


def joint_trace(m, hi):
    """tr Z <= 1 + m + m hi^2 on the joint lift's feasible set, lo >= 0 (module docstring)."""
    return 1 + m + m * hi * hi


def qarr(A):
    return np.vectorize(Q, otypes=[object])(np.asarray(A, dtype=float))


def _integer_matrix(A):
    """The doubles A as Python integers over one power of two D: A = N / D exactly.
    A finite double is an integer over a power of two, n / d, and with D the largest
    of the d every D / d is an integer."""
    A = np.asarray(A, dtype=float)
    if not np.isfinite(A).all():
        raise MlsynthDataError("the exact data need finite values: the series has an infinite or "
                               "missing entry.")
    parts = [v.as_integer_ratio() for v in A.ravel().tolist()]
    D = max((d for _, d in parts), default=1)
    N = np.empty(len(parts), dtype=object)
    N[:] = [n * (D // d) for n, d in parts]
    return N.reshape(A.shape), D


class ExactData:
    """y and X as given (doubles), their exact Gram, cross products and norms.

    The products are of Python integers over one power of two (_integer_matrix):
    with y = n / Dy and X = N / Dx, X'X = N'N / Dx^2, X'y = N'n / (Dx Dy) and
    y'y = n'n / Dy^2, one division each. The Fraction products they replace reduce
    a gcd at every partial sum; on a GeoLift group (90 periods, 36 controls) they
    took 0.19 s a group."""

    def __init__(self, y, X):
        n, Dy = _integer_matrix(y)
        N, Dx = _integer_matrix(X)
        self._y, self._X = y, X
        to_q = np.vectorize(Q, otypes=[object])
        self.G = to_q(N.T @ N, Dx * Dx) if N.size else np.empty((N.shape[1], N.shape[1]), dtype=object)
        self.c = to_q(N.T @ n, Dx * Dy) if N.size else np.empty(N.shape[1], dtype=object)
        self.yy = Q(int(n @ n) if n.size else 0, Dy * Dy)
        self.p = X.shape[1]

    @functools.cached_property
    def yq(self):
        return qarr(self._y)

    @functools.cached_property
    def Xq(self):
        return qarr(self._X)

    def rss(self, C, b):
        """The exact RSS of control set C at slope b (a double): an upper bound
        on the incumbent's value at its best slope."""
        x = self.Xq[:, sorted(C)].sum(axis=1); r = self.yq - Q(float(b)) * x
        return r @ r

    def slope_cap(self):
        """No design whose RSS is below y'y has slope b = y'x/x'x above
        ||y|| / sqrt(lambda_min(G)) (x'x = u'Gu >= m lambda_min(G) >= lambda_min(G)).
        Exact: a rational upper bound on the ratio, or None."""
        lam = lambda_min_lower(_from_fractions(self.G.tolist()))
        if lam is None or lam <= 0:
            return None
        ratio2 = self.yy / lam                 # b^2 <= y'y / lambda_min
        r = Q(math.sqrt(float(ratio2)) * (1 + 1e-12))
        while r * r < ratio2:                   # pragma: no cover - rounding guard
            r *= Q(1000001, 1000000)
        return r


def hull_bound(D, wbar, lo, hi, m):
    """min ||y - X w||^2 over {0 <= w <= hi, m lo <= sum w <= m hi} from below:
    f(wbar) + min_P g'(w - wbar), g the gradient at wbar; valid for any wbar
    because f is convex. The linear minimum over P is a continuous knapsack."""
    lo, hi, m = Q(float(lo)), Q(float(hi)), Q(int(m))
    w = qarr(wbar)
    Gw = D.G @ w
    f = D.yy - 2 * (D.c @ w) + w @ Gw
    g = 2 * (Gw - D.c)
    A, B = m * lo, m * hi
    v = np.full(D.p, Q(0), dtype=object); s = Q(0)
    for j in sorted(range(D.p), key=lambda j: g[j]):
        if s >= B or (g[j] >= 0 and s >= A):
            break
        need = (B - s) if g[j] < 0 else (A - s)
        t = min(hi, need); v[j] = t; s += t
    return f + g @ (v - w)


U = 2.0 ** -53                          # unit roundoff of IEEE double
ETA = 2.0 ** -1074                       # the smallest subnormal double


def _gamma(k):
    """gamma_k = k u / (1 - k u), the rounding constant of Bjorck (2015), Lemma 1.4.2."""
    return k * U / (1.0 - k * U)


def _dot_err(k, moduli):
    """A bound on the rounding of k-term dot products whose moduli are given:
    gamma_k |a|'|b| (Bjorck 2015, eq. (1.4.10)), which rests on the standard
    model (1.4.5) and so holds only if no underflow occurs, plus k ETA for it.
    Derived here from Russinoff (2022): every denormal is an integer multiple of
    the smallest positive denormal, ETA = 2^-1074 for doubles (Lemma 5.11 and
    Definition 5.19), and denormal rounding rounds to that grid (Definition 6.9).
    Every double is a multiple of ETA, so a sum whose exact value lands in the
    subnormal range is representable and exact, and a product rounded to nearest
    there errs by at most ETA / 2. Each of the k products adds at most ETA / 2,
    and the later roundings scale that by at most 1 + gamma_k < 2, so k ETA covers it."""
    return _gamma(k) * moduli + k * ETA


def knapsack_dual(g, lo, hi, m, lam=None):
    """A lower bound on min g'x over {0 <= x <= hi, m lo <= sum x <= m hi} from the
    LP dual of the sum constraint, valid at every multiplier lam (weak duality):

        sum_j hi min(0, g_j - lam) + lam A   (lam >= 0, A = m lo),
        sum_j hi min(0, g_j - lam) + lam B   (lam <  0, B = m hi).

    Derived here: for lam >= 0, g'x >= g'x - lam (sum x - A) on the set, and the
    right side is minimised over the box alone coordinate by coordinate; for
    lam < 0 the same with the upper end. The default lam is the greedy fill's
    marginal cost (Dantzig 1957), where the bound equals the knapsack minimum:
    the cost of the coordinate the fill stops in when it stops at an end of the
    band, and zero when it stops strictly inside, where the sum is slack.
    """
    g = np.asarray(g, dtype=float)
    A, B = m * lo, m * hi
    if lam is None:
        lam = _greedy_lambda(g, lo, hi, m)
    terms = hi * np.minimum(0.0, g - lam)
    return float(terms.sum()) + (lam * A if lam >= 0 else lam * B)


def hull_bound_float(y, X, w, lo, hi, m):
    """hull_bound in floating point, rounded so it never exceeds the exact value.

    hull_bound is f(w) + min_P g'(x - w) with f(w) = ||y - X w||^2, g its gradient
    and P = {0 <= x <= hi, m lo <= sum x <= m hi}. Here each quantity is a double,
    and each rounding error is bounded by the a priori bound for a dot product of
    k terms, |fl(a'b) - a'b| < gamma_k |a|'|b| (Bjorck 2015, Lemma 1.4.2 and
    eq. (1.4.10), in the notation of Higham 2002), and subtracted. Derived here
    from that bound:

      r~ = fl(X w - y),  |r~ - r| <= e = gamma_{p+1} (|X||w| + |y|),
      f(w) >= fl(r~'r~) - gamma_n |r~|'|r~| - 2 |r~|'e        (since e'e >= 0),
      g~ = fl(2 X'r~),   |g~ - g| <= d = 2 gamma_n |X|'|r~| + 2 |X|'e,
      min_P g'x >= knapsack_dual(g~) - err_dual - max(d) m hi  (sum x <= m hi),
      -g'w >= -fl(g~'w) - gamma_p |g~|'|w| - d'|w|,

    with err_dual = gamma_{p+3} (sum_j |hi min(0, g~_j - lam)| + |lam| m max(lo, hi)),
    which covers the rounding of the dual's terms, their sum, lam A and of
    A = fl(m lo), B = fl(m hi) themselves. The error terms are themselves doubles,
    and inflate() raises their total to cover their own rounding. Every dot-product
    bound also carries an allowance for underflow (_dot_err), and an overflow
    returns -inf, a bound that clears nothing. The last combination f + dual - g'w - err is done in
    rationals from those few doubles and rounded down, so it adds no error.

    Bjorck, A. (2015). Numerical Methods in Matrix Computations. Springer.
    Higham, N. J. (2002). Accuracy and Stability of Numerical Algorithms, 2nd ed.
        SIAM.
    Russinoff, D. M. (2022). Formal Verification of Floating-Point Hardware
        Design, 2nd ed. Springer.
    """
    y = np.asarray(y, dtype=float); X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)
    n, p = X.shape
    if w.shape != (p,):
        raise ValueError(f"the point must hold one weight per control: expected ({p},), got {w.shape}")
    if not np.all(np.isfinite(w)):
        raise ValueError("the point must be finite")
    r, e = residual_bound(y, X, w)
    f, f_err = objective_bound(r, e)
    g, d = gradient_bound(X, r, e)
    kn, kn_err = perturbed_knapsack_bound(g, d, lo, hi, m)
    gw, gw_err = linear_bound(g, d, w)
    err = inflate(f_err + kn_err + gw_err)
    if not all(np.isfinite(v) for v in (f, kn, gw, err)):
        return -np.inf                  # an overflow: no bound, and the exact certificate decides
    return round_down(Q(f) + Q(kn) - Q(gw) - Q(err))


def residual_bound(y, X, w):
    """r~ = fl(X w - y) and e with |r~ - (X w - y)| <= e componentwise: a dot
    product of p terms less y_i, so gamma_{p+1} (|X||w| + |y|) (Bjorck 2015, eq. (1.4.10))."""
    r = X @ w - y
    return r, _dot_err(X.shape[1] + 1, np.abs(X) @ np.abs(w) + np.abs(y))


def objective_bound(r, e):
    """fl(r~'r~) and the error with r'r >= fl(r~'r~) - err for every r within e of
    r~: r'r >= r~'r~ - 2 |r~|'e, and fl(r~'r~) errs by at most gamma_n |r~|'|r~|."""
    ar = np.abs(r)
    return float(r @ r), _dot_err(r.shape[0], float(ar @ ar)) + 2.0 * float(ar @ e)


def gradient_bound(X, r, e):
    """g~ = fl(2 X'r~) and d with |g~ - 2 X'r| <= d for every r within e of r~:
    the rounding of X'r~, gamma_n |X|'|r~|, and |X'(r~ - r)| <= |X|'e, doubled."""
    aX = np.abs(X)
    return 2.0 * (X.T @ r), 2.0 * _dot_err(X.shape[0], aX.T @ np.abs(r)) + 2.0 * (aX.T @ e)


def dual_bound(g, lo, hi, m):
    """knapsack_dual at the greedy multiplier, and a bound on its rounding: each
    term hi min(0, g_j - lam) and lam A or lam B within gamma_3 of itself (one
    subtraction, one product, and A = fl(m lo), B = fl(m hi)), their sum within
    gamma_p, so gamma_{p+3} times the moduli bounds the whole (Bjorck 2015, Lemma 1.4.2)."""
    lam = _greedy_lambda(g, lo, hi, m)
    terms = hi * np.minimum(0.0, g - lam)
    err = _dot_err(g.shape[0] + 3, float(np.abs(terms).sum()) + abs(lam) * m * max(lo, hi))
    return knapsack_dual(g, lo, hi, m, lam=lam), err


def perturbed_knapsack_bound(g, d, lo, hi, m):
    """A value and an error with min_P g'x >= value - err for every gradient g
    within d of the computed g~: the dual at g~ less its rounding, and less
    max(d) m hi, since every x in P has 0 <= x and sum x <= m hi, so
    |(g - g~)'x| <= max(d) sum x (derived here)."""
    dual, dual_err = dual_bound(g, lo, hi, m)
    return dual, dual_err + float(d.max()) * (m * hi)


def linear_bound(g, d, w):
    """fl(g~'w) and an error with g'w <= fl(g~'w) + err for every g within d of g~:
    gamma_p |g~|'|w|, the rounding (Bjorck 2015, eq. (1.4.10)), plus d'|w|."""
    aw = np.abs(w)
    return float(g @ w), _dot_err(g.shape[0], float(np.abs(g) @ aw)) + float(d @ aw)


def inflate(err):
    """An error total computed in doubles, raised to cover its own rounding: it is a
    sum of nonnegative products of doubles, within a relative gamma_{n+p+8} of its
    exact value, and 1e-6 exceeds that whenever n + p < 10^9 (derived here)."""
    return err * (1.0 + 1e-6)


def round_down(x):
    """The largest double at most the rational x: float() rounds to nearest, and a
    result above x is stepped down one ulp (an exact comparison decides)."""
    out = float(x)
    if Q(out) > x:
        out = float(np.nextafter(out, -np.inf))
    return out


def _greedy_lambda(g, lo, hi, m):
    """The multiplier knapsack_dual uses by default (see its docstring)."""
    A, B = m * lo, m * hi
    s, last = 0.0, 0.0
    for j in np.argsort(g, kind="stable"):
        if s >= B or (g[j] >= 0 and s >= A):
            break
        last = float(g[j])
        s += min(hi, (B - s) if g[j] < 0 else (A - s))
    return last if (s >= B or last >= 0) else 0.0


def hull_clears(D, y, X, w, lo, hi, m, UB, stats):
    """Whether the hull's first-order certificate at w is at least UB: the float
    bound decides when it clears UB, and the exact one otherwise. The float bound
    never exceeds the exact one, so the decision is the exact certificate's."""
    stats["float"] = stats.get("float", 0) + 1
    fb = hull_bound_float(y, X, w, lo, hi, m)
    if np.isfinite(fb) and Q(fb) >= UB:
        return True
    stats["exact"] = stats.get("exact", 0) + 1
    return hull_bound(D, w, lo, hi, m) >= UB


def shor_bound(lift, D, lo, hi, m, scale, duals=None):
    """Exact bound on the Shor lift (lifts.Lift level 'shor') from its last solve."""
    p = D.p; n = p + 1
    lo, hi, m = Q(float(lo)), Q(float(hi)), Q(int(m))
    s2 = Q(float(scale)) ** 2
    if duals is None:
        duals = [c.dual_value for c in lift.prob.constraints[1:]]
    duals = [qarr(d) * s2 for d in duals]
    Gf = np.full((n, n), Q(0), dtype=object); k = Q(0)
    Gf[1:, 1:] += D.G; Gf[0, 1:] += -2 * D.c; k += D.yy
    tags = ["Z00", "w>=0", "w<=hi", "dW>=lo w", "dW<=hi w", "sw>=m lo", "sw<=m hi",
            "rW>=m lo w", "rW<=m hi w", "W>=0"]
    assert len(tags) == len(duals)
    # A solver returns a 1x1 block or a length-1 vector as a scalar; every block is
    # read at its declared shape, so one candidate control is handled as p of them.
    shapes = {"Z00": (), "sw>=m lo": (), "sw<=m hi": (), "W>=0": (p, p)}
    duals = [np.reshape(d, shapes.get(t, (p,))) for t, d in zip(tags, duals)]
    for t, d in zip(tags, duals):
        if t != "Z00":
            d = np.maximum(d, Q(0))
        if t == "Z00": Gf[0, 0] += d; k -= d
        elif t == "w>=0": Gf[0, 1:] -= d
        elif t == "w<=hi": Gf[0, 1:] += d; k -= hi * d.sum()
        elif t == "dW>=lo w":
            Gf[0, 1:] += lo * d
            for i in range(p): Gf[1 + i, 1 + i] -= d[i]
        elif t == "dW<=hi w":
            Gf[0, 1:] -= hi * d
            for i in range(p): Gf[1 + i, 1 + i] += d[i]
        elif t == "sw>=m lo": Gf[0, 1:] -= d; k += m * lo * d
        elif t == "sw<=m hi": Gf[0, 1:] += d; k -= m * hi * d
        elif t == "rW>=m lo w":
            Gf[0, 1:] += m * lo * d; Gf[1:, 1:] -= d[:, None] * np.ones(p, dtype=object)[None, :]
        elif t == "rW<=m hi w":
            Gf[0, 1:] -= m * hi * d; Gf[1:, 1:] += d[:, None] * np.ones(p, dtype=object)[None, :]
        elif t == "W>=0": Gf[1:, 1:] -= d
    Ssym = (Gf + Gf.T) / 2
    lam = lambda_min_lower(_from_fractions(Ssym.tolist()))
    if lam is None:
        return None, Ssym, k
    return k + min(Q(0), lam) * shor_trace(m, hi), Ssym, k


FACT = {"u": ({"U": 1}, {"V": 1}, {"Ui": 1}),
        "1-u": ({"Uj": 1, "U": -1}, {"Wj": 1, "V": -1}, {"1": 1, "Ui": -1}),
        "a": ({"Vt": 1, "U": "-lo"}, {"W": 1, "V": "-lo"}, {"Wi": 1, "Ui": "-lo"}),
        "c": ({"U": "hi", "Vt": -1}, {"V": "hi", "W": -1}, {"Ui": "hi", "Wi": -1})}


def _lin(d, lo, hi):
    out = {}
    for k, v in d.items():
        v = {"-lo": -lo, "hi": hi}.get(v, v) if isinstance(v, str) else Q(v)
        out[k] = out.get(k, 0) + v
    return out


def _add(*terms):
    out = {}
    for coef, d in terms:
        for k, v in d.items():
            out[k] = out.get(k, 0) + coef * v
    return out


def family(a, g, lo, hi):
    pu, pw, lin = (_lin(x, lo, hi) for x in FACT[a])
    return {"u": pu, "1-u": _add((1, lin), (-1, pu)),
            "a": _add((1, pw), (-lo, pu)), "c": _add((hi, pu), (-1, pw))}[g]




def joint_bound(lift, y, X, lo, hi, m, skip=(), keep_sym=True, duals=None, fixes=(), nogood=None):
    """The certified bound in the units of y, X, from the lift's last solve."""
    p = X.shape[1]; n = 2 * p + 1
    lo, hi, m = Q(float(lo)), Q(float(hi)), Q(int(m))
    s2 = Q(float(lift.scale)) ** 2      # multipliers of the scaled problem, rescaled
    if duals is None:
        duals = [c.dual_value for c in lift.prob.constraints[1:]]
    duals = [qarr(d) * s2 for d in duals]
    Xq, yq = qarr(X), qarr(y)
    G = Xq.T @ Xq; c = Xq.T @ yq; yy = yq @ yq
    Gf = np.full((n, n), Q(0), dtype=object); k = Q(0)
    iu, iw = 1, p + 1
    Gf[iw:, iw:] += G; Gf[0, iw:] += -2 * c; k += yy
    order = ["Z00", "diagU", "diagV", "sumu", "rowU", "colV"] + (["sym"] if keep_sym else []) \
        + [(a, g) for a in FACT for g in FACT if (a, g) not in set(skip)]
    assert len(order) == len(duals), (len(order), len(duals))
    # every block at its declared shape (a 1x1 block or length-1 vector can come back scalar)
    shapes = {"Z00": (), "sumu": (), "sym": (p, p)}
    duals = [np.reshape(d, shapes.get(t, (p, p) if isinstance(t, tuple) else (p,)))
             for t, d in zip(order, duals)]
    for tag, d in zip(order, duals):
        if tag == "Z00": Gf[0, 0] += d; k -= d
        elif tag == "diagU":
            for i in range(p): Gf[iu + i, iu + i] += d[i]; Gf[0, iu + i] -= d[i]
        elif tag == "diagV":
            for i in range(p): Gf[iu + i, iw + i] += d[i]; Gf[0, iw + i] -= d[i]
        elif tag == "sumu": Gf[0, iu:iw] += d; k -= m * d
        elif tag == "rowU":
            Gf[iu:iw, iu:iw] += d[:, None] * np.ones(p, dtype=object)[None, :]; Gf[0, iu:iw] -= m * d
        elif tag == "colV":
            Gf[iu:iw, iw:] += np.ones(p, dtype=object)[:, None] * d[None, :]; Gf[0, iw:] -= m * d
        elif tag == "sym":
            Gf[iu:iw, iw:] += d; Gf[iu:iw, iw:] -= d.T
        else:
            D = -np.maximum(d, Q(0))          # e >= 0 is stored as -e <= 0
            for t, v in family(*tag, lo, hi).items():
                if v == 0: continue
                Dv = D * v
                if t == "U": Gf[iu:iw, iu:iw] += Dv
                elif t == "V": Gf[iu:iw, iw:] += Dv
                elif t == "Vt": Gf[iu:iw, iw:] += Dv.T
                elif t == "W": Gf[iw:, iw:] += Dv
                elif t == "Ui": Gf[0, iu:iw] += Dv.sum(axis=1)
                elif t == "Uj": Gf[0, iu:iw] += Dv.sum(axis=0)
                elif t == "Wi": Gf[0, iw:] += Dv.sum(axis=1)
                elif t == "Wj": Gf[0, iw:] += Dv.sum(axis=0)
                elif t == "1": k += Dv.sum()
    for j, val, d in fixes:            # u_j == val, stored as u_j - val
        d = Q(float(d)) * s2
        Gf[0, iu + j] += d; k -= Q(val) * d
    # sum_C u - sum_notC u - (|C|-1) <= 0 per known design, multipliers >= 0; one
    # (C, d) or a list of them
    cuts = [] if nogood is None else ([nogood] if isinstance(nogood[0], (set, frozenset)) else list(nogood))
    for Cset, d in cuts:
        d = max(Q(float(d)), Q(0)) * s2
        for j in range(p):
            Gf[0, iu + j] += d if j in Cset else -d
        k -= d * (len(Cset) - 1)
    Ssym = (Gf + Gf.T) / 2
    lam = lambda_min_lower(_from_fractions(Ssym.tolist()))
    if lam is None:
        return None, Ssym, k
    trmax = joint_trace(m, hi)
    return k + min(Q(0), lam) * trmax, Ssym, k


# ---------------------------------------------------------------- the joint lift in floating point
#
# joint_bound sums S and k in rationals and bounds lambda_min(S) by an exact
# Cholesky residual. Here the same bound comes from doubles with every rounding
# bounded a priori, and the few doubles left are combined in rationals:
#
#   |fl(S) - S| <= E entrywise, |fl(k) - k| <= e_k           (joint_assembly),
#   lambda_min(S) >= lambda_min(fl(S)) - ||E||_inf            (Weyl; Gershgorin),
#   lambda_min(fl(S)) >= sigma - ||R||_inf, R = fl(S) - sigma I - L L'
#                                                             (lambda_min_lower_float),
#   bound = fl(k) - e_k + T min(0, lambda_low - ||E||_inf).
#
# Weyl's theorem (Golub and Van Loan 2013, Cor. 8.1.6) moves an eigenvalue of a
# symmetric matrix by at most the 2-norm of a symmetric perturbation, and the
# 2-norm of a symmetric matrix is at most its largest absolute row sum, since
# every eigenvalue lies in a Gershgorin disc (Golub and Van Loan 2013,
# Thm 8.1.3). The two steps are those of mlsynth's lambda_min_lower, with the
# residual's rounding bounded instead of computed exactly.
#
# Golub, G. H. and Van Loan, C. F. (2013). Matrix Computations, 4th ed. Johns
#     Hopkins.

_UNIT = SimpleNamespace(scale=1.0)       # a lift whose multipliers are in the data's units


@functools.lru_cache(maxsize=64)
def _coef_error(lo, hi):
    """kappa = max |K~ - K| / |K| over the oracle's coefficient table, K~ its doubles
    and K the same products of lo and hi in rationals, or infinity above 1/4, where
    |K| <= 2 |K~| (joint_assembly's step) fails; a product that underflows has
    kappa up to 1. No entry is zero in one and not the other."""
    Kf = _coef(lo, hi)
    worst = Q(0)
    for i, (a, g) in enumerate(FAMS):
        exact = family(a, g, Q(lo), Q(hi))
        for t, name in enumerate(TERMS):
            e, f = Q(exact.get(name, 0)), Q(float(Kf[i, t]))
            if e == 0:
                assert f == 0, (a, g, name)
                continue
            worst = max(worst, abs(f - e) / abs(e))
    return float(np.nextafter(float(worst), np.inf)) if worst <= Q(1, 4) else np.inf


def joint_terms(O):
    """N, at least the number of terms in any entry of the oracle's S or in k: the
    16 bound-factor families meet each entry of U, V and W twice (the entry and its
    transpose) and sum over p into each entry of the first row (16 p twice), and
    over p^2 into k, with the cuts, the fixings and the ten other blocks besides."""
    p = O.p
    return 16 * p * p + 32 * p + len(O.cuts) + len(O.fixes) + 40


def joint_assembly(O, yv):
    """fl(S), fl(k) and bounds E, e_k on their rounding, entrywise:

        |fl(S) - S| <= E,  |fl(k) - k| <= e_k,

    S and k the exact assembly from the double data and the double multipliers,
    inequality multipliers clipped at zero as joint_bound clips them. Each entry
    is a sum of at most N = joint_terms(O) terms, each a double, a product of two
    doubles, or a product with a coefficient K~ within kappa of K (_coef_error),
    and the symmetrising (A + A')/2 adds one sum; so its rounding is at most
    (gamma_{N+8} + 2 kappa) times the sum of the terms' moduli, which the
    oracle's S_k_moduli assembles (Bjorck 2015, eq. (1.4.10), derived here for
    the coefficient), plus 2 N ETA for underflow (_dot_err). The data's own
    rounding adds gamma_n |X|'|X| to the W block, gamma_n |X|'|y| to the c
    entries and gamma_n |y|'|y| to k, with n the number of periods. The error
    terms are doubles, and inflate() covers their own rounding.
    """
    if O.scale != 1.0:
        raise ValueError("the float certificate reads the data's units: build the oracle with unscaled=True")
    yv = np.where(O.ineq, np.maximum(yv, 0.0), yv)
    S, k = O.S_k(yv)
    M, mk = O.S_k_moduli(yv)
    N = joint_terms(O)
    rel = _gamma(N + 8) + 2.0 * _coef_error(O.lo, O.hi)
    E = rel * M + 2 * N * ETA
    ek = rel * mk + 2 * N * ETA
    aG, ac, ayy = O.data_moduli
    iw = O.p + 1
    E[iw:, iw:] += _dot_err(O.nobs, np.maximum(aG, aG.T))
    dc = _dot_err(O.nobs, ac)
    E[0, iw:] += dc; E[iw:, 0] += dc
    ek += _dot_err(O.nobs, ayy)
    return S, k, inflate(E), inflate(ek)


def cholesky_residual(S, sigma):
    """A Cholesky factor L of fl(S - sigma I), R~ = fl(S - sigma I - L L') and E with
    |R~ - R| <= E entrywise, R the exact residual from the doubles S, sigma and L;
    None when the factorisation fails. Derived here from the a priori bounds: the
    diagonal shift rounds by u |S_ii - sigma|, the n-term dot products of L L' by
    gamma_n |L||L|' (Bjorck 2015, eq. (1.4.10)), the subtraction by u of its
    result, and together gamma_{n+2} (|S| + |sigma| I + |L||L|') bounds them, with
    (n + 2) ETA for underflow (_dot_err)."""
    n = S.shape[0]
    A = S - sigma * np.eye(n)
    try:
        L = np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        return None
    R = A - L @ L.T
    aL = np.abs(L)
    return L, R, inflate(_dot_err(n + 2, np.abs(S) + abs(sigma) * np.eye(n) + aL @ aL.T))


def lambda_min_lower_float(S, tries=40):
    """A double at most lambda_min(S) for the symmetric double S, or None: the steps
    of mlsynth's lambda_min_lower (a shift sigma just under the computed smallest
    eigenvalue, a Cholesky factor of S - sigma I, then Weyl and Gershgorin:
    lambda_min(S) >= sigma - max_i sum_j |R_ij|), with R bounded by
    cholesky_residual instead of computed exactly. The last subtraction is done in
    rationals and rounded down."""
    S = np.asarray(S, dtype=float)
    if not np.all(np.isfinite(S)):
        return None
    n = S.shape[0]
    lam = float(np.linalg.eigvalsh(S)[0])
    size = float(np.abs(S).sum(axis=1).max()) or 1.0
    step = max(abs(lam) * 1e-12, 64 * n * np.finfo(float).eps * size)
    for _ in range(tries):
        sigma = lam - step
        out = cholesky_residual(S, sigma)
        if out is None:
            step *= 4.0
            continue
        _, R, E = out
        rho = inflate(float((np.abs(R) + E).sum(axis=1).max()))
        if not (np.isfinite(rho) and np.isfinite(sigma)):
            return None
        return round_down(Q(sigma) - Q(rho))
    return None  # pragma: no cover - 40 quadruplings of the shift factor any finite symmetric matrix


def joint_bound_float(O, yv):
    """joint_bound in floating point, at most the dual function at yv (see the
    section comment above): fl(k) - e_k + T min(0, lambda_low - ||E||_inf), with T
    the trace bound in rationals. ``O`` is an oracle in the data's units
    (``Oracle(..., unscaled=True)``) and ``yv`` its multipliers in those units. An
    overflow, a non-finite multiplier or a failed factorisation returns -inf, a
    bound that clears nothing."""
    S, k, E, ek = joint_assembly(O, yv)
    if not (np.all(np.isfinite(S)) and np.all(np.isfinite(E)) and np.isfinite(k) and np.isfinite(ek)):
        return -np.inf
    lam = lambda_min_lower_float(S)
    eS = inflate(float(E.sum(axis=1).max()))
    if lam is None or not np.isfinite(eS):
        return -np.inf
    T = joint_trace(Q(int(O.m)), Q(O.hi))
    try:
        return round_down(Q(k) - Q(ek) + T * min(Q(0), Q(lam) - Q(eS)))
    except OverflowError:
        return -np.inf                  # below the most negative double


def joint_clears(O, yv, y, X, lo, hi, m, UB, stats):
    """Whether the joint lift's certificate at the multipliers yv (data units) is at
    least UB: the float bound decides when it clears UB, and the exact certificate
    at the same multipliers otherwise. Both are lower bounds on the node, so a
    prune by either is valid."""
    stats["float"] = stats.get("float", 0) + 1
    fb = joint_bound_float(O, yv)
    if np.isfinite(fb) and Q(fb) >= UB:
        return True
    stats["exact"] = stats.get("exact", 0) + 1
    duals, cuts, fixes = unpack(O, yv, X.shape[1])
    cb, _, _ = joint_bound(_UNIT, y, X, lo, hi, m, duals=duals, nogood=cuts or None, fixes=fixes)
    return cb is not None and cb >= UB
