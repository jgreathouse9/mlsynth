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
import math
from fractions import Fraction as Q

import numpy as np

from mlsynth.utils.marex_helpers.branch import _from_fractions, lambda_min_lower


def qarr(A):
    return np.vectorize(Q, otypes=[object])(np.asarray(A, dtype=float))


class ExactData:
    """y and X as given (doubles), their exact Gram, cross products and norms."""
    def __init__(self, y, X):
        self.yq, self.Xq = qarr(y), qarr(X)
        self.G = self.Xq.T @ self.Xq
        self.c = self.Xq.T @ self.yq
        self.yy = self.yq @ self.yq
        self.p = X.shape[1]

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


def shor_bound(lift, D, lo, hi, m, scale):
    """Exact bound on the Shor lift (lifts.Lift level 'shor') from its last solve."""
    p = D.p; n = p + 1
    lo, hi, m = Q(float(lo)), Q(float(hi)), Q(int(m))
    s2 = Q(float(scale)) ** 2
    duals = [qarr(c.dual_value) * s2 for c in lift.prob.constraints[1:]]
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
    return k + min(Q(0), lam) * (1 + m * hi * hi), Ssym, k


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
    if nogood is not None:             # sum_C u - sum_notC u - (|C|-1) <= 0, multiplier >= 0
        Cset, d = nogood
        d = max(Q(float(d)), Q(0)) * s2
        for j in range(p):
            Gf[0, iu + j] += d if j in Cset else -d
        k -= d * (len(Cset) - 1)
    Ssym = (Gf + Gf.T) / 2
    lam = lambda_min_lower(_from_fractions(Ssym.tolist()))
    if lam is None:
        return None, Ssym, k
    trmax = 1 + m + m * hi * hi
    return k + min(Q(0), lam) * trmax, Ssym, k
