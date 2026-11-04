"""The dual of the joint lift as a function of all its multipliers, in doubles (#965).

theta(y) = k(y) + T min(0, lambda_min(S(y))), with S(y) and k(y) the Lagrangian's
matrix and constant (obj(Z) + sum_k y_k expr_k(Z) = <S, Z> + k) and T the trace
bound 1 + m + m hi^2 (certificate.py). It is a valid lower bound for any
sign-correct y (Jansson, Chaykin and Keil 2007, Thm 3.2), and maximising it over
y is the dual of the lift with the redundant constraint tr Z <= T added, whose
value is the lift's (Liao, Ding and Zheng 2026, Prop. 3.3, for the equivalence of
the penalty T max(lambda_max, 0) and the trace constraint). The inequality
blocks of y (the bound-factor products and the no-good cut) must be >= 0.

The supergradient is expr(Z*) with Z* = T v v', v the eigenvector of
lambda_min, when lambda_min < 0, else expr(0): theta(y) = min over
{Z >= 0, tr Z <= T} of the Lagrangian, a minimum of functions affine in y
over a compact set, attained at Z*, so the gradient in y of the Lagrangian at Z*
is a supergradient (derived here: theta(y') <= L(Z*, y') = theta(y) +
expr(Z*)'(y' - y)). Same
constraint order as lifts.Lift('full'), then the no-good cut, then the fixings.

Jansson, C., Chaykin, D. and Keil, C. (2007). SIAM J. Numer. Anal. 46, 180-200.
Liao, F.-Y., Ding, L. and Zheng, Y. (2026). An overview and comparison of
    spectral bundle methods for primal and dual semidefinite programs.
    arXiv:2307.07651v2.
Helmberg, C. and Kiwiel, K. C. (2002). A spectral bundle method with bounds.
    Math. Program. 93, 173-194.
"""
import numpy as np


FACT = {"u": ({"U": 1}, {"V": 1}, {"Ui": 1}),
        "1-u": ({"Uj": 1, "U": -1}, {"Wj": 1, "V": -1}, {"1": 1, "Ui": -1}),
        "a": ({"Vt": 1, "U": "-lo"}, {"W": 1, "V": "-lo"}, {"Wi": 1, "Ui": "-lo"}),
        "c": ({"U": "hi", "Vt": -1}, {"V": "hi", "W": -1}, {"Ui": "hi", "Wi": -1})}
TERMS = ["U", "V", "Vt", "W", "Ui", "Uj", "Wi", "Wj", "1"]
FAMS = [(a, g) for a in FACT for g in FACT]


def _coef(lo, hi):
    """(16, 9): each family's form as coefficients on TERMS."""
    def lin(d):
        return np.array([{"-lo": -lo, "hi": hi}.get(d.get(t, 0), d.get(t, 0)) if isinstance(d.get(t, 0), str)
                         else float(d.get(t, 0)) for t in TERMS])
    out = np.zeros((16, 9))
    for i, (a, g) in enumerate(FAMS):
        pu, pw, ln = (lin(x) for x in FACT[a])
        out[i] = {"u": pu, "1-u": ln - pu, "a": pw - lo * pu, "c": hi * pu - pw}[g]
    return out


class Oracle:
    def __init__(self, y, X, lo, hi, m, C=None, fixes=(), cuts=None):
        self.scale = float(np.linalg.norm(y))
        y, X = y / self.scale, X / self.scale
        p = X.shape[1]; self.p = p; self.n = 2 * p + 1
        self.G, self.c, self.yy = X.T @ X, X.T @ y, float(y @ y)
        self.lo, self.hi, self.m = float(lo), float(hi), float(m)
        self.K = _coef(self.lo, self.hi)
        self.T = 1.0 + m + m * hi * hi
        # One no-good cut per known design: sum_C u - sum_notC u <= |C| - 1 excludes
        # that design and no other (Balas and Jeroslow 1972). ``C`` is one design.
        self.cuts = [sorted(c) for c in (cuts if cuts is not None else ([] if C is None else [C]))]
        self.C = self.cuts[0] if self.cuts else None
        self.signs = [np.array([1.0 if j in set(c) else -1.0 for j in range(p)]) for c in self.cuts]
        self.fixes = list(fixes)
        sizes = [("z00", 1), ("dU", p), ("dV", p), ("su", 1), ("rU", p), ("cV", p), ("sym", p * p),
                 ("fam", 16 * p * p)] + ([("ng", len(self.cuts))] if self.cuts else []) + [("fix", len(self.fixes))]
        self.sl = {}; o = 0
        for k, s in sizes:
            self.sl[k] = slice(o, o + s); o += s
        self.dim = o
        self.ineq = np.zeros(o, dtype=bool); self.ineq[self.sl["fam"]] = True
        if self.cuts: self.ineq[self.sl["ng"]] = True

    # ---- forward map: expr(Z) for every multiplier, in layout order
    def expr(self, Z):
        p, m = self.p, self.m
        u, w = Z[0, 1:p + 1], Z[0, p + 1:]
        U, V, W = Z[1:p + 1, 1:p + 1], Z[1:p + 1, p + 1:], Z[p + 1:, p + 1:]
        one = np.ones((p, p))
        terms = np.stack([U, V, V.T, W, u[:, None] * one, u[None, :] * one,
                          w[:, None] * one, w[None, :] * one, one])
        fam = -np.einsum("ft,tij->fij", self.K, terms)
        parts = [[Z[0, 0] - 1.0], np.diag(U) - u, np.diag(V) - w, [u.sum() - m],
                 U.sum(1) - m * u, V.sum(0) - m * w, (V - V.T).ravel(), fam.ravel()]
        if self.cuts: parts.append([sg @ u - (len(c) - 1) for sg, c in zip(self.signs, self.cuts)])
        parts.append([u[j] - v for j, v in self.fixes])
        return np.concatenate([np.ravel(x) for x in parts])

    # ---- adjoint: S(y) and k(y)
    def S_k(self, yv):
        p, m, n = self.p, self.m, self.n
        iu, iw = 1, p + 1
        Gf = np.zeros((n, n)); k = self.yy
        Gf[iw:, iw:] += self.G; Gf[0, iw:] += -2 * self.c
        g = lambda key: yv[self.sl[key]]
        d = g("z00")[0]; Gf[0, 0] += d; k -= d
        d = g("dU"); Gf[np.arange(iu, iw), np.arange(iu, iw)] += d; Gf[0, iu:iw] -= d
        d = g("dV"); Gf[np.arange(iu, iw), np.arange(iw, n)] += d; Gf[0, iw:] -= d
        d = g("su")[0]; Gf[0, iu:iw] += d; k -= m * d
        d = g("rU"); Gf[iu:iw, iu:iw] += d[:, None]; Gf[0, iu:iw] -= m * d
        d = g("cV"); Gf[iu:iw, iw:] += d[None, :]; Gf[0, iw:] -= m * d
        D = g("sym").reshape(p, p); Gf[iu:iw, iw:] += D - D.T
        Y = -g("fam").reshape(16, p, p)                   # expr = -form
        Tm = np.einsum("ft,fij->tij", self.K, Y)          # per-term multiplier matrices
        Gf[iu:iw, iu:iw] += Tm[0]; Gf[iu:iw, iw:] += Tm[1] + Tm[2].T; Gf[iw:, iw:] += Tm[3]
        Gf[0, iu:iw] += Tm[4].sum(1) + Tm[5].sum(0); Gf[0, iw:] += Tm[6].sum(1) + Tm[7].sum(0)
        k += Tm[8].sum()
        for d, sg, c in zip(g("ng") if self.cuts else (), self.signs, self.cuts):
            Gf[0, iu:iw] += d * sg; k -= d * (len(c) - 1)
        for (j, v), d in zip(self.fixes, g("fix")):
            Gf[0, iu + j] += d; k -= v * d
        return (Gf + Gf.T) / 2, k

    def theta(self, yv):
        """Bound (scaled units), supergradient, the eigen-decomposition of S."""
        Sm, k = self.S_k(yv)
        lam, Q = np.linalg.eigh(Sm)
        if lam[0] < 0:
            v = Q[:, 0]; Z = self.T * np.outer(v, v)
            return k + self.T * lam[0], self.expr(Z), lam, Q
        return k, self.expr(np.zeros((self.n, self.n))), lam, Q

    def duals_from_cvxpy(self, constraints, nogood=None, fix_cons=()):
        """Pack a Clarabel solve's multipliers (cvxpy order) into this layout."""
        parts = [np.ravel(c.dual_value) for c in constraints]
        if self.cuts: parts.append(np.ravel(nogood.dual_value))
        parts += [np.ravel(c.dual_value) for c in fix_cons]
        return np.concatenate(parts)
