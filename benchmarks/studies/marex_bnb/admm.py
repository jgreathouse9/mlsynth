"""A first-order solver for a lift, whose every iterate gives a certified bound.

A lift (``certified_dnn.Lift``) asks for

    min <C, Y>  over  Y in K  and  Y in Pc,

with ``K = {P T P' : T psd}`` the face of the PSD cone the designs lie on and
``Pc`` the polyhedral conditions: ``Y_00 = 1``, ``Y_ij = 0`` on the entries
fixed at zero, ``Y_ij >= 0`` elsewhere. Without cardinality cuts both sets are
easy to project onto: onto ``K`` by one eigendecomposition of ``Q' X Q``, ``Q``
an orthonormal basis of the range of ``P``; onto ``Pc`` entry by entry. So the
alternating direction method of multipliers (ADMM) on the split ``Y = Z`` needs
no linear system:

    Y <- proj_K(Z - U - C / rho),   Z <- proj_Pc(Y + U),   U <- U + Y - Z.

The certificate. After the ``Z`` step, ``U = X - proj_Pc(X)`` with
``X = Y + U_old``: it is ``alpha`` on entry ``(0, 0)``, free on the entries
fixed at zero and ``min(X_ij, 0) <= 0`` elsewhere. So ``-rho U`` is a valid set
of multipliers: ``y0 = -rho U_00`` on ``Y_00 = 1`` and ``H = -rho U`` elsewhere,
nonnegative where it must be. ``certified_dnn.certify`` turns any such
multipliers into an exact lower bound, ``y0 + tau min(0, lambda_min(S))`` with
``S = P'(C + rho U)P``. The ``Y`` step makes ``C + rho (U_old + Y - Z_old)``
lie in the dual cone of ``K`` exactly, so ``lambda_min(S)`` is held down only
by ``rho (Z - Z_old)``, the dual residual. The bound is valid at every
iterate; convergence only tightens it.

Before certifying, ``y0`` is moved down by the amount that maximises
``y0 - d + tau min(0, lambda_min(S + d e_0 e_0'))``: since ``P' E_00 P =
e_0 e_0'``, lowering ``y0`` adds a rank-one PSD term, and the best ``d`` is a
one-dimensional concave search.

Two devices from Deng et al. (2024), "An enhanced ADMM-based interior point
method for linear and conic optimization" (arXiv 2209.01793): restarts that
replace the iterate by the average over the last cycle (their section 3.2), and,
with ``barrier=mu``, the proximal map of ``-mu log det`` in place of the
projection, ``d -> (d + sqrt(d^2 + 4 mu / rho)) / 2`` on the eigenvalues
(their section 2.4.2), with ``mu`` driven to zero. ``rho`` is balanced against
the primal and dual residuals, as in SCS and OSQP.

A child node's rows are a subset of its parent's, so ``state`` restricted to
them is a warm start.
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from benchmarks.studies.marex_bnb.certified_dnn import certify, objective
from benchmarks.studies.marex_bnb.certify import down


class FaceADMM:
    """ADMM for one lift without cuts; ``bound`` certifies the current iterate."""

    def __init__(self, panel, lift):
        if lift.cuts:
            raise ValueError("FaceADMM projects entrywise; build the lift without cuts")
        self.panel, self.lift = panel, lift
        n = panel.Bc.shape[1]
        diag = max(float(panel.G.frac(j, j)) for j in range(n))
        self.shift = math.frexp(diag)[1]
        self.C_exact = objective(panel.G, lift, self.shift)
        self.C = self.C_exact.float()
        self.P = lift.P.astype(float)
        self.Q = np.linalg.qr(self.P)[0]
        self.N = self.C.shape[0]
        self.free = ~lift.zero
        self.free[0, 0] = False

    # projections -------------------------------------------------------
    def proj_K(self, X, mu=0.0, rho=1.0):
        A = self.Q.T @ X @ self.Q
        d, V = np.linalg.eigh((A + A.T) / 2)
        if mu > 0:
            d = (d + np.sqrt(d * d + 4 * mu / rho)) / 2
        else:
            d = np.maximum(d, 0.0)
        QV = self.Q @ V
        return (QV * d) @ QV.T

    def proj_P(self, X):
        Z = np.where(self.free, np.maximum(X, 0.0), 0.0)
        Z[0, 0] = 1.0
        return Z

    # certificate --------------------------------------------------------
    def _S(self, U, rho):
        M = self.C + rho * U
        return self.P.T @ M @ self.P

    def float_bound(self, U, rho):
        """``(bound, d)``: the floating-point bound after the best shift ``d``.

        ``lambda_min(S + d e_0 e_0')`` is the smallest root of the secular
        equation ``1 + d sum_i c_i^2 / (lambda_i - x) = 0``, ``c = V' e_0``,
        which lies between the two smallest eigenvalues of ``S``.
        """
        lam, V = np.linalg.eigh(self._S(U, rho))
        c2 = V[0] ** 2
        y0 = -rho * U[0, 0]
        tau = self.lift.tau

        def smallest(d):
            if d <= 0 or c2[0] < 1e-300 or lam[1] - lam[0] <= 0:
                return lam[0]
            lo, hi = lam[0], lam[1]
            for _ in range(80):
                x = (lo + hi) / 2
                if 1 + d * np.sum(c2 / (lam - x)) < 0:
                    lo = x
                else:
                    hi = x
            return lo

        def value(d):
            return y0 - d + tau * min(0.0, smallest(d))

        if lam[0] >= 0:
            return value(0.0), 0.0
        lo, hi = 0.0, 4 * tau * abs(lam[0]) + 1e-300
        for _ in range(80):                    # golden section, concave in d
            a = hi - 0.618 * (hi - lo)
            b = lo + 0.618 * (hi - lo)
            if value(a) < value(b):
                lo = a
            else:
                hi = b
        d = (lo + hi) / 2
        return max(value(d), value(0.0)), (d if value(d) > value(0.0) else 0.0)

    def bound(self, U, rho):
        """The exact certified bound at multipliers ``-rho U``, rounded down."""
        _, d = self.float_bound(U, rho)
        y0 = -rho * U[0, 0] - d
        H = -rho * U
        H[0, 0] = 0.0
        lb = certify(self.C_exact, self.lift, float(y0), H, None)
        return down(max(lb, Fraction(0)) * (1 << self.shift))

    # the iteration --------------------------------------------------------
    def solve(self, state=None, iters=3000, rho=None, check=50, target=None,
              stall=1e-6, restart=200, barrier=0.0, mu_decay=0.8, log=None):
        """Run ADMM; return ``(state, float bound, iterations)``.

        Stops when the float bound reaches ``target`` (scaled back to the
        data's units), when it gains less than ``stall`` relative in ``check``
        iterations twice in a row, or after ``iters`` iterations.
        """
        N = self.N
        if state is None:
            Z = np.full((N, N), 1.0 / N)
            Z = self.proj_P(Z)
            U = np.zeros((N, N))
            rho = rho or 1.0
        else:
            Z, U, rho = state["Z"], state["U"], state.get("rho", rho or 1.0)
        scale = 2.0 ** self.shift
        tgt = None if target is None else target / scale
        best, quiet, mu = -np.inf, 0, barrier
        Zs, Us, cyc = np.zeros_like(Z), np.zeros_like(U), 0
        k = 0
        for k in range(1, iters + 1):
            Y = self.proj_K(Z - U - self.C / rho, mu, rho)
            Z_old = Z
            Z = self.proj_P(Y + U)
            U = U + Y - Z
            Zs += Z
            Us += U
            cyc += 1
            if restart and cyc == restart:
                Z, U = self.proj_P(Zs / cyc), Us / cyc
                Zs[:], Us[:], cyc = 0.0, 0.0, 0
            if k % check == 0:
                r_p = np.linalg.norm(Y - Z)
                r_d = rho * np.linalg.norm(Z - Z_old)
                if r_p > 10 * r_d:
                    rho *= 2.0
                    U /= 2.0
                    Us /= 2.0
                elif r_d > 10 * r_p:
                    rho /= 2.0
                    U *= 2.0
                    Us *= 2.0
                if mu > 0:
                    mu *= mu_decay
                lb, _ = self.float_bound(U, rho)
                if log is not None:
                    log.append((k, lb * scale, r_p, r_d, rho))
                if tgt is not None and lb >= tgt:
                    best = lb
                    break
                quiet = quiet + 1 if lb - best <= stall * max(abs(best), 1e-12) else 0
                best = max(best, lb)
                if quiet >= 2:
                    break
        return {"Z": Z, "U": U, "rho": rho}, best * scale, k


def restrict(state, parent_rows, child_rows):
    """The parent's state on the child's rows, a warm start."""
    pos = {key: i for i, key in enumerate(parent_rows)}
    idx = [pos[key] for key in child_rows]
    ix = np.ix_(idx, idx)
    return {"Z": state["Z"][ix].copy(), "U": state["U"][ix].copy(),
            "rho": state["rho"]}
