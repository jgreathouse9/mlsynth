"""The spectral bundle method on a lift's dual, certified at its centre.

The certificate of ``certified_dnn`` is a concave function of the multipliers
``y = (y0, H)``:

    phi(y) = y0 + tau min(0, lambda_min(S(y))),   S(y) = P'(C - y0 E_00 - H) P,

``H`` symmetric, zero on the diagonal, nonnegative off the entries a lift fixes
at zero and free on them. Every sign-feasible ``y`` gives a valid lower bound,
so maximising ``phi`` is the node bound and any iterate is a usable one. This
is the eigenvalue form of a semidefinite program with a trace bound, which the
spectral bundle method of Helmberg and Rendl (2000) maximises; sign constraints
follow Helmberg and Kiwiel (2002).

The model. For any ``W_T`` positive semidefinite with trace at most one,
``<S(y), W_T> >= min(0, lambda_min(S(y)))``; so

    phihat(y) = y0 + tau min(0, min over W_T in Wset of <S(y), W_T>),
    Wset = {alpha Wbar + Q V Q' : alpha >= 0, V psd, alpha + tr V <= 1},

lies above ``phi``. ``Q`` is an orthonormal basis of a few eigenvectors of
``S`` (``T`` space) and ``Wbar`` an aggregate. Modelling the whole span of
``Q`` with a PSD block ``V``, and not its single eigenvectors, is what lets
the model follow a degenerate smallest eigenvalue; at a root the face matrix
has dozens of eigenvalues at zero, and a model of rank-one atoms stalls there.

The step maximises ``phihat(y) - u/2 |y - yhat|^2`` over sign-feasible ``y``,
the norm counting ``y0`` and each off-diagonal pair of ``H`` once. Its dual is
a minimum over ``W_T`` in ``Wset`` of a convex function. With ``W = P W_T P'``
and ``s = b - tau a(W)`` (``a(W)`` has ``W_00`` on ``y0`` and ``2 W_kl`` on the
pair ``kl``), the inner maximum over ``y`` is separable: a free coordinate
contributes ``s yhat + s^2 / 2u`` and a signed one its Huber-like clipping,
``s yhat + s^2 / 2u`` for ``s >= -u yhat`` and ``-u yhat^2 / 2`` below. So
the dual is a smooth convex function of ``(alpha, V)``, minimised by an
accelerated projected gradient; the projection is an eigendecomposition of
``V`` and a projection of ``(alpha, eigenvalues)`` onto the capped simplex.
The step is then ``y = yhat + s / u``, clipped at zero on the signed pairs.

A candidate replaces the centre when it gains a tenth of the increase the
model predicted (a serious step). Otherwise only the bundle changes (a null
step). After either, the master's ``V`` is split: its leading eigenvectors stay
in ``Q``, the rest go into ``Wbar``, and the smallest eigenvectors of ``S`` at
the candidate are added. The method stops when the predicted increase is below
``tol`` relative, when ``phi`` at the centre reaches ``target``, or after
``iters`` steps. ``bound`` certifies the centre exactly.
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from benchmarks.studies.marex_bnb.certified_dnn import best_shift, certify, objective
from benchmarks.studies.marex_bnb.certify import down


def _capped_simplex(x):
    """Projection of ``x`` onto ``{z >= 0, sum z <= 1}``."""
    z = np.maximum(x, 0.0)
    if z.sum() <= 1.0:
        return z
    s = np.sort(x)[::-1]
    cs = np.cumsum(s) - 1.0
    k = np.nonzero(s - cs / np.arange(1, len(s) + 1) > 0)[0][-1]
    return np.maximum(x - cs[k] / (k + 1), 0.0)


class SpectralBundle:
    """The spectral bundle method on one lift (built without cuts)."""

    def __init__(self, panel, lift):
        if lift.cuts:
            raise ValueError("build the lift without cuts")
        self.lift = lift
        n = panel.Bc.shape[1]
        diag = max(float(panel.G.frac(j, j)) for j in range(n))
        self.shift = math.frexp(diag)[1]
        self.C_exact = objective(panel.G, lift, self.shift)
        self.C = self.C_exact.float()
        self.P = lift.P.astype(float)
        self.N, self.K = self.P.shape
        self.tau = float(lift.tau)
        signed = ~lift.zero
        np.fill_diagonal(signed, False)
        self.signed = signed
        free = lift.zero.copy()
        np.fill_diagonal(free, False)
        self.free = free                     # off-diagonal pairs with a free sign

    def S(self, y0, H):
        S = self.P.T @ (self.C - H) @ self.P
        S[0, 0] -= y0
        return (S + S.T) / 2

    def phi(self, y0, H):
        lam, V = np.linalg.eigh(self.S(y0, H))
        return y0 + self.tau * min(0.0, lam[0]), lam, V

    # the master problem ----------------------------------------------------
    def _master(self, y0, H, u, Q, Wbar, inner=400):
        """Minimise the step's dual over ``(alpha, V)``; return the step."""
        tau, P = self.tau, self.P
        Z = P @ Q                                     # N x r
        Wb = P @ Wbar @ P.T
        CQ = Q.T @ (P.T @ self.C @ P) @ Q
        cWb = float(np.vdot(self.C, Wb))
        r = Q.shape[1]
        sg, fr = self.signed, self.free

        def W_of(alpha, V):
            return alpha * Wb + Z @ V @ Z.T

        def dual_and_grad(alpha, V):
            W = W_of(alpha, V)
            s0 = 1.0 - tau * W[0, 0]
            sp = -2.0 * tau * W                       # pair coordinates
            val = s0 * y0 + s0 * s0 / (2 * u)
            hp0 = y0 + s0 / u
            lin = H + sp / u                          # derivative where not clipped
            up = sp >= -u * H
            hpair = np.where(sg, np.where(up, lin, 0.0), np.where(fr, lin, 0.0))
            vpair = np.where(up, sp * H + sp * sp / (2 * u), -u * H * H / 2)
            val += 0.5 * (np.sum(np.where(sg, vpair, 0.0))
                          + np.sum(np.where(fr, sp * H + sp * sp / (2 * u), 0.0)))
            val += tau * (alpha * cWb + float(np.vdot(CQ, V)))
            GW = -tau * hpair                         # d/dW_kl, each ordered pair
            GW[0, 0] = -tau * hp0
            GW = GW + tau * self.C
            ga = float(np.vdot(GW, Wb))
            gV = Z.T @ GW @ Z
            return val, ga, (gV + gV.T) / 2

        def project(alpha, V):
            lam, U = np.linalg.eigh((V + V.T) / 2)
            z = _capped_simplex(np.concatenate([[alpha], lam]))
            return z[0], (U * z[1:]) @ U.T

        alpha, V = 0.0, np.eye(r) / max(r, 1)
        L = self._L if getattr(self, "_L", None) else 1.0
        ax, Vx, t = alpha, V.copy(), 1.0
        val_a = dual_and_grad(alpha, V)[0]
        for _ in range(inner):
            fx, ga, gV = dual_and_grad(ax, Vx)
            while True:                              # backtracking on L
                a_new, V_new = project(ax - ga / L, Vx - gV / L)
                da, dV = a_new - ax, V_new - Vx
                f_new = dual_and_grad(a_new, V_new)[0]
                quad = fx + ga * da + float(np.vdot(gV, dV)) \
                    + L / 2 * (da * da + float(np.vdot(dV, dV)))
                if f_new <= quad + 1e-15 * (1 + abs(fx)):
                    break
                L *= 2.0
            t_new = (1 + math.sqrt(1 + 4 * t * t)) / 2
            mom = (t - 1) / t_new
            if f_new > val_a:                        # restart the momentum
                ax, Vx, t = alpha, V.copy(), 1.0
                L /= 2.0
                continue
            ax = a_new + mom * (a_new - alpha)
            Vx = V_new + mom * (V_new - V)
            done = val_a - f_new <= 1e-12 * (1 + abs(f_new))
            alpha, V, t, val_a = a_new, V_new, t_new, f_new
            L = max(L / 1.5, 1e-12)
            if done:
                break
        self._L = L
        W = W_of(alpha, V)
        s0 = 1.0 - tau * W[0, 0]
        sp = -2.0 * tau * W
        y0c = y0 + s0 / u
        Hc = H + sp / u
        Hc = np.where(sg, np.maximum(Hc, 0.0), np.where(fr, Hc, 0.0))
        return y0c, Hc, alpha, V

    def _model(self, y0, H, Q, Wbar):
        S = self.S(y0, H)
        m = float(np.linalg.eigvalsh(Q.T @ S @ Q)[0])
        if np.trace(Wbar) > 0:
            m = min(m, float(np.vdot(S, Wbar)))
        return y0 + self.tau * min(0.0, m)

    # the method --------------------------------------------------------------
    def solve(self, start=None, iters=300, target=None, tol=1e-6, k_new=3,
              r_max=15, r_keep=8, u=None, inner=400, log=None):
        """Maximise ``phi``; return ``(state, float phi at the centre, steps)``."""
        N, K = self.N, self.K
        y0, H = (0.0, np.zeros((N, N))) if start is None else (start["y0"], start["H"].copy())
        f, lam, Vec = self.phi(y0, H)
        thresh = lam[0] + 1e-9 * max(1.0, abs(lam[0]))
        n0 = max(k_new, int(np.sum(lam <= thresh)))
        Q = Vec[:, :min(r_max, n0)]
        Wbar = np.zeros((K, K))
        u = u or 1.0
        scale = 2.0 ** self.shift
        tgt = None if target is None else target / scale
        k = 0
        for k in range(1, iters + 1):
            if tgt is not None and f >= tgt:
                break
            y0c, Hc, alpha, V = self._master(y0, H, u, Q, Wbar, inner)
            model = self._model(y0c, Hc, Q, Wbar)
            fc, lamc, Vc = self.phi(y0c, Hc)
            gap = model - f
            if log is not None:
                log.append((k, f * scale, fc * scale, gap * scale, u, Q.shape[1]))
            if gap <= tol * (1 + abs(f)):
                break
            gain = fc - f
            if gain >= 0.1 * gap:                     # serious step
                y0, H, f = y0c, Hc, fc
                if gain >= 0.5 * gap:
                    u = max(u / 2, 1e-10)
            else:                                     # null step
                u = min(u * 1.5, 1e10)
            # bundle update: keep V's leading directions, aggregate the rest
            lv, Uv = np.linalg.eigh((V + V.T) / 2)
            order = np.argsort(lv)[::-1]
            keep = [i for i in order[:r_keep] if lv[i] > 1e-10]
            drop = [i for i in order if i not in keep and lv[i] > 0]
            weight = alpha + sum(lv[i] for i in drop)
            if weight > 1e-14:
                Wd = np.zeros((K, K))
                for i in drop:
                    x = Q @ Uv[:, i]
                    Wd += lv[i] * np.outer(x, x)
                Wbar = (alpha * Wbar + Wd) / weight
            basis = [Q @ Uv[:, i] for i in keep] + [Vc[:, i] for i in range(k_new)]
            Qn, _ = np.linalg.qr(np.column_stack(basis))
            Q = Qn[:, :r_max]
        return {"y0": y0, "H": H}, f * scale, k

    def bound(self, state):
        """The exact certified bound at ``state``, rounded down."""
        y0, H = state["y0"], state["H"]
        _, d = best_shift(self.S(y0, H), y0, self.tau)
        lb = certify(self.C_exact, self.lift, float(y0 - d), H, None)
        return down(max(lb, Fraction(0)) * (1 << self.shift))


def restrict(state, parent_rows, child_rows):
    """The parent's multipliers on the child's rows, a warm start."""
    pos = {key: i for i, key in enumerate(parent_rows)}
    idx = [pos[key] for key in child_rows]
    return {"y0": state["y0"], "H": state["H"][np.ix_(idx, idx)].copy()}
