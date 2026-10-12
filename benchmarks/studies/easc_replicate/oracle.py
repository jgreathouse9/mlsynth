"""Readable port of Spruk (2026): the oracle for an EASC build.

Rok Spruk, "Synthetic Control under Out-of-Span Common Shocks: Diagnosis,
Exposure Balance and Correction", arXiv:2610.05535v1 (4 October 2026). The
reference implementation is ``code/ebsc.py`` of the author's replication
package; each function below names the section of the paper it implements and
the reference function it ports.

Notation follows the paper. ``y1`` is the treated unit's outcome path (length
``T``), ``Y0`` the ``T x J`` donor outcome matrix, ``pre`` a boolean mask of the
pre-treatment periods, ``s1`` and ``s0`` the treated and donor exposure indices,
``x`` the observed common shock, and ``base`` the index of the base period
``b`` (the last pre-treatment period in the application).

Two departures from the reference, both chosen so the port is the one a build
would ship:

* synthetic-control weights come from mlsynth's active-set simplex solver,
  where the reference uses cvxpy/CLARABEL (application) or a penalised NNLS
  (Monte Carlo). All three solve the same convex programme;
* the breakdown value is found by Brent's method, where the reference bisects
  sixty times on ``[0, 5]``.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cvxpy as cp
import numpy as np
from scipy.optimize import brentq
from scipy.stats import ncx2, norm

from mlsynth.utils.solvers.active_set import solve_simplex_qp

# --------------------------------------------------------------------- factors


def donor_factors(Y0: np.ndarray, k: int = 2, s: Optional[np.ndarray] = None) -> np.ndarray:
    """Latent factors from the donors' outcomes (Section 4.1; Online Appendix B).

    The donor matrix is demeaned by unit over all periods (donors are untreated
    throughout) and then by period. When the exposure index ``s`` is given, each
    period's cross-sectional projection on the standardised index is removed
    first, so the factors are orthogonal to the exposure channel. The factors
    are the first ``k`` left singular vectors scaled by their singular values
    and by ``1/sqrt(T)``. Ports ``ebsc.donor_factors``.
    """
    Z = Y0 - Y0.mean(axis=0, keepdims=True)
    Z = Z - Z.mean(axis=1, keepdims=True)
    if s is not None:
        sc = (s - s.mean()) / s.std()
        Z = Z - np.outer(Z @ sc / (sc @ sc), sc)
    U, sv, _ = np.linalg.svd(Z, full_matrices=False)
    return U[:, :k] * sv[:k] / np.sqrt(Z.shape[0])


def with_constant(*blocks: np.ndarray) -> np.ndarray:
    """``g_t = (1, blocks...)`` stacked by period."""
    T = blocks[0].shape[0]
    return np.column_stack([np.ones(T), *blocks])


# -------------------------------------------------------------------- leverage


def leverage(G: np.ndarray, pre: np.ndarray) -> np.ndarray:
    """``kappa_t = (g_t' S_0^{-1} g_t)^{1/2}`` for every period (eq. 3, Prop. 1).

    ``S_0`` is the pre-treatment second-moment matrix of the factor vector.
    The pseudo-inverse matches the reference, which allows a singular ``S_0``.
    Ports ``ebsc.shock_leverage``.
    """
    S0 = G[pre].T @ G[pre] / pre.sum()
    Si = np.linalg.pinv(S0)
    return np.sqrt(np.einsum("ti,ij,tj->t", G, Si, G))


def phase_leverage(G: np.ndarray, pre: np.ndarray, mask: np.ndarray) -> float:
    """Leverage of a phase average, ``(gbar' S_0^{-1} gbar)^{1/2}`` (Prop. 1).

    The bias of a phase average is ``Delta' gbar``, so the leverage of the
    averaged factor vector bounds it. Ports ``ebsc.phase_leverage``.
    """
    S0 = G[pre].T @ G[pre] / pre.sum()
    gb = G[mask].mean(axis=0)
    return float(np.sqrt(gb @ np.linalg.pinv(S0) @ gb))


def partial_leverage(x: np.ndarray, F: np.ndarray, pre: np.ndarray) -> Tuple[np.ndarray, float]:
    """Partial leverage ``ell_t`` of the observed shock and its pre-period R^2 (Cor. 1).

    ``x`` is regressed on ``(1, F)`` over the pre-period; ``ell_t`` is the
    residual for every period divided by the residual's pre-period root mean
    square. A pre-period R^2 near one together with a large post-period
    ``|ell_t|`` is the signature of a silent factor. Ports
    ``ebsc.partial_leverage``.
    """
    X = with_constant(F)
    b = np.linalg.lstsq(X[pre], x[pre], rcond=None)[0]
    res = x - X @ b
    sr = np.sqrt(np.mean(res[pre] ** 2))
    return res / sr, float(1.0 - res[pre].var() / x[pre].var())


# ------------------------------------------------------------ weights, balance


def sc_weights(a: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Synthetic-control weights: ``min ||a - A w||^2`` over the simplex (Section 2.1)."""
    return np.asarray(solve_simplex_qp(A, a), dtype=float)


def exposure_imbalance(s1: float, s0: np.ndarray, w: np.ndarray) -> float:
    """``EI(w) = s_1 - sum_j w_j s_j`` (eq. 7). Ports ``ebsc.exposure_imbalance``."""
    return float(s1 - s0 @ w)


def min_imbalance(s1: float, s0: np.ndarray) -> float:
    """``d_min = dist(s_1, conv S)`` for a scalar index (Prop. 3(a))."""
    return float(max(0.0, s1 - s0.max(), s0.min() - s1))


class BalancedWeights:
    """``min ||a - A w||^2`` over the simplex with ``|EI(w)| <= d`` (eq. 11).

    A convex quadratic programme with two linear inequalities, compiled once
    with the tolerance ``d`` as a parameter and re-solved with CLARABEL for each
    ``d``, as ``ebsc.Frontier`` does. ``solve`` returns ``None`` when the solver
    reports no solution. Ports ``ebsc._sc_weights`` with ``tol`` set and
    ``ebsc.Frontier``.
    """

    def __init__(self, a: np.ndarray, A: np.ndarray, s1: float, s0: np.ndarray):
        self.w = cp.Variable(A.shape[1])
        self.d = cp.Parameter(nonneg=True)
        cons = [self.w >= 0, cp.sum(self.w) == 1, cp.abs(s1 - s0 @ self.w) <= self.d]
        self.problem = cp.Problem(cp.Minimize(cp.sum_squares(a - A @ self.w)), cons)

    def solve(self, d: float) -> Optional[np.ndarray]:
        self.d.value = float(d)
        try:
            self.problem.solve(solver=cp.CLARABEL)
        except cp.error.SolverError:  # pragma: no cover - mirrors the reference's guard
            return None
        if self.w.value is None:  # pragma: no cover - the grid starts at d_min, which is feasible
            return None
        v = np.clip(self.w.value, 0, None)
        return v / v.sum()


def frontier(a: np.ndarray, A: np.ndarray, s1: float, s0: np.ndarray, n: int,
             w_sc: Optional[np.ndarray] = None) -> List[Dict]:
    """The exposure-fit frontier ``R(d)`` on ``n`` tolerances (Prop. 3(b), eq. 11).

    The tolerances run evenly from ``d_min`` to the synthetic-control imbalance.
    The first point is EBSC-min. The application (``ebsc.ebsc_path``, 41 points)
    solves the last point with its constraint in place; the Monte Carlo
    (``ebsc.frontier_path``, 21 points) takes the synthetic-control weights
    there. Passing ``w_sc`` selects the second behaviour. Points the solver
    cannot return are skipped, as in the reference.
    """
    dmin = min_imbalance(s1, s0)
    w0 = w_sc if w_sc is not None else sc_weights(a, A)
    d_sc = abs(exposure_imbalance(s1, s0, w0))
    qp = BalancedWeights(a, A, s1, s0)
    out = []
    for g in np.linspace(0.0, 1.0, n):
        tol = dmin + g * max(d_sc - dmin, 0.0) + 1e-9
        w = w_sc if (w_sc is not None and g == 1.0) else qp.solve(tol)
        if w is None:  # pragma: no cover - see BalancedWeights.solve
            continue
        r = a - A @ w
        out.append(dict(frac=float(g), tol=float(tol), w=w, ei=exposure_imbalance(s1, s0, w),
                        rmspe=float(np.sqrt(np.mean(r ** 2)))))
    return out


# ------------------------------------------------------------ the correction


def exposure_effects(Y0: np.ndarray, s0: np.ndarray, pre: np.ndarray, base: int,
                     controls: bool = True, n_sub: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``phi_t`` and its delete-one-donor jackknife (eq. 8; Online Appendix B).

    For every period, donors' outcome changes ``Y_jt - Y_jb`` are regressed
    across donors on ``(1, s_j, c_j)``, where ``c_j`` is the donor's own change
    within each of ``n_sub`` equal sub-periods of the pre-period that end
    before ``b`` (so the controls share no noise with the dependent variable).
    Returns ``phi`` (T,), the jackknife standard error (T,), and the ``J x T``
    jackknife replicates, which phase averages need. Ports
    ``ebsc.exposure_effects`` and ``ebsc.exposure_effects_phase``.
    """
    J = Y0.shape[1]
    D = Y0 - Y0[base][None, :]
    pre_idx = np.where(pre)[0]
    pre_idx = pre_idx[pre_idx < base]
    if controls:
        C = np.column_stack([Y0[th[-1]] - Y0[th[0]] for th in np.array_split(pre_idx, n_sub)])
    else:
        C = np.zeros((J, 0))
    X = np.column_stack([np.ones(J), s0, C])

    def slope(Xm, Dm):
        return np.linalg.lstsq(Xm, Dm.T, rcond=None)[0][1]

    phi = slope(X, D)
    jk = np.array([slope(np.delete(X, j, 0), np.delete(D, j, 1)) for j in range(J)])
    se = np.sqrt((J - 1) / J * ((jk - jk.mean(0)) ** 2).sum(0))
    return phi, se, jk


def jackknife_phase_se(jk: np.ndarray, mask: np.ndarray) -> float:
    """Jackknife standard error of a phase average of ``phi_t``."""
    J = jk.shape[0]
    ja = jk[:, mask].mean(1)
    return float(np.sqrt((J - 1) / J * ((ja - ja.mean()) ** 2).sum()))


def regression_hat_values(Y0: np.ndarray, s0: np.ndarray, pre: np.ndarray, base: int, n_sub: int = 3) -> np.ndarray:
    """Hat values of the exposure regression's design (``code/app_influence.py``)."""
    pre_idx = np.where(pre)[0]
    pre_idx = pre_idx[pre_idx < base]
    C = np.column_stack([Y0[th[-1]] - Y0[th[0]] for th in np.array_split(pre_idx, n_sub)])
    X = np.column_stack([np.ones(Y0.shape[1]), s0, C])
    return np.diag(X @ np.linalg.pinv(X.T @ X) @ X.T)


def easc_gap(y1: np.ndarray, Y0: np.ndarray, w: np.ndarray, s1: float, s0: np.ndarray,
             phi: np.ndarray) -> Tuple[np.ndarray, float]:
    """Exposure-augmented gap ``y_1t - sum_j w_j Y_jt - phi_t EI(w)`` (eq. 9). Ports ``ebsc.easc``."""
    ei = exposure_imbalance(s1, s0, w)
    return y1 - Y0 @ w - phi * ei, ei


def select_frontier_point(path: List[Dict], kappa_lat: np.ndarray, se_phi: np.ndarray, post: np.ndarray) -> Dict:
    """EASC*: the frontier point minimising the heuristic post-period MSE (eq. 12).

    ``mean_t[(kappa^lambda_t)^2 (R(d)^2 - R(dbar)^2) + se(phi_t)^2 EI(w(d))^2]``,
    with ``dbar`` the synthetic-control end (the last point of ``path``). Ties
    keep the first minimiser, as the reference does. Ports ``ebsc.easc_select``
    and the selection step of ``ebsc.easc_tuned``.
    """
    r_sc = path[-1]["rmspe"]
    best = None
    for p in path:
        eta2 = max(0.0, p["rmspe"] ** 2 - r_sc ** 2)
        crit = float(np.mean(kappa_lat[post] ** 2 * eta2 + se_phi[post] ** 2 * p["ei"] ** 2))
        p.update(eta=float(np.sqrt(eta2)), crit=crit)
        if best is None or crit < best["crit"]:
            best = p
    return best


# ------------------------------------------------------------------ inference


def cv_bias_aware(b: float, alpha: float = 0.05) -> float:
    """Smallest ``c`` with ``P(|Z + b| <= c) >= 1 - alpha`` (Prop. 5; Armstrong-Kolesar).

    ``|Z + b|^2`` is a noncentral chi-square with one degree of freedom and
    noncentrality ``b^2``. Ports ``ebsc.cv_bias_aware``.
    """
    b = abs(b)
    if b < 1e-10:
        return float(norm.ppf(1 - alpha / 2))
    return float(np.sqrt(ncx2.ppf(1 - alpha, df=1, nc=b ** 2)))


def bias_aware_halfwidth(M: float, kappa: float, rmspe: float, sd: float, alpha: float = 0.05) -> float:
    """Half-length of the interval with bias bound ``B = M kappa RMSPE`` (eq. 13, Prop. 5)."""
    return cv_bias_aware(M * kappa * rmspe / sd, alpha) * sd


def breakdown(tau: float, kappa: float, rmspe: float, sd: float, alpha: float = 0.05, m_max: float = 5.0) -> float:
    """``M* = sup{M >= 0 : 0 not in CI(B(M))}`` (Prop. 5), capped at ``m_max`` as in the reference.

    The half-length is continuous and increasing in ``M``, so ``M*`` is the
    root of ``|tau| - halfwidth(M)``. Ports ``ebsc.breakdown_M``.
    """
    f = lambda M: abs(tau) - bias_aware_halfwidth(M, kappa, rmspe, sd, alpha)
    if f(0.0) <= 0:
        return 0.0
    if f(m_max) > 0:
        return m_max
    return float(brentq(f, 0.0, m_max, xtol=1e-14))


# ------------------------------------------------------- application pipeline


def analyse(y1: np.ndarray, Y0: np.ndarray, pre: np.ndarray, s1: float, s0: np.ndarray, x: np.ndarray,
            base: int, k: int = 2, controls: bool = True, grid: int = 41) -> Dict:
    """Diagnostics and the three estimators for one donor pool (Sections 4.1-4.2).

    Mirrors ``analyse`` in ``code/app.py``: the leverage of the latent factors
    alone (``kappa``), the exposure-orthogonal factors used for the partial
    leverage and the latent leverage, the exposure regression, the
    synthetic-control weights and gap, EASC at those weights, and the frontier
    with EBSC-min (its first point) and EASC* (its selected point).
    """
    post = ~pre
    F = donor_factors(Y0, k)
    Fl = donor_factors(Y0, k, s0)
    ell, r2 = partial_leverage(x, Fl, pre)
    ell_all, r2_all = partial_leverage(x, F, pre)
    phi, se, jk = exposure_effects(Y0, s0, pre, base, controls)
    w = sc_weights(y1[pre], Y0[pre])
    g_sc = y1 - Y0 @ w
    g_easc, ei_sc = easc_gap(y1, Y0, w, s1, s0, phi)
    kappa_lat = leverage(with_constant(Fl), pre)
    path = frontier(y1[pre], Y0[pre], s1, s0, n=grid)
    best = select_frontier_point(path, kappa_lat, se, post)
    g_tuned, ei_tuned = easc_gap(y1, Y0, best["w"], s1, s0, phi)
    return dict(
        dmin=min_imbalance(s1, s0), F=F, Fl=Fl, kappa=leverage(with_constant(F), pre), kappa_lat=kappa_lat,
        partial=ell, r2=r2, partial_all=ell_all, r2_all=r2_all, phi=phi, se_phi=se, jk_phi=jk,
        w_sc=w, rmspe_sc=float(np.sqrt(np.mean(g_sc[pre] ** 2))), ei_sc=ei_sc, g_sc=g_sc, g_easc=g_easc,
        path=path, best=best, g_easc_tuned=g_tuned, ei_tuned=ei_tuned,
        g_ebsc_min=y1 - Y0 @ path[0]["w"], ei_min=path[0]["ei"], rmspe_min=path[0]["rmspe"],
    )


def _unit(y1, Y0, pre, s1, s0, base, k=2) -> Dict:
    """One unit's SC and EASC gaps, as ``run_unit`` in ``code/app_inference.py``."""
    w = sc_weights(y1[pre], Y0[pre])
    g = y1 - Y0 @ w
    phi, se, jk = exposure_effects(Y0, s0, pre, base)
    ge, ei = easc_gap(y1, Y0, w, s1, s0, phi)
    return dict(g=g, ge=ge, ei=ei, phi=phi, se=se, jk=jk, w=w, rmspe=float(np.sqrt(np.mean(g[pre] ** 2))),
                Fl=donor_factors(Y0, k, s0))


def phase_inference(y1: np.ndarray, Y0: np.ndarray, pre: np.ndarray, s1: float, s0: np.ndarray, x: np.ndarray,
                    base: int, phases: Dict[str, np.ndarray], alpha: float = 0.05,
                    Ms: Sequence[float] = (0.0, 0.5, 1.0), easc_se: str = "period_mean") -> List[Dict]:
    """Placebo noise scale, bias-aware intervals and breakdown values (Section 4.3).

    Mirrors ``code/app_inference.py``. Every donor in turn is the treated unit
    with the remaining donors as controls (the treated unit is never a
    control). The noise scale is the standard deviation of the placebos' phase
    averages, each divided by its own pre-period RMSPE, rescaled by the treated
    unit's RMSPE; the robust version replaces it with 1.4826 times the median
    absolute deviation. For EASC the correction's variance is added in
    quadrature.

    ``easc_se`` chooses that variance. ``"period_mean"`` is what the reference
    code does: ``|EI|`` times the phase mean of the per-period jackknife
    standard errors. ``"phase"`` is what the text describes: ``|EI|`` times the
    jackknife standard error of the phase average of ``phi_t``.
    """
    J = Y0.shape[1]
    tr = _unit(y1, Y0, pre, s1, s0, base)
    plac = []
    for j in range(J):
        keep = [i for i in range(J) if i != j]
        plac.append(_unit(Y0[:, j], Y0[:, keep], pre, float(s0[j]), s0[keep], base))
    Fl = tr["Fl"]
    rows = []
    for est, key in (("SC", "g"), ("EASC", "ge")):
        for ph, m in phases.items():
            tau = float(tr[key][m].mean())
            z = np.array([p[key][m].mean() / p["rmspe"] for p in plac])
            sd = float(z.std(ddof=1) * tr["rmspe"])
            sdm = float(1.4826 * np.median(np.abs(z - np.median(z))) * tr["rmspe"])
            if est == "EASC":
                extra = abs(tr["ei"]) * (tr["se"][m].mean() if easc_se == "period_mean"
                                         else jackknife_phase_se(tr["jk"], m))
                sd = float(np.sqrt(sd ** 2 + extra ** 2))
                sdm = float(np.sqrt(sdm ** 2 + extra ** 2))
                ell, _ = partial_leverage(x, Fl, pre)
                kbar = float(np.sqrt(phase_leverage(with_constant(Fl), pre, m) ** 2 + ell[base] ** 2))
            else:
                kbar = phase_leverage(with_constant(Fl, x), pre, m)
            stat = abs(tau) / tr["rmspe"]
            pval = (1 + sum(abs(p[key][m].mean()) / p["rmspe"] >= stat for p in plac)) / (J + 1)
            ci = {M: (tau - bias_aware_halfwidth(M, kbar, tr["rmspe"], sd, alpha),
                      tau + bias_aware_halfwidth(M, kbar, tr["rmspe"], sd, alpha)) for M in Ms}
            rows.append(dict(est=est, phase=ph, tau=tau, sd=sd, sdm=sdm, kbar=kbar, pval=float(pval), ci=ci,
                             Mstar=breakdown(tau, kbar, tr["rmspe"], sd, alpha),
                             Mstar_mad=breakdown(tau, kbar, tr["rmspe"], sdm, alpha)))
    return rows
