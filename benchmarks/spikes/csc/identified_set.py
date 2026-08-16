"""How wide is the set of ATTs consistent with an optimal CSC fit?

Cross-solver disagreement shows the optimum is not a point, but it conflates
multiplicity with solver failure. This measures the thing directly.

Solve the CSC program for its optimal value ``v*``. Then, over the same
feasible set restricted to fits at least as good as ``v* (1 + tol)``,
maximise and minimise the ATT functional

    ATT(omega, alpha, eta) = mean_{i,t in post} ( y_it - eta_i - sum_j w_ij y_jt )

which is linear in the parameters. Both are convex programs (linear
objective, one convex quadratic constraint), so the interval between them is
the exact set of ATT values the estimator's own objective cannot distinguish
between. A well-identified estimator returns a point; this returns an
interval, and its width is what a practitioner would unknowingly pick from.
"""

from __future__ import annotations

import warnings

import cvxpy as cp
import numpy as np

from core import csc_counterfactual, csc_weights
from dgp import simulate


def att_range(Y1_pre, Y0_pre, X1, Y1_post, Y0_post, *, tol=1e-6):
    """(att_at_optimum, att_min, att_max) over the near-optimal set."""
    n1, T0 = Y1_pre.shape
    n0 = Y0_pre.shape[0]
    K = X1.shape[1]
    T1 = Y0_post.shape[1]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = csc_weights(Y1_pre, Y0_pre, X1)
    v_star = base.objective
    att_hat = float(np.mean(Y1_post - csc_counterfactual(base, Y0_post)))

    omega = cp.Variable(n0)
    alpha = cp.Variable((n0, K))
    eta = cp.Variable(n1)
    W = np.ones((n1, 1)) @ cp.reshape(omega, (1, n0), order="C") + X1 @ alpha.T
    resid = (
        Y1_pre
        - W @ Y0_pre
        - cp.reshape(eta, (n1, 1), order="C") @ np.ones((1, T0))
    )
    att = cp.sum(Y1_post) / (n1 * T1) - (
        cp.sum(W @ Y0_post) + T1 * cp.sum(eta)
    ) / (n1 * T1)
    feasible = [
        W @ np.ones(n0) == np.ones(n1),
        W >= 0,
        cp.sum_squares(resid) <= v_star * (1 + tol) + 1e-9,
    ]

    out = []
    for sense in (cp.Minimize, cp.Maximize):
        prob = cp.Problem(sense(att), feasible)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prob.solve(solver=cp.CLARABEL)
        out.append(float(prob.value) if prob.value is not None else np.nan)
    return att_hat, out[0], out[1]


CONFIGS = {
    "paper baseline (N=100, T=5)": dict(N=100, T=5, prob_tr=0.15),
    "more treated (E[pi]=0.40)": dict(N=100, T=5, prob_tr=0.40),
    "longer pre-period (T=12)": dict(N=100, T=12, prob_tr=0.15),
    "T=12 and E[pi]=0.40": dict(N=100, T=12, prob_tr=0.40),
    "small pool (N=30, E[pi]=0.40)": dict(N=30, T=5, prob_tr=0.40),
    "tiny pool (N=20, T=12, 0.40)": dict(N=20, T=12, prob_tr=0.40),
}


if __name__ == "__main__":
    print(f"{'configuration':<32} {'eq/par':>7} {'ATT':>7} {'identified set':>22} {'width':>8}")
    for name, cfg in CONFIGS.items():
        widths, hats, los, his, ratios = [], [], [], [], []
        for seed in range(10):
            p = simulate(rng=np.random.default_rng(seed), **cfg)
            tr, do = p.treated_idx, p.donor_idx
            if tr.size < 2 or do.size < 2:
                continue
            T0 = p.Y.shape[1] - 1
            X1 = p.X_dummies[tr]
            try:
                hat, lo, hi = att_range(
                    p.Y[tr, :T0], p.Y[do, :T0], X1, p.Y[tr, T0:], p.Y[do, T0:]
                )
            except Exception:  # noqa: BLE001
                continue
            if not np.isfinite([hat, lo, hi]).all():
                continue
            hats.append(hat)
            los.append(lo)
            his.append(hi)
            widths.append(hi - lo)
            ratios.append((tr.size * T0) / (do.size * (1 + X1.shape[1]) + tr.size))
        if not widths:
            print(f"{name:<32} no usable panels")
            continue
        print(
            f"{name:<32} {np.mean(ratios):7.2f} {np.median(hats):7.2f} "
            f"[{np.median(los):+8.2f}, {np.median(his):+8.2f}] {np.median(widths):8.2f}"
        )
    print("\n(true tau = 1.0; medians over 10 panels; tolerance 1e-6 on the fit)")
