"""Confirming the cause: keep the fitting periods out of the reference set.

``whiten_diag.py`` found the refit residuals about 30% smaller in mean square in
the fitting periods, where MAREX chose the treated set, than in the blank and
post periods, which behave alike; persistent factors amplify the gap for a
13-period sum. If that leak is the cause, a reference set drawn from the blank
and post periods alone should restore coverage under every regime.

Arms, all the signed post sum at the true total:

* ``all`` -- the library refit on every period, cyclic shifts over all of them
  (the arm that fell short).
* ``refit_all_ref_bp`` -- the same residuals, cyclic shifts over the blank and
  post periods only.
* ``refit_bp`` -- the control refitted on the blank and post periods alone,
  cyclic shifts over them.
* ``fixed_bp`` -- MAREX's own contrast (weights from the fitting periods, no
  refit), cyclic shifts over the blank and post periods.
* ``refit_all_ref_bp_w`` -- ``refit_all_ref_bp`` with each shift weighted by
  0.97 to its circular distance from the post window.

Run from the repository root::

    python -m benchmarks.studies.marex_cumulative_conformal.leak_fix [draws] [dgp]
"""
import os
import sys
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

import numpy as np

from benchmarks.studies.marex_cumulative_conformal.candidate2 import shapes
from benchmarks.studies.marex_cumulative_conformal.designs import design, panel
from mlsynth.utils.conformal.refit import conformal_refit_gaps

ALPHAS = (0.05, 0.2)
TE, TB, H = 40, 26, 13


def cyclic_p(x, H, rho=None):
    """Signed post sum against all cyclic shifts of ``x``; post is the last ``H``."""
    N = x.size
    idx = (N - H + np.arange(N)[:, None] + np.arange(H)[None, :]) % N
    S = np.abs(x[idx].sum(axis=1))
    hit = S >= S[0] - 1e-9 * (1.0 + S[0])
    if rho is None:
        return float(np.mean(hit))
    k = np.arange(N)
    w = rho ** np.minimum(k, N - k)
    return float(hit @ w / w.sum())


def band_width(fn, C, half=200.0, points=161, alpha=0.05):
    grid = C + np.linspace(-half, half, points)
    ok = np.array([fn(c) > alpha for c in grid])
    if not ok.any():
        return np.nan
    lo, hi = np.flatnonzero(ok)[[0, -1]]
    return np.inf if (ok[0] or ok[-1]) else grid[hi] - grid[lo]


def run(dgp, draws, seed0=0):
    T0 = TE + TB
    N = T0 + H
    paths = {"constant": np.zeros(H), "ramp": 2.0 * shapes(H)["ramp"]}
    hit, widths = {}, {}
    for d in range(draws):
        rng = np.random.default_rng(seed0 + d)
        Y = panel(N, dgp, rng)
        w, v = design(Y, TE, TB, H)
        controls = np.flatnonzero(w <= 1e-8)
        Y0 = Y[:, controls]
        for name, dev in paths.items():
            tau = 1.0 + dev
            C = float(tau.sum())
            y = Y @ w
            y = np.r_[y[:T0], y[T0:] + tau]

            def arms(c):
                yc = np.r_[y[:T0], y[T0:] - c / H]
                r_all = conformal_refit_gaps(yc, Y0, rule="sc")
                r_bp = conformal_refit_gaps(yc[TE:], Y0[TE:], rule="sc")
                g = (yc - Y @ v)[TE:]
                return {"all": cyclic_p(r_all, H),
                        "refit_all_ref_bp": cyclic_p(r_all[TE:], H),
                        "refit_all_ref_bp_w": cyclic_p(r_all[TE:], H, rho=0.97),
                        "refit_bp": cyclic_p(r_bp, H),
                        "fixed_bp": cyclic_p(g, H)}

            p = arms(C)
            for a in ALPHAS:
                for arm, pv in p.items():
                    hit[(arm, name, a)] = hit.get((arm, name, a), 0) + (pv > a)
            if name == "constant" and d < 100:
                for arm in p:
                    widths.setdefault(arm, []).append(
                        band_width(lambda c, arm=arm: arms(c)[arm], C))
    for a in ALPHAS:
        cells = " ".join(f"{k[0]}:{k[1]}={c / draws:.3f}"
                         for k, c in sorted(hit.items()) if k[2] == a)
        print(f"leak {dgp} alpha={a} draws={draws}: {cells}", flush=True)
    ws = " ".join(f"{k}={np.nanmedian(v):.1f}(inf {np.mean(np.isinf(v)):.2f})"
                  for k, v in widths.items())
    print(f"leak {dgp}: median width at 0.05 (first 100 draws, constant path): {ws}", flush=True)


if __name__ == "__main__":
    draws = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    dgps = (sys.argv[2],) if len(sys.argv) > 2 else ("ar", "ar5", "iid", "sorted")
    for g in dgps:
        run(g, draws)
