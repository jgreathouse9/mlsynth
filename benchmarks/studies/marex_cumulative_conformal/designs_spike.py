"""Spike: the detrending arms and the library's CWZ inversion through MAREX designs.

The panel and design are ``designs.py``'s (Section 5 dimensions, ``m_eq = 3``,
exact backend), at 40 fitting, 26 blank and 13 post periods, with the period
effects independent (``iid``), autoregressive (``ar``) or sorted into a trend
(``sorted``). Each draw adds an effect path to the treated units and asks each
arm whether its band covers the total.

Arms:

* ``cyclic`` / ``subsets`` -- the signed-sum inversion on the blank and post
  contrast, no detrending.
* ``cyclic+plugin`` / ``cyclic+exact`` -- the detrending arms of ``detrend.py``.
* ``cwz`` -- :func:`mlsynth.utils.conformal.cumulative_conformal_by_inversion`
  as it ships: the synthetic treated series against the control pool, the
  simplex control refitted on all fitting, blank and post periods under each
  constant-effect null, cyclic shifts, the ``S_1`` statistic.
* ``cwz_signed`` -- the same refit, through the library's
  :func:`~mlsynth.utils.conformal.refit.conformal_refit_gaps`, with the signed
  post sum of CWZ's Remark 1 in place of ``S_1``. Coverage is read at the true
  total only: the null ``tau_t = C / H`` is not rejected.
* ``cwz_signed_w`` -- the same, each shift weighted by ``0.97`` to its circular
  distance from the post window.

Widths are for the constant path at the first level, in outcome units; the
signed arms' are the hull of the accepted totals on a 121-point grid.

Run from the repository root::

    python -m benchmarks.studies.marex_cumulative_conformal.designs_spike [draws] [dgp] [seed0]
"""
import os
import sys
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

import numpy as np

from benchmarks.studies.marex_cumulative_conformal.candidate2 import shapes
from benchmarks.studies.marex_cumulative_conformal.designs import design, panel
from benchmarks.studies.marex_cumulative_conformal.detrend import (
    slope_grid, slope_pvalues, sum_pvalues)
from benchmarks.studies.marex_cumulative_conformal.inversion import band, blocks, reference
from mlsynth.utils.conformal import cumulative_conformal_by_inversion
from mlsynth.utils.conformal.refit import conformal_refit_gaps


def signed_refit_pvalue(y, Y0, T0, H, C, rho=None):
    """CWZ's cyclic-shift p-value of ``tau_t = C / H`` with the signed post sum.

    ``rho`` weights shift ``k`` by ``rho`` to its circular distance from the post
    window, Barber, Candes, Ramdas and Tibshirani's recency weights.
    """
    N = T0 + H
    y0 = np.r_[y[:T0], y[T0:N] - C / H]
    r = conformal_refit_gaps(y0, Y0[:N], rule="sc")
    idx = (T0 + np.arange(N)[:, None] + np.arange(H)[None, :]) % N
    S = np.abs(r[idx].sum(axis=1))
    hit = S >= S[0] - 1e-9 * (1.0 + np.abs(r).sum())
    if rho is None:
        return float(np.mean(hit))
    k = np.arange(N)
    w = rho ** np.minimum(k, N - k)
    return float(hit @ w / w.sum())

def signed_refit_band(y, Y0, T0, H, alpha, centre, half, points=121, rho=None):
    """Hull of the accepted totals on a grid, and whether it reaches an edge."""
    grid = np.linspace(centre - half, centre + half, points)
    ok = np.array([signed_refit_pvalue(y, Y0, T0, H, c, rho) > alpha for c in grid])
    if not ok.any():
        return np.nan, False
    lo, hi = np.flatnonzero(ok)[[0, -1]]
    return grid[hi] - grid[lo], bool(ok[0] or ok[-1])


TE, TB, H = 40, 26, 13
ALPHAS = (0.05, 0.2)


def run(dgp, draws, seed0, R=999):
    T0 = TE + TB
    T = T0 + H
    paths = {"constant": np.zeros(H), "ramp": 2.0 * shapes(H)["ramp"]}
    hit = {}
    widths = {}
    for d in range(draws):
        rng = np.random.default_rng(seed0 + d)
        Y = panel(T, dgp, rng)
        w, v = design(Y, TE, TB, H)
        controls = np.flatnonzero(w <= 1e-8)
        blank = (Y @ w - Y @ v)[TE:T0]
        for name, dev in paths.items():
            tau = 1.0 + dev
            C = float(tau.sum())
            y = Y @ w
            y = np.r_[y[:T0], y[T0:] + tau]
            gaps = y[T0:] - Y[T0:] @ v
            theta_hat, grid = slope_grid(blank)
            p_slope = {s: slope_pvalues(blank, grid, s, R, rng) for s in ("cyclic",)}
            for scheme in ("cyclic", "subsets"):
                idx = blocks(TB, H, scheme, max_subsets=R, rng=rng)
                p_none = sum_pvalues(blank, gaps, np.array([0.0]), C, idx)[0]
                for a in ALPHAS:
                    key = (scheme, name, a)
                    hit[key] = hit.get(key, 0) + (p_none > a)
                if scheme == "cyclic":
                    p_plug = sum_pvalues(blank, gaps, np.array([theta_hat]), C, idx)[0]
                    p_joint = sum_pvalues(blank, gaps, grid, C, idx)
                    for a in ALPHAS:
                        ok = p_slope["cyclic"] > a / 2
                        for arm, h in (("cyclic+plugin", p_plug > a),
                                       ("cyclic+exact", bool(np.any(ok & (p_joint > a / 2))))):
                            hit[(arm, name, a)] = hit.get((arm, name, a), 0) + h
            p_sig = signed_refit_pvalue(y, Y[:, controls], T0, H, C)
            p_sig_w = signed_refit_pvalue(y, Y[:, controls], T0, H, C, rho=0.97)
            for a in ALPHAS:
                hit[("cwz_signed", name, a)] = hit.get(("cwz_signed", name, a), 0) + (p_sig > a)
                hit[("cwz_signed_w", name, a)] = hit.get(("cwz_signed_w", name, a), 0) + (p_sig_w > a)
            for a in ALPHAS:
                b = cumulative_conformal_by_inversion(
                    y, Y[:, controls], T0, H, alpha=a, refit="sc", conformal_type="block")
                covered = (not b.is_empty) and b.lower <= C <= b.upper
                hit[("cwz", name, a)] = hit.get(("cwz", name, a), 0) + covered
                if name == "constant" and a == ALPHAS[0]:
                    idx = blocks(TB, H, "cyclic")
                    bc = band(reference(blank, gaps, idx), a)
                    w_c = bc.upper - bc.lower
                    w_l = b.upper - b.lower if not b.is_empty else np.nan
                    half = 3.0 * np.nanmax([w_c, w_l, 1.0])
                    w_s, edge = signed_refit_band(y, Y[:, controls], T0, H, a,
                                                  float(gaps.sum()), half)
                    w_sw, _ = signed_refit_band(y, Y[:, controls], T0, H, a,
                                                float(gaps.sum()), half, rho=0.97)
                    for k, wv in (("cyclic", w_c), ("cwz", w_l), ("cwz_signed", w_s),
                                  ("cwz_signed_w", w_sw)):
                        widths.setdefault(k, []).append(wv)
                    widths.setdefault("edge", []).append(edge)
    for a in ALPHAS:
        cells = " ".join(f"{k[0]}:{k[1]}={c / draws:.3f}"
                         for k, c in sorted(hit.items()) if k[2] == a)
        print(f"TE={TE} Tb={TB} H={H} {dgp} alpha={a} draws={draws}: {cells}", flush=True)
    ws = " ".join(f"{k}={np.nanmedian(np.asarray(v, float)):.2f}"
                  for k, v in widths.items() if k != "edge")
    print(f"TE={TE} Tb={TB} H={H} {dgp} median width at alpha={ALPHAS[0]}, constant path: {ws}; "
          f"signed refit grid edge reached {np.mean(widths['edge']):.3f}", flush=True)


if __name__ == "__main__":
    draws = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    dgps = (sys.argv[2],) if len(sys.argv) > 2 else ("iid", "ar", "sorted")
    seed0 = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    for g in dgps:
        run(g, draws, seed0)
