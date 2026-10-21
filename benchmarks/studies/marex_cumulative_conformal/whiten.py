"""Spike: whitening the residuals before permuting, for serially dependent errors.

Chernozhukov, Wuthrich and Zhu (2021, Section 4.4.2 and Lemma 7) fit an AR(K)
model to the residuals under the null and permute the innovations, which are
iid under the model, so all permutations are available again. For a total the
statistic should stay a function of the total alone, and a sum of innovations
is not one (the filter carries the last period's effect into it). So the
innovations are permuted and the level paths rebuilt through the same
recursion, and the statistic is the signed post sum of the rebuilt levels. The
identity permutation rebuilds the observed series, so the test is an ordinary
permutation test on the innovation vector: exact under the AR model with its
true coefficients, approximate with estimated ones.

Arms, each a p-value of the constant-effect null at the true total:

* ``cyclic`` -- the signed sum on the residuals, cyclic shifts (no model).
* ``subsets`` -- the same with all permutations of the periods.
* ``white`` / ``white2`` -- AR(1) / AR(2) fitted by least squares with an
  intercept on the null-imputed series, innovations permuted, levels rebuilt.
* ``white_innov`` -- AR(1), the signed post sum of the innovations themselves
  (CWZ's literal two-stage test with the Remark 1 statistic).
* ``white_exact`` -- Vovk, Gammerman and Shafer's detrending route with the AR
  filter as the transformation: for each ``rho`` on a grid the centred
  innovations are tested for exchangeability (lag-1 autocovariance, all
  permutations) at ``alpha / 2``, the level test runs at ``alpha / 2``, and the
  total is covered when some ``rho`` survives both.

Run from the repository root::

    python -m benchmarks.studies.marex_cumulative_conformal.whiten series [draws] [n] [law]
    python -m benchmarks.studies.marex_cumulative_conformal.whiten designs [draws] [dgp]
"""
import os
import sys
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

import numpy as np

from benchmarks.studies.marex_cumulative_conformal.candidate2 import _ar
from benchmarks.studies.marex_cumulative_conformal.inversion import blocks

ALPHAS = (0.05, 0.2)
RHO_GRID = np.linspace(-0.95, 0.95, 39)


def fit_ar(x, K):
    """Least squares of ``x_t`` on an intercept and ``K`` lags: ``(mu, rho)``."""
    N = x.size
    X = np.column_stack([np.ones(N - K)] + [x[K - j - 1:N - j - 1] for j in range(K)])
    coef, *_ = np.linalg.lstsq(X, x[K:], rcond=None)
    return float(coef[0]), coef[1:]


def innovations(x, mu, rho):
    K, N = rho.size, x.size
    return x[K:] - mu - sum(rho[j] * x[K - j - 1:N - j - 1] for j in range(K))


def rebuild(U, init, mu, rho):
    """Level paths from rows of innovations, through the AR recursion."""
    R, M = U.shape
    K = rho.size
    X = np.empty((R, K + M))
    X[:, :K] = init
    for t in range(M):
        X[:, K + t] = mu + U[:, t]
        for j in range(K):
            X[:, K + t] += rho[j] * X[:, K + t - j - 1]
    return X


def perms(M, R, rng):
    return np.vstack([np.arange(M)[None, :], rng.random((R, M)).argsort(axis=1)])


def level_pvalue(x, H, mu, rho, P):
    """Signed post sum of rebuilt levels; row 0 of ``P`` is the identity."""
    K = rho.size
    u = innovations(x, mu, rho)
    X = rebuild(u[P[1:]], x[:K], mu, rho)
    S = np.abs(X[:, -H:].sum(axis=1))
    s0 = abs(x[-H:].sum())
    return (1 + np.sum(S >= s0 - 1e-9 * (1 + s0))) / (P.shape[0])


def innov_pvalue(x, H, mu, rho, P):
    u = innovations(x, mu, rho)
    U = u[P]
    S = np.abs(U[:, -H:].sum(axis=1))
    return float(np.mean(S >= S[0] - 1e-9 * (1 + S[0])))


def exact_covered(x, H, alpha, P1):
    """Some AR(1) coefficient survives both the whiteness test and the level test."""
    xl, xc = x[:-1], x[1:]
    for rho in RHO_GRID:
        d = xc - rho * xl
        mu = d.mean()
        u = d - mu
        U = u[P1]
        ac = np.abs(np.sum(U[:, 1:] * U[:, :-1], axis=1))
        if np.mean(ac >= ac[0] - 1e-12) <= alpha / 2:
            continue
        if level_pvalue(x, H, mu, np.array([rho]), P1) > alpha / 2:
            return True
    return False


def pvalues(x, H, rng, R=999):
    """Every arm's p-value of the null that ``x`` is the residual series."""
    N = x.size
    n = N - H
    out = {}
    for name, scheme in (("cyclic", "cyclic"), ("subsets", "subsets")):
        idx = blocks(n, H, scheme, max_subsets=R, rng=rng)
        S = np.abs(x[idx].sum(axis=1))
        out[name] = float(np.mean(S >= S[0] - 1e-9 * (1 + S[0])))
    P1 = perms(N - 1, R, rng)
    mu, rho = fit_ar(x, 1)
    out["white"] = level_pvalue(x, H, mu, rho, P1)
    out["white_innov"] = innov_pvalue(x, H, mu, rho, P1)
    mu2, rho2 = fit_ar(x, 2)
    out["white2"] = level_pvalue(x, H, mu2, rho2, perms(N - 2, R, rng))
    out["_P1"] = P1
    return out


# ---------------------------------------------------------------------------
# error series

def series_errors(law, N, rng):
    if law == "gauss":
        return rng.standard_normal(N)
    if law.startswith("ar"):
        phi = {"ar5": 0.5, "ar8": 0.8, "arneg5": -0.5}[law]
        return _ar(phi, N, rng)
    raise ValueError(law)


def run_series(n, H, law, draws, seed):
    rng = np.random.default_rng(seed)
    hit = {}
    for _ in range(draws):
        x = series_errors(law, n + H, rng)       # the null-imputed series at the truth
        p = pvalues(x, H, rng)
        P1 = p.pop("_P1")
        for a in ALPHAS:
            for arm, pv in p.items():
                hit[(arm, a)] = hit.get((arm, a), 0) + (pv > a)
            hit[("white_exact", a)] = hit.get(("white_exact", a), 0) + exact_covered(x, H, a, P1)
    for a in ALPHAS:
        cells = "  ".join(f"{k[0]}={c / draws:.3f}" for k, c in sorted(hit.items()) if k[1] == a)
        print(f"series n={n} H={H} {law} alpha={a}: {cells}", flush=True)


# ---------------------------------------------------------------------------
# MAREX designs

def run_designs(dgp, draws, seed0, TE=40, TB=26, H=13):
    from benchmarks.studies.marex_cumulative_conformal.candidate2 import shapes
    from benchmarks.studies.marex_cumulative_conformal.designs import design, panel
    from mlsynth.utils.conformal.refit import conformal_refit_gaps

    T0 = TE + TB
    N = T0 + H
    paths = {"constant": np.zeros(H), "ramp": 2.0 * shapes(H)["ramp"]}
    hit, acf_g, acf_r, widths = {}, [], [], {}
    for d in range(draws):
        rng = np.random.default_rng(seed0 + d)
        Y = panel(N, dgp, rng)
        w, v = design(Y, TE, TB, H)
        controls = np.flatnonzero(w <= 1e-8)
        g = (Y @ w - Y @ v)[TE:]
        g = g - g.mean()
        acf_g.append(float(g[1:] @ g[:-1] / (g @ g)))
        for name, dev in paths.items():
            tau = 1.0 + dev
            C = float(tau.sum())
            y = Y @ w
            y = np.r_[y[:T0], y[T0:] + tau]
            x = conformal_refit_gaps(np.r_[y[:T0], y[T0:] - C / H], Y[:, controls], rule="sc")
            if name == "constant":
                xc = x - x.mean()
                acf_r.append(float(xc[1:] @ xc[:-1] / (xc @ xc)))
            p = pvalues(x, H, rng)
            P1 = p.pop("_P1")
            for a in ALPHAS:
                for arm, pv in p.items():
                    hit[(arm, name, a)] = hit.get((arm, name, a), 0) + (pv > a)
                hit[("white_exact", name, a)] = hit.get(("white_exact", name, a), 0) + \
                    exact_covered(x, H, a, P1)
            if name == "constant" and d < 100:
                # widths on the first 100 draws: hull of accepted totals on a grid,
                # one refit per candidate, the same permutations throughout
                grid = C + np.linspace(-60.0, 60.0, 81)
                acc = {"cyclic": [], "white": []}
                idx = blocks(T0, H, "cyclic")
                for c in grid:
                    xg = conformal_refit_gaps(np.r_[y[:T0], y[T0:] - c / H], Y[:, controls], rule="sc")
                    S = np.abs(xg[idx].sum(axis=1))
                    acc["cyclic"].append(np.mean(S >= S[0] - 1e-9 * (1 + S[0])) > 0.05)
                    mu, rho = fit_ar(xg, 1)
                    acc["white"].append(level_pvalue(xg, H, mu, rho, P1) > 0.05)
                for k, a_ in acc.items():
                    a_ = np.array(a_)
                    if a_.any():
                        lo, hi = np.flatnonzero(a_)[[0, -1]]
                        edge = a_[0] or a_[-1]
                        widths.setdefault(k, []).append(np.inf if edge else grid[hi] - grid[lo])
    for a in ALPHAS:
        cells = " ".join(f"{k[0]}:{k[1]}={c / draws:.3f}" for k, c in sorted(hit.items()) if k[2] == a)
        print(f"designs TE={TE} Tb={TB} H={H} {dgp} alpha={a} draws={draws}: {cells}", flush=True)
    ws = " ".join(f"{k}={np.median(v):.2f} (inf {np.mean(np.isinf(v)):.2f})" for k, v in widths.items())
    print(f"designs {dgp}: lag-1 acf of the blank+post contrast median {np.median(acf_g):+.2f}, "
          f"of the refit residuals {np.median(acf_r):+.2f}; median width at 0.05, first 100 "
          f"draws: {ws}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    draws = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    if mode == "series":
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 48
        laws = (sys.argv[4],) if len(sys.argv) > 4 else ("gauss", "ar5", "ar8", "arneg5")
        for law in laws:
            run_series(n, 13, law, draws, seed=100 * n + len(law))
    else:
        dgps = (sys.argv[3],) if len(sys.argv) > 3 else ("iid", "ar", "ar5", "arneg")
        for g_ in dgps:
            run_designs(g_, draws, 0)
