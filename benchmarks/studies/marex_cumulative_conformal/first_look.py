"""First look: which bands for a cumulative effect keep coverage, on which laws.

MAREX withholds ``T_b`` blank periods from the fit, so its contrast errors there
are out of sample, and the post-period total is the point estimate minus the sum
of ``H`` post errors. Every band below is a rule for covering that sum. The
effect path enters only the two inversion arms, which see the post outcomes.

Arms:

* ``shipped`` -- MAREX's ``cumulative_band``: sign-flipped block paths of the
  blank errors with one level per horizon-length window.
* ``batch`` -- split conformal on the sums of disjoint horizon-length blank
  windows, uncentred. Exact when the windows are exchangeable, and infinite
  until there are ``1 / alpha - 1`` of them.
* ``rolling`` -- split conformal on every consecutive ``H``-sum of the blank
  errors, the overlapping reading SynthPower's rolling origins use.
* ``inv_sum`` / ``inv_abs`` -- invert the constant-effect sharp null with
  Abadie and Zhao's permutation scheme over the blank and post periods: the
  signed post sum (CWZ Remark 1) and the mean absolute post error (Abadie and
  Zhao eq. 16). Coverage of the total ``C`` is the event that the null
  ``tau_t = C / H`` is not rejected, so no search is needed.

Laws for the per-period errors:

* ``gauss`` -- iid normal.
* ``t2`` -- iid Student t on two degrees of freedom.
* ``ar5`` -- a stationary AR(1) with coefficient one half, in time order.
* ``spike_exch`` -- iid normal plus one spike of size 200 at a uniformly drawn
  period: exchangeable, not iid.
* ``spike_iid`` -- iid normal plus an independent spike of size 200 in each
  period with probability ``p``, set to the worst case for the size.

Run from the repository root::

    python -m benchmarks.studies.marex_cumulative_conformal.first_look
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from mlsynth.utils.conformal import block_error_paths
from mlsynth.utils.conformal.quantile import split_conformal_quantile

SPIKE = 200.0
#: The spike probability that maximises ``(1 - (1 - p)^H) (1 - p)^n``, so a band
#: built from the blank errors alone is most often blind to a post spike.
WORST_P = {(47, 13): 0.0186, (104, 13): 0.0090, (247, 13): 0.0039}


def errors(law, n, H, rng):
    N = n + H
    if law == "gauss":
        return rng.standard_normal(N)
    if law == "t2":
        return rng.standard_t(2, N)
    if law == "ar5":
        e = rng.standard_normal(N + 200)
        x = np.empty_like(e)
        x[0] = e[0]
        for t in range(1, e.size):
            x[t] = 0.5 * x[t - 1] + e[t]
        return x[200:]
    if law == "spike_exch":
        e = rng.standard_normal(N)
        e[rng.integers(N)] += SPIKE
        return e
    if law == "spike_iid":
        return rng.standard_normal(N) + SPIKE * (rng.random(N) < WORST_P[(n, H)])
    raise ValueError(law)


def shipped(blank, H, alpha, rng):
    m = blank.size // H
    paths = block_error_paths(blank, horizon=H, block=0, n_sim=2000, rng=rng,
                              n_windows=m if m >= 2 else 0)
    return float(np.quantile(np.abs(paths.sum(axis=1)), 1.0 - alpha))


def batch(blank, H, alpha):
    m = blank.size // H
    return split_conformal_quantile(blank[:m * H].reshape(m, H).sum(axis=1), alpha)


def rolling(blank, H, alpha):
    c = np.concatenate([[0.0], np.cumsum(blank)])
    return split_conformal_quantile(c[H:] - c[:-H], alpha)


def paths(H):
    t = np.arange(1, H + 1, dtype=float)
    out = {"constant": np.ones(H),
           "ramp": 2.0 * t / (H + 1),
           "decay": np.exp(-(t - 1) / 3.0),
           "one_week": np.r_[1.0, np.zeros(H - 1)]}
    # every path has total H, so a unit of effect per period on average
    return {k: v * H / v.sum() for k, v in out.items()}


def invert(blank, post_imputed, stat, R, rng):
    v = np.concatenate([blank, post_imputed])
    H = post_imputed.size
    idx = rng.random((R, v.size)).argpartition(H, axis=1)[:, :H]
    obs = stat(post_imputed[None, :])[0]
    ref = stat(v[idx])
    return (1 + np.sum(ref >= obs - 1e-9)) / (R + 1)


STATS = {"inv_sum": lambda X: np.abs(X.sum(axis=1)),
         "inv_abs": lambda X: np.abs(X).mean(axis=1)}


def coverage(n, H, alphas, laws, draws, seed, R=499, scale=3.0):
    rng = np.random.default_rng(seed)
    P = paths(H)
    for law in laws:
        hit = {}
        inf = {}
        for d in range(draws):
            u = errors(law, n, H, rng)
            blank, post = u[:n], u[n:]
            S = post.sum()
            for a in alphas:
                for arm, half in (("shipped", shipped(blank, H, a, rng)),
                                  ("batch", batch(blank, H, a)),
                                  ("rolling", rolling(blank, H, a))):
                    hit[(arm, a)] = hit.get((arm, a), 0) + (abs(S) <= half)
                    inf[(arm, a)] = inf.get((arm, a), 0) + (not np.isfinite(half))
                for name, tau in P.items():
                    # the post outcome less the estimator's prediction is the
                    # error plus the effect; the null imputes tau_t = C / H
                    imputed = post + scale * (tau - tau.mean())
                    for sname, stat in STATS.items():
                        p = invert(blank, imputed, stat, R, rng)
                        key = (f"{sname}:{name}", a)
                        hit[key] = hit.get(key, 0) + (p > a)
        for a in alphas:
            row = "  ".join(
                f"{k[0]}={hit[k] / draws:.3f}"
                + (f"(inf {inf[k] / draws:.2f})" if inf.get(k) else "")
                for k in sorted(hit) if k[1] == a)
            print(f"n={n} H={H} alpha={a} {law}: {row}", flush=True)


LAWS = ("gauss", "t2", "ar5", "spike_exch", "spike_iid")

if __name__ == "__main__":
    # python -m ...first_look [draws] [n] [law]; omitted arguments run them all
    draws = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    sizes = (int(sys.argv[2]),) if len(sys.argv) > 2 else (47, 104, 247)
    laws = (sys.argv[3],) if len(sys.argv) > 3 else LAWS
    for n in sizes:
        for law in laws:
            coverage(n, 13, (0.05, 0.2), (law,), draws, seed=n * 7 + LAWS.index(law))
